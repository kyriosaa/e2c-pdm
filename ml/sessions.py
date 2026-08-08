"""
Session metadata, measured sample rates, and the GAP MAP.

Why this module exists
----------------------
`data_catcher.py` de-frames the USB stream before writing: it validates the
header and CRC in memory, then writes only the payload bytes to
`vibration.bin`. Lost packets are therefore simply ABSENT from the file, with
no marker of any kind. `vibration.bin` is a splice of discontinuous segments.

Cutting an FFT window across such a splice fabricates a step discontinuity in
the waveform, which shows up as a broadband transient -- indistinguishable from
an impulsive mechanical fault. Training an anomaly detector on that data bakes
false positives into the "healthy" baseline.

`packet_index.csv` records the sequence number, sample count, flags and host
arrival time of every packet, which is enough to reconstruct exactly where in
`vibration.bin` every gap falls. This module turns that into:

  * a list of CONTIGUOUS SEGMENTS in vibration.bin sample coordinates, which is
    the only thing windowing code should ever cut from;
  * a MEASURED sample rate per stream (from sequence numbers and host
    timestamps), because the FFT frequency axis must never use a nominal
    datasheet ODR;
  * loss statistics localised in time, for rig diagnosis.

Results are cached to data/interim/<session>/index.json -- parsing a 366 MB
packet index takes a while and the answer never changes for a finished session.

Run directly for a summary of every session:
    python -m ml.sessions
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from . import config as C

CACHE_VERSION = 3


# ---------------------------------------------------------------------------
# session.json
# ---------------------------------------------------------------------------
def load_session_json(session_dir: Path) -> dict:
    path = Path(session_dir) / C.SESSION_JSON
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- is this a session dir?")
    with open(path) as f:
        return json.load(f)


def accel_fs_g(session: dict) -> float:
    """Accelerometer full scale in g for this session.

    Sessions recorded after the firmware full-scale change carry "accel_fs_g"
    explicitly. The three pilot sessions predate that field and were recorded
    at +/-16 g, so they fall back to the legacy value keyed off the start
    timestamp. Never hardcode full scale at a call site.
    """
    fs = session.get("accel_fs_g")
    if fs is not None:
        return float(fs)
    started = str(session.get("started", ""))
    if started and started < C.ACCEL_FS_CHANGED_AFTER:
        return C.ACCEL_FS_G_LEGACY
    # Unknown and not obviously legacy: refuse to guess silently.
    raise ValueError(
        f"session.json has no 'accel_fs_g' and started={started!r} is not "
        f"before {C.ACCEL_FS_CHANGED_AFTER}. Add accel_fs_g to session.json; "
        f"guessing full scale would silently scale every acceleration value.")


def lsb_per_g(fs_g: float) -> float:
    return 32768.0 / fs_g


# ---------------------------------------------------------------------------
# Packet index -> gap map + measured rates
# ---------------------------------------------------------------------------
def _cache_path(session_dir: Path) -> Path:
    return C.INTERIM_DIR / Path(session_dir).name / "index.json"


def build_index(session_dir: Path, verbose: bool = True) -> dict:
    """Parse packet_index.csv into segments, measured rates, loss stats.

    Single streaming pass. Returns a JSON-serialisable dict.
    """
    session_dir = Path(session_dir)
    path = session_dir / C.PACKET_INDEX_CSV
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Without the packet index there is no way to "
            f"locate gaps in vibration.bin; windowing would silently splice "
            f"across them.")

    # Per-stream running state.
    st: dict[int, dict] = {}

    def stream(t: int) -> dict:
        if t not in st:
            st[t] = {
                "packets": 0, "samples": 0,
                "first_seq": None, "last_seq": None, "last_time": None,
                "first_time": None,
                "gap_events": 0, "missing_packets": 0,
                "seg_start": 0, "segments": [],
                "gaps": [], "resets": [], "overruns": [],
                "flagged_packets": 0,
                "dropped_unknown": 0, "dropped_reported": 0,
                # Rate estimation accumulates only across pairs of packets that
                # belong to the same MCU boot, so an mid-session reset cannot
                # corrupt the estimate (see MAX_PLAUSIBLE_GAP below).
                "acc_intervals": 0, "acc_time": 0.0,
            }
        return st[t]

    # A sequence jump larger than this is treated as an MCU reset / resync
    # artefact rather than a real burst of lost packets: at 417 vib packets/s
    # this is ~40 minutes of loss, which would be a dead link, not a gap.
    MAX_PLAUSIBLE_GAP = 1_000_000

    size_mb = path.stat().st_size / 1e6
    if verbose:
        print(f"  parsing {C.PACKET_INDEX_CSV} ({size_mb:,.0f} MB)...")

    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        ix = {name: i for i, name in enumerate(header)}
        i_t, i_ty = ix["host_time"], ix["type"]
        i_sq, i_n = ix["seq"], ix["n_samples"]
        i_fl, i_dr = ix["flags"], ix["dropped"]

        for row in rdr:
            try:
                ty = int(row[i_ty])
                seq = int(row[i_sq])
                n = int(row[i_n])
                ht = float(row[i_t])
            except (ValueError, IndexError):
                continue                      # truncated final line, etc.

            s = stream(ty)
            if s["first_seq"] is None:
                s["first_seq"] = seq
                s["first_time"] = ht
            else:
                delta = seq - s["last_seq"]
                if delta == 1:
                    # Normal case: contiguous. Count the interval towards the
                    # rate estimate.
                    s["acc_intervals"] += 1
                    s["acc_time"] += ht - s["last_time"]
                elif 1 < delta <= MAX_PLAUSIBLE_GAP:
                    # Real loss: samples were produced but never arrived. The
                    # elapsed time still counts towards the rate estimate,
                    # which is why sequence-derived rates are immune to loss.
                    missing = delta - 1
                    s["acc_intervals"] += delta
                    s["acc_time"] += ht - s["last_time"]
                    s["gap_events"] += 1
                    s["missing_packets"] += missing
                    s["segments"].append([s["seg_start"], s["samples"]])
                    s["gaps"].append({
                        "at_sample": s["samples"],
                        "host_time": round(ht, 4),
                        "missing_packets": missing,
                    })
                    s["seg_start"] = s["samples"]
                else:
                    # delta <= 0, or an implausibly large jump: the MCU
                    # sequence counter restarted (reset / brownout / DTR-assert
                    # on port open) or a resync produced a bogus header. This
                    # interval is NOT counted towards the rate, and it is a
                    # hard discontinuity in the sample stream.
                    s["resets"].append({
                        "at_sample": s["samples"],
                        "host_time": round(ht, 4),
                        "seq_from": s["last_seq"],
                        "seq_to": seq,
                    })
                    s["segments"].append([s["seg_start"], s["samples"]])
                    s["seg_start"] = s["samples"]

            s["last_seq"] = seq
            s["last_time"] = ht
            s["packets"] += 1

            # ON-DEVICE LOSS. FLAG_FIFO_OVERRUN means the IIS3DWB's internal
            # FIFO wrapped and samples were lost INSIDE THE SENSOR, before the
            # MCU ever saw them. The packet itself still arrives with a
            # perfectly contiguous sequence number, so this loss is completely
            # invisible to sequence-gap detection -- but it splices the waveform
            # exactly like a transport gap does. It must break the segment too.
            #
            # The firmware signals the amount as dropped=0xFFFF ("unknown"),
            # because the sensor does not report how many samples it discarded;
            # the bound is the 3 KB FIFO, ~512 samples.
            if int(row[i_fl]) & 0x01:
                s["flagged_packets"] += 1
                s["segments"].append([s["seg_start"], s["samples"]])
                s["overruns"].append({
                    "at_sample": s["samples"],
                    "host_time": round(ht, 4),
                    "seq": seq,
                })
                s["seg_start"] = s["samples"]

            s["samples"] += n
            d = int(row[i_dr])
            if d == 0xFFFF:
                s["dropped_unknown"] += 1
            elif d > 0:
                s["dropped_reported"] += d

    out_streams = {}
    for ty, s in sorted(st.items()):
        s["segments"].append([s["seg_start"], s["samples"]])
        span = (s["last_time"] or 0.0) - (s["first_time"] or 0.0)
        n_per_pkt = (C.VIB_SAMPLES_PER_PACKET if ty == C.PKT_VIB
                     else C.AUDIO_SAMPLES_PER_PACKET if ty == C.PKT_AUDIO
                     else 1)
        # Rate = (packet intervals elapsed x samples per packet) / elapsed time,
        # accumulated only over intervals within one MCU boot. Sequence deltas
        # include packets that never arrived, so the estimate is immune to
        # transport loss; excluding reset boundaries makes it immune to reboots.
        measured_rate = ((s["acc_intervals"] * n_per_pkt / s["acc_time"])
                         if s["acc_time"] > 0 else None)

        missing_samples = s["missing_packets"] * n_per_pkt
        sent = s["samples"] + missing_samples

        out_streams[str(ty)] = {
            "packets": s["packets"],
            "samples": s["samples"],
            # Fraction of what the MCU actually transmitted that reached disk.
            # Distinct from session.json's ratio, which divides by nominal-rate
            # expectation and so folds in rate error and reset dead time.
            "transport_capture_ratio": (round(s["samples"] / sent, 8)
                                        if sent else None),
            "measured_rate_hz": (round(measured_rate, 3)
                                 if measured_rate else None),
            "rate_intervals": s["acc_intervals"],
            "rate_time_s": round(s["acc_time"], 3),
            "first_host_time": s["first_time"],
            "last_host_time": s["last_time"],
            "span_s": round(span, 3),
            "gap_events": s["gap_events"],
            "missing_packets": s["missing_packets"],
            "missing_samples": missing_samples,
            "resets": s["resets"],
            "fifo_overrun_packets": s["flagged_packets"],
            "overruns": s["overruns"],
            # Packets whose header carried dropped=0xFFFF, i.e. "an unknown
            # number of samples was lost in the sensor FIFO". Matches
            # fifo_overrun_packets by construction in the firmware.
            "dropped_unknown_packets": s["dropped_unknown"],
            "dropped_reported_samples": s["dropped_reported"],
            "segments": s["segments"],
            "gaps": s["gaps"],
        }

    return {
        "cache_version": CACHE_VERSION,
        "session": Path(session_dir).name,
        "packet_index_bytes": path.stat().st_size,
        "streams": out_streams,
    }


def get_index(session_dir: Path, rebuild: bool = False,
              verbose: bool = True) -> dict:
    """Cached build_index()."""
    session_dir = Path(session_dir)
    cache = _cache_path(session_dir)
    if cache.exists() and not rebuild:
        try:
            with open(cache) as f:
                idx = json.load(f)
            if idx.get("cache_version") == CACHE_VERSION:
                return idx
        except (json.JSONDecodeError, OSError):
            pass
    idx = build_index(session_dir, verbose=verbose)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(idx, f, indent=2)
    if verbose:
        print(f"  cached -> {cache.relative_to(C.REPO_ROOT)}")
    return idx


def measured_odr(session_dir: Path, **kw) -> float:
    """Measured vibration ODR in Hz. Falls back to nominal with a warning."""
    idx = get_index(session_dir, **kw)
    s = idx["streams"].get(str(C.PKT_VIB), {})
    rate = s.get("measured_rate_hz")
    if not rate:
        print(f"[warn] no measured ODR for {Path(session_dir).name}; using "
              f"nominal {C.NOMINAL_ODR_HZ} Hz. Do not make FFT frequency "
              f"claims on this session.")
        return C.NOMINAL_ODR_HZ
    return float(rate)


def contiguous_segments(session_dir: Path, stream: int = C.PKT_VIB,
                        min_samples: int | None = None, **kw
                        ) -> list[tuple[int, int]]:
    """Half-open [start, end) sample ranges of gap-free data in the stream file.

    Segments shorter than `min_samples` are dropped -- they cannot host a full
    analysis window and are not worth special-casing.
    """
    idx = get_index(session_dir, **kw)
    segs = idx["streams"][str(stream)]["segments"]
    if min_samples:
        segs = [s for s in segs if (s[1] - s[0]) >= min_samples]
    return [(int(a), int(b)) for a, b in segs]


# ---------------------------------------------------------------------------
# status.csv (1 Hz thermal / fault / counter log)
# ---------------------------------------------------------------------------
def load_status(session_dir: Path) -> dict:
    """Returns dict of lists: host_time, obj_c, amb_c, delta_c, running, fault.

    `delta_c` (obj - amb) is the quantity to use for any thermal feature.
    Absolute obj_c is confounded by overnight room cooling -- see
    docs/notes/thermal_channel_confound.md.
    """
    path = Path(session_dir) / C.STATUS_CSV
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    cols: dict[str, list] = {k: [] for k in
                             ("host_time", "obj_c", "amb_c", "delta_c",
                              "running", "fault")}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                o = float(r["obj_temp_c"])
                a = float(r["amb_temp_c"])
            except (ValueError, KeyError):
                continue
            cols["host_time"].append(float(r["host_time"]))
            cols["obj_c"].append(o)
            cols["amb_c"].append(a)
            cols["delta_c"].append(o - a)
            cols["running"].append(int(r["motor_running"]))
            cols["fault"].append(r["fault"])
    return cols


def motor_running_window(session_dir: Path) -> tuple[float, float] | None:
    """(first, last) host_time at which the motor was reported running."""
    st = load_status(session_dir)
    times = [t for t, r in zip(st["host_time"], st["running"]) if r == 1]
    return (times[0], times[-1]) if times else None


# ---------------------------------------------------------------------------
# CLI summary
# ---------------------------------------------------------------------------
def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}"
        n /= 1024.0
    return str(n)


def summarise(session_dir: Path, rebuild: bool = False) -> None:
    session_dir = Path(session_dir)
    sess = load_session_json(session_dir)
    print(f"\n=== {session_dir.name}")
    print(f"  label={sess.get('session')}  elapsed={sess.get('elapsed_s')} s")

    try:
        fs = accel_fs_g(sess)
        print(f"  accel full scale: +/-{fs:g} g  "
              f"({1000.0 / lsb_per_g(fs):.3f} mg/LSB)"
              f"{'  [legacy fallback]' if 'accel_fs_g' not in sess else ''}")
    except ValueError as e:
        print(f"  [error] {e}")

    idx = get_index(session_dir, rebuild=rebuild)
    names = {"0": "vibration", "1": "audio", "2": "status"}
    nominal = {"0": C.NOMINAL_ODR_HZ, "1": float(C.AUDIO_SR_HZ), "2": 1.0}
    for ty, s in idx["streams"].items():
        print(f"  -- stream {ty} ({names.get(ty, '?')})")
        print(f"     packets={s['packets']:,}  samples={s['samples']:,}")
        if s["measured_rate_hz"]:
            nom = nominal.get(ty)
            dev = (f"  ({s['measured_rate_hz'] / nom - 1:+.3%} vs nominal "
                   f"{nom:,.0f} Hz)" if nom else "")
            print(f"     measured rate = {s['measured_rate_hz']:,.3f} Hz{dev}")
        print(f"     transport capture ratio = "
              f"{s['transport_capture_ratio']}  gaps={s['gap_events']:,}  "
              f"missing packets={s['missing_packets']:,} "
              f"({s['missing_samples']:,} samples)")
        if s["resets"]:
            print(f"     [!] MCU sequence RESET x{len(s['resets'])}:")
            for r in s["resets"][:4]:
                at_s = (r["at_sample"] / s["measured_rate_hz"]
                        if s["measured_rate_hz"] else 0)
                print(f"         seq {r['seq_from']:,} -> {r['seq_to']:,} "
                      f"at sample {r['at_sample']:,} (t+{at_s:.1f}s)")
        if ty == "0":
            segs = s["segments"]
            lens = [b - a for a, b in segs]
            odr = s["measured_rate_hz"] or C.NOMINAL_ODR_HZ
            usable = [L for L in lens if L >= C.MIN_SEGMENT_SECONDS * odr]
            print(f"     segments={len(segs):,}  "
                  f"longest={max(lens) / odr:,.1f} s  "
                  f"shortest={min(lens) / odr:.3f} s")
            print(f"     usable (>={C.MIN_SEGMENT_SECONDS:g} s): "
                  f"{len(usable):,} segments, "
                  f"{sum(usable) / odr / 3600:.3f} h "
                  f"({sum(usable) / max(1, sum(lens)):.2%} of samples)")
            if s["fifo_overrun_packets"]:
                print(f"     [!] on-device FIFO overrun packets: "
                      f"{s['fifo_overrun_packets']}")

    vib = session_dir / C.VIB_BIN_NAME
    if vib.exists():
        n = vib.stat().st_size // C.VIB_SAMPLE_BYTES
        idx_n = idx["streams"]["0"]["samples"]
        ok = "OK" if n == idx_n else f"MISMATCH (index says {idx_n:,})"
        print(f"  vibration.bin = {_fmt_bytes(vib.stat().st_size)} "
              f"-> {n:,} samples  [{ok}]")

    try:
        st = load_status(session_dir)
        run = [d for d, r in zip(st["delta_c"], st["running"]) if r == 1]
        if run:
            k = max(1, len(run) // 60)
            print(f"  thermal obj-amb delta while running: "
                  f"start={sum(run[:k]) / k:+.2f} C  "
                  f"end={sum(run[-k:]) / k:+.2f} C  "
                  f"max={max(run):+.2f} C")
    except FileNotFoundError:
        pass


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sessions", nargs="*",
                    help="session dirs or bare names; default = all")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore cached index and re-parse packet_index.csv")
    args = ap.parse_args()

    targets = ([C.session_dir(s) for s in args.sessions]
               if args.sessions else C.all_sessions())
    if not targets:
        print(f"No sessions found under {C.RAW_DIR}")
        return
    for d in targets:
        summarise(d, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
