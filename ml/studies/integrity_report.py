"""
Dataset integrity report across every recorded session.

Produces the table the thesis needs to back the claim that "sample loss is
detectable at every stage of the pipeline, so dataset completeness is measured,
not assumed" (docs/notes/overnight_acquisition_rig.md, §2).

Why this exists rather than just reading session.json
-----------------------------------------------------
session.json reports `vib_capture_ratio = samples / (elapsed * 26667)`, i.e. it
divides by the NOMINAL datasheet ODR. That denominator is wrong by however much
the sensor's real output data rate differs from nominal, so the figure conflates
two completely different things:

  * genuine transport loss (packets that never arrived), and
  * sensor clock error (the IIS3DWB not running at exactly 26.667 kHz).

On the pilot sessions that made a clean rig look like it was dropping 0.5 % of
its data. Measuring the ODR from sequence numbers and host timestamps separates
the two: real transport loss is ~0.001 %, and the IIS3DWB runs ~0.45 % slow.

Run:
    python -m ml.studies.integrity_report
    python -m ml.studies.integrity_report --markdown
"""

from __future__ import annotations

import argparse

from .. import config as C
from .. import sessions as S


def collect(session_dir):
    meta = S.load_session_json(session_dir)
    idx = S.get_index(session_dir, verbose=False)
    vib = idx["streams"][str(C.PKT_VIB)]
    aud = idx["streams"].get(str(C.PKT_AUDIO), {})

    segs = vib["segments"]
    odr = vib["measured_rate_hz"] or C.NOMINAL_ODR_HZ
    min_n = C.MIN_SEGMENT_SECONDS * odr
    usable = sum(b - a for a, b in segs if (b - a) >= min_n)

    try:
        fs_g = S.accel_fs_g(meta)
    except ValueError:
        fs_g = float("nan")

    st = S.load_status(session_dir)
    run_delta = [d for d, r in zip(st["delta_c"], st["running"]) if r == 1]
    k = max(1, len(run_delta) // 60)

    return {
        "session": session_dir.name,
        "hours": vib["samples"] / odr / 3600.0,
        "fs_g": fs_g,
        "odr_measured": odr,
        "odr_dev_pct": (odr / C.NOMINAL_ODR_HZ - 1) * 100.0,
        "audio_sr": aud.get("measured_rate_hz") or float("nan"),
        "audio_dev_pct": ((aud.get("measured_rate_hz") or float("nan"))
                          / C.AUDIO_SR_HZ - 1) * 100.0,
        "json_ratio": meta.get("vib_capture_ratio"),
        "transport_ratio": vib["transport_capture_ratio"],
        "loss_ppm": (1.0 - (vib["transport_capture_ratio"] or 1.0)) * 1e6,
        "gaps": vib["gap_events"],
        "missing_pkts": vib["missing_packets"],
        "resets": len(vib["resets"]),
        "crc_errors": meta.get("crc_errors"),
        "fifo_ovr": vib["fifo_overrun_packets"],
        "dropped_unknown": vib["dropped_unknown_packets"],
        "segments": len(segs),
        "usable_h": usable / odr / 3600.0,
        "usable_pct": usable / max(1, vib["samples"]) * 100.0,
        "delta_start": (sum(run_delta[:k]) / k) if run_delta else float("nan"),
        "delta_end": (sum(run_delta[-k:]) / k) if run_delta else float("nan"),
    }


ROWS = [
    ("duration [h]",                 "hours",           "{:.3f}"),
    ("accel full scale [g]",         "fs_g",            "+/-{:g}"),
    ("measured ODR [Hz]",            "odr_measured",    "{:,.1f}"),
    ("  vs nominal 26667 Hz",        "odr_dev_pct",     "{:+.3f} %"),
    ("measured audio SR [Hz]",       "audio_sr",        "{:,.1f}"),
    ("  vs nominal 16000 Hz",        "audio_dev_pct",   "{:+.3f} %"),
    ("session.json capture ratio",   "json_ratio",      "{:.4f}"),
    ("transport capture ratio",      "transport_ratio", "{:.8f}"),
    ("  transport loss [ppm]",       "loss_ppm",        "{:,.1f}"),
    ("gap events",                   "gaps",            "{:,}"),
    ("missing packets",              "missing_pkts",    "{:,}"),
    ("MCU resets",                   "resets",          "{:,}"),
    ("CRC errors",                   "crc_errors",      "{:,}"),
    ("on-device FIFO overruns",      "fifo_ovr",        "{:,}"),
    ("pkts w/ dropped=unknown",      "dropped_unknown", "{:,}"),
    ("gap-free segments",            "segments",        "{:,}"),
    ("usable contiguous [h]",        "usable_h",        "{:.3f}"),
    ("  of total samples",           "usable_pct",      "{:.2f} %"),
    ("thermal obj-amb start [C]",    "delta_start",     "{:+.2f}"),
    ("thermal obj-amb end [C]",      "delta_end",       "{:+.2f}"),
]


def fmt(val, spec):
    if val is None:
        return "-"
    try:
        return spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--markdown", action="store_true",
                    help="emit a markdown table for pasting into the thesis")
    args = ap.parse_args()

    dirs = C.all_sessions()
    if not dirs:
        raise SystemExit(f"No sessions found under {C.RAW_DIR}")
    data = [collect(d) for d in dirs]

    short = [d["session"].replace("_motor_baseline_", " ") for d in data]
    w = max(28, *(len(s) + 2 for s in short))

    if args.markdown:
        print("| metric | " + " | ".join(short) + " |")
        print("|---|" + "---|" * len(short))
        for label, key, spec in ROWS:
            print(f"| {label} | "
                  + " | ".join(fmt(d[key], spec) for d in data) + " |")
    else:
        print("DATASET INTEGRITY REPORT")
        print("=" * (30 + w * len(short)))
        print(f"{'metric':30s}" + "".join(f"{s:>{w}s}" for s in short))
        print("-" * (30 + w * len(short)))
        for label, key, spec in ROWS:
            print(f"{label:30s}"
                  + "".join(f"{fmt(d[key], spec):>{w}s}" for d in data))

    print()
    print("Notes for the thesis:")
    print("  * 'transport capture ratio' is samples received / samples the MCU")
    print("    actually sent, derived from packet sequence numbers. This is the")
    print("    real integrity figure.")
    print("  * 'session.json capture ratio' divides by the NOMINAL ODR and so")
    print("    understates completeness by the sensor's clock error. Quote the")
    print("    transport ratio instead.")
    print("  * every session contains one MCU reset in the first ~3 s: opening")
    print("    the serial port resets the board. Benign, but it is a hard")
    print("    discontinuity and is excluded by the segment map.")
    print("  * on-device FIFO overruns lose samples INSIDE the sensor while the")
    print("    packet sequence stays contiguous, so they are invisible to")
    print("    sequence-gap detection. They break segments too. Each carries")
    print("    dropped=0xFFFF because the sensor cannot report how many samples")
    print("    it discarded (bounded by the 3 KB FIFO, ~512 samples).")
    print("  * 'usable contiguous' is what survives gap-aware windowing and is")
    print("    the honest denominator for how much data the model trained on.")


if __name__ == "__main__":
    main()
