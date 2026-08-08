"""
Turns a recorded session into per-window feature vectors.

Three things distinguish this from naive windowing:

1. GAP AWARENESS. Windows are cut only from inside gap-free segments supplied
   by ml/sessions.py. vibration.bin is a splice of discontinuous runs -- a
   window straddling a splice contains a step discontinuity whose spectrum is a
   broadband transient, i.e. a fabricated impulsive fault. See the module
   docstring in ml/sessions.py.

2. STREAMING. The vibration file is memory-mapped and converted to engineering
   units one window at a time. Nothing ever holds the 3.2 GB session in RAM.

3. AXES COMBINED IN THE POWER DOMAIN. The three accelerometer axes are turned
   into one spectrum by averaging per-axis PSDs. Taking the rFFT of the
   Euclidean norm ||v|| instead -- the obvious shortcut -- is a nonlinear mix
   that generates sum and difference frequencies which are easy to mistake for
   genuine mechanical sidebands.

Windows recorded while the motor was not running (the 60 s arm delay, and any
later stop) are excluded, so the healthy baseline is genuinely "healthy motor
running" rather than "mostly room noise".

Run:
    python -m ml.features [session ...]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config as C
from . import data_loader as DL
from . import sessions as S


# ---------------------------------------------------------------------------
# Window planning
# ---------------------------------------------------------------------------
def plan_windows(segments: list[tuple[int, int]], win: int, hop: int,
                 skip_before: int = 0) -> list[int]:
    """Start offsets of every window that fits wholly inside a segment.

    Because segments are gap-free by construction, every returned window is
    guaranteed contiguous in time.
    """
    starts: list[int] = []
    for a, b in segments:
        a = max(a, skip_before)
        if b - a < win:
            continue
        starts.extend(range(a, b - win + 1, hop))
    return starts


def sample_to_host_time(idx: dict, sample: int, odr: float) -> float:
    """Approximate host arrival time of a vibration sample offset.

    Linear map from the first packet's host time at the measured ODR. Total
    missing samples are ~0.001 % of a session, so the accumulated error over
    six hours is well under a second -- ample for aligning against a 1 Hz
    thermal channel.
    """
    t0 = idx["streams"][str(C.PKT_VIB)]["first_host_time"] or 0.0
    return t0 + sample / odr


# ---------------------------------------------------------------------------
# Spectra
# ---------------------------------------------------------------------------
def _welch_setup(odr: float, nperseg: int):
    from scipy.signal import get_window
    win = get_window("hann", nperseg)
    # Power-spectral-density scaling, one-sided.
    scale = 1.0 / (odr * (win ** 2).sum())
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / odr)
    return win, scale, freqs


def _welch_psd(x: np.ndarray, win: np.ndarray, scale: float) -> np.ndarray:
    """One-sided PSD of a 1-D signal by Welch averaging with 50 % overlap."""
    nperseg = len(win)
    step = nperseg // 2
    n_seg = 1 + (len(x) - nperseg) // step
    if n_seg < 1:
        raise ValueError("window shorter than WELCH_NPERSEG")
    acc = None
    for i in range(n_seg):
        seg = x[i * step: i * step + nperseg]
        seg = seg - seg.mean()
        p = np.abs(np.fft.rfft(seg * win)) ** 2
        acc = p if acc is None else acc + p
    psd = (acc / n_seg) * (2.0 * scale)
    psd[0] /= 2.0
    if nperseg % 2 == 0:
        psd[-1] /= 2.0
    return psd


def window_spectrum(block_g: np.ndarray, win: np.ndarray, scale: float,
                    keep: int, combine: str) -> np.ndarray:
    """(n,3) block in g -> feature vector in dB."""
    if combine == "magnitude":
        mag = np.linalg.norm(block_g, axis=1)
        psd = _welch_psd(mag, win, scale)[:keep]
        return (10.0 * np.log10(psd + 1e-20)).astype(np.float32)

    psds = [_welch_psd(block_g[:, ax], win, scale)[:keep] for ax in range(3)]
    if combine == "per_axis":
        return np.concatenate(
            [10.0 * np.log10(p + 1e-20) for p in psds]).astype(np.float32)
    # "power_mean" (default): linear average in the power domain, then dB.
    mean_psd = (psds[0] + psds[1] + psds[2]) / 3.0
    return (10.0 * np.log10(mean_psd + 1e-20)).astype(np.float32)


# ---------------------------------------------------------------------------
# Per-session extraction
# ---------------------------------------------------------------------------
def extract_session(session_dir: Path, verbose: bool = True) -> dict:
    session_dir = Path(session_dir)
    info = DL.vibration_info(session_dir)
    odr = info["odr_hz"]
    fs_g = info["accel_fs_g"]
    idx = S.get_index(session_dir, verbose=False)

    win_n = int(round(C.WINDOW_SECONDS * odr))
    hop_n = max(1, int(round(win_n * (1.0 - C.WINDOW_OVERLAP))))
    if win_n < C.WELCH_NPERSEG:
        raise ValueError(
            f"WINDOW_SECONDS={C.WINDOW_SECONDS} at {odr:.0f} Hz gives "
            f"{win_n} samples, shorter than WELCH_NPERSEG="
            f"{C.WELCH_NPERSEG}")

    skip = int(C.ARM_DELAY_SKIP_SECONDS * odr)
    starts = plan_windows(info["segments"], win_n, hop_n, skip_before=skip)

    # Motor-running mask + thermal delta, both from the 1 Hz status log.
    status = S.load_status(session_dir)
    st_t = np.asarray(status["host_time"])
    st_run = np.asarray(status["running"])
    st_delta = np.asarray(status["delta_c"])

    w_time = np.asarray([sample_to_host_time(idx, s, odr) for s in starts])
    # Nearest 1 Hz status sample for each window.
    near = np.clip(np.searchsorted(st_t, w_time), 0, len(st_t) - 1)
    running = st_run[near] == 1
    delta_c = st_delta[near]

    keep_mask = running
    starts_kept = [s for s, k in zip(starts, keep_mask) if k]

    if verbose:
        print(f"  windows: {len(starts):,} planned in gap-free segments, "
              f"{len(starts_kept):,} with motor running "
              f"({len(starts) - len(starts_kept):,} dropped)")

    win, scale, freqs = _welch_setup(odr, C.WELCH_NPERSEG)
    keep = (len(freqs) if C.FFT_BINS_KEEP is None
            else min(C.FFT_BINS_KEEP, len(freqs)))

    vib = DL.open_vibration(session_dir)
    n_feat = keep * (3 if C.AXIS_COMBINE == "per_axis" else 1)
    feats = np.empty((len(starts_kept), n_feat), dtype=np.float32)

    for i, s in enumerate(starts_kept):
        block = DL.read_window_g(vib, s, win_n, fs_g)
        feats[i] = window_spectrum(block, win, scale, keep, C.AXIS_COMBINE)
        if verbose and (i + 1) % 5000 == 0:
            print(f"    {i + 1:,}/{len(starts_kept):,}")

    return {
        "session": session_dir.name,
        "features": feats,
        "freqs": freqs[:keep],
        "starts": np.asarray(starts_kept, dtype=np.int64),
        "host_time": w_time[keep_mask],
        "thermal_delta_c": delta_c[keep_mask].astype(np.float32),
        "odr_hz": odr,
        "accel_fs_g": fs_g,
        "window_samples": win_n,
        "hop_samples": hop_n,
        "axis_combine": C.AXIS_COMBINE,
        "welch_nperseg": C.WELCH_NPERSEG,
    }


def save_session_features(res: dict) -> Path:
    out_dir = C.FEATURES_DIR / res["session"]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "vib_psd_db.npy", res["features"])
    np.save(out_dir / "freqs_hz.npy", res["freqs"])
    np.save(out_dir / "window_start_sample.npy", res["starts"])
    np.save(out_dir / "window_host_time.npy", res["host_time"])
    np.save(out_dir / "thermal_delta_c.npy", res["thermal_delta_c"])
    meta = {k: v for k, v in res.items()
            if k not in ("features", "freqs", "starts", "host_time",
                         "thermal_delta_c")}
    meta["n_windows"] = int(res["features"].shape[0])
    meta["n_bins"] = int(res["features"].shape[1])
    meta["freq_resolution_hz"] = float(res["freqs"][1] - res["freqs"][0])
    meta["band_hz"] = [float(res["freqs"][0]), float(res["freqs"][-1])]
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def load_session_features(session_name: str) -> dict:
    d = C.FEATURES_DIR / session_name
    with open(d / "meta.json") as f:
        meta = json.load(f)
    return {
        "meta": meta,
        "features": np.load(d / "vib_psd_db.npy"),
        "freqs": np.load(d / "freqs_hz.npy"),
        "starts": np.load(d / "window_start_sample.npy"),
        "host_time": np.load(d / "window_host_time.npy"),
        "thermal_delta_c": np.load(d / "thermal_delta_c.npy"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _plot(res: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats, freqs = res["features"], res["freqs"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    for frac in (0.05, 0.5, 0.95):
        i = int(frac * (len(feats) - 1))
        axes[0].plot(freqs, feats[i], lw=0.7, label=f"window @ {frac:.0%}")
    med = np.median(feats, axis=0)
    axes[0].plot(freqs, med, lw=1.4, color="k", ls="--", label="median")
    axes[0].set_xlabel(f"frequency [Hz]  (measured ODR "
                       f"{res['odr_hz']:,.1f} Hz)")
    axes[0].set_ylabel("PSD [dB re g^2/Hz]")
    axes[0].set_title(f"{res['session']} - vibration PSD windows")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Zoom on the rotational signature and label detected peaks.
    lim = freqs <= 2000
    axes[1].plot(freqs[lim], med[lim], lw=1.0, color="tab:blue")
    band = (freqs > 50) & (freqs <= 2000)
    order = np.argsort(med[band])[::-1]
    fb = freqs[band]
    picked: list[float] = []
    for j in order:
        f = float(fb[j])
        if all(abs(f - p) > 25 for p in picked):
            picked.append(f)
        if len(picked) == 6:
            break
    for f in sorted(picked):
        axes[1].axvline(f, color="tab:red", ls=":", lw=0.8)
        axes[1].annotate(f"{f:.0f}", (f, med[lim].max()),
                         fontsize=7, rotation=90, va="top", color="tab:red")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("PSD [dB re g^2/Hz]")
    axes[1].set_title("Median PSD, 50-2000 Hz - rotational signature")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PLOTS_DIR / f"psd_{res['session']}.png"
    fig.savefig(out, dpi=120)
    print(f"  peaks 50-2000 Hz: "
          f"{', '.join(f'{f:.1f}' for f in sorted(picked))} Hz")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build feature windows.")
    ap.add_argument("sessions", nargs="*", help="default = all sessions")
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="stop after N windows (quick smoke test)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    targets = ([C.session_dir(s) for s in args.sessions]
               if args.sessions else C.all_sessions())
    if not targets:
        print(f"No sessions found under {C.RAW_DIR}")
        return

    for d in targets:
        print(f"=== {d.name}")
        res = extract_session(d)
        if args.limit_windows:
            res["features"] = res["features"][:args.limit_windows]
            res["starts"] = res["starts"][:args.limit_windows]
            res["host_time"] = res["host_time"][:args.limit_windows]
            res["thermal_delta_c"] = \
                res["thermal_delta_c"][:args.limit_windows]
        out = save_session_features(res)
        print(f"  features {res['features'].shape} -> "
              f"{out.relative_to(C.REPO_ROOT)}")
        print(f"  freq resolution {res['freqs'][1] - res['freqs'][0]:.3f} Hz, "
              f"band 0-{res['freqs'][-1]:,.0f} Hz")
        if not args.no_plot:
            p = _plot(res)
            print(f"  wrote {p.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
