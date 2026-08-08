"""
Compares raw audio against notch-only, high-pass-only, and notch+high-pass
filtered versions on a representative slice of the most recent session.

Notch frequency is auto-detected as the strongest peak in the 40-70 Hz band
(mains hum / electrical noise). High-pass cutoff defaults to 150 Hz to knock
down low-frequency rumble/handling noise while keeping motor-fault-relevant
content above it.

Run:
    python audio_filter_demo.py [session_dir] [--notch HZ] [--highpass HZ]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, spectrogram, welch, tf2sos

PSD_ZOOM_HZ = 400.0  # upper bound of the low-frequency PSD panel

from .. import config as C  # noqa: E402
from ..data_loader import load_audio  # noqa: E402

DEFAULT_SESSION = C.session_dir("healthy24V_motor_baseline_20260719_010905")
OUT_DIR = C.FIGURES_DIR / "audio_filtering" / "filter_demo"

SEGMENT_SECONDS = 5.0      # slice analyzed/plotted
WAVEFORM_ZOOM_SECONDS = 0.1  # zoomed-in waveform window
HUM_SEARCH_BAND = (40.0, 70.0)
HIGHPASS_DEFAULT_HZ = 150.0
NOTCH_Q = 30.0
HIGHPASS_ORDER = 4


def detect_hum_freq(seg: np.ndarray, sr: int, band: tuple[float, float]) -> float:
    freqs, psd = welch(seg, fs=sr, nperseg=min(len(seg), 8192))
    mask = (freqs >= band[0]) & (freqs <= band[1])
    return float(freqs[mask][np.argmax(psd[mask])])


def notch_sos(freq: float, sr: int) -> np.ndarray:
    b, a = iirnotch(freq, NOTCH_Q, fs=sr)
    return tf2sos(b, a)


def highpass_sos(cutoff: float, sr: int) -> np.ndarray:
    return butter(HIGHPASS_ORDER, cutoff, btype="highpass", fs=sr, output="sos")


def apply(seg: np.ndarray, sos_list: list[np.ndarray]) -> np.ndarray:
    out = seg
    for sos in sos_list:
        out = sosfiltfilt(sos, out)
    return out


def plot_one(seg: np.ndarray, sr: int, title: str, out_path: Path,
             spec_vmax: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(11, 9))

    n_zoom = int(WAVEFORM_ZOOM_SECONDS * sr)
    mid = len(seg) // 2
    t = np.arange(n_zoom) / sr
    axes[0].plot(t, seg[mid: mid + n_zoom], lw=0.7, color="tab:blue")
    axes[0].set_title(f"{title} - waveform ({WAVEFORM_ZOOM_SECONDS * 1000:.0f} ms @ segment midpoint)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("amplitude")

    freqs, psd = welch(seg, fs=sr, nperseg=min(len(seg), 8192))
    psd_db = 10 * np.log10(psd + 1e-12)
    zoom_mask = freqs <= PSD_ZOOM_HZ
    axes[1].plot(freqs[zoom_mask], psd_db[zoom_mask], lw=1.0, color="tab:red")
    axes[1].set_title(f"{title} - low-frequency PSD (0-{PSD_ZOOM_HZ:.0f} Hz, notch/high-pass effects live here)")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("PSD [dB]")
    axes[1].grid(alpha=0.3)

    f, t_spec, sxx = spectrogram(seg, fs=sr, nperseg=1024, noverlap=768)
    sxx_db = 10 * np.log10(sxx + 1e-12)
    im = axes[2].pcolormesh(t_spec, f, sxx_db, shading="auto", cmap="magma",
                            vmin=-100, vmax=spec_vmax)
    axes[2].set_ylim(0, sr / 2)
    axes[2].set_title(f"{title} - spectrogram (full band)")
    axes[2].set_xlabel("time [s]")
    axes[2].set_ylabel("frequency [Hz]")
    fig.colorbar(im, ax=axes[2], label="dB")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", default=str(DEFAULT_SESSION))
    ap.add_argument("--notch", type=float, default=None,
                    help="Override auto-detected notch frequency (Hz)")
    ap.add_argument("--highpass", type=float, default=HIGHPASS_DEFAULT_HZ,
                    help="High-pass cutoff (Hz)")
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    audio, sr = load_audio(session_dir)
    print(f"Loaded {session_dir.name}: {len(audio):,} samples @ {sr} Hz "
          f"({len(audio) / sr / 3600:.2f} h)")

    n_seg = int(SEGMENT_SECONDS * sr)
    mid = len(audio) // 2
    seg = audio[mid: mid + n_seg].astype(np.float64)

    notch_freq = args.notch if args.notch is not None else detect_hum_freq(
        seg, sr, HUM_SEARCH_BAND)
    print(f"Notch frequency: {notch_freq:.1f} Hz (Q={NOTCH_Q})")
    print(f"High-pass cutoff: {args.highpass:.1f} Hz (order {HIGHPASS_ORDER})")

    notch = notch_sos(notch_freq, sr)
    hp = highpass_sos(args.highpass, sr)

    variants = {
        "01_no_filter": seg,
        "02_notch_only": apply(seg, [notch]),
        "03_highpass_only": apply(seg, [hp]),
        "04_both_filters": apply(seg, [notch, hp]),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    titles = {
        "01_no_filter": "No filter",
        "02_notch_only": f"Notch only ({notch_freq:.0f} Hz)",
        "03_highpass_only": f"High-pass only ({args.highpass:.0f} Hz)",
        "04_both_filters": f"Notch ({notch_freq:.0f} Hz) + high-pass ({args.highpass:.0f} Hz)",
    }
    # Shared color scale (derived from the unfiltered signal) so the four
    # spectrograms are visually comparable rather than each auto-normalized.
    _, _, sxx_ref = spectrogram(seg, fs=sr, nperseg=1024, noverlap=768)
    spec_vmax = float((10 * np.log10(sxx_ref + 1e-12)).max())

    for key, sig in variants.items():
        plot_one(sig, sr, titles[key], OUT_DIR / f"{key}.png", spec_vmax)


if __name__ == "__main__":
    main()
