"""
Four-way comparison to justify spectral subtraction as the audio cleaning
method: no filter, notch-only, high-pass-only, spectral subtraction.

Background (see audio_room_noise_compare.py): the broadband noise floor is
identical whether the motor is running or not (it's the room/mic noise
floor), while the ~62 Hz harmonic series (62, 124, 186, ... Hz) only appears
once the motor starts - i.e. it's motor signal, not noise. A notch filter
at 62 Hz or a high-pass therefore both cut into the motor's own signature.
Spectral subtraction instead removes exactly the measured broadband noise
floor (profiled from the pre-motor "room only" segment) and leaves the
harmonic content alone.

Run:
    python audio_filter_compare_4way.py [session_dir]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import (butter, iirnotch, istft, sosfiltfilt, spectrogram,
                          stft, tf2sos, welch)

from .. import config as C  # noqa: E402
from ..data_loader import load_audio  # noqa: E402

DEFAULT_SESSION = C.session_dir("healthy24V_motor_baseline_20260719_010905")
OUT_DIR = C.FIGURES_DIR / "audio_filtering" / "filter_compare_4way"

QUIET_WINDOW_S = (25.0, 55.0)     # pre-motor "room only" reference
SEGMENT_SECONDS = 5.0             # analyzed/plotted motor-running slice
SEGMENT_START_S = 120.0           # well after motor start
WAVEFORM_ZOOM_SECONDS = 0.1
PSD_ZOOM_HZ = 400.0

NOTCH_FREQ_HZ = 62.5
NOTCH_Q = 30.0
HIGHPASS_HZ = 150.0
HIGHPASS_ORDER = 4

STFT_NPERSEG = 1024
STFT_NOVERLAP = 768
SS_ALPHA = 1.0      # oversubtraction factor (1.0 = textbook Boll spectral subtraction)
SS_FLOOR = 0.05      # spectral floor (fraction of original magnitude)


def notch_sos(freq: float, sr: int) -> np.ndarray:
    b, a = iirnotch(freq, NOTCH_Q, fs=sr)
    return tf2sos(b, a)


def highpass_sos(cutoff: float, sr: int) -> np.ndarray:
    return butter(HIGHPASS_ORDER, cutoff, btype="highpass", fs=sr, output="sos")


def spectral_subtract(seg: np.ndarray, noise_profile: np.ndarray, sr: int) -> np.ndarray:
    f, t, Zxx = stft(seg, fs=sr, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)
    mag, phase = np.abs(Zxx), np.angle(Zxx)
    noise = noise_profile[:, None]  # broadcast over time frames
    mag_clean = np.maximum(mag - SS_ALPHA * noise, SS_FLOOR * mag)
    Zxx_clean = mag_clean * np.exp(1j * phase)
    _, cleaned = istft(Zxx_clean, fs=sr, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)
    return cleaned[: len(seg)]


def build_noise_profile(quiet: np.ndarray, sr: int) -> np.ndarray:
    f, t, Zxx = stft(quiet, fs=sr, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP)
    return np.median(np.abs(Zxx), axis=1)  # robust per-bin noise magnitude


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
    axes[1].set_title(f"{title} - low-frequency PSD (0-{PSD_ZOOM_HZ:.0f} Hz)")
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


def plot_summary(freqs: np.ndarray, psds_db: dict, quiet_db: np.ndarray,
                 out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = freqs <= PSD_ZOOM_HZ
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(freqs[mask], quiet_db[mask], lw=1.4, color="black",
           linestyle="--", label="room only (target floor)")
    colors = {"no_filter": "tab:gray", "notch": "tab:red",
             "highpass": "tab:orange", "spectral_subtraction": "tab:green"}
    labels = {"no_filter": "no filter", "notch": "notch (62.5 Hz)",
             "highpass": "high-pass (150 Hz)",
             "spectral_subtraction": "spectral subtraction"}
    for key, psd_db in psds_db.items():
        ax.plot(freqs[mask], psd_db[mask], lw=1.0, color=colors[key],
               label=labels[key], alpha=0.85)
    ax.set_title("Low-frequency PSD vs room-only floor - which method actually matches the target?")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [dB]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", default=str(DEFAULT_SESSION))
    ap.add_argument("--segment-start", type=float, default=SEGMENT_START_S,
                    help="Start of the analyzed/plotted segment (s)")
    ap.add_argument("--tag", default="",
                    help="Filename prefix to avoid overwriting a prior run's outputs")
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    audio, sr = load_audio(session_dir)
    print(f"Loaded {session_dir.name}: {len(audio):,} samples @ {sr} Hz")

    q0, q1 = QUIET_WINDOW_S
    quiet = audio[int(q0 * sr): int(q1 * sr)].astype(np.float64)
    noise_profile = build_noise_profile(quiet, sr)

    n_seg = int(SEGMENT_SECONDS * sr)
    s0 = int(args.segment_start * sr)
    seg = audio[s0: s0 + n_seg].astype(np.float64)
    prefix = f"{args.tag}_" if args.tag else ""

    notch = notch_sos(NOTCH_FREQ_HZ, sr)
    hp = highpass_sos(HIGHPASS_HZ, sr)

    variants = {
        "no_filter": seg,
        "notch": sosfiltfilt(notch, seg),
        "highpass": sosfiltfilt(hp, seg),
        "spectral_subtraction": spectral_subtract(seg, noise_profile, sr),
    }
    titles = {
        "no_filter": "No filter",
        "notch": f"Notch only ({NOTCH_FREQ_HZ:.0f} Hz)",
        "highpass": f"High-pass only ({HIGHPASS_HZ:.0f} Hz)",
        "spectral_subtraction": "Spectral subtraction (room-profile-based)",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    _, _, sxx_ref = spectrogram(seg, fs=sr, nperseg=1024, noverlap=768)
    spec_vmax = float((10 * np.log10(sxx_ref + 1e-12)).max())

    psds_db = {}
    for i, (key, sig) in enumerate(variants.items(), start=1):
        plot_one(sig, sr, titles[key], OUT_DIR / f"{prefix}{i:02d}_{key}.png", spec_vmax)
        f_v, psd_v = welch(sig, fs=sr, nperseg=8192)
        psds_db[key] = 10 * np.log10(psd_v + 1e-12)

    f_q, psd_q = welch(quiet, fs=sr, nperseg=8192)
    quiet_db = 10 * np.log10(psd_q + 1e-12)
    plot_summary(f_q, psds_db, quiet_db, OUT_DIR / f"{prefix}05_summary_vs_room_floor.png")


if __name__ == "__main__":
    main()
