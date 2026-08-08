"""
Compares the PSD of the pre-motor "room only" segment (start of the
recording) against a motor-running segment, to find out what's actually in
the background noise instead of guessing a notch frequency.

Run:
    python audio_room_noise_compare.py [session_dir] [--quiet-end SEC] [--run-start SEC]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import welch

from .. import config as C  # noqa: E402
from ..data_loader import load_audio  # noqa: E402

DEFAULT_SESSION = C.session_dir("healthy24V_motor_baseline_20260719_010905")
OUT_DIR = C.FIGURES_DIR / "audio_filtering" / "room_vs_running"

SEGMENT_SECONDS = 30.0
PLOT_MAX_HZ = 2000.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir", nargs="?", default=str(DEFAULT_SESSION))
    ap.add_argument("--quiet-end", type=float, default=55.0,
                    help="End of the pre-motor quiet window (s)")
    ap.add_argument("--run-start", type=float, default=120.0,
                    help="Start of the motor-running window (s)")
    args = ap.parse_args()

    session_dir = Path(args.session_dir)
    audio, sr = load_audio(session_dir)

    quiet_start_n = max(0, int((args.quiet_end - SEGMENT_SECONDS) * sr))
    quiet_end_n = int(args.quiet_end * sr)
    quiet = audio[quiet_start_n:quiet_end_n].astype(np.float64)

    run_start_n = int(args.run_start * sr)
    run_end_n = run_start_n + int(SEGMENT_SECONDS * sr)
    running = audio[run_start_n:run_end_n].astype(np.float64)

    print(f"Quiet segment:   {args.quiet_end - SEGMENT_SECONDS:.0f}s - {args.quiet_end:.0f}s "
          f"(rms={np.sqrt(np.mean(quiet**2)):.5f})")
    print(f"Running segment: {args.run_start:.0f}s - {args.run_start + SEGMENT_SECONDS:.0f}s "
          f"(rms={np.sqrt(np.mean(running**2)):.5f})")

    f_q, psd_q = welch(quiet, fs=sr, nperseg=8192)
    f_r, psd_r = welch(running, fs=sr, nperseg=8192)
    psd_q_db = 10 * np.log10(psd_q + 1e-12)
    psd_r_db = 10 * np.log10(psd_r + 1e-12)

    # Peaks present in the quiet segment (candidates for what's actually
    # "room hum" as opposed to guessed mains-hum band).
    mask = f_q <= PLOT_MAX_HZ
    order = np.argsort(psd_q_db[mask])[::-1]
    print("\nTop 10 peaks in the QUIET (pre-motor) segment, <=%.0f Hz:" % PLOT_MAX_HZ)
    for i in order[:10]:
        print(f"  {f_q[mask][i]:7.1f} Hz  {psd_q_db[mask][i]:6.1f} dB")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8))

    axes[0].plot(f_q[mask], psd_q_db[mask], lw=1.0, color="tab:green", label="room only (pre-motor)")
    axes[0].plot(f_r[mask], psd_r_db[mask], lw=1.0, color="tab:orange", alpha=0.8, label="motor running")
    axes[0].set_title("PSD overlay: room-only vs motor-running")
    axes[0].set_xlabel("frequency [Hz]")
    axes[0].set_ylabel("PSD [dB]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(f_q[mask], psd_r_db[mask] - psd_q_db[mask], lw=1.0, color="tab:blue")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("Difference (running - room): positive = motor added energy here")
    axes[1].set_xlabel("frequency [Hz]")
    axes[1].set_ylabel("delta PSD [dB]")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "05_room_vs_running_psd.png"
    fig.savefig(out, dpi=130)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
