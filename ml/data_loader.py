"""
Loads one recorded session.

Everything here is MEMORY-MAPPED. A 6-hour session is 3.2 GB of int16
vibration; the previous implementation read the whole file and converted it to
float32 (N,3), which needs ~3.2 GB for the bytes plus ~6.9 GB for the float
array plus ~2.3 GB for the magnitude vector -- it cannot run on a normal
machine. Conversion to engineering units happens per window instead, at the
point of use.

Scaling is per session: the accelerometer full scale is read from
session.json ("accel_fs_g"), falling back to the legacy +/-16 g for the pilot
sessions that predate that field. Never hardcode it.

Run directly for a sanity check:
    python -m ml.data_loader [session]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config as C
from . import sessions as S


# ---------------------------------------------------------------------------
# Vibration
# ---------------------------------------------------------------------------
def open_vibration(session_dir: Path) -> np.memmap:
    """Memory-map vibration.bin as int16 (N, 3) raw counts. No copy, no scaling."""
    path = Path(session_dir) / C.VIB_BIN_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    n = path.stat().st_size // C.VIB_SAMPLE_BYTES
    return np.memmap(path, dtype="<i2", mode="r", shape=(n, 3))


def counts_to_g(block: np.ndarray, fs_g: float) -> np.ndarray:
    """int16 counts -> float32 g, for one small block."""
    return block.astype(np.float32) * (fs_g / 32768.0)


def read_window_g(vib: np.memmap, start: int, n: int, fs_g: float
                  ) -> np.ndarray:
    """Read n samples starting at `start`, converted to g. float32 (n, 3)."""
    return counts_to_g(np.asarray(vib[start:start + n]), fs_g)


def vibration_info(session_dir: Path) -> dict:
    """Everything needed to interpret vibration.bin, in one dict."""
    session_dir = Path(session_dir)
    meta = S.load_session_json(session_dir)
    fs_g = S.accel_fs_g(meta)
    odr = S.measured_odr(session_dir)
    vib = open_vibration(session_dir)
    segs = S.contiguous_segments(
        session_dir, min_samples=int(C.MIN_SEGMENT_SECONDS * odr))
    return {
        "session_dir": session_dir,
        "meta": meta,
        "n_samples": len(vib),
        "odr_hz": odr,
        "accel_fs_g": fs_g,
        "lsb_per_g": S.lsb_per_g(fs_g),
        "duration_s": len(vib) / odr,
        "segments": segs,
        "usable_s": sum(b - a for a, b in segs) / odr,
    }


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
def open_audio(session_dir: Path) -> tuple[np.ndarray, int]:
    """Memory-map audio.wav. Returns (int16 samples, sample_rate)."""
    from scipy.io import wavfile
    path = Path(session_dir) / C.AUDIO_WAV_NAME
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    sr, audio = wavfile.read(path, mmap=True)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sr


def audio_to_float(block: np.ndarray) -> np.ndarray:
    """int16 block -> float32 in [-1, 1)."""
    if block.dtype == np.int16:
        return block.astype(np.float32) / 32768.0
    if block.dtype == np.int32:
        return block.astype(np.float32) / 2147483648.0
    return block.astype(np.float32)


def load_audio(session_dir: Path) -> tuple[np.ndarray, int]:
    """Whole audio track as float32. ~1.4 GB for a 6 h session -- prefer
    open_audio() plus audio_to_float() on slices where possible.

    Kept because the exploratory scripts in ml/studies/ expect this signature.
    """
    audio, sr = open_audio(session_dir)
    return audio_to_float(np.asarray(audio)), sr


def audio_quiet_reference(session_dir: Path, seconds: float = 55.0
                          ) -> tuple[np.ndarray, int]:
    """The motor-off window at the start of a session, as a noise profile.

    data_catcher.py waits ARM_DELAY_S (60 s) before arming the motor, so the
    head of every recording is room-only. This is the measured noise reference
    that the spectral-subtraction decision rests on -- see
    docs/notes/acoustic_noise_filtering.md.
    """
    audio, sr = open_audio(session_dir)
    n = int(seconds * sr)
    return audio_to_float(np.asarray(audio[:n])), sr


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Session sanity check.")
    ap.add_argument("session", nargs="?", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    session_dir = C.session_dir(args.session)
    info = vibration_info(session_dir)

    print(f"=== {session_dir.name}")
    print(f"Measured ODR      : {info['odr_hz']:,.3f} Hz "
          f"({info['odr_hz'] / C.NOMINAL_ODR_HZ - 1:+.3%} vs nominal)")
    print(f"Accel full scale  : +/-{info['accel_fs_g']:g} g "
          f"({1000 / info['lsb_per_g']:.3f} mg/LSB)")
    print(f"Vibration         : {info['n_samples']:,} samples x 3 axes "
          f"({info['duration_s'] / 3600:.3f} h)")
    print(f"Gap-free segments : {len(info['segments']):,} "
          f"({info['usable_s'] / 3600:.3f} h usable, "
          f"{info['usable_s'] / info['duration_s']:.2%})")

    vib = open_vibration(session_dir)
    fs_g = info["accel_fs_g"]
    odr = info["odr_hz"]

    # Compare the motor-off head of the session against a mid-session window,
    # both taken from inside a single gap-free segment.
    n = int(odr)
    quiet = read_window_g(vib, int(20 * odr), n, fs_g)
    a, b = max(info["segments"], key=lambda s: s[1] - s[0])
    mid = (a + b) // 2
    run = read_window_g(vib, mid, n, fs_g)

    for name, blk in (("motor off (t=20s)", quiet), ("running (mid)", run)):
        ac = blk - blk.mean(axis=0)
        rms = float(np.sqrt((ac ** 2).sum(axis=1).mean()))
        pk = float(np.abs(ac).max())
        print(f"  {name:20s} rms={rms:.5f} g  peak={pk:.4f} g  "
              f"({pk / fs_g:.2%} of full scale)")

    audio, sr = open_audio(session_dir)
    print(f"Audio             : {len(audio):,} samples at {sr} Hz "
          f"({len(audio) / sr / 3600:.3f} h)")

    dv, da = info["duration_s"], len(audio) / sr
    if abs(dv - da) > 60:
        print(f"[warn] vibration and audio durations differ by {abs(dv-da):.0f} s")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    t = np.arange(n) / odr
    for i, ax_name in enumerate("XYZ"):
        axes[0].plot(t, run[:, i], lw=0.5, label=ax_name)
    axes[0].set_title(f"Vibration - 1 s mid-segment ({session_dir.name})")
    axes[0].set_ylabel("accel [g]")
    axes[0].legend(loc="upper right")

    aud = audio_to_float(np.asarray(audio[len(audio) // 2:
                                         len(audio) // 2 + sr]))
    axes[1].plot(np.arange(len(aud)) / sr, aud, lw=0.5, color="tab:orange")
    axes[1].set_title("Audio - 1 s from session midpoint")
    axes[1].set_ylabel("amplitude")
    axes[1].set_xlabel("time [s]")

    fig.tight_layout()
    out = C.PLOTS_DIR / f"sanity_{session_dir.name}.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
