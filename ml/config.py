"""
Central configuration for the E2C-PDM machine learning pipeline.

All paths resolve relative to the REPOSITORY ROOT, not the current working
directory, so every module behaves identically no matter where it is invoked
from.

Run modules as package entry points from the repo root:
    python -m ml.data_loader
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (repo-root anchored)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = REPO_ROOT / "data"
RAW_DIR = DATA_ROOT / "raw"           # recorded sessions, untracked
INTERIM_DIR = DATA_ROOT / "interim"   # gap maps, ODR estimates, caches
PROCESSED_DIR = DATA_ROOT / "processed"

FEATURES_DIR = INTERIM_DIR / "features"   # cached .npy feature windows
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"   # models, scalers, tflite
FIGURES_DIR = REPO_ROOT / "docs" / "figures"   # tracked; thesis figures live here
PLOTS_DIR = FIGURES_DIR / "pipeline"           # generated diagnostic plots

# Default session used when a module is run without an explicit path.
DEFAULT_SESSION = "healthy24V_motor_baseline_20260719_010905"

VIB_BIN_NAME = "vibration.bin"
AUDIO_WAV_NAME = "audio.wav"
SESSION_JSON = "session.json"
STATUS_CSV = "status.csv"
PACKET_INDEX_CSV = "packet_index.csv"


def session_dir(name: str | None = None) -> Path:
    """Resolve a session by bare name, or accept an already-complete path."""
    name = name or DEFAULT_SESSION
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    direct = RAW_DIR / name
    if direct.exists() or not RAW_DIR.exists():
        return direct
    # Sessions may be filed one level down in a grouping folder. data/raw/24V/
    # holds the sessions from the motor retired on 2026-08-09. Bare names keep
    # working so nothing that stored a session name has to be rewritten.
    for group in RAW_DIR.iterdir():
        if group.is_dir() and (group / name).exists():
            return group / name
    return direct               # unchanged: report the path we expected


def all_sessions() -> list[Path]:
    """Every recorded session directory under data/raw, sorted by name.

    Looks one level down as well, so sessions grouped into a folder still count.
    A session is a directory holding session.json; a grouping folder is not, so
    the two can never be confused. Sorted by directory NAME, not by path, so
    grouping a session does not move it in the ordering.
    """
    if not RAW_DIR.exists():
        return []
    found = []
    for d in RAW_DIR.iterdir():
        if not d.is_dir():
            continue
        if (d / SESSION_JSON).exists():
            found.append(d)
        else:
            found.extend(s for s in d.iterdir()
                         if s.is_dir() and (s / SESSION_JSON).exists())
    return sorted(found, key=lambda p: p.name)


# ---------------------------------------------------------------------------
# Wire format  (documented here; NOT used to parse vibration.bin)
# ---------------------------------------------------------------------------
# data_catcher.py de-frames the USB stream before writing to disk: it validates
# the header and CRC in memory, then writes ONLY the payload bytes. So the
# on-disk vibration.bin is a bare little-endian int16 x,y,z stream with no
# framing whatsoever -- see data_catcher.py handle_packet(), which calls
# vib_file.write(payload).
#
# The wire header, for reference only (data_catcher.HDR_FMT = '<HBBIHH'):
#
#   magic u16 (0xA55A) | type u8 | flags u8 | seq u32 | payloadLen u16 |
#   dropped u16 | payload[payloadLen] | crc16-ccitt u16
#
#   type 0 = VIB   (payload = 64 x int16 x,y,z  -> 384 B)
#   type 1 = AUDIO (payload = 192 x int16       -> 384 B)
#   type 2 = STATUS
#
# CONSEQUENCE THAT MATTERS: because only payloads are written, lost packets are
# simply ABSENT from vibration.bin with no marker. The file is a splice of
# discontinuous segments. Window extraction must therefore consult the gap map
# derived from packet_index.csv -- see ml/sessions.py. Windowing naively across
# a splice fabricates a broadband transient that looks exactly like an
# impulsive mechanical fault.

VIB_SAMPLES_PER_PACKET = 64
VIB_SAMPLE_BYTES = 6              # int16 x, y, z
AUDIO_SAMPLES_PER_PACKET = 192

PKT_VIB, PKT_AUDIO, PKT_STATUS = 0, 1, 2

# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
# Vibration: STMicroelectronics IIS3DWB, fixed 26.7 kHz ODR, SPI @ 8 MHz,
# 3 KB hardware FIFO in continuous mode. Datasheet nominal ODR is 26667 Hz.
# The MEASURED rate is estimated per session from packet_index.csv timestamps
# (ml/sessions.py: measured_odr) and that is what the FFT axis uses. The
# nominal value below is only a fallback.
NOMINAL_ODR_HZ = 26667.0

# Accelerometer full scale. Firmware register CTRL1_XL selects this.
#   +/-4 g  -> 0.122 mg/LSB   (in effect on hardware since 2026-08-04)
#   +/-16 g -> 0.488 mg/LSB   (everything recorded before that date)
# Never hardcode this in analysis. Sessions carry "accel_fs_g" in session.json;
# loaders fall back to the legacy value below only for the pilot sessions that
# predate the field.
#
# The date is 08-04, not 07-30. The firmware constant was edited on 07-30 but
# the board was not reflashed until 08-04, so the four sessions of 08-01..02
# were recorded at +/-16 g while declaring +/-4 g. Their session.json files have
# been corrected in place and carry accel_fs_g_source="corrected_post_hoc".
# See docs/notes/quirk_register.md §2.1.
ACCEL_FS_G_DEFAULT = 4.0
ACCEL_FS_G_LEGACY = 16.0
# Sessions started before this timestamp were recorded at +/-16 g.
ACCEL_FS_CHANGED_AFTER = "20260804_000000"

G_TO_MS2 = 9.80665

# Audio: InvenSense INMP441, 16 kHz, 16-bit mono (top bits of the 24-bit I2S
# word).
AUDIO_SR_HZ = 16000

# Thermal: Melexis MLX90614 (GY-906) at 1 Hz.
#
# USE THE obj-amb DELTA, NOT ABSOLUTE obj_temp_c. Measured over the pilot
# sessions, ambient falls 1-2 C across a night as the room cools, so absolute
# object temperature TRENDS DOWNWARD even while the motor is running and
# self-heating. A model fed absolute temperature learns the building's
# overnight cooling curve, not motor health. The delta is the physically
# meaningful quantity and it does separate operating points
# (+1.68 C at 15 V vs +2.41 C at 24 V).
THERMAL_USE_DELTA = True

# ---------------------------------------------------------------------------
# Gap handling
# ---------------------------------------------------------------------------
# Windows are only cut from contiguous runs of samples. A contiguous run ends
# wherever packet_index.csv shows a sequence discontinuity. Runs shorter than
# this are discarded rather than padded.
MIN_SEGMENT_SECONDS = 2.0

# Drop the motor-off arm delay at the start of each session from training
# features. It is kept as the acoustic noise-profile reference instead
# (see docs/notes/acoustic_noise_filtering.md).
ARM_DELAY_SKIP_SECONDS = 75.0

# ---------------------------------------------------------------------------
# Windowing / features
# ---------------------------------------------------------------------------
WINDOW_SECONDS = 1.0
WINDOW_OVERLAP = 0.5            # 50 % between analysis windows

# Within each 1 s window the spectrum is a Welch PSD rather than one raw FFT of
# the whole window. Two reasons:
#   1. Variance. Averaging ~6 sub-periodograms per window gives a far more
#      stable feature vector, which is what an autoencoder needs to learn a
#      tight healthy manifold instead of chasing periodogram noise.
#   2. Speed. The measured ODR is ~26547 Hz, whose factorisation contains a
#      large prime, so a full-window rFFT falls back to Bluestein and is very
#      slow. A power-of-two segment length keeps it on the fast path.
# Frequency resolution = measured_odr / WELCH_NPERSEG ~ 3.24 Hz.
WELCH_NPERSEG = 8192

# Keep the first N PSD bins. At 3.24 Hz/bin, 1024 bins covers DC..~3.3 kHz.
#
# Measured power distribution on healthy 24 V data (full band 0-13267 Hz):
#     0- 500 Hz  12.5 %      3315- 5000 Hz   3.1 %
#   500-1000 Hz  45.0 %      5000- 8000 Hz   1.1 %
#  1000-2000 Hz  25.6 %      8000-11000 Hz   9.2 %   <-- distinct band
#  2000-3315 Hz   2.1 %     11000-13270 Hz   1.5 %
#
# So 1024 bins captures ~85 % of total power and the whole rotational harmonic
# comb. But the excluded 8-11 kHz band is NOT noise floor -- at -71 dB mean it
# sits well above the -80 dB floor of its neighbours, so it is a real feature
# (structural resonance, or bearing-related). That is precisely the region where
# incipient bearing defects appear first.
#
# DECISION PENDING: raise this to 4097 (the full band) before recording any
# bearing-fault condition, and compare discriminative power against the 1024-bin
# set. Document whichever the thesis defends. Cost of the full band is 4x the
# feature cache and a 4x wider model input layer.
FFT_BINS_KEEP = 1024

# How to collapse the 3 accelerometer axes into a spectrum.
#   "power_mean" : mean of per-axis power spectra, then dB. Linear in power,
#                  so it introduces no cross-axis intermodulation. Default.
#   "per_axis"   : concatenate the three per-axis dB spectra (3x features).
#   "magnitude"  : rFFT of ||v|| after mean removal. NOT recommended -- the
#                  Euclidean norm is a nonlinear mix of the axes and creates
#                  spurious sum/difference frequencies that are easy to
#                  mistake for real sidebands.
AXIS_COMBINE = "power_mean"

# Audio
AUDIO_N_MELS = 64
AUDIO_WINDOW_SECONDS = 1.0

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TRAIN_FRACTION = 0.8            # first 80 % of session (by time) = train
VAL_FRACTION = 0.1              # next 10 % = validation; remainder = test
BOTTLENECK_UNITS = 16
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
