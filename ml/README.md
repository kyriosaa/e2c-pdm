# ml/ — offline ML pipeline

Consumes sessions recorded by `edge/esp32/tools/data_catcher.py` from
`data/raw/`, and produces model artifacts for the edge tier.

```
ml/
├── config.py             all knobs; repo-root anchored paths
├── sessions.py           gap map, measured sample rates, status log   <- read this first
├── data_loader.py        memory-mapped session access
├── features.py           gap-aware Welch PSD windows
├── train_autoencoder.py  leave-one-session-out or time split
├── evaluate.py           thresholds, false-positive rate, novelty check
├── quantize.py           TFLite dynamic + int8, accuracy/threshold drop table
├── studies/              one-off analyses (audio filtering, integrity report)
└── notebooks/            exploratory
```

Generated output lives outside the package:

```
data/interim/<session>/index.json      cached gap map + measured rates
data/interim/features/<session>/       cached feature windows
ml/artifacts/                          models, scaler, tflite (gitignored)
docs/figures/pipeline/                      diagnostic plots (gitignored)
docs/figures/<study>/                       tracked thesis figures
```

## Run everything as a module, from the repo root

```
python -m ml.sessions
```

Not `python ml/sessions.py`. The package uses relative imports, and every path in
`config.py` resolves from the repository root rather than the working directory,
so results do not depend on where you happen to be standing.

## Python version — read before installing

**Feature extraction needs only numpy + scipy** and runs on any modern Python,
including 3.14.

**Training, evaluation and quantization need TensorFlow, which does not publish
wheels for Python 3.14.** Those three steps will exit with instructions if TF is
missing. Use a separate 3.12 environment for them:

```
py -3.12 -m venv .venv-train
.venv-train\Scripts\activate
pip install -r ml/requirements.txt
```

Cached features under `data/interim/features/` are plain `.npy` and are shared
between both environments, so there is no need to re-extract.

## Run order

```
pip install -r ml/requirements.txt

# 1. Parse packet_index.csv -> gap map + measured sample rates. Cached; the
#    first run takes ~10 s per session for a 366 MB index.
python -m ml.sessions

# 2. Per-session sanity check: measured ODR, full scale, segment count,
#    motor-off vs running RMS.
python -m ml.data_loader [session]

# 3. Feature windows. ~1.5 min per 6 h session.
python -m ml.features [session ...]

# 4. Train. --split session is leave-one-session-out and is the number to report.
#    Restrict the training pool to ONE operating point so the other stays unseen
#    and can serve as the novelty check. With the three pilot sessions:
python -m ml.train_autoencoder --split session \
    --sessions healthy24V_motor_baseline_20260718_001741 \
               healthy24V_motor_baseline_20260719_010905 \
    --holdout  healthy24V_motor_baseline_20260719_010905

# 5. Thresholds + false-positive rate. --novelty scores the 15 V session, which
#    the model has never seen, as an off-manifold sensitivity check.
python -m ml.evaluate --novelty healthy15V_motor_baseline_20260716_002156

# 6. TFLite export, accuracy drop, and threshold shift under quantization.
python -m ml.quantize

# Anytime: the dataset integrity table.
python -m ml.studies.integrity_report [--markdown]
```

## Three things that are easy to get wrong

**1. `vibration.bin` has no framing and no gap markers.** `data_catcher.py`
validates the header and CRC in memory, then writes *only* the payload. Lost
packets are simply absent, so the file is a splice of discontinuous runs. Cutting
an FFT window across a splice fabricates a broadband transient that looks exactly
like an impulsive mechanical fault. `sessions.contiguous_segments()` is the only
safe source of window offsets.

On-device FIFO overruns are the nastier version of this: samples vanish inside
the sensor while the packet sequence stays perfectly contiguous, so they are
invisible to sequence-gap detection. The segment map breaks on the overrun flag
for exactly this reason.

**2. Never use a nominal sample rate.** The IIS3DWB measures 26 533–26 547 Hz
against a 26 667 Hz datasheet nominal — it runs ~0.45 % slow, consistently. On the
same host clock the audio stream is accurate to a few ppm, which localises the
error to the sensor's oscillator. At nominal the frequency axis is wrong by 2.9 Hz
at the 635 Hz peak. `sessions.measured_odr()` derives the rate from sequence
numbers and host timestamps, which makes it immune to packet loss and to MCU
resets.

**3. Accelerometer full scale is per session.** The pilot sessions were recorded
at ±16 g; everything from 2026-07-30 is ±4 g. `session.json` carries `accel_fs_g`,
and `sessions.accel_fs_g()` falls back by start timestamp for the pilot sessions.
It raises rather than guessing for anything ambiguous. Do not hardcode it.

## Memory

Everything is memory-mapped and converted per window. A 6 h session is 3.2 GB of
int16 vibration; materialising it as float32 `(N, 3)` would need ~7 GB on top of
the raw bytes, so `load_vibration`-style whole-file loading is deliberately absent.
`load_audio()` still reads a whole track (~1.4 GB as float32) because the audio
studies want it; prefer `open_audio()` plus `audio_to_float()` on slices.

## What the current results can and cannot claim

Can: measured rig characterisation, dataset integrity figures, a healthy-data
false-positive operating point, and a sensitivity-to-change demonstration
(train on 24 V, score 15 V).

Cannot: any detection rate, ROC or AUC. There is no fault data. See
[../docs/notes/collection_protocol.md](../docs/notes/collection_protocol.md).
