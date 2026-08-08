# data/raw/ — recorded sessions (untracked)

One directory per recording, named `<label>_<YYYYmmdd_HHMMSS>_<Day>`, written by
`edge/esp32/tools/data_catcher.py`. Roughly **4.5 GB per 6-hour session**.

The weekday is a convenience for scanning the directory listing and carries no
meaning downstream. Two constraints it works around: the label stays first
because `ml.train_autoencoder.operating_point()` keys on the first `_`-separated
field, and `session.json`'s `started` stays a bare timestamp because
`ml.sessions.accel_fs_g()` string-compares it against `ACCEL_FS_CHANGED_AFTER`.
The six pilot sessions predate the weekday and keep their original names.

Never renamed after recording: the cached index in `data/interim/<session>/` is
keyed on the directory name.

## Getting sessions onto the analysis machine

Recording happens on a separate machine from analysis, and this directory is
untracked, so sessions travel by removable drive. **Do not drag-and-drop** — use
the export tool, which SHA-256s every file as it writes and leaves the digests
in `export_manifest.json` alongside the data:

```
# recording machine -> drive
python edge/esp32/tools/export_sessions.py E:/pdm_transfer --skip-tests

# analysis machine: copy the drive's contents into data/raw/, including
# export_manifest.json, then verify the copy you will actually use
python edge/esp32/tools/export_sessions.py data/raw --verify
```

Verify `data/raw`, not the drive — verifying the drive only proves the drive is
intact and says nothing about the second copy.

`--verify` exits non-zero on any mismatch, and re-running the export skips files
already present, so an interrupted transfer resumes. Verify before deleting the
source copy: `vibration.bin` is unframed int16 with no checksum of its own, so a
corrupted byte range is indistinguishable from a real impulsive transient once
it reaches the feature extractor.

## Files per session

| File | Contents |
|---|---|
| `vibration.bin` | Bare little-endian int16 x,y,z triplets. **No framing, no gap markers.** Scale with `accel_fs_g` from `session.json`. |
| `audio.wav` | 16 kHz 16-bit mono. First ~60 s is motor-off room reference. |
| `status.csv` | ~0.977 Hz log: obj/amb temperature, motor state, fault code, counters. |
| `packet_index.csv` | Host arrival time, type, seq, n_samples, flags, dropped for **every** packet. This is what makes the gap map possible. ~366 MB. |
| `session.json` | Metadata + integrity summary + acquisition provenance. |

## Read this before analysing anything

`vibration.bin` holds payload bytes only — lost packets are simply absent, so the
file is a splice of discontinuous runs. Windowing across a splice fabricates a
broadband transient that mimics an impulsive fault. Use
`ml.sessions.contiguous_segments()`, never a naive `arange` over the file.

`session.json`'s `vib_capture_ratio` divides by the **nominal** 26 667 Hz ODR and
therefore understates completeness by the sensor's ~0.45 % clock error. For a real
integrity figure use the sequence-derived ratio from `ml.sessions` /
`ml.studies.integrity_report`.

## Current contents

Six 6-hour healthy baselines recorded 2026-07-16 to 07-20 at ±16 g full scale,
plus three short `test_*` rig-shakedown captures from 07-15/16 that are not
analysis data.

| Session | Motor | Capture ratio | CRC errors |
|---|---|---|---|
| `healthy15V_motor_baseline_20260716_002156` | 15 V | 0.9953 | 160 |
| `healthy24V_motor_baseline_20260718_001741` | 24 V | 0.9949 | 83 |
| `day_healthy24V_motor_baseline_20260718_170626` | 24 V, daytime room | 0.9953 | 123 |
| `healthy24V_motor_baseline_20260719_010905` | 24 V | 0.9955 | 115 |
| `healthy24V_motor_baseline_20260720_003416` | 24 V | 0.9946 | **5719** |
| `healthy24V_motor_baseline_20260720_231747` | 24 V | 0.9955 | 375 |

These are **rig validation / pilot data**, not the final healthy baseline: they
predate the ±4 g full-scale change and were recorded with no mechanical load.
See `docs/notes/collection_protocol.md`.

None of them carry the provenance fields (`accel_fs_g`, `motor_voltage_v`,
`mechanical_load`, `fault_condition`, `room_condition`, `notes`) — all six
predate `ACCEL_FS_CHANGED_AFTER`, so `ml.sessions.accel_fs_g()` resolves them to
±16 g by start timestamp. Voltage in the table above comes from the directory
name, which is the only record of it.

### Two things to check before trusting a session

`healthy24V_motor_baseline_20260720_003416` logged **5719 CRC errors**, roughly
50× every other session (5–375). Its capture ratio is normal, so the link was
recovering, but confirm the gap map from `ml.studies.integrity_report` before
using it — link trouble at that rate may have left more or larger splices.

Every `session.json` here also carries a **`measured_odr_hz` field that is not
authoritative**. It was backfilled as `vib_samples / elapsed_s`, which divides
real samples by wall-clock time and so folds packet loss into what looks like a
clock measurement. It happens to land close (26456–26546 Hz) because transport
loss was ~0.001 %, but it is the same conflation `vib_capture_ratio` is warned
about above. Use `ml.sessions.measured_odr()`, which derives the rate from
packet sequence numbers and separates the two.
