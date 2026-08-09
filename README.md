# E2C-PDM — An edge-to-cloud digital twin for predictive maintenance

Thesis project. A three-tier condition-monitoring system built around a
laboratory motor rig: an ESP32-S3 acquisition front-end, a Raspberry Pi
inference tier, and a cloud tier for the digital twin and long-horizon storage.

**Status: data acquisition validated, healthy baseline recorded, no fault data
yet.** The offline feature pipeline runs end to end. Model training is blocked on
seeded-fault data — see
[docs/notes/collection_protocol.md](docs/notes/collection_protocol.md).

---

## Architecture

```
   MOTOR RIG                 EDGE TIER 1            EDGE TIER 2        CLOUD
   ---------                 -----------            -----------        -----
  RS-445PA DC motor
  + BTS7960 driver
        |
  IIS3DWB   26.7 kHz x3 ---+
  INMP441   16 kHz mono ---+--> ESP32-S3 ----USB CDC----> Raspberry Pi ---> InfluxDB
  MLX90614  1 Hz -------- +    (FreeRTOS,   ~200 KB/s    (features,        + digital
                               2 cores,                   inference,         twin
                               raw counts)                cascade gate)
                                    |
                            thermal interlock
                            70 C latched kill
```

Design principle: computation is pushed to the cheapest place. The MCU ships raw
sensor counts and does no unit conversion or feature extraction, which keeps it a
deterministic acquisition front-end and keeps the acquisition path identical
between dataset collection and deployment.

The ESP32→Pi link is infrastructure and is **excluded** from the bandwidth-savings
claim; that claim applies to the Pi→cloud tier only.

## Repository layout

```
edge/
  esp32/firmware/mdc/                  acquisition + safety firmware
  esp32/tools/                         data_catcher.py (host logger), notifier, start_handler
  hardware/edge_asset_monitor_2L/      2-layer PCB, lab-etched (KiCad)
  hardware/edge_asset_monitor_4L/      4-layer PCB + gerbers (KiCad)
  hardware/espressif-kicad-addon/      VENDORED third-party library, see its README
  rpi/                                 Pi-tier scaffolding (early)
cloud/                                 cloud-tier scaffolding (early)
ml/                                    offline pipeline -- see ml/README.md
  studies/                             one-off analyses (audio filtering, integrity)
  notebooks/                           exploratory notebooks
data/                                  UNTRACKED. raw/ interim/ processed/
docs/
  notes/                               decisions + rationale + the quirk
                                       register. READ THIS FIRST.
  figures/                             tracked thesis figures (pipeline/ is generated)
  datasheets/ certifications/          RS-445PA, IIS3DWB, STEVAL-MKI208V1K,
                                       ISO 13732-1, ...
  schematics/ images/                  rev1.0 and rev2.0
references.md                          reference paper bank
```

`data/` is gitignored; sessions are ~4.5 GB each. `docs/notes/` is gitignored
too, so **the design notes linked below are local-only and will not be present
in a clone.** `docs/figures/pipeline/` is gitignored as the pipeline regenerates
it; the rest of `docs/` is tracked.

## Getting oriented

Start with [docs/notes/](docs/notes/):

| Note | What it settles |
|---|---|
| [quirk_register.md](docs/notes/quirk_register.md) | **Read before quoting any number.** The real dataset inventory, known defects, and the measured sensor calibration. |
| [collection_protocol.md](docs/notes/collection_protocol.md) | **How to collect data from here on.** The confound trap, the fault set, the integrity gate. |
| [operating_profile_rationale.md](docs/notes/operating_profile_rationale.md) | Why the rig needs no "real" use case, and what it does need. |
| [motor_signature.md](docs/notes/motor_signature.md) | Measured spectrum, measured sample rates, rig validation against physics. |
| [thermal_channel_confound.md](docs/notes/thermal_channel_confound.md) | Why absolute temperature is unusable and the thermal channel is currently flat. |
| [acoustic_noise_filtering.md](docs/notes/acoustic_noise_filtering.md) | Why spectral subtraction, and why not a notch or high-pass. |
| [overnight_acquisition_rig.md](docs/notes/overnight_acquisition_rig.md) | Acquisition architecture, integrity layers, safety layers, defence Q&A. |
| [thermal_interlock_70c.md](docs/notes/thermal_interlock_70c.md) | Why 70 °C, from ISO 13732-1. |

## Reproducing a recording session

Requires the physical rig. Read
[overnight_acquisition_rig.md](docs/notes/overnight_acquisition_rig.md) §3
before running unattended.

1. Flash `edge/esp32/firmware/mdc/` to the ESP32-S3.
   **Never flash with the 24 V rail energised.** Power sequence: USB first, 24 V
   last; de-energise 24 V first.
2. Set the run parameters and provenance fields at the top of
   `edge/esp32/tools/data_catcher.py` — `SESSION_LABEL`, `RUN_SECONDS`,
   `MOTOR_VOLTAGE_V`, `MECHANICAL_LOAD`, `FAULT_CONDITION`, `ROOM_CONDITION`.
   `ACCEL_FS_G` must match the firmware.
3. `python edge/esp32/tools/data_catcher.py`
   The motor arms after `ARM_DELAY_S` (60 s); that head of the recording is the
   motor-off room-noise reference. No logger means no motor — the firmware
   requires an explicit host arm.
4. Output lands in `data/raw/<label>_<timestamp>/`: `vibration.bin`, `audio.wav`,
   `status.csv`, `packet_index.csv`, `session.json`.
5. Gate the session on integrity before trusting it:
   ```
   python -m ml.sessions <session_name>
   python -m ml.studies.integrity_report
   ```
   Acceptance criteria are in
   [collection_protocol.md](docs/notes/collection_protocol.md).

## Running the offline pipeline

From the repository root. See [ml/README.md](ml/README.md) for detail.

```
pip install -r ml/requirements.txt

python -m ml.sessions                 # gap map + measured sample rates (cached)
python -m ml.data_loader              # per-session sanity check
python -m ml.features                 # gap-aware Welch PSD windows
python -m ml.studies.integrity_report # dataset integrity table

# leave-one-session-out, training pool held to one operating point so the 15 V
# session stays unseen for the novelty check
python -m ml.train_autoencoder --split session \
    --sessions healthy24V_motor_baseline_20260718_001741 \
               healthy24V_motor_baseline_20260719_010905 \
    --holdout  healthy24V_motor_baseline_20260719_010905
python -m ml.evaluate --novelty healthy15V_motor_baseline_20260716_002156
python -m ml.quantize                 # TFLite dynamic + int8
```

Feature extraction needs only numpy + scipy. **Training needs TensorFlow, which
has no Python 3.14 wheels** — use a separate 3.12 environment for the last three
steps; cached features are reusable across both. See ml/README.md.

## Secrets

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored and is the only
place credentials belong.
