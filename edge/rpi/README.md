# edge/rpi/ — Raspberry Pi inference tier

**Status: early. One working script, the rest is planned structure.**

The Pi is the inference tier of the cascade. It receives raw sensor counts from
the ESP32 over USB CDC, does unit conversion and feature extraction (deliberately
NOT done on the MCU), runs the local quantized model, and decides per window
whether the local prediction is confident enough or the window should be offloaded
to the cloud tier.

The Pi->cloud link is the one the bandwidth-savings claim applies to. The
ESP32->Pi link is infrastructure and is excluded from that calculation.

## What exists

| Path | Status |
|---|---|
| `pi_edge_twin.py` | **Working.** Subscribes to an MQTT topic, thresholds `rms_vibration`, writes breaches to InfluxDB 3. This is a functional spike of the confidence gate, not the final design. |
| `config/edge_config.py` | Empty stub. |
| `communications/` | Planned: `usb_listener.py`, `cloud_offload.py`. |
| `data_pipeline/` | Planned: `sensor_processor.py`. |
| `inference/` | Planned: `model.py`, `confidence_gate.py`, `alert_handler.py`. |

## Known gaps

- **`pi_edge_twin.py` uses a fixed `VIBRATION_THRESHOLD` on a single scalar RMS.**
  The offline pipeline in `ml/` produces a 1024-bin PSD feature vector and an
  autoencoder reconstruction error. These need to converge — the deployed gate
  should sit on the same feature and the same threshold the thesis derives in
  `ml/evaluate.py`, not on an unrelated hand-set RMS limit.
- **Two secret stores.** `pi_edge_twin.py` imports credentials from the untracked
  `edge/esp32/tools/private.py`; everything newer uses the gitignored `.env` at
  the repo root. Converge on `.env`.
- **Input path is MQTT, not USB.** The current script assumes something else is
  already publishing to MQTT. `communications/usb_listener.py` is the missing
  piece that would make the Pi read the ESP32 directly, matching the architecture
  described in the root README.
