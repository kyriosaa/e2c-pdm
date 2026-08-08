# cloud/ — cloud tier

**Status: scaffolding only.**

Receives offloaded windows from the Pi when the local model is not confident,
runs the larger cloud model, and backs the digital twin and long-horizon storage
(InfluxDB 3).

| Path | Status |
|---|---|
| `config/cloud_config.py` | Empty stub. |
| `inference/` | Planned: `model.py` (cloud model), `inference_service.py` (API endpoint). |

Credentials belong in the gitignored `.env` at the repo root, not in source.
