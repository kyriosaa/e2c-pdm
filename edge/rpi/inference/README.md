# Planned

- `model.py` — load and run the int8 TFLite autoencoder from `ml/artifacts/`.
- `confidence_gate.py` — threshold the reconstruction error to decide local vs
  cloud. Use the threshold derived on the **quantized** model
  (`ml/quantize.py` reports both); a float-derived threshold does not keep its
  false-positive rate after quantization.
- `alert_handler.py` — act on confident local detections.
