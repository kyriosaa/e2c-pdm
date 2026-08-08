"""
Converts the trained autoencoder to TFLite (dynamic-range and full int8),
then compares reconstruction error against the float model on the test split.
This is the first rung of the edge deployment chain.

Also reports the threshold shift caused by quantization: if the int8 model's
score distribution moves, the threshold picked on the float model no longer has
the false-positive rate it was chosen for. Re-deriving the threshold on the
quantized model is the deployable thing to do.

Run after train_autoencoder.py:
    python -m ml.quantize
"""

from __future__ import annotations

import json

import numpy as np

from . import config as C
from .train_autoencoder import require_tf


def tflite_reconstruct(tflite_model: bytes, x: np.ndarray) -> np.ndarray:
    tf = require_tf()

    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def quant(a, detail):
        scale, zp = detail["quantization"]
        if scale == 0:
            return a.astype(detail["dtype"])
        return (a / scale + zp).round().astype(detail["dtype"])

    def dequant(a, detail):
        scale, zp = detail["quantization"]
        if scale == 0:
            return a.astype(np.float32)
        return (a.astype(np.float32) - zp) * scale

    recon = np.empty_like(x, dtype=np.float32)
    for i in range(len(x)):
        xi = x[i: i + 1].astype(np.float32)
        interp.set_tensor(inp["index"], quant(xi, inp)
                          if inp["dtype"] != np.float32 else xi)
        interp.invoke()
        yi = interp.get_tensor(out["index"])
        recon[i] = dequant(yi, out) if out["dtype"] != np.float32 else yi
    return recon


def main() -> None:
    tf = require_tf()

    scaler = np.load(C.ARTIFACTS_DIR / "scaler.npz")
    with open(C.ARTIFACTS_DIR / "split.json") as f:
        split = json.load(f)

    mu, sd = scaler["mu"], scaler["sd"]
    # Both splits were written by train_autoencoder.py, so this works for the
    # leave-one-session-out protocol as well as the time split.
    x_te = ((np.load(C.ARTIFACTS_DIR / "test_features.npy") - mu) / sd
            ).astype(np.float32)
    x_va = ((np.load(C.ARTIFACTS_DIR / "val_features.npy") - mu) / sd
            ).astype(np.float32)
    # Calibration set for int8: spread across the VALIDATION data, never the
    # held-out test session, or the quantization ranges are fitted to the data
    # the model is about to be judged on.
    step = max(1, len(x_va) // 200)
    x_rep = x_va[::step][:200].astype(np.float32)

    model = tf.keras.models.load_model(C.ARTIFACTS_DIR / "vib_autoencoder.keras")

    # Float baseline
    err_float = np.mean((x_te - model.predict(x_te, verbose=0)) ** 2, axis=1)

    # Dynamic-range quantization
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tfl_dyn = conv.convert()
    (C.ARTIFACTS_DIR / "vib_ae_dynamic.tflite").write_bytes(tfl_dyn)

    # Full int8 quantization (needs a representative dataset)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: (
        [x_rep[i: i + 1]] for i in range(len(x_rep)))
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tfl_int8 = conv.convert()
    (C.ARTIFACTS_DIR / "vib_ae_int8.tflite").write_bytes(tfl_int8)

    err_dyn = np.mean((x_te - tflite_reconstruct(tfl_dyn, x_te)) ** 2, axis=1)
    err_i8 = np.mean((x_te - tflite_reconstruct(tfl_int8, x_te)) ** 2, axis=1)

    # Threshold stability: a threshold chosen on float32 does not keep its
    # false-positive rate if quantization shifts the score distribution.
    err_va_f = np.mean((x_va - model.predict(x_va, verbose=0)) ** 2, axis=1)
    err_va_i8 = np.mean((x_va - tflite_reconstruct(tfl_int8, x_va)) ** 2,
                        axis=1)
    thr_f = float(np.percentile(err_va_f, 99))
    thr_i8 = float(np.percentile(err_va_i8, 99))

    def kb(b):
        return len(b) / 1024
    print(f"Model sizes: dynamic={kb(tfl_dyn):.1f} KB, "
          f"int8={kb(tfl_int8):.1f} KB")
    print(f"Mean test reconstruction error:")
    print(f"  float32      {err_float.mean():.5f}")
    print(f"  dynamic-q    {err_dyn.mean():.5f} "
          f"(delta {100 * (err_dyn.mean() / err_float.mean() - 1):+.2f}%)")
    print(f"  full int8    {err_i8.mean():.5f} "
          f"(delta {100 * (err_i8.mean() / err_float.mean() - 1):+.2f}%)")
    print("\nSmall deltas (<~5%) mean quantization is safe for the edge tier. "
          "Record these numbers - they're a thesis table.")


if __name__ == "__main__":
    main()
