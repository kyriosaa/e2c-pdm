"""
Scores the trained autoencoder and reports thresholds.

What this can and cannot tell you
---------------------------------
CAN: the false-positive rate of a threshold on healthy data it has never seen
(the held-out session), which is the operating point of the cascade's
confidence gate.

CAN: a SENSITIVITY DEMONSTRATION without any fault data. Train on the 24 V
healthy sessions, then score the 15 V session with --novelty. The 15 V motor is
mechanically healthy but runs at a different speed, so its spectrum sits off the
learned manifold. A large score separation shows the model responds to a genuine
change in machine state rather than to noise. This is a change-detection
sanity check, NOT a fault-detection result, and must be labelled as such.

CANNOT: a detection rate, ROC curve, or AUC. Those need seeded-fault data. Any
threshold reported here is a false-positive operating point only.

Run after train_autoencoder.py:
    python -m ml.evaluate
    python -m ml.evaluate --novelty healthy15V_motor_baseline_20260716_002156
"""

from __future__ import annotations

import json

import numpy as np

from . import config as C
from . import features as F
from .train_autoencoder import require_tf


def scores(model, X: np.ndarray, mu: np.ndarray, sd: np.ndarray,
           batch: int = 512) -> np.ndarray:
    """Per-window reconstruction MSE in standardised feature space."""
    x = (X - mu) / sd
    recon = model.predict(x, batch_size=batch, verbose=0)
    return np.mean((x - recon) ** 2, axis=1)


def describe(name: str, e: np.ndarray) -> None:
    print(f"  {name:22s} n={len(e):>7,}  mean={e.mean():.5f}  "
          f"p50={np.percentile(e, 50):.5f}  p95={np.percentile(e, 95):.5f}  "
          f"p99={np.percentile(e, 99):.5f}  max={e.max():.5f}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--novelty", default=None,
                    help="extra session to score as an off-manifold check")
    args = ap.parse_args()

    tf = require_tf()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(C.ARTIFACTS_DIR / "split.json") as f:
        split = json.load(f)
    sc = np.load(C.ARTIFACTS_DIR / "scaler.npz")
    mu, sd = sc["mu"], sc["sd"]
    model = tf.keras.models.load_model(
        C.ARTIFACTS_DIR / "vib_autoencoder.keras")

    X_va = np.load(C.ARTIFACTS_DIR / "val_features.npy")
    X_te = np.load(C.ARTIFACTS_DIR / "test_features.npy")
    err_va = scores(model, X_va, mu, sd)
    err_te = scores(model, X_te, mu, sd)

    print(f"Protocol: {split['protocol']}")
    if split["protocol"] == "leave-one-session-out":
        print(f"  train sessions : {split['train_sessions']}")
        print(f"  held-out       : {split['holdout']}")
    print("\nReconstruction error (MSE):")
    describe("val (healthy)", err_va)
    describe("held-out (healthy)", err_te)

    thr_p99 = float(np.percentile(err_va, 99))
    thr_3sig = float(err_va.mean() + 3 * err_va.std())
    fpr_p99 = float((err_te > thr_p99).mean())
    fpr_3sig = float((err_te > thr_3sig).mean())

    print(f"\nThresholds from the healthy validation distribution:")
    print(f"  p99         = {thr_p99:.5f}  -> held-out FPR {fpr_p99:.2%}")
    print(f"  mean+3sigma = {thr_3sig:.5f}  -> held-out FPR {fpr_3sig:.2%}")
    print(f"  (These are FALSE-POSITIVE operating points only. No detection "
          f"rate is computable without fault data.)")
    if fpr_p99 > 0.10:
        print(f"  [!] held-out FPR of {fpr_p99:.1%} against a nominal 1 % "
              f"threshold means session-to-session variation exceeds "
              f"within-session variation. The model is keying on something "
              f"that differs between recordings -- check ambient conditions, "
              f"mounting, and accelerometer full scale before blaming the "
              f"model.")

    results = {"protocol": split["protocol"],
               "thr_p99": thr_p99, "thr_3sigma": thr_3sig,
               "holdout_fpr_p99": fpr_p99, "holdout_fpr_3sigma": fpr_3sig,
               "val_mean": float(err_va.mean()),
               "holdout_mean": float(err_te.mean())}

    err_nov = None
    if args.novelty:
        d = F.load_session_features(args.novelty)
        err_nov = scores(model, d["features"], mu, sd)
        describe(f"novelty ({args.novelty.split('_')[0]})", err_nov)
        rate = float((err_nov > thr_p99).mean())
        sep = float(err_nov.mean() / max(err_te.mean(), 1e-12))
        print(f"\nNovelty check on {args.novelty}:")
        print(f"  windows above healthy p99 threshold : {rate:.2%}")
        print(f"  mean-score ratio vs held-out healthy: {sep:.2f}x")
        print(f"  Interpretation: this session is mechanically HEALTHY at a "
              f"different operating point. A high flag rate demonstrates "
              f"SENSITIVITY TO CHANGE, not fault detection.")
        results["novelty_session"] = args.novelty
        results["novelty_flag_rate"] = rate
        results["novelty_score_ratio"] = sep

    with open(C.ARTIFACTS_DIR / "eval.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- plots
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].plot(err_te, lw=0.4, label="held-out healthy session")
    axes[0].axhline(thr_p99, color="r", ls=":", lw=1, label="val p99")
    axes[0].set_title("Anomaly score over the held-out healthy session")
    axes[0].set_xlabel("window index (time-ordered)")
    axes[0].set_ylabel("reconstruction MSE")
    axes[0].legend(fontsize=8)

    bins = np.histogram_bin_edges(
        np.concatenate([err_va, err_te] + ([err_nov] if err_nov is not None
                                           else [])), bins=120)
    axes[1].hist(err_va, bins=bins, alpha=0.55, density=True, label="val")
    axes[1].hist(err_te, bins=bins, alpha=0.55, density=True, label="held-out")
    if err_nov is not None:
        axes[1].hist(err_nov, bins=bins, alpha=0.55, density=True,
                     label=f"novelty ({args.novelty.split('_')[0]})")
    axes[1].axvline(thr_p99, color="r", ls=":", lw=1)
    axes[1].set_yscale("log")
    axes[1].set_title("Score distributions (log density)")
    axes[1].set_xlabel("reconstruction MSE")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = C.PLOTS_DIR / "anomaly_score.png"
    fig.savefig(out, dpi=120)
    print(f"\nWrote {out.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
