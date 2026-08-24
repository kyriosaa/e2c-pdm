"""
Fault-detection evaluation: the script to run the day fault data exists.

`ml.evaluate` stops where healthy-only data stops: thresholds and
false-positive rates. This module is the other half -- detection rates, ROC and
AUC -- and it is written BEFORE any fault session exists so the fault campaign
is not blocked on analysis code. Rehearse it today with the 15 V session as a
stand-in fault:

    python -m ml.evaluate_faults --fault healthy15V_motor_baseline_20260716_002156_Thurs

A rehearsal like that is a CHANGE-DETECTION exercise: the 15 V session is
mechanically healthy at a different operating point, so its numbers demonstrate
sensitivity to change, never fault detection. The script detects the case (any
--fault session labelled healthy*) and says so in the output and in the saved
JSON. Real fault sessions -- e.g. imbalanceL1_... -- report real detection
results.

What it reports, per fault session and pooled:

  * window-level detection rate at the deployed thresholds (val p99, 3sigma) --
    directly comparable with the held-out false-positive rates in eval.json;
  * AUC against held-out healthy windows (Mann-Whitney, tie-safe), which is
    threshold-free and the headline number for the thesis;
  * time to first alarm, in windows and seconds, and the rate under a
    k-consecutive-windows rule -- the smoothing a deployed gate would apply.

Threshold provenance: thresholds are recomputed from val_features.npy exactly
as ml.evaluate does. If ml/artifacts/quantize.json exists (written by
ml.quantize) its int8 threshold is reported alongside, because the device runs
int8 and a float threshold does not keep its operating point after quantization
(quirk_register.md 3.5). Scoring here is float; --model int8 rescoring can be
added when the edge inference path is exercised end to end.

Run after train_autoencoder.py:
    python -m ml.evaluate_faults --fault <session ...> [--healthy <session ...>]
"""

from __future__ import annotations

import json

import numpy as np

from . import config as C
from . import features as F
from .train_autoencoder import operating_point, require_tf

OUT_JSON = C.ARTIFACTS_DIR / "eval_faults.json"
K_CONSECUTIVE = 3   # windows in a row above threshold = one confirmed alarm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def auc_mann_whitney(healthy: np.ndarray, fault: np.ndarray) -> float:
    """P(fault window scores higher than healthy window). Tie-safe, exact."""
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([healthy, fault]))
    n_h, n_f = len(healthy), len(fault)
    return float((r[n_h:].sum() - n_f * (n_f + 1) / 2) / (n_h * n_f))


def roc_curve(healthy: np.ndarray, fault: np.ndarray,
              n_points: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """(fpr, tpr) swept over score thresholds, for plotting."""
    thr = np.quantile(np.concatenate([healthy, fault]),
                      np.linspace(0.0, 1.0, n_points))
    fpr = np.array([(healthy > t).mean() for t in thr])
    tpr = np.array([(fault > t).mean() for t in thr])
    order = np.argsort(fpr)
    return fpr[order], tpr[order]


def first_alarm(err: np.ndarray, thr: float, k: int) -> tuple[int, int]:
    """(first window above thr, first window ending k consecutive above thr).

    Either is -1 if it never happens. Windows are time-ordered.
    """
    above = err > thr
    first = int(np.argmax(above)) if above.any() else -1
    if k <= 1:
        return first, first
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= k:
            return first, i
    return first, -1


def confirmed_alarm_rate(err: np.ndarray, thr: float, k: int) -> float:
    """Fraction of windows inside a run of >= k consecutive flags.

    The smoothing a deployed gate applies: one stray window does not ship
    data, k in a row does. Applied identically to healthy and fault scores so
    the two stay comparable.
    """
    above = err > thr
    if not above.any():
        return 0.0
    confirmed = np.zeros(len(err), dtype=bool)
    run_start, run = 0, 0
    for i, a in enumerate(above):
        if a:
            if run == 0:
                run_start = i
            run += 1
            if run >= k:
                confirmed[run_start:i + 1] = True
        else:
            run = 0
    return float(confirmed.mean())


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def scores(model, X: np.ndarray, mu: np.ndarray, sd: np.ndarray,
           batch: int = 512) -> np.ndarray:
    x = (X - mu) / sd
    recon = model.predict(x, batch_size=batch, verbose=0)
    return np.mean((x - recon) ** 2, axis=1)


def hop_seconds(session_name: str) -> float:
    """Seconds between consecutive windows, from the session's own meta."""
    with open(C.FEATURES_DIR / session_name / "meta.json") as f:
        m = json.load(f)
    return m["hop_samples"] / m["odr_hz"]


# ---------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fault", nargs="+", required=True,
                    help="session(s) to score as faulty / off-baseline")
    ap.add_argument("--healthy", nargs="*", default=None,
                    help="healthy reference session(s); default = the "
                         "held-out session from split.json")
    ap.add_argument("--k", type=int, default=K_CONSECUTIVE,
                    help="consecutive windows above threshold that confirm "
                         "an alarm (default %(default)s)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    F.require_raw_counterpart(args.fault + (args.healthy or []))

    tf = require_tf()

    with open(C.ARTIFACTS_DIR / "split.json") as f:
        split = json.load(f)
    sc = np.load(C.ARTIFACTS_DIR / "scaler.npz")
    mu, sd = sc["mu"], sc["sd"]
    model = tf.keras.models.load_model(
        C.ARTIFACTS_DIR / "vib_autoencoder.keras")

    overlap = [n for n in args.fault
               if n in split.get("train_sessions", [])]
    if overlap:
        raise SystemExit(f"--fault session(s) were in the TRAINING pool: "
                         f"{overlap}\nA fault the model trained on is not a "
                         f"detection result. Retrain without them first.")

    # Healthy reference: the artifact's held-out session unless overridden.
    err_va = scores(model, np.load(C.ARTIFACTS_DIR / "val_features.npy"),
                    mu, sd)
    if args.healthy:
        err_h = np.concatenate([
            scores(model, F.load_session_features(n)["features"], mu, sd)
            for n in args.healthy])
        healthy_desc = args.healthy
    else:
        err_h = scores(model, np.load(C.ARTIFACTS_DIR / "test_features.npy"),
                       mu, sd)
        healthy_desc = [split.get("holdout", "test split")]

    # Thresholds, exactly as ml.evaluate derives them.
    thr_p99 = float(np.percentile(err_va, 99))
    thr_3sig = float(err_va.mean() + 3 * err_va.std())
    thr_int8 = None
    qpath = C.ARTIFACTS_DIR / "quantize.json"
    if qpath.exists():
        with open(qpath) as f:
            thr_int8 = json.load(f).get("thr_int8_p99")

    rehearsal = [n for n in args.fault
                 if operating_point(n).startswith("healthy")]
    if rehearsal:
        print("=" * 76)
        print("REHEARSAL: these --fault sessions are labelled healthy:")
        for n in rehearsal:
            print(f"  {n}")
        print("Their numbers demonstrate SENSITIVITY TO CHANGE (different")
        print("operating point), not fault detection. Real detection rates")
        print("need seeded-fault sessions.")
        print("=" * 76)

    print(f"\nHealthy reference : {healthy_desc} ({len(err_h):,} windows)")
    print(f"Thresholds (val)  : p99={thr_p99:.5f}  3sigma={thr_3sig:.5f}")
    if thr_int8 is not None:
        print(f"int8 threshold    : {thr_int8:.5f} (from quantize.json -- the "
              f"deployed one)")
    else:
        print(f"int8 threshold    : not available; run ml.quantize to get the "
              f"deployed operating point")
    print(f"Healthy FPR       : p99 {100 * (err_h > thr_p99).mean():.2f} % | "
          f"3sigma {100 * (err_h > thr_3sig).mean():.2f} % | "
          f"confirmed({args.k}) "
          f"{100 * confirmed_alarm_rate(err_h, thr_p99, args.k):.2f} %")

    results = {
        "healthy_reference": healthy_desc,
        "n_healthy_windows": int(len(err_h)),
        "thr_p99": thr_p99,
        "thr_3sigma": thr_3sig,
        "thr_int8_p99": thr_int8,
        "k_consecutive": args.k,
        "healthy_fpr_p99": float((err_h > thr_p99).mean()),
        "healthy_fpr_confirmed": confirmed_alarm_rate(err_h, thr_p99, args.k),
        "rehearsal_sessions": rehearsal,
        "faults": {},
    }

    per_fault_err = {}
    print(f"\n{'session':<48} {'AUC':>6} {'det p99':>8} {'conf':>7} "
          f"{'1st alarm':>10}")
    print("-" * 84)
    for name in args.fault:
        err_f = scores(model, F.load_session_features(name)["features"],
                       mu, sd)
        per_fault_err[name] = err_f
        hop_s = hop_seconds(name)
        auc = auc_mann_whitney(err_h, err_f)
        det = float((err_f > thr_p99).mean())
        conf = confirmed_alarm_rate(err_f, thr_p99, args.k)
        first, first_k = first_alarm(err_f, thr_p99, args.k)
        first_s = first_k * hop_s if first_k >= 0 else None
        results["faults"][name] = {
            "operating_point": operating_point(name),
            "n_windows": int(len(err_f)),
            "auc": auc,
            "detection_rate_p99": det,
            "detection_rate_3sigma": float((err_f > thr_3sig).mean()),
            "confirmed_rate": conf,
            "first_alarm_window": first,
            "first_confirmed_window": first_k,
            "first_confirmed_seconds": first_s,
            "mean_score_ratio_vs_healthy":
                float(err_f.mean() / max(err_h.mean(), 1e-12)),
            "is_rehearsal": name in rehearsal,
        }
        when = f"{first_s:.1f}s" if first_s is not None else "never"
        print(f"{name:<48} {auc:>6.3f} {det:>7.2%} {conf:>6.2%} {when:>10}")
    print("-" * 84)

    pooled = np.concatenate(list(per_fault_err.values()))
    results["pooled"] = {
        "n_windows": int(len(pooled)),
        "auc": auc_mann_whitney(err_h, pooled),
        "detection_rate_p99": float((pooled > thr_p99).mean()),
    }
    print(f"pooled AUC {results['pooled']['auc']:.3f} | "
          f"detection at p99 {results['pooled']['detection_rate_p99']:.2%} | "
          f"healthy FPR {results['healthy_fpr_p99']:.2%}")
    if rehearsal:
        print("(rehearsal sessions included -- label accordingly)")

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(C.REPO_ROOT)}")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bins = np.histogram_bin_edges(
        np.concatenate([err_h, pooled]), bins=120)
    axes[0].hist(err_h, bins=bins, alpha=0.55, density=True,
                 label="healthy (held out)")
    for name, err_f in per_fault_err.items():
        axes[0].hist(err_f, bins=bins, alpha=0.55, density=True,
                     label=operating_point(name))
    axes[0].axvline(thr_p99, color="r", ls=":", lw=1, label="val p99")
    axes[0].set_yscale("log")
    axes[0].set_title("Score distributions (log density)")
    axes[0].set_xlabel("reconstruction MSE")
    axes[0].legend(fontsize=8)

    for name, err_f in per_fault_err.items():
        fpr, tpr = roc_curve(err_h, err_f)
        axes[1].plot(fpr, tpr, lw=1.2,
                     label=f"{operating_point(name)} "
                           f"(AUC {results['faults'][name]['auc']:.3f})")
    axes[1].plot([0, 1], [0, 1], color="k", ls=":", lw=0.8)
    axes[1].set_title("ROC vs held-out healthy" +
                      (" -- REHEARSAL" if rehearsal else ""))
    axes[1].set_xlabel("false-positive rate (healthy)")
    axes[1].set_ylabel("detection rate (fault)")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PLOTS_DIR / "eval_faults.png"
    fig.savefig(out, dpi=120)
    print(f"Wrote {out.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
