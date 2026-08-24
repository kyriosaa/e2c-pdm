"""
Leave-one-session-out sweep: hold out EVERY healthy session in turn.

Why this exists
---------------
`ml.evaluate` reports one held-out false-positive rate, from one holdout. The
current artifact says 7.3 % against a nominal 1 % p99 threshold. One number
cannot distinguish between the two explanations that matter:

  * session-to-session variation really is ~7 % of windows, in which case a
    seeded fault has to move more than that to be detectable at all; or
  * one session is an outlier and the rest sit near 1 %.

Those imply completely different fault campaigns. Holding out each session in
turn gives the distribution instead of a sample of one, and the spread is the
number to quote -- it is the floor on detectability
(docs/notes/quirk_register.md 3.3).

Nothing here is a detection rate. No fault data exists; every fold is healthy
data scored against a healthy threshold, so every fold reports a false-positive
operating point only.

The validation set, and why it is a flag
----------------------------------------
`ml.train_autoencoder` takes the last 10 % of the CONCATENATED training pool as
validation, which in practice is the tail of one session. The p99 threshold is
therefore set by that one session. --val-mode controls this:

  tail        reproduce train_autoencoder exactly. The default, so the sweep
              is comparable with the number already in ml/artifacts/eval.json.
  stratified  last 10 % of EACH training session, so validation spans the pool.

If the spread collapses under `stratified`, the variation was in the threshold,
not in the data. That is worth knowing before blaming the rig.

This writes no model artifacts and does not touch the ones ml.train_autoencoder
produced.

Run (hours; leave it):
    python -m ml.studies.loso_sweep
    python -m ml.studies.loso_sweep --epochs 10          quick shape-check
    python -m ml.studies.loso_sweep --val-mode stratified
    python -m ml.studies.loso_sweep --resume             continue after a crash
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime

import numpy as np

from .. import config as C
from .. import features as F
from ..train_autoencoder import (assemble, build_model, operating_point,
                                 require_tf)

OUT_JSON = C.ARTIFACTS_DIR / "loso_sweep.json"


# ---------------------------------------------------------------------------
# Pool selection
# ---------------------------------------------------------------------------
def default_pool() -> list[str]:
    """Usable cached sessions, restricted to the largest operating point.

    Mixing operating points would make the sweep measure voltage as much as
    night-to-night variation. The register's pool is the twelve 24 V sessions,
    with 15 V reserved for the novelty check; taking the largest group
    reproduces that without hardcoding a voltage.
    """
    names = F.usable_sessions()
    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(operating_point(n), []).append(n)
    if not groups:
        return []
    best = max(groups.values(), key=len)
    dropped = [n for n in names if n not in best]
    if dropped:
        print(f"[info] pool restricted to {operating_point(best[0])} "
              f"({len(best)} sessions); excluded {len(dropped)}: {dropped}")
        print(f"       Pass --sessions explicitly to override.")
    return best


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------
def split_mask(counts: list[int], val_mode: str) -> np.ndarray:
    """Boolean mask over the concatenated training pool: True = validation.

    tail        last 10 % of the whole pool (what train_autoencoder does)
    stratified  last 10 % of each session
    """
    total = int(sum(counts))
    mask = np.zeros(total, dtype=bool)
    if val_mode == "tail":
        mask[int(total * 0.9):] = True
        return mask
    off = 0
    for n in counts:
        mask[off + int(n * 0.9):off + n] = True
        off += n
    return mask


def batched_mse(model, x: np.ndarray, batch: int = 4096) -> np.ndarray:
    """Per-row reconstruction MSE. x is already standardised."""
    out = np.empty(len(x), dtype=np.float32)
    for i in range(0, len(x), batch):
        blk = x[i:i + batch]
        recon = model.predict(blk, batch_size=batch, verbose=0)
        out[i:i + batch] = np.mean((blk - recon) ** 2, axis=1)
    return out


def run_fold(tf, pool: list[str], holdout: str, epochs: int,
             val_mode: str) -> dict:
    t0 = time.time()
    np.random.seed(C.RANDOM_SEED)
    tf.random.set_seed(C.RANDOM_SEED)

    train_names = [n for n in pool if n != holdout]
    counts = []
    for n in train_names:
        with open(C.FEATURES_DIR / n / "meta.json") as f:
            counts.append(json.load(f)["n_windows"])

    X, _, _ = assemble(train_names)
    val_mask = split_mask(counts, val_mode)
    if len(val_mask) != len(X):
        raise SystemExit(f"meta.json window counts disagree with the loaded "
                         f"features on fold {holdout}: "
                         f"{len(val_mask)} vs {len(X)}")

    # Standardise on the TRAIN rows only, then in place -- a second full copy
    # of a 400k x 1024 float32 matrix is 1.7 GB nobody has to spend.
    mu = X[~val_mask].mean(axis=0)
    sd = X[~val_mask].std(axis=0) + 1e-6
    X -= mu
    X /= sd
    x_tr, x_va = X[~val_mask], X[val_mask]

    model = build_model(X.shape[1])
    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                             patience=4),
    ]
    hist = model.fit(x_tr, x_tr, validation_data=(x_va, x_va), epochs=epochs,
                     batch_size=C.BATCH_SIZE, shuffle=True, callbacks=cb,
                     verbose=2)

    err_va = batched_mse(model, x_va)
    del X, x_tr, x_va
    gc.collect()

    X_te, _, _ = assemble([holdout])
    X_te -= mu
    X_te /= sd
    err_te = batched_mse(model, X_te)
    n_test = len(X_te)
    del X_te
    gc.collect()

    thr_p99 = float(np.percentile(err_va, 99))
    thr_3sig = float(err_va.mean() + 3 * err_va.std())
    res = {
        "holdout": holdout,
        "n_train": int((~val_mask).sum()),
        "n_val": int(val_mask.sum()),
        "n_test": int(n_test),
        "epochs_run": len(hist.history["loss"]),
        "thr_p99": thr_p99,
        "thr_3sigma": thr_3sig,
        "fpr_p99": float((err_te > thr_p99).mean()),
        "fpr_3sigma": float((err_te > thr_3sig).mean()),
        "val_mean": float(err_va.mean()),
        "holdout_mean": float(err_te.mean()),
        "holdout_p99": float(np.percentile(err_te, 99)),
        "seconds": round(time.time() - t0, 1),
    }
    res["score_ratio"] = res["holdout_mean"] / max(res["val_mean"], 1e-12)

    tf.keras.backend.clear_session()
    del model
    gc.collect()
    return res


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def summarise(folds: list[dict]) -> dict:
    f99 = np.array([f["fpr_p99"] for f in folds])
    f3s = np.array([f["fpr_3sigma"] for f in folds])
    worst = max(folds, key=lambda f: f["fpr_p99"])
    best = min(folds, key=lambda f: f["fpr_p99"])
    return {
        "n_folds": len(folds),
        "fpr_p99_median": float(np.median(f99)),
        "fpr_p99_iqr": [float(np.percentile(f99, 25)),
                        float(np.percentile(f99, 75))],
        "fpr_p99_min": float(f99.min()),
        "fpr_p99_max": float(f99.max()),
        "fpr_3sigma_median": float(np.median(f3s)),
        "worst_session": worst["holdout"],
        "best_session": best["holdout"],
    }


def print_table(folds: list[dict], summary: dict) -> None:
    print("")
    print(f"{'holdout':<48} {'windows':>9} {'ep':>3} "
          f"{'FPR p99':>9} {'FPR 3sig':>9} {'score x':>8}")
    print("-" * 92)
    for f in sorted(folds, key=lambda f: f["fpr_p99"]):
        print(f"{f['holdout']:<48} {f['n_test']:>9,} {f['epochs_run']:>3} "
              f"{f['fpr_p99']:>8.2%} {f['fpr_3sigma']:>9.2%} "
              f"{f['score_ratio']:>8.2f}")
    print("-" * 92)
    lo, hi = summary["fpr_p99_iqr"]
    print(f"median held-out FPR (p99 threshold) : "
          f"{summary['fpr_p99_median']:.2%}")
    print(f"interquartile range                 : {lo:.2%} - {hi:.2%}")
    print(f"full range                          : "
          f"{summary['fpr_p99_min']:.2%} - {summary['fpr_p99_max']:.2%}")
    print(f"worst session                       : {summary['worst_session']}")
    print("")
    print("These are FALSE-POSITIVE operating points on healthy data. No")
    print("detection rate is computable without fault data. The MEDIAN is the")
    print("floor on detectability: a seeded fault must flag more windows than")
    print("this to be visible at all.")


def plot(folds: list[dict], summary: dict, val_mode: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = sorted(folds, key=lambda f: f["fpr_p99"])
    labels = [f["holdout"].replace("healthy", "").replace("_motor_baseline", "")
              for f in order]
    fpr = [f["fpr_p99"] for f in order]
    ratio = [f["score_ratio"] for f in order]
    med = summary["fpr_p99_median"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    axes[0].bar(range(len(order)), fpr, color="tab:blue", alpha=0.8)
    axes[0].axhline(med, color="r", ls="--", lw=1, label=f"median {med:.2%}")
    axes[0].axhline(0.01, color="k", ls=":", lw=1,
                    label="nominal 1 % (p99 threshold)")
    axes[0].set_ylabel("held-out false-positive rate")
    axes[0].set_title(f"Leave-one-session-out sweep, {len(order)} healthy "
                      f"folds (val-mode={val_mode})")
    axes[0].legend(fontsize=8)

    axes[1].bar(range(len(order)), ratio, color="tab:orange", alpha=0.8)
    axes[1].axhline(1.0, color="k", ls=":", lw=1,
                    label="held-out = validation")
    axes[1].set_ylabel("mean score, held-out / validation")
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = C.PLOTS_DIR / "loso_sweep.png"
    fig.savefig(out, dpi=120)
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="pool to sweep; default = largest operating point")
    ap.add_argument("--holdouts", nargs="*", default=None,
                    help="only hold these out; default = every pool session")
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--val-mode", choices=("tail", "stratified"),
                    default="tail")
    ap.add_argument("--resume", action="store_true",
                    help="keep folds already in loso_sweep.json")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    pool = args.sessions or default_pool()
    if args.sessions:
        F.require_raw_counterpart(pool)
    if len(pool) < 2:
        raise SystemExit(f"Need >= 2 cached sessions to sweep, found "
                         f"{len(pool)}. Run: python -m ml.features")

    holdouts = args.holdouts or list(pool)
    unknown = [h for h in holdouts if h not in pool]
    if unknown:
        raise SystemExit(f"holdout(s) not in the pool: {unknown}")

    tf = require_tf()

    folds: list[dict] = []
    if args.resume and OUT_JSON.exists():
        with open(OUT_JSON) as f:
            prev = json.load(f)
        if prev.get("val_mode") == args.val_mode and prev.get("pool") == pool:
            folds = prev.get("folds", [])
            done = {f["holdout"] for f in folds}
            holdouts = [h for h in holdouts if h not in done]
            print(f"[resume] {len(folds)} fold(s) already done; "
                  f"{len(holdouts)} to go")
        else:
            print(f"[resume] existing {OUT_JSON.name} used a different pool or "
                  f"val-mode; starting fresh")

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    for i, holdout in enumerate(holdouts, 1):
        print("")
        print(f"=== fold {i}/{len(holdouts)}  holdout={holdout}")
        folds.append(run_fold(tf, pool, holdout, args.epochs, args.val_mode))
        # Written after every fold: the sweep is hours long and a crash in
        # fold 9 must not cost folds 1-8.
        with open(OUT_JSON, "w") as f:
            json.dump({"protocol": "leave-one-session-out sweep",
                       "generated": datetime.now().isoformat(timespec="seconds"),
                       "pool": pool,
                       "val_mode": args.val_mode,
                       "epochs_max": args.epochs,
                       "folds": folds,
                       "summary": summarise(folds)}, f, indent=2)
        r = folds[-1]
        print(f"    FPR p99 {r['fpr_p99']:.2%} | 3sigma {r['fpr_3sigma']:.2%} "
              f"| {r['epochs_run']} epochs | {r['seconds']:.0f}s")

    summary = summarise(folds)
    print_table(folds, summary)
    print(f"Wrote {OUT_JSON.relative_to(C.REPO_ROOT)}")
    if not args.no_plot:
        out = plot(folds, summary, args.val_mode)
        print(f"Wrote {out.relative_to(C.REPO_ROOT)}")


if __name__ == "__main__":
    main()
