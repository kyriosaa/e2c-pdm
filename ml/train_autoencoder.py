"""
Trains a dense autoencoder on healthy-motor vibration PSD windows.

Two split protocols, and the choice matters more than any hyperparameter:

  --split session  (DEFAULT, and the one to report)
      Leave-one-session-out. Train on whole sessions, hold out a different
      session entirely. This measures what the thesis actually claims: that the
      healthy model generalises to a NEW recording of the same healthy machine.

  --split time
      First 80 % / next 10 % / last 10 % of one session, in time order.
      Useful for a within-session drift check, but it FLATTERS the model badly:
      windows overlap by 50 % and consecutive windows of a steady-state motor
      are nearly identical, so train and test are near-duplicates. Do not
      report this number as generalisation performance.

Neither protocol can produce a detection rate. That needs seeded-fault data --
see docs/notes/collection_protocol.md.

Run:
    python -m ml.train_autoencoder --holdout healthy24V_motor_baseline_20260719_010905
"""

from __future__ import annotations

import json

import numpy as np

from . import config as C
from . import features as F


def require_tf():
    """Import TensorFlow with an actionable message if it is unavailable."""
    try:
        import tensorflow as tf
        return tf
    except ImportError as e:
        import sys
        raise SystemExit(
            f"TensorFlow is not importable ({e}).\n"
            f"You are on Python {sys.version_info.major}."
            f"{sys.version_info.minor}. TensorFlow does not publish wheels for "
            f"Python 3.14 yet, so this step cannot run on the default "
            f"interpreter.\n\n"
            f"Create a separate training environment on Python 3.12:\n"
            f"    py -3.12 -m venv .venv-train\n"
            f"    .venv-train\\Scripts\\activate\n"
            f"    pip install -r ml/requirements.txt\n\n"
            f"Feature extraction (ml.sessions, ml.data_loader, ml.features) "
            f"needs only numpy+scipy and runs fine on 3.14, so the cached "
            f"features under data/interim/features/ are reusable as-is.")


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
def available_sessions() -> list[str]:
    """Cached sessions with a recording behind them, orphans excluded."""
    return F.usable_sessions()


def assemble(session_names: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Concatenate feature matrices. Returns (X, session_id_per_row, metas)."""
    mats, ids, metas = [], [], {}
    for i, name in enumerate(session_names):
        d = F.load_session_features(name)
        mats.append(d["features"])
        ids.append(np.full(len(d["features"]), i, dtype=np.int16))
        metas[name] = d["meta"]
    if not mats:
        raise SystemExit(
            f"No cached features found in {C.FEATURES_DIR}. "
            f"Run: python -m ml.features")
    # All sessions must share the feature layout or the model input is
    # meaningless across them.
    widths = {m["n_bins"] for m in metas.values()}
    if len(widths) > 1:
        raise SystemExit(f"Feature width mismatch across sessions: {widths}. "
                         f"Re-run ml.features with one AXIS_COMBINE / "
                         f"FFT_BINS_KEEP setting.")
    return np.concatenate(mats), np.concatenate(ids), metas


def operating_point(session_name: str) -> str:
    """Crude operating-point key from the session label ('healthy15V', ...).

    Good enough to catch the common mistake below. Once sessions carry
    motor_voltage_v / fault_condition in session.json, key on those instead.
    """
    return session_name.split("_")[0]


def warn_mixed_operating_points(train_names: list[str], holdout: str) -> None:
    """Flag a training pool that spans several operating points.

    Mixing operating points in training is a legitimate choice -- it buys a model
    robust to both -- but it has a consequence people miss: any session at one of
    those operating points is then IN-distribution, so it can no longer serve as
    the off-manifold novelty check in ml/evaluate.py.

    With the three pilot sessions the clean protocol is: train on the two 24 V
    sessions, hold one out, and keep 15 V entirely unseen as the novelty check.
    """
    points = sorted({operating_point(n) for n in train_names})
    if len(points) > 1:
        print(f"[warn] training pool spans {len(points)} operating points: "
              f"{points}")
        print(f"       Those are now in-distribution and cannot be used as a "
              f"novelty check.")
        same = [n for n in train_names
                if operating_point(n) == operating_point(holdout)]
        print(f"       For a clean protocol, restrict training to the holdout's "
              f"operating point:")
        print(f"         --sessions {' '.join(same + [holdout])} "
              f"--holdout {holdout}")
        print(f"       and pass the other session(s) to "
              f"`ml.evaluate --novelty`.")


def time_split(n: int) -> tuple[slice, slice, slice]:
    i_tr = int(n * C.TRAIN_FRACTION)
    i_va = int(n * (C.TRAIN_FRACTION + C.VAL_FRACTION))
    return slice(0, i_tr), slice(i_tr, i_va), slice(i_va, n)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(n_bins: int):
    tf = require_tf()
    from tensorflow.keras import layers

    inp = layers.Input(shape=(n_bins,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    z = layers.Dense(C.BOTTLENECK_UNITS, activation="relu",
                     name="bottleneck")(x)
    x = layers.Dense(64, activation="relu")(z)
    x = layers.Dense(256, activation="relu")(x)
    out = layers.Dense(n_bins, activation=None)(x)

    model = tf.keras.Model(inp, out, name="vib_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(C.LEARNING_RATE),
                  loss="mse")
    return model


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=("session", "time"), default="session")
    ap.add_argument("--holdout", default=None,
                    help="session name to hold out (--split session); "
                         "default = the last one alphabetically")
    ap.add_argument("--sessions", nargs="*", default=None,
                    help="restrict to these sessions; default = all cached")
    args = ap.parse_args()

    tf = require_tf()
    np.random.seed(C.RANDOM_SEED)
    tf.random.set_seed(C.RANDOM_SEED)

    names = args.sessions or available_sessions()
    F.require_raw_counterpart(names)
    if not names:
        raise SystemExit(f"No cached features in {C.FEATURES_DIR}. "
                         f"Run: python -m ml.features")

    if args.split == "session":
        if len(names) < 2:
            raise SystemExit(
                f"--split session needs >= 2 sessions, found {len(names)}. "
                f"Use --split time for a single-session drift check, but do "
                f"not report it as generalisation performance.")
        holdout = args.holdout or names[-1]
        if holdout not in names:
            raise SystemExit(f"holdout {holdout!r} not in {names}")
        train_names = [n for n in names if n != holdout]
        warn_mixed_operating_points(train_names, holdout)
        X_tr_all, _, metas = assemble(train_names)
        X_te, _, _ = assemble([holdout])
        # Last 10 % of the training pool (in time order) is validation.
        cut = int(len(X_tr_all) * 0.9)
        X_tr, X_va = X_tr_all[:cut], X_tr_all[cut:]
        split_meta = {"protocol": "leave-one-session-out",
                      "train_sessions": train_names, "holdout": holdout}
        print(f"Protocol : leave-one-session-out")
        print(f"  train  : {train_names}  ({len(X_tr):,} windows)")
        print(f"  val    : last 10 % of train pool ({len(X_va):,})")
        print(f"  test   : {holdout} ({len(X_te):,})")
    else:
        if len(names) != 1:
            print(f"[warn] --split time with {len(names)} sessions "
                  f"concatenates them; using {names[0]} only.")
            names = names[:1]
        X, _, metas = assemble(names)
        tr, va, te = time_split(len(X))
        X_tr, X_va, X_te = X[tr], X[va], X[te]
        split_meta = {"protocol": "within-session-time",
                      "session": names[0],
                      "train_end": tr.stop, "val_end": va.stop}
        print(f"Protocol : within-session time split on {names[0]}")
        print(f"  train {len(X_tr):,} | val {len(X_va):,} | test {len(X_te):,}")
        print(f"  [warn] 50 %-overlapped windows of a steady-state motor make "
              f"train/test near-duplicates. Not a generalisation estimate.")

    # Standardise on TRAIN statistics only.
    mu = X_tr.mean(axis=0)
    sd = X_tr.std(axis=0) + 1e-6
    x_tr = (X_tr - mu) / sd
    x_va = (X_va - mu) / sd

    model = build_model(X_tr.shape[1])
    model.summary()

    cb = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                            patience=4),
    ]
    model.fit(x_tr, x_tr, validation_data=(x_va, x_va), epochs=C.EPOCHS,
              batch_size=C.BATCH_SIZE, shuffle=True, callbacks=cb, verbose=2)

    C.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(C.ARTIFACTS_DIR / "vib_autoencoder.keras")
    np.savez(C.ARTIFACTS_DIR / "scaler.npz", mu=mu, sd=sd)
    np.save(C.ARTIFACTS_DIR / "test_features.npy", X_te)
    np.save(C.ARTIFACTS_DIR / "val_features.npy", X_va)
    with open(C.ARTIFACTS_DIR / "split.json", "w") as f:
        json.dump({**split_meta,
                   "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
                   "n_test": int(len(X_te)),
                   "n_bins": int(X_tr.shape[1]),
                   "feature_meta": metas}, f, indent=2)
    print(f"Saved model, scaler, split -> "
          f"{C.ARTIFACTS_DIR.relative_to(C.REPO_ROOT)}/")


if __name__ == "__main__":
    main()
