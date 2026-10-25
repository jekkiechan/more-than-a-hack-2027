"""Bounded LightGBM comparison against the short-spell-weighted physical HGB.

Same features, augmentation, beta-1.0 spell weights and decoder as
refine_weighted.py; only the gradient-boosting library and its settings change.
The HGB baseline is refitted in the same run so the comparison is like for like.
Five LightGBM configurations are predeclared below (defaults plus four
Kaggle-style settings); each is scored with switch costs 1, 3 and 8 on
validation J03WOY only. Test paths are never constructed or opened.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import refine_features, refine_weighted, robust, supervised  # noqa: E402

TRAIN = refine_weighted.TRAIN
VALIDATION = refine_weighted.VALIDATION
BETA = 1.0
SWITCH_COSTS = (1, 3, 8)
SEED = 17

# Fixed before any fit. "kaggle_*" follow common competition practice:
# lower learning rate with more trees, row/feature subsampling, L2, larger leaves.
CONFIGS = {
    "hgb_baseline": dict(library="sklearn", max_iter=260, max_leaf_nodes=63, learning_rate=0.08,
                         min_samples_leaf=35, l2_regularization=1),
    "lgbm_default": dict(library="lightgbm"),
    "lgbm_kaggle_a": dict(library="lightgbm", n_estimators=1000, learning_rate=0.03, num_leaves=63,
                          min_child_samples=50, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.7, reg_lambda=2.0),
    "lgbm_kaggle_b": dict(library="lightgbm", n_estimators=1500, learning_rate=0.02, num_leaves=127,
                          min_child_samples=100, subsample=0.7, subsample_freq=1,
                          colsample_bytree=0.6, reg_lambda=5.0, max_bin=255),
    "lgbm_kaggle_c": dict(library="lightgbm", n_estimators=800, learning_rate=0.05, num_leaves=31,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0),
    "lgbm_kaggle_d": dict(library="lightgbm", n_estimators=600, learning_rate=0.05, num_leaves=255,
                          max_depth=12, min_child_samples=200, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, reg_lambda=10.0, min_split_gain=0.01),
}


def make_estimator(config):
    config = dict(config)
    library = config.pop("library")
    if library == "sklearn":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **config)
    from lightgbm import LGBMClassifier
    return LGBMClassifier(objective="multiclass", random_state=SEED, n_jobs=1, verbose=-1, **config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "data/external_audit/sportec/processed")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/experiments/refine_weighted",
                        help="Reuse refine_weighted's cached training/validation features if present")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/refine_lgbm")
    parser.add_argument("--only", nargs="+", choices=sorted(CONFIGS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    spec = {"created_utc": datetime.now(timezone.utc).isoformat(), "training_matches": TRAIN,
            "validation_matches": VALIDATION, "beta": BETA, "switch_costs": SWITCH_COSTS,
            "configs": CONFIGS, "selection": robust.SELECTION, "test_access": "none"}
    spec_path = args.out / "predeclared.json"
    if not spec_path.exists():
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    paths = [args.processed / f"{name}.npz" for name in TRAIN]
    cache_path = args.cache / "training_features.joblib"
    if cache_path.exists():
        cache = joblib.load(cache_path)
        x, y, names = cache["x"], cache["y"], cache["names"]
    else:
        x, y, names, manifest = refine_features.prepare_training(paths)
        args.cache.mkdir(parents=True, exist_ok=True)
        joblib.dump(dict(x=x, y=y, names=names, manifest=manifest), cache_path)
    weights, _ = refine_weighted.augmented_weights(paths, y, BETA)

    validation, val_features = [], {}
    for name in VALIDATION:
        data = supervised.load(args.processed / f"{name}.npz")
        validation.append((name, data))
        val_features[name] = {}
        for condition, view in (("clean", robust.tracking_inputs(data)),
                                ("masked", robust.masked_inputs(data, 2026))):
            path = args.cache / f"{name}_{condition}_features.npy"
            if path.exists():
                matrix = np.load(path)
            else:
                matrix, these = refine_features.features(view)
                if these != names:
                    raise ValueError("Validation schema differs")
                np.save(path, matrix)
            val_features[name][condition] = (view, matrix)

    results_path = args.out / "validation.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else []
    done = {r["config"] for r in results}
    for key, config in CONFIGS.items():
        if (args.only and key not in args.only) or key in done:
            continue
        started = time.monotonic()
        estimator = make_estimator(config)
        with threadpool_limits(limits=1):
            estimator.fit(x, y, sample_weight=weights)
        fit_s = time.monotonic() - started
        model = {"estimator": estimator, "feature_names": names, "classes": estimator.classes_.tolist()}
        probabilities = {n: {c: (view, refine_features.probabilities_from_features(m, names, model))
                             for c, (view, m) in views.items()} for n, views in val_features.items()}
        for cost in SWITCH_COSTS:
            candidate = robust.evaluate_cost(validation, probabilities, cost)
            row = {"config": key, "switch_cost": cost, "fit_seconds": round(fit_s, 1),
                   **{c: {k: candidate[c][k] for k in ("mean_macro_f1", "mean_spell_f1")}
                      for c in ("clean", "masked")}}
            results.append(row)
            print(json.dumps(row), flush=True)
        results_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
