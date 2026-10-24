"""One final cached-feature capacity check; no new feature or decoder search.

Train HGB 127 leaves / 400 iterations on the existing uniform-weight augmented
training rows. Compare uniform penalties 1/3/8 and the already fixed team/dead
penalties 3/8. Validation only; no code path extracts or opens test data.
"""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

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
from experiments import refine_decode, refine_features, robust, supervised  # noqa: E402

TRAIN = ("J03WMX", "J03WN1", "J03WOH")
VALIDATION = "J03WOY"
CAPACITY = {"leaves": 127, "iterations": 400}
DECODERS = (
    {"decoder": "uniform", "switch_cost": 1},
    {"decoder": "uniform", "switch_cost": 3},
    {"decoder": "uniform", "switch_cost": 8},
    {"decoder": "separate", "switch_cost": 3, "team_cost": 3., "dead_cost": 8., "boundary_power": 0.},
)


def decode(probabilities, data, config):
    """Explicit float32 decoding matches the stored validation probabilities."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    periods, fps = supervised.model_periods(data), float(data["fps"])
    if config["decoder"] == "uniform":
        return supervised.decode(probabilities, periods, fps, config["switch_cost"])
    if config["decoder"] != "separate":
        raise ValueError("Unknown decoder")
    return refine_decode.decode(probabilities, periods, fps, config)


def predict_probabilities(data, model):
    return refine_features.predict_probabilities(robust.tracking_inputs(data), model).astype(np.float32)


def predict(data, model):
    inputs = robust.tracking_inputs(data)
    return decode(predict_probabilities(inputs, model), inputs, model["config"])


def evaluate(validation, views, probabilities, config):
    candidate = {"config": config}
    for condition in ("clean", "masked"):
        score = robust.compact_score(validation, decode(probabilities[condition], views[condition], config))
        candidate[condition] = {**robust.aggregate({VALIDATION: score}),
                                "scores": {VALIDATION: score},
                                "ball_missing_fraction": {VALIDATION: robust.missing_fraction(views[condition])}}
    return candidate


def self_test():
    rng = np.random.default_rng(7)
    probability = rng.dirichlet([1, 1, 1], 40).astype(np.float32)
    data = {"period": np.r_[np.ones(20), np.full(20, 2)],
            "timestamps": np.arange(40) / 10, "fps": np.array(10.)}
    a = decode(probability, data, {"decoder": "uniform", "switch_cost": 3})
    b = decode(probability, data, {"decoder": "separate", "team_cost": 3., "dead_cost": 3., "boundary_power": 0.})
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, decode(probability, dict(data, labels=np.full(40, "D")), DECODERS[1]))
    print("CHECKS decoder equivalence and target independence passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/experiments/refine_weighted")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/external_audit/sportec/processed")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/refine_scale")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--declare", action="store_true", help="Write protocol before caches are ready, without fitting")
    args = parser.parse_args()
    self_test()
    if args.check:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "frozen_config.json").exists():
        raise ValueError("Capacity check already frozen; do not overwrite")
    specification = {"created_utc": datetime.now(timezone.utc).isoformat(),
                     "train": TRAIN, "validation": VALIDATION, "capacity": CAPACITY,
                     "decoders": DECODERS, "sample_weights": "uniform",
                     "feature_count": 137, "training_rows": 283164,
                     "augmentation": "clean + 41% block-masked; each with team swap",
                     "selection": robust.SELECTION, "stress_used_for_selection": False,
                     "cache_directory": str(args.cache), "test_access": "none"}
    declaration = args.out / "predeclared.json"
    if declaration.exists():
        previous = json.loads(declaration.read_text())
        specification["created_utc"] = previous["created_utc"]
        if previous != json.loads(json.dumps(specification)):
            raise ValueError("The predeclared capacity experiment changed")
    else:
        declaration.write_text(json.dumps(specification, indent=2) + "\n")
    if args.declare:
        print("DECLARED", declaration, flush=True)
        return
    # Wait for the cache owner to signal completion before invoking this CLI.
    cache_path = args.cache / "training_features.joblib"
    val_paths = {condition: args.cache / f"{VALIDATION}_{condition}_features.npy"
                 for condition in ("clean", "masked")}
    for path in (cache_path, *val_paths.values()):
        if not path.exists():
            raise ValueError(f"Cache not ready: {path}; no feature re-extraction fallback")
    started = time.monotonic()
    cache = joblib.load(cache_path)
    x, y, names, manifest = (cache[key] for key in ("x", "y", "names", "manifest"))
    if x.shape != (283164, 137) or len(names) != 137 or y.shape != (len(x),):
        raise ValueError("Unexpected cached training dimensions")
    if tuple(Path(item["path"]).stem for item in manifest) != TRAIN:
        raise ValueError("Cache contains non-training matches")
    for name, item in zip(TRAIN, manifest):
        if robust.sha256(args.processed / f"{name}.npz") != item["sha256"]:
            raise ValueError("Training data changed after caching")
    prior_frozen = json.loads((ROOT / "data/experiments/refine_features/frozen_config.json").read_text())
    for relative, expected in prior_frozen["source_sha256"].items():
        if robust.sha256(ROOT / relative) != expected:
            raise ValueError(f"Feature source changed after previous freeze: {relative}")
    validation_path = args.processed / f"{VALIDATION}.npz"
    validation = supervised.load(validation_path)
    views = {"clean": robust.tracking_inputs(validation), "masked": robust.masked_inputs(validation, 2026)}
    val_features = {condition: np.load(path, allow_pickle=False) for condition, path in val_paths.items()}
    for matrix in val_features.values():
        if matrix.shape != (len(validation["labels"]), 137):
            raise ValueError("Unexpected cached validation dimensions")
    print(f"FIT cached_rows={len(y)} features={len(names)} leaves=127 iterations=400", flush=True)
    estimator = HistGradientBoostingClassifier(
        max_iter=400, max_leaf_nodes=127, learning_rate=.08,
        min_samples_leaf=35, l2_regularization=1, early_stopping=False, random_state=17)
    with threadpool_limits(limits=1):
        estimator.fit(x, y)
    model = {"estimator": estimator, "feature_names": names, "classes": estimator.classes_.tolist(),
             "training": manifest,
             "config": {**CAPACITY, "train_fps": 4, "two_stage": False, "sample_weights": "uniform",
                        "augmentation": specification["augmentation"], "random_state": 17,
                        "features_api": "experiments.refine_features.features",
                        "prediction_api": "experiments.refine_scale.predict"}}
    probabilities = {condition: refine_features.probabilities_from_features(matrix, names, model).astype(np.float32)
                     for condition, matrix in val_features.items()}
    candidates = []
    for config in DECODERS:
        candidate = evaluate(validation, views, probabilities, config)
        candidates.append(candidate)
        print(json.dumps({"config": config,
                          **{condition: {key: candidate[condition][key]
                                         for key in ("mean_macro_f1", "mean_spell_f1")}
                             for condition in ("clean", "masked")}}), flush=True)
    selected = max(candidates, key=robust.selection_key)
    model["config"].update(selected["config"])
    model_path = args.out / "model.joblib"
    joblib.dump(model, model_path)
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "config": model["config"],
              "training": manifest, "train": TRAIN, "validation": VALIDATION,
              "selection": robust.SELECTION, "candidates": DECODERS,
              "test_evaluated": False, "stress_used_for_selection": False,
              "model_sha256": robust.sha256(model_path), "posterior_precision": "float32",
              "training_cache_sha256": robust.sha256(cache_path),
              "validation_cache_sha256": {condition: robust.sha256(path) for condition, path in val_paths.items()},
              "validation_data_sha256": robust.sha256(validation_path),
              "source_sha256": {"experiments/refine_scale.py": robust.sha256(__file__),
                                "experiments/refine_decode.py": robust.sha256(ROOT / "experiments/refine_decode.py"),
                                **prior_frozen["source_sha256"]},
              "runtime_s": time.monotonic()-started}
    (args.out / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (args.out / "validation.json").write_text(json.dumps({"selected": selected, "candidates": candidates}, indent=2) + "\n")
    for condition, probability in probabilities.items():
        np.savez_compressed(args.out / f"{VALIDATION}_{condition}_probabilities.npz",
                            probabilities=probability, period=validation["period"],
                            timestamps=validation["timestamps"], fps=validation["fps"])
    print("FROZEN", json.dumps({"config": model["config"], "model_sha256": frozen["model_sha256"],
                                "seconds": frozen["runtime_s"]}), flush=True)


if __name__ == "__main__":
    main()
