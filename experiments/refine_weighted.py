"""Bounded training-only spell-duration weighting of the physical-feature HGB.

Two predeclared weighting strengths share one fixed backbone and six total
validation candidates. Test paths are never constructed or opened. Inference
uses experiments.refine_features.predict and has no access to spell targets.
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
from experiments import refine_features, robust, supervised  # noqa: E402

TRAIN = ("J03WMX", "J03WN1", "J03WOH")
VALIDATION = ("J03WOY",)
BETAS = (0.5, 1.0)
SWITCH_COSTS = (1, 3, 8)
LEAVES, ITERATIONS = 63, 260


def spell_weights(data, beta):
    """Return unnormalized per-frame weights from TRAINING labels only.

Runs are separated at label changes, period changes, and timestamp gaps.
H/A use clip((5 seconds / run duration)**beta, .25, 4); D/X use 1.
Inference never calls this function.
    """
    if beta not in BETAS:
        raise ValueError("Only the two predeclared weighting strengths are allowed")
    labels = np.asarray(data["labels"])
    periods = supervised.model_periods(data)
    fps = float(data["fps"])
    if labels.ndim != 1 or len(periods) != len(labels) or fps <= 0:
        raise ValueError("Invalid training label/time shape")
    weights = np.ones(len(labels), dtype=np.float64)
    changes = (labels[1:] != labels[:-1]) | (periods[1:] != periods[:-1])
    boundaries = np.r_[0, np.flatnonzero(changes) + 1, len(labels)]
    runs = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if labels[start] not in ("H", "A"):
            continue
        duration = (stop - start) / fps
        weight = float(np.clip((5.0 / duration) ** beta, 0.25, 4.0))
        weights[start:stop] = weight
        runs.append(duration)
    return weights, np.asarray(runs, dtype=float)


def augmented_weights(paths, expected_y, beta):
    """Match prepare_training's per-match clean/swap/masked/swap row order."""
    pieces, summaries, offset = [], [], 0
    for path in paths:
        data = supervised.load(path)
        labels = np.asarray(data["labels"])
        stride = max(1, round(float(data["fps"]) / 4))
        rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::stride]
        y = np.array([np.flatnonzero(supervised.LABELS == label)[0]
                      for label in labels[rows]], dtype=np.int8)
        opposite = np.where(y < 2, 1-y, y)
        expected = np.concatenate((y, opposite, y, opposite))
        if not np.array_equal(expected_y[offset:offset+len(expected)], expected):
            raise ValueError(f"Feature/target augmentation mapping disagrees for {path.stem}")
        full_weights, durations = spell_weights(data, beta)
        sampled = full_weights[rows]
        pieces.extend((sampled, sampled, sampled, sampled))
        summaries.append({"match": path.stem, "sampled_frames": len(rows),
                          "live_runs": len(durations), "live_runs_under_3s": int((durations < 3).sum()),
                          "raw_sample_weight_mean": float(sampled.mean()),
                          "raw_sample_weight_percentiles": np.percentile(sampled, [0, 25, 50, 75, 100]).tolist()})
        offset += len(expected)
    if offset != len(expected_y):
        raise ValueError("Unexpected augmented feature rows")
    weights = np.concatenate(pieces)
    normalizer = float(weights.mean())
    weights /= normalizer
    return weights, {"beta": beta, "normalizing_divisor": normalizer,
                     "normalized_mean": float(weights.mean()), "matches": summaries,
                     "normalized_weight_by_class": {
                         str(label): float(weights[expected_y == index].mean())
                         for index, label in enumerate(supervised.LABELS)}}


def self_test():
    labels = np.array(list("HHHHAADDXHH"))
    data = {"labels": labels, "period": np.array([1]*10 + [2]),
            "timestamps": np.arange(len(labels), dtype=float), "fps": np.array(1.0)}
    weights, durations = spell_weights(data, 1.0)
    np.testing.assert_allclose(weights[:4], 1.25)
    np.testing.assert_allclose(weights[4:6], 2.5)
    np.testing.assert_allclose(weights[6:9], 1)
    np.testing.assert_allclose(weights[9:], 4)
    np.testing.assert_allclose(durations, [4, 2, 1, 1])
    gap = dict(data, labels=np.array(list("HHHH")), period=np.ones(4),
               timestamps=np.array([0., 1., 4., 5.]))
    gap_weights, gap_durations = spell_weights(gap, 0.5)
    np.testing.assert_allclose(gap_durations, [2, 2])
    np.testing.assert_allclose(gap_weights, np.sqrt(2.5))
    print("CHECKS training-run duration, unknown/dead weights, period/time-gap separation passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "data/experiments/refinement_protocol.json")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/external_audit/sportec/processed")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/refine_weighted")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.check:
        return
    started = time.monotonic()
    split = json.loads(args.split.read_text())
    if tuple(split["train"]) != TRAIN or tuple(split["validation"]) != VALIDATION:
        raise ValueError("This round is restricted to the predeclared train/validation matches")
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "frozen_config.json").exists():
        raise ValueError("Already frozen; do not overwrite this round")
    spec = {"created_utc": datetime.now(timezone.utc).isoformat(),
            "training_matches": TRAIN, "validation_matches": VALIDATION,
            "betas": BETAS, "switch_costs": SWITCH_COSTS,
            "leaves": LEAVES, "iterations": ITERATIONS,
            "weight_formula": "H/A: clip((5 / run_duration_seconds)^beta, .25, 4); D:1; global mean normalized to1",
            "augmentation": "clean + 41% block-masked; each with team swap",
            "selection": robust.SELECTION, "stress_used_for_selection": False,
            "test_access": "none", "split_sha256": robust.sha256(args.split)}
    spec_path = args.out / "predeclared.json"
    if spec_path.exists():
        previous = json.loads(spec_path.read_text())
        spec["created_utc"] = previous["created_utc"]
        if previous != json.loads(json.dumps(spec)):
            raise ValueError("Existing predeclaration disagrees")
    else:
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    # Only allowlisted train/validation IDs below construct data paths.
    paths = [args.processed / f"{name}.npz" for name in TRAIN]
    cache_path = args.out / "training_features.joblib"
    if cache_path.exists():
        cache = joblib.load(cache_path)
        x, y, names, manifest = (cache[key] for key in ("x", "y", "names", "manifest"))
        for path, item in zip(paths, manifest):
            if robust.sha256(path) != item["sha256"]:
                raise ValueError("Training data changed after feature caching")
    else:
        x, y, names, manifest = refine_features.prepare_training(paths)
        joblib.dump(dict(x=x, y=y, names=names, manifest=manifest), cache_path)
    if len(names) != 137:
        raise ValueError("Expected the fixed 137-dimensional physical-feature API")

    validation, val_features = [], {}
    for name in VALIDATION:
        data = supervised.load(args.processed / f"{name}.npz")
        validation.append((name, data))
        val_features[name] = {}
        for condition, view in (("clean", robust.tracking_inputs(data)),
                                ("masked", robust.masked_inputs(data, 2026))):
            path = args.out / f"{name}_{condition}_features.npy"
            if path.exists():
                matrix = np.load(path)
            else:
                print(f"FEATURES {name} {condition}", flush=True)
                matrix, these_names = refine_features.features(view)
                if names != these_names:
                    raise ValueError("Validation schema differs")
                np.save(path, matrix)
            val_features[name][condition] = (view, matrix)
    print(f"TRAIN rows={len(y)} features={len(names)} fixed_capacity={LEAVES}/{ITERATIONS}", flush=True)
    candidates, weight_reports = [], []
    best = best_model = best_probabilities = None
    for beta in BETAS:
        weights, weight_report = augmented_weights(paths, y, beta)
        weight_reports.append(weight_report)
        print(f"FIT beta={beta} leaves={LEAVES} iterations={ITERATIONS}", flush=True)
        estimator = HistGradientBoostingClassifier(
            max_iter=ITERATIONS, max_leaf_nodes=LEAVES, learning_rate=0.08,
            min_samples_leaf=35, l2_regularization=1, early_stopping=False, random_state=17)
        with threadpool_limits(limits=1):
            estimator.fit(x, y, sample_weight=weights)
        model = {"estimator": estimator, "feature_names": names,
                 "classes": estimator.classes_.tolist(), "training": manifest,
                 "config": {"leaves": LEAVES, "iterations": ITERATIONS,
                            "weighting_beta": beta, "weighting_reference_s": 5,
                            "weighting_clip": [.25, 4], "weighting_global_mean": 1,
                            "switch_cost": 1, "train_fps": 4, "two_stage": False,
                            "random_state": 17, "augmentation": spec["augmentation"],
                            "features_api": "experiments.refine_features.features",
                            "prediction_api": "experiments.refine_features.predict"}}
        probabilities = {name: {condition: (view, refine_features.probabilities_from_features(matrix, names, model))
                               for condition, (view, matrix) in views.items()}
                         for name, views in val_features.items()}
        for cost in SWITCH_COSTS:
            candidate = robust.evaluate_cost(validation, probabilities, cost)
            candidate.update({"beta": beta, "leaves": LEAVES, "iterations": ITERATIONS})
            candidates.append(candidate)
            print(json.dumps({"beta": beta, "switch_cost": cost,
                  **{condition: {key: candidate[condition][key] for key in ("mean_macro_f1", "mean_spell_f1")}
                     for condition in ("clean", "masked")}}), flush=True)
            if best is None or robust.selection_key(candidate) > robust.selection_key(best):
                best = candidate
                best_model = dict(model, config=dict(model["config"], switch_cost=cost))
                best_probabilities = probabilities
        (args.out / "validation_progress.json").write_text(json.dumps(candidates, indent=2) + "\n")
        (args.out / "weight_summaries.json").write_text(json.dumps(weight_reports, indent=2) + "\n")

    model_path = args.out / "model.joblib"
    joblib.dump(best_model, model_path)
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(),
              "config": best_model["config"], "selection": robust.SELECTION,
              "betas": BETAS, "switch_costs": SWITCH_COSTS,
              "training": manifest, "split": split, "classes": best_model["classes"],
              "feature_count": len(names), "training_rows_after_augmentation": len(y),
              "model_sha256": robust.sha256(model_path),
              "predeclared_sha256": robust.sha256(spec_path),
              "stress_used_for_selection": False, "test_evaluated": False,
              "validation_mask_seed": 2026,
              "source_sha256": {str(p.relative_to(ROOT)): robust.sha256(p) for p in (
                  Path(__file__), ROOT / "experiments/refine_features.py",
                  ROOT / "experiments/robust.py", ROOT / "experiments/supervised.py",
                  ROOT / "experiments/stress.py", ROOT / "experiments/metrics.py")},
              "validation_data_sha256": {name: robust.sha256(args.processed / f"{name}.npz")
                                         for name in VALIDATION}}
    (args.out / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (args.out / "validation.json").write_text(json.dumps({"selected": best, "candidates": candidates}, indent=2) + "\n")
    for name, conditions in best_probabilities.items():
        for condition, (view, probability) in conditions.items():
            np.savez_compressed(args.out / f"{name}_{condition}_probabilities.npz",
                                probabilities=probability.astype(np.float32),
                                period=view["period"], timestamps=view["timestamps"], fps=view["fps"])
    print("FROZEN", json.dumps({"config": frozen["config"], "model_sha256": frozen["model_sha256"],
                                "seconds": time.monotonic()-started}), flush=True)


if __name__ == "__main__":
    main()
