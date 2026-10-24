"""Bounded clean-plus-masked training experiment; validation only, never test.

The missingness process uses tracking/period metadata and a fixed random seed,
never possession labels. Clean validation selects the decoding penalty; masked
validation is a supplementary robustness measurement, not a selection objective.
"""

from __future__ import annotations

import os

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
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
from experiments import supervised  # noqa: E402
from experiments.stress import mask_ball  # noqa: E402

INPUT_KEYS = ("ball", "players", "teams", "period", "fps", "timestamps", "pitch",
              "ball_observed", "player_observed")
SWITCH_COSTS = (0, 3, 8, 16, 32)
MISSING_RATE = 0.41
MEAN_GAP_S = 1.0
BASE_SEED = 2026
SELECTION = "highest clean validation spell F1 among macro F1 >= .90; otherwise highest clean macro F1"


def tracking_inputs(data):
    """Allowlist features so inference cannot accidentally read label fields."""
    return {key: data[key] for key in INPUT_KEYS if key in data}


def masked_inputs(data, seed):
    return mask_ball(tracking_inputs(data), missing_rate=MISSING_RATE,
                     mean_gap_s=MEAN_GAP_S, seed=seed)


def missing_fraction(data):
    return float(np.mean(~np.isfinite(data["ball"]).all(axis=1)))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit(train_paths):
    xs, ys, names, manifest = [], [], None, []
    for match_index, path in enumerate(train_paths):
        data = supervised.load(path)
        labels = np.asarray(data["labels"])
        stride = max(1, round(float(data["fps"]) / 4))
        rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::stride]
        y = np.array([np.flatnonzero(supervised.LABELS == v)[0]
                      for v in labels[rows]], dtype=np.int8)
        views = {"clean": tracking_inputs(data),
                 "masked": masked_inputs(data, BASE_SEED + match_index)}
        for name, view in views.items():
            print(f"FEATURES {path.stem} {name}", flush=True)
            x, these_names = supervised.features(view)
            if names is not None and names != these_names:
                raise ValueError("Inconsistent feature definitions")
            names = these_names
            x = x[rows]
            xs.extend((x, supervised.swap_team_features(x, names)))
            ys.extend((y, np.where(y < 2, 1-y, y)))
        manifest.append({
            "path": str(path), "sha256": sha256(path),
            "labelled_sampled_frames": len(rows), "sample_stride": stride,
            "effective_train_fps": float(data["fps"]) / stride,
            "mask_seed": BASE_SEED + match_index,
            "clean_ball_missing_fraction": missing_fraction(views["clean"]),
            "masked_ball_missing_fraction": missing_fraction(views["masked"]),
        })
    x, y = np.concatenate(xs), np.concatenate(ys)
    print(f"FIT rows={len(y)} features={x.shape[1]}", flush=True)
    estimator = HistGradientBoostingClassifier(
        max_iter=160, max_leaf_nodes=31, learning_rate=0.08,
        min_samples_leaf=35, l2_regularization=1, early_stopping=False,
        random_state=17,
    )
    with threadpool_limits(limits=1):
        estimator.fit(x, y)
    return {
        "estimator": estimator, "feature_names": names,
        "classes": estimator.classes_.tolist(), "training": manifest,
        "config": {
            "iterations": 160, "leaves": 31, "train_fps": 4,
            "two_stage": False, "switch_cost": 0,
            "augmentation": "clean + masked, each with team swap",
            "missing_rate": MISSING_RATE, "mean_gap_s": MEAN_GAP_S,
            "training_mask_seed_base": BASE_SEED,
            "random_state": 17, "training_rows_after_augmentation": len(y),
        },
    }


def predict(data, model):
    return supervised.predict(tracking_inputs(data), model)


def compact_score(data, predictions):
    result = supervised.score(data, predictions)
    keys = ("accuracy", "macro_f1", "macro_f1_present_classes", "spell_f1_mean",
            "spell_f1", "n_gt_spells", "n_pred_spells", "spell_count_error",
            "alive_dead_f1", "team_acc_gt_alive", "per_class", "confusion",
            "label_coverage", "target_classes_present", "three_class_benchmark")
    return {key: result[key] for key in keys}


def aggregate(scores):
    return {"mean_spell_f1": float(np.mean([s["spell_f1_mean"] for s in scores.values()])),
            "mean_macro_f1": float(np.mean([s["macro_f1"] for s in scores.values()]))}


def validation_probabilities(validation, model):
    output = {}
    for name, data in validation:
        output[name] = {}
        for condition, view in (("clean", tracking_inputs(data)),
                                ("masked", masked_inputs(data, BASE_SEED))):
            probabilities = supervised.predict_probabilities(view, model)
            output[name][condition] = (view, probabilities)
    return output


def evaluate_cost(validation, probabilities, cost):
    conditions = {}
    for condition in ("clean", "masked"):
        scores, missing = {}, {}
        for name, data in validation:
            view, probability = probabilities[name][condition]
            predictions = supervised.decode(probability, supervised.model_periods(view),
                                            float(view["fps"]), cost)
            scores[name] = compact_score(data, predictions)
            missing[name] = missing_fraction(view)
        conditions[condition] = {**aggregate(scores), "scores": scores,
                                 "ball_missing_fraction": missing}
    return {"switch_cost": cost, **conditions}


def selection_key(candidate):
    clean = candidate["clean"]
    macro, spell = clean["mean_macro_f1"], clean["mean_spell_f1"]
    return (1, spell, macro) if macro >= 0.90 else (0, macro, spell)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT / "data/external_audit/sportec/split.json")
    parser.add_argument("--processed", type=Path, default=ROOT / "data/external_audit/sportec/processed")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/robust")
    parser.add_argument("--baseline", type=Path, help="Optional frozen clean-training model; validation comparison only")
    args = parser.parse_args()
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "frozen_config.json").exists():
        raise ValueError("Output already frozen; use a new output directory for a new experiment")
    split = json.loads(args.split.read_text())
    train_ids, val_ids, test_ids = map(set, (split["train"], split["validation"], split["test"]))
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Match partitions overlap")
    # No code path constructs or opens a held-out test NPZ.
    train_paths = [args.processed / f"{name}.npz" for name in split["train"]]
    validation = [(name, supervised.load(args.processed / f"{name}.npz"))
                  for name in split["validation"]]
    baseline, baseline_hash = None, None
    if args.baseline:
        baseline_hash = sha256(args.baseline)
        baseline = joblib.load(args.baseline)
    model = fit(train_paths)
    probabilities = validation_probabilities(validation, model)
    candidates = []
    for cost in SWITCH_COSTS:
        candidate = evaluate_cost(validation, probabilities, cost)
        candidates.append(candidate)
        print(json.dumps({"switch_cost": cost,
                          **{condition: {k: candidate[condition][k] for k in ("mean_macro_f1", "mean_spell_f1")}
                             for condition in ("clean", "masked")}}), flush=True)
    selected = max(candidates, key=selection_key)
    model["config"]["switch_cost"] = selected["switch_cost"]
    model_path = args.out / "model.joblib"
    joblib.dump(model, model_path)
    frozen = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "config": model["config"], "training": model["training"],
        "classes": model["classes"], "split": split,
        "selection": SELECTION, "stress_used_for_selection": False,
        "validation_mask_seed": BASE_SEED, "switch_cost_candidates": SWITCH_COSTS,
        "model_sha256": sha256(model_path),
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in (
            ROOT / "experiments/robust.py", ROOT / "experiments/supervised.py",
            ROOT / "experiments/stress.py", ROOT / "experiments/metrics.py")},
        "validation_data_sha256": {name: sha256(args.processed / f"{name}.npz")
                                   for name in split["validation"]},
        "test_evaluated": False,
    }
    (args.out / "frozen_config.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (args.out / "validation.json").write_text(json.dumps({
        "selection": SELECTION, "selected": selected,
        "candidates": candidates, "test_evaluated": False,
    }, indent=2) + "\n")
    print("FROZEN", json.dumps({"config": model["config"], "model_sha256": frozen["model_sha256"]}), flush=True)
    if baseline is not None:
        comparison = evaluate_cost(validation, validation_probabilities(validation, baseline),
                                   baseline["config"]["switch_cost"])
        (args.out / "baseline_validation.json").write_text(json.dumps({
            "model_path": str(args.baseline), "model_sha256": baseline_hash,
            "config": baseline["config"], "scores": comparison,
        }, indent=2) + "\n")
        print("BASELINE", json.dumps({condition: {k: comparison[condition][k]
              for k in ("mean_macro_f1", "mean_spell_f1")}
              for condition in ("clean", "masked")}), flush=True)
    print(f"DONE seconds={time.monotonic()-started:.1f}", flush=True)


if __name__ == "__main__":
    main()
