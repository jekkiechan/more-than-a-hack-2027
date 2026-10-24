"""One fixed equal-weight ensemble; no fitting, weight search or held-out access.

Members are the frozen spell-weighted 63/260 and uniform 127/400 physical-feature
models. Average their float32 H/A/D posteriors, then apply fixed team/dead costs
3/8. Both models share one feature extraction at inference.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import refine_combined, refine_features, robust, supervised

ROOT = Path(__file__).resolve().parents[1]
MEMBERS = ("refine_weighted", "refine_scale")
VALIDATION = "J03WOY"
CONFIG = {"member_weights": [0.5, 0.5], "posterior_precision": "float32",
          "decoder": {"team_cost": 3.0, "dead_cost": 8.0, "boundary_power": 0.0}}


def average_probabilities(first, second):
    first, second = np.asarray(first, dtype=np.float32), np.asarray(second, dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("Member probabilities must have identical H/A/D shapes")
    return (first + second) * np.float32(0.5)


def predict_from_features(matrix, names, data, model):
    """Apply both frozen classifiers to one cached physical-feature matrix."""
    if model["config"] != CONFIG or len(model["members"]) != 2:
        raise ValueError("This experiment permits only its fixed two-member ensemble")
    if names != model["feature_names"]:
        raise ValueError("Physical feature schema differs from the frozen ensemble")
    first, second = [refine_features.probabilities_from_features(matrix, names, member)
                     for member in model["members"]]
    probabilities = average_probabilities(first, second)
    return refine_combined.predict_from_probabilities(probabilities, data, CONFIG["decoder"])


def predict(data, model):
    """Tracking-only inference, extracting shared physical features once."""
    inputs = robust.tracking_inputs(data)
    matrix, names = refine_features.features(inputs)
    return predict_from_features(matrix, names, inputs, model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/refine_ensemble")
    args = parser.parse_args()
    if (args.out / "frozen_config.json").exists():
        raise RuntimeError("Ensemble already frozen; do not overwrite")
    records, models, source_hashes, posterior_paths = [], [], {}, {}
    for name in MEMBERS:
        directory = ROOT / "data/experiments" / name
        config_path, model_path = directory / "frozen_config.json", directory / "model.joblib"
        frozen = json.loads(config_path.read_text())
        if robust.sha256(model_path) != frozen["model_sha256"]:
            raise ValueError(f"Member changed after freeze: {name}")
        for path, expected in frozen["source_sha256"].items():
            if robust.sha256(ROOT / path) != expected:
                raise ValueError(f"Member source changed after freeze: {path}")
            source_hashes[path] = expected
        paths = {condition: directory / f"{VALIDATION}_{condition}_probabilities.npz"
                 for condition in ("clean", "masked")}
        posterior_paths[name] = paths
        records.append({"name": name, "model_path": str(model_path.relative_to(ROOT)),
                        "model_sha256": frozen["model_sha256"],
                        "frozen_config_sha256": robust.sha256(config_path),
                        "config": frozen["config"],
                        "validation_posteriors_sha256": {key: robust.sha256(path)
                                                          for key, path in paths.items()}})
        models.append(joblib.load(model_path))
    if models[0]["feature_names"] != models[1]["feature_names"]:
        raise ValueError("Ensemble members require different features")
    args.out.mkdir(parents=True, exist_ok=True)
    model = {"members": models, "member_records": records,
             "feature_names": models[0]["feature_names"], "classes": [0, 1, 2],
             "config": CONFIG}
    model_path = args.out / "model.joblib"
    joblib.dump(model, model_path)
    for name in ("refine_ensemble", "refine_combined", "refine_decode"):
        path = f"experiments/{name}.py"
        source_hashes[path] = robust.sha256(ROOT / path)
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "config": CONFIG,
              "model_sha256": robust.sha256(model_path), "members": records,
              "source_sha256": source_hashes,
              "prediction_api": "experiments.refine_ensemble.predict",
              "cached_prediction_api": "experiments.refine_ensemble.predict_from_features",
              "policy": "One equal-weight candidate using independently frozen models; no new fitting or parameter search",
              "validation_match": VALIDATION, "n_candidates": 1,
              "test_evaluated": False, "masked_validation_used_for_selection": False}
    content = json.dumps(frozen, indent=2)+"\n"
    (args.out / "frozen_config.json").write_text(content)
    (args.out / "config.json").write_text(content)
    # The immutable model/config are saved before any ensemble scoring.
    data_path = ROOT / f"data/external_audit/sportec/processed/{VALIDATION}.npz"
    with np.load(data_path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("labels", "period", "timestamps", "fps")}
    result = {"conditions": {}, "validation_match": VALIDATION,
              "validation_data_sha256": robust.sha256(data_path), "config": CONFIG}
    for condition in ("clean", "masked"):
        probabilities = []
        for name in MEMBERS:
            with np.load(posterior_paths[name][condition], allow_pickle=False) as archive:
                for key in ("period", "timestamps", "fps"):
                    if not np.array_equal(data[key], archive[key]):
                        raise ValueError(f"Posterior alignment differs: {name}/{condition}/{key}")
                probabilities.append(archive["probabilities"])
        average = average_probabilities(*probabilities)
        predictions = refine_combined.predict_from_probabilities(average, data, CONFIG["decoder"])
        scores = supervised.score(data, predictions)
        result["conditions"][condition] = scores
        np.save(args.out / f"{VALIDATION}_{condition}_predictions.npy", predictions)
        np.savez_compressed(args.out / f"{VALIDATION}_{condition}_probabilities.npz",
                            probabilities=average, period=data["period"],
                            timestamps=data["timestamps"], fps=data["fps"])
        print(condition, json.dumps({key: scores[key] for key in (
            "macro_f1", "spell_f1_mean", "accuracy", "n_gt_spells", "n_pred_spells")}), flush=True)
    (args.out / "validation.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
