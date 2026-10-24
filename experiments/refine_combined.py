"""One fixed composition of the independently selected feature and decoder models.

There is no parameter grid. Team/dead costs 3/8 were selected before this
composition was scored. The CLI loads only the original validation match.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments import refine_decode, refine_features, robust, supervised

ROOT = Path(__file__).resolve().parents[1]
FIXED_CONFIG = {"team_cost": 3.0, "dead_cost": 8.0, "boundary_power": 0.0}
VALIDATION = "J03WOY"


def predict_from_probabilities(probabilities, data, config=None):
    config = FIXED_CONFIG if config is None else config.get("config", config)
    if config != FIXED_CONFIG:
        raise ValueError("This bounded composition uses only the previously fixed costs")
    # The feature experiment archives posteriors as float32. Use that same
    # precision at inference so archived and regenerated decoding agree exactly.
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (len(data["period"]), 3):
        raise ValueError("Expected one H/A/D probability row per frame")
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)):
        raise ValueError("Invalid H/A/D probabilities")
    return refine_decode.decode(probabilities, supervised.model_periods(data),
                                float(data["fps"]), config)


def predict(data, model, config=None):
    """Predict from tracking, the frozen physical-feature model and fixed costs."""
    inputs = robust.tracking_inputs(data)
    return predict_from_probabilities(
        refine_features.predict_probabilities(inputs, model), inputs, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/experiments/refine_features")
    parser.add_argument("--decoder", type=Path, default=ROOT / "data/experiments/refine_decode")
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/refine_combined")
    args = parser.parse_args()
    if (args.out / "frozen_config.json").exists():
        raise RuntimeError("Composition already frozen; do not overwrite")
    feature_config = json.loads((args.features / "frozen_config.json").read_text())
    decoder_config = json.loads((args.decoder / "frozen_config.json").read_text())
    if decoder_config["config"] != FIXED_CONFIG:
        raise ValueError("The independently selected decoder differs from the fixed plan")
    model_path = args.features / "model.joblib"
    if robust.sha256(model_path) != feature_config["model_sha256"]:
        raise ValueError("Physical-feature model changed after freeze")
    for path, expected in feature_config["source_sha256"].items():
        if robust.sha256(ROOT / path) != expected:
            raise ValueError(f"Feature source changed after freeze: {path}")
    if robust.sha256(ROOT / "experiments/refine_decode.py") != decoder_config["source_sha256"]:
        raise ValueError("Decoder source changed after freeze")
    posterior_paths = {condition: args.features / f"{VALIDATION}_{condition}_probabilities.npz"
                       for condition in ("clean", "masked")}
    # These hashes also ensure both completed posterior exports exist before
    # writing an immutable composition manifest.
    posterior_hashes = {condition: robust.sha256(path)
                        for condition, path in posterior_paths.items()}
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "config": FIXED_CONFIG,
        "composition_policy": "One fixed cross-application; no additional grid or parameter selection",
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": feature_config["model_sha256"],
        "feature_config_sha256": robust.sha256(args.features / "frozen_config.json"),
        "decoder_config_sha256": robust.sha256(args.decoder / "frozen_config.json"),
        "source_sha256": {"experiments/refine_combined.py": robust.sha256(__file__),
                          "experiments/refine_decode.py": decoder_config["source_sha256"],
                          **feature_config["source_sha256"]},
        "validation_match": VALIDATION, "validation_posteriors_sha256": posterior_hashes,
        "posterior_precision": "float32, matching archived posteriors and prediction API",
        "prediction_api": "experiments.refine_combined.predict",
        "masked_validation_used_for_selection": False, "test_evaluated": False,
    }
    (args.out / "frozen_config.json").write_text(json.dumps(frozen, indent=2)+"\n")
    source = ROOT / f"data/external_audit/sportec/processed/{VALIDATION}.npz"
    with np.load(source, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("labels", "period", "timestamps", "fps")}
    results = {"config": FIXED_CONFIG, "validation_match": VALIDATION,
               "validation_data_sha256": robust.sha256(source), "conditions": {}}
    for condition, path in posterior_paths.items():
        with np.load(path, allow_pickle=False) as archive:
            probabilities = archive["probabilities"]
            for key in ("period", "timestamps", "fps"):
                if not np.array_equal(archive[key], data[key]):
                    raise ValueError(f"Posterior metadata differs for {condition}: {key}")
        predictions = predict_from_probabilities(probabilities, data, FIXED_CONFIG)
        score = refine_decode.compact_score(data, predictions)
        results["conditions"][condition] = score
        np.save(args.out / f"{VALIDATION}_{condition}_predictions.npy", predictions)
        print(condition, json.dumps({key: score[key] for key in (
            "macro_f1", "spell_f1_mean", "accuracy", "n_gt_spells", "n_pred_spells")}), flush=True)
    (args.out / "validation.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
