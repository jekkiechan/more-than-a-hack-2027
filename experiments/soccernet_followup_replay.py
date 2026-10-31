"""Replay sealed SoccerNet follow-up weights against saved probability arrays.

This checks inference and decoder outputs only; it does not access targets or
recompute metrics.  The sealed evaluation cache object may deserialize its
external labels when loaded.  Run after the evaluator has sealed its arrays.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import factorized_had as fh
from experiments import soccernet_capacity as cap
from experiments import soccernet_capacity_model as cap_model
from experiments import soccernet_extension as ext
from experiments import soccernet_followup_evaluate as evaluation
from experiments import supervised


ARMS = ("reference", "control", "balanced_anchors", "gap_consistency",
        "context_reconstruction")
SEEDS = (2026, 2027)
MEMBERS = ("2026", "2027", "ensemble")


def _ensemble_probability(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Use the exact arithmetic and dtype behavior of the evaluator."""
    return (first + second) * 0.5


def _verify_array(path: Path, probability: np.ndarray,
                  prediction: np.ndarray) -> None:
    with np.load(path, allow_pickle=False) as saved:
        if not {"probability", "prediction"}.issubset(saved.files):
            raise ValueError(f"Saved replay array is incomplete: {path}")
        if not np.array_equal(saved["probability"], probability):
            raise ValueError(f"Recomputed probability differs: {path}")
        if not np.array_equal(saved["prediction"], prediction):
            raise ValueError(f"Recomputed decoded prediction differs: {path}")


def _verify_evaluation_seal(execution: Path) -> tuple[dict, dict]:
    output = execution / "evaluation"
    report_path = output / "aggregate.json"
    rows_path = output / "scored_rows.json"
    seal_path = output / "evaluation_sealed.json"
    for path in (report_path, rows_path, seal_path):
        if not path.is_file():
            raise ValueError(f"Completed evaluator output is missing: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    evaluation._verify_output_seal(report, seal, execution, report_path, rows_path)
    hashes = report.get("private_array_sha256", {})
    expected = {
        f"evaluation/arrays/{cell['item']['match']}_{cell['view']}_{cell['replicate']}_{arm}_{member}.npz"
        for cell in evaluation.expected_cells()
        for arm in ARMS for member in MEMBERS
    }
    if set(hashes) != expected or len(expected) != 225:
        raise ValueError("Evaluation array seal does not contain the exact 225 outputs")
    evaluation._expect_hashes(hashes, base=execution, description="saved replay array")
    return report, seal


def replay(execution_dir: Path, data_dir: Path) -> dict:
    execution = Path(execution_dir).resolve()
    data = Path(data_dir).resolve()
    cap.DATA = data
    ext.DATA = data
    declaration, model_seal = evaluation._verify_declaration(execution, data)
    report, output_seal = _verify_evaluation_seal(execution)

    models = {}
    teachers = {}
    for arm in ARMS:
        models[arm] = {}
        for provider in ("sportec", "metrica", "skillcorner"):
            fold = f"leave_out_{provider}"
            for seed in SEEDS:
                relative = f"models/{arm}/{fold}/seed{seed}.joblib"
                path = execution / relative
                if model_seal["models"].get(relative) is None:
                    raise ValueError(f"Checkpoint is absent from model seal: {relative}")
                models[arm][fold, seed] = joblib.load(path)
    for provider in ("sportec", "metrica", "skillcorner"):
        fold = f"leave_out_{provider}"
        for seed in SEEDS:
            teachers[fold, seed] = joblib.load(ext.control_path(fold, seed))

    checked = {"seed_arrays": 0, "ensemble_arrays": 0, "decoded_arrays": 0}
    for cell in declaration["evaluation_cells"]:
        item, view, replicate = cell["item"], cell["view"], cell["replicate"]
        cached = joblib.load(ext.eval_path(item, view, replicate))
        inputs = cached["inputs"]
        record = ext.inference_record(cached["record"])
        n = len(np.asarray(inputs["period"]))
        fold = f"leave_out_{item['source']}"
        original = {seed: fh.predict(record, teachers[fold, seed]) for seed in SEEDS}
        native_by_arm = {}
        for arm in ARMS:
            native_by_arm[arm] = {}
            for seed in SEEDS:
                teacher = teachers[fold, seed]
                sampled = cap_model.predict(record, models[arm][fold, seed], teacher,
                                            original_probability=original[seed])
                probability = ext.upsample(sampled, record, inputs)
                if probability.shape != (n, 3) or not np.isfinite(probability).all():
                    raise ValueError(f"Invalid recomputed probability shape/content for {arm}/{seed}")
                native_by_arm[arm][seed] = probability
                prediction = supervised.decode(probability,
                    supervised.model_periods(inputs), float(inputs["fps"]), cap.lt.COST)
                array = execution / "evaluation/arrays" / (
                    f"{item['match']}_{view}_{replicate}_{arm}_{seed}.npz")
                _verify_array(array, probability, prediction)
                checked["seed_arrays"] += 1
                checked["decoded_arrays"] += 1
            ensemble = _ensemble_probability(native_by_arm[arm][2026],
                                             native_by_arm[arm][2027])
            prediction = supervised.decode(ensemble,
                supervised.model_periods(inputs), float(inputs["fps"]), cap.lt.COST)
            array = execution / "evaluation/arrays" / (
                f"{item['match']}_{view}_{replicate}_{arm}_ensemble.npz")
            _verify_array(array, ensemble, prediction)
            checked["ensemble_arrays"] += 1
            checked["decoded_arrays"] += 1

    if checked != {"seed_arrays": 150, "ensemble_arrays": 75, "decoded_arrays": 225}:
        raise ValueError(f"Replay did not cover the full sealed grid: {checked}")
    result = {
        "study_id": "soccernet_followup_weight_replay_v1",
        "declaration_sha256": evaluation.sha256(execution / "predeclared.json"),
        "model_seal_sha256": evaluation.sha256(execution / "models_sealed.json"),
        "evaluation_seal_sha256": evaluation.sha256(execution / "evaluation/evaluation_sealed.json"),
        "aggregate_sha256": evaluation.sha256(execution / "evaluation/aggregate.json"),
        "scored_rows_sha256": evaluation.sha256(execution / "evaluation/scored_rows.json"),
        "replay_source_sha256": evaluation.sha256(Path(__file__)),
        "evaluation_cells": len(declaration["evaluation_cells"]),
        "unique_checkpoints": len(model_seal["models"]),
        **checked,
        "probability_replay_exact": True,
        "decoded_prediction_replay_exact": True,
        "labels_used": False,
        "cached_external_labels_deserialized": True,
        "organizer_labels_read": False,
        "metrics_scored": False,
    }
    if len(model_seal["models"]) != 30:
        raise ValueError("Model seal does not contain all 30 expected checkpoints")
    path = execution / "weight_replay_verified.json"
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("MTAH_DATA_DIR", "/workspace/mtah-data")))
    args = parser.parse_args()
    print(json.dumps(replay(args.execution_dir, args.data_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
