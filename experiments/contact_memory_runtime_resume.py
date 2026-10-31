"""Resume the fixed contact-memory score cells after a local replay drift.

The frozen runner had an exact equality guard between two same-checkpoint P(D)
replays. CPU batch shapes produced sub-micro replay drift on one SkillCorner
match. This wrapper changes only that local-to-local guard to a 1e-6 tolerance;
the candidate still copies archived member P(D), and archived member and
ensemble P(D) checks remain exact. No fit, view, decoder, or score cell changes.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import contact_memory_experiment as experiment
from experiments import factorized_had as fh
from experiments import learned_temporal as lt
from experiments import objective_ensemble as oe
from experiments import supervised
from experiments.eval_set.protocol import sha256


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RECEIPT = ROOT / "experiments/results/contact_memory_runtime_correction.json"


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _row_from_saved(cell: dict, spec: dict) -> dict:
    """Rebuild the lost score row from one already-saved cell's predictions."""
    cache_path = experiment.DATA / cell["evaluation_cache"]
    if sha256(cache_path) != cell["evaluation_cache_sha256"]:
        raise ValueError("A frozen evaluation input changed during score recovery")
    cache = joblib.load(cache_path)
    inputs = cache["inputs"]
    labels = np.asarray(cache["labels"]).astype(str)
    key = f"{cell['match']}_{cell['view']}_{cell['replicate']}"
    ensemble_path = experiment.OUT / f"candidate_{key}_ensemble.npz"
    if not ensemble_path.is_file():
        raise ValueError(f"Missing completed ensemble artifact: {key}")
    with np.load(ensemble_path, allow_pickle=False) as archive:
        ensemble_probability = archive["probability"]
        ensemble_prediction = archive["prediction"]
    saved_ensemble_path = experiment.DATA / cell["ensemble_probability"]
    with np.load(saved_ensemble_path, allow_pickle=False) as archive:
        saved_ensemble = archive["probability"]
    if not np.array_equal(ensemble_probability[:, 2], saved_ensemble[:, 2]):
        raise ValueError(f"Saved cell changed exact archived ensemble P(D): {key}")
    if not np.array_equal(ensemble_prediction, supervised.decode(
            ensemble_probability, supervised.model_periods(inputs),
            float(inputs["fps"]), lt.COST).astype("U1")):
        raise ValueError(f"Saved ensemble prediction/probability mismatch: {key}")
    ensemble_row = oe.score_prediction(
        labels, ensemble_prediction, inputs, "contact_memory_ensemble", cell,
        experiment.digest_array(ensemble_probability))
    if (ensemble_row["n_gt_spells"] != cell["ensemble_baseline_row"]["n_gt_spells"]
            or ensemble_row["graded_frames"] !=
            cell["ensemble_baseline_row"]["graded_frames"]):
        raise ValueError(f"Saved ensemble changed graded population: {key}")

    seed_rows = []
    for seed in experiment.TRAIN_SEEDS:
        candidate_path = experiment.OUT / f"candidate_{key}_seed_{seed}.npz"
        if not candidate_path.is_file():
            raise ValueError(f"Missing completed member artifact: {key}/{seed}")
        with np.load(candidate_path, allow_pickle=False) as archive:
            probability = archive["probability"]
            prediction = archive["prediction"]
        baseline = cell["member_baselines"][str(seed)]
        control_path = experiment.DATA / baseline["probability"]
        with np.load(control_path, allow_pickle=False) as archive:
            saved_member = archive["probability"]
        if not np.array_equal(probability[:, 2], saved_member[:, 2]):
            raise ValueError(f"Saved member changed exact archived P(D): {key}/{seed}")
        if not np.array_equal(prediction, supervised.decode(
                probability, supervised.model_periods(inputs),
                float(inputs["fps"]), lt.COST).astype("U1")):
            raise ValueError(f"Saved member prediction/probability mismatch: {key}/{seed}")
        row = oe.score_prediction(
            labels, prediction, inputs, f"contact_memory_seed_{seed}", cell,
            experiment.digest_array(probability))
        baseline_row = baseline["row"]
        if (row["n_gt_spells"] != baseline_row["n_gt_spells"] or
                row["graded_frames"] != baseline_row["graded_frames"]):
            raise ValueError(f"Saved member changed graded population: {key}/{seed}")
        seed_rows.append({
            "candidate": row,
            "control": baseline_row,
            "candidate_probability_file": str(candidate_path.relative_to(experiment.DATA)),
            "candidate_probability_file_sha256": sha256(candidate_path),
            "control_probability_sha256": experiment.digest_array(saved_member),
            "d_probability_exact": True,
        })

    return {
        "cell": cell,
        "control_ensemble": cell["ensemble_baseline_row"],
        "candidate_ensemble": ensemble_row,
        "seed_members": seed_rows,
        "candidate_ensemble_probability_file":
            str(ensemble_path.relative_to(experiment.DATA)),
        "candidate_ensemble_probability_file_sha256": sha256(ensemble_path),
        "d_probability_exact": True,
    }


def _candidate_paths(cell: dict) -> list[Path]:
    key = f"{cell['match']}_{cell['view']}_{cell['replicate']}"
    return [experiment.OUT / f"candidate_{key}_ensemble.npz"] + [
        experiment.OUT / f"candidate_{key}_seed_{seed}.npz"
        for seed in experiment.TRAIN_SEEDS]


def _artifact_info(cell: dict) -> list[dict]:
    paths = _candidate_paths(cell)
    return [{"file": str(path.relative_to(experiment.DATA)),
             "sha256": sha256(path)} for path in paths]


def _probe_failed_cell(cell: dict, spec: dict) -> dict:
    """Measure only same-checkpoint D replay drift; do not score labels."""
    import torch
    torch.set_num_threads(1)
    cache = joblib.load(experiment.DATA / cell["evaluation_cache"])
    inputs = cache["inputs"]
    record = dict(cache["record"])
    record["y"] = np.full(len(record["x"]), -1, dtype=np.int8)
    memory = experiment.cm.memory_features(inputs, record["indices"])
    period = supervised.model_periods(inputs)
    result = {"cell": {key: cell[key] for key in ("source", "match", "view", "replicate")},
              "evaluation_cache_sha256": cell["evaluation_cache_sha256"],
              "per_seed": {}}
    for seed in experiment.TRAIN_SEEDS:
        reference = spec["baseline"]["preparation"]["checkpoints"][cell["source"]][str(seed)]
        fitted_base, base_model = experiment._load_base(experiment.DATA / reference["path"])
        _, adapter = experiment._load_adapter(experiment._adapter_path(cell["source"], seed))
        base = experiment.upsample_probabilities(
            fh.predict(record, fitted_base), record, period)
        candidate = experiment.upsample_probabilities(
            experiment.predict(record, memory, fitted_base, base_model, adapter),
            record, period)
        baseline = cell["member_baselines"][str(seed)]
        with np.load(experiment.DATA / baseline["probability"], allow_pickle=False) as archive:
            archived = archive["probability"]
        result["per_seed"][str(seed)] = {
            "max_abs_candidate_vs_local_base_pd":
                float(np.max(np.abs(candidate[:, 2] - base[:, 2]))),
            "max_abs_local_base_vs_archived_pd":
                float(np.max(np.abs(base[:, 2] - archived[:, 2]))),
        }
    maximum = max(
        value[key]
        for value in result["per_seed"].values()
        for key in ("max_abs_candidate_vs_local_base_pd",
                    "max_abs_local_base_vs_archived_pd"))
    if maximum > 1e-6:
        raise ValueError(f"Local P(D) replay drift exceeds declared tolerance: {maximum}")
    result["maximum_observed_drift"] = maximum
    result["labels_used"] = False
    return result


def _corrected_score_function():
    original = inspect.getsource(experiment._score_cell)
    exact_guard = "if not np.array_equal(candidate_probability[:, 2], base_probability[:, 2]):"
    corrected_guard = "if not np.allclose(candidate_probability[:, 2], base_probability[:, 2], rtol=0, atol=1e-6):"
    if original.count(exact_guard) != 1:
        raise ValueError("Expected exactly one local candidate-vs-base P(D) guard")
    corrected = original.replace(exact_guard, corrected_guard)
    namespace = dict(experiment.__dict__)
    exec(compile(corrected, "<contact-memory-local-d-replay-guard>", "exec"), namespace)
    return namespace["_score_cell"], hashlib.sha256(original.encode()).hexdigest(), hashlib.sha256(
        corrected.encode()).hexdigest()


def run() -> dict:
    if PUBLIC_RECEIPT.exists():
        raise ValueError("Runtime correction receipt is immutable")
    if experiment.FIT_REPORT.exists():
        raise ValueError("Contact-memory score report is already present")
    spec = experiment.check()
    seal = experiment._verify_seal(spec)
    cells = spec["baseline"]["evaluation"]["cells"]
    complete, missing = [], []
    for cell in cells:
        paths = _candidate_paths(cell)
        present = [path.is_file() for path in paths]
        if all(present):
            complete.append(cell)
        elif not any(present):
            missing.append(cell)
        else:
            raise ValueError(f"Partial score artifacts exist for {cell['match']}/{cell['view']}")
    if len(complete) != 12 or len(missing) != 3 or any(
            cell["match"] != "skillcorner_had_1996435" for cell in missing):
        raise ValueError("Expected exactly the twelve prior cells and three failed-match cells")

    # Reconstruct the twelve already-scored rows from their immutable files; do
    # not rerun inference or alter those candidate artifacts.
    prior_rows_by_key = {}
    prior_rows_receipts = []
    for cell in complete:
        row = _row_from_saved(cell, spec)
        key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
        prior_rows_by_key[key] = row
        prior_rows_receipts.append({
            "cell": {name: cell[name] for name in ("source", "match", "view", "replicate")},
            "row_sha256": hashlib.sha256(_canonical(row)).hexdigest(),
            "candidate_artifacts": _artifact_info(cell),
        })

    drift = _probe_failed_cell(missing[0], spec)
    corrected_score, original_function_sha, corrected_function_sha = _corrected_score_function()
    receipt = {
        "study_id": spec["study_id"],
        "declaration_sha256": spec["_declaration_sha256"],
        "original_runner_sha256": spec["source_sha256"][
            "experiments/contact_memory_experiment.py"],
        "runtime_wrapper": str(Path(__file__).resolve().relative_to(ROOT)),
        "runtime_wrapper_sha256": sha256(Path(__file__).resolve()),
        "original_score_function_sha256": original_function_sha,
        "effective_score_function_sha256": corrected_function_sha,
        "correction": {
            "scope": "Only candidate-vs-regenerated-local-baseline member P(D) replay guard",
            "change": "np.array_equal -> np.allclose(rtol=0, atol=1e-6)",
            "reason": "Sub-micro CPU batch-shape replay drift; this guard compared two replays of the same frozen checkpoint.",
            "archived_member_pd": "Still required bit-exact after copying archived P(D); independently verified for every candidate member.",
            "archived_ensemble_pd": "Still required bit-exact; independently verified for every candidate ensemble.",
            "fit_view_loss_decoder_score_cells": "unchanged",
        },
        "drift_probe": drift,
        "sealed_models": seal["models"],
        "reused_completed_cells": prior_rows_receipts,
        "reused_cell_count": len(complete),
        "resumed_missing_cells": [
            {name: cell[name] for name in ("source", "match", "view", "replicate")}
            for cell in missing],
        "organizer_labels_read": False,
        "organizer_scored": False,
    }
    experiment._write_exclusive(PUBLIC_RECEIPT, receipt)

    # The frozen runner is unchanged on disk; this in-memory function applies
    # only the precisely hash-bound local replay tolerance above.
    new_rows_by_key = {}
    base_models, adapters = {}, {}
    for source in experiment.HELDOUT:
        for seed in experiment.TRAIN_SEEDS:
            control = spec["baseline"]["preparation"]["checkpoints"][source][str(seed)]
            fitted_base, base_model = experiment._load_base(experiment.DATA / control["path"])
            fitted_adapter, adapter = experiment._load_adapter(
                experiment._adapter_path(source, seed))
            if fitted_adapter.get("base_checkpoint_sha256") != control["sha256"]:
                raise ValueError(f"Adapter/backbone binding mismatch: {source}/{seed}")
            base_models[(source, seed)] = (fitted_base, base_model)
            adapters[(source, seed)] = (fitted_adapter, adapter)
    for cell in missing:
        row = corrected_score(cell, spec, base_models, adapters)
        new_rows_by_key[(cell["source"], cell["match"], cell["view"], cell["replicate"])] = row

    rows = []
    for cell in cells:
        key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
        row = prior_rows_by_key.get(key, new_rows_by_key.get(key))
        if row is None:
            raise ValueError(f"Missing recovered score row: {key}")
        rows.append(row)

    # Final strict archive-D audit over all fifteen member and ensemble files.
    for cell, row in zip(cells, rows):
        key = f"{cell['match']}_{cell['view']}_{cell['replicate']}"
        ensemble_path = experiment.OUT / f"candidate_{key}_ensemble.npz"
        with np.load(ensemble_path, allow_pickle=False) as archive:
            candidate = archive["probability"]
        with np.load(experiment.DATA / cell["ensemble_probability"], allow_pickle=False) as archive:
            archived = archive["probability"]
        if not np.array_equal(candidate[:, 2], archived[:, 2]):
            raise ValueError(f"Strict final ensemble P(D) check failed: {key}")
        for seed in experiment.TRAIN_SEEDS:
            candidate_path = experiment.OUT / f"candidate_{key}_seed_{seed}.npz"
            with np.load(candidate_path, allow_pickle=False) as archive:
                member = archive["probability"]
            baseline = cell["member_baselines"][str(seed)]
            with np.load(experiment.DATA / baseline["probability"], allow_pickle=False) as archive:
                member_base = archive["probability"]
            if not np.array_equal(member[:, 2], member_base[:, 2]):
                raise ValueError(f"Strict final member P(D) check failed: {key}/{seed}")
    if not all(row["d_probability_exact"] for row in rows):
        raise ValueError("A recovered row lost its strict D-probability receipt")

    result = {
        "study_id": spec["study_id"],
        "declaration_sha256": spec["_declaration_sha256"],
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "new_ensemble_cells": len(rows),
        "new_seed_diagnostic_rows": sum(len(row["seed_members"]) for row in rows),
        "d_probability_exact_for_every_member_and_ensemble": True,
        "runtime_correction_receipt": str(PUBLIC_RECEIPT.relative_to(ROOT)),
        "runtime_correction_receipt_sha256": sha256(PUBLIC_RECEIPT),
        "organizer_labels_read": False,
        "organizer_scored": False,
        "automatic_promotion": False,
    }
    experiment._write_exclusive(experiment.FIT_REPORT, result)
    return experiment.report()


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True, indent=2))
