"""GAR-only evaluator amendment for the selected pooled v2 weight-edit package.

The frozen evaluation module used the 5 Hz dense-record fps in one new phase
count assertion, although the reused GAR phase assembler creates one record
per raw/native 5 Hz offset. This module preserves all frozen helpers and
declarations while binding the corrected assertion to raw ``inputs['fps']``.
It has no public scoring or fitting action.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from experiments import v2_weight_edit_evaluation_v1 as base

ROOT = base.ROOT
OUT = base.OUT
GAMES = OUT / "games"
AMENDMENT = ROOT / "experiments/results/v2_weight_edit_evaluation_v1_gar_execution_amendment.json"
FAILED_ATTEMPT_RECEIPT = ROOT / "experiments/results/v2_weight_edit_evaluation_v1_gar_failed_attempt.json"
RESULT_RECEIPT = ROOT / "experiments/results/v2_weight_edit_evaluation_v1_gar_execution_result_receipt.json"
MODULE_PATH = Path(__file__).resolve()


def _write_once(path: Path, value: dict) -> None:
    base.json_write_once(path, value)


def seal_amendment() -> dict:
    """Record the failed pre-score attempt and corrected executor without changing old seals."""
    if AMENDMENT.exists():
        value = base.json_read(AMENDMENT)
        check_amendment()
        return value
    declaration = base.json_read(base.DECLARATION)
    eval_rel = str(Path(base.__file__).resolve().relative_to(ROOT))
    declared_sha = declaration["source_sha256"].get(eval_rel)
    if declared_sha != base.sha256(Path(base.__file__)):
        raise base.EvaluationError("Frozen evaluation source was changed; restore it before GAR amendment")
    existing = list(GAMES.glob("test__*")) if GAMES.exists() else []
    if existing:
        raise base.EvaluationError("GAR outputs exist; amendment must precede all game scoring")
    failed_attempt = {
        "study": base.STUDY,
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "original_declaration_sha256": base.sha256(base.DECLARATION),
        "frozen_evaluation_module_sha256": declared_sha,
        "failed_log_path": str((OUT / "gar_test.log").relative_to(ROOT)),
        "failed_log_sha256": base.sha256(OUT / "gar_test.log"),
        "failure": "Initial phase-count assertion used sampled dense-record fps (~4.995), expecting 1 while the frozen phase helper returned 6 records.",
        "game_score_artifacts_written": False,
        "model_scoring_completed": False,
        "organizer_reference_access": False,
        "public_score_preceded_failure": True,
        "public_result_sha256": base.sha256(base.PUBLIC_RESULT),
    }
    _write_once(FAILED_ATTEMPT_RECEIPT, failed_attempt)
    result = {
        "study": base.STUDY,
        "amended_utc": datetime.now(timezone.utc).isoformat(),
        "original_declaration_path": str(base.DECLARATION.relative_to(ROOT)),
        "original_declaration_sha256": base.sha256(base.DECLARATION),
        "frozen_evaluation_module_path": eval_rel,
        "frozen_evaluation_module_sha256": declared_sha,
        "amended_executor_path": str(MODULE_PATH.relative_to(ROOT)),
        "amended_executor_sha256": base.sha256(MODULE_PATH),
        "failed_attempt": {
            "receipt_path": str(FAILED_ATTEMPT_RECEIPT.relative_to(ROOT)),
            "receipt_sha256": base.sha256(FAILED_ATTEMPT_RECEIPT),
        },
        "correction": "Use int(round(float(inputs['fps']) / 5.0)) for each clean/v2 view; otherwise call the frozen GAR helpers and adapter unchanged.",
        "winner_id": base.json_read(base.DECLARATION)["selected_package"]["receipt"]["winner_id"],
        "deployment_receipt_sha256": base.json_read(base.DECLARATION)["selected_package"]["receipt_sha256"],
        "public_gate_preceded_amendment": True,
        "public_result_sha256": base.sha256(base.PUBLIC_RESULT),
        "public_scoring_affected": False,
    }
    _write_once(AMENDMENT, result)
    return result


def check_amendment() -> dict:
    if not AMENDMENT.is_file():
        raise base.EvaluationError("Seal the GAR-only executor amendment before scoring")
    value = base.json_read(AMENDMENT)
    declaration = base.json_read(base.DECLARATION)
    eval_rel = str(Path(base.__file__).resolve().relative_to(ROOT))
    if (value.get("original_declaration_sha256") != base.sha256(base.DECLARATION)
            or value.get("frozen_evaluation_module_sha256") != declaration["source_sha256"].get(eval_rel)
            or base.sha256(Path(base.__file__)) != value.get("frozen_evaluation_module_sha256")
            or value.get("amended_executor_path") != str(MODULE_PATH.relative_to(ROOT))
            or value.get("amended_executor_sha256") != base.sha256(MODULE_PATH)
            or value.get("public_result_sha256") != base.sha256(base.PUBLIC_RESULT)
            or value.get("failed_attempt", {}).get("receipt_sha256") != base.sha256(FAILED_ATTEMPT_RECEIPT)
            or base.json_read(FAILED_ATTEMPT_RECEIPT).get("game_score_artifacts_written") is not False):
        raise base.EvaluationError("GAR execution amendment no longer matches its frozen lineage")
    return value


def evaluate_game(game_id: str) -> Path:
    declaration = base.check_declaration()
    amendment = check_amendment()
    if game_id not in declaration["cohort"]["match_ids"]:
        raise base.EvaluationError(f"Undeclared GAR TEST game: {game_id}")
    target = GAMES / f"test__{game_id}.json"
    if target.exists():
        saved = base.json_read(target)
        if (saved.get("declaration_sha256") != base.sha256(base.DECLARATION)
                or saved.get("execution_amendment_sha256") != base.sha256(AMENDMENT)
                or saved.get("execution_module_sha256") != base.sha256(MODULE_PATH)):
            raise base.EvaluationError(f"Existing amended game output has stale provenance: {game_id}")
        return target

    adapter, deployment = base._load_deployment()
    records = base.gar.previous.load_game_records("test", game_id, base.RECORDS)
    captured = base.gar.previous.capture_inputs("test", game_id)
    clean_inputs, labels, saved_x, names = base.gar.previous.build_game(records, captured)
    calibration = base.json_read(base.gar.previous.CALIBRATION)["calibration"]
    sampled_fps = float(records[0]["fps"])
    scores, predictions, phase_counts = {}, {}, {}
    for view in base.VIEWS:
        inputs = (clean_inputs if view == "clean" else
                  base.gar.previous.feed_view(clean_inputs, labels, calibration,
                      base.corruption_seed_key(game_id), int(view[-1]))[0])
        phases = base.gar.previous.phase_set(inputs, names, sampled_fps)
        phase_counts[view] = len(phases)
        expected_phases = int(round(float(inputs["fps"]) / 5.0))
        if phase_counts[view] != expected_phases:
            raise base.EvaluationError(
                f"GAR {view} phase count {phase_counts[view]} != native grid {expected_phases}")
        if view == "clean" and not np.array_equal(phases[0]["x"], saved_x, equal_nan=True):
            raise base.EvaluationError("Clean phase-0 features differ from frozen GAR dense records")
        native = base.audit.tc.inputs_only(inputs)
        period = base.audit.supervised.model_periods(native)
        fps_for_decode = float(inputs["fps"])
        phase_probabilities = [base._candidate_probability(record, native, adapter, deployment)
                               for record in phases]
        if len({p.shape for p in phase_probabilities}) != 1:
            raise base.EvaluationError("Native phase probability arrays are misaligned")
        probabilities = np.mean(np.stack(phase_probabilities, axis=0), axis=0)
        predicted = np.asarray(adapter.decode_probabilities(probabilities, period, fps_for_decode,
            declaration["selected_package"]["receipt"]))
        if predicted.shape != labels.shape:
            raise base.EvaluationError("Selected adapter decoder returned a prediction with the wrong frame count")
        cell = {"labels": labels, "period": inputs["period"],
                "timestamps": inputs["timestamps"], "fps": fps_for_decode}
        measured = base.audit.score(cell, predicted)
        scores[view] = {key: measured[key] for key in base.METRICS if key in measured}
        predictions[view] = predicted.astype("S1")
    counts = {(row["graded_frames"], row["reference_spell_count"]) for row in scores.values()}
    if len(counts) != 1:
        raise base.EvaluationError("Grading population or reference count changed across candidate views")
    GAMES.mkdir(parents=True, exist_ok=True)
    native_path = target.with_suffix(".npz")
    np.savez_compressed(native_path, labels=labels.astype("S1"), period=clean_inputs["period"],
        timestamps=clean_inputs["timestamps"], fps=clean_inputs["fps"], **predictions)
    _write_once(target, {
        "study": base.STUDY, "split": "test", "game_id": game_id,
        "declaration_sha256": base.sha256(base.DECLARATION),
        "package_receipt_sha256": base.sha256(base.PACKAGE_RECEIPT),
        "execution_amendment_sha256": base.sha256(AMENDMENT),
        "execution_module_sha256": base.sha256(MODULE_PATH),
        "records": len(records), "native_frames": len(labels),
        "graded_frames": int((labels != "X").sum()), "phase_counts": phase_counts,
        "seed_namespace_reused": base.corruption_seed_key(game_id), "scores": scores,
        "prediction_sha256": base.sha256(native_path),
        "grading_and_reference_counts_equal": True,
    })
    return target


def games(workers: int = 3) -> None:
    check_amendment()
    declaration = base.check_declaration()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_game, game_id) for game_id in declaration["cohort"]["match_ids"]]
        for future in as_completed(futures):
            print("DONE", future.result(), flush=True)


def report_receipt() -> dict:
    """Bind the frozen standard report to the GAR-only amended executor."""
    amendment = check_amendment()
    result = base.json_read(base.RESULT)
    cells = []
    for game in base.json_read(base.DECLARATION)["cohort"]["match_ids"]:
        cell_path = GAMES / f"test__{game}.json"
        cell = base.json_read(cell_path)
        if (cell.get("execution_amendment_sha256") != base.sha256(AMENDMENT)
                or cell.get("execution_module_sha256") != base.sha256(MODULE_PATH)
                or cell.get("package_receipt_sha256") != base.json_read(base.DECLARATION)["selected_package"]["receipt_sha256"]):
            raise base.EvaluationError(f"Amended GAR cell has mismatched lineage: {game}")
        cells.append(base.sha256(cell_path))
    receipt = {
        "study": base.STUDY, "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_declaration_sha256": base.sha256(base.DECLARATION),
        "execution_amendment_sha256": base.sha256(AMENDMENT),
        "execution_module_sha256": base.sha256(MODULE_PATH),
        "result_path": str(base.RESULT.relative_to(ROOT)),
        "result_sha256": base.sha256(base.RESULT),
        "game_result_sha256": cells,
        "reported_candidate_v2_spell_f1": result["summary"]["v2"]["spell_f1"]["candidate_equal_game_mean"],
        "organizer_reference_access": False, "public_scoring": False,
    }
    _write_once(RESULT_RECEIPT, receipt)
    return receipt


def report() -> dict:
    """Verify all amended cells, then run the frozen aggregation and bind its result."""
    check_amendment()
    declaration = base.check_declaration()
    for game in declaration["cohort"]["match_ids"]:
        path = GAMES / f"test__{game}.json"
        cell = base.json_read(path)
        if (cell.get("execution_amendment_sha256") != base.sha256(AMENDMENT)
                or cell.get("execution_module_sha256") != base.sha256(MODULE_PATH)
                or cell.get("package_receipt_sha256") != declaration["selected_package"]["receipt_sha256"]):
            raise base.EvaluationError(f"Amended GAR cell is incomplete or unbound: {game}")
    base.summarize()
    return report_receipt()


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal-amendment", "games", "report"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    result = seal_amendment() if args.action == "seal-amendment" else (
        games(args.workers) if args.action == "games" else report())
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
