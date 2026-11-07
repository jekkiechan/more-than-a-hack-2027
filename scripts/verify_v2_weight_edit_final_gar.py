#!/usr/bin/env python3
"""Finalize independent GAR verification after the sealed amended run.

This wrapper binds the GAR-only execution amendment and final report receipt,
then delegates native prediction metric/aggregation replay to the preserved
development verifier. It never opens the organizer reference or scores public.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts import verify_v2_weight_edit as verifier

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
GAR_ROOT = ROOT / "data/experiments/v2_weight_edit_evaluation_v1/games"
GAR_DECL = RESULTS / "v2_weight_edit_evaluation_v1_declaration.json"
GAR_RESULT = RESULTS / "v2_weight_edit_evaluation_v1.json"
GAR_MODULE = ROOT / "experiments/v2_weight_edit_evaluation_v1.py"
EXECUTOR = ROOT / "experiments/v2_weight_edit_gar_execution_amended_v1.py"
AMENDMENT = RESULTS / "v2_weight_edit_evaluation_v1_gar_execution_amendment.json"
FAILED_ATTEMPT = RESULTS / "v2_weight_edit_evaluation_v1_gar_failed_attempt.json"
RESULT_RECEIPT = RESULTS / "v2_weight_edit_evaluation_v1_gar_execution_result_receipt.json"
PUBLIC_RESULT = RESULTS / "v2_weight_edit_evaluation_v1_public_result.json"
PUBLIC_RAW = ROOT / "data/experiments/v2_weight_edit_evaluation_v1_public/score_raw.json"
DEPLOYMENT = ROOT / "data/experiments/v2_weight_edit_pooled_deployment_v1/deployment.json"
EARLY_REPORT = RESULTS / "v2_weight_edit_development_v1_independent_verification.json"
FINAL_REPORT = RESULTS / "v2_weight_edit_final_verification.json"
VERIFIER_SOURCE = Path(verifier.__file__).resolve()
FINAL_WRAPPER_SOURCE = Path(__file__).resolve()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_final_gar() -> dict:
    needed = (GAR_DECL, GAR_RESULT, GAR_MODULE, EXECUTOR, AMENDMENT,
              FAILED_ATTEMPT, RESULT_RECEIPT, PUBLIC_RESULT, PUBLIC_RAW,
              DEPLOYMENT, EARLY_REPORT)
    missing = [str(path) for path in needed if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final GAR verification awaits saved artifacts: {missing}")

    declaration, amendment = read(GAR_DECL), read(AMENDMENT)
    failed = read(FAILED_ATTEMPT)
    deployment, result, receipt = map(read, (DEPLOYMENT, GAR_RESULT, RESULT_RECEIPT))
    early = read(EARLY_REPORT)
    eval_rel = str(GAR_MODULE.relative_to(ROOT))
    expected_module_sha = declaration.get("source_sha256", {}).get(eval_rel)
    if (not expected_module_sha or verifier.sha(GAR_MODULE) != expected_module_sha
            or amendment.get("original_declaration_sha256") != verifier.sha(GAR_DECL)
            or amendment.get("frozen_evaluation_module_path") != eval_rel
            or amendment.get("frozen_evaluation_module_sha256") != expected_module_sha
            or amendment.get("amended_executor_path") != str(EXECUTOR.relative_to(ROOT))
            or amendment.get("amended_executor_sha256") != verifier.sha(EXECUTOR)):
        raise AssertionError("amended GAR executor/declaration provenance mismatch")
    if (amendment.get("study") != declaration.get("study")
            or amendment.get("winner_id") != declaration["selected_package"]["receipt"]["winner_id"]
            or amendment.get("deployment_receipt_sha256") != verifier.sha(DEPLOYMENT)
            or amendment.get("public_result_sha256") != verifier.sha(PUBLIC_RESULT)
            or amendment.get("public_scoring_affected") is not False):
        raise AssertionError("GAR amendment winner/package/public binding mismatch")

    failed_path = ROOT / amendment["failed_attempt"]["receipt_path"]
    if (failed_path != FAILED_ATTEMPT
            or verifier.sha(FAILED_ATTEMPT) != amendment["failed_attempt"]["receipt_sha256"]
            or failed.get("original_declaration_sha256") != verifier.sha(GAR_DECL)
            or failed.get("frozen_evaluation_module_sha256") != expected_module_sha
            or failed.get("failed_log_sha256") != verifier.sha(ROOT / failed["failed_log_path"])
            or failed.get("game_score_artifacts_written") is not False
            or failed.get("model_scoring_completed") is not False
            or failed.get("organizer_reference_access") is not False
            or failed.get("public_result_sha256") != verifier.sha(PUBLIC_RESULT)):
        raise AssertionError("failed initial attempt receipt is inconsistent")

    public_check = verifier.verify_public_saved_result()
    if (public_check is None or amendment.get("public_result_sha256") != public_check["public_result_sha256"]
            or amendment.get("public_result_sha256") != verifier.sha(PUBLIC_RESULT)):
        raise AssertionError("saved public report changed after the GAR amendment")

    # This replays all 10×5 saved native predictions and official per-cell
    # metrics, then independently reproduces equal-game means and paired
    # bootstrap intervals. It never runs the adapter or reads public labels.
    gar_replay = verifier.verify_gar_readout()
    if gar_replay is None or gar_replay.get("gar_game_count") != 10:
        raise AssertionError("GAR report or all ten saved game artifacts are missing")

    game_ids = [str(game) for game in declaration["cohort"]["match_ids"]]
    if len(set(game_ids)) != 10:
        raise AssertionError("GAR declaration does not contain ten unique TEST games")
    game_receipt_hashes = []
    for game in game_ids:
        sidecar = GAR_ROOT / f"test__{game}.json"
        array = GAR_ROOT / f"test__{game}.npz"
        cell = read(sidecar)
        if (cell.get("execution_amendment_sha256") != verifier.sha(AMENDMENT)
                or cell.get("execution_module_sha256") != verifier.sha(EXECUTOR)
                or cell.get("package_receipt_sha256") != declaration["selected_package"]["receipt_sha256"]
                or cell.get("declaration_sha256") != verifier.sha(GAR_DECL)
                or cell.get("prediction_sha256") != verifier.sha(array)):
            raise AssertionError(f"amended GAR cell receipt mismatch: {game}")
        with __import__("numpy").load(array, allow_pickle=False) as data:
            fps = float(data["fps"].item())
            expected_count = int(round(fps / 5.0))
            if expected_count != 6:
                raise AssertionError(f"Expected six native 5 Hz phases for GAR {game}, got {expected_count}")
            if set(cell.get("phase_counts", {})) != set(verifier.GAR_VIEWS):
                raise AssertionError(f"GAR view-phase population is incomplete: {game}")
            if any(int(cell["phase_counts"][view]) != expected_count for view in verifier.GAR_VIEWS):
                raise AssertionError(f"GAR native phase count mismatch: {game}")
            if any(view not in data.files for view in verifier.GAR_VIEWS):
                raise AssertionError(f"GAR prediction array missing: {game}")
        game_receipt_hashes.append(verifier.sha(sidecar))

    receipt_result_path = ROOT / receipt["result_path"]
    if (receipt.get("study") != declaration.get("study")
            or receipt.get("original_declaration_sha256") != verifier.sha(GAR_DECL)
            or receipt.get("execution_amendment_sha256") != verifier.sha(AMENDMENT)
            or receipt.get("execution_module_sha256") != verifier.sha(EXECUTOR)
            or receipt_result_path != GAR_RESULT
            or receipt.get("result_sha256") != verifier.sha(GAR_RESULT)
            or receipt.get("game_result_sha256") != game_receipt_hashes
            or receipt.get("organizer_reference_access") is not False
            or receipt.get("public_scoring") is not False):
        raise AssertionError("final GAR result receipt does not bind the amended run outputs")
    if (result.get("declaration_sha256") != verifier.sha(GAR_DECL)
            or result.get("organizer_reference_access") is not False
            or result.get("public_scoring") is not False
            or early.get("primary_result_sha256") != verifier.sha(RESULTS / "v2_weight_edit_development_v1.json")):
        raise AssertionError("final GAR result or early development report lineage mismatch")
    reported_mean = result["summary"]["v2"]["spell_f1"]["candidate_equal_game_mean"]
    if abs(float(receipt["reported_candidate_v2_spell_f1"]) - float(reported_mean)) > verifier.EPS:
        raise AssertionError("GAR result receipt v2 spell mean differs from the saved report")

    return {
        "verification": "passed", "gar_native_predictions_replayed": 50,
        "gar_games": len(game_ids), "native_phase_count_per_view": 6,
        "gar_metrics_aggregates_and_bootstrap_replayed": True,
        "amendment_sha256": verifier.sha(AMENDMENT),
        "amended_executor_sha256": verifier.sha(EXECUTOR),
        "failed_attempt_receipt_sha256": verifier.sha(FAILED_ATTEMPT),
        "gar_result_sha256": verifier.sha(GAR_RESULT),
        "gar_result_receipt_sha256": verifier.sha(RESULT_RECEIPT),
        "deployment_receipt_sha256": verifier.sha(DEPLOYMENT),
        "development_verification_report_sha256": verifier.sha(EARLY_REPORT),
        "preserved_verifier_source_sha256": verifier.sha(VERIFIER_SOURCE),
        "final_wrapper_source_sha256": verifier.sha(FINAL_WRAPPER_SOURCE),
        "public_result_sha256": verifier.sha(PUBLIC_RESULT),
        "public_raw_receipt_sha256": verifier.sha(PUBLIC_RAW),
        "public_reference_file_opened": False, "public_score_recomputed": False,
        "candidate_inference_rerun": False,
        "gar_candidate_v2_spell_f1_equal_game_mean": float(reported_mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report = verify_final_gar()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.write_report:
        with FINAL_REPORT.open("x") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
