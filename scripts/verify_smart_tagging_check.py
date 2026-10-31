"""Independent artifact-only verification for the one Smart Tagging public check.

This script never opens evaluation/gt or runs an evaluator. It writes only a
small aggregate receipt; frame predictions and inputs remain local.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ORIGINAL = ROOT / "data/experiments/smart_tagging_validation_v1"
FINAL = ROOT / "data/experiments/smart_tagging_validation_v1_final"
REPORT = ROOT / "experiments/results/smart_tagging_public_check.json"
BASELINE_REPORT = ROOT / "experiments/results/combined_objective_public_check.json"
GLOBAL_GATE = ROOT / "experiments/results/smart_tagging_public_check_started.json"
RECEIPT = ROOT / "experiments/results/smart_tagging_verification.json"
BASELINE_CSV_SHA = "a9beb9ddee36182f926e4d693f648c19f4a19c55afd1653ea93e19f3dd32634b"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(actual: float, expected: float, name: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-14):
        raise AssertionError(f"aggregate mismatch: {name}")


def f1_from_counts(row: dict, gt: int, pred: int) -> float:
    matched = int(row["n_matched"])
    missed = int(row["n_missed"])
    spurious = int(row["n_spurious"])
    if matched + missed != gt or matched + spurious != pred:
        raise AssertionError("saved spell counts do not reconcile")
    denominator = gt + pred
    return 2.0 * matched / denominator if denominator else 0.0


def main() -> None:
    # Imports are limited to public model/inference utilities. No GT module is
    # imported, and the only evaluator inputs below are saved integer counts.
    from experiments import challenge, current_baseline, smart_tagging_prior, supervised
    from experiments import smart_tagging_check
    from starter.load_smart_tagging import load as load_tags

    final_decl = load_json(FINAL / "declaration.json")
    old_decl = load_json(ORIGINAL / "declaration.json")
    seal = load_json(FINAL / "prediction_seal.json")
    report = load_json(REPORT)
    baseline_source = load_json(BASELINE_REPORT)
    global_gate = load_json(GLOBAL_GATE)
    local_gate = load_json(FINAL / "public_check_started.json")
    completion = load_json(FINAL / "public_check_completed.json")
    # Ensure moving this post hoc verifier out of experiments/ preserves the
    # exact frozen source inventory and all declared artifact seals.
    verified_spec = smart_tagging_check.read_spec(FINAL)
    if verified_spec != final_decl:
        raise AssertionError("frozen declaration readback differs")

    if old_decl["files_sha256"] != final_decl["files_sha256"]:
        raise AssertionError("input/model/manifest/policy provenance changed across declarations")
    changed_source_keys = {
        key for key in set(old_decl["source_sha256"]) | set(final_decl["source_sha256"])
        if old_decl["source_sha256"].get(key) != final_decl["source_sha256"].get(key)
    }
    if changed_source_keys != {"experiments/smart_tagging_check.py"}:
        raise AssertionError("source amendments exceed the readout checker")

    # Verify source and declared source/data/model provenance without printing
    # any model or input contents.
    for relative, expected in final_decl["source_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise AssertionError("frozen source hash mismatch")
    for relative, expected in final_decl["files_sha256"].items():
        if sha(ROOT / relative) != expected:
            raise AssertionError("frozen declared input/model hash mismatch")
    for relative in (
        "data/challenge/inputs.npz",
        "data/tactical/BAR-ATH_tactical_smart_tagging.xml",
        "experiments/baseline_manifest.json",
        "experiments/model_decision_policy.json",
    ):
        if relative not in final_decl["files_sha256"]:
            raise AssertionError("required provenance hash missing")

    # Original and final inference products must be byte-identical despite the
    # checker-only corrections. Do not expose either file's contents.
    artifact_names = (
        "baseline.csv", "baseline.npz",
        "smart_tagging_missing_ball.csv", "smart_tagging_missing_ball.npz",
    )
    byte_identity = {}
    for name in artifact_names:
        before, after = ORIGINAL / name, FINAL / name
        if not before.is_file() or not after.is_file() or sha(before) != sha(after):
            raise AssertionError("original/final inference artifact differs")
        byte_identity[name] = sha(after)

    if sha(FINAL / "baseline.csv") != BASELINE_CSV_SHA:
        raise AssertionError("replayed baseline CSV differs from frozen scored baseline")
    if seal.get("declaration_sha256") != sha(FINAL / "declaration.json"):
        raise AssertionError("prediction seal declaration hash mismatch")
    if seal.get("files_sha256") != {
        f"data/experiments/smart_tagging_validation_v1_final/{name}": byte_identity[name]
        for name in artifact_names
    }:
        raise AssertionError("prediction seal artifacts do not match local files")
    if seal.get("rows") != 199319 or seal.get("reference_labels_read") is not False:
        raise AssertionError("prediction seal metadata mismatch")

    # Verify the saved candidate is exactly produced by the frozen adapter on
    # the sealed baseline probabilities and the numeric tracking/tag inputs.
    inputs = current_baseline.load_inputs(ROOT / "data/challenge/inputs.npz")
    manifest = load_json(ROOT / "experiments/baseline_manifest.json")
    with np.load(FINAL / "baseline.npz", allow_pickle=False) as archive:
        base_p = np.array(archive["probabilities"], copy=True)
        base_labels = np.array(archive["prediction"], copy=True)
    with np.load(FINAL / "smart_tagging_missing_ball.npz", allow_pickle=False) as archive:
        candidate_p = np.array(archive["probabilities"], copy=True)
        candidate_labels = np.array(archive["prediction"], copy=True)
    events = load_tags()
    recomputed = smart_tagging_prior.apply_smart_tagging_prior(
        base_p, events, final_decl["team_to_side"], float(inputs["fps"]),
        inputs["period"], inputs["ball_observed"],
        missing_ball_only=final_decl["missing_ball_only"],
        timestamps=inputs["timestamps"],
    )
    if recomputed.tobytes() != candidate_p.tobytes():
        raise AssertionError("candidate probabilities do not replay from frozen prior")
    if base_p.shape != (199319, 3) or candidate_p.shape != base_p.shape:
        raise AssertionError("native probability shape mismatch")
    if base_p[:, 2].tobytes() != candidate_p[:, 2].tobytes():
        raise AssertionError("native P(D) changed")
    observed = np.asarray(inputs["ball_observed"], dtype=bool)
    if not np.array_equal(base_p[observed], candidate_p[observed]):
        raise AssertionError("prior changed probability rows with observed ball")

    periods = supervised.model_periods(inputs)
    decoded_base = supervised.decode(base_p, periods, float(inputs["fps"]), 3)
    decoded_candidate = supervised.decode(candidate_p, periods, float(inputs["fps"]), 3)
    decoded_base[np.asarray(inputs["period"]) <= 0] = "X"
    decoded_candidate[np.asarray(inputs["period"]) <= 0] = "X"
    with tempfile.TemporaryDirectory(prefix="smart-tagging-verification-") as temp:
        for filename, labels in (("baseline.csv", decoded_base),
                                 ("smart_tagging_missing_ball.csv", decoded_candidate)):
            generated = Path(temp) / filename
            challenge.write_submission(labels, generated)
            if sha(generated) != sha(FINAL / filename):
                raise AssertionError("sealed CSV does not decode from saved native probabilities")
    if not np.array_equal(candidate_labels, decoded_candidate) or not np.array_equal(base_labels, decoded_base):
        raise AssertionError("saved NPZ labels differ from fixed cost-3 decode")

    # Validate aggregate-only public report arithmetic; this does not invoke
    # metrics or open reference labels.
    candidate_metrics = report["candidate"]
    for threshold, row in candidate_metrics["spell_f1"].items():
        assert_close(row["f1"], f1_from_counts(row, candidate_metrics["n_gt_spells"],
                                              candidate_metrics["n_pred_spells"]),
                     f"candidate IoU {threshold}")
    candidate_mean = float(np.mean([row["f1"] for row in candidate_metrics["spell_f1"].values()]))
    assert_close(candidate_metrics["spell_f1_mean"], candidate_mean, "candidate mean spell F1")
    baseline = report["saved_baseline"]
    expected_baseline = baseline_source["scores"]["combined_objective_ensemble"]
    if baseline != expected_baseline:
        raise AssertionError("saved baseline readout differs from referenced aggregate")
    for threshold, row in baseline["spell_by_iou"].items():
        assert_close(row["f1"], f1_from_counts(row, baseline["ground_truth_spells"],
                                               baseline["predicted_spells"]),
                     f"baseline IoU {threshold}")
    assert_close(baseline["spell_f1_mean"],
                 np.mean([row["f1"] for row in baseline["spell_by_iou"].values()]),
                 "baseline mean spell F1")
    deltas = report["candidate_minus_baseline"]
    assert_close(deltas["spell_f1_mean"], candidate_metrics["spell_f1_mean"] - baseline["spell_f1_mean"],
                 "spell F1 delta")
    assert_close(deltas["macro_f1"], candidate_metrics["macro_f1"] - baseline["macro_f1"],
                 "macro F1 delta")
    if int(candidate_metrics["n_pred_spells"]) - int(baseline["predicted_spells"]) != deltas["predicted_spells"]:
        raise AssertionError("predicted spell delta mismatch")
    for label, row in (("baseline", baseline), ("candidate", candidate_metrics)):
        expected = .7 * row["spell_f1_mean"] + .3 * row["macro_f1"]
        assert_close(report["auxiliary_70_30"][label], expected, f"{label} auxiliary")

    if GLOBAL_GATE.read_bytes() != (FINAL / "public_check_started.json").read_bytes():
        raise AssertionError("global/local gate records differ")
    if global_gate["declaration_sha256"] != sha(FINAL / "declaration.json"):
        raise AssertionError("gate declaration differs")
    if completion["report_sha256"] != sha(REPORT):
        raise AssertionError("completion receipt does not seal report")
    # The exclusive gate must reject any other output directory.
    from experiments import smart_tagging_check
    try:
        smart_tagging_check.require_open(ROOT / "data/experiments/verification_alternate")
    except ValueError:
        pass
    else:
        raise AssertionError("exclusive public gate is not closed globally")

    receipt = {
        "study_id": report["study_id"],
        "verification": "passed",
        "scope": "artifact integrity and aggregate arithmetic only; no reference read or rescore",
        "reference_labels_opened": False,
        "organizer_evaluator_rerun": False,
        "provenance": {
            "verifier_script": "scripts/verify_smart_tagging_check.py",
            "verifier_source_sha256": sha(Path(__file__).resolve()),
            "final_declaration_sha256": sha(FINAL / "declaration.json"),
            "prediction_seal_sha256": sha(FINAL / "prediction_seal.json"),
            "global_gate_sha256": sha(GLOBAL_GATE),
            "public_report_sha256": sha(REPORT),
            "public_reference_file_sha256_reused_from_saved_report": report["public_gt_sha256"],
            "inputs_model_xml_manifest_policy_hashes_verified": True,
            "original_and_final_prediction_artifacts_byte_identical": True,
            "artifact_sha256": byte_identity,
        },
        "prediction_checks": {
            "baseline_csv_matches_saved_scored_baseline_sha256": True,
            "candidate_reproduces_from_frozen_prior": True,
            "both_csvs_reproduce_from_native_probabilities_at_cost_3": True,
            "native_p_d_bitwise_identical": True,
            "probability_changes_confined_to_missing_ball_frames": True,
        },
        "aggregate_checks": {
            "candidate_all_iou_f1_recomputed_from_integer_counts": True,
            "baseline_all_iou_f1_recomputed_from_integer_counts": True,
            "saved_baseline_and_deltas_verified": True,
            "auxiliary_70_30_verified": True,
            "candidate_spell_f1_mean": candidate_metrics["spell_f1_mean"],
            "baseline_spell_f1_mean": baseline["spell_f1_mean"],
            "candidate_macro_f1": candidate_metrics["macro_f1"],
            "baseline_macro_f1": baseline["macro_f1"],
            "candidate_predicted_spells": candidate_metrics["n_pred_spells"],
            "baseline_predicted_spells": baseline["predicted_spells"],
        },
        "gate_checks": {
            "global_local_started_markers_match": True,
            "closed_across_alternate_output_directories": True,
            "completion_seals_public_report": True,
        },
        "private_artifact_availability": "local ignored data/ artifacts present; this verifier performed no upload",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Smart Tagging artifact verification passed; aggregate receipt written.")


if __name__ == "__main__":
    main()
