"""Package the fixed combined-loss two-seed ensemble after a verified comparison.

This script does not fit, score, or infer. It only consumes sealed deployment
checkpoints, the saved full-provider comparison, and a separately generated
label-free submission/validation receipt.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import shutil

from experiments.eval_set.protocol import sha256


ROOT = Path(__file__).resolve().parents[1]
MODEL = "joint_gru_combined_ensemble"
PACKAGE = MODEL + "_v1"
MEMBERS = {"joint_gru_2026": 0.5, "joint_gru_2027": 0.5}
USER_REQUEST = "try compare and if it works, consider bumping baseline"
REPORT = ROOT / "data/experiments/objective_ensemble_v1/report.json"
COMPARISON_DECLARATION = ROOT / "data/experiments/objective_ensemble_v1/declaration.json"
COMPARISON_RUNS = ROOT / "data/experiments/objective_ensemble_v1/runs.json"
COMPARISON_VERIFICATION = ROOT / "experiments/results/objective_ensemble_verification.json"
EXPECTED_COMPARISON_VERIFICATION_SHA256 = (
    "837bda7c7dbacc1a2215179a73b2c7dfbc7be9ece63c3576bbe17bc1f20e3df2"
)
DEPLOYMENT_OUT = ROOT / "data/experiments/combined_objective_deployment_v1"
DEPLOYMENT_SEAL = DEPLOYMENT_OUT / "checkpoints_frozen.json"
BASELINE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
PRIOR_PROMOTION = ROOT / "experiments/results/baseline_seed_ensemble_promotion.json"
PACKAGE_DIRECTORY = ROOT / "data/models" / PACKAGE
PREVIOUS_MANIFEST = ROOT / "experiments/baselines/joint_gru_seed_ensemble_v1.json"
PREVIOUS_POLICY = ROOT / "experiments/baselines/model_decision_policy_joint_gru_seed_ensemble_v1.json"
CSV_PATH = ROOT / "data/experiments/current_baseline/joint_gru_combined_ensemble.csv"
VALIDATION_PATH = CSV_PATH.with_suffix(".json")
PROMOTION_RECEIPT = ROOT / "experiments/results/baseline_combined_objective_promotion.json"
INPUTS = ROOT / "data/challenge/inputs.npz"
EXPECTED_INPUT_SHA256 = "7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3"
EXPECTED_PRIOR_CSV_SHA256 = "3635b160d7397ca845d86fc2da6331b49df7b3eabf05c6a8eeca9d9d472a57b9"
EXPECTED_ROWS = 199_319
PROVIDER_COHORT = {
    "sportec": {"J03WOY"},
    "metrica": {"metrica_3"},
    "skillcorner": {
        "skillcorner_had_1996435", "skillcorner_had_1996436",
        "skillcorner_had_2006229",
    },
}


def _json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _verify_prediction_sources(source_hashes: object) -> dict[str, str]:
    """Verify every transitive inference source sealed by the prediction run."""
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Prediction declaration is missing its inference source map")
    root = ROOT.resolve()
    paths = {}
    for relative, expected in source_hashes.items():
        if (not isinstance(relative, str) or not relative
                or not isinstance(expected, str) or len(expected) != 64):
            raise ValueError("Prediction declaration contains a malformed inference source entry")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Prediction declaration contains a path outside the repository")
        source = (root / relative_path).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Prediction declaration contains a path outside the repository")
        paths[relative] = (source, expected)
    predictor = "scripts/predict_combined_objective.py"
    if predictor not in paths:
        raise ValueError("Prediction declaration did not bind its label-free predictor")
    verified = {}
    for relative, (source, expected) in paths.items():
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"Prediction inference source changed or is missing: {relative}")
        verified[relative] = expected
    return verified


def verify_comparison(report_path: Path = REPORT) -> dict:
    """Bind the passing report to its fixed, independent verification receipt."""
    report_path = Path(report_path)
    report = _json(report_path)
    declaration_path = report_path.parent / "declaration.json"
    runs_path = report_path.parent / "runs.json"
    verification_path = COMPARISON_VERIFICATION
    if sha256(verification_path) != EXPECTED_COMPARISON_VERIFICATION_SHA256:
        raise ValueError("Independent comparison verification receipt changed")
    verification = _json(verification_path)
    verified_inputs = verification.get("inputs", {})
    if (report.get("study_id") != "objective_ensemble_v1"
            or report.get("declaration_sha256") != sha256(declaration_path)
            or report.get("runs_sha256") != sha256(runs_path)
            or verification.get("study_id") != "objective_ensemble_v1"
            or verified_inputs.get("report_sha256") != sha256(report_path)
            or verified_inputs.get("runs_sha256") != sha256(runs_path)
            or verified_inputs.get("declaration_sha256") != sha256(declaration_path)):
        raise ValueError("Independent verifier checked different comparison artifacts")
    verification_checks = verification.get("checks", {})
    required_verification = {
        "expected_cells": 15,
        "baseline_saved_score_rows_match_prior_report_at_abs_tolerance_1e_12": 15,
        "baseline_probability_and_decode_digests_reproduced": 15,
        "candidate_two_seed_native_probability_means_exact": 15,
        "candidate_ensemble_decode_prediction_digests_reproduced": 15,
        "candidate_output_and_member_file_array_digests_match": 15,
        "independently_recomputed_candidate_minus_control_deltas_match": True,
        "independently_recomputed_provider_and_equal_provider_aggregates_max_abs_delta": 0.0,
        "post_hoc_selection_gate_passes": True,
    }
    if any(verification_checks.get(key) != value
           for key, value in required_verification.items()):
        raise ValueError("Independent verification did not pass all comparison integrity checks")
    verifier_gate = verification_checks.get("selection_gate_checks", {})
    if not verifier_gate or any(value is not True for value in verifier_gate.values()):
        raise ValueError("Independent verification found a failed comparison screen")

    active = _json(BASELINE)
    policy = _json(POLICY)
    baseline_facts = report.get("baseline", {})
    baseline_sha, policy_sha = sha256(BASELINE), sha256(POLICY)
    if (active.get("model"), active.get("package_version")) != (
            "joint_gru_seed_ensemble", "joint_gru_seed_ensemble_v1"):
        raise ValueError("The expected current two-seed baseline is no longer active")
    if (baseline_facts.get("manifest_sha256") != baseline_sha
            or baseline_facts.get("decision_policy_sha256") != policy_sha):
        raise ValueError("Comparison used a different baseline or decision policy")
    if (baseline_facts.get("model") != active["model"]
            or baseline_facts.get("package_version") != active["package_version"]
            or baseline_facts.get("model_sha256") != active.get("models_sha256")):
        raise ValueError("Comparison baseline model inventory differs from the active package")
    if policy.get("working_candidate", {}).get("model") != active["model"]:
        raise ValueError("Decision policy does not name the active baseline")
    if policy["working_candidate"].get("baseline_manifest_sha256") != baseline_sha:
        raise ValueError("Decision policy does not bind the active manifest")
    if baseline_facts.get("promotion_receipt_sha256") != sha256(PRIOR_PROMOTION):
        raise ValueError("Comparison did not bind the active baseline promotion receipt")
    active_evidence = active.get("evidence", {})
    if (baseline_facts.get("saved_report_sha256") != active_evidence.get("report_sha256")
            or baseline_facts.get("verification_sha256") != active_evidence.get("verification_sha256")):
        raise ValueError("Comparison baseline evidence differs from the active manifest")
    if verified_inputs.get("prior_baseline_report_sha256") != active_evidence.get("report_sha256"):
        raise ValueError("Independent verifier used a different baseline report")
    if report.get("cohort") is None:
        raise ValueError("Comparison report is missing its evaluation cohort")
    cohort = {source: set() for source in PROVIDER_COHORT}
    for row in report["cohort"]:
        cohort[row["source"]].add(row["match"])
    if cohort != PROVIDER_COHORT:
        raise ValueError("Comparison evaluation cohort differs from the frozen five matches")
    status = report.get("evaluation_status", {})
    if status.get("organizer_labels_or_scores_used") is not False:
        raise ValueError("Organizer labels/scores must not enter the comparison")
    if status.get("independent_confirmation") is not False:
        raise ValueError("Comparison must retain its historical-development qualification")
    comparison = report.get("comparison", {})
    if (comparison.get("candidate") != "combined_objective_equal_probability_ensemble"
            or comparison.get("control") != "current_baseline_equal_probability_ensemble"
            or "cost 3" not in comparison.get("decoder", "").lower()
            or "equal arithmetic mean" not in comparison.get("probability_rule", "").lower()):
        raise ValueError("Comparison does not match the declared two-seed probability comparison")

    selection = report.get("result", {}).get("selection_gate", {})
    if (selection.get("automatic_promotion") is not False
            or selection.get("post_hoc") is not True
            or selection.get("passes") is not True
            or selection.get("checks") != verifier_gate):
        raise ValueError("The comparison screen did not pass, or incorrectly enables auto-promotion")
    result = report["result"]
    deltas = result["candidate_minus_control"]
    for view in ("clean", "feed_like"):
        delta = deltas[view]["equal_provider_mean"]["spell_f1"]
        if (delta <= 0
                or abs(float(verification["equal_provider_spell_f1_delta"][view])
                       - float(delta)) > 1e-12):
            raise ValueError(f"Independent {view} spell delta differs or is not positive")

    return dict(
        report=str(report_path.relative_to(ROOT)),
        report_sha256=sha256(report_path),
        independent_verification=str(verification_path.relative_to(ROOT)),
        independent_verification_sha256=sha256(verification_path),
        declaration_sha256=report["declaration_sha256"],
        runs_sha256=report["runs_sha256"],
        prior_baseline_manifest_sha256=baseline_sha,
        prior_decision_policy_sha256=policy_sha,
        prior_promotion_receipt_sha256=baseline_facts["promotion_receipt_sha256"],
        selection_gate=selection,
        result=result,
        post_hoc=bool(status.get("post_hoc")),
        independent_confirmation=False,
        organizer_labels_or_scores_used=False,
    )


def _verify_replay_file(replay_path: str, csv_path: Path) -> Path:
    replay_file = (ROOT / replay_path).resolve()
    # The ignored data directory may be mounted or linked outside the checkout.
    # Keep the replay beside the authorized candidate, with its exact prior seal.
    if (replay_file.parent != csv_path.resolve().parent
            or sha256(replay_file) != EXPECTED_PRIOR_CSV_SHA256):
        raise ValueError("Current-baseline replay file differs from its byte-parity seal")
    return replay_file


def verify_label_free_validation(receipt_path: Path = VALIDATION_PATH,
                                 csv_path: Path = CSV_PATH,
                                 model_hashes: dict | None = None) -> dict:
    """Check the label-free submission receipt and its actual CSV bytes."""
    receipt_path, csv_path = Path(receipt_path), Path(csv_path)
    receipt = _json(receipt_path)
    declaration_path = csv_path.with_suffix(".declaration.json")
    if (not declaration_path.is_file()
            or receipt.get("declaration_sha256") != sha256(declaration_path)):
        raise ValueError("Prediction receipt does not bind its source/input declaration")
    declaration = _json(declaration_path)
    if (declaration.get("input_sha256") != EXPECTED_INPUT_SHA256
            or declaration.get("current_manifest_sha256") != sha256(BASELINE)
            or declaration.get("organizer_reference_labels_read") is not False
            or declaration.get("organizer_scored") is not False):
        raise ValueError("Prediction declaration used unexpected inputs or baseline")
    source_hashes = _verify_prediction_sources(declaration.get("sources_sha256"))
    if sha256(INPUTS) != EXPECTED_INPUT_SHA256 or receipt.get("input_sha256") != EXPECTED_INPUT_SHA256:
        raise ValueError("Prediction receipt does not use the sealed challenge input")
    if not csv_path.is_file() or sha256(csv_path) != receipt.get("output_sha256"):
        raise ValueError("Candidate CSV is missing or differs from the validation receipt")
    if receipt.get("rows") != EXPECTED_ROWS:
        raise ValueError("Candidate CSV has the wrong native row count")
    format_result = receipt.get("format_validation", {})
    if format_result.get("valid") is not True or format_result.get("rows") != EXPECTED_ROWS:
        raise ValueError("Candidate CSV has no passing format-validation receipt")
    if (model_hashes is not None
            and (receipt.get("model_sha256") != model_hashes
                 or declaration.get("model_sha256") != model_hashes)):
        raise ValueError("CSV receipt used different deployment checkpoints")
    member_hashes = receipt.get("member_probability_sha256", {})
    if set(member_hashes) != set(MEMBERS) or any(
            len(str(value)) != 64 for value in member_hashes.values()):
        raise ValueError("CSV receipt is missing the two member probability hashes")
    blended_hash = receipt.get("blended_probability_sha256")
    if not isinstance(blended_hash, str) or len(blended_hash) != 64:
        raise ValueError("CSV receipt is missing the blended probability hash")
    if receipt.get("current_baseline_replay_sha256") != EXPECTED_PRIOR_CSV_SHA256:
        raise ValueError("Current-baseline replay did not reproduce its sealed CSV")
    replay_path = receipt.get("current_baseline_replay_path")
    if replay_path:
        _verify_replay_file(replay_path, csv_path)
    if receipt.get("organizer_reference_labels_read") is not False:
        raise ValueError("Organizer reference labels must not be read")
    if receipt.get("organizer_scored") is not False or receipt.get("organizer_score") is not None:
        raise ValueError("Promotion receipt must not include organizer scoring")
    if receipt.get("organizer_labels_read", False) is not False:
        raise ValueError("Organizer labels must not be read")
    # Count rows without inspecting any reference targets.
    with csv_path.open(newline="", encoding="utf-8") as stream:
        row_count = sum(1 for _ in csv.reader(stream))
    if row_count not in (EXPECTED_ROWS, EXPECTED_ROWS + 1):
        raise ValueError(f"Candidate CSV contains {row_count} physical rows, expected {EXPECTED_ROWS}")
    return dict(
        receipt=str(receipt_path.relative_to(ROOT)),
        receipt_sha256=sha256(receipt_path),
        declaration=str(declaration_path.relative_to(ROOT)),
        declaration_sha256=sha256(declaration_path),
        csv=str(csv_path.relative_to(ROOT)),
        csv_sha256=sha256(csv_path),
        input_sha256=receipt["input_sha256"],
        rows=EXPECTED_ROWS,
        current_baseline_replay_sha256=receipt["current_baseline_replay_sha256"],
        model_sha256=receipt["model_sha256"],
        prediction_sources_sha256=source_hashes,
        member_probability_sha256=member_hashes,
        blended_probability_sha256=blended_hash,
        format_validation=format_result,
        organizer_reference_labels_read=False,
        organizer_scored=False,
        organizer_score=None,
    )


def _deployment_assets() -> tuple[dict, dict[str, Path], dict[str, str]]:
    """Load only the sealed deployment outputs; never launch a fit."""
    from experiments import combined_objective_deployment as deployment

    declaration = deployment.check()
    declaration_path = Path(deployment.OUT) / "predeclared.json"
    if not declaration_path.is_file() or _json(declaration_path) != declaration:
        raise ValueError("Deployment check did not return its authoritative frozen declaration")
    seal_path = getattr(deployment, "SEAL", DEPLOYMENT_SEAL)
    if not Path(seal_path).is_file():
        raise ValueError("Both deployment fits must be sealed before packaging")
    sealed = _json(Path(seal_path))
    plan = declaration.get("split_plan", declaration.get("pool_plan"))
    expected_matches = [item["match"] for item in plan["train"]]
    if (len(expected_matches) != 13 or declaration.get("objective") != "combined"
            or declaration.get("loss_coefficients") != {"spell": 1.0, "transition": 0.1}):
        raise ValueError("Deployment declaration does not bind the pooled combined objective")
    paths, hashes = {}, {}
    frozen_rows = sealed.get("checkpoints", sealed.get("new_checkpoints", []))
    frozen_files = {row.get("file"): row for row in frozen_rows}
    for seed in (2026, 2027):
        key = f"joint_gru_{seed}"
        path = Path(deployment.checkpoint(seed))
        if not path.is_file():
            raise ValueError(f"Missing combined-objective deployment checkpoint: {path}")
        digest = sha256(path)
        receipt_path = path.with_suffix(".json")
        if not receipt_path.is_file():
            raise ValueError(f"Missing deployment fit receipt: {receipt_path}")
        fit_receipt = _json(receipt_path)
        model_receipt_hash = sha256(receipt_path)
        if fit_receipt.get("sha256", digest) != digest:
            raise ValueError(f"Deployment fit receipt does not seal checkpoint {path.name}")
        if frozen_files:
            row = frozen_files.get(path.name)
            if row is None or row.get("sha256") != digest:
                raise ValueError(f"Checkpoint is absent from the sealed deployment manifest: {path.name}")
        deployment.verify_model(seed, path)
        model = deployment.load_model(seed)
        if (model.get("seed") != seed or model.get("objective") != "combined"
                or model.get("loss_coefficients") != {"spell": 1.0, "transition": 0.1}
                or model.get("player_only") is not True):
            raise ValueError(f"Unexpected combined deployment model contents: {path.name}")
        if fit_receipt.get("training_matches") != expected_matches:
            raise ValueError(f"Deployment fit omitted or added training matches: {path.name}")
        if fit_receipt.get("seed") != seed or fit_receipt.get("objective") != "combined":
            raise ValueError(f"Deployment fit receipt has wrong provenance: {receipt_path}")
        paths[key] = path
        hashes[f"{key}.joblib"] = digest
        hashes[f"{key}.receipt.json"] = model_receipt_hash
    return dict(declaration=declaration,
                declaration_path=str(declaration_path.relative_to(ROOT)),
                declaration_sha256=sha256(declaration_path),
                seal_sha256=sha256(Path(seal_path))), paths, hashes


def _preserve(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing historical/package file differs: {path}")
    else:
        path.write_bytes(content)


def promote() -> dict:
    """Promote previously verified deployment artifacts; never fit or score."""
    active = _json(BASELINE)
    policy = _json(POLICY)
    if active.get("model") == MODEL:
        if not PROMOTION_RECEIPT.is_file():
            raise ValueError("New baseline is active but its promotion receipt is missing")
        receipt = _json(PROMOTION_RECEIPT)
        if (receipt.get("baseline_manifest_sha256") != sha256(BASELINE)
                or receipt.get("models_sha256") != active.get("models_sha256")
                or policy.get("working_candidate", {}).get("model") != MODEL
                or policy.get("working_candidate", {}).get("baseline_manifest_sha256") != sha256(BASELINE)):
            raise ValueError("Existing promotion receipt, active manifest, and policy differ")
        for filename, digest in active.get("models_sha256", {}).items():
            if sha256(PACKAGE_DIRECTORY / filename) != digest:
                raise ValueError(f"Active combined-objective checkpoint changed: {filename}")
        return receipt
    if (active.get("model"), active.get("package_version")) != (
            "joint_gru_seed_ensemble", "joint_gru_seed_ensemble_v1"):
        raise ValueError("Unexpected active baseline; refusing to overwrite it")
    if PROMOTION_RECEIPT.exists():
        raise ValueError("Promotion receipt already exists; inspect before proceeding")

    comparison = verify_comparison()
    deployment, sources, deployment_hashes = _deployment_assets()
    package_hashes = {f"{name}.joblib": digest for name, digest in
                      ((member, sha256(path)) for member, path in sources.items())}
    validation = verify_label_free_validation(model_hashes=package_hashes)

    package = PACKAGE_DIRECTORY
    package.mkdir(parents=True, exist_ok=True)
    for member, source in sources.items():
        target = package / f"{member}.joblib"
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError(f"Package conflict for {target.name}")
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256(target) != sha256(source):
            raise ValueError(f"Copied deployment checkpoint differs: {target}")
    hashes = {f"{member}.joblib": sha256(package / f"{member}.joblib")
              for member in MEMBERS}

    architecture = dict(
        kind="joint_gru_ensemble",
        members=MEMBERS,
        combination="Equal mean of native H/A/D probabilities before unchanged Viterbi decoder",
        network="Two factorized bidirectional GRUs, hidden24 each; player-only D path",
        rate_hz=5, core_seconds=10, context_each_side_seconds=5,
        decoder_switch_cost=3,
        training="Pooled thirteen external TRAIN matches; fixed combined spell/transition objective; seeds 2026/2027",
    )
    candidate = dict(
        version=5,
        model=MODEL,
        package_version=PACKAGE,
        status="User-authorized working development baseline; independent confirmation pending",
        promoted_utc=datetime.now(timezone.utc).isoformat(),
        replaced_baseline=active["model"],
        previous_manifest=str(PREVIOUS_MANIFEST.relative_to(ROOT)),
        previous_manifest_sha256=sha256(BASELINE),
        previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)),
        previous_policy_sha256=sha256(POLICY),
        model_directory=str(package.relative_to(ROOT)),
        models_sha256=hashes,
        default_output=str(CSV_PATH.relative_to(ROOT)),
        architecture=architecture,
        evidence=dict(
            comparison=comparison,
            deployment_declaration=deployment["declaration_path"],
            deployment_declaration_sha256=deployment["declaration_sha256"],
            deployment_seal_sha256=deployment["seal_sha256"],
            deployment_model_hashes={key: value for key, value in deployment_hashes.items()
                                     if key.endswith(".joblib")},
            deployment_fit_receipt_hashes={key: value for key, value in deployment_hashes.items()
                                           if key.endswith(".receipt.json")},
            label_free_validation=validation,
        ),
        promotion=dict(
            type="explicit_user_authorized_conditional_working_baseline_update",
            user_request=USER_REQUEST,
            comparison_gate_passed=True,
            automatic_promotion=False,
            fitting_during_promotion=False,
            organizer_rescoring_performed=False,
            organizer_reference_labels_used=False,
        ),
        organizer_public_diagnostic=dict(
            report="experiments/results/combined_objective_public_check.json",
            status="Separately requested diagnostic; completion and scores live in the report",
            known_at_promotion=False,
            used_for_training_tuning_selection_or_adoption=False,
            user_request="looks good. don't forget to update pages + also compute public score",
        ),
        organizer_reference_labels_used_for_training=False,
        historical_selection_records_preserved=True,
        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in (
            Path(__file__), ROOT / "experiments/current_baseline.py",
            ROOT / "experiments/objective_ensemble.py",
            ROOT / "experiments/combined_objective_deployment.py",
        )},
    )
    manifest_bytes = (json.dumps(candidate, indent=2, allow_nan=False) + "\n").encode()
    new_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    new_policy = dict(policy)
    new_policy["version"] = 5
    new_policy["working_candidate"] = dict(
        family="Two-seed combined-objective joint GRU probability ensemble",
        model=MODEL,
        status=candidate["status"],
        baseline_manifest="experiments/baseline_manifest.json",
        baseline_manifest_sha256=new_manifest_sha,
        prediction_entrypoint="python -m experiments.current_baseline predict",
        reason=("Post-hoc full-provider comparison passed: clean spell .633205 and "
                "feed_like spell .498830 versus .626531/.487419 for the prior ensemble."),
        promotion_record=str(PROMOTION_RECEIPT.relative_to(ROOT)),
        previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)),
        public_diagnostic_known_before_user_request=False,
    )
    new_policy["stress_guardrail"]["reference_recipe"] = MODEL
    new_policy["stress_guardrail"]["rule"] = (
        "Full-pipeline candidate feed_like spell >= adopted ensemble -.01; report "
        "component seeds separately and protect provider clean D-F1/macro within .02")
    policy_bytes = (json.dumps(new_policy, indent=2, allow_nan=False) + "\n").encode()

    _preserve(PREVIOUS_MANIFEST, BASELINE.read_bytes())
    _preserve(PREVIOUS_POLICY, POLICY.read_bytes())
    _preserve(package / "baseline_manifest.json", manifest_bytes)
    receipt = dict(
        package=PACKAGE,
        models_sha256=hashes,
        baseline_manifest_sha256=new_manifest_sha,
        previous_manifest_sha256=sha256(PREVIOUS_MANIFEST),
        previous_policy_sha256=sha256(PREVIOUS_POLICY),
        model_decision_policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
        evidence=candidate["evidence"],
        validation=validation,
        organizer_labels_read=False,
        organizer_scored=False,
        organizer_score=None,
    )
    _preserve(PROMOTION_RECEIPT,
              (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode())
    # Active pointers are changed only after every upstream artifact is sealed.
    BASELINE.write_bytes(manifest_bytes)
    POLICY.write_bytes(policy_bytes)
    return receipt


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("promote",))
    parser.parse_args()
    promote()
