"""One bounded public-window score of two sealed combined-objective CSVs.

The fixed pair is the promoted combined-objective ensemble and the byte-sealed
replay of the preceding original-loss ensemble. The exclusive public gate is
claimed before this module opens the organizer reference file. The result is a
requested diagnostic only; it cannot select or change the active baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# The fixed predictor writes under ROOT/data, which is the repository's
# authorized private-data link in this checkout.
DATA = ROOT / "data"
OUT = DATA / "experiments/combined_objective_public_check_v1"
PREDECLARED = OUT / "public_check_predeclared.json"
STARTED = OUT / "public_check_started.json"
PRIVATE_REPORT = OUT / "public_check_result.json"
PUBLIC_REPORT = ROOT / "experiments/results/combined_objective_public_check.json"

COMBINED_CSV = DATA / "experiments/current_baseline/joint_gru_combined_ensemble.csv"
PRECEDING_CSV = DATA / "experiments/current_baseline/preceding_seed_ensemble_replay.csv"
PREDICTION_RECEIPT = COMBINED_CSV.with_suffix(".json")
PREDICTION_DECLARATION = COMBINED_CSV.with_suffix(".declaration.json")
INPUTS = DATA / "challenge/inputs.npz"
BASELINE_MANIFEST = ROOT / "experiments/baseline_manifest.json"
DECISION_POLICY = ROOT / "experiments/model_decision_policy.json"
PROMOTION_RECEIPT = ROOT / "experiments/results/baseline_combined_objective_promotion.json"
COMPARISON_REPORT = ROOT / "experiments/results/objective_ensemble.json"
COMPARISON_VERIFICATION = ROOT / "experiments/results/objective_ensemble_verification.json"
COMPARISON_RUNS = DATA / "experiments/objective_ensemble_v1/runs.json"
HISTORICAL_PUBLIC_REPORT = ROOT / "experiments/results/current_baseline_public_check.json"
REFERENCE_CSV = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"

STUDY_ID = "combined_objective_public_check_v1"
EXPECTED_INPUT_SHA256 = "7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3"
EXPECTED_PRECEDING_CSV_SHA256 = "3635b160d7397ca845d86fc2da6331b49df7b3eabf05c6a8eeca9d9d472a57b9"
EXPECTED_FRAMES = 199_319
EXPECTED_GRADED = 7_500
EXPECTED_GT_SPELLS = 17
BOOTSTRAP_DRAWS = 1_000
IOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)

# These sources affect the frozen prediction, its frame mask, or the official
# public metric. The candidate prediction declaration adds its own transitive
# source map, which is also rechecked in full.
SOURCE_RELATIVE = (
    "experiments/combined_objective_public_check.py",
    "experiments/promote_combined_objective_baseline.py",
    "experiments/current_baseline.py",
    "experiments/combined_objective_deployment.py",
    "experiments/learned_temporal.py",
    "experiments/supervised.py",
    "experiments/factorized_had.py",
    "experiments/live_gap_experiment.py",
    "experiments/robust.py",
    "experiments/challenge.py",
    "experiments/refine_features.py",
    "experiments/refine_domain.py",
    "experiments/stress.py",
    "experiments/eval_set/protocol.py",
    "scripts/predict_combined_objective.py",
    "evaluation/possession_metrics.py",
    "games.py",
)
PREDICTION_REQUIRED_SOURCES = {
    "scripts/predict_combined_objective.py",
    "experiments/current_baseline.py",
    "experiments/combined_objective_deployment.py",
    "experiments/learned_temporal.py",
    "experiments/supervised.py",
    "experiments/factorized_had.py",
    "experiments/live_gap_experiment.py",
    "experiments/robust.py",
    "experiments/challenge.py",
    "evaluation/possession_metrics.py",
}


class PublicCheckError(RuntimeError):
    """Raised for a changed seal, invalid fixed output, or closed gate."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PublicCheckError(f"Refusing to overwrite sealed public-check artifact: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _display_path(path: Path) -> str:
    try:
        return Path(path).relative_to(ROOT).as_posix()
    except ValueError:
        return str(Path(path))


def _source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_RELATIVE:
        path = ROOT / relative
        if not path.is_file():
            raise PublicCheckError(f"Missing public-check source: {relative}")
        result[relative] = sha256(path)
    return result


def _artifact_paths(model_manifest: dict[str, Any] | None = None) -> dict[str, Path]:
    paths = {
        "combined_csv": COMBINED_CSV,
        "preceding_ensemble_replay_csv": PRECEDING_CSV,
        "prediction_receipt": PREDICTION_RECEIPT,
        "prediction_declaration": PREDICTION_DECLARATION,
        "challenge_inputs": INPUTS,
        "active_baseline_manifest": BASELINE_MANIFEST,
        "active_decision_policy": DECISION_POLICY,
        "promotion_receipt": PROMOTION_RECEIPT,
        "objective_comparison_report": COMPARISON_REPORT,
        "objective_comparison_verification": COMPARISON_VERIFICATION,
        "objective_comparison_runs": COMPARISON_RUNS,
        "historical_public_check_report": HISTORICAL_PUBLIC_REPORT,
    }
    if model_manifest is not None:
        directory = Path(model_manifest["model_directory"])
        for filename in sorted(model_manifest["models_sha256"]):
            if directory.is_absolute():
                model_path = directory / filename
            elif directory.parts and directory.parts[0] == "data":
                model_path = DATA.joinpath(*directory.parts[1:], filename)
            else:
                model_path = ROOT / directory / filename
            paths[f"model:{filename}"] = model_path
        package_manifest = paths["active_baseline_manifest"].parent
        if directory.parts and directory.parts[0] == "data":
            package_manifest = DATA.joinpath(*directory.parts[1:])
        elif directory.is_absolute():
            package_manifest = directory
        else:
            package_manifest = ROOT / directory
        paths["model_package_manifest"] = package_manifest / "baseline_manifest.json"
    return paths


def _mask_facts(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as inputs:
        if "period" not in inputs.files:
            raise PublicCheckError("Sealed challenge input lacks its period mask")
        period = np.asarray(inputs["period"])
    if period.shape != (EXPECTED_FRAMES,):
        raise PublicCheckError(f"Expected {EXPECTED_FRAMES} input periods, got {period.shape}")
    out_of_play = period <= 0
    return {
        "period_sha256": array_sha256(period),
        "out_of_play_mask_sha256": array_sha256(out_of_play.astype(np.uint8)),
        "out_of_play_frames": int(out_of_play.sum()),
        "rule": "For both fixed CSVs, exactly the input frames with period <= 0 are X.",
    }


def _model_manifest_path(manifest: dict[str, Any], filename: str) -> Path:
    directory = Path(manifest["model_directory"])
    if directory.is_absolute():
        return directory / filename
    if directory.parts and directory.parts[0] == "data":
        return DATA.joinpath(*directory.parts[1:], filename)
    return ROOT / directory / filename


def _verify_csv(path: Path, expected_mask: np.ndarray) -> None:
    from evaluation import possession_metrics as pm

    if not path.is_file():
        raise PublicCheckError(f"Missing fixed public-check CSV: {path}")
    frame = pm.read_label_csv(path)
    error = pm.validate_submission(frame, EXPECTED_FRAMES)
    if error:
        raise PublicCheckError(f"Invalid frozen CSV {path.name}: {error}")
    labels = np.asarray([str(value).strip().upper() for value in frame["label"]])
    if not np.array_equal(labels == "X", expected_mask):
        raise PublicCheckError(f"Out-of-play X mask differs from the frozen tracking input: {path.name}")


def _verify_prediction_artifacts(manifest: dict[str, Any], promotion: dict[str, Any]) -> dict[str, Any]:
    receipt = _json(PREDICTION_RECEIPT)
    declaration = _json(PREDICTION_DECLARATION)
    if receipt.get("declaration_sha256") != sha256(PREDICTION_DECLARATION):
        raise PublicCheckError("Candidate validation receipt does not bind its declaration")
    expected_models = manifest.get("models_sha256", {})
    if (receipt.get("model_sha256") != expected_models
            or declaration.get("model_sha256") != expected_models):
        raise PublicCheckError("Candidate prediction did not use the active combined package")
    if (declaration.get("current_manifest_sha256") != promotion.get("previous_manifest_sha256")
            or declaration.get("input_sha256") != EXPECTED_INPUT_SHA256
            or receipt.get("input_sha256") != EXPECTED_INPUT_SHA256
            or sha256(INPUTS) != EXPECTED_INPUT_SHA256):
        raise PublicCheckError("Candidate input or prior-baseline provenance differs from the promotion record")
    if (receipt.get("output_sha256") != sha256(COMBINED_CSV)
            or receipt.get("rows") != EXPECTED_FRAMES
            or receipt.get("format_validation", {}).get("valid") is not True
            or receipt.get("format_validation", {}).get("rows") != EXPECTED_FRAMES):
        raise PublicCheckError("Candidate CSV does not match its label-free validation receipt")
    if (receipt.get("organizer_reference_labels_read") is not False
            or receipt.get("organizer_scored") is not False
            or receipt.get("organizer_score") is not None
            or declaration.get("organizer_reference_labels_read") is not False
            or declaration.get("organizer_scored") is not False):
        raise PublicCheckError("Candidate generation must remain label-free")
    if (receipt.get("current_baseline_replay_sha256") != EXPECTED_PRECEDING_CSV_SHA256
            or receipt.get("current_baseline_replay_path") != _display_path(PRECEDING_CSV)
            or sha256(PRECEDING_CSV) != EXPECTED_PRECEDING_CSV_SHA256):
        raise PublicCheckError("Preceding-original-ensemble replay differs from its byte seal")
    if not isinstance(receipt.get("blended_probability_sha256"), str):
        raise PublicCheckError("Candidate validation receipt lacks its blended probability digest")
    member_probabilities = receipt.get("member_probability_sha256", {})
    if set(member_probabilities) != {"joint_gru_2026", "joint_gru_2027"}:
        raise PublicCheckError("Candidate validation receipt must bind both seed probabilities")

    source_map = declaration.get("sources_sha256", {})
    if not PREDICTION_REQUIRED_SOURCES.issubset(source_map):
        raise PublicCheckError("Candidate prediction declaration omits required inference sources")
    for relative, expected in source_map.items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != expected:
            raise PublicCheckError(f"Candidate prediction source changed: {relative}")
    return {
        "receipt_sha256": sha256(PREDICTION_RECEIPT),
        "declaration_sha256": sha256(PREDICTION_DECLARATION),
        "receipt": receipt,
        "declaration": declaration,
    }


def _build_declaration() -> dict[str, Any]:
    manifest = _json(BASELINE_MANIFEST)
    policy = _json(DECISION_POLICY)
    promotion = _json(PROMOTION_RECEIPT)
    prior_public = _json(HISTORICAL_PUBLIC_REPORT)
    comparison = _json(COMPARISON_REPORT)
    comparison_verification = _json(COMPARISON_VERIFICATION)
    comparison_evidence = promotion.get("evidence", {}).get("comparison", {})

    if (manifest.get("model") != "joint_gru_combined_ensemble"
            or manifest.get("package_version") != "joint_gru_combined_ensemble_v1"):
        raise PublicCheckError("The combined-objective package must be promoted before this public check")
    baseline_sha = sha256(BASELINE_MANIFEST)
    policy_sha = sha256(DECISION_POLICY)
    if (policy.get("working_candidate", {}).get("model") != manifest["model"]
            or policy["working_candidate"].get("baseline_manifest_sha256") != baseline_sha):
        raise PublicCheckError("Active decision policy does not bind the promoted manifest")
    if (promotion.get("baseline_manifest_sha256") != baseline_sha
            or promotion.get("models_sha256") != manifest.get("models_sha256")
            or promotion.get("model_decision_policy_sha256") != policy_sha
            or promotion.get("organizer_scored") is not False
            or promotion.get("organizer_labels_read") is not False):
        raise PublicCheckError("Promotion receipt does not bind the active model and policy")
    public_sidecar = manifest.get("organizer_public_diagnostic", {})
    if (public_sidecar.get("report") != "experiments/results/combined_objective_public_check.json"
            or public_sidecar.get("known_at_promotion") is not False
            or public_sidecar.get("used_for_training_tuning_selection_or_adoption") is not False):
        raise PublicCheckError("The promoted manifest must mark this public check as future and non-selective")
    if not isinstance(prior_public.get("organizer_gt_sha256"), str) or len(prior_public["organizer_gt_sha256"]) != 64:
        raise PublicCheckError("Historical public-check receipt lacks the organizer reference SHA")
    if comparison.get("study_id") != "objective_ensemble_v1" or comparison_verification.get("study_id") != "objective_ensemble_v1":
        raise PublicCheckError("Expected the verified objective-ensemble development comparison")
    if (comparison_verification.get("inputs", {}).get("report_sha256") != sha256(COMPARISON_REPORT)
            or comparison_verification.get("inputs", {}).get("runs_sha256") != sha256(COMPARISON_RUNS)
            or comparison_evidence.get("report_sha256") != sha256(COMPARISON_REPORT)
            or comparison_evidence.get("independent_verification_sha256") != sha256(COMPARISON_VERIFICATION)):
        raise PublicCheckError("Promotion and independent audit do not bind the saved development comparison")

    prediction = _verify_prediction_artifacts(manifest, promotion)
    if (COMBINED_CSV.name != "joint_gru_combined_ensemble.csv"
            or PRECEDING_CSV.name != "preceding_seed_ensemble_replay.csv"):
        raise PublicCheckError("Only the two fixed ensemble CSVs may enter this check")
    mask = _mask_facts(INPUTS)
    with np.load(INPUTS, allow_pickle=False) as inputs:
        expected_mask = np.asarray(inputs["period"]) <= 0
    _verify_csv(COMBINED_CSV, expected_mask)
    _verify_csv(PRECEDING_CSV, expected_mask)

    artifact_paths = _artifact_paths(manifest)
    artifacts = {}
    for role, path in artifact_paths.items():
        if not path.is_file():
            raise PublicCheckError(f"Missing pre-score artifact {role}: {path}")
        artifacts[role] = {"path": _display_path(path), "sha256": sha256(path),
                           "bytes": path.stat().st_size}
    models = {
        filename: {"path": _display_path(_model_manifest_path(manifest, filename)),
                   "sha256": sha256(_model_manifest_path(manifest, filename))}
        for filename in sorted(manifest["models_sha256"])
    }
    for filename, info in models.items():
        if info["sha256"] != manifest["models_sha256"][filename]:
            raise PublicCheckError(f"Promoted model file differs from manifest: {filename}")

    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "user_request": "looks good. don’t forget to update pages + also compute public score",
        "purpose": "One fixed descriptive public-window score of the promoted combined ensemble and exact preceding-ensemble replay.",
        "candidate_set": [
            {"name": "combined_objective_ensemble", "role": "promoted fixed model", "path": _display_path(COMBINED_CSV)},
            {"name": "preceding_seed_ensemble_replay", "role": "matched prior fixed model", "path": _display_path(PRECEDING_CSV)},
        ],
        "primary_metric": "Official spell_f1_mean; macro_f1 is the exact-tie secondary; 70/30 is auxiliary only.",
        "scope": "One public five-minute diagnostic; 199,319 rows, 7,500 graded frames and 17 possession spells. It is not a full-match or independent-match estimate.",
        "bootstrap": {
            "scorer": "Unchanged evaluation.possession_metrics.evaluate",
            "draws": BOOTSTRAP_DRAWS,
            "reported_interval": "Official separate 5th–95th percentile block-bootstrap intervals for spell F1 over spells and macro F1 over GT segments.",
            "paired_difference_interval": None,
            "interpretation": "Intervals are conditional on this single public match; separate intervals do not establish paired significance.",
        },
        "reference": {
            "path": _display_path(REFERENCE_CSV),
            "organizer_gt_sha256": prior_public["organizer_gt_sha256"],
            "sha_source": _display_path(HISTORICAL_PUBLIC_REPORT),
            "historical_report_sha256": sha256(HISTORICAL_PUBLIC_REPORT),
            "opened_before_exclusive_gate": False,
        },
        "active_package": {
            "model": manifest["model"],
            "package_version": manifest["package_version"],
            "manifest_sha256": baseline_sha,
            "decision_policy_sha256": policy_sha,
            "promotion_receipt_sha256": sha256(PROMOTION_RECEIPT),
            "models": models,
        },
        "prediction_validation": {
            "receipt_sha256": prediction["receipt_sha256"],
            "declaration_sha256": prediction["declaration_sha256"],
            "combined_csv_sha256": sha256(COMBINED_CSV),
            "preceding_replay_csv_sha256": sha256(PRECEDING_CSV),
            "input_sha256": sha256(INPUTS),
            "input_mask": mask,
        },
        "artifacts": artifacts,
        "source_sha256": _source_hashes(),
        "prior_gates": "All earlier public-check gates and historical reports remain unchanged; this request uses a new one-time exclusive gate.",
        "guardrails": {
            "model_selection_after_public_score": False,
            "baseline_or_policy_change_after_public_score": False,
            "organizer_public_diagnostic_used_for_adoption": False,
            "full_match_claim": False,
        },
        "organizer_reference_labels_read_before_gate": False,
        "organizer_scored_before_gate": False,
    }


def _verify_frozen(spec: dict[str, Any]) -> None:
    if spec.get("study_id") != STUDY_ID:
        raise PublicCheckError("Unexpected public-check declaration")
    if spec.get("candidate_set") != [
        {"name": "combined_objective_ensemble", "role": "promoted fixed model", "path": _display_path(COMBINED_CSV)},
        {"name": "preceding_seed_ensemble_replay", "role": "matched prior fixed model", "path": _display_path(PRECEDING_CSV)},
    ]:
        raise PublicCheckError("The frozen public candidate set changed")
    if spec.get("source_sha256") != _source_hashes():
        raise PublicCheckError("Public-check source changed after freeze")
    for role, fact in spec.get("artifacts", {}).items():
        path = _artifact_paths(_json(BASELINE_MANIFEST)).get(role)
        if path is None or not path.is_file() or sha256(path) != fact.get("sha256") or path.stat().st_size != fact.get("bytes"):
            raise PublicCheckError(f"Frozen public-check artifact changed: {role}")

    manifest = _json(BASELINE_MANIFEST)
    policy = _json(DECISION_POLICY)
    promotion = _json(PROMOTION_RECEIPT)
    if (sha256(BASELINE_MANIFEST) != spec["active_package"]["manifest_sha256"]
            or sha256(DECISION_POLICY) != spec["active_package"]["decision_policy_sha256"]
            or promotion.get("baseline_manifest_sha256") != spec["active_package"]["manifest_sha256"]
            or policy.get("working_candidate", {}).get("model") != manifest.get("model")):
        raise PublicCheckError("Active promoted package or policy changed after freeze")
    if sha256(HISTORICAL_PUBLIC_REPORT) != spec["reference"]["historical_report_sha256"]:
        raise PublicCheckError("Historical public-check receipt changed after freeze")
    history = _json(HISTORICAL_PUBLIC_REPORT)
    if history.get("organizer_gt_sha256") != spec["reference"]["organizer_gt_sha256"]:
        raise PublicCheckError("Historical organizer reference SHA changed after freeze")

    prediction = _verify_prediction_artifacts(manifest, promotion)
    if (prediction["receipt_sha256"] != spec["prediction_validation"]["receipt_sha256"]
            or prediction["declaration_sha256"] != spec["prediction_validation"]["declaration_sha256"]):
        raise PublicCheckError("Candidate validation receipt changed after freeze")
    with np.load(INPUTS, allow_pickle=False) as inputs:
        expected_mask = np.asarray(inputs["period"]) <= 0
    mask = _mask_facts(INPUTS)
    if mask != spec["prediction_validation"]["input_mask"]:
        raise PublicCheckError("Input period or out-of-play mask changed after freeze")
    _verify_csv(COMBINED_CSV, expected_mask)
    _verify_csv(PRECEDING_CSV, expected_mask)
    if (sha256(COMBINED_CSV) != spec["prediction_validation"]["combined_csv_sha256"]
            or sha256(PRECEDING_CSV) != spec["prediction_validation"]["preceding_replay_csv_sha256"]
            or sha256(INPUTS) != spec["prediction_validation"]["input_sha256"]):
        raise PublicCheckError("Frozen candidate CSV or tracking input changed")


def freeze() -> dict[str, Any]:
    """Freeze the exact two CSVs and current v5 package without touching GT."""
    if STARTED.exists() or PRIVATE_REPORT.exists() or PUBLIC_REPORT.exists():
        raise PublicCheckError("Public-check gate is closed; do not freeze or score again")
    if PREDECLARED.exists():
        spec = _json(PREDECLARED)
        _verify_frozen(spec)
        return spec
    spec = _build_declaration()
    _write_exclusive(PREDECLARED, spec)
    return spec


def _score_summary(score: dict[str, Any]) -> dict[str, Any]:
    spell_iou = {}
    for threshold in IOU_THRESHOLDS:
        detail = score["spell_f1"][threshold]
        spell_iou[f"{threshold:.1f}"] = {
            key: detail[key] for key in ("precision", "recall", "f1", "n_matched", "n_spurious", "n_missed")
        }
    dead = score["per_class"]["D"]
    spell = float(score["spell_f1_mean"])
    macro = float(score["macro_f1"])
    return {
        "primary_metric": "spell_f1_mean",
        "spell_f1_mean": spell,
        "macro_f1": macro,
        "auxiliary_70_30": 0.7 * spell + 0.3 * macro,
        "d_precision": float(dead["precision"]),
        "d_recall": float(dead["recall"]),
        "d_f1": float(dead["f1"]),
        "spell_by_iou": spell_iou,
        "headline_iou": float(score["spell_iou_headline"]),
        "spell_f1_at_headline": float(score["spell_f1_at_headline"]),
        "spell_f1_bootstrap_5_95": [score["spell_bootstrap_low"], score["spell_bootstrap_high"]],
        "macro_f1_bootstrap_5_95": [score["bootstrap_low"], score["bootstrap_high"]],
        "bootstrap_draws": int(score["bootstrap_n"]),
        "frames": int(score["n_frames"]),
        "graded_frames": int(score["n_graded"]),
        "ground_truth_spells": int(score["n_gt_spells"]),
        "predicted_spells": int(score["n_pred_spells"]),
    }


def _score_fixed_csvs(reference_path: Path) -> dict[str, Any]:
    """Read reference and invoke the official evaluator exactly once per fixed CSV."""
    from evaluation import possession_metrics as pm

    reference = pm.read_label_csv(reference_path)
    error = pm.validate_submission(reference, EXPECTED_FRAMES)
    if error:
        raise PublicCheckError(f"Public reference format differs: {error}")
    truth = pm.label_codes(reference)
    scores = {}
    for role, path in (("combined_objective_ensemble", COMBINED_CSV),
                       ("preceding_seed_ensemble_replay", PRECEDING_CSV)):
        predicted = pm.read_label_csv(path)
        error = pm.validate_submission(predicted, EXPECTED_FRAMES)
        if error:
            raise PublicCheckError(f"Frozen {role} CSV is invalid: {error}")
        score = pm.evaluate(truth, pm.label_codes(predicted), bootstrap_n=BOOTSTRAP_DRAWS)
        if (score["n_frames"] != EXPECTED_FRAMES
                or score["n_graded"] != EXPECTED_GRADED
                or score["n_gt_spells"] != EXPECTED_GT_SPELLS
                or score["bootstrap_n"] != BOOTSTRAP_DRAWS):
            raise PublicCheckError("Official public reference coverage differs from the frozen five-minute window")
        scores[role] = _score_summary(score)
    return scores


def _deltas(scores: dict[str, Any]) -> dict[str, Any]:
    candidate = scores["combined_objective_ensemble"]
    preceding = scores["preceding_seed_ensemble_replay"]
    fields = ("spell_f1_mean", "macro_f1", "auxiliary_70_30", "d_precision", "d_recall", "d_f1")
    result = {field: candidate[field] - preceding[field] for field in fields}
    result["spell_by_iou_f1"] = {
        threshold: candidate["spell_by_iou"][threshold]["f1"]
        - preceding["spell_by_iou"][threshold]["f1"]
        for threshold in (f"{value:.1f}" for value in IOU_THRESHOLDS)
    }
    return result


def check() -> dict[str, Any]:
    """Claim the new gate, then score the two sealed CSVs once."""
    if STARTED.exists() or PRIVATE_REPORT.exists() or PUBLIC_REPORT.exists():
        raise PublicCheckError("Public-check gate is closed; saved report is final")
    if not PREDECLARED.is_file():
        raise PublicCheckError("Freeze the two public CSVs before scoring")
    spec = _json(PREDECLARED)
    _verify_frozen(spec)
    marker = {
        "study_id": STUDY_ID,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": sha256(PREDECLARED),
        "candidate_csv_sha256": spec["prediction_validation"]["combined_csv_sha256"],
        "preceding_replay_csv_sha256": spec["prediction_validation"]["preceding_replay_csv_sha256"],
        "reference_sha256_from_historical_receipt": spec["reference"]["organizer_gt_sha256"],
        "reference_opened_before_marker": False,
    }
    _write_exclusive(STARTED, marker)
    # A second no-reference verification after the exclusive marker closes a
    # hash-check/read race. Any failure now leaves the gate permanently closed.
    _verify_frozen(spec)
    if sha256(REFERENCE_CSV) != spec["reference"]["organizer_gt_sha256"]:
        raise PublicCheckError("Public reference file does not match the previously recorded SHA")
    scores = _score_fixed_csvs(REFERENCE_CSV)
    # Keep the result tied to the same frozen inputs and active baseline even
    # if a concurrent process changed a file while scoring was in progress.
    _verify_frozen(spec)
    if sha256(REFERENCE_CSV) != spec["reference"]["organizer_gt_sha256"]:
        raise PublicCheckError("Public reference changed during the one-time scoring pass")
    result = {
        "study_id": STUDY_ID,
        "reported_utc": datetime.now(timezone.utc).isoformat(),
        "scope": spec["scope"],
        "primary_metric": spec["primary_metric"],
        "frozen": spec,
        "organizer_reference_labels_read": True,
        "organizer_scored": True,
        "organizer_gt_sha256": spec["reference"]["organizer_gt_sha256"],
        "score_source": "Official evaluation.possession_metrics.evaluate; exactly two fixed CSVs, 1,000 bootstrap draws each.",
        "scores": scores,
        "candidate_minus_preceding_point_deltas": _deltas(scores),
        "interpretation": {
            "public_check_only": True,
            "full_match_score": None,
            "independent_confirmation": False,
            "paired_significance_test": False,
            "no_model_selection_or_tuning_after_public_score": True,
            "baseline_or_policy_changed_after_public_score": False,
            "existing_closed_gates_changed": False,
            "interval_note": "Separate official 5th–95th percentile intervals are conditional on this single public match; they are not a paired interval for the score difference.",
            "reporting_order": "Spell F1 is primary; macro F1 is the exact-tie secondary; 70/30 is auxiliary and not a ranking metric.",
        },
    }
    _write_exclusive(PRIVATE_REPORT, result)
    _write_exclusive(PUBLIC_REPORT, result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    result = freeze() if args.action == "freeze" else check()
    print(json.dumps(result if args.action == "freeze" else result["scores"],
                     sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
