"""Version and activate frozen f062 using saved evidence and exact CSV replay."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from experiments import calibrated_dense_baseline as inference
from experiments import feed_spell_push_run as study
from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "joint_gru_sn_dense_calibrated_ensemble"
PACKAGE = MODEL + "_v1"
PREDECESSOR = "joint_gru_sn_dense_ensemble_v1"
DIRECTORY = ROOT / "data/models" / PACKAGE
STAGED = DIRECTORY / "baseline_manifest.json"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
PREVIOUS = ROOT / "experiments/baselines" / (PREDECESSOR + ".json")
PREVIOUS_POLICY = ROOT / "experiments/baselines" / ("model_decision_policy_" + PREDECESSOR + ".json")
PREVIOUS_SOURCE = DIRECTORY / "previous_source/current_baseline.py"
RECEIPT = ROOT / "experiments/results/baseline_calibrated_dense_promotion.json"
CSV = ROOT / "data/experiments/current_baseline" / (MODEL + ".csv")
INPUTS = ROOT / "data/challenge/inputs.npz"
SEAL = ROOT / "data/experiments/feed_spell_public_v1/seal.json"
PUBLIC_RESULT = ROOT / "experiments/results/feed_spell_public_v1.json"
USER_REQUEST = "i think we should promote this?"
SOURCES = tuple(dict.fromkeys(("experiments/current_baseline.py",
    "experiments/calibrated_dense_baseline.py", "experiments/promote_calibrated_dense_baseline.py",
    "experiments/sn_dense_baseline.py", "experiments/sn_dense_train.py",
    "experiments/soccernet_capacity_model.py", "experiments/factorized_had.py",
    "experiments/learned_temporal.py", "experiments/live_gap_experiment.py") + study.SOURCES))


def read(path):
    return json.loads(Path(path).read_text())


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def preserve(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace preserved artifact: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(content)


def frozen_evidence():
    """Verify original seals against their preserved predecessor files.

    Promotion changes active metadata and the shared baseline dispatch. The
    historical seal itself is never rewritten to accept the promoted files.
    """
    seal = read(SEAL)
    aliases = {"experiments/baseline_manifest.json": PREVIOUS,
               "experiments/model_decision_policy.json": PREVIOUS_POLICY,
               "experiments/current_baseline.py": PREVIOUS_SOURCE}
    for name, digest in (seal["source_sha256"] | seal["dependency_sha256"]).items():
        if sha256(aliases.get(name, ROOT / name)) != digest:
            raise ValueError("Historical sealed evidence differs: " + name)
    for name, digest in seal["models_sha256"].items():
        if sha256(ROOT.parent / name) != digest:
            raise ValueError("Original deployment checkpoint changed")
    for item in seal["predictions"].values():
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("Original sealed prediction changed")
    public = read(PUBLIC_RESULT)
    if public["prediction_seal_sha256"] != sha256(SEAL):
        raise ValueError("Public result does not bind the original prediction seal")
    aggregate = read(study.AGGREGATE)
    config = aggregate["selected_config"]
    if config != inference.CONFIG or aggregate["eligible_ranked"][0] != config["id"]:
        raise ValueError("Promotion must use the externally selected f062")
    candidate = aggregate["finalists"]["f062"]["ensemble"]
    previous = aggregate["baseline"]
    if (not study.eligible(candidate, previous)
            or candidate["clean"]["equal_provider_mean"]["spell_f1"] <
               previous["clean"]["equal_provider_mean"]["spell_f1"] + .01):
        raise ValueError("The saved candidate fails its promotion guards or clean screen")
    return seal, public, aggregate


def stage():
    if STAGED.exists():
        return read(STAGED)
    previous = read(ACTIVE)
    if previous["package_version"] != PREDECESSOR:
        raise ValueError("Expected the uncalibrated dense predecessor")
    preserve(PREVIOUS, ACTIVE.read_bytes())
    preserve(PREVIOUS_POLICY, POLICY.read_bytes())
    if not PREVIOUS_SOURCE.is_file():
        raise ValueError("Preserve the predecessor baseline dispatch before editing it")
    seal, public, aggregate = frozen_evidence()
    original = ROOT / previous["model_directory"]
    for name, digest in previous["models_sha256"].items():
        source = original / name
        if sha256(source) != digest:
            raise ValueError("Predecessor model hash mismatch: " + name)
        preserve(DIRECTORY / name, source.read_bytes())
    architecture = dict(previous["architecture"], kind=inference.KIND,
        postprocessing=inference.POSTPROCESSING,
        combination="Normalize each member, average native H/A/D probabilities, soften D log odds by temperature 4, preserve conditional H/A, decode with cost 3")
    summary = aggregate["finalists"]["f062"]["ensemble"]
    baseline = aggregate["baseline"]
    development = {
        "two_seed_ensemble_clean_spell": summary["clean"]["equal_provider_mean"]["spell_f1"],
        "two_seed_ensemble_feed_like_spell": summary["feed_like"]["equal_provider_mean"]["spell_f1"],
        "previous_adopted_clean_spell": baseline["clean"]["equal_provider_mean"]["spell_f1"],
        "previous_adopted_feed_like_spell": baseline["feed_like"]["equal_provider_mean"]["spell_f1"],
        "provider_and_feed_like_guards_pass": True,
        "clean_improvement_screen_plus_01_passes": True,
        "previously_exposed_development_matches": True,
        "synthetic_feed_like_stress": True,
    }
    diagnostic = public["candidate_scores"]["final_external_winner"]
    manifest = dict(version=8, model=MODEL, package_version=PACKAGE,
        status="User-authorized working development baseline; independent confirmation pending",
        promoted_utc=datetime.now(timezone.utc).isoformat(), replaced_baseline=previous["model"],
        previous_manifest=str(PREVIOUS.relative_to(ROOT)), previous_manifest_sha256=sha256(PREVIOUS),
        previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)), previous_policy_sha256=sha256(PREVIOUS_POLICY),
        model_directory=str(DIRECTORY.relative_to(ROOT)), models_sha256=previous["models_sha256"],
        default_output=str(CSV.relative_to(ROOT)), architecture=architecture, training=previous["training"],
        evidence=dict(declaration=str(study.DECLARATION.relative_to(ROOT)), declaration_sha256=sha256(study.DECLARATION),
            aggregate=str(study.AGGREGATE.relative_to(ROOT)), aggregate_sha256=sha256(study.AGGREGATE),
            frozen_config=inference.CONFIG, development_full_provider_holdout=development,
            provider_and_seed_summaries=aggregate["finalists"]["f062"],
            prediction_seal_sha256=sha256(SEAL),
            prediction_csv_sha256=seal["predictions"]["final_external_winner"]["sha256"]),
        promotion=dict(type="explicit_user_authorized_decoder_promotion", user_request=USER_REQUEST,
            original_screen_passes=True, automatic_promotion=False, fitting_during_promotion=False,
            organizer_rescoring_performed=False, public_diagnostic_known_before_authorization=True,
            model_checkpoints_unchanged=True),
        organizer_public_diagnostic=dict(report=str(PUBLIC_RESULT.relative_to(ROOT)), known_at_promotion=True,
            diagnostic_only=True, spell_f1=diagnostic["spell_f1_mean"], macro_f1=diagnostic["macro_f1"],
            d_f1=diagnostic["d_f1"], previous_spell_f1=public["adopted_baseline_reused"]["spell_f1_mean"],
            used_for_model_or_hyperparameter_selection=False, independent_confirmation=False),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        historical_source=dict(repository_commit_before_promotion=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            predecessor_dispatch_sha256=sha256(PREVIOUS_SOURCE),
            note="Original sealed studies require predecessor metadata and source; their receipts and hashes are unchanged"),
        organizer_reference_labels_used_for_training=False, historical_selection_records_preserved=True)
    preserve(STAGED, encode(manifest))
    return manifest


def updated_policy(previous, candidate):
    policy = json.loads(json.dumps(previous))
    policy.update(version=8, confirmed_utc_date=candidate["promoted_utc"][:10], primary_evaluation_view="feed_like")
    policy["working_candidate"] = dict(family="Dense-SoccerNet two-seed ensemble with frozen D-logit temperature 4",
        model=MODEL, status=candidate["status"], baseline_manifest="experiments/baseline_manifest.json",
        baseline_manifest_sha256=sha256(STAGED), prediction_entrypoint="python -m experiments.current_baseline predict",
        reason="Explicit user promotion of the externally selected feed-like leader; both seeds and all providers improve spell in both views; all guards and +.01 clean screen pass",
        promotion_record=str(RECEIPT.relative_to(ROOT)), previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)),
        public_diagnostic_known_before_user_request=True)
    policy["stress_guardrail"]["reference_recipe"] = MODEL
    policy["clean_improvement_screen"] = dict(default_delta=.01, historical_failures_preserved=True,
        current_screen_passes=True, current_adoption_exception=None)
    return policy


def promote():
    if RECEIPT.exists():
        receipt = read(RECEIPT)
        if sha256(ACTIVE) != receipt["baseline_manifest_sha256"] or sha256(POLICY) != receipt["policy_sha256"]:
            raise ValueError("Completed promotion metadata changed")
        return receipt
    candidate = stage()
    seal, _, _ = frozen_evidence()
    for name, digest in candidate["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Staged source changed: " + name)
    if not CSV.exists():
        subprocess.run([sys.executable, "-m", "experiments.current_baseline", "predict",
                        "--manifest", str(STAGED)], cwd=ROOT, check=True)
    validation = read(CSV.with_suffix(".json"))
    expected = seal["predictions"]["final_external_winner"]["sha256"]
    if (sha256(CSV) != expected or validation.get("output_sha256") != expected
            or validation.get("baseline_manifest_sha256") != sha256(STAGED)
            or validation.get("target_labels_read") is not False or validation.get("organizer_scored") is not False):
        raise ValueError("New baseline CLI must exactly reproduce the sealed f062 CSV")
    subprocess.run([sys.executable, "evaluation/validate_submission.py", str(CSV)], cwd=ROOT, check=True)
    policy = updated_policy(read(PREVIOUS_POLICY), candidate)
    if ACTIVE.read_bytes() not in (PREVIOUS.read_bytes(), encode(candidate)):
        raise ValueError("Active predecessor changed during promotion")
    if POLICY.read_bytes() not in (PREVIOUS_POLICY.read_bytes(), encode(policy)):
        raise ValueError("Active predecessor policy changed during promotion")
    preserve(ROOT / "experiments/baselines" / (PACKAGE + ".json"), encode(candidate))
    ACTIVE.write_bytes(encode(candidate))
    POLICY.write_bytes(encode(policy))
    receipt = dict(package_version=PACKAGE, promoted_utc=candidate["promoted_utc"],
        baseline_manifest_sha256=sha256(ACTIVE), policy_sha256=sha256(POLICY),
        previous_manifest_sha256=sha256(PREVIOUS), previous_policy_sha256=sha256(PREVIOUS_POLICY),
        authorization=candidate["promotion"], evidence=candidate["evidence"],
        models_sha256=candidate["models_sha256"], model_checkpoints_unchanged=True,
        csv_equals_sealed_candidate=True, candidate_csv_sha256=sha256(CSV), rows=199319,
        validation_receipt_sha256=sha256(CSV.with_suffix(".json")),
        fitting_performed=False, organizer_reference_accessed=False, organizer_rescored=False,
        historical_score_gates_and_declarations_preserved=True)
    preserve(RECEIPT, encode(receipt))
    preserve(DIRECTORY / "promotion_receipt.json", encode(receipt))
    preserve(DIRECTORY / "model_decision_policy.json", encode(policy))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "promote"))
    args = parser.parse_args()
    value = stage() if args.action == "stage" else promote()
    print(args.action.upper(), value["package_version"])
