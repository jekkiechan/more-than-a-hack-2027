"""Adopt already-sealed original_full artifacts; never fit or access references."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from experiments import original_full_baseline as inference, sn_dense_baseline
from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "joint_gru_sn_dense_original_diversity_ensemble"
PACKAGE = MODEL + "_v1"
PREDECESSOR = "joint_gru_sn_dense_calibrated_ensemble_v1"
DIRECTORY = ROOT / "data/models" / PACKAGE
STAGED = DIRECTORY / "baseline_manifest.json"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
PREVIOUS = ROOT / "experiments/baselines" / (PREDECESSOR + ".json")
PREVIOUS_POLICY = ROOT / "experiments/baselines" / ("model_decision_policy_" + PREDECESSOR + ".json")
PREVIOUS_SOURCE = DIRECTORY / "previous_source/current_baseline.py"
RECEIPT = ROOT / "experiments/results/baseline_original_full_promotion.json"
CSV = ROOT / "data/experiments/current_baseline" / (MODEL + ".csv")
INPUTS = ROOT / "data/challenge/inputs.npz"
ORIGINAL = ROOT / "data/experiments/short_quality_public_v1"
DECLARATION = ROOT / "experiments/results/short_quality_public_v1_declaration.json"
RESULT = ROOT / "experiments/results/short_quality_public_v1_result.json"
AGGREGATE = ROOT / "experiments/results/short_original_diversity_v1_aggregate.json"
USER_REQUEST = "for original_full what is the public score? i do agree that we should promote"
EXCEPTION = ("Explicit user adoption on 2026-11-01 despite the preserved failed +.01 clean screen; "
             "matched clean/feed spell gains and passing provider guards support a working baseline, "
             "with SkillCorner regression and independent-confirmation limits disclosed")


def read(path):
    return json.loads(Path(path).read_text())


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def preserve(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Refusing to replace preserved artifact: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def evidence():
    declaration, seal, public, aggregate = (
        read(DECLARATION), read(ORIGINAL / "prediction_seal.json"), read(RESULT), read(AGGREGATE))
    if (seal["declaration_sha256"] != sha256(DECLARATION)
            or public["prediction_seal_sha256"] != sha256(ORIGINAL / "prediction_seal.json")
            or public["candidate_csv_sha256"] != seal["candidate_sha256"]
            or not (ORIGINAL / "public_check_started.json").is_file()):
        raise ValueError("Expected the existing sealed candidate and permanently closed public gate")
    for name, digest in (seal["models_sha256"] | seal["files_sha256"]).items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Sealed candidate artifact differs: " + name)
    if (sha256(INPUTS) != declaration["input_sha256"]["data/challenge/inputs.npz"]
            or aggregate["winner"] != "original_full"
            or not aggregate["guards_pass"]["original_full"]
            or aggregate["historical_clean_screen_pass"]["original_full"]):
        raise ValueError("Expected fixed original_full winner, passing guards and preserved failed screen")
    aliases = {"experiments/current_baseline.py": PREVIOUS_SOURCE,
               "experiments/baseline_manifest.json": PREVIOUS,
               "experiments/model_decision_policy.json": PREVIOUS_POLICY}
    # Preserve the historical inference/training sources, rather than rewriting
    # the old declaration to accept the new dispatcher or active metadata.
    for name, digest in declaration["source_sha256"].items():
        if sha256(aliases.get(name, ROOT / name)) != digest:
            raise ValueError("Historical source differs: " + name)
    if (sha256(PREVIOUS) != declaration["baseline_manifest_sha256"]
            or sha256(PREVIOUS_POLICY) != declaration["policy_sha256"]):
        raise ValueError("Historical baseline metadata differs")
    verification = ROOT / "experiments/results/short_quality_public_v1_artifact_verification.json"
    checked = read(verification)
    if not checked["passed"] or checked["result_sha256"] != sha256(RESULT):
        raise ValueError("Saved reference-free artifact verification differs")
    return declaration, seal, public, aggregate


def stage():
    if STAGED.exists():
        return read(STAGED)
    previous = read(ACTIVE)
    if previous["package_version"] != PREDECESSOR:
        raise ValueError("Expected the calibrated dense predecessor")
    preserve(PREVIOUS, ACTIVE.read_bytes())
    preserve(PREVIOUS_POLICY, POLICY.read_bytes())
    declaration, seal, public, aggregate = evidence()
    hashes = {}
    loaded = {}
    for seed in (2026, 2027):
        for name, source in ((f"dense_{seed}.joblib", ORIGINAL / "models" / f"pooled_{seed}.joblib"),
                             (f"teacher_{seed}.joblib", ROOT / previous["model_directory"] / f"teacher_{seed}.joblib")):
            digest = sha256(source)
            expected = (seal["models_sha256"][str(source.relative_to(ROOT))] if name.startswith("dense")
                        else previous["models_sha256"][name])
            if digest != expected:
                raise ValueError("Checkpoint identity differs: " + name)
            preserve(DIRECTORY / name, source.read_bytes())
            hashes[name] = digest
            loaded[name.removesuffix(".joblib")] = joblib.load(source)
        receipt = read(ORIGINAL / "models" / f"pooled_{seed}.json")
        if (receipt["declaration_sha256"] != sha256(DECLARATION)
                or receipt["sha256"] != hashes[f"dense_{seed}.joblib"]
                or receipt["teacher_sha256"] != hashes[f"teacher_{seed}.joblib"]
                or receipt["old_matches"] != sorted(x["match"] for x in declaration["train_plan"]["train"])
                or len(receipt["config"]["soccernet_games"]) != 45
                or any(receipt["sampling"]["counts"][s]["draws"] != 32768 for s in ("old", "new"))):
            raise ValueError("Sealed pooled training receipt differs")
    architecture = dict(previous["architecture"], kind=inference.KIND,
        postprocessing=inference.POSTPROCESSING,
        combination="Equal native two-seed probability mean, normalize pooled vector, frozen D-logit temperature 4 and cost-3 decoder",
        phase_tta=False)
    sn_dense_baseline.validate_models(loaded, architecture)
    summary = aggregate["summaries"]["original_full"]["ensemble"]
    # The latest stacking report supplies the exact same adopted control.
    control = read(ROOT / "experiments/results/smart_stacking_v1_aggregate.json")["summaries"]["adopted"]["ensemble"]
    manifest = dict(version=9, model=MODEL, package_version=PACKAGE,
        status="User-authorized working development baseline; independent confirmation pending",
        promoted_utc=datetime.now(timezone.utc).isoformat(), replaced_baseline=previous["model"],
        previous_manifest=str(PREVIOUS.relative_to(ROOT)), previous_manifest_sha256=sha256(PREVIOUS),
        previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)), previous_policy_sha256=sha256(PREVIOUS_POLICY),
        model_directory=str(DIRECTORY.relative_to(ROOT)), models_sha256=hashes,
        default_output=str(CSV.relative_to(ROOT)), architecture=architecture,
        training=dict(recipe="original_full", seeds=[2026, 2027], steps=2048,
            original_train_matches=13, gar_train_games=45, sampler="OriginalDiversitySampler",
            frozen_model_receipts={str((ORIGINAL / "models" / f"pooled_{s}.json").relative_to(ROOT)):
                sha256(ORIGINAL / "models" / f"pooled_{s}.json") for s in (2026, 2027)}),
        evidence=dict(declaration=str(DECLARATION.relative_to(ROOT)), declaration_sha256=sha256(DECLARATION),
            aggregate=str(AGGREGATE.relative_to(ROOT)), aggregate_sha256=sha256(AGGREGATE),
            provider_and_seed_summaries=aggregate["summaries"]["original_full"],
            previous_adopted_summaries=control,
            development_full_provider_holdout=dict(two_seed_ensemble_clean_spell=summary["clean"]["equal_provider_mean"]["spell_f1"],
                two_seed_ensemble_feed_like_spell=summary["feed_like"]["equal_provider_mean"]["spell_f1"],
                provider_and_feed_like_guards_pass=True, clean_improvement_screen_plus_01_passes=False,
                previously_exposed_development_matches=True, synthetic_feed_like_stress=True),
            prediction_seal_sha256=sha256(ORIGINAL / "prediction_seal.json"),
            prediction_csv_sha256=seal["candidate_sha256"]),
        promotion=dict(type="explicit_user_authorized_recipe_promotion", user_request=USER_REQUEST,
            original_screen_passes=False, clean_screen_exception=EXCEPTION,
            automatic_promotion=False, fitting_during_promotion=False, phase_tta_added=False,
            organizer_rescoring_performed=False, public_diagnostic_known_before_authorization=True),
        organizer_public_diagnostic=dict(report=str(RESULT.relative_to(ROOT)), diagnostic_only=True,
            known_at_promotion=True, spell_f1=public["candidate"]["spell_f1_mean"],
            macro_f1=public["candidate"]["macro_f1"], d_f1=public["candidate"]["d_f1"],
            previous_spell_f1=public["adopted_baseline_reused"]["spell_f1_mean"],
            used_for_model_or_hyperparameter_selection=False, independent_confirmation=False),
        source_sha256={name: sha256(ROOT / name) for name in sorted(set(declaration["source_sha256"]) |
            {"experiments/original_full_baseline.py", "experiments/promote_original_full_baseline.py"})},
        historical_source=dict(predecessor_dispatch_sha256=sha256(PREVIOUS_SOURCE),
            predecessor_dispatch_path=str(PREVIOUS_SOURCE.relative_to(ROOT))),
        organizer_reference_labels_used_for_training=False, historical_selection_records_preserved=True)
    preserve(STAGED, encode(manifest))
    return manifest


def updated_policy(previous, candidate):
    policy = json.loads(json.dumps(previous))
    policy.update(version=9, confirmed_utc_date=candidate["promoted_utc"][:10])
    policy["working_candidate"] = dict(family="Two-seed original-possession diversity with frozen temperature-4 decoding",
        model=MODEL, status=candidate["status"], baseline_manifest="experiments/baseline_manifest.json",
        baseline_manifest_sha256=sha256(STAGED), prediction_entrypoint="python -m experiments.current_baseline predict",
        reason=EXCEPTION, promotion_record=str(RECEIPT.relative_to(ROOT)),
        previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)), public_diagnostic_known_before_user_request=True)
    policy["stress_guardrail"]["reference_recipe"] = MODEL
    policy["clean_improvement_screen"] = dict(default_delta=.01, historical_failures_preserved=True,
        current_screen_passes=False, current_adoption_exception=EXCEPTION)
    return policy


def promote():
    if RECEIPT.exists():
        receipt = read(RECEIPT)
        if sha256(ACTIVE) != receipt["baseline_manifest_sha256"] or sha256(POLICY) != receipt["policy_sha256"]:
            raise ValueError("Completed promotion metadata changed")
        return receipt
    candidate = stage()
    _, seal, _, _ = evidence()
    for name, digest in candidate["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Staged source changed: " + name)
    if not CSV.exists():
        subprocess.run([sys.executable, "-m", "experiments.current_baseline", "predict", "--manifest", str(STAGED)],
                       cwd=ROOT, check=True)
    validation = read(CSV.with_suffix(".json"))
    if (sha256(CSV) != seal["candidate_sha256"] or validation.get("output_sha256") != seal["candidate_sha256"]
            or validation.get("baseline_manifest_sha256") != sha256(STAGED)
            or validation.get("input_sha256") != sha256(INPUTS)
            or validation.get("target_labels_read") is not False or validation.get("organizer_scored") is not False):
        raise ValueError("Baseline CLI must exactly reproduce the already-sealed original_full CSV")
    subprocess.run([sys.executable, "evaluation/validate_submission.py", str(CSV)], cwd=ROOT, check=True)
    policy = updated_policy(read(PREVIOUS_POLICY), candidate)
    if ACTIVE.read_bytes() != PREVIOUS.read_bytes() or POLICY.read_bytes() != PREVIOUS_POLICY.read_bytes():
        raise ValueError("Active predecessor changed during promotion")
    preserve(ROOT / "experiments/baselines" / (PACKAGE + ".json"), encode(candidate))
    ACTIVE.write_bytes(encode(candidate))
    POLICY.write_bytes(encode(policy))
    receipt = dict(package_version=PACKAGE, promoted_utc=candidate["promoted_utc"],
        baseline_manifest_sha256=sha256(ACTIVE), policy_sha256=sha256(POLICY),
        previous_manifest_sha256=sha256(PREVIOUS), previous_policy_sha256=sha256(PREVIOUS_POLICY),
        authorization=candidate["promotion"], models_sha256=candidate["models_sha256"],
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
