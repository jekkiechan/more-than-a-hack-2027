"""Stage and explicitly adopt the already frozen spell-frontier candidate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

from experiments import spell_frontier_baseline as inference
from experiments.eval_set.protocol import sha256
from experiments.promote_original_full_baseline import read, encode, preserve

ROOT = Path(__file__).resolve().parents[1]
MODEL = "joint_gru_sn_dense_spell_frontier_ensemble"
PACKAGE = MODEL + "_v1"
PREDECESSOR = "joint_gru_sn_dense_original_diversity_phase_ensemble_v1"
DIRECTORY = ROOT / "data/models" / PACKAGE
STAGED = DIRECTORY / "baseline_manifest.json"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
PREVIOUS = ROOT / "experiments/baselines" / (PREDECESSOR + ".json")
PREVIOUS_POLICY = ROOT / "experiments/baselines" / ("model_decision_policy_" + PREDECESSOR + ".json")
SELECTION = ROOT / "experiments/results/v2_spell_frontier_v1_selection.json"
SEAL = ROOT / "data/experiments/v2_spell_frontier_v1/public_prediction_seal.json"
GATE = SEAL.with_name("public_gate_claimed.json")
PUBLIC = ROOT / "experiments/results/v2_spell_frontier_v1_public_result.json"
RECEIPT = ROOT / "experiments/results/baseline_spell_frontier_promotion.json"
CSV = ROOT / "data/experiments/current_baseline" / (MODEL + ".csv")
EXCEPTION = ("Explicit user promotion of the frozen 0.593619 v2 spell blend. "
    "The >0.60 target and historical +.01 clean screen remain unmet. "
    "All three spell means improve over the predecessor; slight v1 macro regression is accepted. "
    "Provider regressions and repeated development/public exposure remain disclosed; "
    "independent continuous-match gold confirmation is pending.")


def stage():
    if STAGED.exists():
        manifest = read(STAGED)
        for name, digest in manifest["models_sha256"].items():
            if sha256(DIRECTORY / name) != digest:
                raise ValueError("Staged model changed: " + name)
        return manifest
    old, selection, seal = read(ACTIVE), read(SELECTION), read(SEAL)
    if old["package_version"] != PREDECESSOR:
        raise ValueError("Expected the phase-average predecessor")
    if selection["pipeline_id"] != "base__ce0.5_cal0.375_ad0.125_arithmetic_ha3":
        raise ValueError("Only the user-approved frozen candidate may be promoted")
    if sha256(SEAL) != "9f35ef18d8c892c86b68db5edc8909356f1f18f51990fbf1263dbae4c2fffc77":
        raise ValueError("Frozen prediction seal changed")
    if sha256(GATE) != "6e38e4bb52d6dbb470a8cc8f7c6f19c3a9356151b83c9a8385ffb5460bbb852e":
        raise ValueError("Closed public gate changed")
    for path, digest in seal["development_result_bindings"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Frozen development evidence changed: " + path)
    preserve(PREVIOUS, ACTIVE.read_bytes())
    preserve(PREVIOUS_POLICY, POLICY.read_bytes())
    preserve(DIRECTORY / "previous_source/current_baseline.py", (ROOT / "experiments/current_baseline.py").read_bytes())
    models = {}
    for role, source in seal["model_role_paths"].items():
        expected = seal["model_sha256"][source]
        if sha256(source) != expected:
            raise ValueError("Frozen checkpoint changed: " + role)
        name = role.replace("/", "_") + ".joblib"
        preserve(DIRECTORY / name, Path(source).read_bytes())
        models[name] = expected
    public = read(PUBLIC)
    manifest = dict(version=11, model=MODEL, package_version=PACKAGE,
        status="User-authorized provisional working baseline; independent gold confirmation pending",
        promoted_utc=datetime.now(timezone.utc).isoformat(), replaced_baseline=old["model"],
        previous_manifest=str(PREVIOUS.relative_to(ROOT)), previous_manifest_sha256=sha256(PREVIOUS),
        previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)), previous_policy_sha256=sha256(PREVIOUS_POLICY),
        model_directory=str(DIRECTORY.relative_to(ROOT)), models_sha256=models,
        default_output=str(CSV.relative_to(ROOT)),
        architecture=dict(kind=inference.KIND, recipe=inference.RECIPE,
            seeds=list(inference.SEEDS), phase_order="seeds_then_phases",
            D="Exact predecessor decoded D mask", roles=list(inference.ROLES)),
        training=dict(ce_stage=1024, seeds=[2026, 2027], original_train_matches=13,
            gar_train_games=45, fitting_during_promotion=False,
            pooled_seal="experiments/results/v2_spell_balanced_v1_pooled_seal.json"),
        evidence=dict(report="experiments/V2_SPELL_FRONTIER.md",
            selection_sha256=sha256(SELECTION), original_development=selection["aggregate"],
            prediction_csv_sha256=seal["candidate_csv_sha256"],
            historical_clean_screen_passes=False, independent_continuous_gold_confirmation=False),
        promotion=dict(type="explicit_user_authorized_spell_frontier_upgrade",
            user_request="this is great. let's promote it", exception=EXCEPTION,
            automatic_promotion=False, fitting_during_promotion=False,
            organizer_rescoring_performed=False, public_diagnostic_known_before_authorization=True),
        organizer_public_diagnostic=dict(report=str(PUBLIC.relative_to(ROOT)),
            report_sha256=sha256(PUBLIC), diagnostic_only=True,
            used_for_model_or_hyperparameter_selection=False, independent_confirmation=False),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (
            Path(inference.__file__), Path(__file__),
            ROOT / "experiments/v2_spell_balanced_eval.py",
            ROOT / "experiments/v2_spell_balanced_public.py",
            ROOT / "experiments/v2_iterative_runtime.py")},
        organizer_reference_labels_used_for_training=False,
        historical_selection_records_preserved=True)
    preserve(STAGED, encode(manifest))
    return manifest


def promote():
    if RECEIPT.exists():
        receipt = read(RECEIPT)
        if sha256(ACTIVE) != receipt["baseline_manifest_sha256"] or sha256(POLICY) != receipt["policy_sha256"]:
            raise ValueError("Completed promotion differs from active metadata")
        return receipt
    manifest = stage()
    for path, digest in manifest["source_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError("Staged source changed: " + path)
    gar = ROOT / "experiments/results/spell_frontier_gar_v1.json"
    if not gar.is_file():
        raise ValueError("Finish the exact pooled GAR TEST/VALID check before promotion")
    if not CSV.exists():
        subprocess.run([sys.executable, "-m", "experiments.current_baseline", "predict",
                        "--manifest", str(STAGED)], cwd=ROOT, check=True)
    replay, seal = read(CSV.with_suffix(".json")), read(SEAL)
    if (sha256(CSV) != seal["candidate_csv_sha256"] or
        replay["baseline_manifest_sha256"] != sha256(STAGED) or
        replay["input_sha256"] != seal["input_sha256"] or
        replay["target_labels_read"] or replay["organizer_scored"]):
        raise ValueError("Normal inference must reproduce the sealed CSV without scoring")
    subprocess.run([sys.executable, "evaluation/validate_submission.py", str(CSV)], cwd=ROOT, check=True)
    if ACTIVE.read_bytes() != PREVIOUS.read_bytes() or POLICY.read_bytes() != PREVIOUS_POLICY.read_bytes():
        raise ValueError("Active predecessor changed during promotion")
    policy = read(PREVIOUS_POLICY)
    policy.update(version=11, confirmed_utc_date=manifest["promoted_utc"][:10])
    policy["working_candidate"] = dict(family="Three-expert conditional H/A with frozen predecessor D",
        model=MODEL, status=manifest["status"], baseline_manifest="experiments/baseline_manifest.json",
        baseline_manifest_sha256=sha256(STAGED), prediction_entrypoint="python -m experiments.current_baseline predict",
        reason=EXCEPTION, promotion_record=str(RECEIPT.relative_to(ROOT)),
        previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)))
    policy["clean_improvement_screen"].update(current_screen_passes=False, current_adoption_exception=EXCEPTION)
    policy["stress_guardrail"]["reference_recipe"] = MODEL
    preserve(ROOT / "experiments/baselines" / (PACKAGE + ".json"), encode(manifest))
    ACTIVE.write_bytes(encode(manifest))
    POLICY.write_bytes(encode(policy))
    receipt = dict(package_version=PACKAGE, authorization=manifest["promotion"],
        baseline_manifest_sha256=sha256(ACTIVE), policy_sha256=sha256(POLICY),
        previous_manifest_sha256=sha256(PREVIOUS), previous_policy_sha256=sha256(PREVIOUS_POLICY),
        candidate_csv_sha256=sha256(CSV), rows=199319, csv_equals_sealed_candidate=True,
        models_sha256=manifest["models_sha256"], gar_report_sha256=sha256(gar),
        dispatcher_sha256=sha256(ROOT / "experiments/current_baseline.py"),
        organizer_reference_accessed=False, organizer_rescored=False, fitting_performed=False,
        closed_public_gate_sha256=sha256(GATE), historical_declarations_and_results_preserved=True)
    preserve(RECEIPT, encode(receipt))
    preserve(DIRECTORY / "promotion_receipt.json", encode(receipt))
    preserve(DIRECTORY / "model_decision_policy.json", encode(policy))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "promote"))
    print((stage if parser.parse_args().action == "stage" else promote)()["package_version"])
