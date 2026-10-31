"""Package the user-approved sn_dense_v1 D2 ensemble without fitting or scoring.

``stage`` copies the two sealed pooled D2 checkpoints and their pooled teachers
into a hash-verified package and writes a staged manifest. ``promote`` replays
the package through ``current_baseline predict`` and requires the CSV to equal
the sealed public-check prediction byte for byte, then archives the previous
manifest/policy and activates the new ones. Historical screens stay recorded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import joblib

from experiments import sn_dense_baseline as inference
from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "joint_gru_sn_dense_ensemble"
PACKAGE = MODEL + "_v1"
STUDY = ROOT / "data/experiments/sn_dense_v1"
PUBLIC = STUDY / "public"
DIRECTORY = ROOT / "data/models" / PACKAGE
STAGED = DIRECTORY / "manifest.json"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
PREVIOUS = ROOT / "experiments/baselines/joint_gru_soccernet_capacity_ensemble_v1.json"
PREVIOUS_POLICY = ROOT / "experiments/baselines/model_decision_policy_joint_gru_soccernet_capacity_ensemble_v1.json"
RECEIPT = ROOT / "experiments/results/baseline_sn_dense_promotion.json"
CSV = ROOT / "data/experiments/current_baseline" / (MODEL + ".csv")
INPUTS = ROOT / "data/challenge/inputs.npz"
PUBLIC_RESULT = ROOT / "experiments/results/sn_dense_v1_public_check.json"
PROGRESS = ROOT / "experiments/results/sn_dense_v1_progress.md"
DECLARATION = ROOT / "experiments/results/sn_dense_v1_declaration.json"
USER_REQUEST = "let's promote this"
SOURCES = ("experiments/current_baseline.py", "experiments/sn_dense_baseline.py",
           "experiments/promote_sn_dense_baseline.py", "experiments/sn_dense_train.py",
           "experiments/sn_dense_public.py", "experiments/soccernet_capacity_model.py",
           "experiments/factorized_had.py", "experiments/learned_temporal.py",
           "experiments/live_gap_experiment.py", "experiments/supervised.py")


def read(path):
    return json.loads(Path(path).read_text())


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def preserve(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing package/history differs: {path}")
    else:
        with path.open("xb") as stream:
            stream.write(content)


def sealed():
    seal = read(PUBLIC / "competition_prediction_seal.json")
    if (sha256(PUBLIC / "competition_candidate.csv") != seal["prediction_sha256"]
            or sha256(INPUTS) != seal["input_sha256"]
            or sha256(DECLARATION) != seal["declaration_sha256"]):
        raise ValueError("Sealed public-check prediction, input or declaration changed")
    result = read(PUBLIC_RESULT)
    if result["prediction_seal_sha256"] != sha256(PUBLIC / "competition_prediction_seal.json"):
        raise ValueError("Public result is not bound to the sealed prediction")
    return seal, result


def stage():
    seal, public = sealed()
    old = read(ACTIVE)
    if old.get("package_version") != "joint_gru_soccernet_capacity_ensemble_v1":
        raise ValueError("Expected the SoccerNet-capacity predecessor")
    architecture = dict(kind="sn_dense_ensemble", members=inference.MEMBERS,
                        teachers=inference.TEACHERS, hidden_ha=96, hidden_d=24,
                        core_seconds=10, context_each_side_seconds=5, rate_hz=5,
                        decoder_switch_cost=3,
                        combination="Mean of two native H/A/D probability arrays before Viterbi; each member uses its fine-tuned D path and conditional H/A head with its pooled teacher's normalization")
    hashes, models = {}, {}
    for seed in (2026, 2027):
        sources = {f"dense_{seed}": PUBLIC / f"D2_f050_g45_pooled_s{seed}.joblib",
                   f"teacher_{seed}": ROOT / f"data/models/joint_gru_combined_ensemble_v1/joint_gru_{seed}.joblib"}
        for name, source in sources.items():
            if sha256(source) != seal["models_sha256"][str(source.relative_to(ROOT / "data"))]:
                raise ValueError("Deployment checkpoint differs from the public seal: " + name)
            target = DIRECTORY / (name + ".joblib")
            preserve(target, source.read_bytes())
            hashes[target.name] = sha256(target)
            models[name] = joblib.load(target)
    inference.validate_models(models, architecture)
    manifest = dict(
        version=7, model=MODEL, package_version=PACKAGE,
        status="User-authorized working development baseline; independent confirmation pending",
        promoted_utc=(read(STAGED)["promoted_utc"] if STAGED.exists()
                      else datetime.now(timezone.utc).isoformat()),
        replaced_baseline=old["model"], previous_manifest=str(PREVIOUS.relative_to(ROOT)),
        previous_manifest_sha256=sha256(ACTIVE), previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)),
        previous_policy_sha256=sha256(POLICY), model_directory=str(DIRECTORY.relative_to(ROOT)),
        models_sha256=hashes, default_output=str(CSV.relative_to(ROOT)), architecture=architecture,
        training=dict(recipe="D2_f050_g45: dense touch-stream H/A/D targets with complete-spell and transition losses on SoccerNet, D-path fine-tune, fraction .5",
                      original_train_matches=13, soccernet_train_games=45, learning_rate=3e-4,
                      d_learning_rate=1e-4, optimizer_updates=2048, seeds=[2026, 2027],
                      dense_gar_possession_labels=True, d_path_trained=True,
                      normalization_frozen=True),
        evidence=dict(
            declaration=str(DECLARATION.relative_to(ROOT)), declaration_sha256=sha256(DECLARATION),
            progress=str(PROGRESS.relative_to(ROOT)), progress_sha256=sha256(PROGRESS),
            development_full_provider_holdout=dict(
                two_seed_ensemble_clean_spell=0.6472, two_seed_ensemble_feed_like_spell=0.5058,
                previous_adopted_clean_spell=0.641790, previous_adopted_feed_like_spell=0.502586,
                seed_2026_clean_spell_vs_rebuilt_reference=[0.6355, 0.6288],
                seed_2027_clean_spell=0.6372, provider_and_feed_like_guards_pass=True,
                clean_improvement_screen_plus_01_passes=False,
                previously_exposed_development_matches=True),
            gar_valid_silver_note="GAR-valid dense silver partly rewards agreement with the labelling rule; not neutral evidence",
            prediction_seal_sha256=sha256(PUBLIC / "competition_prediction_seal.json"),
            prediction_csv_sha256=seal["prediction_sha256"]),
        promotion=dict(type="explicit_user_authorized_post_hoc_screen_exception", user_request=USER_REQUEST,
                       original_screen_passes=False, automatic_promotion=False,
                       fitting_during_promotion=False, organizer_rescoring_performed=False,
                       public_diagnostic_known_before_authorization=True),
        organizer_public_diagnostic=dict(
            report=str(PUBLIC_RESULT.relative_to(ROOT)), known_at_promotion=True, diagnostic_only=True,
            spell_f1=public["candidate"]["spell_f1_mean"], macro_f1=public["candidate"]["macro_f1"],
            previous_spell_f1=public["adopted_baseline"]["spell_f1_mean"],
            used_for_model_or_hyperparameter_selection=False, independent_confirmation=False),
        source_sha256={name: sha256(ROOT / name) for name in SOURCES},
        organizer_reference_labels_used_for_training=False,
        historical_selection_records_preserved=True)
    preserve(STAGED, encode(manifest))
    return manifest


def updated_policy(previous, candidate):
    policy = json.loads(json.dumps(previous))
    policy.update(version=7, confirmed_utc_date=candidate["promoted_utc"][:10])
    policy["working_candidate"] = dict(
        family="Two-seed dense-SoccerNet GRU ensemble with fine-tuned D paths",
        model=MODEL, status=candidate["status"], baseline_manifest="experiments/baseline_manifest.json",
        baseline_manifest_sha256=sha256(STAGED),
        prediction_entrypoint="python -m experiments.current_baseline predict",
        reason="User-authorized post-hoc promotion: positive clean/feed-like development gains in both seeds; provider guards pass; +.01 clean screen remains failed.",
        promotion_record=str(RECEIPT.relative_to(ROOT)), previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)),
        public_diagnostic_known_before_user_request=True)
    policy["stress_guardrail"]["reference_recipe"] = MODEL
    policy["clean_improvement_screen"] = dict(
        default_delta=.01, historical_failures_preserved=True,
        current_adoption_exception="Explicit user decision; two-seed clean gain +.0054 over the previous package; not a retroactive screen pass.")
    return policy


def promote():
    if RECEIPT.exists():
        receipt = read(RECEIPT)
        if sha256(ACTIVE) != receipt["baseline_manifest_sha256"] or sha256(POLICY) != receipt["policy_sha256"]:
            raise ValueError("Completed promotion metadata differs")
        return receipt
    candidate = stage()
    seal, _ = sealed()
    if not CSV.exists():
        subprocess.run([sys.executable, "-m", "experiments.current_baseline", "predict",
                        "--manifest", str(STAGED)], cwd=ROOT, check=True)
    validation = read(CSV.with_suffix(".json"))
    if (sha256(CSV) != seal["prediction_sha256"] or validation.get("output_sha256") != seal["prediction_sha256"]
            or validation.get("baseline_manifest_sha256") != sha256(STAGED)
            or validation.get("target_labels_read") is not False
            or validation.get("organizer_scored") is not False):
        raise ValueError("Baseline CLI did not reproduce the sealed prediction without labels")
    for name, digest in candidate["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Staged inference source changed: " + name)
    preserve(PREVIOUS, ACTIVE.read_bytes())
    preserve(PREVIOUS_POLICY, POLICY.read_bytes())
    policy = updated_policy(read(POLICY), candidate)
    ACTIVE.write_bytes(encode(candidate))
    POLICY.write_bytes(encode(policy))
    receipt = dict(package_version=PACKAGE, promoted_utc=candidate["promoted_utc"],
                   baseline_manifest_sha256=sha256(ACTIVE), policy_sha256=sha256(POLICY),
                   previous_manifest_sha256=sha256(PREVIOUS), previous_policy_sha256=sha256(PREVIOUS_POLICY),
                   models_sha256=candidate["models_sha256"], evidence=candidate["evidence"],
                   authorization=candidate["promotion"],
                   validation_receipt_sha256=sha256(CSV.with_suffix(".json")),
                   candidate_csv_sha256=sha256(CSV), csv_equals_sealed_public_prediction=True,
                   rows=199319, fitting_performed=False, organizer_reference_accessed=False,
                   organizer_rescored=False, historical_score_gates_and_declarations_preserved=True)
    preserve(RECEIPT, encode(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "promote"))
    args = parser.parse_args()
    value = stage() if args.action == "stage" else promote()
    print(args.action.upper(), value.get("package_version"))


if __name__ == "__main__":
    main()
