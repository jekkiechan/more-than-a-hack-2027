"""Package the user-approved SoccerNet baseline without fitting or scoring.

Stage a hash-verified, self-contained package, replay it through the baseline
CLI, then promote those exact bytes. The historical +.01 screen stays failed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import joblib

from experiments import soccernet_baseline as inference
from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "joint_gru_soccernet_capacity_ensemble"
PACKAGE = MODEL + "_v1"
OUT = ROOT / "data/experiments/soccernet_capacity_v1"
DIRECTORY = ROOT / "data/models" / PACKAGE
STAGED = DIRECTORY / "manifest.json"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
PREVIOUS = ROOT / "experiments/baselines/joint_gru_combined_ensemble_v1.json"
PREVIOUS_POLICY = ROOT / "experiments/baselines/model_decision_policy_joint_gru_combined_ensemble_v1.json"
RECEIPT = ROOT / "experiments/results/baseline_soccernet_capacity_promotion.json"
CSV = ROOT / "data/experiments/current_baseline" / (MODEL + ".csv")
INPUTS = ROOT / "data/challenge/inputs.npz"
VERIFY = ROOT / "experiments/results/soccernet_capacity_verification.json"
EXPECTED_VERIFY = "162715bda90c0a5d1edcf41ae8223804baf4fece819b07149a8f5efbae72ec07"
EXPECTED_DECLARATION = "27e16319fd5d8ab430429732f74f4c3821651cce9f109e9fbabfc037a3951dc7"
EXPECTED_PREDICTION_SEAL = "4a48909e18e45fae3ae133d9af1a3bc2c16569a5f93cb036aa0d49e1c3900449"
EXPECTED_CSV = "68aa3a3e1e908e60f61ab2a68fef987504e097262f500c7374549c89332dcc0a"
EXPECTED_PROBABILITY = "426663c9ea8b70251971fa736cad5a9632ac24581bccd17349a65c987a980621"
REPLAY_VERIFIER = ROOT / "scripts/verify_soccernet_baseline.py"
EXPECTED_REPLAY_VERIFIER = "cbc0fa90d41a000b16a484564b1e12c3a4dd71707303b95f0a36d2ede2bc1f61"
TRANSACTION = DIRECTORY / "promotion_transaction.json"


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


def selection_evidence(report, control, decision):
    """Verify positive matched gains and guards without relaxing the old screen."""
    if (decision.get("selected_recipe") != "w96_lr3e-4_step2048"
            or decision.get("selected_width") != 96
            or decision.get("adopt_model") is not False
            or decision.get("improvement_screen_passes") is not False):
        raise ValueError("Historical selection or failed screen changed")
    summaries = report["summaries"]
    baseline, candidate = summaries["baseline"], summaries["w96"]
    deltas = {view: candidate[view]["equal_provider_mean"]["spell_f1"]
              - baseline[view]["equal_provider_mean"]["spell_f1"]
              for view in ("clean", "feed_like")}
    provider_guards = {
        f"{provider}_{metric}": values[metric] >= baseline["clean"]["per_provider"][provider][metric] - .02
        for provider, values in candidate["clean"]["per_provider"].items()
        for metric in ("d_f1", "macro_f1")
    }
    if (set(candidate["clean"]["per_provider"]) != {"metrica", "skillcorner", "sportec"}
            or not all(provider_guards.values()) or deltas["feed_like"] < -.01
            or not 0 < deltas["clean"] < .01 or deltas["feed_like"] <= 0
            or abs(deltas["clean"] - decision["clean_spell_delta_vs_baseline"]) > 1e-12
            or not all(decision["guards"]["w96"].values())):
        raise ValueError("Matched development gains or provider/stress guards changed")
    control_deltas = {view: candidate[view]["equal_provider_mean"]["spell_f1"]
                      - control["summaries"]["old_only"][view]["equal_provider_mean"]["spell_f1"]
                      for view in deltas}
    if any(value <= 0 for value in control_deltas.values()):
        raise ValueError("SoccerNet no longer beats the matched old-only control")
    return dict(view="full_provider_holdout", previously_exposed_development_matches=True,
                candidate=candidate, previous_baseline=baseline,
                clean_and_feed_like_spell_deltas=deltas, matched_control_spell_deltas=control_deltas,
                provider_clean_guards=provider_guards, feed_like_guard_passes=True,
                historical_improvement_screen_passes=False, historical_adopt_model=False,
                original_clean_screen=.01, exception_is_post_hoc=True,
                independent_confirmation=False)


def verified_inputs():
    if sha256(VERIFY) != EXPECTED_VERIFY or sha256(OUT / "predeclared.json") != EXPECTED_DECLARATION:
        raise ValueError("Frozen declaration or independent verification changed")
    verification = read(VERIFY)
    if verification.get("status") != "PASS":
        raise ValueError("Independent verification did not pass")
    public_names = {
        "coarse_decision": "soccernet_capacity_coarse_decision.json",
        "coarse_report": "soccernet_capacity_coarse_report.json",
        "control_report": "soccernet_capacity_control_report.json",
        "final_decision": "soccernet_capacity_final_decision.json",
        "final_report": "soccernet_capacity_final_report.json",
    }
    for key, filename in public_names.items():
        if sha256(VERIFY.parent / filename) != verification["provenance_hashes"]["aggregate_public_file_sha256"][key]:
            raise ValueError("Verified aggregate evidence changed: " + filename)
    for stage, filename in (("coarse", "verification_coarse.json"), ("final", "verification_final.json"),
                            ("control", "verification_control.json"), ("pooled", "verification_pooled.json"),
                            ("public", "verification_public.json")):
        path = OUT / filename
        if (sha256(path) != verification["provenance_hashes"]["stage_verification_sha256"][stage]
                or read(path).get("status") != "PASS"):
            raise ValueError("Independent stage receipt changed: " + stage)
    if sha256(OUT / "competition_prediction_seal.json") != EXPECTED_PREDICTION_SEAL:
        raise ValueError("Sealed prediction provenance changed")
    seal = read(OUT / "competition_prediction_seal.json")
    for filename, digest in seal["external_evidence_sha256"].items():
        if sha256(OUT / filename) != digest:
            raise ValueError("Private external selection evidence changed: " + filename)
    if sha256(INPUTS) != seal["input_sha256"] or sha256(OUT / "competition_candidate.csv") != EXPECTED_CSV:
        raise ValueError("Sealed input or prediction changed")
    # The active metadata and dispatch are deliberately updated by promotion.
    # Validate their original bytes against the preserved study snapshot.
    replaced = {"experiments/current_baseline.py", "experiments/baseline_manifest.json",
                "experiments/model_decision_policy.json"}
    for relative, digest in read(OUT / "predeclared.json")["source_sha256"].items():
        path = OUT / "source_snapshot" / relative if relative in replaced else ROOT / relative
        if sha256(path) != digest:
            raise ValueError("Frozen study source changed: " + relative)
    result = selection_evidence(read(VERIFY.parent / public_names["final_report"]),
                               read(VERIFY.parent / public_names["control_report"]),
                               read(OUT / "final_decision.json"))
    return seal, result


def stage():
    seal, evidence = verified_inputs()
    old = read(ACTIVE)
    if old.get("package_version") != "joint_gru_combined_ensemble_v1":
        raise ValueError("Expected the combined-objective predecessor")
    original_hashes = read(OUT / "predeclared.json")["baseline"]
    if sha256(ACTIVE) != original_hashes["manifest_sha256"] or sha256(POLICY) != original_hashes["policy_sha256"]:
        raise ValueError("Active predecessor manifest or policy changed")
    architecture = dict(kind="soccernet_capacity_ensemble", members=inference.MEMBERS,
                        teachers=inference.TEACHERS, hidden_ha=96, hidden_d=24,
                        core_seconds=10, context_each_side_seconds=5, rate_hz=5,
                        decoder_switch_cost=3,
                        combination="Mean of two native H/A/D probability arrays before Viterbi; each H/A conditioned on its matching unchanged teacher P(D)")
    hashes, models = {}, {}
    for seed in (2026, 2027):
        paths = {
            f"capacity_{seed}": OUT / f"pooled/w96_lr3e-4_step2048/seed{seed}.joblib",
            f"teacher_{seed}": ROOT / old["model_directory"] / f"joint_gru_{seed}.joblib",
        }
        for name, source in paths.items():
            expected = seal["models_sha256"][str(source.relative_to(ROOT / "data"))]
            if sha256(source) != expected:
                raise ValueError("Deployment checkpoint changed: " + name)
            target = DIRECTORY / (name + ".joblib")
            preserve(target, source.read_bytes())
            hashes[target.name] = sha256(target)
            models[name] = joblib.load(target)
    inference.validate_models(models, architecture)
    sources = ["experiments/current_baseline.py", "experiments/soccernet_baseline.py",
               "experiments/promote_soccernet_baseline.py", "experiments/soccernet_capacity_model.py",
               "experiments/objective_losses.py", "experiments/factorized_had.py",
               "experiments/learned_temporal.py", "experiments/live_gap_experiment.py",
               "experiments/supervised.py", "experiments/robust.py", "experiments/refine_features.py",
               "experiments/refine_domain.py", "experiments/stress.py", "experiments/challenge.py"]
    manifest = dict(version=6, model=MODEL, package_version=PACKAGE,
                    status="User-authorized working development baseline; independent confirmation pending",
                    promoted_utc=(read(STAGED)["promoted_utc"] if STAGED.exists()
                                  else datetime.now(timezone.utc).isoformat()), replaced_baseline=old["model"],
                    previous_manifest=str(PREVIOUS.relative_to(ROOT)), previous_manifest_sha256=sha256(ACTIVE),
                    previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)), previous_policy_sha256=sha256(POLICY),
                    model_directory=str(DIRECTORY.relative_to(ROOT)), models_sha256=hashes,
                    default_output=str(CSV.relative_to(ROOT)), architecture=architecture,
                    training=dict(recipe="w96_lr3e-4_step2048", original_train_matches=13,
                                  soccernet_train_games=45, sparse_ha_anchors=42919,
                                  retained_tracking_hours=44.67, gar_batch_fraction=.5,
                                  learning_rate=3e-4, optimizer_updates=2048, seeds=[2026, 2027],
                                  dense_gar_possession_labels=False, normalization_and_d_frozen=True),
                    evidence=dict(comparison=evidence, verification=str(VERIFY.relative_to(ROOT)),
                                  verification_sha256=sha256(VERIFY), declaration_sha256=EXPECTED_DECLARATION,
                                  prediction_seal_sha256=EXPECTED_PREDICTION_SEAL,
                                  prediction_csv_sha256=EXPECTED_CSV),
                    promotion=dict(type="explicit_user_authorized_post_hoc_screen_exception",
                                   user_request="yeah let's go", prior_context="Promote width 96 as the working development baseline; explicitly preserve failed +.01 clean screen.",
                                   original_screen_passes=False, automatic_promotion=False,
                                   fitting_during_promotion=False, organizer_rescoring_performed=False,
                                   public_diagnostic_known_before_authorization=True),
                    organizer_public_diagnostic=dict(report="experiments/results/soccernet_capacity_public_check.json",
                                                     known_at_promotion=True, diagnostic_only=True,
                                                     used_for_model_or_hyperparameter_selection=False,
                                                     independent_confirmation=False),
                    artifact_availability=dict(local_hash_verified=True, remote_uploaded=False,
                                               private_cache_default_still_predecessor=True,
                                               receipt="experiments/results/soccernet_capacity_artifacts.json"),
                    source_sha256={name: sha256(ROOT / name) for name in sources},
                    organizer_reference_labels_used_for_training=False,
                    historical_selection_records_preserved=True)
    preserve(STAGED, encode(manifest))
    return manifest


def verify_replay(candidate):
    """Bind replay attestations to the exact sealed input, models and arrays."""
    import numpy as np
    replay_path = DIRECTORY / "probability_replay.json"
    replay = read(replay_path)
    with np.load(OUT / "competition_probability.npz", allow_pickle=False) as saved:
        probability_sha = hashlib.sha256(saved["probability"].tobytes()).hexdigest()
    expected = dict(package_version=PACKAGE, baseline_manifest_sha256=sha256(STAGED),
                    input_sha256=sha256(INPUTS), models_sha256=candidate["models_sha256"],
                    probability_exact=True, decoded_prediction_exact=True, csv_exact=True,
                    candidate_csv_sha256=EXPECTED_CSV, rows=199319,
                    probability_sha256=probability_sha,
                    verifier_source_sha256=EXPECTED_REPLAY_VERIFIER,
                    organizer_reference_accessed=False, organizer_scored=False,
                    model_training_performed=False)
    if (probability_sha != EXPECTED_PROBABILITY
            or sha256(REPLAY_VERIFIER) != EXPECTED_REPLAY_VERIFIER
            or any(replay.get(key) != value for key, value in expected.items())):
        raise ValueError("Exact replay receipt is not bound to the sealed inputs, models, CSV and verifier")
    return replay_path


def updated_policy(previous, candidate):
    policy = json.loads(json.dumps(previous))
    policy.update(version=6, confirmed_utc_date=candidate["promoted_utc"][:10])
    policy["working_candidate"] = dict(
        family="Two-seed SoccerNet conditional H/A GRU ensemble with frozen original D paths",
        model=MODEL, status=candidate["status"], baseline_manifest="experiments/baseline_manifest.json",
        baseline_manifest_sha256=sha256(STAGED), prediction_entrypoint="python -m experiments.current_baseline predict",
        reason="User-authorized post-hoc promotion: positive clean/feed-like gains and matched-control gains; provider guards pass; +.01 clean screen remains failed.",
        promotion_record=str(RECEIPT.relative_to(ROOT)), previous_policy=str(PREVIOUS_POLICY.relative_to(ROOT)),
        public_diagnostic_known_before_user_request=True)
    policy["stress_guardrail"]["reference_recipe"] = MODEL
    policy["clean_improvement_screen"] = dict(default_delta=.01, historical_failures_preserved=True,
                                             current_adoption_exception="Explicit user decision; measured clean gain +.008584746; not a retroactive screen pass.")
    return policy


def atomic_replace(path, content):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        staged = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(staged, path)
    finally:
        staged.unlink(missing_ok=True)


def finish_transaction():
    """Complete a prepared promotion after a crash between metadata writes."""
    pending = read(TRANSACTION)
    candidate = read(STAGED)
    if (pending["manifest"] != candidate
            or pending["policy"] != updated_policy(read(PREVIOUS_POLICY), candidate)
            or sha256(PREVIOUS) != candidate["previous_manifest_sha256"]
            or sha256(PREVIOUS_POLICY) != candidate["previous_policy_sha256"]):
        raise ValueError("Promotion journal or archived predecessor differs")
    verified_inputs()
    replay_path = verify_replay(candidate)
    for name, digest in candidate["models_sha256"].items():
        if sha256(DIRECTORY / name) != digest:
            raise ValueError("Pending promotion model changed")
    for name, digest in candidate["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Pending promotion source changed")
    receipt = pending["receipt"]
    if (receipt["baseline_manifest_sha256"] != sha256(STAGED)
            or receipt["policy_sha256"] != hashlib.sha256(encode(pending["policy"])).hexdigest()
            or receipt["probability_replay_receipt_sha256"] != sha256(replay_path)
            or receipt["candidate_csv_sha256"] != sha256(CSV)
            or receipt["validation_receipt_sha256"] != sha256(CSV.with_suffix(".json"))):
        raise ValueError("Promotion journal no longer matches verified outputs")
    targets = [(ACTIVE, encode(candidate), candidate["previous_manifest_sha256"]),
               (POLICY, encode(pending["policy"]), candidate["previous_policy_sha256"])]
    for target, content, old_digest in targets:
        new_digest = hashlib.sha256(content).hexdigest()
        if sha256(target) not in (old_digest, new_digest):
            raise ValueError("Active metadata changed outside the prepared promotion")
    for target, content, _ in targets:
        atomic_replace(target, content)
    preserve(RECEIPT, encode(receipt))
    return receipt


def promote():
    if RECEIPT.exists():
        receipt = read(RECEIPT)
        if (sha256(ACTIVE) != receipt["baseline_manifest_sha256"]
                or sha256(POLICY) != receipt["policy_sha256"]):
            raise ValueError("Completed promotion metadata differs")
        from experiments import current_baseline
        current_baseline.load_models()
        return receipt
    if TRANSACTION.exists():
        return finish_transaction()
    candidate = stage()
    validation = read(CSV.with_suffix(".json"))
    if (sha256(CSV) != EXPECTED_CSV or validation.get("output_sha256") != EXPECTED_CSV
            or validation.get("baseline_manifest_sha256") != sha256(STAGED)
            or validation.get("input_sha256") != sha256(INPUTS)
            or validation.get("target_labels_read") is not False
            or validation.get("organizer_scored") is not False):
        raise ValueError("Actual baseline CLI did not reproduce the sealed candidate without labels")
    replay_path = verify_replay(candidate)
    for name, digest in candidate["source_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise ValueError("Staged inference source changed: " + name)
    preserve(PREVIOUS, ACTIVE.read_bytes())
    preserve(PREVIOUS_POLICY, POLICY.read_bytes())
    policy = updated_policy(read(POLICY), candidate)
    receipt = dict(package_version=PACKAGE, promoted_utc=candidate["promoted_utc"],
                   baseline_manifest_sha256=sha256(STAGED), policy_sha256=hashlib.sha256(encode(policy)).hexdigest(),
                   previous_manifest_sha256=sha256(PREVIOUS), previous_policy_sha256=sha256(PREVIOUS_POLICY),
                   models_sha256=candidate["models_sha256"], comparison=candidate["evidence"]["comparison"],
                   authorization=candidate["promotion"], validation_receipt_sha256=sha256(CSV.with_suffix(".json")),
                   probability_replay_receipt_sha256=sha256(replay_path), candidate_csv_sha256=sha256(CSV),
                   rows=199319, fitting_performed=False, organizer_reference_accessed=False,
                   organizer_rescored=False, remote_archive_uploaded=False,
                   historical_score_gates_and_declarations_preserved=True)
    preserve(TRANSACTION, encode(dict(manifest=candidate, policy=policy, receipt=receipt)))
    return finish_transaction()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "promote"))
    args = parser.parse_args()
    value = stage() if args.action == "stage" else promote()
    print(args.action.upper(), value.get("package_version"))


if __name__ == "__main__":
    main()
