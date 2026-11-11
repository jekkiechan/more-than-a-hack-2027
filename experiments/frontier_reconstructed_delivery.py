"""Verify and promote the reconstructed pooled frontier + fixed Fable package.

This runner consumes the separately sealed pooled reconstruction outputs. It
does not fit or score during package delivery. All predictions use the ordinary
label-free inference modules; the GAR adapter temporarily redirects only the
legacy runner's paths and loader, then restores them.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from experiments import ball_exit_delivery as legacy_delivery
from experiments import ball_exit_gar as legacy_gar
from experiments import ball_exit_runtime, current_baseline

ROOT = Path(__file__).resolve().parents[1]
STUDY = "ball_exit_reconstructed_v1"
PACKAGE = "joint_gru_sn_dense_spell_frontier_ball_exit_reconstructed_ensemble_v1"
MODEL = PACKAGE.removesuffix("_v1")
OUT = ROOT / "data/experiments" / STUDY
RECON = ROOT / "data/experiments/frontier_pooled_reconstruction_v1/models"
MODEL_DIR = ROOT / "data/models" / PACKAGE
TRAIN_RECEIPT = ROOT / "experiments/results/frontier_pooled_reconstruction_v1_seal.json"
RECON_PLAN = ROOT / "experiments/results/frontier_pooled_reconstruction_v1_plan.json"
RECON_DECL = ROOT / "experiments/results/frontier_pooled_reconstruction_v1_declaration.json"
RECON_VERIFY = ROOT / "experiments/results/frontier_pooled_reconstruction_v1_verification.json"
FRONTIER = ROOT / "experiments/baselines/joint_gru_sn_dense_spell_frontier_ensemble_v1.json"
PARENT = ROOT / "experiments/baselines/joint_gru_sn_dense_spell_frontier_ball_exit_ensemble_v1.json"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
STAGED = OUT / "baseline_manifest.json"
CONTROL = OUT / "frontier_control.csv"
CONTROL_MANIFEST = OUT / "frontier_control_manifest.json"
CSV = OUT / "candidate.csv"
SEAL = OUT / "prediction_seal.json"
GATE = OUT / "public_gate_claimed.json"
PUBLIC = ROOT / f"experiments/results/{STUDY}_public.json"
GAR = ROOT / f"experiments/results/{STUDY}_gar.json"
GAR_DECL = ROOT / f"experiments/results/{STUDY}_gar_declaration.json"
RECEIPT = ROOT / f"experiments/results/{STUDY}.json"
ARCHIVE_VERIFY = ROOT / f"experiments/results/{STUDY}_archive_verification.json"
EXPECTED_FRONTIER_CSV = "26af7348940626da1cbe05a0c73b2a9db41ad9cedcb3ae6adc5de2771b6ce5dc"
SEEDS = (2026, 2027)
_legacy_source_inventory = legacy_delivery.sources


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def reconstruction_models():
    return {f"ce_{seed}.joblib": RECON / f"stage1024_{seed}.joblib" for seed in SEEDS}


def training_model_paths():
    return {f"stage{stage}_{seed}.joblib": RECON / f"stage{stage}_{seed}.joblib"
            for stage in (512, 1024) for seed in SEEDS}


def sources():
    names = dict(_legacy_source_inventory())
    names[str(Path(legacy_gar.__file__).relative_to(ROOT))] = sha(legacy_gar.__file__)
    names[str(ROOT / "experiments/frontier_pooled_reconstruction.py").replace(str(ROOT) + "/", "")] = sha(
        ROOT / "experiments/frontier_pooled_reconstruction.py")
    names[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    return names


def check_training_seal():
    plan = read(RECON_PLAN)
    declaration = read(RECON_DECL)
    seal = read(TRAIN_RECEIPT)
    if plan.get("study") != "frontier_pooled_reconstruction_v1":
        raise ValueError("Unexpected reconstruction plan")
    if (declaration.get("plan_sha256") != sha(RECON_PLAN) or
            seal.get("declaration_sha256") != sha(RECON_DECL)):
        raise ValueError("Training declaration/seal does not bind the predeclared plan")
    files = training_model_paths()
    fits = seal.get("fits", [])
    recorded = {fit.get("path"): fit.get("sha256") for fit in fits}
    expected_paths = {str(path.relative_to(ROOT)): sha(path) for path in files.values() if path.is_file()}
    if (len(fits) != 4 or len(recorded) != 4 or recorded != expected_paths or
            seal.get("models_sha256") != expected_paths):
        raise ValueError("Reconstruction seal does not bind exactly all four staged fits")
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if recorded.get(str(path.relative_to(ROOT))) != sha(path):
            raise ValueError(f"Reconstructed model is not sealed: {name}")
    if seal.get("all_four_fits_complete") is not True or seal.get("original_byte_identity") is not False:
        raise ValueError("Reconstruction seal is incomplete or misstates artifact identity")
    for fit in fits:
        if fit.get("adopted_d_state_unchanged") is not True:
            raise ValueError("Reconstructed fit did not retain the frozen adopted D state")
    verification = read(RECON_VERIFY)
    if (verification.get("seal_sha256") != sha(TRAIN_RECEIPT) or
            verification.get("declaration_sha256") != sha(RECON_DECL) or
            verification.get("all_passed") is not True or verification.get("fit_count") != 4 or
            verification.get("models_sha256") != expected_paths):
        raise ValueError("Independent reconstruction verification is missing or failed")
    return seal


def load_models(manifest_path=STAGED):
    manifest = read(manifest_path)
    for rel, expected in manifest["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError("Staged source changed: " + rel)
    seal = check_training_seal()
    if manifest["reconstruction"]["training_seal_sha256"] != sha(TRAIN_RECEIPT):
        raise ValueError("Staged training seal changed")
    if manifest["reconstruction"]["verification_sha256"] != sha(RECON_VERIFY):
        raise ValueError("Staged reconstruction verification changed")
    recorded = {fit["path"]: fit["sha256"] for fit in seal["fits"]}
    if recorded != {str(v.relative_to(ROOT)): sha(v) for v in training_model_paths().values()}:
        raise ValueError("Reconstructed models changed after verification")
    return current_baseline.load_models(manifest_path=manifest_path)


def _materialize_package_models(parent, model_hashes):
    source_dir = ROOT / parent["model_directory"]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in model_hashes.items():
        source = reconstruction_models()[name] if name.startswith("ce_") else source_dir / name
        target = MODEL_DIR / name
        if sha(source) != expected:
            raise ValueError("Source model hash mismatch before packaging: " + name)
        if target.exists():
            if sha(target) != expected:
                raise ValueError("Existing reconstructed package model differs: " + name)
        else:
            shutil.copyfile(source, target)
        if sha(target) != expected:
            raise ValueError("Packaged model hash mismatch: " + name)


def stage():
    if ACTIVE.read_bytes() != PARENT.read_bytes():
        raise ValueError("Active predecessor changed; preserve it and do not replace a newer baseline")
    parent = read(PARENT)
    if parent.get("previous_manifest_sha256") != sha(FRONTIER):
        raise ValueError("Ball-exit predecessor is not bound to the frozen frontier manifest")
    parent_dir = ROOT / parent["model_directory"]
    for name, expected in parent["models_sha256"].items():
        if name.startswith("ce_"):
            continue  # These are the specifically missing pooled weights being reconstructed.
        if sha(parent_dir / name) != expected:
            raise ValueError("Frozen non-CE predecessor model is missing or changed: " + name)
    check_training_seal()
    frontier = read(FRONTIER)
    model_hashes = dict(parent["models_sha256"])
    for name, path in reconstruction_models().items():
        model_hashes[name] = sha(path)
    _materialize_package_models(parent, model_hashes)
    training = dict(parent["training"])
    training.update(source_pooled_seal=training.get("pooled_seal"),
        pooled_seal=str(TRAIN_RECEIPT.relative_to(ROOT)), reconstruction_declaration=str(RECON_DECL.relative_to(ROOT)),
        reconstruction_declaration_sha256=sha(RECON_DECL),
        reconstruction_verification=str(RECON_VERIFY.relative_to(ROOT)),
        reconstruction_verification_sha256=sha(RECON_VERIFY),
        fitting_during_promotion=False, reconstructed_roles=["ce_2026", "ce_2027"],
        original_checkpoint_byte_identity=False)
    manifest = dict(parent)
    manifest.update(version=13, model=MODEL, package_version=PACKAGE,
        status="Reconstructed pooled frontier package staged; exact reference-free parity and fixed Fable regression checks required",
        previous_manifest=str(PARENT.relative_to(ROOT)), previous_manifest_sha256=sha(PARENT),
        replaced_baseline=parent["model"], default_output=str(CSV.relative_to(ROOT)),
        model_directory=str(MODEL_DIR.relative_to(ROOT)), models_sha256=model_hashes,
        training=training,
        architecture=dict(parent["architecture"],
            reconstruction="Pooled CE stage-1024 weights reconstructed from the predeclared original recipe",
            reconstructed_frontier_control_csv_parity="pending_exact_199319_row_replay"),
        reconstruction={"plan": str(RECON_PLAN.relative_to(ROOT)), "plan_sha256": sha(RECON_PLAN),
            "source_frontier_manifest_sha256": sha(FRONTIER),
            "training_seal": str(TRAIN_RECEIPT.relative_to(ROOT)), "training_seal_sha256": sha(TRAIN_RECEIPT),
            "verification": str(RECON_VERIFY.relative_to(ROOT)), "verification_sha256": sha(RECON_VERIFY),
            "training_outputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                                 for name, path in reconstruction_models().items()},
            "historical_frontier_csv_sha256": frontier["evidence"]["prediction_csv_sha256"],
            "independent_continuous_gold_confirmation": False},
        evidence=dict(parent["evidence"], reconstructed_frontier=True,
            reconstructed_frontier_control_csv_parity="pending_exact_199319_row_replay",
            reconstruction_plan_sha256=sha(RECON_PLAN), training_seal_sha256=sha(TRAIN_RECEIPT)),
        promotion={"user_authorized": True, "fitting_during_promotion": False,
                   "public_used_for_selection": False, "organizer_rescoring_performed": False,
                   "automatic_promotion": False, "model_checkpoints_unchanged": False,
                   "reconstructed_model_roles": ["ce_2026", "ce_2027"],
                   "unchanged_predecessor_model_roles": [name.removesuffix(".joblib") for name in model_hashes
                                                          if not name.startswith("ce_")],
                   "source_status": "reconstructed_not_original_checkpoint_restore",
                   "reason": "The original pooled CE checkpoint bytes were unavailable; the fixed pooled recipe was reconstructed from the sealed source and requires exact prediction parity before deployment checks."},
        source_sha256=sources())
    manifest.pop("promoted_utc", None)
    manifest.pop("regression_checks", None)
    write_once(STAGED, manifest)
    control_manifest = dict(read(FRONTIER), model_directory=str(MODEL_DIR.relative_to(ROOT)),
        models_sha256=model_hashes,
        reconstruction={"source_manifest_sha256": sha(FRONTIER), "training_seal_sha256": sha(TRAIN_RECEIPT)})
    write_once(CONTROL_MANIFEST, control_manifest)
    return manifest


def check_manifest():
    manifest = read(STAGED)
    if manifest.get("package_version") != PACKAGE:
        raise ValueError("Unexpected staged package")
    ball_exit_runtime.frontier_manifest(manifest)
    if manifest["previous_manifest_sha256"] != sha(PARENT):
        raise ValueError("Predecessor manifest changed")
    if manifest["reconstruction"]["source_frontier_manifest_sha256"] != sha(FRONTIER):
        raise ValueError("Frozen frontier source manifest changed")
    for rel, digest in manifest["source_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("Staged source changed: " + rel)
    if (manifest["reconstruction"]["training_seal_sha256"] != sha(TRAIN_RECEIPT) or
            manifest["reconstruction"]["verification_sha256"] != sha(RECON_VERIFY)):
        raise ValueError("Staged training seal changed")
    load_models(STAGED)
    return manifest


def _cli_predict(manifest_path, output):
    subprocess.run([sys.executable, "-m", "experiments.current_baseline", "predict",
                    "--manifest", str(manifest_path), "--output", str(output)], cwd=ROOT, check=True)


def predict():
    manifest = check_manifest()
    if (SEAL.exists() or GATE.exists() or CONTROL.exists() or CSV.exists() or
            CONTROL.with_suffix(".json").exists() or CSV.with_suffix(".json").exists()):
        raise FileExistsError("Preserve any existing prediction or gate")
    load_models(STAGED)
    _cli_predict(CONTROL_MANIFEST, CONTROL)
    if sha(CONTROL) != EXPECTED_FRONTIER_CSV:
        raise ValueError("Reconstructed frontier full-CSV parity failed; stop before GAR/public checks")
    _cli_predict(STAGED, CSV)
    for path in (CONTROL, CSV):
        subprocess.run([sys.executable, "evaluation/validate_submission.py", str(path)], cwd=ROOT, check=True)
    public_baseline = ROOT / "experiments/results/v2_spell_frontier_v1_public_result.json"
    write_once(SEAL, {"study": STUDY, "candidate_sha256": sha(CSV), "control_sha256": sha(CONTROL),
        "historical_control_sha256": EXPECTED_FRONTIER_CSV, "control_parity_passed": True,
        "manifest_sha256": sha(STAGED), "input_sha256": sha(ROOT / "data/challenge/inputs.npz"),
        "control_manifest_sha256": sha(CONTROL_MANIFEST),
        "models_sha256": manifest["models_sha256"], "training_seal_sha256": sha(TRAIN_RECEIPT),
        "saved_public_baseline_sha256": sha(public_baseline),
        "source_sha256": sources(), "reference_access": False, "organizer_scored": False})


def check_seal():
    manifest = check_manifest()
    seal = read(SEAL)
    if seal["manifest_sha256"] != sha(STAGED) or seal["candidate_sha256"] != sha(CSV):
        raise ValueError("Sealed candidate or manifest changed")
    if seal["control_sha256"] != sha(CONTROL) or seal["control_sha256"] != EXPECTED_FRONTIER_CSV:
        raise ValueError("Historical full-CSV control parity is not exact")
    if seal["input_sha256"] != sha(ROOT / "data/challenge/inputs.npz"):
        raise ValueError("Sealed prediction input changed")
    if seal["control_manifest_sha256"] != sha(CONTROL_MANIFEST):
        raise ValueError("Sealed frontier control manifest changed")
    if seal["source_sha256"] != sources() or seal["models_sha256"] != manifest["models_sha256"]:
        raise ValueError("Sealed source or model set changed")
    baseline = ROOT / "experiments/results/v2_spell_frontier_v1_public_result.json"
    if seal["saved_public_baseline_sha256"] != sha(baseline):
        raise ValueError("Saved public baseline changed after prediction sealing")
    return seal


def public():
    seal = check_seal()
    if GATE.exists() or PUBLIC.exists():
        raise FileExistsError("The exclusive reconstructed public gate is already claimed")
    baseline_path = ROOT / "experiments/results/v2_spell_frontier_v1_public_result.json"
    if seal["saved_public_baseline_sha256"] != sha(baseline_path):
        raise ValueError("Saved public baseline changed")
    reference = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    write_once(GATE, {"seal_sha256": sha(SEAL), "claimed_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": "User-authorized one fixed reconstructed-candidate public diagnostic on 2026-11-11"})
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    result = pm.evaluate(pm.label_codes(pm.read_label_csv(reference)), pm.label_codes(pm.read_label_csv(CSV)), bootstrap_n=1000)
    if (result["n_frames"], result["n_graded"], result["n_gt_spells"]) != (199319, 7500, 17):
        raise ValueError("Unexpected public grading population; gate remains closed")
    candidate = _score_summary(result)
    baseline = read(baseline_path)["candidate"]
    write_once(PUBLIC, {"study": STUDY, "candidate": candidate, "saved_frontier": baseline,
        "delta": {k: candidate[k] - baseline[k] for k in ("spell_f1_mean", "macro_f1", "d_f1")},
        "gate_sha256": sha(GATE), "seal_sha256": sha(SEAL), "selection_or_tuning": False})


@contextmanager
def legacy_gar_context():
    """Redirect the frozen GAR runner to this package while restoring globals."""
    updates = {
        legacy_delivery: {"STUDY": STUDY, "PACKAGE": PACKAGE, "MODEL": MODEL, "OUT": OUT,
            "STAGED": STAGED, "PARENT": PARENT, "ACTIVE": ACTIVE, "POLICY": POLICY,
            "CSV": CSV, "SEAL": SEAL, "GATE": GATE, "PUBLIC": PUBLIC, "GAR": GAR,
            "RECEIPT": RECEIPT, "sources": sources, "check_manifest": check_manifest},
        legacy_gar: {"OUT": OUT / "gar", "RECORDS": OUT / "gar/records", "DECL": GAR_DECL},
    }
    saved = {module: {name: getattr(module, name) for name in values} for module, values in updates.items()}
    try:
        for module, values in updates.items():
            for name, value in values.items():
                setattr(module, name, value)
        yield
    finally:
        for module, values in saved.items():
            for name, value in values.items():
                setattr(module, name, value)


def prepare_gar():
    check_seal()
    with legacy_gar_context():
        legacy_gar.prepare()


def run_gar():
    check_seal()
    with legacy_gar_context():
        legacy_gar.run()


def report_gar():
    check_seal()
    with legacy_gar_context():
        legacy_gar.report()


def activate():
    manifest = check_manifest()
    seal = check_seal()
    if not GAR.is_file() or not PUBLIC.is_file():
        raise FileNotFoundError("Finish GAR TEST and the exclusive public diagnostic before promotion")
    gar, diagnostic = read(GAR), read(PUBLIC)
    if not ARCHIVE_VERIFY.is_file():
        raise FileNotFoundError("Privately archived package and fresh restore verification are required before promotion")
    archive = read(ARCHIVE_VERIFY)
    if (gar.get("manifest_sha256") != sha(STAGED) or
            gar.get("independent_prediction_replay_passed") is not True or
            gar.get("organizer_reference_access") is not False or
            diagnostic.get("seal_sha256") != sha(SEAL) or
            diagnostic.get("gate_sha256") != sha(GATE) or
            gar.get("spell_regression") is not False or
            diagnostic.get("delta", {}).get("spell_f1_mean", -1) < 0 or
            seal.get("control_parity_passed") is not True or
            seal.get("control_sha256") != EXPECTED_FRONTIER_CSV or
            seal.get("input_sha256") != sha(ROOT / "data/challenge/inputs.npz") or
            archive.get("manifest_sha256") != sha(STAGED) or
            archive.get("models_sha256") != manifest["models_sha256"] or
            archive.get("control_csv_sha256") != seal["control_sha256"] or
            archive.get("input_sha256") != seal["input_sha256"] or
            archive.get("remote_restore_verified") is not True):
        raise ValueError("Regression reports do not bind the sealed, parity-verified candidate")
    if ACTIVE.read_bytes() != PARENT.read_bytes():
        raise ValueError("Active baseline changed")
    # Preserve the provisional state byte-for-byte before installing the final version.
    archive_manifest = ROOT / "experiments/baselines" / f"{PARENT.stem}.provisional-before-reconstructed.json"
    archive_policy = ROOT / "experiments/baselines/model_decision_policy_provisional_before_reconstructed.json"
    if not archive_manifest.exists():
        with archive_manifest.open("xb") as stream:
            stream.write(ACTIVE.read_bytes())
    elif archive_manifest.read_bytes() != ACTIVE.read_bytes():
        raise ValueError("Provisional manifest archive differs")
    if not archive_policy.exists():
        with archive_policy.open("xb") as stream:
            stream.write(POLICY.read_bytes())
    elif archive_policy.read_bytes() != POLICY.read_bytes():
        raise ValueError("Provisional policy archive differs")
    manifest.update(status="User-authorized reconstructed provisional working baseline; independent continuous-gold confirmation pending",
        promoted_utc=datetime.now(timezone.utc).isoformat(),
        architecture=dict(manifest["architecture"], reconstructed_frontier_control_csv_parity=True,
            verified_control_csv_sha256=seal["control_sha256"], verified_control_input_sha256=seal["input_sha256"]),
        evidence=dict(manifest["evidence"], reconstructed_frontier_control_csv_parity=True,
            verified_control_csv_sha256=seal["control_sha256"], verified_control_input_sha256=seal["input_sha256"],
            original_checkpoint_byte_identity=False, organizer_public_diagnostic_performed=True),
        regression_checks={"gar_test10": sha(GAR), "organizer_public": sha(PUBLIC), "prediction_seal": sha(SEAL)})
    manifest["promotion"].update(type="explicit_user_authorized_reconstruction_promotion",
        deployment_verified=True, reconstruction_prediction_parity=True,
        regression_reports_verified=True, organizer_rescoring_performed=True,
        model_checkpoints_unchanged=False,
        reason="The original pooled CE checkpoint bytes were unavailable. The reconstructed fixed recipe matched the historical frontier CSV exactly, then passed fixed Fable GAR/public checks; CE checkpoint artifact identity remains reconstructed.")
    manifest["archive_verification"] = {"receipt": str(ARCHIVE_VERIFY.relative_to(ROOT)),
        "receipt_sha256": sha(ARCHIVE_VERIFY), "verified_staged_manifest_sha256": sha(STAGED),
        "remote_restore_verified": True, "models_sha256": manifest["models_sha256"],
        "control_csv_sha256": seal["control_sha256"], "input_sha256": seal["input_sha256"]}
    versioned = ROOT / "experiments/baselines" / f"{PACKAGE}.json"
    write_once(versioned, manifest)
    policy = read(POLICY)
    policy.update(version=13, confirmed_utc_date="2026-11-11")
    policy["working_candidate"].update(model=MODEL, family="Reconstructed pooled spell frontier with frozen ball-exit D correction",
        status="User-authorized reconstructed provisional working baseline; independent continuous-gold confirmation pending",
        baseline_manifest=str(ACTIVE.relative_to(ROOT)), baseline_manifest_sha256=sha(versioned),
        promotion_record=str(RECEIPT.relative_to(ROOT)), previous_policy=str(archive_policy.relative_to(ROOT)),
        reason="Pooled stage-1024 CE weights reconstructed under the predeclared recipe; exact frontier CSV parity and fixed deployment checks passed.")
    policy["stress_guardrail"]["reference_recipe"] = MODEL
    policy["clean_improvement_screen"]["current_adoption_exception"] = (
        "Explicit user-authorized reconstruction promotion after exact full-CSV frontier parity and fixed GAR/public regression checks; historical failures remain preserved.")
    ACTIVE.write_bytes(versioned.read_bytes())
    POLICY.write_text(json.dumps(policy, indent=2) + "\n")
    write_once(RECEIPT, {"package": PACKAGE, "manifest_sha256": sha(ACTIVE), "policy_sha256": sha(POLICY),
        "candidate_csv_sha256": sha(CSV), "control_csv_sha256": sha(CONTROL), "gar_result_sha256": sha(GAR),
        "public_result_sha256": sha(PUBLIC), "training_seal_sha256": sha(TRAIN_RECEIPT),
        "staged_manifest_sha256": sha(STAGED), "final_versioned_manifest_sha256": sha(versioned),
        "archive_verified_staged_manifest_sha256": archive["manifest_sha256"],
        "verified_control_csv_sha256": seal["control_sha256"], "verified_control_input_sha256": seal["input_sha256"],
        "final_manifest_model_inventory_matches_archive": archive["models_sha256"] == manifest["models_sha256"],
        "archive_verification_sha256": sha(ARCHIVE_VERIFY),
        "provisional_manifest_archive_sha256": sha(archive_manifest), "provisional_policy_archive_sha256": sha(archive_policy),
        "user_authorized": True, "fitting": False, "reconstructed": True})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "predict", "public", "prepare-gar", "run-gar", "report-gar", "activate"))
    args = parser.parse_args()
    {"stage": stage, "predict": predict, "public": public, "prepare-gar": prepare_gar,
     "run-gar": run_gar, "report-gar": report_gar, "activate": activate}[args.action]()
