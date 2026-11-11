"""Stage, seal, diagnose and activate the user-authorized frozen ball-exit package.

Missing or changed pooled checkpoint bytes block staging. Historical public
gates are never reused; the new diagnostic has its own exclusive claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from experiments import ball_exit_runtime as runtime, current_baseline
from experiments.ball_exit_followup import sha, write_once

ROOT = Path(__file__).resolve().parents[1]
STUDY = "ball_exit_promotion_v1"
PACKAGE = "joint_gru_sn_dense_spell_frontier_ball_exit_ensemble_v1"
MODEL = PACKAGE.removesuffix("_v1")
OUT = ROOT / "data/experiments" / STUDY
STAGED = OUT / "baseline_manifest.json"
PARENT = ROOT / "experiments/baselines/joint_gru_sn_dense_spell_frontier_ensemble_v1.json"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
CSV = OUT / "candidate.csv"
SEAL = OUT / "prediction_seal.json"
GATE = OUT / "public_gate_claimed.json"
PUBLIC = ROOT / f"experiments/results/{STUDY}_public.json"
GAR = ROOT / f"experiments/results/{STUDY}_gar.json"
RECEIPT = ROOT / f"experiments/results/{STUDY}.json"


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    names = ("ball_exit_delivery.py", "ball_exit_gar.py", "ball_exit_runtime.py", "ball_exit_dmask_v1.py",
             "spell_frontier_baseline.py", "current_baseline.py", "v2_spell_balanced_eval.py",
             "v2_spell_balanced_public.py", "v2_iterative_runtime.py", "supervised.py",
             "feed_benchmark_audit_v1.py", "feed_spell_push.py", "sn_dense_train.py",
             "soccernet_extension.py", "tracking_corruption.py")
    return {f"experiments/{name}": sha(ROOT / "experiments" / name) for name in names}


def stage():
    parent = read(PARENT)
    if ACTIVE.read_bytes() != PARENT.read_bytes():
        raise ValueError("Active predecessor changed; do not silently replace a newer baseline")
    current_baseline.load_models(manifest_path=PARENT)
    result = read(ROOT / "experiments/results/ball_exit_crossing_v1.json")
    verification = read(ROOT / "experiments/results/ball_exit_crossing_v1_verification.json")
    if result["selected"] != "snap" or not verification["all_passed"]:
        raise ValueError("Expected verified retention of unchanged snap_m0.0")
    if verification["result_sha256"] != sha(ROOT / "experiments/results/ball_exit_crossing_v1.json"):
        raise ValueError("Development readout binding changed")
    manifest = dict(parent)
    manifest.update(version=12, model=MODEL, package_version=PACKAGE,
        status="User-authorized ball-exit candidate staged; activation requires exact-package regression checks",
        previous_manifest=str(PARENT.relative_to(ROOT)), previous_manifest_sha256=sha(PARENT),
        replaced_baseline=parent["model"], default_output=str(CSV.relative_to(ROOT)),
        architecture=dict(parent["architecture"], kind=runtime.KIND, ball_exit=runtime.CORRECTION,
                          D="Predecessor mask corrected by frozen snap_m0.0"),
        evidence={"report": "experiments/BALL_EXIT_PROMOTION.md", "original_development": result["aggregate"]["snap"],
                  "source_result_sha256": sha(ROOT / "experiments/results/ball_exit_crossing_v1.json"),
                  "independent_continuous_gold_confirmation": False},
        promotion={"user_authorized": True, "date": "2026-11-11", "fitting": False,
                   "public_used_for_selection": False, "automatic_promotion": False},
        source_sha256=sources())
    manifest.pop("promoted_utc", None)
    manifest.pop("organizer_public_diagnostic", None)
    manifest["training"] = dict(parent["training"], fitting_during_promotion=False)
    write_once(STAGED, manifest)
    return manifest


def check_manifest():
    manifest = read(STAGED)
    runtime.frontier_manifest(manifest)
    if manifest["previous_manifest_sha256"] != sha(PARENT):
        raise ValueError("Predecessor manifest changed")
    for rel, digest in manifest["source_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("Staged source changed: " + rel)
    current_baseline.load_models(manifest_path=STAGED)
    return manifest


def predict():
    manifest = check_manifest()
    if SEAL.exists() or GATE.exists() or CSV.exists():
        raise FileExistsError("Preserve prior prediction/gate")
    control = OUT / "frontier_replay.csv"
    for path, output in ((PARENT, control), (STAGED, CSV)):
        subprocess.run([sys.executable, "-m", "experiments.current_baseline", "predict",
                        "--manifest", str(path), "--output", str(output)], cwd=ROOT, check=True)
    if sha(control) != read(PARENT)["evidence"]["prediction_csv_sha256"]:
        raise ValueError("Exact frontier full-CSV replay failed before public access")
    write_once(SEAL, {"candidate_sha256": sha(CSV), "control_sha256": sha(control),
        "manifest_sha256": sha(STAGED), "input_sha256": sha(ROOT / "data/challenge/inputs.npz"),
        "models_sha256": manifest["models_sha256"], "source_sha256": sources(),
        "development_result_sha256": manifest["evidence"]["source_result_sha256"],
        "saved_public_baseline_sha256": sha(ROOT / "experiments/results/v2_spell_frontier_v1_public_result.json"),
        "reference_access": False})


def check_seal():
    manifest, seal = check_manifest(), read(SEAL)
    if seal["manifest_sha256"] != sha(STAGED) or seal["candidate_sha256"] != sha(CSV):
        raise ValueError("Sealed candidate changed")
    if seal["input_sha256"] != sha(ROOT / "data/challenge/inputs.npz") or seal["source_sha256"] != sources():
        raise ValueError("Sealed input/source changed")
    if seal["models_sha256"] != manifest["models_sha256"]:
        raise ValueError("Sealed models changed")
    return seal


def public():
    seal = check_seal()
    baseline_path = ROOT / "experiments/results/v2_spell_frontier_v1_public_result.json"
    if seal["saved_public_baseline_sha256"] != sha(baseline_path):
        raise ValueError("Saved public baseline changed")
    reference = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    if not reference.is_file():
        raise FileNotFoundError(reference)
    if PUBLIC.exists():
        raise FileExistsError("Public result already exists")
    write_once(GATE, {"seal_sha256": sha(SEAL), "claimed_utc": datetime.now(timezone.utc).isoformat(),
                      "authorization": "User explicitly requested public spell regression diagnostic on 2026-11-11"})
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    result = pm.evaluate(pm.label_codes(pm.read_label_csv(reference)),
                         pm.label_codes(pm.read_label_csv(CSV)), bootstrap_n=1000)
    if (result["n_frames"], result["n_graded"], result["n_gt_spells"]) != (199319, 7500, 17):
        raise ValueError("Unexpected public grading population; gate remains closed")
    candidate, baseline = _score_summary(result), read(baseline_path)["candidate"]
    write_once(PUBLIC, {"study": STUDY, "candidate": candidate, "saved_frontier": baseline,
        "delta": {k: candidate[k] - baseline[k] for k in ("spell_f1_mean", "macro_f1", "d_f1")},
        "gate_sha256": sha(GATE), "seal_sha256": sha(SEAL), "selection_or_tuning": False})


def activate():
    manifest = check_manifest()
    check_seal()
    if not GAR.is_file() or not PUBLIC.is_file():
        raise FileNotFoundError("Finish and review GAR TEST/public diagnostics before activation")
    gar, diagnostic = read(GAR), read(PUBLIC)
    if (gar.get("manifest_sha256") != sha(STAGED) or
            gar.get("independent_prediction_replay_passed") is not True or
            diagnostic.get("seal_sha256") != sha(SEAL) or
            diagnostic.get("gate_sha256") != sha(GATE)):
        raise ValueError("Regression reports do not bind this sealed package")
    if ACTIVE.read_bytes() != PARENT.read_bytes():
        raise ValueError("Active baseline changed")
    # A failed diagnostic requires a disclosed decision; it never changes the rule.
    if gar.get("spell_regression") or diagnostic["delta"]["spell_f1_mean"] < 0:
        raise ValueError("Regression observed; report it before activating this candidate")
    old_policy = ROOT / "experiments/baselines/model_decision_policy_joint_gru_sn_dense_spell_frontier_ensemble_v1.json"
    if not old_policy.exists():
        old_policy.write_bytes(POLICY.read_bytes())
    elif old_policy.read_bytes() != POLICY.read_bytes():
        raise ValueError("Archived predecessor policy differs")
    manifest.update(status="User-authorized provisional working baseline; independent gold confirmation pending",
                    promoted_utc=datetime.now(timezone.utc).isoformat())
    manifest["regression_checks"] = {"gar": sha(GAR), "public": sha(PUBLIC)}
    versioned = ROOT / "experiments/baselines" / f"{PACKAGE}.json"
    write_once(versioned, manifest)
    policy = read(POLICY)
    policy.update(version=12, confirmed_utc_date="2026-11-11")
    policy["working_candidate"].update(model=MODEL, family="Spell frontier with physical ball-exit D starts",
        baseline_manifest_sha256=sha(versioned), promotion_record=str(RECEIPT.relative_to(ROOT)),
        previous_policy=str(old_policy.relative_to(ROOT)), reason="Explicit user promotion of unchanged Fable snap_m0.0 after regression checks")
    policy["stress_guardrail"]["reference_recipe"] = MODEL
    policy["clean_improvement_screen"].update(current_screen_passes=True,
        current_adoption_exception="No new exception: snap improves matched frontier clean spell by .011371. Historical failures remain preserved.")
    ACTIVE.write_bytes(versioned.read_bytes())
    POLICY.write_text(json.dumps(policy, indent=2) + "\n")
    write_once(RECEIPT, {"package": PACKAGE, "manifest_sha256": sha(ACTIVE), "policy_sha256": sha(POLICY),
        "csv_sha256": sha(CSV), "gar_result_sha256": sha(GAR), "public_result_sha256": sha(PUBLIC),
        "user_authorized": True, "fitting": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stage", "predict", "public", "activate"))
    args = parser.parse_args()
    globals()[args.action]()
