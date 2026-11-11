"""Explicit provisional adoption, separate from exact deployment verification.

The user delegated the adoption decision after the missing weights were disclosed.
This changes the default recipe; it does not certify checkpoint availability,
full-match inference, or GAR/public regression results. Frozen delivery gates stay
unchanged. Once weights are restored, stage this exact manifest for those gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments import ball_exit_delivery as delivery
from experiments.ball_exit_followup import sha, write_once

ROOT = delivery.ROOT
VERSIONED = ROOT / "experiments/baselines" / f"{delivery.PACKAGE}.json"
OLD_POLICY = ROOT / "experiments/baselines/model_decision_policy_joint_gru_sn_dense_spell_frontier_ensemble_v1.json"
RECEIPT = ROOT / "experiments/results/ball_exit_adoption_v1.json"
REASON = (
    "User delegated the choice and requested promotion after the missing exact weights "
    "and unmeasured GAR/public regressions were disclosed. Adopt unchanged Fable snap_m0.0 "
    "as the provisional default: +.005071 v2 spell over the frontier. The complementary "
    "candidate adds only .000090 on exposed development and requires additional pooled "
    "models; choose the simpler package as an explicit deployment tradeoff. "
    "Exact deployment verification remains pending; no regression pass is implied."
)


def checkpoint_inventory(manifest):
    verified, missing = {}, {}
    for name, digest in manifest["models_sha256"].items():
        path = ROOT / manifest["model_directory"] / name
        if not path.is_file():
            missing[name] = digest
        elif sha(path) != digest:
            raise ValueError("Checkpoint hash mismatch: " + name)
        else:
            verified[name] = digest
    return {"verified": verified, "missing": missing}


def adopt():
    if delivery.ACTIVE.read_bytes() != delivery.PARENT.read_bytes():
        raise ValueError("Active predecessor changed; refuse to replace another adoption")
    for path in (VERSIONED, RECEIPT, delivery.STAGED, delivery.GATE, delivery.SEAL):
        if path.exists():
            raise FileExistsError(path)
    if OLD_POLICY.exists() and OLD_POLICY.read_bytes() != delivery.POLICY.read_bytes():
        raise ValueError("Archived predecessor policy differs")
    result_path = ROOT / "experiments/results/ball_exit_crossing_v1.json"
    verification_path = ROOT / "experiments/results/ball_exit_crossing_v1_verification.json"
    result, verification = delivery.read(result_path), delivery.read(verification_path)
    if (result["selected"] != "snap" or not verification["all_passed"] or
            verification["result_sha256"] != sha(result_path)):
        raise ValueError("Expected independently verified unchanged Fable result")
    parent = delivery.read(delivery.PARENT)
    inventory = checkpoint_inventory(parent)
    manifest = dict(parent)
    manifest.update(
        version=12, model=delivery.MODEL, package_version=delivery.PACKAGE,
        status="User-authorized provisional default; exact deployment and GAR/public checks pending",
        promoted_utc=datetime.now(timezone.utc).isoformat(),
        previous_manifest=str(delivery.PARENT.relative_to(ROOT)),
        previous_manifest_sha256=sha(delivery.PARENT),
        previous_policy=str(OLD_POLICY.relative_to(ROOT)), previous_policy_sha256=sha(delivery.POLICY),
        replaced_baseline=parent["model"],
        default_output=f"data/experiments/current_baseline/{delivery.MODEL}.csv",
        architecture=dict(parent["architecture"], kind=delivery.runtime.KIND,
                          ball_exit=delivery.runtime.CORRECTION,
                          D="Predecessor mask corrected by frozen snap_m0.0"),
        evidence={"report": "experiments/BALL_EXIT_PROMOTION.md",
                  "original_development": result["aggregate"]["snap"],
                  "source_result_sha256": sha(result_path),
                  "verification_sha256": sha(verification_path),
                  "historical_clean_screen_passes": True,
                  "independent_continuous_gold_confirmation": False},
        promotion={"type": "explicit_user_authorized_provisional_adoption",
                   "user_request": "promote the version you think is best", "reason": REASON,
                   "automatic_promotion": False, "fitting_during_promotion": False,
                   "model_checkpoints_unchanged": True, "organizer_rescoring_performed": False,
                   "public_used_for_selection": False, "deployment_verified": False},
        regression_checks={"gar_test10": "pending", "organizer_public": "pending"},
        source_sha256=delivery.sources())
    manifest["source_sha256"][str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    manifest.pop("organizer_public_diagnostic", None)
    policy = delivery.read(delivery.POLICY)
    policy.update(version=12, confirmed_utc_date="2026-11-11")
    policy["working_candidate"].update(
        model=delivery.MODEL, family="Spell frontier with physical ball-exit D starts",
        status=manifest["status"], reason=REASON,
        promotion_record=str(RECEIPT.relative_to(ROOT)), previous_policy=str(OLD_POLICY.relative_to(ROOT)))
    policy["stress_guardrail"]["reference_recipe"] = delivery.MODEL
    policy["clean_improvement_screen"].update(current_screen_passes=True,
        current_adoption_exception="Clean +.011371 passes; provisional adoption precedes exact deployment/GAR/public checks. Historical failures are preserved.")
    if not OLD_POLICY.exists():
        OLD_POLICY.write_bytes(delivery.POLICY.read_bytes())
    write_once(VERSIONED, manifest)
    policy["working_candidate"]["baseline_manifest_sha256"] = sha(VERSIONED)
    delivery.ACTIVE.write_bytes(VERSIONED.read_bytes())
    delivery.POLICY.write_text(json.dumps(policy, indent=2) + "\n")
    write_once(RECEIPT, {"package": delivery.PACKAGE, "status": manifest["status"],
        "user_request": manifest["promotion"]["user_request"], "reason": REASON,
        "manifest_sha256": sha(delivery.ACTIVE), "policy_sha256": sha(delivery.POLICY),
        "previous_manifest_sha256": sha(delivery.PARENT), "previous_policy_sha256": sha(OLD_POLICY),
        "checkpoint_inventory": inventory, "deployment_verified": False,
        "gar_scored": False, "public_scored": False, "fitting": False})
    return manifest


def stage():
    """Prepare the adopted snapshot for existing exact-package diagnostic gates."""
    receipt = delivery.read(RECEIPT)
    if (sha(VERSIONED) != receipt["manifest_sha256"] or
            delivery.ACTIVE.read_bytes() != VERSIONED.read_bytes()):
        raise ValueError("Adopted snapshot changed")
    manifest = delivery.read(VERSIONED)
    delivery.runtime.frontier_manifest(manifest)
    for rel, digest in manifest["source_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("Adopted source changed: " + rel)
    delivery.current_baseline.load_models(manifest_path=VERSIONED)
    write_once(delivery.STAGED, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("adopt", "stage"))
    args = parser.parse_args()
    manifest = globals()[args.action]()
    print(manifest["package_version"], manifest["status"])
