"""One-purpose execution adapter for the frozen HA boundary seed readout.

The frozen runner stores per-seed members as (phase, probability) but its
shared pooling helper expects (phase, seed, probability). This adapter only
normalizes that in-memory tuple shape for the already-selected seed diagnostic.
It does not change arms, cached probabilities, pooling, metrics, or selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments import v2_calibration_ha_boundary as study

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "experiments/results/v2_calibration_ha_boundary_seed_adapter.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_adapter() -> dict:
    if RECEIPT.exists():
        raise RuntimeError("Seed adapter receipt already exists")
    if not study.DECLARATION.is_file() or not study.PREFLIGHT.is_file():
        raise RuntimeError("Frozen boundary declaration and controls-only preflight are required")
    preflight = json.loads(study.PREFLIGHT.read_text())
    if preflight.get("preflight_passed") is not True:
        raise RuntimeError("Boundary preflight did not pass")
    receipt = {
        "study_id": study.STUDY,
        "purpose": "repair only per-seed tuple arity in selected-arm diagnostics after fixed grid completed but before result was written",
        "frozen_declaration_sha256": sha(study.DECLARATION),
        "preflight_sha256": sha(study.PREFLIGHT),
        "adapter_source_sha256": sha(Path(__file__)),
        "change": "map each (phase, probability) pair to (phase, requested_seed, probability) before calling the frozen phase pooling helper",
        "arms_or_parameters_changed": False,
        "candidate_grid_or_selection_changed": False,
        "re_inference": False,
        "fits": 0,
        "preflight_controls": preflight["checks"],
        "status": "frozen_before_replay_of_same_fixed_grid_and_seed_readout",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt


def run() -> dict:
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("adapter_source_sha256") != sha(Path(__file__)):
        raise RuntimeError("Seed adapter source changed after freeze")
    if receipt.get("frozen_declaration_sha256") != sha(study.DECLARATION):
        raise RuntimeError("Boundary declaration changed after adapter freeze")
    original = study._phase_ensemble

    def normalized_phase_ensemble(members, phase_count, seed=None):
        if seed is None:
            return original(members, phase_count, seed=None)
        if members and len(members[0]) == 2:
            members = [(phase, seed, probability) for phase, probability in members]
        return original(members, phase_count, seed=seed)

    study._phase_ensemble = normalized_phase_ensemble
    try:
        return study.run()
    finally:
        study._phase_ensemble = original


if __name__ == "__main__":
    import argparse
    action = argparse.ArgumentParser().add_subparsers(dest="action", required=True)
    action.add_parser("freeze")
    action.add_parser("run")
    selected = __import__("sys").argv[1]
    {"freeze": freeze_adapter, "run": run}[selected]()
