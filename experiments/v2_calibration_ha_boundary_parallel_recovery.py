"""Fixed-process recovery orchestration for the frozen HA boundary study.

Scores the exact frozen 25 cells once with three workers, then injects those
same per-cell results into the frozen reducer and runs only selected-seed
readouts serially with the separately frozen tuple-shape adapter.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_calibration_ha_boundary as study
from experiments import v2_calibration_ha_boundary_seed_adapter as adapter

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "experiments/results/v2_calibration_ha_boundary_parallel_recovery.json"
CANDIDATE_ROWS = ROOT / "data/experiments/v2_calibration_ha_boundary_v1/candidate_rows_recovery.json"
WORKERS = 3


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jobs():
    return [(p, m, view, rep) for p, m, _ in study.AUDIT.MATCHES
            for view, reps in study.VIEWS for rep in reps]


def _smoke_seed_tuple_shape() -> dict:
    cases = []
    for provider, desired_phases in (("Sportec", 5), ("SkillCorner", 2)):
        provider, match, _ = next(row for row in study.AUDIT.MATCHES if row[0] == provider)
        view, reps = study.VIEWS[0]
        _, _, seed_outputs, phases = study._load_cell_probabilities(provider, match, view, reps[0])
        if phases != desired_phases:
            raise RuntimeError(f"Expected {desired_phases} phases for {provider}, found {phases}")
        for recipe in study.RECIPES:
            for seed in study.SEEDS:
                pairs = seed_outputs[recipe][seed]
                assert len(pairs) == phases and all(len(pair) == 2 for pair in pairs)
                normalized = [(phase, seed, probability) for phase, probability in pairs]
                pooled = study._phase_ensemble(normalized, phases, seed=seed)
                assert pooled.ndim == 2 and pooled.shape[1] == 3
                assert np.isfinite(pooled).all() and np.allclose(pooled.sum(axis=1), 1, atol=1e-5)
        cases.append({"provider": provider, "match": match, "view": view, "rep": reps[0],
                      "native_phases": phases, "recipes": list(study.RECIPES),
                      "seeds": list(study.SEEDS), "normalization_passed": True,
                      "metrics_or_scores_called": False})
    return {"cases": cases}


def freeze() -> dict:
    if RECEIPT.exists() or study.RESULT.exists():
        raise RuntimeError("Recovery receipt/result already exists")
    if CANDIDATE_ROWS.exists():
        raise RuntimeError("Recovery candidate-row cache already exists")
    if not study.DECLARATION.is_file() or not study.PREFLIGHT.is_file():
        raise RuntimeError("Frozen study declaration and preflight are required")
    if not json.loads(study.PREFLIGHT.read_text()).get("preflight_passed"):
        raise RuntimeError("Frozen controls did not pass")
    smoke = _smoke_seed_tuple_shape()
    jobs = _jobs()
    if len(jobs) != 25:
        raise RuntimeError(f"Expected exact 25-cell grid, found {len(jobs)}")
    receipt = {
        "study_id": study.STUDY,
        "status": "frozen_fixed_grid_parallel_execution_recovery_before_replay_scores",
        "reason": "first exact grid scoring completed but result write was blocked by a two-tuple selected-seed adapter error; this does not add configurations or change selection",
        "frozen_study_declaration_sha256": sha(study.DECLARATION),
        "frozen_preflight_sha256": sha(study.PREFLIGHT),
        "tuple_adapter_source_sha256": sha(Path(adapter.__file__)),
        "tuple_adapter_receipt_sha256": sha(adapter.RECEIPT),
        "orchestration_source_sha256": sha(Path(__file__)),
        "worker_count": WORKERS,
        "candidate_rows_path": str(CANDIDATE_ROWS.relative_to(ROOT)),
        "cell_jobs": [list(job) for job in jobs],
        "candidate_configurations": len(study.ARMS),
        "candidate_scores": len(jobs) * len(study.ARMS),
        "re_inference": False, "fits": 0,
        "smoke_test": smoke,
        "candidate_grid_or_selection_changes": False,
        "status_note": "repeat exact same 500 fixed candidate scores after discarded process-local scores; then reducer uses persisted-in-memory parallel results without a second candidate scoring pass",
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt


def run() -> dict:
    receipt = json.loads(RECEIPT.read_text())
    if receipt.get("orchestration_source_sha256") != sha(Path(__file__)):
        raise RuntimeError("Recovery orchestration source changed")
    if receipt.get("frozen_study_declaration_sha256") != sha(study.DECLARATION):
        raise RuntimeError("Frozen study declaration changed")
    if receipt.get("tuple_adapter_source_sha256") != sha(Path(adapter.__file__)):
        raise RuntimeError("Tuple adapter source changed")
    jobs = _jobs()
    if [list(job) for job in jobs] != receipt["cell_jobs"]:
        raise RuntimeError("Cell job inventory changed")

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(study._score_job, jobs))
    cache = dict(results)
    if len(cache) != 25 or any(len(scores) != 20 for scores in cache.values()):
        raise RuntimeError("Parallel scoring result inventory is not exactly 25×20")
    rows_receipt = {
        "study_id": study.STUDY,
        "frozen_study_declaration_sha256": sha(study.DECLARATION),
        "frozen_preflight_sha256": sha(study.PREFLIGHT),
        "parallel_recovery_receipt_sha256": sha(RECEIPT),
        "candidate_cell_count": len(cache),
        "candidate_arm_count_per_cell": 20,
        "candidate_score_count": sum(len(scores) for scores in cache.values()),
        "cells": cache,
    }
    CANDIDATE_ROWS.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATE_ROWS.open("x") as stream:
        json.dump(rows_receipt, stream, indent=1, sort_keys=True)
        stream.write("\n")
    original_score_job = study._score_job
    original_ensemble = study._phase_ensemble
    adapter_original_ensemble = adapter.study._phase_ensemble

    def cached_score_job(job):
        provider, match, view, rep = job
        key = f"{provider}__{match}__{view}__{rep}"
        return key, cache[key]

    def normalized_phase_ensemble(members, phase_count, seed=None):
        if seed is None:
            return original_ensemble(members, phase_count, seed=None)
        if members and len(members[0]) == 2:
            members = [(phase, seed, probability) for phase, probability in members]
        return original_ensemble(members, phase_count, seed=seed)

    study._score_job = cached_score_job
    study._phase_ensemble = normalized_phase_ensemble
    try:
        result = study.run()
    finally:
        study._score_job = original_score_job
        study._phase_ensemble = original_ensemble
        adapter.study._phase_ensemble = adapter_original_ensemble
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
