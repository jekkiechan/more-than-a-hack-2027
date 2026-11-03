"""Readout-only correction for the frozen HA boundary selected-seed diagnostic.

The frozen study passes a seed-keyed mapping to its pooling helper. This
adapter selects the requested seed then normalizes (phase, probability) pairs
for the unchanged frozen phase averaging function. Candidate metrics are read
verbatim from the already-persisted 500-row cache.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_calibration_ha_boundary as study

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ROWS = ROOT / "data/experiments/v2_calibration_ha_boundary_v1/candidate_rows_recovery.json"
RECEIPT = ROOT / "experiments/results/v2_calibration_ha_boundary_seed_mapping_adapter.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _actual_cache_smoke() -> dict:
    checks = []
    for provider, expected_phases in (("Sportec", 5), ("SkillCorner", 2)):
        provider, match, _ = next(row for row in study.AUDIT.MATCHES if row[0] == provider)
        view, reps = study.VIEWS[0]
        _, _, per_seed, phases = study._load_cell_probabilities(provider, match, view, reps[0])
        if phases != expected_phases:
            raise RuntimeError(f"Unexpected native phase count for {provider}: {phases}")
        for recipe in study.RECIPES:
            for seed in study.SEEDS:
                pairs = per_seed[recipe][seed]
                if len(pairs) != phases or any(len(pair) != 2 for pair in pairs):
                    raise RuntimeError("Actual cache did not return (phase, probability) pairs")
                normalized = [(phase, seed, probability) for phase, probability in pairs]
                pooled = study._phase_ensemble(normalized, phases, seed)
                if (pooled.shape[1] != 3 or not np.isfinite(pooled).all() or
                        not np.allclose(pooled.sum(axis=1), 1, atol=1e-5)):
                    raise RuntimeError("Actual-cache seed pooling failed normalization")
        checks.append({"provider": provider, "match": match, "native_phases": phases,
                       "recipes": list(study.RECIPES), "seeds": list(study.SEEDS),
                       "normalized": True, "scoring": False})
    return {"actual_cache_shape_checks": checks}


def freeze() -> dict:
    if RECEIPT.exists() or study.RESULT.exists():
        raise RuntimeError("Seed-mapping adapter receipt or study result already exists")
    declaration = json.loads(study.DECLARATION.read_text())
    preflight = json.loads(study.PREFLIGHT.read_text())
    rows = json.loads(CANDIDATE_ROWS.read_text())
    if preflight.get("preflight_passed") is not True:
        raise RuntimeError("Frozen preflight is not passed")
    if (rows.get("frozen_study_declaration_sha256") != sha(study.DECLARATION) or
            rows.get("frozen_preflight_sha256") != sha(study.PREFLIGHT) or
            rows.get("candidate_cell_count") != 25 or rows.get("candidate_score_count") != 500):
        raise RuntimeError("Persisted candidate rows do not match frozen inputs or exact score budget")
    if len(rows.get("cells", {})) != 25 or any(len(row) != 20 for row in rows["cells"].values()):
        raise RuntimeError("Persisted candidate score matrix is not exactly 25×20")
    receipt = {
        "study_id": study.STUDY,
        "status": "frozen_selected_seed_mapping_readout_recovery_before_readout_only",
        "reason": "frozen runner passed a seed-keyed mapping instead of one seed's phase-pair list to its phase reducer",
        "frozen_declaration_sha256": sha(study.DECLARATION),
        "frozen_preflight_sha256": sha(study.PREFLIGHT),
        "candidate_rows_path": str(CANDIDATE_ROWS.relative_to(ROOT)),
        "candidate_rows_sha256": sha(CANDIDATE_ROWS),
        "candidate_score_count": 500,
        "prior_tuple_adapter_receipt_sha256": sha(ROOT / "experiments/results/v2_calibration_ha_boundary_seed_adapter.json"),
        "parallel_recovery_receipt_sha256": sha(ROOT / "experiments/results/v2_calibration_ha_boundary_parallel_recovery.json"),
        "adapter_source_sha256": sha(Path(__file__)),
        "fix": "select members[requested_seed], map (phase, probability) to unchanged helper's (phase, seed, probability); preserve phase averaging",
        "candidate_scores_reused_verbatim": True,
        "grid_or_selection_changed": False,
        "fits": 0,
        "new_inference": False,
        "smoke_test": _actual_cache_smoke(),
    }
    RECEIPT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return receipt


def run() -> dict:
    receipt = json.loads(RECEIPT.read_text())
    if (receipt.get("adapter_source_sha256") != sha(Path(__file__)) or
            receipt.get("candidate_rows_sha256") != sha(CANDIDATE_ROWS) or
            receipt.get("frozen_declaration_sha256") != sha(study.DECLARATION)):
        raise RuntimeError("Frozen seed adapter or candidate rows changed")
    rows = json.loads(CANDIDATE_ROWS.read_text())
    cache = rows["cells"]
    original_job = study._score_job
    original_ensemble = study._phase_ensemble

    def cached_job(job):
        provider, match, view, rep = job
        key = f"{provider}__{match}__{view}__{rep}"
        return key, cache[key]

    def seed_aware_ensemble(members, phase_count, seed=None):
        if seed is None:
            return original_ensemble(members, phase_count, seed=None)
        if isinstance(members, dict):
            members = members[seed]
        if members and len(members[0]) == 2:
            members = [(phase, seed, probability) for phase, probability in members]
        return original_ensemble(members, phase_count, seed=seed)

    study._score_job = cached_job
    study._phase_ensemble = seed_aware_ensemble
    try:
        return study.run()
    finally:
        study._score_job = original_job
        study._phase_ensemble = original_ensemble


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.action]()
