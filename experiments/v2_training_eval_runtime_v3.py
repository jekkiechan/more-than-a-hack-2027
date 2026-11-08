"""Mechanical typed-cache reader correction for the frozen v2 eval runner.

The frozen cache reader stringifies integer identity fields before comparison.
This adapter preserves scalar types, validates the complete existing cache set,
forbids cache misses, and delegates all three arms to the unchanged evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_training_eval as frozen
from experiments import v2_training_eval_runtime_v2 as runtime_v2


ROOT = frozen.ROOT
STUDY = "v2_training_eval_runtime_v3"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
PARTIAL_FAILURE = ROOT / "data/experiments/v2_training_eval/partial_run_failure_receipt.json"
PREFILL = ROOT / "data/experiments/v2_training_eval/skillcorner_prefill_receipt.json"
_VERIFIED_DECLARATION = None
_VERIFIED_SEAL = None


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(view, phase_record, provider, match, view_name, rep, seed, phase,
              declaration_sha, model_sha):
    teacher_path = frozen.upstream.frozen._teacher_path(provider, seed)
    calibrated_path = frozen.upstream.frozen._model_path("adop", provider, seed)
    return {
        "declaration_sha256": declaration_sha,
        "model_sha256": model_sha,
        "teacher_sha256": _sha(teacher_path),
        "calibrated_model_sha256": _sha(calibrated_path),
        "input_sha256": frozen.upstream.frozen._input_sha(view),
        "record_sha256": frozen._record_sha(phase_record),
        "feature_names_sha256": frozen._array_sha(
            np.asarray(list(phase_record["names"]), dtype="U")),
        "provider": provider,
        "match": match,
        "view": view_name,
        "replicate": int(rep),
        "seed": int(seed),
        "phase": int(phase),
    }


def _validate_cache(path, expected, archived_emission_facts):
    with np.load(path, allow_pickle=False) as saved:
        # `.item()` deliberately preserves integer versus string identity types.
        actual = {key: saved[key].item() for key in expected}
        if actual != expected:
            raise RuntimeError(f"Typed candidate emission identity mismatch: {path}")
        new_full, calibrated = saved["new_full"], saved["calibrated"]
        if (str(saved["new_full_sha256"].item()) != frozen._array_sha(new_full)
                or str(saved["calibrated_sha256"].item()) != frozen._array_sha(calibrated)):
            raise RuntimeError(f"Candidate probability-array hash mismatch: {path}")
    provider, match, view_name = expected["provider"], expected["match"], expected["view"]
    rep, seed, phase = expected["replicate"], expected["seed"], expected["phase"]
    archived_key = frozen._archived_cache_key(provider, match, view_name, rep, seed, phase)
    archived_fact = archived_emission_facts.get(archived_key)
    archived_path = frozen.upstream.frozen._cache_path(
        provider, match, view_name, rep, "adop", phase, seed)
    if (archived_fact is None or not archived_path.is_file()
            or _sha(archived_path) != archived_fact["sha256"]):
        raise RuntimeError(f"Sealed calibrated emission is missing or changed: {archived_path}")
    with np.load(archived_path, allow_pickle=False) as saved:
        archived = saved["probabilities"]
    if not np.array_equal(calibrated, archived):
        raise RuntimeError(f"Candidate cache calibrated member differs from sealed emission: {path}")
    return new_full, calibrated


def _expected_emission_inventory(seal):
    expected, identities = {}, {}
    declaration_sha = _sha(frozen.DECLARATION)
    for provider, match, _ in frozen.upstream.frozen.audit.MATCHES:
        for view_name, reps in frozen.VIEWS:
            for rep in reps:
                view, _, _, phases, _ = frozen.upstream.frozen._prepare_cell(
                    provider, match, view_name, rep)
                for phase, phase_record in enumerate(phases):
                    for seed in frozen.SEEDS:
                        fold = frozen.upstream.frozen.audit.FOLD[provider]
                        model_path = runtime_v2.corrected_fit_path(fold, seed)
                        model_sha = _sha(model_path)
                        identity = _identity(view, phase_record, provider, match, view_name,
                                             rep, seed, phase, declaration_sha, model_sha)
                        path = frozen._emission_path(provider, match, view_name, rep, seed, phase)
                        key = str(path.relative_to(ROOT))
                        expected[key] = _sha(path) if path.is_file() else None
                        identities[key] = (path, identity)
    return expected, identities


def declare() -> dict:
    if DECLARATION.exists():
        raise RuntimeError("Runtime v3 correction declaration already exists")
    if frozen.RESULT.exists():
        raise RuntimeError("Candidate result exists; cache correction is too late")
    base, seal, preflight = frozen._require_seal()
    runtime_v2._verify()
    if not PARTIAL_FAILURE.is_file() or not PREFILL.is_file():
        raise FileNotFoundError("Preserve partial-failure and SkillCorner-prefill receipts first")
    if json.loads(PARTIAL_FAILURE.read_text())["partial_effects"]["cell_scores_saved"]:
        raise RuntimeError("Partial cell scores exist; this replay protocol assumes none were saved")
    expected, identities = _expected_emission_inventory(seal)
    actual_paths = {str(path.relative_to(ROOT)) for path in (frozen.OUT / "emissions").glob("*.npz")}
    if actual_paths != set(expected) or len(expected) != 160:
        raise RuntimeError(f"Expected exactly 160 candidate emissions; found expected={len(expected)} actual={len(actual_paths)}")
    for key, (path, identity) in identities.items():
        if expected[key] is None:
            raise FileNotFoundError(f"Candidate cache missing; new inference is forbidden: {path}")
        _validate_cache(path, identity, seal["archived_calibrated_emissions"])
    prefill = json.loads(PREFILL.read_text())
    if prefill.get("candidate_scoring") is not False or prefill.get("emission_file_count") != 60:
        raise RuntimeError("SkillCorner prefill receipt does not match the fixed inference-only scope")
    model_hashes = {key: value["sha256"] for key, value in seal["models"].items()}
    value = {
        "study_id": STUDY,
        "status": "mechanical_cache_reader_correction_frozen_before_complete_fixed_score_replay",
        "correction": "Compare NPZ scalar identity values with type-preserving `.item()`; integer replicate/seed/phase remain integers. Require every sealed cache file and forbid inference on any cache miss.",
        "partial_failure": {"path": str(PARTIAL_FAILURE.relative_to(ROOT)),
                            "sha256": _sha(PARTIAL_FAILURE),
                            "scored_cells_not_saved": 10,
                            "partial_aggregates_written": False,
                            "selection_or_tuning": False,
                            "complete_replay_required": True},
        "prior_runtime_correction": {"path": str(runtime_v2.DECLARATION.relative_to(ROOT)),
                                      "sha256": _sha(runtime_v2.DECLARATION)},
        "original_plan": {"declaration_sha256": _sha(frozen.DECLARATION),
                          "runner_sha256": _sha(frozen.__file__),
                          "seal_sha256": _sha(frozen.SEAL),
                          "upstream_preflight_sha256": seal["upstream_preflight"]["sha256"],
                          "arms": base["arms"], "cohort": base["cohort"],
                          "budget": base["budget"]},
        "typed_candidate_cache_inventory": expected,
        "candidate_cache_count": len(expected),
        "skillcorner_prefill_receipt": {"path": str(PREFILL.relative_to(ROOT)),
                                         "sha256": _sha(PREFILL)},
        "model_hashes": model_hashes,
        "runtime_source_sha256": {
            str(Path(__file__).relative_to(ROOT)): _sha(__file__),
            str(Path(runtime_v2.__file__).relative_to(ROOT)): _sha(runtime_v2.__file__),
        },
        "replay_scope": {"complete_cells": 25, "fixed_arms": base["arms"],
                         "pooled_arm_cell_scores": 75,
                         "component_seed_cell_scores": 150,
                         "probability_inference_after_correction": False,
                         "aggregate_write_once": True},
        "organizer_reference_access": False,
        "public_scoring": False,
        "promotion": False,
    }
    with DECLARATION.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True)
        stream.write("\n")
    return value


def _verify():
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze runtime v3 cache correction before complete replay")
    value = json.loads(DECLARATION.read_text())
    if value.get("study_id") != STUDY:
        raise RuntimeError("Unexpected runtime v3 correction study ID")
    for rel, digest in value["runtime_source_sha256"].items():
        if _sha(ROOT / rel) != digest:
            raise RuntimeError(f"Runtime source changed: {rel}")
    if value["original_plan"]["declaration_sha256"] != _sha(frozen.DECLARATION):
        raise RuntimeError("Original pre-fit plan changed")
    if value["original_plan"]["seal_sha256"] != _sha(frozen.SEAL):
        raise RuntimeError("Original six-model seal changed")
    if _sha(PARTIAL_FAILURE) != value["partial_failure"]["sha256"]:
        raise RuntimeError("Partial failure receipt changed")
    base, seal, preflight = frozen._require_seal()
    if base["arms"] != value["original_plan"]["arms"]:
        raise RuntimeError("Fixed arm list changed")
    if preflight.get("preflight_passed") is not True:
        raise RuntimeError("Upstream C0 preflight no longer passes")
    return value, seal


def corrected_emissions(view, record, provider, match, view_name, rep, seed, phase,
                        phase_record, declaration_sha, model_sha, archived_emission_facts):
    """Read and validate existing emissions only; never infer on a cache miss."""
    if _VERIFIED_DECLARATION is None or _VERIFIED_SEAL is None:
        raise RuntimeError("Runtime v3 correction was not verified before cache reads")
    value = _VERIFIED_DECLARATION
    path = frozen._emission_path(provider, match, view_name, rep, seed, phase)
    relative = str(path.relative_to(ROOT))
    sealed_cache_sha = value["typed_candidate_cache_inventory"].get(relative)
    if not path.is_file() or sealed_cache_sha is None or _sha(path) != sealed_cache_sha:
        raise RuntimeError(f"Frozen candidate emission unavailable or changed: {path}")
    expected = _identity(view, phase_record, provider, match, view_name, rep, seed, phase,
                         declaration_sha, model_sha)
    return _validate_cache(path, expected, archived_emission_facts)


def run():
    global _VERIFIED_DECLARATION, _VERIFIED_SEAL
    _VERIFIED_DECLARATION, _VERIFIED_SEAL = _verify()
    frozen._fit_path = runtime_v2.corrected_fit_path
    frozen._emissions = corrected_emissions
    return frozen.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "run"))
    action = parser.parse_args().action
    {"declare": declare, "run": run}[action]()


if __name__ == "__main__":
    main()
