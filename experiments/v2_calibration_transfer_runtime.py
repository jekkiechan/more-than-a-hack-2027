"""Mechanical execution wrapper for the frozen calibration-transfer study.

The frozen transfer runner compares cached NPZ integer identity fields as
strings, and its seal lookup uses short provider aliases where the six-model
seal uses ``leave_out_*`` keys. This wrapper corrects only those readers,
binds the main evaluation's cache-only v3 correction, and delegates the fixed
three-arm transfer seal and run without changing emissions or scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_calibration_transfer as frozen
from experiments import v2_training_eval_runtime_v3 as main_v3


ROOT = frozen.ROOT
AMENDMENT = ROOT / "experiments/results/v2_calibration_transfer_execution_amendment.json"
COMPLETION = ROOT / "experiments/results/v2_calibration_transfer_runtime_receipt.json"

# Keep the fold-to-model correction used by the active main evaluation.
frozen.base._fit_path = main_v3.runtime_v2.corrected_fit_path


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_facts():
    paths = (Path(__file__), Path(frozen.__file__), Path(main_v3.__file__),
             Path(main_v3.runtime_v2.__file__), Path(frozen.base.__file__),
             frozen.DECLARATION, main_v3.DECLARATION,
             main_v3.runtime_v2.DECLARATION, main_v3.runtime_v2.PREVIOUS_CORRECTION,
             frozen.base.DECLARATION, frozen.base.SEAL,
             frozen.calibration.RESULT, frozen.calibration.DECLARATION,
             frozen.upstream.frozen.DECLARATION, frozen.upstream.frozen.PREFLIGHT)
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): _sha(path)
            for path in paths}


def _normalized_cache_identity(provider, match, view, rep, seed, phase, phase_record, view_data):
    fold_alias = frozen.upstream.frozen.audit.FOLD[provider]
    model_fact = json.loads(frozen.base.SEAL.read_text())["models"][f"leave_out_{fold_alias}/{seed}"]
    teacher_path = frozen.upstream.frozen._teacher_path(provider, seed)
    calibrated_path = frozen.upstream.frozen._model_path("adop", provider, seed)
    identity = {
        "declaration_sha256": _sha(frozen.base.DECLARATION),
        "model_sha256": model_fact["sha256"],
        "teacher_sha256": _sha(teacher_path),
        "calibrated_model_sha256": _sha(calibrated_path),
        "input_sha256": frozen.upstream.frozen._input_sha(view_data),
        "record_sha256": frozen.base._record_sha(phase_record),
        "feature_names_sha256": frozen.base._array_sha(
            np.asarray(list(phase_record["names"]), dtype="U")),
        "provider": provider,
        "match": match,
        "view": view,
        "replicate": int(rep),
        "seed": int(seed),
        "phase": int(phase),
    }
    # The frozen transfer reader stringifies NPZ scalar values before comparison.
    return {key: str(value) for key, value in identity.items()}


frozen._cache_identity = _normalized_cache_identity


def freeze_amendment() -> dict:
    """Bind the fixed transfer declaration to the main cache-only correction."""
    if AMENDMENT.exists() or COMPLETION.exists():
        raise RuntimeError("Transfer execution amendment/completion receipt already exists")
    transfer_decl = frozen._verify_declaration()
    main_correction, _ = main_v3._verify()
    value = {
        "study_id": "v2_calibration_transfer_execution_amendment",
        "status": "mechanical_cache_reader_and_seal_key_correction_frozen_before_transfer_seal_or_scores",
        "scope": "Normalize expected cache identity scalars to strings for the frozen transfer reader and map short provider aliases to the exact leave_out_* model seal keys; do not alter arrays, selected decoder, arms, labels, scoring or cohort.",
        "transfer_declaration": {"path": str(frozen.DECLARATION.relative_to(ROOT)),
                                 "sha256": _sha(frozen.DECLARATION),
                                 "study_id": transfer_decl["study_id"],
                                 "arms": transfer_decl["arms"],
                                 "config": transfer_decl["fixed_calibration"]},
        "main_evaluation_cache_correction": {
            "path": str(main_v3.DECLARATION.relative_to(ROOT)),
            "sha256": _sha(main_v3.DECLARATION),
            "source_path": str(Path(main_v3.__file__).relative_to(ROOT)),
            "source_sha256": _sha(main_v3.__file__),
            "cache_inventory_count": main_correction["candidate_cache_count"],
            "cache_only_no_inference": True,
        },
        "prior_main_runtime_correction": {
            "path": str(main_v3.runtime_v2.PREVIOUS_CORRECTION.relative_to(ROOT)),
            "sha256": _sha(main_v3.runtime_v2.PREVIOUS_CORRECTION),
        },
        "transfer_source_bindings": _source_facts(),
        "cache_identity_rules": {
            "integer_scalar_fields": ["replicate", "seed", "phase"],
            "comparison": "normalize expected values to strings to match the frozen transfer reader's string conversion",
            "model_seal_key": "leave_out_{provider_fold}/{seed}",
        },
        "new_fits": 0,
        "new_decoder_settings": 0,
        "new_emissions": 0,
        "organizer_reference_access": False,
        "public_scoring": False,
        "promotion": False,
    }
    with AMENDMENT.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True)
        stream.write("\n")
    return value


def _verify_amendment():
    if not AMENDMENT.is_file():
        raise RuntimeError("Freeze transfer execution amendment before sealing or scoring")
    amendment = json.loads(AMENDMENT.read_text())
    if amendment.get("study_id") != "v2_calibration_transfer_execution_amendment":
        raise RuntimeError("Unexpected transfer execution amendment")
    if amendment["transfer_declaration"]["sha256"] != _sha(frozen.DECLARATION):
        raise RuntimeError("Frozen transfer declaration changed")
    if amendment["main_evaluation_cache_correction"]["sha256"] != _sha(main_v3.DECLARATION):
        raise RuntimeError("Main cache correction declaration changed")
    if amendment["main_evaluation_cache_correction"]["source_sha256"] != _sha(main_v3.__file__):
        raise RuntimeError("Main cache correction source changed")
    if amendment["prior_main_runtime_correction"]["sha256"] != _sha(main_v3.runtime_v2.PREVIOUS_CORRECTION):
        raise RuntimeError("Prior model-path correction changed")
    if amendment["transfer_source_bindings"] != _source_facts():
        raise RuntimeError("Transfer execution source/input bindings changed")
    correction, _ = main_v3._verify()
    if correction["candidate_cache_count"] != amendment["main_evaluation_cache_correction"]["cache_inventory_count"]:
        raise RuntimeError("Main cache inventory count changed")
    return amendment


def seal():
    _verify_amendment()
    return frozen.seal()


def run():
    _verify_amendment()
    result = frozen.run()
    if COMPLETION.exists():
        raise RuntimeError("Transfer completion receipt already exists")
    completion = {
        "study_id": frozen.STUDY,
        "execution_amendment_sha256": _sha(AMENDMENT),
        "transfer_declaration_sha256": _sha(frozen.DECLARATION),
        "transfer_seal_sha256": _sha(frozen.SEAL),
        "transfer_result_sha256": _sha(frozen.RESULT),
        "main_evaluation_cache_correction_sha256": _sha(main_v3.DECLARATION),
        "main_evaluation_result_sha256": _sha(frozen.base.RESULT),
        "candidate_cells": result["candidate_cells"],
        "candidate_arms": result["candidate_arms"],
        "cache_only_no_inference": True,
        "organizer_reference_access": False,
        "public_scoring": False,
        "promotion": False,
    }
    with COMPLETION.open("x") as stream:
        json.dump(completion, stream, indent=1, sort_keys=True)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-amendment", "seal", "run"))
    action = parser.parse_args().action
    {"freeze-amendment": freeze_amendment, "seal": seal, "run": run}[action]()


if __name__ == "__main__":
    main()
