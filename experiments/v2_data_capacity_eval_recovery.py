"""Recover the controls-only receipt from immutable cached emissions.

This v2 execution amendment only re-scores the saved control probabilities.
It performs no feature inference, model inference, fitting, or candidate score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_data_capacity_eval as core
from experiments import v2_data_capacity_eval_execution as execution


ROOT = core.ROOT
STUDY = "v2_data_capacity_eval_recovery_v2"
AMENDMENT = ROOT / f"experiments/results/{STUDY}_amendment.json"
FAILURE_LOG = ROOT / "data/experiments/v2_data_capacity_v1/control_replay.log"
_BASE_JOB = core._control_job
_BASE_SAVE = core._save_cell_arrays
_BASE_EQUAL_PROVIDER = core.equal_provider
CHANGE = ("bind parent artifact roots, replay official metrics from the 35 sealed control NPZs, "
          "preserve cache bytes, and canonically order aggregate reduction; no inference, metric, "
          "arm, cohort, budget or threshold changes")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _roots():
    core.runtime.frozen._roots()
    if str(core.audit.DATA) != "/workspace/mtah-dt":
        raise RuntimeError(f"Unexpected parent data root: {core.audit.DATA}")
    if str(core.audit.SQD) != "/workspace/mtah-sqd/experiments/short_quality_diversity_delivery_v1/artifacts":
        raise RuntimeError(f"Unexpected parent SQD root: {core.audit.SQD}")


def _cache_facts():
    facts = {}
    for provider, match, view, rep in core._views():
        path = core._cell_cache_path(provider, match, view, rep)
        if not path.is_file():
            raise FileNotFoundError(f"Original controls-only emission missing: {path}")
        facts[f"{provider}|{match}|{view}|{rep}"] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    if len(facts) != 35:
        raise RuntimeError(f"Expected 35 saved control emissions, found {len(facts)}")
    return facts


def _prepared_facts():
    facts = {}
    for provider, match, view, rep in core._views():
        identity = {**execution._identity(provider, match, view, rep),
                    "amendment_sha256": sha(execution.AMENDMENT)}
        path = execution._tuple_path(provider, match, view, rep)
        receipt_path = path.with_suffix(".receipt.json")
        if not path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Prepared tuple/receipt missing: {path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != identity or receipt.get("tuple_sha256") != sha(path):
            raise RuntimeError(f"Prepared tuple receipt mismatch: {path}")
        facts[f"{provider}|{match}|{view}|{rep}"] = {
            "path": str(path.relative_to(ROOT)), "sha256": sha(path),
            "receipt_path": str(receipt_path.relative_to(ROOT)),
            "receipt_sha256": sha(receipt_path), "identity": identity}
    if len(facts) != 35:
        raise RuntimeError(f"Expected 35 existing prepared tuples, found {len(facts)}")
    return facts


def freeze():
    if AMENDMENT.exists():
        raise RuntimeError("Recovery amendment already exists")
    if not core.DECLARATION.is_file() or core.REPLAY.exists():
        raise RuntimeError("Need frozen core declaration and no existing replay receipt")
    _roots()
    if not FAILURE_LOG.is_file():
        raise FileNotFoundError(FAILURE_LOG)
    value = {"study_id": STUDY,
             "core_declaration_sha256": sha(core.DECLARATION),
             "prior_execution_amendment_sha256": sha(execution.AMENDMENT),
             "core_source_sha256": sha(Path(core.__file__)),
             "execution_wrapper_sha256": sha(Path(execution.__file__)),
             "recovery_source_sha256": sha(Path(__file__)),
             "failure_log_sha256": sha(FAILURE_LOG),
             "prepared_tuple_inventory": _prepared_facts(),
             "immutable_control_emissions": _cache_facts(),
             "change": CHANGE,
             "candidate_scoring": False, "organizer_reference_access": False}
    AMENDMENT.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    return value


def _bind():
    if not AMENDMENT.is_file():
        raise RuntimeError("Freeze recovery amendment first")
    _roots()
    amendment = json.loads(AMENDMENT.read_text())
    expected = {"study_id": STUDY,
                "core_declaration_sha256": sha(core.DECLARATION),
                "prior_execution_amendment_sha256": sha(execution.AMENDMENT),
                "core_source_sha256": sha(Path(core.__file__)),
                "execution_wrapper_sha256": sha(Path(execution.__file__)),
                "recovery_source_sha256": sha(Path(__file__)),
                "failure_log_sha256": sha(FAILURE_LOG), "change": CHANGE}
    if any(amendment.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Recovery amendment source/declaration binding changed")
    for key, fact in amendment["immutable_control_emissions"].items():
        if sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Original control emission changed: {key}")
    for key, fact in amendment["prepared_tuple_inventory"].items():
        if (sha(ROOT / fact["path"]) != fact["sha256"] or
                sha(ROOT / fact["receipt_path"]) != fact["receipt_sha256"]):
            raise RuntimeError(f"Prepared tuple changed: {key}")
    return amendment


def _replay_cached_cell(job):
    """Load immutable arrays, then use core's official score path only."""
    _roots()
    provider, match, view, rep = job
    path = execution._tuple_path(provider, match, view, rep)
    if not path.is_file() or not path.with_suffix(".receipt.json").is_file():
        raise FileNotFoundError(f"Recovery requires an existing prepared tuple: {path}")
    prep_path, _native, _labels, phases, _period = execution.cached_raw_cache(
        provider, match, view, rep)
    raw_identity = execution._identity(provider, match, view, rep)
    key = f"{provider}|{match}|{view}|{rep}"
    emission = core._cell_cache_path(provider, match, view, rep)
    identity = {"study_id": core.STUDY, "declaration_sha256": sha(core.DECLARATION),
                "provider": provider, "match": match, "view": view, "replicate": rep,
                "input_sha256": raw_identity["raw_sha256"], "phase_count": len(phases)}
    controls = core._load_cell_arrays(emission, identity)
    row = core._score_cell(provider, match, view, rep, control_cache=controls)
    if row["input_sha256"] != raw_identity["raw_sha256"]:
        raise RuntimeError(f"Cached tuple input identity mismatch: {key}")
    return row


def _control_job(job):
    return _replay_cached_cell(job)


def _canonical_equal_provider(values):
    return _BASE_EQUAL_PROVIDER(dict(sorted(values.items())))


def _assert_unchanged_cache(path, row, identity):
    expected = {key: np.asarray(value) for key, value in row.items()
                if key in {"labels", "timestamps", "period", "model_period", "fps"}}
    expected.update({f"prob__{name}": np.asarray(value)
                     for name, value in row["probabilities"].items()})
    expected.update({f"identity__{key}": np.asarray(str(value))
                     for key, value in identity.items()})
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise RuntimeError(f"Cached emission key set differs: {path}")
        for key, value in expected.items():
            actual = saved[key]
            equal = (np.array_equal(actual, value, equal_nan=True)
                     if actual.dtype.kind in "fci" and value.dtype.kind in "fci"
                     else np.array_equal(actual, value))
            if not equal:
                raise RuntimeError(f"Cached control emission differs from replay: {path}/{key}")
    return sha(path)


def replay_controls():
    amendment = _bind()
    execution._install()
    _roots()
    original_job, original_save, original_equal = (
        core._control_job, core._save_cell_arrays, core.equal_provider)
    core._control_job = _control_job
    core._save_cell_arrays = _assert_unchanged_cache
    core.equal_provider = _canonical_equal_provider
    try:
        return core.replay_controls()
    finally:
        core._control_job, core._save_cell_arrays, core.equal_provider = (
            original_job, original_save, original_equal)


def seal():
    _bind()
    execution._install()
    _roots()
    original_equal = core.equal_provider
    core.equal_provider = _canonical_equal_provider
    try:
        return core.seal()
    finally:
        core.equal_provider = original_equal


def run():
    _bind()
    execution._install()
    _roots()
    original_equal = core.equal_provider
    core.equal_provider = _canonical_equal_provider
    try:
        return core.run()
    finally:
        core.equal_provider = original_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay-controls", "seal", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "replay-controls": replay_controls,
     "seal": seal, "run": run}[action]()


if __name__ == "__main__":
    main()
