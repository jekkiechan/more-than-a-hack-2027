"""Additive execution amendment for cached v2 data/capacity feature tuples.

This wrapper leaves the frozen evaluator byte-for-byte unchanged. It memoizes
its deterministic raw-view to prepared-phase tuple so controls and candidate
recipes share one feature build per development cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib

from experiments import v2_data_capacity_eval as core


ROOT = core.ROOT
STUDY = "v2_data_capacity_eval_execution"
AMENDMENT = ROOT / f"experiments/results/{STUDY}_amendment.json"
OUT = ROOT / "data/experiments" / STUDY / "prepared"
_ORIGINAL_RAW_CACHE = core._raw_cache


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(provider: str, match: str, view: str, rep: int) -> dict:
    core.runtime.frozen._roots()
    filename = core.audit.FILE[match]
    raw_path = (core.audit.EVAL_INPUTS / f"{filename}_feed_like_corrupt_{rep}.joblib"
                if view == "feed_like_v1" else
                core.audit.EVAL_INPUTS / f"{filename}_clean_clean.joblib")
    if not raw_path.is_file():
        raise FileNotFoundError(raw_path)
    return {"provider": provider, "match": match, "view": view, "replicate": int(rep),
            "raw_path": str(raw_path), "raw_sha256": sha(raw_path),
            "v2_calibration_sha256": sha(core.runtime.frozen.CALIBRATION)
            if view == "feed_like_v2" else "not_applicable",
            "core_source_sha256": sha(Path(core.__file__)),
            "wrapper_source_sha256": sha(Path(__file__))}


def _tuple_path(provider, match, view, rep):
    label = "clean" if view == "clean" else f"{view}_r{rep}"
    return OUT / f"{provider}__{match}__{label}.joblib"


def cached_raw_cache(provider, match, view, rep):
    identity = _identity(provider, match, view, rep)
    path = _tuple_path(provider, match, view, rep)
    receipt_path = path.with_suffix(".receipt.json")
    expected = {**identity, "amendment_sha256": sha(AMENDMENT)}
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"Incomplete prepared tuple cache: {path}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") != expected or receipt.get("tuple_sha256") != sha(path):
            raise RuntimeError(f"Prepared tuple cache identity/hash mismatch: {path}")
        return joblib.load(path)
    value = _ORIGINAL_RAW_CACHE(provider, match, view, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value, path, compress=0)
    receipt = {"study_id": STUDY, "identity": expected, "tuple_sha256": sha(path)}
    receipt_path.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    return value


def _bind():
    if not core.DECLARATION.is_file():
        raise FileNotFoundError("Frozen core evaluator declaration is required first")
    declaration_sha = sha(core.DECLARATION)
    if not AMENDMENT.is_file():
        raise RuntimeError(f"Freeze execution amendment first: {AMENDMENT}")
    amendment = json.loads(AMENDMENT.read_text())
    expected = {"study_id": STUDY, "core_declaration_sha256": declaration_sha,
                "core_source_sha256": sha(Path(core.__file__)),
                "wrapper_source_sha256": sha(Path(__file__)),
                "change": "memoize deterministic _raw_cache tuples and use verified raw-path-only lookup during core freeze; no inference, metric, arm, cohort, budget or threshold changes"}
    if any(amendment.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Execution amendment or frozen core binding changed")
    return amendment


def freeze_amendment():
    if AMENDMENT.exists():
        raise RuntimeError("Execution amendment already exists")
    if not core.DECLARATION.is_file():
        raise FileNotFoundError("Freeze the core evaluator declaration before its execution amendment")
    value = {"study_id": STUDY, "core_declaration_sha256": sha(core.DECLARATION),
             "core_source_sha256": sha(Path(core.__file__)),
             "wrapper_source_sha256": sha(Path(__file__)),
             "change": "memoize deterministic _raw_cache tuples and use verified raw-path-only lookup during core freeze; no inference, metric, arm, cohort, budget or threshold changes",
             "cache_root": str(OUT.relative_to(ROOT)), "organizer_reference_access": False}
    AMENDMENT.parent.mkdir(parents=True, exist_ok=True)
    AMENDMENT.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    core._raw_cache = cached_raw_cache
    return value


def _install():
    _bind()
    core._raw_cache = cached_raw_cache


def freeze_core():
    """Run core freeze with path-only input checks, restoring its helper afterward."""
    if core.DECLARATION.exists():
        raise RuntimeError("Core evaluator declaration already exists")
    original = core._raw_cache
    def raw_path_only(provider, match, view, rep):
        identity = _identity(provider, match, view, rep)
        return (Path(identity["raw_path"]),)
    try:
        core._raw_cache = raw_path_only
        return core.freeze()
    finally:
        core._raw_cache = original


def replay_controls():
    _install()
    return core.replay_controls()


def seal():
    _install()
    return core.seal()


def run():
    _install()
    return core.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-core", "freeze-amendment", "replay-controls", "seal", "run"))
    action = parser.parse_args().action
    {"freeze-core": freeze_core, "freeze-amendment": freeze_amendment, "replay-controls": replay_controls,
     "seal": seal, "run": run}[action]()


if __name__ == "__main__":
    main()
