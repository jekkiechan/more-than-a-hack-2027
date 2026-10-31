"""Run one declared objective fit with normalized arrays backed by read-only mmap.

This adapter calls the frozen objective-alignment runner unchanged. Its only
training-time patch replaces ``learned_temporal.normalize`` with a wrapper that
computes the same array, writes that array to a temporary ``.npy`` file, and
returns a read-only memory map. The wrapper and its temporary files are removed
when the fit exits.

The adapter was introduced after candidate fitting had begun. Its runtime
addendum records that timing and binds this adapter plus the frozen training
sources before any subsequent fit uses it.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time

import numpy as np

from experiments import (corruption_experiment as ce,
                         factorized_had,
                         learned_temporal as lt,
                         objective_alignment as oa,
                         objective_losses)
from experiments.eval_set.protocol import sha256


ROOT = oa.ROOT
ADDENDUM_NAME = "runtime_addendum.json"
_INITIAL_FITS_WITHOUT_ADAPTER = (
    "skillcorner/spell_sqrt/2026",
    "sportec/spell_sqrt/2026",
    "skillcorner/combined/2027",
)
_BASE_NORMALIZE = lt.normalize
_NORMALIZE_PATCH_LOCK = threading.Lock()


def _runtime_bindings(out: Path) -> dict:
    test_path = ROOT / "experiments/tests/test_objective_filebacked.py"
    return dict(
        version=1,
        study="objective_alignment_v1",
        adapter_sha256=sha256(Path(__file__)),
        parity_test_sha256=sha256(test_path) if test_path.exists() else None,
        predeclared_sha256=sha256(out / "predeclared.json"),
        runner_sha256=sha256(Path(oa.__file__)),
        objective_losses_sha256=sha256(Path(objective_losses.__file__)),
        factorized_had_sha256=sha256(Path(factorized_had.__file__)),
        learned_temporal_sha256=sha256(Path(lt.__file__)),
        normalization_contract=(
            "Call the frozen learned_temporal.normalize exactly once per original "
            "and alternate record, save its returned values without transformation, "
            "and pass read-only npy-backed memmaps to the unchanged Dataset/DataLoader."),
        timing_disclosure=(
            "The adapter and this addendum were introduced after candidate fits had "
            "started. The listed initial fits use the original in-memory path; this "
            "addendum applies only to fits launched after its creation."),
        initial_fits_started_without_adapter=list(_INITIAL_FITS_WITHOUT_ADAPTER),
    )


def freeze_runtime_addendum(out: Path = oa.DEFAULT) -> dict:
    """Freeze the post-hoc runtime adapter before any adapter-backed fit."""
    out = Path(out)
    spec = oa.check(out)
    expected = _runtime_bindings(out)
    if expected["parity_test_sha256"] is None:
        raise FileNotFoundError("The synthetic file-backed parity test is missing")
    addendum_path = out / ADDENDUM_NAME
    fixed_runtime_fields = dict(
        after_initial_fits_started=True,
        adapter_scope="Subsequent objective fits only; no in-progress fit is switched mid-run.",
        training_recipe_changed=False,
        model_architecture_changed=False,
        arrays_transformed=False,
        optimizer_or_rng_changed=False,
    )
    if addendum_path.exists():
        addendum = json.loads(addendum_path.read_text())
        for key, value in dict(**expected, **fixed_runtime_fields).items():
            if addendum.get(key) != value:
                raise ValueError(f"Frozen runtime addendum changed: {key}")
        _require(set(addendum) == set(expected) | set(fixed_runtime_fields) | {"created_utc"},
                 "Frozen runtime addendum has unexpected or missing fields")
        return addendum

    # `oa.check` above verifies that this addition applies to the existing
    # frozen objective declaration and that its source/input bindings remain.
    addendum = dict(**expected, **fixed_runtime_fields,
                    created_utc=datetime.now(timezone.utc).isoformat())
    ce.write_json(addendum_path, addendum, exclusive=True)
    return addendum


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_runtime_addendum(out: Path) -> tuple[dict, Path]:
    addendum_path = out / ADDENDUM_NAME
    _require(addendum_path.exists(),
             "Freeze runtime_addendum.json before starting a file-backed fit")
    addendum = json.loads(addendum_path.read_text())
    expected = _runtime_bindings(out)
    for key, value in expected.items():
        _require(addendum.get(key) == value,
                 f"Runtime addendum no longer matches current sources: {key}")
    _require(addendum.get("after_initial_fits_started") is True,
             "Runtime addendum lacks the required post-hoc timing disclosure")
    return addendum, addendum_path


@contextmanager
def filebacked_normalize(temp_root: Path | None = None):
    """Temporarily make exact normalized arrays read-only, file-backed maps.

    The original normalizer owns all numerical work. Serialization and loading
    preserve the resulting dtype, shape, strides and element bits. The global
    patch is process-local and restored even if normalization or fitting fails.
    One fit per process is supported.
    """
    if not _NORMALIZE_PATCH_LOCK.acquire(blocking=False):
        raise RuntimeError("A file-backed normalization fit is already active in this process")
    if lt.normalize is not _BASE_NORMALIZE:
        _NORMALIZE_PATCH_LOCK.release()
        raise RuntimeError("learned_temporal.normalize changed before adapter entry")

    if temp_root is not None and not Path(temp_root).is_dir():
        _NORMALIZE_PATCH_LOCK.release()
        raise FileNotFoundError(f"Temporary root is not a directory: {temp_root}")
    try:
        directory = tempfile.TemporaryDirectory(prefix="objective-normalized-",
                                                 dir=str(temp_root) if temp_root else None)
    except Exception:
        _NORMALIZE_PATCH_LOCK.release()
        raise
    temp_path = Path(directory.name)
    stats = dict(
        temp_directory=str(temp_path),
        normalize_calls=0,
        mapped_arrays=[],
        all_mappings_read_only=True,
        normalizer_restored=False,
        temporary_files_removed=False,
    )
    original_normalize = lt.normalize

    def mapped_normalize(x, median, scale):
        normalized = original_normalize(x, median, scale)
        index = stats["normalize_calls"]
        path = temp_path / f"normalized_{index:05d}.npy"
        np.save(path, normalized, allow_pickle=False)
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        same_layout = (mapped.shape == normalized.shape
                       and mapped.dtype == normalized.dtype
                       and mapped.strides == normalized.strides
                       and mapped.flags.c_contiguous == normalized.flags.c_contiguous)
        if not same_layout or mapped.flags.writeable:
            raise RuntimeError("Read-only mmap changed normalized array layout or mutability")
        stats["mapped_arrays"].append(dict(
            shape=list(mapped.shape), dtype=mapped.dtype.str,
            nbytes=int(mapped.nbytes), writeable=bool(mapped.flags.writeable)))
        stats["normalize_calls"] += 1
        del normalized
        return mapped

    try:
        lt.normalize = mapped_normalize
        yield stats
    finally:
        lt.normalize = original_normalize
        stats["normalizer_restored"] = lt.normalize is original_normalize
        try:
            directory.cleanup()
            stats["temporary_files_removed"] = not temp_path.exists()
        finally:
            _NORMALIZE_PATCH_LOCK.release()


def fit_filebacked(out: Path, source: str, arm: str, seed: int,
                   temp_root: Path | None = None) -> dict:
    """Call the frozen runner for one fit and write its adapter receipt."""
    out = Path(out)
    spec = oa.check(out)
    addendum, addendum_path = _check_runtime_addendum(out)
    _require(source in spec["training_sources"],
             f"Unknown heldout provider {source!r}")
    _require(arm in oa.NEW_ARMS, f"File-backed adapter accepts only new arms: {arm!r}")
    _require(seed in oa.SEEDS, f"Undeclared fit seed: {seed!r}")
    _require(not (out / "checkpoints_frozen.json").exists(),
             "Model fitting is closed after checkpoint sealing")

    model_path = oa.checkpoint(out, source, arm, seed)
    receipt_path = model_path.with_suffix(".runtime.json")
    _require(not model_path.exists(), f"Candidate already exists; refusing to claim an adapter fit: {model_path}")
    _require(not receipt_path.exists(), f"Runtime receipt already exists: {receipt_path}")
    started = time.monotonic()
    with filebacked_normalize(temp_root) as storage:
        oa.fit(out, source, arm, seed)
    _require(storage["normalizer_restored"], "Normalizer wrapper was not restored")
    _require(storage["temporary_files_removed"], "Temporary mapped arrays were not cleaned up")
    expected_calls = 2 * 4 * len(spec["training_matches"][source])
    _require(storage["normalize_calls"] == expected_calls,
             f"Unexpected normalization call count: {storage['normalize_calls']} != {expected_calls}")
    _require(storage["all_mappings_read_only"]
             and all(not row["writeable"] for row in storage["mapped_arrays"]),
             "A normalized array was not read-only")
    _require(model_path.exists(), "Frozen runner did not produce the candidate checkpoint")
    model = ce.load_model(out, model_path)
    model_receipt_path = model_path.with_suffix(".json")
    model_receipt = json.loads(model_receipt_path.read_text())
    objective = oa.OBJECTIVE_BY_ARM[arm]
    _require((model.get("objective"), model.get("seed"), model_receipt.get("heldout"),
              model_receipt.get("arm"), model_receipt.get("seed"))
             == (objective, seed, source, arm, seed),
             "Candidate checkpoint provenance differs from the requested fit")
    control = oa._load_control_model(spec, source, seed)
    for key in ("parameters", "optimizer_updates", "windows_per_epoch"):
        _require(model[key] == control[key], f"Candidate changed control budget: {key}")
    for key in ("median", "scale"):
        _require(np.array_equal(model[key], control[key]),
                 f"Candidate changed control normalization: {key}")
    normalization_hashes = {
        key: hashlib.sha256(np.ascontiguousarray(model[key]).tobytes()).hexdigest()
        for key in ("median", "scale")
    }
    runtime_receipt = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        heldout=source, arm=arm, objective=objective, seed=seed,
        elapsed_seconds=time.monotonic() - started,
        adapter_sha256=sha256(Path(__file__)),
        runtime_addendum_sha256=sha256(addendum_path),
        objective_predeclared_sha256=sha256(out / "predeclared.json"),
        model_sha256=sha256(model_path),
        model_receipt_sha256=sha256(model_receipt_path),
        parameters=int(model["parameters"]),
        optimizer_updates=int(model["optimizer_updates"]),
        windows_per_epoch=int(model["windows_per_epoch"]),
        normalization_sha256=normalization_hashes,
        loss_coefficients=model["loss_coefficients"],
        normalized_mmap_count=storage["normalize_calls"],
        normalized_mmap_bytes=sum(row["nbytes"] for row in storage["mapped_arrays"]),
        normalized_mmap_shapes=sorted({tuple(row["shape"]) for row in storage["mapped_arrays"]}),
        normalized_mmap_dtypes=sorted({row["dtype"] for row in storage["mapped_arrays"]}),
        checks=dict(
            model_loaded_and_validated=True,
            model_provenance_matched=True,
            model_budget_matches_control=True,
            normalization_matches_control=True,
            normalizer_restored=storage["normalizer_restored"],
            temporary_files_removed=storage["temporary_files_removed"],
            all_mappings_read_only=storage["all_mappings_read_only"],
            normalization_call_count_matched=True,
        ),
    )
    ce.write_json(receipt_path, runtime_receipt, exclusive=True)
    # Include no temporary paths in the receipt; they have already been deleted.
    _ = addendum
    return runtime_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-runtime", "fit"))
    parser.add_argument("--out", type=Path, default=oa.DEFAULT)
    parser.add_argument("--source")
    parser.add_argument("--arm", choices=oa.NEW_ARMS)
    parser.add_argument("--seed", type=int, choices=oa.SEEDS)
    parser.add_argument("--temp-root", type=Path,
                        help="Optional directory for temporary normalized .npy files")
    args = parser.parse_args()
    if args.action == "freeze-runtime":
        print(json.dumps(freeze_runtime_addendum(args.out), indent=2, sort_keys=True))
        return
    if args.source is None or args.arm is None or args.seed is None:
        parser.error("fit requires --source, --arm and --seed")
    receipt = fit_filebacked(args.out, args.source, args.arm, args.seed, args.temp_root)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
