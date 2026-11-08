"""Freeze and cache the 35 native frontier input/phase tuples for recovery.

This is input preparation only: it loads verified raw tracking, applies the
frozen deterministic v2 corruption where selected, recomputes shared features,
and saves aligned phases. It loads no model checkpoints and scores no labels.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import joblib
import numpy as np
from experiments import v2_data_capacity_eval as capacity
from experiments import v2_spell_target_runtime as runtime
from experiments import tracking_corruption
from experiments import v2_short_original_diversity as recipe

ROOT = Path(__file__).resolve().parents[1]
STUDY = "frontier_input_prepare_v1"
OUT = ROOT / "data/experiments/frontier_recovery_v1/prepared_inputs"
DECLARATION = ROOT / "experiments/results/frontier_input_prepare_v1_declaration.json"
MANIFEST = OUT / "manifest.json"
PREDECESSOR_MANIFEST = ROOT / "data/experiments/complementary_spell_v1/native_probability/preparation_manifest.json"
SEEDS = (2026, 2027)
VIEWS = (("clean", (0,)), ("feed_like_v1", (1, 2)), ("feed_like_v2", (1, 2, 3, 4)))
AUDIT = runtime.frozen.audit
runtime.frozen.tc = tracking_corruption


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _configure() -> tuple[Path, Path]:
    data = Path(os.environ.get("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))).resolve()
    sqd = Path(os.environ.get("MTAH_SQD_DIR", str(ROOT.parent / "mtah-sqd"))).resolve()
    runtime.frozen._roots()
    runtime.frozen.tc = tracking_corruption
    return data, sqd


def _source_paths() -> list[Path]:
    return sorted({Path(__file__), Path(capacity.__file__), Path(runtime.__file__),
        Path(runtime.frozen.__file__), Path(AUDIT.__file__), Path(recipe.v2.__file__),
        Path(recipe.prep.__file__), Path(recipe.ms.__file__), Path(tracking_corruption.__file__),
        Path(recipe.rf.supervised.__file__), Path(recipe.rf.sn.__file__),
        Path(recipe.rf.old.__file__), ROOT / "experiments/refine_features.py",
        Path(recipe.rf.sc.__file__), Path(recipe.rf.dl.__file__),
        Path(recipe.rf.sn.fh.__file__), Path(recipe.rf.sn.lt.__file__),
        Path(recipe.rf.sn.ol.__file__), Path(recipe.rf.deployment.__file__),
        ROOT / "experiments/feed_benchmark_audit_v1.py",
        ROOT / "experiments/feed_like_v2.py", ROOT / "experiments/tracking_corruption.py",
        ROOT / "experiments/soccernet_capacity_model.py", ROOT / "experiments/factorized_had.py",
        ROOT / "experiments/learned_temporal.py", ROOT / "experiments/stress_components.py",
        ROOT / "experiments/datasets.py", ROOT / "experiments/short_label_quality.py",
        ROOT / "experiments/feed_short_sampler.py", ROOT / "experiments/dense_short_team_v1.py",
        ROOT / "experiments/results/feed_like_v2_calibration.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_round1_declaration.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_selection.json"})


def _source_hashes() -> dict:
    return {str(p.relative_to(ROOT)): sha(p) for p in _source_paths()}


def _jobs() -> list[tuple[str, str, str, int]]:
    return [(provider, match, view, rep) for provider, match, _ in AUDIT.MATCHES
            for view, reps in VIEWS for rep in reps]


def _key(job: tuple[str, str, str, int]) -> str:
    return "|".join((job[0], job[1], job[2], str(job[3])))


def _predecessor(job: tuple[str, str, str, int], *, load_arrays: bool = True) -> tuple[dict, dict | None, str]:
    manifest = json.loads(PREDECESSOR_MANIFEST.read_text())
    fact = manifest["cells"].get(_key(job))
    if not isinstance(fact, dict):
        raise KeyError(f"Missing predecessor metadata: {_key(job)}")
    path = ROOT / fact["path"]
    if not path.is_file() or sha(path) != fact["sha256"]:
        raise ValueError(f"Changed predecessor cache: {_key(job)}")
    arrays = None
    if load_arrays:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    return fact, arrays, sha(PREDECESSOR_MANIFEST)


def _artifact_input(path: Path, data_root: Path) -> dict:
    manifest = data_root / "seed_gap_context_cache_manifest.json"
    relative = path.relative_to(data_root).as_posix()
    fact = runtime.frozen._artifact_fact(manifest, data_root, relative)
    return {"path": str(path), "sha256": fact["sha256"],
            "manifest_path": str(manifest), "manifest_sha256": sha(manifest)}


def freeze() -> dict:
    if DECLARATION.exists() or MANIFEST.exists():
        raise FileExistsError("Input preparation declaration/output already exists")
    data_root, _ = _configure()
    if not PREDECESSOR_MANIFEST.is_file():
        raise FileNotFoundError(PREDECESSOR_MANIFEST)
    source_hashes = _source_hashes()
    input_facts = {}
    predecessor_facts = {}
    for job in _jobs():
        provider, match, view, rep = job
        raw_path = (AUDIT.EVAL_INPUTS / f"{AUDIT.FILE[match]}_clean_clean.joblib" if view in ("clean", "feed_like_v2")
                    else AUDIT.EVAL_INPUTS / f"{AUDIT.FILE[match]}_feed_like_corrupt_{rep}.joblib")
        input_facts[_key(job)] = _artifact_input(raw_path, data_root)
        fact, arrays, predecessor_manifest_sha = _predecessor(job, load_arrays=False)
        expected_raw_sha = fact["identity"].get("source_sha256", {}).get(input_facts[_key(job)]["path"])
        if expected_raw_sha != input_facts[_key(job)]["sha256"]:
            raise ValueError(f"View raw-input SHA differs from predecessor source receipt: {_key(job)}")
        predecessor_facts[_key(job)] = {"path": fact["path"], "sha256": fact["sha256"],
            "manifest_sha256": predecessor_manifest_sha, "phase_count": fact["phase_count"],
            "rows": fact["rows"], "input_sha256": fact["identity"]["input_sha256"],
            "view_raw_input_sha256": expected_raw_sha,
            "calibration_sha256": fact["identity"]["calibration_sha256"]}
        if fact["identity"]["calibration_sha256"] != sha(runtime.frozen.CALIBRATION):
            raise ValueError(f"Calibration SHA differs from predecessor receipt: {_key(job)}")
    value = {"schema": "frontier_input_prepare_v1_declaration", "study_id": STUDY,
        "status": "frozen_before_input_preparation", "expected_cells": sorted(input_facts),
        "source_hashes": source_hashes, "raw_input_facts": input_facts,
        "predecessor_facts": predecessor_facts,
        "predecessor_manifest": {"path": str(PREDECESSOR_MANIFEST.relative_to(ROOT)),
                                 "sha256": sha(PREDECESSOR_MANIFEST)},
        "calibration_sha256": sha(runtime.frozen.CALIBRATION),
        "seed_protocol": "feed_like_v2_benchmark/{rep}/{match}",
        "thread_limit": 1, "max_workers": 2, "scoring_performed": False,
        "candidate_scoring": False, "organizer_reference_access": False}
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    with DECLARATION.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    return value


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _prepare_cell(job: tuple[str, str, str, int]) -> dict:
    declaration = json.loads(DECLARATION.read_text())
    key = _key(job)
    if declaration["source_hashes"] != _source_hashes():
        raise ValueError("Input preparation sources changed after freeze")
    raw_fact = declaration["raw_input_facts"][key]
    raw_path = Path(raw_fact["path"])
    if sha(raw_path) != raw_fact["sha256"]:
        raise ValueError(f"Raw source changed after freeze: {key}")
    provider, match, view, rep = job
    returned_path, inputs, labels, phases, model_period = capacity._raw_cache(provider, match, view, rep)
    if returned_path != raw_path:
        raise ValueError(f"Shared input loader selected another raw source: {key}")
    previous_fact, previous, predecessor_manifest_sha = _predecessor(job)
    metadata = {"labels": labels, "period": inputs["period"], "timestamps": inputs["timestamps"],
                "model_period": model_period, "fps": inputs["fps"]}
    for name, value in metadata.items():
        if not np.array_equal(np.asarray(value), np.asarray(previous[name])):
            raise ValueError(f"Native phase input metadata differs from predecessor: {key}/{name}")
    if len(phases) != previous_fact["phase_count"]:
        raise ValueError(f"Native phase count differs from predecessor: {key}")
    seed = (int(hashlib.sha256(f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8], 16)
            if view == "feed_like_v2" else None)
    payload = {"path": str(returned_path), "inputs": inputs, "labels": labels,
               "phases": phases, "model_period": model_period}
    cache_path = OUT / "cells" / f"{key.replace('|','__')}.joblib"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    joblib.dump(payload, temp, compress=0)
    os.replace(temp, cache_path)
    receipt = {"cell": key, "provider": provider, "match": match, "view": view, "rep": rep,
        "source_status": "reconstructed", "raw_input_sha256": raw_fact["sha256"],
        "input_manifest_sha256": raw_fact["manifest_sha256"],
        "calibration_sha256": declaration["calibration_sha256"], "corruption_seed": seed,
        "predecessor_cache_sha256": previous_fact["sha256"],
        "predecessor_manifest_sha256": predecessor_manifest_sha,
        "phase_count": len(phases), "rows": len(labels), "fps": float(inputs["fps"]),
        "cache_sha256": sha(cache_path), "cache_path": str(cache_path.relative_to(ROOT)),
        "scoring_performed": False}
    return receipt


def prepare(workers: int = 2) -> dict:
    if workers not in (1, 2):
        raise ValueError("Use at most two workers")
    if not DECLARATION.is_file() or MANIFEST.exists():
        raise FileExistsError("Freeze declaration first; preserve any existing manifest")
    declaration = json.loads(DECLARATION.read_text())
    if declaration["source_hashes"] != _source_hashes():
        raise ValueError("Source code differs from frozen input preparation declaration")
    receipts = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_prepare_cell, job): job for job in _jobs()}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            receipts[row["cell"]] = row
            print(f"PREPARED NATIVE INPUT {index}/35 {row['cell']}", flush=True)
    manifest = {"schema": "frontier_input_prepare_v1_manifest", "study_id": STUDY,
        "declaration_sha256": sha(DECLARATION), "source_hashes": declaration["source_hashes"],
        "predecessor_manifest": declaration["predecessor_manifest"],
        "expected_cells": declaration["expected_cells"], "cells": receipts,
        "source_status": "reconstructed_inputs", "feature_recomputed_after_corruption": True,
        "native_fps_and_phase_counts_preserved": True, "scoring_performed": False,
        "candidate_scoring": False, "organizer_reference_access": False,
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "joblib": joblib.__version__}}
    with MANIFEST.open("x") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def load(key: str) -> dict:
    declaration = json.loads(DECLARATION.read_text())
    manifest = json.loads(MANIFEST.read_text())
    if manifest["declaration_sha256"] != sha(DECLARATION):
        raise ValueError("Input cache manifest differs from its declaration")
    if declaration["source_hashes"] != _source_hashes():
        raise ValueError("Input preparation source changed")
    fact = manifest["cells"].get(key)
    if not fact:
        raise KeyError(key)
    path = ROOT / fact["cache_path"]
    if not path.is_file() or sha(path) != fact["cache_sha256"]:
        raise ValueError(f"Input cache changed: {key}")
    return joblib.load(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare"))
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    _configure()
    value = freeze() if args.action == "freeze" else prepare(args.workers)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
