"""Cache native per-phase dev emissions for the v2 postprocessing study.

Study `v2_postproc_v1`. Labels are stored alongside emissions for later scoring
but never enter inference. Families: `cand` (adopted original-full fold models)
and `adop` (calibrated dense D2 fold models), both provider-excluded.
Cells: five exposed development matches x (clean, feed_like v1 r1-2, v2 r1-4).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import v2_spell_target_runtime as runtime  # noqa: F401  (alias fix)
from experiments import v2_spell_target as base
from experiments import preparation as prep
from experiments import supervised

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_postproc_v1"
OUT = ROOT / "data/experiments" / STUDY / "emissions"
SEEDS = tuple(int(x) for x in os.environ.get("EMIT_SEEDS", "2026,2027").split(","))
FAMILIES = ("cand", "adop")
VIEWS = (("clean", (0,)), ("feed_like_v1", (1, 2)), ("feed_like_v2", (1, 2, 3, 4)))


def cells():
    return [(p, m, v, r) for p, m, _ in base.audit.MATCHES for v, reps in VIEWS for r in reps]


def cell_key(provider, match, view, rep):
    return f"{provider}__{match}__{view}__{rep}"


def prepare(provider, match, view, rep):
    base._roots()
    if view in ("clean", "feed_like_v2"):
        return base._prepare_cell(provider, match, "clean" if view == "clean" else view, rep)
    cached = joblib.load(base.audit.EVAL_INPUTS / f"{base.audit.FILE[match]}_feed_like_corrupt_{rep}.joblib")
    inputs, labels = cached["inputs"], np.asarray(cached["labels"]).astype(str)
    record = prep.build_record(inputs, labels, provider, base.audit.FILE[match])
    return inputs, labels, record, base._phase_records(inputs, record), supervised.model_periods(inputs)


CE_STAGE = int(os.environ.get("CE_STAGE", "1024"))
CE_TAG = "" if os.environ.get("V2_POSTPROC_WARM", "cand") == "cand" else os.environ["V2_POSTPROC_WARM"]
CE_OUT = ROOT / "data/experiments" / STUDY / f"ce_emissions_stage{CE_STAGE}{CE_TAG}"
_CE = {}


def predict_ce(inputs, record, provider, seed):
    """Stage-1024 rebuilt CE fold model with its original paired teacher."""
    from experiments import v2_postproc_v1_train as train
    fold = base.audit.FOLD[provider]
    key = (fold, seed)
    if key not in _CE:
        _CE[key] = (joblib.load(train.model_path(f"leave_out_{fold}", seed, CE_STAGE)),
                    joblib.load(base._teacher_path(provider, train.base_seed(seed))))
    record = base.old.inference_record(record)
    record["names"] = list(record["names"])
    model, teacher = _CE[key]
    return base.old.upsample(base.sn.predict(record, model, teacher), record, base.tc.inputs_only(inputs))


def run_ce_cell(job):
    provider, match, view, rep = job
    path = CE_OUT / f"{cell_key(*job)}{os.environ.get('CE_SUFFIX', '')}.npz"
    if path.is_file():
        return str(path)
    inputs, _labels, _record, phases, _ = prepare(provider, match, view, rep)
    arrays = {f"phase__ce{CE_TAG}{CE_STAGE}__s{seed}": np.stack([predict_ce(inputs, rec, provider, seed) for rec in phases]).astype(np.float32)
              for seed in SEEDS}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)
    return str(path)


def run_cell(job):
    provider, match, view, rep = job
    path = OUT / f"{cell_key(*job)}.npz"
    if path.is_file():
        return str(path)
    inputs, labels, _record, phases, model_period = prepare(provider, match, view, rep)
    arrays = {}
    for family in FAMILIES:
        for seed in SEEDS:
            arrays[f"phase__{family}__s{seed}"] = np.stack([
                base.predict_seed(inputs, rec, family, provider, seed) for rec in phases]).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, labels=labels, period=np.asarray(inputs["period"]),
                        timestamps=np.asarray(inputs["timestamps"]), fps=float(inputs["fps"]),
                        model_period=np.asarray(model_period), **arrays)
    tmp.rename(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "3")))
    parser.add_argument("--only", default="")
    parser.add_argument("--family", choices=("base", "ce"), default="base")
    args = parser.parse_args()
    jobs = [c for c in cells() if args.only in cell_key(*c)]
    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(run_ce_cell if args.family == "ce" else run_cell, j): j for j in jobs}
        for f in as_completed(futures):
            print("done", f.result(), flush=True)


if __name__ == "__main__":
    main()
