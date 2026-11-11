"""Cache per-phase development emissions for `sc2020_realfeed_v1` fold models.

SkillCorner cells of SkillCorner-2020 arms reuse the control emission because
that fold never sees the new data. Labels never enter inference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

import joblib
import numpy as np

from experiments import sc2020_realfeed_common as C
from experiments import sc2020_realfeed_score as S
from experiments import sc2020_realfeed_train as T


def emit(job):
    arm, step, cell = job
    C.configure()
    provider = cell[0]
    key = C.cell_key(*cell)
    name = f"{arm}_{step}"
    path = S.emission_path(name, key)
    if path.is_file():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fold = f"leave_out_{C.base.audit.FOLD[provider]}"
    if arm == T.REBUILD_ARM and fold == "leave_out_skillcorner":
        # The rebuild's SkillCorner fold is the approved frontier CE itself.
        cell = C.frontier_cell(key)
        count = sum(1 for k in cell if k.startswith("prob__ce__seed2026__phase"))
        np.savez_compressed(path.with_suffix(".tmp.npz"), **{
            f"seed{seed}": np.stack([cell[f"prob__ce__seed{seed}__phase{ph}"] for ph in range(count)])
            for seed in C.SEEDS})
        path.with_suffix(".tmp.npz").rename(path)
        return str(path)
    if T.ARMS[arm] and fold == "leave_out_skillcorner":
        source = S.emission_path(f"ctrl_{step}", key)
        if not source.is_file():
            raise FileNotFoundError(f"Emit ctrl first: {source}")
        os.link(source, path)
        return str(path)
    payload = C.prepared(*cell)
    arrays = {}
    for seed in C.SEEDS:
        model = joblib.load(T.model_path(arm, fold, seed, step))
        teacher = joblib.load(C.base._teacher_path(provider, seed))
        arrays[f"seed{seed}"] = C.predict(payload, model, teacher).astype(np.float32)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="ctrl,sc4,sc8")
    parser.add_argument("--steps", default="256,512")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    for arm in args.arms.split(","):          # ctrl first so SkillCorner cells can be linked
        jobs = [(arm, int(step), cell) for step in args.steps.split(",") for cell in C.cells()]
        with ProcessPoolExecutor(args.workers) as pool:
            for future in as_completed([pool.submit(emit, j) for j in jobs]):
                future.result()
        print("EMITTED", arm, flush=True)


if __name__ == "__main__":
    main()
