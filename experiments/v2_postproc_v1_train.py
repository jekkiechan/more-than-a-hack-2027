"""Rebuild provider-excluded stage-1024 CE continuations for `v2_postproc_v1`.

The frontier package's provider-excluded continuation weights were kept only in
private local storage. This runner refits them in a new namespace with the
same recipe: short-original-diversity fold warm start, 512 CE updates at 3e-4
(iterative `ce_control`), then 512 more at 1e-4 (spell-balanced stage 1024),
each with the original-diversity sampler and round-robin four-view v2 TRAIN
windows. Bitwise reproduction of the archived weights is not claimed.
Only TRAIN data is read; D state and normalization stay fixed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import v2_iterative_train as iterative
from experiments import v2_short_original_diversity as recipe

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_postproc_v1"
OUT = ROOT / "data/experiments" / STUDY / "train"
STAGES = ((512, 3e-4), (1024, 1e-4))
# Extra continuation seeds warm-start from the same fold checkpoint/teacher as
# their base seed; only the Torch/dropout and sampler/draw seeds change.
EXTRA_SEEDS = {2126: 2026, 2127: 2027}


def base_seed(seed):
    return EXTRA_SEEDS.get(seed, seed)


def bind():
    recipe.OUT = OUT
    recipe.ARCHIVE_ROOT = Path(os.environ["MTAH_RF_ARCHIVE_DIR"])
    recipe.TRAIN_RECORDS = OUT / "gar"
    target = Path(os.environ["V2_ITERATIVE_GAR_ARCHIVE"])
    link = recipe.TRAIN_RECORDS / "train"
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    return recipe


WARM = os.environ.get("V2_POSTPROC_WARM", "cand")  # cand (original_full) or adop (calibrated D2)


def prior_path(fold, seed):
    if WARM == "adop":
        return (Path(os.environ["MTAH_DT_DIR"]) / "experiments/dense_targeted_followups_v1/artifacts/"
                "control_D2/models" / f"control_D2__{fold.removeprefix('leave_out_')}__s{seed}.joblib")
    root = Path(os.environ["MTAH_SQD_DIR"])
    return (root / "experiments/short_quality_diversity_delivery_v1/artifacts/"
            "short_original_diversity_v1/models" / f"{fold}_{seed}.joblib")


def model_path(fold, seed, stage):
    models = "models" if WARM == "cand" else f"models_{WARM}"
    return OUT / models / f"stage{stage}" / f"{fold}_{seed}.joblib"


def fit_one(job):
    fold, seed = job
    bind()
    iterative.SEEDS = tuple(sorted(set(iterative.SEEDS) | set(EXTRA_SEEDS)))
    teacher = joblib.load(recipe.rf.sc.initial_path(fold, base_seed(seed)))
    old = recipe.training(fold)
    gar = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    checkpoint = joblib.load(prior_path(fold, base_seed(seed)))
    d_state = {k: np.array(v, copy=True) for k, v in checkpoint["d_state"].items()}
    receipts = []
    for stage, lr in STAGES:
        path = model_path(fold, seed, stage)
        if path.exists():
            checkpoint = joblib.load(path)
            continue
        sampler = recipe.short.OriginalDiversitySampler(old, gar, list(teacher["feature_names"]), seed)
        old_lr, old_study = iterative.LR, iterative.STUDY
        iterative.LR, iterative.STUDY = lr, STUDY
        start = time.monotonic()
        try:
            with recipe._round_robin_v2_windows(old):
                with recipe.short.parent.base.installed(sampler):
                    result = iterative.fit_from_checkpoints(
                        teacher, checkpoint, old, gar, seed=seed, arm="ce_control", verbose=True)
        finally:
            iterative.LR, iterative.STUDY = old_lr, old_study
        model = result["model"]
        for key, value in d_state.items():
            np.testing.assert_array_equal(model["d_state"][key], value)
        model.setdefault("source", {})["checkpoint_sha256"] = checkpoint.get("source", {}).get("checkpoint_sha256")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        receipt = dict(study_id=STUDY, fold=fold, seed=seed, stage=stage, lr=lr,
                       wall_seconds=time.monotonic() - start, loss_intervals=result["loss_intervals"],
                       old_matches=sorted({str(x["match"]) for x in old}),
                       gar_games=len({str(x["match"]) for x in gar}),
                       organizer_reference_access=False)
        Path(str(path) + ".json").write_text(json.dumps(receipt, indent=1, default=float) + "\n")
        receipts.append(receipt)
        print("FIT", fold, seed, stage, round(receipt["wall_seconds"], 1), flush=True)
        checkpoint = model
    return receipts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "fit"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", default="2026,2027")
    args = parser.parse_args()
    bind()
    if args.action == "prepare":
        recipe.prepare(workers=args.workers)
        return
    jobs = [(f, int(s)) for f in iterative.FOLDS for s in args.seeds.split(",")]
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(fit_one, j) for j in jobs]):
            future.result()


if __name__ == "__main__":
    main()
