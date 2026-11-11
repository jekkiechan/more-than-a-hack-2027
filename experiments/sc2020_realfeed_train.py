"""Frontier CE continuation with real broadcast-tracking windows (`sc2020_realfeed_v1`).

Continues the reconstructed stage-1024 frontier CE fold checkpoints with the
shared dense objective. A third, optional batch stream draws windows from the
six SkillCorner 2019/20 TRAIN matches; those windows replace GAR windows so the
original-provider half of every batch is unchanged. With no SkillCorner 2020
windows the loop is the shared `v2_iterative_train` objective (parity-tested).

SkillCorner 2020 is treated as SkillCorner for provider exclusion, so the
SkillCorner-held-out fold never sees it. Its three declared evaluation matches
(3518, 3749, 4039) are never loaded. D state and normalization stay fixed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import joblib
import numpy as np

from experiments import sc2020_realfeed_common as C
from experiments import v2_iterative_train as iterative

cap, sn, ol = iterative.cap, iterative.sn, iterative.ol
recipe = C.recipe
SOURCE = "skillcorner2020"
SPLIT = Path(os.environ["MTAH_DATA_DIR"]) / "external/split_skillcorner2020.json"
BATCH = 32
LR = 1e-4
SNAPSHOTS = (256, 512)
SEED_OFFSET = 7919          # fresh Torch/sampler/draw schedule relative to stage 1024
ARMS = {"ctrl": 0, "sc4": 4, "sc8": 8}   # SkillCorner 2020 windows per 16 external windows


def sc_records():
    """Feature records for the six declared SkillCorner 2020 TRAIN matches."""
    from experiments import multisource_eval as ms, preparation as prep, supervised
    cache = C.OUT / "sc2020_records.joblib"
    if cache.is_file():
        return joblib.load(cache)
    split = json.loads(SPLIT.read_text())
    records = []
    for match in split["train"]:
        path = SPLIT.parent / "processed" / f"{match}.npz"
        inputs, labels = ms.view(path)          # unobserved positions masked; <12 visible players -> X
        record = prep.build_record(inputs, labels, SOURCE, match)
        full_y = np.full(len(labels), -1, np.int8)
        for i, label in enumerate(supervised.LABELS):
            full_y[labels == label] = i
        record.update(full_y=full_y, full_period=supervised.model_periods(inputs),
                      fps=float(inputs["fps"]), names=list(record["names"]),
                      source=SOURCE, match=match)
        np.testing.assert_array_equal(record["y"], full_y[record["indices"]])
        records.append(record)
    cache.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(records, cache, compress=3)
    return records


def fit_loop(initial, old_records, gar_records, extra_records, *, seed, extra_count, lr=LR,
             snapshots=SNAPSHOTS, draw_seed=None, verbose=False):
    """Shared dense H/A objective with `extra_count` of 16 external windows from `extra_records`."""
    import torch
    from torch.nn import functional as F
    names, median, scale, input_features = cap._model_input_width(initial)
    old_groups, old_ids = cap._window_index(old_records, names, old=True)
    new_groups, new_ids = cap._window_index(gar_records, names, old=False)
    old_spell, old_pairs, old_weights, _ = cap._old_objective_targets(old_records, require_full_native=True)
    new_spell, new_pairs, new_weights, _ = sn.objective_targets(gar_records, require_full_native=True)
    half = BATCH // 2
    if not 0 <= extra_count < half or (extra_count and not extra_records):
        raise ValueError("extra_count must leave GAR windows and have records")
    if extra_count:
        ex_groups, ex_ids = cap._window_index(extra_records, names, old=False)
        ex_spell, ex_pairs, ex_weights, _ = sn.objective_targets(extra_records, require_full_native=True)
        ex_rng = random.Random(int(seed if draw_seed is None else draw_seed) + 15485863)
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = cap.network(input_features, 96, cap.INPUT_DROPOUT)
    model.load_state_dict({k: torch.as_tensor(v) for k, v in initial["state"].items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cap.WEIGHT_DECAY)
    rng = random.Random(int(seed if draw_seed is None else draw_seed))
    base = dict(median=median, scale=scale, feature_names=names, input_features=input_features)
    plan = ((0, half), (1, half - extra_count), (2, extra_count))
    group_weight = {0: 1.0 - iterative.FRACTION_GAR,
                    1: iterative.FRACTION_GAR * (half - extra_count) / half,
                    2: iterative.FRACTION_GAR * extra_count / half}
    config = dict(study=C.STUDY, seed=int(seed), lr=lr, batch_size=BATCH, extra_source=SOURCE,
                  windows_per_batch=dict(original=half, gar=half - extra_count, extra=extra_count),
                  loss_weights={str(k): v for k, v in group_weight.items()}, train_d=False,
                  warm_start_sha256=cap._checkpoint_sha(initial),
                  teacher_sha256=initial["_original_teacher_sha256"])
    exposures, losses, accum, exported = defaultdict(int), [], [], {}
    model.train()
    for step in range(1, max(snapshots) + 1):
        rows = []
        for group, count in plan:
            for _ in range(count):
                if group == 0:
                    ri, start, size, full, alt = cap._draw_old(rng, old_groups, old_records)
                    record, spell, pairs, weights, swapped = old_records[ri], old_spell, old_pairs, old_weights, False
                    exposures["old/" + "/".join(old_ids[ri])] += 1
                elif group == 1:
                    ri, start, size, full, alt = cap._draw_new(rng, new_groups)
                    record, spell, pairs, weights = gar_records[ri], new_spell, new_pairs, new_weights
                    swapped = rng.random() < 0.5
                    exposures["new/" + new_ids[ri][1]] += 1
                else:
                    game = ex_rng.choice(sorted(ex_groups))
                    ri, start, size, full, _ = ex_rng.choice(ex_groups[game])
                    alt = False
                    record, spell, pairs, weights = extra_records[ri], ex_spell, ex_pairs, ex_weights
                    swapped = ex_rng.random() < 0.5
                    exposures["extra/" + ex_ids[ri][1]] += 1
                x, y = cap._prepare_window(record, start, size, full, alt, median=median, scale=scale, swap=swapped)
                sf = np.ones(cap.CORE, dtype=np.float32); sf[:size] = spell[ri][start:start + size]
                target = np.zeros(cap.CORE, dtype=np.float32); mask = np.zeros(cap.CORE, dtype=bool)
                weight = np.zeros(cap.CORE, dtype=np.float32)
                target[:size] = pairs[ri]["live_target"][start:start + size]
                mask[:size] = pairs[ri]["live_mask"][start:start + size]
                weight[:size] = weights[ri][start:start + size]
                rows.append((x, y, group, sf, target, mask, weight))
        order = list(range(BATCH)); rng.shuffle(order)
        col = lambda i, dtype: torch.as_tensor(np.stack([rows[j][i] for j in order]), dtype=dtype)
        xt, yt, st = col(0, torch.float32), col(1, torch.long), col(3, torch.float32)
        ttar, tmask, twt = col(4, torch.float32), col(5, torch.bool), col(6, torch.float32)
        gt = torch.as_tensor([rows[j][2] for j in order], dtype=torch.int64)
        all_logits = model(xt)
        logits = all_logits[:, cap.CONTEXT:cap.CONTEXT + cap.CORE]
        labelled = (yt == 0) | (yt == 1)
        safe_y = torch.where(labelled, yt, torch.zeros_like(yt))
        row_loss = F.cross_entropy(logits.transpose(1, 2), safe_y, reduction="none")
        count = labelled.sum(dim=1)
        if torch.any(count == 0):
            raise RuntimeError("Every sampled window must have a known H/A label")
        factors = 1.0 + cap.SPELL_COEFFICIENT * (st - 1.0)
        per_window = (row_loss * labelled * factors).sum(dim=1) / count
        loss = 0.0
        for group, _ in plan:
            if group_weight[group] > 0:
                loss = loss + group_weight[group] * per_window[gt == group].mean()
        q_left = torch.softmax(logits, dim=-1)
        q_right = torch.softmax(all_logits[:, cap.CONTEXT + 1:cap.CONTEXT + cap.CORE + 1], dim=-1)
        p_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
        edge_loss = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), ttar, reduction="none")
        for group, _ in plan:
            if group_weight[group] > 0:
                transition = ol.group_balanced_binary_loss(edge_loss, ttar, tmask & (gt == group)[:, None], twt)
                if transition is not None:
                    loss = loss + group_weight[group] * cap.LIVE_TRANSITION_COEFFICIENT * transition
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cap.GRADIENT_CLIP)
        optimizer.step()
        accum.append(float(loss.detach()))
        if step % 32 == 0:
            losses.append(dict(step=step, mean_loss=float(np.mean(accum)))); accum.clear()
        if step in snapshots:
            snap = sn._export(model, None, base=base, teacher_sha=initial["_original_teacher_sha256"],
                              config=dict(config, steps=step), step=step)
            if "d_state" in initial:
                snap["d_state"] = {k: np.array(v, copy=True) for k, v in initial["d_state"].items()}
            snap["metadata"] = dict(exposures=dict(sorted(exposures.items())), loss_intervals=list(losses))
            exported[step] = snap
            model.train()
            if verbose:
                print(f"{C.STUDY} seed={seed} extra={extra_count} step={step} loss={losses[-1]['mean_loss']:.5f}", flush=True)
    return exported


def model_path(arm, fold, seed, step):
    return C.OUT / "models" / f"{arm}_{step}" / f"{fold}_{seed}.joblib"


def fit_one(arm, fold, seed):
    C.configure()
    extra_count = ARMS[arm]
    if extra_count and fold == "leave_out_skillcorner":
        raise ValueError("SkillCorner 2020 is excluded from the SkillCorner-held-out fold; reuse ctrl")
    if all(model_path(arm, fold, seed, s).is_file() for s in SNAPSHOTS):
        return
    teacher = joblib.load(recipe.rf.sc.initial_path(fold, seed))
    checkpoint = joblib.load(C.FRONTIER / "models/stage1024" / f"{fold}_{seed}.joblib")
    old = recipe.training(fold)
    gar = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    extra = sc_records() if extra_count else []
    if any(str(r["source"]).lower().startswith(fold.removeprefix("leave_out_")) for r in list(old) + list(extra)):
        raise ValueError(f"Held-out provider leaked into {fold}")
    initial = iterative.warmstart_state(teacher, checkpoint)
    sampler = recipe.short.OriginalDiversitySampler(old, gar, list(teacher["feature_names"]), seed + SEED_OFFSET)
    start = time.monotonic()
    with recipe._round_robin_v2_windows(old):
        with recipe.short.parent.base.installed(sampler):
            exported = fit_loop(initial, old, gar, extra, seed=seed + SEED_OFFSET,
                                extra_count=extra_count, verbose=True)
    for step, model in exported.items():
        for key, value in checkpoint["d_state"].items():
            np.testing.assert_array_equal(model["d_state"][key], value)
        path = model_path(arm, fold, seed, step)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        Path(str(path) + ".json").write_text(json.dumps(dict(
            study_id=C.STUDY, arm=arm, fold=fold, seed=seed, step=step, config=model["training"],
            exposures=model["metadata"]["exposures"], loss_intervals=model["metadata"]["loss_intervals"],
            wall_seconds=time.monotonic() - start, organizer_reference_access=False), indent=1, default=float) + "\n")
    print("FIT", arm, fold, seed, round(time.monotonic() - start, 1), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("records", "fit", "one"))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--job", default="")
    args = parser.parse_args()
    C.configure()
    if args.action == "records":
        for r in sc_records():
            y = np.asarray(r["y"])
            print(r["match"], len(y), "known H/A rows", int(np.isin(y, (0, 1)).sum()))
        return
    if args.action == "one":
        arm, fold, seed = args.job.split(":")
        fit_one(arm, fold, int(seed))
        return
    # Fresh process per fit, serially: two concurrent fits exhausted memory in the frontier recovery.
    for arm in args.arms.split(","):
        for fold in C.FOLDS:
            if ARMS[arm] and fold == "leave_out_skillcorner":
                continue
            for seed in C.SEEDS:
                subprocess.run([sys.executable, "-m", "experiments.sc2020_realfeed_train", "one",
                                "--job", f"{arm}:{fold}:{seed}"], check=True)


if __name__ == "__main__":
    main()
