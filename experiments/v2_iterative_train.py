"""Warm-started H/A boundary-loss continuation study for feed-like v2.

This module is intentionally a trainer component only. It reuses the frozen
v2 short-original-diversity TRAIN preparation and the shared window/objective
helpers, while adding one bounded H/A transition-margin term. It never reads
evaluation labels or changes the D model.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import random
from typing import Any

import numpy as np

from experiments import objective_losses as ol
from experiments import soccernet_capacity_model as cap
from experiments import sn_dense_train as sn

STUDY = "v2_iterative_train"
FOLDS = ("leave_out_metrica", "leave_out_skillcorner", "leave_out_sportec")
SEEDS = (2026, 2027)
ARMS = ("ce_control", "boundary_margin")
STEPS = 512
BATCH_SIZE = 32
FRACTION_GAR = 0.5
LR = 3e-4
BOUNDARY_COEFFICIENT = 0.1
BOUNDARY_MARGIN = 1.0
BOUNDARY_RADIUS = 1


def warmstart_state(teacher: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Build an H/A initialization from a fold-matched archived checkpoint.

    Normalization and feature schema must be identical to the original teacher;
    this prevents silently applying a trained GRU to a different coordinate
    system. The returned object carries no D state, so fitting cannot update it.
    """
    if checkpoint.get("kind") != "soccernet_capacity" or int(checkpoint.get("hidden", -1)) != 96:
        raise ValueError("Warm-start must be a width-96 archived capacity model")
    for key in ("feature_names",):
        if list(checkpoint.get(key, ())) != list(teacher.get(key, ())):
            raise ValueError(f"Warm-start {key} differs from original teacher")
    for key in ("median", "scale"):
        left, right = np.asarray(checkpoint.get(key)), np.asarray(teacher.get(key))
        if left.shape != right.shape or not np.array_equal(left, right):
            raise ValueError(f"Warm-start {key} differs from original teacher")
    state = checkpoint.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Warm-start checkpoint has no H/A state")
    initialized = dict(teacher)
    initialized["_original_teacher_sha256"] = cap._checkpoint_sha(teacher)
    initialized["state"] = {name: np.array(value, copy=True) for name, value in state.items()}
    if "d_state" in checkpoint:
        initialized["d_state"] = {name: np.array(value, copy=True)
                                  for name, value in checkpoint["d_state"].items()}
    initialized["median"] = np.array(checkpoint["median"], copy=True)
    initialized["scale"] = np.array(checkpoint["scale"], copy=True)
    initialized["feature_names"] = list(checkpoint["feature_names"])
    return initialized


def boundary_rows(targets: np.ndarray, radius: int = BOUNDARY_RADIUS,
                  periods: np.ndarray | None = None,
                  valid_edges: np.ndarray | None = None) -> np.ndarray:
    """Mark known H/A rows within ``radius`` samples of a true H/A edge.

    Edges require two adjacent known H/A labels with different teams. D, X,
    unknown, padding and separated segments break the mask naturally.
    """
    y = np.asarray(targets)
    if y.ndim < 1 or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("targets must have a time axis and radius must be nonnegative")
    if not np.isin(y, (-1, 0, 1, 2)).all():
        raise ValueError("targets may contain only H/A/D/unknown ids")
    p = None if periods is None else np.asarray(periods)
    if p is not None and p.shape != y.shape:
        raise ValueError("periods must match targets")
    ve = None if valid_edges is None else np.asarray(valid_edges, dtype=bool)
    expected_edges = y.shape[:-1] + (max(y.shape[-1] - 1, 0),)
    if ve is not None and ve.shape != expected_edges:
        raise ValueError("valid_edges must match the target adjacent-pair shape")
    result = np.zeros(y.shape, dtype=bool)
    if y.shape[-1] < 2:
        return result
    same_period = np.ones(y.shape[:-1] + (y.shape[-1] - 1,), dtype=bool)
    if p is not None:
        same_period = p[..., :-1] == p[..., 1:]
    if ve is not None:
        same_period &= ve
    edges = (same_period & np.isin(y[..., :-1], (0, 1)) &
             np.isin(y[..., 1:], (0, 1)) & (y[..., :-1] != y[..., 1:]))
    # Edge j lies between rows j and j+1. Expand one model sample on both
    # sides of that edge, then intersect with known H/A rows.
    for offset in range(-radius, radius + 2):
        source_start = max(0, -offset)
        source_stop = min(edges.shape[-1], y.shape[-1] - offset)
        if source_stop <= source_start:
            continue
        row_start = source_start + offset
        row_stop = source_stop + offset
        candidate = edges[..., source_start:source_stop]
        if p is not None:
            candidate = candidate & (p[..., row_start:row_stop] ==
                                     p[..., source_start:source_stop])
        result[..., row_start:row_stop] |= candidate
    return result & np.isin(y, (0, 1))


def boundary_margin_loss(logits, targets, *, margin: float = BOUNDARY_MARGIN,
                         periods=None, valid_edges=None):
    """Mean softplus margin loss on boundary-local rows; zero when no edge exists."""
    import torch
    from torch.nn import functional as F

    if logits.ndim != 3 or logits.shape[-1] != 2 or tuple(logits.shape[:2]) != tuple(targets.shape):
        raise ValueError("logits must be [batch,time,2] and match target shape")
    if not np.isfinite(float(margin)) or margin <= 0:
        raise ValueError("margin must be positive and finite")
    periods_np = None if periods is None else periods.detach().cpu().numpy()
    edges_np = None if valid_edges is None else valid_edges.detach().cpu().numpy()
    mask_np = boundary_rows(targets.detach().cpu().numpy(), periods=periods_np,
                            valid_edges=edges_np)
    mask = torch.as_tensor(mask_np, dtype=torch.bool, device=logits.device)
    known = (targets == 0) | (targets == 1)
    mask &= known
    if not bool(mask.any()):
        return logits.sum() * 0.0
    safe_target = targets.clamp(0, 1)
    correct = logits.gather(-1, safe_target.unsqueeze(-1)).squeeze(-1)
    other = logits.gather(-1, (1 - safe_target).unsqueeze(-1)).squeeze(-1)
    return F.softplus(float(margin) - (correct - other))[mask].mean()


def _fit_loop(initial: dict[str, Any], old_records: list[dict], gar_records: list[dict], *,
              seed: int, arm: str, verbose: bool = False,
              steps: int = STEPS) -> dict[str, Any]:
    """Fit one warm-started H/A component with the shared dense objective."""
    if arm not in ARMS or seed not in SEEDS or int(steps) != steps or steps <= 0 or steps > STEPS:
        raise ValueError("Use a declared arm and seed")
    names, median, scale, input_features = cap._model_input_width(initial)
    old_groups, old_ids = cap._window_index(old_records, names, old=True)
    new_groups, new_ids = cap._window_index(gar_records, names, old=False)
    if not old_groups or not new_groups:
        raise ValueError("Both provider-excluded and GAR TRAIN windows are required")
    old_spell, old_pairs, old_weights, has_native = cap._old_objective_targets(
        old_records, require_full_native=True)
    new_spell, new_pairs, new_weights, _ = sn.objective_targets(
        gar_records, require_full_native=True)

    import torch
    from torch.nn import functional as F
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    model = cap.network(input_features, 96, cap.INPUT_DROPOUT)
    state = initial.get("state", {})
    expected = model.state_dict()
    if set(state) != set(expected):
        raise ValueError("Warm-start state keys do not match the width-96 H/A model")
    for key, target in expected.items():
        if tuple(np.asarray(state[key]).shape) != tuple(target.shape):
            raise ValueError(f"Warm-start state shape mismatch for {key}")
    model.load_state_dict({key: torch.as_tensor(value) for key, value in state.items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=cap.WEIGHT_DECAY)
    rng = random.Random(int(seed))
    base = dict(median=median, scale=scale, feature_names=names,
                input_features=input_features)
    config = dict(study=STUDY, arm=arm, width=96, seed=int(seed), steps=int(steps),
                  batch_size=BATCH_SIZE, fraction_gar=FRACTION_GAR, lr=LR,
                  input_dropout=cap.INPUT_DROPOUT, weight_decay=cap.WEIGHT_DECAY,
                  gradient_clip=cap.GRADIENT_CLIP,
                  spell_coefficient=cap.SPELL_COEFFICIENT,
                  live_transition_coefficient=cap.LIVE_TRANSITION_COEFFICIENT,
                  warm_start="short_original_diversity_v1 fold/seed checkpoint",
                  warm_start_sha256=cap._checkpoint_sha(initial),
                  teacher_sha256=initial["_original_teacher_sha256"],
                  train_d=False, boundary_coefficient=(BOUNDARY_COEFFICIENT if arm == "boundary_margin" else 0.0),
                  boundary_margin=BOUNDARY_MARGIN, boundary_radius_samples=BOUNDARY_RADIUS,
                  require_full_native=has_native)
    exposures: dict[str, int] = defaultdict(int)
    losses: list[dict[str, float]] = []
    accum: list[float] = []
    old_count = new_count = BATCH_SIZE // 2
    model.train()
    for step in range(1, int(steps) + 1):
        bx, by, bg, bs, bt, bm, bw, bp, be = [], [], [], [], [], [], [], [], []
        for group, count in ((0, old_count), (1, new_count)):
            for _ in range(count):
                if group == 0:
                    ri, start, size, full, alt = cap._draw_old(rng, old_groups, old_records)
                    record = old_records[ri]
                    spell, pairs, weights = old_spell, old_pairs, old_weights
                    swapped = False
                    exposures["old/" + "/".join(old_ids[ri])] += 1
                else:
                    ri, start, size, full, alt = cap._draw_new(rng, new_groups)
                    record = gar_records[ri]
                    spell, pairs, weights = new_spell, new_pairs, new_weights
                    swapped = rng.random() < 0.5
                    exposures["new/" + new_ids[ri][1]] += 1
                x, y = cap._prepare_window(record, start, size, full, alt,
                    median=median, scale=scale, swap=swapped)
                sf = np.ones(cap.CORE, dtype=np.float32); sf[:size] = spell[ri][start:start + size]
                target = np.zeros(cap.CORE, dtype=np.float32)
                mask = np.zeros(cap.CORE, dtype=bool)
                weight = np.zeros(cap.CORE, dtype=np.float32)
                period = np.full(cap.CORE, -1, dtype=np.int64)
                valid_edges = np.zeros(max(cap.CORE - 1, 0), dtype=bool)
                target[:size] = pairs[ri]["live_target"][start:start + size]
                mask[:size] = pairs[ri]["live_mask"][start:start + size]
                weight[:size] = weights[ri][start:start + size]
                period[:size] = np.asarray(record["period"])[start:start + size]
                valid_edges[:max(size - 1, 0)] = pairs[ri]["live_mask"][start:start + max(size - 1, 0)]
                bx.append(x); by.append(y); bg.append(group); bs.append(sf)
                bt.append(target); bm.append(mask); bw.append(weight); bp.append(period); be.append(valid_edges)
        order = list(range(BATCH_SIZE)); rng.shuffle(order)
        xt = torch.as_tensor(np.stack([bx[i] for i in order]), dtype=torch.float32)
        yt = torch.as_tensor(np.stack([by[i] for i in order]), dtype=torch.long)
        gt = torch.as_tensor([bg[i] for i in order], dtype=torch.int64)
        st = torch.as_tensor(np.stack([bs[i] for i in order]), dtype=torch.float32)
        ttar = torch.as_tensor(np.stack([bt[i] for i in order]), dtype=torch.float32)
        tmask = torch.as_tensor(np.stack([bm[i] for i in order]), dtype=torch.bool)
        twt = torch.as_tensor(np.stack([bw[i] for i in order]), dtype=torch.float32)
        pt = torch.as_tensor(np.stack([bp[i] for i in order]), dtype=torch.long)
        et = torch.as_tensor(np.stack([be[i] for i in order]), dtype=torch.bool)
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
        old_loss, new_loss = per_window[gt == 0].mean(), per_window[gt == 1].mean()
        loss = (1.0 - FRACTION_GAR) * old_loss + FRACTION_GAR * new_loss

        q_left = torch.softmax(logits, dim=-1)
        q_right = torch.softmax(all_logits[:, cap.CONTEXT + 1:cap.CONTEXT + cap.CORE + 1], dim=-1)
        p_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
        edge_loss = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), ttar, reduction="none")
        for group, group_weight in ((0, 1.0 - FRACTION_GAR), (1, FRACTION_GAR)):
            tm = tmask & (gt == group)[:, None]
            transition = ol.group_balanced_binary_loss(edge_loss, ttar, tm, twt)
            if transition is not None:
                loss = loss + group_weight * cap.LIVE_TRANSITION_COEFFICIENT * transition
        if arm == "boundary_margin":
            loss = loss + BOUNDARY_COEFFICIENT * boundary_margin_loss(
                logits, yt, periods=pt, valid_edges=et)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cap.GRADIENT_CLIP)
        optimizer.step()
        accum.append(float(loss.detach()))
        if step % 32 == 0 or step == steps:
            losses.append(dict(step=step, mean_loss=float(np.mean(accum))))
            accum.clear()
        if verbose and (step % 128 == 0 or step == steps):
            print(f"{STUDY} arm={arm} seed={seed} step={step}/{steps} loss={loss.item():.5g}", flush=True)
    exported = sn._export(model, None, base=base,
        teacher_sha=initial["_original_teacher_sha256"], config=config, step=int(steps))
    if "d_state" in initial:
        exported["d_state"] = {name: np.array(value, copy=True)
                               for name, value in initial["d_state"].items()}
    exported["metadata"] = dict(exposures=dict(sorted(exposures.items())), loss_intervals=losses)
    return dict(model=exported, config=config, exposures=dict(exposures), loss_intervals=losses,
                warm_start_sha256=cap._checkpoint_sha(initial))


def fit_from_checkpoints(teacher: dict[str, Any], checkpoint: dict[str, Any],
                         old_records: list[dict], gar_records: list[dict], *,
                         seed: int, arm: str, verbose: bool = True) -> dict[str, Any]:
    """Orchestrator API: train from paired, verified in-memory fold artifacts."""
    initial = warmstart_state(teacher, checkpoint)
    return _fit_loop(initial, old_records, gar_records, seed=seed, arm=arm, verbose=verbose)


def fit(fold: str, seed: int, arm: str, *, verbose: bool = True) -> dict[str, Any]:
    """Explicit fit entry point. Caller must freeze and verify its declaration first."""
    if fold not in FOLDS or seed not in SEEDS or arm not in ARMS:
        raise ValueError("Use only the fixed folds, seeds and arms")
    from experiments import feed_robustness_v1 as rf
    from experiments import short_original_diversity_v1 as short
    from experiments import v2_short_original_diversity as v2
    import joblib
    teacher = joblib.load(rf.sc.initial_path(fold, seed))
    prior = joblib.load(short.OUT / "models" / f"{fold}_{seed}.joblib")
    old_records = v2.training(fold)
    gar_records = rf.dl.load_dense_records("train", output=v2.TRAIN_RECORDS)
    return fit_from_checkpoints(teacher, prior, old_records, gar_records,
                                seed=seed, arm=arm, verbose=verbose)
