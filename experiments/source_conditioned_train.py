"""C/H/HD training and inference integration for source-conditioned heads.

Frozen D2 remains the C trainer. This module implements only H/HD using its
window sampler, losses, normalization, initialization and update schedule.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import random

import numpy as np

from experiments import (factorized_had as fh, learned_temporal as lt,
                         objective_losses as ol, source_conditioned_heads as sch,
                         soccernet_capacity_model as cap, sn_dense_train as sn,
                         supervised)

CORE, CONTEXT = cap.CORE, cap.CONTEXT


def _head_params(net):
    return list(net.ha_sequence.parameters()) + [p for h in net.ha.values() for p in h.parameters()]


def _dead_params(net):
    return list(net.dead_sequence.parameters()) + [p for h in net.dead.values() for p in h.parameters()]


def _export(net, initial, config, step):
    return dict(kind="source_conditioned_dense", arm=config["arm"],
                state=sch.export_state(net), hidden=int(net.ha_sequence.hidden_size),
                d_hidden=int(net.dead_sequence.hidden_size),
                input_dropout=float(net.ha_input_dropout.p),
                separate_ha=sch.SOURCE_CONDITIONED_HA[config["arm"]],
                separate_d=sch.SOURCE_CONDITIONED_D[config["arm"]],
                median=np.array(initial["median"], copy=True),
                scale=np.array(initial["scale"], copy=True),
                feature_names=list(initial["feature_names"]),
                input_features=int(initial["input_features"]),
                player_indices=list(initial["player_indices"]),
                source=dict(kind="factorized_had", checkpoint_sha256=cap._checkpoint_sha(initial)),
                step=int(step), training=copy.deepcopy(config))


def train(initial_model: dict, old_records: list[dict], new_records: list[dict], *,
          arm: str, seed: int, steps: int = 2048, fraction: float = 0.5,
          batch_size: int = 32, width: int = 96, d_width: int = 24,
          lr: float = 3e-4, d_lr: float = 1e-4,
          input_dropout: float = 0.1, require_full_native: bool = True,
          log_every: int = 256, verbose: bool = True) -> dict:
    """Fit one H or HD model with the frozen D2 batch/loss recipe."""
    if arm not in ("C", "H", "HD"):
        raise ValueError("Source-conditioned trainer supports C, H or HD")
    if int(steps) != steps or steps <= 0 or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("steps and batch_size must be positive integers")
    if not 0 <= float(fraction) <= 1:
        raise ValueError("fraction must be in [0, 1]")
    new_count = int(round(batch_size * float(fraction)))
    if not np.isclose(new_count / batch_size, fraction, atol=1e-12, rtol=0):
        raise ValueError("batch_size * fraction must be an integer")
    old_count, new_count = batch_size - new_count, new_count
    if (old_count and not old_records) or (new_count and not new_records):
        raise ValueError("Both source record lists are required at the declared fraction")

    names, median, scale, input_features = cap._model_input_width(initial_model)
    if not bool(initial_model.get("player_only", False)):
        raise ValueError("The D2 fold teacher must provide the player-only D pathway")
    old_groups, old_ids, old_spell, old_pairs, old_weights = {}, [], [], [], []
    if old_count:
        old_groups, old_ids = cap._window_index(old_records, names, old=True)
        if not old_groups:
            raise ValueError("Original records contain no H/A-supervised windows")
        old_spell, old_pairs, old_weights, _ = cap._old_objective_targets(
            old_records, require_full_native=require_full_native)
    new_groups, new_ids = {}, []
    if new_count:
        new_groups, new_ids = cap._window_index(new_records, names, old=False)
        if not new_groups:
            raise ValueError("SoccerNet records contain no H/A-supervised windows")
        new_spell, new_pairs, new_weights, _ = sn.objective_targets(
            new_records, require_full_native=True)

    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    net = sch.network(input_features, int(width), list(initial_model["player_indices"]),
                      d_hidden=int(d_width), input_dropout=float(input_dropout),
                      separate_ha=sch.SOURCE_CONDITIONED_HA[arm],
                      separate_d=sch.SOURCE_CONDITIONED_D[arm])
    sch.initialize_from_teacher(net, initial_model)
    ha_optimizer = torch.optim.AdamW(_head_params(net), lr=float(lr),
                                     weight_decay=cap.WEIGHT_DECAY)
    d_optimizer = torch.optim.AdamW(_dead_params(net), lr=float(d_lr),
                                    weight_decay=cap.WEIGHT_DECAY)
    rng = random.Random(int(seed))
    config = dict(arm=arm, seed=int(seed), steps=int(steps), fraction=float(fraction),
        batch_size=int(batch_size), width=int(width), d_width=int(d_width), lr=float(lr),
        d_lr=float(d_lr), input_dropout=float(input_dropout), weight_decay=cap.WEIGHT_DECAY,
        gradient_clip=cap.GRADIENT_CLIP, ha_target="dense", train_d=True,
        spell_coefficient=cap.SPELL_COEFFICIENT,
        live_transition_coefficient=cap.LIVE_TRANSITION_COEFFICIENT,
        teacher_sha256=cap._checkpoint_sha(initial_model),
        old_windows=sum(map(len, old_groups.values())),
        new_windows=sum(map(len, new_groups.values())),
        old_windows_per_batch=old_count, new_windows_per_batch=new_count,
        initialization="fold teacher widened HA; exact player-only teacher D; GAR projections cloned",
        torch_rng_draw_parity="HA GRU/output and full-width D GRU/3-logit initialization draw order; duplicate heads deep-copied")

    exposures = defaultdict(int)
    label_counts = {"target": defaultdict(int), "soccernet": defaultdict(int)}
    gradient_diagnostics = []
    loss_intervals, accum = [], defaultdict(list)
    net.train()
    for step in range(1, int(steps) + 1):
        bx, by, bg, bs, bt, bm, bw = [], [], [], [], [], [], []
        for group, count in ((0, old_count), (1, new_count)):
            for _ in range(count):
                if group == 0:
                    ri, start, size, full, alt = cap._draw_old(rng, old_groups, old_records)
                    record, swapped = old_records[ri], False
                else:
                    ri, start, size, full, alt = cap._draw_new(rng, new_groups)
                    record = new_records[ri]
                    swapped = rng.random() < 0.5
                x, y = cap._prepare_window(record, start, size, full, alt,
                    median=median, scale=scale, swap=swapped)
                if not np.isin(y, (0, 1)).any():
                    raise RuntimeError("Sampler emitted a window without an H/A anchor")
                spell = np.ones(CORE, dtype=np.float32)
                tt = np.zeros(CORE, dtype=np.float32)
                tm = np.zeros(CORE, dtype=bool)
                tw = np.zeros(CORE, dtype=np.float32)
                if group == 0:
                    spell[:size] = old_spell[ri][start:start + size]
                    pr = old_pairs[ri]
                    tt[:size] = pr["live_target"][start:start + size]
                    tm[:size] = pr["live_mask"][start:start + size]
                    tw[:size] = old_weights[ri][start:start + size]
                    exposures["old/" + "/".join(old_ids[ri])] += 1
                else:
                    spell[:size] = new_spell[ri][start:start + size]
                    pr = new_pairs[ri]
                    tt[:size] = pr["live_target"][start:start + size]
                    tm[:size] = pr["live_mask"][start:start + size]
                    tw[:size] = new_weights[ri][start:start + size]
                    exposures["new/" + new_ids[ri][1]] += 1
                bx.append(x); by.append(y); bg.append(group); bs.append(spell)
                bt.append(tt); bm.append(tm); bw.append(tw)
        order = list(range(batch_size)); rng.shuffle(order)
        xt = torch.as_tensor(np.stack([bx[i] for i in order]), dtype=torch.float32)
        yt = torch.as_tensor(np.stack([by[i] for i in order]), dtype=torch.long)
        gt = torch.as_tensor([bg[i] for i in order], dtype=torch.int64)
        st = torch.as_tensor(np.stack([bs[i] for i in order]), dtype=torch.float32)
        ttar = torch.as_tensor(np.stack([bt[i] for i in order]), dtype=torch.float32)
        tmask = torch.as_tensor(np.stack([bm[i] for i in order]), dtype=torch.bool)
        twt = torch.as_tensor(np.stack([bw[i] for i in order]), dtype=torch.float32)

        outputs = net(xt, gt == 1)
        logits = outputs["ha"][:, CONTEXT:CONTEXT + CORE]
        dead = outputs["d"][:, CONTEXT:CONTEXT + CORE]
        labelled = (yt == 0) | (yt == 1)
        safe_y = torch.where(labelled, yt, torch.zeros_like(yt))
        row_loss = F.cross_entropy(logits.transpose(1, 2), safe_y, reduction="none")
        count = labelled.sum(dim=1)
        if torch.any(count == 0):
            raise RuntimeError("Every sampled window must contain a known H/A label")
        factors = 1.0 + cap.SPELL_COEFFICIENT * (st - 1.0)
        per_window = (row_loss * labelled * factors).sum(dim=1) / count
        old_mask, new_mask = gt == 0, gt == 1
        terms = []
        source_loss = {"target": None, "soccernet": None}
        old_loss = per_window[old_mask].mean() if old_count else None
        new_loss = per_window[new_mask].mean() if new_count else None
        if old_loss is not None:
            old_ha_term = (1.0 - fraction) * old_loss if new_count else old_loss
            terms.append(old_ha_term)
            source_loss["target"] = old_ha_term
        if new_loss is not None:
            new_ha_term = fraction * new_loss if old_count else new_loss
            terms.append(new_ha_term)
            source_loss["soccernet"] = new_ha_term
        loss = terms[0] if len(terms) == 1 else terms[0] + terms[1]

        q_left = torch.softmax(outputs["ha"][:, CONTEXT:CONTEXT + CORE], dim=-1)
        q_right = torch.softmax(outputs["ha"][:, CONTEXT + 1:CONTEXT + CORE + 1], dim=-1)
        p_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
        edge = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6), ttar, reduction="none")
        if old_count:
            t_old = ol.group_balanced_binary_loss(edge, ttar, tmask & old_mask[:, None], twt)
            if t_old is not None:
                old_transition = (1.0 - fraction) * cap.LIVE_TRANSITION_COEFFICIENT * t_old
                loss = loss + old_transition
                source_loss["target"] = (source_loss["target"] + old_transition
                                          if source_loss["target"] is not None else old_transition)
        if new_count:
            t_new = ol.group_balanced_binary_loss(edge, ttar, tmask & new_mask[:, None], twt)
            if t_new is not None:
                new_transition = fraction * cap.LIVE_TRANSITION_COEFFICIENT * t_new
                loss = loss + new_transition
                source_loss["soccernet"] = (source_loss["soccernet"] + new_transition
                                            if source_loss["soccernet"] is not None else new_transition)

        known = yt >= 0
        bce = F.binary_cross_entropy_with_logits(dead, (yt == 2).float(), reduction="none")
        per_d = (bce * known).sum(dim=1) / known.sum(dim=1).clamp(min=1)
        has = known.any(dim=1)
        d_terms = []
        d_source_terms = {"target": None, "soccernet": None}
        if (old_mask & has).any():
            d_source_terms["target"] = ((1.0 - fraction) if new_count else 1.0) * per_d[old_mask & has].mean()
            d_terms.append(d_source_terms["target"])
        if (new_mask & has).any():
            d_source_terms["soccernet"] = (fraction if old_count else 1.0) * per_d[new_mask & has].mean()
            d_terms.append(d_source_terms["soccernet"])
        d_value = sum(d_terms)
        loss = loss + d_value
        for source in ("target", "soccernet"):
            if d_source_terms[source] is not None:
                source_loss[source] = (source_loss[source] + d_source_terms[source]
                                       if source_loss[source] is not None else d_source_terms[source])
        label_counts["target"]["ha_frames"] += int(((yt == 0) | (yt == 1))[old_mask].sum())
        label_counts["target"]["d_loss_valid_frames"] += int((yt >= 0)[old_mask].sum())
        label_counts["target"]["dead_class_frames"] += int((yt == 2)[old_mask].sum())
        label_counts["soccernet"]["ha_frames"] += int(((yt == 0) | (yt == 1))[new_mask].sum())
        label_counts["soccernet"]["d_loss_valid_frames"] += int((yt >= 0)[new_mask].sum())
        label_counts["soccernet"]["dead_class_frames"] += int((yt == 2)[new_mask].sum())
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        ha_optimizer.zero_grad(set_to_none=True)
        d_optimizer.zero_grad(set_to_none=True)
        if step == 1 or step % 512 == 0 or step == steps:
            shared = {"ha_encoder": list(net.ha_sequence.parameters()),
                      "d_encoder": list(net.dead_sequence.parameters())}
            diagnostic = {"step": step, "sources": {}}
            for source, component in source_loss.items():
                diagnostic["sources"][source] = {}
                for name, params in shared.items():
                    grad = torch.autograd.grad(component, params, retain_graph=True,
                                               allow_unused=True) if component is not None else ()
                    norm = float(torch.sqrt(sum((g.detach().square().sum() for g in grad
                                                  if g is not None), torch.tensor(0.0))))
                    diagnostic["sources"][source][name] = norm
            gradient_diagnostics.append(diagnostic)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(_head_params(net), cap.GRADIENT_CLIP)
        ha_optimizer.step()
        torch.nn.utils.clip_grad_norm_(_dead_params(net), cap.GRADIENT_CLIP)
        d_optimizer.step()
        accum["loss"].append(float(loss.detach()))
        if old_loss is not None: accum["old_loss"].append(float(old_loss.detach()))
        if new_loss is not None: accum["new_loss"].append(float(new_loss.detach()))
        accum["d_loss"].append(float(d_value.detach()))
        if step % 32 == 0 or step == steps:
            loss_intervals.append(dict(step=step, **{k: float(np.mean(v)) for k, v in accum.items()}))
            accum.clear()
        if verbose and log_every and step % log_every == 0:
            print(f"SOURCE_HEAD arm={arm} seed={seed} step={step}/{steps} loss={float(loss.detach()):.5g}", flush=True)

    model = _export(net, initial_model, config, steps)
    model["metadata"] = dict(exposures=dict(sorted(exposures.items())),
        effective_label_counts={k: dict(v) for k, v in label_counts.items()},
        shared_encoder_gradient_norms=gradient_diagnostics, loss_intervals=loss_intervals)
    return dict(model=model, final=model, config=config, loss_intervals=loss_intervals,
                exposures=dict(exposures), steps_run=int(steps))


def predict(record: dict, model: dict) -> np.ndarray:
    """Target-route H/A/D inference with D2 chunks, frozen normalization and team symmetry."""
    import torch

    if list(record["names"]) != list(model["feature_names"]):
        raise ValueError("Feature schema differs from source-conditioned model")
    feature_record = {key: record[key] for key in ("x", "names", "period")}
    for optional in ("timestamps", "fps"):
        if optional in record: feature_record[optional] = record[optional]
    windows = cap._record_chunks(feature_record)
    net = sch.network(model["input_features"], model["hidden"], model["player_indices"],
                      d_hidden=model["d_hidden"], input_dropout=model["input_dropout"],
                      separate_ha=model["separate_ha"], separate_d=model["separate_d"])
    net.load_state_dict({k: torch.as_tensor(v) for k, v in model["state"].items()})
    net.eval()
    x_raw = np.asarray(feature_record["x"])
    n = len(x_raw)
    ha_views, dead_views = [], []
    with torch.inference_mode():
        for swap in (False, True):
            raw = supervised.swap_team_features(x_raw, feature_record["names"]) if swap else x_raw
            normalized = lt.normalize(raw, model["median"], model["scale"])
            conditional = np.empty((n, 2), dtype=np.float32)
            dead_probability = np.empty(n, dtype=np.float32)
            for offset in range(0, len(windows), 64):
                batch = windows[offset:offset + 64]
                xb = torch.as_tensor(np.stack([normalized[full] for _, _, full in batch]),
                                     dtype=torch.float32)
                outputs = net(xb, sch.TARGET)
                q = torch.softmax(outputs["ha"][:, CONTEXT:CONTEXT + CORE], dim=-1).cpu().numpy()
                d = torch.nn.functional.logsigmoid(
                    outputs["d"][:, CONTEXT:CONTEXT + CORE]).exp().cpu().numpy()
                for i, (start, size, _) in enumerate(batch):
                    conditional[start:start + size] = q[i, :size]
                    dead_probability[start:start + size] = d[i, :size]
            ha_views.append(conditional[:, [1, 0]] if swap else conditional)
            dead_views.append(dead_probability)
    q = (ha_views[0] + ha_views[1]) * 0.5
    d = (dead_views[0] + dead_views[1]) * 0.5
    result = np.empty((n, 3), dtype=np.float32)
    result[:, :2] = q * (1 - d[:, None])
    result[:, 2] = d
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-6, rtol=0):
        raise RuntimeError("Invalid source-conditioned probabilities")
    return result
