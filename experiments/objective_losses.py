"""Isolated objective ablations for the factorized H/A/D GRU.

The network and inference contract are exactly the ones in
``factorized_had``. This module changes only the training loss: complete
training-only H/A spell duration weights and a probability-based transition
surrogate. The transition terms use marginal D probability from the
player-only D path and conditional H/A probabilities; they add no head.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt


REFERENCE_SECONDS = 5.0
SPELL_ALPHA = 0.5
SPELL_MIN_WEIGHT = 0.5
SPELL_MAX_WEIGHT = 2.0
TRANSITION_COEFFICIENT = 0.1
OBJECTIVES = {
    "control": (0.0, 0.0),
    "spell_sqrt": (1.0, 0.0),
    "transition": (0.0, TRANSITION_COEFFICIENT),
    "combined": (1.0, TRANSITION_COEFFICIENT),
}


def capped_spell_weight(duration_seconds):
    """Raw pre-normalization factor for one complete live spell."""
    duration = float(duration_seconds)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("spell duration must be positive and finite")
    return float(np.clip((REFERENCE_SECONDS / duration) ** SPELL_ALPHA,
                         SPELL_MIN_WEIGHT, SPELL_MAX_WEIGHT))


def _labels(values, name):
    """Return canonical H/A/D ids (0/1/2), with unknown represented by -1."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.dtype.kind in "USO":
        table = {"H": 0, "A": 1, "D": 2, "X": -1, "-1": -1}
        try:
            result = np.asarray([table[str(value)] for value in values], dtype=np.int8)
        except KeyError as exc:
            raise ValueError(f"Unknown label in {name}: {exc.args[0]}") from exc
    else:
        if not np.isin(values, (-1, 0, 1, 2)).all():
            raise ValueError(f"{name} must contain only -1, 0, 1, 2")
        result = values.astype(np.int8, copy=False)
    return result


def spell_duration_factors(full_y, full_period, indices, sampled_y, fps):
    """Map full-resolution, training-only spell weights onto sampled rows.

    Only fully observed H/A runs receive ``clip(sqrt(5 s / duration), .5, 2)``.
    A run touching a time-block edge or an unknown target is censored and keeps
    factor 1. D and unknown rows also keep factor 1; the trainer separately
    gives unknown rows zero NLL weight. Only eligible complete-spell factors
    are normalized to mean one, preserving D, censored-live, and total
    eligible live-team mass.

    ``full_period`` must already encode every match/half/time discontinuity,
    as ``supervised.model_periods`` does. ``indices`` maps the 5 Hz record back
    to raw training rows. H/A-swapped augmentation may reuse these arrays:
    swapping team ids does not change run lengths or censoring.
    """
    full_y = _labels(full_y, "full_y")
    sampled_y = _labels(sampled_y, "sampled_y")
    full_period = np.asarray(full_period)
    indices = np.asarray(indices)
    if full_period.ndim != 1 or len(full_period) != len(full_y):
        raise ValueError("full_period must match full_y")
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("indices must be a one-dimensional integer array")
    if len(indices) != len(sampled_y):
        raise ValueError("indices must match sampled_y")
    if len(indices) and (indices.min() < 0 or indices.max() >= len(full_y)):
        raise ValueError("sample index is outside the full-resolution labels")
    if len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("sample indices must be strictly increasing")
    if not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("fps must be positive and finite")
    if len(indices):
        if not np.array_equal(sampled_y >= 0, full_y[indices] >= 0):
            raise ValueError("sampled and full-resolution known-label masks disagree")
        if not np.array_equal(sampled_y == 2, full_y[indices] == 2):
            raise ValueError("sampled and full-resolution D/live masks disagree")

    factors = np.ones(len(full_y), dtype=np.float64)
    eligible = np.zeros(len(full_y), dtype=bool)
    starts = np.r_[0, np.flatnonzero(full_period[1:] != full_period[:-1]) + 1]
    stops = np.r_[starts[1:], len(full_y)]
    for block_start, block_stop in zip(starts, stops):
        at = int(block_start)
        while at < block_stop:
            label = int(full_y[at])
            if label not in (0, 1):
                at += 1
                continue
            stop = at + 1
            while stop < block_stop and full_y[stop] == label:
                stop += 1
            left_censored = at == block_start or full_y[at - 1] < 0
            right_censored = stop == block_stop or full_y[stop] < 0
            if not (left_censored or right_censored):
                duration = (stop - at) / float(fps)
                factor = capped_spell_weight(duration)
                factors[at:stop] = factor
                eligible[at:stop] = True
            at = stop

    mapped = factors[indices].copy()
    eligible_sample = eligible[indices] & (sampled_y >= 0)
    # Keep the D/live gate and censored live mass fixed. Normalize only the
    # complete spells whose conditional H/A loss is actually reweighted.
    if eligible_sample.any():
        mapped[eligible_sample] /= mapped[eligible_sample].mean()
    return mapped.astype(np.float32)


def transition_targets(y, period, start=0, size=None, core_size=None,
                       full_y=None, full_period=None, indices=None):
    """Build pair targets with each edge assigned to its left core row.

    The final core row can pair with the next frame in right context, so core
    chunk seams are retained. A pair is valid only when both labels are known,
    consecutive in this record, and in the same uninterrupted time block.
    The dead/live term accepts all H/A/D pairs; the H/A term accepts only pairs
    whose two labels are H/A. X/unknown never becomes a transition target.
    """
    y = _labels(y, "y")
    period = np.asarray(period)
    if period.ndim != 1 or len(period) != len(y):
        raise ValueError("period must match y")
    if any(value is not None for value in (full_y, full_period, indices)):
        if any(value is None for value in (full_y, full_period, indices)):
            raise ValueError("full_y, full_period, and indices must be supplied together")
        full_y = _labels(full_y, "full_y")
        full_period = np.asarray(full_period)
        indices = np.asarray(indices)
        if (full_period.ndim != 1 or len(full_period) != len(full_y)
                or indices.ndim != 1 or len(indices) != len(y)
                or not np.issubdtype(indices.dtype, np.integer)):
            raise ValueError("invalid full-resolution transition metadata")
        if len(indices) and (indices.min() < 0 or indices.max() >= len(full_y)):
            raise ValueError("sample index is outside the full-resolution labels")
        if len(indices) > 1 and np.any(np.diff(indices) <= 0):
            raise ValueError("sample indices must be strictly increasing")
        if len(indices):
            if not np.array_equal((y >= 0), (full_y[indices] >= 0)):
                raise ValueError("sampled and full-resolution known-label masks disagree")
            if not np.array_equal((y == 2), (full_y[indices] == 2)):
                raise ValueError("sampled and full-resolution D/live masks disagree")
            if not np.array_equal(period, full_period[indices]):
                raise ValueError("sampled and full-resolution time blocks disagree")
    if size is None:
        size = len(y) - int(start)
    if core_size is None:
        core_size = lt.CORE
    start, size, core_size = int(start), int(size), int(core_size)
    if start < 0 or size < 0 or core_size < 0 or start + size > len(y):
        raise ValueError("invalid core bounds")
    if size > core_size:
        raise ValueError("size cannot exceed core_size")

    dead_target = np.zeros(core_size, dtype=np.float32)
    dead_mask = np.zeros(core_size, dtype=bool)
    live_target = np.zeros(core_size, dtype=np.float32)
    live_mask = np.zeros(core_size, dtype=bool)
    if size == 0:
        return dict(dead_target=dead_target, dead_mask=dead_mask,
                    live_target=live_target, live_mask=live_mask)

    left = start + np.arange(size)
    right = left + 1
    inside = right < len(y)
    same_block = np.zeros(size, dtype=bool)
    same_block[inside] = period[left[inside]] == period[right[inside]]
    known_pair = np.zeros(size, dtype=bool)
    known_pair[inside] = ((y[left[inside]] >= 0) & (y[left[inside]] <= 2)
                          & (y[right[inside]] >= 0) & (y[right[inside]] <= 2))
    if full_y is not None:
        full_interval_known = np.zeros(size, dtype=bool)
        for offset in np.flatnonzero(inside & same_block & known_pair):
            native_left = int(indices[left[offset]])
            native_right = int(indices[right[offset]])
            if native_right <= native_left:
                continue
            interval_y = full_y[native_left:native_right + 1]
            interval_period = full_period[native_left:native_right + 1]
            full_interval_known[offset] = (
                len(interval_y) == native_right-native_left+1
                and np.all((interval_y >= 0) & (interval_y <= 2))
                and np.all(interval_period == interval_period[0])
                and period[left[offset]] == interval_period[0]
            )
        known_pair &= full_interval_known
    valid_dead = same_block & known_pair
    safe_right = np.minimum(right, len(y)-1)
    valid_live = valid_dead & (y[left] < 2) & (y[safe_right] < 2)
    dead_mask[:size] = valid_dead
    live_mask[:size] = valid_live
    dead_target[:size] = ((y[left] == 2) != (y[safe_right] == 2)).astype(np.float32)
    live_target[:size] = (y[left] != y[safe_right]).astype(np.float32)
    return dict(dead_target=dead_target, dead_mask=dead_mask,
                live_target=live_target, live_mask=live_mask)


def _record_full_fields(record):
    missing = [key for key in ("full_y", "full_period", "indices", "fps") if key not in record]
    if missing:
        raise ValueError("spell_sqrt requires training-only full-resolution fields: "
                         + ", ".join(missing))
    return spell_duration_factors(record["full_y"], record["full_period"],
                                  record["indices"], record["y"], record["fps"])


def frame_nll_losses(log_prob, targets, *, spell_factors=None, spell_coefficient=0.0):
    """Per-frame factorized NLL with optional conditional H/A-only weighting."""
    import torch
    losses = torch.nn.functional.nll_loss(log_prob.transpose(1, 2), targets,
                                          reduction="none")
    if spell_coefficient == 0:
        return losses
    if spell_factors is None or not 0 <= spell_coefficient <= 1:
        raise ValueError("spell factors and a coefficient in [0,1] are required")
    conditional_log_prob = log_prob[..., :2] - torch.logsumexp(
        log_prob[..., :2], dim=-1, keepdim=True)
    conditional_targets = targets.clamp(min=0, max=1)
    conditional_loss = torch.nn.functional.nll_loss(
        conditional_log_prob.transpose(1, 2), conditional_targets,
        reduction="none")
    live_rows = (targets >= 0) & (targets < 2)
    factor = 1 + spell_coefficient * (spell_factors - 1)
    return losses + conditional_loss * live_rows * (factor - 1)


def group_balanced_binary_loss(losses, targets, valid_mask, weights):
    """Half-weight stable/change means; absent groups contribute zero."""
    import torch
    terms = []
    for label in (0, 1):
        group_weight = weights * valid_mask * (targets == label)
        if group_weight.sum() > 0:
            terms.append((losses * group_weight).sum() / group_weight.sum())
    if not terms:
        return None
    return 0.5 * torch.stack(terms).sum()


def _record_pair_weights(records, pairs, component):
    """Group-balance change/stable edges while retaining source/record mass."""
    source_counts = Counter(record["source"] for record in records)
    weights = []
    mask_key = f"{component}_mask"
    target_key = f"{component}_target"
    for record, pair in zip(records, pairs):
        mask = pair[mask_key]
        target = pair[target_key]
        row = np.zeros(len(mask), dtype=np.float32)
        for group in (0, 1):
            group_mask = mask & (target == group)
            count = int(group_mask.sum())
            if count:
                row[group_mask] = 1.0 / (source_counts[record["source"]] * count)
        weights.append(row)
    return weights


def fit(records, *, player_only, seed, objective="control",
        spell_coefficient=None, transition_coefficient=None):
    """Fit one fixed loss arm using the unchanged factorized-HAD model.

    Optional coefficients support explicit zero-coefficient parity checks.
    The normal experiment arms use the constants in ``OBJECTIVES``.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown objective {objective!r}; expected one of {tuple(OBJECTIVES)}")
    default_spell, default_transition = OBJECTIVES[objective]
    spell_coefficient = default_spell if spell_coefficient is None else float(spell_coefficient)
    transition_coefficient = (default_transition if transition_coefficient is None
                              else float(transition_coefficient))
    if not 0.0 <= spell_coefficient <= 1.0:
        raise ValueError("spell_coefficient must be between 0 and 1")
    if not np.isfinite(transition_coefficient) or transition_coefficient < 0:
        raise ValueError("transition_coefficient must be finite and nonnegative")

    # This is the exact frozen-control training path, including its batching,
    # RNG use, loss arithmetic, and checkpoint contents.
    if spell_coefficient == 0 and transition_coefficient == 0:
        return fh.fit(records, player_only=player_only, seed=seed)

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    mixed = [dict(r, x=np.where(r["replace_rows"][:, None], r["alternate"], r["x"]))
             for r in records]
    median, scale = lt.normalization(mixed)
    del mixed
    original = [lt.normalize(r["x"], median, scale) for r in records]
    alternate = [lt.normalize(r["alternate"], median, scale) for r in records]
    samples = [(i, a, size, full, use)
               for i, record in enumerate(records)
               for (a, size, full), use in zip(lt.chunks(record), record["replace_windows"])]
    masses = np.array([1 / (sum(v["source"] == r["source"] for v in records)
                           * max(1, np.sum(r["y"] >= 0))) for r in records])
    masses /= masses.mean()

    use_spell = spell_coefficient != 0
    use_transition = transition_coefficient != 0
    frame_factors = ([_record_full_fields(r) for r in records] if use_spell
                     else [None] * len(records))
    pair_targets = ([transition_targets(
                        r["y"], r["period"], core_size=len(r["y"]),
                        full_y=r["full_y"], full_period=r["full_period"],
                        indices=r["indices"])
                     for r in records] if use_transition else [None] * len(records))
    if use_transition:
        dead_pair_weights = _record_pair_weights(records, pair_targets, "dead")
        live_pair_weights = _record_pair_weights(records, pair_targets, "live")
    else:
        dead_pair_weights = live_pair_weights = [None] * len(records)

    class Windows(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            i, a, size, full, use = samples[index]
            record = records[i]
            target = np.zeros(lt.CORE, np.int64)
            weight = np.zeros(lt.CORE, np.float32)
            y = record["y"][a:a+size]
            target[:size] = np.maximum(y, 0)
            weight[:size] = (y >= 0) * masses[i]
            if use_spell:
                spell_factor = np.ones(lt.CORE, dtype=np.float32)
                spell_factor[:size] = frame_factors[i][a:a+size]

            if use_transition:
                dead_target = np.zeros(lt.CORE, np.float32)
                dead_mask = np.zeros(lt.CORE, bool)
                live_target = np.zeros(lt.CORE, np.float32)
                live_mask = np.zeros(lt.CORE, bool)
                dead_weight = np.zeros(lt.CORE, np.float32)
                live_weight = np.zeros(lt.CORE, np.float32)
                for key, destination in (("dead_target", dead_target),
                                         ("dead_mask", dead_mask),
                                         ("live_target", live_target),
                                         ("live_mask", live_mask)):
                    destination[:size] = pair_targets[i][key][a:a+size]
                dead_weight[:size] = dead_pair_weights[i][a:a+size]
                live_weight[:size] = live_pair_weights[i][a:a+size]
            else:
                dead_target = dead_mask = live_target = live_mask = None
                dead_weight = live_weight = None
            result = ((alternate if use else original)[i][full], target, weight)
            if use_spell:
                result += (spell_factor,)
            if use_transition:
                result += (dead_target, dead_mask, dead_weight,
                           live_target, live_mask, live_weight)
            return result

    loader = DataLoader(Windows(), batch_size=64, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    columns = fh.player_columns(records[0]["names"])
    model = fh.network(len(median) * 2, columns, player_only)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history, nll_history, dead_history, live_history = [], [], [], []
    updates = 0
    for epoch in range(lt.EPOCHS):
        model.train()
        numerator = denominator = 0.0
        dead_num = dead_den = live_num = live_den = 0.0
        dead_group_values = {"stable": [], "changed": []}
        live_group_values = {"stable": [], "changed": []}
        for batch in loader:
            x, y, weight = batch[:3]
            if not weight.sum():
                continue
            batch_at = 3
            if use_spell:
                spell_factor = batch[batch_at]
                batch_at += 1
            else:
                spell_factor = None
            optimizer.zero_grad()
            log_prob = model(x)
            core_log_prob = log_prob[:, lt.CONTEXT:lt.CONTEXT+lt.CORE]
            if use_spell:
                # Weight only conditional H/A CE; the D/live gate keeps its
                # control mass and D rows are unchanged.
                losses = frame_nll_losses(core_log_prob, y,
                                          spell_factors=spell_factor,
                                          spell_coefficient=spell_coefficient)
            else:
                losses = torch.nn.functional.nll_loss(
                    core_log_prob.transpose(1, 2), y, reduction="none")
            nll_loss = (losses * weight).sum() / weight.sum()
            transition_terms = []
            transition_metrics = []
            if use_transition:
                (dead_target, dead_mask, dead_weight,
                 live_target, live_mask, live_weight) = batch[batch_at:batch_at+6]
                # Both edge endpoints are scored from this window. The left
                # endpoint is in the core; the right may be in right context.
                left_log = log_prob[:, lt.CONTEXT:lt.CONTEXT+lt.CORE]
                right_log = log_prob[:, lt.CONTEXT+1:lt.CONTEXT+lt.CORE+1]

                if dead_mask.any() and dead_weight.sum() > 0:
                    p_dead_left = left_log[..., 2].exp()
                    p_dead_right = right_log[..., 2].exp()
                    p_dead_change = p_dead_left * (1-p_dead_right) + (1-p_dead_left) * p_dead_right
                    dead_loss = torch.nn.functional.binary_cross_entropy(
                        p_dead_change.clamp(1e-6, 1-1e-6), dead_target, reduction="none")
                    dead_term = group_balanced_binary_loss(
                        dead_loss, dead_target, dead_mask, dead_weight)
                    if dead_term is not None:
                        transition_terms.append(0.5 * dead_term)
                        transition_metrics.append(("dead", dead_term))
                        for group, label in ((0, "stable"), (1, "changed")):
                            group_weight = dead_weight * dead_mask * (dead_target == group)
                            if group_weight.sum() > 0:
                                group_value = (dead_loss * group_weight).sum() / group_weight.sum()
                                dead_group_values[label].append(float(group_value.detach()))

                if live_mask.any() and live_weight.sum() > 0:
                    q_left = torch.softmax(left_log[..., :2], dim=-1)
                    q_right = torch.softmax(right_log[..., :2], dim=-1)
                    p_live_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
                    live_loss = torch.nn.functional.binary_cross_entropy(
                        p_live_change.clamp(1e-6, 1-1e-6), live_target, reduction="none")
                    live_term = group_balanced_binary_loss(
                        live_loss, live_target, live_mask, live_weight)
                    if live_term is not None:
                        transition_terms.append(0.5 * live_term)
                        transition_metrics.append(("live", live_term))
                        for group, label in ((0, "stable"), (1, "changed")):
                            group_weight = live_weight * live_mask * (live_target == group)
                            if group_weight.sum() > 0:
                                group_value = (live_loss * group_weight).sum() / group_weight.sum()
                                live_group_values[label].append(float(group_value.detach()))

            if transition_terms:
                # Each active component already carries its fixed 1/2 share.
                transition_loss = torch.stack(transition_terms).sum()
                total_loss = nll_loss + transition_coefficient * transition_loss
            else:
                transition_loss = None
                total_loss = nll_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            updates += 1
            numerator += float((losses.detach() * weight).sum())
            denominator += float(weight.sum())
            if transition_loss is not None:
                for name, term_loss in transition_metrics:
                    value = float(term_loss.detach())
                    if name == "dead":
                        dead_num += value
                        dead_den += 1.0
                    else:
                        live_num += value
                        live_den += 1.0

        nll_epoch = numerator / denominator if denominator else 0.0
        nll_history.append(nll_epoch)
        if dead_den:
            dead_component = dead_num / dead_den
            dead_history.append(dict(
                component=dead_component,
                stable=float(np.mean(dead_group_values["stable"]))
                if dead_group_values["stable"] else None,
                changed=float(np.mean(dead_group_values["changed"]))
                if dead_group_values["changed"] else None))
        else:
            dead_history.append(None)
        if live_den:
            live_component = live_num / live_den
            live_history.append(dict(
                component=live_component,
                stable=float(np.mean(live_group_values["stable"]))
                if live_group_values["stable"] else None,
                changed=float(np.mean(live_group_values["changed"]))
                if live_group_values["changed"] else None))
        else:
            live_history.append(None)
        # Fixed lambda/2 per component: a missing component contributes zero,
        # so batch composition cannot double the remaining component's weight.
        aux_epoch = 0.5 * (dead_num/dead_den if dead_den else 0.0)
        aux_epoch += 0.5 * (live_num/live_den if live_den else 0.0)
        history.append(nll_epoch + transition_coefficient * aux_epoch)
        print("OBJECTIVE", objective, seed, epoch+1,
              round(nll_epoch, 5), round(aux_epoch, 5), flush=True)

    return dict(kind="factorized_had", player_only=player_only, seed=seed,
                player_indices=columns,
                state={key: value.detach().numpy() for key, value in model.state_dict().items()},
                median=median, scale=scale, history=history, nll_history=nll_history,
                dead_transition_history=dead_history, live_transition_history=live_history,
                input_features=len(median)*2, feature_names=records[0]["names"],
                windows_per_epoch=len(samples), optimizer_updates=updates,
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                objective=objective,
                loss_coefficients=dict(spell=spell_coefficient,
                                       transition=transition_coefficient),
                transition_definition=(
                    "fixed lambda/2 each for group-balanced dead/live change from player-only P(D) "
                    "and conditional H/A change; full-native-interval masks; pairwise probability "
                    "surrogate with independent endpoint marginals and cost-sensitive stable/change "
                    "balancing, not a calibrated transition likelihood; no auxiliary head"),
                spell_weight_definition=(
                    "complete full-resolution H/A spells only, applied to conditional H/A CE; "
                    "raw sqrt(5s/duration) clipped [.5,2], censored/D/unknown neutral; "
                    "eligible complete spells among known sampled live rows are normalized "
                    "within each match; D/censored live mass stays fixed and effective "
                    "weights may exceed raw caps after normalization"))


def predict(record, model):
    """Reuse the frozen factorized-HAD inference implementation."""
    return fh.predict(record, model)
