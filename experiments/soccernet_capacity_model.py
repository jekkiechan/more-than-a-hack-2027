"""Capacity-scaled conditional H/A warm-starts for sparse SoccerNet records.

The model predicts only conditional home/away probabilities. The original
factorized checkpoint remains the source of P(D) at inference. Wider GRUs are
initialized to preserve the teacher's conditional H/A function: old recurrent
units remain isolated from new units, and the conditional output initially
ignores all new hidden units.

Records are independent sampled-grid segments. ``lt.chunks`` clips windows at
each record's period/time breaks; this module never joins segments or reads
target arrays during prediction.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import hashlib
import random
from typing import Any, Callable

import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt, objective_losses as ol, supervised


OLD_HIDDEN = 24
INPUT_DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
LIVE_TRANSITION_COEFFICIENT = 0.05
SPELL_COEFFICIENT = 1.0
LEARNING_RATES = (1e-4, 3e-4)
CORE = 50
CONTEXT = 25


def network(n_features: int, hidden: int, input_dropout: float = INPUT_DROPOUT):
    """One-layer bidirectional GRU with a conditional two-class output."""
    import torch
    from torch import nn

    if int(hidden) not in (24, 48, 96):
        raise ValueError("hidden must be one of 24, 48, or 96")
    if not 0 <= float(input_dropout) < 1:
        raise ValueError("input_dropout must be in [0, 1)")

    class ConditionalNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_dropout = nn.Dropout(float(input_dropout))
            self.sequence = nn.GRU(int(n_features), int(hidden), batch_first=True,
                                   bidirectional=True)
            self.output = nn.Linear(2 * int(hidden), 2)

        def forward(self, x):
            return self.output(self.sequence(self.input_dropout(x))[0])

    return ConditionalNetwork()


def _state_copy(model: dict) -> dict[str, np.ndarray]:
    state = model.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Teacher checkpoint has no model state")
    return {str(key): np.array(value, copy=True) for key, value in state.items()}


def _checkpoint_sha(model: dict) -> str:
    h = hashlib.sha256()
    h.update(str(model.get("kind", "")).encode())
    for name in model.get("feature_names", []):
        h.update(str(name).encode("utf-8")); h.update(b"\0")
    for name in ("median", "scale"):
        arr = np.ascontiguousarray(np.asarray(model[name]))
        h.update(name.encode()); h.update(arr.dtype.str.encode()); h.update(arr.tobytes())
    for key, value in sorted(_state_copy(model).items()):
        arr = np.ascontiguousarray(value)
        h.update(key.encode()); h.update(arr.dtype.str.encode()); h.update(arr.shape.__repr__().encode()); h.update(arr.tobytes())
    return h.hexdigest()


def _copy_teacher_function(teacher: dict, widened) -> None:
    """Copy each old GRU gate into the low-index block of a widened GRU."""
    import torch

    old = _state_copy(teacher)
    target = widened.state_dict()
    old_width = OLD_HIDDEN
    new_width = widened.sequence.hidden_size
    for suffix in ("", "_reverse"):
        ih = f"sequence.weight_ih_l0{suffix}"
        hh = f"sequence.weight_hh_l0{suffix}"
        bih = f"sequence.bias_ih_l0{suffix}"
        bhh = f"sequence.bias_hh_l0{suffix}"
        for gate in range(3):
            old_rows = slice(gate * old_width, (gate + 1) * old_width)
            new_rows = slice(gate * new_width, gate * new_width + old_width)
            target[ih][new_rows].copy_(torch.as_tensor(old[ih][old_rows]))
            target[hh][new_rows, :old_width].copy_(
                torch.as_tensor(old[hh][old_rows, :old_width]))
            if new_width > old_width:
                # Old hidden states do not receive recurrent input from new units.
                target[hh][new_rows, old_width:].zero_()
            target[bih][new_rows].copy_(torch.as_tensor(old[bih][old_rows]))
            target[bhh][new_rows].copy_(torch.as_tensor(old[bhh][old_rows]))

    # Teacher output layout is [forward old units, reverse old units].
    # The widened layout is [forward width units, reverse width units].
    old_output = old["output.weight"]
    if old_output.shape != (3, 2 * old_width):
        raise ValueError("Unexpected teacher output dimensions")
    target["output.weight"].zero_()
    target["output.weight"][:, :old_width].copy_(
        torch.as_tensor(old_output[:2, :old_width]))
    target["output.weight"][:, new_width:new_width + old_width].copy_(
        torch.as_tensor(old_output[:2, old_width:]))
    target["output.bias"].copy_(torch.as_tensor(old["output.bias"][:2]))
    widened.load_state_dict(target)


def _identity(value: Any, field: str) -> str:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"Record {field} must be a scalar identity")
    scalar = arr.reshape(()).item()
    if scalar is None or str(scalar) == "":
        raise ValueError(f"Record {field} must be nonempty")
    return str(scalar)


def _validate_record(record: dict, names: list[str]) -> tuple[str, str]:
    for key in ("x", "y", "period", "names", "source", "match"):
        if key not in record:
            raise ValueError(f"Training record is missing {key}")
    x, y, period = np.asarray(record["x"]), np.asarray(record["y"]), np.asarray(record["period"])
    if (x.ndim != 2 or x.shape[1] != len(names) or len(x) == 0 or
            y.shape != (len(x),) or period.shape != (len(x),)):
        raise ValueError("Training record x/y/period dimensions differ")
    if not np.isfinite(period).all():
        raise ValueError("Period identifiers must be finite")
    if list(record["names"]) != names:
        raise ValueError("Training feature schema differs from the teacher")
    if not np.isin(y, (-1, 0, 1, 2)).all():
        raise ValueError("Targets must use -1/0/1/2 for unknown/H/A/D")
    if np.isinf(x).any():
        raise ValueError("Training features may contain NaN masks but not infinities")
    _record_chunks(record)
    return _identity(record["source"], "source"), _identity(record["match"], "match")


def _record_chunks(record: dict):
    """Chunk only within prepared period blocks and declared time gaps."""
    period = np.asarray(record["period"])
    if "timestamps" in record:
        timestamps = np.asarray(record["timestamps"])
        fps = float(record.get("fps", 0))
        if (timestamps.shape != period.shape or not np.isfinite(timestamps).all() or
                not np.isfinite(fps) or fps <= 0):
            raise ValueError("Timestamped records require aligned timestamps and positive fps")
        derived = supervised.model_periods(dict(period=period, timestamps=timestamps, fps=fps))
        if np.any((np.diff(derived) != 0) & (np.diff(period) == 0)):
            raise ValueError("period must preserve every timestamp discontinuity")
    return list(lt.chunks(dict(period=period)))


def _window_index(records: list[dict], names: list[str], *, old: bool):
    """Build source→match→window old groups or game→window GAR groups."""
    groups: dict[Any, list[tuple[int, int, int, np.ndarray, bool]]] = defaultdict(list)
    identities = []
    for record_index, record in enumerate(records):
        source, match = _validate_record(record, names)
        identities.append((source, match))
        windows = _record_chunks(record)
        replacements = record.get("replace_windows") if old else None
        if replacements is not None and len(replacements) != len(windows):
            raise ValueError("Prepared old replacement choices do not align with windows")
        for window_index, (start, size, full) in enumerate(windows):
            y = np.asarray(record["y"])[start:start + size]
            if not np.isin(y, (0, 1)).any():
                continue
            use_alternate = bool(replacements[window_index]) if replacements is not None else False
            if use_alternate:
                alternate = np.asarray(record.get("alternate"))
                if alternate.shape != np.asarray(record["x"]).shape:
                    raise ValueError("Selected old alternate features are missing or malformed")
            key = (source, match) if old else match
            groups[key].append((record_index, start, size, full, use_alternate))
    return groups, identities


def _draw_old(rng: random.Random, groups: dict, records: list[dict]):
    sources = sorted({source for source, _ in groups})
    source = rng.choice(sources)
    matches = sorted(match for candidate, match in groups if candidate == source)
    match = rng.choice(matches)
    record_index, start, size, full, alternate = rng.choice(groups[(source, match)])
    return record_index, start, size, full, alternate


def _draw_new(rng: random.Random, groups: dict):
    game = rng.choice(sorted(groups))
    record_index, start, size, full, _ = rng.choice(groups[game])
    return record_index, start, size, full, False


def _prepare_window(record: dict, start: int, size: int, full: np.ndarray,
                    alternate: bool, *, median: np.ndarray, scale: np.ndarray,
                    swap: bool) -> tuple[np.ndarray, np.ndarray]:
    raw_all = np.asarray(record["alternate"] if alternate else record["x"])
    raw = raw_all[full]
    y = np.asarray(record["y"])[start:start + size].astype(np.int64, copy=True)
    if swap:
        raw = supervised.swap_team_features(raw, record["names"])
        known_ha = np.isin(y, (0, 1))
        y[known_ha] = 1 - y[known_ha]
    x = lt.normalize(raw, median, scale)
    target = np.full(CORE, -1, dtype=np.int64)
    target[:size] = y
    return x, target


def _old_objective_targets(records: list[dict], *, require_full_native: bool):
    """Reuse the tested complete-spell and native-interval transition targets."""
    fields = ("full_y", "full_period", "indices", "fps")
    presence = [all(key in record for key in fields) for record in records]
    if any(presence) and not all(presence):
        raise ValueError("Original records must consistently include full-native objective fields")
    has_full_native = bool(presence and all(presence))
    if require_full_native and not has_full_native:
        raise ValueError("Real capacity fits require full_y/full_period/indices/fps on every original record")
    spell_factors = []
    pair_targets = []
    for record in records:
        if has_full_native:
            spell_factors.append(ol._record_full_fields(record))
            pair_targets.append(ol.transition_targets(
                record["y"], record["period"], core_size=len(record["y"]),
                full_y=record["full_y"], full_period=record["full_period"],
                indices=record["indices"]))
        else:
            # Synthetic unit tests may omit native fields. In that case the
            # existing helper uses only sample-adjacent, same-period known pairs.
            spell_factors.append(np.ones(len(record["y"]), dtype=np.float32))
            pair_targets.append(ol.transition_targets(
                record["y"], record["period"], core_size=len(record["y"])))
    # Source/match balance is already imposed by hierarchical window sampling.
    # Unit edge weights avoid applying a second inverse-frequency correction;
    # the objective helper below balances stable/change groups within a batch.
    pair_weights = [pair["live_mask"].astype(np.float32) for pair in pair_targets]
    return spell_factors, pair_targets, pair_weights, has_full_native


def _model_input_width(initial: dict) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    names = list(initial.get("feature_names", []))
    median = np.array(initial.get("median"), copy=True)
    scale = np.array(initial.get("scale"), copy=True)
    if (not names or median.shape != (len(names),) or scale.shape != median.shape or
            not np.isfinite(median).all() or not np.isfinite(scale).all() or
            np.any(scale <= 0)):
        raise ValueError("Teacher normalizer/schema is invalid")
    input_features = int(initial.get("input_features", -1))
    if input_features != 2 * len(names):
        raise ValueError("Teacher input width does not match normalized schema")
    if initial.get("kind") != "factorized_had":
        raise ValueError("Expected a factorized H/A/D teacher checkpoint")
    return names, median, scale, input_features


def _snapshot(model, optimizer, *, teacher_sha: str, config: dict,
              next_step: int, rng: random.Random, losses: list[dict],
              old_exposures: dict, old_alternate_exposures: dict,
              new_exposures: dict, new_swap_exposures: dict,
              gradient_norms: dict) -> dict:
    import torch

    model_state = {key: value.detach().cpu().numpy().copy()
                   for key, value in model.state_dict().items()}
    result = dict(kind="soccernet_capacity", hidden=int(model.sequence.hidden_size),
        state=model_state, median=config["median"].copy(), scale=config["scale"].copy(),
        feature_names=list(config["feature_names"]), input_features=config["input_features"],
        source=dict(kind="factorized_had", checkpoint_sha256=teacher_sha,
                    seed=config.get("teacher_seed")),
        schema=dict(feature_names=list(config["feature_names"]),
                    input_features=config["input_features"],
                    normalized_input_features=2 * len(config["feature_names"])),
        step=int(next_step), training=dict(config),
        metadata=dict(loss_intervals=copy.deepcopy(losses),
            old_window_exposures_by_source_match=dict(sorted(old_exposures.items())),
            old_alternate_window_exposures_by_source_match=dict(sorted(old_alternate_exposures.items())),
            new_window_exposures_by_game=dict(sorted(new_exposures.items())),
            new_team_swapped_windows_by_game=dict(sorted(new_swap_exposures.items())),
            capacity_gradient_norms_by_step=dict(sorted(gradient_norms.items()))),
        continuation=dict(step=int(next_step), optimizer_state=copy.deepcopy(optimizer.state_dict()),
            python_rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state().clone(),
            loss_intervals=copy.deepcopy(losses), old_exposures=dict(old_exposures),
            old_alternate_exposures=dict(old_alternate_exposures),
            new_exposures=dict(new_exposures), new_swap_exposures=dict(new_swap_exposures),
            gradient_norms=dict(gradient_norms),
            teacher_sha256=teacher_sha, config={k: v for k, v in config.items()
                                                if k not in ("median", "scale", "feature_names", "steps")}))
    return result


def train(initial_model: dict, old_records: list[dict], new_records: list[dict],
          width: int, lr: float, seed: int, steps: int = 2048,
          milestones: tuple[int, ...] = (512, 2048), fraction: float = 0.5,
          batch_size: int = 32, resume_from: dict | None = None,
          input_dropout: float = INPUT_DROPOUT,
          require_full_native: bool = False,
          on_checkpoint: Callable[[int, dict], None] | None = None) -> dict[int, dict]:
    """Train conditional H/A and return sealed model snapshots by step.

    A snapshot includes optimizer plus Python/PyTorch RNG state and can be
    passed as ``resume_from`` to continue bit-for-bit. ``steps`` is the target
    total update count, not additional updates after resumption.
    """
    if isinstance(width, bool) or int(width) not in (24, 48, 96):
        raise ValueError("width must be 24, 48, or 96")
    if isinstance(steps, bool) or int(steps) != steps or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not np.isfinite(lr) or not any(np.isclose(lr, candidate, rtol=0, atol=1e-12)
                                      for candidate in LEARNING_RATES):
        raise ValueError("lr must be one of the declared rates: 1e-4 or 3e-4")
    if not 0 <= float(fraction) < 1:
        raise ValueError("fraction must be in [0, 1)")
    new_count = int(round(batch_size * float(fraction)))
    if not np.isclose(new_count / batch_size, fraction, atol=1e-12, rtol=0):
        raise ValueError("batch_size * fraction must be an integer")
    if fraction > 0 and new_count == 0:
        raise ValueError("Positive GAR fraction requires at least one GAR row per batch")
    old_count = int(batch_size) - new_count
    if old_count <= 0:
        raise ValueError("The original-data group must remain in every batch")
    milestones = tuple(int(value) for value in milestones)
    if (not milestones or tuple(sorted(set(milestones))) != milestones or
            milestones[-1] != int(steps) or any(m < 1 or m > steps for m in milestones)):
        raise ValueError("milestones must be sorted, unique, and end at steps")
    if not old_records:
        raise ValueError("At least one original-data record is required")

    names, median, scale, input_features = _model_input_width(initial_model)
    old_groups, old_ids = _window_index(old_records, names, old=True)
    if not old_groups:
        raise ValueError("Original records contain no H/A-supervised windows")
    spell_factors, pair_targets, pair_weights, has_full_native = _old_objective_targets(
        old_records, require_full_native=require_full_native)
    if fraction > 0 and not new_records:
        raise ValueError("Positive GAR fraction requires GAR records")
    new_groups, new_ids = _window_index(new_records if fraction else [], names, old=False)
    if fraction > 0 and not new_groups:
        raise ValueError("GAR records contain no H/A-supervised windows")

    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    teacher_sha = _checkpoint_sha(initial_model)
    torch.manual_seed(int(seed))
    model = network(input_features, int(width), float(input_dropout))
    _copy_teacher_function(initial_model, model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr),
                                  weight_decay=WEIGHT_DECAY)
    rng = random.Random(int(seed))
    start_step = 0
    losses: list[dict] = []
    old_exposures: dict[str, int] = defaultdict(int)
    old_alternate_exposures: dict[str, int] = defaultdict(int)
    new_exposures: dict[str, int] = defaultdict(int)
    new_swap_exposures: dict[str, int] = defaultdict(int)
    gradient_norms: dict[int, float] = {}
    config = dict(width=int(width), lr=float(lr), seed=int(seed), steps=int(steps),
        fraction=float(fraction), batch_size=int(batch_size), input_dropout=float(input_dropout),
        weight_decay=WEIGHT_DECAY, gradient_clip=GRADIENT_CLIP,
        feature_names=names, input_features=input_features, median=median, scale=scale,
        teacher_seed=initial_model.get("seed"), teacher_sha256=teacher_sha,
        loss="mean per-window conditional H/A cross-entropy; unknown/D masked",
        sampling="old source→match→window; GAR game→window",
        old_record_count=len(old_records), new_record_count=len(new_records) if fraction else 0,
        old_source_match_count=len(old_groups), new_game_count=len(new_groups),
        old_windows=sum(map(len, old_groups.values())),
        new_windows=sum(map(len, new_groups.values())),
        old_windows_per_batch=old_count, new_windows_per_batch=new_count,
        old_spell_coefficient=SPELL_COEFFICIENT,
        old_objective="conditional H/A row CE weighted by complete-spell factors; censored/D/unknown neutral",
        old_spell_factor_source="objective_losses._record_full_fields" if has_full_native else "neutral synthetic fallback",
        old_transition_source="objective_losses.transition_targets with full-native interval mask" if has_full_native else "objective_losses.transition_targets with sample-adjacent/same-period mask",
        old_live_transition_coefficient=LIVE_TRANSITION_COEFFICIENT,
        gar_objective="conditional H/A CE at sparse anchors only; duration and transitions absent",
        require_full_native=bool(require_full_native))
    if resume_from is not None:
        continuation = resume_from.get("continuation")
        if (resume_from.get("kind") != "soccernet_capacity" or not continuation or
                resume_from.get("hidden") != int(width) or
                continuation.get("teacher_sha256") != teacher_sha or
                continuation.get("config") != {k: v for k, v in config.items()
                    if k not in ("median", "scale", "feature_names", "steps")}):
            raise ValueError("Resume checkpoint does not match teacher and training configuration")
        start_step = int(continuation["step"])
        if start_step >= steps:
            raise ValueError("Resume checkpoint is already at or beyond requested steps")
        model.load_state_dict({key: torch.as_tensor(value)
                               for key, value in resume_from["state"].items()})
        optimizer.load_state_dict(copy.deepcopy(continuation["optimizer_state"]))
        rng.setstate(continuation["python_rng_state"])
        torch.set_rng_state(continuation["torch_rng_state"].clone())
        losses = copy.deepcopy(continuation["loss_intervals"])
        old_exposures.update(continuation["old_exposures"])
        old_alternate_exposures.update(continuation["old_alternate_exposures"])
        new_exposures.update(continuation["new_exposures"])
        new_swap_exposures.update(continuation["new_swap_exposures"])
        gradient_norms.update(continuation["gradient_norms"])
    else:
        pass
    if start_step >= steps or any(m <= start_step for m in milestones):
        raise ValueError("Requested milestones must be after the resume step")

    model.train()
    loss_accum: list[float] = []
    old_accum: list[float] = []
    new_accum: list[float] = []
    transition_accum: list[float] = []
    snapshots: dict[int, dict] = {}
    for step in range(start_step + 1, int(steps) + 1):
        batch_x, batch_y, batch_group = [], [], []
        batch_spell, batch_transition_target = [], []
        batch_transition_mask, batch_transition_weight = [], []
        for group, count in ((0, old_count), (1, new_count)):
            for _ in range(count):
                if group == 0:
                    record_index, start, size, full, alternate = _draw_old(rng, old_groups, old_records)
                    record = old_records[record_index]
                    swapped = False  # Cached paired records preserve their fixed swap/replace choice.
                else:
                    record_index, start, size, full, alternate = _draw_new(rng, new_groups)
                    record = new_records[record_index]
                    swapped = rng.random() < 0.5
                x, y = _prepare_window(record, start, size, full, alternate,
                    median=median, scale=scale, swap=swapped)
                if not np.isin(y, (0, 1)).any():
                    raise RuntimeError("Sampler emitted a window without an H/A anchor")
                batch_x.append(x); batch_y.append(y); batch_group.append(group)
                spell = np.ones(CORE, dtype=np.float32)
                transition_target = np.zeros(CORE, dtype=np.float32)
                transition_mask = np.zeros(CORE, dtype=bool)
                transition_weight = np.zeros(CORE, dtype=np.float32)
                if group == 0:
                    spell[:size] = spell_factors[record_index][start:start + size]
                    pair = pair_targets[record_index]
                    transition_target[:size] = pair["live_target"][start:start + size]
                    transition_mask[:size] = pair["live_mask"][start:start + size]
                    transition_weight[:size] = pair_weights[record_index][start:start + size]
                batch_spell.append(spell)
                batch_transition_target.append(transition_target)
                batch_transition_mask.append(transition_mask)
                batch_transition_weight.append(transition_weight)
                if group == 0:
                    source, match = old_ids[record_index]
                    old_exposures[f"{source}/{match}"] += 1
                    old_alternate_exposures[f"{source}/{match}"] += int(alternate)
                else:
                    _, game = new_ids[record_index]
                    new_exposures[game] += 1
                    new_swap_exposures[game] += int(swapped)

        order = list(range(batch_size)); rng.shuffle(order)
        x_tensor = torch.as_tensor(np.stack([batch_x[i] for i in order]), dtype=torch.float32)
        y_tensor = torch.as_tensor(np.stack([batch_y[i] for i in order]), dtype=torch.long)
        group_tensor = torch.as_tensor([batch_group[i] for i in order], dtype=torch.int64)
        spell_tensor = torch.as_tensor(np.stack([batch_spell[i] for i in order]), dtype=torch.float32)
        transition_target = torch.as_tensor(np.stack(
            [batch_transition_target[i] for i in order]), dtype=torch.float32)
        transition_mask = torch.as_tensor(np.stack(
            [batch_transition_mask[i] for i in order]), dtype=torch.bool)
        transition_weight = torch.as_tensor(np.stack(
            [batch_transition_weight[i] for i in order]), dtype=torch.float32)
        all_logits = model(x_tensor)
        logits = all_logits[:, CONTEXT:CONTEXT + CORE]
        labelled = (y_tensor == 0) | (y_tensor == 1)
        safe_y = torch.where(labelled, y_tensor, torch.zeros_like(y_tensor))
        row_loss = F.cross_entropy(logits.transpose(1, 2), safe_y, reduction="none")
        per_window_count = labelled.sum(dim=1)
        if torch.any(per_window_count == 0):
            raise RuntimeError("Every sampled window must contain a known H/A label")
        factors = torch.where(group_tensor[:, None] == 0,
                              1.0 + SPELL_COEFFICIENT * (spell_tensor - 1.0),
                              torch.ones_like(spell_tensor))
        per_window = (row_loss * labelled * factors).sum(dim=1) / per_window_count
        old_mask = group_tensor == 0
        old_loss = per_window[old_mask].mean()
        new_loss = per_window[~old_mask].mean() if new_count else None
        q_left = torch.softmax(all_logits[:, CONTEXT:CONTEXT + CORE], dim=-1)
        q_right = torch.softmax(all_logits[:, CONTEXT + 1:CONTEXT + CORE + 1], dim=-1)
        p_change = q_left[..., 0] * q_right[..., 1] + q_left[..., 1] * q_right[..., 0]
        edge_loss = F.binary_cross_entropy(p_change.clamp(1e-6, 1 - 1e-6),
                                           transition_target, reduction="none")
        old_edge_mask = transition_mask & old_mask[:, None]
        transition_loss = ol.group_balanced_binary_loss(
            edge_loss, transition_target, old_edge_mask, transition_weight)
        transition_value = (torch.zeros((), dtype=old_loss.dtype)
                            if transition_loss is None else transition_loss)
        loss = old_loss if fraction == 0 else (1.0 - fraction) * old_loss + fraction * new_loss
        loss = loss + (1.0 - fraction) * LIVE_TRANSITION_COEFFICIENT * transition_value
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if int(width) > OLD_HIDDEN:
            # Capture actual gradient flow into the added recurrent units.
            extra = []
            for suffix in ("", "_reverse"):
                name = f"weight_ih_l0{suffix}"
                parameter = getattr(model.sequence, name)
                parameter_extra = torch.cat([
                    parameter.grad[gate * int(width) + OLD_HIDDEN:(gate + 1) * int(width)]
                    for gate in range(3)])
                extra.append(float(parameter_extra.norm().detach()))
                hidden_parameter = getattr(model.sequence, f"weight_hh_l0{suffix}")
                hidden_extra = torch.cat([
                    hidden_parameter.grad[gate * int(width) + OLD_HIDDEN:(gate + 1) * int(width)]
                    for gate in range(3)])
                extra.append(float(hidden_extra.norm().detach()))
            gradient_norms[step] = float(np.linalg.norm(extra))
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        current_loss = float(loss.detach())
        loss_accum.append(current_loss)
        old_accum.append(float(old_loss.detach()))
        if new_loss is not None:
            new_accum.append(float(new_loss.detach()))
        transition_accum.append(float(transition_value.detach()))

        if step % 128 == 0:
            print(f"CAPACITY_TRAIN width={int(width)} lr={float(lr):g} seed={int(seed)} "
                  f"step={step}/{int(steps)} loss={current_loss:.6g}", flush=True)

        if step % 32 == 0 or step in milestones:
            losses.append(dict(step=step, loss=float(np.mean(loss_accum)),
                old_loss=float(np.mean(old_accum)),
                new_loss=float(np.mean(new_accum)) if new_accum else None,
                old_live_transition_loss=float(np.mean(transition_accum))))
            loss_accum.clear(); old_accum.clear(); new_accum.clear()
            transition_accum.clear()
        if step in milestones:
            checkpoint = _snapshot(model, optimizer, teacher_sha=teacher_sha,
                config=config, next_step=step, rng=rng, losses=losses,
                old_exposures=old_exposures,
                old_alternate_exposures=old_alternate_exposures,
                new_exposures=new_exposures, new_swap_exposures=new_swap_exposures,
                gradient_norms=gradient_norms)
            snapshots[step] = checkpoint
            if on_checkpoint is not None:
                # The callback receives an isolated checkpoint copy; it cannot
                # mutate live optimizer, model, RNG, or retained snapshots.
                on_checkpoint(step, copy.deepcopy(checkpoint))
    return snapshots


def predict(record: dict, model: dict, original: dict,
            original_probability: np.ndarray | None = None) -> np.ndarray:
    """Symmetrize conditional H/A and combine with exact original P(D).

    Only feature arrays, schema, period breaks, and optional sampled timestamps
    and rate enter the inference path. Target and native-label fields are never
    copied or read.
    """
    if model.get("kind") != "soccernet_capacity":
        raise ValueError("Expected a conditional capacity checkpoint")
    feature_record = {key: record[key] for key in ("x", "names", "period")}
    if "timestamps" in record:
        feature_record["timestamps"] = record["timestamps"]
    if "fps" in record:
        feature_record["fps"] = record["fps"]
    if "timestamps" in feature_record and "fps" not in feature_record:
        raise ValueError("Timestamped inference records require fps")
    _record_chunks(feature_record)
    if list(feature_record["names"]) != list(model.get("feature_names", [])):
        raise ValueError("Inference schema differs from capacity checkpoint")
    if (not np.array_equal(model["median"], original.get("median")) or
            not np.array_equal(model["scale"], original.get("scale")) or
            list(model["feature_names"]) != list(original.get("feature_names", []))):
        raise ValueError("Teacher and capacity model must share frozen normalization/schema")

    if original_probability is None:
        baseline = fh.predict(feature_record, original)
    else:
        baseline = np.asarray(original_probability)
    x_raw = np.asarray(feature_record["x"])
    n = len(x_raw)
    if (baseline.shape != (n, 3) or not np.issubdtype(baseline.dtype, np.floating) or
            not np.isfinite(baseline).all() or np.any(baseline < 0) or np.any(baseline > 1)):
        raise ValueError("Original probabilities must be finite categorical H/A/D rows")

    import torch
    torch.set_num_threads(1)
    net = network(int(model["input_features"]), int(model["hidden"]),
                  float(model["training"]["input_dropout"]))
    net.load_state_dict({key: torch.as_tensor(value) for key, value in model["state"].items()})
    net.eval()
    windows = _record_chunks(feature_record)
    views = []
    with torch.inference_mode():
        for swap in (False, True):
            raw = (supervised.swap_team_features(x_raw, feature_record["names"])
                   if swap else x_raw)
            normalized = lt.normalize(raw, model["median"], model["scale"])
            conditional = np.empty((n, 2), dtype=np.float32)
            for offset in range(0, len(windows), 64):
                batch = windows[offset:offset + 64]
                x = torch.as_tensor(np.stack([normalized[full]
                                               for _, _, full in batch]), dtype=torch.float32)
                probabilities = torch.softmax(net(x)[:, CONTEXT:CONTEXT + CORE], dim=-1).cpu().numpy()
                for i, (start, size, _) in enumerate(batch):
                    conditional[start:start + size] = probabilities[i, :size]
            views.append(conditional[:, [1, 0]] if swap else conditional)
    q = (views[0] + views[1]) * 0.5
    result = np.empty((n, 3), dtype=baseline.dtype)
    result[:, :2] = q * (1.0 - baseline[:, 2:3])
    result[:, 2] = baseline[:, 2]
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1.0, atol=1e-6, rtol=0):
        raise RuntimeError("Invalid combined categorical probabilities")
    return result


def teacher_conditional(record: dict, teacher: dict) -> np.ndarray:
    """Return unsymmetrized conditional H/A probabilities for tests/audits."""
    import torch

    torch.set_num_threads(1)
    net = fh.network(int(teacher["input_features"]), list(teacher["player_indices"]),
                     bool(teacher["player_only"]))
    net.load_state_dict({key: torch.as_tensor(value) for key, value in teacher["state"].items()})
    net.eval()
    windows = _record_chunks(record)
    out = np.empty((len(record["x"]), 2), dtype=np.float32)
    with torch.inference_mode():
        normalized = lt.normalize(record["x"], teacher["median"], teacher["scale"])
        for offset in range(0, len(windows), 64):
            batch = windows[offset:offset + 64]
            x = torch.as_tensor(np.stack([normalized[full] for _, _, full in batch]), dtype=torch.float32)
            logits = net.output(net.sequence(x)[0])[..., :2]
            q = torch.softmax(logits[:, CONTEXT:CONTEXT + CORE], dim=-1).cpu().numpy()
            for i, (start, size, _) in enumerate(batch):
                out[start:start + size] = q[i, :size]
    return out
