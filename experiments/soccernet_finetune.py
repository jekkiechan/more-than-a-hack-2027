"""Sparse SoccerNet warm-start fine-tuning for conditional H/A probabilities.

The warm-start checkpoint supplies both the initial state and the frozen
normalizer. Training uses only conditional H/A loss; inference combines the
tuned H/A conditional probabilities with P(D) from the original checkpoint.
Record ``x``, ``y``, and ``period`` fields are sampled-grid arrays. SoccerNet
records contain 24 sampled rows per clip; they are kept separate from each
other and from native-grid audit arrays. Each record is chunked independently,
so clip and time-block boundaries are never crossed.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import random

import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt, supervised


BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
CORE = 50
CONTEXT = 25


def _validate_record(record: dict, *, names: list[str], sparse: bool) -> None:
    required = ("x", "y", "period", "names", "source", "match")
    if any(key not in record for key in required):
        raise ValueError("Record is missing features, labels, periods, or identity")
    x = np.asarray(record["x"])
    y = np.asarray(record["y"])
    period = np.asarray(record["period"])
    if (x.ndim != 2 or x.shape[1] != len(names) or len(x) == 0 or
            y.shape != (len(x),) or period.shape != (len(x),)):
        raise ValueError("Record feature, label, and period shapes differ")
    if list(record["names"]) != names:
        raise ValueError("Record feature schema differs from warm-start checkpoint")
    if not np.isin(y, (-1, 0, 1, 2)).all():
        raise ValueError("Expected H/A/D or unknown labels encoded as -1/0/1/2")
    if sparse:
        if len(y) != 24 or np.count_nonzero(np.isin(y, (0, 1))) != 1:
            raise ValueError("Each SoccerNet record must have 24 samples and one H/A anchor")
        if np.any(y == 2):
            raise ValueError("SoccerNet sparse records cannot contain D labels")


def _record_windows(records: list[dict], *, sparse: bool,
                    expected_names: list[str] | None = None) -> dict:
    """Index labelled H/A windows grouped by source and match/game."""
    groups: dict[tuple[str, str], list[tuple[int, int, int, np.ndarray, bool]]] = defaultdict(list)
    if sparse and not records:
        raise ValueError("Positive SoccerNet mixture requires sparse records")
    names = (list(expected_names) if expected_names is not None else
             (list(records[0]["names"]) if records else []))
    for record_index, record in enumerate(records):
        _validate_record(record, names=names, sparse=sparse)
        if sparse and str(record["source"]).lower() != "soccernet":
            raise ValueError("Sparse records must use source='soccernet'")
        windows = list(lt.chunks(record))
        choices = record.get("replace_windows") if not sparse else None
        if choices is not None and len(choices) != len(windows):
            raise ValueError("Old-record replacement choices do not match temporal windows")
        for window_index, (start, size, full) in enumerate(windows):
            y = np.asarray(record["y"])[start:start + size]
            if not np.isin(y, (0, 1)).any():
                continue
            use_alternate = bool(choices[window_index]) if choices is not None else False
            if use_alternate and "alternate" not in record:
                raise ValueError("Selected legacy augmentation has no alternate features")
            groups[(str(record["source"]), str(record["match"]))].append(
                (record_index, start, size, full, use_alternate))
    return groups


def _draw_window(rng: random.Random, groups: dict,
                 records: list[dict]) -> tuple[dict, int, int, np.ndarray, bool]:
    """Draw source, then match, then one prepared variant/window uniformly."""
    sources = sorted({source for source, _ in groups})
    source = rng.choice(sources)
    matches = sorted(match for candidate, match in groups if candidate == source)
    match = rng.choice(matches)
    entry = rng.choice(groups[(source, match)])
    record_index, start, size, full, use_alternate = entry
    return records[record_index], start, size, full, use_alternate


def _normalized_window(record: dict, start: int, size: int, full: np.ndarray,
                       use_alternate: bool, *, median: np.ndarray,
                       scale: np.ndarray, swap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(record["alternate"] if use_alternate else record["x"])[full]
    y = np.asarray(record["y"])[start:start + size].astype(np.int64, copy=True)
    if swap:
        raw = supervised.swap_team_features(raw, record["names"])
        y = np.where(np.isin(y, (0, 1)), 1 - y, y)
    x = lt.normalize(raw, median, scale)
    target = np.full(CORE, -1, dtype=np.int64)
    target[:size] = y
    return x, target


def _model_state(model: dict) -> dict:
    state = model.get("state")
    if not isinstance(state, dict) or not state:
        raise ValueError("Warm-start checkpoint has no factorized model state")
    return {key: np.array(value, copy=True) for key, value in state.items()}


def train(initial_model: dict, old_records: list[dict], new_records: list[dict],
          fraction: float, seed: int, updates: int = 256) -> dict:
    """Fine-tune conditional H/A logits from one provider-fold checkpoint.

    `old_records` are the verified paired deployment records. Their saved
    replacement-window decisions select the original/alternate input per
    sampled-grid window. SoccerNet clips are separate 24-row sampled-grid
    records with exactly one known H/A anchor; native-grid arrays are not
    concatenated or used by this trainer. Batch composition is fixed at
    0/16/32 new windows for the supported 0/.25/.5 fractions. Unknown and D
    rows have no H/A loss.
    """
    if fraction not in (0.0, 0.25, 0.5):
        raise ValueError("fraction must be 0, .25, or .5")
    if isinstance(updates, bool) or int(updates) != updates or updates <= 0:
        raise ValueError("updates must be a positive integer")
    if not old_records:
        raise ValueError("At least one old training record is required")
    if initial_model.get("kind") != "factorized_had":
        raise ValueError("Expected a factorized H/A/D warm-start checkpoint")
    names = list(initial_model.get("feature_names", []))
    median = np.array(initial_model.get("median"), copy=True)
    scale = np.array(initial_model.get("scale"), copy=True)
    if (not names or median.shape != (len(names),) or scale.shape != median.shape or
            not np.isfinite(median).all() or not np.isfinite(scale).all() or
            np.any(scale <= 0)):
        raise ValueError("Warm-start normalizer or feature schema is invalid")
    if initial_model.get("input_features") != 2 * len(names):
        raise ValueError("Warm-start network input width differs from its feature schema")

    old_groups = _record_windows(old_records, sparse=False, expected_names=names)
    if not old_groups:
        raise ValueError("Old records contain no labelled H/A windows")
    new_groups = (_record_windows(new_records, sparse=True, expected_names=names)
                  if fraction > 0 else {})
    if fraction > 0 and not new_groups:
        raise ValueError("Positive SoccerNet mixture requires labelled clip windows")
    new_count = int(BATCH_SIZE * fraction)
    old_count = BATCH_SIZE - new_count

    import torch
    from torch.nn import functional as F

    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    rng = random.Random(int(seed))
    model = fh.network(int(initial_model["input_features"]),
                        list(initial_model["player_indices"]),
                        bool(initial_model["player_only"]))
    model.load_state_dict({key: torch.as_tensor(value).clone()
                           for key, value in _model_state(initial_model).items()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)

    interval_losses: list[dict] = []
    interval_old: dict[str, int] = defaultdict(int)
    interval_new: dict[str, int] = defaultdict(int)
    exposure_old: dict[str, int] = defaultdict(int)
    exposure_new: dict[str, int] = defaultdict(int)
    swaps_by_match: dict[str, int] = defaultdict(int)
    total_old_swaps = total_new_swaps = 0
    loss_window: list[float] = []
    old_loss_window: list[float] = []
    new_loss_window: list[float] = []
    model.train()
    for step in range(1, int(updates) + 1):
        batch_x: list[np.ndarray] = []
        batch_y: list[np.ndarray] = []
        batch_group: list[int] = []
        for group, count, groups, records in (
                (0, old_count, old_groups, old_records),
                (1, new_count, new_groups, new_records)):
            for _ in range(count):
                record, start, size, full, use_alternate = _draw_window(rng, groups, records)
                swap = group == 1 and rng.random() < 0.5
                x, y = _normalized_window(record, start, size, full, use_alternate,
                                          median=median, scale=scale, swap=swap)
                batch_x.append(x)
                batch_y.append(y)
                batch_group.append(group)
                match_key = f"{record['source']}/{record['match']}"
                if group == 0:
                    exposure_old[match_key] += 1
                    interval_old[match_key] += 1
                    total_old_swaps += int(use_alternate)
                else:
                    exposure_new[match_key] += 1
                    interval_new[match_key] += 1
                    swaps_by_match[match_key] += int(swap)
                    total_new_swaps += int(swap)

        order = list(range(BATCH_SIZE))
        rng.shuffle(order)
        x_tensor = torch.as_tensor(np.stack([batch_x[i] for i in order]),
                                   dtype=torch.float32)
        y_tensor = torch.as_tensor(np.stack([batch_y[i] for i in order]),
                                   dtype=torch.long)
        group_tensor = torch.as_tensor([batch_group[i] for i in order],
                                       dtype=torch.int64)
        # Train only the conditional H/A distribution. P(D) from this clone is
        # intentionally discarded by predict(); its shared GRU can change.
        sequence = model.sequence(x_tensor)[0]
        live_logits = model.output(sequence)[..., :2]
        core_logits = live_logits[:, CONTEXT:CONTEXT + CORE]
        labelled = (y_tensor >= 0) & (y_tensor < 2)
        # D and unknown rows have no conditional H/A target. Give cross-
        # entropy a harmless in-range index before masking those rows out.
        safe_y = torch.where(labelled, y_tensor, torch.zeros_like(y_tensor))
        row_loss = F.cross_entropy(core_logits.transpose(1, 2), safe_y,
                                   reduction="none")
        window_count = labelled.sum(dim=1)
        if torch.any(window_count == 0):
            raise RuntimeError("Sampler emitted a batch window without an H/A anchor")
        window_loss = (row_loss * labelled).sum(dim=1) / window_count
        old_mask = group_tensor == 0
        old_loss = window_loss[old_mask].mean()
        if new_count:
            new_loss = window_loss[~old_mask].mean()
            loss = (1.0 - fraction) * old_loss + fraction * new_loss
        else:
            new_loss = torch.zeros((), dtype=old_loss.dtype)
            loss = old_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        loss_window.append(float(loss.detach()))
        old_loss_window.append(float(old_loss.detach()))
        if new_count:
            new_loss_window.append(float(new_loss.detach()))
        if step % 32 == 0 or step == int(updates):
            interval_losses.append({
                "step": step,
                "loss": float(np.mean(loss_window)),
                "old_loss": float(np.mean(old_loss_window)),
                "new_loss": (float(np.mean(new_loss_window)) if new_loss_window else None),
            })
            loss_window.clear()
            old_loss_window.clear()
            new_loss_window.clear()

    state = {key: value.detach().cpu().numpy().copy()
             for key, value in model.state_dict().items()}
    fitted = copy.deepcopy(initial_model)
    fitted.update(kind="factorized_had", state=state, median=median, scale=scale,
                  seed=int(seed), input_features=int(initial_model["input_features"]),
                  feature_names=names,
                  player_indices=list(initial_model["player_indices"]),
                  player_only=bool(initial_model["player_only"]),
                  history=[entry["loss"] for entry in interval_losses],
                  windows_per_epoch=sum(len(value) for value in old_groups.values()) +
                  sum(len(value) for value in new_groups.values()),
                  optimizer_updates=int(updates),
                  parameters=sum(parameter.numel() for parameter in model.parameters()))
    fitted["finetune"] = {
        "method": "conditional_HA_warm_start; original checkpoint supplies inference P(D)",
        "fraction": float(fraction), "seed": int(seed), "updates": int(updates),
        "batch_size": BATCH_SIZE, "old_windows_per_batch": old_count,
        "soccernet_windows_per_batch": new_count,
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP,
        "normalizer": "copied unchanged from initial fold checkpoint",
        "sparse_loss": "per-window mean conditional H/A cross-entropy over known H/A anchors only",
        "team_swap": "SoccerNet window independently swapped with probability .5; old paired-record swaps retained",
        "loss_intervals": interval_losses,
        "old_window_exposures_by_source_match": dict(sorted(exposure_old.items())),
        "soccernet_window_exposures_by_game": dict(sorted(exposure_new.items())),
        "soccernet_team_swaps_by_game": dict(sorted(swaps_by_match.items())),
        "old_legacy_alternate_window_exposures": int(total_old_swaps),
        "soccernet_team_swapped_windows": int(total_new_swaps),
        "soccernet_matches": len({str(record["match"]) for record in new_records}) if fraction else 0,
    }
    return fitted


def predict(record: dict, tuned: dict, original: dict,
            original_probability: np.ndarray | None = None) -> np.ndarray:
    """Combine tuned conditional H/A with the original checkpoint's exact P(D).

    `record` is feature-only: labels are neither required nor accessed. Pass a
    precomputed native probability array for `original_probability` to avoid
    running the original model twice across fine-tune arms.
    """
    if original_probability is None:
        original_probability = fh.predict(record, original)
    baseline = np.asarray(original_probability)
    n = len(np.asarray(record.get("x", [])))
    if (baseline.shape != (n, 3) or not np.issubdtype(baseline.dtype, np.floating) or
            not np.isfinite(baseline).all() or
            np.any(baseline < 0) or np.any(baseline > 1)):
        raise ValueError("Original probabilities must be a finite (n, 3) array")
    if list(record.get("names", [])) != list(tuned.get("feature_names", [])):
        raise ValueError("Inference feature schema differs from tuned checkpoint")
    if (list(original.get("feature_names", [])) != list(tuned.get("feature_names", [])) or
            not np.array_equal(original.get("median"), tuned.get("median")) or
            not np.array_equal(original.get("scale"), tuned.get("scale"))):
        raise ValueError("Original and tuned checkpoints must share schema and normalizer")
    import torch

    torch.set_num_threads(1)
    net = fh.network(int(tuned["input_features"]), list(tuned["player_indices"]),
                     bool(tuned["player_only"]))
    net.load_state_dict({key: torch.as_tensor(value)
                         for key, value in tuned["state"].items()})
    net.eval()
    windows = list(lt.chunks(record))
    conditional_views = []
    with torch.inference_mode():
        for swap in (False, True):
            raw = (supervised.swap_team_features(record["x"], record["names"])
                   if swap else np.asarray(record["x"]))
            normalized = lt.normalize(raw, tuned["median"], tuned["scale"])
            probabilities = np.empty((n, 2), dtype=np.float32)
            for offset in range(0, len(windows), BATCH_SIZE):
                batch = windows[offset:offset + BATCH_SIZE]
                x = torch.as_tensor(np.stack([normalized[full]
                                               for _, _, full in batch]),
                                    dtype=torch.float32)
                live_logits = net.output(net.sequence(x)[0])[..., :2]
                q = torch.softmax(
                    live_logits[:, CONTEXT:CONTEXT + CORE], dim=-1).cpu().numpy()
                for index, (start, size, _) in enumerate(batch):
                    probabilities[start:start + size] = q[index, :size]
            if swap:
                probabilities = probabilities[:, [1, 0]]
            conditional_views.append(probabilities)
    # Average conditional distributions across team orientations. The tuned
    # clone's D logit is deliberately never read, so it cannot reweight either
    # orientation before this average.
    conditional = (conditional_views[0] + conditional_views[1]) * 0.5
    result = np.empty((n, 3), dtype=baseline.dtype)
    result[:, :2] = conditional * (1.0 - baseline[:, 2:3])
    result[:, 2] = baseline[:, 2]
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1.0,
                                                       rtol=0.0, atol=2e-7):
        raise RuntimeError("Combined probabilities are invalid or fail normalization")
    return result
