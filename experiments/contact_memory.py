"""Sparse, label-free contact events for the contact-memory GRU experiment.

This is deliberately not another per-frame anchor feature expansion. Reliable
contact runs are compressed to event tokens; each query row receives only its
nearest strictly earlier and later event, with explicit age and validity. The
learned bridge lives in ``contact_memory_experiment`` and reads these tokens.
"""
from __future__ import annotations

import numpy as np

from experiments import possession_geometry as geometry, supervised


MAX_AGE_SECONDS = 10.0
MIN_STABLE_SECONDS = 0.2
MAX_EVENT_SECONDS = 2.0
GEOMETRY_HALO_FRAMES = 2
EVENT_FIELDS = (
    "home_strength",
    "away_strength",
    "home_margin",
    "control_ambiguity",
    "support_fraction",
)
MEMORY_SPEC = {
    "source": "tracking-only native ball/player observations and masks",
    "candidate": "best team control >= .2 and absolute normalized H/A margin >= .2",
    "reliability": "same signed margin for at least .2 seconds within one continuous input block",
    "compression": "median over a query-local clipped stable contact event; never summarizes across the query",
    "query": "nearest completed prior and starting future stable contact event, same block",
    "maximum_age_seconds": MAX_AGE_SECONDS,
    "minimum_stable_seconds": MIN_STABLE_SECONDS,
    "event_duration_normalization_seconds": MAX_EVENT_SECONDS,
    "geometry_derivative_halo_frames": GEOMETRY_HALO_FRAMES,
    "token_support": "query-local contact samples only; prior runs end before the query, future runs start after it",
    "native_masks": "finite coordinates are ignored when the corresponding observed mask is false",
    "labels_or_events_read": False,
    "unknown_targets": "not consulted; no possession target is inferred",
    "time_breaks": "supervised.model_periods, including half and timestamp discontinuities",
    "fields": EVENT_FIELDS,
}


def _candidate_runs(data: dict, geom: np.ndarray, names: list[str]):
    """Return stable contact intervals grouped by continuous input block."""
    columns = {name: index for index, name in enumerate(names)}
    home_index = columns["H_geometry_best_control"]
    away_index = columns["A_geometry_best_control"]
    margin_index = columns["home_geometry_control_margin"]
    ambiguity_index = columns["geometry_control_ambiguity"]
    home = geom[:, home_index]
    away = geom[:, away_index]
    margin = geom[:, margin_index]
    ambiguity = geom[:, ambiguity_index]
    reliable = np.isfinite(margin) & (np.maximum(home, away) >= .2)
    reliable &= np.abs(margin) >= .2
    side = np.where(reliable, np.sign(margin), 0).astype(np.int8)

    times = np.asarray(data.get("timestamps", np.arange(len(side)) / float(data["fps"])),
                       dtype=float)
    fps = float(data["fps"])
    period = supervised.model_periods(data)
    runs_by_block = {}
    for block_start, block_stop in supervised.blocks(period):
        block_runs = []
        at = block_start
        while at < block_stop:
            team_side = int(side[at])
            if team_side == 0:
                at += 1
                continue
            stop = at + 1
            while stop < block_stop and side[stop] == team_side:
                stop += 1
            duration = float(times[stop - 1] - times[at] + 1.0 / fps)
            if duration + 1e-9 >= MIN_STABLE_SECONDS:
                block_runs.append((at, stop, team_side))
            at = stop
        if block_runs:
            runs_by_block[int(period[block_start])] = {
                "runs": block_runs,
                "starts": np.fromiter((run[0] for run in block_runs), dtype=np.int64),
                "stops": np.fromiter((run[1] for run in block_runs), dtype=np.int64),
                "start": int(block_start),
                "stop": int(block_stop),
            }
    return runs_by_block, home, away, margin, ambiguity, times, period, fps


def memory_features(inputs: dict, sample_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Build two-slot contact memory aligned to the 5 Hz model rows.

    Returned arrays have shapes ``[T, 2, 5]``, ``[T, 2]`` and ``[T, 2]`` for
    event tokens, age in seconds, and validity. Slot 0 is the preceding event;
    slot 1 is the following event. Invalid token/age rows are exact zeros.
    """
    if "labels" in inputs or "targets" in inputs or "events" in inputs:
        # These keys are harmless metadata, but rejecting them makes accidental
        # target-aware construction conspicuous at the feature boundary.
        inputs = {key: value for key, value in inputs.items()
                  if key not in ("labels", "targets", "events")}
    geom, names = geometry.features(inputs)
    (runs_by_block, home, away, margin, ambiguity, times, period,
     fps) = _candidate_runs(inputs, geom, names)
    if sample_indices is None:
        from experiments import learned_temporal as lt
        sample_indices = lt.sampled_indices(inputs)
    sample_indices = np.asarray(sample_indices)
    if sample_indices.ndim != 1 or not np.issubdtype(sample_indices.dtype, np.integer):
        raise ValueError("sample_indices must be a one-dimensional integer array")
    if sample_indices.size and (sample_indices.min() < 0 or sample_indices.max() >= len(geom)):
        raise ValueError("sample index is outside the native tracking rows")

    tokens = np.zeros((len(sample_indices), 2, len(EVENT_FIELDS)), dtype=np.float32)
    ages = np.zeros((len(sample_indices), 2), dtype=np.float32)
    valid = np.zeros((len(sample_indices), 2), dtype=bool)
    for row, native_index in enumerate(sample_indices.astype(np.int64, copy=False)):
        block_id = int(period[native_index])
        block = runs_by_block.get(block_id)
        if not block:
            continue
        runs, starts, stops = block["runs"], block["starts"], block["stops"]
        block_start, block_stop = block["start"], block["stop"]
        query_time = float(times[native_index])
        halo = GEOMETRY_HALO_FRAMES / fps
        local_times = times[block_start:block_stop]
        lower = block_start + int(np.searchsorted(
            local_times, query_time - MAX_AGE_SECONDS + halo, side="left"))
        upper = block_start + int(np.searchsorted(
            local_times, query_time + MAX_AGE_SECONDS - halo, side="right"))
        lower = max(block_start, lower)
        upper = min(block_stop, upper)

        # Use disjoint completed/starting runs. A single long contact run can
        # never appear as both sides of a supposed bridge.
        before_position = int(np.searchsorted(stops, native_index, side="right")) - 1
        after_position = int(np.searchsorted(starts, native_index, side="right"))
        for side, position in ((0, before_position), (1, after_position)):
            if position < 0 or position >= len(runs):
                continue
            run_start, run_stop, _ = runs[position]
            if side == 0:
                clip_start, clip_stop = max(run_start, lower), run_stop
                if clip_stop > native_index:
                    raise ValueError("Prior contact event did not end before query")
                anchor = clip_stop - 1
            else:
                clip_start, clip_stop = run_start, min(run_stop, upper)
                if clip_start <= native_index:
                    raise ValueError("Future contact event did not start after query")
                anchor = clip_start
            if clip_start >= clip_stop:
                continue
            age = abs(float(times[anchor] - query_time))
            if age > MAX_AGE_SECONDS + 1e-9:
                continue
            duration = float(times[clip_stop - 1] - times[clip_start] + 1.0 / fps)
            if duration + 1e-9 < MIN_STABLE_SECONDS:
                continue
            token = np.asarray((
                np.nanmedian(home[clip_start:clip_stop]),
                np.nanmedian(away[clip_start:clip_stop]),
                np.nanmedian(margin[clip_start:clip_stop]),
                np.nanmedian(ambiguity[clip_start:clip_stop]),
                min(duration, MAX_EVENT_SECONDS) / MAX_EVENT_SECONDS,
            ), dtype=np.float32)
            token[~np.isfinite(token)] = 0.0
            tokens[row, side] = token
            ages[row, side] = age
            valid[row, side] = True
    return {"tokens": tokens, "ages": ages, "valid": valid}


def swap_memory(memory: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Apply the H/A symmetry transform to contact evidence."""
    tokens = np.asarray(memory["tokens"], dtype=np.float32).copy()
    if tokens.ndim != 3 or tokens.shape[1:] != (2, len(EVENT_FIELDS)):
        raise ValueError("Contact tokens must have shape [T, 2, len(EVENT_FIELDS)]")
    tokens[..., 0], tokens[..., 1] = memory["tokens"][..., 1], memory["tokens"][..., 0]
    tokens[..., 2] *= -1
    return {"tokens": tokens,
            "ages": np.asarray(memory["ages"], dtype=np.float32).copy(),
            "valid": np.asarray(memory["valid"], dtype=bool).copy()}


def validate_memory(memory: dict[str, np.ndarray], length: int) -> None:
    """Check event arrays before they enter a training/inference batch."""
    tokens = np.asarray(memory.get("tokens"))
    ages = np.asarray(memory.get("ages"))
    valid = np.asarray(memory.get("valid"))
    if tokens.shape != (length, 2, len(EVENT_FIELDS)):
        raise ValueError("Contact token shape does not match the model rows")
    if ages.shape != (length, 2) or valid.shape != (length, 2):
        raise ValueError("Contact age/valid shape does not match the model rows")
    if not np.isfinite(tokens).all() or not np.isfinite(ages).all() or (ages < 0).any():
        raise ValueError("Contact memory must be finite with nonnegative ages")
    if (tokens[~valid] != 0).any() or (ages[~valid] != 0).any():
        raise ValueError("Invalid contact slots must be zero-filled")
    if (ages[valid] > MAX_AGE_SECONDS + 1e-6).any():
        raise ValueError("Contact memory exceeded its declared age limit")

