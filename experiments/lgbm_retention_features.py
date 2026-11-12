"""Feature and training-weight helpers for the bounded D-retention follow-up."""
from __future__ import annotations

import numpy as np

from experiments import lgbm_public_runtime
from experiments import supervised


_VIEW_WEIGHTS = {"clean": 1.0, "feed_like_v1": 0.5, "feed_like_v2": 0.25}


def _forward_fill(values, valid, blocks, default=0.0):
    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("Forward-fill values and validity mask must be aligned vectors")
    result = np.full(len(values), float(default), dtype=np.float64)
    for start, stop in blocks:
        block_values, block_valid = values[start:stop], valid[start:stop]
        if not block_valid.any():
            continue
        positions = np.arange(stop - start)
        last = np.maximum.accumulate(np.where(block_valid, positions, -1))
        first = int(np.flatnonzero(block_valid)[0])
        last = np.where(last < 0, first, last)
        result[start:stop] = block_values[last]
    return result


def _nomiss(names, matrix, inputs):
    """Reproduce R2_nomiss: remove ball_* features and use block-local filled cues."""
    keep = [i for i, name in enumerate(names) if not name.startswith("ball_")]
    output_names = [names[i] for i in keep]
    output = matrix[:, keep]

    ball = np.asarray(inputs["ball"], dtype=np.float64)
    observed = np.asarray(inputs.get("ball_observed", np.isfinite(ball).all(axis=1)), dtype=bool)
    if observed.shape != (len(ball),):
        raise ValueError("ball_observed must align with ball frames")
    period = supervised.model_periods(inputs)
    blocks = supervised.blocks(period)
    speed_idx = names.index("ball_speed")
    speed = np.asarray(matrix[:, speed_idx], dtype=np.float64)
    speed_filled = _forward_fill(speed, np.isfinite(speed), blocks)

    length, width = map(float, inputs["pitch"])
    x, y = np.nan_to_num(ball[:, 0]), np.nan_to_num(ball[:, 1])
    beyond = np.maximum.reduce((-x, x - length, -y, y - width))
    beyond_filled = _forward_fill(beyond, observed, blocks)

    result = np.column_stack((output, speed_filled, beyond_filled)).astype(np.float32)
    output_names.extend(("ball_speed_ff", "ball_beyond_last_ff"))
    if result.shape != (len(ball), len(output_names)):
        raise ValueError("R2_nomiss feature rows do not align")
    return output_names, result


def features(inputs, components, feature_set="full"):
    """Return frozen full R2 features or its exact missingness-free ablation."""
    if feature_set not in ("full", "nomiss"):
        raise ValueError("feature_set must be 'full' or 'nomiss'")
    names, matrix = lgbm_public_runtime.features(inputs, components)
    if feature_set == "full":
        return names, matrix
    return _nomiss(names, matrix, inputs)


def view_weight(view):
    """Raw per-row factor yielding equal total mass for 1/2/4 clean/v1/v2 cells."""
    try:
        return _VIEW_WEIGHTS[view]
    except KeyError as exc:
        raise ValueError(f"Unknown training view: {view!r}") from exc

