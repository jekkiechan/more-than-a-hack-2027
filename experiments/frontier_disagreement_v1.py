"""Cross-expert disagreement gated H/A decoding for the current frontier.

Inputs are the three already-pooled conditional H/A expert probabilities,
the frozen frontier D/X prediction, and model-period identifiers. No targets,
provider identity, or evaluation metadata enter the decoder.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from experiments import decoder_search

WEIGHTS = (0.5, 0.375, 0.125)  # CE, calibrated, adopted
LAMBDAS = (0.5, 1.0)
HA_COST = 3.0
WINDOW_SECONDS = 0.4


def _experts(probabilities: Sequence[np.ndarray] | Mapping[str, np.ndarray], n: int,
             weights=WEIGHTS) -> tuple[list[np.ndarray], tuple[float, ...]]:
    if isinstance(probabilities, Mapping):
        names = ("ce", "calibrated", "adopted")
        try:
            arrays = [np.asarray(probabilities[name], dtype=np.float64) for name in names]
        except KeyError as exc:
            raise ValueError("Expected CE, calibrated, and adopted expert arrays") from exc
    else:
        arrays = [np.asarray(value, dtype=np.float64) for value in probabilities]
    weights = tuple(float(value) for value in weights)
    if (len(arrays) != 3 or len(weights) != 3 or not np.isfinite(weights).all()
            or any(value < 0 for value in weights) or not np.isclose(sum(weights), 1.0)):
        raise ValueError("Exactly three experts and normalized nonnegative weights are required")
    result = []
    for index, array in enumerate(arrays):
        if (array.ndim != 2 or array.shape[0] != n or array.shape[1] not in (2, 3)
                or not np.isfinite(array).all() or (array < 0).any()):
            raise ValueError(f"Invalid expert probability array {index}")
        ha = array[:, :2].copy()
        mass = ha.sum(axis=1, keepdims=True)
        if (mass <= 0).any():
            raise ValueError(f"Expert {index} has a row with no H/A probability mass")
        ha /= mass
        result.append(ha)
    return result, weights


def disagreement(experts: Sequence[np.ndarray] | Mapping[str, np.ndarray],
                 weights=WEIGHTS) -> np.ndarray:
    """Return the specified per-frame, weighted H-probability variance score."""
    if isinstance(experts, Mapping):
        try:
            arrays = [np.asarray(experts[name], dtype=np.float64)
                      for name in ("ce", "calibrated", "adopted")]
        except KeyError as exc:
            raise ValueError("Expected CE, calibrated, and adopted expert arrays") from exc
    else:
        arrays = [np.asarray(value, dtype=np.float64) for value in experts]
    if len(arrays) != 3 or any(a.ndim != 2 or a.shape[1] not in (2, 3) for a in arrays):
        raise ValueError("Expected three Nx2 or Nx3 expert arrays")
    normalized, w = _experts(arrays, len(arrays[0]), weights)
    home = np.stack([array[:, 0] for array in normalized], axis=0)
    mean = np.sum(np.asarray(w)[:, None] * home, axis=0)
    gate = 4.0 * np.sum(np.asarray(w)[:, None] * (home - mean[None, :]) ** 2, axis=0)
    # Weighted variance of values in [0, 1] is at most .25.
    return np.clip(gate, 0.0, 1.0)


def _smooth_segment(values: np.ndarray, half_width: int) -> np.ndarray:
    """Centered uniform smoother with nearest-value edge padding."""
    if half_width <= 0:
        return values.copy()
    width = 2 * half_width + 1
    padded = np.pad(values, (half_width, half_width), mode="edge")
    prefix = np.r_[0.0, np.cumsum(padded, dtype=np.float64)]
    return (prefix[width:width + len(values)] - prefix[:len(values)]) / width


def _edge_gate(gate: np.ndarray, labels: np.ndarray, period: np.ndarray,
               fps: float) -> np.ndarray:
    """Smoothed gate, reset at every fixed-D/X or model-period boundary."""
    half = round(WINDOW_SECONDS * fps / 2.0)
    smoothed = np.zeros(len(gate), dtype=np.float64)
    active = np.isin(labels, ("H", "A"))
    starts = np.flatnonzero(active & np.r_[True, (~active[:-1]) | (period[1:] != period[:-1])])
    stops = []
    for start in starts:
        stop = start + 1
        while stop < len(gate) and active[stop] and period[stop] == period[start]:
            stop += 1
        stops.append((start, stop))
        smoothed[start:stop] = _smooth_segment(gate[start:stop], half)
    edge = np.zeros(len(gate), dtype=np.float64)
    for start, stop in stops:
        # Index t denotes the transition into frame t, matching decoder_search._path.
        for t in range(start + 1, stop):
            edge[t] = 0.5 * (smoothed[t - 1] + smoothed[t])
    return edge


def decode(expert_probabilities: Sequence[np.ndarray] | Mapping[str, np.ndarray],
           predecessor, model_period, fps: float, lambda_value: float,
           weights=WEIGHTS) -> np.ndarray:
    """Decode H/A within frozen frontier live spans, preserving D and X exactly.

    Each expert array is either Nx2 conditional H/A probability or Nx3
    probability. Seeds and phases must already have been pooled within each
    expert using the frozen frontier recipe.
    """
    lam = float(lambda_value)
    y = np.asarray(predecessor).astype("U1", copy=False)
    period = np.asarray(model_period)
    if (not np.isfinite(lam) or lam < 0 or y.ndim != 1 or len(y) == 0
            or period.shape != y.shape or not np.isin(y, ("H", "A", "D", "X")).all()
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Invalid predecessor, model period, fps, or lambda")
    arrays, w = _experts(expert_probabilities, len(y), weights)
    gate = disagreement(arrays, w)
    mixture = np.sum(np.asarray(w)[:, None, None] * np.stack(arrays), axis=0)
    mixture /= mixture.sum(axis=1, keepdims=True)
    edge_gate = _edge_gate(gate, y, period, float(fps))
    output = y.copy()
    costs = np.array([[0.0, HA_COST], [HA_COST, 0.0]], dtype=np.float64) * (float(fps) / 10.0)
    active = np.isin(y, ("H", "A"))
    starts = np.flatnonzero(active & np.r_[True, (~active[:-1]) | (period[1:] != period[:-1])])
    for start in starts:
        stop = start + 1
        while stop < len(y) and active[stop] and period[stop] == period[start]:
            stop += 1
        logp = np.log(np.maximum(mixture[start:stop], 1e-7))
        scale = 1.0 + lam * edge_gate[start:stop]
        path = decoder_search._path(logp, costs, scale)
        output[start:stop] = np.where(path == 0, "H", "A")
    frozen = np.isin(y, ("D", "X"))
    if not np.array_equal(output[frozen], y[frozen]):
        raise RuntimeError("Disagreement decoder changed a frozen D/X frame")
    return output
