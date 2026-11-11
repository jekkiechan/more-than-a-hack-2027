"""Count-preserving relocation of decoded H/A boundaries using posterior evidence.

Inference is label-free. ``period`` must already encode period and timestamp
breaks (``supervised.model_periods`` is suitable). X and D frames are immutable.
"""
from __future__ import annotations

import numpy as np

STRENGTHS = {"baseline": 0.0, "shift_0.2": 0.2, "shift_0.4": 0.4, "shift_0.8": 0.8}
DISPLACEMENT_PENALTY = 0.1  # natural-log posterior evidence units per frame moved


def relocate(probabilities, labels, period, timestamps, fps, strength="baseline"):
    """Relocate H/A transitions within runs; return a copy of labels.

    Inputs are frame probabilities in H,A,D order, decoded H/A/D/X labels,
    model-period ids (including timestamp gaps), seconds timestamps, and fps.
    Each eligible boundary receives a posterior-median estimate within a
    non-overlapping corridor. Boundaries are then applied simultaneously.
    """
    if strength not in STRENGTHS:
        raise ValueError(f"Unknown strength: {strength}")
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels)
    period = np.asarray(period)
    t = np.asarray(timestamps, dtype=float)
    if (p.ndim != 2 or p.shape[1] != 3 or y.ndim != 1 or period.ndim != 1
            or t.ndim != 1 or len(y) == 0 or len(p) != len(y)
            or len(period) != len(y) or len(t) != len(y)
            or not np.isfinite(p).all() or (p < 0).any()
            or not np.isfinite(t).all() or not np.isfinite(fps) or fps <= 0
            or np.any(~np.isin(y, ["H", "A", "D", "X"]))):
        raise ValueError("Invalid probabilities, labels, periods, timestamps, or fps")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probabilities must sum to one per frame")
    out = y.copy()
    limit = round(STRENGTHS[strength] * fps)
    if limit == 0:
        return out
    # Break further on irregular time intervals even if caller's period ids omit them.
    delta = np.diff(t)
    breaks = np.r_[True, (period[1:] != period[:-1]) | (delta <= 0) | (delta > 1.5 / fps)]
    starts = np.flatnonzero(breaks)
    stops = np.r_[starts[1:], len(y)]
    for a, b in zip(starts, stops):
        _relocate_block(p[a:b], out[a:b], limit, a)
    return out


def _relocate_block(p, y, limit, offset):
    edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
    states = y[edges[:-1]]
    # Only an uninterrupted sequence of H/A runs can be moved. D/X borders pin it.
    i = 0
    while i < len(states):
        if states[i] not in ("H", "A"):
            i += 1
            continue
        j = i + 1
        while j < len(states) and states[j] in ("H", "A"):
            j += 1
        if j - i >= 2:
            _optimize_segment(p, y, edges, states, i, j, limit)
        i = j


def _optimize_segment(p, y, edges, states, first, last, limit):
    old = edges[first + 1:last]
    # Independent posterior-median estimates, each confined to its Voronoi
    # corridor between neighboring original boundaries. Corridors cannot
    # overlap/cross; the fixed endpoints preserve every run simultaneously.
    chosen_edges = []
    for k, boundary in enumerate(old):
        lo = max(edges[first]+1, int(boundary)-limit)
        hi = min(edges[last]-1, int(boundary)+limit)
        if k:
            lo = max(lo, (int(old[k-1])+int(boundary))//2 + 1)
        if k+1 < len(old):
            hi = min(hi, (int(boundary)+int(old[k+1]))//2)
        candidates = np.arange(lo, hi+1)
        if len(candidates) == 0:
            chosen_edges.append(int(boundary))
            continue
        # Score complete segment labels under conditional H/A posteriors.
        left_team = 0 if states[first+k] == "H" else 1
        right_team = 1-left_team
        start = edges[first+k]
        stop = edges[first+k+2]
        lp = np.log(np.maximum(p[start:stop, :2], 1e-12))
        pref = np.vstack([np.zeros(2), np.cumsum(lp, axis=0)])
        scores = np.array([pref[int(c)-start, left_team] +
                           pref[-1, right_team]-pref[int(c)-start, right_team] -
                           DISPLACEMENT_PENALTY*abs(int(c)-int(boundary))
                           for c in candidates])
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        chosen_edges.append(int(candidates[np.searchsorted(np.cumsum(weights), .5, side="left")]))
    bounds = [int(edges[first]), *chosen_edges, int(edges[last])]
    for k, label in enumerate(states[first:last]):
        y[bounds[k]:bounds[k+1]] = label
