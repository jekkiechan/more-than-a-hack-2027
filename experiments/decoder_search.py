"""Target-independent postprocessing alternatives for a bounded decoder study.

Numba is an optional accelerator (study runtime: 0.61.2). The same functions
execute as plain Python when it is absent; no accelerator enters the model.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

from experiments import supervised

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f


@njit(cache=True)
def _path(logp, costs, edge):
    n, k = logp.shape
    back = np.zeros((n, k), dtype=np.int32)
    value = logp[0].copy()
    for t in range(1, n):
        nxt = np.empty(k)
        for j in range(k):
            best, source = -np.inf, 0
            for i in range(k):
                candidate = value[i] - costs[i, j] * edge[t]
                if candidate > best:
                    best, source = candidate, i
            back[t, j] = source
            nxt[j] = best + logp[t, j]
        value = nxt
    state = np.argmax(value)
    result = np.empty(n, dtype=np.int8)
    for t in range(n-1, -1, -1):
        result[t] = state
        state = back[t, state]
    return result


@njit(cache=True)
def _minimum_duration(logp, costs, minimum):
    """Exact segment DP with lower duration bounds; first/last runs censored."""
    n, k = logp.shape
    prefix = np.zeros((n+1, k))
    for t in range(n):
        prefix[t+1] = prefix[t] + logp[t]
    score = np.full((n+1, k), -np.inf)
    starts = np.zeros((n+1, k), dtype=np.int32)
    prev = np.full((n+1, k), -1, dtype=np.int32)
    best = np.zeros(k)  # start at zero, no entry cost
    best_start = np.zeros(k, dtype=np.int32)
    best_prev = np.full(k, -1, dtype=np.int32)
    for t in range(1, n+1):
        for j in range(k):
            u = t - minimum[j]
            if u > 0:
                for i in range(k):
                    if i == j:
                        continue
                    candidate = score[u, i] - costs[i, j] - prefix[u, j]
                    if candidate > best[j]:
                        best[j], best_start[j], best_prev[j] = candidate, u, i
            score[t, j] = prefix[t, j] + best[j]
            starts[t, j], prev[t, j] = best_start[j], best_prev[j]
    # The recording can end part-way through a spell.
    for j in range(k):
        for u in range(max(1, n-minimum[j]+1), n):
            for i in range(k):
                if i != j:
                    candidate = score[u, i] - costs[i, j] + prefix[n, j] - prefix[u, j]
                    if candidate > score[n, j]:
                        score[n, j], starts[n, j], prev[n, j] = candidate, u, i
    result = np.empty(n, dtype=np.int8)
    t, state = n, np.argmax(score[n])
    while t > 0:
        u = starts[t, state]
        result[u:t] = state
        state, t = prev[t, state], u
    return result


@njit(cache=True)
def _hysteresis(p, threshold, dwell):
    """Backdate a persistent high-confidence challenger to its first frame."""
    n = len(p)
    result = np.empty(n, dtype=np.int8)
    state = np.argmax(p[0])
    challenger, count = -1, 0
    for t in range(n):
        top = np.argmax(p[t])
        if top != state and p[t, top] >= threshold:
            if challenger != top:
                challenger, count = top, 0
            count += 1
            if count >= dwell:
                state = top
                result[t-count+1:t] = state
                challenger, count = -1, 0
        else:
            challenger, count = -1, 0
        result[t] = state
    return result


def configurations():
    """Finite search space, fixed before scoring (no post-result extensions)."""
    arms = [{"family": "matrix", "ha": h, "dead": d}
            for h in (0, .5, 1, 2, 3, 4, 6, 9, 15)
            for d in (1, 3, 6, 12)]
    arms += [{"family": "factorized", "ha": h, "dead": d}
             for h in (0, .5, 1, 2, 3, 6, 12) for d in (1, 3, 6)]
    arms += [{"family": kind, "seconds": s, "cost": c}
             for kind in ("gaussian", "median")
             for s in (.1, .25, .5, 1, 2) for c in (0, 3)]
    arms += [{"family": "duration", "ha": h, "dead": 3, "seconds": s}
             for h in (1, 3, 6) for s in (.2, .5, 1, 2)]
    arms += [{"family": "hysteresis", "threshold": q, "seconds": s}
             for q in (.5, .65, .8) for s in (.1, .3, .6)]
    arms += [{"family": "clip", "cap": cap, "ha": h, "dead": 3}
             for cap in (.5, 1, 2) for h in (1, 3, 6)]
    arms += [{"family": "adaptive", "gain": g, "ha": h, "dead": 3}
             for g in (1, 4, 16) for h in (1, 3, 6)]
    arms += [{"family": "prune", "seconds": s, "confidence": q}
             for s in (.5, 1, 2, 3) for q in (.55, .7, .85)]
    arms += [{"family": "dead_bias", "bias": b, "ha": 3, "dead": 3}
             for b in (-.75, -.5, -.25, .25, .5, .75)]
    for i, arm in enumerate(arms):
        arm["id"] = f"a{i:03d}"
    return arms


def _costs(ha, dead, fps):
    return np.array([[0, ha, dead], [ha, 0, dead], [dead, dead, 0]]) * (fps / 10)


def decode(probabilities, model_period, fps, config):
    """H/A/D inference; accepts no truth labels, provider ID or target mask."""
    p = np.asarray(probabilities, dtype=np.float64)
    period = np.asarray(model_period)
    if p.ndim != 2 or p.shape != (len(period), 3) or not len(p):
        raise ValueError("expected nonempty Nx3 probabilities and N periods")
    if (not np.isfinite(p).all() or (p < 0).any() or
            (p.sum(axis=1) <= 0).any() or not np.isfinite(fps) or fps <= 0):
        raise ValueError("invalid probabilities or fps")
    p = p / p.sum(axis=1, keepdims=True)
    available = p.sum(axis=0) > 0
    family = config["family"]
    result = np.empty(len(p), dtype=np.int8)
    for a, b in supervised.blocks(period):
        q = p[a:b].copy()
        logp = np.log(np.maximum(q, 1e-7))
        cost = _costs(config.get("ha", config.get("cost", 3)),
                      config.get("dead", config.get("cost", 3)), fps)
        edge = np.ones(b-a)
        if family in ("gaussian", "median"):
            width = config["seconds"] * fps
            q = (gaussian_filter1d(q, width, axis=0, mode="nearest")
                 if family == "gaussian" else
                 median_filter(q, size=(max(1, round(width)*2+1), 1), mode="nearest"))
            q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-30)
            logp = np.log(np.maximum(q, 1e-7))
        elif family == "dead_bias":
            logp[:, 2] += config["bias"]
        elif family == "clip":
            # Cap conditional-team evidence while preserving the independent D head.
            odds = np.clip(logp[:, 0]-logp[:, 1], -config["cap"], config["cap"])
            logp[:, 0] = np.log(np.maximum(1-q[:, 2], 1e-7)) - np.logaddexp(0, -odds)
            logp[:, 1] = np.log(np.maximum(1-q[:, 2], 1e-7)) - np.logaddexp(0, odds)
        elif family == "adaptive":
            smooth = gaussian_filter1d(q, max(.1*fps, 1), axis=0, mode="nearest")
            lag = max(1, round(.25*fps))
            left = smooth[np.maximum(np.arange(b-a)-lag, 0)]
            right = smooth[np.minimum(np.arange(b-a)+lag, b-a-1)]
            edge = 1 / (1 + config["gain"] * np.abs(right-left).sum(axis=1))
        logp[:, ~available] = -np.inf
        if family == "factorized" and available[:2].all():
            conditional = logp[:, :2] - np.log(np.maximum(1-q[:, 2:3], 1e-7))
            teams = _path(conditional, cost[:2, :2], edge)
            alive_dead = np.column_stack((np.log(np.maximum(1-q[:, 2], 1e-7)), logp[:, 2]))
            dead_path = _path(alive_dead, np.array([[0., cost[0, 2]], [cost[0, 2], 0.]]), edge)
            path = np.where(dead_path == 1, 2, teams).astype(np.int8)
        elif family == "duration":
            classes = np.flatnonzero(available)
            minimum = np.array([
                max(1, round(config["seconds"]*fps)),
                max(1, round(config["seconds"]*fps)), 1])
            path = classes[_minimum_duration(logp[:, classes], cost[np.ix_(classes, classes)],
                                              minimum[classes])].astype(np.int8)
        elif family == "hysteresis":
            path = _hysteresis(q, config["threshold"], max(1, round(config["seconds"]*fps)))
        else:
            path = _path(logp, cost, edge)
            if family == "prune":
                # Remove uncertain, short islands only when both neighbors agree.
                edges = np.r_[0, np.flatnonzero(np.diff(path))+1, len(path)]
                original = path.copy()
                for i in range(1, len(edges)-2):
                    u, v = edges[i:i+2]
                    if (v-u <= config["seconds"]*fps and original[u-1] == original[v]
                            and q[u:v, original[u]].mean() < config["confidence"]):
                        path[u:v] = original[u-1]
        result[a:b] = path
    return supervised.LABELS[result]
