"""Label-free postprocessing pipelines for `v2_postproc_v1`.

Every pipeline accepts one cell's cached emissions plus inference time breaks
and returns H/A/D labels. Labels never enter any function in this module.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from experiments import v2_postproc_v1_score as S
from experiments import v2_spell_balanced_eval as balanced
from experiments import feed_spell_push as fsp


def cond(p):
    ha = np.asarray(p, dtype=np.float64)[:, :2]
    return ha / ha.sum(axis=1, keepdims=True)


def mixture(cell, weights, pooling="arithmetic"):
    """weights: {family: w}. Conditional H/A expert pooling (as the frontier)."""
    items = [(cond(S.family_mean(cell, f)), w) for f, w in weights.items() if w > 0]
    if pooling == "arithmetic":
        out = sum(w * p for p, w in items)
    else:
        logits = sum(w * np.log(np.clip(p, 1e-12, 1)) for p, w in items)
        logits -= logits.max(axis=1, keepdims=True)
        out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def d_mask_pred(cell, family="cand", **decoder):
    if family == "mix":
        p = 0.5 * (S.family_mean(cell, "cand") + S.family_mean(cell, "adop"))
    else:
        p = S.family_mean(cell, family)
    if not decoder:
        return S.decode_f062(p, cell)
    config = dict(S.base.audit.CONFIG, **decoder)
    return fsp.decode(p[None], cell["model_period"], cell["fps"], config)


def family_f062(cell, family):
    return S.decode_f062(S.family_mean(cell, family), cell)


def blend(cell, weights, ha_cost=3.0, pooling="arithmetic", d_family="cand", d_decoder=None):
    d_pred = d_mask_pred(cell, d_family, **(d_decoder or {}))
    prob = mixture(cell, weights, pooling)
    return balanced.decode_fixed_d(prob, d_pred, cell["model_period"], cell["fps"], ha_cost)


def P(fn, **kw):
    return partial(fn, **kw)


def _runs(mask):
    m = np.concatenate(([False], mask, [False]))
    d = np.diff(m.astype(np.int8))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def insert_short(pred, prob, cell, q=0.4, min_s=0.2, max_s=3.0, smooth_s=0.0):
    """Relabel short runs where the opposing team's conditional H/A probability
    exceeds q inside a decoded live span. Uses probabilities and fps only."""
    from scipy.ndimage import uniform_filter1d
    pred = np.asarray(pred).copy()
    p_home = np.asarray(prob, dtype=np.float64)[:, 0]
    fps = cell["fps"]
    if smooth_s > 0:
        p_home = uniform_filter1d(p_home, max(1, int(round(smooth_s * fps))), mode="nearest")
    lo, hi = int(round(min_s * fps)), int(round(max_s * fps))
    for team, other, p_other in (("H", "A", 1 - p_home), ("A", "H", p_home)):
        for a, b in _runs((pred == team) & (p_other > q)):
            if lo <= b - a <= hi:
                pred[a:b] = other
    return pred


def blend_insert(cell, weights, ha_cost=3.0, q=0.4, min_s=0.2, max_s=3.0, smooth_s=0.0,
                 d_family="cand"):
    base_pred = blend(cell, weights, ha_cost=ha_cost, d_family=d_family)
    prob = mixture(cell, weights)
    return insert_short(base_pred, prob, cell, q=q, min_s=min_s, max_s=max_s, smooth_s=smooth_s)
