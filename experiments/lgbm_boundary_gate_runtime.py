"""Label-free runtime for the confidence-gated boundary refiner (`boundary_conf0.6`).

The pooled LightGBM dead-ball refiner proposes a new D mask. Only edits that move
the start or end of an existing baseline D run are applied, and only when the
refiner's own mean P(D) inside the edit is decisive. Runs are never created,
deleted, split or merged, so the D-run count of the adopted baseline is kept.
Tracking and model probabilities are the only inputs; no labels are read.
"""
from __future__ import annotations

import numpy as np

from experiments import ball_exit_runtime
from experiments import feed_benchmark_audit_v1 as audit
from experiments import lgbm_public_runtime as refiner_runtime
from experiments import spell_frontier_baseline as frontier
from experiments import supervised

THRESHOLD = 0.6
BLEND = 0.5


def gate_edits(base_dead, refined_dead, refined_pd, model_period, threshold=THRESHOLD):
    """Return the baseline D mask with confident boundary shifts applied.

    An edit is a maximal run of frames where the two masks differ inside one
    baseline run. It is kept when it touches exactly one end of that run (a
    boundary shift) and the mean refiner P(D) over the edit is at least
    ``threshold`` for live-to-D edits or at most ``1 - threshold`` for D-to-live.
    """
    base = np.asarray(base_dead, dtype=bool)
    cand = np.asarray(refined_dead, dtype=bool)
    pd = np.asarray(refined_pd, dtype=np.float64)
    period = np.asarray(model_period)
    if not (base.shape == cand.shape == pd.shape == period.shape) or base.ndim != 1:
        raise ValueError("Masks, probabilities and periods must be aligned vectors")
    if not 0.5 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0.5, 1]")
    out = base.copy()
    if not len(base):
        return out
    change = (base[1:] != base[:-1]) | (np.diff(period) != 0)
    bounds = np.r_[0, np.flatnonzero(change) + 1, len(base)]
    run_of = np.repeat(np.arange(len(bounds) - 1), np.diff(bounds))
    diff = base != cand
    code = np.where(diff, run_of + 1, 0)
    edges = np.r_[0, np.flatnonzero(code[1:] != code[:-1]) + 1, len(code)]
    for start, stop in zip(edges[:-1], edges[1:]):
        if not diff[start]:
            continue
        run_start, run_stop = bounds[run_of[start]], bounds[run_of[start] + 1]
        if (start == run_start) == (stop == run_stop):
            continue  # whole-run deletion/fill, or an island inside the run
        mean = float(pd[start:stop].mean())
        if (mean <= 1.0 - threshold) if base[start] else (mean >= threshold):
            out[start:stop] = cand[start:stop]
    return out


def dead_masks(inputs, components, refined_pd):
    """Baseline and unrestricted-refiner D masks after the frozen ball-exit snap."""
    adopted = np.asarray(components["adopted"], dtype=np.float64)
    period, fps = supervised.model_periods(inputs), float(inputs["fps"])
    base = np.asarray(audit.decode(adopted, period, fps, audit.CONFIG)) == "D"
    base = ball_exit_runtime.corrected_mask(inputs, base)
    p_dead = BLEND * refined_pd + (1.0 - BLEND) * adopted[:, 2]
    ha = adopted[:, :2]
    denom = ha.sum(axis=1, keepdims=True)
    conditional = np.divide(ha, denom, out=np.full_like(ha, 0.5), where=denom > 0)
    blended = np.column_stack((conditional * (1.0 - p_dead[:, None]), p_dead))
    refined = audit.decode(blended, period, fps, dict(audit.CONFIG, dt=4, dead=3)) == "D"
    return base, ball_exit_runtime.corrected_mask(inputs, refined)


def refined_probability(inputs, components, booster):
    names, matrix = refiner_runtime.features(inputs, components)
    expected = getattr(booster, "feature_name", lambda: names)()
    if expected and list(expected) != list(names):
        raise ValueError("LightGBM feature schema mismatch")
    refined = np.asarray(booster.predict(matrix), dtype=np.float64).reshape(-1)
    if refined.shape != (len(matrix),) or not np.isfinite(refined).all() or np.any((refined < 0) | (refined > 1)):
        raise ValueError("Booster must return one P(D) value in [0, 1] per frame")
    return refined


def predict_labels(inputs, components, booster, threshold=THRESHOLD):
    """Adopted pipeline with the gated refiner D mask and the unchanged H/A decode."""
    refined_pd = refined_probability(inputs, components, booster)
    base, refined = dead_masks(inputs, components, refined_pd)
    period, fps = supervised.model_periods(inputs), float(inputs["fps"])
    dead = gate_edits(base, refined, refined_pd, period, threshold)
    probability = frontier.blend.recipe_probability(
        {k: components[k] for k in ("adopted", "calibrated", "ce")},
        frontier.RECIPE["ce_weight"], frontier.RECIPE["calibrated_weight"], frontier.RECIPE["pooling"])
    return frontier.blend.decode_fixed_d(probability, np.where(dead, "D", "H"), period, fps,
                                         frontier.RECIPE["ha_cost"])
