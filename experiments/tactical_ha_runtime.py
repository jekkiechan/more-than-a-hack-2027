"""Deployment adapter: Fable snap_m0.0 plus a direction-aware tactical H/A expert.

The adopted pipeline is decoded first, unchanged. A tracking-only tactical expert
(attacking direction, ball progress, team shape; `tactical_ha_features`) then adds
log-odds evidence to the conditional H/A mixture, except inside adopted spells shorter
than `protect_short_s`, and H/A is re-decoded inside the same frozen D mask.
No labels enter prediction; the D mask is never changed.
"""
from __future__ import annotations

import numpy as np

from experiments import ball_exit_runtime as fable
from experiments import spell_frontier_baseline as frontier
from experiments import supervised, tactical_ha_features as tactical

KIND = "sn_dense_spell_frontier_ball_exit_tactical_ensemble"
TACTICAL = {"weight": 0.25, "protect_short_s": 5.0, "probability_clip": 1e-6}
ROLE = "tactical_ha"


def _logit(p):
    c = TACTICAL["probability_clip"]
    p = np.clip(np.asarray(p, dtype=np.float64), c, 1 - c)
    return np.log(p / (1 - p))


def tactical_probability(inputs, model):
    """P(H | live) per native frame: mean of the direct and team-swapped passes."""
    direct, names = tactical.features(inputs)
    swapped, swapped_names = tactical.features(tactical.swap_inputs(inputs))
    if list(names) != list(model["names"]) or swapped_names != names:
        raise ValueError("Tactical feature schema mismatch")
    estimator = model["estimator"]
    return 0.5 * (estimator.predict_proba(direct)[:, 1] + 1 - estimator.predict_proba(swapped)[:, 1])


def evidence_weight(labels, period, fps):
    """Zero inside adopted H/A spells shorter than the protection length, else one."""
    labels, period = np.asarray(labels), np.asarray(period)
    weight = np.ones(len(labels))
    change = np.flatnonzero((labels[1:] != labels[:-1]) | (period[1:] != period[:-1])) + 1
    for a, b in zip(np.r_[0, change], np.r_[change, len(labels)]):
        if labels[a] in ("H", "A") and (b - a) / fps < TACTICAL["protect_short_s"]:
            weight[a:b] = 0.0
    return weight


def combine(probability, tactical_h, weight):
    logit = _logit(probability[:, 0]) + TACTICAL["weight"] * weight * _logit(tactical_h)
    home = 1.0 / (1.0 + np.exp(-logit))
    return np.column_stack((home, 1.0 - home))


def decode_components(inputs, components, tactical_h, return_baseline=False):
    period, fps = supervised.model_periods(inputs), float(inputs["fps"])
    predecessor = frontier.audit.decode(components["adopted"], period, fps, frontier.audit.CONFIG)
    dead = fable.corrected_mask(inputs, np.asarray(predecessor) == "D")
    mask = np.where(dead, "D", "H")
    probability = frontier.blend.recipe_probability(
        components, frontier.RECIPE["ce_weight"], frontier.RECIPE["calibrated_weight"],
        frontier.RECIPE["pooling"])
    baseline = frontier.blend.decode_fixed_d(probability, mask, period, fps, frontier.RECIPE["ha_cost"])
    weight = evidence_weight(baseline, period, fps)
    labels = frontier.blend.decode_fixed_d(combine(probability, tactical_h, weight), mask, period, fps,
                                           frontier.RECIPE["ha_cost"])
    if not np.array_equal(np.asarray(labels) == "D", dead):
        raise RuntimeError("The frozen D mask changed")
    return (labels, baseline) if return_baseline else labels


def frontier_manifest(manifest):
    arch = manifest["architecture"]
    if arch.get("kind") != KIND or arch.get("ball_exit") != fable.CORRECTION or arch.get("tactical") != TACTICAL:
        raise ValueError("Expected the frozen tactical deployment recipe")
    parent = {k: v for k, v in arch.items() if k != "tactical"}
    parent["kind"] = frontier.KIND
    return dict(manifest, architecture=parent)


def predict_labels(inputs, models, manifest):
    parent = frontier_manifest(manifest)
    gru = {name: model for name, model in models.items() if name != ROLE}
    components = frontier.native_components(inputs, gru, parent)
    return decode_components(inputs, components, tactical_probability(inputs, models[ROLE]))
