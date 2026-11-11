"""Tracking-only deployment adapter for Fable's frozen snap_m0.0 rule."""
from __future__ import annotations

import numpy as np

from experiments import ball_exit_dmask_v1 as frozen
from experiments import spell_frontier_baseline as frontier
from experiments import supervised, tracking_corruption

KIND = "sn_dense_spell_frontier_ball_exit_ensemble"
CORRECTION = {"rule": "snap", "margin_m": 0.0, "delay_window_s": 5.0,
              "advance_window_s": 3.0}


def corrected_mask(inputs, dead):
    clean = tracking_corruption.inputs_only(inputs)
    inside, outside = frozen.ball_state(clean["ball"], clean["pitch"], 0.0)
    return frozen.edit_mask(dead, inside, outside, supervised.model_periods(clean),
                            float(clean["fps"]), "snap")


def frontier_manifest(manifest):
    if (manifest["architecture"].get("kind") != KIND or
            manifest["architecture"].get("ball_exit") != CORRECTION):
        raise ValueError("Expected the frozen snap_m0.0 deployment recipe")
    return dict(manifest, architecture=dict(manifest["architecture"], kind=frontier.KIND))


def decode_components(inputs, components):
    period, fps = supervised.model_periods(inputs), float(inputs["fps"])
    predecessor = frontier.audit.decode(components["adopted"], period, fps, frontier.audit.CONFIG)
    dead = corrected_mask(inputs, np.asarray(predecessor) == "D")
    probability = frontier.blend.recipe_probability(
        components, frontier.RECIPE["ce_weight"], frontier.RECIPE["calibrated_weight"],
        frontier.RECIPE["pooling"])
    return frontier.blend.decode_fixed_d(probability, np.where(dead, "D", "H"),
                                         period, fps, frontier.RECIPE["ha_cost"])


def predict_labels(inputs, models, manifest):
    parent = frontier_manifest(manifest)
    return decode_components(inputs, frontier.native_components(inputs, models, parent))
