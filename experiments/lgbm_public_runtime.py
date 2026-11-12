"""Tracking-only runtime adapter for the frozen R2 LightGBM public arm.

The R1 feature builder uses temporal shifts and rolling windows over a record.
This adapter invokes it independently on each contiguous supervised model block
so no feature can leak across a period or timestamp discontinuity.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments import ball_exit_runtime
from experiments import feed_benchmark_audit_v1 as audit
from experiments import spell_frontier_baseline as frontier
from experiments import supervised
from experiments import v2_target_push_refiner as R1


def _probability(components, name, n):
    value = np.asarray(components[name], dtype=np.float64)
    if value.shape != (n, 3) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite H/A/D probabilities with shape (N, 3)")
    if np.any(value < 0) or not np.allclose(value.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError(f"{name} contains invalid probabilities")
    return value


def features(inputs, components):
    """Return R1 names and block-isolated features from tracking/probabilities only."""
    n = len(inputs["ball"])
    ball = np.asarray(inputs["ball"])
    players = np.asarray(inputs["players"])
    if ball.shape != (n, 2) or players.shape[0] != n:
        raise ValueError("Tracking arrays have inconsistent frame counts")
    fps = float(inputs["fps"])
    period = supervised.model_periods(inputs)
    predecessor = _probability(components, "adopted", n)
    calibrated = _probability(components, "calibrated", n)
    obs = np.asarray(inputs.get("ball_observed", np.isfinite(ball).all(axis=1)), dtype=bool)
    if obs.shape != (n,):
        raise ValueError("ball_observed must have shape (N,)")

    names = None
    matrices = []
    for start, stop in supervised.blocks(period):
        cell = SimpleNamespace(
            n=stop - start,
            fps=fps,
            pD={"predecessor": predecessor[start:stop, 2],
                "calibrated": calibrated[start:stop, 2]},
            ball=ball[start:stop],
            ball_obs=obs[start:stop],
            pitch=np.asarray(inputs["pitch"]),
            _players=players[start:stop],
        )
        cell.players = lambda c=cell: c._players
        block_names, matrix = R1.features(cell)
        if names is None:
            names = block_names
        elif block_names != names:
            raise ValueError("R1 feature schema changed across blocks")
        matrices.append(matrix)
    if names is None:
        raise ValueError("At least one frame is required")
    matrix = np.concatenate(matrices, axis=0)
    if matrix.shape != (n, len(names)):
        raise ValueError("Feature rows do not align with tracking frames")
    return names, matrix


def predict_labels(inputs, components, booster):
    """Apply fixed R2_full_w0.5, snap, and frontier decoding to tracking only."""
    n = len(inputs["ball"])
    adopted = _probability(components, "adopted", n)
    calibrated = _probability(components, "calibrated", n)
    ce = _probability(components, "ce", n)
    names, matrix = features(inputs, components)
    expected_names = getattr(booster, "feature_name", lambda: names)()
    if expected_names and list(expected_names) != list(names):
        raise ValueError("LightGBM feature schema mismatch")
    refined = np.asarray(booster.predict(matrix), dtype=np.float64).reshape(-1)
    if refined.shape != (n,) or not np.isfinite(refined).all() or np.any((refined < 0) | (refined > 1)):
        raise ValueError("Booster must return one P(D) value in [0, 1] per frame")

    p_dead = 0.5 * refined + 0.5 * adopted[:, 2]
    # Preserve the predecessor's conditional H/A probabilities, as in R2.
    ha = adopted[:, :2]
    denom = ha.sum(axis=1, keepdims=True)
    conditional = np.divide(ha, denom, out=np.full_like(ha, 0.5), where=denom > 0)
    blended = np.column_stack((conditional * (1.0 - p_dead[:, None]), p_dead))
    period, fps = supervised.model_periods(inputs), float(inputs["fps"])
    dead = audit.decode(blended, period, fps, dict(audit.CONFIG, dt=4, dead=3)) == "D"
    dead = ball_exit_runtime.corrected_mask(inputs, dead)

    # R2 changes only D probability; frontier's deployed H/A recipe is fixed.
    probability = frontier.blend.recipe_probability(
        {"adopted": adopted, "calibrated": calibrated, "ce": ce},
        frontier.RECIPE["ce_weight"], frontier.RECIPE["calibrated_weight"],
        frontier.RECIPE["pooling"],
    )
    return frontier.blend.decode_fixed_d(
        probability, np.where(dead, "D", "H"), period, fps, frontier.RECIPE["ha_cost"]
    )
