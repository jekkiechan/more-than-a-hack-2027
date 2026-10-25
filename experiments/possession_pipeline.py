"""Observation repair, control-context features and probability-level ensembling.

All inference accepts the tracking allowlist only. Parameters are physical or
predeclared experimental choices; nothing is inferred from competition labels.
Historical feature extractors and frozen experiments are left unchanged.
"""

from __future__ import annotations

import numpy as np

from experiments import refine_domain, refine_features, robust, supervised
from experiments.stress import mask_ball


def _repair(values, periods, fps, maximum_speed, gap_seconds):
    """Remove isolated impossible spikes and bridge only plausible short gaps.

    No extrapolation, cross-period interpolation, or long-gap trajectory guesses.
    A spike requires TWO implausible edges and a plausible neighbour-to-neighbour
    bridge, so a persistent displacement (e.g. an identity switch) is not silently
    replaced with the previous track. Both coordinates share one validity mask.
    """
    out = np.array(values, dtype=float, copy=True)
    out[~np.isfinite(out).all(axis=-1)] = np.nan
    removed = np.zeros(out.shape[:-1], dtype=bool)
    filled = np.zeros_like(removed)
    for start, stop in supervised.blocks(periods):
        part = out[start:stop]
        if len(part) >= 3:
            speed = np.linalg.norm(np.diff(part, axis=0), axis=-1) * fps
            bridge = np.linalg.norm(part[2:] - part[:-2], axis=-1) * fps / 2
            spike = ((speed[:-1] > maximum_speed) & (speed[1:] > maximum_speed)
                     & (bridge <= maximum_speed))
            removed[start + 1:stop - 1] = spike
            part[1:-1][spike] = np.nan
        valid = np.flatnonzero(np.isfinite(part).all(axis=1))
        for left, right in zip(valid[:-1], valid[1:]):
            gap = right - left - 1
            if not 0 < gap <= round(gap_seconds * fps):
                continue
            speed = np.linalg.norm(part[right] - part[left]) * fps / (right - left)
            if speed > maximum_speed:
                continue
            alpha = np.arange(1, right - left) / (right - left)
            part[left + 1:right] = part[left] + alpha[:, None] * (part[right] - part[left])
            filled[start + left + 1:start + right] = True
    return out, removed, filled


def preprocess(data):
    """Return independent tracking arrays and observation-quality diagnostics.

    Ball spikes use a generous 65 m/s bound. Player gaps up to 0.2 s are filled
    only below 13 m/s. Ball repair here does not fill natural gaps: the historical
    physical extractor retains its original 0.5 s ball interpolation. New control
    context uses a separate speed-checked 0.5 s interpolation.
    """
    inputs = robust.tracking_inputs(data)
    fps = float(inputs["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    periods = supervised.model_periods(inputs)
    ball = np.array(inputs["ball"], dtype=float, copy=True)
    players = np.array(inputs["players"], dtype=float, copy=True)
    if ball.shape != (len(periods), 2) or players.shape != (len(periods), len(inputs["teams"]), 2):
        raise ValueError("Expected aligned ball/player XY arrays")
    ball_seen = np.isfinite(ball).all(axis=1)
    player_seen = np.isfinite(players).all(axis=2)
    if "ball_observed" in inputs:
        ball_seen &= np.asarray(inputs["ball_observed"], dtype=bool)
    if "player_observed" in inputs:
        player_seen &= np.asarray(inputs["player_observed"], dtype=bool)
    ball[~ball_seen] = np.nan
    players[~player_seen] = np.nan
    ball, ball_spike, _ = _repair(ball, periods, fps, 65., 0.)
    player_spike = np.zeros_like(player_seen)
    player_fill = np.zeros_like(player_seen)
    for slot in range(players.shape[1]):
        players[:, slot], player_spike[:, slot], player_fill[:, slot] = _repair(
            players[:, slot], periods, fps, 13., .2)
    ball_seen &= ~ball_spike
    player_seen &= ~player_spike
    out = dict(inputs, ball=ball, players=players, ball_observed=ball_seen,
               player_observed=player_seen)
    return out, {"ball_spike": ball_spike, "player_spike": player_spike,
                 "player_filled": player_fill}


def degraded_inputs(data, seed):
    """Label-free gaps and same-team slot switches, local to continuous blocks.

    Synthetic stress is supplementary; this is not calibrated camera occlusion.
    The existing 41% ball mask is augmented with 8% per-player missingness and
    three deterministic-random same-team slot switches per block.
    """
    out = robust.tracking_inputs(data)
    periods = supervised.model_periods(out)
    out = mask_ball(dict(out, period=periods), seed=seed)
    out["period"] = np.asarray(data["period"]).copy()
    players = np.array(out["players"], copy=True)
    observed = np.asarray(out.get("player_observed", np.isfinite(players).all(axis=2))).copy()
    rng = np.random.default_rng(seed + 10000)
    fps = float(out["fps"])
    for slot in range(players.shape[1]):
        masked = mask_ball({"ball": players[:, slot], "ball_observed": observed[:, slot],
                            "period": periods, "fps": fps}, missing_rate=.08,
                           mean_gap_s=.6, seed=int(rng.integers(2**31)))
        players[:, slot], observed[:, slot] = masked["ball"], masked["ball_observed"]
    teams = np.asarray(out["teams"])
    for start, stop in supervised.blocks(periods):
        if stop - start < 2:
            continue
        for team in (0, 1):
            slots = np.flatnonzero((teams == team) & np.isfinite(players[start:stop]).any(axis=(0, 2)))
            if len(slots) < 2:
                continue
            for _ in range(3):
                left, right = rng.choice(slots, 2, replace=False)
                cut = int(rng.integers(start + 1, stop))
                players[cut:stop, [left, right]] = players[cut:stop, [right, left]]
                observed[cut:stop, [left, right]] = observed[cut:stop, [right, left]]
    return dict(out, players=players, player_observed=observed)


def _memory(evidence, periods, fps, reverse=False, horizon=5.):
    """Nearest unambiguous control evidence, expiring at a fixed time horizon."""
    value, age = np.zeros(len(evidence)), np.full(len(evidence), horizon)
    for start, stop in supervised.blocks(periods):
        last, elapsed = 0., horizon
        indices = range(stop - 1, start - 1, -1) if reverse else range(start, stop)
        for row in indices:
            if evidence[row] > 0:
                last, elapsed = evidence[row], 0.
            value[row] = last * np.exp(-elapsed / 2.) if elapsed < horizon else 0.
            age[row] = elapsed
            elapsed = min(horizon, elapsed + 1. / fps)
    return value, age


def control_features(inputs, quality):
    """Team-symmetric possession context that remains defined across ball loss."""
    periods = supervised.model_periods(inputs)
    fps = float(inputs["fps"])
    ball, _, filled = _repair(inputs["ball"], periods, fps, 65., .5)
    players = np.asarray(inputs["players"])
    teams = np.asarray(inputs["teams"])
    distance = np.linalg.norm(players - ball[:, None], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    nearest = np.column_stack([np.min(distance[:, teams == team], axis=1) for team in (0, 1)])
    valid = np.isfinite(nearest).all(axis=1)
    gap = np.zeros(len(ball))
    np.subtract(nearest[:, 1], nearest[:, 0], out=gap, where=valid)
    columns, names = [], []

    def add(name, values):
        names.append(name)
        columns.append(np.asarray(values, dtype=float))

    add("context_ball_interpolated", filled)
    add("context_ball_spike", quality["ball_spike"])
    before, after = [], []
    for team, prefix in enumerate(("H", "A")):
        slots = teams == team
        visible = np.isfinite(players[:, slots]).all(axis=2)
        count = visible.sum(axis=1)
        repaired = quality["player_filled"][:, slots].sum(axis=1)
        add(f"{prefix}_context_repaired_fraction", repaired / np.maximum(count, 1))
        advantage = gap if team == 0 else -gap
        evidence = np.where(valid & (nearest[:, team] <= 2.5) & (advantage >= .75),
                            np.exp(-.5 * (nearest[:, team] / 1.8)**2), 0.)
        add(f"{prefix}_context_control", evidence)
        run = np.zeros(len(ball))
        for start, stop in supervised.blocks(periods):
            elapsed = 0.
            for row in range(start, stop):
                elapsed = min(3., elapsed + 1. / fps) if evidence[row] > 0 else 0.
                run[row] = elapsed
        add(f"{prefix}_context_control_run_s", run)
        for reverse, name, collection in ((False, "before", before), (True, "after", after)):
            memory, age = _memory(evidence, periods, fps, reverse)
            add(f"{prefix}_context_control_{name}", memory)
            add(f"{prefix}_context_control_age_{name}_s", age)
            collection.append(memory)
        for seconds in (.25, 1., 3.):
            add(f"{prefix}_context_control_mean_{seconds}s",
                supervised._rolling(evidence, 2 * round(seconds * fps) + 1, periods))
        # Fractions avoid exposing the provider's padded roster size.
        for radius in (3, 8, 15):
            fraction = ((distance[:, slots] <= radius) & visible).sum(axis=1) / np.maximum(count, 1)
            fraction[~np.isfinite(ball).all(axis=1) | (count == 0)] = np.nan
            add(f"{prefix}_context_support_{radius}m", fraction)
    add("home_context_before_advantage", before[0] - before[1])
    add("home_context_after_advantage", after[0] - after[1])
    add("context_control_agreement", (before[0] - before[1]) * (after[0] - after[1]))
    matrix = np.column_stack(columns).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names


def feature_views(data):
    """Three controlled ablations sharing the same tracking-only observations."""
    raw, names = refine_features.features(robust.tracking_inputs(data))
    clean, quality = preprocess(data)
    repaired, repaired_names = refine_features.features(clean)
    if repaired_names != names:
        raise ValueError("Physical feature schemas differ")
    extra, extra_names = control_features(clean, quality)
    return {"baseline": (raw, names), "preprocessed": (repaired, names),
            "context": (np.column_stack((repaired, extra)), names + extra_names)}, clean


def blend_probabilities(base, context, player_dead, context_weight=.5, player_dead_weight=.5):
    """Average team conditionals separately from the independent D probability.

    This avoids an expert's high D probability suppressing its H/A vote. There
    is no fitted stacker and no ground-truth-dependent gating at inference.
    """
    if not 0 <= context_weight <= 1 or not 0 <= player_dead_weight <= 1:
        raise ValueError("Blend weights must be in [0, 1]")
    base, context = np.asarray(base), np.asarray(context)
    player_dead = np.asarray(player_dead)
    if base.ndim != 2 or base.shape[1] != 3 or context.shape != base.shape or player_dead.shape != (len(base),):
        raise ValueError("Expected matching N x 3 probabilities and N dead probabilities")
    for values in (base, context, player_dead):
        if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
            raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.allclose(base.sum(axis=1), 1) or not np.allclose(context.sum(axis=1), 1):
        raise ValueError("Class probabilities must sum to one")
    def teams(probabilities):
        alive = probabilities[:, :2].sum(axis=1, keepdims=True)
        return np.divide(probabilities[:, :2], alive, out=np.full_like(probabilities[:, :2], .5, dtype=float),
                         where=alive > 1e-12)
    team = (1 - context_weight) * teams(base) + context_weight * teams(context)
    full_dead = (1 - context_weight) * base[:, 2] + context_weight * context[:, 2]
    dead = (1 - player_dead_weight) * full_dead + player_dead_weight * player_dead
    return np.column_stack((team * (1 - dead[:, None]), dead))


def predict(data, model):
    """Run a frozen validation-selected model on tracking, with no label access."""
    from experiments.refine_decode import decode

    matrices, clean = feature_views(data)
    probabilities = {name: refine_features.probabilities_from_features(*matrices[name], member)
                     for name, member in model["members"].items()}
    dead_x, names = refine_domain.features(clean)
    if names != model["dead_feature_names"]:
        raise ValueError("Dead-head feature schema differs")
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        dead = model["dead_estimator"].predict_proba(dead_x)[:, 1]
    config = model["candidate"]
    left, right = config["members"]
    p = blend_probabilities(probabilities[left], probabilities[right], dead,
                            config["context_weight"], config["player_dead_weight"])
    return decode(p, supervised.model_periods(clean), float(clean["fps"]), config)
