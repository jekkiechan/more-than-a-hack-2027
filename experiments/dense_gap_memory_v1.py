"""Label-free carry of the last reliable team signal through ball gaps.

This is inference-only and uses the fixed declared rule: it uses the current
team probabilities at observed-ball frames as the reliability evidence, then
adds a fixed exponentially decaying bias during missing-ball frames.
"""
from __future__ import annotations

import numpy as np


RELIABILITY = 0.80
MIN_TEAM_MARGIN = 0.20
MAX_LOG_ODDS = 0.25
HALF_LIFE_SECONDS = 5.0
STUDY_ID = "dense_gap_memory_v1"
HYPOTHESIS = (
    "A last reliable, observed-ball H/A control estimate with a decaying age "
    "bias improves dense SoccerNet ensemble spell recovery during missing-ball "
    "gaps without harming clean or feed-like full-provider holdout performance."
)
ARMS = ("seed_2026", "seed_2027", "equal_probability_ensemble")
VIEWS = ("clean", "feed_like")
DECLARED_LIMITS = {
    "reliability": RELIABILITY,
    "minimum_team_margin": MIN_TEAM_MARGIN,
    "maximum_log_odds": MAX_LOG_ODDS,
    "half_life_seconds": HALF_LIFE_SECONDS,
    "gap_only": True,
    "team_symmetric": True,
    "learned_components": False,
    "inference_labels": False,
}


def apply_memory(probabilities, ball_observed, timestamps, period, *,
                 reliability=RELIABILITY, min_team_margin=MIN_TEAM_MARGIN,
                 max_log_odds=MAX_LOG_ODDS, half_life_seconds=HALF_LIFE_SECONDS):
    """Adjust categorical H/A/D[,X] probabilities using the last reliable anchor.

    The probability columns are H,A,D[,X]. An anchor is accepted only on an
    observed-ball frame when its larger H/A probability is at least reliability
    and its H/A margin meets min_team_margin. On a missing-ball frame, a bounded
    log-odds bias is applied to that same team, decaying with elapsed age. Any
    period change or non-increasing timestamp clears the anchor. All evidence
    comes from inference inputs; labels are never accepted by this function.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    observed_raw = np.asarray(ball_observed)
    if observed_raw.dtype.kind != "b":
        raise ValueError("ball_observed must be a boolean mask")
    observed = observed_raw.astype(bool, copy=False)
    times = np.asarray(timestamps, dtype=np.float64)
    periods = np.asarray(period)
    if p.ndim != 2 or p.shape[1] not in (3, 4):
        raise ValueError("probabilities must have columns H,A,D[,X]")
    n = len(p)
    if any(x.shape != (n,) for x in (observed, times, periods)):
        raise ValueError("observation, timestamps and period must match rows")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError("probabilities must be finite categorical rows")
    params = np.asarray([reliability, min_team_margin, max_log_odds, half_life_seconds],
                        dtype=np.float64)
    if (not np.isfinite(times).all() or not np.isfinite(params).all() or
            reliability <= 0.5 or reliability >= 1):
        raise ValueError("invalid timestamps or reliability")
    if min_team_margin < 0 or max_log_odds < 0 or half_life_seconds <= 0:
        raise ValueError("invalid memory parameters")

    out = p.copy()
    anchor_team = None
    anchor_time = None
    previous_time = None
    previous_period = None
    for i in range(n):
        if (previous_time is not None and
                (times[i] <= previous_time or periods[i] != previous_period)):
            anchor_team = anchor_time = None
        if observed[i]:
            h, a = p[i, 0], p[i, 1]
            team = int(a > h)
            if max(h, a) >= reliability and abs(h - a) >= min_team_margin:
                anchor_team, anchor_time = team, times[i]
            else:
                anchor_team = anchor_time = None
        elif anchor_team is not None:
            age = max(0.0, times[i] - anchor_time)
            bias = max_log_odds * 2.0 ** (-age / half_life_seconds)
            favored, opposed = (1, 0) if anchor_team else (0, 1)
            # Transfer only between H and A, preserving every other class.
            log_ratio = np.log(max(p[i, favored], 1e-12) / max(p[i, opposed], 1e-12))
            new_ratio = log_ratio + bias
            mass = p[i, favored] + p[i, opposed]
            favored_p = mass / (1.0 + np.exp(-np.clip(new_ratio, -60, 60)))
            out[i, favored] = favored_p
            out[i, opposed] = mass - favored_p
        previous_time, previous_period = times[i], periods[i]
    return out
