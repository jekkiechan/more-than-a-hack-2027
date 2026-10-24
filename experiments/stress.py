"""Synthetic observation-gap stress test; not a simulation of camera occlusion.

The mask is independent of ownership, ball status, and every reference label.
Uniform random missingness is easier than real visibility failures concentrated
around duels and transitions, so these results are supplementary only.
"""

import numpy as np


def mask_ball(data, missing_rate=0.41, mean_gap_s=1.0, seed=2026):
    """Remove ball observations in deterministic Markov blocks within each half.

    missing_rate describes the generated mask, not the final fraction missing
    when the source already contains gaps. The stationary expected rate is used;
    the realized rate is reported by callers and is not forced using any labels.
    """
    if not 0 <= missing_rate < 1 or mean_gap_s <= 0:
        raise ValueError("Need 0 <= missing_rate < 1 and positive mean gap")
    result = dict(data)
    ball = np.asarray(data["ball"]).copy()
    fps = float(data["fps"])
    period = np.asarray(data["period"])
    if fps <= 0 or period.shape != (len(ball),):
        raise ValueError("Invalid fps or period shape")
    rng = np.random.default_rng(seed)
    missing = np.zeros(len(ball), dtype=bool)
    if missing_rate:
        end_gap = min(1.0, 1.0 / (mean_gap_s * fps))
        start_gap = min(1.0, missing_rate * end_gap / (1.0 - missing_rate))
        hidden = False
        for i in range(len(ball)):
            if i == 0 or period[i] != period[i - 1]:
                hidden = rng.random() < missing_rate
            elif hidden:
                hidden = not (rng.random() < end_gap)
            else:
                hidden = rng.random() < start_gap
            missing[i] = hidden
    ball[missing] = np.nan
    result["ball"] = ball
    source_observed = np.asarray(data.get(
        "ball_observed", np.isfinite(data["ball"]).all(axis=1)), dtype=bool)
    result["ball_observed"] = source_observed & ~missing
    return result
