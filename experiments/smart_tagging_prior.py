"""Conservative, bounded H/A evidence from public Smart Tagging intervals.

This adapter consumes the records returned by ``starter.load_smart_tagging``.
Smart Tagging describes broad phases, not possession labels. Only explicitly
classified attack/defense phase codes contribute evidence; goals, restarts,
generic phases, and unknown codes are deliberately neutral.

No target labels or model fitting are accepted here. D probability is copied
unchanged, while a fixed, capped log-odds offset adjusts only the conditional
H/A ratio. The default evidence is disabled on observed-ball frames; set
``missing_ball_only=False`` to apply it throughout tagged intervals.
"""
from __future__ import annotations

import math
import re

import numpy as np


# Official DATA.md codes only. Unrecognized codes stay neutral. In particular,
# set pieces, goals, and player-count tags are absent.
ATTACKING_CODES = frozenset({
    "build up",
    "progression",
    "ball in final third",
    "ball in the box",
    "creating chances",
    "long ball",
    "attacking transition",
})

DEFENDING_CODES = frozenset({
    "defending in defensive third",
    "defending in middle third",
    "defending in attacking third",
    "defensive transition",
})

MAX_LOG_ODDS = math.log(2.0)
TAPER_SECONDS = 1.0


def _norm(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", str(value).strip().lower()))


def classify_smart_tagging_event(event, team_to_side):
    """Return ``(category, side)`` using exact official code names.

    ``category`` is ``attacking``, ``defending``, or ``ignored``. ``side`` is
    the mapped H/A value of the tagged club, or ``None`` for an unknown team.
    A defending event implies possession by the opposite side.
    """
    if not isinstance(event, dict):
        return "ignored", None
    code = _norm(event.get("code"))
    if code in ATTACKING_CODES:
        category = "attacking"
    elif code in DEFENDING_CODES:
        category = "defending"
    else:
        return "ignored", None
    mapping = {_norm(team): side for team, side in (team_to_side or {}).items()}
    return category, mapping.get(_norm((event.get("tags") or {}).get("Team")))


def _half_period(tags, period_values):
    """Resolve a supplied Half label to the common numeric period encoding."""
    half = _norm((tags or {}).get("Half"))
    if half in {"1st half", "first half", "1", "period 1"}:
        return 1
    if half in {"2nd half", "second half", "2", "period 2"}:
        return 2
    # If the event has no Half tag, allow it only when the input has one
    # positive contiguous period. This avoids guessing across a half break.
    positive = np.unique(period_values[period_values > 0])
    if len(positive) == 1:
        return positive[0]
    return None


def apply_smart_tagging_prior(
    probabilities,
    events,
    team_to_side,
    fps,
    period,
    ball_observed=None,
    *,
    missing_ball_only=True,
    timestamps=None,
):
    """Apply capped Smart Tagging evidence to native ``[H, A, D]`` probabilities.

    Args:
        probabilities: finite ``(n, 3)`` probability array with columns H, A, D.
        events: loader output records (``code``, frame bounds, and ``tags``).
        team_to_side: explicit club-name mapping to ``"H"`` or ``"A"``.
        fps: positive frame rate used for the fixed one-second edge taper.
        period: length-n period IDs; nonpositive rows and period transitions are
            excluded. Event ``Half`` tags are matched to numeric periods 1/2.
        ball_observed: optional length-n boolean ball-validity mask. If omitted,
            ``missing_ball_only`` causes a no-op.
        missing_ball_only: gate tag evidence to missing-ball frames (default).
        timestamps: optional length-n seconds; discontinuities reset the taper.

    Returns a fresh array with the input floating dtype. Duplicate tags are invariant, simultaneous
    opposing tags cancel to neutral, total absolute conditional H/A log-odds
    change is at most log(2), and D values are copied bit-for-bit.
    """
    p = np.asarray(probabilities)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3) in H/A/D order")
    if not np.issubdtype(p.dtype, np.floating):
        raise ValueError("probabilities must use a floating-point dtype")
    n = p.shape[0]
    periods = np.asarray(period)
    if periods.shape != (n,):
        raise ValueError("period must have shape (n,)")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("probabilities must be finite, nonnegative, and row-normalized")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    if ball_observed is not None:
        ball_observed = np.asarray(ball_observed, dtype=bool)
        if ball_observed.shape != (n,):
            raise ValueError("ball_observed must have shape (n,)")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.shape != (n,) or not np.isfinite(timestamps).all():
            raise ValueError("timestamps must be a finite length-n array")

    # A true no-op preserves dtype and every bit when no usable evidence exists.
    if not events or (missing_ball_only and ball_observed is None):
        return p.copy()

    mapping = {_norm(team): side for team, side in (team_to_side or {}).items()}
    if any(side not in {"H", "A"} for side in mapping.values()):
        raise ValueError("team_to_side values must be 'H' or 'A'")

    # Track strongest tapered evidence for each side. max (not sum) makes
    # duplicates and correlated overlaps unable to increase confidence.
    evidence = np.zeros((n, 2), dtype=np.float64)
    side_col = {"H": 0, "A": 1}
    half_values = np.asarray(periods)
    for event in events:
        category, side = classify_smart_tagging_event(event, team_to_side)
        if category == "ignored" or side is None:
            continue
        tags = event.get("tags") or {}
        start, end = event.get("start_frame"), event.get("end_frame")
        if not isinstance(start, (int, np.integer)) or not isinstance(end, (int, np.integer)):
            continue
        start, end = max(0, int(start)), min(n, int(end))
        if end <= start:
            continue
        event_period = _half_period(tags, half_values)
        if event_period is None:
            continue
        frame = np.arange(start, end)
        valid = (half_values[start:end] == event_period) & (half_values[start:end] > 0)
        if missing_ball_only:
            valid &= ~ball_observed[start:end]
        if not valid.any():
            continue
        # Taper independently inside each contiguous run. This resets at data
        # gaps and period/time breaks, so evidence cannot bridge a break.
        indices = frame[valid]
        discontinuity = np.zeros(max(0, len(indices) - 1), dtype=bool)
        if len(indices) > 1:
            discontinuity |= np.diff(indices) != 1
            discontinuity |= half_values[indices[1:]] != half_values[indices[:-1]]
            if timestamps is not None:
                discontinuity |= np.abs(np.diff(timestamps[indices]) - 1.0 / fps) > 0.1 / fps
        split = np.flatnonzero(discontinuity) + 1
        for run in np.split(indices, split):
            if not len(run):
                continue
            taper = np.minimum(run - run[0], run[-1] - run) / float(fps * TAPER_SECONDS)
            taper = np.clip(taper, 0.0, 1.0)
            # A phase marked as defending implies the opposing side is the
            # attacking/possessing side. Only explicit defense codes are flipped.
            possessor = side if category == "attacking" else ("A" if side == "H" else "H")
            evidence[run, side_col[possessor]] = np.maximum(
                evidence[run, side_col[possessor]], taper * MAX_LOG_ODDS
            )

    # If both sides have an active tag, treat the frame as semantically
    # conflicted and leave the model's conditional odds alone.
    conflict = (evidence[:, 0] > 0) & (evidence[:, 1] > 0)
    delta = np.clip(evidence[:, 0] - evidence[:, 1], -MAX_LOG_ODDS, MAX_LOG_ODDS)
    delta[conflict] = 0.0
    active = delta != 0
    if not active.any():
        return p.copy()

    out = p.copy()
    live_mass = p[active, 0].astype(np.float64) + p[active, 1].astype(np.float64)
    # Symmetric odds tilt preserves exact zeros and original live mass.
    h = p[active, 0].astype(np.float64) * np.exp(delta[active] / 2.0)
    a = p[active, 1].astype(np.float64) * np.exp(-delta[active] / 2.0)
    denom = h + a
    out[active, 0] = np.divide(live_mass * h, denom, out=np.zeros_like(live_mass), where=denom > 0)
    out[active, 1] = np.divide(live_mass * a, denom, out=np.zeros_like(live_mass), where=denom > 0)
    # ``out`` retains the input dtype, so D is copied bit-for-bit even for
    # float32 input arrays.
    return out
