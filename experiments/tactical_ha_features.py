"""Direction-aware tactical H/A features (tracking only; no labels, events or possession metadata).

The adopted GRU sees only ball-local geometry. These features add what it cannot see:
each team's attacking direction (from where its players stand over the half), ball position
and progress along that direction, and collective team shape/motion.
All columns are antisymmetric or exchanged under a team swap by construction of the
orientation: swapping `teams` negates the orientation and exchanges H_/A_ columns.
"""
from __future__ import annotations

import warnings

import numpy as np

from experiments import supervised, tracking_corruption

SAMPLE_HZ = 5.0


def _roll(values, seconds, fps, period):
    return supervised._rolling(values, 2 * int(round(seconds * fps)) + 1, period)


def _shifted_mean(values, lo_s, hi_s, fps, period):
    """Mean of values over the offset window [lo_s, hi_s] seconds relative to each frame."""
    out = np.full(len(values), np.nan)
    lo, hi = int(round(lo_s * fps)), int(round(hi_s * fps))
    for start, stop in supervised.blocks(period):
        v = values[start:stop]
        ok = np.isfinite(v)
        cs = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
        cn = np.concatenate(([0], np.cumsum(ok)))
        n = stop - start
        idx = np.arange(n)
        a = np.clip(idx + lo, 0, n)
        b = np.clip(idx + hi + 1, 0, n)
        total, count = cs[b] - cs[a], cn[b] - cn[a]
        out[start:stop] = np.divide(total, count, out=np.full(n, np.nan), where=count > 0)
    return out


def orientation(players, teams, period):
    """+1 where team 0 (H) attacks +x in that period block, else -1. Tracking only."""
    sign = np.ones(len(players))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for start, stop in supervised.blocks(period):
            xs = players[start:stop, :, 0]
            h = np.nanmean(xs[:, teams == 0])
            a = np.nanmean(xs[:, teams == 1])
            sign[start:stop] = 1.0 if not (np.isfinite(h) and np.isfinite(a)) or a >= h else -1.0
    return sign


def features(data):
    data = tracking_corruption.inputs_only(data)
    fps = float(data["fps"])
    period = supervised.model_periods(data)
    native_period = np.asarray(data["period"])
    players = np.asarray(data["players"], dtype=np.float64)
    teams = np.asarray(data["teams"])
    pitch = np.asarray(data.get("pitch", [105, 68]), dtype=float)
    ball = supervised._short_fill(np.asarray(data["ball"], dtype=np.float64), period, fps)
    # Orientation is constant over a native half; time breaks inside a half do not change ends.
    sign = orientation(players, teams, native_period)
    px = sign[:, None] * (players[:, :, 0] - pitch[0] / 2)
    py = players[:, :, 1] - pitch[1] / 2
    bx = sign * (ball[:, 0] - pitch[0] / 2)
    by = ball[:, 1] - pitch[1] / 2
    names, cols = [], []

    def add(name, values):
        names.append(name)
        cols.append(np.asarray(values, dtype=np.float32))

    def multi(name, values, windows=(1.0, 3.0)):
        add(name, values)
        for w in windows:
            add(f"{name}_mean{w:g}s", _roll(values, w, fps, period))

    stats = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for team, prefix, direction in ((0, "H", 1.0), (1, "A", -1.0)):
            x, y = px[:, teams == team], py[:, teams == team]
            seen = np.isfinite(x) & np.isfinite(y)
            n = seen.sum(axis=1)
            cx, cy = np.nanmean(x, axis=1), np.nanmean(y, axis=1)
            sx, sy = np.nanstd(x, axis=1), np.nanstd(y, axis=1)
            # Along the team's own attacking direction: positive = further up the pitch.
            own = direction * x
            ordered = np.sort(np.where(seen, own, np.inf), axis=1)
            deep2 = np.where(n >= 3, ordered[:, 1], np.nan)       # deepest outfield line (skips keeper)
            ordered_hi = np.sort(np.where(seen, own, -np.inf), axis=1)
            high1 = np.where(n >= 3, ordered_hi[:, -1], np.nan)   # most advanced player
            ball_own = direction * bx
            ahead = np.divide((seen & (own > ball_own[:, None])).sum(axis=1), n,
                              out=np.full(len(n), np.nan), where=n > 0)
            ahead[~np.isfinite(bx)] = np.nan
            dist = np.sqrt((x - bx[:, None]) ** 2 + (y - by[:, None]) ** 2)
            stats[prefix] = dict(n=n, cx=cx, cy=cy, sx=sx, sy=sy, deep2=deep2, high1=high1,
                                 ahead=ahead, ball_own=ball_own, dist_mean=np.nanmean(dist, axis=1),
                                 near10=np.where(np.isfinite(bx), (dist < 10).sum(axis=1), np.nan),
                                 near20=np.where(np.isfinite(bx), (dist < 20).sum(axis=1), np.nan),
                                 direction=direction)

    def velocity(values, smooth_s):
        v = np.full(len(values), np.nan)
        for start, stop in supervised.blocks(period):
            if stop - start > 1:
                v[start:stop] = np.gradient(values[start:stop]) * fps
        v[np.abs(v) > 45] = np.nan
        return _roll(v, smooth_s, fps, period)

    add("ball_x", bx)
    add("ball_abs_y", np.abs(by))
    bvx = velocity(bx, 0.2)
    multi("ball_vx", bvx)
    for horizon in (2.0, 5.0):
        add(f"ball_progress_{horizon:g}s", _shifted_mean(bx, 0.0, horizon, fps, period)
            - _shifted_mean(bx, -horizon, 0.0, fps, period))
    for prefix in ("H", "A"):
        s = stats[prefix]
        d = s["direction"]
        add(f"{prefix}_n_observed", s["n"])
        multi(f"{prefix}_ball_minus_centroid", s["ball_own"] - d * s["cx"])
        multi(f"{prefix}_ball_minus_deep_line", s["ball_own"] - s["deep2"])
        multi(f"{prefix}_ball_minus_front", s["ball_own"] - s["high1"])
        multi(f"{prefix}_fraction_ahead_of_ball", s["ahead"])
        multi(f"{prefix}_centroid_own_x", d * s["cx"])
        multi(f"{prefix}_centroid_forward_speed", d * velocity(s["cx"], 0.5))
        multi(f"{prefix}_deep_line_forward_speed", velocity(s["deep2"], 0.5))
        multi(f"{prefix}_spread_x", s["sx"])
        multi(f"{prefix}_spread_y", s["sy"])
        multi(f"{prefix}_mean_distance_to_ball", s["dist_mean"])
        multi(f"{prefix}_players_within10m", s["near10"])
        multi(f"{prefix}_players_within20m", s["near20"])
        for horizon in (2.0, 5.0):
            c = d * s["cx"]
            add(f"{prefix}_centroid_progress_{horizon:g}s", _shifted_mean(c, 0.0, horizon, fps, period)
                - _shifted_mean(c, -horizon, 0.0, fps, period))
    h, a = stats["H"], stats["A"]
    multi("centroid_gap_x", a["cx"] - h["cx"])
    multi("spread_x_H_minus_A", h["sx"] - a["sx"])
    multi("spread_y_H_minus_A", h["sy"] - a["sy"])
    multi("both_centroids_vx", velocity((h["cx"] + a["cx"]) / 2, 0.5))
    matrix = np.column_stack(cols).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names


def swap_inputs(data):
    out = dict(data)
    out["teams"] = 1 - np.asarray(data["teams"])
    return out


def sample_indices(n, fps, period):
    """5 Hz native sample indices, phase 0 inside every model block."""
    step = max(1, int(round(fps / SAMPLE_HZ)))
    return np.concatenate([np.arange(a, b, step) for a, b in supervised.blocks(period)])
