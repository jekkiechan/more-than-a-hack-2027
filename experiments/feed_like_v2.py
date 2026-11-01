"""Challenge-calibrated feed simulation (`feed_like_v2`).

The original `feed_like` view (stress_components with coordinates, ball_gaps, slot_gaps
and id_swaps) places ball gaps and short player-slot gaps independently of game state.
In the organizer tracking, ball loss is concentrated in dead play, live-play loss mostly
extends dead gaps across restarts, and player dropouts are rare, long, more frequent in
dead play and concentrated at or beyond the pitch boundary.

`calibrate` derives every rate and duration from organizer tracking inputs plus a
label-free prediction CSV (no organizer labels). `corrupt` applies them to a native
development or training view; it needs that view's H/A/D/X targets only to *place*
simulated defects by game state, exactly as the real feed's loss depends on state.
Targets never enter features or inference, and every returned array is tracking only.
Non-ball, non-dropout defects (camera drift, local noise, jumps, ID swaps) are drawn by
the frozen `stress_components.corrupt` with the same seed.
"""
from __future__ import annotations

import numpy as np

from experiments import stress_components as stc, supervised, tracking_corruption as tc

VERSION = 2
QUANTILES = np.linspace(0, 1, 101)
EDGE_BINS = (-np.inf, 0., 3., 10., np.inf)
PLAYER_GAP_CAP_S = 60.
DEAD_SHORT_S = 3.
# Fixed after two single-realization fidelity checks on Sportec/Metrica development views,
# comparing model-predicted-state profiles with the organizer profile (no labels on either
# side of that comparison, no model scores consulted). Development views place defects by
# true state while the profile conditions on predicted state, which inflates isolated live
# gaps; dead-play dropouts on clean full-pitch data start further from the boundary.
SCALES = {"dropout_scale": {"dead": 1.25, "live": 1.}, "isolated_scale": .5}


def runs(mask):
    b = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0]))
    return list(zip(b[::2].tolist(), b[1::2].tolist()))


def edge_distance(xy, pitch):
    """Signed distance to the nearest pitch boundary; negative outside the pitch."""
    return np.minimum.reduce([xy[..., 0], pitch[0]-xy[..., 0], xy[..., 1], pitch[1]-xy[..., 1]])


def _q(values, cap=None):
    values = np.asarray(values, dtype=float)
    if cap is not None:
        values = np.minimum(values, cap)
    return np.quantile(values, QUANTILES).tolist() if len(values) else [0.]*len(QUANTILES)


def _sample(rng, quantiles):
    return float(np.interp(rng.random(), QUANTILES, quantiles))


def _observed(inputs):
    ball = np.asarray(inputs["ball"], dtype=float)
    players = np.asarray(inputs["players"], dtype=float)
    ball_obs = np.isfinite(ball).all(axis=1)
    player_obs = np.isfinite(players).all(axis=2)
    if "ball_observed" in inputs:
        ball_obs &= np.asarray(inputs["ball_observed"], dtype=bool)
    if "player_observed" in inputs:
        player_obs &= np.asarray(inputs["player_observed"], dtype=bool)
    return ball, players, ball_obs, player_obs


def calibrate(inputs, prediction):
    """Rates/durations from real tracking and label-free predicted states (H/A/D)."""
    fps = float(inputs["fps"])
    ball, players, ball_obs, player_obs = _observed(inputs)
    pitch = np.asarray(inputs.get("pitch", [105., 68.]), dtype=float)
    inplay = np.asarray(inputs["period"]) > 0
    dead = (np.asarray(prediction).astype(str) == "D") & inplay
    live = ~dead & inplay
    missing = ~ball_obs & inplay
    cal = {"version": VERSION, "fps": fps, **SCALES}
    # Dead runs: probability of any ball gap, onset after dead start, end before dead end.
    for name, keep in (("dead_short", lambda s: s < DEAD_SHORT_S), ("dead_long", lambda s: s >= DEAD_SHORT_S)):
        onset, end, n, hit, cover = [], [], 0, 0, []
        for a, b in runs(dead):
            if not keep((b-a)/fps):
                continue
            n += 1
            w = np.flatnonzero(missing[a:b])
            cover.append(len(w)/(b-a))
            if len(w):
                hit += 1
                onset.append(w[0]/fps)
                end.append((b-a-1-w[-1])/fps)
        cal[name] = dict(runs=n, p_any=hit/max(1, n), onset_q=_q(onset), end_q=_q(end),
                         mean_coverage=float(np.mean(cover)) if cover else 0.)
    # Live-play loss touching a dead run (gap extends across a restart/stoppage edge).
    before, after, isolated = [], [], []
    for a, b in runs(missing & live):
        touches_after = a > 0 and dead[a-1]
        touches_before = b < len(dead) and dead[b]
        if touches_before:
            before.append((b-a)/fps)
        elif touches_after:
            after.append((b-a)/fps)
        else:
            isolated.append((a, b))
    edges_start = sum(1 for a, _ in runs(dead) if a > 0 and live[a-1])
    edges_end = sum(1 for _, b in runs(dead) if b < len(dead) and live[b])
    cal["extend_before_dead"] = dict(p=len(before)/max(1, edges_start), duration_q=_q(before))
    cal["extend_after_dead"] = dict(p=len(after)/max(1, edges_end), duration_q=_q(after))
    live_seconds = live.sum()/fps
    iso_d = [(b-a)/fps for a, b in isolated]
    cal["isolated_live"] = dict(rate_per_s=len(isolated)/max(1., live_seconds), duration_q=_q(iso_d))
    # Player dropouts: hazard per observed player-second by state and boundary distance.
    edge = edge_distance(players, pitch)
    starts = np.zeros(player_obs.shape, dtype=bool)
    durations = {"dead": [], "live": []}
    for s in range(players.shape[1]):
        for a, b in runs(~player_obs[:, s] & inplay):
            if a == 0 or not player_obs[a-1, s] or b >= len(players):
                continue  # never-seen slots and censored runs do not define dropout hazard
            starts[a-1, s] = True
            durations["dead" if dead[a] else "live"].append((b-a)/fps)
    hazard = {}
    for state, mask in (("dead", dead), ("live", live)):
        # A dropout starting at frame a is attributed to the state of frame a; exposure is
        # the last observed frame before it (a-1), with that frame's boundary distance.
        following = np.r_[mask[1:], mask[-1:]]
        rows = []
        for lo, hi in zip(EDGE_BINS[:-1], EDGE_BINS[1:]):
            exposed = player_obs & following[:, None] & (edge >= lo) & (edge < hi)
            rows.append(float(starts[exposed].sum()/max(1., exposed.sum()/fps)))
        hazard[state] = rows
    cal["player_dropout"] = dict(edge_bins=list(EDGE_BINS[1:-1]), hazard_per_s=hazard,
                                 duration_q={k: _q(v, PLAYER_GAP_CAP_S) for k, v in durations.items()})
    return cal


def _ball_gaps(ball, state, fps, rng, cal):
    dead = state == "D"
    live = ~dead
    n = len(ball)
    blank = np.zeros(n, dtype=bool)
    for a, b in runs(dead):
        c = cal["dead_short" if (b-a)/fps < DEAD_SHORT_S else "dead_long"]
        if rng.random() < c["p_any"]:
            u = a + int(round(_sample(rng, c["onset_q"])*fps))
            v = b - int(round(_sample(rng, c["end_q"])*fps))
            if v > u:
                blank[u:v] = True
        if a > 0 and live[a-1] and rng.random() < cal["extend_before_dead"]["p"]:
            d = max(1, int(round(_sample(rng, cal["extend_before_dead"]["duration_q"])*fps)))
            blank[max(0, a-d):a] = True
        if b < n and live[b] and rng.random() < cal["extend_after_dead"]["p"]:
            d = max(1, int(round(_sample(rng, cal["extend_after_dead"]["duration_q"])*fps)))
            blank[b:min(n, b+d)] = True
    iso = cal["isolated_live"]
    native_iso = sum(1 for a, b in runs(~np.isfinite(ball).all(axis=1) & live)
                     if not ((a > 0 and dead[a-1]) or (b < n and dead[b])))/max(1., live.sum()/fps)
    p_start = max(0., iso["rate_per_s"]*cal.get("isolated_scale", 1.)-native_iso)/fps
    for a, b in runs(live):
        starts = a + np.flatnonzero(rng.random(b-a) < p_start)
        for s in starts:
            d = max(1, int(round(_sample(rng, iso["duration_q"])*fps)))
            blank[s:min(b, s+d)] = True
    ball[blank] = np.nan
    return ball


def native_dropout_rate(players, state, fps):
    """Native dropout starts per visible player-second by state (for top-up only)."""
    visible = np.isfinite(players).all(axis=2)
    out = {}
    for key, mask in (("dead", state == "D"), ("live", state != "D")):
        starts = (visible[:-1] & ~visible[1:])[mask[1:]].sum()
        exposure = visible[:-1][mask[1:]].sum()/fps
        out[key] = float(starts/max(1., exposure))
    return out


def target_dropout_rate(cal, players, state, pitch):
    """Mean calibrated hazard over this view's visible player-frames, by state."""
    c = cal["player_dropout"]
    edge = edge_distance(players, pitch)
    which = np.digitize(np.nan_to_num(edge, nan=np.inf), np.asarray(c["edge_bins"]))
    visible = np.isfinite(players).all(axis=2)
    out = {}
    for key, mask in (("dead", state == "D"), ("live", state != "D")):
        sel = visible & np.r_[mask[1:], mask[-1:]][:, None]
        out[key] = float(np.asarray(c["hazard_per_s"][key])[which[sel]].mean()) if sel.any() else 0.
    return out


def _player_dropouts(players, state, fps, rng, cal, pitch):
    c = cal["player_dropout"]
    bins = np.asarray(c["edge_bins"])
    edge = edge_distance(players, pitch)
    which = np.digitize(np.nan_to_num(edge, nan=np.inf), bins)
    scale = cal.get("dropout_scale", {"dead": 1., "live": 1.})
    native, target = native_dropout_rate(players, state, fps), target_dropout_rate(cal, players, state, pitch)
    # Top up to the calibrated rate: providers with heavier native loss receive fewer extra dropouts.
    keep = {k: scale[k]*max(0., 1.-native[k]/max(1e-12, target[k]*scale[k])) for k in ("dead", "live")}
    following = np.r_[state[1:], state[-1:]] == "D"  # a dropout at t blanks from t+1
    rate = np.where(following[:, None], keep["dead"]*np.asarray(c["hazard_per_s"]["dead"])[which],
                    keep["live"]*np.asarray(c["hazard_per_s"]["live"])[which])
    visible = np.isfinite(players).all(axis=2)
    candidate = visible & (rng.random(visible.shape) < rate/fps)
    n = len(players)
    for s in range(players.shape[1]):
        busy = -1
        for t in np.flatnonzero(candidate[:, s]):
            if t <= busy:
                continue
            key = "dead" if following[t] else "live"
            d = max(1, int(round(_sample(rng, c["duration_q"][key])*fps)))
            players[t+1:min(n, t+1+d), s] = np.nan
            busy = t + d
    return players


def _crossing_swaps(players, teams, fps, rng, interval=(60., 120.), radius=1., search_s=10.):
    """Same-team identity switches where two tracks pass within `radius`, without a gap.

    Uses the original feed_like swap frequency; real trackers switch identities at
    close crossings, and the organizer feed contains no player gaps shorter than 2 s.
    """
    n = len(players)
    for start in tc._events(0, n, fps, rng, interval):
        stop = min(n, start+int(search_s*fps))
        team = rng.integers(2)
        slots = np.flatnonzero(teams == team)
        if len(slots) < 2 or stop <= start:
            continue
        p = players[start:stop][:, slots]
        dist = np.linalg.norm(p[:, :, None]-p[:, None], axis=3)
        dist[:, np.arange(len(slots)), np.arange(len(slots))] = np.inf
        hit = np.argwhere(np.nan_to_num(dist, nan=np.inf) < radius)
        if not len(hit):
            continue
        t, i, j = hit[0]
        pair = slots[[i, j]]
        players[start+t:, pair] = players[start+t:, pair[::-1]].copy()
    return players


def corrupt(data, labels, calibration, seed):
    """One deterministic challenge-calibrated view; inputs and targets are not mutated."""
    native = tc.inputs_only(data)
    if "ball_observed" in native:
        native["ball"][~np.asarray(native["ball_observed"], dtype=bool)] = np.nan
    if "player_observed" in native:
        native["players"][~np.asarray(native["player_observed"], dtype=bool)] = np.nan
    # Frozen camera drift, local noise, jumps and ID swaps; no state-blind gaps.
    out = stc.corrupt(native, {"gap_edges_seconds": [.08, .2], "gap_bin_probability": [1.],
                               "ball_missing_fraction": .05}, seed, ("coordinates",))
    labels = np.asarray(labels).astype(str)
    if labels.shape != (len(out["ball"]),):
        raise ValueError("Expected targets aligned with tracking frames")
    rng = np.random.default_rng([seed, VERSION])
    fps = float(out["fps"])
    pitch = np.asarray(out.get("pitch", [105., 68.]), dtype=float)
    for a, b in supervised.blocks(supervised.model_periods(out)):
        if out["period"][a] <= 0:
            continue
        state = labels[a:b]
        out["ball"][a:b] = _ball_gaps(out["ball"][a:b], state, fps, rng, calibration)
        out["players"][a:b] = _crossing_swaps(out["players"][a:b], np.asarray(out["teams"]), fps, rng)
        out["players"][a:b] = _player_dropouts(out["players"][a:b], state, fps, rng, calibration, pitch)
    out["ball_observed"] = np.isfinite(out["ball"]).all(axis=1)
    out["player_observed"] = np.isfinite(out["players"]).all(axis=2)
    return out


def profile(inputs, prediction):
    """Label-free fidelity statistics, comparable between real and simulated feeds."""
    fps = float(inputs["fps"])
    ball, players, ball_obs, player_obs = _observed(inputs)
    pitch = np.asarray(inputs.get("pitch", [105., 68.]), dtype=float)
    inplay = np.asarray(inputs["period"]) > 0
    dead = (np.asarray(prediction).astype(str) == "D") & inplay
    live = ~dead & inplay
    missing = ~ball_obs & inplay
    minutes = inplay.sum()/fps/60
    gaps = runs(missing)
    dur = np.array([(b-a)/fps for a, b in gaps]) if gaps else np.zeros(1)
    lm = runs(missing & live)
    touching = [(a > 0 and dead[a-1]) or (b < len(dead) and dead[b]) for a, b in lm]
    nv = player_obs.sum(axis=1)
    ref = np.median(nv[live]) if live.any() else 0
    starts = {"dead": 0, "live": 0}
    pgaps, outside = [], []
    edge = edge_distance(players, pitch)
    for s in range(players.shape[1]):
        for a, b in runs(~player_obs[:, s] & inplay):
            if a == 0 or not player_obs[a-1, s] or b >= len(players):
                continue
            starts["dead" if dead[a] else "live"] += 1
            pgaps.append((b-a)/fps)
            outside.append(edge[a-1, s] < 0)
    pgaps = np.array(pgaps) if pgaps else np.zeros(1)
    return dict(
        ball_missing=float(missing[inplay].mean()),
        ball_missing_given_pred_dead=float(missing[dead].mean()) if dead.any() else None,
        ball_missing_given_pred_live=float(missing[live].mean()) if live.any() else None,
        ball_gaps_per_min=len(gaps)/minutes,
        ball_gap_time_share_ge15s=float(dur[dur >= 15].sum()/max(dur.sum(), 1e-9)),
        live_missing_runs_touching_dead=float(np.mean(touching)) if touching else None,
        isolated_live_gaps_per_live_min=float((len(lm)-sum(touching))/max(1e-9, live.sum()/fps/60)),
        visible_players_dead_minus_live=float(nv[dead].mean()-nv[live].mean()) if dead.any() and live.any() else None,
        frames_dead_below_live_median_by_3plus=float((nv[dead] <= ref-3).mean()) if dead.any() else None,
        frames_live_below_live_median_by_3plus=float((nv[live] <= ref-3).mean()) if live.any() else None,
        player_dropouts_per_min_dead=starts["dead"]/max(1e-9, dead.sum()/fps/60),
        player_dropouts_per_min_live=starts["live"]/max(1e-9, live.sum()/fps/60),
        player_gap_median_s=float(np.median(pgaps)),
        player_gap_share_lt1s=float((pgaps < 1).mean()),
        dropout_start_outside_pitch=float(np.mean(outside)) if outside else None,
    )
