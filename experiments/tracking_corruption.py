"""Label-free, block-local corruption of tracking before feature extraction.

Gap durations can be calibrated from unlabeled inputs. Coordinate error ranges
are explicit assumptions, not measurements of tracking accuracy. No target,
owner, event or ball-status field is accepted or returned.
"""
from __future__ import annotations

import numpy as np
from experiments import robust, supervised

GAP_EDGES = [.08, .2, .5, 1., 2., 5., 10., 30.]
RECIPE = {
    'version': 1,
    'clean_training_weight': .5,
    'ball_gap': 'alternating observed/gap runs; empirical duration histogram, capped at 30 s',
    'regional_dropout': {'interval_seconds': [35., 70.], 'duration_seconds': [1., 5.],
                         'radius_m': [10., 22.]},
    'player_dropout': {'interval_seconds': [40., 100.], 'duration_seconds': [.2, 3.]},
    'drift': {'knot_seconds': 20., 'translation_sd_m': .35, 'scale_sd': .003},
    'jumps': {'interval_seconds': [60., 120.], 'duration_seconds': [.3, 1.5], 'sd_m': 1.},
    'fragmentation': {'interval_seconds': [60., 120.], 'reacquisition_seconds': [.2, 1.]},
    'sampling': {'block_seconds': 10., 'missing_probability': [.02, .08]},
    'scope': 'All applied before features; timestamps, frame grid and labels unchanged; no cross-block interpolation',
}


def inputs_only(data):
    out = {k: np.array(v, copy=True) for k, v in robust.tracking_inputs(data).items()}
    for key, mask in (('ball', 'ball_observed'), ('players', 'player_observed')):
        if key not in out:
            continue
        out[key] = np.asarray(out[key], dtype=float)
        if mask in out:
            out[key][~out[mask].astype(bool)] = np.nan
    return out


def gap_lengths(observed, periods, fps):
    lengths = []
    for a, b in supervised.blocks(periods):
        missing = ~observed[a:b]
        edges = np.diff(np.r_[False, missing, False].astype(np.int8))
        lengths.extend((np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1))/fps)
    return np.asarray(lengths)


def profile(data):
    """Aggregate input statistics only; period-zero frames are excluded."""
    data = inputs_only(data)
    active = np.asarray(data['period']) > 0
    periods = supervised.model_periods(data)
    observed = np.isfinite(data['ball']).all(axis=1)
    # Do not join live halves across the halftime region.
    lengths = []
    for a, b in supervised.blocks(periods):
        if active[a]:
            lengths.extend(gap_lengths(observed[a:b], np.zeros(b-a), float(data['fps'])))
    lengths = np.clip(lengths, GAP_EDGES[0], GAP_EDGES[-1])
    counts, _ = np.histogram(lengths, bins=GAP_EDGES)
    if not counts.sum():
        counts = np.ones(len(GAP_EDGES)-1)
    visible = np.isfinite(data['players']).all(axis=2).sum(axis=1)[active]
    return dict(ball_missing_fraction=float((~observed[active]).mean()),
                gap_edges_seconds=GAP_EDGES, gap_bin_probability=(counts/counts.sum()).tolist(),
                gap_count=int(len(lengths)),
                gap_quantiles_seconds=np.quantile(lengths, [.1, .5, .9, .99]).tolist() if len(lengths) else [],
                visible_player_quantiles=np.quantile(visible, [.1, .25, .5, .75, .9]).tolist(),
                calibration='Unlabeled tracking only; missingness does not identify true occlusion causes',
                coordinate_errors='Assumed ranges; no coordinate ground truth available')


def _events(a, b, fps, rng, interval):
    start = a + round(rng.uniform(*interval)*fps)
    while start < b:
        yield start
        start += max(1, round(rng.uniform(*interval)*fps))


def _smooth(n, fps, rng, knot_s, sd, dimensions):
    knots = np.arange(0, n+round(knot_s*fps), max(1, round(knot_s*fps)))
    if len(knots) == 1:
        knots = np.r_[knots, n]
    values = rng.normal(0, sd, (len(knots), dimensions))
    sample = np.arange(n)
    left = np.minimum(np.searchsorted(knots, sample, side='right')-1, len(knots)-2)
    alpha = (sample-knots[left])/(knots[left+1]-knots[left])
    alpha = (1-np.cos(np.pi*alpha))*.5
    return values[left]*(1-alpha[:, None])+values[left+1]*alpha[:, None]


def corrupt(data, calibration, seed):
    """One deterministic view. Original arrays and targets are never mutated."""
    out = inputs_only(data)
    rng = np.random.default_rng(seed)
    fps = float(out['fps'])
    pitch = np.asarray(out.get('pitch', [105., 68.]))
    periods = supervised.model_periods(out)
    gap_edges = np.asarray(calibration['gap_edges_seconds'])
    gap_probability = calibration['gap_bin_probability']
    missing = float(np.clip(calibration['ball_missing_fraction'], .05, .65))
    teams = out['teams']
    for a, b in supervised.blocks(periods):
        if out['period'][a] <= 0:
            continue
        n = b-a
        players, ball = out['players'][a:b], out['ball'][a:b]
        # Shared camera-to-pitch error preserves relative ball/player geometry.
        shift = _smooth(n, fps, rng, 20., .35, 2)
        scale = 1+_smooth(n, fps, rng, 20., .003, 1)
        players[:] = (players-pitch/2)*scale[:, None]+pitch/2+shift[:, None]
        ball[:] = (ball-pitch/2)*scale+pitch/2+shift
        # Small object-specific, temporally smooth localization errors.
        ball += _smooth(n, fps, rng, 1., .2, 2)
        players += _smooth(n, fps, rng, 3., .12, players.shape[1]*2).reshape(players.shape)
        for start in _events(0, n, fps, rng, (60., 120.)):
            stop = min(n, start+round(rng.uniform(.3, 1.5)*fps))
            delta = rng.normal(0, 1., 2)
            players[start:stop] += delta
            ball[start:stop] += delta
        # Renewal gaps include long live-play gaps without consulting labels.
        at = 0
        while at < n:
            bin_index = rng.choice(len(gap_probability), p=gap_probability)
            duration = rng.uniform(gap_edges[bin_index], gap_edges[bin_index+1])
            visible_s = duration*(1-missing)/missing
            at += max(1, round(rng.exponential(visible_s)*fps))
            stop = min(n, at+max(1, round(duration*fps)))
            ball[at:stop] = np.nan
            at = stop
        # Spatially correlated loss, with no arbitrary independent frame noise.
        for start in _events(0, n, fps, rng, (35., 70.)):
            stop = min(n, start+round(rng.uniform(1., 5.)*fps))
            seen = np.flatnonzero(np.isfinite(players[start]).all(axis=1))
            if len(seen):
                center = players[start, rng.choice(seen)].copy()
                region = np.linalg.norm(players[start:stop]-center, axis=2) < rng.uniform(10., 22.)
                players[start:stop][region] = np.nan
        for slot in range(players.shape[1]):
            if not np.isfinite(players[:, slot]).any():
                continue
            for start in _events(0, n, fps, rng, (40., 100.)):
                stop = min(n, start+round(rng.uniform(.2, 3.)*fps))
                players[start:stop, slot] = np.nan
        # Reassign a pair of same-team track slots after an observation break.
        # The player set is preserved; derivatives cannot bridge track identities.
        for start in _events(0, n, fps, rng, (60., 120.)):
            team = rng.integers(2)
            slots = np.flatnonzero((teams == team) & np.isfinite(players[start]).all(axis=1))
            if len(slots) >= 2:
                pair = rng.choice(slots, 2, replace=False)
                players[start:, pair] = players[start:, pair[::-1]].copy()
                stop = min(n, start+max(2, round(rng.uniform(.2, 1.)*fps)))
                players[start:stop, pair] = np.nan
        for start in range(0, n, max(1, round(10*fps))):
            stop = min(n, start+round(10*fps))
            rate = rng.uniform(.02, .08)
            lost = rng.random(stop-start) < rate
            players[start:stop][lost] = np.nan
            ball[start:stop][lost] = np.nan
    out['ball_observed'] = np.isfinite(out['ball']).all(axis=1)
    out['player_observed'] = np.isfinite(out['players']).all(axis=2)
    return out


def visibility(data):
    active = np.asarray(data['period']) > 0
    return dict(ball_missing_fraction=float((~np.isfinite(data['ball'][active]).all(axis=1)).mean()),
                mean_visible_players=float(np.isfinite(data['players'][active]).all(axis=2).sum(axis=1).mean()))
