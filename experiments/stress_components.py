"""Component-gated copy of the frozen composite corruption.

``corrupt(data, calibration, seed, keep)`` draws every random event exactly as
``tracking_corruption.corrupt`` does, on a shadow copy that always receives the
full composite. Only the components named in ``keep`` are written to the
returned view, so every view of one seed shares identical event realizations.
With all components kept, the output equals the frozen composite exactly.
"""
from __future__ import annotations

import numpy as np
from experiments import supervised, tracking_corruption as tc

COMPONENTS = ('coordinates', 'ball_gaps', 'regional', 'slot_gaps', 'id_swaps', 'sampling')


def corrupt(data, calibration, seed, keep):
    keep = set(keep)
    if keep - set(COMPONENTS):
        raise ValueError(f'Unknown components: {sorted(keep-set(COMPONENTS))}')
    out = tc.inputs_only(data)
    shadow = tc.inputs_only(data)
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
        sp, sb = shadow['players'][a:b], shadow['ball'][a:b]
        op, ob = out['players'][a:b], out['ball'][a:b]
        coordinates = 'coordinates' in keep
        shift = tc._smooth(n, fps, rng, 20., .35, 2)
        scale = 1+tc._smooth(n, fps, rng, 20., .003, 1)
        targets = ((sp, sb), (op, ob)) if coordinates else ((sp, sb),)
        for players, ball in targets:
            players[:] = (players-pitch/2)*scale[:, None]+pitch/2+shift[:, None]
            ball[:] = (ball-pitch/2)*scale+pitch/2+shift
        ball_noise = tc._smooth(n, fps, rng, 1., .2, 2)
        player_noise = tc._smooth(n, fps, rng, 3., .12, sp.shape[1]*2).reshape(sp.shape)
        for players, ball in targets:
            ball += ball_noise
            players += player_noise
        for start in tc._events(0, n, fps, rng, (60., 120.)):
            stop = min(n, start+round(rng.uniform(.3, 1.5)*fps))
            delta = rng.normal(0, 1., 2)
            for players, ball in targets:
                players[start:stop] += delta
                ball[start:stop] += delta
        at = 0
        while at < n:
            bin_index = rng.choice(len(gap_probability), p=gap_probability)
            duration = rng.uniform(gap_edges[bin_index], gap_edges[bin_index+1])
            visible_s = duration*(1-missing)/missing
            at += max(1, round(rng.exponential(visible_s)*fps))
            stop = min(n, at+max(1, round(duration*fps)))
            sb[at:stop] = np.nan
            if 'ball_gaps' in keep:
                ob[at:stop] = np.nan
            at = stop
        for start in tc._events(0, n, fps, rng, (35., 70.)):
            stop = min(n, start+round(rng.uniform(1., 5.)*fps))
            seen = np.flatnonzero(np.isfinite(sp[start]).all(axis=1))
            if len(seen):
                center = sp[start, rng.choice(seen)].copy()
                region = np.linalg.norm(sp[start:stop]-center, axis=2) < rng.uniform(10., 22.)
                sp[start:stop][region] = np.nan
                if 'regional' in keep:
                    op[start:stop][region] = np.nan
        for slot in range(sp.shape[1]):
            if not np.isfinite(sp[:, slot]).any():
                continue
            for start in tc._events(0, n, fps, rng, (40., 100.)):
                stop = min(n, start+round(rng.uniform(.2, 3.)*fps))
                sp[start:stop, slot] = np.nan
                if 'slot_gaps' in keep:
                    op[start:stop, slot] = np.nan
        for start in tc._events(0, n, fps, rng, (60., 120.)):
            team = rng.integers(2)
            slots = np.flatnonzero((teams == team) & np.isfinite(sp[start]).all(axis=1))
            if len(slots) >= 2:
                pair = rng.choice(slots, 2, replace=False)
                stop = min(n, start+max(2, round(rng.uniform(.2, 1.)*fps)))
                sp[start:, pair] = sp[start:, pair[::-1]].copy()
                sp[start:stop, pair] = np.nan
                if 'id_swaps' in keep:
                    op[start:, pair] = op[start:, pair[::-1]].copy()
                    op[start:stop, pair] = np.nan
        for start in range(0, n, max(1, round(10*fps))):
            stop = min(n, start+round(10*fps))
            rate = rng.uniform(.02, .08)
            lost = rng.random(stop-start) < rate
            sp[start:stop][lost] = np.nan
            sb[start:stop][lost] = np.nan
            if 'sampling' in keep:
                op[start:stop][lost] = np.nan
                ob[start:stop][lost] = np.nan
    out['ball_observed'] = np.isfinite(out['ball']).all(axis=1)
    out['player_observed'] = np.isfinite(out['players']).all(axis=2)
    return out
