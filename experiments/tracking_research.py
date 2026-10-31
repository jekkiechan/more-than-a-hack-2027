"""Label-free prototypes for challenge-like views and long-ball-gap features.

Research only: these functions do not change the adopted inference pipeline.
All gap and derivative operations reset at period/timestamp discontinuities.
"""
from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from experiments import refine_features, supervised, tracking_corruption as tc

STATE_NAMES = ['speed', 'abs_vx', 'abs_vy', 'longitudinal_edge', 'lateral_edge',
               'near5', 'near10', 'visible_players']
SEEDS = (2026, 2027, 2028)


def runs(mask, periods):
    """Half-open true intervals, split even when masks touch across blocks."""
    for a, b in supervised.blocks(periods):
        edges = np.diff(np.r_[False, mask[a:b], False].astype(np.int8))
        yield from zip(a+np.flatnonzero(edges == 1), a+np.flatnonzero(edges == -1))


def deduplicate(data, radius=1.):
    """Suppress close same-team observations, preferring longer-lived slots.

    This is conservative observation suppression, not verified player identity.
    Genuine teammates within radius can also be suppressed; measure that risk.
    No interpolation, team reassignment or cross-slot derivative is introduced.
    """
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('radius must be positive')
    out = tc.inputs_only(data)
    players, teams = out['players'], out['teams']
    visible = np.isfinite(players).all(axis=2)
    retained = visible.copy()
    for team in np.unique(teams):
        slots = np.flatnonzero(teams == team)
        slots = slots[np.argsort(-visible[:, slots].sum(axis=0), kind='stable')]
        for rank, slot in enumerate(slots):
            if rank:
                previous = slots[:rank]
                near = np.linalg.norm(players[:, previous]-players[:, slot, None], axis=2) <= radius
                retained[:, slot] &= ~(near & retained[:, previous]).any(axis=1)
    players[~retained] = np.nan
    out['player_observed'] = retained
    return out


def gap_features(data):
    """Offline endpoint context, only inside actual ball observation gaps.

    An unavailable endpoint remains NaN. Future endpoints are permitted for
    offline labeling; they never cross halves, time gaps or missing blocks.
    """
    data = tc.inputs_only(data)
    ball, players, teams = data['ball'], data['players'], data['teams']
    fps, period = float(data['fps']), supervised.model_periods(data)
    n = len(ball)
    names = ['gap_present', 'gap_duration_s', 'gap_elapsed_s', 'gap_remaining_s',
             'gap_fraction', 'gap_endpoint_speed', 'gap_start_x', 'gap_start_y',
             'gap_end_x', 'gap_end_y']
    names += [f'{team}_gap_{edge}_distance' for team in ('H', 'A') for edge in ('start', 'end')]
    x = np.full((n, len(names)), np.nan, dtype=np.float32)
    x[:, :5] = 0
    observed = np.isfinite(ball).all(axis=1)
    for a, b in runs(~observed, period):
        duration = (b-a)/fps
        x[a:b, 0] = 1
        x[a:b, 1] = duration
        x[a:b, 2] = (np.arange(b-a)+1)/fps
        x[a:b, 3] = np.arange(b-a, 0, -1)/fps
        x[a:b, 4] = (np.arange(b-a)+.5)/(b-a)
        left = a-1 if a > 0 and period[a-1] == period[a] else None
        right = b if b < n and period[b] == period[a] else None
        for index, endpoint in enumerate((left, right)):
            if endpoint is None or not observed[endpoint]:
                continue
            x[a:b, 6+2*index:8+2*index] = ball[endpoint]
            for team in (0, 1):
                distances = np.linalg.norm(players[endpoint, teams == team]-ball[endpoint], axis=1)
                valid = distances[np.isfinite(distances)]
                if len(valid):
                    x[a:b, 10+2*team+index] = valid.min()
        if left is not None and right is not None:
            x[a:b, 5] = np.linalg.norm(ball[right]-ball[left])/((right-left)/fps)
    return x, names


def features(data, gap=True, dedup=True):
    inputs = deduplicate(data) if dedup else tc.inputs_only(data)
    base, names = refine_features.features(inputs)
    if gap:
        extra, extra_names = gap_features(inputs)
        return np.column_stack((base, extra)), list(names)+extra_names
    return base, names


def states(data, rate=5):
    """Observable state at a grid point, using only its current/past inputs."""
    data = tc.inputs_only(data)
    fps = float(data['fps'])
    stride = max(1, round(fps/rate))
    period = supervised.model_periods(data)
    rows = np.concatenate([np.arange(a, b, stride) for a, b in supervised.blocks(period)])
    ball, players = data['ball'][rows], data['players'][rows]
    velocity = np.full_like(ball, np.nan)
    for a, b in supervised.blocks(period[rows]):
        if b-a > 1:
            velocity[a+1:b] = np.diff(ball[a:b], axis=0)/(stride/fps)
    velocity[np.linalg.norm(velocity, axis=1) > 65] = np.nan
    pitch = data.get('pitch', np.array([105., 68.]))
    distance = np.linalg.norm(players-ball[:, None], axis=2)
    x = np.column_stack((np.linalg.norm(velocity, axis=1), np.abs(velocity),
                         np.minimum(ball[:, 0], pitch[0]-ball[:, 0]),
                         np.minimum(ball[:, 1], pitch[1]-ball[:, 1]),
                         (distance < 5).sum(axis=1), (distance < 10).sum(axis=1),
                         np.isfinite(players).all(axis=2).sum(axis=1)))
    return rows, x


def profile(data):
    data = tc.inputs_only(data)
    active = data['period'] > 0
    period, fps = supervised.model_periods(data), float(data['fps'])
    visible = np.isfinite(data['players']).all(axis=2)
    clean = deduplicate(data)
    removed = visible.sum(axis=1)-clean['player_observed'].sum(axis=1)
    tracklets = [(b-a)/fps for slot in range(visible.shape[1])
                 for a, b in runs(visible[:, slot] & active, period)]
    ball_gaps = [(b-a)/fps for a, b in runs(~np.isfinite(data['ball']).all(axis=1) & active, period)]
    def quantiles(values):
        return np.quantile(values, [.1, .5, .9, .99]).tolist() if len(values) else []
    return dict(active_frames=int(active.sum()), slots=int(len(data['teams'])),
                ball_missing_fraction=float((~np.isfinite(data['ball'][active]).all(axis=1)).mean()),
                ball_gap_count=len(ball_gaps), ball_gap_seconds=quantiles(ball_gaps),
                tracklet_seconds=quantiles(tracklets),
                visible_players=quantiles(visible[active].sum(axis=1)),
                close_pair_frame_fraction=float((removed[active] > 0).mean()),
                suppressed_observation_fraction=float(removed[active].sum()/max(1, visible[active].sum())),
                mean_suppressed_per_frame=float(removed[active].mean()),
                close_pair_definition='Greedy same-team suppression within 1 m; not verified duplicate truth')


def fit_missingness(data):
    """Fit grid-step gap-onset hazard on unlabeled tracking, not possession.

    Complete gap durations and onset states are kept in the private cache.
    The next state and future endpoint never enter the onset predictor.
    """
    inputs = tc.inputs_only(data)
    rows, x = states(inputs)
    period = supervised.model_periods(inputs)[rows]
    seen = np.isfinite(inputs['ball'][rows]).all(axis=1)
    eligible = seen[:-1] & (period[:-1] == period[1:]) & (inputs['period'][rows[:-1]] > 0)
    y = ~seen[1:]
    if len(np.unique(y[eligible])) < 2:
        raise ValueError('Need observed continuations and gap onsets')
    model = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                          StandardScaler(), LogisticRegression(C=.2, max_iter=500))
    model.fit(x[:-1][eligible], y[eligible])
    duration, onset = [], []
    for a, b in runs(~seen, period):
        if a > 0 and b < len(rows) and period[a-1] == period[a] == period[b] and seen[a-1] and seen[b]:
            if inputs['period'][rows[a]] > 0:
                duration.append((rows[b]-rows[a])/float(inputs['fps']))
                onset.append(x[a-1])
    if not duration:
        raise ValueError('No complete internal gaps')
    return dict(version=1, estimator=model, duration_seconds=np.array(duration),
                onset_states=np.array(onset), onset_probability=float(y[eligible].mean()),
                profile=profile(inputs), state_names=STATE_NAMES, rate_hz=5,
                scope='Unlabeled tracking only; duration is sampled after onset, never an onset input')


def challenge_view(data, calibration, seed, players=True):
    """State-conditioned ball gaps plus separately controlled slot defects.

    Targets are not accepted by any feature/mask decision. Natural missingness
    is retained. This remains a simulation, not real broadcast confirmation.
    """
    out = tc.inputs_only(data)
    rng = np.random.default_rng(seed)
    rows, x = states(out)
    period = supervised.model_periods(out)
    seen = np.isfinite(out['ball']).all(axis=1)
    p = calibration['estimator'].predict_proba(x)[:, 1]
    eligible = seen[rows] & (out['period'][rows] > 0)
    if not eligible.any():
        return out
    # Account for pre-existing missingness: adding a reference-sized mask on
    # top of Metrica's natural gaps would overshoot the organizer defect rate.
    active = out['period'] > 0
    budget = max(0, round(calibration['profile']['ball_missing_fraction']*active.sum())
                 - int((~seen & active).sum()))
    reference = calibration['estimator'][:-1].transform(calibration['onset_states'])
    transformed = calibration['estimator'][:-1].transform(x)
    # An exponential race samples onset locations in proportion to their
    # observable-state hazard, without possession-dependent rejection rules.
    priority = -np.log(np.maximum(rng.random(len(rows)), 1e-12))/np.maximum(p, 1e-12)
    stride = max(1, round(float(out['fps'])/5))
    injected = np.zeros(len(seen), dtype=bool)
    for index in np.argsort(priority):
        row = rows[index]
        if budget <= 0:
            break
        if not eligible[index] or injected[max(0, row-stride):row+1].any():
            continue
        distance = np.square(reference-transformed[index]).mean(axis=1)
        nearest = np.argsort(distance)[:min(20, len(distance))]
        seconds = rng.choice(calibration['duration_seconds'][nearest])
        stop = min(len(seen), row+max(1, round(seconds*float(out['fps']))))
        # Never cross a period/time discontinuity; do not fill natural gaps.
        end = row+int(np.argmax(period[row:stop] != period[row])) if (period[row:stop] != period[row]).any() else stop
        if injected[row:min(len(seen), end+stride)].any():
            continue
        added = np.flatnonzero(seen[row:end])
        if len(added) > budget:
            end = row+added[budget-1]+1
        count = int(seen[row:end].sum())
        out['ball'][row:end] = np.nan
        injected[row:end] = True
        budget -= count
    out['ball_observed'] = np.isfinite(out['ball']).all(axis=1)
    if players:
        out = fragment_and_duplicate(out, calibration['profile'], rng)
    return out


def fragment_and_duplicate(data, target, rng):
    """Slot fragmentation and close copies; locations are never fabricated far away.

    Whole team permutations preserve the physical player set. Copies use spare
    slots and cannot overwrite a genuine detection. No possession labels enter.
    """
    out = tc.inputs_only(data)
    original, teams = out['players'], np.asarray(out['teams'])
    n, slots = original.shape[:2]
    desired = max(slots+2, target['slots'])
    extras = np.arange(desired-slots) % 2
    out['teams'] = np.r_[teams, extras]
    result = np.full((n, desired, 2), np.nan)
    fps = float(out['fps'])
    periods = supervised.model_periods(out)
    median = max(.4, target['tracklet_seconds'][1])
    for a, b in supervised.blocks(periods):
        for team in (0, 1):
            source = np.flatnonzero(teams == team)
            destinations = np.flatnonzero(out['teams'] == team)
            start = a
            while start < b:
                stop = min(b, start+max(2, round(rng.exponential(median)*fps)))
                chosen = rng.choice(destinations, len(source), replace=False)
                result[start:stop, chosen] = original[start:stop, source]
                # A single missing frame prevents a derivative across new identities.
                if start > a:
                    result[start, chosen] = np.nan
                start = stop
        # Copies persist for short runs. Their duty fraction is empirical, but
        # their localization noise (.15 m) is an explicit unverified assumption.
        chunk = max(1, round(fps))
        for start in range(a, b, chunk):
            if rng.random() >= target['close_pair_frame_fraction']:
                continue
            stop = min(b, start+chunk)
            for team in (0, 1):
                observed = np.isfinite(result[start]).all(axis=1)
                free = np.flatnonzero((out['teams'] == team) & ~np.isfinite(result[start:stop]).any(axis=(0, 2)))
                sources = np.flatnonzero((out['teams'] == team) & observed)
                if len(free) and len(sources):
                    result[start:stop, free[0]] = result[start:stop, rng.choice(sources)]+rng.normal(0, .15, 2)
    out['players'] = result
    out['player_observed'] = np.isfinite(result).all(axis=2)
    return out
