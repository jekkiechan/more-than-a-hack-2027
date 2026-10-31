"""Native-observation control geometry and reliable team-contact anchors.

These tracking-only proxies are not ownership labels. No coordinates are filled.
Geometry and anchors are separate feature groups for the full H/A/D ablation.
"""
from __future__ import annotations

import numpy as np

from experiments import supervised

TEAM_SUFFIXES = (
    'best_control', 'second_control', 'control_separation', 'control_candidate_fraction',
    'approach_time_s', 'approach_miss_distance_m', 'approach_improvement_m',
    'approach_candidate_fraction',
)
GEOMETRY_NAMES = tuple(f'{team}_geometry_{name}' for team in ('H', 'A') for name in TEAM_SUFFIXES) + (
    'home_geometry_control_margin', 'geometry_control_strength', 'geometry_control_ambiguity',
    'geometry_collective_acceleration_median', 'geometry_collective_braking_fraction',
    'geometry_collective_accelerating_fraction', 'geometry_acceleration_observation_fraction',
)
ANCHOR_NAMES = tuple(name for side in ('before', 'after') for name in (
    f'H_reliable_contact_{side}', f'A_reliable_contact_{side}',
    f'home_reliable_margin_{side}', f'reliable_contact_age_{side}_s',
    f'reliable_contact_valid_{side}',
)) + ('reliable_contact_same_team', 'home_reliable_margin_after_minus_before',
      'reliable_contact_span_s', 'reliable_contact_bracket_valid')

SPEC = {
    'native_observations_only': True, 'ball_speed_limit_mps': 65.,
    'player_speed_limit_mps': 13., 'player_acceleration_limit_mps2': 8.,
    'control_distance_scale_m': 1.8, 'control_relative_speed_scale_mps': 5.,
    'minimum_approach_relative_speed_mps': .5, 'approach_horizon_s': 2.,
    'signed_acceleration_threshold_mps2': .5, 'control_margin_epsilon': .001,
    'anchor_minimum_best_control': .2, 'anchor_minimum_absolute_margin': .2,
    'anchor_minimum_stable_duration_s': .2, 'anchor_maximum_age_s': 10.,
    'anchor_direction': 'strictly before and strictly after target within same input block',
    'derivatives': 'native timestamps, centered interior and one-sided block endpoints; unknown coordinates stay NaN',
    'geometry_names': GEOMETRY_NAMES, 'anchor_names': ANCHOR_NAMES,
}


def _fraction(numerator, denominator):
    return np.divide(numerator, denominator, out=np.full(len(denominator), np.nan), where=denominator > 0)


def _gradient(values, times):
    if len(values) < 2:
        return np.full_like(values, np.nan, dtype=float)
    return np.gradient(values, times, axis=0)


def _speed_limited_velocity(positions, times, limit):
    velocity = _gradient(positions, times)
    valid = np.isfinite(positions).all(axis=-1) & np.isfinite(velocity).all(axis=-1)
    valid &= np.linalg.norm(velocity, axis=-1) <= limit
    return np.where(valid[..., None], velocity, np.nan)


def _observed_slice(data, key, mask_key, start, stop):
    values = np.array(data[key][start:stop], dtype=float, copy=True)
    if mask_key in data:
        values[~np.asarray(data[mask_key][start:stop], dtype=bool)] = np.nan
    return values


def _chunk_geometry(data, start, stop, times, teams):
    ball = _observed_slice(data, 'ball', 'ball_observed', start, stop)
    players = _observed_slice(data, 'players', 'player_observed', start, stop)
    bv = _speed_limited_velocity(ball, times, SPEC['ball_speed_limit_mps'])
    pv = _speed_limited_velocity(players, times, SPEC['player_speed_limit_mps'])
    relative = bv[:, None, :] - pv
    offset = ball[:, None, :] - players
    distance = np.linalg.norm(offset, axis=-1)
    relative_speed = np.linalg.norm(relative, axis=-1)
    visible = np.isfinite(players).all(axis=-1)
    control_valid = np.isfinite(distance) & np.isfinite(relative_speed)
    control = np.exp(-.5*(distance/SPEC['control_distance_scale_m'])**2)
    control *= np.exp(-.5*(relative_speed/SPEC['control_relative_speed_scale_mps'])**2)
    control[~control_valid] = np.nan

    # Stationary relative motion is a valid control candidate, but cannot
    # determine a projected closest-approach time. It remains missing here.
    moving = control_valid & (relative_speed > SPEC['minimum_approach_relative_speed_mps'])
    tau = np.divide(-np.sum(offset*relative, axis=-1), relative_speed**2,
                    out=np.full_like(distance, np.nan), where=moving)
    tau = np.clip(tau, 0, SPEC['approach_horizon_s'])
    miss = np.linalg.norm(offset+tau[..., None]*relative, axis=-1)
    output = []
    bests = []
    for team in (0, 1):
        selected = teams == team
        valid = control_valid[:, selected]
        observed_count = visible[:, selected].sum(axis=1)
        count = valid.sum(axis=1)
        ordered = np.sort(np.where(valid, control[:, selected], -np.inf), axis=1)
        best = np.where(count > 0, ordered[:, -1], np.nan)
        second = np.where(count > 1, ordered[:, -2], np.nan) if ordered.shape[1] > 1 else np.full(len(ball), np.nan)
        bests.append(best)
        # Resolve projection ties by minimum miss, then minimum time, then
        # minimum initial distance, using scalar reductions instead of slot IDs.
        tm = miss[:, selected]
        tt = tau[:, selected]
        td = distance[:, selected]
        best_miss = np.min(np.where(np.isfinite(tm), tm, np.inf), axis=1)
        tied_miss = np.isfinite(tm) & (tm == best_miss[:, None])
        best_tau = np.min(np.where(tied_miss, tt, np.inf), axis=1)
        tied_time = tied_miss & (tt == best_tau[:, None])
        best_initial = np.min(np.where(tied_time, td, np.inf), axis=1)
        supported = np.isfinite(best_miss) & np.isfinite(best_tau) & np.isfinite(best_initial)
        best_miss = np.where(supported, best_miss, np.nan)
        best_tau = np.where(supported, best_tau, np.nan)
        improvement = np.where(supported, best_initial-best_miss, np.nan)
        output.extend((best, second, best-second, _fraction(count, observed_count),
                       best_tau, best_miss, improvement,
                       _fraction(moving[:, selected].sum(axis=1), observed_count)))
    total = bests[0]+bests[1]
    denominator = total+SPEC['control_margin_epsilon']
    output.extend(((bests[0]-bests[1])/denominator, total,
                   2*np.minimum(bests[0], bests[1])/denominator))
    signed_acceleration = _gradient(np.linalg.norm(pv, axis=-1), times)
    signed_acceleration[~np.isfinite(pv).all(axis=-1)] = np.nan
    signed_acceleration[np.abs(signed_acceleration)>SPEC['player_acceleration_limit_mps2']] = np.nan
    valid_acceleration = np.isfinite(signed_acceleration)
    count = valid_acceleration.sum(axis=1)
    # Finite sorted row medians avoid all-NaN warnings on missing observations.
    ordered = np.sort(np.where(valid_acceleration, signed_acceleration, np.inf), axis=1)
    row = np.arange(len(ball))
    lo = np.maximum(count-1, 0)//2
    hi = count//2
    median = np.where(count>0, (ordered[row,lo]+ordered[row,hi])/2, np.nan)
    threshold = SPEC['signed_acceleration_threshold_mps2']
    output.extend((median, _fraction((signed_acceleration < -threshold).sum(axis=1), count),
                   _fraction((signed_acceleration > threshold).sum(axis=1), count),
                   _fraction(count, visible.sum(axis=1))))
    matrix = np.column_stack(output).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _anchors(geometry, times, period, fps):
    index = {name:i for i,name in enumerate(GEOMETRY_NAMES)}
    home = geometry[:, index['H_geometry_best_control']]
    away = geometry[:, index['A_geometry_best_control']]
    margin = geometry[:, index['home_geometry_control_margin']]
    reliable = np.isfinite(margin) & (np.maximum(home,away)>=SPEC['anchor_minimum_best_control'])
    reliable &= np.abs(margin)>=SPEC['anchor_minimum_absolute_margin']
    side = np.where(reliable,np.sign(margin),0)
    minimum = SPEC['anchor_minimum_stable_duration_s']
    anchors = np.zeros(len(times),dtype=bool)
    for a,b in supervised.blocks(period):
        edges = np.r_[a,np.flatnonzero(side[a+1:b]!=side[a:b-1])+a+1,b]
        for left,right in zip(edges[:-1],edges[1:]):
            # Add one nominal frame interval so five consecutive 25Hz samples
            # represent .2s of observed evidence rather than four intervals.
            if side[left] and times[right-1]-times[left]+1/fps >= minimum-1e-9:
                anchors[left:right] = True
    columns=[]
    sides=[]
    for direction in ('before','after'):
        selected=np.full(len(times),-1,dtype=int)
        for a,b in supervised.blocks(period):
            last=-1
            sequence=range(a,b) if direction=='before' else range(b-1,a-1,-1)
            for t in sequence:
                # Assign before updating to exclude the target frame itself.
                if last>=0 and abs(times[t]-times[last])<=SPEC['anchor_maximum_age_s']+1e-9:
                    selected[t]=last
                if anchors[t]:
                    last=t
        valid=selected>=0
        safe=np.maximum(selected,0)
        age=np.where(valid,np.abs(times-times[safe]),np.nan)
        stored_margin=np.where(valid,margin[safe],np.nan)
        columns.extend((np.where(valid,home[safe],np.nan),np.where(valid,away[safe],np.nan),
                        stored_margin,age,valid.astype(float)))
        sides.append((valid,stored_margin,age))
    before,after=sides
    both=before[0]&after[0]
    columns.extend((np.where(both,np.sign(before[1])==np.sign(after[1]),np.nan),
                    after[1]-before[1],before[2]+after[2],both.astype(float)))
    return np.column_stack(columns).astype(np.float32)


def features(inputs, include_anchors=False):
    """Return native-rate features, optionally followed by reliable anchors.

    Only positions, team IDs, observation masks, periods, timestamps and fps are
    read. Derivative halos make chunking equivalent to within-block processing;
    the two-frame halo covers the second derivative used for acceleration.
    """
    ball=np.asarray(inputs['ball'])
    players=np.asarray(inputs['players'])
    teams=np.asarray(inputs['teams'])
    n=len(ball)
    fps=float(inputs['fps'])
    if n==0 or ball.shape!=(n,2) or players.ndim!=3 or players.shape[0]!=n or players.shape[2]!=2:
        raise ValueError('Expected nonempty XY ball and player arrays')
    if teams.shape!=(players.shape[1],) or not np.isin(teams,(0,1)).all() or not all(np.any(teams==t) for t in (0,1)):
        raise ValueError('Expected player slots for both teams')
    if not np.isfinite(fps) or fps<=0 or np.asarray(inputs['period']).shape!=(n,):
        raise ValueError('Invalid frame rate or periods')
    for mask,shape in (('ball_observed',(n,)),('player_observed',players.shape[:2])):
        if mask in inputs and np.asarray(inputs[mask]).shape!=shape:
            raise ValueError(f'Invalid {mask} shape')
    times=np.asarray(inputs.get('timestamps',np.arange(n)/fps),dtype=float)
    if times.shape!=(n,) or not np.isfinite(times).all():
        raise ValueError('Expected finite native timestamps')
    period=supervised.model_periods(inputs)
    geometry=np.full((n,len(GEOMETRY_NAMES)),np.nan,dtype=np.float32)
    for a,b in supervised.blocks(period):
        for start in range(a,b,8192):
            stop=min(start+8192,b)
            left,right=max(a,start-2),min(b,stop+2)
            chunk=_chunk_geometry(inputs,left,right,times[left:right],teams)
            geometry[start:stop]=chunk[start-left:stop-left]
    if not include_anchors:
        return geometry,list(GEOMETRY_NAMES)
    return np.column_stack((geometry,_anchors(geometry,times,period,fps))),list(GEOMETRY_NAMES+ANCHOR_NAMES)
