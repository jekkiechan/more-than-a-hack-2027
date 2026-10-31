"""Tracking-only gap/contact strata and official, globally matched spell errors.

Contacts are the existing possession_geometry reliable-contact proxy, uncapped
in age for diagnosis. They are not possession annotations. No scorer is changed.
"""
from __future__ import annotations

import numpy as np
from evaluation import possession_metrics as pm
from experiments import supervised, possession_geometry as geometry, learned_temporal as lt
from experiments.metrics import prepare_tracking_codes

GAP_EDGES = (0., 3., 10., 20., 40., np.inf)
DISTANCE_EDGES = (0., 1., 3., 5., 10., 20., np.inf)


def runs(mask, a, b):
    edges = np.r_[a, np.flatnonzero(mask[a+1:b] != mask[a:b-1])+a+1, b]
    return [(int(l), int(r)) for l, r in zip(edges[:-1], edges[1:]) if mask[l]]


def nearest_contacts(contact, period, fps):
    n = len(contact)
    before, after = np.full(n, -1, int), np.full(n, -1, int)
    for a, b in supervised.blocks(period):
        ids = np.flatnonzero(contact[a:b])+a
        if not len(ids):
            continue
        where = np.searchsorted(ids, np.arange(a, b), side='right')-1
        valid = where >= 0
        before[a:b][valid] = ids[where[valid]]
        where = np.searchsorted(ids, np.arange(a, b), side='left')
        valid = where < len(ids)
        after[a:b][valid] = ids[where[valid]]
    i = np.arange(n)
    distance = np.minimum(np.where(before >= 0, i-before, np.inf),
                          np.where(after >= 0, after-i, np.inf))/fps
    return before, after, distance


def covariates(inputs):
    fps = float(inputs['fps'])
    period = supervised.model_periods(inputs)
    n = len(period)
    observed = np.isfinite(inputs['ball']).all(axis=-1)
    if 'ball_observed' in inputs:
        observed &= inputs['ball_observed']
    gap = np.zeros(n)
    gaps = []
    for a, b in supervised.blocks(period):
        for l, r in runs(~observed, a, b):
            gap[l:r] = (r-l)/fps
            gaps.append(dict(seconds=(r-l)/fps, touches_block_edge=l == a or r == b))
    x, names = geometry.features(inputs)
    home = x[:, names.index('H_geometry_best_control')]
    away = x[:, names.index('A_geometry_best_control')]
    margin = x[:, names.index('home_geometry_control_margin')]
    reliable = observed & np.isfinite(margin)
    reliable &= np.maximum(home, away) >= geometry.SPEC['anchor_minimum_best_control']
    reliable &= np.abs(margin) >= geometry.SPEC['anchor_minimum_absolute_margin']
    side = np.where(reliable, np.sign(margin), 0)
    contact = np.zeros(n, bool)
    for a, b in supervised.blocks(period):
        edges = np.r_[a, np.flatnonzero(side[a+1:b] != side[a:b-1])+a+1, b]
        for l, r in zip(edges[:-1], edges[1:]):
            if side[l] and (r-l)/fps >= geometry.SPEC['anchor_minimum_stable_duration_s']-1e-9:
                contact[l:r] = True
    before, after, distance = nearest_contacts(contact, period, fps)
    # Actual input-window endpoints (including clipping at input block edges).
    indices = lt.sampled_indices(inputs)
    sampled_period = period[indices]
    bracket = {}
    for seconds in (5, 15):
        covered = np.zeros(n, bool)
        for a, b in supervised.blocks(sampled_period):
            for start in range(a, b, lt.CORE):
                stop = min(start+lt.CORE, b)
                left = indices[max(a, start-seconds*lt.RATE)]
                right = indices[min(b-1, start+lt.CORE+seconds*lt.RATE-1)]
                native_left = indices[start]
                native_right = indices[stop] if stop < b else (
                    np.flatnonzero(period == sampled_period[a])[-1]+1)
                sl = slice(native_left, native_right)
                covered[sl] = (before[sl] >= left) & (after[sl] >= 0) & (after[sl] <= right)
        bracket[str(seconds)] = covered
    return dict(gap=gap, distance=distance, contact=contact, before=before, after=after,
                bracket=bracket, gaps=gaps)


def ratio(a, b):
    return float(a/b) if b else None


def strata(gap, distance):
    yield 'all', np.ones(len(gap), bool)
    yield 'gap:no_gap', gap == 0
    for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:]):
        g = (gap > 0) & (gap >= lo) & (gap < hi)
        key = f'gap:{lo:g}-{hi:g}s'
        yield key, g
    for lo, hi in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
        yield f'contact:{lo:g}-{hi:g}s', (distance >= lo) & (distance < hi)
    yield 'contact:none_in_block', ~np.isfinite(distance)
    for lo, hi in zip(GAP_EDGES[:-1], GAP_EDGES[1:]):
        for dl, dh in zip(DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:]):
            yield f'joint:gap{lo:g}-{hi:g}s_contact{dl:g}-{dh:g}s', (
                (gap > 0) & (gap >= lo) & (gap < hi) & (distance >= dl) & (distance < dh))


def frame_counts(labels, prediction, mask, cov=None):
    live = mask & np.isin(labels, ('H', 'A'))
    dead = mask & (labels == 'D')
    correct = int((live & (labels == prediction)).sum())
    wrong = int((live & np.isin(prediction, ('H', 'A')) & (labels != prediction)).sum())
    false_d = int((live & (prediction == 'D')).sum())
    d = dict(graded_frames=int((live | dead).sum()), live_frames=int(live.sum()),
             correct_team=correct, wrong_team=wrong, false_dead=false_d,
             abstained_live=int((live & (prediction == 'X')).sum()),
             dead_frames=int(dead.sum()), missed_dead=int((dead & (prediction != 'D')).sum()),
             team_accuracy=ratio(correct, live.sum()), wrong_team_rate=ratio(wrong, live.sum()),
             false_dead_rate=ratio(false_d, live.sum()))
    if cov is not None:
        d['both_contacts_in_window'] = {k: int((live & v).sum()) for k,v in cov['bracket'].items()}
    return d


def analyze(inputs, labels, prediction, cov=None):
    cov = covariates(inputs) if cov is None else cov
    fps = float(inputs['fps'])
    frames = {k: frame_counts(labels, prediction, mask, cov)
              for k, mask in strata(cov['gap'], cov['distance'])}
    _, truth, guess, inserts = prepare_tracking_codes(labels, prediction, inputs['period'], fps, inputs['timestamps'])
    valid = pm.valid_runs(truth)
    filled = pm.fill_abstentions(guess, valid)
    gt, pred = pm.possession_spells(truth, valid), pm.possession_spells(filled, valid)
    overlaps = pm.spell_overlaps(gt, pred)
    matched = pm.matched_gt_flags(overlaps, len(gt), .5)
    degree = np.zeros(len(gt), int)
    for _, gi, _ in overlaps:
        degree[gi] += 1
    padded_gap = np.insert(cov['gap'], inserts, 0)
    padded_dist = np.insert(cov['distance'], inserts, np.inf)
    # The midpoint locates a spell once. Also report maximum intersected gap.
    mid = np.array([(a+b-1)//2 for a,b,_ in gt], int)
    durations = np.array([(b-a)/fps for a,b,_ in gt])
    sg = np.array([padded_gap[a:b].max() for a,b,_ in gt])
    spell = {}
    groups = list(strata(padded_gap[mid], padded_dist[mid]))
    groups += [(f'intersects_gap_at_least_{s}s', sg >= s) for s in (3,10,20)]
    for k, mask in groups:
        short = mask & (durations < 3)
        spell[k] = dict(reference_spells=int(mask.sum()), missed=int((mask & ~matched).sum()),
                        fragmented=int((mask & (degree > 1)).sum()),
                        short_reference=int(short.sum()), short_missed=int((short & ~matched).sum()),
                        short_recall=ratio((short & matched).sum(), short.sum()))
    # Direct H<->A transitions only, reset at every graded run, including X.
    reference, predicted = [], []
    run_id = np.full(len(truth), -1, int)
    for rid, (a,b) in enumerate(valid):
        run_id[a:b] = rid
        for values, dest in ((truth, reference), (filled, predicted)):
            for i in range(a+1, b):
                if values[i-1] in (pm.H,pm.A) and values[i] in (pm.H,pm.A) and values[i-1] != values[i]:
                    dest.append((i, int(values[i]), rid))
    candidates = sorted((abs(p-g), gi, pi, (p-g)/fps)
                        for gi,(g,team,rid) in enumerate(reference)
                        for pi,(p,pt,pr) in enumerate(predicted)
                        if team == pt and rid == pr and abs(p-g)/fps <= 5)
    used_gt, used_pred, timing = set(), set(), {}
    for _,gi,pi,error in candidates:
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi); used_pred.add(pi); timing[gi] = error
    ti = np.array([r[0] for r in reference], int)
    turnovers = {}
    for k, mask in strata(padded_gap[ti], padded_dist[ti]):
        ids = np.flatnonzero(mask)
        errors = [timing[i] for i in ids if i in timing]
        turnovers[k] = dict(reference=int(mask.sum()), matched_within_5s=len(errors),
            missed_within_5s=int(mask.sum())-len(errors),
            matched_within_1s=sum(abs(v) <= 1 for v in errors),
            signed_error_seconds_sum=float(sum(errors)),
            absolute_error_seconds_sum=float(sum(abs(v) for v in errors)),
            mean_signed_seconds=float(np.mean(errors)) if errors else None,
            median_absolute_seconds=float(np.median(np.abs(errors))) if errors else None)
    return dict(frames=frames, spells=spell, turnovers=turnovers,
                unmatched_predicted_turnovers=len(predicted)-len(used_pred),
                reference_spells_touching_run_edge=sum(a in {l for l,_ in valid} or b in {r for _,r in valid} for a,b,_ in gt),
                gap_intervals=len(cov['gaps']), contact_frames=int(cov['contact'].sum()))


def complementarity(labels, predictions):
    a, b = predictions
    live = np.isin(labels, ('H','A'))
    ca, cb = a == labels, b == labels
    return dict(graded=int(np.isin(labels, ('H','A','D')).sum()), live_frames=int(live.sum()),
                both_correct=int((live & ca & cb).sum()), only_2026_correct=int((live & ca & ~cb).sum()),
                only_2027_correct=int((live & ~ca & cb).sum()), both_wrong=int((live & ~ca & ~cb).sum()),
                decoded_disagreement=int((live & (a != b)).sum()))
