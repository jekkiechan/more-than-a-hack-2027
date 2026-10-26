"""Tracking-only broadcast/tactical alignment and conservative ball-gap fusion.

No video, Smart Tagging, possession labels or manual anchors are consumed.
Both feeds must use the same pitch coordinates and home/away convention.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
from starter.baseline import team_of_each_track
from experiments import current_baseline, supervised, tracking_corruption as tc
from experiments.eval_set.protocol import sha256

SPEC = dict(anchor_seconds=30, coarse_seconds=1, candidates=200,
            max_median_player_error_m=2., match_radius_m=3., min_matches_each_team=3,
            cluster_radius_seconds=1., min_cluster_anchors=3,
            max_anchor_extrapolation_seconds=30., minimum_alignment_run_seconds=1.,
            scope='Unlabeled player geometry only; same pitch/team convention required')


def load_atd(meta, raw, pitch=(105.3, 68.)):
    teams = team_of_each_track(meta)
    players, balls = [], []
    with Path(raw).open() as handle:
        for i, line in enumerate(handle):
            fields = line.strip().split(':')
            if len(fields) != 3 or int(fields[0]) != i+1:
                raise ValueError('ATD rows must be contiguous one-based frame IDs')
            p = np.fromstring(fields[1].replace(';', ','), sep=',', dtype=np.float32)
            ball = np.fromstring(fields[2], sep=',', dtype=np.float32)
            if p.size != 3*len(teams) or ball.size != 3:
                raise ValueError('Unexpected ATD slot dimensions')
            players.append(p.reshape(-1, 3)[:, :2]*pitch)
            balls.append(ball[:2]*pitch)
    n = len(balls)
    return dict(players=np.array(players, dtype=np.float32), ball=np.array(balls, dtype=np.float32),
                teams=teams, fps=np.array(25.), period=np.ones(n, dtype=int),
                timestamps=np.arange(n)/25., pitch=np.array(pitch))


def descriptor(players, teams):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.concatenate([np.nanquantile(players[:, teams == team], [.25, .5, .75], axis=1)
                               .transpose(1, 0, 2).reshape(len(players), -1) for team in (0, 1)], axis=1)


def geometry(tactical_players, tactical_teams, broadcast_players, broadcast_teams):
    """One-to-one same-team assignment; no many-to-one duplicate rewards."""
    distances, counts = [], []
    for team in (0, 1):
        a = tactical_players[(tactical_teams == team) & np.isfinite(tactical_players).all(axis=1)]
        b = broadcast_players[(broadcast_teams == team) & np.isfinite(broadcast_players).all(axis=1)]
        if min(len(a), len(b)) < SPEC['min_matches_each_team']:
            return np.inf, (0, 0)
        distance = np.linalg.norm(a[:, None]-b[None], axis=2)
        row, col = linear_sum_assignment(distance)
        values = distance[row, col]
        distances.extend(values.tolist())
        counts.append(int((values <= SPEC['match_radius_m']).sum()))
    return float(np.median(distances)), tuple(counts)


def accepted(score):
    return score[0] <= SPEC['max_median_player_error_m'] and min(score[1]) >= SPEC['min_matches_each_team']


def align(tactical, broadcast):
    fps = float(tactical['fps'])
    if not np.isclose(fps, float(broadcast['fps'])):
        raise ValueError('Equal frame rates are required; resample explicitly first')
    if not np.allclose(tactical['pitch'], broadcast['pitch']):
        raise ValueError('Pitch coordinate systems differ')
    tactical, broadcast = tc.inputs_only(tactical), tc.inputs_only(broadcast)
    stride = max(1, round(fps*SPEC['coarse_seconds']))
    sample = np.arange(0, len(tactical['ball']), stride)
    desc = descriptor(tactical['players'][sample], tactical['teams'])
    valid = np.isfinite(desc).all(axis=1) & (tactical['period'][sample] > 0)
    sample, desc = sample[valid], desc[valid]
    if not len(sample):
        raise ValueError('No usable tactical player constellations')
    tree = cKDTree(desc)
    anchors = []
    step = max(1, round(fps*SPEC['anchor_seconds']))
    for index in range(0, len(broadcast['ball']), step):
        d = descriptor(broadcast['players'][index:index+1], broadcast['teams'])[0]
        if not np.isfinite(d).all():
            continue
        _, candidates = tree.query(d, k=min(SPEC['candidates'], len(sample)))
        ranked = []
        for candidate in sample[np.atleast_1d(candidates)]:
            score = geometry(tactical['players'][candidate], tactical['teams'],
                             broadcast['players'][index], broadcast['teams'])
            ranked.append((score[0], int(candidate)))
        best = (np.inf, -1, (0, 0))
        for _, center in sorted(ranked)[:3]:
            for candidate in range(max(0, center-stride), min(len(tactical['ball']), center+stride+1)):
                if tactical['period'][candidate] <= 0:
                    continue
                score = geometry(tactical['players'][candidate], tactical['teams'],
                                 broadcast['players'][index], broadcast['teams'])
                if score[0] < best[0]:
                    best = score[0], candidate, score[1]
        if accepted((best[0], best[2])):
            anchors.append(dict(broadcast=index, tactical=best[1], error=best[0]))
    if len(anchors) < SPEC['min_cluster_anchors']:
        raise ValueError('Insufficient player-geometry anchors; fusion is unavailable')
    offsets = np.array([v['tactical']-v['broadcast'] for v in anchors])
    groups = DBSCAN(eps=fps*SPEC['cluster_radius_seconds'], min_samples=SPEC['min_cluster_anchors']).fit_predict(offsets[:, None])
    alignment = np.full(len(broadcast['ball']), -1, dtype=np.int32)
    errors = np.full(len(alignment), np.inf)
    clusters = []
    # Never extrapolate a constant offset across a distant cut or a half boundary.
    for group in sorted(set(groups)-{-1}):
        selected = [a for a, g in zip(anchors, groups) if g == group]
        offset = int(round(np.median([a['tactical']-a['broadcast'] for a in selected])))
        positions = np.array([a['broadcast'] for a in selected])
        clusters.append(dict(offset_frames=offset, anchors=len(selected)))
        for index in range(len(alignment)):
            if np.min(np.abs(positions-index)) > fps*SPEC['max_anchor_extrapolation_seconds']:
                continue
            candidate = index+offset
            if candidate < 0 or candidate >= len(tactical['ball']) or tactical['period'][candidate] <= 0:
                continue
            score = geometry(tactical['players'][candidate], tactical['teams'],
                             broadcast['players'][index], broadcast['teams'])
            if accepted(score) and score[0] < errors[index]:
                errors[index], alignment[index] = score[0], candidate
    # Reject isolated constellation coincidences and offset/period discontinuities.
    edges = np.r_[True, (np.diff(alignment) != 1) | (alignment[1:] < 0), True]
    starts = np.flatnonzero(edges)
    for a, b in zip(starts[:-1], starts[1:]):
        if alignment[a] >= 0 and b-a < fps*SPEC['minimum_alignment_run_seconds']:
            alignment[a:b] = -1
    return alignment, dict(spec=SPEC, candidate_anchors=len(anchors), clusters=clusters,
                           accepted_broadcast_frames=int((alignment >= 0).sum()),
                           accepted_fraction=float((alignment >= 0).mean()))


def fuse(tactical, broadcast, alignment):
    """Use measured broadcast observations only; preserve observed tactical ball."""
    out = tc.inputs_only(tactical)
    alignment = np.asarray(alignment)
    if alignment.shape != (len(broadcast['ball']),):
        raise ValueError('One alignment index is required per broadcast frame')
    if ((alignment < -1) | (alignment >= len(out['ball']))).any():
        raise ValueError('Invalid tactical frame index')
    # Multiple broadcast observations of the same tactical frame are ambiguous.
    mapped = alignment[alignment >= 0]
    count = np.bincount(mapped, minlength=len(out['ball']))
    seen = np.isfinite(out['ball']).all(axis=1)
    source = tc.inputs_only(broadcast)['ball']
    valid = np.isfinite(source).all(axis=1)
    valid &= ((source >= -5) & (source <= np.asarray(out['pitch'])+5)).all(axis=1)
    indices = np.flatnonzero((alignment >= 0) & valid)
    indices = indices[count[alignment[indices]] == 1]
    destination = alignment[indices]
    active = out['period'][destination] > 0
    indices, destination = indices[active], destination[active]
    overlap = seen[destination]
    residual = np.linalg.norm(out['ball'][destination[overlap]]-source[indices[overlap]], axis=1)
    gap = ~seen[destination]
    out['ball'][destination[gap]] = source[indices[gap]]
    out['ball_observed'] = np.isfinite(out['ball']).all(axis=1)
    eligible = (out['period'] > 0) & ~seen
    return out, dict(tactical_missing_active_frames=int(eligible.sum()),
                     recovered_frames=int(gap.sum()),
                     fraction_missing_recovered=float(gap.sum()/max(1, eligible.sum())),
                     original_missing_fraction=float(eligible.sum()/max(1, (out['period'] > 0).sum())),
                     fused_missing_fraction=float((~out['ball_observed'][out['period'] > 0]).mean()),
                     overlapping_ball_frames=len(residual),
                     overlap_ball_error_quantiles_m=np.quantile(residual, [.5, .9, .99]).tolist() if len(residual) else [],
                     overlap_error_over5m_fraction=float((residual > 5).mean()) if len(residual) else None,
                     ambiguous_tactical_frames=int((count > 1).sum()),
                     target_labels_read=False, organizer_scored=False,
                     limitation='Coverage and inter-feed agreement are not coordinate ground truth or possession accuracy')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tactical', type=Path, default=Path('data/challenge/inputs.npz'))
    p.add_argument('--broadcast-meta', type=Path, default=Path('data/broadcast/BAR-ATH_broadcast_atd.xml'))
    p.add_argument('--broadcast-raw', type=Path, default=Path('data/broadcast/BAR-ATH_broadcast_atd.txt'))
    p.add_argument('--out', type=Path, default=Path('data/experiments/broadcast_fusion'))
    args = p.parse_args()
    if (args.out/'report.json').exists():
        raise ValueError('Use a new output directory; do not overwrite a completed audit')
    args.out.mkdir(parents=True, exist_ok=True)
    tactical = current_baseline.load_inputs(args.tactical)
    broadcast = load_atd(args.broadcast_meta, args.broadcast_raw, tactical['pitch'])
    alignment, alignment_report = align(tactical, broadcast)
    fused, report = fuse(tactical, broadcast, alignment)
    np.savez_compressed(args.out/'fused_inputs.npz', **fused)
    np.save(args.out/'alignment.npy', alignment)
    result = dict(alignment=alignment_report, fusion=report,
                  inputs_sha256={p.name:sha256(p) for p in (args.tactical, args.broadcast_meta, args.broadcast_raw)})
    (args.out/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
