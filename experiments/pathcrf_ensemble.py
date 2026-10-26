# SPDX-License-Identifier: MPL-2.0
# PathCRF feature/decode conventions follow the MPL-2.0 upstream implementation.
"""External-only PathCRF H/A ablation with the promoted GRU/HGB D head."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from experiments import challenge_pathcrf, learned_temporal as lt, multisource_eval, pathcrf_eval, supervised
from experiments.eval_set.protocol import paired_source_bootstrap, sha256
from experiments.temporal_report import source_mean

ROOT = lt.ROOT
CACHE = ROOT/'data/experiments/learned_temporal'
OUT = ROOT/'data/experiments/pathcrf_ensemble_v2'
MATCHES = ('metrica_3', 'skillcorner_had_1996435', 'skillcorner_had_1996436', 'skillcorner_had_2006229')
METHODS = ('gru_hgb_baseline', 'gru_hgb_pathcrf_team', 'gru_hgb_team_blend')
CHECKPOINT_SHA = '55757d9dca9777d60356ba368dc315867ea1ae38e90c332e935b9c5be72d83c9'


def path_for(match):
    if match not in MATCHES:
        raise ValueError('Match not in predeclared uncontaminated-source cohort')
    return ROOT/'data/external/processed'/f'{match}.npz'


def source_for(match):
    return 'metrica' if match.startswith('metrica') else 'skillcorner'


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = pathcrf_eval.UPSTREAM/pathcrf_eval.CHECKPOINT
    if sha256(checkpoint) != CHECKPOINT_SHA:
        raise ValueError('Unexpected pretrained checkpoint')
    paths = [Path(__file__), Path(pathcrf_eval.__file__), Path(challenge_pathcrf.__file__), checkpoint]
    for match in MATCHES:
        paths += [path_for(match), CACHE/f'{match}_attribution.npz']
        for fold in ('pooled', f'without_{source_for(match)}'):
            paths += [CACHE/f'{fold}_{m}_{match}_prediction.npz' for m in ('bigru', 'ordered_hgb')]
    spec = dict(checkpoint_commit=pathcrf_eval.COMMIT, checkpoint_sha256=CHECKPOINT_SHA,
        inputs_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths}, matches=MATCHES,
        methods=METHODS, D='Fixed 50/50 GRU + ordered HGB probabilities',
        team='Original HGB conditional team, PathCRF CRF vote share, or fixed 50/50 average of these',
        weights='No sweep or calibration', decoder_switch_cost=lt.COST,
        geometry='Existing challenge_pathcrf roster stabilization within each continuous nonzero-period block; original 10s/5s-stride PathCRF adapter; neutral 50/50 shares outside periods',
        PathCRF_training=['J03WMX', 'J03WN1', 'J03WOH', 'J03WOY', 'J03WPY'],
        PathCRF_validation=['J03WQQ'], excluded='All Sportec matches; J03WOY was checkpoint training data',
        interpretation='Historical Metrica/SkillCorner development comparison; provider holdout applies to D only',
        organizer='No organizer data, predictions or labels read; old public gates unchanged')
    spec = json.loads(json.dumps(spec))
    path = out/'predeclared.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k, v in spec.items()):
            raise ValueError('Frozen input/source/spec changed')
    else:
        with path.open('x') as handle:
            json.dump(dict(spec, created_utc=datetime.now(timezone.utc).isoformat()), handle, indent=2)
            handle.write('\n')
    return spec


def prepare_geometry(path):
    """Read geometry and observed masks only, never target labels or ball."""
    data = pathcrf_eval.load_geometry(path)
    with np.load(path, allow_pickle=False) as source:
        if 'player_observed' in source.files:
            data['players'] = np.array(data['players'], copy=True)
            data['players'][~source['player_observed']] = np.nan
    players = np.full((len(data['period']), 22, 2), np.nan, dtype=np.float32)
    observed = np.zeros((len(data['period']), 22), dtype=bool)
    audits = []
    for start, stop in pathcrf_eval.blocks(data):
        if data['period'][start] <= 0:
            continue
        prepared, mask, audit = challenge_pathcrf.prepare_half(data, np.arange(start, stop))
        players[start:stop], observed[start:stop] = prepared['players'], mask
        audits.append(dict(start=int(start), stop=int(stop), teams=audit))
    geometry = dict(players=players, teams=np.repeat([0, 1], 11),
                    **{k: data[k] for k in ('period', 'fps', 'timestamps', 'pitch')})
    return geometry, observed, audits


def active_phases(data):
    for start, stop in pathcrf_eval.blocks(data):
        if data['period'][start] <= 0:
            continue
        block = {k: data[k][start:stop] if k in ('players', 'period', 'timestamps') else data[k]
                 for k in pathcrf_eval.GEOMETRY_KEYS if k in data}
        for begin, end, features, teams in pathcrf_eval.feature_phases(block):
            yield start+begin, start+end, features, teams


def vote_shares(data, model, batch_size=4):
    """Expose original CRF triangular overlap vote shares; not calibrated posteriors."""
    import torch
    fps = float(data['fps'])
    step = round(fps/5)
    if step < 1 or not np.isclose(fps/step, 5):
        raise ValueError('Expected source frame rate divisible by 5')
    result = np.full((len(data['period']), 2), .5)
    started = time.monotonic()
    stats = dict(phases=0, windows=0, outside_source_steps=0, decoded_steps=0)
    with torch.inference_mode():
        for begin, end, full, node_teams in active_phases(data):
            features = full[::step]
            n = len(features)
            width = min(50, n)
            starts = list(range(0, max(1, n-width+1), 25))
            if starts[-1] != n-width:
                starts.append(n-width)
            votes, total = np.zeros((n, 2)), np.zeros(n)
            weights = 1 + np.minimum(np.arange(width), np.arange(width)[::-1])
            valid_players = np.flatnonzero(node_teams >= 0)
            for offset in range(0, len(starts), batch_size):
                chunk = starts[offset:offset+batch_size]
                tensor = torch.from_numpy(np.stack([features[s:s+width] for s in chunk]))
                macro, micro, edges = model(tensor)
                valid_edges = model.valid_orig_edge_ids
                decoded = model.crf.decode(micro.index_select(2, valid_edges), edges.index_select(2, valid_edges))
                sources = model.comp_src[decoded].numpy()
                macro_prob = torch.softmax(macro[:, :, valid_players, 0], dim=-1).numpy()
                team_prob = np.stack([macro_prob[:, :, node_teams[valid_players] == team].sum(axis=-1)
                                      for team in (0, 1)], axis=-1)
                for row, start in enumerate(chunk):
                    teams = node_teams[sources[row]].copy()
                    stats['outside_source_steps'] += int(np.count_nonzero(teams < 0))
                    stats['decoded_steps'] += width
                    for frame in range(width):
                        if teams[frame] < 0:
                            teams[frame] = teams[frame-1] if frame else int(team_prob[row, frame].argmax())
                    votes[start:start+width] += np.eye(2)[teams]*weights[:, None]
                    total[start:start+width] += weights
                stats['windows'] += len(chunk)
            if (total <= 0).any():
                raise ValueError('Uncovered PathCRF frame')
            up = np.minimum(np.arange(end-begin)//step, n-1)
            result[begin:end] = (votes/total[:, None])[up]
            stats['phases'] += 1
            print(f'PATHCRF phase={stats["phases"]} frame={end}/{len(result)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1):
        raise ValueError('Invalid PathCRF vote shares')
    stats['inference_seconds'] = time.monotonic()-started
    return result, stats


def predict(match, out):
    if (out/f'{match}_inference.json').exists():
        print('Already frozen', match)
        return
    start = time.monotonic()
    geometry, observed, audit = prepare_geometry(path_for(match))
    np.savez_compressed(out/f'{match}_geometry.npz', **geometry, player_observed=observed)
    (out/f'{match}_geometry_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print('GEOMETRY', match, 'imputed', round(float(1-observed.mean()), 4), flush=True)
    model = pathcrf_eval.load_model(threads=1)
    shares, stats = vote_shares(geometry, model)
    path = out/f'{match}_team.npz'
    np.savez_compressed(path, team=shares, period=geometry['period'], timestamps=geometry['timestamps'])
    (out/f'{match}_inference.json').write_text(json.dumps(dict(
        prediction_sha256=sha256(path), checkpoint_sha256=CHECKPOINT_SHA,
        geometry_imputed_fraction=float(1-observed.mean()), labels_read=False,
        seconds_including_geometry=time.monotonic()-start, **stats), indent=2)+'\n')


def team_variants(base, pathcrf):
    alive = base[:, :2].sum(axis=1)
    conditional = np.divide(base[:, :2], alive[:, None], out=np.full_like(base[:, :2], .5),
                            where=alive[:, None] > 1e-12)
    return {'gru_hgb_baseline': base,
            'gru_hgb_pathcrf_team': np.column_stack((pathcrf, np.zeros(len(base)))),
            'gru_hgb_team_blend': np.column_stack(((conditional+pathcrf)*.5, np.zeros(len(base))))}


def score(out, report_path):
    if report_path.exists():
        raise ValueError('Comparison already scored')
    # Check every inference receipt before opening external targets.
    for match in MATCHES:
        receipt = json.loads((out/f'{match}_inference.json').read_text())
        if receipt['prediction_sha256'] != sha256(out/f'{match}_team.npz'):
            raise ValueError('Prediction changed after freeze')
    rows = []
    for match in MATCHES:
        source = source_for(match)
        inputs, labels = multisource_eval.view(path_for(match))
        with np.load(out/f'{match}_team.npz', allow_pickle=False) as data:
            q_path = data['team']
            np.testing.assert_array_equal(data['timestamps'], inputs['timestamps'])
            np.testing.assert_array_equal(data['period'], inputs['period'])
        with np.load(CACHE/f'{match}_attribution.npz', allow_pickle=False) as data:
            base = data['probabilities']
        for fold in ('pooled', f'without_{source}'):
            members = []
            for member in ('bigru', 'ordered_hgb'):
                with np.load(CACHE/f'{fold}_{member}_{match}_prediction.npz', allow_pickle=False) as data:
                    members.append(data['dead'])
            dead = np.mean(members, axis=0)
            for method, probabilities in team_variants(base, q_path).items():
                summary, pred = lt.summarize(inputs, labels, probabilities, dead)
                live = np.isin(labels, ['H', 'A'])
                ha = np.array(['H', 'A'])[probabilities[:, :2].argmax(axis=1)]
                row = dict(model=method, fold=fold, source=source, match=match,
                           ha_accuracy_gt_live=float(np.mean(ha[live] == labels[live])), **summary)
                rows.append(row)
                np.savez_compressed(out/f'{fold}_{method}_{match}.npz', prediction=pred)
                print(json.dumps({k: row[k] for k in ('model', 'fold', 'source', 'match', 'spell_f1', 'macro_f1', 'ha_accuracy_gt_live')}), flush=True)
    summaries, deltas = [], []
    for view in ('pooled', 'provider_holdout_D'):
        subset = [r for r in rows if (r['fold'] == 'pooled') == (view == 'pooled')]
        for method in METHODS:
            selected = [r for r in subset if r['model'] == method]
            summary = dict(view=view, model=method,
                **{m: source_mean(selected, m) for m in ('spell_f1', 'macro_f1', 'd_f1', 'ha_accuracy_gt_live')},
                per_source={s: {m: float(np.mean([r[m] for r in selected if r['source'] == s]))
                               for m in ('spell_f1', 'macro_f1', 'ha_accuracy_gt_live')}
                            for s in ('metrica', 'skillcorner')})
            summary['weighted_summary'] = .7*summary['spell_f1']+.3*summary['macro_f1']
            summaries.append(summary)
            deltas.append(dict(view=view, model=method, **paired_source_bootstrap(
                [dict(r, eval_source=r['source']) for r in subset if r['model'] == 'gru_hgb_baseline'],
                [dict(r, eval_source=r['source']) for r in selected])))
    report = dict(scope='Historical external-only Metrica/SkillCorner comparison; no Sportec checkpoint-training overlap',
        spec=json.loads((out/'predeclared.json').read_text()), summaries=summaries, rows=rows,
        paired_deltas=deltas, organizer_check_performed=False,
        inference={match: json.loads((out/f'{match}_inference.json').read_text()) for match in MATCHES})
    with report_path.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for row in summaries:
        print('SUMMARY', row['view'], row['model'], row['spell_f1'], row['macro_f1'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'predict', 'score'))
    parser.add_argument('--match', choices=MATCHES)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--report', type=Path, default=ROOT/'experiments/results/pathcrf_ensemble_development.json')
    args = parser.parse_args()
    freeze(args.out)
    if args.action == 'predict':
        if args.match is None:
            parser.error('predict requires --match')
        predict(args.match, args.out)
    elif args.action == 'score':
        score(args.out, args.report)


if __name__ == '__main__':
    main()
