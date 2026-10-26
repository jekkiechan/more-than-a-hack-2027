"""Fixed probability ensembles of frozen D heads, on external development only.

The earlier public check and competition selection are never opened or changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from experiments import learned_temporal as lt, multisource_eval, supervised
from experiments.eval_set.protocol import paired_source_bootstrap, sha256
from experiments.temporal_report import source_mean

MEMBERS = ('bigru', 'ordered_hgb', 'lgbm_002')
ENSEMBLES = {
    'gru_hgb': {'bigru': .5, 'ordered_hgb': .5},
    'gru_lgbm': {'bigru': .5, 'lgbm_002': .5},
    'hgb_lgbm': {'ordered_hgb': .5, 'lgbm_002': .5},
    'gru_hgb_lgbm': {name: 1/3 for name in MEMBERS},
}
DEFAULT_CACHE = lt.ROOT/'data/experiments/learned_temporal'


def blend(probabilities, weights):
    if not weights or any(not np.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError('Weights must be positive and finite')
    if not np.isclose(sum(weights.values()), 1):
        raise ValueError('Weights must sum to one')
    shape = None
    result = None
    for name, weight in weights.items():
        p = np.asarray(probabilities[name], dtype=float)
        if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError('Expected finite one-dimensional probabilities in [0,1]')
        if shape is not None and p.shape != shape:
            raise ValueError('Member frame shapes differ')
        shape = p.shape
        result = weight*p if result is None else result + weight*p
    return result


def cases():
    for fold in lt.FOLDS:
        for source, split in lt.dead_transfer.SPLITS.items():
            if fold != 'pooled' and fold != f'without_{source}':
                continue
            for path in split['eval']:
                yield fold, source, path


def freeze(cache, out):
    files = {cache/'predeclared.json', lt.ROOT/'experiments/eval_set/manifest.json'}
    for fold, _, path in cases():
        files.update((path, cache/f'{path.stem}_attribution.npz', cache/f'{fold}_results.json'))
        for member in MEMBERS:
            files.update((cache/f'{fold}_{member}_{path.stem}_prediction.npz',
                          cache/f'{fold}_{member}_frozen.json'))
    spec = {
        'source_sha256': sha256(__file__), 'members': MEMBERS, 'ensembles': ENSEMBLES,
        'inputs_sha256': {str(p.relative_to(lt.ROOT)): sha256(p) for p in sorted(files)},
        'averaging': 'Linear average of native-rate P(D), before unchanged conditional H/A combination and Viterbi',
        'controls': {'conditional_HA': 'same fixed pooled HGB backbone', 'switch_cost': lt.COST,
                     'weights': 'four fixed equal-weight blends; no grid/stacking/calibration fit'},
        'scope': 'Separate external-only development experiment after original frozen experiment; prior organizer results known, never read for this evaluation',
        'primary': 'equal-provider mean official spell F1 on D-head provider-holdout folds',
        'limitations': 'Historical development matches, not fresh tests; provider holdout applies to D only',
        'organizer': 'No organizer inputs/labels/predictions read; original selection and public gates unchanged',
    }
    path = out/'predeclared.json'
    canonical = json.loads(json.dumps(spec))
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k, v in canonical.items()):
            raise ValueError('Ensemble source/spec/inputs changed; use a new output directory')
    else:
        with path.open('x') as handle:
            json.dump(dict(canonical, created_utc=datetime.now(timezone.utc).isoformat()), handle, indent=2)
            handle.write('\n')
    return canonical


def diagnostics(probabilities, labels):
    keep = np.isin(labels, supervised.LABELS)
    y = (labels[keep] == 'D').astype(float)
    p = {name: values[keep] for name, values in probabilities.items()}
    pairs = []
    for left, right in combinations(MEMBERS, 2):
        a, b = p[left]-y, p[right]-y
        ea, eb = (p[left] >= .5) != y, (p[right] >= .5) != y
        union = (ea | eb).sum()
        pairs.append(dict(members=[left, right],
                          residual_correlation=float(np.corrcoef(a, b)[0, 1])
                          if a.std() > 0 and b.std() > 0 else None,
                          raw_D_disagreement=float(np.mean((p[left] >= .5) != (p[right] >= .5))),
                          error_jaccard=float((ea & eb).sum()/union) if union else None))
    return dict(graded_frames=int(keep.sum()),
                brier={name: float(np.mean((values-y)**2)) for name, values in p.items()}, pairs=pairs)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    spec = freeze(args.cache, args.out)
    progress = args.out/'progress.json'
    rows = json.loads(progress.read_text()) if progress.exists() else []
    analyses = []
    for fold, source, path in cases():
        inputs, labels = multisource_eval.view(path)
        with np.load(args.cache/f'{path.stem}_attribution.npz', allow_pickle=False) as data:
            team = data['probabilities']
        probabilities = {}
        for member in MEMBERS:
            with np.load(args.cache/f'{fold}_{member}_{path.stem}_prediction.npz', allow_pickle=False) as data:
                probabilities[member] = data['dead']
        if any(v.shape != labels.shape for v in probabilities.values()) or team.shape != (len(labels), 3):
            raise ValueError('Cached probability/native tracking shape mismatch')
        analysis = diagnostics(probabilities, labels)
        analyses.append(dict(fold=fold, source=source, match=path.stem, **analysis))
        old_rows = json.loads((args.cache/f'{fold}_results.json').read_text())
        for member in MEMBERS:
            if any(r['fold'] == fold and r['match'] == path.stem and r['model'] == member for r in rows):
                continue
            old = next(r for r in old_rows if r['model'] == member and r['match'] == path.stem)
            rows.append(dict(old, kind='single', raw_d_brier=analysis['brier'][member]))
        for name, weights in ENSEMBLES.items():
            if any(r['fold'] == fold and r['match'] == path.stem and r['model'] == name for r in rows):
                continue
            dead = blend(probabilities, weights)
            summary, predicted = lt.summarize(inputs, labels, team, dead)
            keep = np.isin(labels, supervised.LABELS)
            brier = float(np.mean((dead[keep]-(labels[keep] == 'D'))**2))
            row = dict(fold=fold, source=source, match=path.stem, model=name,
                       kind='ensemble', weights=weights, raw_d_brier=brier, **summary)
            rows.append(row)
            np.savez_compressed(args.out/f'{fold}_{name}_{path.stem}.npz', dead=dead, prediction=predicted)
            progress.write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps({k: row[k] for k in ('fold', 'source', 'match', 'model', 'spell_f1', 'macro_f1', 'd_f1', 'raw_d_brier')}), flush=True)
    expected = {(fold, path.stem, model) for fold, _, path in cases() for model in (*MEMBERS, *ENSEMBLES)}
    actual = {(r['fold'], r['match'], r['model']) for r in rows}
    if actual != expected or len(actual) != len(rows):
        raise ValueError('Incomplete/duplicated comparison; refusing to summarize')
    summaries, deltas = [], []
    for view in ('pooled', 'provider_holdout_D'):
        subset = [r for r in rows if (r['fold'] == 'pooled') == (view == 'pooled')]
        for name in (*MEMBERS, *ENSEMBLES):
            these = [r for r in subset if r['model'] == name]
            summaries.append(dict(view=view, model=name,
                **{metric: source_mean(these, metric) for metric in ('spell_f1', 'macro_f1', 'd_f1', 'raw_d_brier')},
                per_source={s: {metric: float(np.mean([r[metric] for r in these if r['source'] == s]))
                               for metric in ('spell_f1', 'd_f1')}
                            for s in lt.dead_transfer.SPLITS}))
            for baseline in ('bigru', 'ordered_hgb'):
                for metric in ('spell_f1', 'd_f1'):
                    deltas.append(dict(view=view, model=name, baseline=baseline,
                        **paired_source_bootstrap(
                            [dict(r, eval_source=r['source']) for r in subset if r['model'] == baseline],
                            [dict(r, eval_source=r['source']) for r in these], metric)))
    best = max((r for r in summaries if r['view'] == 'provider_holdout_D'),
               key=lambda r: (r['spell_f1'], r['macro_f1']))
    result = dict(scope=spec['scope'], spec=spec, summaries=summaries, paired_deltas=deltas,
                  diagnostics=analyses, rows=rows, best_external_development=best,
                  organizer_check_performed=False, prior_competition_selection_changed=False)
    with args.report.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for row in summaries:
        print('SUMMARY', row['view'], row['model'], round(row['spell_f1'], 5), round(row['d_f1'], 5), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--out', type=Path, default=lt.ROOT/'data/experiments/temporal_ensemble')
    parser.add_argument('--report', type=Path, default=lt.ROOT/'experiments/results/temporal_ensemble_development.json')
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Report already frozen; do not overwrite it')
    run(args)


if __name__ == '__main__':
    main()
