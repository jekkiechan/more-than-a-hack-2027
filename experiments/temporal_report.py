"""Freeze the externally selected temporal-D comparison before a public check."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
from experiments import learned_temporal as lt
from experiments.eval_set.protocol import paired_source_bootstrap


def source_mean(rows, key):
    return float(np.mean([np.mean([r[key] for r in rows if r['source'] == s])
                          for s in sorted({r['source'] for r in rows})]))


def report(out, destination):
    if destination.exists():
        raise ValueError('A frozen development report already exists')
    rows = [row for fold in lt.FOLDS for row in json.loads((out/f'{fold}_results.json').read_text())]
    expected = {(fold, kind, source, p.stem) for fold in lt.FOLDS for kind in lt.MODELS
                for source, split in lt.dead_transfer.SPLITS.items()
                if fold == 'pooled' or fold == f'without_{source}' for p in split['eval']}
    observed = {(r['fold'], r['model'], r['source'], r['match']) for r in rows}
    if len(observed) != len(rows) or observed != expected:
        raise ValueError('Incomplete/duplicated predeclared comparison; do not select a partial run')
    for row in rows:
        row['d_boundary_evaluation_supported'] = any(v['n_gt'] > 0 for v in row['d_boundaries'].values())
    summary, paired = [], []
    for view in ('pooled', 'provider_holdout_D'):
        subset = [r for r in rows if (r['fold'] == 'pooled') == (view == 'pooled')]
        baseline = [dict(r, eval_source=r['source']) for r in subset if r['model'] == 'hgb']
        for kind in lt.MODELS:
            current = [r for r in subset if r['model'] == kind]
            item = dict(view=view, model=kind,
                        **{metric: source_mean(current, metric) for metric in
                           ('spell_f1', 'macro_f1', 'd_f1', 'd_precision', 'd_recall', 'raw_d_f1', 'd_auroc')},
                        per_source={s: {metric: float(np.mean([r[metric] for r in current if r['source'] == s]))
                                      for metric in ('spell_f1', 'macro_f1', 'd_f1')}
                                    for s in lt.dead_transfer.SPLITS})
            summary.append(item)
            for metric in ('spell_f1', 'macro_f1', 'd_f1'):
                paired.append(dict(view=view, model=kind,
                    **paired_source_bootstrap(baseline, [dict(r, eval_source=r['source']) for r in current], metric)))
    chosen = max((r for r in summary if r['view'] == 'provider_holdout_D'),
                 key=lambda r: (r['spell_f1'], r['macro_f1']))
    selected = chosen['model']
    model_path = out/f'pooled_{selected}.joblib'
    selection = dict(selected_model=selected, selected_model_sha256=lt.digest(model_path),
                     predeclared_spec_sha256=lt.digest(out/'predeclared.json'),
                     attribution_model_sha256=lt.digest(out/'attribution.joblib'),
                     selection_rule=lt.SPEC['selection'], frozen_utc=datetime.now(timezone.utc).isoformat(),
                     public_check_performed=False, organizer_assets_available=False)
    with (out/'selection.json').open('x') as handle:
        json.dump(selection, handle, indent=2)
        handle.write('\n')
    versions = {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'lightgbm', 'torch', 'scipy')}
    result = dict(scope='Historical external development matches; no fresh test and no organizer result',
                  spec=json.loads((out/'predeclared.json').read_text()), versions=versions,
                  boundary_limitations='SkillCorner D intervals have X margins: zero eligible reference boundaries. Its official boundary zeros are unavailable evidence, not measured timing failure; inspect n_gt and d_boundary_evaluation_supported. No cross-provider boundary mean is computed.',
                  selection=selection, summary=summary, paired_deltas=paired, rows=rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for row in summary:
        print(row['view'], row['model'], 'spell', round(row['spell_f1'], 4),
              'macro', round(row['macro_f1'], 4), 'D', round(row['d_f1'], 4))
    print('SELECTED', selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=lt.ROOT/'data/experiments/learned_temporal')
    parser.add_argument('--destination', type=Path, default=lt.ROOT/'experiments/results/learned_temporal_development.json')
    args = parser.parse_args()
    report(args.out, args.destination)


if __name__ == '__main__':
    main()
