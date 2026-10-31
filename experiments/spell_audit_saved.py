"""Posthoc spell diagnostics of saved external predictions; never fit or load organizer data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.metrics import evaluate_tracking
from experiments.spell_diagnostics import spell_diagnostics

ROOT = Path(__file__).resolve().parents[1]
# The fixed historical cohort, not a new selection or confirmation split.
COHORT = {'sportec': ['J03WOY'], 'metrica': ['metrica_3'],
          'skillcorner': ['skillcorner_had_1996435', 'skillcorner_had_1996436', 'skillcorner_had_2006229']}
ARMS = ('baseline', 'had_gru')
VIEWS = ('clean', 'corrupt_1', 'corrupt_2')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(archive):
    spec_path = archive/'predeclared.json'
    spec = json.loads(spec_path.read_text())
    if spec['evaluation_matches'] != COHORT:
        raise ValueError('Unexpected archive cohort')
    rows = []
    for source, matches in COHORT.items():
        saved = json.loads((archive/f'{source}_results.json').read_text())
        for match in matches:
            paths = [ROOT/p for p in spec['data_sha256'] if Path(p).stem == match]
            if len(paths) != 1:
                raise ValueError('Missing or ambiguous dataset')
            path = paths[0]
            if digest(path) != spec['data_sha256'][str(path.relative_to(ROOT))]:
                raise ValueError(f'Dataset hash mismatch: {match}')
            with np.load(path, allow_pickle=False) as data:
                labels = data['labels'].astype(str)
                if source == 'skillcorner':
                    labels[data['player_observed'].sum(axis=1) < 12] = 'X'
                kwargs = dict(period=data['period'], timestamps=data['timestamps'], fps=float(data['fps']))
            for view in VIEWS:
                for arm in ARMS:
                    prior = [r for r in saved if (r['match'], r['view'], r['arm']) == (match, view, arm)]
                    if len(prior) != 1:
                        raise ValueError('Missing or duplicate archived score')
                    pred_path = archive/f'{source}_{match}_{view}_{arm}.npz'
                    with np.load(pred_path, allow_pickle=False) as prediction:
                        pred = prediction['prediction']
                    score = evaluate_tracking(labels, pred, **kwargs)
                    for old, new in [('spell_f1', 'spell_f1_mean'), ('macro_f1', 'macro_f1'),
                                     ('n_gt_spells', 'n_gt_spells'), ('n_pred_spells', 'n_pred_spells')]:
                        if not np.isclose(prior[0][old], score[new], atol=1e-12, rtol=0):
                            raise ValueError(f'Archive parity failed: {match}/{view}/{arm}/{old}')
                    rows.append(dict(source=source, match=match, view=view, arm=arm,
                                     prediction_sha256=digest(pred_path),
                                     **spell_diagnostics(labels, pred, **kwargs)))
    summaries = []
    for arm in ARMS:
        for condition, views in [('clean', ['clean']), ('stress', ['corrupt_1', 'corrupt_2'])]:
            selected = [r for r in rows if r['arm'] == arm and r['view'] in views]
            def balanced(fn):
                return float(np.mean([np.mean([fn(r) for r in selected if r['source'] == s]) for s in COHORT]))
            summaries.append(dict(arm=arm, condition=condition,
                spell_f1_mean=balanced(lambda r:r['spell_f1_mean']),
                per_iou={str(t):{k:balanced(lambda r,t=t,k=k:r['per_iou'][str(t)][k])
                          for k in ('precision','recall','f1')} for t in (.3,.4,.5,.6,.7)},
                short_recall_at_0_5=balanced(lambda r:r['per_iou']['0.5']['reference_recall_groups']['under_3_seconds']['recall']),
                run_edge_spell_fraction=balanced(lambda r:r['reference_spells_touching_graded_run_edge']/r['n_gt_spells']),
                fragmented_reference_fraction=balanced(lambda r:r['overlap_geometry']['reference_overlapped_by_multiple_same_team_predictions']/r['n_gt_spells']),
                merged_prediction_fraction=balanced(lambda r:r['overlap_geometry']['predictions_overlapping_multiple_same_team_references']/max(1,r['n_pred_spells']))))
    return dict(scope='Posthoc diagnostics of fixed archived external full-pipeline provider holdouts',
                models='Original unrestricted joint GRU and matched baseline, not the later player-only D GRU',
                aggregation='Mean over views/matches within provider, then equal provider mean',
                organizer_labels_read=False, fresh_confirmation=False, models_refit=False,
                archived_spec_sha256=digest(spec_path), archive_score_parity='all 30 rows within 1e-12',
                official_scorer_sha256=digest(ROOT/'evaluation/possession_metrics.py'),
                input_sha256={p:h for p,h in spec['data_sha256'].items() if Path(p).stem in sum(COHORT.values(), [])},
                source_sha256={str(p.relative_to(ROOT)):digest(p) for p in
                    [Path(__file__),ROOT/'experiments/spell_diagnostics.py',ROOT/'experiments/metrics.py']},
                summaries=summaries, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT/'data/experiments/live_gap_had_v1')
    parser.add_argument('--out', type=Path, default=ROOT/'experiments/results/spell_metric_audit.json')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite a completed diagnostic')
    result = run(args.archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result['summaries'], indent=2))


if __name__ == '__main__':
    main()
