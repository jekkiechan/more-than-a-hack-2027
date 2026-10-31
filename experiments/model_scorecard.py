"""Compare saved external metrics with spell-first and weighted rankings.

No models are evaluated and no organizer results are read. Historical experiment
selection records remain unchanged; the weighted column is a retrospective view.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORTS = (ROOT/'experiments/results/learned_temporal_development.json',
                   ROOT/'experiments/results/temporal_ensemble_development.json')


def combined_score(spell, macro, spell_weight=.7):
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (spell, macro, spell_weight)):
        raise ValueError('F1 values and spell weight must be finite in [0,1]')
    return spell_weight*spell + (1-spell_weight)*macro


def collect(reports, view):
    if view not in ('pooled', 'provider_holdout_D'):
        raise ValueError('Unsupported external comparison view')
    candidates = {}
    for report in reports:
        if report.get('organizer_check_performed') or not report.get('rows'):
            raise ValueError('Expected saved external match rows, not organizer results')
        for row in report['rows']:
            fold, source = row['fold'], row['source']
            if fold != 'pooled' and fold != f'without_{source}':
                raise ValueError('Unexpected provider fold')
            if (fold == 'pooled') != (view == 'pooled'):
                continue
            key = source, row['match']
            metrics = (float(row['spell_f1']), float(row['macro_f1']))
            combined_score(*metrics)
            rows = candidates.setdefault(row['model'], {})
            if key in rows and rows[key] != metrics:
                raise ValueError('Conflicting metrics for the same model and match')
            rows[key] = metrics
    if not candidates:
        raise ValueError('No external candidates')
    cohorts = [set(rows) for rows in candidates.values()]
    if any(cohort != cohorts[0] for cohort in cohorts[1:]):
        raise ValueError('Compare candidates on exactly the same source/match cohort')
    return candidates


def scorecard(candidates, spell_weight):
    combined_score(0., 0., spell_weight)
    result = []
    for model, rows in candidates.items():
        sources = sorted({source for source, _ in rows})
        by_source = {source: {
            'spell_f1': mean(v[0] for (s, _), v in rows.items() if s == source),
            'macro_f1': mean(v[1] for (s, _), v in rows.items() if s == source),
        } for source in sources}
        spell = mean(v['spell_f1'] for v in by_source.values())
        macro = mean(v['macro_f1'] for v in by_source.values())
        result.append(dict(model=model, spell_f1=spell, macro_f1=macro,
                           weighted_score=combined_score(spell, macro, spell_weight),
                           per_source=by_source))
    for field, key in (
        ('spell_first_rank', lambda r: (r['spell_f1'], r['macro_f1'])),
        ('weighted_rank', lambda r: (r['weighted_score'], r['spell_f1'], r['macro_f1'])),
    ):
        previous, rank = None, 0
        for position, row in enumerate(sorted(result, key=key, reverse=True), 1):
            if key(row) != previous:
                rank = position
            row[field] = rank
            previous = key(row)
    return sorted(result, key=lambda r: (r['spell_first_rank'], r['model']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, nargs='+', default=list(DEFAULT_REPORTS))
    parser.add_argument('--view', choices=('pooled', 'provider_holdout_D'), default='provider_holdout_D')
    parser.add_argument('--policy', type=Path, default=ROOT/'experiments/model_decision_policy.json')
    parser.add_argument('--spell-weight', type=float,
                        help='Weight in the additional combined column; macro weight is 1 minus this')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/results/model_scorecard_development.json')
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    if policy['primary_order'] != ['spell_f1', 'macro_f1'] or policy['weighted_summary']['used_for_primary_ranking']:
        parser.error('This scorecard preserves the confirmed competition headline ranking')
    if not math.isclose(policy['weighted_summary']['spell_weight'] + policy['weighted_summary']['macro_weight'], 1):
        parser.error('Policy weights must sum to one')
    weight = policy['weighted_summary']['spell_weight'] if args.spell_weight is None else args.spell_weight
    reports = [json.loads(path.read_text()) for path in args.reports]
    candidates = collect(reports, args.view)
    rows = scorecard(candidates, weight)
    sensitivity = []
    for alternative in (0., .25, .5, .7, .8, 1.):
        ranking = sorted(scorecard(candidates, alternative), key=lambda r: r['weighted_rank'])
        sensitivity.append(dict(spell_weight=alternative, macro_weight=1-alternative,
                                order=[r['model'] for r in ranking]))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), view=args.view,
        scope='Retrospective external development scorecard; not a new evaluation or organizer selection',
        competition_rule='spell F1 descending, then macro F1 for exact ties; weighted score is not the official metric',
        weighted_score=dict(spell_weight=weight, macro_weight=1-weight,
                            purpose='additional internal summary, configurable before future experiments'),
        policy=policy, policy_sha256=hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        scorecard_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        aggregation='Mean within each provider, then equal mean across providers',
        report_sha256={(str(p.resolve().relative_to(ROOT)) if p.resolve().is_relative_to(ROOT) else str(p.resolve())):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in args.reports},
        candidates=rows, weight_sensitivity=sensitivity)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for row in rows:
        print(row['model'], 'spell', f"{row['spell_f1']:.4f}", 'macro', f"{row['macro_f1']:.4f}",
              'combined', f"{row['weighted_score']:.4f}")


if __name__ == '__main__':
    main()
