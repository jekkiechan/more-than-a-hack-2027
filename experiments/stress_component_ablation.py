"""Which composite-stress components drive the frozen models' stress loss?

Re-scores existing frozen provider-holdout checkpoints only; nothing is
trained. New views keep subsets of the frozen composite's components, with
the composite's own two seeds, so every view shares the composite's event
realizations. Clean and composite views are recomputed and checked against
the archived scores before any new view is reported. No organizer labels are
read.

    python -m experiments.stress_component_ablation freeze
    python -m experiments.stress_component_ablation run --match J03WOY --view ball_gaps
    python -m experiments.stress_component_ablation report
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
from experiments import (corruption_experiment as ce, dead_transfer as dt, gru_robustness_experiment as gr,
                         factorized_had, joint_lgbm as tree, joint_lgbm_experiment as joint,
                         learned_temporal as lt, live_gap_experiment as old, multisource_eval as ms,
                         possession_geometry as geometry, refine_domain, refine_features,
                         stress_components as sc, supervised)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
OUT = ROOT/'data/experiments/stress_component_ablation_v1'
RESULT = ROOT/'experiments/results/stress_component_ablation.json'
ALL = set(sc.COMPONENTS)
VIEWS = {
    'clean': (),
    'ball_gaps': ('ball_gaps',),
    'feed_like': ('ball_gaps', 'coordinates', 'slot_gaps', 'id_swaps'),
    'no_sampling': tuple(sorted(ALL-{'sampling'})),
    'no_regional': tuple(sorted(ALL-{'regional'})),
    'composite': tuple(sc.COMPONENTS),
}
REPLICATES = ('corrupt_1', 'corrupt_2')
GRUS = tuple(f'{a}_{s}' for a in gr.ARMS for s in gr.SEEDS)
RECIPES = (('baseline',) + GRUS + ('full_127',) + tuple(f'full_127+{g}' for g in GRUS)
           + tuple(f'geometry_63+control_{s}' for s in gr.SEEDS))


def units():
    for source, split in dt.SPLITS.items():
        for path in split['eval']:
            for view in VIEWS:
                for replicate in (('clean',) if view == 'clean' else REPLICATES):
                    yield source, path, view, replicate


def fingerprints():
    return {str(Path(m.__file__).relative_to(ROOT)): sha256(Path(m.__file__))
            for m in (sc, ce, gr, factorized_had, tree, joint, lt, old, geometry, refine_domain,
                      refine_features, supervised, __import__('experiments.stress_component_ablation',
                                                              fromlist=['x']))}


def checkpoints(source):
    paths = {'baseline': old.archive_paths(source)}
    for g in GRUS:
        paths[g] = gr.checkpoint(gr.DEFAULT, source, *g.rsplit('_', 1)[:1], int(g.rsplit('_', 1)[1]))
    for config in ('full_127', 'geometry_63'):
        paths[config] = joint.checkpoint(joint.DEFAULT, source, config)
    return paths


def flat(paths):
    for v in paths.values():
        yield from (v.values() if isinstance(v, dict) else [v])


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    calibration = json.loads((ROOT/'experiments/results/realistic_corruption_development.json').read_text())['spec']['calibration']
    spec = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        user_request='Isolate which stress components cause the GRU stress loss; check whether the LightGBM adds value under feed-like stress.',
        question='Fraction of each recipe\'s clean-to-composite spell loss that remains when unrealistic components are removed.',
        organizer_labels_read=False, training=False,
        views=VIEWS, replicates=REPLICATES, recipes=RECIPES, calibration=calibration,
        view_rationale=dict(
            ball_gaps='Calibrated renewal ball gaps only (organizer median 10.7 s).',
            feed_like='Ball gaps, coordinate drift/noise/jumps, per-slot gaps and same-team ID swaps; '
                      'drops whole-frame sampling loss and regional blanking, which the organizer feed '
                      'does not show (0 empty-player frames; 0.25 vs ~70 frames/min losing >=25% of players).',
            no_sampling='Composite minus 2-8% per-frame loss of every object.',
            no_regional='Composite minus 1-5 s spatial player blanking.'),
        decision_rules=dict(
            attribution='Report retained loss = (clean - view)/(clean - composite) per recipe.',
            gru_guardrail_on_feed_like='Each GRU seed vs adopted baseline: feed_like spell delta >= -0.01.',
            lgbm_value='Full127 + slot_dropout blend minus its GRU on feed_like spell: >= +0.02 in both seeds '
                       'means the tree robustness value survives realistic stress; otherwise it does not.',
            caveat='Five previously exposed external matches; descriptive, no adoption.'),
        aggregation='Replicates averaged within match, matches within provider, providers equally.',
        source_sha256=fingerprints(),
        checkpoint_sha256={str(p.relative_to(ROOT)): sha256(p) for s in dt.SPLITS for p in flat(checkpoints(s))},
        data_sha256={str(p.relative_to(ROOT)): sha256(p) for s in dt.SPLITS for p in dt.SPLITS[s]['eval']},
    )
    with (OUT/'predeclared.json').open('x') as handle:
        json.dump(spec, handle, indent=1)
    print('FROZEN', sha256(OUT/'predeclared.json'))


def check():
    spec = json.loads((OUT/'predeclared.json').read_text())
    if spec['source_sha256'] != fingerprints():
        raise ValueError('Source changed after freeze')
    for key, digest in {**spec['checkpoint_sha256'], **spec['data_sha256']}.items():
        if sha256(ROOT/key) != digest:
            raise ValueError(f'Frozen file changed: {key}')
    return spec


def load(source):
    paths = checkpoints(source)
    models = {'baseline': {k: __import__('joblib').load(p) for k, p in paths['baseline'].items()}}
    for g in GRUS:
        models[g] = ce.load_model(gr.DEFAULT, paths[g])
    for config in ('full_127', 'geometry_63'):
        models[config] = ce.load_model(joint.DEFAULT, paths[config])
    return models


def probabilities(inputs, labels, source, match, models):
    """All recipe probability vectors for one view, reusing one feature pass."""
    team_x, names = refine_features.features(inputs)
    player, player_names = lt.player_features(inputs)
    indices = lt.sampled_indices(inputs)
    periods = supervised.model_periods(inputs)
    y = np.full(len(labels), -1, np.int8)
    for i, label in enumerate(supervised.LABELS):
        y[labels == label] = i
    common = dict(source=source, match=match, indices=indices, native_n=len(labels),
                  fps=float(inputs['fps']), period=periods[indices])
    record = dict(common, x=np.column_stack((team_x[indices], player[indices])), y=y[indices],
                  names=names+['D_'+n for n in player_names])
    dead = dict(common, x=player[indices], names=player_names, variant='eval',
                y=np.where(y[indices] < 0, -1, y[indices] == 2).astype(np.int8))
    up = lambda sampled, rec: np.column_stack([lt.upsample(sampled[:, k], rec, periods) for k in range(3)])
    out = {}
    base = models['baseline']
    team = refine_features.probabilities_from_features(team_x, names, base['team'])
    d = .5*sum(lt.upsample(lt.predict_sampled(dead, base[k]), dead, periods) for k in ('bigru', 'ordered_hgb'))
    out['baseline'] = refine_domain.combine(team, d, 0.)
    for g in GRUS:
        out[g] = up(factorized_had.predict(record, models[g]), record)
    extra, extra_names = geometry.features(inputs, include_anchors=True)
    augmented = joint.augment(record, dict(x=extra[indices], names=extra_names, indices=indices))
    for config in ('full_127', 'geometry_63'):
        out[config] = up(tree.predict(augmented, models[config]), augmented)
    for g in GRUS:
        out[f'full_127+{g}'] = tree.blend(out['full_127'], out[g])
    for s in gr.SEEDS:
        out[f'geometry_63+control_{s}'] = tree.blend(out['geometry_63'], out[f'control_{s}'])
    return out, periods


def parity(source, match, view, probs, rows):
    """Clean/composite must reproduce the archived per-cell scores."""
    archived_view = 'clean' if view == 'clean' else view
    issues = []
    for g in GRUS:
        arm, seed = g.rsplit('_', 1)
        with np.load(gr.DEFAULT/f'{source}_{match}_{archived_view}_{arm}_{seed}.npz') as saved:
            if not np.allclose(saved['probability'], probs[g], atol=1e-6):
                issues.append(f'{g} probability')
    joint_rows = json.loads((ROOT/'experiments/results/joint_lgbm_development.json').read_text())['rows']
    for r in joint_rows:
        if (r['match'], r['view'], r['config'], r['recipe']) == (match, archived_view, 'full_127', 'standalone'):
            if r['native_probability_sha256'] != hashlib.sha256(np.ascontiguousarray(probs['full_127']).tobytes()).hexdigest():
                score = next(x for x in rows if x['recipe'] == 'full_127')['spell_f1']
                if abs(score-r['spell_f1']) > 1e-9:
                    issues.append('full_127 score')
    base_rows = json.loads((old.DEFAULT/f'{source}_results.json').read_text())
    for r in base_rows:
        if (r['match'], r['view'], r['arm']) == (match, archived_view, 'baseline'):
            score = next(x for x in rows if x['recipe'] == 'baseline')['spell_f1']
            if abs(score-r['spell_f1']) > 1e-9:
                issues.append(f'baseline score {score} vs {r["spell_f1"]}')
    return issues


def run(match, view):
    spec = check()
    source, path = next((s, p) for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] if p.stem == match)
    models = load(source)
    original, labels = ms.view(path)
    for replicate in (('clean',) if view == 'clean' else REPLICATES):
        dest = OUT/f'{match}_{view}_{replicate}.json'
        if dest.exists():
            continue
        started = time.monotonic()
        inputs = (tc_inputs(original) if view == 'clean'
                  else sc.corrupt(original, spec['calibration'], ce.seed(match, replicate), VIEWS[view]))
        probs, periods = probabilities(inputs, labels, source, match, models)
        rows = []
        for recipe in RECIPES:
            prediction = supervised.decode(probs[recipe], periods, float(inputs['fps']), lt.COST)
            m = supervised.score(dict(inputs, labels=labels), prediction)
            rows.append(dict(source=source, match=match, view=view, replicate=replicate, recipe=recipe,
                             spell_f1=m['spell_f1_mean'], macro_f1=m['macro_f1'],
                             d_f1=m['per_class']['D']['f1'], d_precision=m['per_class']['D']['precision'],
                             d_recall=m['per_class']['D']['recall'], n_pred_spells=m['n_pred_spells'],
                             n_gt_spells=m['n_gt_spells'],
                             spell_thresholds={str(k): v['f1'] for k, v in m['spell_f1'].items()}))
        issues = parity(source, match, 'clean' if view == 'clean' else replicate, probs, rows) \
            if view in ('clean', 'composite') else []
        result = dict(rows=rows, parity_issues=issues, seconds=time.monotonic()-started,
                      predeclared_sha256=sha256(OUT/'predeclared.json'))
        with dest.open('x') as handle:
            json.dump(result, handle, indent=1)
        print('DONE', match, view, replicate, round(result['seconds']), 'parity', issues or 'ok',
              {r['recipe']: round(r['spell_f1'], 4) for r in rows if r['recipe'] in ('baseline', 'slot_dropout_2026', 'full_127+slot_dropout_2026')},
              flush=True)


def tc_inputs(data):
    from experiments import tracking_corruption as tc
    return tc.inputs_only(data)


def mean(xs):
    return float(np.mean(xs))


def report():
    spec = check()
    rows, issues = [], []
    for source, path, view, replicate in units():
        result = json.loads((OUT/f'{path.stem}_{view}_{replicate}.json').read_text())
        if result['predeclared_sha256'] != sha256(OUT/'predeclared.json'):
            raise ValueError('Result from another declaration')
        rows.extend(result['rows'])
        issues.extend(f'{path.stem}/{view}/{replicate}: {i}' for i in result['parity_issues'])
    keys = ('spell_f1', 'macro_f1', 'd_f1', 'd_precision', 'd_recall')

    def summary(recipe, view, source=None):
        per = {}
        for s in ([source] if source else dt.SPLITS):
            per[s] = {k: mean([np.mean([r[k] for r in rows if (r['recipe'], r['view'], r['match']) == (recipe, view, p.stem)])
                               for p in dt.SPLITS[s]['eval']]) for k in keys}
        return {k: mean([v[k] for v in per.values()]) for k in keys}, per

    table = {recipe: {view: summary(recipe, view)[0] for view in VIEWS} for recipe in RECIPES}
    providers = {recipe: {view: summary(recipe, view)[1] for view in VIEWS} for recipe in RECIPES}
    retained = {recipe: {view: (table[recipe]['clean']['spell_f1']-table[recipe][view]['spell_f1'])
                         / (table[recipe]['clean']['spell_f1']-table[recipe]['composite']['spell_f1'])
                         for view in VIEWS if view not in ('clean', 'composite')} for recipe in RECIPES}
    guard = {g: table[g]['feed_like']['spell_f1']-table['baseline']['feed_like']['spell_f1'] for g in GRUS}
    lgbm = {s: table[f'full_127+slot_dropout_{s}']['feed_like']['spell_f1']-table[f'slot_dropout_{s}']['feed_like']['spell_f1']
            for s in gr.SEEDS}
    out = dict(spec_sha256=sha256(OUT/'predeclared.json'), parity_issues=issues, summary=table,
               per_provider=providers, retained_loss_fraction=retained,
               gru_minus_baseline_feed_like=guard, lgbm_blend_minus_gru_feed_like=lgbm, rows=rows)
    RESULT.write_text(json.dumps(out, indent=1))
    print('parity issues:', issues or 'none')
    print(f"{'recipe':34s}" + ''.join(f'{v:>12s}' for v in VIEWS))
    for recipe in RECIPES:
        print(f'{recipe:34s}' + ''.join(f"{table[recipe][v]['spell_f1']:12.4f}" for v in VIEWS))
    print('\nretained fraction of composite loss')
    for recipe in RECIPES:
        print(f'{recipe:34s}' + ''.join(f'{retained[recipe][v]:12.2f}' for v in retained[recipe]))
    print('\nGRU - baseline (feed_like):', {k: round(v, 4) for k, v in guard.items()})
    print('Full127+slot GRU - slot GRU (feed_like):', {k: round(v, 4) for k, v in lgbm.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'report'))
    parser.add_argument('--match')
    parser.add_argument('--view', choices=tuple(VIEWS))
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if args.action == 'freeze':
            freeze()
        elif args.action == 'run':
            run(args.match, args.view)
        else:
            report()


if __name__ == '__main__':
    main()
