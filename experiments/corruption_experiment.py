"""Fixed realistic-corruption ablation against the adopted GRU/HGB baseline.

All evaluations are on historical external development matches. Organizer
inputs are read only to freeze aggregate, label-free gap statistics. This
module has no organizer scoring or baseline-promotion command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from experiments import (current_baseline, dead_transfer, learned_temporal as lt,
                         multisource_eval, refine_domain, refine_features,
                         robust, supervised, tracking_corruption as tc)
from experiments.eval_set.protocol import paired_source_bootstrap, sha256

ROOT = lt.ROOT
DEFAULT = ROOT/'data/experiments/realistic_corruption_v1'
ARMS = {'baseline': ('baseline', 'baseline'), 'team_only': ('augmented', 'baseline'),
        'D_only': ('baseline', 'augmented'), 'both': ('augmented', 'augmented')}
VIEWS = ('clean', 'corrupt_1', 'corrupt_2')


def seed(match, view):
    return int(hashlib.sha256(f'20261026:{match}:{view}'.encode()).hexdigest()[:8], 16)


def write_json(path, value, exclusive=False):
    with path.open('x' if exclusive else 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def sources_hash():
    paths = [Path(__file__), Path(tc.__file__), Path(lt.__file__), Path(refine_domain.__file__),
             Path(refine_features.__file__), Path(multisource_eval.__file__), Path(robust.__file__),
             Path(supervised.__file__), Path(dead_transfer.__file__), Path(current_baseline.__file__),
             ROOT/'experiments/metrics.py', ROOT/'evaluation/possession_metrics.py',
             ROOT/'experiments/eval_set/protocol.py', ROOT/'experiments/eval_set/metrics.py',
             ROOT/'experiments/model_decision_policy.json', ROOT/'experiments/baseline_manifest.json']
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def freeze(out, inputs):
    out.mkdir(parents=True, exist_ok=True)
    target = out/'predeclared.json'
    if target.exists():
        check(out)
        return
    calibration = tc.profile(current_baseline.load_inputs(inputs))
    existing = json.loads((ROOT/'experiments/eval_set/manifest.json').read_text())
    files = {}
    for source, values in existing['sources'].items():
        for role in ('train', 'selection'):
            for entry in values[role]:
                path = ROOT/entry['path']
                if sha256(path) != entry['sha256']:
                    raise ValueError(f'External dataset hash mismatch: {path}')
                files[entry['path']] = entry['sha256']
    # Verify the unchanged comparison checkpoints before any new fitting.
    current_baseline.load_models()
    spec = dict(version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sources_hash(), data_sha256=files,
        organizer_input_sha256=sha256(inputs), calibration=calibration, recipe=tc.RECIPE,
        arms=ARMS, views=VIEWS, folds=lt.FOLDS,
        training='One clean and one deterministic corrupted copy, each half the training mass; same model capacities and eight GRU epochs',
        baseline='Exact packaged baseline for pooled; original clean+slot-dropout D protocol refit for each D provider holdout',
        team_training='Unchanged multisource HGB fit/4 Hz/team-swap; replaces original ball-gap copy with composite corruption',
        D_training='Unchanged learned_temporal models, normalization, 5 Hz, provider/match weights; replaces slot-dropout copy',
        evaluation='Labels/graded masks fixed from original views even after synthetic player loss; identical frame grids in all arms',
        holdout_scope='D heads only; both team models remain pooled and have seen training matches from all providers',
        ranking='Official spell F1, exact-tie macro F1; clean and corrupted reported separately; no automatic promotion',
        weighted_summary={'spell': .7, 'macro': .3, 'ranking': False},
        corruption_replicates='Two fixed seeds per match; average seeds within each match before equal-provider means',
        organizer_labels_read=False, organizer_recheck=False, historical_tests_accessed=False,
        fresh_confirmation=False, decision='Retain adopted baseline pending external results and independent confirmation')
    write_json(target, spec, exclusive=True)
    print('FROZEN', target, flush=True)


def check(out):
    spec = json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256'] != sources_hash():
        raise ValueError('Frozen experiment source changed; use a new output directory')
    return spec


def record(inputs, labels, source, match, variant):
    indices = lt.sampled_indices(inputs)
    x, names = lt.player_features(inputs)
    y = np.where(np.isin(labels[indices], supervised.LABELS), labels[indices] == 'D', -1).astype(np.int8)
    return dict(source=source, match=match, variant=variant, x=x[indices], y=y,
                period=supervised.model_periods(inputs)[indices], names=names, indices=indices,
                native_n=len(labels), fps=float(inputs['fps']))


def training_team(view, labels):
    stride = max(1, round(float(view['fps'])/multisource_eval.TRAIN_HZ))
    rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::stride]
    y = np.array([np.flatnonzero(supervised.LABELS == value)[0]
                  for value in labels[rows]], dtype=np.int8)
    x, names = refine_features.features(view)
    x = x[rows]
    return (np.concatenate((x, supervised.swap_team_features(x, names))),
            np.concatenate((y, np.where(y < 2, 1-y, y))), names)


def prepare(out):
    spec = check(out)
    for source, split in dead_transfer.SPLITS.items():
        for i, path in enumerate(split['train']):
            cache = out/f'train_{path.stem}.joblib'
            if cache.exists():
                continue
            start = time.monotonic()
            inputs, labels = multisource_eval.view(path)
            clean = record(inputs, labels, source, path.stem, 0)
            dropout = refine_domain.drop_player_slots(inputs, lt.SEED+i)
            old = record(dropout, labels, source, path.stem, 1)
            augmented = tc.corrupt(inputs, spec['calibration'], seed(path.stem, 'train'))
            new = record(augmented, labels, source, path.stem, 1)
            a, b = training_team(inputs, labels), training_team(augmented, labels)
            team = (np.concatenate((a[0], b[0])), np.concatenate((a[1], b[1])), a[2])
            joblib.dump(dict(baseline=[clean, old], augmented=[clean, new], team=team,
                             visibility={'clean': tc.visibility(inputs), 'augmented': tc.visibility(augmented)}), cache)
            print('PREPARED', path.stem, round(time.monotonic()-start, 1), flush=True)


def freeze_model(out, path, value, provenance):
    joblib.dump(value, path)
    write_json(path.with_suffix('.json'), dict(sha256=sha256(path),
               predeclared_sha256=sha256(out/'predeclared.json'), **provenance), exclusive=True)


def load_model(out, path):
    receipt = json.loads(path.with_suffix('.json').read_text())
    if receipt['sha256'] != sha256(path) or receipt['predeclared_sha256'] != sha256(out/'predeclared.json'):
        raise ValueError(f'Frozen checkpoint changed: {path}')
    return joblib.load(path)


def fit_team(out):
    check(out)
    path = out/'team_augmented.joblib'
    if path.exists():
        load_model(out, path)
        return
    cache = {}
    for source, split in dead_transfer.SPLITS.items():
        records = [joblib.load(out/f'train_{p.stem}.joblib')['team'] for p in split['train']]
        cache[source] = (np.concatenate([r[0] for r in records]),
                         np.concatenate([r[1] for r in records]), records[0][2])
    start = time.monotonic()
    model = multisource_eval.fit(tuple(cache), cache)
    freeze_model(out, path, model, dict(training='pooled augmented H/A backbone', seconds=time.monotonic()-start))
    print('FROZEN TEAM', round(time.monotonic()-start, 1), flush=True)


def fit_d(out, fold, arm):
    check(out)
    if fold == 'pooled' and arm == 'baseline':
        return  # Exact packaged checkpoint, not an approximate refit.
    sources = [s for s in dead_transfer.SPLITS if fold != f'without_{s}']
    records = [r for source in sources for p in dead_transfer.SPLITS[source]['train']
               for r in joblib.load(out/f'train_{p.stem}.joblib')[arm]]
    for kind in ('ordered_hgb', 'bigru'):
        path = out/f'{fold}_{arm}_{kind}.joblib'
        if path.exists():
            load_model(out, path)
            continue
        start = time.monotonic()
        model = lt.fit(kind, records)
        freeze_model(out, path, model, dict(fold=fold, arm=arm, sources=sources,
                     training_matches=sorted({r['match'] for r in records}), seconds=time.monotonic()-start))
        print('FROZEN D', fold, arm, kind, round(time.monotonic()-start, 1), flush=True)


def prepare_eval(out):
    spec = check(out)
    models = current_baseline.load_models()
    new_team = load_model(out, out/'team_augmented.joblib')
    for source, split in dead_transfer.SPLITS.items():
        for path in split['eval']:
            original, labels = multisource_eval.view(path)
            for view in VIEWS:
                cache = out/f'eval_{path.stem}_{view}.joblib'
                if cache.exists():
                    continue
                start = time.monotonic()
                inputs = original if view == 'clean' else tc.corrupt(original, spec['calibration'], seed(path.stem, view))
                x, names = refine_features.features(inputs)
                old = refine_features.probabilities_from_features(x, names, models['attribution'])
                new = refine_features.probabilities_from_features(x, names, new_team)
                r = record(inputs, labels, source, path.stem, view)
                joblib.dump(dict(record=r, baseline=old, augmented=new, visibility=tc.visibility(inputs)), cache)
                print('EVAL FEATURES', path.stem, view, round(time.monotonic()-start, 1), flush=True)


def evaluate(out, fold):
    check(out)
    # Refuse partial arm comparisons: all models must be frozen first.
    baseline = current_baseline.load_models()
    d_models = {}
    for arm in ('baseline', 'augmented'):
        d_models[arm] = {kind: baseline[kind] if fold == 'pooled' and arm == 'baseline'
                         else load_model(out, out/f'{fold}_{arm}_{kind}.joblib')
                         for kind in ('bigru', 'ordered_hgb')}
    result_path = out/f'{fold}_results.json'
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    for source, split in dead_transfer.SPLITS.items():
        if fold != 'pooled' and fold != f'without_{source}':
            continue
        for path in split['eval']:
            inputs, labels = multisource_eval.view(path)
            periods = supervised.model_periods(inputs)
            for view in VIEWS:
                cached = joblib.load(out/f'eval_{path.stem}_{view}.joblib')
                r = cached['record']
                probabilities = {}
                for arm, members in d_models.items():
                    probabilities[arm] = sum(lt.upsample(lt.predict_sampled(r, model), r, periods)
                                             for model in members.values())*.5
                for arm, (team, d) in ARMS.items():
                    if any(v['match'] == path.stem and v['view'] == view and v['model'] == arm for v in rows):
                        continue
                    summary, predicted = lt.summarize(inputs, labels, cached[team], probabilities[d])
                    alive = np.isin(labels, ['H', 'A'])
                    team_pred = supervised.LABELS[np.argmax(cached[team][:, :2], axis=1)]
                    row = dict(fold=fold, source=source, match=path.stem, view=view, model=arm,
                        conditional_team_accuracy=float(np.mean(team_pred[alive] == labels[alive])),
                        visibility=cached['visibility'], **summary)
                    row['raw_d_brier'] = float(np.mean((probabilities[d][np.isin(labels, supervised.LABELS)]
                         -(labels[np.isin(labels, supervised.LABELS)] == 'D'))**2))
                    rows.append(row)
                    np.savez_compressed(out/f'{fold}_{arm}_{path.stem}_{view}.npz', prediction=predicted)
                    write_json(result_path, rows)
                    print('SCORED', fold, path.stem, view, arm, round(row['spell_f1'], 5),
                          round(row['macro_f1'], 5), flush=True)


METRICS = ('spell_f1', 'macro_f1', 'd_f1', 'conditional_team_accuracy', 'raw_d_brier')


def match_means(rows, model, view, holdout):
    result = []
    for source, split in dead_transfer.SPLITS.items():
        for path in split['eval']:
            subset = [r for r in rows if r['source'] == source and r['match'] == path.stem
                      and r['model'] == model and ((r['fold'] != 'pooled') == holdout)
                      and ((r['view'] != 'clean') if view == 'corrupted' else r['view'] == 'clean')]
            expected = 2 if view == 'corrupted' else 1
            if len(subset) != expected:
                raise ValueError(f'Incomplete cohort: {source}/{path.stem}/{model}/{view}/{holdout}')
            result.append(dict(eval_source=source, match=path.stem,
                               **{k: float(np.mean([r[k] for r in subset])) for k in METRICS}))
    return result


def report(out, destination):
    spec = check(out)
    rows = [r for fold in lt.FOLDS for r in json.loads((out/f'{fold}_results.json').read_text())]
    summaries, deltas = [], []
    for holdout in (False, True):
        for view in ('clean', 'corrupted'):
            old = match_means(rows, 'baseline', view, holdout)
            for model in ARMS:
                matches = match_means(rows, model, view, holdout)
                per_source = {s: {k: float(np.mean([r[k] for r in matches if r['eval_source'] == s]))
                                  for k in METRICS} for s in dead_transfer.SPLITS}
                summary = dict(evaluation='provider_holdout_D' if holdout else 'pooled',
                               view=view, model=model, per_source=per_source,
                               **{k: float(np.mean([r[k] for r in per_source.values()])) for k in METRICS})
                summary['weighted_70_30'] = .7*summary['spell_f1']+.3*summary['macro_f1']
                summaries.append(summary)
                if model != 'baseline':
                    deltas.append(dict(evaluation=summary['evaluation'], view=view, model=model,
                                       **paired_source_bootstrap(old, matches)))
    write_json(destination, dict(spec=spec, summaries=summaries, paired_deltas=deltas, rows=rows,
        organizer_labels_read=False, organizer_check_performed=False, baseline_changed=False,
        limitations=['Historical development only; no fresh confirmation',
                     'Provider holdouts apply only to D; H/A remains pooled',
                     'Synthetic corruption is a sensitivity test, not measured broadcast performance',
                     'Singleton Sportec/Metrica development matches cannot quantify provider uncertainty',
                     'SkillCorner silver D labels have censored boundaries and original >=12-player grading']), exclusive=True)
    for r in summaries:
        print('SUMMARY', r['evaluation'], r['view'], r['model'], round(r['spell_f1'], 5), round(r['macro_f1'], 5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'prepare', 'fit-team', 'fit-d', 'prepare-eval', 'evaluate', 'report'))
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--inputs', type=Path, default=ROOT/'data/challenge/inputs.npz')
    p.add_argument('--fold', choices=lt.FOLDS, default='pooled')
    p.add_argument('--arm', choices=('baseline', 'augmented'), default='augmented')
    p.add_argument('--report', type=Path, default=ROOT/'experiments/results/realistic_corruption_development.json')
    args = p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if args.action == 'freeze': freeze(args.out, args.inputs)
        elif args.action == 'prepare': prepare(args.out)
        elif args.action == 'fit-team': fit_team(args.out)
        elif args.action == 'fit-d': fit_d(args.out, args.fold, args.arm)
        elif args.action == 'prepare-eval': prepare_eval(args.out)
        elif args.action == 'evaluate': evaluate(args.out, args.fold)
        elif args.action == 'report': report(args.out, args.report)


if __name__ == '__main__':
    main()
