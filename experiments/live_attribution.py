"""Bounded live-only H/A training ablation; frozen GRU/HGB D and decoder.

Only the existing external train/development splits are opened. No organizer
inputs, labels, historical tests, or competition gates are accessed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from experiments import (current_baseline, dead_transfer, error_analysis,
                         learned_temporal as lt, multisource_eval,
                         refine_features, refine_weighted, robust, supervised)
from experiments.eval_set.protocol import paired_source_bootstrap, sha256
from experiments.temporal_report import source_mean

ROOT = lt.ROOT
CANDIDATES = ('baseline', 'live_only', 'live_spell_weighted')
CONDITIONS = ('clean', 'ball_masked')
MASK_SEED = 112026
SPEC = {
    'hypothesis': 'Separate conditional team learning from D; test one short-spell loss weighting ablation',
    'candidates': list(CANDIDATES),
    'training': 'Existing pooled 13 external training matches, same 4-Hz sampled graded rows as baseline, then filter H/A',
    'augmentation': 'Unchanged clean + Markov ball gap copies, team swap for both; seed 2026 + match index within source',
    'capacity': dict(max_iter=260, max_leaf_nodes=63, learning_rate=.08,
                     min_samples_leaf=35, l2_regularization=1,
                     early_stopping=False, random_state=17),
    'weighting': 'live_only: uniform within provider; live_spell_weighted: clip(sqrt(5 / training spell seconds), .25, 4), then unit mass per provider',
    'controls': 'Identical packaged pooled GRU/HGB P(D), 50/50; unchanged physical features, team-swap inference, Viterbi cost 3',
    'conditions': 'Clean main view; independently seeded existing Markov ball mask supplementary; unchanged player-only D in both',
    'selection': 'Equal-provider mean official spell F1 on five whole development matches; macro F1 exact-tie break; no baseline promotion',
    'auxiliary_weights': dict(spell_f1=.7, macro_f1=.3),
    'scope': 'Pooled training with whole-match development separation; no provider holdout, no fresh confirmation',
    'organizer_access': False,
    'seed_scope': 'One fixed HGB seed; no seed-robustness claim or adaptive search',
    'mask_seed': MASK_SEED,
}


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def freeze(out):
    files = [Path(__file__), current_baseline.MANIFEST,
             ROOT/'experiments/model_decision_policy.json',
             ROOT/'evaluation/possession_metrics.py']
    files += [Path(module.__file__) for module in
              (current_baseline, dead_transfer, error_analysis, lt, multisource_eval,
               refine_features, refine_weighted, robust, supervised)]
    files += [p for split in dead_transfer.SPLITS.values() for paths in split.values() for p in paths]
    manifest = json.loads(current_baseline.MANIFEST.read_text())
    files += [ROOT/manifest['model_directory']/name for name in manifest['models_sha256']]
    spec = dict(SPEC, inputs_and_source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in files},
                splits={s: {role: [p.stem for p in paths] for role, paths in split.items()}
                        for s, split in dead_transfer.SPLITS.items()},
                versions={name: importlib.metadata.version(name)
                          for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'joblib')})
    path = out/'predeclared.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(key) != value for key, value in spec.items()):
            raise ValueError('Frozen inputs/source/spec changed; use a new output directory')
    else:
        write_new(path, dict(spec, created_utc=datetime.now(timezone.utc).isoformat()))
    return spec


def training_metadata(inputs, labels):
    """Keep baseline sampling/augmentation order; weights use training targets only."""
    stride = max(1, round(float(inputs['fps'])/multisource_eval.TRAIN_HZ))
    rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::stride]
    target = np.array([np.flatnonzero(supervised.LABELS == v)[0] for v in labels[rows]], dtype=np.int8)
    weights, _ = refine_weighted.spell_weights(dict(inputs, labels=labels), .5)
    y = np.concatenate((target, np.where(target < 2, 1-target, target)) * 2)
    w = np.tile(weights[rows], 4)
    return rows, y, w


def prepare_source(source, out):
    path = out/f'{source}_training.joblib'
    if path.exists():
        return joblib.load(path)
    xs, ys, ws, names = [], [], [], None
    started = time.monotonic()
    for i, match in enumerate(dead_transfer.SPLITS[source]['train']):
        inputs, labels = multisource_eval.view(match)
        rows, y, w = training_metadata(inputs, labels)
        pieces = []
        for variant in (inputs, robust.masked_inputs(inputs, 2026+i)):
            x, schema = refine_features.features(variant)
            if names is not None and schema != names:
                raise ValueError('Training feature schemas differ')
            names = schema
            pieces.extend((x[rows], supervised.swap_team_features(x[rows], names)))
        x = np.concatenate(pieces)
        live = y < 2
        xs.append(x[live])
        ys.append(y[live])
        ws.append(w[live])
        print(f'FEATURES {source} {match.stem} live_augmented_rows={live.sum()}', flush=True)
    result = dict(x=np.concatenate(xs), y=np.concatenate(ys), w=np.concatenate(ws), names=names)
    joblib.dump(result, path)
    print(f'PREPARED {source} seconds={time.monotonic()-started:.1f}', flush=True)
    return result


def pooled_arrays(cache, weighted):
    xs, ys, ws, names = [], [], [], None
    for part in cache.values():
        if names is not None and names != part['names']:
            raise ValueError('Provider schemas differ')
        names = part['names']
        xs.append(part['x'])
        ys.append(part['y'])
        w = part['w'] if weighted else np.ones(len(part['y']))
        ws.append(w/w.sum())
    weights = np.concatenate(ws)
    return np.concatenate(xs), np.concatenate(ys), weights/weights.mean(), names


def prepare_cached_source(source, out):
    # Workers write independent provider caches; large matrices stay off IPC.
    prepare_source(source, out)


def train(out):
    needed = [name for name in CANDIDATES[1:] if not (out/f'{name}_frozen.json').exists()]
    if needed:
        with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs = [pool.submit(prepare_cached_source, source, out) for source in dead_transfer.SPLITS]
            for job in jobs:
                job.result()
        cache = {source: joblib.load(out/f'{source}_training.joblib') for source in dead_transfer.SPLITS}
        for name in needed:
            path = out/f'{name}.joblib'
            if path.exists():
                raise ValueError('Unfrozen model found; inspect interrupted run')
            x, y, weights, names = pooled_arrays(cache, name == 'live_spell_weighted')
            if set(np.unique(y)) != {0, 1}:
                raise ValueError('Binary team targets must be H/A only')
            started = time.monotonic()
            estimator = HistGradientBoostingClassifier(**SPEC['capacity'])
            with threadpool_limits(limits=1):
                estimator.fit(x, y, sample_weight=weights)
            model = dict(estimator=estimator, feature_names=names, classes=estimator.classes_.tolist())
            joblib.dump(model, path)
            write_new(out/f'{name}_frozen.json', dict(model_sha256=sha256(path),
                      spec_sha256=sha256(out/'predeclared.json'), seconds=time.monotonic()-started,
                      training_rows=len(y), training_classes=model['classes']))
            print(f'FROZEN {name} seconds={time.monotonic()-started:.1f}', flush=True)
            del x, y, weights, estimator, model
    models = {}
    for name in CANDIDATES[1:]:
        path = out/f'{name}.joblib'
        receipt = json.loads((out/f'{name}_frozen.json').read_text())
        if receipt['model_sha256'] != sha256(path) or receipt['spec_sha256'] != sha256(out/'predeclared.json'):
            raise ValueError('Frozen model/spec hash mismatch')
        models[name] = joblib.load(path)
    return models


def summarize(inputs, labels, team, dead):
    summary, predicted = lt.summarize(inputs, labels, team, dead)
    data = dict(inputs, labels=labels)
    diagnostics = error_analysis.analyze(data, predicted)
    live = np.isin(labels, ['H', 'A'])
    summary.update(team_accuracy_live=float(np.mean(supervised.LABELS[np.argmax(team[live, :2], axis=1)] == labels[live])),
                   short_spells=diagnostics['by_gt_duration'],
                   weighted_70_30=.7*summary['spell_f1']+.3*summary['macro_f1'])
    return summary, predicted


def make_summary(rows):
    expected = {(condition, name, source, path.stem) for condition in CONDITIONS for name in CANDIDATES
                for source, split in dead_transfer.SPLITS.items() for path in split['eval']}
    observed = {(r['condition'], r['model'], r['source'], r['match']) for r in rows}
    if observed != expected or len(rows) != len(expected):
        raise ValueError('Incomplete or duplicate comparison')
    summaries, deltas = [], []
    for condition in CONDITIONS:
        base = [dict(r, eval_source=r['source']) for r in rows
                if r['condition'] == condition and r['model'] == 'baseline']
        for name in CANDIDATES:
            subset = [r for r in rows if r['condition'] == condition and r['model'] == name]
            summaries.append(dict(condition=condition, model=name,
                **{key: source_mean(subset, key) for key in
                   ('spell_f1', 'macro_f1', 'd_f1', 'team_accuracy_live', 'weighted_70_30')},
                per_source={s: {k: float(np.mean([r[k] for r in subset if r['source'] == s]))
                                for k in ('spell_f1', 'macro_f1', 'd_f1', 'team_accuracy_live')}
                            for s in dead_transfer.SPLITS}))
            if name != 'baseline':
                for metric in ('spell_f1', 'macro_f1'):
                    deltas.append(dict(condition=condition, model=name,
                        **paired_source_bootstrap(base, [dict(r, eval_source=r['source']) for r in subset], metric)))
    return summaries, deltas


def run(args):
    if args.report.exists():
        raise ValueError('Report already exists; do not overwrite frozen results')
    args.out.mkdir(parents=True, exist_ok=True)
    spec = freeze(args.out)
    baseline = current_baseline.load_models()
    models = dict(baseline=baseline['attribution'], **train(args.out))
    receipt = args.out/'evaluation_started.json'
    if not receipt.exists():
        write_new(receipt, dict(created_utc=datetime.now(timezone.utc).isoformat(),
                               spec_sha256=sha256(args.out/'predeclared.json'),
                               models={name: sha256(args.out/f'{name}.joblib') for name in CANDIDATES[1:]}))
    rows = []
    for source, split in dead_transfer.SPLITS.items():
        for path in split['eval']:
            inputs, labels = multisource_eval.view(path)
            dead = current_baseline.dead_probability(inputs, baseline)
            dead_path = args.out/f'{path.stem}_dead.npy'
            np.save(dead_path, dead, allow_pickle=False)
            for condition in CONDITIONS:
                view = inputs if condition == 'clean' else robust.masked_inputs(inputs, MASK_SEED)
                x, names = refine_features.features(view)
                for name, model in models.items():
                    team = refine_features.probabilities_from_features(x, names, model)
                    summary, predicted = summarize(view, labels, team, dead)
                    prediction = args.out/f'{path.stem}_{condition}_{name}.npz'
                    np.savez_compressed(prediction, team=team, prediction=predicted)
                    row = dict(source=source, match=path.stem, condition=condition, model=name,
                               input_sha256=sha256(path), prediction_sha256=sha256(prediction),
                               shared_dead_sha256=sha256(dead_path),
                               ball_missing_fraction=robust.missing_fraction(view), **summary)
                    rows.append(row)
                    (args.out/'progress.json').write_text(json.dumps(rows, indent=2)+'\n')
                    print(json.dumps({k: row[k] for k in ('source', 'match', 'condition', 'model', 'spell_f1', 'macro_f1')}), flush=True)
    summaries, deltas = make_summary(rows)
    result = dict(spec=spec, summaries=summaries, paired_deltas=deltas, rows=rows,
                  predeclared_sha256=sha256(args.out/'predeclared.json'),
                  model_receipts={name: json.loads((args.out/f'{name}_frozen.json').read_text()) for name in CANDIDATES[1:]},
                  organizer_check_performed=False, baseline_changed=False)
    write_new(args.report, result)
    for row in summaries:
        print('SUMMARY', row['condition'], row['model'], row['spell_f1'], row['macro_f1'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/experiments/live_attribution')
    parser.add_argument('--report', type=Path, default=ROOT/'experiments/results/live_attribution_development.json')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
