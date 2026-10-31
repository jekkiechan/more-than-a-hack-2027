"""Pooled deployment fits from the thirteen external TRAIN matches only.

This helper never loads competition inputs, organizer labels, or evaluation
records. The caller must freeze ``declaration()`` under ``pooled_fit`` in its
new predeclared.json before fitting. Historical study sources stay immutable.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import (
    corruption_experiment as ce, dead_transfer as dt, factorized_had,
    gru_robustness_experiment as gr, joint_lgbm as tree,
    joint_lgbm_experiment as joint, learned_temporal as lt,
)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
TREE_CONFIGS = ('geometry_63', 'full_127')
GRU_ARMS = ('control', 'slot_dropout')
SEED = 2026
TREE_THREADS = 2


def fingerprints():
    return dict(joint.fingerprints(), **{
        str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))})


def declaration():
    """Verify reusable TRAIN inputs/caches and return JSON-safe frozen metadata."""
    previous = joint.check(joint.DEFAULT)
    gr.check(gr.DEFAULT)
    training = {s: [p.stem for p in split['train']] for s, split in dt.SPLITS.items()}
    excluded = {s: [p.stem for p in split['eval']] for s, split in dt.SPLITS.items()}
    train_ids = [m for ids in training.values() for m in ids]
    eval_ids = [m for ids in excluded.values() for m in ids]
    if (len(train_ids) != 13 or len(set(train_ids)) != 13
            or len(eval_ids) != 5 or len(set(eval_ids)) != 5
            or set(train_ids) & set(eval_ids)):
        raise ValueError('Expected thirteen unique TRAIN and five unique disjoint evaluation matches')
    dependencies = {}
    for path in (joint.DEFAULT/'predeclared.json', gr.DEFAULT/'predeclared.json'):
        dependencies[str(path.relative_to(ROOT))] = sha256(path)
    for split in dt.SPLITS.values():
        for path in split['train']:
            key = str(path.relative_to(ROOT))
            if sha256(path) != previous['data_sha256'][key]:
                raise ValueError('External training input changed')
            dependencies[key] = previous['data_sha256'][key]
            for directory, prefix in ((gr.DEFAULT, 'train_'), (joint.DEFAULT, 'geometry_')):
                cache = directory/f'{prefix}{path.stem}.joblib'
                receipt_path = cache.with_suffix('.json')
                receipt = json.loads(receipt_path.read_text())
                digest = sha256(cache)
                if digest != receipt['sha256'] or receipt['predeclared_sha256'] != sha256(directory/'predeclared.json'):
                    raise ValueError('Reusable training cache changed')
                if directory == gr.DEFAULT and digest != previous['training_caches_sha256'][str(cache.relative_to(ROOT))]:
                    raise ValueError('Joint study used a different GRU cache')
                dependencies[str(cache.relative_to(ROOT))] = digest
                dependencies[str(receipt_path.relative_to(ROOT))] = sha256(receipt_path)
    sampled = previous['sampled_training_matches']
    if {r['match'] for r in sampled} != set(train_ids):
        raise ValueError('Original sampling manifest is incomplete')
    parameters = dict(tree.COMMON, n_jobs=TREE_THREADS)
    return dict(
        source_sha256=fingerprints(), dependencies_sha256=dependencies,
        training_matches=training, excluded_evaluation_matches=excluded,
        training_rows=sum(r['training_rows_four_views'] for r in sampled),
        original_known_5hz_rows_four_views=4*sum(r['known5Hz_frames'] for r in sampled),
        tree_configs={c: tree.CONFIGS[c] for c in TREE_CONFIGS},
        tree_parameters=parameters,
        tree_feature_counts={f: previous['feature_counts'][f] for f in ('geometry', 'full')},
        tree_sampling=previous['sampling'],
        tree_weighting='Equal provider/match/view mass; same four coherent original/swapped views and within-run Voronoi weights as joint_lgbm_v1.',
        tree_execution='Two CPU threads; deterministic=True and force_col_wise=True retained. Thread count is an execution change, not a new statistical arm.',
        gru_arms=list(GRU_ARMS), gru_seed=SEED, gru_player_only=True,
        gru_threads=1, gru_epochs=lt.EPOCHS,
        gru_training='Unchanged factorized_had.fit and paired_records; all thirteen TRAIN matches; training-only normalization; fixed seed2026. Cache arrays mmap read-only; normalized arrays follow the original implementation.',
        organizer_labels_read=False, selection_matches_used_for_fitting=False,
    )


def require_open(out):
    # Import lazily: the parent runner imports this helper while declaring fits.
    from experiments.temporal_competition_check import require_open as guard
    guard(Path(out))


def check(out):
    out = Path(out)
    spec = json.loads((out/'predeclared.json').read_text())
    if spec.get('pooled_fit') != declaration():
        raise ValueError('Pooled fit metadata, source, or verified TRAIN dependencies changed')
    return spec['pooled_fit']


def tree_path(out, config):
    if config not in TREE_CONFIGS:
        raise ValueError('Undeclared tree configuration')
    return Path(out)/f'tree_{config}.joblib'


def gru_path(out, arm):
    if arm not in GRU_ARMS:
        raise ValueError('Undeclared GRU arm')
    return Path(out)/f'gru_{arm}_{SEED}.joblib'


def _training_ids(meta):
    return [m for ids in meta['training_matches'].values() for m in ids]


def matrix(out, family):
    """Stream one match at a time into a checked, disk-backed pooled matrix."""
    out = Path(out)
    require_open(out)
    if family not in ('geometry', 'full'):
        raise ValueError('Undeclared pooled feature family')
    meta = check(out)
    previous = joint.check(joint.DEFAULT)
    manifest = previous['feature_manifest'][family]
    prefix = out/f'matrix_pooled_{family}'
    xp, yp, rp = (prefix.with_suffix(s) for s in ('.npy', '.npz', '.json'))
    with prefix.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if rp.exists():
            receipt = json.loads(rp.read_text())
            if (receipt['predeclared_sha256'] != sha256(out/'predeclared.json')
                    or receipt['x_sha256'] != sha256(xp)
                    or receipt['y_weight_sha256'] != sha256(yp)):
                raise ValueError('Pooled matrix changed')
            with np.load(yp) as values:
                y, weight = values['y'], values['weight']
            return np.load(xp, mmap_mode='r'), y, weight, manifest
        if xp.exists() or yp.exists():
            raise ValueError('Unsealed pooled matrix exists; inspect interrupted preparation')
        n = meta['training_rows']
        x = np.lib.format.open_memmap(xp, mode='w+', dtype=np.float32,
                                     shape=(n, len(manifest)))
        y, weight = np.empty(n, np.int8), np.empty(n, float)
        at = 0
        counts, source_mass, class_mass = {}, {}, np.zeros(3)
        # A nonexistent excluded provider includes only SPLITS[*]['train']; the
        # original immutable iterator still verifies every cache and view.
        for record in joint.training_records(joint.DEFAULT, 'pooled'):
            source, match = record['source'], record['match']
            if match not in meta['training_matches'].get(source, []):
                raise ValueError('Unexpected training match')
            counts[match] = counts.get(match, 0)+1
            rows, mass = tree.sampling(record['y'], record['period'])
            full, actual = tree.training_design(record, family)
            if actual != manifest:
                raise ValueError('Pooled feature schema differs from frozen study')
            stop = at+len(rows)
            x[at:stop], y[at:stop] = full[rows], record['y'][rows]
            weight[at:stop] = mass/(max(1, np.sum(record['y'] >= 0))
                                   * len(meta['training_matches'][source])*4)
            source_mass[source] = source_mass.get(source, 0.)+float(weight[at:stop].sum())
            for k in range(3):
                class_mass[k] += weight[at:stop][y[at:stop] == k].sum()
            at = stop
        if at != n or counts != dict.fromkeys(_training_ids(meta), 4):
            raise ValueError('Pooled cohort or four-view row count differs')
        if not all(np.isclose(v, 1.) for v in source_mass.values()):
            raise ValueError('Pooled provider mass is not equal')
        x.flush()
        del x
        weight /= weight.mean()
        np.savez_compressed(yp, y=y, weight=weight)
        require_open(out)
        ce.write_json(rp, dict(
            x_sha256=sha256(xp), y_weight_sha256=sha256(yp),
            predeclared_sha256=sha256(out/'predeclared.json'),
            rows=n, features=len(manifest), training_matches=_training_ids(meta),
            records_by_match=counts, provider_mass_before_mean_normalization=source_mass,
            class_mass_before_mean_normalization=class_mass.tolist()), exclusive=True)
        return np.load(xp, mmap_mode='r'), y, weight, manifest


def fit_tree(out, config):
    from lightgbm import LGBMClassifier, record_evaluation, log_evaluation
    out = Path(out)
    require_open(out)
    path = tree_path(out, config)
    meta = check(out)
    if path.exists():
        ce.load_model(out, path)
        return path
    if (out/'checkpoints_frozen.json').exists():
        raise ValueError('Pooled fitting is closed')
    start = time.monotonic()
    settings = dict(meta['tree_configs'][config])
    family = settings.pop('family')
    x, y, weight, manifest = matrix(out, family)
    if not np.array_equal(np.unique(y), [0, 1, 2]):
        raise ValueError('Pooled fit requires all three classes')
    estimator = LGBMClassifier(**meta['tree_parameters'], **settings)
    history = {}
    estimator.fit(x, y, sample_weight=weight, eval_set=[(x, y)],
                  eval_sample_weight=[weight], eval_names=['training'],
                  eval_metric='multi_logloss', callbacks=[record_evaluation(history), log_evaluation(period=64)])
    fitted = dict(kind='joint_multiclass_lgbm', config=config, family=family,
                  estimator=estimator, feature_manifest=manifest,
                  training_rows=len(y), history=history, seed=SEED)
    require_open(out)
    ce.freeze_model(out, path, fitted, dict(
        deployment='pooled_external_train_only', training_matches=_training_ids(meta),
        excluded_evaluation_matches=meta['excluded_evaluation_matches'],
        config=config, family=family, seed=SEED, cpu_threads=TREE_THREADS,
        seconds=time.monotonic()-start, training_rows=len(y), n_features=len(manifest),
        boosting_rounds=estimator.n_estimators_, class_trees=estimator.booster_.num_trees(),
        training_loss=history['training']['multi_logloss'],
        matrix_receipt_sha256=sha256(out/f'matrix_pooled_{family}.json')))
    print('POOLED TREE FROZEN', config, round(time.monotonic()-start, 1), flush=True)
    return path


def fit_gru(out, arm):
    out = Path(out)
    require_open(out)
    path = gru_path(out, arm)
    meta = check(out)
    if path.exists():
        ce.load_model(out, path)
        return path
    if (out/'checkpoints_frozen.json').exists():
        raise ValueError('Pooled fitting is closed')
    start = time.monotonic()
    records = []
    for source, split in dt.SPLITS.items():
        for native in split['train']:
            cache = joblib.load(gr.DEFAULT/f'train_{native.stem}.joblib', mmap_mode='r')
            paired = gr.paired_records(cache, native.stem, arm)
            if any(r['source'] != source or r['match'] != native.stem for r in paired):
                raise ValueError('GRU cache cohort differs')
            records.extend(paired)
    if len(records) != 52:
        raise ValueError('Expected four paired records per TRAIN match')
    fitted = factorized_had.fit(records, player_only=True, seed=SEED)
    require_open(out)
    ce.freeze_model(out, path, fitted, dict(
        deployment='pooled_external_train_only', training_matches=_training_ids(meta),
        excluded_evaluation_matches=meta['excluded_evaluation_matches'],
        arm=arm, seed=SEED, player_only=True, cpu_threads=1,
        seconds=time.monotonic()-start, parameters=fitted['parameters'],
        optimizer_updates=fitted['optimizer_updates'],
        windows_per_epoch=fitted['windows_per_epoch'], training_loss=fitted['history']))
    print('POOLED GRU FROZEN', arm, round(time.monotonic()-start, 1), flush=True)
    return path
