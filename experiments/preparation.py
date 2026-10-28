"""Reusable, hash-verified external split plans and unnormalized feature caches.

No fitting, scoring, organizer input discovery, or changes to frozen studies.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'experiments/eval_set/manifest.json'
VIEWS = ('clean', 'legacy', 'feed_like', 'composite')
# Include transitive feature/view dependencies, not just this wrapper. Conservative
# invalidation is preferable to silently restoring features from older code.
SOURCES = ('preparation.py', 'supervised.py', 'refine_features.py',
           'refine_domain.py', 'learned_temporal.py', 'multisource_eval.py',
           'dead_transfer.py', 'robust.py', 'stress.py', 'tracking_corruption.py',
           'stress_components.py', 'eval_set/protocol.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def split_plan(manifest_path=MANIFEST, fold='pooled'):
    """Resolve existing whole-match folds without reading any tracking arrays."""
    from experiments.eval_set.protocol import validate_manifest
    manifest = json.loads(Path(manifest_path).read_text())
    validate_manifest(manifest)
    folds = [f for f in manifest['folds'] if f['name'] == fold]
    if len(folds) != 1:
        raise ValueError(f'Expected one manifest fold named {fold!r}')
    spec = folds[0]
    if any(len(spec[k]) != len(set(spec[k])) for k in ('train_sources', 'selection_sources')):
        raise ValueError('Duplicate providers in fold')
    rows = {}
    matches, paths = set(), set()
    for source, values in manifest['sources'].items():
        for role in ('train', 'selection'):
            for item in values[role]:
                path = Path(item['path'])
                if (path.is_absolute() or '..' in path.parts or
                        len(path.parts) < 2 or path.parts[0] != 'data' or
                        path.suffix != '.npz' or path.stem != item['match']):
                    raise ValueError('Expected data-relative NPZ named for its match')
                if item['match'] in matches or str(path) in paths:
                    raise ValueError('Duplicate match or file across source/role')
                if item['match'] in manifest['historical_tests_excluded']:
                    raise ValueError('Historical test is excluded from development')
                if len(item['sha256']) != 64 or any(c not in '0123456789abcdef' for c in item['sha256']):
                    raise ValueError('Expected a SHA-256 input seal')
                matches.add(item['match'])
                paths.add(str(path))
                rows.setdefault((source, role), []).append(dict(item, source=source, role=role))
    def select(role, sources):
        return [r for s in sources for r in rows.get((s, role), [])]
    train = select('train', spec['train_sources'])
    selection = select('selection', spec['selection_sources'])
    if not train or not selection:
        raise ValueError('Empty training or selection cohort')
    if fold != 'pooled' and set(spec['train_sources']) & set(spec['selection_sources']):
        raise ValueError('Provider holdout includes selection provider in training')
    return dict(schema=1, manifest_sha256=sha256(manifest_path), fold=fold,
                train=train, selection=selection,
                interpretation=spec['interpretation'],
                normalization='Fit on these training records only, separately for every learned component',
                evidence='Historical development; splitting does not create fresh confirmation')


def data_path(item, data_dir):
    """Relocate the manifest's data/ prefix; reject paths escaping the root."""
    root = Path(data_dir).resolve()
    path = (root / Path(item['path']).relative_to('data')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Input escapes data directory')
    return path


def recipe(view='clean', seed=2026, calibration=None):
    if view not in VIEWS:
        raise ValueError(f'Unknown view: {view}')
    if view in ('feed_like', 'composite') and calibration is None:
        raise ValueError('Stress views require explicitly frozen calibration')
    if view in ('clean', 'legacy') and calibration is not None:
        raise ValueError('Calibration is unused for this view')
    return dict(schema=1, view=view, seed=None if view == 'clean' else int(seed), calibration=calibration,
                features='joint_refine_plus_player_v1', rate_hz=5,
                source_sha256={p: sha256(ROOT/'experiments'/p) for p in SOURCES},
                versions={p: importlib.metadata.version(p) for p in
                          ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')})


def build_record(inputs, labels, source, match):
    """Same joint/dead feature schema as live_gap_experiment.feature_view.

    Features receive tracking fields only. Targets stay on the native grid until
    block-local 5 Hz sampling; no unknown rows are dropped or imputed.
    """
    from experiments import learned_temporal as lt, refine_features, supervised
    from experiments.tracking_corruption import inputs_only
    inputs = inputs_only(inputs)
    labels = np.asarray(labels).astype(str)
    if labels.shape != (len(inputs['ball']),) or not len(labels) or not np.isin(labels, ['H', 'A', 'D', 'X']).all():
        raise ValueError('Expected nonempty aligned H/A/D/X targets')
    team, names = refine_features.features(inputs)
    player, player_names = lt.player_features(inputs)
    indices = lt.sampled_indices(inputs)
    y = np.full(len(labels), -1, np.int8)
    for i, label in enumerate(supervised.LABELS):
        y[labels == label] = i
    return dict(x=np.column_stack((team[indices], player[indices])),
                y=y[indices], names=np.asarray(names+['D_'+n for n in player_names]),
                indices=indices, period=supervised.model_periods(inputs)[indices],
                native_n=np.asarray(len(labels)), fps=np.asarray(float(inputs['fps'])),
                source=np.asarray(source), match=np.asarray(match))


def load_record(directory, expected_identity=None):
    """Verify receipts before loading an array cache; never unpickle a cache."""
    directory = Path(directory)
    receipt = json.loads((directory/'receipt.json').read_text())
    identity = receipt['identity']
    if fingerprint(identity) != directory.name or (expected_identity is not None and identity != expected_identity):
        raise ValueError('Cache identity mismatch')
    if sha256(directory/'record.npz') != receipt['record_sha256']:
        raise ValueError('Cache payload checksum mismatch')
    with np.load(directory/'record.npz', allow_pickle=False) as stored:
        record = {k: stored[k] for k in stored.files}
    record['names'] = record['names'].tolist()
    record['source'], record['match'] = str(record['source']), str(record['match'])
    return record


def prepare_match(item, data_dir, cache_dir, config):
    """Cache one match/view independently of fold; no fitted state is cached."""
    from experiments import multisource_eval as ms, robust, stress_components as sc
    from experiments.tracking_corruption import inputs_only
    path = data_path(item, data_dir)
    if sha256(path) != item['sha256']:
        raise ValueError(f'Input checksum mismatch: {path}')
    # Role/fold are intentionally absent: raw features may be shared, fitted
    # normalization and estimators must not be shared across held-out providers.
    identity = dict(source=item['source'], match=item['match'], group_id=item['group_id'],
                    input_sha256=item['sha256'], recipe=config)
    cache_dir = Path(cache_dir)
    destination = cache_dir/fingerprint(identity)
    if destination.exists():
        load_record(destination, identity)
        return destination
    inputs, labels = ms.view(path)  # Native SkillCorner grading BEFORE corruption.
    native = inputs_only(inputs)
    seed = int(fingerprint(dict(match=item['match'], seed=config['seed']))[:8], 16)
    if config['view'] == 'clean':
        viewed = native
    elif config['view'] == 'legacy':
        viewed = robust.masked_inputs(native, seed)
    else:
        keep = sc.COMPONENTS if config['view'] == 'composite' else tuple(
            c for c in sc.COMPONENTS if c not in ('sampling', 'regional'))
        viewed = sc.corrupt(native, config['calibration'], seed, keep)
    record = build_record(viewed, labels, item['source'], item['match'])
    record.update(native_labels=np.asarray(labels), native_period=native['period'],
                  native_timestamps=native['timestamps'],
                  native_ball_observed=np.isfinite(native['ball']).all(axis=1),
                  native_player_observed=np.isfinite(native['players']).all(axis=2))
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.prepare-', dir=cache_dir) as tmp:
        staging = Path(tmp)/'entry'
        staging.mkdir()
        np.savez_compressed(staging/'record.npz', **record)
        receipt = dict(identity=identity, record_sha256=sha256(staging/'record.npz'))
        (staging/'receipt.json').write_text(canonical(receipt)+'\n')
        try:
            staging.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            load_record(destination, identity)  # Concurrent identical preparation.
    load_record(destination, identity)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'prepare'))
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    parser.add_argument('--fold', default='pooled')
    parser.add_argument('--data-dir', type=Path, default=Path(os.environ.get('MTAH_DATA_DIR', ROOT/'data')))
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--view', choices=VIEWS, default='clean')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--calibration', type=Path, help='Frozen calibration object as JSON')
    args = parser.parse_args()
    plan = split_plan(args.manifest, args.fold)
    if args.action == 'prepare':
        calibration = json.loads(args.calibration.read_text()) if args.calibration else None
        config = recipe(args.view, args.seed, calibration)
        cache = args.cache_dir or args.data_dir/'experiments/prepared_v1'
        for role in ('train', 'selection'):
            for item in plan[role]:
                directory = prepare_match(item, args.data_dir, cache, config)
                item['cache_key'] = directory.name
        plan['recipe'] = config
        plans = cache/'plans'
        plans.mkdir(parents=True, exist_ok=True)
        destination = plans/(fingerprint(plan)+'.json')
        if destination.exists():
            if json.loads(destination.read_text()) != plan:
                raise ValueError('Prepared plan checksum mismatch')
        else:
            with destination.open('x') as handle:
                handle.write(canonical(plan)+'\n')
    print(json.dumps(plan, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
