"""Frozen source/match feature preparation for the objective-alignment study.

This module is deliberately separate from fitting and scoring so its source
hash can be frozen before parallel cache preparation starts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import (corruption_experiment as ce,
                         dead_transfer as dt,
                         gru_robustness_experiment as gr,
                         learned_temporal as lt,
                         light_corruption as lc,
                         live_gap_experiment as old,
                         multisource_eval as ms,
                         refine_domain,
                         robust,
                         supervised,
                         eval_set)
from experiments.eval_set.protocol import sha256


ROOT = lt.ROOT
DEFAULT = ROOT / 'data/experiments/objective_alignment_v1'
PREPARATION_SOURCE_PATHS = (
    Path(__file__), Path(gr.__file__), Path(old.__file__), Path(lt.__file__),
    Path(lc.__file__), Path(robust.__file__), Path(refine_domain.__file__),
    Path(ms.__file__), Path(supervised.__file__), Path(ce.__file__),
    Path(eval_set.__file__), ROOT / 'experiments/eval_set/protocol.py',
    ROOT / 'experiments/eval_set/manifest.json',
)
ALLOWED_HISTORICAL_SOURCE_DIFFERENCES = {
    'experiments/current_baseline.py',
    'experiments/baseline_manifest.json',
    'experiments/model_decision_policy.json',
}
VIEWS = ('clean', 'gap', 'dropout_legacy', 'dropout_gap')


def preparation_fingerprint() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in PREPARATION_SOURCE_PATHS}


def _reconcile_prior_source() -> dict:
    prior = json.loads((gr.DEFAULT / 'predeclared.json').read_text())
    current = gr.fingerprints()
    old_hashes = prior['source_sha256']
    changed = {key for key in set(old_hashes) | set(current)
               if old_hashes.get(key) != current.get(key)}
    if changed != ALLOWED_HISTORICAL_SOURCE_DIFFERENCES:
        raise ValueError(f'Unexpected historical source differences: {sorted(changed)}')
    return dict(
        prior_declaration_sha256=sha256(gr.DEFAULT / 'predeclared.json'),
        prior_source_sha256=old_hashes,
        current_source_sha256=current,
        allowed_differences={key: dict(archived=old_hashes[key], current=current[key])
                             for key in sorted(changed)},
        explanation=('The prior runner snapshot bound the former working baseline. '
                     'Only the active baseline manifest, decision policy, and '
                     'baseline dispatcher changed after its declaration; all '
                     'feature extraction, training-record assembly, and objective '
                     'training sources remain byte-identical.'),
    )


def _data_hashes() -> dict[str, str]:
    manifest = json.loads((ROOT / 'experiments/eval_set/manifest.json').read_text())
    result = {}
    for provider in manifest['sources'].values():
        for role in ('train', 'selection'):
            for entry in provider[role]:
                path = ROOT / entry['path']
                actual = sha256(path)
                if actual != entry['sha256']:
                    raise ValueError(f'External data checksum differs: {path}')
                result[entry['path']] = actual
    return result


def freeze_preparation(out: Path = DEFAULT) -> dict:
    """Bind preprocessing code and inputs before any feature work begins."""
    out.mkdir(parents=True, exist_ok=True)
    target = out / 'preparation_declaration.json'
    if target.exists():
        return check_preparation(out)
    manifest = json.loads((ROOT / 'experiments/eval_set/manifest.json').read_text())
    calibration_path = ROOT / 'experiments/results/realistic_corruption_development.json'
    calibration_report = json.loads(calibration_path.read_text())
    reconciliation = _reconcile_prior_source()
    data_sha256 = _data_hashes()
    spec = dict(
        version=1,
        created_utc=datetime.now(timezone.utc).isoformat(),
        study='objective_alignment_v1',
        source_sha256=preparation_fingerprint(),
        data_sha256=data_sha256,
        baseline_manifest_sha256=sha256(ROOT / 'experiments/baseline_manifest.json'),
        decision_policy_sha256=sha256(ROOT / 'experiments/model_decision_policy.json'),
        calibration_report_sha256=sha256(calibration_path),
        calibration=calibration_report['spec']['calibration'],
        provider_train_matches={source: [path.stem for path in dt.SPLITS[source]['train']]
                                for source in dt.SPLITS},
        train_hz=ms.TRAIN_HZ,
        source_seed='Use the original gr.prepare convention lt.SEED + match_index within provider.',
        views=VIEWS,
        record_source='old.feature_view(...)["joint"] with original training labels and exact cached sampled indices.',
        cache_assembly=(
            'Replicate the frozen gr.prepare raw views: clean, robust.masked_inputs '
            'legacy, calibrated ball-gap, then gr.slot_view on legacy and gap. Cache '
            'only the four joint records consumed by gr.paired_records(control=' 
            'slot_dropout): clean, gap, dropout_legacy, dropout_gap. Keep full '
            'resolution H/A/D/X ids and model periods alongside those records for '
            'the new loss API; these are training targets only.'),
        historical_source_reconciliation=reconciliation,
        prior_data_sha256=json.loads((gr.DEFAULT / 'predeclared.json').read_text())['data_sha256'],
        evaluation_features=('Reuse the separate, hash-verified seed_gap_context_v1 '
                             'feature archive; this preparation declaration covers '
                             'training features only.'),
        organizer_labels_read=False,
        baseline_changed=False,
    )
    ce.write_json(target, spec, exclusive=True)
    return spec


def check_preparation(out: Path = DEFAULT) -> dict:
    spec = json.loads((out / 'preparation_declaration.json').read_text())
    if spec['source_sha256'] != preparation_fingerprint():
        raise ValueError('Preparation source changed; use a fresh output directory')
    if spec['baseline_manifest_sha256'] != sha256(ROOT / 'experiments/baseline_manifest.json'):
        raise ValueError('Baseline manifest changed during preparation')
    if spec['decision_policy_sha256'] != sha256(ROOT / 'experiments/model_decision_policy.json'):
        raise ValueError('Decision policy changed during preparation')
    if spec['calibration_report_sha256'] != sha256(
            ROOT / 'experiments/results/realistic_corruption_development.json'):
        raise ValueError('Stress calibration changed during preparation')
    _reconcile_prior_source()
    return spec


def match_provider(match: str):
    return next(((source, index, path) for source in dt.SPLITS
                 for index, path in enumerate(dt.SPLITS[source]['train'])
                 if path.stem == match), None)


def prepare_match(match: str, out: Path = DEFAULT) -> Path:
    spec = check_preparation(out)
    selected = match_provider(match)
    if selected is None:
        raise ValueError(f'Unknown external training match: {match}')
    source, index, path = selected
    target = out / f'train_{match}.joblib'
    receipt_path = target.with_suffix('.json')
    if target.exists():
        if not receipt_path.exists():
            raise ValueError(f'Incomplete prepared cache: {target}')
        receipt = json.loads(receipt_path.read_text())
        if (receipt.get('sha256') != sha256(target)
                or receipt.get('preparation_sha256') != sha256(out / 'preparation_declaration.json')
                or receipt.get('input_sha256') != spec['data_sha256'][str(path.relative_to(ROOT))]):
            raise ValueError(f'Prepared cache changed: {target}')
        return target

    started = time.monotonic()
    inputs, labels = ms.view(path)
    input_digest = sha256(path)
    if input_digest != spec['data_sha256'][str(path.relative_to(ROOT))]:
        raise ValueError(f'Training input changed after preparation freeze: {path}')
    seed = lt.SEED + index
    legacy = robust.masked_inputs(inputs, seed)
    gap = lc.ball_gap_view(inputs, spec['calibration'], ce.seed(match, 'train'))
    raw_views = {
        'clean': inputs,
        'gap': gap,
        'dropout_legacy': gr.slot_view(legacy, seed),
        'dropout_gap': gr.slot_view(gap, seed),
    }
    records = {name: old.feature_view(view, labels, source, match)['joint']
               for name, view in raw_views.items()}
    full_y = np.full(len(labels), -1, dtype=np.int8)
    for index, label in enumerate(supervised.LABELS):
        full_y[labels == label] = index
    metadata = dict(source=source, match=match, records=records,
                    full_y=full_y,
                    full_period=supervised.model_periods(inputs).astype(np.int64),
                    fps=float(inputs['fps']), input_sha256=input_digest,
                    preparation_sha256=sha256(out / 'preparation_declaration.json'))
    joblib.dump(metadata, target, compress=0)
    ce.write_json(receipt_path, dict(
        sha256=sha256(target), bytes=target.stat().st_size,
        input_sha256=input_digest,
        preparation_sha256=sha256(out / 'preparation_declaration.json'),
        source=source, match=match, seed=seed, view_names=VIEWS,
        frames=len(full_y), seconds=time.monotonic() - started), exclusive=True)
    print('PREPARED', source, match, round(time.monotonic() - started, 1), flush=True)
    return target


def _cache_inventory(out: Path) -> dict:
    spec = check_preparation(out)
    result = {}
    for source in dt.SPLITS:
        for path in dt.SPLITS[source]['train']:
            target = out / f'train_{path.stem}.joblib'
            receipt_path = target.with_suffix('.json')
            if not target.exists() or not receipt_path.exists():
                raise ValueError(f'Missing prepared training cache: {target}')
            digest = sha256(target)
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get('sha256') != digest
                    or receipt.get('preparation_sha256') != sha256(out / 'preparation_declaration.json')
                    or receipt.get('input_sha256') != spec['data_sha256'][str(path.relative_to(ROOT))]
                    or receipt.get('bytes') != target.stat().st_size):
                raise ValueError(f'Prepared training cache receipt differs: {target}')
            result[path.stem] = dict(
                cache=str(target.relative_to(ROOT)), sha256=digest, bytes=target.stat().st_size,
                receipt=str(receipt_path.relative_to(ROOT)), receipt_sha256=sha256(receipt_path),
                input_sha256=receipt['input_sha256'],
            )
    return result


def seal_preparation(out: Path = DEFAULT) -> dict:
    """Seal all regenerated joint feature and label caches before candidate fits."""
    spec = check_preparation(out)
    inventory = _cache_inventory(out)
    value = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        preparation_declaration_sha256=sha256(out / 'preparation_declaration.json'),
        source_sha256=spec['source_sha256'],
        caches=inventory,
        count=len(inventory),
        baseline_checkpoints_retrained=False,
    )
    target = out / 'prepared_training_manifest.json'
    if target.exists():
        existing = json.loads(target.read_text())
        if existing['caches'] != inventory:
            raise ValueError('Prepared training manifest changed')
        return existing
    ce.write_json(target, value, exclusive=True)
    return value


def load_training_cache(match: str, out: Path = DEFAULT) -> dict:
    """Load one cache only after checking its sealed prep receipt."""
    check_preparation(out)
    manifest_path = out / 'prepared_training_manifest.json'
    if not manifest_path.exists():
        raise ValueError('Run seal-preparation before fitting any objective arm')
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['caches'].get(match)
    if entry is None:
        raise ValueError(f'Match is absent from prepared feature manifest: {match}')
    path = ROOT / entry['cache']
    receipt_path = ROOT / entry['receipt']
    if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
        raise ValueError(f'Prepared training cache changed: {path}')
    if sha256(receipt_path) != entry['receipt_sha256']:
        raise ValueError(f'Prepared cache receipt changed: {receipt_path}')
    return joblib.load(path, mmap_mode='r')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'prepare', 'seal'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--match')
    parser.add_argument('--source', choices=tuple(dt.SPLITS))
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze_preparation(args.out)
    elif args.action == 'prepare':
        if args.match is not None:
            prepare_match(args.match, args.out)
        else:
            sources = (args.source,) if args.source else tuple(dt.SPLITS)
            for source in sources:
                for path in dt.SPLITS[source]['train']:
                    prepare_match(path.stem, args.out)
    else:
        seal_preparation(args.out)


if __name__ == '__main__':
    main()
