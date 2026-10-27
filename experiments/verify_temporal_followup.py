"""Independently verify completed external follow-up reports and frozen artifacts.

Reads only saved external results and predictions; never fits or reads organizer
labels. Writes a new receipt exclusively, without modifying experiment records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import joblib

from experiments import dead_transfer, multisource_eval

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_frozen(directory):
    spec = read(directory / 'predeclared.json')
    for category in ('source_sha256', 'data_sha256'):
        for relative, expected in spec[category].items():
            if digest(ROOT / relative) != expected:
                raise ValueError(f'Changed {category}: {relative}')
    sealed = read(directory / 'checkpoints_frozen.json')['checkpoints']
    for model in sealed:
        path = directory / model['file']
        if digest(path) != model['sha256'] or digest(path.with_suffix('.json')) != model['receipt_sha256']:
            raise ValueError(f'Changed checkpoint: {path.name}')
    return len(sealed)


def verify_means(rows, summaries, identity, stress_name, metric_names):
    """Recompute match means, then provider means, without runner summaries."""
    for summary in summaries:
        selected = [r for r in rows if all(r[k] == summary[k] for k in identity)
                    and ((r['view'] == 'clean') == (summary['view'] == 'clean'))]
        if summary['view'] not in ('clean', stress_name) or not selected:
            raise ValueError('Unexpected summary population')
        providers = sorted({r['source'] for r in selected})
        for metric in metric_names:
            per_provider = []
            for provider in providers:
                matches = sorted({r['match'] for r in selected if r['source'] == provider})
                values = [np.mean([r[metric] for r in selected if r['source'] == provider and r['match'] == match])
                          for match in matches]
                value = float(np.mean(values))
                if not np.isclose(value, summary['per_source'][provider][metric], atol=1e-12, rtol=0):
                    raise ValueError(f'Provider mean mismatch: {provider} {metric}')
                per_provider.append(value)
            if not np.isclose(np.mean(per_provider), summary[metric], atol=1e-12, rtol=0):
                raise ValueError(f'Equal-provider mean mismatch: {metric}')


def verify(gru, lgbm, output):
    a, b = read(gru / 'report.json'), read(lgbm / 'report.json')
    model_counts = {'gru': verify_frozen(gru), 'lightgbm': verify_frozen(lgbm)}
    expected_gru = {(s, p.stem, v, arm, seed)
                    for s, split in dead_transfer.SPLITS.items() for p in split['eval']
                    for v in ('clean', 'corrupt_1', 'corrupt_2')
                    for arm in ('control', 'slot_dropout') for seed in (2026, 2027)}
    expected_lgbm = {(s, p.stem, v, config, seed, weight)
                     for s, split in dead_transfer.SPLITS.items() for p in split['eval']
                     for v in ('clean', 'corrupt_1', 'corrupt_2')
                     for config in b['spec']['configs'] for seed in (2026, 2027)
                     for weight in (.5, 1.)}
    actual_gru = {(r['source'], r['match'], r['view'], r['arm'], r['seed']) for r in a['rows']}
    actual_lgbm = {(r['source'], r['match'], r['view'], r['config'], r['seed'], r['weight']) for r in b['rows']}
    if len(a['rows']) != len(expected_gru) or actual_gru != expected_gru or len(b['rows']) != len(expected_lgbm) or actual_lgbm != expected_lgbm:
        raise ValueError('Incomplete or duplicated declared cohort')
    if model_counts != {'gru': 12, 'lightgbm': 12}:
        raise ValueError('Declared model count differs')
    metric_names = ('spell_f1', 'macro_f1', 'd_f1', 'd_precision', 'd_recall')
    verify_means(a['rows'] + a['archived_comparators'], a['summaries'],
                 ('arm', 'seed'), 'corrupted', metric_names[:3])
    verify_means(b['rows'] + b['controls'], b['summaries'],
                 ('config', 'weight', 'seed'), 'stress', metric_names)
    baseline = {(r['source'], r['match'], r['view']): r for r in a['archived_comparators']}
    controls = {(r['source'], r['match'], r['view'], r['seed']): r
                for r in a['rows'] if r['arm'] == 'control'}
    label_support = {}
    for source, split in dead_transfer.SPLITS.items():
        for path in split['eval']:
            _, labels = multisource_eval.view(path)
            label_support[(source, path.stem)] = int(np.isin(labels, ['H', 'A', 'D']).sum())
            for view in ('clean', 'corrupt_1', 'corrupt_2'):
                cached = joblib.load(gru/f'eval_{path.stem}_{view}.joblib')
                if not np.array_equal(labels, cached['labels']):
                    raise ValueError('Shared evaluation labels differ from original masked cohort')
                if cached['joint']['match'] != path.stem or cached['joint']['source'] != source:
                    raise ValueError('Shared evaluation cache identity differs')
    for r in a['rows']:
        key = (r['source'], r['match'], r['view'])
        original = baseline[key]
        archived_support = sum(original['per_class'][label]['support'] for label in ('H', 'A', 'D'))
        if r['n_gt_spells'] != original['n_gt_spells'] or r['graded_frames'] != archived_support or r['graded_frames'] != label_support[key[:2]]:
            raise ValueError(f'GRU grading mismatch: {key}')
        probabilities = np.load(gru / f"{r['source']}_{r['match']}_{r['view']}_{r['arm']}_{r['seed']}.npz")
        p = probabilities['probability']
        if p.shape != (len(probabilities['prediction']), 3) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6):
            raise ValueError('Invalid saved native GRU probability array')
    for r in b['rows'] + b['controls']:
        reference = controls[(r['source'], r['match'], r['view'], r['seed'])]
        for field in ('n_gt_spells', 'graded_frames'):
            if r[field] != reference[field]:
                raise ValueError(f'LightGBM grading mismatch: {field}')
        if r['config'] == 'gru_control':
            for metric in metric_names:
                if r[metric] != reference[metric]:
                    raise ValueError(f'Reused control differs: {metric}')
    # Preserve every source/input/model guard in the reports, and report actual
    # score populations separately from the additional verified archived rows.
    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                  verifier_sha256=digest(__file__), model_counts=model_counts,
                  new_scored_rows={'gru': len(a['rows']), 'lightgbm': len(b['rows'])},
                  archived_baseline_rows=len(a['archived_comparators']),
                  reused_gru_control_rows=len(b['controls']),
                  independent_summary_and_grading_checks=True,
                  valid_native_gru_probabilities=True,
                  organizer_labels_read=False,
                  reports={str(p.relative_to(ROOT)): digest(p) for p in (gru/'report.json', lgbm/'report.json')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gru', type=Path, default=ROOT/'data/experiments/gru_slot_robustness_v2')
    parser.add_argument('--lgbm', type=Path, default=ROOT/'data/experiments/temporal_lgbm_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'data/experiments/temporal_followup_verification.json')
    args = parser.parse_args()
    verify(args.gru, args.lgbm, args.output)
