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

from experiments import dead_transfer, multisource_eval, supervised

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
    """Check the corrected full H/A/D study; superseded D-only runs are unscored."""
    a, b = read(gru / 'report.json'), read(lgbm / 'report.json')
    model_counts = {'gru': verify_frozen(gru), 'joint_lightgbm': verify_frozen(lgbm)}
    if model_counts != {'gru': 12, 'joint_lightgbm': 12}:
        raise ValueError('Declared model count differs')
    if b['gru_report_sha256'] != digest(gru/'report.json'):
        raise ValueError('Shared GRU report changed')
    for relative, expected in b['evaluation_inputs']['files_sha256'].items():
        if digest(ROOT/relative) != expected:
            raise ValueError(f'Shared evaluation artifact changed: {relative}')
    for name, relative in (('baseline_manifest_sha256', 'experiments/baseline_manifest.json'),
                           ('policy_sha256', 'experiments/model_decision_policy.json')):
        if b['spec'][name] != digest(ROOT/relative):
            raise ValueError('Adopted package or decision policy changed')
    expected_gru = {(s, p.stem, v, arm, seed)
                    for s, split in dead_transfer.SPLITS.items() for p in split['eval']
                    for v in ('clean', 'corrupt_1', 'corrupt_2')
                    for arm in ('control', 'slot_dropout') for seed in (2026, 2027)}
    expected_lgbm = {(s, p.stem, v, config, recipe)
                     for s, split in dead_transfer.SPLITS.items() for p in split['eval']
                     for v in ('clean', 'corrupt_1', 'corrupt_2')
                     for config in b['spec']['configs'] for recipe in b['spec']['recipes']}
    actual_gru = {(r['source'], r['match'], r['view'], r['arm'], r['seed']) for r in a['rows']}
    actual_lgbm = {(r['source'], r['match'], r['view'], r['config'], r['recipe']) for r in b['rows']}
    if len(a['rows']) != len(expected_gru) or actual_gru != expected_gru or len(b['rows']) != len(expected_lgbm) or actual_lgbm != expected_lgbm:
        raise ValueError('Incomplete or duplicated declared cohort')
    metric_names = ('spell_f1', 'macro_f1', 'd_f1', 'd_precision', 'd_recall')
    verify_means(a['rows'] + a['archived_comparators'], a['summaries'],
                 ('arm', 'seed'), 'corrupted', metric_names[:3])
    verify_means(b['rows'] + b['gru_controls'] + b['archived_baseline'], b['summaries'],
                 ('config', 'recipe'), 'stress', metric_names + ('weighted_summary',))
    summary_index = {(r['config'], r['recipe'], r['view']): r for r in b['summaries']}
    base_clean = summary_index[('adopted_recipe', 'baseline', 'clean')]
    base_stress = summary_index[('adopted_recipe', 'baseline', 'stress')]
    for screen in b['screens']:
        clean = summary_index[(screen['config'], screen['recipe'], 'clean')]
        stress = summary_index[(screen['config'], screen['recipe'], 'stress')]
        cd, sd = clean['spell_f1']-base_clean['spell_f1'], stress['spell_f1']-base_stress['spell_f1']
        pd = min(clean['per_source'][s][k]-base_clean['per_source'][s][k]
                 for s in dead_transfer.SPLITS for k in ('d_f1', 'macro_f1'))
        expected = cd >= .01 and sd >= -.01 and pd >= -.02
        if screen['passes'] != expected or any(not np.isclose(screen[k], value, atol=1e-12, rtol=0)
                for k, value in (('clean_spell_delta', cd), ('stress_spell_delta', sd), ('worst_provider_d_or_macro_delta', pd))):
            raise ValueError('Reported guardrail outcome differs')
    baseline = {(r['source'], r['match'], r['view']): r for r in a['archived_comparators']}
    controls = {(r['source'], r['match'], r['view'], f"{r['arm']}_{r['seed']}"): r for r in a['rows']}
    tree_rows = {(r['source'], r['match'], r['view'], r['config'], r['recipe']): r for r in b['rows']}
    for r in b['gru_controls']:
        original = controls[(r['source'], r['match'], r['view'], r['recipe'])]
        if any(r[k] != original[k] for k in original):
            raise ValueError('Reused GRU row differs')
    for r in b['archived_baseline']:
        original = baseline[(r['source'], r['match'], r['view'])]
        if any(r[k] != original[k] for k in original):
            raise ValueError('Reused baseline differs')
    native_gru_count = native_tree_count = blend_count = decoded_count = 0
    for source, split in dead_transfer.SPLITS.items():
        for path in split['eval']:
            _, labels = multisource_eval.view(path)
            support = int(np.isin(labels, ['H', 'A', 'D']).sum())
            for view in ('clean', 'corrupt_1', 'corrupt_2'):
                cached = joblib.load(gru/f'eval_{path.stem}_{view}.joblib')
                if not np.array_equal(labels, cached['labels']):
                    raise ValueError('Shared labels differ from original masked cohort')
                if cached['joint']['match'] != path.stem or cached['joint']['source'] != source:
                    raise ValueError('Shared cache identity differs')
                reference = baseline[(source, path.stem, view)]
                archived_support = sum(reference['per_class'][label]['support'] for label in ('H', 'A', 'D'))
                if archived_support != support:
                    raise ValueError('Archived baseline grading differs')
                gru_probabilities = {}
                for arm in ('control', 'slot_dropout'):
                    for seed in (2026, 2027):
                        recipe = f'{arm}_{seed}'
                        r = controls[(source, path.stem, view, recipe)]
                        if (r['n_gt_spells'], r['graded_frames']) != (reference['n_gt_spells'], support):
                            raise ValueError('GRU grading mismatch')
                        with np.load(gru/f'{source}_{path.stem}_{view}_{recipe}.npz') as saved:
                            p = saved['probability'].copy()
                            valid_probability(p, len(labels))
                            if len(saved['prediction']) != len(labels):
                                raise ValueError('Native GRU prediction length differs')
                        gru_probabilities[recipe] = p
                        native_gru_count += 1
                for config in b['spec']['configs']:
                    with np.load(lgbm/f'{source}_{path.stem}_{view}_{config}_probability.npz') as saved:
                        p = saved['probability'].copy()
                        valid_probability(p, len(labels))
                    native_tree_count += 1
                    for recipe in b['spec']['recipes']:
                        r = tree_rows[(source, path.stem, view, config, recipe)]
                        if (r['n_gt_spells'], r['graded_frames']) != (reference['n_gt_spells'], support):
                            raise ValueError('Joint LightGBM grading mismatch')
                        combined = p if recipe == 'standalone' else .5*p + .5*gru_probabilities[recipe]
                        if array_digest(combined) != r['native_probability_sha256']:
                            raise ValueError('Full H/A/D blend differs from scored probabilities')
                        prediction_path = lgbm/f'{source}_{path.stem}_{view}_{config}_{recipe}.npz'
                        with np.load(prediction_path) as saved:
                            prediction = saved['prediction']
                            if len(prediction) != len(labels) or not np.isin(prediction, ['H', 'A', 'D']).all():
                                raise ValueError('Invalid native tree prediction')
                            if array_digest(prediction) != r['prediction_sha256']:
                                raise ValueError('Saved prediction changed')
                            # All vector hashes are checked; a fixed representative
                            # cell per provider/recipe also verifies decoding.
                            if path == split['eval'][0] and view == 'clean' and config == 'legacy_63':
                                inputs = cached['inputs']
                                decoded = supervised.decode(combined, supervised.model_periods(inputs), float(inputs['fps']), 3)
                                if not np.array_equal(decoded, prediction):
                                    raise ValueError('Probability-to-decoded-label mismatch')
                                decoded_count += 1
                        blend_count += 1
    result = dict(verified_utc=datetime.now(timezone.utc).isoformat(),
                  verifier_sha256=digest(__file__), model_counts=model_counts,
                  new_scored_rows={'gru': len(a['rows']), 'joint_lightgbm': len(b['rows'])},
                  archived_baseline_rows=len(a['archived_comparators']),
                  reused_gru_rows=len(b['gru_controls']),
                  independent_summary_and_grading_checks=True,
                  native_gru_probabilities_verified=native_gru_count,
                  native_tree_probabilities_verified=native_tree_count,
                  full_vector_blends_verified=blend_count,
                  representative_decodes_verified=decoded_count,
                  organizer_labels_read=False,
                  reports={str(p.relative_to(ROOT)): digest(p) for p in (gru/'report.json', lgbm/'report.json')})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


def array_digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def valid_probability(p, n):
    if p.shape != (n, 3) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-6):
        raise ValueError('Invalid saved native H/A/D probability array')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gru', type=Path, default=ROOT/'data/experiments/gru_slot_robustness_v2')
    parser.add_argument('--lgbm', type=Path, default=ROOT/'data/experiments/joint_lgbm_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'data/experiments/temporal_followup_verification.json')
    args = parser.parse_args()
    verify(args.gru, args.lgbm, args.output)
