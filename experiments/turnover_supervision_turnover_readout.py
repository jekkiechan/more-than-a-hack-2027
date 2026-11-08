"""Descriptive turnover-error readout for the three pooled CE candidates.

Reuses the approved bidirectional event auditor on saved native predictions.
This diagnostic is separate from the selection guards and retains aggregates
only; it does not fit, infer, or score the official possession metric.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import turnover_error_audit as auditor
from experiments import turnover_supervision_eval as evaluation
from experiments import turnover_supervision_train as training

ROOT = evaluation.ROOT
STUDY = 'turnover_supervision_turnover_readout_v1'
ARMS = ('fresh_control', 'uniform_turnover', 'hard_turnover')
DECLARATION = ROOT / 'experiments/results/turnover_supervision_turnover_readout_declaration.json'
RESULT = ROOT / 'experiments/results/turnover_supervision_turnover_readout.json'
BASELINE_AUDIT = ROOT / 'experiments/results/turnover_error_audit.json'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _source_facts() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): _sha(path) for path in
        (Path(__file__), Path(auditor.__file__), Path(evaluation.__file__),
         Path(training.__file__),
         ROOT / 'experiments/tests/test_turnover_supervision_turnover_readout.py',
         evaluation.NATIVE_MANIFEST, BASELINE_AUDIT)}


def freeze() -> dict:
    if DECLARATION.exists() or RESULT.exists():
        raise FileExistsError('preserve existing turnover-specific readout artifacts')
    candidate_cells = evaluation.OUT / 'cells'
    if evaluation.RESULT.exists() or (candidate_cells.exists() and any(candidate_cells.glob('*.npz'))):
        raise ValueError('freeze this descriptive diagnostic before candidate inference or evaluation results')
    if not BASELINE_AUDIT.is_file() or not evaluation.DECLARATION.is_file():
        raise FileNotFoundError('baseline turnover audit and frozen evaluation declaration are required')
    baseline = json.loads(BASELINE_AUDIT.read_text())
    eval_decl = json.loads(evaluation.DECLARATION.read_text())
    train_decl = json.loads(training.TRAIN_DECL.read_text())
    native = json.loads(evaluation.NATIVE_MANIFEST.read_text())
    if (baseline.get('schema') != auditor.SCHEMA or len(native.get('cells', {})) != 35 or
            eval_decl.get('study_id') != evaluation.STUDY or train_decl.get('study_id') != evaluation.STUDY):
        raise ValueError('unexpected baseline/evaluation/training inventory')
    value = {
        'schema': STUDY, 'study_id': STUDY, 'status': 'frozen_before_turnover_candidate_readout',
        'question': 'Do the three pooled turnover-supervision candidates recover more eligible direct H/A turnovers than the approved baseline while retaining a fixed D/X mask?',
        'source_sha256': _source_facts(),
        'inputs': {
            'native_manifest': {'path': str(evaluation.NATIVE_MANIFEST.relative_to(ROOT)),
                                'sha256': _sha(evaluation.NATIVE_MANIFEST)},
            'baseline_turnover_audit': {'path': str(BASELINE_AUDIT.relative_to(ROOT)),
                                        'sha256': _sha(BASELINE_AUDIT)},
            'evaluation_declaration': {'path': str(evaluation.DECLARATION.relative_to(ROOT)),
                                       'sha256': _sha(evaluation.DECLARATION)},
            'training_declaration': {'path': str(training.TRAIN_DECL.relative_to(ROOT)),
                                     'sha256': _sha(training.TRAIN_DECL)},
            'cells': sorted(native['cells']),
        },
        'arms': list(ARMS),
        'method': {
            'auditor': 'turnover_error_audit.audit_cell, unchanged',
            'eligible_event': 'direct H↔A edge with >=1s contiguous same-team context before and after in one period/model-time block',
            'recovery': 'directional one-to-one maximum-cardinality match within ±1s, same block',
            'counts': 'eligible, recovered, misses, miss reason, and weak/reversed posterior evidence; predicted H/A edge count is descriptive and is not precision because the reference subset is restricted to eligible turnovers',
            'scope': '35 repeated cache cells spanning five matches; candidate rows are descriptive and not independent samples or selection guards',
            'retention': 'aggregate counts by provider/view and cell only; do not save raw event details',
        },
        'organizer_reference_access': False, 'official_candidate_scoring': False,
        'selection_guard': False, 'promotion': False,
    }
    _write_once(DECLARATION, value)
    return value


def _load_native(fact: dict):
    path = ROOT / fact['path']
    if not path.is_file() or _sha(path) != fact['sha256']:
        raise ValueError(f'native cell hash mismatch: {fact["identity"]["cell"]}')
    return auditor._read_cell(path)


def _candidate_arrays(native_arrays, prediction_path: Path, arm: str, *, key: str, facts: dict):
    labels, period, model_period, timestamps, fps, _base_probability, base_prediction = native_arrays
    with np.load(prediction_path, allow_pickle=False) as z:
        probability = np.asarray(z[f'prob__{arm}'], dtype=np.float64)
        prediction = z[f'pred__{arm}'].astype('U1', copy=True)
    if probability.shape != (len(labels), 2) or prediction.shape != labels.shape:
        raise ValueError(f'candidate probability/prediction mismatch: {arm}')
    if not np.array_equal(prediction == 'D', base_prediction == 'D') or \
            not np.array_equal(prediction == 'X', base_prediction == 'X'):
        raise ValueError(f'candidate changed fixed D/X mask: {arm}')
    digest = hashlib.sha256(prediction.tobytes()).hexdigest()
    if facts.get(arm, {}).get(key) != digest:
        raise ValueError(f'candidate prediction digest mismatch: {key}/{arm}')
    return labels, period, model_period, timestamps, fps, probability, prediction


def _aggregate(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    additive = ('all_direct_HA_edges', 'eligible_HA_edges', 'predicted_HA_edges',
                'recovered_one_to_one_within_1s', 'misses',
                'misses_baseline_D_at_boundary', 'misses_baseline_D_within_1s')
    counted = ('miss_reason_counts', 'miss_posterior_evidence_counts',
               'misses_by_D_and_posterior', 'posterior_evidence_counts_all_eligible',
               'pre_duration_bins_all_edges', 'post_duration_bins_all_edges',
               'pre_post_duration_bins_all_edges')
    result = []
    for values, items in sorted(groups.items()):
        entry = dict(zip(group_keys, values))
        entry['cells'] = len(items)
        for field in additive:
            entry[field] = sum(int(row[field]) for row in items)
        for field in counted:
            total = Counter()
            for row in items:
                total.update(row[field])
            entry[field] = dict(sorted(total.items()))
        result.append(entry)
    return result


def _same_json(actual, expected, where: str) -> None:
    if actual != expected:
        raise ValueError(f'baseline turnover audit replay differs: {where}')


def run_readout() -> dict:
    if RESULT.exists():
        raise FileExistsError('preserve existing turnover-specific readout result')
    if not DECLARATION.is_file():
        raise FileNotFoundError('freeze turnover-specific readout before evaluation result is read')
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get('source_sha256') != _source_facts():
        raise ValueError('turnover-specific readout source changed after freeze')
    for name, fact in declaration['inputs'].items():
        if name == 'cells':
            continue
        if _sha(ROOT / fact['path']) != fact['sha256']:
            raise ValueError(f'readout input changed after freeze: {name}')
    if not evaluation.RESULT.is_file():
        raise FileNotFoundError('sealed turnover evaluation result is required before diagnostic run')
    result_hash = _sha(evaluation.RESULT)
    eval_result = json.loads(evaluation.RESULT.read_text())
    if (eval_result.get('study_id') != evaluation.STUDY or eval_result.get('passed') is not True or
            eval_result.get('declaration_sha256') != _sha(evaluation.DECLARATION) or
            eval_result.get('native_manifest_sha256') != _sha(evaluation.NATIVE_MANIFEST) or
            set(eval_result.get('cells', {})) != set(declaration['inputs']['cells'])):
        raise ValueError('turnover evaluation result is incomplete')
    baseline_report = json.loads(BASELINE_AUDIT.read_text())
    expected_baseline_rows = {row['cell']: row for row in baseline_report['summary']['cells']}
    if set(expected_baseline_rows) != set(declaration['inputs']['cells']):
        raise ValueError('baseline turnover report does not contain all 35 cell rows')
    native_manifest = json.loads(evaluation.NATIVE_MANIFEST.read_text())
    baseline_rows, candidate_rows = {}, {arm: {} for arm in ARMS}
    predicted_counts = {arm: {} for arm in ARMS}
    for key in sorted(declaration['inputs']['cells']):
        native_fact = native_manifest['cells'][key]
        arrays = _load_native(native_fact)
        meta = {'identity': native_fact['identity']}
        base_row, _ = auditor.audit_cell(key, meta, arrays)
        _same_json(base_row, expected_baseline_rows[key], key)
        baseline_rows[key] = base_row
        eval_fact = eval_result['cells'][key]
        prediction_path = ROOT / eval_fact['path']
        if not prediction_path.is_file() or _sha(prediction_path) != eval_fact['sha256']:
            raise ValueError(f'evaluation prediction artifact hash mismatch: {key}')
        for arm in ARMS:
            candidate = _candidate_arrays(arrays, prediction_path, arm, key=key,
                                          facts=eval_result['prediction_sha256'])
            row, _ = auditor.audit_cell(key, meta, candidate)
            candidate_rows[arm][key] = row
            predicted_counts[arm][key] = int(row['predicted_HA_edges'])
    per_cell = {arm: {key: {k: row[k] for k in (
        'provider', 'match', 'view', 'rep', 'eligible_HA_edges',
        'recovered_one_to_one_within_1s', 'misses', 'miss_reason_counts',
        'miss_posterior_evidence_counts', 'misses_by_D_and_posterior',
        'posterior_evidence_counts_all_eligible', 'predicted_HA_edges')}
        for key, row in candidate_rows[arm].items()} for arm in ARMS}
    aggregate = {arm: {'by_provider_view': _aggregate(list(candidate_rows[arm].values()), ('provider', 'view')),
                       'by_provider': _aggregate(list(candidate_rows[arm].values()), ('provider',)),
                       'overall_repeated_cells': _aggregate(list(candidate_rows[arm].values()), ())}
                 for arm in ARMS}
    return {
        'schema': 'turnover_supervision_turnover_readout_result_v1',
        'study_id': STUDY, 'passed': True,
        'declaration_sha256': _sha(DECLARATION), 'evaluation_result_sha256': result_hash,
        'baseline_audit_sha256': _sha(BASELINE_AUDIT),
        'baseline_replay_verified_cells': len(baseline_rows),
        'candidate_audit_rows': sum(map(len, candidate_rows.values())),
        'arms': list(ARMS), 'aggregate': aggregate, 'per_cell': per_cell,
        'predicted_edge_counts_caveat': 'Descriptive count only; denominator is the eligible reference-turnover subset, so this is not precision.',
        'sampling_unit_caveat': '35 cells repeat physical events across views and realizations; counts are not independent observations.',
        'raw_event_details_written': False,
        'organizer_reference_access': False, 'selection_guard': False, 'promotion': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run'))
    args = parser.parse_args()
    report = freeze() if args.action == 'freeze' else run_readout()
    if args.action == 'run':
        _write_once(RESULT, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
