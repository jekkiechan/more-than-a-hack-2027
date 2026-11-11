"""Independent readout for saved turnover-supervision predictions and receipts.

Re-scores the approved baseline and every saved candidate/seed diagnostic with
the official scorer. It performs no fitting or model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import turnover_supervision_eval as run
from experiments import turnover_supervision_train as train
from experiments import frontier_postprocess_readout as official

ROOT = run.ROOT
ARMS = run.ARMS
SEEDS = run.SEEDS
TOL = 1e-10
PROB_TOL = 1e-10
READOUT = ROOT / 'experiments/results/turnover_supervision_readout.json'


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _close(actual, expected, where: str, tol: float = TOL) -> None:
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if int(actual) != int(expected):
            raise ValueError(f'{where}: {actual} != {expected}')
    elif expected is None:
        if actual is not None:
            raise ValueError(f'{where}: {actual} != null')
    elif not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tol):
        raise ValueError(f'{where}: {actual} != {expected}')


def _compare_rows(actual: Mapping, expected: Mapping, where: str) -> None:
    official._compare_row(actual, expected, where)


def _validate_hash_facts(inventory: Mapping[str, str]) -> None:
    if not inventory:
        raise ValueError('empty source hash inventory')
    for relative, digest in inventory.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or _sha(path) != digest:
            raise ValueError(f'missing or changed source: {relative}')


def _pred(archive, name: str) -> np.ndarray:
    pred = official._read_labels(archive[f'pred__{name}'], name)
    if pred.ndim != 1:
        raise ValueError(f'invalid prediction rank: {name}')
    return pred


def _check_dx(prediction: np.ndarray, baseline: np.ndarray, where: str) -> None:
    # D/X rows must match exactly: only H/A predictions are mutable.
    if not np.array_equal(prediction == 'D', baseline == 'D'):
        raise ValueError(f'D mask changed: {where}')
    if not np.array_equal(prediction == 'X', baseline == 'X'):
        raise ValueError(f'X mask changed: {where}')


def _aggregate(rows: Mapping[str, dict]) -> dict:
    return official._aggregate(rows, run.VIEWS)


def _compare_aggregates(actual: Mapping, expected: Mapping, where: str) -> None:
    for view in run.VIEWS:
        for metric in official.balanced.METRICS:
            _close(actual[view][metric]['equal_provider'], expected[view][metric]['equal_provider'],
                   f'{where}/{view}/{metric}/equal_provider', 1e-10)
            if actual[view][metric]['providers'].keys() != expected[view][metric]['providers'].keys():
                raise ValueError(f'{where}/{view}/{metric}: provider set changed')
            for provider in actual[view][metric]['providers']:
                _close(actual[view][metric]['providers'][provider],
                       expected[view][metric]['providers'][provider],
                       f'{where}/{view}/{metric}/{provider}', 1e-10)


def _pool_saved_ce(archive, arm: str, phase_count: int, seed: int | None = None) -> np.ndarray:
    by_phase = []
    for phase in range(phase_count):
        key_prefix = f'prob__ce__{arm}__seed'
        if seed is None:
            left = np.asarray(archive[f'{key_prefix}2026__phase{phase}'], dtype=np.float64)
            right = np.asarray(archive[f'{key_prefix}2027__phase{phase}'], dtype=np.float64)
            if left.shape != right.shape:
                raise ValueError(f'paired CE phase shapes differ: {arm}/{phase}')
            by_phase.append((left + right) / 2.0)
        else:
            by_phase.append(np.asarray(archive[f'{key_prefix}{seed}__phase{phase}'], dtype=np.float64))
    if any(x.ndim != 2 or x.shape[1] != 3 for x in by_phase):
        raise ValueError(f'raw CE phase must be native Nx3: {arm}')
    pooled = np.mean(np.stack(by_phase, axis=0), axis=0)
    return pooled


def _conditional(probability: np.ndarray) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    ha = value[:, :2]
    mass = ha.sum(axis=1, keepdims=True)
    if not np.isfinite(value).all() or np.any(mass <= 0):
        raise ValueError('invalid H/A expert probability mass')
    return ha / mass


def _decode_replay(key: str, cell: Mapping, archive, name: str, *, arm: str, seed: int | None = None) -> None:
    phase_count = len(cell['components']['predecessor'][2026])
    ce = _pool_saved_ce(archive, arm, phase_count, seed)
    saved_ce = np.asarray(archive[f'ce__{arm}'], dtype=np.float64)
    if seed is None and not np.allclose(ce, saved_ce, rtol=0, atol=PROB_TOL):
        raise ValueError(f'pooled CE replay mismatch: {key}/{name}')
    baseline_parts = cell['components']
    calibrated = official._pool_family(baseline_parts['calibrated'], .5)
    predecessor = official._pool_family(baseline_parts['predecessor'], .5)
    mixture = .5 * _conditional(ce) + .375 * _conditional(calibrated) + .125 * _conditional(predecessor)
    mixture /= mixture.sum(axis=1, keepdims=True)
    saved_probability = np.asarray(archive[f'prob__{name}'], dtype=np.float64)
    if not np.allclose(mixture, saved_probability, rtol=0, atol=PROB_TOL):
        raise ValueError(f'candidate mixture replay mismatch: {key}/{name}')
    prediction = _pred(archive, name)
    decoded = official._decode(mixture, cell['decoded_predecessor'], cell, run.HA_COST)
    if not np.array_equal(decoded, prediction):
        raise ValueError(f'candidate decoder replay mismatch: {key}/{name}')
    _check_dx(prediction, cell['baseline_prediction'], f'{key}/{name}')


def _training_audit(declaration: Mapping, seal: Mapping) -> dict:
    train_decl = _load_training_declaration()
    rows = {(r['arm'], r['fold'], int(r['seed'])): r for r in seal['fits']}
    if len(rows) != 18 or set(rows) != run.expected_fit_keys():
        raise ValueError('expected exactly 18 sealed arm/fold/seed fits')
    schedule_checks = 0
    frozen_checks = 0
    request_checks = 0
    catalog_checks = 0
    per_fit = {}
    for fold in train.FOLDS:
        for seed in train.SEEDS:
            base = rows[('fresh_control', fold, seed)]
            for arm in ARMS:
                row = rows[(arm, fold, seed)]
                path = ROOT / row['model_path']
                if not path.is_file() or _sha(path) != row['model_sha256']:
                    raise ValueError(f'model hash changed: {arm}/{fold}/{seed}')
                receipt_path = Path(str(path) + '.json')
                receipt = json.loads(receipt_path.read_text())
                if receipt != row:
                    raise ValueError(f'fit seal and sidecar receipt differ: {arm}/{fold}/{seed}')
                if receipt.get('draw_sequence_sha256') != row.get('draw_sequence_sha256'):
                    raise ValueError(f'receipt sequence mismatch: {arm}/{fold}/{seed}')
                for field in ('view_draws', 'exposures'):
                    if receipt[field] != base[field]:
                        raise ValueError(f'matched source/match/GAR schedule differs: {arm}/{fold}/{seed}/{field}')
                windows = receipt.get('sampled_window_ids', [])
                base_windows = base.get('sampled_window_ids', [])
                if not windows or len(windows) != len(base_windows):
                    raise ValueError(f'missing exact draw schedule: {arm}/{fold}/{seed}')
                if [tuple(x[:3]) for x in windows] != [tuple(x[:3]) for x in base_windows]:
                    raise ValueError(f'draw-level source/match/GAR schedule changed: {arm}/{fold}/{seed}')
                for current, initial in zip(windows, base_windows):
                    if str(current[0]).lower() in ('new', 'gar', '1') and current != initial:
                        raise ValueError(f'GAR window schedule changed: {arm}/{fold}/{seed}')
                old_meta = receipt.get('old_record_index', [])
                base_meta = base.get('old_record_index', [])
                if old_meta != base_meta:
                    raise ValueError(f'original record metadata differs: {arm}/{fold}/{seed}')
                cohorts = train_decl['cohorts'][fold]
                excluded = fold.removeprefix('leave_out_')
                if {row['match'] for row in old_meta} != set(cohorts):
                    raise ValueError(f'original metadata cohort differs from declaration: {arm}/{fold}/{seed}')
                if any(row['match'] not in cohorts or row['source'].lower() == excluded for row in old_meta):
                    raise ValueError(f'original record metadata violates fold exclusion: {arm}/{fold}/{seed}')
                # The sampler's actual old-window stream excludes proposed
                # choices; event replacement may alter start/size, while record
                # and clean/gap orientation must remain identical.
                actual_old = receipt['sampling']['actual_old_window_ids']
                base_old = base['sampling']['actual_old_window_ids']
                if [(x[0], x[3]) for x in actual_old] != [(x[0], x[3]) for x in base_old]:
                    raise ValueError(f'actual old record/orientation schedule changed: {arm}/{fold}/{seed}')
                if receipt['sampling']['turnover_request_count'] != base['sampling']['turnover_request_count'] or \
                        receipt['sampling']['turnover_request_mask_sha256'] != base['sampling']['turnover_request_mask_sha256']:
                    raise ValueError(f'turnover request schedule changed: {arm}/{fold}/{seed}')
                request_checks += 1
                schedule_checks += 1
                model = __import__('joblib').load(path)
                if not receipt.get('d_state_unchanged') or not receipt.get('normalization_unchanged'):
                    raise ValueError(f'frozen D/normalization receipt failed: {arm}/{fold}/{seed}')
                warm = __import__('joblib').load(ROOT / row['warm_start_path'])
                warm_path = (ROOT / row['warm_start_path']).resolve()
                declared_warm = train_decl['input_sha256'][f'stage1024/{fold}/{seed}']['sha256']
                if _sha(warm_path) != declared_warm or _sha(warm_path) != row['warm_start_sha256']:
                    raise ValueError(f'warm-start checkpoint hash differs from frozen source: {fold}/{seed}')
                for name, values in (('d_state', warm.get('d_state', {})),):
                    if set(model.get(name, {})) != set(values):
                        raise ValueError(f'{name} keys differ: {arm}/{fold}/{seed}')
                    for k, v in values.items():
                        np.testing.assert_array_equal(model[name][k], v)
                for name in ('median', 'scale'):
                    np.testing.assert_array_equal(model[name], warm[name])
                if model['feature_names'] != warm['feature_names']:
                    raise ValueError(f'feature normalization schema changed: {arm}/{fold}/{seed}')
                frozen_checks += 1
                per_fit[f'{arm}|{fold}|{seed}'] = {
                    'draw_count': int(receipt['draw_count']),
                    'draw_sequence_sha256': receipt['draw_sequence_sha256'],
                    'turnover_sampling_counts': {k: int(v) for k, v in receipt['sampling'].items()
                        if isinstance(v, (int, np.integer)) and not isinstance(v, bool)},
                    'hardness_catalog_sha256': receipt.get('hardness_catalog_sha256'),
                }
            uniform = rows[('uniform_turnover', fold, seed)]['sampling']
            hard = rows[('hard_turnover', fold, seed)]['sampling']
            uniform_ids = {tuple(x) for x in uniform['turnover_selected_event_ids']}
            hard_ids = {tuple(x) for x in hard['turnover_selected_event_ids']}
            if not hard_ids.issubset(uniform_ids):
                raise ValueError(f'hard turnover catalog is not a uniform subset: {fold}/{seed}')
            hard_row = rows[('hard_turnover', fold, seed)]
            hardness_path = ROOT / hard_row['hardness_catalog_path']
            if not hardness_path.is_file() or _sha(hardness_path) != hard_row['hardness_catalog_sha256']:
                raise ValueError(f'hardness catalog hash mismatch: {fold}/{seed}')
            hardness_catalog = json.loads(hardness_path.read_text())
            if hardness_catalog.get('warm_start_sha256') != hard_row['warm_start_sha256']:
                raise ValueError(f'hardness catalog checkpoint mismatch: {fold}/{seed}')
            hardness = {tuple(row['identity']): row for row in hardness_catalog['events']}
            cohorts = set(train_decl['cohorts'][fold])
            if any(identity[0] not in cohorts for identity in hardness):
                raise ValueError(f'hardness catalog match outside fold TRAIN cohort: {fold}/{seed}')
            if not uniform_ids.issubset(hardness):
                raise ValueError(f'uniform candidate lacks physical hardness evidence: {fold}/{seed}')
            per_match = {}
            for identity in uniform_ids:
                value = hardness[identity]
                score = min(float(value['pre_correct']), float(value['post_correct']))
                per_match.setdefault(identity[0], []).append((score, identity))
            expected_hard = set()
            for entries in per_match.values():
                entries.sort(key=lambda item: (item[0], item[1]))
                expected_hard.update(identity for _score, identity in entries[:(len(entries)+1)//2])
            if expected_hard != hard_ids:
                raise ValueError(f'hardness bottom-half selection mismatch: {fold}/{seed}')
            for receipt in (uniform, hard):
                selected = {tuple(x) for x in receipt['turnover_selected_event_ids']}
                nominated = {tuple(x) for x in receipt['turnover_nominated_event_ids']}
                encountered = {tuple(x) for x in receipt['turnover_events_encountered']}
                if not nominated.issubset(selected) or not encountered.issubset(uniform_ids):
                    raise ValueError(f'nominated event outside selected catalog: {fold}/{seed}')
            catalog_checks += 1
    return {'fit_count': len(rows), 'matched_schedule_checks': schedule_checks,
            'request_schedule_checks': request_checks, 'hardness_catalog_checks': catalog_checks,
            'frozen_d_normalization_checks': frozen_checks, 'fits': per_fit}


def _load_training_declaration() -> dict:
    """Load the frozen cohort/source binding once before any fold audit."""
    value = json.loads(train.TRAIN_DECL.read_text())
    if value.get('study_id') != run.STUDY or not isinstance(value.get('cohorts'), dict):
        raise ValueError('invalid turnover training declaration')
    return value


def verify() -> dict:
    declaration = json.loads(run.DECLARATION.read_text())
    baseline_replay = json.loads(run.BASELINE_REPLAY.read_text())
    result = json.loads(run.RESULT.read_text())
    if (declaration.get('status') != 'frozen_before_candidate_scoring' or
            declaration.get('study_id') != run.STUDY or
            result.get('study_id') != run.STUDY or result.get('passed') is not True):
        raise ValueError('unsupported or incomplete turnover evaluation artifacts')
    _validate_hash_facts(declaration['source_hashes'])
    _, fit_seal, _ = run._training_inputs()
    if (result.get('declaration_sha256') != _sha(run.DECLARATION) or
            baseline_replay.get('declaration_sha256') != _sha(run.DECLARATION) or
            result.get('baseline_replay_sha256') != _sha(run.BASELINE_REPLAY) or
            baseline_replay.get('passed') is not True):
        raise ValueError('evaluation artifacts are not bound to the frozen declaration/replay')
    facts = declaration['native_cells']
    if set(facts) != run.EXPECTED_CELLS or set(result.get('cells', {})) != run.EXPECTED_CELLS:
        raise ValueError('expected all 35 cell artifacts')
    if set(baseline_replay.get('cell_metrics', {})) != run.EXPECTED_CELLS:
        raise ValueError('baseline replay does not cover all 35 cells')
    native_manifest, native_cells = run._native_inputs()
    frontier_hashes = run._frontier_hashes()
    baseline_rows, candidate_rows = {}, {arm: {} for arm in ARMS}
    seed_names = [f'{arm}_seed{seed}' for arm in ARMS for seed in SEEDS]
    seed_rows = {name: {} for name in seed_names}
    prediction_hashes = {name: {} for name in [*ARMS, *seed_names]}
    base_hashes = {}
    for key in sorted(run.EXPECTED_CELLS):
        fact = facts[key]
        if fact['sha256'] != native_cells[key]['sha256']:
            raise ValueError(f'native cell identity differs: {key}')
        native_path = ROOT / fact['path']
        if not native_path.is_file() or _sha(native_path) != fact['sha256']:
            raise ValueError(f'native cache hash mismatch: {key}')
        cache_fact = result['cells'][key]
        path = ROOT / cache_fact['path']
        if not path.is_file() or _sha(path) != cache_fact['sha256']:
            raise ValueError(f'candidate cell prediction hash mismatch: {key}')
        cell = run._load_cell(key, fact, frontier_hashes)
        model_facts = declaration['training']['models']
        # Independent baseline score and exact mask/grading invariants.
        base_prediction = cell['baseline_prediction']
        _check_dx(base_prediction, cell['decoded_predecessor'], f'{key}/approved baseline')
        base_score = official._official_score(cell, base_prediction)
        _compare_rows(base_score, baseline_replay['cell_metrics'][key], f'baseline/{key}')
        baseline_rows[key] = base_score
        base_hashes[key] = hashlib.sha256(base_prediction.tobytes()).hexdigest()
        with np.load(path, allow_pickle=False) as archive:
            for field, expected in cache_fact.get('identity', {}).items():
                stored = str(np.asarray(archive[f'identity__{field}']).item())
                if stored != str(expected):
                    raise ValueError(f'candidate cache identity mismatch: {key}/{field}')
            for arm in ARMS:
                provider, _match, _view, _rep = run._cell_parts(key)
                fold = f"leave_out_{run.recovery.AUDIT.FOLD[provider]}"
                for seed in SEEDS:
                    fact_model = model_facts[f'{arm}|{fold}|{seed}']
                    identity_key = f'{arm}_seed{seed}_model_sha256'
                    if cache_fact['identity'].get(identity_key) != fact_model['sha256']:
                        raise ValueError(f'model hash not bound into prediction identity: {key}/{arm}/{seed}')
            for name in [*ARMS, *seed_names]:
                prediction = _pred(archive, name)
                if prediction.shape != base_prediction.shape:
                    raise ValueError(f'candidate prediction length mismatch: {key}/{name}')
                _check_dx(prediction, base_prediction, f'{key}/{name}')
                score = official._official_score(cell, prediction)
                _compare_rows(score,
                    (result['cell_metrics'][name][key] if name in ARMS else result['seed_diagnostic_metrics'][name][key]),
                    f'{name}/{key}')
                if name in ARMS:
                    candidate_rows[name][key] = score
                else:
                    seed_rows[name][key] = score
                digest = hashlib.sha256(prediction.tobytes()).hexdigest()
                prediction_hashes[name][key] = digest
                if digest != result['prediction_sha256'][name][key]:
                    raise ValueError(f'prediction digest mismatch: {key}/{name}')
            for arm in ARMS:
                _decode_replay(key, cell, archive, arm, arm=arm)
                for seed in SEEDS:
                    name = f'{arm}_seed{seed}'
                    _decode_replay(key, cell, archive, name, arm=arm, seed=seed)
    baseline_aggregate = _aggregate(baseline_rows)
    _compare_aggregates(baseline_aggregate, baseline_replay['aggregate'], 'baseline aggregate')
    if baseline_replay['prediction_sha256'] != base_hashes:
        raise ValueError('baseline replay prediction hashes differ')
    aggregates = {arm: _aggregate(rows) for arm, rows in candidate_rows.items()}
    seed_aggregates = {name: _aggregate(rows) for name, rows in seed_rows.items()}
    for arm in ARMS:
        _compare_aggregates(aggregates[arm], result['aggregate'][arm], f'aggregate/{arm}')
        guards = official._independent_guards(aggregates[arm], baseline_aggregate)
        saved_guards = result['current_baseline_guards'][arm]
        if guards.get('passed') != saved_guards.get('passed') or guards.get('checks', {}).keys() != saved_guards.get('checks', {}).keys():
            raise ValueError(f'guard inventory/status mismatch: {arm}')
        for label, check in guards['checks'].items():
            saved = saved_guards['checks'][label]
            if check.get('passed') != saved.get('passed'):
                raise ValueError(f'guard outcome mismatch: {arm}/{label}')
            _close(check['actual'], saved['actual'], f'guard/{arm}/{label}/actual')
            _close(check['minimum'], saved['minimum'], f'guard/{arm}/{label}/minimum')
    for name in seed_names:
        _compare_aggregates(seed_aggregates[name], result['seed_diagnostic_aggregate'][name], f'seed aggregate/{name}')
    paired = {}
    for arm in ('uniform_turnover', 'hard_turnover'):
        paired[arm] = {view: {metric: aggregates[arm][view][metric]['equal_provider'] -
            aggregates['fresh_control'][view][metric]['equal_provider']
            for metric in official.balanced.METRICS} for view in run.VIEWS}
    if paired != result['paired_fresh_control_deltas']:
        for arm in paired:
            for view in paired[arm]:
                for metric, actual in paired[arm][view].items():
                    _close(actual, result['paired_fresh_control_deltas'][arm][view][metric],
                           f'paired/{arm}/{view}/{metric}')
    training_seal = json.loads(train.FIT_SEAL.read_text())
    training_audit = _training_audit(declaration, training_seal)
    return {'schema': 'turnover_supervision_independent_readout_v1',
        'study_id': run.STUDY, 'passed': True,
        'declaration_sha256': _sha(run.DECLARATION),
        'baseline_replay_sha256': _sha(run.BASELINE_REPLAY), 'result_sha256': _sha(run.RESULT),
        'native_manifest_sha256': _sha(run.NATIVE_MANIFEST),
        'readout_code_sha256': _sha(Path(__file__)),
        'cells': 35, 'baseline_predictions': len(baseline_rows),
        'candidate_predictions': len(candidate_rows) * 35,
        'seed_diagnostic_predictions': len(seed_rows) * 35,
        'baseline_aggregate': baseline_aggregate, 'aggregate': aggregates,
        'seed_diagnostic_aggregate': seed_aggregates,
        'prediction_sha256': prediction_hashes, 'training_audit': training_audit,
        'candidate_d_probability_decode_replayed': True,
        'official_scorer': 'frontier_postprocess_readout._official_score',
        'tolerance': {'metrics': TOL, 'probability_replay': PROB_TOL},
        'organizer_reference_access': False, 'promotion': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=READOUT)
    args = parser.parse_args()
    report = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True, default=float)
        stream.write('\n')
    print(f"Verified {report['baseline_predictions']} baseline, {report['candidate_predictions']} candidate, and {report['seed_diagnostic_predictions']} seed predictions.")


if __name__ == '__main__':
    main()
