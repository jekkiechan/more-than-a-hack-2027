"""Fixed three-point CE mixture follow-up using only sealed saved predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments import frontier_postprocess_readout as official
from experiments import turnover_supervision_eval as evaluation
from experiments import turnover_supervision_readout as independent
from experiments import turnover_supervision_train as training
from experiments import turnover_supervision_turnover_readout as turnover_diag

ROOT = evaluation.ROOT
STUDY = 'turnover_supervision_mix_v1'
FRACTIONS = (0.25, 0.50, 0.75)
ARMS = tuple(f'mix{int(x*100):03d}' for x in FRACTIONS)
SEEDS = evaluation.SEEDS
NATIVE_MANIFEST = evaluation.NATIVE_MANIFEST
DECLARATION = ROOT / 'experiments/results/turnover_supervision_mix_v1_declaration.json'
ENDPOINT = ROOT / 'experiments/results/turnover_supervision_mix_v1_endpoint_replay.json'
RESULT = ROOT / 'experiments/results/turnover_supervision_mix_v1.json'
READOUT = ROOT / 'experiments/results/turnover_supervision_mix_v1_readout.json'
OUT = ROOT / 'data/experiments/turnover_supervision_mix_v1'
TOL = 1e-10


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def sources() -> dict[str, str]:
    paths = (Path(__file__), Path(independent.__file__), Path(official.__file__),
        Path(evaluation.__file__), Path(training.__file__), Path(turnover_diag.__file__),
        ROOT/'experiments/tests/test_turnover_supervision_mix_v1.py',
        ROOT/'experiments/TURNOVER_SUPERVISION_MIX_V1_PLAN.md')
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def _inputs() -> dict:
    required = (evaluation.DECLARATION, evaluation.RESULT,
                evaluation.BASELINE_REPLAY, independent.READOUT,
                turnover_diag.DECLARATION, turnover_diag.RESULT,
                training.TRAIN_DECL, training.FIT_SEAL, NATIVE_MANIFEST)
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError('missing sealed upstream artifacts: ' + ', '.join(missing))
    eval_result = json.loads(evaluation.RESULT.read_text())
    baseline_replay = json.loads(evaluation.BASELINE_REPLAY.read_text())
    if (eval_result.get('passed') is not True or len(eval_result.get('cells', {})) != 35 or
            eval_result.get('declaration_sha256') != sha(evaluation.DECLARATION) or
            eval_result.get('native_manifest_sha256') != sha(NATIVE_MANIFEST) or
            baseline_replay.get('passed') is not True or
            baseline_replay.get('declaration_sha256') != sha(evaluation.DECLARATION)):
        raise ValueError('original 35-cell turnover evaluation is not complete')
    eval_decl = json.loads(evaluation.DECLARATION.read_text())
    train_decl = json.loads(training.TRAIN_DECL.read_text())
    independent_readout = json.loads(independent.READOUT.read_text())
    diag_decl = json.loads(turnover_diag.DECLARATION.read_text())
    diag_result = json.loads(turnover_diag.RESULT.read_text())
    if (independent_readout.get('passed') is not True or
            independent_readout.get('result_sha256') != sha(evaluation.RESULT) or
            diag_result.get('passed') is not True or
            diag_result.get('evaluation_result_sha256') != sha(evaluation.RESULT) or
            diag_result.get('declaration_sha256') != sha(turnover_diag.DECLARATION) or
            diag_result.get('baseline_audit_sha256') != sha(ROOT/'experiments/results/turnover_error_audit.json')):
        raise ValueError('original independent readout/turnover diagnostic is incomplete or stale')
    for inventory in (eval_decl.get('source_hashes', {}), train_decl.get('source_sha256', {}),
                      diag_decl.get('source_sha256', {})):
        if not inventory:
            raise ValueError('an upstream declaration has no source inventory')
        for relative, digest in inventory.items():
            path = (ROOT / relative).resolve()
            if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or sha(path) != digest:
                raise ValueError(f'upstream declared source changed: {relative}')
    return {
        'evaluation_declaration_sha256': sha(evaluation.DECLARATION),
        'evaluation_result_sha256': sha(evaluation.RESULT),
        'baseline_replay_sha256': sha(evaluation.BASELINE_REPLAY),
        'independent_readout_sha256': sha(independent.READOUT),
        'turnover_diagnostic_declaration_sha256': sha(turnover_diag.DECLARATION),
        'turnover_diagnostic_result_sha256': sha(turnover_diag.RESULT),
        'training_declaration_sha256': sha(training.TRAIN_DECL),
        'training_seal_sha256': sha(training.FIT_SEAL),
        'native_manifest_sha256': sha(NATIVE_MANIFEST),
        'candidate_cells': sorted(eval_result['cells']),
    }


def freeze() -> dict:
    if any(path.exists() for path in (DECLARATION, ENDPOINT, RESULT, READOUT)):
        raise FileExistsError('preserve existing mix follow-up evidence')
    inputs = _inputs()
    cell_dir = OUT / 'cells'
    if cell_dir.exists() and any(cell_dir.glob('*.npz')):
        raise FileExistsError('refuse to freeze after any mix prediction file exists')
    declaration = {
        'schema': 'turnover_supervision_mix_v1_declaration', 'study_id': STUDY,
        'status': 'frozen_before_mix_metrics', 'created_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        'question': 'Do fixed coarse blends of fresh-control and hard-turnover CE retain complementary turnover supervision while meeting existing baseline guards?',
        'fractions_hard_turnover': list(FRACTIONS),
        'formula': 'CE_mix=(1-alpha)*CE_fresh_control+alpha*CE_hard_turnover, per native phase then pooled with the existing mean-seed/mean-phase order; apply approved conditional mixture CE .5/calibrated .375/predecessor .125 and fixed-D HA-cost-3 decoder.',
        'baseline_endpoint_replay': 'Re-score fresh_control, uniform_turnover, hard_turnover over all 35 cells against the saved official rows and verify exact D/X masks before making blends.',
        'guard_policy': 'Unchanged current baseline guards: strict feed-like-v2 spell-F1 improvement; clean and feed-like-v1 spell-F1 no regression; short-spell recall no regression in every view; view macro-F1 >= baseline-.001; clean provider macro-F1 and D-F1 >= baseline-.02.',
        'scope': 'Three fixed pooled candidate blends; per-seed blend predictions are descriptive only. No model inference, fitting, candidate search, GAR VALID/TEST, organizer labels, automatic promotion, or new independent confirmation.',
        'inputs': inputs, 'source_sha256': sources(),
        'organizer_reference_access': False, 'promotion': False,
    }
    write_once(DECLARATION, declaration)
    return declaration


def _conditional(probability: np.ndarray) -> np.ndarray:
    arr = np.asarray(probability, dtype=np.float64)
    ha = arr[:, :2]
    mass = ha.sum(axis=1, keepdims=True)
    if arr.ndim != 2 or arr.shape[1] != 3 or not np.isfinite(arr).all() or np.any(mass <= 0):
        raise ValueError('CE/family probabilities must be valid native Nx3')
    return ha / mass


def _mix(ce: np.ndarray, calibrated: np.ndarray, predecessor: np.ndarray) -> np.ndarray:
    parts = .5*_conditional(ce) + .375*_conditional(calibrated) + .125*_conditional(predecessor)
    return parts / parts.sum(axis=1, keepdims=True)


def _assert_probability(actual: np.ndarray, expected: np.ndarray, where: str) -> None:
    if np.asarray(actual).shape != np.asarray(expected).shape or not np.allclose(
            actual, expected, rtol=0, atol=TOL):
        raise ValueError(f'probability replay mismatch: {where}')


def _rebuild_ce(archive, arm: str, phase_count: int, seed: int | None = None) -> np.ndarray:
    phases = []
    for phase in range(phase_count):
        if seed is None:
            left = np.asarray(archive[f'prob__ce__{arm}__seed2026__phase{phase}'], dtype=np.float64)
            right = np.asarray(archive[f'prob__ce__{arm}__seed2027__phase{phase}'], dtype=np.float64)
            phases.append((left + right) / 2.0)
        else:
            phases.append(np.asarray(archive[f'prob__ce__{arm}__seed{seed}__phase{phase}'], dtype=np.float64))
    return np.mean(np.stack(phases), axis=0)


def _rebuild_mix(cell, archive, alpha: float, arm_name: str, seed: int | None = None) -> np.ndarray:
    phase_count = len(cell['components']['predecessor'][2026])
    fresh = _rebuild_ce(archive, 'fresh_control', phase_count, seed)
    hard = _rebuild_ce(archive, 'hard_turnover', phase_count, seed)
    ce = blend_ce(fresh, hard, alpha)
    cal = official._pool_family(cell['components']['calibrated'], .5)
    pred = official._pool_family(cell['components']['predecessor'], .5)
    return _mix(ce, cal, pred)


def blend_ce(fresh: np.ndarray, hard: np.ndarray, alpha: float) -> np.ndarray:
    if alpha not in FRACTIONS:
        raise ValueError('alpha must be one of the frozen fractions')
    fresh, hard = np.asarray(fresh, dtype=np.float64), np.asarray(hard, dtype=np.float64)
    if fresh.shape != hard.shape or fresh.ndim != 2 or fresh.shape[1] != 3:
        raise ValueError('fresh and hard CE arrays must be aligned native Nx3')
    return (1.0-float(alpha))*fresh + float(alpha)*hard


def endpoint_replay() -> dict:
    if not DECLARATION.is_file() or ENDPOINT.exists():
        raise FileExistsError('freeze first and preserve endpoint replay')
    declaration = json.loads(DECLARATION.read_text())
    if declaration['source_sha256'] != sources() or declaration['inputs'] != _inputs():
        raise ValueError('frozen input/source inventory changed')
    result = json.loads(evaluation.RESULT.read_text())
    native_manifest = json.loads(NATIVE_MANIFEST.read_text())
    frontier_hashes = evaluation._frontier_hashes()
    rows = {arm: {} for arm in evaluation.ARMS}
    hashes = {arm: {} for arm in evaluation.ARMS}
    for key in declaration['inputs']['candidate_cells']:
        cell = evaluation._load_cell(key, native_manifest['cells'][key], frontier_hashes)
        fact = result['cells'][key]
        pred_path = ROOT / fact['path']
        if sha(pred_path) != fact['sha256']:
            raise ValueError(f'original prediction file hash changed: {key}')
        with np.load(pred_path, allow_pickle=False) as z:
            for arm in evaluation.ARMS:
                pred = official._read_labels(z[f'pred__{arm}'], arm)
                if not np.array_equal(pred == 'D', cell['baseline_prediction'] == 'D') or not np.array_equal(pred == 'X', cell['baseline_prediction'] == 'X'):
                    raise ValueError(f'original arm changed D/X: {key}/{arm}')
                score = official._official_score(cell, pred)
                official._compare_row(score, result['cell_metrics'][arm][key], f'endpoint/{key}/{arm}')
                rows[arm][key] = score
                hashes[arm][key] = hashlib.sha256(pred.tobytes()).hexdigest()
                if hashes[arm][key] != result['prediction_sha256'][arm][key]:
                    raise ValueError(f'original prediction hash mismatch: {key}/{arm}')
    for arm in evaluation.ARMS:
        aggregate = official._aggregate(rows[arm], evaluation.VIEWS)
        for view in evaluation.VIEWS:
            for metric in official.balanced.METRICS:
                a = aggregate[view][metric]['equal_provider']
                b = result['aggregate'][arm][view][metric]['equal_provider']
                if abs(a-b) > TOL:
                    raise ValueError(f'original aggregate replay mismatch: {arm}/{view}/{metric}')
    receipt = {'schema':'turnover_supervision_mix_v1_endpoint_replay','passed':True,
        'study_id':STUDY,'declaration_sha256':sha(DECLARATION),
        'evaluation_result_sha256':sha(evaluation.RESULT),'cells':len(rows['fresh_control']),
        'arm_rows':{arm:len(value) for arm,value in rows.items()},
        'prediction_sha256':hashes,'official_scores_replayed':True,
        'exact_DX_masks':True,'organizer_reference_access':False}
    write_once(ENDPOINT, receipt)
    return receipt


def evaluate() -> dict:
    if RESULT.exists():
        raise FileExistsError('preserve existing mix result')
    declaration = json.loads(DECLARATION.read_text())
    endpoint = json.loads(ENDPOINT.read_text())
    if (endpoint.get('schema') != 'turnover_supervision_mix_v1_endpoint_replay' or
            endpoint.get('passed') is not True or endpoint.get('declaration_sha256') != sha(DECLARATION) or
            endpoint.get('evaluation_result_sha256') != sha(evaluation.RESULT) or
            endpoint.get('cells') != 35 or endpoint.get('arm_rows') != {a: 35 for a in evaluation.ARMS}):
        raise ValueError('successful endpoint replay required before mix metrics')
    if declaration['source_sha256'] != sources() or declaration['inputs'] != _inputs():
        raise ValueError('frozen source/input inventory changed')
    if ENDPOINT.exists() and endpoint.get('evaluation_result_sha256') != sha(evaluation.RESULT):
        raise ValueError('endpoint replay is stale against the sealed evaluation')
    output_cells = OUT / 'cells'
    if output_cells.exists() and any(output_cells.glob('*.npz')):
        raise FileExistsError('refuse to overwrite or resume partial mix cell outputs')
    original = json.loads(evaluation.RESULT.read_text())
    baseline_replay = json.loads(evaluation.BASELINE_REPLAY.read_text())
    native = json.loads(NATIVE_MANIFEST.read_text())
    frontier_hashes = evaluation._frontier_hashes()
    cells_out, rows = {}, {arm: {} for arm in ARMS}
    seed_rows = {f'{arm}_seed{seed}': {} for arm in ARMS for seed in SEEDS}
    for key in declaration['inputs']['candidate_cells']:
        cell = evaluation._load_cell(key, native['cells'][key], frontier_hashes)
        source_fact = original['cells'][key]
        source_path = ROOT / source_fact['path']
        if sha(source_path) != source_fact['sha256']:
            raise ValueError(f'original candidate probability file changed: {key}')
        outputs = {}
        with np.load(source_path, allow_pickle=False) as z:
            count = len(cell['components']['predecessor'][2026])
            calibrated = official._pool_family(cell['components']['calibrated'], .5)
            predecessor = official._pool_family(cell['components']['predecessor'], .5)
            fresh = np.asarray(z['ce__fresh_control'], dtype=np.float64)
            hard = np.asarray(z['ce__hard_turnover'], dtype=np.float64)
            # Eval stores phase arrays for every arm; verify its pooled CE before mixing.
            fresh_rebuilt = np.mean(np.stack([np.mean(np.stack([
                np.asarray(z[f'prob__ce__fresh_control__seed{s}__phase{p}'], dtype=np.float64)
                for s in SEEDS]),axis=0) for p in range(count)]),axis=0)
            hard_rebuilt = np.mean(np.stack([np.mean(np.stack([
                np.asarray(z[f'prob__ce__hard_turnover__seed{s}__phase{p}'], dtype=np.float64)
                for s in SEEDS]),axis=0) for p in range(count)]),axis=0)
            _assert_probability(fresh, fresh_rebuilt, f'{key}/fresh CE pooling')
            _assert_probability(hard, hard_rebuilt, f'{key}/hard CE pooling')
            for arm, alpha in zip(ARMS, FRACTIONS):
                mixed_ce = blend_ce(fresh, hard, alpha)
                probability = _mix(mixed_ce, calibrated, predecessor)
                prediction = official._decode(probability, cell['decoded_predecessor'], cell, evaluation.HA_COST)
                if not np.array_equal(prediction == 'D', cell['baseline_prediction'] == 'D') or not np.array_equal(prediction == 'X', cell['baseline_prediction'] == 'X'):
                    raise ValueError(f'mix altered fixed D/X: {key}/{arm}')
                outputs[f'prob__{arm}'] = probability.astype(np.float64)
                outputs[f'ce__{arm}'] = mixed_ce.astype(np.float64)
                outputs[f'pred__{arm}'] = prediction.astype('U1')
                score = official._official_score(cell, prediction)
                rows[arm][key] = score
                baseline_counts = baseline_replay['cell_metrics'][key]
                for count_name in ('graded_frames','reference_spell_count','short_reference_spell_count'):
                    if score[count_name] != baseline_counts[count_name]:
                        raise ValueError(f'mix changed fixed-mask score count: {key}/{arm}/{count_name}')
                for seed in SEEDS:
                    # Per-seed CE averages its own native phases; D families
                    # remain the complete two-seed pools from the approved run.
                    hs = np.mean(np.stack([np.asarray(z[f'prob__ce__hard_turnover__seed{seed}__phase{p}'],dtype=np.float64) for p in range(count)]),axis=0)
                    fs = np.mean(np.stack([np.asarray(z[f'prob__ce__fresh_control__seed{seed}__phase{p}'],dtype=np.float64) for p in range(count)]),axis=0)
                    seed_probability = _mix(blend_ce(fs, hs, alpha), calibrated, predecessor)
                    seed_prediction = official._decode(seed_probability, cell['decoded_predecessor'], cell, evaluation.HA_COST)
                    sname=f'{arm}_seed{seed}'
                    outputs[f'prob__{sname}']=seed_probability.astype(np.float64)
                    outputs[f'ce__{sname}']=blend_ce(fs,hs,alpha).astype(np.float64)
                    outputs[f'pred__{sname}']=seed_prediction.astype('U1')
                    seed_rows[sname][key]=official._official_score(cell,seed_prediction)
        out_path = OUT/'cells'/f'{key.replace("|","__")}.npz'
        out_path.parent.mkdir(parents=True,exist_ok=True)
        if out_path.exists(): raise FileExistsError(f'refuse to overwrite {out_path}')
        temp = out_path.with_name(f'.{out_path.name}.{os.getpid()}.tmp.npz')
        if temp.exists(): raise FileExistsError(f'refuse to overwrite {temp}')
        outputs['identity__cell'] = np.asarray(key)
        outputs['identity__source_prediction_sha256'] = np.asarray(source_fact['sha256'])
        outputs['identity__source_native_sha256'] = np.asarray(native['cells'][key]['sha256'])
        np.savez_compressed(temp,**outputs)
        os.replace(temp, out_path)
        cells_out[key]={'path':str(out_path.relative_to(ROOT)),'sha256':sha(out_path),
                        'source_prediction_sha256':source_fact['sha256'],
                        'source_native_sha256':native['cells'][key]['sha256']}
    baseline=json.loads(evaluation.BASELINE_REPLAY.read_text())['aggregate']
    aggregates={arm:official._aggregate(rows[arm],evaluation.VIEWS) for arm in ARMS}
    seeds={name:official._aggregate(v,evaluation.VIEWS) for name,v in seed_rows.items()}
    guards={arm:official._independent_guards(aggregates[arm],baseline) for arm in ARMS}
    report={'schema':'turnover_supervision_mix_v1_result','study_id':STUDY,'passed':True,
        'declaration_sha256':sha(DECLARATION),'endpoint_replay_sha256':sha(ENDPOINT),
        'evaluation_result_sha256':sha(evaluation.RESULT),'cells':cells_out,
        'aggregate':aggregates,'cell_metrics':rows,
        'seed_diagnostic_aggregate':seeds,'seed_diagnostic_cell_metrics':seed_rows,
        'baseline_aggregate':baseline,
        'guards':guards,'candidate_fractions':dict(zip(ARMS,FRACTIONS)),
        'organizer_reference_access':False,'promotion':False}
    write_once(RESULT,report)
    return report


def readout() -> dict:
    if READOUT.exists():
        raise FileExistsError('preserve existing independent mix readout')
    declaration=json.loads(DECLARATION.read_text()); result=json.loads(RESULT.read_text())
    if (declaration['source_sha256']!=sources() or result.get('declaration_sha256')!=sha(DECLARATION) or
            result.get('passed') is not True or result.get('evaluation_result_sha256')!=sha(evaluation.RESULT) or
            result.get('endpoint_replay_sha256')!=sha(ENDPOINT) or set(result.get('cells',{}))!=set(declaration['inputs']['candidate_cells'])):
        raise ValueError('mix declaration/result binding mismatch')
    if _inputs()!=declaration['inputs']:
        raise ValueError('upstream source/input binding changed after mix freeze')
    native=json.loads(NATIVE_MANIFEST.read_text()); rows={a:{} for a in ARMS}; seedrows={}
    frontier_hashes=evaluation._frontier_hashes()
    original=json.loads(evaluation.RESULT.read_text())
    for key,fact in result['cells'].items():
        cell=evaluation._load_cell(key,native['cells'][key],frontier_hashes)
        source_fact=original['cells'][key];source_path=ROOT/source_fact['path']
        if sha(source_path)!=source_fact['sha256'] or fact.get('source_prediction_sha256')!=source_fact['sha256'] or fact.get('source_native_sha256')!=native['cells'][key]['sha256']:
            raise ValueError(f'mix source prediction/native binding mismatch: {key}')
        path=ROOT/fact['path']
        if not path.resolve().is_relative_to((OUT/'cells').resolve()):
            raise ValueError(f'mix cell path escapes private output directory: {key}')
        if sha(path)!=fact['sha256']: raise ValueError(f'mix output hash mismatch: {key}')
        baseline_counts=json.loads(evaluation.BASELINE_REPLAY.read_text())['cell_metrics'][key]
        with np.load(source_path,allow_pickle=False) as source, np.load(path,allow_pickle=False) as z:
            if str(np.asarray(z['identity__cell']).item())!=key or \
                    str(np.asarray(z['identity__source_prediction_sha256']).item())!=source_fact['sha256'] or \
                    str(np.asarray(z['identity__source_native_sha256']).item())!=native['cells'][key]['sha256']:
                raise ValueError(f'mix cell identity mismatch: {key}')
            for name in (*ARMS,*(f'{a}_seed{s}' for a in ARMS for s in SEEDS)):
                alpha=FRACTIONS[ARMS.index(name.split('_seed')[0])] if '_seed' in name else FRACTIONS[ARMS.index(name)]
                arm=name.split('_seed')[0]
                seed=int(name.rsplit('seed',1)[1]) if '_seed' in name else None
                rebuilt=_rebuild_mix(cell,source,alpha,arm,seed)
                saved_prob=np.asarray(z[f'prob__{name}'],dtype=np.float64)
                _assert_probability(saved_prob, rebuilt, f'{key}/{name}')
                if not np.allclose(np.asarray(z[f'ce__{name}'],dtype=np.float64),
                                   blend_ce(_rebuild_ce(source,'fresh_control',len(cell['components']['predecessor'][2026]),seed),
                                            _rebuild_ce(source,'hard_turnover',len(cell['components']['predecessor'][2026]),seed),alpha),rtol=0,atol=TOL):
                    raise ValueError(f'independent CE blend reconstruction failed: {key}/{name}')
                decoded=official._decode(rebuilt,cell['decoded_predecessor'],cell,evaluation.HA_COST)
                saved_pred=official._read_labels(z[f'pred__{name}'],name)
                if not np.array_equal(decoded,saved_pred): raise ValueError(f'independent decode replay failed: {key}/{name}')
                if not np.array_equal(saved_pred=='D',cell['baseline_prediction']=='D') or not np.array_equal(saved_pred=='X',cell['baseline_prediction']=='X'):
                    raise ValueError(f'mix D/X mask mismatch: {key}/{name}')
                score=official._official_score(cell,saved_pred)
                for count in ('graded_frames','reference_spell_count','short_reference_spell_count'):
                    if score[count]!=baseline_counts[count]: raise ValueError(f'mix grading count changed: {key}/{name}/{count}')
                official._compare_row(score,(result['cell_metrics'][name][key] if name in ARMS else result['seed_diagnostic_cell_metrics'][name][key]),f'{key}/{name}')
                if name in ARMS: rows[name][key]=score
                else: seedrows.setdefault(name,{})[key]=score
    aggregates={a:official._aggregate(rows[a],evaluation.VIEWS) for a in ARMS}
    seed_aggregates={n:official._aggregate(v,evaluation.VIEWS) for n,v in seedrows.items()}
    baseline=json.loads(evaluation.BASELINE_REPLAY.read_text())['aggregate']
    guards={a:official._independent_guards(aggregates[a],baseline) for a in ARMS}
    for arm in ARMS:
        if guards[arm] != result['guards'][arm]: raise ValueError(f'mix guard replay mismatch: {arm}')
        official._compare_row(aggregates[arm], result['aggregate'][arm], f'mix aggregate/{arm}')
        official._compare_row(rows[arm], result['cell_metrics'][arm], f'mix cell rows/{arm}')
        for view in evaluation.VIEWS:
            for metric in official.balanced.METRICS:
                if abs(aggregates[arm][view][metric]['equal_provider']-
                       result['aggregate'][arm][view][metric]['equal_provider']) > TOL:
                    raise ValueError(f'mix aggregate replay mismatch: {arm}/{view}/{metric}')
    for name in seedrows:
        official._compare_row(seed_aggregates[name], result['seed_diagnostic_aggregate'][name], f'seed aggregate/{name}')
        official._compare_row(seedrows[name], result['seed_diagnostic_cell_metrics'][name], f'seed rows/{name}')
    report={'schema':'turnover_supervision_mix_v1_readout','study_id':STUDY,
        'passed':True,'declaration_sha256':sha(DECLARATION),'result_sha256':sha(RESULT),
        'aggregate':aggregates,'guards':guards,
        'retained_cell_rows':sum(map(len,rows.values())),
        'seed_diagnostic_cell_rows':sum(map(len,seedrows.values())),
        'probability_decode_replayed':True,
        'organizer_reference_access':False,'promotion':False}
    write_once(READOUT,report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('freeze','endpoint','evaluate','readout'));args=parser.parse_args()
    value={'freeze':freeze,'endpoint':endpoint_replay,'evaluate':evaluate,'readout':readout}[args.action]()
    print(json.dumps(value,indent=2,sort_keys=True,default=float))


if __name__=='__main__':main()
