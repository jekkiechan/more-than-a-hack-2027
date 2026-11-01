"""Exact pooled-package GAR evaluation; VALID is exposed development evidence.

Reuses Claude's preserved GAR input assembler. No fitting, public scoring or
automatic promotion. See GAR_DEPLOYMENT_EVAL_V2.md for the frozen protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import numpy as np

from experiments import current_baseline as baseline
from experiments import feed_benchmark_audit_v1 as audit
from experiments import gar_heldout_eval_v1 as previous
from experiments import soccernet_dense_labels as dense
from experiments import sn_dense_baseline as package
from experiments import sn_dense_train as sn
from experiments import soccernet_extension as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'gar_deployment_eval_v2'
OUT = ROOT / 'data/experiments' / STUDY
RECORDS = OUT / 'records'
DECLARATION = ROOT / 'experiments/results' / f'{STUDY}_declaration.json'
RESULT = ROOT / 'experiments/results' / f'{STUDY}.json'
MANIFESTS = {
    'original_full': ROOT / 'experiments/baseline_manifest.json',
    'calibrated': ROOT / 'experiments/baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json',
}
VIEWS = ('clean', 'v2_1', 'v2_2', 'v2_3', 'v2_4')
MODES = ('phase0', 'phase_avg')
SPLITS = ('test', 'valid')
_PACKAGES = {}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')


def checkpoint_package(arm):
    if arm not in _PACKAGES:
        manifest = read(MANIFESTS[arm])
        models = baseline.load_models(manifest_path=MANIFESTS[arm])
        package.validate_models(models, manifest['architecture'])
        _PACKAGES[arm] = manifest, models
    return _PACKAGES[arm]


def declare():
    if DECLARATION.exists():
        raise ValueError('Declaration already exists; preserve the frozen protocol')
    tracking = dense.tracking_root(dense.DEFAULT_RAW)
    ids = {s: sorted({r['game_id'] for r in dense.read_split_samples(tracking, s)}) for s in SPLITS}
    if len(ids['test']) != 10 or len(ids['valid']) != 9 or set(ids['test']) & set(ids['valid']):
        raise ValueError('Unexpected or overlapping split identities')
    packages = {}
    for arm in MANIFESTS:
        manifest, models = checkpoint_package(arm)
        training = set()
        for member in package.MEMBERS:
            games = models[member]['training']['soccernet_games']
            training.update(g.removeprefix('soccernet_') for g in games)
            if len(games) != 45:
                raise ValueError('Expected the sealed 45-game TRAIN recipe')
        if training & set(ids['test'] + ids['valid']):
            raise ValueError('Evaluation match appears in learned checkpoint training')
        packages[arm] = dict(package=manifest['package_version'], manifest_sha256=digest(MANIFESTS[arm]),
            model_sha256=manifest['models_sha256'], training_game_ids=sorted(training),
            teacher_scope='Original-provider pooled combined-objective teachers; no GAR training',
            normalization='Sealed paired teacher normalization; no target-data fitting')
    inputs = read(ROOT/'experiments/results/gar_heldout_eval_v1_download_receipt.json')
    write_once(DECLARATION, dict(study=STUDY, created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does phase averaging improve exact original_full pooled deployment on held-out GAR TEST?',
        packages=packages, match_ids=ids, views=list(VIEWS), modes=list(MODES),
        source_revision=inputs['revision'], raw_files=inputs['files'],
        calibration_sha256=digest(previous.CALIBRATION),
        exposure=dict(test='TRAIN-disjoint; prior label QA, no known prior model selection/scoring in saved lineage',
            valid='All nine matches previously used for dense-model selection; development only'),
        phase_count='round(native_fps / 5): six phases on GAR 29.97 Hz, five on competition 25 Hz',
        averaging='Equal native probability mean over two sealed pooled seeds, then equal mean over phases',
        decoder='Unchanged f062, D-logit temperature 4, switch costs 3',
        primary='TEST mean per-game v2 spell F1; macro breaks exact ties; clean secondary',
        aggregation='Mean over four realizations within game, then equal mean over games; VALID separate',
        inference_tolerance=1e-6, phase0_feature_tolerance=0.0,
        bootstrap=dict(unit='whole match', draws=2000, seed=20261101),
        guards=dict(clean_spell_delta_min=-.005, clean_macro_delta_min=-.02, clean_d_f1_delta_min=-.02,
            historical_clean_improvement_screen=.01, report_all_game_deltas=True,
            no_automatic_promotion=True),
        limits=['Event-derived silver labels', 'Discontinuous event clips, not full-match coverage',
            'Same GAR provider and label recipe as TRAIN', 'True states place synthetic defects only',
            'V2 calibration uses predecessor predicted D states', 'No corruption or arm reselection'],
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in (
            Path(__file__), Path(previous.__file__), Path(dense.__file__), Path(audit.__file__),
            Path(sn.__file__), Path(package.__file__), ROOT/'experiments/feed_like_v2.py')},
        organizer_reference_access=False, public_scoring=False, fitting=False, promotion=False))


def check_seal():
    d = read(DECLARATION)
    for rel, sha in d['source_sha256'].items():
        if digest(ROOT/rel) != sha:
            raise ValueError('Declared source changed: '+rel)
    if digest(previous.CALIBRATION) != d['calibration_sha256']:
        raise ValueError('Calibration changed')
    for arm, entry in d['packages'].items():
        if digest(MANIFESTS[arm]) != entry['manifest_sha256']:
            raise ValueError('Package changed')
        checkpoint_package(arm)
    return d


def prepare():
    check_seal()
    for split in SPLITS:
        dense.prepare_split(split, output=RECORDS, workers=2)


def native_probability(record, inputs, arm):
    _, models = checkpoint_package(arm)
    record = old.inference_record(record)
    record['names'] = list(record['names'])
    members = [old.upsample(sn.predict(record, models[m], models[t]), record, inputs)
               for m, t in package.TEACHERS.items()]
    p = np.mean(members, axis=0)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1), 1, atol=1e-6, rtol=0):
        raise ValueError('Invalid probability simplex')
    return p


def run_game(task):
    split, game_id = task
    check_seal()
    target = OUT/'games'/f'{split}__{game_id}.json'
    if target.exists():
        return str(target)
    started = time.monotonic()
    records = previous.load_game_records(split, game_id, RECORDS)
    captured = previous.capture_inputs(split, game_id)
    inputs, labels, saved_x, names = previous.build_game(records, captured)
    sampled_fps = float(records[0]['fps'])
    calibration = read(previous.CALIBRATION)['calibration']
    scores, predictions, parity = {}, {}, {}
    for view in VIEWS:
        data = inputs if view == 'clean' else previous.feed_view(inputs, labels, calibration,
            STUDY+'/'+game_id, int(view[-1]))[0]
        phases = previous.phase_set(data, names, sampled_fps)
        if view == 'clean' and not np.array_equal(phases[0]['x'], saved_x, equal_nan=True):
            raise ValueError('Phase-0 features differ from frozen dense records')
        scores[view] = {}
        native = audit.tc.inputs_only(data)
        for arm in MANIFESTS:
            probabilities = [native_probability(r, native, arm) for r in phases]
            if view == 'clean':
                manifest, models = checkpoint_package(arm)
                ref = np.mean(list(package.member_probabilities(native, models, manifest).values()), axis=0)
                diff = float(np.max(np.abs(ref-probabilities[0])))
                if diff > read(DECLARATION)['inference_tolerance']:
                    raise ValueError('Deployment phase-0 parity failed')
                parity[arm] = diff
            scores[view][arm] = {}
            for mode, p in [('phase0', probabilities[0]), ('phase_avg', np.mean(probabilities, axis=0))]:
                metrics, pred = previous.score_game(labels, inputs['period'], inputs['timestamps'],
                    float(inputs['fps']), p)
                scores[view][arm][mode] = metrics
                predictions[f'{view}__{arm}__{mode}'] = pred.astype('S1')
        print('VIEW', split, game_id, view, round(time.monotonic()-started, 1), flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    path = target.with_suffix('.npz')
    np.savez_compressed(path, labels=labels.astype('S1'), period=inputs['period'],
        timestamps=inputs['timestamps'], fps=inputs['fps'], **predictions)
    write_once(target, dict(study=STUDY, split=split, game=game_id,
        declaration_sha256=digest(DECLARATION), records=len(records),
        native_frames=len(labels), graded_frames=int((labels!='X').sum()),
        graded_hours=float((labels!='X').sum()/float(inputs['fps'])/3600),
        phase_count=len(phases), phase0_feature_parity=True, deployment_probability_parity=parity,
        scores=scores, prediction_sha256=digest(path), seconds=time.monotonic()-started))
    return str(target)


def games(workers=2):
    d = check_seal()
    jobs = [(s, g) for s in SPLITS for g in d['match_ids'][s]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_game, job) for job in jobs]):
            print('DONE', future.result(), flush=True)


def aggregate(rows):
    """Realizations belong to their match; never count them as independent games."""
    metrics = sorted(set.intersection(*(set(r['scores']['clean']['original_full']['phase0']) for r in rows)))
    out = {}
    for view, subviews in [('clean', ('clean',)), ('v2', VIEWS[1:])]:
        out[view] = {}
        for arm in MANIFESTS:
            out[view][arm] = {}
            for mode in MODES:
                per_game = {r['game']: {metric: float(np.mean([
                    r['scores'][v][arm][mode][metric] for v in subviews])) for metric in metrics} for r in rows}
                out[view][arm][mode] = dict(per_game=per_game,
                    mean={m: float(np.mean([r[m] for r in per_game.values()])) for m in metrics})
    return out


def report():
    d = check_seal()
    result = dict(study=STUDY, declaration_sha256=digest(DECLARATION), splits={}, comparisons={},
        organizer_reference_access=False, public_scoring=False, fitting=False, promotion=False)
    for split in SPLITS:
        rows = [read(OUT/'games'/f'{split}__{g}.json') for g in d['match_ids'][split]]
        if any(r['declaration_sha256'] != digest(DECLARATION) for r in rows):
            raise ValueError('Mixed declarations')
        summary = aggregate(rows)
        result['splits'][split] = dict(exposure=d['exposure'][split], games=len(rows),
            records=sum(r['records'] for r in rows), graded_frames=sum(r['graded_frames'] for r in rows),
            graded_hours=sum(r['graded_hours'] for r in rows), scores=summary,
            max_phase0_probability_difference=max(max(r['deployment_probability_parity'].values()) for r in rows))
        result['comparisons'][split] = {}
        for view in ('clean', 'v2'):
            pairs = [('phase_gain', ('original_full','phase_avg'), ('original_full','phase0')),
                ('package_with_phase', ('original_full','phase_avg'), ('calibrated','phase_avg')),
                ('package_phase0', ('original_full','phase0'), ('calibrated','phase0'))]
            result['comparisons'][split][view] = {}
            for name, a, b in pairs:
                table = {'a': summary[view][a[0]][a[1]]['per_game'], 'b': summary[view][b[0]][b[1]]['per_game']}
                result['comparisons'][split][view][name] = {
                    m: previous.bootstrap(table, 'a', 'b', list(table['a']), metric=m)
                    for m in ('spell_f1','macro_f1','d_f1')}
    write_once(RESULT, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=('declare','prepare','games','report'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.step == 'games':
        games(args.workers)
    else:
        globals()[args.step]()
