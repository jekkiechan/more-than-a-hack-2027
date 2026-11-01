"""Pre-score execution amendment preserving package-specific seed arithmetic.

The initial source/declaration remain unchanged. GAR phase0 parity uses the
independently assembled dense records, preserving their native 29.97 Hz rate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import time

import numpy as np

from experiments import gar_deployment_eval_v2 as base

AMENDMENT = base.ROOT/'experiments/results/gar_deployment_eval_v2_execution_amendment.json'


def amend():
    base.check_seal()
    base.write_once(AMENDMENT, dict(study=base.STUDY, created_utc=datetime.now(timezone.utc).isoformat(),
        original_declaration_sha256=base.digest(base.DECLARATION), model_scores_observed=False,
        reason=['GAR native rate fails the ordinary deployment integer-multiple-of-5 sampler',
            'Calibrated package normalizes seeds before pooling; original_full pools before normalization'],
        correction=['Use independently prepared native dense records for GAR phase0 feature/probability/decoded parity',
            'Preserve native timestamps and fps; no 30-Hz metadata substitution',
            'Keep per-seed native arrays until package-specific decoding',
            'Original_full: raw mean over phases and seeds, then normalize/decode as one member',
            'Calibrated: mean phases within each seed, then pass two arrays to frozen decoder',
            'Assert unchanged graded/reference counts across every arm, mode and view'],
        remaining_limit='Frozen v2 simulator treats unknown X context as live for defect placement; X remains ungraded',
        source_sha256={str(Path(__file__).relative_to(base.ROOT)):base.digest(__file__)},
        scoring_budget_changed=False, fitting=False, public_scoring=False, promotion=False))


def check():
    declaration = base.check_seal()
    amendment = base.read(AMENDMENT)
    if amendment['original_declaration_sha256'] != base.digest(base.DECLARATION):
        raise ValueError('Original declaration changed')
    for rel, expected in amendment['source_sha256'].items():
        if base.digest(base.ROOT/rel) != expected:
            raise ValueError('Amended source changed')
    return declaration


def seed_probabilities(record, inputs, arm):
    _, models = base.checkpoint_package(arm)
    record = base.old.inference_record(record)
    record['names'] = list(record['names'])
    values = np.stack([base.old.upsample(base.sn.predict(record, models[m], models[t]), record, inputs)
        for m,t in base.package.TEACHERS.items()])
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(2), 1, atol=1e-6, rtol=0)):
        raise ValueError('Invalid per-seed probability simplex')
    return values


def decode(seeds, arm, period, fps):
    members = np.mean(seeds, axis=0)[None] if arm == 'original_full' else seeds
    return base.audit.decoder.decode(members, period, fps, base.audit.CONFIG)


def evaluate_view(inputs, labels, records, saved_x, names, view):
    phases = base.previous.phase_set(inputs, names, float(records[0]['fps']))
    if view == 'clean' and not np.array_equal(phases[0]['x'], saved_x, equal_nan=True):
        raise ValueError('Prepared phase0 feature parity failed')
    native = base.audit.tc.inputs_only(inputs)
    period = base.audit.supervised.model_periods(native)
    fps = float(inputs['fps'])
    scores, predictions, parity = {}, {}, {}
    for arm in base.MANIFESTS:
        values = [seed_probabilities(r, native, arm) for r in phases]
        if view == 'clean':
            # Independent segmentation/feature path: original saved dense records.
            reference = np.concatenate([seed_probabilities(r,
                dict(period=r['full_period'], timestamps=r['full_timestamps'], fps=float(r['native_fps'])), arm)
                for r in records], axis=1)
            difference = float(np.max(np.abs(values[0]-reference)))
            if difference > base.read(base.DECLARATION)['inference_tolerance']:
                raise ValueError('Prepared-record seed probability parity failed')
            ref_labels = decode(reference, arm, period, fps)
            actual_labels = decode(values[0], arm, period, fps)
            if not np.array_equal(ref_labels, actual_labels):
                raise ValueError('Package-specific decoded phase0 parity failed')
            parity[arm] = difference
        scores[arm] = {}
        for mode, seeds in [('phase0',values[0]), ('phase_avg',np.mean(values,axis=0))]:
            pred = decode(seeds, arm, period, fps)
            cell = dict(labels=labels,period=inputs['period'],timestamps=inputs['timestamps'],fps=fps)
            measured = base.audit.score(cell,pred)
            scores[arm][mode] = {k: measured[k] for k in base.previous.KEYS if k in measured}
            predictions[f'{view}__{arm}__{mode}'] = pred.astype('S1')
    return scores, predictions, parity, len(phases)


def run_game(task):
    split, game_id = task
    check()
    target = base.OUT/'games'/f'{split}__{game_id}.json'
    if target.exists():
        result = base.read(target)
        if result.get('execution_amendment_sha256') != base.digest(AMENDMENT):
            raise ValueError('Existing output has different execution provenance')
        return str(target)
    started = time.monotonic()
    records = base.previous.load_game_records(split,game_id,base.RECORDS)
    inputs,labels,saved_x,names = base.previous.build_game(records,base.previous.capture_inputs(split,game_id))
    scores,predictions,parity = {},{},{}
    calibration = base.read(base.previous.CALIBRATION)['calibration']
    for view in base.VIEWS:
        data = inputs if view == 'clean' else base.previous.feed_view(inputs,labels,calibration,
            base.STUDY+'/'+game_id,int(view[-1]))[0]
        scores[view],preds,replay,phase_count = evaluate_view(data,labels,records,saved_x,names,view)
        predictions.update(preds)
        parity.update(replay)
        print('VIEW',split,game_id,view,round(time.monotonic()-started,1),flush=True)
    counts = {(r['graded_frames'],r['reference_spell_count'])
        for v in scores.values() for a in v.values() for r in a.values()}
    if len(counts) != 1:
        raise ValueError('Grading or reference spell counts changed across cells')
    target.parent.mkdir(parents=True,exist_ok=True)
    native_path = target.with_suffix('.npz')
    np.savez_compressed(native_path,labels=labels.astype('S1'),period=inputs['period'],
        timestamps=inputs['timestamps'],fps=inputs['fps'],**predictions)
    base.write_once(target,dict(study=base.STUDY,split=split,game=game_id,
        declaration_sha256=base.digest(base.DECLARATION),execution_amendment_sha256=base.digest(AMENDMENT),
        records=len(records),native_frames=len(labels),graded_frames=int((labels!='X').sum()),
        graded_hours=float((labels!='X').sum()/float(inputs['fps'])/3600),phase_count=phase_count,
        phase0_feature_parity=True,phase0_decoded_parity=True,grading_and_reference_counts_equal=True,
        deployment_probability_parity=parity,scores=scores,prediction_sha256=base.digest(native_path),
        seconds=time.monotonic()-started))
    return str(target)


def games(workers=2):
    d = check()
    jobs = [(s,g) for s in base.SPLITS for g in d['match_ids'][s]]
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_game,t) for t in jobs]):
            print('DONE',future.result(),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('step',choices=('amend','games','report'))
    parser.add_argument('--workers',type=int,default=2)
    args = parser.parse_args()
    if args.step == 'amend':
        amend()
    elif args.step == 'games':
        games(args.workers)
    else:
        check()
        base.report()
