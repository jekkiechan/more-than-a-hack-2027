"""One separately requested public diagnostic after external GAR selection.

Predictions and every model/source/input seal precede the exclusive reference
access marker. The exposed public window cannot affect model selection/tuning.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import numpy as np
import joblib
from experiments import soccernet_extension as study

ROOT,DATA,OUT=study.ROOT,study.DATA,study.OUT
CSV=OUT/'competition_candidate.csv'
SEAL=OUT/'competition_prediction_seal.json'
STARTED=OUT/'competition_check_started.json'
RESULT=OUT/'competition_check_result.json'
BASELINE=ROOT/'experiments/results/combined_objective_public_check.json'
INPUTS=DATA/'challenge/inputs.npz'
INPUT_SHA='7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3'


def exclusive(path,value):
    # Unlike idempotent result receipts, a gate claim must always fail if present.
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())


def assert_open():
    if STARTED.exists() or RESULT.exists():raise ValueError('Public diagnostic gate is permanently closed')


def verify_seal(spec):
    study.check();study.seal()
    expected={str(CSV):spec['prediction_sha256'],str(INPUTS):spec['input_sha256'],
        str(OUT/'decision.json'):spec['decision_sha256'],str(BASELINE):spec['baseline_report_sha256'],
        **{str(DATA/p):v for p,v in spec['models'].items()}}
    if study.sources()!=spec['source_sha256']:raise ValueError('Source seal changed')
    for p,value in expected.items():
        if study.sha(p)!=value:raise ValueError('Public prediction seal changed: '+p)


def predict():
    from experiments import current_baseline, live_gap_experiment, soccernet_finetune as ft
    study.check();study.seal();assert_open()
    if SEAL.exists():verify_seal(study.read(SEAL));return
    decision=study.read(OUT/'decision.json');arm=decision['selected_arm']
    if study.sha(INPUTS)!=INPUT_SHA:raise ValueError('Unexpected organizer tracking input')
    models={};originals={};seals={}
    for seed in study.SEEDS:
        p=study.model_path('pooled',arm,seed);receipt=study.read(p.with_suffix('.json'))
        if study.sha(p)!=receipt['sha256']:raise ValueError('Pooled fit seal changed')
        models[seed]=joblib.load(p);originals[seed]=joblib.load(study.control_path('pooled',seed))
        for path in (p,study.control_path('pooled',seed)):seals[str(path.relative_to(DATA))]=study.sha(path)
    inputs=current_baseline.load_inputs(INPUTS)
    placeholder=np.full(len(inputs['ball']),'X',dtype='U1')
    record=study.inference_record(live_gap_experiment.feature_view(inputs,placeholder,'competition','BAR-ATH')['joint'])
    probabilities=[]
    for seed in study.SEEDS:
        sampled=ft.predict(record,models[seed],originals[seed])
        replay=ft.predict(record,models[seed],originals[seed])
        if not np.array_equal(sampled,replay):raise ValueError('Inference replay differs')
        probabilities.append(study.upsample(sampled,record,inputs))
    probability=(probabilities[0]+probabilities[1])*.5
    prediction=study.supervised.decode(probability,study.supervised.model_periods(inputs),float(inputs['fps']),study.lt.COST)
    prediction=np.asarray(prediction).astype('U1');prediction[np.asarray(inputs['period'])<=0]='X'
    if len(prediction)!=199319:raise ValueError('Unexpected prediction length')
    with CSV.open('x') as handle:
        handle.write('frame,label\n')
        for frame,label in enumerate(prediction):handle.write(f'{frame},{label}\n')
    from evaluation import possession_metrics as pm
    error=pm.validate_submission(pm.read_label_csv(CSV),199319)
    if error:raise ValueError(error)
    np.savez_compressed(OUT/'competition_probability.npz',probability=probability,prediction=prediction)
    historical=study.read(BASELINE)
    spec=dict(study_id='soccernet_extension_public_check_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        selected_arm=arm,decision_sha256=study.sha(OUT/'decision.json'),
        declaration_sha256=study.sha(OUT/'predeclared.json'),prediction_sha256=study.sha(CSV),
        input_sha256=study.sha(INPUTS),baseline_report_sha256=study.sha(BASELINE),
        models=seals,source_sha256=study.sources(),rows=199319,format_valid=True,inference_replay_exact=True,
        expected_reference_sha256=historical['organizer_gt_sha256'],
        reference_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        explicit_user_request='Extend training to include this data; optimize/tune as needed and see if it improves holdout + public spell F1.',
        organizer_reference_labels_read=False,public_used_for_selection=False,
        interpretation='One already-selected candidate; no public-driven training/tuning/promotion.')
    exclusive(SEAL,spec);print('SEALED_PUBLIC_PREDICTION',spec['prediction_sha256'],flush=True)


def score():
    if RESULT.exists():return study.read(RESULT)
    spec=study.read(SEAL);verify_seal(spec)
    exclusive(STARTED,dict(started_utc=datetime.now(timezone.utc).isoformat(),prediction_seal_sha256=study.sha(SEAL)))
    # This is the first reference access in the entire study, after the gate.
    result=subprocess.run(['git','show',f'{spec["reference_git_commit"]}:evaluation/gt/BAR-ATH_possession_gt_public.csv'],cwd=ROOT,check=True,stdout=subprocess.PIPE)
    if hashlib.sha256(result.stdout).hexdigest()!=spec['expected_reference_sha256']:raise ValueError('Reference hash differs')
    path=OUT/'public_reference.csv';path.write_bytes(result.stdout)
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    scored=pm.evaluate(pm.label_codes(pm.read_label_csv(path)),pm.label_codes(pm.read_label_csv(CSV)),bootstrap_n=1000)
    summary=_score_summary(scored)
    if (summary['graded_frames'],summary['ground_truth_spells'])!=(7500,17):raise ValueError('Unexpected public population')
    baseline=study.read(BASELINE)['scores']['combined_objective_ensemble']
    value=dict(study_id='soccernet_extension_public_check_v1',selected_arm=spec['selected_arm'],candidate=summary,saved_baseline=baseline,
        deltas={k:summary[k]-baseline[k] for k in ('spell_f1_mean','macro_f1','d_f1')},
        prediction_seal_sha256=study.sha(SEAL),external_decision_sha256=spec['decision_sha256'],
        organizer_reference_labels_read=True,public_used_for_selection=False,
        interpretation='Exposed five-minute diagnostic with only 17 reference possessions; no further tuning or adoption follows. Full-match performance remains unknown.')
    exclusive(RESULT,value);exclusive(ROOT/'experiments/results/soccernet_extension_public_check.json',value)
    print('PUBLIC_RESULT',value['deltas'],flush=True);return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('predict','score'))
    args=parser.parse_args();globals()[args.action]()

if __name__=='__main__':main()
