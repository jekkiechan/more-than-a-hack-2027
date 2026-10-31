"""One public diagnostic after complete capacity/learning-rate selection.

This module is the sole organizer-reference entrypoint for the capacity study.
No public prediction or scoring is allowed until external reports, attribution
control and pooled model seals are complete. All historical gates stay closed.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import subprocess
import numpy as np
import joblib
from experiments import soccernet_capacity as study, soccernet_capacity_model as model

ROOT,DATA,OUT=study.ROOT,study.DATA,study.OUT
CSV=OUT/'competition_candidate.csv'
SEAL=OUT/'competition_prediction_seal.json'
STARTED=OUT/'competition_check_started.json'
RESULT=OUT/'competition_check_result.json'
BASELINE=ROOT/'experiments/results/combined_objective_public_check.json'
INPUTS=DATA/'challenge/inputs.npz'
INPUT_SHA='7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3'


def exclusive(path,value):
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as handle:
        json.dump(study._jsonable(value),handle,indent=2,allow_nan=False)
        handle.write('\n');handle.flush();os.fsync(handle.fileno())


def assert_external_complete():
    study.check()
    study._check_seal('coarse_checkpoints_frozen.json')
    study._check_seal('checkpoints_confirm_frozen.json')
    study._check_seal('control_checkpoints_frozen.json')
    study._check_seal('pooled_checkpoints_frozen.json')
    for name in ('coarse_report.json','coarse_decision.json','final_report.json',
                 'final_decision.json','control_report.json','public_ready.json'):
        if not (OUT/name).is_file():raise ValueError('External evidence incomplete: '+name)


def assert_open():
    if STARTED.exists() or RESULT.exists():raise ValueError('Public diagnostic gate is permanently closed')


def verify_seal(spec):
    assert_external_complete()
    expected={str(CSV):spec['prediction_sha256'],str(INPUTS):spec['input_sha256'],
              str(BASELINE):spec['baseline_report_sha256']}
    expected.update({str(OUT/name):value for name,value in spec['external_evidence_sha256'].items()})
    expected.update({str(DATA/name):value for name,value in spec['models_sha256'].items()})
    if study._source_hashes()!=spec['source_sha256']:raise ValueError('Public source seal changed')
    for path,value in expected.items():
        if study.sha256(Path(path))!=value:raise ValueError('Public prediction seal changed: '+path)


def predict():
    from experiments import current_baseline,live_gap_experiment
    assert_external_complete();assert_open()
    if SEAL.exists():verify_seal(study.read_json(SEAL));return
    decision=study.read_json(OUT/'final_decision.json')
    width=int(decision['selected_width']);recipe=study._selected_coarse()[width]
    if study.sha256(INPUTS)!=INPUT_SHA:raise ValueError('Unexpected organizer input hash')
    inputs=current_baseline.load_inputs(INPUTS)
    placeholder=np.full(len(inputs['ball']),'X',dtype='U1')
    record=study.old.inference_record(live_gap_experiment.feature_view(inputs,placeholder,'competition','BAR-ATH')['joint'])
    members=[];seals={}
    for seed in study.SEEDS:
        path=study.pooled_path(width,recipe['lr'],recipe['steps'],seed)
        tuned=study._load_sealed_model(path)
        initial_path=study.initial_path('pooled',seed);initial=joblib.load(initial_path)
        probability=model.predict(record,tuned,initial)
        replay=model.predict(record,tuned,initial)
        if not np.array_equal(probability,replay):raise ValueError('Capacity inference is not exactly reproducible')
        members.append(study.old.upsample(probability,record,inputs))
        for p in (path,initial_path):seals[str(p.relative_to(DATA))]=study.sha256(p)
    probability=(members[0]+members[1])*.5
    prediction=study.supervised.decode(probability,study.supervised.model_periods(inputs),float(inputs['fps']),study.lt.COST)
    prediction=np.asarray(prediction).astype('U1');prediction[np.asarray(inputs['period'])<=0]='X'
    if len(prediction)!=199319:raise ValueError('Unexpected organizer row count')
    with CSV.open('x') as handle:
        handle.write('frame,label\n')
        for frame,label in enumerate(prediction):handle.write(f'{frame},{label}\n')
    from evaluation import possession_metrics as pm
    error=pm.validate_submission(pm.read_label_csv(CSV),199319)
    if error:raise ValueError(error)
    np.savez_compressed(OUT/'competition_probability.npz',probability=probability,prediction=prediction)
    historical=study.read_json(BASELINE)
    evidence_names=('predeclared.json','coarse_report.json','coarse_decision.json',
                    'final_report.json','final_decision.json','control_report.json',
                    'coarse_checkpoints_frozen.json','checkpoints_confirm_frozen.json',
                    'control_checkpoints_frozen.json','pooled_checkpoints_frozen.json','public_ready.json')
    spec=dict(study_id='soccernet_capacity_public_check_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        selected_width=width,selected_recipe=recipe['recipe'],
        external_evidence_sha256={name:study.sha256(OUT/name) for name in evidence_names},
        prediction_sha256=study.sha256(CSV),input_sha256=study.sha256(INPUTS),
        baseline_report_sha256=study.sha256(BASELINE),models_sha256=seals,
        source_sha256=study._source_hashes(),rows=199319,format_valid=True,inference_replay_exact=True,
        expected_reference_sha256=historical['organizer_gt_sha256'],
        reference_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        explicit_user_request='Include SoccerNet data; larger models and more tuning; audit preprocessing; compare holdout and public spell F1.',
        organizer_reference_labels_read=False,public_used_for_selection=False,
        interpretation='All model/hyperparameter/step/control choices sealed before reference access; no post-public tuning or automatic promotion.')
    exclusive(SEAL,spec);print('SEALED_CAPACITY_PUBLIC_PREDICTION',spec['prediction_sha256'],flush=True)


def score():
    if RESULT.exists():return study.read_json(RESULT)
    spec=study.read_json(SEAL);verify_seal(spec)
    exclusive(STARTED,dict(started_utc=datetime.now(timezone.utc).isoformat(),prediction_seal_sha256=study.sha256(SEAL)))
    raw=subprocess.run(['git','show',f'{spec["reference_git_commit"]}:evaluation/gt/BAR-ATH_possession_gt_public.csv'],cwd=ROOT,check=True,stdout=subprocess.PIPE).stdout
    if hashlib.sha256(raw).hexdigest()!=spec['expected_reference_sha256']:raise ValueError('Reference hash differs')
    reference=OUT/'public_reference.csv';reference.write_bytes(raw)
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    value=pm.evaluate(pm.label_codes(pm.read_label_csv(reference)),pm.label_codes(pm.read_label_csv(CSV)),bootstrap_n=1000)
    summary=_score_summary(value)
    if (summary['graded_frames'],summary['ground_truth_spells'])!=(7500,17):raise ValueError('Unexpected public population')
    baseline=study.read_json(BASELINE)['scores']['combined_objective_ensemble']
    result=dict(study_id='soccernet_capacity_public_check_v1',selected_width=spec['selected_width'],
        selected_recipe=spec['selected_recipe'],candidate=summary,saved_baseline=baseline,
        deltas={k:summary[k]-baseline[k] for k in ('spell_f1_mean','macro_f1','d_f1')},
        prediction_seal_sha256=study.sha256(SEAL),external_decision_sha256=spec['external_evidence_sha256']['final_decision.json'],
        organizer_reference_labels_read=True,public_used_for_selection=False,
        interpretation='Exposed five-minute pipeline diagnostic, 17 reference possessions; no model tuning/selection/adoption follows this score. Full-match performance remains unknown.')
    exclusive(RESULT,result)
    exclusive(ROOT/'experiments/results/soccernet_capacity_public_check.json',result)
    print('CAPACITY_PUBLIC_RESULT',result['deltas'],flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('predict','score'))
    args=p.parse_args();globals()[args.action]()

if __name__=='__main__':main()
