"""One explicitly requested public pipeline check of the latest D-repaired GRU.

One fixed-seed pooled fit on external training matches; unchanged input handling
and decoder. Fitting/prediction cannot read organizer reference labels. Only the
check action opens those labels, after an exclusive gate and prediction seal.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
from experiments import (deadball_transfer_experiment as dev, live_gap_experiment as old,
                         factorized_had as gru, learned_temporal as lt,
                         corruption_experiment as ce, current_baseline as baseline,
                         challenge, refine_features, robust, supervised)
from experiments.light_public_check import SCORE_KEYS
from experiments.eval_set.protocol import sha256

ROOT=lt.ROOT
DEFAULT=ROOT/'data/experiments/joint_gru_public_v1'
REPORT=ROOT/'experiments/results/joint_gru_public_check.json'
INPUTS=ROOT/'data/challenge/inputs.npz'
INPUT_SHA256='7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3'
SEED=2026
ARM='factorized_player_D'


def sources():
    return dict(dev.fingerprints(),**{str(p.relative_to(ROOT)):sha256(p) for p in
        (Path(__file__),Path(challenge.__file__),ROOT/'experiments/light_public_check.py')})


def training_paths():
    return [(source,p) for source,v in dev.dt.SPLITS.items() for p in v['train']]


def freeze(out=DEFAULT):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'predeclared.json').exists(): return verify_plan(out)
    if REPORT.exists() or (out/'public_check_started.json').exists():
        raise ValueError('Public check already exists')
    original=dev.check(dev.DEFAULT)
    if sha256(INPUTS)!=INPUT_SHA256: raise ValueError('Organizer input cache differs')
    paths=training_paths()
    evaluation={p.stem for s in dev.dt.SPLITS.values() for p in s['eval']}
    if evaluation & {p.stem for _,p in paths}: raise ValueError('Training/evaluation match overlap')
    data={str(p.relative_to(ROOT)):sha256(p) for _,p in paths}
    for name,digest in data.items():
        if original['data_sha256'][name]!=digest: raise ValueError('External training data changed')
    archived=ROOT/'experiments/results/light_corruption_public_check.json'
    previous=ROOT/'experiments/results/current_baseline_public_check.json'
    spec=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources(),
        user_request='calculate comp too. i think we are moving in the right direction. perhaps we just need better preprocessing for corrupted tracking',
        scope='One descriptive five-minute pipeline check of the latest D-repaired joint GRU; not model selection, tuning, promotion or full-match evaluation',
        arm=ARM,seed=SEED,seed_rule='Original default seed, not the better external-score seed; no seed ensemble',
        fit='One pooled final fit on the original 13 training matches only; same architecture, eight epochs, categorical loss, optimizer, clean/coarse and 25% ball-gap views; no retuning',
        training_sources=list(dev.dt.SPLITS),training_matches=[p.stem for _,p in paths],
        excluded_evaluation_matches=sorted(evaluation),data_sha256=data,
        training_caches=original['training_caches'],
        external_development_report_sha256=sha256(dev.DEFAULT/'report.json'),
        external_guardrails='Both D-repaired seeds failed synthetic-stress guardrails; this user-requested diagnostic does not change that decision',
        input_path=str(INPUTS.relative_to(ROOT)),input_sha256=INPUT_SHA256,
        preprocessing='Unchanged baseline tracking-input allowlist and observed-coordinate masks; no new repair/interpolation',
        decoder_switch_cost=lt.COST,bootstrap_draws=1000,
        baseline_report=str(archived.relative_to(ROOT)),baseline_report_sha256=sha256(archived),
        baseline_manifest_sha256=sha256(baseline.MANIFEST),
        organizer_gt_sha256=json.loads(previous.read_text())['organizer_gt_sha256'],
        prior_public_gates='Untouched; separate exclusive marker for this explicitly requested check',
        organizer_labels_used_for_fitting=False,baseline_replaced=False)
    ce.write_json(out/'predeclared.json',spec,exclusive=True)
    return spec


def verify_plan(out=DEFAULT):
    spec=json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256']!=sources(): raise ValueError('Frozen sources changed')
    if spec['external_development_report_sha256']!=sha256(dev.DEFAULT/'report.json'):
        raise ValueError('External development evidence changed')
    if spec['baseline_report_sha256']!=sha256(ROOT/spec['baseline_report']):
        raise ValueError('Archived baseline result changed')
    if spec['baseline_manifest_sha256']!=sha256(baseline.MANIFEST):
        raise ValueError('Baseline manifest changed')
    if spec['input_sha256']!=sha256(ROOT/spec['input_path']):
        raise ValueError('Organizer inputs changed')
    return spec


def require_open_training(out):
    if any((out/name).exists() for name in ('public_check_predeclared.json','public_check_started.json')) or REPORT.exists():
        raise ValueError('Prediction/check sealed; no further fitting or generation')


def fit(out=DEFAULT):
    spec=verify_plan(out);require_open_training(out)
    target=out/'pooled_joint_gru.joblib'
    if target.exists():
        ce.load_model(out,target)
        return
    records=[]
    for source,path in training_paths():
        if sha256(path)!=spec['data_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Training data changed')
        cached=old.DEFAULT/f'train_{path.stem}.joblib'
        if sha256(cached)!=spec['training_caches'][cached.name]: raise ValueError('Training cache changed')
        r=joblib.load(cached)
        for v,name in enumerate(('clean','legacy')):
            for swap in (False,True):
                records.append(old.mixed_record(r[name]['joint'],r['gap']['joint'],path.stem,v,swap))
    started=time.monotonic()
    fitted=gru.fit(records,player_only=True,seed=SEED)
    ce.freeze_model(out,target,fitted,dict(arm=ARM,seed=SEED,sources=spec['training_sources'],
        training_matches=spec['training_matches'],seconds=time.monotonic()-started,
        parameters=fitted['parameters'],windows_per_epoch=fitted['windows_per_epoch'],
        optimizer_updates=fitted['optimizer_updates'],organizer_labels_used=False))
    print('FROZEN pooled external-only joint GRU',flush=True)


def feature_record(inputs):
    clean=robust.tracking_inputs(inputs)
    team,names=refine_features.features(clean)
    player,pnames=lt.player_features(clean)
    indices=lt.sampled_indices(clean)
    periods=supervised.model_periods(clean)
    return dict(x=np.column_stack((team[indices],player[indices])),names=names+['D_'+n for n in pnames],
        indices=indices,native_n=len(periods),period=periods[indices]),periods


def predict(out=DEFAULT):
    spec=verify_plan(out);require_open_training(out)
    destination=out/'candidate.csv'
    if destination.exists(): raise ValueError('Candidate already generated; refusing overwrite')
    model=ce.load_model(out,out/'pooled_joint_gru.joblib')
    inputs=baseline.load_inputs(ROOT/spec['input_path'])
    started=time.monotonic()
    record,periods=feature_record(inputs)
    features_seconds=time.monotonic()-started
    started=time.monotonic()
    p5=gru.predict(record,model)
    p=np.column_stack([lt.upsample(p5[:,k],record,periods) for k in range(3)])
    predicted=supervised.decode(p,periods,float(inputs['fps']),lt.COST)
    predicted[np.asarray(inputs['period'])<=0]='X'
    inference_seconds=time.monotonic()-started
    challenge.write_submission(predicted,destination)
    ce.write_json(destination.with_suffix('.json'),dict(output_sha256=sha256(destination),
        input_sha256=spec['input_sha256'],model_sha256=sha256(out/'pooled_joint_gru.joblib'),
        plan_sha256=sha256(out/'predeclared.json'),feature_seconds=features_seconds,
        inference_seconds=inference_seconds,reference_labels_read=False),exclusive=True)
    print('Generated and format-validated all 199319 frames without reference labels',flush=True)


def verify_files(frozen,root=ROOT):
    for name,digest in frozen['files_sha256'].items():
        if sha256(root/name)!=digest: raise ValueError(f'Frozen check input changed: {name}')


def seal(out=DEFAULT):
    spec=verify_plan(out)
    if REPORT.exists() or (out/'public_check_started.json').exists(): raise ValueError('Public gate is closed')
    ce.load_model(out,out/'pooled_joint_gru.joblib')
    receipt=json.loads((out/'candidate.json').read_text())
    if receipt['output_sha256']!=sha256(out/'candidate.csv') or receipt['model_sha256']!=sha256(out/'pooled_joint_gru.joblib'):
        raise ValueError('Candidate binding changed')
    paths=[out/name for name in ('predeclared.json','pooled_joint_gru.joblib','pooled_joint_gru.json','candidate.csv','candidate.json')]
    paths += [ROOT/spec['baseline_report'],baseline.MANIFEST,Path(__file__),ROOT/'evaluation/possession_metrics.py']
    frozen=dict(created_utc=datetime.now(timezone.utc).isoformat(),arm=ARM,seed=SEED,
        scope=spec['scope'],bootstrap_draws=spec['bootstrap_draws'],organizer_gt_sha256=spec['organizer_gt_sha256'],
        files_sha256={str(p.relative_to(ROOT)):sha256(p) for p in paths})
    path=out/'public_check_predeclared.json'
    if path.exists(): verify_files(json.loads(path.read_text()))
    else: ce.write_json(path,frozen,exclusive=True)
    print('Sealed model and predictions before public-label access',flush=True)


def start(out,frozen):
    ce.write_json(out/'public_check_started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),
        declaration_sha256=sha256(out/'public_check_predeclared.json'),arm=frozen['arm'],seed=frozen['seed']),exclusive=True)


def check(out=DEFAULT):
    if REPORT.exists(): raise ValueError('Public report already exists; no repeat scoring')
    spec=verify_plan(out)
    frozen=json.loads((out/'public_check_predeclared.json').read_text())
    verify_files(frozen)
    start(out,frozen)  # Exclusive gate BEFORE importing/reading reference labels.
    import pandas as pd
    from evaluation import possession_metrics as pm
    gt_path=ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    if sha256(gt_path)!=frozen['organizer_gt_sha256']: raise ValueError('Public reference changed')
    truth=pd.read_csv(gt_path)
    prediction=pd.read_csv(out/'candidate.csv')
    error=pm.validate_submission(prediction,len(truth))
    if error: raise ValueError(error)
    score=pm.evaluate(pm.label_codes(truth.label.to_numpy()),pm.label_codes(prediction.label.to_numpy()),bootstrap_n=frozen['bootstrap_draws'])
    archived=json.loads((ROOT/spec['baseline_report']).read_text())
    current={k:archived['score'][k] for k in SCORE_KEYS}
    result=dict(scope=spec['scope'],frozen=frozen,training_plan=spec,
        score={k:score[k] for k in SCORE_KEYS},archived_baseline=current,
        spell_delta=score['spell_f1_mean']-current['spell_f1_mean'],
        macro_delta=score['macro_f1']-current['macro_f1'],
        auxiliary_70_30=dict(joint=.7*score['spell_f1_mean']+.3*score['macro_f1'],
            baseline=.7*current['spell_f1_mean']+.3*current['macro_f1']),
        baseline_score_reused=True,full_match_score=None,organizer_labels_used_for_fitting=False,
        preprocessing_changed=False,baseline_replaced=False,
        bootstrap_interval='Official 5th–95th percentiles; not a paired significance test')
    ce.write_json(out/'public_check_result.json',result,exclusive=True)
    ce.write_json(REPORT,result,exclusive=True)
    print(json.dumps(dict(joint=result['score'],baseline=current,spell_delta=result['spell_delta']),indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','fit','predict','seal','check'))
    args=p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        globals()[args.action]()


if __name__=='__main__': main()
