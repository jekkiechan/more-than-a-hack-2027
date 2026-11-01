"""One user-requested public diagnostic of frozen original_full, with pooled fits."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import io
import json
import time

import joblib
import numpy as np

from experiments import short_original_diversity_v1 as research
from experiments import current_baseline as baseline, sn_dense_baseline
from experiments import live_gap_experiment
from experiments.short_original_diversity import OriginalDiversitySampler

p=research.p
STUDY='short_quality_public_v1'
OUT=p.ROOT/'data/experiments'/STUDY
DECL=p.RESULTS/f'{STUDY}_declaration.json'
RESULT=p.RESULTS/f'{STUDY}_result.json'
SEAL=OUT/'prediction_seal.json'
STARTED=OUT/'public_check_started.json'
INPUT=p.ROOT/'data/challenge/inputs.npz'
REFERENCE=p.ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
PACKAGE=p.ROOT/'data/models/joint_gru_sn_dense_calibrated_ensemble_v1'


def write(path,value):
    payload=json.dumps(p.sc._jsonable(value),indent=2,sort_keys=True,allow_nan=False)+'\n'
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:f.write(payload)


def require_open():
    if STARTED.exists() or RESULT.exists():raise ValueError('Public gate permanently closed; reuse saved result')


def teacher_path(seed):return PACKAGE/f'teacher_{seed}.joblib'


def check():
    d=p.read(DECL)
    for rel,digest in (d['source_sha256']|d['input_sha256']).items():
        assert p.sha(p.shared._resolve_artifact(rel))==digest,rel
    return d


def declare():
    require_open();research.check();p.check()
    result=p.read(p.RESULTS/f'{research.STUDY}_aggregate.json')
    assert result['winner']=='original_full' and result['guards_pass']['original_full']
    assert p.read(p.RESULTS/f'{research.STUDY}_verification.json')['passed']
    manifest=p.read(baseline.MANIFEST)
    inputs=dict(p.read(p.DECLARATION)['input_sha256'])
    for seed in p.SEEDS:
        path=teacher_path(seed);assert p.sha(path)==manifest['models_sha256'][path.name]
        inputs[str(path.relative_to(p.ROOT))]=p.sha(path)
    inputs[str(INPUT.relative_to(p.ROOT))]=p.sha(INPUT)
    for name in (f'{research.STUDY}_aggregate.json',f'{research.STUDY}_verification.json',
                 'short_quality_fresh_v1_aggregate.json','feed_spell_public_v1.json'):
        path=p.RESULTS/name;inputs[str(path.relative_to(p.ROOT))]=p.sha(path)
    sources=p.source_hashes()
    for path in (p.ROOT/'experiments/tests/test_short_quality_public_v1.py',p.ROOT/'games.py'):
        sources[str(path.relative_to(p.ROOT))]=p.sha(path)
    old=p.read(p.RESULTS/'feed_spell_public_v1.json')
    write(DECL,dict(study=STUDY,utc=datetime.now(timezone.utc).isoformat(),
        user_request='try the public. this looks like a reasonable gain overall in spell F1 cause our baseline is still at 0.5185',
        fixed_candidate='original_full selected on external development before this request; one equal two-seed full H/A/D probability ensemble, frozen f062 decoder (D temperature4/cost3)',
        fit='Two pooled fits only, seeds2026/2027; all13 original TRAIN matches and45 GAR TRAIN games; same OriginalDiversitySampler, width96,2048updates,batch32,50/50sources,HA lr3e-4,D lr1e-4,unchanged labels/losses/normalization/raw-feed views. Warm-start exact adopted combined-objective teacher pair, not adopted dense weights.',
        seeds=list(p.SEEDS),steps=2048,train_plan=p.prep.split_plan(fold='pooled'),
        public='One unique full CSV evaluated once with official scorer and1000bootstrap draws. Expect199319rows,7500gradedframes,17reference possessions. No seed/component scores or variants on public labels.',
        stop='Consume exclusive global/local gate before reference I/O, including failed reads. No public-driven tuning, repairs, reselection, repeat scoring or automatic promotion. All prior gates remain closed.',
        baseline_scores_reused=old['candidate_scores']['final_external_winner'],
        expected_reference_sha256=old['reference_sha256'],
        baseline_csv_sha256=manifest['evidence']['prediction_csv_sha256'],
        baseline_manifest_sha256=p.sha(baseline.MANIFEST),policy_sha256=p.sha(p.ROOT/'experiments/model_decision_policy.json'),
        external_candidate=result['summaries']['original_full']['ensemble'],
        source_sha256=sources,input_sha256=inputs,reference_labels_read=False,model_promoted=False))


def fit(seed):
    require_open();d=check();assert seed in p.SEEDS
    path=OUT/'models'/f'pooled_{seed}.joblib'
    write(path.with_suffix('.lock.json'),dict(seed=seed,declaration_sha256=p.sha(DECL)))
    teacher=joblib.load(teacher_path(seed));old=p.training('pooled')
    assert sorted({str(r['match']) for r in old})==sorted(i['match'] for i in d['train_plan']['train'])
    gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS)
    assert len({str(r['match']) for r in gar})==45
    sampler=OriginalDiversitySampler(old,gar,list(teacher['feature_names']),seed)
    started=time.monotonic()
    with research.parent.base.installed(sampler):
        result=p.sn.train(teacher,old,gar,width=96,lr=3e-4,seed=seed,steps=2048,
            fraction=.5,batch_size=32,ha_target='dense',train_d=True,require_full_native=True,log_every=256,verbose=True)
    receipt=sampler.receipt()
    assert receipt['exposures']==result['exposures']
    assert receipt['counts']['old']['draws']==receipt['counts']['new']['draws']==32768
    assert receipt['eligible_quality_games']==45 and receipt['quality_counts']['quality']==403
    for field in ('median','scale'):np.testing.assert_array_equal(result['model'][field],teacher[field])
    assert not path.exists();joblib.dump(result['model'],path)
    write(path.with_suffix('.json'),dict(seed=seed,sha256=p.sha(path),declaration_sha256=p.sha(DECL),
        sampling=receipt,config=result['config'],wall_seconds=time.monotonic()-started,
        teacher_sha256=p.sha(teacher_path(seed)),old_matches=sorted({str(r['match']) for r in old})))
    print('POOLED FIT COMPLETE',seed,flush=True)


def prediction_record(inputs):
    clean=p.tc.inputs_only(inputs)
    placeholder=np.full(len(clean['ball']),'X',dtype='U1')
    return p.old.inference_record(live_gap_experiment.feature_view(clean,placeholder,'competition','BAR-ATH')['joint'])


def prepare():
    require_open();d=check();models={};model_hashes={}
    for seed in p.SEEDS:
        path=OUT/'models'/f'pooled_{seed}.joblib';r=p.read(path.with_suffix('.json'))
        assert r['declaration_sha256']==p.sha(DECL) and p.sha(path)==r['sha256']
        models[seed]=joblib.load(path);assert models[seed]['step']==2048
        model_hashes[str(path.relative_to(p.ROOT))]=p.sha(path)
    write(OUT/'model_seal.json',dict(models=model_hashes,declaration_sha256=p.sha(DECL)))
    inputs=baseline.load_inputs(INPUT);record=prediction_record(inputs)
    period=p.supervised.model_periods(inputs);fps=float(inputs['fps']);native={}
    for seed in p.SEEDS:
        prob=p.old.upsample(p.sn.predict(record,models[seed],joblib.load(teacher_path(seed))),record,inputs)
        assert prob.shape==(199319,3) and np.isfinite(prob).all()
        np.testing.assert_allclose(prob.sum(axis=1),1,atol=1e-6,rtol=0)
        native[str(seed)]=prob
    pooled=np.mean(list(native.values()),axis=0)
    prediction=p.decode(pooled,period,fps);prediction[np.asarray(inputs['period'])<=0]='X'
    csv=OUT/'candidate.csv';assert not csv.exists();baseline.challenge.write_submission(prediction,csv)
    # Independent feature-only adopted replay verifies the complete challenge
    # record/teacher/calibration/frame mapping without any reference access.
    bm=baseline.load_models(PACKAGE);manifest=p.read(baseline.MANIFEST)
    adopted=sn_dense_baseline.member_probabilities(inputs,bm,manifest)
    basepred=p.decode(np.mean(list(adopted.values()),axis=0),period,fps)
    basepred[np.asarray(inputs['period'])<=0]='X'
    basecsv=OUT/'adopted_replay.csv';assert not basecsv.exists();baseline.challenge.write_submission(basepred,basecsv)
    assert p.sha(basecsv)==d['baseline_csv_sha256'],'Adopted full CSV byte replay failed'
    dest=OUT/'native_probabilities.npz';assert not dest.exists()
    np.savez_compressed(dest,**{'member_'+k:v for k,v in native.items()},ensemble=pooled,
        prediction=prediction,model_period=period,period=inputs['period'],fps=fps)
    paths=[csv,basecsv,dest,OUT/'model_seal.json']
    write(SEAL,dict(study=STUDY,utc=datetime.now(timezone.utc).isoformat(),
        declaration_sha256=p.sha(DECL),models_sha256=model_hashes,
        files_sha256={str(x.relative_to(p.ROOT)):p.sha(x) for x in paths},
        candidate_csv=str(csv.relative_to(p.ROOT)),candidate_sha256=p.sha(csv),
        rows=199319,adopted_replay_exact=True,reference_labels_read=False))
    print('FULL CSV VALIDATED AND SEALED; no public labels read',flush=True)


def verify_seal():
    d=check();s=p.read(SEAL);assert s['declaration_sha256']==p.sha(DECL)
    for rel,digest in (s['models_sha256']|s['files_sha256']).items():assert p.sha(p.ROOT/rel)==digest,rel
    return d,s


def read_reference():
    assert STARTED.exists(),'Reference I/O requires the claimed gate'
    return REFERENCE.read_bytes()


def score():
    if RESULT.exists():return p.read(RESULT)
    require_open();d,s=verify_seal()
    write(STARTED,dict(utc=datetime.now(timezone.utc).isoformat(),prediction_seal_sha256=p.sha(SEAL)))
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    reused=s['candidate_sha256']==d['baseline_csv_sha256']
    if reused:summary=d['baseline_scores_reused']
    else:
        raw=read_reference()
        assert hashlib.sha256(raw).hexdigest()==d['expected_reference_sha256'],'Reference changed; gate remains closed'
        truth=pm.label_codes(pm.read_label_csv(io.BytesIO(raw)))
        candidate=pm.read_label_csv(p.ROOT/s['candidate_csv'])
        assert pm.validate_submission(candidate,199319) is None
        measured=pm.evaluate(truth,pm.label_codes(candidate),bootstrap_n=1000)
        assert (measured['n_frames'],measured['n_graded'],measured['n_gt_spells'])==(199319,7500,17)
        summary=_score_summary(measured)
    prior=d['baseline_scores_reused']
    result=dict(study=STUDY,candidate=summary,adopted_baseline_reused=prior,
        delta_vs_adopted={k:summary[k]-prior[k] for k in ('spell_f1_mean','macro_f1','d_f1')},
        candidate_csv_sha256=s['candidate_sha256'],prediction_seal_sha256=p.sha(SEAL),
        reference_sha256=d['expected_reference_sha256'],reference_labels_read=not reused,
        new_unique_csvs_scored=0 if reused else 1,public_used_for_training_tuning_selection=False,
        model_promoted=False,interpretation='One user-requested fixed diagnostic on17 previously exposed public possessions; no tuning or reselection follows. Full-match performance unknown.')
    write(RESULT,result);print(json.dumps(result,indent=2),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('declare','fit','prepare','score','verify_seal'))
    parser.add_argument('--seed',type=int,choices=p.SEEDS);args=parser.parse_args()
    value=fit(args.seed) if args.action=='fit' else globals()[args.action]()
