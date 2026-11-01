"""User-requested fixed public diagnostic after all five external rounds."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,io,json
from pathlib import Path
import numpy as np
from experiments import current_baseline as baseline, sn_dense_baseline, supervised
from experiments import feed_spell_push as broad_methods, feed_spell_refine as refine_methods
from experiments import feed_spell_more as more_methods, feed_spell_push_run as first
from experiments import decoder_search_run as scoring, spell_mbr_run as shared

ROOT=shared.ROOT;RESULTS=shared.RESULTS
OUT=ROOT/'data/experiments/feed_spell_public_v1'
SEAL=OUT/'seal.json';STARTED=OUT/'started.json';RESULT=RESULTS/'feed_spell_public_v1.json'
INPUT=ROOT/'data/challenge/inputs.npz';MODELS=ROOT.parent/'mtah-data/models/joint_gru_sn_dense_ensemble_v1'
LEGACY_MANIFEST=ROOT/'experiments/baselines/joint_gru_combined_ensemble_v1.json'
LEGACY_MODELS=ROOT.parent/'mtah-data/models/joint_gru_combined_ensemble_v1'


def studies():
    return ('feed_spell_push_v1','feed_spell_refine_v2','feed_spell_iteration_r3','feed_spell_iteration_r4','feed_spell_iteration_r5')


def uses_legacy(c):
    return c['family']=='legacy' or any(uses_legacy(c[k]) for k in ('left','right','parent') if isinstance(c.get(k),dict))


def select_external():
    candidates=[]
    for name in studies():
        result=json.loads((RESULTS/(name+'_aggregate.json')).read_text())
        declpath=RESULTS/(name+'_declaration.json');d=json.loads(declpath.read_text())
        if shared.sha256_file(declpath)!=result['declaration_sha256']:raise ValueError('Unbound external result')
        for path,sha in d['source_sha256'].items():
            if shared.sha256_file(ROOT/path)!=sha:raise ValueError('External source changed')
        c=result['selected_config']
        if c not in d['arms'] or c['id']!=result['eligible_ranked'][0]:raise ValueError('External selected config changed')
        candidates.append((name,c,result['arms'][c['id']]))
    return max(candidates,key=lambda x:first.rank(x[2]))


def prepare():
    if OUT.exists() or RESULT.exists():raise ValueError('Public preparation/gate already exists')
    selected_study,selected_config,summary=select_external()
    inputs=baseline.load_inputs(INPUT);manifest=json.loads(baseline.MANIFEST.read_text())
    models=baseline.load_models(MODELS)
    members=sn_dense_baseline.member_probabilities(inputs,models,manifest)
    x=np.stack([members[k] for k in sn_dense_baseline.MEMBERS]);period=supervised.model_periods(inputs);fps=float(inputs['fps'])
    legacy=None;model_hashes={str((MODELS/name).relative_to(ROOT.parent)):sha for name,sha in manifest['models_sha256'].items()}
    if uses_legacy(selected_config):
        oldmanifest=json.loads(LEGACY_MANIFEST.read_text());oldmodels=baseline.load_models(LEGACY_MODELS,LEGACY_MANIFEST)
        oldmembers=baseline.joint_gru_member_probabilities(inputs,oldmodels,oldmanifest)
        legacy=np.stack(list(oldmembers.values()))
        model_hashes|={str((LEGACY_MODELS/name).relative_to(ROOT.parent)):sha for name,sha in oldmanifest['models_sha256'].items()}
    ordinary=supervised.decode(x.mean(axis=0),period,fps,3)
    broad_config=json.loads(first.AGGREGATE.read_text())['selected_config']
    broad=broad_methods.decode(x,period,fps,broad_config)
    if selected_study=='feed_spell_push_v1':final=broad.copy()
    elif selected_study=='feed_spell_refine_v2':final=refine_methods.decode(x,period,fps,selected_config)
    else:final=more_methods.decode(x,period,fps,selected_config,legacy)
    for y in (ordinary,broad,final):y[np.asarray(inputs['period'])<=0]='X'
    OUT.mkdir(parents=True,exist_ok=False)
    baseline.challenge.write_submission(ordinary,OUT/'adopted_replay.csv')
    if shared.sha256_file(OUT/'adopted_replay.csv')!=manifest['evidence']['prediction_csv_sha256']:
        raise ValueError('Exact adopted deployment CSV replay failed')
    predictions={}
    for name,y in (('broad_winner',broad),('final_external_winner',final)):
        path=OUT/(name+'.csv');baseline.challenge.write_submission(y,path)
        predictions[name]={'path':str(path.relative_to(ROOT)),'sha256':shared.sha256_file(path)}
    previous=ROOT/'data/experiments/feed_spell_push_v1_candidate.csv'
    if predictions['broad_winner']['sha256']!=shared.sha256_file(previous):raise ValueError('Broad candidate exact CSV replay failed')
    np.savez_compressed(OUT/'native_probabilities.npz',dense_members=x,**({'legacy_members':legacy} if legacy is not None else {}))
    sources=list((ROOT/'experiments').glob('*.py'))+[ROOT/'evaluation/possession_metrics.py']
    dependencies={str((RESULTS/(s+suffix)).relative_to(ROOT)):shared.sha256_file(RESULTS/(s+suffix))
                  for s in studies() for suffix in ('_declaration.json','_aggregate.json','_readout.json')}
    amendment=RESULTS/'feed_spell_refine_v2_execution_amendment.json'
    dependencies[str(amendment.relative_to(ROOT))]=shared.sha256_file(amendment)
    dependencies[str(baseline.MANIFEST.relative_to(ROOT))]=shared.sha256_file(baseline.MANIFEST)
    dependencies[str(INPUT.relative_to(ROOT))]=shared.sha256_file(INPUT)
    expected=json.loads((RESULTS/'combined_objective_public_check.json').read_text())['organizer_gt_sha256']
    scoring.write_once(SEAL,{'study':'feed_spell_public_v1','sealed_utc':datetime.now(timezone.utc).isoformat(),
        'explicit_user_request':"don't forget to calculate public score as well ... do several more iterations for improvements",
        'candidate_rule':'Current broad winner and strongest guard-passing externally selected winner across all five completed rounds; duplicate CSV hashes scored once',
        'selected_study':selected_study,'selected_config':selected_config,'selected_external_summary':summary,
        'predictions':predictions,'source_sha256':{str(p.relative_to(ROOT)):shared.sha256_file(p) for p in sources},
        'dependency_sha256':dependencies,'models_sha256':model_hashes,'expected_reference_sha256':expected,
        'baseline_replay_exact':True,'broad_candidate_replay_exact':True,'reference_labels_read':False,
        'no_selection_tuning_or_promotion_after_public':True})
    print('Sealed candidates',selected_study,selected_config,flush=True)


def validate_seal(spec):
    for rel,sha in (spec['source_sha256']|spec['dependency_sha256']).items():
        if shared.sha256_file(ROOT/rel)!=sha:raise ValueError('Sealed artifact changed: '+rel)
    for rel,sha in spec['models_sha256'].items():
        if shared.sha256_file(ROOT.parent/rel)!=sha:raise ValueError('Sealed checkpoint changed')
    for item in spec['predictions'].values():
        if shared.sha256_file(ROOT/item['path'])!=item['sha256']:raise ValueError('Sealed prediction changed')


def score():
    if RESULT.exists():return json.loads(RESULT.read_text())
    if STARTED.exists():raise ValueError('Public gate permanently closed; do not rescore')
    spec=json.loads(SEAL.read_text());validate_seal(spec)
    scoring.write_once(STARTED,{'started_utc':datetime.now(timezone.utc).isoformat(),'seal_sha256':shared.sha256_file(SEAL)})
    raw=(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=spec['expected_reference_sha256']:raise ValueError('Reference SHA mismatch; gate remains closed')
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    truth=pm.label_codes(pm.read_label_csv(io.BytesIO(raw)));by_digest={};scores={}
    for name,item in spec['predictions'].items():
        if item['sha256'] not in by_digest:
            frame=pm.read_label_csv(ROOT/item['path']);error=pm.validate_submission(frame,199319)
            if error:raise ValueError(error)
            value=pm.evaluate(truth,pm.label_codes(frame),bootstrap_n=1000)
            if (value['n_graded'],value['n_gt_spells'])!=(7500,17):raise ValueError('Public grading population changed')
            by_digest[item['sha256']]=_score_summary(value)
        scores[name]=by_digest[item['sha256']]
    adopted=json.loads((RESULTS/'sn_dense_v1_public_check.json').read_text())['candidate']
    result={'study':'feed_spell_public_v1','candidate_scores':scores,'adopted_baseline_reused':adopted,
        'delta_vs_adopted':{name:{k:s[k]-adopted[k] for k in ('spell_f1_mean','macro_f1','d_f1')} for name,s in scores.items()},
        'prediction_seal_sha256':shared.sha256_file(SEAL),'unique_csvs_scored':len(by_digest),
        'selected_study':spec['selected_study'],'selected_config':spec['selected_config'],
        'reference_sha256':spec['expected_reference_sha256'],'public_used_for_training_tuning_selection_or_promotion':False,
        'interpretation':'One fixed, exposed five-minute diagnostic with 17 reference possessions; full-match performance unknown; no tuning or reselection follows'}
    scoring.write_once(RESULT,result);print(json.dumps(result['delta_vs_adopted'],indent=2),flush=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('prepare','score'));args=parser.parse_args()
    prepare() if args.stage=='prepare' else score()
