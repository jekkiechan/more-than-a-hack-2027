"""Fixed within-record short-possession resampling on raw-feed training views."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import joblib
import numpy as np
from experiments import feed_robustness_v1 as p,feed_short_sampler as sampler
from experiments import feed_robustness_confirmation_v1 as fresh_data
from experiments import feed_spell_push_run as selection
from experiments import tracking_corruption as tc,supervised

STUDY='feed_short_sampling_v1'
OUT=p.ROOT/'data/experiments'/STUDY
DECL=p.RESULTS/f'{STUDY}_declaration.json'
AUDIT=p.RESULTS/f'{STUDY}_exposure.json'
ARMS=('short_sampling','fixed_half_blend')


def declare():
    p.check()
    audit=p.read(AUDIT)
    assert audit['source_sha256']==p.sha(Path(sampler.__file__))
    for row in audit['rows']:
        assert row['baseline']['exposures']==row['targeted']['exposures']
        for group in ('old','new'):
            assert row['targeted']['counts'][group]['short_ha_rows']>row['baseline']['counts'][group]['short_ha_rows']
    paths=[Path(__file__),Path(sampler.__file__),p.ROOT/"scripts/audit_feed_short_sampling_v1.py",p.ROOT/"experiments/tests/test_feed_short_sampler.py",AUDIT,p.RESULTS/'feed_robustness_v1_aggregate.json',p.RESULTS/'feed_robustness_v1_model_seal.json',p.RESULTS/'feed_robustness_v1_verification.json',fresh_data.OUT/'preparation.json',p.RESULTS/'feed_robustness_confirmation_v1_aggregate.json']
    p.write(DECL,{'study':STUDY,'utc':datetime.now(timezone.utc).isoformat(),
        'question':'Does increasing exposure to complete short TRAIN possessions improve the raw-feed augmented ensemble?',
        'fits':6,'seeds':list(p.SEEDS),'folds':list(p.FOLDS),'steps':2048,
        'intervention':'With probability .25 replace a base draw by a uniform existing window containing a complete <=3s H/A span, within the same record. Separate seed+773311 RNG. Fallback unchanged if no eligible windows. Native time-block, unknown and record-edge censoring retained.',
        'fixed':'Same provider-excluded teachers, normalizers, raw-feed views, labels, D2 losses, width96, lrHA3e-4, lrD1e-4, batch32, 50/50 original/GAR, original source/match/record RNG, GAR team-swap RNG, 2048 updates; changes window distribution for BOTH H/A and D.',
        'arms':list(ARMS),'blend':'Exactly .5 new native probability + .5 raw-feed augmented probability, within each seed and then across both seeds. No weight or seed selection.',
        'controls':'Archived two-seed raw-feed augmented model (.5252404582) and promoted baseline (.5185274873). No control refits.',
        'primary':'Equal-provider original feed_like official spell F1; macro tie-breaker. Must beat raw-feed augmented control; .55 stretch.',
        'guards':'Against promoted baseline: aggregate clean spell >= baseline-.005; provider clean macro/D F1 >= baseline-.02; provider feed spell >= baseline-.01. Report +.01 historical clean screen separately.',
        'stop':'Six sealed fits; two fixed arms only. No additional rates, thresholds, epochs, weights or decoder settings in this ID.',
        'fresh_diagnostic':'Only if eligible original winner beats augmented control: evaluate that fixed winner on the previous two extra corruption realizations. These results are previously exposed development diagnostics, NOT fresh independent confirmation; no reselection.',
        'evidence':'Same five previously exposed matches and synthetic feed stress; target-derived TRAIN sampling, no inference labels; original teacher reconstruction provenance retained.',
        'organizer_reference_access':False,'promotion':False,
        'parent_declaration_sha256':p.sha(p.DECLARATION),
        'bindings':{str(x.relative_to(p.ROOT)):p.sha(x) for x in paths}})


def check():
    p.check();d=p.read(DECL)
    assert p.sha(p.DECLARATION)==d['parent_declaration_sha256']
    for rel,digest in d['bindings'].items():
        if p.sha(p.ROOT/rel)!=digest:raise ValueError('Frozen binding changed: '+rel)
    return d


def parent_probability(c):
    key=tuple(c[k] for k in ('provider','match','view','member'))
    path=p.OUT/'predictions/feed_augmented'/('__'.join(key)+'.npz')
    expected=p.read(p.RESULTS/'feed_robustness_v1_aggregate.json')['prediction_sha256']
    assert p.sha(path)==expected[str(path.relative_to(p.ROOT))]
    with np.load(path) as z:
        for field in ('labels','period','timestamps','model_period','fps'):np.testing.assert_array_equal(z[field],c[field])
        return z['probabilities']


def preflight():
    check();cells=p.shared._load_cells(p.read(p.DECLARATION));rows=[]
    for c in cells:rows.append(p.score(c,parent_probability(c))[0])
    prior=p.read(p.RESULTS/'feed_robustness_v1_aggregate.json')
    assert rows==prior['rows']['feed_augmented']
    c=next(x for x in cells if x['provider']=='Metrica' and x['view']=='clean' and x['member']=='2026')
    model=joblib.load(p.OUT/'models/leave_out_metrica_2026.joblib');teacher=joblib.load(p.sc.initial_path('leave_out_metrica',2026))
    actual=p.infer(c,model,teacher);expected=parent_probability(c)
    diff=float(np.abs(actual-expected).max())
    assert diff<=2e-6
    assert np.array_equal(p.decode(actual,c['model_period'],c['fps']),p.decode(expected,c['model_period'],c['fps']))
    p.write(p.RESULTS/f'{STUDY}_preflight.json',{'passed':True,'control_cells':45,'runtime_max_probability_difference':diff,'runtime_changed_labels':0,'declaration_sha256':p.sha(DECL)})


def fit(fold,seed):
    check();assert fold in p.FOLDS and seed in p.SEEDS
    assert p.read(p.RESULTS/f'{STUDY}_preflight.json')['passed']
    path=OUT/'models'/f'{fold}_{seed}.joblib'
    p.write(path.with_suffix('.lock.json'),{'declaration_sha256':p.sha(DECL)})
    teacher=joblib.load(p.sc.initial_path(fold,seed));records=p.training(fold)
    gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS)
    sampling=sampler.Resampler(records,gar,list(teacher['feature_names']),seed)
    started=time.monotonic()
    with sampler.installed(sampling):
        result=p.sn.train(teacher,records,gar,width=96,lr=3e-4,seed=seed,steps=2048,fraction=.5,batch_size=32,ha_target='dense',train_d=True,require_full_native=True,log_every=256,verbose=True)
    expected=next(r['targeted'] for r in p.read(AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)
    assert sampling.receipt()==expected
    assert result['exposures']==expected['exposures']
    assert not path.exists();joblib.dump(result['model'],path)
    p.write(path.with_suffix('.json'),{'fold':fold,'seed':seed,'sha256':p.sha(path),'declaration_sha256':p.sha(DECL),'wall_seconds':time.monotonic()-started,'config':result['config'],'old_matches':sorted({str(r['match']) for r in records}),'sampling':sampling.receipt(),'exposures':result['exposures']})
    print('FIT COMPLETE',fold,seed,flush=True)


def seal():
    check();files={}
    for fold in p.FOLDS:
        for seed in p.SEEDS:
            path=OUT/'models'/f'{fold}_{seed}.joblib';r=p.read(path.with_suffix('.json'))
            assert p.sha(path)==r['sha256'] and r['declaration_sha256']==p.sha(DECL)
            files[str(path.relative_to(p.ROOT))]=p.sha(path)
    p.write(p.RESULTS/f'{STUDY}_model_seal.json',{'models':files,'declaration_sha256':p.sha(DECL)})


def models():
    check()
    seal=p.read(p.RESULTS/f'{STUDY}_model_seal.json')
    for rel,digest in seal['models'].items():assert p.sha(p.ROOT/rel)==digest
    return {(f,s):(joblib.load(OUT/'models'/f'{f}_{s}.joblib'),joblib.load(p.sc.initial_path(f,s))) for f in p.FOLDS for s in p.SEEDS}


def save(c,prob,arm,group,rows,files):
    row,pred=p.score(c,prob);rows[arm].append(row)
    path=OUT/'predictions'/group/arm/('__'.join(str(c[k]) for k in ('provider','match','view','member'))+'.npz')
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,probabilities=prob,prediction=pred,**{k:c[k] for k in ('labels','period','timestamps','model_period','fps')})
    files[str(path.relative_to(p.ROOT))]=p.sha(path)


def evaluate():
    fitted=models();p.write(OUT/'score_lock.json',{'declaration_sha256':p.sha(DECL)})
    cells=p.shared._load_cells(p.read(p.DECLARATION));prob={};rows={a:[] for a in ARMS};files={}
    for c in cells:
        if c['member']=='ensemble':continue
        key=tuple(c[k] for k in ('provider','match','view','member'))
        model,teacher=fitted['leave_out_'+c['provider'].lower(),int(c['member'])]
        prob[key]=p.infer(c,model,teacher)
    for c in cells:
        key=tuple(c[k] for k in ('provider','match','view','member'))
        value=np.mean([prob[key[:3]+(str(s),)] for s in p.SEEDS],axis=0) if c['member']=='ensemble' else prob[key]
        save(c,value,'short_sampling','original',rows,files)
        save(c,.5*(value+parent_probability(c)),'fixed_half_blend','original',rows,files)
    summaries={a:{m:p.scoring.summarize(r,m) for m in p.shared.MEMBERS} for a,r in rows.items()}
    prior=p.read(p.RESULTS/'feed_robustness_v1_aggregate.json')['summaries'];base=prior['baseline']['ensemble'];control=prior['feed_augmented']['ensemble']
    guards={a:selection.eligible(v['ensemble'],base) for a,v in summaries.items()}
    eligible=[a for a in ARMS if guards[a] and selection.rank(summaries[a]['ensemble'])>selection.rank(control)]
    winner=max(eligible,key=lambda a:selection.rank(summaries[a]['ensemble'])) if eligible else None
    result={'study':STUDY,'rows':rows,'summaries':summaries,'baseline':base,'control':control,'guards_pass':guards,'winner':winner,'stretch_reached':bool(winner and summaries[winner]['ensemble']['feed_like']['equal_provider_mean']['spell_f1']>=.55),'historical_clean_screen_pass':{a:v['ensemble']['clean']['equal_provider_mean']['spell_f1']>=base['clean']['equal_provider_mean']['spell_f1']+.01 for a,v in summaries.items()},'prediction_sha256':files,'organizer_reference_access':False}
    p.write(p.RESULTS/f'{STUDY}_aggregate.json',result)
    for a,v in summaries.items():print(a,{view:v['ensemble'][view]['equal_provider_mean']['spell_f1'] for view in ('clean','feed_like')},guards[a],flush=True)


def diagnostic():
    result=p.read(p.RESULTS/f'{STUDY}_aggregate.json');winner=result['winner']
    if winner is None:raise ValueError('No eligible improvement; diagnostic not authorized by declaration')
    fitted=models();p.write(OUT/'diagnostic_lock.json',{'winner_fixed':winner,'aggregate_sha256':p.sha(p.RESULTS/f'{STUDY}_aggregate.json')})
    prior=p.read(p.RESULTS/'feed_robustness_confirmation_v1_aggregate.json');rows={winner:[r for r in result['rows'][winner] if r['view']=='clean']};files={}
    for entry in p.read(fresh_data.OUT/'preparation.json')['files']:
        path=p.ROOT/entry['path'];assert p.sha(path)==entry['sha256'];cached=joblib.load(path)
        inputs=tc.inputs_only(cached['inputs']);record=p.old.inference_record(cached['record']);record['names']=np.asarray(record['names']).tolist()
        c=dict(provider=p.shared._provider_for_match(entry['match']),match=entry['match'].removeprefix('skillcorner_had_'),view=f"feed_like_fresh_{entry['replicate']}",labels=cached['labels'],period=inputs['period'],model_period=supervised.model_periods(inputs),timestamps=inputs['timestamps'],fps=float(inputs['fps']))
        values=[]
        for seed in p.SEEDS:
            model,teacher=fitted['leave_out_'+entry['provider'],seed];value=p.old.upsample(p.sn.predict(record,model,teacher),record,inputs)
            if winner=='fixed_half_blend':
                path=fresh_data.OUT/'predictions/feed_augmented'/f"{entry['match']}__fresh_{entry['replicate']}__{seed}.npz"
                assert p.sha(path)==prior['prediction_sha256'][str(path.relative_to(p.ROOT))]
                with np.load(path) as z:value=.5*(value+z['probabilities'])
            values.append(value);save(dict(c,member=str(seed)),value,winner,'diagnostic',rows,files)
        save(dict(c,member='ensemble'),np.mean(values,axis=0),winner,'diagnostic',rows,files)
    p.write(p.RESULTS/f'{STUDY}_diagnostic.json',{'winner_fixed':winner,'rows':rows,'summaries':{m:p.scoring.summarize(rows[winner],m) for m in p.shared.MEMBERS},'prediction_sha256':files,'previously_exposed_corruptions':True,'independent_confirmation':False,'organizer_reference_access':False})


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('action',choices=('declare','preflight','fit','seal','evaluate','diagnostic'));q.add_argument('--fold',choices=p.FOLDS);q.add_argument('--seed',type=int,choices=p.SEEDS);a=q.parse_args()
    fit(a.fold,a.seed) if a.action=='fit' else globals()[a.action]()
