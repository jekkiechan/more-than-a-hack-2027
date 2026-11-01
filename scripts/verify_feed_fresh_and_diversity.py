"""Official-score replay and manual aggregation for fresh and four-seed evidence."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from experiments import feed_robustness_v1 as p, feed_robustness_confirmation_v1 as c
from experiments import feed_seed_diversity_v1 as d, spell_mbr_run as shared, metrics

def scored_array(path,row,files):
    assert p.sha(path)==files[str(path.relative_to(ROOT))]
    with np.load(path) as z:
        arrays={k:z[k] for k in z.files}
    np.testing.assert_array_equal(arrays['prediction'],p.decode(arrays['probabilities'],arrays['model_period'],float(arrays['fps'])))
    official=metrics.evaluate_tracking(arrays['labels'],arrays['prediction'],period=arrays['period'],timestamps=arrays['timestamps'],fps=float(arrays['fps']))
    for key,value in [('spell_f1',official['spell_f1_mean']),('macro_f1',official['macro_f1']),('d_f1',official['per_class']['D']['f1'])]:
        assert abs(value-row[key])<=1e-12
    return arrays

def aggregate(rows,member,summary):
    for view in ('clean','feed_like'):
        for metric in ('spell_f1','macro_f1','d_f1'):
            providers=[]
            for provider,matches in shared.EXPECTED_MATCHES.items():
                means=[]
                for match in matches:
                    values=[r[metric] for r in rows if r['member']==member and r['provider']==provider and r['match']==match and (r['view']=='clean' if view=='clean' else r['view'].startswith('feed_like_'))]
                    assert len(values)==(1 if view=='clean' else 2)
                    means.append(sum(values)/len(values))
                providers.append(sum(means)/len(means))
            assert abs(sum(providers)/len(providers)-summary[view]['equal_provider_mean'][metric])<=1e-12

def confirmation():
    result=p.read(p.RESULTS/'feed_robustness_confirmation_v1_aggregate.json')
    baseline={ (x['provider'],x['match']):x for x in shared._load_cells(p.check()) if x['member']=='ensemble' and x['view']=='clean'}
    count=0; all_arrays={}
    for arm,rows in result['rows'].items():
        for row in rows:
            if row['view']=='clean':continue
            match=('skillcorner_had_'+row['match']) if row['provider']=='SkillCorner' else row['match']
            replica=row['view'].removeprefix('feed_like_')
            path=c.OUT/'predictions'/arm/f"{match}__{replica}__{row['member']}.npz"
            arrays=scored_array(path,row,result['prediction_sha256'])
            base=baseline[row['provider'],row['match']]
            for k in ('labels','period','timestamps','model_period','fps'):
                np.testing.assert_array_equal(arrays[k],base[k])
            key=tuple(row[k] for k in ('provider','match','view','member'))
            all_arrays[arm,key]=arrays['probabilities'];count+=1
        for member in shared.MEMBERS:aggregate(rows,member,result['summaries'][arm][member])
    for (arm,key),probability in all_arrays.items():
        if key[-1]=='ensemble':
            np.testing.assert_array_equal(probability,np.mean([all_arrays[arm,key[:-1]+(str(s),)] for s in p.SEEDS],axis=0))
    assert count==60
    return {'new_official_cells':count,'unchanged_grading':True,'seed_ensemble_parity':True,'aggregate_sha256':p.sha(p.RESULTS/'feed_robustness_confirmation_v1_aggregate.json')}

def diversity():
    original=p.read(p.RESULTS/'feed_seed_diversity_v1_aggregate.json')
    fresh=p.read(p.RESULTS/'feed_seed_diversity_v1_fresh.json')
    baseline={(x['provider'],x['match'],x['view']):x for x in shared._load_cells(p.check()) if x['member']=='ensemble'}
    previous=p.read(p.RESULTS/'feed_robustness_v1_aggregate.json')
    members={}; count=0
    for row in original['rows']:
        key=tuple(row[k] for k in ('provider','match','view','member'))
        path=d.OUT/'predictions/original'/('__'.join(key)+'.npz')
        arrays=scored_array(path,row,original['prediction_sha256'])
        for field in ('labels','period','timestamps','model_period','fps'):
            np.testing.assert_array_equal(arrays[field],baseline[key[:3]][field])
        members[key]=arrays['probabilities'];count+=1
    for key,value in members.items():
        if key[-1]!='four_seed':continue
        values=[]
        for seed in p.SEEDS:
            path=p.OUT/'predictions/feed_augmented'/('__'.join(key[:3]+(str(seed),))+'.npz')
            assert p.sha(path)==previous['prediction_sha256'][str(path.relative_to(ROOT))]
            with np.load(path) as z:values.append(z['probabilities'])
        values.extend(members[key[:3]+(str(seed),)] for seed in d.SEEDS)
        np.testing.assert_array_equal(value,np.mean(values,axis=0))
    for member in ('2028','2029','four_seed'):aggregate(original['rows'],member,original['summaries'][member])
    for row in fresh['rows']:
        if row['view']=='clean':continue
        key=tuple(row[k] for k in ('provider','match','view','member'))
        path=d.OUT/'predictions/fresh'/('__'.join(key)+'.npz')
        arrays=scored_array(path,row,fresh['prediction_sha256'])
        for field in ('labels','period','timestamps','model_period','fps'):
            np.testing.assert_array_equal(arrays[field],baseline[key[0],key[1],'clean'][field])
        count+=1
    aggregate(fresh['rows'],'four_seed',fresh['summary'])
    for fold in p.FOLDS:
        expected=sorted(i['match'] for i in p.prep.split_plan(fold=fold)['train'])
        for seed in d.SEEDS:
            path=d.OUT/'models'/f'{fold}_{seed}.joblib'
            receipt=p.read(path.with_suffix('.json'));model=joblib.load(path)
            assert receipt['old_matches']==expected and receipt['seed']==seed and receipt['teacher_seed']==d.teacher_seed(seed)
            assert model['step']==2048 and len(receipt['config']['soccernet_games'])==45
            assert p.sha(path)==receipt['sha256']
    assert count==55
    return {'new_official_cells':count,'models':6,'original_four_seed_probability_mean_exact':True,'unchanged_grading':True,'original_aggregate_sha256':p.sha(p.RESULTS/'feed_seed_diversity_v1_aggregate.json'),'fresh_aggregate_sha256':p.sha(p.RESULTS/'feed_seed_diversity_v1_fresh.json')}

if __name__=='__main__':
    mode=sys.argv[1]
    result=(confirmation() if mode=='confirmation' else diversity())
    result.update(passed=True,verifier_sha256=p.sha(Path(__file__)),organizer_reference_access=False)
    path=p.RESULTS/f'feed_{mode}_independent_verification_v1.json'
    p.write(path,result)
    print(json.dumps(result,indent=2))
