"""Replay official scores and independently check sampling/ensemble provenance."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import joblib
from experiments import feed_short_sampling_v1 as study,feed_short_head_v1 as head
from scripts.verify_feed_fresh_and_diversity import scored_array,aggregate
p=study.p

def main(mode):
    study.check()
    module=head if mode=='head' else study
    path=p.RESULTS/f'{module.STUDY}_aggregate.json';result=p.read(path)
    cells={tuple(c[k] for k in ('provider','match','view','member')):c for c in p.shared._load_cells(p.read(p.DECLARATION))}
    groups={'targeted_ha':result['rows']} if mode=='head' else result['rows']
    saved={};count=0
    for arm,rows in groups.items():
        for row in rows:
            key=tuple(row[k] for k in ('provider','match','view','member'));c=cells[key]
            file=(head.OUT/'predictions' if mode=='head' else study.OUT/'predictions/original'/arm)/('__'.join(key)+'.npz')
            arrays=scored_array(file,row,result['prediction_sha256'])
            for field in ('labels','period','timestamps','model_period','fps'):np.testing.assert_array_equal(arrays[field],c[field])
            saved[arm,key]=arrays['probabilities'];count+=1
        for m in p.shared.MEMBERS:aggregate(rows,m,result['summaries'][m] if mode=='head' else result['summaries'][arm][m])
    for (arm,key),prob in saved.items():
        if key[-1]=='ensemble':
            expected=np.mean([saved[arm,key[:3]+(str(s),)] for s in p.SEEDS],axis=0)
            np.testing.assert_allclose(prob,expected,atol=1e-15,rtol=0)
        if mode=='head' and key[-1]!='ensemble':
            prior=p.read(p.RESULTS/f'{study.STUDY}_aggregate.json');f=study.OUT/'predictions/original/short_sampling'/('__'.join(key)+'.npz')
            assert p.sha(f)==prior['prediction_sha256'][str(f.relative_to(ROOT))]
            with np.load(f) as z:ha=z['probabilities']
            dead=study.parent_probability(cells[key]);mass=ha[:,:2].sum(axis=1)
            home=np.divide(ha[:,0],mass,out=np.full(len(mass),.5),where=mass>0)
            expected=np.column_stack((home*(1-dead[:,2]),(1-home)*(1-dead[:,2]),dead[:,2]))
            np.testing.assert_allclose(prob,expected,atol=1e-15,rtol=0)
        if mode!='head' and arm=='fixed_half_blend':
            np.testing.assert_array_equal(prob,.5*(saved['short_sampling',key]+study.parent_probability(cells[key])))
    models=0
    if mode!='head':
        audit=p.read(study.AUDIT)
        for row in audit['rows']:
            file=study.OUT/'models'/f"{row['fold']}_{row['seed']}.joblib";r=p.read(file.with_suffix('.json'));model=joblib.load(file)
            assert p.sha(file)==r['sha256'] and model['step']==2048
            assert r['sampling']==row['targeted'] and r['exposures']==row['baseline']['exposures']
            assert r['old_matches']==sorted(x['match'] for x in p.prep.split_plan(fold=row['fold'])['train'])
            assert len(r['config']['soccernet_games'])==45
            models+=1
    assert count==(45 if mode=='head' else 90)
    receipt={'passed':True,'official_cells':count,'models':models,'unchanged_native_grading':True,'ensemble_and_blend_or_cross_checked':True,'aggregate_sha256':p.sha(path),'verifier_sha256':p.sha(Path(__file__)),'organizer_reference_access':False}
    p.write(p.RESULTS/f'{module.STUDY}_verification.json',receipt);print(json.dumps(receipt,indent=2))

def diagnostic():
    path=p.RESULTS/f'{study.STUDY}_diagnostic.json';result=p.read(path);arm=result['winner_fixed']
    original=p.read(p.RESULTS/f'{study.STUDY}_aggregate.json');assert arm==original['winner']
    clean={(c['provider'],c['match']):c for c in p.shared._load_cells(p.read(p.DECLARATION)) if c['member']=='ensemble' and c['view']=='clean'}
    saved={};count=0
    for row in result['rows'][arm]:
        if row['view']=='clean':continue
        key=tuple(row[k] for k in ('provider','match','view','member'))
        file=study.OUT/'predictions/diagnostic'/arm/('__'.join(key)+'.npz')
        arrays=scored_array(file,row,result['prediction_sha256'])
        for field in ('labels','period','timestamps','model_period','fps'):np.testing.assert_array_equal(arrays[field],clean[key[:2]][field])
        saved[key]=arrays['probabilities'];count+=1
    for key,value in saved.items():
        if key[-1]=='ensemble':np.testing.assert_array_equal(value,np.mean([saved[key[:3]+(str(seed),)] for seed in p.SEEDS],axis=0))
    for member in p.shared.MEMBERS:aggregate(result['rows'][arm],member,result['summaries'][member])
    assert count==30
    receipt={'passed':True,'official_cells':count,'unchanged_native_grading':True,'ensemble_parity':True,'previously_exposed_development':True,'aggregate_sha256':p.sha(path),'verifier_sha256':p.sha(Path(__file__)),'organizer_reference_access':False}
    p.write(p.RESULTS/f'{study.STUDY}_diagnostic_verification.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':
    mode=sys.argv[1] if len(sys.argv)>1 else 'sampling'
    diagnostic() if mode=='diagnostic' else main(mode)
