"""Independent score, component and sampling replay for game-level access."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from experiments import feed_short_game_v1 as study
from scripts.verify_feed_fresh_and_diversity import scored_array,aggregate
p=study.p

def main():
    study.check();path=p.RESULTS/f'{study.STUDY}_aggregate.json';result=p.read(path)
    cells={tuple(c[k] for k in ('provider','match','view','member')):c for c in p.shared._load_cells(p.read(p.DECLARATION))}
    saved={};count=0
    for arm in study.ARMS:
        for row in result['rows'][arm]:
            key=tuple(row[k] for k in ('provider','match','view','member'));c=cells[key]
            file=study.OUT/'predictions/original'/arm/('__'.join(key)+'.npz')
            arrays=scored_array(file,row,result['prediction_sha256'])
            for field in ('labels','period','timestamps','model_period','fps'):np.testing.assert_array_equal(arrays[field],c[field])
            saved[arm,key]=arrays['probabilities'];count+=1
        for member in p.shared.MEMBERS:aggregate(result['rows'][arm],member,result['summaries'][arm][member])
    for (arm,key),value in saved.items():
        if key[-1]=='ensemble':np.testing.assert_array_equal(value,np.mean([saved[arm,key[:3]+(str(seed),)] for seed in p.SEEDS],axis=0))
        elif arm=='game_ha':
            ha=saved['game_sampling',key];dead=study.previous.parent_probability(cells[key]);mass=ha[:,:2].sum(axis=1)
            home=np.divide(ha[:,0],mass,out=np.full(len(mass),.5),where=mass>0)
            expected=np.column_stack((home*(1-dead[:,2]),(1-home)*(1-dead[:,2]),dead[:,2]))
            np.testing.assert_allclose(value,expected,rtol=0,atol=1e-15)
    for row in p.read(study.AUDIT)['rows']:
        f=study.OUT/'models'/f"{row['fold']}_{row['seed']}.joblib";r=p.read(f.with_suffix('.json'));model=joblib.load(f)
        old=p.read(study.previous.OUT/'models'/f"{row['fold']}_{row['seed']}.json")
        assert p.sha(f)==r['sha256'] and model['step']==2048
        assert r['sampling']==row['game_sampling'] and r['exposures']==old['exposures']
        assert r['sampling']['counts']['old']==old['sampling']['counts']['old']
        assert r['old_matches']==sorted(x['match'] for x in p.prep.split_plan(fold=row['fold'])['train'])
        assert len(r['config']['soccernet_games'])==45
    for rel,digest in p.read(study.previous_head.DECL)['source_sha256'].items():assert p.sha(ROOT/rel)==digest
    assert count==90
    receipt={'passed':True,'official_cells':count,'models':6,'unchanged_native_grading':True,'ensemble_and_cross_checked':True,'old_sampling_matches_within_record_fit':True,'aggregate_sha256':p.sha(path),'verifier_sha256':p.sha(Path(__file__)),'organizer_reference_access':False}
    p.write(p.RESULTS/f'{study.STUDY}_verification.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
