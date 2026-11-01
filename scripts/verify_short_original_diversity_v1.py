"""Official score and probability/sampling algebra replay for the continuation."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from experiments import short_original_diversity_v1 as study
from scripts.verify_feed_fresh_and_diversity import scored_array,aggregate
p=study.p


def main():
    study.check();path=p.RESULTS/f'{study.STUDY}_aggregate.json';result=p.read(path)
    cells={tuple(c[k] for k in ('provider','match','view','member')):c for c in p.shared._load_cells(p.read(p.DECLARATION))}
    saved={};count=0
    for arm in study.ARMS:
        for row in result['rows'][arm]:
            key=tuple(row[k] for k in ('provider','match','view','member'))
            arrays=scored_array(study.OUT/'predictions'/arm/('__'.join(key)+'.npz'),row,result['prediction_sha256'])
            for field in ('labels','period','timestamps','model_period','fps'):
                np.testing.assert_array_equal(arrays[field],cells[key][field])
            saved[arm,key]=arrays['probabilities'];count+=1
        for member in p.shared.MEMBERS:aggregate(result['rows'][arm],member,result['summaries'][arm][member])
    for (arm,key),value in saved.items():
        if key[-1]=='ensemble':
            np.testing.assert_array_equal(value,np.mean([saved[arm,key[:3]+(str(s),)] for s in p.SEEDS],axis=0))
        elif arm=='original_ha':
            ha=saved['original_full',key];dead=study.parent.prior.parent_probability(cells[key]);mass=ha[:,:2].sum(axis=1)
            home=np.divide(ha[:,0],mass,out=np.full(len(mass),.5),where=mass>0)
            np.testing.assert_allclose(value,np.column_stack((home*(1-dead[:,2]),(1-home)*(1-dead[:,2]),dead[:,2])),rtol=0,atol=1e-15)
        elif arm=='original_leader_half':
            np.testing.assert_array_equal(value,.5*(saved['original_ha',key]+study.parent.leader_probability(cells[key])))
    for row in p.read(study.AUDIT)['rows']:
        fold,seed=row['fold'],row['seed'];file=study.OUT/'models'/f'{fold}_{seed}.joblib'
        r=p.read(file.with_suffix('.json'));m=joblib.load(file)
        assert p.sha(file)==r['sha256'] and m['step']==2048 and r['sampling']==row['sampling']
        expected=next(x['sampling']['diversity'] for x in p.read(study.parent.AUDIT)['rows'] if x['fold']==fold and x['seed']==seed)
        assert r['sampling']['counts']['new']==expected['counts']['new'] and r['exposures']==expected['exposures']
        assert r['old_matches']==sorted(x['match'] for x in p.prep.split_plan(fold=fold)['train'])
        teacher=joblib.load(p.sc.initial_path(fold,seed))
        for field in ('median','scale'):np.testing.assert_array_equal(m[field],teacher[field])
    assert count==135
    receipt=dict(passed=True,official_cells=count,models=6,unchanged_native_grading=True,
        ensemble_cross_blend_checked=True,full_provider_exclusions_checked=True,
        frozen_gar_schedule_and_source_exposure_exact=True,aggregate_sha256=p.sha(path),
        verifier_sha256=p.sha(Path(__file__)),organizer_reference_access=False)
    p.write(p.RESULTS/f'{study.STUDY}_verification.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
