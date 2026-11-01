"""Independent native-array and official-score replay for component transfer."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments import feed_robustness_v1 as p, feed_head_transfer_v1 as h
from experiments import spell_mbr_run as shared, metrics

def main():
    result=p.read(p.RESULTS/'feed_head_transfer_v1_aggregate.json')
    parent=p.read(p.RESULTS/'feed_robustness_v1_aggregate.json')
    cells=shared._load_cells(p.check())
    by_key={tuple(c[k] for k in ('provider','match','view','member')):c for c in cells}
    saved={}; max_error=0.
    for arm in h.ARMS:
        for row in result['rows'][arm]:
            key=tuple(row[k] for k in ('provider','match','view','member'))
            c=by_key[key]
            path=h.OUT/'predictions'/arm/('__'.join(key)+'.npz')
            assert p.sha(path)==result['prediction_sha256'][str(path.relative_to(ROOT))]
            with np.load(path) as z:
                prob=z['probabilities']; pred=z['prediction']
                for k in ('labels','timestamps','period','model_period','fps'):
                    np.testing.assert_array_equal(z[k],c[k])
            saved[arm,key]=prob
            np.testing.assert_array_equal(pred,p.decode(prob,c['model_period'],c['fps']))
            v=metrics.evaluate_tracking(c['labels'],pred,period=c['period'],timestamps=c['timestamps'],fps=c['fps'])
            for k,a in [('spell_f1',v['spell_f1_mean']),('macro_f1',v['macro_f1']),('d_f1',v['per_class']['D']['f1'])]:
                error=abs(a-row[k]);max_error=max(max_error,error)
                assert error<=1e-12
            if key[-1]!='ensemble':
                cp=p.OUT/'predictions/feed_augmented'/('__'.join(key)+'.npz')
                assert p.sha(cp)==parent['prediction_sha256'][str(cp.relative_to(ROOT))]
                with np.load(cp) as z:new=z['probabilities']
                team,dead=(new,c['probabilities']) if arm=='robust_ha' else (c['probabilities'],new)
                total=team[:,:2].sum(axis=1)
                home=np.divide(team[:,0],total,out=np.full(len(total),.5),where=total>0)
                expected=np.column_stack([home*(1-dead[:,2]),(1-home)*(1-dead[:,2]),dead[:,2]])
                np.testing.assert_allclose(prob,expected,rtol=0,atol=3e-16)
        for key in by_key:
            if key[-1]=='ensemble':
                np.testing.assert_array_equal(saved[arm,key],np.mean([saved[arm,key[:-1]+(str(s),)] for s in p.SEEDS],axis=0))
        for member in shared.MEMBERS:
            for view in ('clean','feed_like'):
                for metric in ('spell_f1','macro_f1','d_f1'):
                    providers=[]
                    for provider,matches in shared.EXPECTED_MATCHES.items():
                        match_means=[]
                        for match in matches:
                            values=[r[metric] for r in result['rows'][arm] if r['member']==member and r['provider']==provider and r['match']==match and (r['view']=='clean' if view=='clean' else r['view'].startswith('feed_like_'))]
                            assert len(values)==(1 if view=='clean' else 2)
                            match_means.append(sum(values)/len(values))
                        providers.append(sum(match_means)/len(match_means))
                    assert abs(sum(providers)/len(providers)-result['summaries'][arm][member][view]['equal_provider_mean'][metric])<=1e-12
    p.write(p.RESULTS/'feed_head_transfer_v1_verification.json',{'passed':True,'cells':90,'native_files':90,'maximum_score_difference':max_error,'component_and_ensemble_parity':True,'unchanged_native_grading':True,'aggregate_sha256':p.sha(p.RESULTS/'feed_head_transfer_v1_aggregate.json'),'verifier_sha256':p.sha(Path(__file__)),'organizer_reference_access':False})
    print('Verified 90 component-transfer cells')

if __name__=='__main__':main()
