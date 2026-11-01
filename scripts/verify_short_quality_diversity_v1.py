"""Separate official-scorer, ensemble, component and sampling evidence replay."""
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from experiments import short_quality_diversity_v1 as study
from scripts.verify_feed_fresh_and_diversity import scored_array, aggregate

p=study.p


def main():
    study.check();p.check()
    path=p.RESULTS/f'{study.STUDY}_aggregate.json';result=p.read(path)
    cells={tuple(c[k] for k in ('provider','match','view','member')):c
           for c in p.shared._load_cells(p.read(p.DECLARATION))}
    saved={};count=0
    for arm in study.ARMS:
        for row in result['rows'][arm]:
            key=tuple(row[k] for k in ('provider','match','view','member'));c=cells[key]
            file=study.OUT/'predictions'/arm/('__'.join(key)+'.npz')
            arrays=scored_array(file,row,result['prediction_sha256'])
            for field in ('labels','period','timestamps','model_period','fps'):
                np.testing.assert_array_equal(arrays[field],c[field])
            saved[arm,key]=arrays['probabilities'];count+=1
        for member in p.shared.MEMBERS:
            aggregate(result['rows'][arm],member,result['summaries'][arm][member])
    for (arm,key),value in saved.items():
        if key[-1]=='ensemble':
            np.testing.assert_array_equal(value,np.mean([saved[arm,key[:3]+(str(seed),)] for seed in p.SEEDS],axis=0))
        elif arm.endswith('_ha'):
            ha=saved[arm.removesuffix('_ha')+'_full',key]
            dead=study.prior.parent_probability(cells[key]);mass=ha[:,:2].sum(axis=1)
            home=np.divide(ha[:,0],mass,out=np.full(len(mass),.5),where=mass>0)
            expected=np.column_stack((home*(1-dead[:,2]),(1-home)*(1-dead[:,2]),dead[:,2]))
            np.testing.assert_allclose(value,expected,rtol=0,atol=1e-15)
        elif arm=='diversity_leader_half':
            expected=.5*(saved['diversity_ha',key]+study.leader_probability(cells[key]))
            np.testing.assert_array_equal(value,expected)
    for row in p.read(study.AUDIT)['rows']:
        fold,seed=row['fold'],row['seed']
        previous=next(r for r in p.read(study.prior.AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)['targeted']
        for variant in study.VARIANTS:
            file=study.OUT/'models'/variant/f'{fold}_{seed}.joblib'
            r=p.read(file.with_suffix('.json'));m=joblib.load(file)
            assert p.sha(file)==r['sha256'] and m['step']==2048
            assert r['sampling']==row['sampling'][variant]
            assert r['sampling']['counts']['old']==previous['counts']['old']
            assert r['exposures']==previous['exposures']
            assert r['old_matches']==sorted(x['match'] for x in p.prep.split_plan(fold=fold)['train'])
            assert r['sampling']['eligible_quality_games']==45
            teacher=joblib.load(p.sc.initial_path(fold,seed))
            for field in ('median','scale'):np.testing.assert_array_equal(m[field],teacher[field])
    assert count==225
    baseline=result['baseline'];control=result['control']
    guards={a:bool(study.prior.selection.eligible(s['ensemble'],baseline)) for a,s in result['summaries'].items()}
    assert guards==result['guards_pass']
    eligible=[a for a in study.ARMS if guards[a] and study.prior.selection.rank(result['summaries'][a]['ensemble'])>study.prior.selection.rank(control)]
    winner=max(eligible,key=lambda a:study.prior.selection.rank(result['summaries'][a]['ensemble'])) if eligible else None
    assert winner==result['winner']
    receipt=dict(passed=True,official_cells=count,models=12,unchanged_native_grading=True,
        ensemble_component_and_blend_checked=True,full_provider_exclusions_checked=True,
        training_only_normalizers_checked=True,source_and_game_exposure_exact=True,
        aggregate_sha256=p.sha(path),verifier_sha256=p.sha(Path(__file__)),organizer_reference_access=False)
    p.write(p.RESULTS/f'{study.STUDY}_verification.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
