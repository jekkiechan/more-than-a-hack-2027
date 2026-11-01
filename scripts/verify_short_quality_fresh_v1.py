"""Replay the fixed winner's new-corruption score and component algebra."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments import short_quality_fresh_v1 as study
from scripts.verify_feed_fresh_and_diversity import scored_array,aggregate
p=study.p


def main():
    study.check();path=p.RESULTS/f'{study.STUDY}_aggregate.json';result=p.read(path)
    original={(c['provider'],c['match']):c for c in p.shared._load_cells(p.read(p.DECLARATION))
              if c['member']=='ensemble' and c['view']=='clean'}
    saved={};count=0;arm=result['selection']['arm']
    for name,rows in result['rows'].items():
        for row in rows:
            if row['view']=='clean':continue
            match=('skillcorner_had_'+row['match']) if row['provider']=='SkillCorner' else row['match']
            replica=row['view'].removeprefix('feed_like_')
            file=study.OUT/'predictions'/name/f'{match}__{replica}__{row["member"]}.npz'
            arrays=scored_array(file,row,result['prediction_sha256'])
            base=original[row['provider'],row['match']]
            for field in ('labels','period','timestamps','model_period','fps'):
                np.testing.assert_array_equal(arrays[field],base[field])
            key=(match,replica,row['member']);saved[name,key]=arrays['probabilities'];count+=1
        for member in p.shared.MEMBERS:aggregate(rows,member,result['summaries'][name][member])
    def cross(ha,raw):
        mass=ha[:,:2].sum(axis=1)
        home=np.divide(ha[:,0],mass,out=np.full(len(mass),.5),where=mass>0)
        return np.column_stack((home*(1-raw[:,2]),(1-home)*(1-raw[:,2]),raw[:,2]))
    for (name,key),value in saved.items():
        if key[-1]=='ensemble':
            np.testing.assert_array_equal(value,np.mean([saved[name,key[:2]+(str(seed),)] for seed in p.SEEDS],axis=0))
            continue
        file=study.OUT/'components'/('__'.join(key)+'.npz')
        assert p.sha(file)==result['component_sha256'][str(file.relative_to(ROOT))]
        with np.load(file) as z:
            raw,lead,can=z['raw_feed'],z['leader_model'],z['candidate_model']
        before=cross(lead,raw);candidate=cross(can,raw)
        expected=before if name=='leader' else can if arm.endswith('_full') else .5*(candidate+before) if arm.endswith('_half') else candidate
        np.testing.assert_allclose(value,expected,rtol=0,atol=1e-15)
    assert count==60 and result['new_scored_cells']==60 and not result['reselection']
    assert result['selection']==p.read(study.OUT/'selection.json')
    receipt=dict(passed=True,official_new_cells=count,unchanged_native_grading=True,
        fixed_selection_and_probability_algebra_checked=True,independent_match_confirmation=False,
        aggregate_sha256=p.sha(path),verifier_sha256=p.sha(Path(__file__)),organizer_reference_access=False)
    p.write(p.RESULTS/f'{study.STUDY}_verification.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
