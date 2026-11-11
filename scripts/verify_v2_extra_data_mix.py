"""Independent probability/decoder/official-score audit of the four cached blends."""
from pathlib import Path
import json
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import v2_extra_data_mix_eval as study
from scripts import verify_v2_extra_data as independent

OUT=ROOT/'experiments/results/v2_extra_data_mix_v1_verification.json'


def close(actual, expected, where):
    if abs(actual-expected)>1e-12:raise AssertionError(where)


def verify():
    declaration=study.require_declaration()
    result=json.loads(study.RESULT.read_text())
    if result['declaration_sha256']!=study.sha(study.DECLARATION):
        raise ValueError('Result declaration changed')
    recipes=result['recipes']
    if recipes!=declaration['recipes'] or len(recipes)!=4:raise ValueError('Recipe inventory changed')
    rows,seed_rows={},{}
    for index,(cell,source) in enumerate(sorted(declaration['upstream']['prediction_cache'].items()),1):
        control=independent._load_control(cell,declaration['upstream']['control_inputs'][cell])
        old=independent.shared.runtime.frozen.audit.decode(control['adopted'],control['model_period'],control['fps'],independent.shared.runtime.frozen.audit.CONFIG)
        fact=result['prediction_cache'][cell]
        path=ROOT/fact['path']
        if study.sha(path)!=fact['sha256']:raise ValueError('Candidate predictions changed')
        rows[cell],seed_rows[cell]={},{}
        with np.load(ROOT/source['path'],allow_pickle=False) as previous, np.load(path,allow_pickle=False) as saved:
            if str(saved['identity__cell'].item())!=cell or str(saved['identity__declaration_sha256'].item())!=study.sha(study.DECLARATION):
                raise ValueError('Prediction identity mismatch')
            for recipe in recipes:
                rid,weights=recipe['recipe_id'],recipe['components']
                ce=sum(weight*previous[f'pooled__{arm}'] for arm,weight in weights.items())
                rows[cell][rid]=None;seed_rows[cell][rid]={}
                for seed in (None,*study.SEEDS):
                    if seed is None:
                        p=independent._blend(ce,control['calibrated'],control['adopted'],study.BASE_RECIPE)
                        probkey,predkey=f'prob__{rid}',f'pred__{rid}'
                    else:
                        pieces={}
                        for arm in weights:
                            prefix=f'phase__{arm}__{seed}__'
                            phases=sorted(int(k[len(prefix):]) for k in previous.files if k.startswith(prefix))
                            expected=list(range(round(control['fps']/5)))
                            if phases!=expected:raise ValueError('Missing native phases')
                            pieces[arm]=np.mean([previous[f'{prefix}{phase}'] for phase in phases],axis=0)
                        seed_ce=sum(weight*pieces[arm] for arm,weight in weights.items())
                        p=independent._blend(seed_ce,control[f'adop_seed{seed}'],control[f'cand_seed{seed}'],study.BASE_RECIPE)
                        probkey,predkey=f'seedprob__{rid}__{seed}',f'seedpred__{rid}__{seed}'
                    np.testing.assert_allclose(p,saved[probkey],rtol=0,atol=1e-12)
                    pred=independent._fixed_d_decode(p,old,control['model_period'],control['fps'])
                    decoded=np.asarray(['H','A','D','X'])[saved[predkey]]
                    np.testing.assert_array_equal(pred,decoded)
                    np.testing.assert_array_equal(pred=='D',old=='D')
                    scored=independent._score(control,pred)
                    wanted=result['cell_metrics'][cell][rid] if seed is None else result['seed_cells'][cell][rid][str(seed)]
                    for metric in study.METRICS:close(scored[metric],wanted[metric],f'{cell}/{rid}/{seed}/{metric}')
                    if seed is None:rows[cell][rid]=scored
                    else:seed_rows[cell][rid][str(seed)]=scored
        print(f'MIX AUDIT {index}/35 {cell}',flush=True)
    frontier=json.loads(study.PRIOR_REPLAY.read_text())['aggregate']
    aggregates={};gates={}
    for recipe in recipes:
        rid=recipe['recipe_id']
        aggregates[rid]=independent._aggregate({c:r[rid] for c,r in rows.items()})
        for seed in (None,*study.SEEDS):
            agg=aggregates[rid] if seed is None else independent._aggregate({c:r[rid][str(seed)] for c,r in seed_rows.items()})
            saved=result['aggregate'][rid] if seed is None else result['seed_diagnostics'][rid][str(seed)]
            for view,_ in study.VIEWS:
                for metric in study.METRICS:
                    close(agg[view][metric]['equal_provider'],saved[view][metric]['equal_provider'],f'{rid}/{seed}/{view}/{metric}')
                    for provider,x in agg[view][metric]['providers'].items():
                        close(x,saved[view][metric]['providers'][provider],f'{rid}/{seed}/{view}/{metric}/{provider}')
        checks={}
        for view,_ in study.VIEWS:
            for metric in ('spell_f1','macro_f1','short_spell_recall_iou_0_5_lt3s'):
                actual=aggregates[rid][view][metric]['equal_provider']
                floor=frontier[view][metric]['equal_provider']-(.001 if metric=='macro_f1' else 0)
                passed=actual>floor if view=='feed_like_v2' and metric=='spell_f1' else actual>=floor
                recorded=result['eligibility'][rid]['checks'][f'{view}/{metric}']
                close(actual,recorded['actual'],'Guard actual');close(floor,recorded['minimum'],'Guard floor')
                if passed!=recorded['passed']:raise AssertionError('Guard verdict mismatch')
                checks[f'{view}/{metric}']=passed
        gates[rid]=all(checks.values())
        recorded_gate=result['eligibility'][rid]
        if not recorded_gate['d_metrics_unchanged'] or not recorded_gate['reference_spell_counts_invariant']:
            raise AssertionError('D or reference-count guard failed')
        if gates[rid]!=result['eligibility'][rid]['eligible']:raise AssertionError('Eligibility mismatch')
    ranking=sorted((rid for rid,passed in gates.items() if passed),key=lambda rid:(-aggregates[rid]['feed_like_v2']['spell_f1']['equal_provider'],-aggregates[rid]['feed_like_v2']['macro_f1']['equal_provider'],rid))
    if ranking!=result['eligible_ranking']:raise AssertionError('Ranking changed')
    receipt=dict(study_id=study.STUDY,passed=True,cells=35,candidates=4,official_scorecards_recomputed=420,
        declaration_sha256=study.sha(study.DECLARATION),result_sha256=study.sha(study.RESULT),
        source_sha256={str(p.relative_to(ROOT)):study.sha(p) for p in (Path(__file__),Path(independent.__file__))},
        eligible_ranking=ranking,aggregate=aggregates,exact_prediction_and_D_replay=True,
        public_scoring=False,organizer_reference_access=False)
    with OUT.open('x') as f:json.dump(receipt,f,indent=2,sort_keys=True);f.write('\n')
    return {'passed':True,'eligible_ranking':ranking}

if __name__=='__main__':print(verify())
