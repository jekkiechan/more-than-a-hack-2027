"""Read saved evidence into the spell-balanced report/dashboard comparison."""
import json
from pathlib import Path
from experiments import v2_spell_balanced_eval as study

ROOT = study.ROOT
OUT = ROOT / f'experiments/results/{study.STUDY}_readout.json'


def build():
    selection_path=ROOT/f'experiments/results/{study.STUDY}_selection.json'
    public_path=ROOT/f'experiments/results/{study.STUDY}_public_result.json'
    paths=[study.CONTROL_REPLAY, study.RESULT,
           ROOT/f'experiments/results/{study.STUDY}_round2.json',selection_path,public_path]
    replay,first,second,selected,public=[json.loads(p.read_text()) for p in paths]
    metrics=study.METRICS
    def values(aggregate):
        result={v:{m:aggregate[v][m]['equal_provider'] for m in metrics} for v,_ in study.VIEWS}
        for row in result.values(): row['auxiliary_70_30']=.7*row['spell_f1']+.3*row['macro_f1']
        return result
    baseline=values(replay['adopted'])
    rows=[dict(id='adopted',label='Adopted baseline',stage=0,selected=False,highlight=True,
               eligible=None,failed=[],values=baseline,providers=replay['adopted'])]
    anchor=study.recipe_key(study.ANCHOR)
    for stage,result in ((512,first),(1024,second)):
        for recipe in result['recipes']:
            rid=recipe['recipe_id']; gates=result['eligibility'][rid]
            winner=stage==selected['training_stage'] and rid==selected['recipe']['recipe_id']
            failed=[key for key,check in gates['checks'].items() if not check['passed']]
            failed += [key for key in ('d_metrics_unchanged','reference_spell_counts_invariant') if not gates[key]]
            label=f"{recipe['ce_weight']*100:g}/{recipe['calibrated_weight']*100:g}/{recipe['adopted_weight']*100:g} {'mean' if recipe['pooling']=='arithmetic' else 'log'} · HA {recipe['ha_cost']:g}"
            if stage==512 and rid==anchor: label='Original promising blend'
            if winner: label='Selected · '+label
            rowvalues=values(result['aggregate'][rid])
            rows.append(dict(id=f'{stage}/{rid}',recipe=recipe,label=label,stage=stage,
                selected=winner,highlight=winner or (stage==512 and rid==anchor) or rid==selected['recipe']['recipe_id'],
                eligible=gates['eligible'],failed=failed,values=rowvalues,
                delta_vs_adopted={v:{m:value-baseline[v][m] for m,value in row.items()} for v,row in rowvalues.items()},
                providers=result['aggregate'][rid],
                seeds=result.get('seed_diagnostics',{}).get(rid),
                exceeds_v2_0_60=rowvalues['feed_like_v2']['spell_f1']>.60))
    winner=next(row for row in rows if row['selected'])
    anchorrow=next(row for row in rows if row['stage']==512 and row['recipe']['recipe_id']==anchor)
    leader=max(rows[1:],key=lambda row:(row['values']['feed_like_v2']['spell_f1'],row['values']['feed_like_v2']['macro_f1']))
    result=dict(study_id=study.STUDY,date='2026-11-07',rows=rows,
        selected_id=winner['id'],numerical_leader_id=leader['id'],
        eligible_counts={str(stage):sum(row['stage']==stage and bool(row['eligible']) for row in rows) for stage in (512,1024)},
        selected_delta_vs_starting_arm={v:{m:value-anchorrow['values'][v][m] for m,value in row.items()} for v,row in winner['values'].items()},
        public=public, target_reached=winner['exceeds_v2_0_60'],
        source_sha256=study.sha(Path(__file__)),
        evidence_sha256={str(p.relative_to(ROOT)):study.sha(p) for p in paths},
        evidence_scope='Five exposed development matches; four fixed synthetic v2 realizations; reused 17-possession public diagnostic; no independent confirmation',
        adoption_changed=False)
    study._write_once(OUT,result)
    return result


if __name__=='__main__':
    r=build();print(json.dumps({'selected':r['selected_id'],'eligible_counts':r['eligible_counts'],'target_reached':r['target_reached']},indent=2))
