"""Combine saved balanced/frontier evidence without rerunning evaluation."""
import json
from pathlib import Path
from experiments import v2_spell_frontier_eval as frontier
from experiments import v2_spell_balanced_eval as balanced
ROOT=frontier.ROOT
OUT=ROOT/f'experiments/results/{frontier.STUDY}_readout.json'


def build():
    def read(name):return json.loads((ROOT/'experiments/results'/name).read_text())
    chosen=read(frontier.STUDY+'_selection.json')
    public=read(frontier.STUDY+'_public_result.json')
    base=read('v2_spell_balanced_v1_control_replay.json')
    def values(a):
        result={v:{m:a[v][m]['equal_provider'] for m in balanced.METRICS} for v,_ in balanced.VIEWS}
        for row in result.values():row['auxiliary_70_30']=.7*row['spell_f1']+.3*row['macro_f1']
        return result
    baseline=values(base['adopted'])
    rows=[dict(id='adopted',label='Adopted baseline',stage=0,selected=False,highlight=True,eligible=None,
               failed=[],values=baseline,providers=base['adopted'])]
    counts={}
    paths=[ROOT/'experiments/results'/name for name in
           ('v2_spell_balanced_v1_control_replay.json',frontier.STUDY+'_selection.json',frontier.STUDY+'_public_result.json')]
    for study,stage,round_number in (('v2_spell_balanced_v1',512,1),('v2_spell_balanced_v1',1024,2),
                                    (frontier.STUDY,1024,1),(frontier.STUDY,1536,2)):
        path=ROOT/f'experiments/results/{study}_round{round_number}.json';paths.append(path)
        r=json.loads(path.read_text())
        pipelines=r.get('pipelines') or [dict(pipeline_id=x['recipe_id'],recipe=x,training_arm='base',training_stage=stage) for x in r['recipes']]
        count=0
        for pipeline in pipelines:
            key=pipeline['pipeline_id'];recipe=pipeline['recipe'];arm=pipeline['training_arm'];gate=r['eligibility'][key]
            count+=int(gate['eligible'])
            rid=recipe['recipe_id']
            winner=study==frontier.STUDY and stage==chosen['training_stage'] and arm==chosen['training_arm'] and rid==chosen['recipe']['recipe_id']
            label=f"{recipe['ce_weight']*100:g}/{recipe['calibrated_weight']*100:g}/{recipe['adopted_weight']*100:g} {'mean' if recipe['pooling']=='arithmetic' else 'log'} · HA {recipe['ha_cost']:g}"
            label=f'{study.removeprefix("v2_spell_").removesuffix("_v1")} · {arm} · '+label
            anchor=stage==512 and rid==balanced.recipe_key(balanced.ANCHOR)
            parent=study=='v2_spell_balanced_v1' and stage==1024 and rid==frontier.LEADER_IDS[0]
            shorter=study=='v2_spell_balanced_v1' and stage==512 and rid==frontier.LEADER_IDS[0]
            if anchor:label='Original promising blend'
            if winner:label='Final selection · '+label
            failures=[k for k,c in gate['checks'].items() if not c['passed']]
            failures += [k for k in ('d_metrics_unchanged','reference_spell_counts_invariant') if not gate[k]]
            val=values(r['aggregate'][key])
            rows.append(dict(id=f'{study}/r{round_number}/{key}',label=label,stage=stage,training_arm=arm,recipe=recipe,
                selected=winner,highlight=winner or anchor or parent or shorter,eligible=gate['eligible'],failed=failures,
                values=val,providers=r['aggregate'][key],seeds=r.get('seed_diagnostics',{}).get(key),
                delta_vs_adopted={v:{m:x-baseline[v][m] for m,x in row.items()} for v,row in val.items()}))
        counts[f'{study}/r{round_number}']={'eligible':count,'total':len(pipelines)}
    selected=next(row for row in rows if row['selected'])
    result=dict(study_id=frontier.STUDY,date='2026-11-07',rows=rows,selected_id=selected['id'],eligible_counts=counts,
                public=public,target_reached=chosen['aspiration_above_0_60'],adoption_changed=False,
                source_sha256=frontier.sha(Path(__file__)),
                evidence_sha256={str(p.relative_to(ROOT)):frontier.sha(p) for p in paths},
                evidence_scope='Exposed five-match development cohort; synthetic stress; one final reused 17-possession public diagnostic')
    frontier._write_once(OUT,result)
    return result

if __name__=='__main__':
    r=build();print(json.dumps({k:r[k] for k in ('selected_id','eligible_counts','target_reached')},indent=2))
