"""Build an aggregate-only readout and dashboard payload from sealed results."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'experiments/results'
VIEWS = ('clean', 'feed_like_v1', 'feed_like_v2')
METRICS = ('spell_f1','macro_f1','d_f1','d_precision','d_recall','short_spell_recall_iou_0_5_lt3s')

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def build():
    replay_path = RESULTS/'v2_iterative_v1_control_replay.json'
    replay=json.loads(replay_path.read_text())
    assert replay['passed']
    sources={str(replay_path.relative_to(ROOT)):sha(replay_path)}
    rows=[dict(id='adopted',label='Adopted original-full',round=0,passed=False,retention_passed=True,failed=[],
               values={v:{m:replay['adopted_aggregate'][v][m]['equal_provider'] for m in METRICS} for v in VIEWS})]
    labels={
      'joint_probability_blend_ha3':'Joint archived blend',
      'fixed_d_conditional_ha3':'Archived team blend · cost 3',
      'fixed_d_conditional_ha1_5':'Archived team blend · cost 1.5',
      'fixed_d_logit_ha3':'Archived logit blend',
      'ce_control_fixed_d_ha3':'Continuation control',
      'boundary_margin_fixed_d_ha3':'Boundary margin',
      'ce_control_conditional_blend_fixed_d_ha3':'Control + adopted',
      'boundary_margin_conditional_blend_fixed_d_ha3':'Boundary + adopted',
    }
    round_results={}
    for number,name in ((1,'v2_iterative_v1_round1'),(2,'v2_iterative_v1_round2'),(3,'v2_iterative_v1_round3')):
        path=RESULTS/(name+'.json')
        result=json.loads(path.read_text())
        round_results[number]=result
        sources[str(path.relative_to(ROOT))]=sha(path)
        for arm,gate in result['strict_success_gates'].items():
            agg=result['aggregate'][arm]
            values={v:{m:(agg[m][v] if number==1 else agg[v][m])['equal_provider'] for m in METRICS} for v in VIEWS}
            failed=[k for k,c in gate['checks'].items() if not c['passed']]
            retention=all(c['passed'] for k,c in gate['checks'].items() if 'strictly_above' not in k)
            rows.append(dict(id=arm,label=labels.get(arm,arm.replace('_',' ')),round=number,
                             passed=gate['all_passed'],retention_passed=retention,failed=failed,values=values))
    candidates=rows[1:]
    ranked=sorted(candidates,key=lambda r:(-r['values']['feed_like_v2']['spell_f1'],-r['values']['feed_like_v2']['macro_f1'],r['id']))
    value=dict(study_id='v2_iterative_v1',date='2026-11-06',rows=rows,
               result_sha256=sources,qualifying_candidates=[r['id'] for r in ranked if r['passed']],
               best_v2_candidate=ranked[0]['id'],exposure='Five previously selected-on development matches; synthetic stress, GAR silver TRAIN supervision; no independent confirmation',
               promotion=False,raw_artifacts='Locally verified only')
    path=RESULTS/'v2_iterative_v1_readout.json'
    with path.open('x') as f: json.dump(value,f,indent=1,sort_keys=True);f.write('\n')
    lines=['## Completed scorecard','',
           'Equal-provider official scores. All displayed values are rounded; decisions use the saved unrounded values. The strict retention check covers six quality metrics on each of clean, v1 and v2.','',
           '| Round / pipeline | Clean spell | v1 spell | v2 spell | v2 macro | Retains all metrics | >.60 and retention |',
           '| --- | ---: | ---: | ---: | ---: | --- | --- |']
    for r in rows:
        a=r['values']; lines.append(f"| {r['round']} / {r['label']} | {a['clean']['spell_f1']:.6f} | {a['feed_like_v1']['spell_f1']:.6f} | {a['feed_like_v2']['spell_f1']:.6f} | {a['feed_like_v2']['macro_f1']:.6f} | {'Yes' if r['retention_passed'] else 'No'} | {'Yes' if r['passed'] else 'No'} |")
    leader=ranked[0]; leader_result=round_results[leader['round']]; arm=leader['id']; base=rows[0]['values']
    lines.extend(['','### Numerical leader: protected-metric deltas','',
                  '| View | Spell | Macro | D F1 | D precision | D recall | Short recall |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for view in VIEWS:
        lines.append('| '+view+' | '+' | '.join(f"{leader['values'][view][m]-base[view][m]:+.6f}" for m in METRICS)+' |')
    lines.extend(['','### Numerical leader: provider and seed diagnostics','',
                  '| View | Metrica spell | SkillCorner spell | Sportec spell | Seed 2026 spell | Seed 2027 spell | 70/30 auxiliary |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for view in VIEWS:
        agg=leader_result['aggregate'][arm]
        provider=(agg['spell_f1'][view] if leader['round']==1 else agg[view]['spell_f1'])['providers']
        if leader['round']==1: seed_values=leader_result['seed_diagnostics'][arm]
        elif leader['round']==3: seed_values=leader_result['seed_aggregate'][arm]
        else:
            recipe='boundary_margin' if arm.startswith('boundary') else 'ce_control'
            variant='conditional_blend' if 'conditional_blend' in arm else 'fixed_d'
            seed_values=leader_result['seed_aggregate'][recipe][variant]
        values=[provider[p] for p in ('Metrica','SkillCorner','Sportec')]
        values += [seed_values[str(seed)][view]['spell_f1']['equal_provider'] for seed in (2026,2027)]
        values += [.7*leader['values'][view]['spell_f1']+.3*leader['values'][view]['macro_f1']]
        lines.append('| '+view+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines.extend(['','Two fixed seeds are sensitivity diagnostics, not a confidence interval; neither seed is selected. Provider weighting gives each of the three providers equal mass. The auxiliary score does not override failed checks.'])
    lines.extend(['','### Every failed check',''])
    for r in candidates: lines.append(f"- **{r['label']}:** "+(', '.join('`'+k+'`' for k in r['failed']) or 'None')+'.')
    lines.extend(['','The [machine-readable readout](results/v2_iterative_v1_readout.json) contains every protected metric; the round result files also retain provider, individual-seed and spell-count diagnostics.'])
    report=ROOT/'experiments/V2_ITERATIVE_DATA.md'
    body=report.read_text().split('\n## Completed scorecard')[0]
    report.write_text(body+'\n'+'\n'.join(lines)+'\n')
    print(json.dumps({'best':ranked[0]['id'],'v2_spell':ranked[0]['values']['feed_like_v2']['spell_f1'],'qualifying':value['qualifying_candidates']}))
    return value

if __name__=='__main__': build()
