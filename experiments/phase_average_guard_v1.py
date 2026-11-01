"""Complete missing metrics of the already-fixed phase-average candidate.

This reuses exposed original development matches. It supplies no new independent
evidence, fits no model, changes no recipe and never reads organizer labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import gar_deployment_eval_v2 as common
from experiments import feed_benchmark_audit_v1 as audit

ROOT=common.ROOT
OUT=ROOT/'data/experiments/phase_average_guard_v1'
ASSETS=OUT/'assets/experiments'
DECLARATION=ROOT/'experiments/results/phase_average_guard_v1_declaration.json'
RESULT=ROOT/'experiments/results/phase_average_guard_v1.json'
MODEL=ASSETS/'short_quality_diversity_delivery_v1/artifacts/short_original_diversity_v1/models'
TEACHER=ASSETS/'objective_alignment_v1'
INPUTS=ASSETS/'seed_gap_context_v1'
VIEWS=('clean','feed_like_corrupt_1','feed_like_corrupt_2')


def declare():
    restored=common.read(OUT/'restore_receipt.json')
    for f in restored['files']:
        if common.digest(OUT/'assets'/f['path'])!=f['sha256']:
            raise ValueError('Restored asset mismatch')
    common.write_once(DECLARATION,dict(study='phase_average_guard_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Complete provider D/macro guards and seed diagnostics for the already-fixed original_full phase-average config',
        exposure='All five matches and existing recipe/decoder were previously exposed; not fresh confirmation',
        post_TEST_origin='Astra requested missing active-policy guard evidence after the fixed GAR TEST comparison',
        models='Exact archived provider-excluded original_full folds and paired combined-objective teachers; no refits',
        cells=[dict(provider=p,match=m,view=v) for p,m,_ in audit.MATCHES for v in VIEWS],
        modes=['phase0','phase_avg'],members=['ensemble','2026','2027'],score_cells=90,
        inference='Native masks/breaks preserved; all native 5Hz phases; equal raw seed probability mean then phase mean',
        decoder='Unchanged f062 temperature4/cost3',
        guards=dict(feed_spell_floor=-.01,provider_clean_macro_floor=-.02,provider_clean_d_f1_floor=-.02,
            historical_clean_screen=.01,component_seeds='reported, not new selection arms'),
        replay_tolerance=1e-12,
        expected_metric_sources={n:common.digest(ROOT/'experiments/results'/n) for n in
            ['feed_benchmark_audit_v1.json','feed_like_v2_scorecard.json']},
        assets=restored['files'],
        source_sha256={str(p.relative_to(ROOT)):common.digest(p) for p in
            [Path(__file__),Path(audit.__file__),Path(common.sn.__file__),ROOT/'experiments/soccernet_capacity_model.py']},
        fitting=False,organizer_reference_access=False,public_scoring=False,promotion=False))


def check():
    d=common.read(DECLARATION)
    for name,sha in d['source_sha256'].items():
        if common.digest(ROOT/name)!=sha:raise ValueError('Frozen source changed')
    return d


def run():
    check()
    for provider,match,filename in audit.MATCHES:
        fold=audit.FOLD[provider]
        models={s:joblib.load(MODEL/f'leave_out_{fold}_{s}.joblib') for s in (2026,2027)}
        teachers={s:joblib.load(TEACHER/f'without_{fold}_combined_{s}.joblib') for s in (2026,2027)}
        for view in VIEWS:
            dest=OUT/'cells'/f'{match}__{view}.json'
            if dest.exists():continue
            suffix='clean_clean' if view=='clean' else view.replace('feed_like_corrupt_','feed_like_corrupt_')
            cached=joblib.load(INPUTS/f'{filename}_{suffix}.joblib')
            inputs,labels,record=cached['inputs'],np.asarray(cached['labels']).astype(str),cached['record']
            phases=audit.phase_records(inputs,record)
            if not np.array_equal(phases[0]['x'],record['x'],equal_nan=True):
                raise ValueError('Archived phase0 feature replay failed')
            native=audit.tc.inputs_only(inputs)
            values=[]
            for phase in phases:
                phase=common.old.inference_record(phase)
                phase['names']=list(phase['names'])
                values.append(np.stack([common.old.upsample(common.sn.predict(phase,models[s],teachers[s]),phase,native)
                    for s in (2026,2027)]))
            bymode={'phase0':values[0],'phase_avg':np.mean(values,axis=0)}
            period=common.audit.supervised.model_periods(inputs)
            cell=dict(labels=labels,period=inputs['period'],timestamps=inputs['timestamps'],fps=float(inputs['fps']))
            rows,predictions={},{}
            for mode,seeds in bymode.items():
                ensemble=np.mean(seeds,axis=0) if mode=='phase0' else np.mean([np.mean(x,axis=0) for x in values],axis=0)
                for member,p in [('ensemble',ensemble),('2026',seeds[0]),('2027',seeds[1])]:
                    pred=audit.decode(p,period,float(inputs['fps']))
                    rows[f'{mode}/{member}']=audit.score(cell,pred)
                    predictions[f'{mode}__{member}']=pred.astype('S1')
            counts={(r['graded_frames'],r['reference_spell_count']) for r in rows.values()}
            if len(counts)!=1:raise ValueError('Grading changed across fixed arms')
            dest.parent.mkdir(parents=True,exist_ok=True)
            native_path=dest.with_suffix('.npz')
            np.savez_compressed(native_path,labels=labels.astype('S1'),period=inputs['period'],timestamps=inputs['timestamps'],
                fps=inputs['fps'],**predictions)
            common.write_once(dest,dict(provider=provider,match=match,view=view,scores=rows,
                declaration_sha256=common.digest(DECLARATION),prediction_sha256=common.digest(native_path)))
            print('DONE',provider,match,view,flush=True)


def report():
    d=check()
    rows=[common.read(OUT/'cells'/f"{c['match']}__{c['view']}.json") for c in d['cells']]
    summaries={}
    metrics=('spell_f1','macro_f1','d_f1','d_precision','d_recall','short_spell_recall_iou_0_5_lt3s',
        'reference_spell_count','predicted_spell_count','graded_frames','weighted_70_30')
    for member in d['members']:
        summaries[member]={}
        for mode in d['modes']:
            summaries[member][mode]={}
            for view,sub in [('clean',('clean',)),('feed_like',VIEWS[1:])]:
                table={}
                for metric in metrics:
                    mean,providers=audit.equal_provider({(r['provider'],r['match'],r['view']):r['scores'][f'{mode}/{member}'][metric]
                        for r in rows if r['view'] in sub})
                    table[metric]=dict(equal_provider_mean=mean,providers=providers)
                summaries[member][mode][view]=table
    # Verify both known modes against every published provider spell/macro cell.
    expected=common.read(ROOT/'experiments/results/feed_benchmark_audit_v1.json')['tta']
    errors=[]
    for mode,key in [('phase0','cand_phase0'),('phase_avg','cand_phase_tta')]:
        for view in ('clean','feed_like'):
            for metric,field in [('spell','spell_f1'),('macro','macro_f1')]:
                actual=summaries['ensemble'][mode][view][field]
                errors.append(abs(actual['equal_provider_mean']-expected[key][metric][view]['equal_provider_mean']))
                errors.extend(abs(v-expected[key][metric][view]['providers'][p]) for p,v in actual['providers'].items())
    replay=max(errors)
    if replay>d['replay_tolerance']:raise ValueError(f'Archived score replay failed: {replay}')
    initial,phase=summaries['ensemble']['phase0'],summaries['ensemble']['phase_avg']
    checks={'feed_spell_floor':phase['feed_like']['spell_f1']['equal_provider_mean']>=initial['feed_like']['spell_f1']['equal_provider_mean']-.01}
    for metric in ('macro_f1','d_f1'):
        for provider,v in phase['clean'][metric]['providers'].items():
            checks[f'{provider}_clean_{metric}_floor']=v>=initial['clean'][metric]['providers'][provider]-.02
    common.write_once(RESULT,dict(study='phase_average_guard_v1',declaration_sha256=common.digest(DECLARATION),
        score_cells=90,summaries=summaries,active_policy_guard_checks=checks,all_active_policy_guards_pass=all(checks.values()),
        historical_clean_screen_passes=False,historical_clean_screen_delta=phase['clean']['spell_f1']['equal_provider_mean']-initial['clean']['spell_f1']['equal_provider_mean'],
        max_published_score_replay_error=replay,organizer_reference_access=False,public_scoring=False,fitting=False,promotion=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('step',choices=('declare','run','report'))
    args=parser.parse_args()
    globals()[args.step]()
