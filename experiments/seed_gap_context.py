"""Two-seed probability ensemble, baseline gap audit, conditional context test.

External development only. Existing checkpoint probabilities are averaged
before the unchanged decoder. No organizer labels, promotion or gate reset.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from experiments import (learned_temporal as lt, factorized_had as model,
    gru_robustness_experiment as gr, live_gap_experiment as old,
    corruption_experiment as ce, dead_transfer as dt, multisource_eval as ms,
    supervised, stress_components as sc, tracking_corruption as tc,
    refine_features, robust, light_corruption as lc, gap_diagnostics as gd)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
OUT = ROOT/'data/experiments/seed_gap_context_v1'
RESULT = ROOT/'experiments/results/seed_gap_context.json'
DECLARATION = ROOT/'experiments/results/seed_gap_context_declaration.json'
FEED = ('ball_gaps','coordinates','slot_gaps','id_swaps')
SEEDS = (2026,2027)


def write(path, value, exclusive=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w') as f:
        json.dump(value, f, indent=1, allow_nan=False)
        f.write('\n')


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def fingerprints():
    # Seal the complete executable research surface, without historical reports.
    return {str(p.relative_to(ROOT)):sha256(p) for base in ('experiments','evaluation')
            for p in sorted((ROOT/base).rglob('*.py'))}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT.parent/'mtah-data/gru_slot_robustness_cache_manifest.json'
    if sha256(cache) != '46702a6be49685047e4a748d4154fd0dcee7fabe0e2bdcb27c46e1155210c512':
        raise ValueError('Wrong archived cache manifest')
    restored = json.loads(cache.read_text())
    # All 253 restored objects, not LFS pointers, must match the known receipt.
    for r in restored['files']:
        p = ROOT/'data'/r['path']
        if p.stat().st_size != r['bytes'] or sha256(p) != r['sha256']:
            raise ValueError(f'Archive mismatch {p}')
    inputs = {str(p.relative_to(ROOT)):sha256(p) for split in dt.SPLITS.values()
              for role in ('train','eval') for p in split[role]}
    checkpoints = {str(gr.checkpoint(gr.DEFAULT,s,'slot_dropout',seed).relative_to(ROOT)):
                   sha256(gr.checkpoint(gr.DEFAULT,s,'slot_dropout',seed))
                   for s in dt.SPLITS for seed in SEEDS}
    spec = dict(id='seed_gap_context_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        base_commit='7615dd5c738adbc63e92d19af233962777cc98eb',
        hypotheses=[
            'Equal averaging of the two frozen full H/A/D vectors improves spell recovery if seed errors complement.',
            'Long ball gaps and distance to reliable observed contact concentrate wrong-team errors on the current GRU.',
            'Conditional on that diagnosis, 15s rather than 5s context each side improves recovery while keeping 10s output cores.'],
        arms=['seed_2026','seed_2027','equal_probability_ensemble'],
        probability_ensemble='0.5 * (native H/A/D probability seed2026 + seed2027), then unchanged Viterbi cost3. No calibration.',
        views={'clean':['clean'],'feed_like':['corrupt_1','corrupt_2']},
        feed_components=FEED, calibration=json.loads((ROOT/'experiments/results/realistic_corruption_development.json').read_text())['spec']['calibration'],
        aggregation='Replicates within match, matches within provider, equal provider mean. Do not pool frames for headline model ranking.',
        holdout='Entire evaluation provider excluded from H/A, D and normalization; same five previously exposed external development matches.',
        cohort={s:{role:[p.stem for p in split[role]] for role in ('train','eval')} for s,split in dt.SPLITS.items()},
        inputs_sha256=inputs, checkpoint_sha256=checkpoints, source_sha256=fingerprints(),
        archived_manifest_sha256=sha256(cache), restored_files=len(restored['files']),
        policy_sha256=sha256(ROOT/'experiments/model_decision_policy.json'),
        baseline_manifest_sha256=sha256(ROOT/'experiments/baseline_manifest.json'),
        parity='All 10 clean individual-seed spell/macro/D-F1 rows and 20 feed-like rows must reproduce saved stress-component ablation to 1e-9; clean native probabilities to atol1e-6.',
        diagnostics=dict(gap_edges_seconds=[0,3,10,20,40,'infinity'],
            contact_distance_edges_seconds=[0,1,3,5,10,20,'infinity'],
            contacts='Existing geometry best-control >=.2, absolute H/A margin >=.2, stable .2s, native observed masks; no 10s age cap in diagnostic. Proxy, not ground truth.',
            gaps='Consecutive missing-ball intervals clipped at half/time breaks; frame strata apply inside gaps; no-gap separate. Duration includes whole interval, including unknown-label frames.',
            spells='Global official IoU.5 matches; midpoint strata plus maximum intersected gap; short <3s; fragmentation any positive same-team overlap with >=2 predictions. Graded-run edges reported.',
            turnovers='Direct adjacent H/A changes within graded runs. Greedy nearest same-direction one-to-one matches within 5s; report missed, within1s and signed/absolute timing; X/D breaks excluded.',
            distinction='On reference-live frames, wrong H/A, false D and X are disjoint; missed D reported separately.'),
        context_condition='Seed2026 feed-like: provider-balanced wrong-team rate > false-D rate among live gap>=10s; and wrong-team rate there exceeds live no-gap rate. This is a descriptive development trigger, not a significance test.',
        context_experiment=dict(context_each_side_seconds=[5,15], core_seconds=10, new_fits=6,
            seeds=SEEDS, folds=list(dt.SPLITS), hidden=24, epochs=8, batch=64, optimizer='AdamW .001/.0001, clip1',
            fixed='Features, labels, normalization population, augmentation whole-window choices, source/match weights, parameters, window count, update count, decoder3. Only context span changes.',
            falsifier='No positive matched spell delta and long-gap team-accuracy delta in both seeds; guardrail failure prevents recommending follow-up.',
            guardrails='Each seed clean spell >= own 5s-control -.01; feed-like spell >= own control -.01; each provider clean D-F1 and macro >= own control -.02.',
            budget='Six new fits, no early stopping, eight epochs, 30 new match/view/seed cells. Seal all six before scoring. No extra contexts or seed selection.'),
        organizer_labels_read=False, fresh_confirmation=False, promotion=False)
    write(DECLARATION,spec)
    write(OUT/'predeclared.json',spec)
    print('FROZEN',sha256(DECLARATION),flush=True)


def check():
    spec=json.loads(DECLARATION.read_text())
    if spec['source_sha256'] != fingerprints():
        raise ValueError('Executable source changed after freeze')
    for key,value in {**spec['inputs_sha256'],**spec['checkpoint_sha256']}.items():
        if sha256(ROOT/key) != value:
            raise ValueError(f'Frozen artifact changed: {key}')
    return spec


def cells():
    for source,split in dt.SPLITS.items():
        for path in split['eval']:
            for view in ('clean','feed_like'):
                for replicate in (('clean',) if view=='clean' else ('corrupt_1','corrupt_2')):
                    yield source,path,view,replicate


def native_probability(record, fitted, inputs):
    sampled=model.predict(record,fitted)
    period=supervised.model_periods(inputs)
    return np.column_stack([lt.upsample(sampled[:,k],record,period) for k in range(3)])


def average_probabilities(a,b):
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError('Expected matched full H/A/D arrays')
    for x in (a,b):
        if not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x.sum(1),1,atol=1e-5):
            raise ValueError('Invalid categorical probabilities')
    return .5*(a+b)


def score(inputs,labels,probability):
    prediction=supervised.decode(probability,supervised.model_periods(inputs),float(inputs['fps']),lt.COST)
    m=supervised.score(dict(inputs,labels=labels),prediction)
    result=dict(spell_f1=m['spell_f1_mean'],macro_f1=m['macro_f1'],
        weighted_70_30=.7*m['spell_f1_mean']+.3*m['macro_f1'],
        d_f1=m['per_class']['D']['f1'],d_precision=m['per_class']['D']['precision'],
        d_recall=m['per_class']['D']['recall'],n_gt_spells=m['n_gt_spells'],n_pred_spells=m['n_pred_spells'],
        spell_thresholds={str(k):v['f1'] for k,v in m['spell_f1'].items()},
        probability_sha256=digest_array(probability),prediction_sha256=digest_array(prediction))
    return result,prediction


def evaluate(match=None):
    spec=check()
    saved=json.loads((ROOT/'experiments/results/stress_component_ablation.json').read_text())['rows']
    for source,path,view,replicate in cells():
        if match is not None and path.stem != match:
            continue
        dest=OUT/f'{path.stem}_{view}_{replicate}.json'
        if dest.exists():
            continue
        start=time.monotonic()
        inputs,labels=ms.view(path)
        inputs=tc.inputs_only(inputs) if view=='clean' else sc.corrupt(inputs,spec['calibration'],ce.seed(path.stem,replicate),FEED)
        record=old.feature_view(inputs,labels,source,path.stem)['joint']
        probabilities=[]
        for seed in SEEDS:
            fitted=ce.load_model(gr.DEFAULT,gr.checkpoint(gr.DEFAULT,source,'slot_dropout',seed))
            p=native_probability(record,fitted,inputs)
            if view=='clean':
                with np.load(gr.DEFAULT/f'{source}_{path.stem}_clean_slot_dropout_{seed}.npz') as original:
                    if not np.allclose(p,original['probability'],atol=1e-6):
                        raise ValueError('Clean probability parity failed')
            probabilities.append(p)
        probabilities.append(average_probabilities(*probabilities))
        cov=gd.covariates(inputs)
        rows,predictions=[],[]
        for name,p in zip(('seed_2026','seed_2027','equal_probability_ensemble'),probabilities):
            scores,prediction=score(inputs,labels,p)
            if name.startswith('seed_'):
                seed=name.split('_')[1]
                oldrow=next(r for r in saved if (r['source'],r['match'],r['view'],r['replicate'],r['recipe']) ==
                    (source,path.stem,view,replicate,f'slot_dropout_{seed}'))
                for key in ('spell_f1','macro_f1','d_f1','d_precision','d_recall','n_gt_spells','n_pred_spells'):
                    if abs(scores[key]-oldrow[key]) > 1e-9:
                        raise ValueError(f'Archived score parity failed: {key}')
            row=dict(source=source,match=path.stem,view=view,replicate=replicate,recipe=name,**scores,
                diagnostics=gd.analyze(inputs,labels,prediction,cov))
            rows.append(row);predictions.append(prediction)
        if len({r['n_gt_spells'] for r in rows}) != 1:
            raise ValueError('Reference population changed')
        joblib.dump(dict(inputs=inputs,labels=labels,record=record,cov=cov),OUT/f'{path.stem}_{view}_{replicate}.joblib',compress=3)
        np.savez_compressed(OUT/f'{path.stem}_{view}_{replicate}.npz',**{
            name:p for name,p in zip(('seed_2026','seed_2027','equal_probability_ensemble'),probabilities)})
        write(dest,dict(rows=rows,complementarity=gd.complementarity(labels,predictions[:2]),
              declaration_sha256=sha256(DECLARATION), parity=True,seconds=time.monotonic()-start))
        print('EVALUATED',path.stem,view,replicate,round(time.monotonic()-start,1),
              {r['recipe']:round(r['spell_f1'],5) for r in rows},flush=True)


def summarize(rows):
    keys=('spell_f1','macro_f1','d_f1','d_precision','d_recall','weighted_70_30')
    out=[]
    for recipe in sorted({r['recipe'] for r in rows}):
        for view in ('clean','feed_like'):
            selected=[r for r in rows if r['recipe']==recipe and r['view']==view]
            providers={}
            for source in dt.SPLITS:
                matches={r['match'] for r in selected if r['source']==source}
                providers[source]={k:float(np.mean([np.mean([r[k] for r in selected if r['match']==m]) for m in matches])) for k in keys}
            out.append(dict(recipe=recipe,view=view,per_provider=providers,
                       **{k:float(np.mean([p[k] for p in providers.values()])) for k in keys}))
    return out


def balanced_frame_rate(rows,stratum,key):
    values=[]
    for source in dt.SPLITS:
        matches={r['match'] for r in rows if r['source']==source}
        within=[]
        for match in matches:
            rr=[r['diagnostics']['frames'][stratum] for r in rows if r['match']==match]
            rates=[gd.ratio(r[key],r['live_frames']) for r in rr]
            rates=[r for r in rates if r is not None]
            if rates: within.append(np.mean(rates))
        if within: values.append(np.mean(within))
    return float(np.mean(values)) if values else None


def report():
    spec=check()
    bundles=[json.loads((OUT/f'{p.stem}_{v}_{r}.json').read_text()) for _,p,v,r in cells()]
    if any(b['declaration_sha256'] != sha256(DECLARATION) for b in bundles):
        raise ValueError('Mixed declarations')
    rows=[r for b in bundles for r in b['rows']]
    assert len(rows)==45
    current=[r for r in rows if r['recipe']=='seed_2026' and r['view']=='feed_like']
    # Sum >=10s strata within a cell, then retain provider/match weighting.
    augmented=[]
    for row in current:
        row=json.loads(json.dumps(row))
        counts=[row['diagnostics']['frames'][k] for k in ('gap:10-20s','gap:20-40s','gap:40-infs')]
        row['diagnostics']['frames']['long_gap']={k:sum(c[k] for c in counts) for k in ('live_frames','wrong_team','false_dead')}
        augmented.append(row)
    wrong=balanced_frame_rate(augmented,'long_gap','wrong_team')
    false=balanced_frame_rate(augmented,'long_gap','false_dead')
    no_gap=balanced_frame_rate(augmented,'gap:no_gap','wrong_team')
    trigger=dict(long_gap_wrong_team_rate=wrong,long_gap_false_dead_rate=false,no_gap_wrong_team_rate=no_gap,
                 holds=wrong is not None and false is not None and no_gap is not None and wrong>false and wrong>no_gap)
    result=dict(spec=spec,rows=rows,summaries=summarize(rows),context_trigger=trigger,
        parity_cells=30,expected_parity_cells=30,complementarity=[dict(match=p.stem,view=v,replicate=r,**b['complementarity']) for (_,p,v,r),b in zip(cells(),bundles)],
        completed_cells=45,organizer_labels_read=False,baseline_changed=False)
    write(RESULT,result)
    print('REPORT',json.dumps(result['summaries']),flush=True)
    print('CONTEXT_TRIGGER',json.dumps(trigger),flush=True)


@contextmanager
def context(seconds):
    """Reuse the frozen library verbatim in a serial process; always restore it.

    Both chunking and train/predict crop offsets read lt.CONTEXT at call time.
    CORE and augmentation choices stay fixed. Never use this across threads.
    """
    previous=lt.CONTEXT
    try:
        lt.CONTEXT=int(seconds*lt.RATE)
        yield
    finally:
        lt.CONTEXT=previous


def context_check():
    spec=check()
    if not json.loads(RESULT.read_text())['context_trigger']['holds']:
        raise ValueError('Predeclared context condition does not hold')
    return spec


def prepare(source):
    spec=context_check()
    for i,path in enumerate(dt.SPLITS[source]['train']):
        dest=OUT/f'train_{path.stem}.joblib'
        if dest.exists():
            if sha256(dest) != json.loads(dest.with_suffix('.json').read_text())['sha256']:
                raise ValueError('Training cache changed')
            continue
        inputs,labels=ms.view(path)
        legacy=robust.masked_inputs(inputs,lt.SEED+i)
        gap=lc.ball_gap_view(inputs,spec['calibration'],ce.seed(path.stem,'train'))
        views=dict(clean=inputs,gap=gap,dropout_legacy=gr.slot_view(legacy,lt.SEED+i),
                   dropout_gap=gr.slot_view(gap,lt.SEED+i))
        records={name:dict(joint=old.feature_view(view,labels,source,path.stem)['joint']) for name,view in views.items()}
        joblib.dump(records,dest,compress=3)
        write(dest.with_suffix('.json'),dict(sha256=sha256(dest),input_sha256=sha256(path)))
        print('PREPARED',source,path.stem,flush=True)


def fit(source,seed):
    context_check()
    path=OUT/f'without_{source}_context15_{seed}.joblib'
    if path.exists():
        ce.load_model(OUT,path); return
    if (OUT/'context_checkpoints_frozen.json').exists():
        raise ValueError('Training closed')
    records=[]
    for provider in old.training_sources(source):
        for p in dt.SPLITS[provider]['train']:
            cache=OUT/f'train_{p.stem}.joblib'
            if sha256(cache) != json.loads(cache.with_suffix('.json').read_text())['sha256']:
                raise ValueError('Training cache changed')
            records.extend(gr.paired_records(joblib.load(cache),p.stem,'slot_dropout'))
    start=time.monotonic()
    with context(15):
        fitted=model.fit(records,player_only=True,seed=seed)
    baseline=ce.load_model(gr.DEFAULT,gr.checkpoint(gr.DEFAULT,source,'slot_dropout',seed))
    for key in ('median','scale'):
        if not np.array_equal(fitted[key],baseline[key]):
            raise ValueError(f'Matched normalization differs: {key}')
    for key in ('parameters','optimizer_updates','windows_per_epoch','feature_names'):
        if fitted[key] != baseline[key]:
            raise ValueError(f'Matched budget differs: {key}')
    fitted['context_seconds']=15
    ce.freeze_model(OUT,path,fitted,dict(source=source,seed=seed,context_seconds=15,
        matched_normalization=True,parameters=fitted['parameters'],optimizer_updates=fitted['optimizer_updates'],
        windows_per_epoch=fitted['windows_per_epoch'],seconds=time.monotonic()-start))
    print('FROZEN_CONTEXT',source,seed,round(time.monotonic()-start,1),flush=True)


def seal():
    context_check()
    paths=[OUT/f'without_{s}_context15_{seed}.joblib' for s in dt.SPLITS for seed in SEEDS]
    for path in paths: ce.load_model(OUT,path)
    receipt={p.name:sha256(p) for p in paths}
    dest=OUT/'context_checkpoints_frozen.json'
    if dest.exists():
        if json.loads(dest.read_text()) != receipt: raise ValueError('Sealed model changed')
    else: write(dest,receipt)


def evaluate_context(match=None):
    seal()
    for source,path,view,replicate in cells():
        if match is not None and path.stem != match: continue
        dest=OUT/f'{path.stem}_{view}_{replicate}_context.json'
        if dest.exists(): continue
        cached=joblib.load(OUT/f'{path.stem}_{view}_{replicate}.joblib')
        inputs,labels=cached['inputs'],cached['labels']
        rows=[]
        for seed in SEEDS:
            fitted=ce.load_model(OUT,OUT/f'without_{source}_context15_{seed}.joblib')
            with context(15): p=native_probability(cached['record'],fitted,inputs)
            scores,prediction=score(inputs,labels,p)
            rows.append(dict(source=source,match=path.stem,view=view,replicate=replicate,
                recipe=f'context15_{seed}',**scores,
                diagnostics=gd.analyze(inputs,labels,prediction,cached['cov'])))
            np.savez_compressed(OUT/f'{path.stem}_{view}_{replicate}_context15_{seed}.npz',probability=p,prediction=prediction)
        write(dest,dict(rows=rows,declaration_sha256=sha256(DECLARATION)))
        print('CONTEXT_SCORED',path.stem,view,replicate,{r['recipe']:r['spell_f1'] for r in rows},flush=True)


def context_report():
    seal()
    rows=[r for _,p,v,k in cells() for r in json.loads((OUT/f'{p.stem}_{v}_{k}_context.json').read_text())['rows']]
    if len(rows)!=30: raise ValueError('Incomplete context cohort')
    base=json.loads(RESULT.read_text())
    for row in rows:
        other=next(r for r in base['rows'] if (r['match'],r['view'],r['replicate'],r['recipe']) ==
                   (row['match'],row['view'],row['replicate'],row['recipe'].replace('context15_','seed_')))
        if row['n_gt_spells'] != other['n_gt_spells']: raise ValueError('Grading changed')
    summaries=summarize(rows)
    guards=[]
    for seed in SEEDS:
        c=next(r for r in summaries if r['recipe']==f'context15_{seed}' and r['view']=='clean')
        f=next(r for r in summaries if r['recipe']==f'context15_{seed}' and r['view']=='feed_like')
        bc=next(r for r in base['summaries'] if r['recipe']==f'seed_{seed}' and r['view']=='clean')
        bf=next(r for r in base['summaries'] if r['recipe']==f'seed_{seed}' and r['view']=='feed_like')
        delta=c['spell_f1']-bc['spell_f1']; fd=f['spell_f1']-bf['spell_f1']
        worst=min(c['per_provider'][s][k]-bc['per_provider'][s][k] for s in dt.SPLITS for k in ('d_f1','macro_f1'))
        guards.append(dict(seed=seed,clean_spell_delta=delta,feed_like_spell_delta=fd,
                           worst_provider_d_or_macro_delta=worst,passes=delta>=-.01 and fd>=-.01 and worst>=-.02))
    write(ROOT/'experiments/results/seed_gap_context_long.json',dict(rows=rows,summaries=summaries,guardrails=guards,
        completed_cells=30,checkpoints=json.loads((OUT/'context_checkpoints_frozen.json').read_text()),
        declaration_sha256=sha256(DECLARATION),baseline_changed=False,organizer_labels_read=False))
    print('CONTEXT_REPORT',json.dumps(summaries),json.dumps(guards),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','evaluate','report','prepare','fit','seal','evaluate-context','context-report'))
    parser.add_argument('--source',choices=tuple(dt.SPLITS))
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--match')
    args=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if args.action=='freeze': freeze()
        elif args.action=='evaluate': evaluate(args.match)
        elif args.action=='report': report()
        elif args.action=='prepare':
            for s in ([args.source] if args.source else dt.SPLITS): prepare(s)
        elif args.action=='fit':
            if args.source is None or args.seed is None: parser.error('fit needs source/seed')
            fit(args.source,args.seed)
        elif args.action=='seal': seal()
        elif args.action=='evaluate-context': evaluate_context(args.match)
        else: context_report()


if __name__=='__main__': main()
