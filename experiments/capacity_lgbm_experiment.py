"""User-requested deeper LightGBM extension to the frozen temporal study.

The original declaration and four configurations are immutable. This study
changes capacity and shrinkage together, using the same engineered features,
rows, augmentation, held-out providers, GRU controls, and decoder.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import joblib
import numpy as np
from experiments import temporal_lgbm_experiment as base, temporal_lgbm as model
from experiments import corruption_experiment as ce, supervised, spell_diagnostics
from experiments.eval_set.protocol import sha256

ROOT=base.ROOT
DEFAULT=ROOT/'data/experiments/capacity_lgbm_v1'
CONFIGS={
    'context_63_slow': dict(learning_rate=.01,n_estimators=1280,num_leaves=63,max_depth=8,
        min_child_samples=120,min_child_weight=.01,reg_alpha=.2,reg_lambda=10.,
        colsample_bytree=.7,subsample=.8,subsample_freq=1,max_bin=127),
    'context_127_slow': dict(learning_rate=.01,n_estimators=1280,num_leaves=127,max_depth=10,
        min_child_samples=120,min_child_weight=.01,reg_alpha=.2,reg_lambda=10.,
        colsample_bytree=.7,subsample=.8,subsample_freq=1,max_bin=127),
}


def fingerprints():
    return dict(base.fingerprints(),**{'experiments/capacity_lgbm_experiment.py':sha256(__file__)})


def freeze(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'predeclared.json').exists():return check(out)
    previous=base.check(base.DEFAULT)
    caches={}
    for provider,split in base.dt.SPLITS.items():
        for path in split['train']:
            cache=base.DEFAULT/f'train_{path.stem}.joblib'
            base.validate_cache(base.DEFAULT,cache,previous)
            caches[str(cache.relative_to(ROOT))]=sha256(cache)
    spec=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=fingerprints(),
        original_spec_sha256=sha256(base.DEFAULT/'predeclared.json'),
        data_sha256=previous['data_sha256'],training_caches_sha256=caches,
        baseline_manifest_sha256=previous['baseline_manifest_sha256'],policy_sha256=previous['policy_sha256'],
        configs=CONFIGS,tree_seed=2026,gru_seeds=base.SEEDS,blend_lgbm_weights=base.WEIGHTS,
        views=base.VIEWS,training_matches=previous['training_matches'],evaluation_matches=previous['evaluation_matches'],
        request='User requested more num_leaves and max_depth with smaller learning rate while original study was fitting; original scores had not been opened.',
        hypothesis='More leaf/depth capacity with half the learning rate and twice the trees improves the rich625-feature D companion. Two combined capacity/shrinkage hypotheses, not isolated causal tests of each parameter.',
        comparator='Original context_regularized:31leaves,depth7,rate.02,640trees. All remaining parameters and625features are identical. New63/depth8 and127/depth10 use rate.01/1280trees; learning_rate*n_estimators stays12.8.',
        unchanged='Same2.5Hz rows built from full5Hz context, source/match/view weights, clean+slot dropout, seed2026, all full-provider exclusions, GRUconditionalH/A and fixed Dreplacement/equalblend, Viterbi cost3.',
        budget='6newfits (2configs x3providerfolds),120new match/view/config/GRUseed/blend cells. No early stopping, adaptive arms, threshold/decoder search or public scoring. All6checkpoints sealed before extension scoring.',
        primary=previous['primary'],guardrail=previous['guardrail'],diagnostics=previous['diagnostics'],evidence=previous['evidence'],
        original_study_preserved=True,organizer_labels_read=False,automatic_promotion=False)
    ce.write_json(out/'predeclared.json',spec,exclusive=True)
    return spec


def check(out):
    spec=json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256']!=fingerprints() or spec['original_spec_sha256']!=sha256(base.DEFAULT/'predeclared.json'):
        raise ValueError('Frozen extension or original declaration changed')
    base.check(base.DEFAULT)
    return spec


def checkpoint(out,source,config):return out/f'without_{source}_{config}.joblib'


def fit(out,source,config):
    from lightgbm import LGBMClassifier
    spec=check(out);target=checkpoint(out,source,config)
    if target.exists():ce.load_model(out,target);return
    if (out/'checkpoints_frozen.json').exists():raise ValueError('Fitting closed')
    records=[]
    for provider,split in base.dt.SPLITS.items():
        if provider==source:continue
        for p in split['train']:
            cache=base.DEFAULT/f'train_{p.stem}.joblib'
            if sha256(cache)!=spec['training_caches_sha256'][str(cache.relative_to(ROOT))]:raise ValueError('Training cache changed')
            records.extend(joblib.load(cache))
    started=time.monotonic()
    x,y,weight,names=model.matrix(records,True)
    estimator=LGBMClassifier(objective='binary',random_state=2026,n_jobs=1,
        deterministic=True,force_col_wise=True,verbosity=-1,**CONFIGS[config])
    estimator.fit(x,y,sample_weight=weight)
    fitted=dict(kind='temporal_lgbm',config=config,seed=2026,estimator=estimator,
        engineered=True,feature_names=names,training_rows=len(y),params=CONFIGS[config])
    importance=estimator.booster_.feature_importance(importance_type='gain')
    grouped={}
    for name,value in zip(names,importance):
        family='original' if '__' not in name else name.split('__')[1].rsplit('_',1)[0]
        bucket=grouped.setdefault(family,dict(total_gain=0.,features=0))
        bucket['total_gain']+=float(value);bucket['features']+=1
    ce.freeze_model(out,target,fitted,dict(heldout=source,training_matches=spec['training_matches'][source],
        config=config,seconds=time.monotonic()-started,training_rows=len(y),n_features=len(names),
        training_gain_by_family=grouped,top_training_gain_features=sorted([dict(feature=n,gain=float(v)) for n,v in zip(names,importance)],key=lambda r:-r['gain'])[:30]))
    print('CAPACITY FROZEN',source,config,round(time.monotonic()-started,1),flush=True)


def seal(out):
    check(out);rows=[]
    for s in base.dt.SPLITS:
        for c in CONFIGS:
            p=checkpoint(out,s,c);ce.load_model(out,p)
            rows.append(dict(file=p.name,sha256=sha256(p),receipt_sha256=sha256(p.with_suffix('.json'))))
    target=out/'checkpoints_frozen.json'
    if target.exists():
        if json.loads(target.read_text())['checkpoints']!=rows:raise ValueError('Sealed extension models changed')
    else:ce.write_json(target,dict(checkpoints=rows),exclusive=True)


def seal_dependencies(out):
    check(out);base.seal(base.DEFAULT);base.seal_gru(base.DEFAULT);base.seal_evaluation_inputs(base.DEFAULT)
    paths=[base.DEFAULT/'predeclared.json',base.DEFAULT/'checkpoints_frozen.json',
        base.DEFAULT/'gru_dependencies.json',base.DEFAULT/'evaluation_inputs_frozen.json']
    paths.extend(base.checkpoint(base.DEFAULT,s,'context_regularized') for s in base.dt.SPLITS)
    payload={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    target=out/'dependencies_frozen.json'
    if target.exists():
        if json.loads(target.read_text())!=payload:raise ValueError('Original/GRU dependencies changed')
    else:ce.write_json(target,payload,exclusive=True)


def evaluate(out,source):
    check(out);seal(out);seal_dependencies(out)
    destination=out/f'{source}_results.json'
    rows=json.loads(destination.read_text()) if destination.exists() else []
    fitted={c:ce.load_model(out,checkpoint(out,source,c)) for c in CONFIGS}
    for path in base.dt.SPLITS[source]['eval']:
        for view in base.VIEWS:
            cache=joblib.load(base.GRU/f'eval_{path.stem}_{view}.joblib')
            joint,inputs,labels=cache['joint'],cache['inputs'],cache['labels']
            cols=[i for i,n in enumerate(joint['names']) if n.startswith('D_')]
            record=dict(joint,x=joint['x'][:,cols],names=[joint['names'][i][2:] for i in cols])
            periods=supervised.model_periods(inputs)
            for config,head in fitted.items():
                dead=base.lt.upsample(model.predict(record,head),record,periods)
                for seed in base.SEEDS:
                    with np.load(base.GRU/f'{source}_{path.stem}_{view}_control_{seed}.npz') as saved:gru=saved['probability']
                    for weight in base.WEIGHTS:
                        key=(path.stem,view,config,seed,weight)
                        if any((r['match'],r['view'],r['config'],r['seed'],r['weight'])==key for r in rows):continue
                        probability=model.fuse(gru,dead,weight)
                        pred=supervised.decode(probability,periods,float(inputs['fps']),base.lt.COST)
                        result=supervised.score(dict(inputs,labels=labels),pred)
                        row=dict(source=source,match=path.stem,view=view,config=config,seed=seed,weight=weight,
                            spell_f1=result['spell_f1_mean'],macro_f1=result['macro_f1'],weighted_summary=.7*result['spell_f1_mean']+.3*result['macro_f1'],
                            d_f1=result['per_class']['D']['f1'],d_precision=result['per_class']['D']['precision'],d_recall=result['per_class']['D']['recall'],
                            n_gt_spells=result['n_gt_spells'],n_pred_spells=result['n_pred_spells'],graded_frames=int(np.isin(labels,supervised.LABELS).sum()),
                            diagnostics=spell_diagnostics.spell_diagnostics(labels,pred,period=inputs['period'],fps=float(inputs['fps']),timestamps=inputs.get('timestamps')))
                        rows.append(row);ce.write_json(destination,rows)
                        np.savez_compressed(out/f'{source}_{path.stem}_{view}_{config}_{seed}_{weight}.npz',prediction=pred)
                        print('CAPACITY SCORE',source,path.stem,view,config,seed,weight,round(row['spell_f1'],5),flush=True)


def report(out):
    check(out);seal(out);seal_dependencies(out)
    rows=[r for s in base.dt.SPLITS for r in json.loads((out/f'{s}_results.json').read_text())]
    expected={(s,p.stem,v,c,seed,w) for s in base.dt.SPLITS for p in base.dt.SPLITS[s]['eval'] for v in base.VIEWS for c in CONFIGS for seed in base.SEEDS for w in base.WEIGHTS}
    if len(rows)!=len(expected) or {(r['source'],r['match'],r['view'],r['config'],r['seed'],r['weight']) for r in rows}!=expected:raise ValueError('Incomplete or duplicate extension cells')
    original=json.loads((base.DEFAULT/'report.json').read_text())
    controls=original['controls'];reference=[r for r in original['rows'] if r['config']=='context_regularized']
    for r in rows:
        c=next(c for c in controls if (c['source'],c['match'],c['view'],c['seed'])==(r['source'],r['match'],r['view'],r['seed']))
        if (r['n_gt_spells'],r['graded_frames'])!=(c['n_gt_spells'],c['graded_frames']):raise ValueError('Grading changed')
    summaries=base.summarize(rows+reference+controls);screens=[];deltas=[]
    for config in CONFIGS:
        for weight in base.WEIGHTS:
            for seed in base.SEEDS:
                get=lambda c,w,v:next(r for r in summaries if (r['config'],r['weight'],r['seed'],r['view'])==(c,w,seed,v))
                clean,stress=get(config,weight,'clean'),get(config,weight,'stress')
                control,control_stress=get('gru_control',0.,'clean'),get('gru_control',0.,'stress')
                cd=clean['spell_f1']-control['spell_f1'];sd=stress['spell_f1']-control_stress['spell_f1']
                pd=min(clean['per_source'][s][k]-control['per_source'][s][k] for s in base.dt.SPLITS for k in ('macro_f1','d_f1'))
                screens.append(dict(config=config,weight=weight,seed=seed,clean_spell_delta=cd,stress_spell_delta=sd,worst_provider_d_or_macro_delta=pd,passes=cd>=-.01 and sd>=.02 and pd>=-.02))
                for view in ('clean','stress'):
                    a,b=get(config,weight,view),get('context_regularized',weight,view)
                    deltas.append(dict(config=config,weight=weight,seed=seed,view=view,spell_delta=a['spell_f1']-b['spell_f1'],macro_delta=a['macro_f1']-b['macro_f1']))
    ce.write_json(out/'report.json',dict(spec=check(out),rows=rows,controls=controls,original_regularized_rows=reference,
        summaries=summaries,screens=screens,deltas_vs_original_regularized=deltas,
        model_receipts={p.name:json.loads(p.read_text()) for p in out.glob('without_*.json')},
        original_report_sha256=sha256(base.DEFAULT/'report.json'),dependencies=json.loads((out/'dependencies_frozen.json').read_text()),baseline_changed=False),exclusive=True)
    for r in summaries:print('SUMMARY',r['config'],r['weight'],r['seed'],r['view'],round(r['spell_f1'],5),round(r['macro_f1'],5),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','fit','seal','evaluate','report'))
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--source',choices=tuple(base.dt.SPLITS));parser.add_argument('--config',choices=tuple(CONFIGS))
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.out)
    elif args.action=='seal':seal(args.out)
    elif args.action=='report':report(args.out)
    elif args.action=='fit':
        if args.source is None or args.config is None:parser.error('fit requires source and config')
        fit(args.out,args.source,args.config)
    else:
        for s in ([args.source] if args.source else base.dt.SPLITS):evaluate(args.out,s)


if __name__=='__main__':main()
