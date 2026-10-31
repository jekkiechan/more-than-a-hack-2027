"""Frozen feature and capacity study for a player-only LightGBM D companion.

Full provider holdouts, fixed GRU conditional team probabilities and cost 3.
No organizer path, optimizer selected checkpoint, or baseline promotion exists.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
from experiments import (temporal_lgbm as model, corruption_experiment as ce,
    dead_transfer as dt, learned_temporal as lt, multisource_eval as ms,
    refine_domain, supervised, factorized_had, spell_diagnostics,
    gru_robustness_experiment as gr, tracking_corruption as tc)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
DEFAULT = ROOT/'data/experiments/temporal_lgbm_v1'
GRU = ROOT/'data/experiments/gru_slot_robustness_v2'
VIEWS = ('clean', 'corrupt_1', 'corrupt_2')
SEEDS = (2026, 2027)  # Fixed GRU controls; LightGBM itself uses seed 2026.
WEIGHTS = (0.5, 1.0)


def fingerprints():
    paths = ['experiments/temporal_lgbm.py', 'experiments/temporal_lgbm_experiment.py',
             'experiments/factorized_had.py', 'experiments/spell_diagnostics.py',
             'experiments/gru_robustness_experiment.py',
             'experiments/metrics.py', 'evaluation/possession_metrics.py',
             'experiments/refine_domain.py', 'experiments/multisource_eval.py']
    return dict(ce.sources_hash(), **{p: sha256(ROOT/p) for p in paths})


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'predeclared.json').exists():
        return check(out)
    manifest = json.loads((ROOT/'experiments/eval_set/manifest.json').read_text())
    data = {}
    for source in manifest['sources'].values():
        for role in ('train', 'selection'):
            for item in source[role]:
                if sha256(ROOT/item['path']) != item['sha256']:
                    raise ValueError('External input differs from manifest')
                data[item['path']] = item['sha256']
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=fingerprints(),
        data_sha256=data, baseline_manifest_sha256=sha256(ROOT/'experiments/baseline_manifest.json'),
        policy_sha256=sha256(ROOT/'experiments/model_decision_policy.json'),
        configs=model.CONFIGS, tree_seed=2026, gru_seeds=SEEDS, blend_lgbm_weights=WEIGHTS,
        views=VIEWS, decoder_switch_cost=3, sample_hz=5, training_stride=2,
        training_matches={s:[p.stem for provider, split in dt.SPLITS.items() if provider!=s for p in split['train']] for s in dt.SPLITS},
        evaluation_matches={s:[p.stem for p in split['eval']] for s,split in dt.SPLITS.items()},
        training='Equal source/match/view weights; clean plus target-independent player-slot dropout, seed2026+i per provider; unknown targets excluded. Same 2.5Hz rows for all configurations, temporal features computed first on full5Hz blocks.',
        hypothesis='Separate before/after motion and coverage improve D probability decisions and complement the latest joint GRU. Original67 vs context_slow isolates feature expansion; shallow and regularized test capacity/regularization with those same features.',
        features='Original67 plus primitive finite masks; +/-1,3,5s values and differences; separate past/future means, standard deviations, coverage and mean contrasts, excluding target. No temporal feature crosses match/half/time discontinuity.',
        hyperparameters='Four fixed configurations, no early stopping or adaptive sweep. min_child_samples/weight, leaves/depth, L1/L2, row/column subsampling and binning constrain correlated-context overfit. subsample_freq=1 makes bagging effective.',
        ensemble='D-only probability replacement or equal blend with each fixed player-onlyD joint GRU control seed; GRU conditional H/A remains unchanged. A mean of seed scores is not an inference seed ensemble.',
        controls='Shared full H/A+D provider holdout GRU controls from gru_slot_robustness_v2. Their checkpoints and receipts must be sealed before scoring. No models trained on excluded-provider matches or normalization.',
        budget='12 LightGBM fits, 4 configurations x3provider folds; 240 new paired match/view/config/GRUseed/blend rows plus30 reused GRU control rows. One CPU thread per fit; no extra arms.',
        primary='Unrounded equal-provider clean official spell F1, macro exact-tie. Two stress views averaged within match; report both GRU seeds, then mean. Fixed auxiliary .7spell+.3macro not used for ranking.',
        guardrail='For each GRU seed, clean spell no more than .01 below matched GRU control; stress spell improves by at least .02 over that GRU; worst-provider clean D-F1 and macro loss <=.02. Separately show adopted-recipe full-provider baseline comparison; no automatic adoption.',
        diagnostics='Exact per-IoU TP/FP/FN; short-spell and graded-edge recall, split/merge geometry and complete-boundary errors; D precision/recall, spell counts. LightGBM training gain grouped by feature family is descriptive, not causal validation importance.',
        evidence='Five previously exposed development matches with native grading; SkillCorner X margins and >=12visibility filter retained. Synthetic stress and silver labels are not independent broadcast confirmation. Tree seed variance not measured.',
        organizer_labels_read=False, automatic_promotion=False)
    ce.write_json(out/'predeclared.json', spec, exclusive=True)
    return spec


def check(out):
    spec=json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256'] != fingerprints():
        raise ValueError('Frozen source changed; do not resume changed experiment')
    return spec


def prepare(out, source):
    spec=check(out)
    for i,path in enumerate(dt.SPLITS[source]['train']):
        target=out/f'train_{path.stem}.joblib'
        if target.exists():
            validate_cache(out,target,spec);continue
        if sha256(path)!=spec['data_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Training input changed')
        started=time.monotonic()
        inputs,labels=ms.view(path)
        records=[ce.record(inputs,labels,source,path.stem,'clean'),
                 ce.record(refine_domain.drop_player_slots(inputs,2026+i),labels,source,path.stem,'dropout')]
        joblib.dump(records,target)
        ce.write_json(target.with_suffix('.json'),dict(sha256=sha256(target),
            predeclared_sha256=sha256(out/'predeclared.json'),input_sha256=sha256(path),
            match=path.stem,source=source),exclusive=True)
        print('D CACHE',path.stem,round(time.monotonic()-started,1),flush=True)


def validate_cache(out,path,spec):
    receipt=json.loads(path.with_suffix('.json').read_text())
    if receipt['sha256']!=sha256(path) or receipt['predeclared_sha256']!=sha256(out/'predeclared.json'):
        raise ValueError('Feature cache changed')
    if receipt['input_sha256'] not in spec['data_sha256'].values():
        raise ValueError('Unrecognized feature-cache input')


def checkpoint(out,source,config):
    return out/f'without_{source}_{config}.joblib'


def fit(out,source,config):
    spec=check(out);target=checkpoint(out,source,config)
    if target.exists():
        ce.load_model(out,target);return
    if (out/'checkpoints_frozen.json').exists():
        raise ValueError('Fitting closed')
    records=[]
    for provider,split in dt.SPLITS.items():
        if provider==source:continue
        for p in split['train']:
            cache=out/f'train_{p.stem}.joblib';validate_cache(out,cache,spec)
            records.extend(joblib.load(cache))
    started=time.monotonic();fitted=model.fit(records,config)
    importance=fitted['estimator'].booster_.feature_importance(importance_type='gain')
    grouped={}
    for name,value in zip(fitted['feature_names'],importance):
        family='original' if '__' not in name else name.split('__')[1].rsplit('_',1)[0]
        bucket=grouped.setdefault(family,dict(total_gain=0.,features=0))
        bucket['total_gain']+=float(value);bucket['features']+=1
    ce.freeze_model(out,target,fitted,dict(heldout=source,training_matches=spec['training_matches'][source],
        config=config,seconds=time.monotonic()-started,training_rows=fitted['training_rows'],
        n_features=len(fitted['feature_names']),training_gain_by_family=grouped,
        top_training_gain_features=sorted([dict(feature=n,gain=float(v)) for n,v in zip(fitted['feature_names'],importance)],key=lambda x:-x['gain'])[:30]))
    print('LGBM FROZEN',source,config,round(time.monotonic()-started,1),flush=True)


def seal(out):
    check(out);rows=[]
    for s in dt.SPLITS:
        for c in model.CONFIGS:
            p=checkpoint(out,s,c);ce.load_model(out,p)
            rows.append(dict(file=p.name,sha256=sha256(p),receipt_sha256=sha256(p.with_suffix('.json'))))
    target=out/'checkpoints_frozen.json'
    if target.exists():
        if json.loads(target.read_text())['checkpoints']!=rows:raise ValueError('Sealed models changed')
    else:ce.write_json(target,dict(checkpoints=rows),exclusive=True)


def seal_gru(out):
    """Bind the independently frozen controls before any LGBM score is opened."""
    spec=check(out);gr.check(GRU);rows=[]
    for source in dt.SPLITS:
        for seed in SEEDS:
            p=GRU/f'without_{source}_control_{seed}.joblib'
            fitted=ce.load_model(GRU,p)
            receipt=json.loads(p.with_suffix('.json').read_text())
            if set(receipt['training_matches'])!=set(spec['training_matches'][source]) or not fitted['player_only']:
                raise ValueError('GRU control provider/information-path mismatch')
            rows.append(dict(path=str(p.relative_to(ROOT)),sha256=sha256(p),receipt_sha256=sha256(p.with_suffix('.json'))))
    payload=dict(controls=rows,gru_spec_sha256=sha256(GRU/'predeclared.json'))
    target=out/'gru_dependencies.json'
    if target.exists():
        if json.loads(target.read_text())!=payload:raise ValueError('GRU controls changed')
    else:ce.write_json(target,payload,exclusive=True)


def seal_evaluation_inputs(out):
    """Seal every shared view and control probability before opening new scores."""
    spec=check(out);gru_spec=gr.check(GRU);seal_gru(out)
    target=out/'evaluation_inputs_frozen.json'
    paths=[]
    for source,split in dt.SPLITS.items():
        for path in split['eval']:
            for view in VIEWS:
                paths.append(GRU/f'eval_{path.stem}_{view}.joblib')
                paths.extend(GRU/f'{source}_{path.stem}_{view}_control_{seed}.npz' for seed in SEEDS)
    current={str(p.relative_to(ROOT)):sha256(p) for p in paths}
    if target.exists():
        if json.loads(target.read_text())['files_sha256']!=current:
            raise ValueError('Sealed evaluation inputs changed')
        return
    for source,split in dt.SPLITS.items():
        for path in split['eval']:
            if sha256(path)!=spec['data_sha256'][str(path.relative_to(ROOT))]:
                raise ValueError('Evaluation input changed')
            original,labels=ms.view(path)
            for view in VIEWS:
                cache=joblib.load(GRU/f'eval_{path.stem}_{view}.joblib')
                expected=original if view=='clean' else tc.corrupt(original,gru_spec['calibration'],ce.seed(path.stem,view))
                expected=tc.inputs_only(expected)
                if set(cache['inputs'])!=set(expected):raise ValueError('Raw view schema changed')
                for key,value in expected.items():
                    if not np.array_equal(cache['inputs'][key],value,equal_nan=True):
                        raise ValueError(f'Raw view mismatch: {path.stem}/{view}/{key}')
                record=cache['joint']
                if (not np.array_equal(labels,cache['labels']) or record['match']!=path.stem
                    or record['source']!=source or not np.array_equal(record['indices'],lt.sampled_indices(expected))
                    or not np.array_equal(record['period'],supervised.model_periods(expected)[record['indices']])):
                    raise ValueError('Shared evaluation cohort/clock mismatch')
                for seed in SEEDS:
                    with np.load(GRU/f'{source}_{path.stem}_{view}_control_{seed}.npz') as saved:
                        prob=saved['probability']
                        if prob.shape!=(len(labels),3) or not np.isfinite(prob).all() or np.any(prob<0) or not np.allclose(prob.sum(1),1):
                            raise ValueError('Invalid GRU probabilities')
                        prediction=supervised.decode(prob,supervised.model_periods(expected),float(expected['fps']),lt.COST)
                        if not np.array_equal(prediction,saved['prediction']):
                            raise ValueError('Control prediction/probability mismatch')
    ce.write_json(target,dict(files_sha256=current,gru_dependencies_sha256=sha256(out/'gru_dependencies.json'),
        views_verified=15,controls_verified=30),exclusive=True)


def evaluate(out,source):
    spec=check(out);seal(out);seal_gru(out);seal_evaluation_inputs(out)
    destination=out/f'{source}_results.json'
    rows=json.loads(destination.read_text()) if destination.exists() else []
    models={c:ce.load_model(out,checkpoint(out,source,c)) for c in model.CONFIGS}
    for path in dt.SPLITS[source]['eval']:
        if sha256(path)!=spec['data_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Evaluation input changed')
        _, expected_labels=ms.view(path)
        for view in VIEWS:
            cache=joblib.load(GRU/f'eval_{path.stem}_{view}.joblib')
            joint,inputs,labels=cache['joint'],cache['inputs'],cache['labels']
            if not np.array_equal(labels,expected_labels) or joint['match']!=path.stem or joint['source']!=source:
                raise ValueError('Evaluation cache cohort or grading mismatch')
            cols=[i for i,n in enumerate(joint['names']) if n.startswith('D_')]
            record=dict(joint,x=joint['x'][:,cols],names=[joint['names'][i][2:] for i in cols])
            periods=supervised.model_periods(inputs)
            for config,fitted in models.items():
                sampled=model.predict(record,fitted)
                dead=lt.upsample(sampled,record,periods)
                for seed in SEEDS:
                    archived=np.load(GRU/f'{source}_{path.stem}_{view}_control_{seed}.npz')
                    gru=archived['probability']
                    for weight in WEIGHTS:
                        key=(path.stem,view,config,seed,weight)
                        if any((r['match'],r['view'],r['config'],r['seed'],r['weight'])==key for r in rows):continue
                        probability=model.fuse(gru,dead,weight)
                        pred=supervised.decode(probability,periods,float(inputs['fps']),lt.COST)
                        result=supervised.score(dict(inputs,labels=labels),pred)
                        diag=spell_diagnostics.spell_diagnostics(labels,pred,period=inputs['period'],fps=float(inputs['fps']),timestamps=inputs.get('timestamps'))
                        row=dict(source=source,match=path.stem,view=view,config=config,seed=seed,weight=weight,
                            spell_f1=result['spell_f1_mean'],macro_f1=result['macro_f1'],
                            weighted_summary=.7*result['spell_f1_mean']+.3*result['macro_f1'],
                            d_f1=result['per_class']['D']['f1'],d_precision=result['per_class']['D']['precision'],d_recall=result['per_class']['D']['recall'],
                            n_gt_spells=result['n_gt_spells'],n_pred_spells=result['n_pred_spells'],graded_frames=int(np.isin(labels,supervised.LABELS).sum()),diagnostics=diag)
                        rows.append(row);ce.write_json(destination,rows)
                        np.savez_compressed(out/f'{source}_{path.stem}_{view}_{config}_{seed}_{weight}.npz',prediction=pred)
                        print('LGBM SCORE',source,path.stem,view,config,seed,weight,round(row['spell_f1'],5),flush=True)


def summarize(rows):
    keys=('spell_f1','macro_f1','weighted_summary','d_f1','d_precision','d_recall')
    result=[]
    for config,weight,seed in sorted({(r['config'],r['weight'],r['seed']) for r in rows}):
        for view in ('clean','stress'):
            per={}
            for source in dt.SPLITS:
                selected=[r for r in rows if (r['config'],r['weight'],r['seed'],r['source'])==(config,weight,seed,source) and (r['view']=='clean')==(view=='clean')]
                if not selected:raise ValueError('Missing provider summary')
                per[source]={k:float(np.mean([r[k] for r in selected])) for k in keys}
            result.append(dict(config=config,weight=weight,seed=seed,view=view,per_source=per,**{k:float(np.mean([v[k] for v in per.values()])) for k in keys}))
    return result


def report(out):
    check(out);seal(out);seal_gru(out)
    rows=[r for s in dt.SPLITS for r in json.loads((out/f'{s}_results.json').read_text())]
    expected={(s,p.stem,v,c,seed,w) for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] for v in VIEWS for c in model.CONFIGS for seed in SEEDS for w in WEIGHTS}
    actual={(r['source'],r['match'],r['view'],r['config'],r['seed'],r['weight']) for r in rows}
    if actual!=expected or len(rows)!=len(expected):raise ValueError('Incomplete or duplicate experiment cells')
    controls=[]
    for source in dt.SPLITS:
        for r in json.loads((GRU/f'{source}_results.json').read_text()):
            if r['arm']!='control':continue
            controls.append(dict(r,config='gru_control',weight=0.,weighted_summary=.7*r['spell_f1']+.3*r['macro_f1']))
    expected_controls={(s,p.stem,v,seed) for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] for v in VIEWS for seed in SEEDS}
    if len(controls)!=len(expected_controls) or {(r['source'],r['match'],r['view'],r['seed']) for r in controls}!=expected_controls:
        raise ValueError('Incomplete control cohort')
    for r in rows:
        control=next(c for c in controls if (c['source'],c['match'],c['view'],c['seed'])==(r['source'],r['match'],r['view'],r['seed']))
        if r['n_gt_spells']!=control['n_gt_spells'] or r['graded_frames']!=control['graded_frames']:raise ValueError('Grading changed')
    summaries=summarize(rows+controls);screens=[]
    for c in model.CONFIGS:
        for w in WEIGHTS:
            for seed in SEEDS:
                get=lambda config,weight,view: next(r for r in summaries if (r['config'],r['weight'],r['seed'],r['view'])==(config,weight,seed,view))
                clean,stress=get(c,w,'clean'),get(c,w,'stress')
                baseline,baseline_stress=get('gru_control',0.,'clean'),get('gru_control',0.,'stress')
                clean_delta=clean['spell_f1']-baseline['spell_f1'];stress_delta=stress['spell_f1']-baseline_stress['spell_f1']
                provider_delta=min(clean['per_source'][s][k]-baseline['per_source'][s][k] for s in dt.SPLITS for k in ('macro_f1','d_f1'))
                screens.append(dict(config=c,weight=w,seed=seed,clean_spell_delta=clean_delta,stress_spell_delta=stress_delta,worst_provider_d_or_macro_delta=provider_delta,passes=clean_delta>=-.01 and stress_delta>=.02 and provider_delta>=-.02))
    result=dict(spec=check(out),rows=rows,controls=controls,summaries=summaries,screens=screens,
        model_receipts={p.name:json.loads(p.read_text()) for p in out.glob('without_*.json')},
        gru_dependencies=json.loads((out/'gru_dependencies.json').read_text()),
        evaluation_inputs=json.loads((out/'evaluation_inputs_frozen.json').read_text()),baseline_changed=False)
    ce.write_json(out/'report.json',result,exclusive=True)
    for r in summaries:print('SUMMARY',r['config'],r['weight'],r['seed'],r['view'],round(r['spell_f1'],5),round(r['macro_f1'],5),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','prepare','fit','seal','evaluate','report'))
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--source',choices=tuple(dt.SPLITS));parser.add_argument('--config',choices=tuple(model.CONFIGS))
    args=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if args.action=='freeze':freeze(args.out)
        elif args.action=='seal':seal(args.out)
        elif args.action=='report':report(args.out)
        elif args.action=='fit':
            if args.source is None or args.config is None:parser.error('fit requires source and config')
            fit(args.out,args.source,args.config)
        else:
            for source in ([args.source] if args.source else dt.SPLITS):
                (prepare if args.action=='prepare' else evaluate)(args.out,source)


if __name__=='__main__':main()
