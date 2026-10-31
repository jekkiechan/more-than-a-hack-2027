"""Full three-class LightGBM feature ablation and complete-vector GRU ensemble.

User-corrected scope: these trees predict H/A/D directly. No D-only head
replacement, organizer scoring or automatic baseline promotion is performed.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import hashlib
import fcntl
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
from experiments import (joint_lgbm as model,possession_geometry as geometry,
    gru_robustness_experiment as gr,live_gap_experiment as old,
    corruption_experiment as ce,dead_transfer as dt,learned_temporal as lt,
    multisource_eval as ms,light_corruption as lc,robust,supervised,
    tracking_corruption as tc,spell_diagnostics)
from experiments.eval_set.protocol import sha256

ROOT=lt.ROOT
DEFAULT=ROOT/'data/experiments/joint_lgbm_v1'
VIEWS=('clean','corrupt_1','corrupt_2')
RECIPES=('standalone','control_2026','control_2027','slot_dropout_2026','slot_dropout_2027')


def fingerprints():
    return dict(gr.fingerprints(),**{str(p.relative_to(ROOT)):sha256(p) for p in
        (Path(__file__),Path(model.__file__),Path(geometry.__file__),ROOT/'experiments/temporal_lgbm.py')})


def augment(record,extra):
    if not np.array_equal(record['indices'],extra['indices']):raise ValueError('Geometry/sample clock mismatch')
    if record['x'].shape[1]!=204:raise ValueError('Expected actual204-column GRU base schema')
    return dict(record,x=np.column_stack((record['x'],extra['x'])),
        names=list(record['names'])+list(extra['names']),base_count=204)


def freeze(out):
    import importlib.metadata,platform
    out.mkdir(parents=True,exist_ok=True)
    if (out/'predeclared.json').exists():return check(out)
    previous=gr.check(gr.DEFAULT)
    caches={};sampled=[];manifest=None
    for source,split in dt.SPLITS.items():
        for path in split['train']:
            cached=gr.DEFAULT/f'train_{path.stem}.joblib'
            receipt=json.loads(cached.with_suffix('.json').read_text())
            if sha256(cached)!=receipt['sha256'] or receipt['predeclared_sha256']!=sha256(gr.DEFAULT/'predeclared.json'):
                raise ValueError('GRU training cache changed')
            if sha256(path)!=previous['data_sha256'][str(path.relative_to(ROOT))]:raise ValueError('Native training input changed')
            caches[str(cached.relative_to(ROOT))]=sha256(cached)
            r=joblib.load(cached)['clean']['joint']
            rows,mass=model.sampling(r['y'],r['period'])
            counts=[int(np.sum(r['y']==k)) for k in range(3)]
            represented=[float(mass[r['y'][rows]==k].sum()) for k in range(3)]
            if represented!=counts:raise ValueError('Sampling changed original5Hz class mass')
            sampled.append(dict(source=source,match=path.stem,known5Hz_frames=int(np.sum(r['y']>=0)),
                retained_per_view=len(rows),training_rows_four_views=4*len(rows),class5Hz_counts=counts,represented_class_mass=represented))
            if manifest is None:
                # Read real base names; zero examples below inspect only the fixed schema.
                dummy=dict(r,x=np.zeros((2,204+len(geometry.GEOMETRY_NAMES)+len(geometry.ANCHOR_NAMES)),np.float32),
                    period=np.ones(2),names=list(r['names'])+list(geometry.GEOMETRY_NAMES)+list(geometry.ANCHOR_NAMES),base_count=204)
                manifest={family:model.design(dummy,family)[1] for family in ('legacy','geometry','full')}
    spec=dict(created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=fingerprints(),
        gru_spec_sha256=sha256(gr.DEFAULT/'predeclared.json'),training_caches_sha256=caches,
        data_sha256=previous['data_sha256'],calibration=previous['calibration'],
        baseline_manifest_sha256=sha256(ROOT/'experiments/baseline_manifest.json'),
        policy_sha256=sha256(ROOT/'experiments/model_decision_policy.json'),
        training_matches=previous['training_matches'],evaluation_matches=previous['evaluation_matches'],
        configs=model.CONFIGS,common_parameters=model.COMMON,geometry_spec=geometry.SPEC,
        feature_manifest=manifest,feature_counts={f:len(m) for f,m in manifest.items()},
        sampled_training_matches=sampled,
        training_rows_by_fold={heldout:sum(r['training_rows_four_views'] for r in sampled if r['source']!=heldout) for heldout in dt.SPLITS},
        views=VIEWS,recipes=RECIPES,decoder_switch_cost=3,
        hypothesis='Genuinely new all-player control/interception geometry, reliable native-observation contact anchors, and ordered before/after support improve FULL H/A/D spell recovery; 127leaves separately tests additional capacity on identical full features.',
        scope='Three-class objective and full H/A/D probability vectors; standalone trees plus fixed50/50 complete-vector GRU blends. No conditional-team preservation or D-only replacement.',
        training='Exact clean/coarse-ball joint-GRU base records;25% same frozen whole-core ball-gap replacement, direct+H/A-swapped records. Build nonlinear/temporal features separately on coherent original and alternate views before replacing cores. Equal provider/match/view total mass. All learned components exclude evaluation provider.',
        sampling='Training-only: split at input discontinuities and contiguous known H/A/D label runs; retain first/last3 samples of each run at5Hz, interiors stride4 (1.25Hz). Integer within-run nearest-selected Voronoi weights preserve original5Hz class/time mass. Unknown labels have zero mass. Every tiny run remains represented.',
        symmetry='Swap raw204+geometry+anchor channels BEFORE nonlinear temporal engineering; labels0/1 exchange, D2 unchanged. Predict direct+swapped, mapH/A back and average. Explicit even/odd/pair manifest verifies transformations.',
        budget='12fits (4arms x3providerfolds);640multiclass boosting rounds per fit,3class trees per round (up to1920trees). This is a bounded initial budget, not mathematically equivalent to1280binary rounds or a convergence claim. Record full training multi_logloss progression; no early stopping or outcome-driven extra fits.',
        evaluation='300new cells:4configs x15match/views x5recipes (standalone plus bothcontrol/dropoutGRUseeds). Reuse60GRUrows. Freeze all12tree checkpoints and all12GRUcheckpoints before new scoring; seal every native sharedview/probability hash.',
        primary='Unrounded equal-provider clean official spellF1; macroF1 exact-tie. Average matches within provider, stress replicates within match, then equal providers. Fixed .7spell+.3macro auxiliary only; report GRU seeds separately, no seed-selected checkpoint.',
        guardrails='Versus adopted-recipe matched full-provider archived baseline: clean spell >= baseline+.01; worst-provider clean D-F1/macro loss <=.02; stress spell >= baseline-.01. Report paired blend-vs-corresponding-GRU changes. Every recipe is assessed independently; no automatic promotion.',
        limitations='Five historically exposed external development matches, including silver labels and SkillCorner visibility/X censoring; synthetic stress is not independent broadcast confirmation. Full multiclass trees can use ball-source shortcuts; D is NOT player-only. Tree seed variance and1280-round fullHDA budgets are unmeasured.',
        diagnostics='Exact per-IoU spell counts, short/edge recall, overlapping fragmentation/merge geometry, complete-boundary timing; Dprecision/recall; training-gain families and training-loss progression are descriptive, not held-out causal importance.',
        runtime=dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in ('numpy','scikit-learn','lightgbm','torch','joblib','threadpoolctl')},threads_per_fit=1),
        organizer_labels_read=False,automatic_promotion=False,supersedes_scope='D-only temporal_lgbm_v1 and capacity_lgbm_v1 retained with scope-correction receipts; no scores opened in them')
    ce.write_json(out/'predeclared.json',spec,exclusive=True)
    print('FROZEN FULL H/A/D',spec['feature_counts'],spec['training_rows_by_fold'],flush=True)
    return spec


def check(out):
    spec=json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256']!=fingerprints() or spec['gru_spec_sha256']!=sha256(gr.DEFAULT/'predeclared.json'):
        raise ValueError('Frozen H/A/D source or GRU declaration changed')
    return spec


def prepare(out,source):
    spec=check(out)
    for i,path in enumerate(dt.SPLITS[source]['train']):
        target=out/f'geometry_{path.stem}.joblib'
        if target.exists():
            if json.loads(target.with_suffix('.json').read_text())['sha256']!=sha256(target):raise ValueError('Geometry cache changed')
            continue
        if sha256(path)!=spec['data_sha256'][str(path.relative_to(ROOT))]:raise ValueError('Native training input changed')
        started=time.monotonic();inputs,labels=ms.view(path)
        views=dict(clean=inputs,legacy=robust.masked_inputs(inputs,2026+i),
            gap=lc.ball_gap_view(inputs,spec['calibration'],ce.seed(path.stem,'train')))
        records={}
        for view,data in views.items():
            x,names=geometry.features(data,include_anchors=True);indices=lt.sampled_indices(data)
            records[view]=dict(x=x[indices],names=names,indices=indices)
        joblib.dump(records,target)
        ce.write_json(target.with_suffix('.json'),dict(sha256=sha256(target),input_sha256=sha256(path),
            predeclared_sha256=sha256(out/'predeclared.json'),seconds=time.monotonic()-started),exclusive=True)
        print('GEOMETRY PREPARED',path.stem,round(time.monotonic()-started,1),flush=True)


def training_records(out,source):
    spec=check(out)
    for provider,split in dt.SPLITS.items():
        if provider==source:continue
        for path in split['train']:
            original=gr.DEFAULT/f'train_{path.stem}.joblib'
            if sha256(original)!=spec['training_caches_sha256'][str(original.relative_to(ROOT))]:raise ValueError('Original training cache changed')
            extra_path=out/f'geometry_{path.stem}.joblib';receipt=json.loads(extra_path.with_suffix('.json').read_text())
            if receipt['sha256']!=sha256(extra_path) or receipt['predeclared_sha256']!=sha256(out/'predeclared.json'):raise ValueError('Geometry cache changed')
            cached=joblib.load(original);extra=joblib.load(extra_path)
            augmented={name:augment(cached[name]['joint'],extra[name]) for name in ('clean','legacy','gap')}
            for variant,name in enumerate(('clean','legacy')):
                for swapped in (False,True):
                    yield old.mixed_record(augmented[name],augmented['gap'],path.stem,variant,swapped)


def matrix(out,source,family):
    # Different capacity fits may share one full-feature matrix. Serialize its
    # preparation and checksum validation, never duplicate or partly read it.
    with (out/f'matrix_without_{source}_{family}.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _matrix(out,source,family)


def _matrix(out,source,family):
    """Disk-backed rows bound RAM and reuse identical full features at63/127leaves."""
    spec=check(out);prefix=out/f'matrix_without_{source}_{family}'
    xp=prefix.with_suffix('.npy');yp=prefix.with_suffix('.npz');rp=prefix.with_suffix('.json')
    if rp.exists():
        receipt=json.loads(rp.read_text())
        if receipt['x_sha256']!=sha256(xp) or receipt['y_weight_sha256']!=sha256(yp) or receipt['predeclared_sha256']!=sha256(out/'predeclared.json'):
            raise ValueError('Prepared matrix changed')
        arrays=np.load(yp);return np.load(xp,mmap_mode='r'),arrays['y'],arrays['weight'],spec['feature_manifest'][family]
    if xp.exists() or yp.exists():raise ValueError('Unsealed matrix exists; inspect interrupted preparation')
    n=spec['training_rows_by_fold'][source];manifest=spec['feature_manifest'][family]
    x=np.lib.format.open_memmap(xp,mode='w+',dtype=np.float32,shape=(n,len(manifest)))
    y=np.empty(n,np.int8);weight=np.empty(n,float);at=0
    matches={s:len(split['train']) for s,split in dt.SPLITS.items()}
    class_masses=np.zeros(3)
    for r in training_records(out,source):
        rows,mass=model.sampling(r['y'],r['period'])
        full,actual=model.training_design(r,family)
        if actual!=manifest:raise ValueError('Training feature manifest mismatch')
        stop=at+len(rows);x[at:stop]=full[rows];y[at:stop]=r['y'][rows]
        weight[at:stop]=mass/(max(1,np.sum(r['y']>=0))*matches[r['source']]*4)
        for k in range(3):class_masses[k]+=weight[at:stop][y[at:stop]==k].sum()
        at=stop
    if at!=n:raise ValueError('Declared row count changed')
    x.flush();del x
    weight/=weight.mean();np.savez_compressed(yp,y=y,weight=weight)
    ce.write_json(rp,dict(x_sha256=sha256(xp),y_weight_sha256=sha256(yp),
        predeclared_sha256=sha256(out/'predeclared.json'),rows=n,features=len(manifest),
        provider_weighted_class_mass_before_mean_normalization=class_masses.tolist()),exclusive=True)
    return np.load(xp,mmap_mode='r'),y,weight,manifest


def checkpoint(out,source,config):return out/f'without_{source}_{config}.joblib'


def fit(out,source,config):
    from lightgbm import LGBMClassifier,record_evaluation
    spec=check(out);target=checkpoint(out,source,config)
    if target.exists():ce.load_model(out,target);return
    if (out/'checkpoints_frozen.json').exists():raise ValueError('Fitting closed')
    started=time.monotonic();settings=dict(model.CONFIGS[config]);family=settings.pop('family')
    x,y,weight,manifest=matrix(out,source,family)
    if not np.array_equal(np.unique(y),[0,1,2]):raise ValueError('All three training classes required')
    estimator=LGBMClassifier(**model.COMMON,**settings);history={}
    estimator.fit(x,y,sample_weight=weight,eval_set=[(x,y)],eval_sample_weight=[weight],
        eval_names=['training'],eval_metric='multi_logloss',callbacks=[record_evaluation(history)])
    fitted=dict(kind='joint_multiclass_lgbm',config=config,family=family,estimator=estimator,
        feature_manifest=manifest,training_rows=len(y),history=history,seed=2026)
    gain=estimator.booster_.feature_importance(importance_type='gain');grouped={}
    for entry,value in zip(manifest,gain):
        row=grouped.setdefault(entry['family'],dict(total_gain=0.,features=0));row['total_gain']+=float(value);row['features']+=1
    loss=history['training']['multi_logloss']
    ce.freeze_model(out,target,fitted,dict(heldout=source,training_matches=spec['training_matches'][source],
        config=config,family=family,seconds=time.monotonic()-started,training_rows=len(y),n_features=len(manifest),
        boosting_rounds=estimator.n_estimators_,class_trees=estimator.booster_.num_trees(),
        training_loss=loss,training_gain_by_family=grouped,
        top_training_gain_features=sorted([dict(feature=m['name'],gain=float(v)) for m,v in zip(manifest,gain)],key=lambda r:-r['gain'])[:30]))
    print('JOINT TREE FROZEN',source,config,round(time.monotonic()-started,1),'rounds',estimator.n_estimators_,'loss',loss[0],loss[-1],flush=True)


def seal(out):
    check(out);rows=[]
    for s in dt.SPLITS:
        for c in model.CONFIGS:
            p=checkpoint(out,s,c);ce.load_model(out,p)
            rows.append(dict(file=p.name,sha256=sha256(p),receipt_sha256=sha256(p.with_suffix('.json'))))
    target=out/'checkpoints_frozen.json'
    if target.exists():
        if json.loads(target.read_text())['checkpoints']!=rows:raise ValueError('Frozen checkpoints changed')
    else:ce.write_json(target,dict(checkpoints=rows),exclusive=True)


def seal_inputs(out):
    spec=check(out);gr.check(gr.DEFAULT);gr.seal(gr.DEFAULT)
    paths=[gr.DEFAULT/'predeclared.json',gr.DEFAULT/'checkpoints_frozen.json',gr.DEFAULT/'report.json']
    for source,split in dt.SPLITS.items():
        for arm in gr.ARMS:
            for seed in gr.SEEDS:
                p=gr.checkpoint(gr.DEFAULT,source,arm,seed);ce.load_model(gr.DEFAULT,p)
                r=json.loads(p.with_suffix('.json').read_text())
                if set(r['training_matches'])!=set(spec['training_matches'][source]):raise ValueError('GRU provider isolation changed')
                paths.extend([p,p.with_suffix('.json')])
        for p in split['eval']:
            for view in VIEWS:
                paths.append(gr.DEFAULT/f'eval_{p.stem}_{view}.joblib')
                paths.extend(gr.DEFAULT/f'{source}_{p.stem}_{view}_{arm}_{seed}.npz' for arm in gr.ARMS for seed in gr.SEEDS)
    payload={str(p.relative_to(ROOT)):sha256(p) for p in paths};target=out/'evaluation_inputs_frozen.json'
    if target.exists():
        if json.loads(target.read_text())['files_sha256']!=payload:raise ValueError('Shared evaluation dependencies changed')
        return
    for source,split in dt.SPLITS.items():
        for path in split['eval']:
            if sha256(path)!=spec['data_sha256'][str(path.relative_to(ROOT))]:raise ValueError('Native evaluation input changed')
            original,labels=ms.view(path)
            for view in VIEWS:
                cache=joblib.load(gr.DEFAULT/f'eval_{path.stem}_{view}.joblib')
                expected=original if view=='clean' else tc.corrupt(original,spec['calibration'],ce.seed(path.stem,view))
                expected=tc.inputs_only(expected)
                if set(expected)!=set(cache['inputs']) or not np.array_equal(labels,cache['labels']):raise ValueError('Shared raw view or labels differ')
                for key,value in expected.items():
                    if not np.array_equal(cache['inputs'][key],value,equal_nan=True):raise ValueError(f'Shared input mismatch:{key}')
                joint=cache['joint']
                if (joint['match']!=path.stem or joint['source']!=source or not np.array_equal(joint['indices'],lt.sampled_indices(expected))
                    or not np.array_equal(joint['period'],supervised.model_periods(expected)[joint['indices']])):
                    raise ValueError('Shared feature cohort/clock differs')
                for arm in gr.ARMS:
                    for seed in gr.SEEDS:
                        with np.load(gr.DEFAULT/f'{source}_{path.stem}_{view}_{arm}_{seed}.npz') as saved:
                            prob=saved['probability']
                            if prob.shape!=(len(labels),3) or not np.isfinite(prob).all() or np.any(prob<0) or not np.allclose(prob.sum(1),1):raise ValueError('Invalid GRU probability')
                            pred=supervised.decode(prob,supervised.model_periods(expected),float(expected['fps']),3)
                            if not np.array_equal(pred,saved['prediction']):raise ValueError('GRU probability/prediction mismatch')
    ce.write_json(target,dict(files_sha256=payload,views_verified=15,gru_probabilities_verified=60),exclusive=True)


def evaluate(out,source):
    check(out);seal(out);seal_inputs(out)
    dest=out/f'{source}_results.json';rows=json.loads(dest.read_text()) if dest.exists() else []
    fitted={c:ce.load_model(out,checkpoint(out,source,c)) for c in model.CONFIGS}
    for path in dt.SPLITS[source]['eval']:
        for view in VIEWS:
            cache=joblib.load(gr.DEFAULT/f'eval_{path.stem}_{view}.joblib')
            original,inputs,labels=cache['joint'],cache['inputs'],cache['labels']
            extra,names=geometry.features(inputs,include_anchors=True)
            record=augment(original,dict(x=extra[original['indices']],names=names,indices=original['indices']))
            periods=supervised.model_periods(inputs)
            for config,tree in fitted.items():
                sampled=model.predict(record,tree)
                probability=np.column_stack([lt.upsample(sampled[:,k],record,periods) for k in range(3)])
                artifact=out/f'{source}_{path.stem}_{view}_{config}_probability.npz'
                if artifact.exists():
                    with np.load(artifact) as saved:
                        if not np.array_equal(saved['probability'],probability) or not np.array_equal(saved['sampled'],sampled) or not np.array_equal(saved['indices'],record['indices']):
                            raise ValueError('Saved standalone probabilities changed')
                else:
                    with artifact.open('xb') as handle:np.savez_compressed(handle,probability=probability,sampled=sampled,indices=record['indices'])
                probability_receipt=dict(sha256=sha256(artifact),model_sha256=sha256(checkpoint(out,source,config)),
                    input_seal_sha256=sha256(out/'evaluation_inputs_frozen.json'))
                if artifact.with_suffix('.json').exists():
                    if json.loads(artifact.with_suffix('.json').read_text())!=probability_receipt:raise ValueError('Probability provenance changed')
                else:ce.write_json(artifact.with_suffix('.json'),probability_receipt,exclusive=True)
                for recipe in RECIPES:
                    key=(path.stem,view,config,recipe)
                    if any((r['match'],r['view'],r['config'],r['recipe'])==key for r in rows):continue
                    if recipe=='standalone':combined=probability
                    else:
                        arm,seed=recipe.rsplit('_',1)
                        with np.load(gr.DEFAULT/f'{source}_{path.stem}_{view}_{arm}_{seed}.npz') as saved:
                            combined=model.blend(probability,saved['probability'])
                    prediction=supervised.decode(combined,periods,float(inputs['fps']),3)
                    metrics=supervised.score(dict(inputs,labels=labels),prediction)
                    row=dict(source=source,match=path.stem,view=view,config=config,recipe=recipe,
                        probability_artifact=artifact.name,native_probability_sha256=hashlib.sha256(np.ascontiguousarray(combined).tobytes()).hexdigest(),
                        native_probability_dtype=combined.dtype.str,prediction_sha256=hashlib.sha256(np.ascontiguousarray(prediction).tobytes()).hexdigest(),prediction_dtype=prediction.dtype.str,
                        spell_f1=metrics['spell_f1_mean'],macro_f1=metrics['macro_f1'],weighted_summary=.7*metrics['spell_f1_mean']+.3*metrics['macro_f1'],
                        d_f1=metrics['per_class']['D']['f1'],d_precision=metrics['per_class']['D']['precision'],d_recall=metrics['per_class']['D']['recall'],
                        per_class=metrics['per_class'],n_gt_spells=metrics['n_gt_spells'],n_pred_spells=metrics['n_pred_spells'],graded_frames=int(np.isin(labels,supervised.LABELS).sum()),
                        diagnostics=spell_diagnostics.spell_diagnostics(labels,prediction,period=inputs['period'],fps=float(inputs['fps']),timestamps=inputs.get('timestamps')))
                    prediction_path=out/f'{source}_{path.stem}_{view}_{config}_{recipe}.npz'
                    if prediction_path.exists():
                        with np.load(prediction_path) as saved:
                            if not np.array_equal(saved['prediction'],prediction):raise ValueError('Saved prediction changed')
                    else:
                        with prediction_path.open('xb') as handle:np.savez_compressed(handle,prediction=prediction)
                    row['prediction_file_sha256']=sha256(prediction_path)
                    rows.append(row);ce.write_json(dest,rows)
                    print('JOINT SCORE',source,path.stem,view,config,recipe,round(row['spell_f1'],5),flush=True)


def summarize(rows):
    keys=('spell_f1','macro_f1','weighted_summary','d_f1','d_precision','d_recall')
    summaries=[]
    for config,recipe in sorted({(r['config'],r['recipe']) for r in rows}):
        for view in ('clean','stress'):
            per={}
            for source in dt.SPLITS:
                chosen=[r for r in rows if (r['config'],r['recipe'],r['source'])==(config,recipe,source) and (r['view']=='clean')==(view=='clean')]
                if not chosen:raise ValueError('Incomplete provider summary')
                per[source]={k:float(np.mean([r[k] for r in chosen])) for k in keys}
            summaries.append(dict(config=config,recipe=recipe,view=view,per_source=per,**{k:float(np.mean([r[k] for r in per.values()])) for k in keys}))
    return summaries


def report(out):
    check(out);seal(out);seal_inputs(out)
    rows=[r for s in dt.SPLITS for r in json.loads((out/f'{s}_results.json').read_text())]
    expected={(s,p.stem,v,c,recipe) for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] for v in VIEWS for c in model.CONFIGS for recipe in RECIPES}
    if len(rows)!=len(expected) or {(r['source'],r['match'],r['view'],r['config'],r['recipe']) for r in rows}!=expected:raise ValueError('Incomplete or duplicate joint cells')
    gru_report=json.loads((gr.DEFAULT/'report.json').read_text())
    controls=[dict(r,config='gru',recipe=f"{r['arm']}_{r['seed']}",weighted_summary=.7*r['spell_f1']+.3*r['macro_f1']) for r in gru_report['rows']]
    baseline=[dict(r,config='adopted_recipe',recipe='baseline',d_precision=r['per_class']['D']['precision'],d_recall=r['per_class']['D']['recall'],weighted_summary=.7*r['spell_f1']+.3*r['macro_f1']) for r in gru_report['archived_comparators']]
    expected_controls={(s,p.stem,v,f'{arm}_{seed}') for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] for v in VIEWS for arm in gr.ARMS for seed in gr.SEEDS}
    expected_baseline={(s,p.stem,v) for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] for v in VIEWS}
    if len(controls)!=60 or {(r['source'],r['match'],r['view'],r['recipe']) for r in controls}!=expected_controls:raise ValueError('Incomplete or duplicate GRU comparator cohort')
    if len(baseline)!=15 or {(r['source'],r['match'],r['view']) for r in baseline}!=expected_baseline:raise ValueError('Incomplete or duplicate archived baseline cohort')
    for r in rows:
        c=next(c for c in controls if (r['source'],r['match'],r['view'])==(c['source'],c['match'],c['view']))
        if (r['n_gt_spells'],r['graded_frames'])!=(c['n_gt_spells'],c['graded_frames']):raise ValueError('Joint grading population changed')
    summaries=summarize(rows+controls+baseline);base={r['view']:r for r in summaries if r['config']=='adopted_recipe'}
    screens=[];paired=[]
    for config in model.CONFIGS:
        for recipe in RECIPES:
            get=lambda c,v:next(r for r in summaries if (r['config'],r['recipe'],r['view'])==(c,recipe,v))
            clean,stress=get(config,'clean'),get(config,'stress')
            cd=clean['spell_f1']-base['clean']['spell_f1'];sd=stress['spell_f1']-base['stress']['spell_f1']
            pd=min(clean['per_source'][s][k]-base['clean']['per_source'][s][k] for s in dt.SPLITS for k in ('d_f1','macro_f1'))
            screens.append(dict(config=config,recipe=recipe,clean_spell_delta=cd,stress_spell_delta=sd,worst_provider_d_or_macro_delta=pd,passes=cd>=.01 and sd>=-.01 and pd>=-.02))
            if recipe!='standalone':
                for view in ('clean','stress'):
                    a,b=get(config,view),get('gru',view)
                    paired.append(dict(config=config,recipe=recipe,view=view,spell_delta=a['spell_f1']-b['spell_f1'],macro_delta=a['macro_f1']-b['macro_f1']))
    ce.write_json(out/'report.json',dict(spec=check(out),rows=rows,gru_controls=controls,archived_baseline=baseline,
        summaries=summaries,screens=screens,paired_blend_deltas=paired,
        model_receipts={p.name:json.loads(p.read_text()) for p in out.glob('without_*.json')},
        evaluation_inputs=json.loads((out/'evaluation_inputs_frozen.json').read_text()),
        gru_report_sha256=sha256(gr.DEFAULT/'report.json'),baseline_changed=False),exclusive=True)
    for r in summaries:print('SUMMARY',r['config'],r['recipe'],r['view'],round(r['spell_f1'],5),round(r['macro_f1'],5),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','prepare','matrix','fit','seal','seal-inputs','evaluate','report'))
    parser.add_argument('--out',type=Path,default=DEFAULT);parser.add_argument('--source',choices=tuple(dt.SPLITS));parser.add_argument('--config',choices=tuple(model.CONFIGS));parser.add_argument('--family',choices=('legacy','geometry','full'))
    args=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if args.action=='freeze':freeze(args.out)
        elif args.action=='seal':seal(args.out)
        elif args.action=='seal-inputs':seal_inputs(args.out)
        elif args.action=='report':report(args.out)
        elif args.action=='fit':
            if args.source is None or args.config is None:parser.error('fit requires source and config')
            fit(args.out,args.source,args.config)
        elif args.action=='matrix':
            if args.source is None or args.family is None:parser.error('matrix requires source and family')
            matrix(args.out,args.source,args.family)
        else:
            for source in ([args.source] if args.source else dt.SPLITS):(prepare if args.action=='prepare' else evaluate)(args.out,source)


if __name__=='__main__':main()
