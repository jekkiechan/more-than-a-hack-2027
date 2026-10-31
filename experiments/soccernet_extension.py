"""Declared sparse SoccerNet H/A transfer with matched warm-start controls.

No organizer reference labels are accessed here. Historical runners and active
baseline remain unchanged. P(D) always comes from the original fold checkpoint.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from experiments import preparation as prep, learned_temporal as lt, factorized_had as fh
from experiments import combined_objective_deployment as deployment
from experiments import supervised, objective_ensemble as scoring

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('MTAH_DATA_DIR', ROOT/'data')).resolve()
OUT = DATA/'experiments/soccernet_extension_v1'
GAR = DATA/'independent/soccernet_gar'
SEEDS = (2026, 2027)
FOLDS = ('leave_out_sportec','leave_out_metrica','leave_out_skillcorner')
ARMS = {'old_only': 0., 'gar25': .25, 'gar50': .5}
VIEWS = (('clean','clean'),('feed_like','corrupt_1'),('feed_like','corrupt_2'))
UPDATES = 256
SOURCE_PATHS = (
 'experiments/soccernet_extension.py','experiments/soccernet_data.py',
 'experiments/soccernet_finetune.py','experiments/soccernet_public_check.py',
 'experiments/preparation.py','experiments/refine_features.py','experiments/refine_domain.py',
 'experiments/supervised.py','experiments/learned_temporal.py','experiments/factorized_had.py',
 'experiments/combined_objective_deployment.py','experiments/objective_ensemble.py',
 'experiments/metrics.py','experiments/spell_diagnostics.py',
 'experiments/eval_set/manifest.json','experiments/baseline_manifest.json',
 'experiments/model_decision_policy.json')
METRICS = ('spell_f1','macro_f1','weighted_70_30','d_f1','d_precision','d_recall','short_spell_recall')

sha = prep.sha256

def read(path):
    return json.loads(Path(path).read_text())

def write_once(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value: raise ValueError('Refusing to overwrite sealed artifact: '+str(path))
        return
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n')
        handle.flush();os.fsync(handle.fileno())

def sources():
    paths = {ROOT/p for p in SOURCE_PATHS}
    paths.update(p for folder in ('experiments','evaluation')
                 for p in (ROOT/folder).rglob('*.py') if 'tests' not in p.parts)
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}

def control_path(fold,seed):
    if seed not in SEEDS or fold not in (*FOLDS,'pooled'): raise ValueError('Unknown fold/seed')
    return (DATA/f'models/joint_gru_combined_ensemble_v1/joint_gru_{seed}.joblib' if fold=='pooled'
        else DATA/f'experiments/objective_alignment_v1/without_{fold.removeprefix("leave_out_")}_combined_{seed}.joblib')

def model_path(fold,arm,seed):
    if arm not in ARMS: raise ValueError('Unknown arm')
    return OUT/f'{fold}_{arm}_{seed}.joblib'

def selection_items():
    value=read(ROOT/'experiments/eval_set/manifest.json')
    return [dict(i,source=s) for s,roles in value['sources'].items() for i in roles['selection']]

def eval_path(item,view,replicate):
    return DATA/f'experiments/seed_gap_context_v1/{item["match"]}_{view}_{replicate}.joblib'

def historical_path(item,view,replicate,seed):
    return DATA/f'experiments/objective_alignment_v1/{item["source"]}_{item["match"]}_{view}_{replicate}_combined_{seed}.npz'

def old_spec():
    return read(DATA/'experiments/objective_alignment_v1/predeclared.json')

def load_old(fold):
    """Reuse verified unnormalized historical features and augmentation exactly."""
    plan=prep.split_plan(fold=fold);old=old_spec();records=[]
    for item in plan['train']:
        entry=old['prepared_training_caches'][item['match']]
        path=DATA/Path(entry['cache']).relative_to('data')
        receipt=DATA/Path(entry['receipt']).relative_to('data')
        if sha(path)!=entry['sha256'] or sha(receipt)!=entry['receipt_sha256']:
            raise ValueError('Historical training cache checksum differs')
        cache=joblib.load(path)
        if cache['match']!=item['match'] or cache['source']!=item['source'] or cache['input_sha256']!=item['sha256']:
            raise ValueError('Training identity mismatch')
        records.extend(deployment.paired_records(cache,item['match']))
    excluded=fold.removeprefix('leave_out_')
    if fold!='pooled' and any(r['source']==excluded for r in records): raise ValueError('Provider leakage')
    return records

def freeze():
    from experiments import soccernet_data, soccernet_finetune
    del soccernet_data,soccernet_finetune
    if (OUT/'predeclared.json').exists(): return check()
    prepared=OUT/'gar_prepared/records.joblib';receipt=OUT/'gar_prepared/manifest.json'
    gar_info=read(receipt)
    files={}
    restored=read(OUT/'restore_manifest.json')['files']
    for entry in restored:
        path=DATA/entry['path']
        if sha(path)!=entry['sha256']: raise ValueError('Restored dependency differs: '+entry['path'])
        files[entry['path']]=entry['sha256']
    controls={}
    for fold in (*FOLDS,'pooled'):
        for seed in SEEDS:
            p=control_path(fold,seed);controls[str(p.relative_to(DATA))]=sha(p)
    plans={fold:prep.split_plan(fold=fold) for fold in (*FOLDS,'pooled')}
    provenance={}
    for fold in FOLDS:
        for seed in SEEDS:
            path=control_path(fold,seed);r=read(path.with_suffix('.json'))
            plan=plans[fold]
            if (r['sha256']!=sha(path) or r['heldout']!=fold.removeprefix('leave_out_')
                or r['seed']!=seed or r['objective']!='combined'
                or set(r['training_matches'])!={i['match'] for i in plan['train']}
                or set(r['sources'])!={i['source'] for i in plan['train']}):
                raise ValueError('Original fold checkpoint provenance differs')
            provenance[f'{fold}:{seed}']=r
    raw_hashes={}
    for entry in read(GAR/'acquisition_manifest.json')['files']:
        if entry['path'].startswith('tracking-full/train/'):
            actual=sha(GAR/entry['path'])
            if actual!=entry['sha256']: raise ValueError('GAR raw source checksum differs')
            raw_hashes[entry['path']]=actual
    # Validate feature schemas and sparse labels before freezing, never score here.
    new=joblib.load(prepared)
    expected_names=joblib.load(control_path('pooled',2026))['feature_names']
    for r in new:
        if np.sum(np.isin(r['y'],(0,1)))!=1 or np.any(np.asarray(r['y'])==2): raise ValueError('GAR target is not sparse H/A')
        if list(r['names'])!=expected_names: raise ValueError('Feature schema mismatch')
    value=dict(study_id='soccernet_extension_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does sparse native-broadcast event-team supervision improve possession spell F1 beyond equal-update original-data fine-tuning?',
        source_sha256=sources(),dependency_sha256=files,initial_checkpoint_sha256=controls,
        gar_cache_sha256=sha(prepared),gar_manifest_sha256=sha(receipt),gar_preparation=gar_info,
        plans=plans,initial_checkpoint_provenance=provenance,gar_raw_sha256=raw_hashes,
        arms=ARMS,seeds=SEEDS,folds=FOLDS,updates=UPDATES,batch_size=64,
        training='Warm-start conditional H/A only; AdamW lr1e-4 wd1e-4 clip1; frozen fold normalization; source/match-balanced original windows, game-balanced GAR clips; 25%/50% GAR windows; separate group-mean losses; matched old-only control.',
        d_path='Frozen original checkpoint P(D); tuned clone only contributes conditional H/A. Decoded D metrics can still move.',
        gar_targets='Only selected PASS/CROSS/SHOT centers; all other rows unknown. Event-team silver, not continuous possession gold. Visible-only raw coordinates; no provider events/identity metadata as features.',
        evaluation=[dict(item=i,view=v,replicate=r) for i in selection_items() for v,r in VIEWS],
        aggregation='Mean within provider across matches/replicates, then equal mean over three providers; seeds separately and actual equal-probability ensemble.',
        selection='Among GAR arms passing provider clean D/macro >= baseline-.02 and feed_like spell >= baseline-.01, choose highest unrounded clean ensemble spell; tie macro then lower GAR fraction. If none passes, choose highest clean GAR solely for requested diagnostic and mark ineligible. Never use public results for selection.',
        improvement_screen='Selected GAR clean spell >= baseline+.01 AND strictly above old-only matched fine-tune control; all safety guards pass. No automatic adoption.',
        public_diagnostic='One user-requested selected GAR ensemble after external decision seal and separate pooled fits; distinct exclusive gate before reference access. Reuse saved baseline scores. No tuning follows.',
        prior_exposure='All five original development matches and public five-minute diagnostic previously exposed. No fresh independent confirmation; GAR valid/test not used for model selection/training.',
        fixture_overlap='No identifier overlap across providers; no complete fixture-level crosswalk. Do not claim new independent test evidence.',
        organizer_labels_read=False,automatic_promotion=False)
    write_once(OUT/'predeclared.json',value)
    print('FROZEN',sha(OUT/'predeclared.json'),flush=True)
    return value

def check():
    spec=read(OUT/'predeclared.json')
    if spec['source_sha256']!=sources(): raise ValueError('Frozen source changed')
    if spec['arms']!=ARMS or spec['updates']!=UPDATES: raise ValueError('Frozen configuration changed')
    for rel,expected in spec['initial_checkpoint_sha256'].items():
        if sha(DATA/rel)!=expected: raise ValueError('Initial checkpoint changed')
    if sha(OUT/'gar_prepared/records.joblib')!=spec['gar_cache_sha256'] or sha(OUT/'gar_prepared/manifest.json')!=spec['gar_manifest_sha256']:
        raise ValueError('Prepared GAR data changed')
    return spec

def fit(fold,arm,seed):
    from experiments import soccernet_finetune as ft
    spec=check();path=model_path(fold,arm,seed)
    if fold=='pooled':
        decision=read(OUT/'decision.json')
        if arm!=decision['selected_arm']: raise ValueError('Pooled fit must use externally selected arm')
    elif fold not in FOLDS: raise ValueError('Unknown fit fold')
    if path.exists():
        if sha(path)!=read(path.with_suffix('.json'))['sha256']: raise ValueError('Model seal mismatch')
        return
    old=load_old(fold);new=joblib.load(OUT/'gar_prepared/records.joblib') if ARMS[arm] else []
    initial=joblib.load(control_path(fold,seed));start=time.monotonic()
    model=ft.train(initial,old,new,ARMS[arm],seed,updates=UPDATES)
    if not np.array_equal(model['median'],initial['median']) or not np.array_equal(model['scale'],initial['scale']):
        raise ValueError('Normalizer changed')
    joblib.dump(model,path)
    receipt=dict(fold=fold,arm=arm,seed=seed,sha256=sha(path),initial_sha256=sha(control_path(fold,seed)),
        declaration_sha256=sha(OUT/'predeclared.json'),seconds=time.monotonic()-start,
        metadata=model.get('finetune',model.get('finetune_metadata',{})),
        old_training_matches=sorted({r['match'] for r in old}),
        new_training_matches=sorted({r['match'] for r in new}),normalization_frozen=True,
        original_dead_checkpoint=str(control_path(fold,seed).relative_to(DATA)))
    write_once(path.with_suffix('.json'),receipt)
    print('FIT',fold,arm,seed,round(receipt['seconds'],1),flush=True)

def seal():
    check();files={}
    for fold in FOLDS:
        for arm in ARMS:
            for seed in SEEDS:
                p=model_path(fold,arm,seed);r=read(p.with_suffix('.json'))
                if sha(p)!=r['sha256']: raise ValueError('Fit seal changed')
                files[str(p.relative_to(DATA))]=r['sha256']
    result=dict(declaration_sha256=sha(OUT/'predeclared.json'),models=files)
    write_once(OUT/'checkpoints_frozen.json',result)
    return result

def inference_record(record):
    return {k:record[k] for k in ('x','names','period','indices','native_n','fps')}

def upsample(sampled,record,inputs):
    periods=supervised.model_periods(inputs)
    return np.column_stack([lt.upsample(sampled[:,k],record,periods) for k in range(3)])

def metric_row(probability,inputs,labels,item,view,replicate,arm,member):
    prediction=supervised.decode(probability,supervised.model_periods(inputs),float(inputs['fps']),lt.COST)
    cell=dict(source=item['source'],match=item['match'],view=view,replicate=replicate,arm=arm,member=member)
    return scoring.score_prediction(labels,prediction,inputs,arm,cell,scoring.digest_array(probability)),prediction

def evaluate(match=None):
    from experiments import soccernet_finetune as ft
    spec=check();sealed=seal()
    for item in selection_items():
        if match and item['match']!=match: continue
        fold='leave_out_'+item['source']
        models={arm:{seed:joblib.load(model_path(fold,arm,seed)) for seed in SEEDS} for arm in ARMS}
        originals={seed:joblib.load(control_path(fold,seed)) for seed in SEEDS}
        for view,replicate in VIEWS:
            output=OUT/f'cell_{item["match"]}_{view}_{replicate}.json'
            if output.exists(): continue
            p=eval_path(item,view,replicate)
            if sha(p)!=spec['dependency_sha256'][str(p.relative_to(DATA))]: raise ValueError('Evaluation cache changed')
            cached=joblib.load(p);inputs,labels=cached['inputs'],cached['labels'];record=inference_record(cached['record'])
            raw={seed:fh.predict(record,originals[seed]) for seed in SEEDS}
            native={seed:upsample(raw[seed],record,inputs) for seed in SEEDS}
            for seed in SEEDS:
                hp=historical_path(item,view,replicate,seed)
                if sha(hp)!=spec['dependency_sha256'][str(hp.relative_to(DATA))]: raise ValueError('Historical control array changed')
                with np.load(hp,allow_pickle=False) as prior:
                    if not np.allclose(native[seed],prior['probability'],atol=1e-7,rtol=0): raise ValueError('Historical baseline probability parity failed')
            rows=[];files={};n_gt=set();graded=set()
            all_prob={'baseline':native}
            for arm in ARMS:
                all_prob[arm]={}
                for seed in SEEDS:
                    probability=ft.predict(record,models[arm][seed],originals[seed],original_probability=raw[seed])
                    if not np.array_equal(probability[:,2],raw[seed][:,2]): raise ValueError('Sampled P(D) changed')
                    all_prob[arm][seed]=upsample(probability,record,inputs)
                    if not np.array_equal(all_prob[arm][seed][:,2],native[seed][:,2]): raise ValueError('Native P(D) changed')
            for arm,probabilities in all_prob.items():
                probabilities=dict(probabilities,ensemble=(probabilities[2026]+probabilities[2027])*.5)
                for member,probability in probabilities.items():
                    row,prediction=metric_row(probability,inputs,labels,item,view,replicate,arm,str(member))
                    n_gt.add(row['n_gt_spells']);graded.add(row['graded_frames']);rows.append(row)
                    f=OUT/f'{item["match"]}_{view}_{replicate}_{arm}_{member}.npz'
                    np.savez_compressed(f,probability=probability,prediction=prediction)
                    files[str(f.relative_to(DATA))]=sha(f)
            if len(n_gt)!=1 or len(graded)!=1: raise ValueError('Grading population differs')
            write_once(output,dict(declaration_sha256=sha(OUT/'predeclared.json'),checkpoint_seal_sha256=sha(OUT/'checkpoints_frozen.json'),rows=rows,files=files,baseline_probability_parity=True,exact_dead_probability_parity=True))
            print('CELL',item['match'],view,replicate,flush=True)


def summarize(rows,arm,view,member='ensemble'):
    selected=[r for r in rows if (r['arm'],r['view'],r['member'])==(arm,view,member)]
    providers={};keys=(*METRICS,'n_gt_spells','n_pred_spells','fragmented_reference_spells')
    for source in ('sportec','metrica','skillcorner'):
        values=[r for r in selected if r['source']==source]
        providers[source]={k:float(np.mean([r[k] for r in values if r.get(k) is not None])) for k in keys}
        providers[source]['spell_by_iou']={k:float(np.mean([r['spell_by_iou'][k] for r in values])) for k in ('0.3','0.4','0.5','0.6','0.7')}
    mean={k:float(np.mean([v[k] for v in providers.values()])) for k in keys}
    mean['spell_by_iou']={k:float(np.mean([v['spell_by_iou'][k] for v in providers.values()])) for k in ('0.3','0.4','0.5','0.6','0.7')}
    return dict(equal_provider_mean=mean,per_provider=providers)


def select_arm(summaries):
    baseline=summaries['baseline'];eligible=[];guards={}
    for arm in ('gar25','gar50'):
        s=summaries[arm]
        checks={f'{p}_{m}':s['clean']['per_provider'][p][m]>=baseline['clean']['per_provider'][p][m]-.02
            for p in ('sportec','metrica','skillcorner') for m in ('d_f1','macro_f1')}
        checks['feed_like_spell']=s['feed_like']['equal_provider_mean']['spell_f1']>=baseline['feed_like']['equal_provider_mean']['spell_f1']-.01
        guards[arm]=checks
        if all(checks.values()):eligible.append(arm)
    pool=eligible or ['gar25','gar50']
    chosen=max(pool,key=lambda arm:(summaries[arm]['clean']['equal_provider_mean']['spell_f1'],summaries[arm]['clean']['equal_provider_mean']['macro_f1'],-ARMS[arm]))
    clean=summaries[chosen]['clean']['equal_provider_mean']['spell_f1']
    gains=dict(clean_vs_baseline=clean-baseline['clean']['equal_provider_mean']['spell_f1'],
        feed_like_vs_baseline=summaries[chosen]['feed_like']['equal_provider_mean']['spell_f1']-baseline['feed_like']['equal_provider_mean']['spell_f1'],
        clean_vs_old_only=clean-summaries['old_only']['clean']['equal_provider_mean']['spell_f1'])
    return dict(selected_arm=chosen,selected_is_guardrail_eligible=chosen in eligible,guards=guards,gains=gains,
        improvement_screen_passes=chosen in eligible and gains['clean_vs_baseline']>=.01 and gains['clean_vs_old_only']>0,
        adopt_model=False,reason='External selection only; no automatic promotion. One requested public diagnostic follows regardless of screen.',public_used_for_selection=False)


def report():
    check();seal();rows=[]
    for item in selection_items():
        for view,replicate in VIEWS:
            result=read(OUT/f'cell_{item["match"]}_{view}_{replicate}.json')
            if (result['declaration_sha256']!=sha(OUT/'predeclared.json') or
                result['checkpoint_seal_sha256']!=sha(OUT/'checkpoints_frozen.json')):
                raise ValueError('Cell declaration/checkpoint identity differs')
            for rel,expected in result['files'].items():
                if sha(DATA/rel)!=expected: raise ValueError('Saved evaluation array changed')
            rows.extend(result['rows'])
    keys=[(r['source'],r['match'],r['view'],r['replicate'],r['arm'],r['member']) for r in rows]
    expected={(i['source'],i['match'],v,r,arm,member) for i in selection_items()
              for v,r in VIEWS for arm in ('baseline',*ARMS)
              for member in ('2026','2027','ensemble')}
    if len(keys)!=len(expected) or set(keys)!=expected:
        raise ValueError('Incomplete or duplicate evaluation cohort')
    summaries={arm:{v:summarize(rows,arm,v) for v in ('clean','feed_like')} for arm in ('baseline',*ARMS)}
    historical=read(DATA/'experiments/objective_ensemble_v1/report.json')['result']['recipes']['combined_objective_equal_probability_ensemble']
    for view in ('clean','feed_like'):
        for key in METRICS:
            if abs(summaries['baseline'][view]['equal_provider_mean'][key]-historical[view]['equal_provider_mean'][key])>1e-10:
                raise ValueError('Saved baseline aggregate parity failed: '+key)
    decision=select_arm(summaries)
    value=dict(study_id='soccernet_extension_v1',declaration_sha256=sha(OUT/'predeclared.json'),
        checkpoint_seal_sha256=sha(OUT/'checkpoints_frozen.json'),summaries=summaries,
        seed_summaries={arm:{str(seed):{v:summarize(rows,arm,v,str(seed)) for v in ('clean','feed_like')} for seed in SEEDS} for arm in ('baseline',*ARMS)},
        decision=decision,rows=rows,organizer_labels_read=False,
        limitations=['Previously exposed five-match development cohort; feed_like is synthetic.',
          'Sparse event-team silver; no SoccerNet continuous spell evaluation.',
          'P(D) is bitwise preserved, while decoder competition can change D precision/recall.',
          'Warm-start H/A extension rather than a full combined-loss refit; original normalizers stay frozen.'])
    write_once(OUT/'report.json',value);write_once(OUT/'decision.json',decision)
    write_once(ROOT/'experiments/results/soccernet_extension.json',value)
    print(json.dumps({arm:{v:summaries[arm][v]['equal_provider_mean']['spell_f1'] for v in ('clean','feed_like')} for arm in summaries}),flush=True)
    print(json.dumps(decision),flush=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','check','fit','seal','evaluate','report'))
    parser.add_argument('--fold',choices=(*FOLDS,'pooled'))
    parser.add_argument('--arm',choices=tuple(ARMS))
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--match')
    args=parser.parse_args()
    if args.action=='fit':fit(args.fold,args.arm,args.seed)
    elif args.action=='evaluate':evaluate(args.match)
    else:globals()[args.action]()

if __name__=='__main__':main()
