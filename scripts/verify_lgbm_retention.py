#!/usr/bin/env python3
"""Independent verification of saved development-only LGBM retention outputs.

Run only after the study owner signals that the saved result is ready.
This script never opens public assets.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import feed_benchmark_audit_v1 as audit
from experiments import sc2020_realfeed_score as scoring

STUDY='lgbm_spell_retention_v1'
OUT=ROOT/'data/experiments'/STUDY
RES=ROOT/'experiments/results'
DECL=RES/f'{STUDY}_declaration.json'
PREP=OUT/'preparation.json'
CONTROL=RES/f'{STUDY}_control.json'
TRAIN=RES/f'{STUDY}_training.json'
RESULT=RES/f'{STUDY}.json'
PRIVATE_RESULT=OUT/'result_private.json'
OUTPUT_POLICY=RES/f'{STUDY}_output_policy.json'
EVAL_ENTRY=ROOT/'scripts/run_lgbm_retention_evaluation.py'
RECEIPT=RES/f'{STUDY}_verification.json'
VERIFIER=Path(__file__).resolve()
SEEDS=('2026','2027')
RECIPES=('full','equalview','nomiss')
ARMS=('full','equalview','nomiss','topology_both','topology_start','topology_end')
MEMBERS=('ensemble','2026','2027')
METRICS=tuple(scoring.METRICS)+('d_precision','d_recall','weighted_70_30','graded_frames','short_reference_spell_count')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read(path): return json.loads(Path(path).read_text())
def close(a,b,tol=1e-12):
    if a is None or b is None: return a is b
    if isinstance(a,(int,np.integer)) and isinstance(b,(int,np.integer)): return int(a)==int(b)
    return abs(float(a)-float(b)) <= tol

def equal_provider(rows):
    out={}
    for view in scoring.VIEW_NAMES:
        providers={}
        for key,row in rows.items():
            provider,match,v,rep=key.split('__')
            if v==view:
                providers.setdefault(provider,{}).setdefault(match,[]).append(row)
        out[view]={}
        for metric in METRICS:
            per={p:float(np.mean([np.mean([r[metric] for r in reps]) for reps in matches.values()]))
                 for p,matches in providers.items()}
            out[view][metric]=float(np.mean(list(per.values())))
            out[view][metric+'/providers']=per
    return out

def verify():
    if RECEIPT.exists(): raise FileExistsError(RECEIPT)
    for p in (DECL,PREP,CONTROL,TRAIN,RESULT,PRIVATE_RESULT,OUTPUT_POLICY,EVAL_ENTRY):
        if not p.is_file(): raise FileNotFoundError(p)
    decl,prep,control,train,result=map(read,(DECL,PREP,CONTROL,TRAIN,RESULT))
    private=read(PRIVATE_RESULT)
    policy=read(OUTPUT_POLICY)
    if policy['declaration_sha256']!=sha(DECL): raise AssertionError('output policy declaration binding')
    if policy['entrypoint_sha256']!=sha(EVAL_ENTRY): raise AssertionError('output policy runner binding')
    if result.get('private_result_sha256')!=sha(PRIVATE_RESULT): raise AssertionError('private result hash binding')
    if result.get('output_policy_sha256')!=sha(OUTPUT_POLICY): raise AssertionError('output policy hash binding')
    for field in ('control','aggregate','verdicts','eligible','selected_for_confirmation','cell_rows',
                  'prediction_sha256','declaration_sha256','training_sha256','public_access','gar_access','adoption'):
        if result.get(field)!=private.get(field): raise AssertionError(f'public/private result mismatch {field}')
    def sanitize_diagnostics(value):
        return {key:{name:{field:(len(v) if isinstance(v,list) else v)
                           for field,v in diag.items()}
                     for name,diag in arms.items()}
                for key,arms in value.items()}
    if result.get('topology_diagnostics')!=sanitize_diagnostics(private['topology_diagnostics']):
        raise AssertionError('public topology diagnostics are not the private count projection')
    if len(result['topology_diagnostics'])!=35: raise AssertionError('public topology cell count')
    for key,arms in result['topology_diagnostics'].items():
        if len(arms)!=9: raise AssertionError(f'public topology arm count: {key}')
        for name,diag in arms.items():
            if set(diag)!={'mode','blocks','proposed_runs','accepted_runs','rejected_runs','fallback_blocks',
                           'changed_frames','d_runs_before','d_runs_after'}:
                raise AssertionError(f'public topology diagnostic field inventory {key}/{name}')
            if diag['mode'] not in ('both','start','end'): raise AssertionError(f'public topology mode {key}/{name}')
            if any(not isinstance(diag[f],int) or diag[f]<0 for f in set(diag)-{'mode'}):
                raise AssertionError(f'public topology diagnostic contains non-count data {key}/{name}')
    # Freeze chain and all declaration-bound source bytes.
    source_checks={}
    for rel,expected in decl['source_sha256'].items():
        p=ROOT/rel
        actual=sha(p)
        if actual!=expected: raise AssertionError(f'source hash mismatch: {rel}')
        source_checks[rel]=actual
    if prep['declaration_sha256']!=sha(DECL): raise AssertionError('preparation declaration binding')
    if prep['control_sha256']!=sha(CONTROL): raise AssertionError('preparation control binding')
    if train['declaration_sha256']!=sha(DECL): raise AssertionError('training declaration binding')
    if train['preparation_sha256']!=sha(PREP): raise AssertionError('training preparation binding')
    if result['declaration_sha256']!=sha(DECL): raise AssertionError('result declaration binding')
    if result['training_sha256']!=sha(TRAIN): raise AssertionError('result training binding')

    expected_cells={f'{provider}__{match}__{view}__{rep}'
        for provider,match,_ in audit.MATCHES
        for view,reps in (('clean',(0,)),('feed_like_v1',(1,2)),('feed_like_v2',(1,2,3,4)))
        for rep in reps}
    entries={r['key']:r for r in prep['cells']}
    if set(entries)!=expected_cells or len(entries)!=35: raise AssertionError('preparation cell roster')
    cells={}
    for key,r in entries.items():
        p=ROOT/r['path']
        if sha(p)!=r['sha256']: raise AssertionError(f'prepared cell hash: {key}')
        with np.load(p,allow_pickle=False) as z: cells[key]={k:z[k].copy() for k in z.files}
        c=cells[key]
        n=len(c['labels'])
        if len(c['baseline'])!=n or len(c['model_period'])!=n: raise AssertionError(f'cell frame alignment: {key}')

    if len(train['fits'])!=18 or train.get('fit_count')!=18: raise AssertionError('expected exactly 18 fits')
    fit_index={}
    for fit in train['fits']:
        ident=(fit['recipe'],fit['held_out_provider'],str(fit['seed']))
        if ident in fit_index: raise AssertionError(f'duplicate fit {ident}')
        fit_index[ident]=fit
        model=ROOT/fit['path']
        if sha(model)!=fit['sha256']: raise AssertionError(f'model hash {ident}')
        expected_keys={k for k in expected_cells if k.split('__')[0]!=fit['held_out_provider']}
        if set(fit['training_keys'])!=expected_keys: raise AssertionError(f'provider exclusion keys {ident}')
        if any(k.split('__')[0]==fit['held_out_provider'] for k in fit['training_keys']):
            raise AssertionError(f'held-out provider in train keys {ident}')
        nrows=0
        for k in fit['training_keys']:
            labels=cells[k]['labels']
            idx=np.arange(0,len(labels),5)
            nrows+=int(np.count_nonzero(labels[idx]!='X'))
        if fit['training_rows']!=nrows: raise AssertionError(f'training row count {ident}')
    expected_fit_index={(recipe,p,s) for recipe in RECIPES for p in ('Metrica','SkillCorner','Sportec') for s in SEEDS}
    if set(fit_index)!=expected_fit_index: raise AssertionError('fit inventory')
    model_files={str(p.relative_to(ROOT)):p for p in (OUT/'models').glob('*.txt')}
    listed_models={fit['path'] for fit in train['fits']}
    if len(model_files)!=18 or set(model_files)!=listed_models: raise AssertionError('model file inventory')
    prob_inventory=train['probabilities']
    if len(prob_inventory)!=210: raise AssertionError('expected 210 fold predictions')
    for rel,h in prob_inventory.items():
        if sha(OUT/rel)!=h: raise AssertionError(f'fold probability hash: {rel}')

    expected_arms={f'{arm}/{member}' for arm in ARMS for member in MEMBERS}
    pred_hashes=result['prediction_sha256']
    if set(pred_hashes)!=expected_cells: raise AssertionError('prediction file roster')
    saved_rows={arm:{} for arm in expected_arms}
    baseline_recomputed={}
    for key,c in cells.items():
        path=OUT/'predictions'/f'{key}.npz'
        if sha(path)!=pred_hashes[key]: raise AssertionError(f'prediction archive hash: {key}')
        with np.load(path,allow_pickle=False) as z:
            expected_encoded={name.replace('/','__') for name in expected_arms}
            if set(z.files)!=expected_encoded: raise AssertionError(f'arm roster: {key}')
            for name in sorted(expected_arms):
                pred=z[name.replace('/','__')]
                if pred.shape!=c['labels'].shape or not np.isin(pred,('H','A','D','X')).all():
                    raise AssertionError(f'bad saved labels: {key}/{name}')
                row=audit.score(c,pred)
                saved_rows[name][key]={m:float(row[m]) for m in METRICS}
                for pop in ('graded_frames','reference_spell_count','short_reference_spell_count'):
                    if row[pop]!=control['rows'][key][pop]:
                        raise AssertionError(f'grading count changed {key}/{name}/{pop}')
        base_row=audit.score(c,c['baseline'])
        baseline_recomputed[key]={m:float(base_row[m]) for m in METRICS}
        for m in METRICS:
            if not close(base_row[m],control['rows'][key][m]):
                raise AssertionError(f'baseline cell parity {key}/{m}')

    aggregates={}
    for name in sorted(expected_arms):
        agg=equal_provider(saved_rows[name]); aggregates[name]=agg
        if name not in result['cell_rows']: raise AssertionError(f'missing scalar rows {name}')
        if set(result['cell_rows'][name])!=expected_cells: raise AssertionError(f'scalar row roster {name}')
        for key in expected_cells:
            for m in METRICS:
                if not close(saved_rows[name][key][m],result['cell_rows'][name][key][m]):
                    raise AssertionError(f'scalar row parity {name}/{key}/{m}')
        if name not in result['aggregate']: raise AssertionError(f'missing aggregate {name}')
        for view in scoring.VIEW_NAMES:
            for m in METRICS:
                got=aggregates[name][view][m]
                if not close(got,result['aggregate'][name][view][m]):
                    raise AssertionError(f'aggregate parity {name}/{view}/{m}')
                gotp=aggregates[name][view][m+'/providers']
                refp=result['aggregate'][name][view][m+'/providers']
                if set(gotp)!=set(refp) or any(not close(gotp[k],refp[k]) for k in gotp):
                    raise AssertionError(f'provider aggregate parity {name}/{view}/{m}')
    # Independent replay of all saved baseline predictions and exact aggregate.
    base_agg=equal_provider(baseline_recomputed)
    for view in scoring.VIEW_NAMES:
        for m in METRICS:
            if not close(base_agg[view][m],control['aggregate'][view][m]):
                raise AssertionError(f'control aggregate parity {view}/{m}')
            if not close(base_agg[view][m],result['control'][view][m]):
                raise AssertionError(f'result control parity {view}/{m}')

    receipt={
      'study':STUDY,
      'verified_utc':datetime.now(timezone.utc).isoformat(),
      'declaration_sha256':sha(DECL),'preparation_sha256':sha(PREP),
      'control_sha256':sha(CONTROL),'training_sha256':sha(TRAIN),
      'result_sha256':sha(RESULT),
      'private_result_sha256':sha(PRIVATE_RESULT),
      'output_policy_sha256':sha(OUTPUT_POLICY),
      'evaluation_entrypoint_sha256':sha(EVAL_ENTRY),
      'verifier_sha256':sha(VERIFIER),
      'source_hashes_verified':len(source_checks),
      'prepared_cells_verified':len(cells),
      'prediction_cells_verified':len(pred_hashes),
      'prediction_arms_per_cell':len(expected_arms),
      'saved_cell_scores_recomputed':len(expected_cells)*len(expected_arms),
      'fit_models_verified':len(fit_index),
      'fold_probability_files_verified':len(prob_inventory),
      'heldout_provider_exclusion_verified':True,
      'official_scalar_score_parity':True,
      'equal_provider_aggregate_parity':True,
      'control_baseline_cell_and_aggregate_parity':True,
      'private_result_hash_and_public_topology_count_projection_verified':True,
      'grading_population_invariants_verified':True,
      'public_assets_or_labels_read':False,
    }
    with RECEIPT.open('x') as f: json.dump(receipt,f,indent=2,sort_keys=True); f.write('\n')
    return receipt

if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
