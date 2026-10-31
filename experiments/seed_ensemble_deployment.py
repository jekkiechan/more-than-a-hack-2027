"""One frozen pooled seed-2027 deployment fit for the user-selected ensemble.

The six external holdout models already tested the recipe. This fit uses only
the thirteen TRAIN matches; no scoring, selection or organizer inputs occur.
Promotion is a separate operation and never fits a model.
"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import shutil
import time
import joblib
import numpy as np
from experiments import (preparation as prep, learned_temporal as lt,
    factorized_had as model, gru_robustness_experiment as gr,
    corruption_experiment as ce)
from experiments.eval_set.protocol import sha256

ROOT=lt.ROOT
OUT=ROOT/'data/experiments/seed_ensemble_deployment_v1'
CONTROL=ROOT/'data/models/joint_gru_slotdrop_v1/joint_gru.joblib'
MODEL=OUT/'pooled_slot_dropout_2027.joblib'
SOURCES=('factorized_had.py','learned_temporal.py','gru_robustness_experiment.py',
 'live_gap_experiment.py','refine_features.py','refine_domain.py','supervised.py',
 'robust.py','stress.py','tracking_corruption.py','light_corruption.py','preparation.py')


def sources():
    paths=[ROOT/'experiments'/p for p in SOURCES]+[Path(__file__),ROOT/'scripts/fit_context_filebacked.py']
    return {str(p.relative_to(ROOT)):sha256(p) for p in paths}


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'predeclared.json').exists():return check()
    manifest=ROOT/'experiments/baseline_manifest.json'
    active=json.loads(manifest.read_text())
    if active['package_version']!='joint_gru_slotdrop_v1' or sha256(CONTROL)!=active['models_sha256']['joint_gru.joblib']:
        raise ValueError('Restore verified original deployment model first')
    shutil.copyfile(manifest,OUT/'previous_baseline_manifest.json')
    plan=prep.split_plan(fold='pooled')
    dependencies={str(CONTROL.relative_to(ROOT)):sha256(CONTROL)}
    for item in plan['train']:
        native=prep.data_path(item,ROOT/'data')
        if sha256(native)!=item['sha256']:raise ValueError('Training input changed')
        cache=ROOT/f"data/experiments/seed_gap_context_v1/train_{item['match']}.joblib"
        receipt=json.loads(cache.with_suffix('.json').read_text())
        if sha256(cache)!=receipt['sha256'] or receipt['input_sha256']!=item['sha256']:
            raise ValueError('Incompatible or altered reusable training cache')
        for p in (native,cache,cache.with_suffix('.json')):dependencies[str(p.relative_to(ROOT))]=sha256(p)
    spec=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Deployment completion of explicitly user-adopted equal 2026/2027 probability ensemble',
        user_request="this seems like an improvement. let's make this the updated baseline.",
        seed=2027,budget='Exactly one pooled fit; eight epochs, 10s core +5s context; no early stopping or scoring',
        split_plan=plan,split_plan_sha256=prep.fingerprint(plan),
        recipe='Exact legacy slot_dropout paired_records and validated unnormalized caches from the completed study; retain original within-provider augmentation indices, rather than substitute the newer shared augmentation seeds.',
        unchanged='Features, hidden24 player-only-D factorized GRU, augmentation masks, H/A swaps, equal provider/match weights, normalization population, AdamW and decoder3.',
        source_sha256=sources(),dependencies_sha256=dependencies,
        original_manifest_sha256=sha256(manifest),
        baseline_control_restored=True,organizer_inputs_read=False,organizer_labels_read=False,
        selection_matches_used_for_fitting=False,external_comparison='Reuse completed ensemble evidence without rescoring')
    ce.write_json(OUT/'predeclared.json',spec,exclusive=True)
    ce.write_json(ROOT/'experiments/results/seed_ensemble_deployment_declaration.json',spec,exclusive=True)
    return spec


def check():
    spec=json.loads((OUT/'predeclared.json').read_text())
    if sources()!=spec['source_sha256']:raise ValueError('Frozen fit source changed')
    for path,digest in spec['dependencies_sha256'].items():
        if sha256(ROOT/path)!=digest:raise ValueError(f'Frozen dependency changed: {path}')
    if prep.fingerprint(spec['split_plan'])!=spec['split_plan_sha256']:
        raise ValueError('Split plan changed')
    return spec


def fit():
    spec=check()
    if MODEL.exists():return ce.load_model(OUT,MODEL)
    if lt.CONTEXT!=25 or lt.CORE!=50 or lt.EPOCHS!=8:raise ValueError('Original window/budget required')
    records=[]
    for item in spec['split_plan']['train']:
        cache=joblib.load(ROOT/f"data/experiments/seed_gap_context_v1/train_{item['match']}.joblib")
        records.extend(gr.paired_records(cache,item['match'],'slot_dropout'))
    if len(records)!=52:raise ValueError('Expected exactly four records for each of thirteen TRAIN matches')
    from scripts.fit_context_filebacked import file_backed_fit
    started=time.monotonic()
    with file_backed_fit():fitted=model.fit(records,player_only=True,seed=2027)
    control=joblib.load(CONTROL)
    for key in ('median','scale'):
        if not np.array_equal(fitted[key],control[key]):raise ValueError(f'Pooled control normalization differs: {key}')
    for key in ('feature_names','parameters','windows_per_epoch','optimizer_updates'):
        if fitted[key]!=control[key]:raise ValueError(f'Pooled control parity differs: {key}')
    ce.freeze_model(OUT,MODEL,fitted,dict(seed=2027,deployment='pooled_external_train_only',
        training_matches=[i['match'] for i in spec['split_plan']['train']],
        excluded_selection_matches=[i['match'] for i in spec['split_plan']['selection']],
        seconds=time.monotonic()-started,matched_normalization=True,
        parameters=fitted['parameters'],windows_per_epoch=fitted['windows_per_epoch'],
        optimizer_updates=fitted['optimizer_updates'],organizer_labels_read=False))
    print('POOLED_SEED_2027_FROZEN',sha256(MODEL),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','fit'))
    args=parser.parse_args()
    (freeze if args.action=='freeze' else fit)()
