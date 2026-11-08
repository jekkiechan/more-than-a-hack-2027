"""Declared execution for the paired warm-start boundary-loss experiment."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from experiments import v2_iterative_train as trainer
from experiments import v2_short_original_diversity as recipe

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'v2_iterative_train_v1'
OUT = ROOT / 'data/experiments' / STUDY
DECL = ROOT / f'experiments/results/{STUDY}_declaration.json'
SEAL = ROOT / f'experiments/results/{STUDY}_seal.json'
PARENT = ROOT / 'experiments/results/v2_iterative_v1_plan.json'
PRIOR_SEAL = ROOT / 'experiments/results/short_original_diversity_v1_model_seal.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write('\n')


def bind():
    recipe.OUT = OUT
    recipe.ARCHIVE_ROOT = Path(os.environ.get('MTAH_RF_ARCHIVE_DIR', '/workspace/mtah-rf/experiments/feed_robustness_v1/artifacts'))
    recipe.TRAIN_RECORDS = OUT / 'gar'
    # Only TRAIN records are made available through this adapter path.
    target = Path(os.environ.get('V2_ITERATIVE_GAR_ARCHIVE', '/workspace/mtah-dt/experiments/dense_targeted_followups_v1/artifacts/train_records'))
    link = recipe.TRAIN_RECORDS / 'train'
    if not link.exists():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target.resolve(): raise RuntimeError('GAR train path changed')
    return recipe


def prior_path(fold, seed):
    root = Path(os.environ.get('MTAH_SQD_DIR', '/workspace/mtah-sqd'))
    return root / 'experiments/short_quality_diversity_delivery_v1/artifacts/short_original_diversity_v1/models' / f'{fold}_{seed}.joblib'


def model_path(arm, fold, seed):
    if arm not in trainer.ARMS or fold not in trainer.FOLDS or seed not in trainer.SEEDS:
        raise ValueError('Undeclared fit')
    return OUT / 'models' / arm / f'{fold}_{seed}.joblib'


def sources():
    paths = {Path(__file__), Path(trainer.__file__), PARENT, PRIOR_SEAL,
             ROOT / 'experiments/v2_iterative_runtime.py',
             ROOT / 'experiments/v2_iterative_execution.py',
             ROOT / 'experiments/results/v2_iterative_v1_execution_amendment.json',
             *recipe._source_files(), *recipe._view_pipeline_files()}
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def input_facts():
    bind()
    paths = {OUT / 'preparation.json', recipe.CALIBRATION}
    spec = recipe.rf.old.old_spec()
    for item in recipe.prep.split_plan()['train']:
        paths.add(recipe.prep.data_path(item, recipe.rf.old.DATA))
        paths.add(recipe.rf.old.DATA / Path(spec['prepared_training_caches'][item['match']]['cache']).relative_to('data'))
        paths.add(recipe._archive_root() / 'views' / f"{item['match']}.joblib")
        for rep in range(4):
            p = recipe._view_path(item['match'], rep)
            paths.update((p, Path(str(p) + '.json')))
    manifest = recipe.TRAIN_RECORDS / 'train/manifest.json'
    paths.add(manifest)
    m = json.loads(manifest.read_text())
    if len(m['game_files']) != 45: raise ValueError('Expected exactly 45 GAR TRAIN games')
    for name, digest in m['game_files'].items():
        p = manifest.parent / 'games' / name
        if sha(p) != digest: raise ValueError(f'GAR record seal mismatch: {name}')
        paths.add(p)
    for fold in trainer.FOLDS:
        for seed in trainer.SEEDS:
            prior = prior_path(fold, seed)
            receipt_path = prior.with_suffix('.json')
            receipt = json.loads(receipt_path.read_text())
            expected = json.loads(PRIOR_SEAL.read_text())['models'][
                f'data/experiments/short_original_diversity_v1/models/{fold}_{seed}.joblib']
            if (sha(prior) != expected or receipt['sha256'] != expected
                    or receipt['fold'] != fold or receipt['seed'] != seed):
                raise ValueError('Archived warm-start model/receipt differs from original seal')
            paths.update((prior, receipt_path, recipe.rf.sc.initial_path(fold, seed)))
    return {str(p): sha(p) for p in sorted(paths)}


def freeze():
    if DECL.exists() or SEAL.exists(): raise FileExistsError('Training study is frozen')
    bind()
    cohorts = {f: recipe.prep.split_plan(fold=f)['train'] for f in trainer.FOLDS}
    for fold, cohort in cohorts.items():
        if any(r['source'].lower() == fold.removeprefix('leave_out_') for r in cohort):
            raise ValueError('Held-out provider in training cohort')
    for fold in trainer.FOLDS:
        for seed in trainer.SEEDS:
            teacher = joblib.load(recipe.rf.sc.initial_path(fold, seed))
            prior = joblib.load(prior_path(fold, seed))
            if prior['source']['checkpoint_sha256'] != recipe.rf.sn.cap._checkpoint_sha(teacher):
                raise ValueError('Warm-start and teacher canonical hash mismatch')
            trainer.warmstart_state(teacher, prior)
    d = dict(study_id=STUDY, parent_plan_sha256=sha(PARENT),
        hypothesis='Warm-start continuation with a boundary-local signed H/A margin objective improves transition timing beyond matched continuation with the existing objective.',
        arms=list(trainer.ARMS), folds=list(trainer.FOLDS), seeds=list(trainer.SEEDS),
        budget=dict(fits=12, updates_per_fit=trainer.STEPS, batch_size=trainer.BATCH_SIZE,
                    gar_fraction=trainer.FRACTION_GAR, lr=trainer.LR, workers_max=2),
        treatment=dict(coefficient=trainer.BOUNDARY_COEFFICIENT, margin=trainer.BOUNDARY_MARGIN,
                       radius_samples=trainer.BOUNDARY_RADIUS,
                       masks='Only known contiguous H/A boundaries; preserve time/period, unknown and padding breaks'),
        initialization='Exact archived original_full H/A state and normalization; original teacher pairing; copied adopted d_state unchanged',
        sampling='Shared OriginalDiversitySampler, four deterministic PR82 v2 TRAIN views round robin, identical arm/seed draw schedules',
        evaluation_pipelines=['each recipe fixed adopted D / H/A cost3', 'each recipe half conditional-H/A blend with adopted / fixed adopted D / H/A cost3'],
        train_cohorts=cohorts, input_sha256=input_facts(), source_sha256=sources(),
        organizer_reference_access=False, gar_valid_test_access=False, promotion=False,
        reconstruction='New continuation checkpoints from verified archived adopted weights, not reconstruction of unavailable PR82-84 weights')
    once(DECL, d)
    return d


def check(full=True):
    bind()
    d = json.loads(DECL.read_text())
    if d['study_id'] != STUDY or d['source_sha256'] != sources(): raise RuntimeError('Frozen training source changed')
    if full and d['input_sha256'] != input_facts(): raise RuntimeError('Frozen training inputs changed')
    return d


def fit_one(job):
    arm, fold, seed = job
    check(full=False)
    path = model_path(arm, fold, seed)
    receipt_path = Path(str(path) + '.json')
    if path.exists() or receipt_path.exists(): raise FileExistsError(f'Fit already attempted: {path}')
    teacher = joblib.load(recipe.rf.sc.initial_path(fold, seed))
    prior = joblib.load(prior_path(fold, seed))
    old = recipe.training(fold)
    gar = recipe.rf.dl.load_dense_records('train', output=recipe.TRAIN_RECORDS)
    sampler = recipe.short.OriginalDiversitySampler(old, gar, list(teacher['feature_names']), seed)
    start = time.monotonic()
    with recipe._round_robin_v2_windows(old) as views:
        with recipe.short.parent.base.installed(sampler):
            result = trainer.fit_from_checkpoints(teacher, prior, old, gar, seed=seed, arm=arm, verbose=True)
    model = result['model']
    if set(model['d_state']) != set(prior['d_state']): raise ValueError('D state keys changed')
    for key in prior['d_state']:
        np.testing.assert_array_equal(model['d_state'][key], prior['d_state'][key])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    r = dict(study_id=STUDY, arm=arm, fold=fold, seed=seed, path=str(path.relative_to(ROOT)),
             sha256=sha(path), declaration_sha256=sha(DECL), wall_seconds=time.monotonic()-start,
             config=result['config'], sampling=sampler.receipt(), view_draws=dict(views),
             exposures=result['exposures'], loss_intervals=result['loss_intervals'],
             warmstart_sha256=sha(prior_path(fold, seed)),
             teacher_sha256=sha(recipe.rf.sc.initial_path(fold, seed)),
             unchanged_d_state=True, old_matches=sorted({str(x['match']) for x in old}),
             gar_games=sorted({str(x['match']) for x in gar}), organizer_reference_access=False)
    once(receipt_path, r)
    print('FIT COMPLETE', arm, fold, seed, round(r['wall_seconds'], 2), flush=True)
    return r


def fit_all(workers):
    if workers not in (1, 2): raise ValueError('Use one or two workers')
    check()
    if SEAL.exists(): raise FileExistsError(SEAL)
    jobs = [(a,f,s) for f in trainer.FOLDS for s in trainer.SEEDS for a in trainer.ARMS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fit_one,j) for j in jobs]
        for future in as_completed(futures): rows.append(future.result())
    rows.sort(key=lambda r:(r['fold'],r['seed'],r['arm']))
    for fold in trainer.FOLDS:
        for seed in trainer.SEEDS:
            pair = [r for r in rows if r['fold']==fold and r['seed']==seed]
            for key in ('sampling','view_draws','exposures','old_matches','gar_games'):
                if pair[0][key] != pair[1][key]: raise ValueError(f'Unmatched draw schedule: {fold}/{seed}/{key}')
    once(SEAL, dict(study_id=STUDY, declaration_sha256=sha(DECL), fits=rows,
                    all_12_complete=True, paired_draw_schedules_equal=True,
                    organizer_reference_access=False, promotion=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','freeze','fit'])
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    bind()
    if args.action=='prepare': recipe.prepare(workers=args.workers)
    elif args.action=='freeze': freeze()
    else: fit_all(args.workers)

if __name__=='__main__': main()
