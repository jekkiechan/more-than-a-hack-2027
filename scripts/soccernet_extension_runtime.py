"""Audited metadata-only runtime correction for frozen SoccerNet study.

Keep the frozen experiment source and completed checkpoints unchanged. The
original fit driver saved models before failing to hash NumPy scalar match IDs
in its receipt. Recover those receipts without refitting, and scalarize identity
fields only when new pooled fits load the GAR cache. No feature/target changes.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
from experiments import soccernet_extension as study


def scalar_ids(records):
    return [dict(r,**{k:str(r[k]) for k in ('source','match','clip_id') if k in r}) for r in records]


def record_amendment():
    spec=study.check()
    value=dict(study_id='soccernet_extension_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        declaration_sha256=study.sha(study.OUT/'predeclared.json'),
        runtime_source_sha256=study.sha(__file__),frozen_source_sha256=spec['source_sha256'],
        error='TypeError: unhashable type numpy.ndarray in post-fit new_training_matches receipt set',
        correction='Convert only source/match/clip_id zero-dimensional identifiers to Python strings; recover completed fit receipts from saved standard checkpoints and their training metadata.',
        models_changed=False,features_changed=False,targets_changed=False,optimizer_budget_changed=False,
        prior_scoring=False,completed_weights_before_recovery={p.name:study.sha(p) for p in sorted(study.OUT.glob('leave_out_*.joblib'))})
    path=study.OUT/'runtime_amendment.json'
    if path.exists():
        old=study.read(path)
        if old['runtime_source_sha256']!=value['runtime_source_sha256']:raise ValueError('Runtime amendment source changed')
        return old
    study.write_once(path,value);return value


def recover(fold,arm,seed):
    path=study.model_path(fold,arm,seed)
    if not path.exists() or path.with_suffix('.json').exists():return
    spec=study.check();model=joblib.load(path);initial=joblib.load(study.control_path(fold,seed));meta=model['finetune']
    expected_old=sorted(i['match'] for i in spec['plans'][fold]['train'])
    new=joblib.load(study.OUT/'gar_prepared/records.joblib') if study.ARMS[arm] else []
    expected_new=sorted({str(r['match']) for r in new})
    if (meta['updates']!=study.UPDATES or model['optimizer_updates']!=study.UPDATES
        or meta['fraction']!=study.ARMS[arm] or meta['seed']!=seed
        or not np.array_equal(model['median'],initial['median'])
        or not np.array_equal(model['scale'],initial['scale'])
        or model['feature_names']!=initial['feature_names']
        or sorted(k.split('/',1)[1] for k in meta['old_window_exposures_by_source_match'])!=expected_old
        or sorted(k.split('/',1)[1] for k in meta['soccernet_window_exposures_by_game'])!=expected_new):
        raise ValueError('Completed checkpoint recovery contract failed')
    receipt=dict(fold=fold,arm=arm,seed=seed,sha256=study.sha(path),
        initial_sha256=study.sha(study.control_path(fold,seed)),declaration_sha256=study.sha(study.OUT/'predeclared.json'),
        seconds=None,metadata=meta,old_training_matches=expected_old,new_training_matches=expected_new,
        normalization_frozen=True,original_dead_checkpoint=str(study.control_path(fold,seed).relative_to(study.DATA)),
        receipt_recovery='Saved weights and optimizer metadata predate failed receipt serialization; no refit. Runtime amendment records the executed-source boundary.',
        runtime_amendment_sha256=study.sha(study.OUT/'runtime_amendment.json'))
    study.write_once(path.with_suffix('.json'),receipt)
    print('RECOVERED_RECEIPT',path.name,receipt['sha256'],flush=True)


@contextmanager
def scalarized_loader():
    load=joblib.load
    target=(study.OUT/'gar_prepared/records.joblib').resolve()
    def wrapped(path,*args,**kwargs):
        result=load(path,*args,**kwargs)
        return scalar_ids(result) if isinstance(path,(str,Path)) and Path(path).resolve()==target else result
    joblib.load=wrapped
    try:yield
    finally:joblib.load=load


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('amend','recover','fit'))
    p.add_argument('--fold',choices=(*study.FOLDS,'pooled'));p.add_argument('--arm',choices=tuple(study.ARMS));p.add_argument('--seed',type=int,choices=study.SEEDS)
    args=p.parse_args();record_amendment()
    if args.action=='recover':
        for fold in study.FOLDS:
            for arm in study.ARMS:
                for seed in study.SEEDS:recover(fold,arm,seed)
    elif args.action=='fit':
        recover(args.fold,args.arm,args.seed)
        with scalarized_loader():study.fit(args.fold,args.arm,args.seed)

if __name__=='__main__':main()
