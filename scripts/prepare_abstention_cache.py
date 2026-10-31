"""Parallel cache preparation for the frozen abstention experiment.

Run from the repository root. This only changes execution scheduling: the frozen
experiment module remains authoritative and seals all cells before scoring.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import confidence_abstention as ca


def prepare_cell(task):
    import numpy as np
    source, path, view, calibration = task
    target = ca.OUT/f'{path.stem}_{view}.npz'
    if target.exists():
        return target.name
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        models = ca.cb.load_models()
        for kind in ('bigru', 'ordered_hgb'):
            models[kind] = ca.ce.load_model(ca.ce.DEFAULT, ca.ce.DEFAULT/f'without_{source}_baseline_{kind}.joblib')
        inputs, labels = ca.multisource_eval.view(path)
        periods = ca.supervised.model_periods(inputs)
        tracking = inputs if view == 'clean' else ca.tc.corrupt(inputs, calibration, ca.ce.seed(path.stem, view))
        probability = ca.cb.predict_probabilities(tracking, models)
        decoded = ca.supervised.decode(probability, periods, float(inputs['fps']), ca.lt.COST)
        staged = target.with_suffix('.pending.npz')
        np.savez_compressed(staged, probability=probability, baseline=decoded,
            labels=labels, period=inputs['period'], timestamps=inputs['timestamps'],
            model_periods=periods, fps=inputs['fps'])
        staged.rename(target)
    return target.name


def main():
    spec = ca.check()
    receipt = ca.OUT/'cache_execution.json'
    if not receipt.exists():
        ca.ce.write_json(receipt, dict(created_utc=datetime.now(timezone.utc).isoformat(),
            declaration_sha256=ca.sha256(ca.DECLARATION), executor_sha256=ca.sha256(Path(__file__)),
            workers=4, reason='Independent per-cell feature generation; no parameter/protocol changes',
            scoring_performed=False), exclusive=True)
    else:
        import json
        prior = json.loads(receipt.read_text())
        if prior['executor_sha256'] != ca.sha256(Path(__file__)) or prior['declaration_sha256'] != ca.sha256(ca.DECLARATION):
            raise ValueError('Cache execution receipt changed')
    tasks = [(s, p, v, spec['calibration']) for s, split in ca.ce.dead_transfer.SPLITS.items()
             for p in split['eval'] for v in ca.ce.VIEWS if not (ca.OUT/f'{p.stem}_{v}.npz').exists()]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context('spawn')) as pool:
        for name in pool.map(prepare_cell, tasks):
            print('PREPARED', name, flush=True)
    ca.prepare()


if __name__ == '__main__':
    main()
