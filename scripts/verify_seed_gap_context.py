"""Replay probability ensembles and context outputs through the official scorer.

This independent read-only verifier does no training or organizer scoring.
"""
from pathlib import Path
import sys
import json
import numpy as np
import joblib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import seed_gap_context as study, supervised
from experiments.metrics import evaluate_tracking
from experiments.eval_set.protocol import sha256


def replay(inputs,labels,probabilities,row):
    if study.digest_array(probabilities) != row['probability_sha256']:
        raise ValueError('Probability hash mismatch')
    prediction=supervised.decode(probabilities,supervised.model_periods(inputs),float(inputs['fps']),3)
    if study.digest_array(prediction) != row['prediction_sha256']:
        raise ValueError('Prediction hash mismatch')
    score=evaluate_tracking(labels,prediction,inputs['period'],float(inputs['fps']),inputs['timestamps'])
    for key,actual in dict(spell_f1=score['spell_f1_mean'],macro_f1=score['macro_f1'],
        d_f1=score['per_class']['D']['f1'],n_gt_spells=score['n_gt_spells'],n_pred_spells=score['n_pred_spells']).items():
        if abs(actual-row[key]) > 1e-12: raise ValueError(f'Official replay mismatch: {key}')
    for key,value in row['spell_thresholds'].items():
        if abs(score['spell_f1'][float(key)]['f1']-value)>1e-12:
            raise ValueError('IoU replay mismatch')


def main():
    study.check()
    base=json.loads(study.RESULT.read_text())
    long_path=ROOT/'experiments/results/seed_gap_context_long.json'
    long=json.loads(long_path.read_text()) if long_path.exists() else None
    replayed=0
    for source,path,view,replicate in study.cells():
        cached=joblib.load(study.OUT/f'{path.stem}_{view}_{replicate}.joblib')
        with np.load(study.OUT/f'{path.stem}_{view}_{replicate}.npz') as saved:
            mean=.5*(saved['seed_2026']+saved['seed_2027'])
            if not np.array_equal(mean,saved['equal_probability_ensemble']):
                raise ValueError('Actual ensemble differs from full-vector average')
            for seed in (2026,2027):
                row=next(r for r in base['rows'] if (r['match'],r['view'],r['replicate'],r['recipe'])==
                    (path.stem,view,replicate,f'seed_{seed}'))
                if study.digest_array(saved[f'seed_{seed}'])!=row['probability_sha256']:
                    raise ValueError('Individual probability hash mismatch')
            row=next(r for r in base['rows'] if (r['match'],r['view'],r['replicate'],r['recipe'])==
                (path.stem,view,replicate,'equal_probability_ensemble'))
            replay(cached['inputs'],cached['labels'],mean,row);replayed+=1
        if long:
            for seed in (2026,2027):
                with np.load(study.OUT/f'{path.stem}_{view}_{replicate}_context15_{seed}.npz') as saved:
                    row=next(r for r in long['rows'] if (r['match'],r['view'],r['replicate'],r['recipe'])==
                        (path.stem,view,replicate,f'context15_{seed}'))
                    replay(cached['inputs'],cached['labels'],saved['probability'],row);replayed+=1
                    if study.digest_array(saved['prediction'])!=row['prediction_sha256']:
                        raise ValueError('Stored context prediction differs')
        print('VERIFIED',path.stem,view,replicate,flush=True)
    study.write(ROOT/'experiments/results/seed_gap_context_verification.json',dict(
        verifier_sha256=sha256(__file__),declaration_sha256=sha256(study.DECLARATION),
        base_report_sha256=sha256(study.RESULT),long_report_sha256=sha256(long_path) if long else None,
        exact_probability_ensemble_cells=15,individual_probability_hashes=30,
        independently_replayed_score_cells=replayed,archived_individual_parity_cells=base['parity_cells'],
        official_scorer_sha256=sha256(ROOT/'evaluation/possession_metrics.py'),
        organizer_labels_read=False,baseline_changed=False))


if __name__=='__main__': main()
