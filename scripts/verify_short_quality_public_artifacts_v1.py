"""Verify sealed models/CSV/probabilities and saved results; never re-read GT."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
from experiments import short_quality_public_v1 as study

p=study.p


def main():
    d,s=study.verify_seal();result=p.read(study.RESULT)
    assert study.STARTED.exists() and result['prediction_seal_sha256']==p.sha(study.SEAL)
    assert result['candidate_csv_sha256']==s['candidate_sha256']
    path=study.OUT/'native_probabilities.npz'
    with np.load(path) as z:
        values={k:z[k] for k in z.files}
    mean=np.mean([values['member_'+str(seed)] for seed in p.SEEDS],axis=0)
    np.testing.assert_array_equal(mean,values['ensemble'])
    prediction=p.decode(mean,values['model_period'],float(values['fps']))
    prediction[values['period']<=0]='X'
    np.testing.assert_array_equal(prediction,values['prediction'])
    from evaluation import possession_metrics as pm
    frame=pm.read_label_csv(ROOT/s['candidate_csv'])
    assert pm.validate_submission(frame,199319) is None
    np.testing.assert_array_equal(prediction,frame['label'].to_numpy())
    for seed in p.SEEDS:
        path=study.OUT/'models'/f'pooled_{seed}.joblib';model=joblib.load(path)
        r=p.read(path.with_suffix('.json'));teacher=joblib.load(study.teacher_path(seed))
        assert model['step']==2048 and model['training']['seed']==seed
        for field in ('median','scale'):np.testing.assert_array_equal(model[field],teacher[field])
        assert r['old_matches']==sorted(i['match'] for i in d['train_plan']['train'])
        assert len(r['config']['soccernet_games'])==45
        assert r['sampling']['counts']['old']['draws']==r['sampling']['counts']['new']['draws']==32768
    assert result['adopted_baseline_reused']==d['baseline_scores_reused']
    assert result['new_unique_csvs_scored'] in (0,1)
    for field in ('spell_f1_mean','macro_f1','d_f1'):
        assert result['delta_vs_adopted'][field]==result['candidate'][field]-d['baseline_scores_reused'][field]
    receipt=dict(passed=True,models=2,full_rows=199319,native_mean_decode_and_csv_exact=True,
        frozen_training_and_teacher_norms_checked=True,public_reference_read_by_verifier=False,
        public_metrics_recomputed=False,result_sha256=p.sha(study.RESULT),verifier_sha256=p.sha(Path(__file__)))
    study.write(p.RESULTS/f'{study.STUDY}_artifact_verification.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
