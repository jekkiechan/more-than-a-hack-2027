"""One requested public diagnostic of the fixed expansion candidate.

External decisions are sealed first. No competition label is read during data
preparation, fitting, selection, or prediction. Historical gates are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from experiments import skillcorner_expansion as study

ROOT, DATA, OUT = study.ROOT, study.DATA, study.OUT
SEAL = OUT / 'competition_prediction_seal.json'
STARTED = OUT / 'competition_check_started.json'
RESULT = OUT / 'competition_check_result.json'
CSV = OUT / 'competition_candidate.csv'
INPUTS = DATA / 'challenge/inputs.npz'
BASELINE = ROOT / 'experiments/results/combined_objective_public_check.json'
EXPECTED_INPUT_SHA256 = '7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3'


def exclusive(path, value):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())


def verify_seal(spec):
    study.check()
    study.seal()
    if (study.sha(CSV) != spec['prediction_sha256']
            or study.sha(INPUTS) != spec['input_sha256']
            or study.sha(OUT / 'decision.json') != spec['external_decision_sha256']
            or study.sha(BASELINE) != spec['saved_baseline_report_sha256']
            or study.source_hashes() != spec['source_sha256']):
        raise ValueError('Competition prediction seal changed')
    for path, expected in spec['checkpoints'].items():
        if study.sha(DATA / path) != expected:
            raise ValueError('Competition checkpoint changed')


def predict():
    study.check()
    study.seal()
    decision = study.read(OUT / 'decision.json')
    if STARTED.exists() or RESULT.exists():
        raise ValueError('Public diagnostic gate is already closed')
    if SEAL.exists():
        verify_seal(study.read(SEAL))
        return
    if study.sha(INPUTS) != EXPECTED_INPUT_SHA256:
        raise ValueError('Unexpected competition tracking input')
    manifest = dict(model='joint_gru_combined_expanded_candidate',
        model_directory=str(OUT),
        models_sha256={f'pooled_{seed}.joblib': study.sha(study.candidate_path('pooled', seed)) for seed in study.SEEDS},
        architecture=dict(kind='joint_gru_ensemble', members={f'pooled_{seed}': .5 for seed in study.SEEDS}),
        status='Fixed diagnostic candidate; public results never change the external decision')
    manifest_path = OUT / 'competition_candidate_manifest.json'
    study.write(manifest_path, manifest)
    command = [sys.executable, '-m', 'experiments.current_baseline', 'predict',
               '--inputs', str(INPUTS), '--manifest', str(manifest_path), '--output', str(CSV)]
    subprocess.run(command, cwd=ROOT, check=True)
    replay = OUT / 'competition_candidate_replay.csv'
    subprocess.run(command[:-1] + [str(replay)], cwd=ROOT, check=True)
    if study.sha(replay) != study.sha(CSV):
        raise ValueError('Normal prediction entrypoint did not reproduce its CSV')
    from evaluation import possession_metrics as pm
    import numpy as np
    frame = pm.read_label_csv(CSV)
    error = pm.validate_submission(frame, 199319)
    if error:
        raise ValueError(error)
    with np.load(INPUTS, allow_pickle=False) as archive:
        if not np.array_equal(frame['label'].to_numpy() == 'X', archive['period'] <= 0):
            raise ValueError('Candidate X mask does not match input period mask')
    historical = study.read(BASELINE)
    spec = dict(study_id='skillcorner_expansion_public_check_v1',
        created_utc=datetime.now(timezone.utc).isoformat(),
        explicit_user_request='test the behavior first; if useful add without approval; also see if it helps competition spell F1',
        external_decision_sha256=study.sha(OUT / 'decision.json'),
        external_adopt_dataset=decision['adopt_dataset'],
        declaration_sha256=study.sha(OUT / 'predeclared.json'),
        prediction_sha256=study.sha(CSV), replay_sha256=study.sha(replay),
        input_sha256=study.sha(INPUTS), rows=199319, format_valid=True,
        checkpoints={str(study.candidate_path('pooled', seed).relative_to(DATA)): study.sha(study.candidate_path('pooled', seed)) for seed in study.SEEDS},
        source_sha256=study.source_hashes(), saved_baseline_report_sha256=study.sha(BASELINE),
        expected_reference_sha256=historical['organizer_gt_sha256'],
        reference_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        organizer_reference_labels_read=False,
        interpretation='One fixed candidate, compared with saved baseline; no public-driven selection, tuning, seed changes or automatic promotion')
    exclusive(SEAL, spec)
    print('SEALED_COMPETITION_PREDICTION', spec['prediction_sha256'], flush=True)


def score():
    if RESULT.exists():
        return study.read(RESULT)
    spec = study.read(SEAL)
    verify_seal(spec)
    # Exclusive, never-reset marker is created before fetching/opening labels.
    exclusive(STARTED, dict(started_utc=datetime.now(timezone.utc).isoformat(), seal_sha256=study.sha(SEAL)))
    reference = OUT / 'public_reference.csv'
    result = subprocess.run(['git', 'show', f'{spec["reference_git_commit"]}:evaluation/gt/BAR-ATH_possession_gt_public.csv'],
                            cwd=ROOT, check=True, stdout=subprocess.PIPE)
    if hashlib.sha256(result.stdout).hexdigest() != spec['expected_reference_sha256']:
        raise ValueError('Published reference does not match the historical reference hash')
    reference.write_bytes(result.stdout)
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    truth = pm.label_codes(pm.read_label_csv(reference))
    predicted = pm.read_label_csv(CSV)
    scored = pm.evaluate(truth, pm.label_codes(predicted), bootstrap_n=1000)
    summary = _score_summary(scored)
    if (summary['graded_frames'], summary['ground_truth_spells']) != (7500, 17):
        raise ValueError('Unexpected public diagnostic population')
    baseline = study.read(BASELINE)['scores']['combined_objective_ensemble']
    value = dict(study_id='skillcorner_expansion_public_check_v1',
        candidate=summary, saved_baseline=baseline,
        deltas={k: summary[k]-baseline[k] for k in ('spell_f1_mean', 'macro_f1', 'd_f1')},
        seal_sha256=study.sha(SEAL), external_decision_sha256=spec['external_decision_sha256'],
        external_adopt_dataset=spec['external_adopt_dataset'],
        organizer_reference_labels_read=True, public_used_for_selection=False,
        interpretation='Already-exposed five-minute diagnostic, 17 possessions. Full-match performance remains unknown. No decisions or model changes follow this score.')
    exclusive(RESULT, value)
    exclusive(ROOT / 'experiments/results/skillcorner_expansion_public_check.json', value)
    print(value['deltas'], flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('predict', 'score'))
    args = parser.parse_args()
    globals()[args.action]()


if __name__ == '__main__':
    main()
