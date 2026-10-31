"""One fixed, user-requested Smart Tagging diagnostic; no fitting or tuning.

freeze and predict never read reference labels. check consumes a distinct,
exclusive gate after verifying the full-length prediction seal. Existing gates
and the adopted baseline stay unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/experiments/smart_tagging_validation_v1'
REPORT = ROOT / 'experiments/results/smart_tagging_public_check.json'
TEAM_MAP = {'Barcelona': 'H', 'Atletic Club': 'A'}
INPUT_SHA = '7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3'
BASELINE_CSV_SHA = 'a9beb9ddee36182f926e4d693f648c19f4a19c55afd1653ea93e19f3dd32634b'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def source_files():
    paths = {p for folder in ('experiments', 'starter', 'evaluation')
             for p in (ROOT/folder).glob('*.py')}
    paths.add(ROOT/'games.py')
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def require_open(out):
    if ((out/'public_check_started.json').exists() or REPORT.exists()
            or REPORT.with_name('smart_tagging_public_check_started.json').exists()):
        raise ValueError('Smart Tagging diagnostic gate is closed; do not repeat')


def verify(mapping):
    for relative, expected in mapping.items():
        path = Path(relative)
        if path.is_absolute() or '..' in path.parts or sha(ROOT/path) != expected:
            raise ValueError('Frozen artifact differs: '+relative)


def read_spec(out):
    spec = json.loads((out/'declaration.json').read_text())
    if source_files() != spec['source_sha256']:
        raise ValueError('Frozen source changed')
    verify(spec['files_sha256'])
    return spec


def freeze(out):
    require_open(out)
    paths = [ROOT/'data/challenge/inputs.npz',
             ROOT/'data/tactical/BAR-ATH_tactical_smart_tagging.xml',
             ROOT/'experiments/baseline_manifest.json',
             ROOT/'experiments/model_decision_policy.json',
             ROOT/'experiments/results/combined_objective_public_check.json']
    manifest = json.loads(paths[2].read_text())
    if manifest['package_version'] != 'joint_gru_combined_ensemble_v1':
        raise ValueError('Unexpected adopted baseline')
    if sha(paths[0]) != INPUT_SHA:
        raise ValueError('Unexpected tracking cache')
    for name, expected in manifest['models_sha256'].items():
        path = ROOT/manifest['model_directory']/name
        if sha(path) != expected:
            raise ValueError('Model hash differs')
        paths.append(path)
    spec = dict(study_id='smart_tagging_validation_v1', created_utc=now(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        user_request="maybe we can see how it impacts our public set. i know we can't tune on it but that's the only way i can see us validating it",
        source_sha256=source_files(), files_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},
        team_to_side=TEAM_MAP, missing_ball_only=True, max_conditional_odds_multiplier=2,
        taper_seconds=1, decoder_switch_cost=3, candidates=['smart_tagging_missing_ball'],
        expected_baseline_csv_sha256=BASELINE_CSV_SHA,
        fitting=False, new_models=0, selection_or_promotion=False,
        hypotheses='Soft phase-team evidence may improve missing-ball H/A intervals; it may instead fragment or merge possessions.',
        budget='One fixed candidate, no sweeps, one exclusive public diagnostic. No changes after scoring.',
        primary='Official spell_f1_mean, macro_f1 exact-tie secondary, 70/30 auxiliary only',
        public_scope='Descriptive previously exposed 7500-frame / 17-possession check; not independent confirmation or full-match performance.',
        reference_labels_read=False, existing_gates_unchanged=True)
    write(out/'declaration.json', spec)
    print('FROZEN one fixed Smart Tagging candidate; no reference labels read')


def predict(out):
    require_open(out)
    spec = read_spec(out)
    from experiments import challenge, current_baseline, supervised
    from experiments.smart_tagging_prior import apply_smart_tagging_prior
    from starter.load_smart_tagging import load
    inputs = current_baseline.load_inputs(ROOT/'data/challenge/inputs.npz')
    manifest = json.loads((ROOT/'experiments/baseline_manifest.json').read_text())
    models = current_baseline.load_models()
    probability = current_baseline.predict_probabilities(inputs, models, manifest)
    adjusted = apply_smart_tagging_prior(probability, load(), TEAM_MAP,
        float(inputs['fps']), inputs['period'], inputs['ball_observed'],
        timestamps=inputs['timestamps'], missing_ball_only=True)
    if probability[:, 2].tobytes() != adjusted[:, 2].tobytes():
        raise ValueError('Tag adapter changed native P(D)')
    artifacts = {}
    for name, p in [('baseline', probability), ('smart_tagging_missing_ball', adjusted)]:
        labels = supervised.decode(p, supervised.model_periods(inputs), float(inputs['fps']), 3)
        labels[inputs['period'] <= 0] = 'X'
        csv = out/(name+'.csv')
        if csv.exists():
            raise ValueError('Refusing to overwrite prediction')
        challenge.write_submission(labels, csv)
        if name == 'baseline' and sha(csv) != spec['expected_baseline_csv_sha256']:
            raise ValueError('Restored baseline does not reproduce the saved scored CSV')
        npz = out/(name+'.npz')
        with npz.open('xb') as handle:
            np.savez_compressed(handle, probabilities=p, prediction=labels)
        artifacts.update({str(path.relative_to(ROOT)):sha(path) for path in (csv,npz)})
    write(out/'prediction_seal.json', dict(sealed_utc=now(),
        declaration_sha256=sha(out/'declaration.json'), files_sha256=artifacts,
        rows=199319, reference_labels_read=False, native_p_d_bitwise_equal=True,
        baseline_reproduces_saved_scored_csv=True, full_length_csvs_validated=True))
    print('Both full-length predictions sealed; baseline replay exact; no reference labels read')


def check(out):
    require_open(out)
    spec = read_spec(out)
    seal = json.loads((out/'prediction_seal.json').read_text())
    expected_paths = {str((out/(name+suffix)).relative_to(ROOT))
                      for name in ('baseline','smart_tagging_missing_ball')
                      for suffix in ('.csv','.npz')}
    if set(seal['files_sha256']) != expected_paths or seal.get('rows') != 199319:
        raise ValueError('Prediction artifact seal is incomplete')
    if seal['declaration_sha256'] != sha(out/'declaration.json'):
        raise ValueError('Declaration seal differs')
    verify(seal['files_sha256'])
    if (seal['reference_labels_read'] or not seal['baseline_reproduces_saved_scored_csv']
            or not seal.get('full_length_csvs_validated') or not seal.get('native_p_d_bitwise_equal')):
        raise ValueError('Prediction seal invalid')
    # Consume gate BEFORE any reference access, including if scoring fails.
    gate = dict(started_utc=now(),
        declaration_sha256=sha(out/'declaration.json'),
        prediction_seal_sha256=sha(out/'prediction_seal.json'), candidates=spec['candidates'])
    # This fixed marker also blocks repeat checks through a different --out.
    write(REPORT.with_name('smart_tagging_public_check_started.json'), gate)
    write(out/'public_check_started.json', gate)
    import pandas as pd
    from evaluation import possession_metrics as pm
    gt_path = ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    candidate_path = out/'smart_tagging_missing_ball.csv'
    truth = pd.read_csv(gt_path)
    candidate = pd.read_csv(candidate_path)
    for frame in (truth,candidate):
        if pm.validate_submission(frame,199319) is not None:
            raise ValueError('Invalid full-length frame file')
    result = pm.evaluate(pm.label_codes(truth['label'].to_numpy()),
                         pm.label_codes(candidate['label'].to_numpy()), bootstrap_n=1000)
    saved = json.loads((ROOT/'experiments/results/combined_objective_public_check.json').read_text())
    baseline = saved['scores']['combined_objective_ensemble']
    report = dict(study_id=spec['study_id'], completed_utc=now(),
        declaration_sha256=sha(out/'declaration.json'), prediction_seal_sha256=sha(out/'prediction_seal.json'),
        gate_sha256=sha(out/'public_check_started.json'), public_gt_sha256=sha(gt_path),
        candidate=result, saved_baseline=baseline,
        saved_baseline_source='experiments/results/combined_objective_public_check.json',
        candidate_minus_baseline={
            'spell_f1_mean':result['spell_f1_mean']-baseline['spell_f1_mean'],
            'macro_f1':result['macro_f1']-baseline['macro_f1'],
            'predicted_spells':result['n_pred_spells']-baseline['predicted_spells']},
        auxiliary_70_30={'baseline':.7*baseline['spell_f1_mean']+.3*baseline['macro_f1'],
                        'candidate':.7*result['spell_f1_mean']+.3*result['macro_f1']},
        baseline_rescored=False, baseline_replaced=False, tuning_after_check=False,
        scope=spec['public_scope'], files_sha256=seal['files_sha256'])
    write(REPORT, report)
    write(out/'public_check_completed.json', dict(report_sha256=sha(REPORT), completed_utc=now()))
    print('Completed one fixed diagnostic; saved aggregate:',REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','predict','check'))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args()
    {'freeze':freeze,'predict':predict,'check':check}[args.action](args.out.resolve())


if __name__ == '__main__':
    main()
