"""Seal and format-check a label-free combined-objective deployment prediction.

Run only after the two pooled deployment checkpoints have been verified. This
also replays the preceding ensemble and requires its existing CSV byte digest.
No reference labels or scoring entrypoint are loaded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import current_baseline as baseline, challenge, learned_temporal as lt, supervised
from experiments.eval_set.protocol import sha256
from evaluation import possession_metrics as pm

ROOT = baseline.ROOT
OUT = ROOT / 'data/experiments/current_baseline/joint_gru_combined_ensemble.csv'
INPUT_SHA = '7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3'
REPLAY_SHA = '3635b160d7397ca845d86fc2da6331b49df7b3eabf05c6a8eeca9d9d472a57b9'


def preserve(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f'Existing artifact differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(payload)


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_prediction(inputs, probability, path):
    labels = supervised.decode(probability, supervised.model_periods(inputs),
                               float(inputs['fps']), lt.COST)
    labels[np.asarray(inputs['period']) <= 0] = 'X'
    if path.exists():
        raise ValueError(f'Prediction already exists: {path}')
    challenge.write_submission(labels, path)
    table = pm.read_label_csv(str(path))
    reason = pm.validate_submission(table, 199319)
    if reason:
        raise ValueError(reason)
    return sha256(path)


def predict(seed2026, seed2027):
    from experiments import combined_objective_deployment as deploy
    deploy.check()
    paths = {f'joint_gru_{seed}.joblib': Path(path).resolve()
             for seed, path in ((2026, seed2026), (2027, seed2027))}
    for seed in (2026, 2027):
        deploy.verify_model(seed, paths[f'joint_gru_{seed}.joblib'])
    inputs_path = ROOT / 'data/challenge/inputs.npz'
    if sha256(inputs_path) != INPUT_SHA:
        raise ValueError('Organizer tracking input seal changed')
    manifest_path = ROOT / 'experiments/baseline_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['package_version'] != 'joint_gru_seed_ensemble_v1':
        raise ValueError('Replay must precede replacement of the current baseline')
    sources = [Path(__file__), Path(baseline.__file__), Path(deploy.__file__),
               Path(lt.__file__), Path(supervised.__file__)]
    sources.extend(ROOT / rel for rel in (
        'experiments/factorized_had.py', 'experiments/live_gap_experiment.py',
        'experiments/robust.py', 'experiments/challenge.py',
        'experiments/refine_features.py', 'experiments/refine_domain.py',
        'experiments/stress.py', 'evaluation/possession_metrics.py',
        'games.py', 'starter/baseline.py'))
    declaration = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Label-free deployment prediction and preceding-baseline replay; no scoring',
        input_sha256=INPUT_SHA, current_manifest_sha256=sha256(manifest_path),
        model_sha256={name: sha256(path) for name, path in paths.items()},
        sources_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sources},
        organizer_reference_labels_read=False, organizer_scored=False)
    preserve(OUT.with_suffix('.declaration.json'),
             (json.dumps(declaration, indent=2) + '\n').encode())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    inputs = baseline.load_inputs(inputs_path)
    controls = baseline.load_models()
    print('REPLAYING_PREVIOUS_BASELINE', flush=True)
    previous = baseline.joint_gru_member_probabilities(inputs, controls, manifest)
    replay = OUT.with_name('preceding_seed_ensemble_replay.csv')
    replay_digest = write_prediction(inputs, .5 * sum(previous.values()), replay)
    if replay_digest != REPLAY_SHA:
        raise ValueError('Preceding baseline CSV does not reproduce byte-for-byte')
    models = {name.removesuffix('.joblib'): joblib.load(path) for name, path in paths.items()}
    if any(model.get('objective') != 'combined' for model in models.values()):
        raise ValueError('Expected combined-objective deployment checkpoints')
    print('PREDICTING_COMBINED_OBJECTIVE_ENSEMBLE', flush=True)
    pieces = baseline.joint_gru_member_probabilities(inputs, models, manifest)
    probability = .5 * sum(pieces.values())
    output_digest = write_prediction(inputs, probability, OUT)
    receipt = dict(input_sha256=INPUT_SHA, output_sha256=output_digest, rows=len(probability),
        model_sha256=declaration['model_sha256'],
        member_probability_sha256={name: array_sha(p) for name, p in pieces.items()},
        blended_probability_sha256=array_sha(probability),
        current_baseline_replay_sha256=replay_digest,
        current_baseline_replay_path=str(replay.relative_to(ROOT)),
        format_validation=dict(valid=True, rows=199319),
        organizer_reference_labels_read=False, organizer_scored=False, organizer_score=None,
        declaration_sha256=sha256(OUT.with_suffix('.declaration.json')))
    preserve(OUT.with_suffix('.json'), (json.dumps(receipt, indent=2) + '\n').encode())
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed2026', type=Path, required=True)
    parser.add_argument('--seed2027', type=Path, required=True)
    args = parser.parse_args()
    predict(args.seed2026, args.seed2027)
