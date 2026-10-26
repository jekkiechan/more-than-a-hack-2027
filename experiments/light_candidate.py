"""Generate the externally selected light-corruption candidate without scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import (challenge, corruption_experiment as ce, current_baseline,
                         light_corruption as light, supervised)
from experiments.eval_set.protocol import sha256


def generate(out, inputs_path, destination):
    light.check(out)
    selection_path = out/'selection.json'
    selection = json.loads(selection_path.read_text())
    for name, expected in selection['pooled_checkpoints_sha256'].items():
        if sha256(out/name) != expected: raise ValueError('Externally frozen checkpoint changed')
    arm = selection['selected']['model']
    team, d = light.ARMS[arm]
    models = current_baseline.load_models()
    if team == 'light': models['attribution'] = ce.load_model(out, out/'team_light.joblib')
    if d == 'light':
        for kind in ('bigru', 'ordered_hgb'):
            models[kind] = ce.load_model(out, out/f'pooled_{kind}.joblib')
    if destination.exists(): raise ValueError('Refusing to overwrite a generated candidate')
    inputs = current_baseline.load_inputs(inputs_path)
    probability = current_baseline.predict_probabilities(inputs, models)
    prediction = supervised.decode(probability, supervised.model_periods(inputs),
                                   float(inputs['fps']), light.lt.COST)
    prediction[np.asarray(inputs['period']) <= 0] = 'X'
    challenge.write_submission(prediction, destination)
    ce.write_json(destination.with_suffix('.json'), dict(
        arm=arm, selection_sha256=sha256(selection_path), input_sha256=sha256(inputs_path),
        output_sha256=sha256(destination), exporter_sha256=sha256(__file__),
        baseline_manifest_sha256=sha256(current_baseline.MANIFEST),
        reference_labels_read=False, organizer_scored=False, baseline_replaced=False), exclusive=True)
    print('Generated and validated', arm, 'without reference labels', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=light.DEFAULT)
    p.add_argument('--inputs', type=Path, default=light.ROOT/'data/challenge/inputs.npz')
    p.add_argument('--output', type=Path, default=light.DEFAULT/'candidate.csv')
    args = p.parse_args()
    generate(args.out, args.inputs, args.output)


if __name__ == '__main__': main()
