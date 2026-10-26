"""Generate the externally selected light-corruption candidate without scoring."""
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import (challenge, corruption_experiment as ce, current_baseline,
                         light_corruption as light, supervised)
from experiments.eval_set.protocol import sha256


def diagnostic_selection(report, working):
    eligible = [r for r in report['summaries'] if r['evaluation'] == 'provider_holdout_D'
                and r['view'] == 'clean' and r['model'] != 'baseline']
    best = max(eligible, key=lambda r: (r['spell_f1'], r['macro_f1']))
    return dict(working, selected=best, diagnostic_only=True,
        working_model=working['selected']['model'],
        rule='Best new experimental arm by external clean provider-held-out spell F1, macro for exact ties; reporting only, working selection unchanged')


def generate(out, inputs_path, destination, selection_path=None):
    light.check(out)
    selection_path = selection_path or out/'selection.json'
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
        arm=arm, selection_file=selection_path.name, selection_sha256=sha256(selection_path),
        diagnostic_only=selection.get('diagnostic_only', False), input_sha256=sha256(inputs_path),
        output_sha256=sha256(destination), exporter_sha256=sha256(__file__),
        baseline_manifest_sha256=sha256(current_baseline.MANIFEST),
        reference_labels_read=False, organizer_scored=False, baseline_replaced=False), exclusive=True)
    print('Generated and validated', arm, 'without reference labels', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=light.DEFAULT)
    p.add_argument('--inputs', type=Path, default=light.ROOT/'data/challenge/inputs.npz')
    p.add_argument('--output', type=Path)
    p.add_argument('--diagnostic-best-new', action='store_true',
                   help='Export best new arm for the user-requested diagnostic; preserve working selection')
    args = p.parse_args()
    selection_path = args.out/'selection.json'
    if args.diagnostic_best_new:
        if (args.out/'public_check_started.json').exists():
            raise ValueError('Competition check started; no new diagnostic selection')
        working = json.loads(selection_path.read_text())
        report_path = light.ROOT/'experiments/results/light_corruption_development.json'
        selected = diagnostic_selection(json.loads(report_path.read_text()), working)
        selected['working_selection_sha256'] = sha256(selection_path)
        selected['external_report_sha256'] = sha256(report_path)
        selection_path = args.out/'diagnostic_selection.json'
        ce.write_json(selection_path, selected, exclusive=True)
    destination = args.output or args.out/('experimental_candidate.csv' if args.diagnostic_best_new else 'candidate.csv')
    generate(args.out, args.inputs, destination, selection_path)


if __name__ == '__main__': main()
