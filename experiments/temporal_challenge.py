"""Generate a frozen temporal-D challenge candidate and one public pipeline check.

The check cannot select or change a model. Organizer inputs and per-frame outputs
remain under ignored data/. A complete external report/selection is required.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from experiments import challenge, learned_temporal as lt, refine_features, robust, supervised


def generate(out, inputs_path):
    selection = json.loads((out/'selection.json').read_text())
    kind = selection['selected_model']
    model_path = out/f'pooled_{kind}.joblib'
    attribution_path = out/'attribution.joblib'
    if lt.digest(model_path) != selection['selected_model_sha256'] or lt.digest(attribution_path) != selection['attribution_model_sha256']:
        raise ValueError('Frozen model hashes changed')
    if lt.digest(out/'predeclared.json') != selection['predeclared_spec_sha256']:
        raise ValueError('Frozen experiment spec changed')
    target = out/'challenge'
    target.mkdir(exist_ok=True)
    if (target/'public_check_started.json').exists():
        raise ValueError('Public check started: candidate regeneration is locked')
    # Missing inputs fail before any organizer GT or prediction is accessed.
    if inputs_path.exists():
        inputs = robust.tracking_inputs(supervised.load(inputs_path))
    else:
        inputs = robust.tracking_inputs(challenge.load_challenge(inputs_path))
    indices = lt.sampled_indices(inputs)
    x, names = lt.player_features(inputs)
    record = dict(x=x[indices], names=names, period=supervised.model_periods(inputs)[indices],
                  indices=indices, native_n=len(x))
    dead = lt.upsample(lt.predict_sampled(record, joblib.load(model_path)), record, supervised.model_periods(inputs))
    base = refine_features.predict_probabilities(inputs, joblib.load(attribution_path))
    for label, d in (('selected', dead), ('hgb_baseline', lt.upsample(lt.predict_sampled(record,
                          joblib.load(out/'pooled_hgb.joblib')), record, supervised.model_periods(inputs)))):
        p = lt.refine_domain.combine(base, d, 0.)
        pred = supervised.decode(p, supervised.model_periods(inputs), float(inputs['fps']), lt.COST)
        pred[np.asarray(inputs['period']) <= 0] = 'X'
        challenge.write_submission(pred, target/f'{label}.csv')
    frozen = dict(selection=selection, input_sha256=lt.digest(inputs_path),
                  csv_sha256={name: lt.digest(target/f'{name}.csv') for name in ('selected', 'hgb_baseline')},
                  created_utc=datetime.now(timezone.utc).isoformat())
    (target/'frozen.json').write_text(json.dumps(frozen, indent=2)+'\n')
    print('Generated and validated both full-length candidates; no organizer labels read')


def check(out, previous_csv):
    target = out/'challenge'
    frozen = json.loads((target/'frozen.json').read_text())
    for name, expected in frozen['csv_sha256'].items():
        if lt.digest(target/f'{name}.csv') != expected:
            raise ValueError('Frozen predictions changed')
    paths = {'selected': target/'selected.csv', 'hgb_baseline': target/'hgb_baseline.csv'}
    if previous_csv is not None:
        if not previous_csv.is_file():
            raise ValueError('Previous candidate is missing')
        paths['previous_candidate'] = previous_csv
    marker = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                  candidates={name: lt.digest(path) for name, path in paths.items()})
    # Exclusive marker prevents accidental public-window iteration/retry.
    with (target/'public_check_started.json').open('x') as handle:
        json.dump(marker, handle, indent=2)
    from evaluation import possession_metrics as pm
    gt = pd.read_csv(lt.ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv')
    result = {}
    for name, path in paths.items():
        predicted = pd.read_csv(path)
        error = pm.validate_submission(predicted, len(gt))
        if error:
            raise ValueError(error)
        score = pm.evaluate(pm.label_codes(gt.label.to_numpy()), pm.label_codes(predicted.label.to_numpy()))
        result[name] = {k: score[k] for k in ('macro_f1', 'spell_f1_mean', 'per_class', 'n_gt_spells', 'n_pred_spells')}
    report = dict(scope='One descriptive five-minute pipeline check; no model/threshold selection or full-match claims',
                  frozen=frozen, scores=result)
    (target/'public_check.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'check'))
    parser.add_argument('--out', type=Path, default=lt.ROOT/'data/experiments/learned_temporal')
    parser.add_argument('--inputs', type=Path, default=lt.ROOT/'data/challenge/inputs.npz')
    parser.add_argument('--previous-csv', type=Path)
    args = parser.parse_args()
    if args.action == 'generate':
        generate(args.out, args.inputs)
    else:
        check(args.out, args.previous_csv)


if __name__ == '__main__':
    main()
