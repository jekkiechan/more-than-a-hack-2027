"""User-requested descriptive check of the two already frozen live-only heads.

Generate without reference labels, then use the existing one-shot public-check
guard. Historical baseline scores are reused after exact CSV reproduction.
This module cannot train, select, promote, or reset previous evaluation gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import (challenge, current_baseline, live_attribution as la,
                         refine_features, supervised, temporal_challenge)
from experiments.eval_set.protocol import sha256

ROOT = la.ROOT
HISTORY = ROOT/'experiments/results/current_baseline_public_check.json'
DEVELOPMENT = ROOT/'experiments/results/live_attribution_development.json'
REPORT = ROOT/'experiments/results/live_attribution_public_check.json'
DEFAULT_OUT = ROOT/'data/experiments/live_attribution/organizer_check'
NAMES = la.CANDIDATES[1:]
SCOPE = ('One user-requested descriptive public-five-minute pipeline check of the two frozen live-only heads; '
         'no tuning, model selection, baseline promotion or full-match accuracy claim; earlier gates preserved')


def read(path):
    return json.loads(Path(path).read_text())


def frozen_models():
    report = read(DEVELOPMENT)
    audit = read(DEVELOPMENT.with_name('live_attribution_verification.json'))
    if sha256(DEVELOPMENT) != audit['report_sha256'] or len(report['rows']) != 30:
        raise ValueError('External development result is not the completed verified experiment')
    for path, expected in report['spec']['inputs_and_source_sha256'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f'Frozen external input/source/checkpoint changed: {path}')
    directory = ROOT/'data/experiments/live_attribution'
    models, hashes = {}, {}
    for name in NAMES:
        receipt = read(directory/f'{name}_frozen.json')
        path = directory/f'{name}.joblib'
        if (receipt != report['model_receipts'][name]
                or sha256(path) != receipt['model_sha256']
                or sha256(directory/'predeclared.json') != receipt['spec_sha256']):
            raise ValueError(f'Externally frozen model/spec changed: {name}')
        models[name] = joblib.load(path)
        hashes[name] = sha256(path)
    return models, hashes


def generate(out, inputs_path):
    target = out/'challenge'
    target.mkdir(parents=True, exist_ok=True)
    if any((target/name).exists() for name in ('public_check_started.json', 'frozen.json', 'public_predeclared.json')):
        raise ValueError('Check already declared/generated; inspect existing artifacts rather than regenerate')
    if not inputs_path.is_file():
        raise FileNotFoundError(inputs_path)
    models, hashes = frozen_models()
    baseline = current_baseline.load_models()
    inputs = current_baseline.load_inputs(inputs_path)
    historical = read(HISTORY)
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in
               (Path(__file__), Path(temporal_challenge.__file__),
                ROOT/'evaluation/possession_metrics.py', DEVELOPMENT, HISTORY,
                current_baseline.MANIFEST)}
    declaration = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope=SCOPE,
        user_request='check with organizer eval too', candidates=list(NAMES),
        model_sha256=hashes, input_sha256=sha256(inputs_path), source_sha256=sources,
        frozen_external_spec_sha256=sha256(ROOT/'data/experiments/live_attribution/predeclared.json'),
        decoder_cost=la.lt.COST, D_members=dict(bigru=.5, ordered_hgb=.5),
        baseline_control_expected_sha256=historical['frozen']['candidate_sha256'])
    la.write_new(target/'public_predeclared.json', declaration)
    dead = current_baseline.dead_probability(inputs, baseline)
    np.save(target/'shared_dead.npy', dead, allow_pickle=False)
    print('Generated shared frozen GRU/HGB D probabilities without reference labels', flush=True)
    x, names = refine_features.features(inputs)
    print('Generated shared tracking-only H/A features without reference labels', flush=True)
    hashes_csv = {}
    for name, model in dict(baseline=baseline['attribution'], **models).items():
        team = refine_features.probabilities_from_features(x, names, model)
        probabilities = la.lt.refine_domain.combine(team, dead, 0.)
        labels = supervised.decode(probabilities, supervised.model_periods(inputs), float(inputs['fps']), la.lt.COST)
        labels[np.asarray(inputs['period']) <= 0] = 'X'
        path = target/f'{name}.csv'
        if path.exists():
            raise ValueError('Refusing to overwrite a generated organizer candidate')
        challenge.write_submission(labels, path)
        hashes_csv[name] = sha256(path)
        print(f'Generated and format-validated {name}: {len(labels)} frames', flush=True)
    if hashes_csv.pop('baseline') != historical['frozen']['candidate_sha256']:
        raise ValueError('Baseline CSV differs from its historical check; stop before reading public labels')
    la.write_new(target/'frozen.json', dict(declaration, csv_sha256=hashes_csv,
        baseline_control_sha256=sha256(target/'baseline.csv'),
        shared_dead_sha256=sha256(target/'shared_dead.npy'),
        public_predeclared_sha256=sha256(target/'public_predeclared.json')))
    print('All candidates frozen; baseline CSV reproduces history exactly; no reference labels read', flush=True)


def check(out):
    if REPORT.exists():
        raise ValueError('Organizer check report already exists; do not repeat the check')
    target = out/'challenge'
    frozen = read(target/'frozen.json')
    if set(frozen['csv_sha256']) != set(NAMES):
        raise ValueError('The full declared pair of frozen candidates is required')
    for path, expected in frozen['source_sha256'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError(f'Generation source/report changed: {path}')
    for path, expected in ((target/'public_predeclared.json', frozen['public_predeclared_sha256']),
                           (target/'baseline.csv', frozen['baseline_control_sha256']),
                           (target/'shared_dead.npy', frozen['shared_dead_sha256'])):
        if sha256(path) != expected:
            raise ValueError('Frozen organizer control/probability/declaration changed')
    # The existing implementation verifies CSV hashes and creates an exclusive
    # public_check_started.json before opening the public reference CSV.
    temporal_challenge.check(out, None)
    result = read(target/'public_check.json')
    historical = read(HISTORY)
    gt_hash = sha256(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv')
    if gt_hash != historical['organizer_gt_sha256']:
        raise ValueError('Public reference differs from the historical baseline check')
    scores = dict(baseline=historical['scores']['gru_hgb'], **result['scores'])
    columns = ('macro_f1', 'spell_f1_mean', 'per_class', 'n_gt_spells', 'n_pred_spells', 'spell_f1',
               'accuracy', 'team_acc_gt_alive', 'bootstrap_n', 'bootstrap_low', 'bootstrap_high',
               'spell_bootstrap_low', 'spell_bootstrap_high')
    compact = {name: {key: score[key] for key in columns if key in score} for name, score in scores.items()}
    for score in compact.values():
        score['weighted_70_30'] = .7*score['spell_f1_mean']+.3*score['macro_f1']
    report = dict(scope=SCOPE, frozen=frozen, scores=compact,
        organizer_gt_sha256=gt_hash, graded_frames=historical['scores']['gru_hgb']['n_graded'],
        baseline_scores='Reused historical check after exact full-length baseline CSV hash reproduction',
        full_match_predictions_format_valid=True, full_match_accuracy_measured=False,
        baseline_changed=False, historical_public_gates_preserved=True,
        raw_private_report_sha256=sha256(target/'public_check.json'))
    la.write_new(REPORT, report)
    for name, score in compact.items():
        print('PUBLIC SUMMARY', name, score['spell_f1_mean'], score['macro_f1'], score['per_class']['D']['f1'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('generate', 'check'))
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--inputs', type=Path, default=ROOT/'data/challenge/inputs.npz')
    args = parser.parse_args()
    if args.action == 'generate':
        generate(args.out, args.inputs)
    else:
        check(args.out)


if __name__ == '__main__':
    main()
