"""One user-requested public check after external selection and CSV freeze.

This never fits, selects or regenerates a model. Earlier experiment gates and
reports remain unchanged. Only the single externally selected candidate is read.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from experiments import current_baseline, light_corruption as light
from experiments.eval_set.protocol import sha256

ROOT = light.ROOT
REPORT = ROOT/'experiments/results/light_corruption_public_check.json'
SCORE_KEYS = ('spell_f1_mean', 'macro_f1', 'accuracy', 'per_class', 'spell_f1',
              'n_frames', 'n_graded', 'n_gt_spells', 'n_pred_spells',
              'team_acc_gt_alive', 'spell_bootstrap_low', 'spell_bootstrap_high',
              'bootstrap_low', 'bootstrap_high', 'bootstrap_n')


def verify(frozen, root=ROOT):
    for name, expected in frozen['files_sha256'].items():
        if sha256(root/name) != expected:
            raise ValueError(f'Frozen competition-check input changed: {name}')


def start(out, frozen):
    light.ce.write_json(out/'public_check_started.json', dict(
        started_utc=datetime.now(timezone.utc).isoformat(),
        declaration_sha256=sha256(out/'public_check_predeclared.json'),
        selected_model=frozen['selected_model']), exclusive=True)


def freeze(out, candidate='candidate.csv'):
    light.check(out)
    if REPORT.exists() or (out/'public_check_started.json').exists():
        raise ValueError('This public check has already started; no repeat scoring')
    candidate_path = out/candidate
    receipt_path = candidate_path.with_suffix('.json')
    receipt = json.loads(receipt_path.read_text())
    selection_path = out/receipt.get('selection_file', 'selection.json')
    selection = json.loads(selection_path.read_text())
    if receipt['arm'] != selection['selected']['model']:
        raise ValueError('Candidate is not the externally selected arm')
    if receipt['selection_sha256'] != sha256(selection_path):
        raise ValueError('Candidate selection hash changed')
    if receipt['output_sha256'] != sha256(candidate_path):
        raise ValueError('Frozen candidate hash changed')
    for name, expected in selection['pooled_checkpoints_sha256'].items():
        if sha256(out/name) != expected: raise ValueError('Externally selected weights changed')
    paths = [candidate_path, receipt_path, selection_path, out/'selection.json',
             ROOT/'experiments/results/light_corruption_development.json',
             ROOT/'experiments/results/current_baseline_public_check.json',
             ROOT/'experiments/baseline_manifest.json', Path(__file__),
             ROOT/'evaluation/possession_metrics.py']
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        selected_model=selection['selected']['model'], bootstrap_draws=1000,
        candidate_file=candidate_path.name, diagnostic_only=selection.get('diagnostic_only', False),
        working_model=json.loads((out/'selection.json').read_text())['selected']['model'],
        files_sha256={str(p.relative_to(ROOT)): sha256(p) for p in paths},
        user_request="try next experiment and also report to me the comp's score",
        scope='One descriptive five-minute check after external selection; no fitting, tuning, reselection or full-match claim',
        prior_public_gates='Preserved; this user-requested check has its own exclusive marker',
        baseline_policy='If the selected CSV is byte-identical to the archived baseline, reuse that score without reading GT')
    target = out/'public_check_predeclared.json'
    if target.exists():
        old = json.loads(target.read_text())
        if old['files_sha256'] != spec['files_sha256']: raise ValueError('Declaration changed')
        verify(old)
    else: light.ce.write_json(target, spec, exclusive=True)
    print('Frozen the single externally selected competition candidate', flush=True)


def check(out):
    if REPORT.exists(): raise ValueError('Public report already exists; do not repeat')
    frozen = json.loads((out/'public_check_predeclared.json').read_text())
    verify(frozen)
    start(out, frozen)
    previous = json.loads((ROOT/'experiments/results/current_baseline_public_check.json').read_text())
    candidate_path = out/frozen['candidate_file']
    identical = sha256(candidate_path) == previous['frozen']['candidate_sha256']
    if identical:
        score = previous['scores']['gru_hgb']
        source = 'Reused exact existing baseline CSV score; organizer reference labels not reread'
    else:
        import pandas as pd
        from evaluation import possession_metrics as pm
        gt_path = ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'
        if sha256(gt_path) != previous['organizer_gt_sha256']:
            raise ValueError('Public reference differs from the historical check')
        predicted = pd.read_csv(candidate_path)
        gt = pd.read_csv(gt_path)
        error = pm.validate_submission(predicted, len(gt))
        if error: raise ValueError(error)
        score = pm.evaluate(pm.label_codes(gt.label.to_numpy()),
                            pm.label_codes(predicted.label.to_numpy()),
                            bootstrap_n=frozen['bootstrap_draws'])
        source = 'New one-time user-requested public pipeline check'
    result = dict(scope=frozen['scope'], frozen=frozen, score_source=source,
        organizer_reference_labels_read=not identical,
        score={k: score[k] for k in SCORE_KEYS},
        archived_baseline={k: previous['scores']['gru_hgb'][k]
                           for k in ('spell_f1_mean', 'macro_f1', 'n_gt_spells')},
        full_match_score=None, external_selection_changed=False,
        baseline_replaced=False, bootstrap_interval='Official 5th–95th percentiles when available')
    light.ce.write_json(REPORT, result, exclusive=True)
    light.ce.write_json(out/'public_check_result.json', result, exclusive=True)
    print(json.dumps(result['score'], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'check'))
    p.add_argument('--out', type=Path, default=light.DEFAULT)
    p.add_argument('--candidate', default='candidate.csv')
    args = p.parse_args()
    if args.action == 'freeze': freeze(args.out, args.candidate)
    else: check(args.out)


if __name__ == '__main__': main()
