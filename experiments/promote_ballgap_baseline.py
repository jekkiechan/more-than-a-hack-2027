"""Package the user-authorized ball-gap team baseline without fitting or scoring.

The historical clean-winner selection remains intact. This is an explicit
working-baseline decision with its external tradeoff recorded, not a replay of
the original experiment's selection criterion.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

from experiments import current_baseline
from experiments.eval_set.protocol import sha256

ROOT = current_baseline.ROOT
MODEL = 'gru_hgb_ballgap25'
PACKAGE = 'gru_hgb_ballgap25_v1'
HISTORY = ROOT/'experiments/baselines'
REPORT = ROOT/'experiments/results/baseline_ballgap25_promotion.json'


def evidence(report):
    def row(model, view):
        return next(r for r in report['summaries'] if r['evaluation'] == 'provider_holdout_D'
                    and r['view'] == view and r['model'] == model)
    cohorts = {model: {(r['source'], r['match'], r['view']) for r in report['rows']
                       if r['fold'] != 'pooled' and r['model'] == model}
               for model in ('baseline', 'ball_team25')}
    if not cohorts['baseline'] or cohorts['baseline'] != cohorts['ball_team25']:
        raise ValueError('Candidate and previous baseline must use identical nonempty cohorts')
    clean, previous_clean = row('ball_team25', 'clean'), row('baseline', 'clean')
    stress, previous_stress = row('ball_team25', 'corrupted'), row('baseline', 'corrupted')
    return dict(clean=clean, previous_clean=previous_clean,
                corrupted=stress, previous_corrupted=previous_stress,
                clean_spell_delta=clean['spell_f1']-previous_clean['spell_f1'],
                corrupted_spell_delta=stress['spell_f1']-previous_stress['spell_f1'],
                clean_rule_candidate_wins=current_baseline.official_better(clean, previous_clean),
                cohort=[list(v) for v in sorted(cohorts['baseline'])])


def preserve(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != data:
        raise ValueError(f'Refusing to replace historical artifact: {path}')
    if not path.exists(): path.write_bytes(data)


def promote():
    active = current_baseline.MANIFEST
    old = json.loads(active.read_text())
    if old['model'] == MODEL:
        current_baseline.load_models()
        print('Requested baseline is already active; hashes verified')
        return
    if old['model'] != 'gru_hgb' or REPORT.exists():
        raise ValueError('Unexpected current baseline or an existing promotion record')
    experiment = ROOT/'data/experiments/light_corruption_v1'
    external = ROOT/'experiments/results/light_corruption_development.json'
    public = ROOT/'experiments/results/light_corruption_public_check.json'
    policy_path = ROOT/'experiments/model_decision_policy.json'
    old_manifest_bytes, old_policy_bytes = active.read_bytes(), policy_path.read_bytes()
    diagnostic = json.loads((experiment/'diagnostic_selection.json').read_text())
    candidate_receipt = json.loads((experiment/'experimental_candidate.json').read_text())
    public_record = json.loads(public.read_text())
    if (diagnostic['selected']['model'] != 'ball_team25'
            or candidate_receipt['arm'] != 'ball_team25'
            or candidate_receipt['baseline_manifest_sha256'] != sha256(active)):
        raise ValueError('Diagnostic candidate does not bind the expected previous baseline')
    if sha256(external) != diagnostic['external_report_sha256']:
        raise ValueError('External report changed')
    if sha256(experiment/'diagnostic_selection.json') != candidate_receipt['selection_sha256']:
        raise ValueError('Diagnostic selection changed')
    if sha256(experiment/'experimental_candidate.csv') != candidate_receipt['output_sha256']:
        raise ValueError('Previously validated candidate changed')
    old_models = ROOT/old['model_directory']
    for name, expected in old['models_sha256'].items():
        if sha256(old_models/name) != expected: raise ValueError('Previous baseline model changed')
    team_source = experiment/'team_light.joblib'
    team_receipt = json.loads(team_source.with_suffix('.json').read_text())
    if (sha256(team_source) != diagnostic['pooled_checkpoints_sha256']['team_light.joblib']
            or sha256(team_source) != team_receipt['sha256']
            or sha256(experiment/'predeclared.json') != team_receipt['predeclared_sha256']):
        raise ValueError('Frozen team checkpoint changed')
    for name, expected in public_record['frozen']['files_sha256'].items():
        if sha256(ROOT/name) != expected: raise ValueError(f'Frozen public-check input changed: {name}')
    internal = evidence(json.loads(external.read_text()))
    if internal['corrupted_spell_delta'] <= 0:
        raise ValueError('Expected external corruption-robustness evidence is absent')
    historical_manifest = HISTORY/'gru_hgb_v1.json'
    historical_policy = HISTORY/'model_decision_policy_gru_hgb_v1.json'
    preserve(historical_manifest, old_manifest_bytes)
    preserve(historical_policy, old_policy_bytes)
    package = ROOT/'data/models'/PACKAGE
    package.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, source in [('bigru.joblib', old_models/'bigru.joblib'),
                         ('ordered_hgb.joblib', old_models/'ordered_hgb.joblib'),
                         ('attribution.joblib', team_source)]:
        target = package/name
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError(f'Existing package differs: {name}')
        shutil.copyfile(source, target)
        files[name] = sha256(target)
    scores = public_record['score']
    manifest = dict(version=2, model=MODEL, package_version=PACKAGE,
        status='User-authorized working development baseline; independent confirmation pending',
        promoted_utc=datetime.now(timezone.utc).isoformat(), replaced_baseline=old['model'],
        previous_manifest=str(historical_manifest.relative_to(ROOT)),
        previous_manifest_sha256=sha256(historical_manifest),
        model_directory=str(package.relative_to(ROOT)), models_sha256=files,
        default_output=f'data/experiments/current_baseline/{MODEL}.csv',
        architecture=dict(D_members={'bigru': .5, 'ordered_hgb': .5},
                          team='Pooled contextual HGB; 25% ball-gap-only training replacement',
                          decoder_switch_cost=3, dead_heads_unchanged=True),
        promotion=dict(type='explicit_user_authorized_working_baseline',
            user_request="so this is good on our internal evals and the public. if so let's update this to the new baseline",
            basis='External corruption robustness with near-tied clean performance; user-authorized tradeoff',
            frozen_clean_winner='previous gru_hgb baseline',
            prior_external_selection_rewritten=False,
            public_results_known_before_request=True,
            raw_reference_labels_read_for_promotion=False,
            retraining_performed=False, public_rescoring_performed=False),
        evidence=dict(report=str(external.relative_to(ROOT)), report_sha256=sha256(external),
                      view='provider_holdout_D', **internal),
        existing_public_diagnostic=dict(report=str(public.relative_to(ROOT)), report_sha256=sha256(public),
            role='Previously recorded descriptive public-window check; no full-match claim',
            spell_f1=scores['spell_f1_mean'], macro_f1=scores['macro_f1'],
            d_f1=scores['per_class']['D']['f1'], graded_frames=scores['n_graded'],
            reference_possessions=scores['n_gt_spells']),
        validated_candidate=dict(path=str((experiment/'experimental_candidate.csv').relative_to(ROOT)),
                                 sha256=candidate_receipt['output_sha256']),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in
            (Path(__file__), Path(current_baseline.__file__), ROOT/'experiments/refine_features.py',
             ROOT/'experiments/learned_temporal.py')},
        organizer_reference_labels_used_for_training=False, historical_selection_records_preserved=True)
    encoded = (json.dumps(manifest, indent=2, allow_nan=False)+'\n').encode()
    preserve(package/'baseline_manifest.json', encoded)
    staged = active.with_suffix('.json.tmp')
    staged.write_bytes(encoded)
    os.replace(staged, active)
    receipt = dict(model=MODEL, package=PACKAGE, baseline_manifest_sha256=sha256(active),
        previous_manifest_sha256=sha256(historical_manifest),
        previous_policy_sha256=sha256(historical_policy), models_sha256=files,
        expected_prediction_sha256=candidate_receipt['output_sha256'],
        historical_external_report_sha256=sha256(external),
        historical_public_report_sha256=sha256(public),
        organizer_reference_labels_read=False, retrained=False, public_rescored=False)
    preserve(REPORT, (json.dumps(receipt, indent=2)+'\n').encode())
    current_baseline.load_models()
    print('Promoted', MODEL, 'with unchanged D heads and a versioned team checkpoint')


if __name__ == '__main__': promote()
