"""User-authorized phase-average promotion with exact sealed-CSV replay."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from experiments import original_full_phase_baseline as inference
from experiments.promote_original_full_baseline import preserve, encode, read
from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "joint_gru_sn_dense_original_diversity_phase_ensemble"
PACKAGE = MODEL + "_v1"
PREDECESSOR = "joint_gru_sn_dense_original_diversity_ensemble_v1"
ACTIVE = ROOT / "experiments/baseline_manifest.json"
POLICY = ROOT / "experiments/model_decision_policy.json"
DIRECTORY = ROOT / "data/models" / PACKAGE
STAGED = DIRECTORY / "baseline_manifest.json"
RECEIPT = ROOT / "experiments/results/baseline_original_full_phase_promotion.json"
CSV = ROOT / "data/experiments/current_baseline" / (MODEL + ".csv")
USER_REQUEST = "update AGENTS.md as necessary, update the site, and promote the latest model"
EXCEPTION = ("Explicit user adoption of original_full phase averaging; historical +.01 clean screen remains failed. "
             "All seven historical active guards pass; original clean/v1/v2 and GAR TEST paired gains support "
             "a provisional upgrade. TEST silver-label interval includes zero; calibrated+phase remains spell "
             "point leader. Independent continuous-match gold confirmation remains pending.")


def promote():
    if RECEIPT.exists():
        receipt = read(RECEIPT)
        if sha256(ACTIVE) != receipt['baseline_manifest_sha256'] or sha256(POLICY) != receipt['policy_sha256']:
            raise ValueError('Completed promotion differs from active metadata')
        return receipt
    old = read(ACTIVE)
    if old['package_version'] != PREDECESSOR:
        raise ValueError('Expected the original_full phase0 predecessor')
    previous = ROOT / 'experiments/baselines' / (PREDECESSOR + '.json')
    previous_policy = ROOT / 'experiments/baselines' / ('model_decision_policy_' + PREDECESSOR + '.json')
    preserve(previous, ACTIVE.read_bytes())
    preserve(previous_policy, POLICY.read_bytes())
    replay = read(ROOT/'experiments/results/original_full_phase_average_replay_v1.json')
    scorecard = read(ROOT/'experiments/results/original_full_phase_average_scorecard_v2.json')
    guards = scorecard['active_policy_guard_completion']
    if not guards['all_active_policy_guards_pass'] or guards['historical_clean_screen_passes']:
        raise ValueError('Expected passing active guards and failed historical clean screen')
    inputs = ROOT/'data/challenge/inputs.npz'
    if sha256(inputs) != replay['input_sha256'] or sha256(previous) != replay['manifest_sha256']:
        raise ValueError('Sealed input or predecessor identity changed')
    if old['models_sha256'] != replay['models_sha256']:
        raise ValueError('Sealed model identities changed')
    for name, expected in old['models_sha256'].items():
        source = ROOT/old['model_directory']/name
        if sha256(source) != expected:
            raise ValueError('Model hash mismatch: ' + name)
        preserve(DIRECTORY/name, source.read_bytes())
    # Keep the historical dispatcher separately; do not alter old source receipts.
    historical_dispatch = subprocess.check_output(['git', 'show', 'HEAD:experiments/current_baseline.py'], cwd=ROOT)
    preserve(DIRECTORY/'previous_source/current_baseline.py', historical_dispatch)
    evidence_names = ['original_full_phase_average_scorecard_v2.json', 'original_full_phase_average_replay_v1.json',
        'original_full_phase_average_adapter_replay_v1.json', 'phase_average_guard_v1.json',
        'phase_average_guard_v1_verification.json', 'gar_deployment_eval_v2.json',
        'gar_deployment_eval_v2_verification.json', 'feed_like_v2_public_result.json']
    public = read(ROOT/'experiments/results/feed_like_v2_public_result.json')
    if public['candidate_csv_sha256'] != replay['candidate_csv_sha256']:
        raise ValueError('Candidate seals disagree')
    if STAGED.exists():
        manifest = read(STAGED)
    else:
        manifest = dict(version=10, model=MODEL, package_version=PACKAGE,
            status='User-authorized provisional working baseline; independent gold confirmation pending',
            promoted_utc=datetime.now(timezone.utc).isoformat(), replaced_baseline=old['model'],
            previous_manifest=str(previous.relative_to(ROOT)), previous_manifest_sha256=sha256(previous),
            previous_policy=str(previous_policy.relative_to(ROOT)), previous_policy_sha256=sha256(previous_policy),
            model_directory=str(DIRECTORY.relative_to(ROOT)), models_sha256=old['models_sha256'],
            default_output=str(CSV.relative_to(ROOT)), training=old['training'],
            architecture=dict(old['architecture'], kind=inference.KIND, phase_tta=True,
                phase_averaging=inference.PHASE_AVERAGING,
                combination='Uniform native seed mean per phase, uniform five-phase mean, then frozen f062 normalization/temperature-4/cost-3 decoding'),
            evidence=dict(report='experiments/ORIGINAL_FULL_PHASE_AVERAGE.md',
                artifacts_sha256={'experiments/results/'+name:sha256(ROOT/'experiments/results'/name) for name in evidence_names},
                original_development=scorecard['original_five_match_aggregates'],
                prediction_csv_sha256=replay['candidate_csv_sha256'],
                historical_active_guards_pass=True, historical_clean_screen_passes=False,
                gar_test_games=10, gar_valid_games=9, gar_valid_previously_selected_on=True,
                independent_continuous_gold_confirmation=False),
            promotion=dict(type='explicit_user_authorized_phase_upgrade', user_request=USER_REQUEST,
                clean_screen_exception=EXCEPTION, original_screen_passes=False, automatic_promotion=False,
                fitting_during_promotion=False, phase_tta_added=True, organizer_rescoring_performed=False,
                public_diagnostic_known_before_authorization=True),
            organizer_public_diagnostic=dict(report='experiments/results/feed_like_v2_public_result.json',
                diagnostic_only=True, known_at_promotion=True, metrics=public['candidate'],
                used_for_model_or_hyperparameter_selection=False, independent_confirmation=False),
            source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [
                ROOT/'experiments/current_baseline.py', Path(inference.__file__), Path(__file__),
                ROOT/'experiments/feed_like_v2_public.py', ROOT/'experiments/feed_benchmark_audit_v1.py']},
            organizer_reference_labels_used_for_training=False, historical_selection_records_preserved=True)
        preserve(STAGED, encode(manifest))
    for name, expected in manifest['source_sha256'].items():
        if sha256(ROOT/name) != expected:
            raise ValueError('Staged inference source changed: ' + name)
    if not CSV.exists():
        subprocess.run([sys.executable, '-m', 'experiments.current_baseline', 'predict', '--manifest', str(STAGED)], cwd=ROOT, check=True)
    validation = read(CSV.with_suffix('.json'))
    if (sha256(CSV) != replay['candidate_csv_sha256'] or validation['baseline_manifest_sha256'] != sha256(STAGED)
            or validation['input_sha256'] != replay['input_sha256'] or validation['target_labels_read']
            or validation['organizer_scored']):
        raise ValueError('Default CLI must reproduce the sealed phase-average CSV without reference access')
    subprocess.run([sys.executable, 'evaluation/validate_submission.py', str(CSV)], cwd=ROOT, check=True)
    policy = read(previous_policy)
    policy.update(version=10, confirmed_utc_date=manifest['promoted_utc'][:10], primary_evaluation_view='feed_like_v2')
    policy['working_candidate'] = dict(family='Original-full two-seed ensemble with uniform phase averaging',
        model=MODEL, status=manifest['status'], baseline_manifest='experiments/baseline_manifest.json',
        baseline_manifest_sha256=sha256(STAGED), prediction_entrypoint='python -m experiments.current_baseline predict',
        reason=EXCEPTION, promotion_record=str(RECEIPT.relative_to(ROOT)), previous_policy=str(previous_policy.relative_to(ROOT)))
    policy['stress_guardrail'] = dict(view='feed_like_v2', reference_recipe=MODEL, minimum_realizations=4,
        rule='Matched full-provider candidate v2 spell >= adopted -.01; protect each provider clean macro/D-F1 within .02; report component seeds separately',
        changed_from='feed_like', changed_after_v2_results=True,
        historical_guard_results_preserved=True, diagnostic_only=['feed_like_v1','composite','no_sampling','no_regional','ball_gaps'],
        evidence=['experiments/FEED_LIKE_V2.md','experiments/GAR_DEPLOYMENT_EVAL_V2.md'])
    policy['evaluation_roles'] = dict(development='Original five matches, full-provider holdout, clean plus feed_like_v2, equal provider means',
        frozen_candidate_check='GAR TEST ten games; exact deployment weights; clean plus four v2 realizations; equal game means',
        gar_valid='Nine previously selected-on games, development only; never pool with TEST for confirmation',
        organizer_public='17-possession pipeline diagnostic only; no tuning, selection or fresh scoring',
        limitations='Repeated TEST reuse becomes development exposure; silver segments are not continuous-match gold; new independent gold remains needed')
    policy['clean_improvement_screen'] = dict(default_delta=.01, historical_failures_preserved=True,
        current_screen_passes=False, current_adoption_exception=EXCEPTION)
    if ACTIVE.read_bytes() != previous.read_bytes() or POLICY.read_bytes() != previous_policy.read_bytes():
        raise ValueError('Active predecessor changed during replay')
    preserve(ROOT/'experiments/baselines'/(PACKAGE+'.json'), encode(manifest))
    ACTIVE.write_bytes(encode(manifest)); POLICY.write_bytes(encode(policy))
    receipt = dict(package_version=PACKAGE, promoted_utc=manifest['promoted_utc'], authorization=manifest['promotion'],
        baseline_manifest_sha256=sha256(ACTIVE), policy_sha256=sha256(POLICY),
        previous_manifest_sha256=sha256(previous), previous_policy_sha256=sha256(previous_policy),
        models_sha256=manifest['models_sha256'], candidate_csv_sha256=sha256(CSV), rows=199319,
        csv_equals_sealed_candidate=True, validation_receipt_sha256=sha256(CSV.with_suffix('.json')),
        fitting_performed=False, organizer_reference_accessed=False, organizer_rescored=False,
        historical_score_gates_and_declarations_preserved=True)
    preserve(RECEIPT, encode(receipt)); preserve(DIRECTORY/'promotion_receipt.json', encode(receipt))
    preserve(DIRECTORY/'model_decision_policy.json', encode(policy))
    return receipt


if __name__ == '__main__':
    print('PROMOTED', promote()['package_version'])
