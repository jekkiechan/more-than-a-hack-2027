"""Package and verify the explicitly user-adopted two-seed GRU baseline.

No fitting or reference scoring occurs here. Seed 2026 is restored; the one
missing pooled deployment seed is fitted and sealed by a separate runner.
"""
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from experiments import current_baseline as baseline, seed_ensemble_deployment as deploy, corruption_experiment as ce
from experiments import supervised, learned_temporal as lt, challenge
from experiments.eval_set.protocol import sha256

ROOT=baseline.ROOT
MODEL='joint_gru_seed_ensemble'
PACKAGE=MODEL+'_v1'
RECEIPT=ROOT/'experiments/results/baseline_seed_ensemble_promotion.json'
MEMBERS={'joint_gru_2026':.5,'joint_gru_2027':.5}
PAGE='https://chatgpt.com/space/page_1df518ea32448191bf885da4952259a9'


def preserve(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=value:raise ValueError(f'Existing historical file differs: {path}')
    else:path.write_bytes(value)


def evidence():
    report=ROOT/'experiments/results/seed_gap_context.json'
    verify=ROOT/'experiments/results/seed_gap_context_verification.json'
    r=json.loads(report.read_text()); v=json.loads(verify.read_text())
    if r['parity_cells']!=30 or v['base_report_sha256']!=sha256(report) or v['exact_probability_ensemble_cells']!=15:
        raise ValueError('The actual probability ensemble is not verified')
    lookup=lambda name,view:next(x for x in r['summaries'] if x['recipe']==name and x['view']==view)
    e={view:lookup('equal_probability_ensemble',view) for view in ('clean','feed_like')}
    previous={view:lookup('seed_2026',view) for view in e}
    other={view:lookup('seed_2027',view) for view in e}
    worst=min(e['clean']['per_provider'][s][k]-previous['clean']['per_provider'][s][k]
              for s in e['clean']['per_provider'] for k in ('macro_f1','d_f1'))
    guards=dict(clean_spell_delta=e['clean']['spell_f1']-previous['clean']['spell_f1'],
        feed_like_spell_delta=e['feed_like']['spell_f1']-previous['feed_like']['spell_f1'],
        worst_provider_clean_d_or_macro_delta=worst)
    guards['passes']=guards['clean_spell_delta']>=.01 and guards['feed_like_spell_delta']>=-.01 and worst>=-.02
    if not guards['passes']:raise ValueError('Ensemble fails the current baseline guardrails')
    return dict(report=str(report.relative_to(ROOT)),report_sha256=sha256(report),
        verification=str(verify.relative_to(ROOT)),verification_sha256=sha256(verify),
        view='full_provider_holdout',ensemble=e,previous_single_seed=previous,
        other_single_seed=other,guardrails=guards,previously_exposed_development_matches=True)


def write_prediction(inputs,probability,path):
    labels=supervised.decode(probability,supervised.model_periods(inputs),float(inputs['fps']),lt.COST)
    labels[np.asarray(inputs['period'])<=0]='X'
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.csv',delete=False) as f:staged=Path(f.name)
    try:
        challenge.write_submission(labels,staged)
        digest=sha256(staged)
        if path.exists():
            if sha256(path)!=digest:raise ValueError('Existing prediction differs')
        else:os.replace(staged,path)
        return digest
    finally:staged.unlink(missing_ok=True)


def promote():
    active=ROOT/'experiments/baseline_manifest.json'
    old=json.loads(active.read_text())
    if old['model']==MODEL:
        baseline.load_models();return
    if old['package_version']!='joint_gru_slotdrop_v1' or RECEIPT.exists():
        raise ValueError('Unexpected baseline or completed promotion')
    deploy.check()
    fitted=ce.load_model(deploy.OUT,deploy.MODEL)
    if fitted['seed']!=2027:raise ValueError('Wrong deployment seed')
    observed=evidence()
    package=ROOT/'data/models'/PACKAGE
    package.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,source in (('joint_gru_2026',deploy.CONTROL),('joint_gru_2027',deploy.MODEL)):
        target=package/(name+'.joblib')
        if target.exists() and sha256(target)!=sha256(source):raise ValueError('Package conflict')
        if not target.exists():shutil.copyfile(source,target)
        hashes[target.name]=sha256(target)
    import joblib
    models={p.removesuffix('.joblib'):joblib.load(package/p) for p in hashes}
    architecture=dict(kind='joint_gru_ensemble',members=MEMBERS,
        combination='Equal mean of native H/A/D probabilities before unchanged Viterbi decoder',
        network='Two original bidirectional GRUs, hidden24 each; player-only D path',
        rate_hz=5,core_seconds=10,context_each_side_seconds=5,decoder_switch_cost=3,
        training='Same thirteen external TRAIN matches, exact slot-dropout recipe, seeds2026/2027')
    inputs_path=ROOT/'data/challenge/inputs.npz'
    check=json.loads((ROOT/'experiments/results/temporal_competition_check.json').read_text())
    if sha256(inputs_path)!=check['declaration']['files_sha256'][str(inputs_path.relative_to(ROOT))]:
        raise ValueError('Organizer input seal changed')
    inputs=baseline.load_inputs(inputs_path)
    pieces=baseline.joint_gru_member_probabilities(inputs,models,dict(architecture=architecture))
    output=ROOT/f'data/experiments/current_baseline/{MODEL}.csv'
    old_output=ROOT/'data/experiments/current_baseline/single_seed_2026_promotion_replay.csv'
    control_digest=write_prediction(inputs,pieces['joint_gru_2026'],old_output)
    if control_digest!=old['validated_candidate']['sha256']:
        raise ValueError('Original seed-2026 CSV fails byte parity')
    probability=.5*(pieces['joint_gru_2026']+pieces['joint_gru_2027'])
    digest=write_prediction(inputs,probability,output)
    validation=dict(input_sha256=sha256(inputs_path),output_sha256=digest,
        original_seed_2026_replay_sha256=control_digest,original_prediction_parity=True,
        probability_sha256=hashlib.sha256(np.ascontiguousarray(probability).tobytes()).hexdigest(),
        member_probability_sha256={k:hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() for k,p in pieces.items()},
        native_rows=len(probability),full_vector_average=True,organizer_reference_labels_read=False,
        organizer_scored=False,full_match_score=None)
    policy_path=ROOT/'experiments/model_decision_policy.json'
    history=ROOT/'experiments/baselines'
    previous_manifest=history/'joint_gru_slotdrop_v1.json'
    previous_policy=history/'model_decision_policy_joint_gru_slotdrop_v1.json'
    preserve(previous_manifest,active.read_bytes());preserve(previous_policy,policy_path.read_bytes())
    manifest=dict(version=4,model=MODEL,package_version=PACKAGE,
        status='User-authorized working development baseline; independent confirmation pending',
        promoted_utc=datetime.now(timezone.utc).isoformat(),replaced_baseline=old['model'],
        previous_manifest=str(previous_manifest.relative_to(ROOT)),previous_manifest_sha256=sha256(previous_manifest),
        model_directory=str(package.relative_to(ROOT)),models_sha256=hashes,
        default_output=str(output.relative_to(ROOT)),architecture=architecture,evidence=observed,
        promotion=dict(type='explicit_user_authorized_working_baseline',
            user_request="this seems like an improvement. let's make this the updated baseline.",
            selected='Actual equal 2026/2027 probability ensemble; original five-second context',
            restored_members=[2026],newly_fitted_pooled_deployment_members=[2027],
            comparison_retraining=False,fitting_during_promotion=False,public_rescoring_performed=False),
        deployment_fit_declaration='experiments/results/seed_ensemble_deployment_declaration.json',
        deployment_fit_receipt_sha256=sha256(deploy.MODEL.with_suffix('.json')),
        validated_candidate=dict(path=str(output.relative_to(ROOT)),sha256=digest),validation=validation,
        previous_single_seed_public_diagnostic={**old['existing_public_diagnostic'],
            'model':old['model'],'role':'Historical single-seed result; not an ensemble score'},
        organizer_public_diagnostic=None,organizer_reference_labels_used_for_training=False,
        progress_page=PAGE,historical_selection_records_preserved=True,
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in (Path(__file__),Path(baseline.__file__),Path(deploy.__file__))})
    encoded=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
    preserve(package/'baseline_manifest.json',encoded)
    active.write_bytes(encoded)
    policy=json.loads(policy_path.read_text())
    policy['version']=4
    policy['working_candidate']=dict(family='Two-seed joint GRU probability ensemble',model=MODEL,
        status=manifest['status'],baseline_manifest='experiments/baseline_manifest.json',
        baseline_manifest_sha256=sha256(active),prediction_entrypoint='python -m experiments.current_baseline predict',
        reason='User adopted actual equal-probability ensemble: clean/feed-like spell .6265/.4874; passes matched former-baseline guards',
        promotion_record=str(RECEIPT.relative_to(ROOT)),previous_policy=str(previous_policy.relative_to(ROOT)),
        public_diagnostic_known_before_user_request=False)
    policy['stress_guardrail']['reference_recipe']=MODEL
    policy['stress_guardrail']['rule']='Full-pipeline candidate feed-like spell >= adopted ensemble -.01; report component seeds separately and protect provider clean D-F1/macro within .02'
    policy_path.write_text(json.dumps(policy,indent=2)+'\n')
    preserve(RECEIPT,(json.dumps(dict(package=PACKAGE,models_sha256=hashes,
        baseline_manifest_sha256=sha256(active),previous_manifest_sha256=sha256(previous_manifest),
        previous_policy_sha256=sha256(previous_policy),model_decision_policy_sha256=sha256(policy_path),
        deployment_fit_receipt=json.loads(deploy.MODEL.with_suffix('.json').read_text()),
        validation=validation,evidence=observed,organizer_labels_read=False),indent=2)+'\n').encode())
    baseline.load_models()
    print('PROMOTED',PACKAGE,observed['guardrails'],flush=True)


if __name__=='__main__':promote()
