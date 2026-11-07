"""Freeze the frontier choice using audited development evidence only."""
from __future__ import annotations
import json
from pathlib import Path
from experiments import v2_spell_frontier_eval as evaluation

ROOT = evaluation.ROOT
STUDY = evaluation.STUDY
SELECTION = ROOT / f'experiments/results/{STUDY}_selection.json'
ROUND2 = ROOT / f'experiments/results/{STUDY}_round2.json'


def rank_candidates(first, second, fallback):
    candidates = []
    recipes = {r['recipe_id']: r for r in first['recipes']}
    pipelines = [dict(pipeline_id='base__'+key, recipe=recipe, training_arm='base', training_stage=1024,
                      result_key=key, result=first) for key, recipe in recipes.items()]
    pipelines += [dict(row, result_key=row['pipeline_id'], result=second) for row in second['pipelines']]
    for result, keys in ((first, set(recipes)), (second, {r['pipeline_id'] for r in second['pipelines']})):
        if set(result['aggregate']) != keys or set(result['eligibility']) != keys:
            raise ValueError('Result and declared pipeline inventories differ')
    floor = fallback['aggregate']['feed_like_v2']['spell_f1']['equal_provider']
    for row in pipelines:
        result, key = row.pop('result'), row.pop('result_key')
        gates, aggregate = result['eligibility'][key], result['aggregate'][key]
        if not gates['eligible']:
            continue
        if not (gates['d_metrics_unchanged'] and gates['reference_spell_counts_invariant'] and
                all(c['passed'] for c in gates['checks'].values())):
            raise ValueError('Eligible pipeline fails a required check')
        if aggregate['feed_like_v2']['spell_f1']['equal_provider'] <= floor:
            continue
        candidates.append(dict(row, aggregate=aggregate, eligibility=gates))
    candidates.sort(key=lambda row:(-row['aggregate']['feed_like_v2']['spell_f1']['equal_provider'],
                                    -row['aggregate']['feed_like_v2']['macro_f1']['equal_provider'],
                                    row['pipeline_id']))
    retained = dict(pipeline_id='base__'+fallback['recipe']['recipe_id'], recipe=fallback['recipe'],
                    training_arm='base', training_stage=1024, aggregate=fallback['aggregate'],
                    eligibility=fallback['eligibility'])
    return candidates + [retained]


def freeze_selection():
    from experiments import v2_spell_frontier_round2 as round2
    from experiments import v2_spell_frontier_train as train
    evaluation.require_declaration()
    round2.require_declaration()
    if SELECTION.exists():
        raise FileExistsError('Preserve final selection')
    for study in (STUDY, 'v2_spell_balanced_v1'):
        if ((ROOT/f'data/experiments/{study}/public_gate_claimed.json').exists() or
                (ROOT/f'experiments/results/{study}_public_result.json').exists()):
            raise RuntimeError('Public scoring must follow all development selection')
    paths = [evaluation.RESULT, ROUND2]
    first, second = [json.loads(path.read_text()) for path in paths]
    for result, declaration in zip((first, second), (evaluation.DECLARATION, round2.DECLARATION)):
        if result.get('declaration_sha256') != evaluation.sha(declaration):
            raise ValueError('Result declaration mismatch')
    audits = [ROOT/f'experiments/results/{STUDY}_independent_round{n}.json' for n in (1,2)]
    for number, audit_path, result_path in zip((1,2), audits, paths):
        audit = json.loads(audit_path.read_text())
        if (audit.get('passed') is not True or audit.get('cell_count') != 35 or
                audit.get(f'round{number}_result_sha256') != evaluation.sha(result_path)):
            raise ValueError('Both complete bound audits are required')
    audit_plan_path = ROOT/f'experiments/results/{STUDY}_audit_plan.json'
    audit_plan = json.loads(audit_plan_path.read_text())
    for raw, expected in audit_plan['verifier_and_test_sha256'].items():
        if evaluation.sha(ROOT/raw) != expected:
            raise ValueError('Audit implementation or tests changed after freeze')
    for path in audits:
        audit = json.loads(path.read_text())
        if (audit.get('audit_plan_sha256') != evaluation.sha(audit_plan_path) or
                audit.get('verifier_source_sha256') != evaluation.sha(ROOT/'scripts/verify_v2_spell_frontier.py')):
            raise ValueError('Audit does not bind the frozen verifier and plan')
    fallback = json.loads(evaluation.SELECTION.read_text())
    ranking = rank_candidates(first, second, fallback)
    winner = ranking[0]
    sources = [evaluation.PLAN, evaluation.DECLARATION, evaluation.REPLAY, evaluation.RESULT,
               round2.DECLARATION, ROUND2, train.DECL, train.SEAL, evaluation.SELECTION,
               *audits, ROOT/f'experiments/results/{STUDY}_audit_plan.json',
               ROOT/'scripts/verify_v2_spell_frontier.py', Path(__file__)]
    recipe = dict(winner['recipe'], id=winner['recipe']['recipe_id'], training_stage=winner['training_stage'])
    selection = dict(study_id=STUDY, final_selection=True, **{k:winner[k] for k in
        ('pipeline_id','training_arm','training_stage','aggregate','eligibility')}, recipe=recipe,
        replaces_balanced_selection=winner['aggregate']['feed_like_v2']['spell_f1']['equal_provider'] >
            fallback['aggregate']['feed_like_v2']['spell_f1']['equal_provider'],
        public_input_sha256=evaluation.sha(ROOT/'data/challenge/inputs.npz'),
        development_result_bindings={str(p.relative_to(ROOT)):evaluation.sha(p) for p in sources},
        ranking=[{k:r[k] for k in ('pipeline_id','training_arm','training_stage')} |
                 {'v2_spell_f1':r['aggregate']['feed_like_v2']['spell_f1']['equal_provider'],
                  'v2_macro_f1':r['aggregate']['feed_like_v2']['macro_f1']['equal_provider']} for r in ranking],
        aspiration_above_0_60=winner['aggregate']['feed_like_v2']['spell_f1']['equal_provider'] > .60,
        public_labels_used=False, seed_diagnostics_used_for_selection=False, automatic_promotion=False,
        evidence='Five exposed development matches and model-conditioned synthetic views; not independent confirmation')
    evaluation._write_once(SELECTION, selection)
    return selection


if __name__ == '__main__':
    selected=freeze_selection()
    print(json.dumps({k:selected[k] for k in ('pipeline_id','training_arm','training_stage','aspiration_above_0_60')},indent=2))
