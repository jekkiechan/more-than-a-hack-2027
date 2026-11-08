"""Freeze the final development choice before pooled fitting and public access."""
from __future__ import annotations

import json
from pathlib import Path
from experiments import v2_spell_balanced_eval as evaluation

ROOT = evaluation.ROOT
STUDY = evaluation.STUDY
SELECTION = ROOT / f'experiments/results/{STUDY}_selection.json'
ROUND2 = ROOT / f'experiments/results/{STUDY}_round2.json'


def rank_candidates(rounds: list[tuple[int, dict]]) -> list[dict]:
    """Rank eligible ensemble results; seed diagnostics never enter selection."""
    candidates = []
    for stage, result in rounds:
        if stage not in (512, 1024):
            raise ValueError('Undeclared continuation stage')
        recipes = {r['recipe_id']: r for r in result['recipes']}
        if set(result['aggregate']) != set(recipes) or set(result['eligibility']) != set(recipes):
            raise ValueError('Recipe, aggregate and eligibility inventories differ')
        for key, recipe in recipes.items():
            gates = result['eligibility'][key]
            if not gates['eligible']:
                continue
            if not gates['d_metrics_unchanged'] or not gates['reference_spell_counts_invariant']:
                raise ValueError('An eligible candidate fails a required invariant')
            if not all(check['passed'] for check in gates['checks'].values()):
                raise ValueError('An eligible candidate has failed metric checks')
            candidates.append(dict(training_stage=stage, recipe=recipe,
                                   aggregate=result['aggregate'][key], eligibility=gates))
    return sorted(candidates, key=lambda row: (
        -row['aggregate']['feed_like_v2']['spell_f1']['equal_provider'],
        -row['aggregate']['feed_like_v2']['macro_f1']['equal_provider'],
        row['recipe']['recipe_id'], row['training_stage']))


def freeze_selection() -> dict:
    from experiments import v2_spell_balanced_round2 as round2
    from experiments import v2_spell_balanced_execution as execution
    evaluation.require_declaration()
    execution.require_round2_declaration()
    if SELECTION.exists():
        raise FileExistsError('Final selection already exists')
    for path in (ROOT / f'experiments/results/{STUDY}_public_result.json',
                 evaluation.OUT / 'public_gate_claimed.json'):
        if path.exists():
            raise RuntimeError('Development selection cannot follow public scoring')
    first = json.loads(evaluation.RESULT.read_text())
    second = json.loads(ROUND2.read_text())
    receipt = json.loads(execution.RECEIPT.read_text())
    if (receipt.get('result_sha256') != evaluation.sha(ROUND2) or
            receipt.get('amendment_sha256') != evaluation.sha(execution.AMENDMENT) or
            receipt.get('completed') is not True):
        raise ValueError('Round-two execution receipt does not bind the completed corrected run')
    audit_paths = [ROOT / f'experiments/results/{STUDY}_independent_round{number}.json'
                   for number in (1, 2)]
    for number, path, result_path in zip((1, 2), audit_paths, (evaluation.RESULT, ROUND2)):
        audit = json.loads(path.read_text())
        if (audit.get('passed') is not True or audit.get('cell_count') != 35 or
                audit.get(f'round{number}_result_sha256') != evaluation.sha(result_path)):
            raise ValueError('Final selection requires both complete, bound independent audits')
    if (first['declaration_sha256'] != evaluation.sha(evaluation.DECLARATION) or
            second['declaration_sha256'] != evaluation.sha(round2.DECLARATION)):
        raise ValueError('Development results do not bind their frozen declarations')
    ranked = rank_candidates([(512, first), (1024, second)])
    if not ranked:
        raise ValueError('No eligible candidate, including the replayed anchor')
    winner = ranked[0]
    bindings = {str(path.relative_to(ROOT)): evaluation.sha(path) for path in (
        evaluation.PLAN, evaluation.DECLARATION, evaluation.CONTROL_REPLAY,
        evaluation.RESULT, round2.DECLARATION, ROUND2,
        Path(execution.__file__), execution.AMENDMENT, execution.RECEIPT,
        *audit_paths,
        ROOT / f'experiments/results/{STUDY}_audit_plan.json',
        ROOT / f'experiments/results/{STUDY}_audit_execution_amendment.json',
        ROOT / 'scripts/verify_v2_spell_balanced.py',
        ROOT / 'scripts/verify_v2_spell_balanced_execution.py',
        Path(__file__))}
    recipe = dict(winner['recipe'])
    recipe['id'] = recipe['recipe_id']
    result = dict(study_id=STUDY, final_selection=True,
                  training_stage=winner['training_stage'], recipe=recipe,
                  public_input_sha256=evaluation.sha(ROOT / 'data/challenge/inputs.npz'),
                  aggregate=winner['aggregate'], eligibility=winner['eligibility'],
                  development_result_bindings=bindings,
                  ranking=[{'training_stage': r['training_stage'],
                            'recipe_id': r['recipe']['recipe_id'],
                            'v2_spell_f1': r['aggregate']['feed_like_v2']['spell_f1']['equal_provider'],
                            'v2_macro_f1': r['aggregate']['feed_like_v2']['macro_f1']['equal_provider']}
                           for r in ranked],
                  aspiration_above_0_60=winner['aggregate']['feed_like_v2']['spell_f1']['equal_provider'] > .60,
                  public_labels_used=False, seed_diagnostics_used_for_selection=False,
                  automatic_promotion=False,
                  evidence='Exposed development matches and synthetic views; selection is not independent confirmation')
    evaluation._write_once(SELECTION, result)
    return result


if __name__ == '__main__':
    selected = freeze_selection()
    print(json.dumps({key: selected[key] for key in ('training_stage', 'recipe', 'aspiration_above_0_60')}, indent=2))
