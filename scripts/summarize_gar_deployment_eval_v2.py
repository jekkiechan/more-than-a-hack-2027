"""Apply the already-declared readout guards without promoting a package."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments import gar_deployment_eval_v2 as study
from experiments import gar_deployment_eval_v2_execution as execution


def summarize():
    declaration = execution.check()
    results = study.read(study.RESULT)
    comparison = results['comparisons']['test']
    clean, stress = comparison['clean']['phase_gain'], comparison['v2']['phase_gain']
    bounds = declaration['guards']
    guards = {
        'clean_spell_floor': clean['spell_f1']['delta'] >= bounds['clean_spell_delta_min'],
        'clean_macro_floor': clean['macro_f1']['delta'] >= bounds['clean_macro_delta_min'],
        'clean_d_f1_floor': clean['d_f1']['delta'] >= bounds['clean_d_f1_delta_min'],
    }
    payload = dict(study=study.STUDY, result_sha256=study.digest(study.RESULT),
        execution_amendment_sha256=study.digest(execution.AMENDMENT),
        primary_test_v2_spell_gain=stress['spell_f1'], clean_test_spell_gain=clean['spell_f1'],
        declared_guards=guards, all_declared_guards_pass=all(guards.values()),
        primary_point_estimate_improves=stress['spell_f1']['delta']>0,
        phase_comparator_test_v2=comparison['v2']['package_with_phase'],
        interpretation='Read paired game spread and confidence intervals; this receipt does not authorize adoption',
        original_five_match_clean_screen=dict(required_delta=.01,observed_delta=.0049650854525323,passes=False),
        active_policy_changed=False, promotion=False, public_scoring=False)
    path = study.ROOT/'experiments/results/gar_deployment_eval_v2_decision.json'
    study.write_once(path,payload)
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    summarize()
