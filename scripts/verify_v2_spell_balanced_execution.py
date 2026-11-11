"""Preserved-auditor correction for seed-cache grid order versus ranked shortlist."""
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from scripts import verify_v2_spell_balanced as auditor

ROOT=auditor.ROOT
AMENDMENT=ROOT/'experiments/results/v2_spell_balanced_v1_audit_execution_amendment.json'
TEST=ROOT/'experiments/tests/test_verify_v2_spell_balanced_execution.py'


def facts():
    return {str(p.relative_to(ROOT)):auditor.sha(p) for p in (
        Path(__file__),Path(auditor.__file__),TEST,auditor.AUDIT_PLAN,auditor.R1_RESULT)}


def declare():
    for n in (1,2):
        if (ROOT/f'experiments/results/v2_spell_balanced_v1_independent_round{n}.json').exists():
            raise RuntimeError('Do not amend a completed audit')
    value=dict(study_id=auditor.study.STUDY,
        failure='First R1 audit stopped at the first cell: R1 seed cache shortlist identity mismatch. Cache recipes follow frozen grid order; result shortlist follows score ranking.',
        correction='Verify the exact expected grid-order recipe identity before normalizing that identity list to ranked-shortlist order solely for the preserved auditor comparison. All cache hashes, values, metrics, predictions, eligibility and ranking checks remain unchanged.',
        original_auditor_preserved=True, actual_candidate_outputs_unchanged=True,
        audit_plan_sha256=auditor.sha(auditor.AUDIT_PLAN),source_and_input_sha256=facts())
    auditor._write_once(AMENDMENT,json.dumps(value,indent=2,sort_keys=True)+'\n')


@contextmanager
def corrected_identity_order(result):
    original=auditor.ast
    ranked=result['selected_for_round2_seed_diagnostics']
    expected=[r['recipe_id'] for r in result['recipes'] if r['recipe_id'] in ranked]
    if len(expected)!=len(ranked) or set(expected)!=set(ranked):
        raise AssertionError('Shortlist recipe identity is not unique and complete')
    def parse_identity(text):
        actual=original.literal_eval(text)
        if actual!=expected:
            raise AssertionError('Seed identity is not the exact frozen recipe-grid order')
        return list(ranked)
    auditor.ast=SimpleNamespace(literal_eval=parse_identity)
    try: yield
    finally: auditor.ast=original


def run(round_number, output):
    amendment=json.loads(AMENDMENT.read_text())
    if amendment['source_and_input_sha256']!=facts():
        raise RuntimeError('Independent audit correction source/input changed')
    if round_number==1:
        with corrected_identity_order(json.loads(auditor.R1_RESULT.read_text())):
            receipt=auditor.verify_round1()
    else:
        receipt=auditor.verify_round2()
    receipt.update(verifier_source_sha256=auditor.sha(Path(auditor.__file__)),
                   execution_source_sha256=auditor.sha(Path(__file__)),
                   audit_plan_sha256=auditor.sha(auditor.AUDIT_PLAN),
                   audit_execution_amendment_sha256=auditor.sha(AMENDMENT))
    auditor._write_once(Path(output),json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'round':round_number,'passed':True,'output':str(output)},indent=2),flush=True)
    return receipt


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['declare','run']);p.add_argument('--round',type=int,choices=[1,2]);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.action=='declare':declare()
    elif a.round is None or a.output is None:p.error('run requires --round and --output')
    else:run(a.round,a.output)
