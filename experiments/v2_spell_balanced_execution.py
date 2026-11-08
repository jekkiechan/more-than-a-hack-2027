"""Bounded pre-inference correction of the R2 shortlist stage-label check."""
from contextlib import contextmanager
from pathlib import Path
import json

from experiments import v2_spell_balanced_round2 as r2

ROOT = r2.ROOT
AMENDMENT = ROOT / 'experiments/results/v2_spell_balanced_v1_round2_execution_amendment.json'
RECEIPT = ROOT / 'experiments/results/v2_spell_balanced_v1_round2_execution_receipt.json'
TEST = ROOT / 'experiments/tests/test_v2_spell_balanced_execution.py'


def _facts():
    return {str(p.relative_to(ROOT)): r2.sha(p) for p in (
        Path(__file__), TEST, Path(r2.__file__), r2.DECLARATION,
        r2.r1.RESULT, r2.trainer.TRAIN_SEAL)}


def declare():
    if AMENDMENT.exists() or r2.RESULT.exists() or any(r2.OUT.glob('predictions/*.npz')):
        raise RuntimeError('Amend only the preserved pre-inference failure, before candidate outputs')
    r2._write_once(AMENDMENT, dict(study_id=r2.STUDY,
        reason='Frozen R2 declaration correctly labels recipes stage1024, but require_declaration compares them with the R1 stage512 metadata returned by _shortlist. It fails before jobs or inference.',
        observed_error='RuntimeError: Round-two recipes differ from the frozen round-one shortlist',
        correction='During R2 validation/execution only, normalize the already-validated shortlist training_stage metadata to1024. Keep every recipe parameter, weight, cell, model, objective, metric and gate unchanged.',
        original_runner_and_declaration_preserved=True,
        candidate_scores_before_correction=False, organizer_reference_access=False,
        source_and_input_sha256=_facts()))


@contextmanager
def corrected_stage_labels():
    original = r2._shortlist
    def corrected():
        return [{**row, 'training_stage': 1024} for row in original()]
    r2._shortlist = corrected
    try:
        yield
    finally:
        r2._shortlist = original


def check():
    amendment = json.loads(AMENDMENT.read_text())
    if amendment['source_and_input_sha256'] != _facts():
        raise RuntimeError('R2 execution correction source or input changed')
    return amendment


def require_round2_declaration():
    check()
    with corrected_stage_labels():
        return r2.require_declaration()


def run():
    check()
    if RECEIPT.exists():
        raise FileExistsError(RECEIPT)
    with corrected_stage_labels():
        result = r2.run_round2()
    r2._write_once(RECEIPT, dict(study_id=r2.STUDY,
        amendment_sha256=r2.sha(AMENDMENT), result_sha256=r2.sha(r2.RESULT),
        source_and_input_sha256=_facts(), completed=True,
        organizer_reference_access=False))
    return result


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['declare','run'])
    action=p.parse_args().action
    declare() if action == 'declare' else run()
