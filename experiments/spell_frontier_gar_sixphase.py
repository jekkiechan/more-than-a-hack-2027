"""Six-phase compatibility for the preserved frozen GAR deployment comparison."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from experiments import spell_frontier_gar_execution as execution
from experiments import spell_frontier_gar as study

AMENDMENT = study.ROOT / 'experiments/results/spell_frontier_gar_v1_sixphase_amendment.json'
TEST = study.ROOT / 'experiments/tests/test_spell_frontier_gar_sixphase.py'
ORIGINAL_POOL = study.package._pool_seed_phase_probabilities


def sixphase_pool(seed_phases):
    values = np.asarray(seed_phases, dtype=np.float64)
    if values.ndim == 4 and values.shape[1] in (2, 5):
        return ORIGINAL_POOL(seed_phases)
    if (values.ndim != 4 or values.shape[0] != 2 or values.shape[1] != 6 or
            values.shape[-1] != 3 or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError('Expected aligned two-seed, six-phase H/A/D probabilities')
    if not np.allclose(values.sum(axis=-1), 1., rtol=0, atol=1e-6):
        raise ValueError('Every probability row must sum to one')
    return values.mean(axis=0).mean(axis=0)


@contextmanager
def installed():
    previous = study.package._pool_seed_phase_probabilities
    study.package._pool_seed_phase_probabilities = sixphase_pool
    try:
        yield
    finally:
        study.package._pool_seed_phase_probabilities = previous


def freeze():
    execution.check()
    log = Path('/tmp/frontier_gar_first.log')
    value = dict(study=study.STUDY, reason='Frozen public pooling helper only accepts 2/5 phases; GAR requires 6. Failure occurred in first clean component pooling before decoding or scoring.',
        declaration_sha256=study.digest(study.DECLARATION),
        runtime_sha256=study.digest(execution.AMENDMENT),
        source_sha256={str(p.relative_to(study.ROOT)):study.digest(p) for p in (Path(__file__), TEST)},
        failed_attempt_log=log.read_text(), failed_attempt_log_sha256=study.digest(log),
        candidate_metrics_saved=False, scoring_started=False,
        correction='Allow exactly six phases using the identical float64 seed mean then phase mean. Existing 2/5 paths call the original helper unchanged.',
        recipe_changed=False, weights_changed=False, seeds_changed=False,
        comparison_changed=False, fitting=False, public_scoring=False)
    study.write_once(AMENDMENT, value)


def check():
    execution.check()
    value=study.read(AMENDMENT)
    for rel, sha in value['source_sha256'].items():
        if study.digest(study.ROOT / rel) != sha:
            raise ValueError('Frozen compatibility source changed: '+rel)
    if value['declaration_sha256'] != study.digest(study.DECLARATION):
        raise ValueError('Frozen GAR declaration changed')


def worker_init():
    execution.worker_init()
    study.package._pool_seed_phase_probabilities = sixphase_pool


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze','first','games'))
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    if args.action=='freeze':
        freeze()
    else:
        check()
        execution.worker_init=worker_init
        execution.run(1 if args.action=='first' else args.workers, first_only=args.action=='first')
