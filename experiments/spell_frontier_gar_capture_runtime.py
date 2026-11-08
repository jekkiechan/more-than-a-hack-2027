"""Apply the already verified exact quantile runtime during GAR input capture."""
from __future__ import annotations
import argparse
from pathlib import Path
from experiments import spell_frontier_gar_sixphase as fix
from experiments import v2_iterative_runtime as runtime

study, execution = fix.study, fix.execution
AMENDMENT=study.ROOT/'experiments/results/spell_frontier_gar_v1_capture_runtime.json'
ORIGINAL_CAPTURE=study.heldout.capture_inputs


def capture(*args, **kwargs):
    with runtime.installed():
        return ORIGINAL_CAPTURE(*args, **kwargs)


def worker_init():
    fix.worker_init()
    study.heldout.capture_inputs=capture


def freeze():
    fix.check()
    study.write_once(AMENDMENT, dict(study=study.STUDY,
        reason='Input capture repeats dense feature assembly before the already-patched phase feature path. Apply the same exact quantile acceleration used for record preparation and phase features during capture too.',
        source_sha256={str(p.relative_to(study.ROOT)):study.digest(p) for p in (Path(__file__),Path(runtime.__file__),Path(fix.__file__))},
        compatibility_amendment_sha256=study.digest(fix.AMENDMENT),
        declaration_sha256=study.digest(study.DECLARATION),
        data_or_features_changed=False, metric_or_tolerance_changed=False,
        exact_clean_phase0_feature_guard_retained=True,
        exact_all_game_predecessor_score_guards_retained=True,
        fitting=False, public_scoring=False, max_workers=4))


def check():
    fix.check()
    amendment=study.read(AMENDMENT)
    for rel,digest in amendment['source_sha256'].items():
        if study.digest(study.ROOT/rel)!=digest:raise ValueError('Frozen capture runtime changed: '+rel)
    if study.digest(fix.AMENDMENT)!=amendment['compatibility_amendment_sha256']:
        raise ValueError('Compatibility amendment changed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('freeze','games'))
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.action=='freeze':freeze()
    else:
        check()
        execution.worker_init=worker_init
        execution.run(args.workers)
