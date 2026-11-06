"""Additive, parity-verified runtime amendment; frozen study sources stay intact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from experiments import v2_iterative_eval as evaluation
from experiments import v2_iterative_train_run as training

ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = ROOT / 'experiments/results/v2_iterative_v1_execution_amendment.json'
PARITY = ROOT / 'experiments/results/v2_iterative_v1_runtime_parity.json'


def facts():
    from experiments import v2_iterative_runtime as runtime
    paths = [Path(__file__), Path(runtime.__file__), PARITY,
             evaluation.DECLARATION, training.PARENT]
    return {str(p.relative_to(ROOT)): training.sha(p) for p in paths}


def freeze():
    parity = json.loads(PARITY.read_text())
    if not parity.get('passed'): raise RuntimeError('Runtime parity must pass first')
    training.once(AMENDMENT, dict(study_id='v2_iterative_v1_execution',
        change='Vectorize float64 2D row-wise linear nanquantiles at q=.5/.9. Preserve every other call through NumPy. Canonically sort aggregation keys during control replay to match saved-cell candidate reduction order and avoid scheduling-dependent roundoff in zero-tolerance comparisons.',
        model_feature_metric_protocol_changes=False,
        original_sources_unchanged=True, bindings=facts(),
        prior_attempt='Slow preparation/control processes stopped and partial artifacts preserved in ignored attempt1_slow directory. Completed TRAIN feature views retained with their original hashes; no completed score receipt reset.',
        frozen_sources=evaluation.sha(evaluation.DECLARATION),
        organizer_reference_access=False))


def run(action, workers):
    from experiments import v2_iterative_runtime as runtime
    amendment=json.loads(AMENDMENT.read_text())
    if amendment['bindings'] != facts(): raise RuntimeError('Runtime amendment bindings changed')
    original_mean = evaluation.capacity.equal_provider
    def canonical_mean(values):
        return original_mean(dict(sorted(values.items())))
    evaluation.capacity.equal_provider = canonical_mean
    try:
        with runtime.installed():
            if action == 'prepare':
                training.bind()
                training.recipe.prepare(workers=workers)
            else:
                evaluation.prepare_controls()
    finally:
        evaluation.capacity.equal_provider = original_mean


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['freeze','prepare','controls'])
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    if args.action=='freeze': freeze()
    else: run(args.action,args.workers)

if __name__=='__main__': main()
