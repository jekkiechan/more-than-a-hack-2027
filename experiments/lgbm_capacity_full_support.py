"""Exact match coverage on all eligible training rows for 32 selected trees.

Stream one training match/view at a time. This is an additional read-only audit,
not a fit, prediction evaluation, or modification of a frozen declaration.
"""
import json
from pathlib import Path

from experiments.lgbm_capacity_audit import ROOT, sha256, structure, sampled_match_support


def main():
    out=ROOT/'experiments/results/lgbm_capacity_full_support.json'
    if out.exists():
        raise ValueError('Refusing to overwrite completed audit')
    cache=ROOT/'data/experiments/temporal_lgbm_v1'
    rows=[]
    for path in sorted(cache.glob('without_*_context_regularized.joblib')):
        row,model=structure(path)
        # The helper's evenly-spaced selection becomes the complete training
        # index set whenever the requested bound exceeds its length.
        support=sampled_match_support(model,json.loads(path.with_suffix('.json').read_text()),
                                      cache,max_rows=10**9)
        if not all(r['sampled_rows']==r['eligible_training_rows'] for r in support['samples']):
            raise ValueError('Incomplete training row coverage')
        support['selection']='ALL exact training-eligible stride2 rows, both views of every training match; first16 and last16 trees only.'
        support['interpretation']='Exact full-training-input match coverage for selected trees, including each tree\'s out-of-bag rows. This is routing support, not the per-tree in-bag fit population, independent spell count, or heldout generalization.'
        support['all_eligible_training_rows_used']=True
        row['selected_tree_full_row_match_support']=support
        rows.append(row)
        print(path.name,json.dumps(support['groups']['all_32']),flush=True)
        del model
    if len(rows)!=3:
        raise ValueError('Expected all three regularized provider folds')
    result=dict(scope='Exact full-row training-input routing support, first and last16 trees, three regularized models; no scoring/fitting',
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'experiments/lgbm_capacity_audit.py']},
        model_rows=rows)
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
