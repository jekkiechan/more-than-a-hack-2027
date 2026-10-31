"""Read-only tree structure and sampled training-match support audit.

Never fit, score labels, or change a frozen model. Match support is a bounded
sample diagnostic, not an estimate of independent observations in a leaf.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import supervised, temporal_lgbm

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def percentiles(values):
    values = np.asarray(values)
    return {str(q): float(np.percentile(values, q)) for q in (0, 10, 25, 50, 75, 90, 100)} if len(values) else None


def leaves(node, depth=0):
    if 'leaf_index' in node:
        return [dict(index=int(node['leaf_index']), depth=depth,
                     training_count=int(node.get('leaf_count', 0)),
                     training_hessian=float(node.get('leaf_weight', 0)))]
    return leaves(node['left_child'], depth+1)+leaves(node['right_child'], depth+1)


def structure(path):
    receipt_path = path.with_suffix('.json')
    receipt = json.loads(receipt_path.read_text())
    if sha256(path) != receipt['sha256']:
        raise ValueError(f'Model hash mismatch: {path}')
    if sha256(path.parent/'predeclared.json') != receipt['predeclared_sha256']:
        raise ValueError(f'Model declaration mismatch: {path}')
    model = joblib.load(path)
    dump = model['estimator'].booster_.dump_model()
    params = model['params']
    leaf_cap = params['num_leaves']
    depth_cap = params.get('max_depth', -1)
    support_floor = params['min_child_samples']
    all_leaves = []
    trees = []
    for tree in dump['tree_info']:
        ls = leaves(tree['tree_structure'])
        count = sum(l['training_count'] for l in ls)
        trees.append(dict(tree_index=int(tree['tree_index']), leaves=len(ls),
                          max_depth=max(l['depth'] for l in ls),
                          training_count=count,
                          training_count_weighted_leaf_depth=sum(l['training_count']*l['depth'] for l in ls)/count))
        all_leaves.extend(ls)
    counts = np.array([l['training_count'] for l in all_leaves])
    depths = np.array([l['depth'] for l in all_leaves])
    result = dict(file=str(path.relative_to(ROOT)), model_sha256=sha256(path),
        receipt_sha256=sha256(receipt_path), predeclared_sha256=receipt['predeclared_sha256'],
        heldout=receipt['heldout'], config=model['config'], feature_count=len(model['feature_names']),
        training_matches=receipt['training_matches'], training_rows=model['training_rows'],
        params=params, n_trees=len(trees),
        tree_leaf_count_percentiles=percentiles([t['leaves'] for t in trees]),
        tree_max_depth_percentiles=percentiles([t['max_depth'] for t in trees]),
        tree_weighted_leaf_depth_percentiles=percentiles([t['training_count_weighted_leaf_depth'] for t in trees]),
        leaf_depth_percentiles=percentiles(depths), leaf_training_count_percentiles=percentiles(counts),
        fraction_trees_at_leaf_cap=float(np.mean([t['leaves']==leaf_cap for t in trees])),
        fraction_trees_at_depth_cap=float(np.mean([t['max_depth']==depth_cap for t in trees])) if depth_cap>0 else None,
        fraction_leaves_below_min_child_samples=float(np.mean(counts<support_floor)),
        fraction_leaves_at_or_below_1_25_times_min_child_samples=float(np.mean(counts<=1.25*support_floor)),
        fraction_leaves_at_or_below_2_times_min_child_samples=float(np.mean(counts<=2*support_floor)),
        training_count_weighted_mean_leaf_depth=float(np.average(depths, weights=counts)),
        early_tree_max_depth_percentiles=percentiles([t['max_depth'] for t in trees[:16]]),
        late_tree_max_depth_percentiles=percentiles([t['max_depth'] for t in trees[-16:]]))
    del dump, all_leaves
    return result, model


def sampled_match_support(model, receipt, cache_dir, max_rows=1000, trees_per_end=16):
    booster = model['estimator'].booster_
    total_trees = booster.num_trees()
    starts = (0, total_trees-trees_per_end)
    if total_trees < 2*trees_per_end:
        raise ValueError('Not enough trees for disjoint first/last samples')
    match_ids = receipt['training_matches']
    if len(set(match_ids)) != len(match_ids):
        raise ValueError('Duplicate training match')
    tree_ids = [i for start in starts for i in range(start, start+trees_per_end)]
    n_leaves = model['params']['num_leaves']
    # Counts are sampled assignments, separately for each original match.
    assignment = np.zeros((len(tree_ids), n_leaves, len(match_ids)), dtype=np.int32)
    samples, cache_receipts = [], []
    for mi, match in enumerate(match_ids):
        path = cache_dir/f'train_{match}.joblib'
        meta = json.loads(path.with_suffix('.json').read_text())
        if meta['sha256'] != sha256(path) or meta['predeclared_sha256'] != sha256(cache_dir/'predeclared.json'):
            raise ValueError(f'Cache integrity mismatch: {match}')
        records = joblib.load(path)
        cache_receipts.append(dict(match=match, sha256=meta['sha256']))
        for record in records:
            if record['match'] != match or record['source'] == receipt['heldout']:
                raise ValueError('Training record identity or provider isolation mismatch')
            eligible = np.zeros(len(record['x']), dtype=bool)
            for a,b in supervised.blocks(record['period']):
                eligible[a:b:2] = True
            eligible &= np.asarray(record['y']) >= 0
            indices = np.flatnonzero(eligible)
            selected = indices[np.linspace(0, len(indices)-1, min(max_rows,len(indices)), dtype=int)]
            # Context is computed on the full single-record 5 Hz grid before
            # selecting rows, exactly as in training. Never stack all matches.
            x, schema = temporal_lgbm.design(record, model['engineered'])
            if schema != model['feature_names']:
                raise ValueError('Feature schema mismatch')
            chosen = np.ascontiguousarray(x[selected]); del x
            offset=0
            for start in starts:
                hits = booster.predict(chosen, pred_leaf=True, start_iteration=start,
                                       num_iteration=trees_per_end, num_threads=1)
                if hits.shape != (len(selected), trees_per_end):
                    raise ValueError('Unexpected leaf prediction dimensions')
                for ti in range(trees_per_end):
                    assignment[offset+ti,:,mi] += np.bincount(hits[:,ti], minlength=n_leaves)
                offset += trees_per_end
            samples.append(dict(match=match,source=record['source'],variant=record['variant'],
                                eligible_training_rows=len(indices),sampled_rows=len(selected)))
            del chosen
        del records
        gc.collect()
    dump = booster.dump_model()
    selected_tree_leaf_counts = {t['tree_index']:t['num_leaves'] for t in dump['tree_info'] if t['tree_index'] in tree_ids}
    groups = {}
    for name, offset in [('first_16',0),('last_16',trees_per_end),('all_32',None)]:
        positions = range(len(tree_ids)) if offset is None else range(offset,offset+trees_per_end)
        arrays = [assignment[i,:selected_tree_leaf_counts[tree_ids[i]]] for i in positions]
        counts = np.concatenate(arrays)
        seen = counts.sum(1)>0
        coverage = (counts>0).sum(1)
        dominance = counts[seen].max(1)/counts[seen].sum(1)
        groups[name] = dict(n_leaves=len(counts),observed_leaves=int(seen.sum()),
            unseen_leaf_fraction=float((~seen).mean()),
            distinct_sampled_matches_percentiles=percentiles(coverage[seen]),
            observed_leaf_fraction_seen_in_one_match=float(np.mean(coverage[seen]==1)),
            observed_leaf_fraction_seen_in_at_most_two_matches=float(np.mean(coverage[seen]<=2)),
            observed_leaf_fraction_seen_in_all_training_matches=float(np.mean(coverage[seen]==len(match_ids))),
            dominant_match_sample_share_percentiles=percentiles(dominance),
            observed_leaf_fraction_over_80pct_samples_from_one_match=float(np.mean(dominance>.8)),
            sample_assignment_weighted_distinct_matches=float(np.average(coverage[seen],weights=counts[seen].sum(1))))
    return dict(tree_indices=tree_ids, max_rows_per_match_view=max_rows,
        selection='Evenly spaced among exact training-eligible stride2 rows per match/view; context built before selecting.',
        interpretation='Distinct match coverage among sampled rows. Missing sampled support does not prove no full-training support; match count does not count independent spells. Clean/dropout views remain the same match.',
        samples=samples, feature_cache_receipts=cache_receipts, groups=groups)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,nargs='+',required=True)
    p.add_argument('--cache-dir',type=Path,default=ROOT/'data/experiments/temporal_lgbm_v1')
    p.add_argument('--sample-regularized',action='store_true')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite completed audit')
    rows=[]
    for path in sorted(p.resolve() for p in args.models):
        row,model=structure(path)
        if args.sample_regularized and model['config']=='context_regularized':
            row['sampled_match_support']=sampled_match_support(model,json.loads(path.with_suffix('.json').read_text()),args.cache_dir)
        rows.append(row)
        print(path.stem,json.dumps({k:row[k] for k in ['n_trees','tree_max_depth_percentiles','fraction_trees_at_leaf_cap','fraction_trees_at_depth_cap']}),flush=True)
        del model
        gc.collect()
    result=dict(scope='Read-only post-fit tree-capacity audit; no labels scored and no fits run',
        audit_source_sha256=sha256(Path(__file__)),model_rows=rows,
        support_caveat='LightGBM leaf_count reflects training rows in each tree, with bagging and repeated/augmented correlated observations. min_child_samples is Hessian-approximated; occasional lower counts are permitted. Row counts are not independent matches, spells, or unique seconds.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
