"""Read-only multiclass tree structure and training-loss audit.

Requires every declared full H/A/D checkpoint. Reads no evaluation scores or
labels, does not predict, and never changes a fitted model or declaration.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    values=np.asarray(values,dtype=float)
    values=values[np.isfinite(values)]
    if not len(values):return None
    return {f'p{p}':float(np.percentile(values,p)) for p in (0,10,25,50,75,90,100)}


def leaf_rows(node,depth=0):
    if 'left_child' not in node:
        return [(depth,int(node.get('leaf_count',0)),float(node.get('leaf_weight',0.)))]
    return leaf_rows(node['left_child'],depth+1)+leaf_rows(node['right_child'],depth+1)


def weighted_mean(values,weights):
    return float(np.average(values,weights=weights)) if np.sum(weights)>0 else None


def structure_summary(trees,leaves,leaf_cap,depth_cap,min_rows):
    leaves=np.asarray(leaves,dtype=float)
    depth,count,hessian=leaves.T
    depth_hist={str(int(d)):sum(t['maximum_depth']==d for t in trees)
                for d in sorted({t['maximum_depth'] for t in trees})}
    return dict(class_trees=len(trees),total_leaves=len(leaves),
        leaves_per_tree=quantiles([t['leaves'] for t in trees]),
        maximum_depth_per_tree=quantiles([t['maximum_depth'] for t in trees]),
        tree_maximum_depth_counts=depth_hist,
        fraction_trees_at_leaf_cap=float(np.mean([t['leaves']==leaf_cap for t in trees])),
        fraction_trees_at_depth_cap=float(np.mean([t['maximum_depth']==depth_cap for t in trees])),
        all_leaf_depth=quantiles(depth),leaf_training_row_count=quantiles(count),
        leaf_training_hessian_sum=quantiles(hessian),
        row_count_weighted_leaf_depth=weighted_mean(depth,count),
        hessian_weighted_leaf_depth=weighted_mean(depth,hessian),
        per_tree_row_count_weighted_leaf_depth=quantiles([t['row_depth'] for t in trees if t['row_depth'] is not None]),
        fraction_leaves_below_min_child_samples=float(np.mean(count<min_rows)),
        fraction_leaves_at_or_below_1_25_times_min_child_samples=float(np.mean(count<=1.25*min_rows)),
        fraction_leaves_at_or_below_2_times_min_child_samples=float(np.mean(count<=2*min_rows)),
        first_48_class_trees_maximum_depth=quantiles([t['maximum_depth'] for t in trees[:48]]),
        last_48_class_trees_maximum_depth=quantiles([t['maximum_depth'] for t in trees[-48:]]))


def loss_summary(values):
    loss=np.asarray(values,dtype=float)
    if not len(loss) or not np.isfinite(loss).all():raise ValueError('Invalid training loss history')
    rounds=len(loss)
    checkpoints=sorted({1,rounds,*[i for i in (64,128,256,512) if i<=rounds]})
    result=dict(rounds=rounds,training_multi_logloss_by_round={str(r):float(loss[r-1]) for r in checkpoints},
        first=float(loss[0]),last=float(loss[-1]),
        minimum=float(loss.min()),minimum_round=int(loss.argmin()+1),
        increases_between_successive_rounds=int(np.sum(np.diff(loss)>0)))
    if rounds>64:
        start=rounds-65
        result['last_64_rounds']=dict(start_round=start+1,end_round=rounds,
            absolute_decrease=float(loss[start]-loss[-1]),
            relative_decrease_percent=float(100*(loss[start]-loss[-1])/loss[start]) if loss[start]>0 else None)
    return result


def audit_model(path,spec):
    receipt_path=path.with_suffix('.json')
    receipt=json.loads(receipt_path.read_text())
    if receipt['sha256']!=digest(path) or receipt['predeclared_sha256']!=digest(path.parent/'predeclared.json'):
        raise ValueError(f'Checkpoint provenance differs: {path.name}')
    if sorted(receipt['training_matches'])!=sorted(spec['training_matches'][receipt['heldout']]):
        raise ValueError('Provider training cohort differs')
    fitted=joblib.load(path)
    config=fitted['config']
    settings=dict(spec['common_parameters'],**{k:v for k,v in spec['configs'][config].items() if k!='family'})
    actual=fitted['estimator'].get_params()
    for key,value in settings.items():
        if actual.get(key)!=value:raise ValueError(f'Fitted parameter differs: {key}')
    if fitted['feature_manifest']!=spec['feature_manifest'][fitted['family']]:
        raise ValueError('Fitted feature manifest differs')
    booster=fitted['estimator'].booster_
    rounds=booster.current_iteration()
    total=booster.num_trees()
    if total!=3*rounds or receipt['boosting_rounds']!=rounds or receipt['class_trees']!=total:
        raise ValueError('Unexpected three-class tree count')
    if list(fitted['estimator'].classes_)!=[0,1,2]:raise ValueError('Class order differs')
    if fitted['history']['training']['multi_logloss']!=receipt['training_loss']:
        raise ValueError('Training loss receipt differs from fitted model')
    trees={k:[] for k in ('H','A','D')}
    leaves={k:[] for k in trees}
    ordered_trees=[]
    # Chunk by boosting iterations to avoid loading up to half a million nested
    # split dictionaries at once for a 127-leaf, 1,920-tree model.
    for start in range(0,rounds,64):
        dumped=booster.dump_model(start_iteration=start,num_iteration=min(64,rounds-start))
        if dumped['num_tree_per_iteration']!=3:raise ValueError('Expected multiclass dump')
        for offset,tree in enumerate(dumped['tree_info']):
            label=('H','A','D')[offset%3]
            ls=leaf_rows(tree['tree_structure'])
            depth=np.array([r[0] for r in ls]);count=np.array([r[1] for r in ls])
            if len(ls)!=tree['num_leaves'] or depth.max()>settings['max_depth']:
                raise ValueError('Tree shape differs from configuration')
            tree_summary=dict(leaves=len(ls),maximum_depth=int(depth.max()),
                              row_depth=weighted_mean(depth,count))
            trees[label].append(tree_summary)
            ordered_trees.append(tree_summary)
            leaves[label].extend(ls)
        del dumped
    if sum(len(v) for v in trees.values())!=total:raise ValueError('Incomplete dumped tree coverage')
    summary_args=(settings['num_leaves'],settings['max_depth'],settings['min_child_samples'])
    result=dict(file=str(path.relative_to(ROOT)),model_sha256=digest(path),
        receipt_sha256=digest(receipt_path),heldout=receipt['heldout'],config=config,family=fitted['family'],
        feature_count=len(fitted['feature_manifest']),training_rows=fitted['training_rows'],
        training_matches=receipt['training_matches'],parameters=settings,
        boosting_rounds=rounds,class_trees=total,
        overall=structure_summary(ordered_trees,sum(leaves.values(),[]),*summary_args),
        by_class={k:structure_summary(trees[k],leaves[k],*summary_args) for k in trees},
        training_loss=loss_summary(receipt['training_loss']))
    del fitted,booster,trees,leaves,ordered_trees
    gc.collect()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,default=ROOT/'data/experiments/joint_lgbm_v1')
    p.add_argument('--out',type=Path,default=ROOT/'experiments/results/joint_lgbm_capacity_audit.json')
    args=p.parse_args();directory=args.models.resolve()
    if args.out.exists():raise ValueError('Refusing to overwrite completed audit')
    spec=json.loads((directory/'predeclared.json').read_text())
    paths=[directory/f'without_{source}_{config}.joblib' for source in spec['training_matches'] for config in spec['configs']]
    if any(not p.exists() or not p.with_suffix('.json').exists() for p in paths):
        raise ValueError('All declared checkpoints and receipts must exist before auditing')
    rows=[]
    for path in paths:
        row=audit_model(path,spec);rows.append(row)
        s=row['overall']
        print(row['heldout'],row['config'],'trees',row['class_trees'],
              'leaf_cap_fraction',round(s['fraction_trees_at_leaf_cap'],4),
              'depth_cap_fraction',round(s['fraction_trees_at_depth_cap'],4),
              'row_weighted_depth',None if s['row_count_weighted_leaf_depth'] is None else round(s['row_count_weighted_leaf_depth'],4),
              'loss',row['training_loss']['first'],row['training_loss']['last'],flush=True)
    result=dict(scope='Full H/A/D multiclass tree structure and training loss only; no evaluation scores, predictions, labels or fits',
        predeclared_sha256=digest(directory/'predeclared.json'),audit_source_sha256=digest(Path(__file__)),
        classes_in_tree_order=['H','A','D'],dump_chunk_boosting_rounds=64,model_rows=rows,
        interpretation=['Leaf_count measures correlated training rows, not independent matches or unique seconds.',
            'Leaf_weight is a Hessian sum. Hessian-weighted depth is not exact raw sample-weighted depth.',
            'Class-time sample weights, duplicated training views and bagging affect support interpretation.',
            'Reaching a tree cap or continuing training-loss improvement does not establish underfitting or heldout benefit.',
            'Training-loss minima are descriptive; all checkpoints retain their predeclared fixed budget.'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
