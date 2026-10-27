"""Full H/A/D LightGBM with curated, ordered possession evidence.

Inference accepts only feature records; labels are used solely by the explicit
training sampler. Team swapping precedes every nonlinear temporal transform.
"""
from __future__ import annotations
import numpy as np
from experiments import supervised
from experiments.temporal_lgbm import _window_stats

RATE=5
ROUNDS=640
CONFIGS={
    'legacy_63':dict(family='legacy',num_leaves=63,max_depth=8),
    'geometry_63':dict(family='geometry',num_leaves=63,max_depth=8),
    'full_63':dict(family='full',num_leaves=63,max_depth=8),
    'full_127':dict(family='full',num_leaves=127,max_depth=10),
}
COMMON=dict(objective='multiclass',num_class=3,learning_rate=.01,n_estimators=ROUNDS,
    min_child_samples=120,min_child_weight=.01,reg_alpha=.2,reg_lambda=10.,
    colsample_bytree=.7,subsample=.8,subsample_freq=1,max_bin=127,
    n_jobs=1,random_state=2026,deterministic=True,force_col_wise=True,verbosity=-1)
CORE_NAMES=(
    'ball_observed','physical_ball_speed','physical_ball_acceleration','physical_ball_turn',
    'missing_before_s','missing_after_s','home_distance_advantage',
    'H_distance1','A_distance1','H_physical_nearest1_radial_closing','A_physical_nearest1_radial_closing',
    'H_physical_nearest1_speed_alignment','A_physical_nearest1_speed_alignment',
    'D_speed_q0.5','D_slow_fraction_2.0','D_fast_fraction_4.0','D_motion_coherence',
    'D_valid_motion_fraction','D_team_centroid_distance','D_touchline_fraction',
)
# New physical names are finalized with possession_geometry before declaration.
EXTRA_NAMES=(
    'H_geometry_best_control','A_geometry_best_control','home_geometry_control_margin','geometry_control_ambiguity',
    'H_geometry_approach_miss_distance_m','A_geometry_approach_miss_distance_m',
    'H_geometry_approach_time_s','A_geometry_approach_time_s',
    'geometry_collective_acceleration_median','geometry_collective_braking_fraction','geometry_collective_accelerating_fraction',
)


def raw_swap_name(name):
    if name.startswith('H_'):return 'A_'+name[2:],1
    if name.startswith('A_'):return 'H_'+name[2:],1
    if name.startswith('home_'):return name,-1
    return name,1


def sampling(y, period, stride=4, edge_samples=3):
    """Retain every tiny spell and 5Hz run edges; thin interiors to1.25Hz.

    Integer Voronoi cells lie within the same known-target run and input block.
    Their sizes preserve *exactly* each original known class/time mass; unknown
    targets contribute zero. This is a training-only stratified approximation.
    """
    y=np.asarray(y)
    if y.ndim!=1 or len(y)!=len(period) or stride<1 or edge_samples<1 or not np.isin(y,[-1,0,1,2]).all():
        raise ValueError('Invalid training labels, blocks or sampling policy')
    if not len(y):return np.empty(0,np.int64),np.empty(0,float)
    rows=[];masses=[]
    for a,b in supervised.blocks(period):
        edges=np.r_[a,np.flatnonzero(y[a+1:b]!=y[a:b-1])+a+1,b]
        for start,stop in zip(edges[:-1],edges[1:]):
            if y[start]<0:continue
            selected=np.unique(np.r_[np.arange(start,stop,stride),
                np.arange(start,min(stop,start+edge_samples)),
                np.arange(max(start,stop-edge_samples),stop)])
            cuts=(selected[:-1]+selected[1:])//2+1
            mass=np.diff(np.r_[start,cuts,stop])
            rows.extend(selected.tolist());masses.extend(mass.tolist())
    return np.asarray(rows,dtype=np.int64),np.asarray(masses,dtype=np.float64)


def _selected_raw(record,family):
    from experiments import possession_geometry as geometry
    names=list(record['names']);base=int(record.get('base_count',204))
    if names[base:] != list(geometry.GEOMETRY_NAMES)+list(geometry.ANCHOR_NAMES):
        raise ValueError('Joint raw geometry/anchor schema mismatch')
    n=base if family=='legacy' else base+len(geometry.GEOMETRY_NAMES) if family=='geometry' else len(names)
    return np.asarray(record['x'][:,:n],np.float32),names[:n]


def design(record,family):
    """Return matrix plus exact family/swap metadata for every feature."""
    if family not in ('legacy','geometry','full'):raise ValueError('Unknown feature family')
    x,names=_selected_raw(record,family)
    parts=[x];manifest=[]
    for name in names:
        partner,sign=raw_swap_name(name)
        from experiments import possession_geometry as geometry
        group='legacy' if len(manifest)<204 else 'geometry' if name in geometry.GEOMETRY_NAMES else 'reliable_contact_anchor'
        manifest.append(dict(name=name,family=group,swap_name=partner,swap_sign=sign))
    if family!='full':return x,manifest
    index={n:i for i,n in enumerate(names)}
    selected=CORE_NAMES+EXTRA_NAMES
    if len(set(selected))!=len(selected) or any(n not in index for n in selected):
        raise ValueError('Curated temporal schema missing or duplicated')
    primitive=x[:,[index[n] for n in selected]]
    period=record['period']
    def add(values,operation,seconds=None,even=False):
        parts.append(np.asarray(values,np.float32))
        suffix=operation if seconds is None else f'{operation}_{seconds:g}s'
        for name in selected:
            partner,sign=raw_swap_name(name)
            manifest.append(dict(name=f'context__{name}__{suffix}',family=f'temporal_{operation}',
                swap_name=f'context__{partner}__{suffix}',swap_sign=1 if even else sign))
    add(np.isfinite(primitive),'observed',even=True)
    for seconds in (.4,1.,3.):
        lag=round(seconds*RATE);before=np.full_like(primitive,np.nan);after=np.full_like(primitive,np.nan)
        for a,b in supervised.blocks(period):
            if b-a>lag:
                before[a+lag:b]=primitive[a:b-lag];after[a:b-lag]=primitive[a+lag:b]
        add(before,'before',seconds);add(after,'after',seconds);add(after-before,'after_minus_before',seconds)
    for seconds in (1.,3.):
        width=round(seconds*RATE)
        pm,fm,ps,fs=[np.full_like(primitive,np.nan) for _ in range(4)]
        pc,fc=[np.zeros_like(primitive) for _ in range(2)]
        for a,b in supervised.blocks(period):
            n=b-a;at=np.arange(n)
            m,s,c=_window_stats(primitive[a:b],np.maximum(0,at-width),at)
            pm[a:b],ps[a:b],pc[a:b]=m,s,c/width
            m,s,c=_window_stats(primitive[a:b],np.minimum(n,at+1),np.minimum(n,at+1+width))
            fm[a:b],fs[a:b],fc[a:b]=m,s,c/width
        for value,op,even in ((pm,'past_mean',False),(fm,'future_mean',False),(fm-pm,'future_minus_past_mean',False),
            (ps,'past_std',True),(fs,'future_std',True),(pc,'past_coverage',True),(fc,'future_coverage',True)):
            add(value,op,seconds,even)
    # Team-level persistence uses real control evidence, not persistent player IDs.
    home=x[:,index['H_geometry_best_control']];away=x[:,index['A_geometry_best_control']]
    valid=np.isfinite(home)&np.isfinite(away)
    margin=home-away
    winners=np.column_stack((np.where(valid,margin>.05,np.nan),np.where(valid,margin<-.05,np.nan)))
    for seconds in (1.,3.,5.):
        width=round(seconds*RATE)
        for side in ('past','future'):
            means=np.full((len(x),2),np.nan);coverage=np.zeros(len(x))
            for a,b in supervised.blocks(period):
                n=b-a;at=np.arange(n)
                lo,hi=(np.maximum(0,at-width),at) if side=='past' else (np.minimum(n,at+1),np.minimum(n,at+1+width))
                mean,_,count=_window_stats(winners[a:b],lo,hi)
                means[a:b]=mean;coverage[a:b]=count[:,0]/width
            parts.extend((means,coverage[:,None]))
            for team,other in (('H','A'),('A','H')):
                manifest.append(dict(name=f'{team}_control_{side}_fraction_{seconds:g}s',family='control_persistence',
                    swap_name=f'{other}_control_{side}_fraction_{seconds:g}s',swap_sign=1))
            name=f'control_{side}_coverage_{seconds:g}s'
            manifest.append(dict(name=name,family='control_persistence_coverage',swap_name=name,swap_sign=1))
    matrix=np.column_stack(parts).astype(np.float32);matrix[~np.isfinite(matrix)]=np.nan
    if matrix.shape[1]!=len(manifest) or len({r['name'] for r in manifest})!=len(manifest):raise ValueError('Feature manifest mismatch')
    return matrix,manifest


def training_design(record,family):
    """Engineer coherent original/alternate views before selecting output cores."""
    x,manifest=design(record,family)
    alternate,other=design(dict(record,x=record['alternate']),family)
    if manifest!=other or np.asarray(record['replace_rows']).shape!=(len(x),):
        raise ValueError('Training view/schema mismatch')
    return np.where(record['replace_rows'][:,None],alternate,x),manifest


def swap_engineered(x,manifest):
    """Explicit parity helper for verification; prediction swaps the raw inputs."""
    columns={r['name']:i for i,r in enumerate(manifest)}
    return np.column_stack([x[:,columns[r['swap_name']]]*r['swap_sign'] for r in manifest])


def swapped_record(record):
    return dict(record,x=supervised.swap_team_features(record['x'],record['names']))


def predict(record,fitted):
    probabilities=[]
    for swapped in (False,True):
        r=swapped_record(record) if swapped else record
        x,manifest=design(r,fitted['family'])
        if manifest!=fitted['feature_manifest']:raise ValueError('Prediction schema changed')
        p=fitted['estimator'].predict_proba(x)
        if not np.array_equal(fitted['estimator'].classes_,[0,1,2]):raise ValueError('Expected class order H/A/D')
        probabilities.append(p[:,[1,0,2]] if swapped else p)
    return (probabilities[0]+probabilities[1])*.5


def blend(tree,gru,tree_weight=.5):
    tree,gru=np.asarray(tree),np.asarray(gru)
    if tree.shape!=gru.shape or tree.ndim!=2 or tree.shape[1]!=3 or not 0<=tree_weight<=1:
        raise ValueError('Expected aligned three-class probabilities')
    for p in (tree,gru):
        if not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1):raise ValueError('Invalid probabilities')
    return tree_weight*tree+(1-tree_weight)*gru
