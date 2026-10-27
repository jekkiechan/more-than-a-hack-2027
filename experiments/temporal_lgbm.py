"""Offline, block-safe player-only temporal features and bounded LightGBM heads.

The feature API accepts sampled tracking features only: neither labels nor ball
inputs can enter the dead-ball estimator. Models retain exact feature schemas.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
from experiments import learned_temporal as lt, supervised

RATE = 5
# One matched feature ablation, then two capacity/regularization hypotheses.
CONFIGS = {
    'raw_slow': dict(engineered=False, learning_rate=.02, n_estimators=640,
                     num_leaves=31, min_child_samples=40, reg_alpha=0., reg_lambda=1.,
                     colsample_bytree=1., subsample=1., subsample_freq=0, max_bin=255),
    'context_slow': dict(engineered=True, learning_rate=.02, n_estimators=640,
                         num_leaves=31, min_child_samples=40, reg_alpha=0., reg_lambda=1.,
                         colsample_bytree=1., subsample=1., subsample_freq=0, max_bin=255),
    'context_shallow': dict(engineered=True, learning_rate=.03, n_estimators=480,
                            num_leaves=15, max_depth=5, min_child_samples=160,
                            min_child_weight=.01, reg_alpha=.1, reg_lambda=5.,
                            colsample_bytree=.8, subsample=.8, subsample_freq=1, max_bin=127),
    'context_regularized': dict(engineered=True, learning_rate=.02, n_estimators=640,
                                num_leaves=31, max_depth=7, min_child_samples=120,
                                min_child_weight=.01, reg_alpha=.2, reg_lambda=10.,
                                colsample_bytree=.7, subsample=.8, subsample_freq=1, max_bin=127),
}


def _window_stats(x, lo, hi):
    """Finite-only half-open window statistics from prefix sums, no imputation."""
    valid = np.isfinite(x)
    value = np.where(valid, x, 0).astype(np.float64)
    def sums(a):
        prefix = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(a, axis=0)))
        return prefix[hi]-prefix[lo]
    count = sums(valid)
    total, squares = sums(value), sums(value*value)
    mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    var = np.divide(squares, count, out=np.full_like(total, np.nan), where=count > 0)-mean*mean
    std = np.sqrt(np.maximum(var, 0))
    # At block edges the denominator remains the requested width: unavailable
    # context is distinguished from a complete window containing measured zeros.
    return mean, std, count


def features(x, names, period):
    """Add direction, onset/offset and observation-support features at 5 Hz.

    ``period`` must encode continuous input blocks, as model_periods does. Each
    half window excludes the target. Out-of-block lags stay NaN, never repeat an
    endpoint. Centered rolling source columns are retained but not re-expanded.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != len(names) or len(x) != len(period):
        raise ValueError('Feature dimensions disagree')
    if len(set(names)) != len(names):
        raise ValueError('Feature names must be unique')
    raw = [i for i, name in enumerate(names) if '_mean_' not in name]
    if not raw:
        raise ValueError('No primitive player features')
    primitive = x[:, raw]
    output, output_names = [x], list(names)
    def append(values, suffix):
        output.append(values)
        output_names.extend(f'{names[i]}__{suffix}' for i in raw)
    append(np.isfinite(primitive).astype(np.float32), 'observed')
    for seconds in (1, 3, 5):
        lag = seconds*RATE
        before = np.full_like(primitive, np.nan)
        after = np.full_like(primitive, np.nan)
        past_mean, future_mean, past_std, future_std = [np.full_like(primitive, np.nan) for _ in range(4)]
        past_coverage, future_coverage = [np.zeros_like(primitive) for _ in range(2)]
        for a, b in supervised.blocks(period):
            n = b-a
            if n > lag:
                before[a+lag:b] = primitive[a:b-lag]
                after[a:b-lag] = primitive[a+lag:b]
            at = np.arange(n)
            pm, ps, pc = _window_stats(primitive[a:b], np.maximum(0, at-lag), at)
            fm, fs, fc = _window_stats(primitive[a:b], np.minimum(n, at+1), np.minimum(n, at+1+lag))
            past_mean[a:b], past_std[a:b], past_coverage[a:b] = pm, ps, pc/lag
            future_mean[a:b], future_std[a:b], future_coverage[a:b] = fm, fs, fc/lag
        for value, label in ((before,'before'), (after,'after'), (after-before,'after_minus_before'),
                             (past_mean,'past_mean'), (future_mean,'future_mean'),
                             (future_mean-past_mean,'future_minus_past_mean'),
                             (past_std,'past_std'), (future_std,'future_std'),
                             (past_coverage,'past_coverage'), (future_coverage,'future_coverage')):
            append(value, f'{label}_{seconds}s')
    result = np.column_stack(output).astype(np.float32)
    result[~np.isfinite(result)] = np.nan
    return result, output_names


def design(record, engineered=True):
    if engineered:
        return features(record['x'], record['names'], record['period'])
    return np.asarray(record['x'], np.float32), list(record['names'])


def matrix(records, engineered, stride=2):
    """Equal mass per provider, match and view; preserve context before thinning.

    Training uses every second sampled frame (2.5 Hz); inference stays 5 Hz.
    This bound applies identically to all four configurations.
    """
    if not records or stride < 1:
        raise ValueError('Nonempty records and positive stride required')
    counts = Counter((r['source'], r['match']) for r in records)
    matches = Counter(s for s, _ in counts)
    xs, ys, weights = [], [], []
    names = None
    for r in records:
        xx, schema = design(r, engineered)
        if names is not None and schema != names:
            raise ValueError('Training feature schema differs')
        names = schema
        select = np.zeros(len(xx), bool)
        for a, b in supervised.blocks(r['period']):
            select[a:b:stride] = True
        select &= np.asarray(r['y']) >= 0
        if not select.any():
            raise ValueError('Training view has no known targets')
        yy = np.asarray(r['y'])[select]
        if not np.isin(yy, (0, 1)).all():
            raise ValueError('Expected binary dead-ball targets')
        xs.append(xx[select]); ys.append(yy)
        mass = 1/(matches[r['source']]*counts[(r['source'], r['match'])])
        weights.append(np.full(select.sum(), mass/select.sum()))
    w = np.concatenate(weights)
    return np.concatenate(xs), np.concatenate(ys), w/w.mean(), names


def fit(records, config, seed=2026):
    from lightgbm import LGBMClassifier
    params = dict(CONFIGS[config])
    engineered = params.pop('engineered')
    x, y, weight, names = matrix(records, engineered)
    if len(np.unique(y)) != 2:
        raise ValueError('Both training classes required')
    estimator = LGBMClassifier(objective='binary', random_state=seed, n_jobs=1,
        deterministic=True, force_col_wise=True, verbosity=-1, **params)
    estimator.fit(x, y, sample_weight=weight)
    return dict(kind='temporal_lgbm', config=config, seed=seed, estimator=estimator,
                engineered=engineered, feature_names=names, training_rows=len(y), params=params)


def predict(record, model):
    x, names = design(record, model['engineered'])
    if names != model['feature_names']:
        raise ValueError('Prediction feature schema differs')
    return model['estimator'].predict_proba(x)[:, 1]


def fuse(gru, dead, weight):
    """Blend only P(D), preserving the GRU's conditional live-team attribution."""
    gru, dead = np.asarray(gru), np.asarray(dead)
    if gru.ndim != 2 or gru.shape[1] != 3 or dead.shape != (len(gru),):
        raise ValueError('Expected aligned H/A/D and D probabilities')
    if not 0 <= weight <= 1 or not np.isfinite(gru).all() or not np.isfinite(dead).all():
        raise ValueError('Invalid blend')
    if np.any(gru < 0) or np.any(dead < 0) or np.any(dead > 1) or not np.allclose(gru.sum(1), 1):
        raise ValueError('Invalid probabilities')
    live = gru[:, :2].sum(1)
    conditional = np.divide(gru[:, :2], live[:, None], out=np.full((len(gru), 2), .5), where=live[:, None] > 1e-12)
    p = (1-weight)*gru[:, 2] + weight*dead
    return np.column_stack((conditional*(1-p[:, None]), p))
