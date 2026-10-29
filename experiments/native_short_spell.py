"""Small conditional H/A residuals over the frozen GRU and native geometry.

This module contains only the core feature/residual contract for the proposed
native-rate short-spell study. It does not load data, fit a research arm, score
matches, or change a frozen runner.
"""
from __future__ import annotations

import numpy as np

from experiments import possession_geometry, supervised


L2 = 0.02
FIT_UPDATES = 512
BATCH_SIZE = 2048
LEARNING_RATE = 0.01
LOGIT_CLIP = 12.0


def conditional_home_logit(probability: np.ndarray) -> np.ndarray:
    """Return frozen prior log(P(H)/P(A)) from finite native H/A/D rows."""
    values = np.asarray(probability, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or
            not np.isfinite(values).all() or (values < 0).any() or
            not np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-5)):
        raise ValueError("Expected normalized H/A/D probabilities")
    tiny = np.finfo(np.float64).tiny
    live_home = np.maximum(values[:, 0], tiny)
    live_away = np.maximum(values[:, 1], tiny)
    return np.clip(np.log(live_home) - np.log(live_away),
                   -LOGIT_CLIP, LOGIT_CLIP)


def preserve_dead_probability(home_logit: np.ndarray,
                              frozen_probability: np.ndarray) -> np.ndarray:
    """Apply an H/A log-odds correction while copying P(D) bit-exactly."""
    logit = np.asarray(home_logit, dtype=np.float64)
    frozen = np.asarray(frozen_probability, dtype=np.float64)
    if (frozen.ndim != 2 or frozen.shape[1] != 3 or
            logit.shape != (len(frozen),) or not np.isfinite(logit).all() or
            not np.isfinite(frozen).all() or (frozen < 0).any() or
            (frozen > 1).any() or
            not np.allclose(frozen.sum(axis=1), 1.0, rtol=0, atol=1e-5)):
        raise ValueError("Expected aligned finite H/A/D probabilities and logits")
    dead = frozen[:, 2].copy()
    # The branch-stable sigmoid avoids overflow for large residuals.
    q_home = np.empty_like(logit)
    positive = logit >= 0
    q_home[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exp_value = np.exp(logit[~positive])
    q_home[~positive] = exp_value / (1.0 + exp_value)
    result = np.empty_like(frozen)
    result[:, 0] = (1.0 - dead) * q_home
    result[:, 1] = (1.0 - dead) - result[:, 0]
    result[:, 2] = dead
    return result


def apply_residual(probability: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Add a residual to the frozen native prior, preserving zero-row parity."""
    frozen = np.asarray(probability, dtype=np.float64)
    correction = np.asarray(residual, dtype=np.float64)
    if (frozen.ndim != 2 or frozen.shape[1] != 3 or
            correction.shape != (len(frozen),) or
            not np.isfinite(correction).all()):
        raise ValueError("Expected aligned native H/A/D probabilities and residual")
    result = frozen.copy()
    changed = correction != 0.0
    if changed.any():
        base_logit = conditional_home_logit(frozen[changed])
        result[changed] = preserve_dead_probability(
            base_logit + correction[changed], frozen[changed])
    return result


def upsample_blockwise(sampled: np.ndarray, indices: np.ndarray,
                       period: np.ndarray) -> np.ndarray:
    """Interpolate one sampled residual per 5 Hz row within each time block."""
    values = np.asarray(sampled, dtype=np.float64)
    indices = np.asarray(indices)
    period = np.asarray(period)
    if (values.ndim != 1 or indices.shape != values.shape or
            not np.issubdtype(indices.dtype, np.integer) or period.ndim != 1 or
            not len(period) or not np.isfinite(values).all()):
        raise ValueError("Expected aligned sampled values, integer indices, and period")
    if ((indices < 0).any() or (indices >= len(period)).any() or
            (np.diff(indices) <= 0).any()):
        raise ValueError("Sample indices must be strictly increasing and in range")
    out = np.zeros(len(period), dtype=np.float64)
    seen = np.zeros(len(period), dtype=bool)
    for start, stop in supervised.blocks(period):
        rows = (indices >= start) & (indices < stop)
        if not rows.any():
            raise ValueError("Every continuous block needs at least one sampled row")
        grid = np.arange(start, stop)
        out[start:stop] = np.interp(grid, indices[rows], values[rows])
        seen[start:stop] = True
    if not seen.all():
        raise ValueError("Period blocks do not cover the native frame grid")
    return out


def equal_support_weights(row_count: int, source_match_count: int,
                          view_count: int = 4, swap_count: int = 2) -> np.ndarray:
    """Give each provider, match, view and team-swap group equal total mass."""
    values = (int(row_count), int(source_match_count), int(view_count), int(swap_count))
    if any(value < 1 for value in values):
        raise ValueError("Support mass requires positive row, match, view and swap counts")
    mass = 1.0 / (values[0] * values[1] * values[2] * values[3])
    return np.full(values[0], mass, dtype=np.float32)


def geometry_features(inputs: dict) -> tuple[np.ndarray, list[str]]:
    """Build native, instantaneous geometry without reliable-contact anchors."""
    matrix, names = possession_geometry.features(inputs, include_anchors=False)
    if tuple(names) != possession_geometry.GEOMETRY_NAMES:
        raise ValueError("Native geometry feature contract changed")
    if matrix.shape != (len(inputs["ball"]), len(names)):
        raise ValueError("Native geometry is not aligned to tracking rows")
    return matrix, names


def _normalization(features: np.ndarray, sample_weight: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(features, dtype=np.float32)
    weight = np.asarray(sample_weight, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("Expected nonempty feature rows")
    if weight.shape != (len(matrix),) or not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Expected positive finite training sample weights")
    impute = np.empty(matrix.shape[1], dtype=np.float64)
    scale = np.empty(matrix.shape[1], dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        if not finite.any():
            raise ValueError("Every geometry feature needs finite training support")
        supported_weight = weight[finite]
        observed = values[finite].astype(np.float64, copy=False)
        total = supported_weight.sum()
        impute[column] = np.dot(observed, supported_weight) / total
        centered = observed - impute[column]
        scale[column] = np.sqrt(np.dot(centered * centered, supported_weight) / total)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    return impute.astype(np.float32), scale.astype(np.float32)


def _normalize(features: np.ndarray, median: np.ndarray,
               scale: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(median):
        raise ValueError("Geometry feature shape differs from fitted normalization")
    output = np.empty((len(matrix), 2 * matrix.shape[1]), dtype=np.float32)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        output[:, column] = np.where(
            finite, np.clip((values - median[column]) / scale[column], -10.0, 10.0), 0.0)
        output[:, matrix.shape[1] + column] = (~finite).astype(np.float32)
    return output


def fit_residual(features: np.ndarray, prior_home_logit: np.ndarray,
                 target_home: np.ndarray, *, backbone_seed: int,
                 sample_weight: np.ndarray | None = None,
                 l2: float = L2, updates: int = FIT_UPDATES,
                 batch_size: int = BATCH_SIZE,
                 learning_rate: float = LEARNING_RATE) -> dict:
    """Fit a train-only linear logistic residual using a frozen prior offset.

    ``target_home`` contains 0/1 only for known H/A rows. Unknown and dead-ball
    rows must be removed by the caller. The fitted scale, weights and bias are
    all learned from the same training fold.
    """
    import torch
    from torch.nn import functional as F

    x = np.asarray(features)
    prior = np.asarray(prior_home_logit, dtype=np.float64)
    target = np.asarray(target_home, dtype=np.float64)
    if (x.ndim != 2 or not len(x) or prior.shape != (len(x),) or
            target.shape != (len(x),) or not np.isfinite(prior).all() or
            not np.isfinite(target).all() or not np.isin(target, (0.0, 1.0)).all()):
        raise ValueError("Expected labeled H/A training rows and finite prior logits")
    weights_np = (np.ones(len(x), dtype=np.float32) if sample_weight is None
                  else np.asarray(sample_weight, dtype=np.float32))
    if (weights_np.shape != (len(x),) or not np.isfinite(weights_np).all() or
            (weights_np <= 0).any()):
        raise ValueError("Expected positive finite sample weights")
    if (not np.isfinite(l2) or l2 < 0 or int(updates) < 1 or
            int(batch_size) < 1 or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("Invalid logistic fit settings")
    median, scale = _normalization(x, weights_np)
    normalized = _normalize(x, median, scale)
    torch.set_num_threads(1)
    seed = int(backbone_seed)
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    torch_x = torch.from_numpy(np.ascontiguousarray(normalized))
    torch_prior = torch.as_tensor(prior, dtype=torch.float32)
    torch_target = torch.as_tensor(target, dtype=torch.float32)
    torch_weight = torch.as_tensor(weights_np, dtype=torch.float32)
    model_weights = torch.nn.Parameter(torch.zeros(normalized.shape[1]))
    bias = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam((model_weights, bias), lr=float(learning_rate))
    last_loss = float("nan")
    for _ in range(int(updates)):
        indices = generator.integers(0, len(x), size=min(int(batch_size), len(x)))
        ix = torch.as_tensor(indices, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        logits = torch_prior[ix] + torch_x[ix] @ model_weights + bias
        row_loss = F.binary_cross_entropy_with_logits(
            logits, torch_target[ix], reduction="none")
        batch_weight = torch_weight[ix]
        loss = (row_loss * batch_weight).sum() / batch_weight.sum()
        loss = loss + 0.5 * float(l2) * torch.sum(model_weights * model_weights)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach())
    fitted_weights = model_weights.detach().cpu().numpy()
    fitted_bias = float(bias.detach())
    if not np.isfinite(fitted_weights).all() or not np.isfinite(fitted_bias):
        raise ValueError("Residual logistic fit returned nonfinite parameters")
    feature_names = list(possession_geometry.GEOMETRY_NAMES)
    expanded_names = feature_names + [f"missing_{name}" for name in feature_names]
    return {
        "kind": "native_geometry_logistic_residual",
        "backbone_seed": int(backbone_seed),
        "geometry_feature_names": feature_names,
        "feature_names": expanded_names,
        "median": median.astype(np.float32),
        "scale": scale.astype(np.float32),
        "weights": fitted_weights.astype(np.float32),
        "bias": fitted_bias,
        "l2": float(l2),
        "optimizer_updates": int(updates),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "training_rows": int(len(target)),
        "training_last_batch_loss": last_loss,
        "missing_values_use_train_fold_imputation_and_finite_mask": True,
    }


def predict_residual(features: np.ndarray, model: dict,
                     feature_names: list[str] | None = None) -> np.ndarray:
    """Apply the sealed linear geometry correction to aligned rows."""
    names = list(feature_names or possession_geometry.GEOMETRY_NAMES)
    if (model.get("kind") != "native_geometry_logistic_residual" or
            names != model.get("geometry_feature_names")):
        raise ValueError("Residual model feature schema mismatch")
    median = np.asarray(model["median"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    if median.shape != scale.shape or weights.shape != (2 * len(median),):
        raise ValueError("Residual model parameter shape mismatch")
    return _normalize(features, median, scale) @ weights + float(model["bias"])


def predict_symmetric_residual(features: np.ndarray, model: dict,
                               feature_names: list[str] | None = None) -> np.ndarray:
    """Enforce exact H/A antisymmetry using the same small head twice."""
    names = list(feature_names or possession_geometry.GEOMETRY_NAMES)
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(names):
        raise ValueError("Geometry rows and feature names do not align")
    swapped = supervised.swap_team_features(matrix, names)
    direct = predict_residual(matrix, model, names)
    opposite = predict_residual(swapped, model, names)
    return 0.5 * (direct - opposite)
