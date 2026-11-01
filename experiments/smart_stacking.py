"""Factorized, team-symmetric probability stacking with bounded temporal context.

Inference accepts probabilities and feature-only time blocks. Provider and match
identifiers, targets, durations and grading masks are never inference features.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import minimize
from scipy.special import expit, logit

from experiments import supervised

PENALTY = .02


def validate(probabilities, period):
    p = np.asarray(probabilities, dtype=np.float64)
    period = np.asarray(period)
    if (p.ndim != 3 or p.shape[1:] != (len(period), 3) or not len(period)
            or period.ndim != 1 or not np.isfinite(p).all()
            or (p < 0).any() or (p.sum(axis=2) <= 0).any()):
        raise ValueError("Expected experts x native frames x H/A/D probabilities")
    return p / p.sum(axis=2, keepdims=True)


def features(probabilities, period, fps, temporal=False):
    p = validate(probabilities, period)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame rate")
    # Symmetric clipping keeps an exact sign change under H/A interchange.
    live = p[:, :, :2].sum(axis=2)
    conditional = np.divide(p[:, :, 0], live, out=np.full_like(live, .5), where=live > 0)
    ha = logit(np.clip(conditional, 1e-6, 1-1e-6)).T
    dead = logit(np.clip(p[:, :, 2], 1e-6, 1-1e-6)).T
    if temporal:
        # Current-minus-local mean can sharpen or soften short evidence without
        # borrowing across a half or timestamp discontinuity.
        smooth = np.empty_like(ha)
        width = 2 * max(1, round(.5*fps)) + 1
        for a, b in supervised.blocks(period):
            smooth[a:b] = uniform_filter1d(ha[a:b], width, axis=0, mode="nearest")
        ha = np.column_stack((ha, ha-smooth))
    return ha, np.column_stack((dead, np.ones(len(period))))


def short_weights(labels, period, fps):
    """Training-only short-spell weights; never used to prepare inference."""
    labels = np.asarray(labels)
    weights = np.ones(len(labels), dtype=float)
    for a, b in supervised.blocks(period):
        ends = np.r_[a, a+1+np.flatnonzero(labels[a+1:b] != labels[a:b-1]), b]
        for start, stop in zip(ends[:-1], ends[1:]):
            if labels[start] in ("H", "A"):
                weights[start:stop] = min(4., max(1., np.sqrt(3./max((stop-start)/fps, .2))))
    return weights


def fit_binary(x, y, weight, anchor):
    x, y, weight, anchor = map(np.asarray, (x, y, weight, anchor))
    if not len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Both training classes required")
    weight = weight / weight.sum()
    def objective(coef):
        value = x @ coef
        residual = coef-anchor
        loss = np.sum(weight*(np.logaddexp(0., value)-y*value)) + .5*PENALTY*np.dot(residual, residual)
        grad = x.T @ (weight*(expit(value)-y)) + PENALTY*residual
        return loss, grad
    fit = minimize(objective, anchor.copy(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
    if not fit.success:
        raise RuntimeError("Stack optimization failed: " + str(fit.message))
    return dict(coef=fit.x.tolist(), objective=float(fit.fun), iterations=int(fit.nit))


def fit(cells, excluded_provider, expert_indices, temporal=False):
    """All cells must use base checkpoints from this SAME outer provider fold.

    Each meta-training match is absent from base TRAIN; the target provider is
    absent from base TRAIN, the teacher/normalizer, and meta-training labels.
    Cell weights balance provider, match, then clean/feed-like view.
    """
    if not cells or any(c["provider"] == excluded_provider for c in cells):
        raise ValueError("Outer provider leaked into meta-training")
    if any(c["outer_provider"] != excluded_provider for c in cells):
        raise ValueError("Mixed outer checkpoint folds would leak provider exposure")
    providers = sorted({c["provider"] for c in cells})
    data = {k: [] for k in ("ha_x", "ha_y", "ha_w", "d_x", "d_y", "d_w")}
    for c in cells:
        p = c["probabilities"][expert_indices]
        hx, dx = features(p, c["model_period"], c["fps"], temporal)
        labels = np.asarray(c["labels"])
        if not np.isin(labels, ("H", "A", "D", "X")).all():
            raise ValueError("Unknown target code")
        # Deterministic native-rate stride; never concatenate time blocks.
        indices = np.concatenate([np.arange(a, b, max(1, round(c["fps"]/5)))
                                  for a, b in supervised.blocks(c["model_period"])])
        count_matches = len({r["match"] for r in cells if r["provider"] == c["provider"]})
        replicas = sum(r["match"] == c["match"] and (r["view"] != "clean") == (c["view"] != "clean") for r in cells)
        mass = 1./len(providers)/count_matches/2/replicas
        sw = short_weights(labels, c["model_period"], c["fps"])
        for head, mask, target in (("ha", np.isin(labels[indices], ("H", "A")), labels[indices] == "H"),
                                   ("d", labels[indices] != "X", labels[indices] == "D")):
            use = indices[mask]
            weight = sw[use] if head == "ha" else np.ones(len(use))
            if not len(use):
                raise ValueError("Meta-training cell lacks graded targets")
            data[head+"_x"].append((hx if head == "ha" else dx)[use])
            data[head+"_y"].append(target[mask].astype(float))
            data[head+"_w"].append(weight/weight.sum()*mass)
    n = len(expert_indices)
    ha_anchor = np.r_[np.full(n, 1./n), np.zeros(n) if temporal else np.empty(0)]
    d_anchor = np.r_[np.full(n, 1./n), 0.]
    model = dict(temporal=temporal, expert_indices=list(expert_indices), excluded_provider=excluded_provider,
                 train_matches=sorted({c["match"] for c in cells}), train_providers=providers)
    for head, anchor in (("ha", ha_anchor), ("d", d_anchor)):
        arrays = [np.concatenate(data[head+suffix]) for suffix in ("_x", "_y", "_w")]
        model[head] = fit_binary(*arrays, anchor)
        model[head]["training_rows"] = len(arrays[1])
    return model


def predict(probabilities, period, fps, model, *, dead_expert=None):
    p = validate(probabilities, period)
    hx, dx = features(p[model["expert_indices"]], period, fps, model["temporal"])
    home = expit(hx @ np.asarray(model["ha"]["coef"]))
    dead = (p[dead_expert, :, 2] if dead_expert is not None
            else expit(dx @ np.asarray(model["d"]["coef"])))
    return np.column_stack((home*(1-dead), (1-home)*(1-dead), dead))


def cross(ha, dead):
    ha = np.asarray(ha, dtype=np.float64)
    dead = np.asarray(dead, dtype=np.float64)
    q = ha[:, :2]/np.maximum(ha[:, :2].sum(axis=1, keepdims=True), 1e-30)
    d = dead[:, 2]/dead.sum(axis=1)
    return np.column_stack((q*(1-d[:, None]), d))
