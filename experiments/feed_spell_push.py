"""Feed-first probability pooling and calibration; no targets enter inference."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.special import expit, logit

from experiments import decoder_refinement as refinement, decoder_search as core, supervised


def configurations():
    arms = [dict(family="reference", dead=d) for d in (6, 12)]
    arms += [dict(family="calibration", dt=dt, ht=ht, ha=h, dead=d)
             for dt in (1.5, 2, 3, 4, 6) for ht in (.65, 1, 1.5)
             for h in (3, 6) for d in (3, 6, 12)]
    arms += [dict(family="dead_bias", bias=b, dt=2, ht=1, ha=3, dead=d)
             for b in (-.5, -.25, .25, .5) for d in (3, 6, 12)]
    arms += [dict(family="filter", kind=k, seconds=s, dt=2, ht=1, ha=3, dead=d)
             for k in ("gaussian", "median") for s in (.1, .3, .6)
             for d in (3, 6, 12)]
    arms += [dict(family="pool", power=r, dt=dt, ht=1, ha=3, dead=d)
             for r in (-1, 0, 2) for dt in (1.5, 3) for d in (3, 6, 12)]
    arms += [dict(family="disagreement", gain=g, dt=dt, ht=1, ha=3, dead=d)
             for g in (1, 3, 9) for dt in (1.5, 3) for d in (3, 6, 12)]
    arms += [dict(family="entropy", gain=g, dt=dt, ht=1, ha=3, dead=d)
             for g in (-.5, .5, 1.5) for dt in (1.5, 3) for d in (3, 6, 12)]
    arms += [dict(family="anchors", confidence=q, seconds=s, dt=dt, ht=1)
             for q in (.5, .6, .7) for s in (.2, .6, 1) for dt in (1.5, 3)]
    for i, c in enumerate(arms):
        c["id"] = f"f{i:03d}"
    return arms


def _validated(members, period, fps):
    x = np.asarray(members, dtype=np.float64)
    period = np.asarray(period)
    if (x.ndim != 3 or x.shape[0] not in (1, 2) or x.shape[1:] != (len(period), 3)
            or len(period) == 0 or period.ndim != 1 or not np.isfinite(x).all()
            or (x < 0).any() or (x.sum(axis=2) <= 0).any()
            or not np.isfinite(fps) or fps <= 0):
        raise ValueError("Expected one or two nonempty native Nx3 probability arrays")
    return x, period


def decode(members, period, fps, config):
    """Pool matching native member probabilities, transform, then segment.

    Only probabilities and inference time breaks are accepted. Neither provider
    identifiers nor observed/unknown ground-truth masks are part of this API.
    A one-member call is a descriptive component-seed diagnostic.
    """
    x, period = _validated(members, period, fps)
    if config["family"] == "reference":
        # Preserve the exact historical arithmetic and implementation.
        return refinement.decode(x.mean(axis=0), period, fps,
            dict(family="dead_temperature", temperature=1.5, ha=3, dead=config["dead"]))
    x = x / x.sum(axis=2, keepdims=True)
    available = x.sum(axis=(0, 1)) > 0
    p = x.mean(axis=0)
    if config["family"] == "pool":
        power = config["power"]
        safe = np.maximum(x, 1e-7)
        p = (np.exp(np.log(safe).mean(axis=0)) if power == 0 else
             np.mean(safe**power, axis=0)**(1/power))
        p[:, ~available] = 0
        p /= p.sum(axis=1, keepdims=True)
    output = np.empty(len(period), dtype="U1")
    for a, b in supervised.blocks(period):
        q = p[a:b].copy()
        if config["family"] == "filter":
            if config["kind"] == "gaussian":
                q = gaussian_filter1d(q, config["seconds"]*fps, axis=0, mode="nearest")
            else:
                q = median_filter(q, size=(2*max(1, round(config["seconds"]*fps))+1, 1), mode="nearest")
            q /= q.sum(axis=1, keepdims=True)
        conditional = q[:, :2] / np.maximum(q[:, :2].sum(axis=1, keepdims=True), 1e-30)
        dtemp = np.full(b-a, config["dt"], dtype=float)
        htemp = np.full(b-a, config["ht"], dtype=float)
        if config["family"] == "entropy":
            entropy = -(conditional*np.log(np.maximum(conditional, 1e-30))).sum(axis=1)/np.log(2)
            dtemp *= 1+config["gain"]*entropy
        elif config["family"] == "disagreement":
            spread = np.abs(x[0, a:b]-x[-1, a:b]).sum(axis=1)/2
            htemp *= 1+config["gain"]*spread
            dtemp *= 1+config["gain"]*spread
        odds = np.log(np.maximum(conditional[:, 0], 1e-7))-np.log(np.maximum(conditional[:, 1], 1e-7))
        home = expit(odds/htemp)
        dead = q[:, 2].copy()
        if available[2] and available[:2].any():
            dead = expit(logit(np.clip(dead, 1e-7, 1-1e-7))/dtemp+config.get("bias", 0))
        transformed = np.column_stack((home*(1-dead), (1-home)*(1-dead), dead))
        transformed[:, ~available] = 0
        transformed /= np.maximum(transformed.sum(axis=1, keepdims=True), 1e-30)
        if config["family"] == "anchors":
            settings = dict(family="anchor_segments", confidence=config["confidence"],
                            seconds=config["seconds"], sigma=0)
            output[a:b] = refinement.decode(transformed, period[a:b], fps, settings)
        else:
            output[a:b] = core.decode(transformed, period[a:b], fps,
                dict(family="matrix", ha=config["ha"], dead=config["dead"]))
    return output
