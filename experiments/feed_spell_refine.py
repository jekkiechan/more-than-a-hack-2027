"""Second, explicitly adaptive feed-first round; calibration order is isolated."""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit

from experiments import feed_spell_push as first


def configurations():
    arms = [dict(family="reference", dead=12),
            dict(family="calibration", dt=4, ht=1, ha=3, dead=3)]
    arms += [dict(family="calibration", dt=t, bias=b, ht=1, ha=h, dead=d)
             for t in (2.5, 3.5, 4.5, 5.5) for b in (-.15, 0, .15)
             for h in (2, 3) for d in (2, 3, 4)]
    arms += [dict(family="member_calibration", dt=t, ht=1, ha=h, dead=d)
             for t in (3, 4, 5) for h in (2, 3) for d in (2, 3, 4)]
    for i, c in enumerate(arms):
        c["id"] = f"g{i:03d}"
    return arms


def decode(members, period, fps, config):
    if config["family"] != "member_calibration":
        return first.decode(members, period, fps, config)
    x, period = first._validated(members, period, fps)
    x = x/x.sum(axis=2,keepdims=True)
    available = x.sum(axis=(0,1)) > 0
    if available[2] and available[:2].any():
        conditional = x[:,:,:2]/np.maximum(x[:,:,:2].sum(axis=2,keepdims=True),1e-30)
        dead = expit(logit(np.clip(x[:,:,2],1e-7,1-1e-7))/config["dt"])
        x = np.concatenate((conditional*(1-dead[:,:,None]),dead[:,:,None]),axis=2)
        x[:,:,~available] = 0
        x /= x.sum(axis=2,keepdims=True)
    # Calibration has already happened independently in each seed.
    return first.decode(x, period, fps, dict(config, family="calibration", dt=1))
