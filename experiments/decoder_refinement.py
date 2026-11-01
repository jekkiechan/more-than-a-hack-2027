"""Target-independent refinements around the split-transition decoder."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logit

from experiments import decoder_search as core, supervised


def configurations():
    arms = [dict(family="matrix", ha=h, dead=d)
            for h in (2, 2.5, 3, 3.5, 4) for d in (6, 9, 12, 18, 24, 36)]
    arms += [dict(family="dead_duration", ha=3, dead=d, seconds=s)
             for d in (6, 12, 24) for s in (.25, .5, 1, 2)]
    arms += [dict(family="dead_closing", ha=3, dead=d, seconds=s, confidence=q)
             for d in (6, 12, 24) for s in (.2, .5, 1, 2) for q in (.6, .8)]
    arms += [dict(family="dead_temperature", ha=3, dead=d, temperature=t)
             for d in (6, 12, 24) for t in (.5, .75, 1.25, 1.5)]
    arms += [dict(family="dead_filter", ha=3, dead=d, seconds=s)
             for d in (6, 12, 24) for s in (.15, .3, .6, 1.2)]
    arms += [dict(family="anchor_segments", confidence=q, seconds=s, sigma=g)
             for q in (.55, .7, .85) for s in (.1, .3, .6) for g in (0, .2)]
    arms += [dict(family="total_variation", seconds=s, norm=norm, iterations=100)
             for s in (.05, .1, .2, .4, .8) for norm in ("component", "joint")]
    for i, config in enumerate(arms):
        config["id"] = f"r{i:03d}"
    return arms


def decode(probabilities, model_period, fps, config):
    if config["family"] in ("anchor_segments", "total_variation"):
        return _non_viterbi(probabilities, model_period, fps, config)
    # Validate through the shared inference API before specialized transforms.
    base = dict(family="matrix", ha=config["ha"], dead=config["dead"])
    if config["family"] == "matrix":
        return core.decode(probabilities, model_period, fps, base)
    p = np.asarray(probabilities, dtype=np.float64)
    period = np.asarray(model_period)
    baseline = core.decode(p, period, fps, base)
    p = p / p.sum(axis=1, keepdims=True)
    available = p.sum(axis=0) > 0
    result = baseline.copy()
    for a, b in supervised.blocks(period):
        q = p[a:b].copy()
        family = config["family"]
        if family == "dead_closing":
            original = baseline[a:b]
            bounds = np.r_[0, np.flatnonzero(original[1:] != original[:-1])+1, len(original)]
            for i in range(1, len(bounds)-2):
                u, v = bounds[i:i+2]
                if (original[u] == "D" and original[u-1] == original[v]
                        and original[v] in ("H", "A") and v-u <= config["seconds"]*fps
                        and q[u:v, 2].mean() < config["confidence"]):
                    result[a+u:a+v] = original[v]
        elif family == "dead_duration":
            classes = np.flatnonzero(available)
            minimum = np.array([1, 1, max(1, round(config["seconds"]*fps))])
            logp = np.log(np.maximum(q, 1e-7))
            costs = core._costs(config["ha"], config["dead"], fps)
            path = core._minimum_duration(logp[:, classes], costs[np.ix_(classes, classes)], minimum[classes])
            result[a:b] = supervised.LABELS[classes[path]]
        elif family in ("dead_temperature", "dead_filter"):
            dead = q[:, 2]
            if available[2] and available[:2].any():
                dead = (expit(logit(np.clip(dead, 1e-7, 1-1e-7))/config["temperature"])
                        if family == "dead_temperature" else
                        gaussian_filter1d(dead, config["seconds"]*fps, mode="nearest"))
            # Leave the conditional H/A evidence unchanged, including missing classes.
            conditional = q[:, :2] / np.maximum(q[:, :2].sum(axis=1, keepdims=True), 1e-30)
            transformed = np.column_stack((conditional*(1-dead[:, None]), dead))
            result[a:b] = core.decode(transformed, period[a:b], fps, base)
        else:
            raise ValueError(f"Unknown refinement family: {family}")
    return result


def _non_viterbi(probabilities, model_period, fps, config):
    p = np.asarray(probabilities, dtype=np.float64)
    period = np.asarray(model_period)
    if (p.ndim != 2 or p.shape != (len(period),3) or len(p)==0 or period.ndim!=1
            or not np.isfinite(p).all() or (p<0).any() or (p.sum(axis=1)<=0).any()
            or not np.isfinite(fps) or fps<=0):
        raise ValueError("Invalid probabilities, periods or fps")
    p = p / p.sum(axis=1,keepdims=True)
    available = p.sum(axis=0)>0
    output = np.empty(len(p),dtype=np.int8)
    for a,b in supervised.blocks(period):
        q = p[a:b].copy()
        if config["family"] == "total_variation":
            # Projected dual ascent for 1/2 ||q-p||² + lambda * TV(q).
            # component: sum of per-class absolute gradients; joint: L2 edge norm.
            dual = np.zeros((max(len(q)-1,0),3))
            penalty = config["seconds"]*fps
            for _ in range(config["iterations"]):
                adjoint = np.zeros_like(q)
                adjoint[:-1] -= dual
                adjoint[1:] += dual
                q = p[a:b]-adjoint
                dual += .249*(q[1:]-q[:-1])
                if config["norm"] == "component":
                    np.clip(dual,-penalty,penalty,out=dual)
                else:
                    dual /= np.maximum(1,np.linalg.norm(dual,axis=1,keepdims=True)/penalty)
            adjoint = np.zeros_like(q)
            adjoint[:-1] -= dual
            adjoint[1:] += dual
            q = p[a:b]-adjoint
            q[:,~available] = -np.inf
            output[a:b] = np.argmax(q,axis=1)
        else:
            if config["sigma"]>0:
                q = gaussian_filter1d(q,config["sigma"]*fps,axis=0,mode="nearest")
            top = np.argmax(q,axis=1)
            confident = np.max(q,axis=1)>=config["confidence"]
            # An anchor must persist; isolated confident frames cannot create a spell.
            bounds=np.r_[0,np.flatnonzero((top[1:]!=top[:-1]) | (confident[1:]!=confident[:-1]))+1,len(q)]
            anchors=[]
            for u,v in zip(bounds[:-1],bounds[1:]):
                if confident[u] and v-u>=max(1,round(config["seconds"]*fps)):
                    anchors.append((int(u),int(v),int(top[u])))
            if not anchors:
                # No confident segment: choose the largest integrated probability.
                output[a:b] = np.argmax(q.sum(axis=0))
                continue
            path = np.full(len(q),anchors[0][2],dtype=np.int8)
            for left,right in zip(anchors[:-1],anchors[1:]):
                u,v,old=left
                x,y,new=right
                if old==new:
                    path[v:y]=new
                    continue
                # One boundary in this uncertain interval, maximizing soft evidence.
                odds=np.log(np.maximum(q[v:x,old],1e-7))-np.log(np.maximum(q[v:x,new],1e-7))
                split=v+int(np.argmax(np.r_[0,np.cumsum(odds)]))
                path[v:split]=old
                path[split:y]=new
            path[anchors[-1][1]:]=anchors[-1][2]
            output[a:b]=path
    return supervised.LABELS[output]
