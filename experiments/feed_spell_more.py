"""Additional bounded mechanisms: logit denoising, recipe diversity and consensus."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit, logit
from experiments import feed_spell_push as initial, decoder_search as core, supervised

BASE = dict(family="base", dt=4, ha=3, dead=3)


def configurations(round_number, parent=None, other=None):
    arms=[BASE.copy()]
    if round_number==3:
        arms += [dict(family="entropy",dt=t,gain=g,ha=3,dead=d)
                 for t in (3,4,5) for g in (-.5,.5,1.5,3) for d in (1.5,3,6)]
        arms += [dict(family="logit_filter",dt=t,ha=3,dead=3,hs=h,ds=d)
                 for t in (3,4) for h in (0,.15,.4,.8) for d in (0,.15,.4,.8) if h or d]
        arms += [dict(family="edge",dt=4,ha=h,dead=d,gain=g)
                 for h in (2,3) for d in (2,3,6) for g in (-.75,-.25,.5,1.5)]
    elif round_number==4:
        arms += [dict(family="legacy",dt=t,ha=3,dead=3,weight=w,scope=s,pooling=p)
                 for t in (3,4,5) for w in (.1,.25,.5,.75)
                 for s in ("team","dead","joint") for p in ("arithmetic","geometric")]
        arms += [dict(family="legacy",dt=4,ha=3,dead=3,weight=w,scope="joint",pooling="arithmetic",parent=parent)
                 for w in (.1,.25,.5)]
        arms += [dict(family="seed_weight",dt=4,ha=3,dead=3,weight=w) for w in (.25,.75)]
    elif round_number==5:
        arms += [dict(family="consensus",left=parent,right=other,weight=w,pooling=p,ha=h,dead=3)
                 for w in (.25,.5,.75) for p in ("arithmetic","geometric") for h in (2,3,4)]
        arms += [dict(family="duration",parent=parent,ha=h,dead=d,hs=s,ds=t)
                 for h in (2,3) for d in (2,3) for s in (.1,.3,.6) for t in (.2,.5,1)]
        arms += [dict(family="cap",parent=parent,cap=c,ha=h,dead=3)
                 for c in (.5,1,2,4) for h in (1.5,3,6)]
    else:raise ValueError("Expected round 3, 4 or 5")
    for i,c in enumerate(arms):c['id']=f'm{round_number}_{i:03d}'
    return arms


def _pool(p,q,w,kind):
    available=(p+q).sum(axis=0)>0
    out=((1-w)*p+w*q if kind=="arithmetic" else
         np.exp((1-w)*np.log(np.maximum(p,1e-7))+w*np.log(np.maximum(q,1e-7))))
    out[:,~available]=0
    return out/out.sum(axis=1,keepdims=True)


def emissions(members,period,fps,c,legacy=None):
    x,period=initial._validated(members,period,fps)
    x=x/x.sum(axis=2,keepdims=True);p=x.mean(axis=0)
    family=c['family']
    if family=="consensus":
        return _pool(emissions(members,period,fps,c['left'],legacy),
                     emissions(members,period,fps,c['right'],legacy),c['weight'],c['pooling'])
    if family in ("duration","cap"):
        p=emissions(members,period,fps,c['parent'],legacy)
        if family=="cap":
            odds=np.clip(np.log(np.maximum(p[:,0],1e-7))-np.log(np.maximum(p[:,1],1e-7)),-c['cap'],c['cap'])
            home=expit(odds);d=p[:,2];available=p.sum(axis=0)>0
            p=np.column_stack((home*(1-d),(1-home)*(1-d),d));p[:,~available]=0;p/=p.sum(axis=1,keepdims=True)
        return p
    if family=="seed_weight":p=(1-c['weight'])*x[0]+c['weight']*x[-1]
    if family=="legacy":
        if legacy is None:raise ValueError("Recipe diversity requires paired legacy probabilities")
        y,_=initial._validated(legacy,period,fps);q=y.mean(axis=0);q/=q.sum(axis=1,keepdims=True)
        if c.get('parent'):
            p=emissions(members,period,fps,c['parent'],legacy)
        blend=_pool(p,q,c['weight'],c['pooling'])
        if c['scope']=="team":
            d=p[:,2];conditional=blend[:,:2]/np.maximum(blend[:,:2].sum(axis=1,keepdims=True),1e-30)
            p=np.column_stack((conditional*(1-d[:,None]),d))
        elif c['scope']=="dead":
            conditional=p[:,:2]/np.maximum(p[:,:2].sum(axis=1,keepdims=True),1e-30);d=blend[:,2]
            p=np.column_stack((conditional*(1-d[:,None]),d))
        else:p=blend
        if c.get('parent'):return p
    available=p.sum(axis=0)>0
    transformed=np.empty_like(p)
    for a,b in supervised.blocks(period):
        q=p[a:b];odds=np.log(np.maximum(q[:,0],1e-7))-np.log(np.maximum(q[:,1],1e-7))
        dlog=logit(np.clip(q[:,2],1e-7,1-1e-7))
        if family=="logit_filter":
            if c['hs']:odds=gaussian_filter1d(odds,c['hs']*fps,mode='nearest')
            if c['ds']:dlog=gaussian_filter1d(dlog,c['ds']*fps,mode='nearest')
        temperature=np.full(b-a,c.get('dt',4),dtype=float)
        if family=="entropy":
            conditional=q[:,:2]/np.maximum(q[:,:2].sum(axis=1,keepdims=True),1e-30)
            entropy=-(conditional*np.log(np.maximum(conditional,1e-30))).sum(axis=1)/np.log(2)
            temperature*=1+c['gain']*entropy
        dead=expit(dlog/temperature) if available[2] and available[:2].any() else q[:,2]
        home=expit(odds)
        transformed[a:b]=np.column_stack((home*(1-dead),(1-home)*(1-dead),dead))
    transformed[:,~available]=0
    return transformed/transformed.sum(axis=1,keepdims=True)


def decode(members,period,fps,c,legacy=None):
    p=emissions(members,period,fps,c,legacy)
    available=p.sum(axis=0)>0;out=np.empty(len(period),dtype='U1')
    for a,b in supervised.blocks(period):
        q=p[a:b];logp=np.log(np.maximum(q,1e-7));logp[:,~available]=-np.inf
        cost=core._costs(c.get('ha',3),c.get('dead',3),fps)
        if c['family']=="duration":
            indices=np.flatnonzero(available)
            minimum=np.array([max(1,round(c['hs']*fps)),max(1,round(c['hs']*fps)),max(1,round(c['ds']*fps))])
            path=indices[core._minimum_duration(logp[:,indices],cost[np.ix_(indices,indices)],minimum[indices])]
        else:
            edge=np.ones(b-a)
            if c['family']=="edge":
                entropy=-(q*np.log(np.maximum(q,1e-30))).sum(axis=1)/np.log(3)
                edge=1+c['gain']*entropy
            path=core._path(logp,cost,edge)
        out[a:b]=supervised.LABELS[path]
    return out
