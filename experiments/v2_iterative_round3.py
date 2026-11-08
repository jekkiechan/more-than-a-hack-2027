"""Final bounded combinations of the selected continuation and archived experts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from experiments import v2_iterative_eval as base
from experiments import v2_iterative_followup as previous

ROOT=base.ROOT
STUDY='v2_iterative_v1_round3'
DECL=ROOT/f'experiments/results/{STUDY}_declaration.json'
RESULT=ROOT/f'experiments/results/{STUDY}.json'
OUT=ROOT/'data/experiments'/STUDY
ARMS=('trained_calibrated_mean','trained_calibrated_log','three_expert_mean','three_expert_log')
sha=base.sha


def combinations(trained, calibrated, adopted):
    """Fixed symmetric conditional-H/A means; accepts no labels or metadata."""
    values=[base._conditional(p) for p in (trained,calibrated,adopted)]
    if any(x.shape != values[0].shape for x in values): raise ValueError('Unaligned experts')
    out={}
    for arm in ARMS:
        members=values if arm.startswith('three_') else values[:2]
        if arm.endswith('_log'):
            log=np.mean([np.log(np.clip(x,1e-12,1.0)) for x in members],axis=0)
            p=np.exp(log-log.max(axis=1,keepdims=True))
        else: p=np.mean(members,axis=0)
        p/=p.sum(axis=1,keepdims=True)
        out[arm]=np.column_stack((p,np.zeros(len(p))))
    return out


def sources():
    paths=[Path(__file__),Path(base.__file__),Path(previous.__file__),
           Path(base.audit.__file__),Path(base.capacity.__file__),
           Path(base.audit.decoder.__file__),Path(base.audit.metrics.__file__),
           ROOT/'evaluation/possession_metrics.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def freeze():
    if DECL.exists() or RESULT.exists(): raise FileExistsError('Final round already frozen')
    r2=json.loads(previous.RESULT.read_text())
    replay=json.loads(base.REPLAY.read_text())
    if not replay['passed']: raise ValueError('Control replay did not pass')
    # Explicit post-round-two development selection, then no further arm tuning.
    selected=max(previous.TRAIN_ARMS,key=lambda a:(
        r2['aggregate'][a+'_fixed_d_ha3']['feed_like_v2']['spell_f1']['equal_provider'],
        r2['aggregate'][a+'_fixed_d_ha3']['feed_like_v2']['macro_f1']['equal_provider']))
    facts={key:r2['candidate_emissions'][selected+'/'+key] for key in replay['emission_cache']}
    for fact in [*facts.values(),*replay['emission_cache'].values()]:
        if sha(ROOT/fact['path'])!=fact['sha256']: raise ValueError('Changed sealed input')
    base._write_once(DECL,dict(study_id=STUDY,round=3,arms=list(ARMS),selected_training_arm=selected,
        selection='Higher unrounded round-two standalone fixed-D v2 spell; macro for exact tie. Exposed development choice, not independent confirmation.',
        selected_arm_retention=r2['strict_success_gates'][selected+'_fixed_d_ha3'],
        question='Do fixed conditional-H/A combinations of the selected continuation and archived experts retain more quality while improving v2 spell?',
        methods='Two-expert (trained/calibrated) and three-expert (trained/calibrated/adopted) uniform arithmetic or log-probability means; fixed adopted decoded D and H/A cost3 for every arm.',
        new_fits=0,new_inference=0,new_parameters=0,cohort_cells=35,seeds=list(base.SEEDS),
        seed_interpretation='Each seed supplies its three expert probabilities; decoded adopted ensemble D mask stays fixed, as in round two.',
        previous_result_sha256=sha(previous.RESULT),control_replay_sha256=sha(base.REPLAY),
        controls=replay['emission_cache'],candidate_emissions=facts,source_sha256=sources(),
        success='Same zero-tolerance clean/v1/v2 six-metric retention and v2 spell strictly >.60.',
        organizer_reference_access=False,public_or_GAR_scoring=False,promotion=False))


def run():
    d=json.loads(DECL.read_text())
    if RESULT.exists(): raise FileExistsError('Final round is complete')
    if d['source_sha256']!=sources() or d['previous_result_sha256']!=sha(previous.RESULT) or d['control_replay_sha256']!=sha(base.REPLAY):
        raise ValueError('Frozen final-round source or evidence changed')
    rows,seeds,artifacts={},{},{}
    for key,fact in sorted(d['controls'].items()):
        control=base._load_control(fact)
        c=d['candidate_emissions'][key]
        if sha(ROOT/c['path'])!=c['sha256']: raise ValueError('Changed continuation emissions')
        with np.load(ROOT/c['path'],allow_pickle=False) as f:
            trained={k:f[k].copy() for k in ('prob__pooled','prob__seed2026','prob__seed2027')}
        groups={'ensemble':combinations(trained['prob__pooled'],control['calibrated'],control['adopted'])}
        for seed in base.SEEDS:
            groups[str(seed)]=combinations(trained[f'prob__seed{seed}'],control[f'adop_seed{seed}'],control[f'cand_seed{seed}'])
        cell={k:control[k] for k in ('labels','period','timestamps','fps')}
        reference=base.audit.score(cell,base.audit.decode(control['adopted'],control['model_period'],control['fps']))['reference_spell_count']
        saved={}
        for member,probs in groups.items():
            scores={}
            for arm,p in probs.items():
                pred=base.fixed_d_prediction(p,control['adopted'],control['model_period'],control['fps'],3.0)
                scores[arm]=base.audit.score(cell,pred)
                if scores[arm]['reference_spell_count']!=reference: raise ValueError('Changed reference population')
                saved[f'pred__{member}__{arm}']=pred
            if member=='ensemble': rows[key]=scores
            else: seeds[f'{key}|{member}']=scores
        path=OUT/(key.replace('|','__')+'.npz')
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**saved)
        artifacts[key]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path))
        print('ROUND3',len(rows),'/35',key,flush=True)
    aggregate,gates,seed_aggregate={},{},{}
    replay=json.loads(base.REPLAY.read_text())
    adopted={v:{m:replay['adopted_aggregate'][v][m]['equal_provider'] for m in base.METRICS} for v,_ in base.VIEWS}
    for arm in ARMS:
        aggregate[arm]={}
        for view,reps in base.VIEWS:
            aggregate[arm][view]={}
            for metric in base.METRICS:
                vals={(p,m,r):row[arm][metric] for key,row in rows.items()
                      for p,m,v,r in [base._split_key(key)] if v==view and r in reps}
                mean,providers=previous._equal_provider(vals)
                aggregate[arm][view][metric]=dict(equal_provider=mean,providers=providers)
        candidate={v:{m:aggregate[arm][v][m]['equal_provider'] for m in base.METRICS} for v,_ in base.VIEWS}
        gates[arm]=base.hard_success(candidate,adopted)
        seed_aggregate[arm]={}
        for seed in base.SEEDS:
            seed_aggregate[arm][str(seed)]={}
            for view,reps in base.VIEWS:
                seed_aggregate[arm][str(seed)][view]={}
                for metric in base.QUALITY_METRICS:
                    vals={}
                    for key,row in seeds.items():
                        c,s=key.rsplit('|',1);p,m,v,r=base._split_key(c)
                        if int(s)==seed and v==view and r in reps: vals[(p,m,r)]=row[arm][metric]
                    mean,providers=previous._equal_provider(vals)
                    seed_aggregate[arm][str(seed)][view][metric]=dict(equal_provider=mean,providers=providers)
    base._write_once(RESULT,dict(study_id=STUDY,round=3,declaration_sha256=sha(DECL),aggregate=aggregate,
        strict_success_gates=gates,matches=rows,seed_metrics=seeds,seed_aggregate=seed_aggregate,
        predictions=artifacts,selected_training_arm=d['selected_training_arm'],organizer_reference_access=False,
        public_or_GAR_scoring=False,promotion=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','run']);a=p.parse_args()
    (freeze if a.action=='freeze' else run)()
if __name__=='__main__': main()
