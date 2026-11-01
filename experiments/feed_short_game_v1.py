"""Conditional game-level GAR short-window access after within-record sampling."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import numpy as np
import joblib
from experiments import feed_short_sampling_v1 as previous,feed_short_head_v1 as previous_head
from experiments import feed_short_sampler as sampler,feed_head_transfer_v1 as cross
from experiments.feed_short_game_sampler import GameResampler
p=previous.p
STUDY='feed_short_game_v1'
OUT=p.ROOT/'data/experiments'/STUDY
DECL=p.RESULTS/f'{STUDY}_declaration.json'
AUDIT=p.RESULTS/f'{STUDY}_exposure.json'
ARMS=('game_sampling','game_ha')


def declare():
    previous.check()
    if (p.RESULTS/f'{previous.STUDY}_aggregate.json').exists():raise ValueError('Freeze before first-round scores')
    audit=p.read(AUDIT)
    for row in audit['rows']:
        old=next(r for r in p.read(previous.AUDIT)['rows'] if r['fold']==row['fold'] and r['seed']==row['seed'])
        assert row['game_sampling']['counts']['old']==old['targeted']['counts']['old']
        assert row['game_sampling']['exposures']==old['targeted']['exposures']
        assert row['game_sampling']['eligible_gar_games']==45
    paths=[Path(__file__),p.ROOT/'experiments/feed_short_game_sampler.py',p.ROOT/'experiments/tests/test_feed_short_game_sampler.py',p.ROOT/'scripts/audit_feed_short_game_v1.py',AUDIT]
    p.write(DECL,{'study':STUDY,'utc':datetime.now(timezone.utc).isoformat(),
        'condition':'Run six fits only if within-record sampling and its fixed H/A cross both miss .55 under guards.',
        'hypothesis':'Record fragmentation limits GAR short-window access. Sampling within the selected game supplies eligible windows while preserving game balance.',
        'intervention':'Original-provider sampler exactly identical to within-record round. GAR draw with probability .25 replaced by uniform eligible existing window from same game, across records. Same complete <=3s native eligibility. Base and within-record RNG consumption preserved; independent game RNG seed+884422.',
        'fits':6,'seeds':list(p.SEEDS),'folds':list(p.FOLDS),'steps':2048,
        'fixed':'Same teachers, normalizers, raw-feed views, features, targets, losses, width96, batch32, 50/50 sources, HA lr3e-4/D lr1e-4 and f062 decoder. Window distribution changes both heads. Target-derived sampling is TRAIN-only.',
        'arms':{'game_sampling':'Equal probability ensemble of both fixed seeds','game_ha':'Conditional H/A from game_sampling with raw-feed control D, cross within seed then mean'},
        'controls':'Raw-feed augmented .525240 plus within-record results. Primary original feed spell; macro tie-break. Same promoted-baseline guards and separately reported historical clean screen.',
        'stop':'Six sealed fits, two fixed arms, no replacement-rate/duration/seed/decoder/blend search. No extra-corruption selection or organizer reference scoring.',
        'evidence':'Same five exposed development matches, synthetic stress, silver labels. Repeated TRAIN examples, no new supervision or independent evaluation.',
        'parent_declaration_sha256':p.sha(previous.DECL),'head_declaration_sha256':p.sha(previous_head.DECL),
        'bindings':{str(x.relative_to(p.ROOT)):p.sha(x) for x in paths},'organizer_reference_access':False,'promotion':False})


def check():
    previous.check();d=p.read(DECL)
    assert d['parent_declaration_sha256']==p.sha(previous.DECL) and d['head_declaration_sha256']==p.sha(previous_head.DECL)
    for rel,digest in d['bindings'].items():assert p.sha(p.ROOT/rel)==digest,rel
    return d


def fit(fold,seed):
    check();assert fold in p.FOLDS and seed in p.SEEDS
    parent=p.read(p.RESULTS/f'{previous.STUDY}_aggregate.json');head=p.read(p.RESULTS/f'{previous_head.STUDY}_aggregate.json')
    assert not parent['stretch_reached'] and not (head['stretch_reached'] and head['guards_pass'])
    assert p.read(p.RESULTS/f'{previous.STUDY}_verification.json')['passed'] and p.read(p.RESULTS/f'{previous_head.STUDY}_verification.json')['passed']
    path=OUT/'models'/f'{fold}_{seed}.joblib';p.write(path.with_suffix('.lock.json'),{'declaration_sha256':p.sha(DECL)})
    teacher=joblib.load(p.sc.initial_path(fold,seed));old=p.training(fold);gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS)
    sampling=GameResampler(old,gar,list(teacher['feature_names']),seed);start=time.monotonic()
    with sampler.installed(sampling):
        result=p.sn.train(teacher,old,gar,width=96,lr=3e-4,seed=seed,steps=2048,fraction=.5,batch_size=32,ha_target='dense',train_d=True,require_full_native=True,log_every=256,verbose=True)
    expected=next(r['game_sampling'] for r in p.read(AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)
    assert sampling.receipt()==expected and result['exposures']==expected['exposures']
    assert not path.exists();joblib.dump(result['model'],path)
    p.write(path.with_suffix('.json'),{'fold':fold,'seed':seed,'sha256':p.sha(path),'declaration_sha256':p.sha(DECL),'wall_seconds':time.monotonic()-start,'config':result['config'],'sampling':sampling.receipt(),'exposures':result['exposures'],'old_matches':sorted({str(r['match']) for r in old})})
    print('FIT COMPLETE',fold,seed,flush=True)


def seal():
    check();files={}
    for f in p.FOLDS:
        for s in p.SEEDS:
            path=OUT/'models'/f'{f}_{s}.joblib';receipt=p.read(path.with_suffix('.json'))
            assert p.sha(path)==receipt['sha256'] and receipt['declaration_sha256']==p.sha(DECL)
            files[str(path.relative_to(p.ROOT))]=p.sha(path)
    p.write(p.RESULTS/f'{STUDY}_model_seal.json',{'models':files,'declaration_sha256':p.sha(DECL)})


def evaluate():
    check();sealed=p.read(p.RESULTS/f'{STUDY}_model_seal.json')
    for rel,digest in sealed['models'].items():assert p.sha(p.ROOT/rel)==digest
    p.write(OUT/'score_lock.json',{'declaration_sha256':p.sha(DECL),'model_seal_sha256':p.sha(p.RESULTS/f'{STUDY}_model_seal.json')})
    cells=p.shared._load_cells(p.read(p.DECLARATION));values={a:{} for a in ARMS};rows={a:[] for a in ARMS};files={}
    for f in p.FOLDS:
        for seed in p.SEEDS:
            model=joblib.load(OUT/'models'/f'{f}_{seed}.joblib');teacher=joblib.load(p.sc.initial_path(f,seed))
            for c in cells:
                if 'leave_out_'+c['provider'].lower()!=f or c['member']!=str(seed):continue
                key=tuple(c[k] for k in ('provider','match','view','member'));prob=p.infer(c,model,teacher)
                values['game_sampling'][key]=prob;values['game_ha'][key]=cross.cross(prob,previous.parent_probability(c))
    for a in ARMS:
        for c in cells:
            key=tuple(c[k] for k in ('provider','match','view','member'))
            prob=np.mean([values[a][key[:3]+(str(seed),)] for seed in p.SEEDS],axis=0) if c['member']=='ensemble' else values[a][key]
            row,pred=p.score(c,prob);rows[a].append(row)
            path=OUT/'predictions/original'/a/('__'.join(key)+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError(path)
            np.savez_compressed(path,probabilities=prob,prediction=pred,**{k:c[k] for k in ('labels','period','timestamps','model_period','fps')});files[str(path.relative_to(p.ROOT))]=p.sha(path)
    summaries={a:{m:p.scoring.summarize(rows[a],m) for m in p.shared.MEMBERS} for a in ARMS}
    parent=p.read(p.RESULTS/f'{previous.STUDY}_aggregate.json');baseline=parent['baseline'];control=parent['control'];guards={a:previous.selection.eligible(s['ensemble'],baseline) for a,s in summaries.items()}
    eligible=[a for a in ARMS if guards[a] and previous.selection.rank(summaries[a]['ensemble'])>previous.selection.rank(control)]
    winner=max(eligible,key=lambda a:previous.selection.rank(summaries[a]['ensemble'])) if eligible else None
    p.write(p.RESULTS/f'{STUDY}_aggregate.json',{'study':STUDY,'rows':rows,'summaries':summaries,'baseline':baseline,'control':control,'guards_pass':guards,'winner':winner,'stretch_reached':bool(winner and summaries[winner]['ensemble']['feed_like']['equal_provider_mean']['spell_f1']>=.55),'historical_clean_screen_pass':{a:s['ensemble']['clean']['equal_provider_mean']['spell_f1']>=baseline['clean']['equal_provider_mean']['spell_f1']+.01 for a,s in summaries.items()},'prediction_sha256':files,'organizer_reference_access':False})
    for a,s in summaries.items():print(a,{v:s['ensemble'][v]['equal_provider_mean']['spell_f1'] for v in ('clean','feed_like')},guards[a],flush=True)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('action',choices=('declare','fit','seal','evaluate'));q.add_argument('--fold',choices=p.FOLDS);q.add_argument('--seed',type=int,choices=p.SEEDS);a=q.parse_args();fit(a.fold,a.seed) if a.action=='fit' else globals()[a.action]()
