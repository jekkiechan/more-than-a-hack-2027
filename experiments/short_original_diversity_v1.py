"""Conditional original-provider possession diversity with fixed GAR quality bags."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
import joblib
import numpy as np

from experiments import short_quality_diversity_v1 as parent
from experiments.short_original_diversity import OriginalDiversitySampler
p=parent.p
STUDY='short_original_diversity_v1'
OUT=p.ROOT/'data/experiments'/STUDY
DECL=p.RESULTS/f'{STUDY}_declaration.json'
AUDIT=p.RESULTS/f'{STUDY}_training_audit.json'
ARMS=('original_full','original_ha','original_leader_half')


def audit():
    parent.check();gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS);rows=[]
    for fold in p.FOLDS:
        old=p.training(fold);names=list(joblib.load(p.sc.initial_path(fold,2026))['feature_names'])
        for seed in p.SEEDS:
            sampler=OriginalDiversitySampler(old,gar,names,seed)
            result=parent.base.replay(sampler,seed)
            before=next(r for r in p.read(parent.AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)['sampling']['diversity']
            assert result['exposures']==before['exposures']
            assert result['counts']['new']==before['counts']['new']
            for key in ('replacement_distinct_possessions','replacement_distinct_windows','replacement_effective_possessions'):
                assert result[key]==before[key]
            rows.append(dict(fold=fold,seed=seed,sampling=result))
            print('AUDIT',fold,seed,flush=True)
    p.write(AUDIT,dict(train_only=True,rows=rows,source_sha256=p.sha(Path(__file__).with_name('short_original_diversity.py'))))


def declare():
    parent.check();assert not (p.RESULTS/f'{parent.STUDY}_aggregate.json').exists()
    paths=[Path(__file__),Path(__file__).with_name('short_original_diversity.py'),AUDIT,
           p.ROOT/'experiments/tests/test_short_original_diversity.py']
    p.write(DECL,dict(study=STUDY,utc=datetime.now(timezone.utc).isoformat(),
        condition='Run only if the five fixed quality/diversity parent arms fail to reach .55 under guards. Frozen before parent scores.',
        question='Does cycling distinct native short original-provider possessions improve over window-uniform original sampling, with the GAR quality/diversity recipe fixed?',
        intervention='With probability .25 replace original draw with a window overlapping the next unique complete <=3s native possession from a shuffled bag within the same selected record. Preserve source/match/paired view/orientation and base/GAR RNG schedules. Exhaust each bag before reuse. Ordinary draws and every target unchanged.',
        fixed='GAR quality/diversity bags identical to parent diversity variant. Six same teacher/fold/seed fits, width96, 2048 updates, batch32, half original/GAR, identical losses/raw-feed views/normalization/learning rates and f062 decoder.',
        arms=list(ARMS),fits=6,folds=list(p.FOLDS),seeds=list(p.SEEDS),steps=2048,
        algebra='Full model; conditional H/A with raw-feed control D within seed then averaged; fixed equal native blend of that cross with archived .527396 leader.',
        selection='Same official original feed spell ranking and promoted-baseline guards as parent; compare against archived leader and all fixed parent arms. Report macro, 70/30, providers, seeds and historical +.01 clean screen.',
        stop='Six sealed fits and three fixed arms, no new duration/rate/weight/decoder/seed settings. Stop this continuation. A new numerical leader may be checked on two new synthetic corruption views without reselection.',
        limitations='Exposed five-match development, silver labels, more even repeated examples without new independent supervision. Original paired records duplicate physical spans and are counted separately; do not call them new examples.',
        parent_declaration_sha256=p.sha(parent.DECL),
        bindings={str(x.relative_to(p.ROOT)):p.sha(x) for x in paths},
        organizer_reference_access=False,promotion=False))


def check():
    parent.check();d=p.read(DECL)
    assert d['parent_declaration_sha256']==p.sha(parent.DECL)
    for rel,digest in d['bindings'].items():assert p.sha(p.ROOT/rel)==digest,rel
    return d


def fit(fold,seed):
    check();assert not p.read(p.RESULTS/f'{parent.STUDY}_aggregate.json')['stretch_reached']
    assert p.read(p.RESULTS/f'{parent.STUDY}_verification.json')['passed']
    assert fold in p.FOLDS and seed in p.SEEDS
    path=OUT/'models'/f'{fold}_{seed}.joblib';p.write(path.with_suffix('.lock.json'),dict(declaration_sha256=p.sha(DECL)))
    teacher=joblib.load(p.sc.initial_path(fold,seed));old=p.training(fold)
    gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS)
    sampler=OriginalDiversitySampler(old,gar,list(teacher['feature_names']),seed);start=time.monotonic()
    with parent.base.installed(sampler):
        result=p.sn.train(teacher,old,gar,width=96,lr=3e-4,seed=seed,steps=2048,
            fraction=.5,batch_size=32,ha_target='dense',train_d=True,require_full_native=True,log_every=256,verbose=True)
    expected=next(r['sampling'] for r in p.read(AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)
    assert sampler.receipt()==expected and result['exposures']==expected['exposures']
    assert not path.exists();joblib.dump(result['model'],path)
    p.write(path.with_suffix('.json'),dict(fold=fold,seed=seed,sha256=p.sha(path),
        declaration_sha256=p.sha(DECL),wall_seconds=time.monotonic()-start,config=result['config'],
        sampling=sampler.receipt(),exposures=result['exposures'],old_matches=sorted({str(r['match']) for r in old})))
    print('FIT COMPLETE',fold,seed,flush=True)


def seal():
    check();files={}
    for fold in p.FOLDS:
        for seed in p.SEEDS:
            path=OUT/'models'/f'{fold}_{seed}.joblib';r=p.read(path.with_suffix('.json'))
            assert p.sha(path)==r['sha256'] and r['declaration_sha256']==p.sha(DECL)
            files[str(path.relative_to(p.ROOT))]=p.sha(path)
    p.write(p.RESULTS/f'{STUDY}_model_seal.json',dict(models=files,declaration_sha256=p.sha(DECL)))


def evaluate():
    check();sealed=p.read(p.RESULTS/f'{STUDY}_model_seal.json')
    for rel,digest in sealed['models'].items():assert p.sha(p.ROOT/rel)==digest
    p.write(OUT/'score_lock.json',dict(declaration_sha256=p.sha(DECL),model_seal_sha256=p.sha(p.RESULTS/f'{STUDY}_model_seal.json')))
    cells=p.shared._load_cells(p.read(p.DECLARATION));values={a:{} for a in ARMS}
    for fold in p.FOLDS:
        for seed in p.SEEDS:
            model=joblib.load(OUT/'models'/f'{fold}_{seed}.joblib');teacher=joblib.load(p.sc.initial_path(fold,seed))
            for c in cells:
                if c['member']!=str(seed) or 'leave_out_'+c['provider'].lower()!=fold:continue
                key=tuple(c[k] for k in ('provider','match','view','member'))
                prob=p.infer(c,model,teacher);values['original_full'][key]=prob
                values['original_ha'][key]=parent.cross.cross(prob,parent.prior.parent_probability(c))
                values['original_leader_half'][key]=.5*(values['original_ha'][key]+parent.leader_probability(c))
    rows={a:[] for a in ARMS};files={}
    for arm in ARMS:
        for c in cells:
            key=tuple(c[k] for k in ('provider','match','view','member'))
            prob=np.mean([values[arm][key[:3]+(str(s),)] for s in p.SEEDS],axis=0) if c['member']=='ensemble' else values[arm][key]
            row,pred=p.score(c,prob);rows[arm].append(row)
            path=OUT/'predictions'/arm/('__'.join(key)+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():raise FileExistsError(path)
            np.savez_compressed(path,probabilities=prob,prediction=pred,**{k:c[k] for k in ('labels','period','timestamps','model_period','fps')})
            files[str(path.relative_to(p.ROOT))]=p.sha(path)
    summaries={a:{m:p.scoring.summarize(rows[a],m) for m in p.shared.MEMBERS} for a in ARMS}
    prior=p.read(p.RESULTS/f'{parent.STUDY}_aggregate.json');baseline=prior['baseline'];control=prior['control']
    guards={a:bool(parent.prior.selection.eligible(s['ensemble'],baseline)) for a,s in summaries.items()}
    eligible=[a for a in ARMS if guards[a] and parent.prior.selection.rank(summaries[a]['ensemble'])>parent.prior.selection.rank(control)]
    winner=max(eligible,key=lambda a:parent.prior.selection.rank(summaries[a]['ensemble'])) if eligible else None
    p.write(p.RESULTS/f'{STUDY}_aggregate.json',dict(study=STUDY,rows=rows,summaries=summaries,
        baseline=baseline,control=control,guards_pass=guards,winner=winner,
        stretch_reached=bool(winner and summaries[winner]['ensemble']['feed_like']['equal_provider_mean']['spell_f1']>=.55),
        historical_clean_screen_pass={a:bool(s['ensemble']['clean']['equal_provider_mean']['spell_f1']>=baseline['clean']['equal_provider_mean']['spell_f1']+.01) for a,s in summaries.items()},
        prediction_sha256=files,organizer_reference_access=False,independent_confirmation=False))
    for a,s in summaries.items():print(a,{v:s['ensemble'][v]['equal_provider_mean']['spell_f1'] for v in ('clean','feed_like')},guards[a],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('audit','declare','fit','seal','evaluate'))
    parser.add_argument('--fold',choices=p.FOLDS);parser.add_argument('--seed',type=int,choices=p.SEEDS);args=parser.parse_args()
    fit(args.fold,args.seed) if args.action=='fit' else globals()[args.action]()
