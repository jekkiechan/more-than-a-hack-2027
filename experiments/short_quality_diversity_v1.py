"""Bounded quality and unique-possession sampling comparison; frozen decoder."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import feed_robustness_v1 as p
from experiments import feed_short_sampling_v1 as prior
from experiments import feed_short_head_v1 as leader
from experiments import feed_head_transfer_v1 as cross
from experiments import feed_short_sampler as base
from experiments import short_label_quality as quality

STUDY = 'short_quality_diversity_v1'
OUT = p.ROOT/'data/experiments'/STUDY
DECL = p.RESULTS/f'{STUDY}_declaration.json'
AUDIT = p.RESULTS/f'{STUDY}_training_audit.json'
VARIANTS = ('quality', 'diversity')
ARMS = ('quality_full', 'quality_ha', 'diversity_full', 'diversity_ha', 'diversity_leader_half')


def audit():
    p.check()
    gar = p.dl.load_dense_records('train', output=p.TRAIN_RECORDS)
    rows = []
    for fold in p.FOLDS:
        old = p.training(fold)
        names = list(joblib.load(p.sc.initial_path(fold, 2026))['feature_names'])
        for seed in p.SEEDS:
            previous = next(r for r in p.read(prior.AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)['targeted']
            values = {}
            for variant in VARIANTS:
                sampler = quality.QualitySampler(old, gar, names, seed, diverse=variant=='diversity')
                result = base.replay(sampler, seed)
                assert result['counts']['old'] == previous['counts']['old']
                assert result['exposures'] == previous['exposures']
                assert result['eligible_quality_games'] == 45
                values[variant] = result
                print('AUDIT', fold, seed, variant, result['replacement_distinct_possessions'], flush=True)
            rows.append(dict(fold=fold, seed=seed, sampling=values))
    p.write(AUDIT, dict(train_only=True, rows=rows,
        source_sha256=p.sha(Path(quality.__file__)), organizer_reference_access=False))


def declare():
    p.check()
    a = p.read(AUDIT)
    assert a['source_sha256'] == p.sha(Path(quality.__file__))
    files = [Path(__file__), Path(quality.__file__), AUDIT,
        p.ROOT/'experiments/tests/test_short_label_quality.py',
        p.RESULTS/'feed_short_head_v1_aggregate.json',
        p.RESULTS/'feed_short_sampling_v1_aggregate.json',
        p.RESULTS/'feed_short_game_v1_aggregate.json',
        p.RESULTS/'feed_robustness_v1_aggregate.json']
    d = p.read(p.DECLARATION)
    p.write(DECL, dict(study=STUDY, utc=datetime.now(timezone.utc).isoformat(),
        question='Can corroborating short GAR silver labels and cycling distinct possessions improve feed-like spell F1 beyond .527396?',
        prefit_train_findings='1427 complete <=3s GAR possessions; 583 lack sampled observed-ball evidence; 72 contain conflicting sparse anchors. 403 spans across all 45 TRAIN games meet the declared conservative proxy.',
        quality_rule='Existing complete <=3s H/A spans only. Require an overlapping same-team uncensored controlled touch, no opposing H/A sparse anchor in the span, and >=2 sampled observed-ball rows where own nearest-player distance is >=.5m smaller than opposing distance. Native masks and full-period/unknown censoring preserved. No label is changed or removed from ordinary draws.',
        quality_variant='Replace .25 of GAR draws by uniform existing eligible windows from the already selected game. Reuse previous within-record original-provider draws exactly.',
        diversity_variant='Same quality pool, rate and game selection. Shuffle distinct native possession IDs per game and exhaust each bag before reuse; choose uniformly among existing windows overlapping that possession. No record duplication or synthetic labels.',
        interpretation='Geometry and anchors are conservative silver-label sampling proxies, not independent gold verification. They may favor observable/easy cases. Duration fixed at <=3s, no alternate threshold sweep.',
        variants=list(VARIANTS), arms=list(ARMS), fits=12,
        seeds=list(p.SEEDS), folds=list(p.FOLDS), steps=2048,
        fixed='Same held-out teachers and training-only normalizers, raw-feed views, 45 GAR TRAIN games, original TRAIN matches, labels/losses, width96, batch32, 50/50 sources, HA lr3e-4/D lr1e-4, 2048 updates. Both heads see changed window distribution. No early stopping.',
        output_arms='For each variant: whole H/A/D and conditional H/A crossed with raw-feed control D within matching seed then averaged. One fixed .5 native-probability blend of diversity_ha and archived numerical leader, within seed then averaged.',
        decoder='Unchanged f062: D temperature4 then cost3. No calibration/decoder/rate/weight/seed search.',
        controls='Archived promoted baseline .518527, raw-feed control .525240 and within-record H/A leader .527396; no control refits.',
        primary='Official original feed_like spell F1; macro exact-tie break; average replicas within match, matches within provider, providers equally. Auxiliary 70/30 spell/macro reported only.',
        guards='Against promoted baseline: clean aggregate spell >= baseline-.005; each provider clean macro and D-F1 >= baseline-.02; each provider feed spell >= baseline-.01. Historical +.01 clean screen separately. Must beat numerical leader to claim progress; .55 stretch.',
        stop='Seal all twelve checkpoints before candidate scoring. Five fixed arms / 225 cells; stop without rates, weights or decoder refinements. If a guard-passing arm beats the numerical leader, freeze its identity and compare it with leader on two newly generated target-independent corruption realizations per match; no reselection.',
        exposure='All five matches are previously exposed development; every learned H/A, D and normalization component excludes evaluated provider. New corruption is not new matches. GAR silver labels are not gold possession labels.',
        source_sha256={str(f.relative_to(p.ROOT)):p.sha(f) for f in files},
        parent_declaration_sha256=p.sha(p.DECLARATION),
        input_sha256=d['input_sha256'], organizer_reference_access=False, promotion=False))


def check():
    d = p.read(DECL)
    assert d['parent_declaration_sha256'] == p.sha(p.DECLARATION)
    for rel, digest in d['source_sha256'].items():
        assert p.sha(p.ROOT/rel) == digest, rel
    return d


def leader_probability(c):
    path=leader.OUT/'predictions'/('__'.join(c[k] for k in ('provider','match','view','member'))+'.npz')
    expected=p.read(p.RESULTS/'feed_short_head_v1_aggregate.json')['prediction_sha256']
    assert p.sha(path)==expected[str(path.relative_to(p.ROOT))]
    with np.load(path) as z:
        for k in ('labels','period','timestamps','model_period','fps'):
            np.testing.assert_array_equal(z[k],c[k])
        return z['probabilities']


def preflight():
    check(); p.check()
    cells=p.shared._load_cells(p.read(p.DECLARATION))
    rows=[p.score(c,leader_probability(c))[0] for c in cells]
    assert rows==p.read(p.RESULTS/'feed_short_head_v1_aggregate.json')['rows']
    rows=[p.score(c,prior.parent_probability(c))[0] for c in cells]
    assert rows==p.read(p.RESULTS/'feed_robustness_v1_aggregate.json')['rows']['feed_augmented']
    c=next(c for c in cells if c['provider']=='Metrica' and c['view']=='clean' and c['member']=='2026')
    m=joblib.load(p.OUT/'models/leave_out_metrica_2026.joblib')
    t=joblib.load(p.sc.initial_path('leave_out_metrica',2026))
    actual=p.infer(c,m,t);expected=prior.parent_probability(c)
    diff=float(np.abs(actual-expected).max());assert diff<=2e-6
    np.testing.assert_array_equal(p.decode(actual,c['model_period'],c['fps']),p.decode(expected,c['model_period'],c['fps']))
    p.write(p.RESULTS/f'{STUDY}_preflight.json',dict(passed=True,control_cells=90,
        runtime_max_probability_difference=diff,runtime_changed_labels=0,declaration_sha256=p.sha(DECL)))


def fit(variant,fold,seed):
    check()
    assert variant in VARIANTS and fold in p.FOLDS and seed in p.SEEDS
    assert p.read(p.RESULTS/f'{STUDY}_preflight.json')['passed']
    path=OUT/'models'/variant/f'{fold}_{seed}.joblib'
    p.write(path.with_suffix('.lock.json'),dict(declaration_sha256=p.sha(DECL)))
    teacher=joblib.load(p.sc.initial_path(fold,seed));old=p.training(fold)
    gar=p.dl.load_dense_records('train',output=p.TRAIN_RECORDS)
    sampler=quality.QualitySampler(old,gar,list(teacher['feature_names']),seed,diverse=variant=='diversity')
    start=time.monotonic()
    with base.installed(sampler):
        result=p.sn.train(teacher,old,gar,width=96,lr=3e-4,seed=seed,steps=2048,
            fraction=.5,batch_size=32,ha_target='dense',train_d=True,
            require_full_native=True,log_every=256,verbose=True)
    expected=next(r for r in p.read(AUDIT)['rows'] if r['fold']==fold and r['seed']==seed)['sampling'][variant]
    assert sampler.receipt()==expected and result['exposures']==expected['exposures']
    assert not path.exists();joblib.dump(result['model'],path)
    p.write(path.with_suffix('.json'),dict(variant=variant,fold=fold,seed=seed,
        sha256=p.sha(path),declaration_sha256=p.sha(DECL),wall_seconds=time.monotonic()-start,
        config=result['config'],sampling=sampler.receipt(),exposures=result['exposures'],
        old_matches=sorted({str(r['match']) for r in old})))
    print('FIT COMPLETE',variant,fold,seed,flush=True)


def seal():
    check();files={}
    for variant in VARIANTS:
        for fold in p.FOLDS:
            for seed in p.SEEDS:
                path=OUT/'models'/variant/f'{fold}_{seed}.joblib';r=p.read(path.with_suffix('.json'))
                assert p.sha(path)==r['sha256'] and r['declaration_sha256']==p.sha(DECL)
                files[str(path.relative_to(p.ROOT))]=p.sha(path)
    p.write(p.RESULTS/f'{STUDY}_model_seal.json',dict(models=files,declaration_sha256=p.sha(DECL)))


def evaluate():
    check();sealed=p.read(p.RESULTS/f'{STUDY}_model_seal.json')
    for rel,digest in sealed['models'].items():assert p.sha(p.ROOT/rel)==digest
    p.write(OUT/'score_lock.json',dict(declaration_sha256=p.sha(DECL),model_seal_sha256=p.sha(p.RESULTS/f'{STUDY}_model_seal.json')))
    cells=p.shared._load_cells(p.read(p.DECLARATION));values={a:{} for a in ARMS}
    for variant in VARIANTS:
        for fold in p.FOLDS:
            for seed in p.SEEDS:
                model=joblib.load(OUT/'models'/variant/f'{fold}_{seed}.joblib')
                teacher=joblib.load(p.sc.initial_path(fold,seed))
                for c in cells:
                    if c['member']!=str(seed) or 'leave_out_'+c['provider'].lower()!=fold:continue
                    key=tuple(c[k] for k in ('provider','match','view','member'))
                    prob=p.infer(c,model,teacher)
                    values[variant+'_full'][key]=prob
                    values[variant+'_ha'][key]=cross.cross(prob,prior.parent_probability(c))
                    if variant=='diversity':values['diversity_leader_half'][key]=.5*(values['diversity_ha'][key]+leader_probability(c))
                    print('INFERRED',variant,key,flush=True)
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
    control=p.read(p.RESULTS/'feed_short_head_v1_aggregate.json')['summaries']['ensemble']
    baseline=p.read(p.RESULTS/'feed_robustness_v1_aggregate.json')['summaries']['baseline']['ensemble']
    guards={a:bool(prior.selection.eligible(s['ensemble'],baseline)) for a,s in summaries.items()}
    eligible=[a for a in ARMS if guards[a] and prior.selection.rank(summaries[a]['ensemble'])>prior.selection.rank(control)]
    winner=max(eligible,key=lambda a:prior.selection.rank(summaries[a]['ensemble'])) if eligible else None
    p.write(p.RESULTS/f'{STUDY}_aggregate.json',dict(study=STUDY,rows=rows,summaries=summaries,
        baseline=baseline,control=control,guards_pass=guards,winner=winner,
        stretch_reached=bool(winner and summaries[winner]['ensemble']['feed_like']['equal_provider_mean']['spell_f1']>=.55),
        historical_clean_screen_pass={a:bool(s['ensemble']['clean']['equal_provider_mean']['spell_f1']>=baseline['clean']['equal_provider_mean']['spell_f1']+.01) for a,s in summaries.items()},
        prediction_sha256=files,organizer_reference_access=False,independent_confirmation=False))
    for a,s in summaries.items():print(a,{v:s['ensemble'][v]['equal_provider_mean']['spell_f1'] for v in ('clean','feed_like')},guards[a],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('audit','declare','preflight','fit','seal','evaluate'))
    parser.add_argument('--variant',choices=VARIANTS);parser.add_argument('--fold',choices=p.FOLDS);parser.add_argument('--seed',type=int,choices=p.SEEDS)
    args=parser.parse_args()
    fit(args.variant,args.fold,args.seed) if args.action=='fit' else globals()[args.action]()
