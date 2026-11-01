"""One bounded decoder calibration on the fixed eligible short-training leader."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from experiments import feed_short_sampling_v1 as record,feed_short_head_v1 as head,feed_short_game_v1 as game
p=record.p
STUDY='feed_short_calibration_v1'
OUT=p.ROOT/'data/experiments'/STUDY
DECL=p.RESULTS/f'{STUDY}_declaration.json'


def configs():
    return [dict(id=f'c{i:02d}',family='calibration',dt=t,ht=h,ha=c,dead=c) for i,(t,h,c) in enumerate((t,h,c) for t in (2,4,6) for h in (.75,1,1.25) for c in (2,3,4))]


def declare():
    game.check()
    if (p.RESULTS/f'{game.STUDY}_aggregate.json').exists():raise ValueError('Declare before game-level outcomes')
    p.write(DECL,{'study':STUDY,'utc':datetime.now(timezone.utc).isoformat(),
        'question':'Does a fixed small calibration grid improve the eligible short-training leader, whose probabilities differ from the model used for the old decoder search?',
        'condition':'After all earlier declared comparisons and replays. Only if eligible original leader beats .525240 but remains below .55.',
        'model_selection':'Choose highest original feed spell among raw-feed control, within-record direct/blend, targeted H/A cross, and game direct/H/A cross, subject to promoted-baseline guards; macro tie-break. Lock this model before calibration. Never select the model using grid results.',
        'configs':configs(),'fits':0,'cells':405,'control':'dt4/ht1/cost3 must replay the selected model exactly before grid scoring.',
        'primary':'Original equal-provider feed spell; macro tie-break. Same promoted-baseline guards; historical clean+.01 separate.',
        'stop':'27 fixed configurations on one probability ensemble; no subsequent local refinement, mixture or seed search. No extra-corruption selection or organizer references. Selected setting gets 30 individual-seed diagnostic cells with no reselection.',
        'evidence':'Further tuning on five exposed development matches and synthetic defects; not independent confirmation.',
        'bindings':{str(Path(__file__).relative_to(p.ROOT)):p.sha(Path(__file__)),str(game.DECL.relative_to(p.ROOT)):p.sha(game.DECL)},'organizer_reference_access':False,'promotion':False})


def candidates():
    d=p.read(p.RESULTS/f'{record.STUDY}_aggregate.json');baseline=d['baseline']
    rows=[dict(name='raw_control',summary=d['control'],folder=p.OUT/'predictions/feed_augmented',aggregate=p.RESULTS/'feed_robustness_v1_aggregate.json')]
    for arm in record.ARMS:rows.append(dict(name=arm,summary=d['summaries'][arm]['ensemble'],folder=record.OUT/'predictions/original'/arm,aggregate=p.RESULTS/f'{record.STUDY}_aggregate.json'))
    h=p.read(p.RESULTS/f'{head.STUDY}_aggregate.json');rows.append(dict(name='targeted_ha',summary=h['summaries']['ensemble'],folder=head.OUT/'predictions',aggregate=p.RESULTS/f'{head.STUDY}_aggregate.json'))
    if (p.RESULTS/f'{game.STUDY}_aggregate.json').exists():
        g=p.read(p.RESULTS/f'{game.STUDY}_aggregate.json')
        for arm in game.ARMS:rows.append(dict(name=arm,summary=g['summaries'][arm]['ensemble'],folder=game.OUT/'predictions/original'/arm,aggregate=p.RESULTS/f'{game.STUDY}_aggregate.json'))
    return baseline,rows


def run():
    game.check();d=p.read(DECL)
    for rel,digest in d['bindings'].items():assert p.sha(p.ROOT/rel)==digest
    for name in (record.STUDY,head.STUDY,game.STUDY):
        if (p.RESULTS/f'{name}_aggregate.json').exists():assert p.read(p.RESULTS/f'{name}_verification.json')['passed']
    baseline,choices=candidates();eligible=[r for r in choices if record.selection.eligible(r['summary'],baseline)]
    leader=max(eligible,key=lambda r:record.selection.rank(r['summary']));score=leader['summary']['feed_like']['equal_provider_mean']['spell_f1']
    if not .5252404582412046<score<.55:
        p.write(p.RESULTS/f'{STUDY}_skipped.json',{'leader':leader['name'],'score':score,'reason':'No eligible improvement below stretch'});return
    cells=p.shared._load_cells(p.read(p.DECLARATION));prob={};bindings={};source=p.read(leader['aggregate'])
    for c in cells:
        key=tuple(c[k] for k in ('provider','match','view','member'));path=leader['folder']/('__'.join(key)+'.npz')
        assert p.sha(path)==source['prediction_sha256'][str(path.relative_to(p.ROOT))]
        bindings[str(path.relative_to(p.ROOT))]=p.sha(path)
        with np.load(path) as z:
            for field in ('labels','period','timestamps','model_period','fps'):np.testing.assert_array_equal(z[field],c[field])
            prob[key]=z['probabilities']
    replay=[p.score(c,prob[tuple(c[k] for k in ('provider','match','view','member'))])[0] for c in cells if c['member']=='ensemble']
    replay_summary=p.scoring.summarize(replay,'ensemble')
    assert replay_summary==leader['summary']
    p.write(OUT/'score_lock.json',{'leader':leader['name'],'leader_aggregate_sha256':p.sha(leader['aggregate']),'declaration_sha256':p.sha(DECL),'probability_sources':bindings,'exact_control_replay':True})
    rows={c['id']:[] for c in configs()};files={}
    def score_save(c,config,group):
        key=tuple(c[k] for k in ('provider','match','view','member'));prediction=p.decoder.decode(prob[key][None],c['model_period'],c['fps'],config)
        row={k:c[k] for k in ('provider','match','view','member')};row.update(p.shared._score_prediction(c['labels'],prediction,c['period'],c['timestamps'],c['fps']))
        path=OUT/'predictions'/group/config['id']/('__'.join(key)+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,prediction=prediction);files[str(path.relative_to(p.ROOT))]=p.sha(path)
        return row
    for cfg in configs():
        for c in cells:
            if c['member']=='ensemble':rows[cfg['id']].append(score_save(c,cfg,'grid'))
        print('SCORED',cfg['id'],flush=True)
    summaries={k:p.scoring.summarize(v,'ensemble') for k,v in rows.items()};guards={k:record.selection.eligible(v,baseline) for k,v in summaries.items()}
    winner=max((k for k in summaries if guards[k]),key=lambda k:record.selection.rank(summaries[k]));config=next(c for c in configs() if c['id']==winner)
    diagnostics=[score_save(c,config,'selected_seeds') for c in cells if c['member']!='ensemble']
    p.write(p.RESULTS/f'{STUDY}_aggregate.json',{'leader':leader['name'],'leader_summary':leader['summary'],'rows':rows,'summaries':summaries,'guards_pass':guards,'winner':winner,'config':config,'winner_summary':summaries[winner],'stretch_reached':summaries[winner]['feed_like']['equal_provider_mean']['spell_f1']>=.55,'historical_clean_screen_pass':{k:v['clean']['equal_provider_mean']['spell_f1']>=baseline['clean']['equal_provider_mean']['spell_f1']+.01 for k,v in summaries.items()},'seed_diagnostics':diagnostics,'seed_summaries':{str(s):p.scoring.summarize(diagnostics,str(s)) for s in p.SEEDS},'prediction_sha256':files,'model_probability_sha256':bindings,'organizer_reference_access':False,'independent_confirmation':False})
    print(winner,config,{v:summaries[winner][v]['equal_provider_mean']['spell_f1'] for v in ('clean','feed_like')},flush=True)


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('action',choices=('declare','run'));globals()[q.parse_args().action]()
