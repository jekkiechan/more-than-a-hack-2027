"""Shared bounded runner for three further feed-first rounds; no public labels."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import json,multiprocessing
from pathlib import Path
import numpy as np
from experiments import feed_spell_more as methods, feed_spell_push_run as first
from experiments import decoder_search_run as scoring, spell_mbr_run as shared, supervised

ROOT=shared.ROOT;RESULTS=shared.RESULTS;CELLS=[]
LEGACY=ROOT.parent/'feed-private'
SOURCES=first.SOURCES+('experiments/feed_spell_more.py','experiments/feed_spell_more_run.py')


def paths(number):
    study=f'feed_spell_iteration_r{number}'
    return study,RESULTS/(study+'_declaration.json'),RESULTS/(study+'_aggregate.json'),ROOT/'data/experiments'/study


def legacy_path(cell,member):
    source=cell['provider'].lower();match=cell['match']
    if source=='skillcorner':match='skillcorner_had_'+match
    view='clean' if cell['view']=='clean' else 'feed_like'
    replicate='clean' if view=='clean' else cell['view'].removeprefix('feed_like_')
    return LEGACY/f'experiments/objective_alignment_v1/{source}_{match}_{view}_{replicate}_combined_{member}.npz'


def load(number,d):
    cells=first.paired_cells(d)
    if number>=4:
        for c in cells:
            ps=[]
            for seed in (('2026','2027') if c['member']=='ensemble' else (c['member'],)):
                path=legacy_path(c,seed)
                rel=str(path.relative_to(LEGACY))
                if shared.sha256_file(path)!=d['legacy_files'][rel]:raise ValueError('Legacy probability hash changed')
                with np.load(path,allow_pickle=False) as z:p=z['probability'].copy()
                if p.shape!=c['probabilities'].shape or not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(axis=1),1,atol=1e-6):
                    raise ValueError('Legacy native probability contract failed')
                ps.append(p)
            c['legacy']=np.stack(ps)
    return cells


def freeze(number):
    study,decl,agg,out=paths(number)
    upstream=json.loads(first.AGGREGATE.read_text());source=json.loads(first.DECLARATION.read_text())
    parent=other=None;dependencies={str(first.AGGREGATE.relative_to(ROOT)):shared.sha256_file(first.AGGREGATE)}
    if number>=4:
        prior=paths(3)[2];p=json.loads(prior.read_text());parent=p['selected_config'];dependencies[str(prior.relative_to(ROOT))]=shared.sha256_file(prior)
    if number>=5:
        prior=paths(4)[2];p=json.loads(prior.read_text());other=p['selected_config'];dependencies[str(prior.relative_to(ROOT))]=shared.sha256_file(prior)
        if first.rank(json.loads(paths(3)[2].read_text())['arms'][parent['id']]) < first.rank(upstream['arms']['f062']):parent=methods.BASE.copy()
        if first.rank(p['arms'][other['id']]) < first.rank(upstream['arms']['f062']):other=methods.BASE.copy()
    arms=methods.configurations(number,parent,other)
    d={"study":study,"frozen_utc":datetime.now(timezone.utc).isoformat(),"round":number,
       "arms":arms,"ensemble_cells":15*len(arms),"component_seed_budget":60,
       "question":{3:'Can logit denoising and uncertainty-dependent emissions or edge costs improve the fixed winner?',
                   4:'Does recipe-diverse ensembling with provider-excluded combined-objective models improve feed-like spell?',
                   5:'Do consensus emissions, calibrated minimum-duration decoding or H/A evidence caps improve the strongest fixed rule?'}[number],
       "primary":source['primary'],"aggregation":source['aggregation'],"guards":source['guards'],
       "cohort":source['cohort'],"reconstruction_status":source['reconstruction_status'],
       "prediction_files":source['prediction_files'],"evaluation_input_hashes":source['evaluation_input_hashes'],
       "model_checkpoint_hashes":source['model_checkpoint_hashes'],
       "source_sha256":{s:shared.sha256_file(ROOT/s) for s in SOURCES},"dependencies":dependencies,
       "new_fits":0,"organizer_reference_access":False,"public_plan":"Only after round 5, score fixed broad winner and final external winner once, deduplicating identical CSVs; no public-driven tuning",
       "selection":"Feed spell then macro among guard-passing arms; top two component-seed checks descriptive only",
       "legacy_files":{}}
    if number>=4:
        manifest=LEGACY/'objective_alignment_cache_manifest.json'
        expected=json.loads((RESULTS/'objective_ensemble_declaration.json').read_text())['private_archive_manifests']['objective_alignment_archive']['sha256']
        if shared.sha256_file(manifest)!=expected:raise ValueError('Legacy manifest provenance changed')
        entries={e['path']:e for e in json.loads(manifest.read_text())['files']}
        for c in first.paired_cells(source):
            for seed in ('2026','2027'):
                path=legacy_path(c,seed);rel=str(path.relative_to(LEGACY));e=entries[rel]
                if path.stat().st_size!=e['bytes'] or shared.sha256_file(path)!=e['sha256']:raise ValueError('Legacy restore hash failed')
                d['legacy_files'][rel]=e['sha256']
        d['legacy_provenance']='Exact objective-alignment native arrays, full-provider-excluded combined-loss seeds. NPZ lacks time/mask metadata; require native shape and historical score replay on current aligned cohort before blending.'
    scoring.write_once(decl,d)


def evaluate(config):
    rows=[]
    for c in CELLS:
        if c['member']=='ensemble':
            y=methods.decode(c['members'],c['model_period'],c['fps'],config,c.get('legacy'))
            rows.append(first.identity(c)|scoring.quick_score(c,y))
    return config,rows,scoring.summarize(rows)


def run(number):
    global CELLS
    study,decl,agg,out=paths(number);d=json.loads(decl.read_text())
    for path,sha in (d['source_sha256']|d['dependencies']).items():
        if shared.sha256_file(ROOT/path)!=sha:raise ValueError('Frozen dependency changed: '+path)
    out.mkdir(parents=True,exist_ok=False);scoring.write_once(out/'lock.json',{'declaration_sha256':shared.sha256_file(decl)})
    CELLS=load(number,d);broad=json.loads(first.AGGREGATE.read_text());baseline=broad['baseline']
    control=[]
    for c in CELLS:
        y=methods.decode(c['members'],c['model_period'],c['fps'],methods.BASE,c.get('legacy'))
        with np.load(first.OUT/'f062'/Path(c['relative_path']).name) as saved:
            if not np.array_equal(y,saved['prediction']):raise ValueError('f062 exact path replay failed')
        control.append(first.identity(c)|scoring.quick_score(c,y))
    old_control=None
    if number>=4:
        old_rows=[]
        for c in CELLS:
            if c['member']=='ensemble':
                y=supervised.decode(c['legacy'].mean(axis=0),c['model_period'],c['fps'],3)
                old_rows.append(first.identity(c)|scoring.quick_score(c,y))
        old_control=scoring.summarize(old_rows)
        for view,expected in (('clean',.6332050169638032),('feed_like',.498829607824332)):
            if abs(old_control[view]['equal_provider_mean']['spell_f1']-expected)>1e-10:raise ValueError('Legacy historical score replay failed')
    scoring.write_once(RESULTS/(study+'_control_replay.json'),{'passed':True,'exact_f062_paths':45,'legacy_replay':old_control})
    summaries={};arm_rows={}
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        for config,rows,summary in pool.map(evaluate,d['arms']):
            for row in rows:
                ref=next(r for r in control if all(r[k]==row[k] for k in first.identity(row)))
                if any(row[k]!=ref[k] for k in ('graded_frames','reference_spell_count')):raise ValueError('Grading population changed')
            arm=config['id'];summaries[arm]=summary;arm_rows[arm]=rows
            scoring.write_once(out/(arm+'.json'),{'config':config,'rows':rows})
            print(arm,config['family'],summary['clean']['equal_provider_mean']['spell_f1'],summary['feed_like']['equal_provider_mean']['spell_f1'],flush=True)
    ranked=sorted(summaries,key=lambda a:first.rank(summaries[a]),reverse=True);allowed=[a for a in ranked if first.eligible(summaries[a],baseline)]
    finalists={}
    for arm in allowed[:2]:
        config=next(c for c in d['arms'] if c['id']==arm);rows=[]
        for c in CELLS:
            y=methods.decode(c['members'],c['model_period'],c['fps'],config,c.get('legacy'))
            row=first.identity(c)|shared._score_prediction(c['labels'],y,c['period'],c['timestamps'],c['fps'])
            if c['member']=='ensemble':
                old=next(r for r in arm_rows[arm] if all(r[k]==row[k] for k in first.identity(row)))
                if abs(old['spell_f1']-row['spell_f1'])>1e-12:raise ValueError('Official score replay failed')
            rows.append(row);path=out/arm/Path(c['relative_path']).name;path.parent.mkdir(exist_ok=True);np.savez_compressed(path,prediction=y)
        scoring.write_once(out/(arm+'_final.json'),{'rows':rows});finalists[arm]={m:scoring.summarize(rows,m) for m in shared.MEMBERS}
    cross={}
    for held in shared.PROVIDERS:
        train=[p for p in shared.PROVIDERS if p!=held];choices=[a for a in summaries if first.eligible(summaries[a],baseline,train)]
        chosen=max(choices,key=lambda a:first.rank(summaries[a],train));cross[held]={'selected':chosen,'selection_providers':train,'scores':{v:summaries[chosen][v]['by_provider'][held] for v in ('clean','feed_like')}}
    scoring.write_once(agg,{'study':study,'declaration_sha256':shared.sha256_file(decl),'arms':summaries,'ranked':ranked,'eligible_ranked':allowed,
        'selected_config':next(c for c in d['arms'] if c['id']==allowed[0]),'finalists':finalists,'baseline':baseline,'baseline_seeds':broad['baseline_seeds'],
        'broad_winner':broad['arms']['f062'],'cross_provider_selection':cross,'completed_ensemble_cells':15*len(summaries),'component_seed_cells':60,
        'organizer_labels_accessed':False,'private_prediction_hashes':{str(p.relative_to(ROOT)):shared.sha256_file(p) for p in out.rglob('*.npz')}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('freeze','run'));parser.add_argument('--round',type=int,choices=(3,4,5),required=True)
    args=parser.parse_args();freeze(args.round) if args.stage=='freeze' else run(args.round)
