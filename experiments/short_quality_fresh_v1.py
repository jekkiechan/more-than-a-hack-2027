"""Two new synthetic corruption views for one fixed original-benchmark winner."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import joblib
import numpy as np
from experiments import short_quality_diversity_v1 as parent
from experiments import short_original_diversity_v1 as extension

p=parent.p
STUDY='short_quality_fresh_v1'
OUT=p.ROOT/'data/experiments'/STUDY
DECL=p.RESULTS/f'{STUDY}_declaration.json'
SEEDS=(746201,746202)


def declare():
    parent.check();extension.check()
    assert not (p.RESULTS/f'{parent.STUDY}_aggregate.json').exists()
    items=p.old.selection_items()
    for item in items:assert p.sha(p.prep.data_path(item,p.old.DATA))==item['sha256']
    p.write(DECL,dict(study=STUDY,utc=datetime.now(timezone.utc).isoformat(),
        seeds=list(SEEDS),matches=items,calibration=p.sn.default_ball_calibration(),
        rule='Freeze the highest eligible original-benchmark arm across both declared studies using original feed spell, macro tie-break; compare only if it exceeds the archived .527396 numerical leader. No fresh-view reselection.',
        evidence='Two previously unused synthetic realizations on the same exposed matches. Not independent-match or gold-label confirmation.',
        source_sha256=p.sha(Path(__file__)),parent_declaration_sha256=p.sha(parent.DECL),
        extension_declaration_sha256=p.sha(extension.DECL),organizer_reference_access=False,promotion=False))


def check():
    d=p.read(DECL)
    assert p.sha(Path(__file__))==d['source_sha256']
    assert p.sha(parent.DECL)==d['parent_declaration_sha256']
    assert p.sha(extension.DECL)==d['extension_declaration_sha256']
    return d


def prepare_cell(task):
    item,index=task;d=check();path=p.prep.data_path(item,p.old.DATA)
    assert p.sha(path)==item['sha256']
    inputs,labels=p.ms.view(path)
    seed=int(hashlib.sha256(f'{STUDY}/{SEEDS[index]}/{item["match"]}'.encode()).hexdigest()[:8],16)
    viewed=p.raw_view(inputs,d['calibration'],seed)
    record=p.prep.build_record(viewed,labels,item['source'],item['match'])
    path=OUT/'inputs'/f'{item["match"]}__fresh_{index+1}.joblib'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    joblib.dump(dict(inputs=viewed,labels=labels,record=record),path)
    print('PREPARED',path.name,flush=True)
    return dict(path=str(path.relative_to(p.ROOT)),sha256=p.sha(path),match=item['match'],provider=item['source'],replicate=index+1,seed=seed)


def prepare():
    d=check();p.write(OUT/'prepare_lock.json',dict(declaration_sha256=p.sha(DECL)))
    with ProcessPoolExecutor(max_workers=2) as pool:
        files=list(pool.map(prepare_cell,[(i,n) for i in d['matches'] for n in range(2)]))
    p.write(OUT/'preparation.json',dict(files=files,declaration_sha256=p.sha(DECL)))


def select():
    check();options=[]
    for study in (parent,extension):
        path=p.RESULTS/f'{study.STUDY}_aggregate.json'
        if not path.exists():continue
        assert p.read(p.RESULTS/f'{study.STUDY}_verification.json')['passed']
        r=p.read(path)
        for a,s in r['summaries'].items():
            if r['guards_pass'][a] and parent.prior.selection.rank(s['ensemble'])>parent.prior.selection.rank(r['control']):
                options.append((parent.prior.selection.rank(s['ensemble']),study.STUDY,a,p.sha(path)))
    if not options:
        p.write(p.RESULTS/f'{STUDY}_skipped.json',dict(reason='No eligible improvement over archived numerical leader',scored=False));return
    _,study,arm,digest=max(options)
    p.write(OUT/'selection.json',dict(study=study,arm=arm,aggregate_sha256=digest,
        selected_before_fresh_scores=True,declaration_sha256=p.sha(DECL)))


def run():
    check();selection=p.read(OUT/'selection.json');receipt=p.read(OUT/'preparation.json')
    selected=parent if selection['study']==parent.STUDY else extension
    result=p.read(p.RESULTS/f'{selected.STUDY}_aggregate.json')
    assert p.sha(p.RESULTS/f'{selected.STUDY}_aggregate.json')==selection['aggregate_sha256']
    p.write(OUT/'score_lock.json',dict(selection_sha256=p.sha(OUT/'selection.json'),preparation_sha256=p.sha(OUT/'preparation.json')))
    arm=selection['arm'];before=p.read(p.RESULTS/'feed_short_head_v1_aggregate.json')
    rows={'leader':[r for r in before['rows'] if r['view']=='clean'],
          'fixed_candidate':[r for r in result['rows'][arm] if r['view']=='clean']}
    files={};components={};loaded={}
    for fold in p.FOLDS:
        for seed in p.SEEDS:
            rawpath=p.OUT/'models'/f'{fold}_{seed}.joblib'
            leaderpath=parent.prior.OUT/'models'/f'{fold}_{seed}.joblib'
            if selected==parent:
                variant='quality' if arm.startswith('quality') else 'diversity'
                candidatepath=parent.OUT/'models'/variant/f'{fold}_{seed}.joblib'
            else:candidatepath=extension.OUT/'models'/f'{fold}_{seed}.joblib'
            for path,seal in ((rawpath,p.RESULTS/'feed_robustness_v1_model_seal.json'),
                              (leaderpath,p.RESULTS/'feed_short_sampling_v1_model_seal.json'),
                              (candidatepath,p.RESULTS/f'{selected.STUDY}_model_seal.json')):
                assert p.sha(path)==p.read(seal)['models'][str(path.relative_to(p.ROOT))]
            loaded[fold,seed]=(joblib.load(rawpath),joblib.load(leaderpath),joblib.load(candidatepath),joblib.load(p.sc.initial_path(fold,seed)))
    for entry in receipt['files']:
        path=p.ROOT/entry['path'];assert p.sha(path)==entry['sha256'];cached=joblib.load(path)
        inputs=p.tc.inputs_only(cached['inputs']);record=p.old.inference_record(cached['record']);record['names']=list(record['names'])
        c=dict(provider=p.shared._provider_for_match(entry['match']),match=entry['match'].removeprefix('skillcorner_had_'),
            view=f'feed_like_fresh_{entry["replicate"]}',labels=cached['labels'],period=inputs['period'],
            model_period=p.supervised.model_periods(inputs),timestamps=inputs['timestamps'],fps=float(inputs['fps']))
        values={a:{} for a in rows}
        for seed in p.SEEDS:
            raw,lead,candidate,teacher=loaded['leave_out_'+entry['provider'],seed]
            probs=[p.old.upsample(p.sn.predict(record,m,teacher),record,inputs) for m in (raw,lead,candidate)]
            lead_cross=parent.cross.cross(probs[1],probs[0]);candidate_cross=parent.cross.cross(probs[2],probs[0])
            final=probs[2] if arm.endswith('_full') else .5*(candidate_cross+lead_cross) if arm.endswith('_half') else candidate_cross
            values['leader'][str(seed)]=lead_cross;values['fixed_candidate'][str(seed)]=final
            dest=OUT/'components'/f'{entry["match"]}__fresh_{entry["replicate"]}__{seed}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise FileExistsError(dest)
            np.savez_compressed(dest,raw_feed=probs[0],leader_model=probs[1],candidate_model=probs[2])
            components[str(dest.relative_to(p.ROOT))]=p.sha(dest)
        for name in values:
            values[name]['ensemble']=np.mean(list(values[name].values()),axis=0)
            for member,prob in values[name].items():
                cell=dict(c,member=member);row,pred=p.score(cell,prob);rows[name].append(row)
                dest=OUT/'predictions'/name/f'{entry["match"]}__fresh_{entry["replicate"]}__{member}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():raise FileExistsError(dest)
                np.savez_compressed(dest,probabilities=prob,prediction=pred,**{k:cell[k] for k in ('labels','period','timestamps','model_period','fps')})
                files[str(dest.relative_to(p.ROOT))]=p.sha(dest)
        print('SCORED',entry['match'],entry['replicate'],flush=True)
    summaries={a:{m:p.scoring.summarize(r,m) for m in p.shared.MEMBERS} for a,r in rows.items()}
    p.write(p.RESULTS/f'{STUDY}_aggregate.json',dict(selection=selection,rows=rows,summaries=summaries,
        prediction_sha256=files,component_sha256=components,new_scored_cells=60,reused_clean_cells=30,
        independent_match_confirmation=False,organizer_reference_access=False,reselection=False))
    print({a:s['ensemble']['feed_like']['equal_provider_mean']['spell_f1'] for a,s in summaries.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('declare','prepare','select','run'))
    globals()[parser.parse_args().action]()
