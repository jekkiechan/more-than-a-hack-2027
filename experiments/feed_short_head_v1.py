"""Predeclared targeted H/A plus unchanged raw-feed D probability cross."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from experiments import feed_short_sampling_v1 as study,feed_head_transfer_v1 as head
p=study.p
STUDY='feed_short_head_v1'
OUT=p.ROOT/'data/experiments'/STUDY
DECL=p.RESULTS/f'{STUDY}_declaration.json'


def declare():
    study.check()
    if (p.RESULTS/f'{study.STUDY}_aggregate.json').exists():raise ValueError('Declare before parent scores')
    p.write(DECL,{'study':STUDY,'utc':datetime.now(timezone.utc).isoformat(),
        'question':'Can targeted H/A sampling help without the accompanying resampled D path?',
        'condition':'Only if parent misses .55; one fixed arm, 45 original score cells, zero fits.',
        'arm':'Conditional H/A from short_sampling + native D from raw-feed augmented control, within matched seed, then equal mean across both seeds; fixed f062 decoder.',
        'guards':'Same promoted-baseline guards as parent; must improve .525240 control. Report historical +.01 clean screen separately.',
        'stop':'No other crosses, weights or decoders. Original development evidence only; no organizer scoring or promotion.',
        'parent_declaration_sha256':p.sha(study.DECL),
        'source_sha256':{str(Path(__file__).relative_to(p.ROOT)):p.sha(Path(__file__)),str(Path(head.__file__).relative_to(p.ROOT)):p.sha(Path(head.__file__))}})


def run():
    d=p.read(DECL);study.check();assert d['parent_declaration_sha256']==p.sha(study.DECL)
    for rel,digest in d['source_sha256'].items():assert p.sha(p.ROOT/rel)==digest
    previous=p.read(p.RESULTS/f'{study.STUDY}_aggregate.json')
    if previous['stretch_reached']:
        p.write(p.RESULTS/f'{STUDY}_skipped.json',{'reason':'Parent reached .55','scored':False});return
    assert p.read(p.RESULTS/f'{study.STUDY}_verification.json')['passed']
    p.write(OUT/'score_lock.json',{'declaration_sha256':p.sha(DECL),'parent_aggregate_sha256':p.sha(p.RESULTS/f'{study.STUDY}_aggregate.json')})
    cells=p.shared._load_cells(p.read(p.DECLARATION));values={};rows=[];files={}
    for c in cells:
        if c['member']=='ensemble':continue
        key=tuple(c[k] for k in ('provider','match','view','member'))
        path=study.OUT/'predictions/original/short_sampling'/('__'.join(key)+'.npz')
        assert p.sha(path)==previous['prediction_sha256'][str(path.relative_to(p.ROOT))]
        with np.load(path) as z:values[key]=head.cross(z['probabilities'],study.parent_probability(c))
    for c in cells:
        key=tuple(c[k] for k in ('provider','match','view','member'))
        value=np.mean([values[key[:3]+(str(s),)] for s in p.SEEDS],axis=0) if c['member']=='ensemble' else values[key]
        row,pred=p.score(c,value);rows.append(row)
        path=OUT/'predictions'/('__'.join(key)+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,probabilities=value,prediction=pred,**{k:c[k] for k in ('labels','period','timestamps','model_period','fps')});files[str(path.relative_to(p.ROOT))]=p.sha(path)
    summaries={m:p.scoring.summarize(rows,m) for m in p.shared.MEMBERS};s=summaries['ensemble'];base=previous['baseline'];control=previous['control']
    p.write(p.RESULTS/f'{STUDY}_aggregate.json',{'rows':rows,'summaries':summaries,'guards_pass':study.selection.eligible(s,base),'improves_control':study.selection.rank(s)>study.selection.rank(control),'stretch_reached':s['feed_like']['equal_provider_mean']['spell_f1']>=.55,'historical_clean_screen_pass':s['clean']['equal_provider_mean']['spell_f1']>=base['clean']['equal_provider_mean']['spell_f1']+.01,'prediction_sha256':files,'organizer_reference_access':False})
    print({v:s[v]['equal_provider_mean']['spell_f1'] for v in ('clean','feed_like')},flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=('declare','run'));globals()[a.parse_args().action]()
