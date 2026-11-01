"""Recover a complete grid readout after a NumPy-bool JSON write failure.

Preserve the frozen runner, scoring lock, all decoded arrays and partial output.
Recompute only readout metrics from those arrays, with no inference or new arms.
"""
from pathlib import Path
import json
import numpy as np
from experiments import feed_short_calibration_v1 as s
p=s.p
ORIGINAL=p.RESULTS/f'{s.STUDY}_aggregate.json'
RESULT=p.RESULTS/f'{s.STUDY}_readout_v2.json'
AMENDMENT=p.RESULTS/f'{s.STUDY}_readout_recovery_v2.json'


def native(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:native(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [native(v) for v in value]
    return value


def write_complete(path,value):
    value=native(value)
    json.dumps(value,allow_nan=False) # Fail before opening a result file.
    p.write(path,value)


def run():
    s.game.check();decl=p.read(s.DECL)
    for rel,digest in decl['bindings'].items():assert p.sha(p.ROOT/rel)==digest
    failure=s.OUT/'initial_readout_failure.log';assert failure.exists() and 'not JSON serializable' in failure.read_text()
    try:p.read(ORIGINAL)
    except json.JSONDecodeError:pass
    else:raise ValueError('Recovery requires the preserved incomplete output')
    files=sorted((s.OUT/'predictions').rglob('*.npz'));assert len(files)==435
    seed_folders={x.parent.name for x in (s.OUT/'predictions/selected_seeds').rglob('*.npz')};assert len(seed_folders)==1
    recorded_winner=next(iter(seed_folders));lock=p.read(s.OUT/'score_lock.json')
    write_complete(AMENDMENT,{'reason':'NumPy bool failed JSON serialization after all 405 grid and 30 selected-seed outputs were saved','partial_aggregate_sha256':p.sha(ORIGINAL),'failure_log_sha256':p.sha(failure),'lock_sha256':p.sha(s.OUT/'score_lock.json'),'saved_predictions':{str(f.relative_to(p.ROOT)):p.sha(f) for f in files},'winner_recorded_before_failure':recorded_winner,'no_new_inference_or_arms':True,'source_sha256':p.sha(Path(__file__))})
    cells={tuple(c[k] for k in ('provider','match','view','member')):c for c in p.shared._load_cells(p.read(p.DECLARATION))}
    probability={}
    for rel,digest in lock['probability_sources'].items():
        path=p.ROOT/rel;assert p.sha(path)==digest
        with np.load(path) as z:probability[tuple(path.stem.split('__'))]=z['probabilities']
    rows={cfg['id']:[] for cfg in s.configs()};diagnostics=[];hashes={};cfgs={c['id']:c for c in s.configs()}
    for file in files:
        key=tuple(file.stem.split('__'));c=cells[key];cfg=cfgs[file.parent.name]
        with np.load(file) as z:prediction=z['prediction']
        np.testing.assert_array_equal(prediction,p.decoder.decode(probability[key][None],c['model_period'],c['fps'],cfg))
        row={k:c[k] for k in ('provider','match','view','member')};row.update(p.shared._score_prediction(c['labels'],prediction,c['period'],c['timestamps'],c['fps']))
        (diagnostics if 'selected_seeds' in file.parts else rows[cfg['id']]).append(row)
        hashes[str(file.relative_to(p.ROOT))]=p.sha(file)
    baseline,choices=s.candidates();leader=next(x for x in choices if x['name']==lock['leader'])
    assert p.sha(leader['aggregate'])==lock['leader_aggregate_sha256']
    summaries={k:p.scoring.summarize(v,'ensemble') for k,v in rows.items()};guards={k:s.record.selection.eligible(v,baseline) for k,v in summaries.items()}
    winner=max((k for k in summaries if guards[k]),key=lambda k:s.record.selection.rank(summaries[k]));assert winner==recorded_winner
    result={'leader':leader['name'],'leader_summary':leader['summary'],'rows':rows,'summaries':summaries,'guards_pass':guards,'winner':winner,'config':cfgs[winner],'winner_summary':summaries[winner],'stretch_reached':summaries[winner]['feed_like']['equal_provider_mean']['spell_f1']>=.55,'historical_clean_screen_pass':{k:v['clean']['equal_provider_mean']['spell_f1']>=baseline['clean']['equal_provider_mean']['spell_f1']+.01 for k,v in summaries.items()},'seed_diagnostics':diagnostics,'seed_summaries':{str(seed):p.scoring.summarize(diagnostics,str(seed)) for seed in p.SEEDS},'prediction_sha256':hashes,'model_probability_sha256':lock['probability_sources'],'organizer_reference_access':False,'independent_confirmation':False,'recovery_amendment_sha256':p.sha(AMENDMENT)}
    write_complete(RESULT,result);print(winner,cfgs[winner],{v:summaries[winner][v]['equal_provider_mean']['spell_f1'] for v in ('clean','feed_like')},flush=True)

if __name__=='__main__':run()
