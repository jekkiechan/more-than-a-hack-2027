"""Replay the fixed calibration grid and selected-seed diagnostics."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from experiments import feed_short_calibration_v1 as s,metrics
from scripts.verify_feed_fresh_and_diversity import aggregate
p=s.p

def main():
    result=p.read(p.RESULTS/f'{s.STUDY}_aggregate.json');decl=p.read(s.DECL);s.game.check()
    for rel,digest in decl['bindings'].items():assert p.sha(ROOT/rel)==digest
    cells={tuple(c[k] for k in ('provider','match','view','member')):c for c in p.shared._load_cells(p.read(p.DECLARATION))}
    probabilities={}
    for rel,digest in result['model_probability_sha256'].items():
        file=ROOT/rel;assert p.sha(file)==digest
        key=tuple(file.stem.split('__'))
        with np.load(file) as z:
            probabilities[key]=z['probabilities']
            for field in ('labels','period','timestamps','model_period','fps'):np.testing.assert_array_equal(z[field],cells[key][field])
    count=0
    def check(row,cfg,group):
        nonlocal count
        key=tuple(row[k] for k in ('provider','match','view','member'));c=cells[key]
        file=s.OUT/'predictions'/group/cfg['id']/('__'.join(key)+'.npz');assert p.sha(file)==result['prediction_sha256'][str(file.relative_to(ROOT))]
        with np.load(file) as z:pred=z['prediction']
        np.testing.assert_array_equal(pred,p.decoder.decode(probabilities[key][None],c['model_period'],c['fps'],cfg))
        score=metrics.evaluate_tracking(c['labels'],pred,period=c['period'],timestamps=c['timestamps'],fps=c['fps'])
        for k,v in [('spell_f1',score['spell_f1_mean']),('macro_f1',score['macro_f1']),('d_f1',score['per_class']['D']['f1'])]:assert abs(row[k]-v)<=1e-12
        count+=1
    for cfg in s.configs():
        for row in result['rows'][cfg['id']]:check(row,cfg,'grid')
        aggregate(result['rows'][cfg['id']],'ensemble',result['summaries'][cfg['id']])
    for row in result['seed_diagnostics']:check(row,result['config'],'selected_seeds')
    for seed in p.SEEDS:aggregate(result['seed_diagnostics'],str(seed),result['seed_summaries'][str(seed)])
    control=next(c['id'] for c in s.configs() if c['dt']==4 and c['ht']==1 and c['ha']==3)
    assert result['summaries'][control]==result['leader_summary']
    baseline,_=s.candidates()
    expected={k:s.record.selection.eligible(v,baseline) for k,v in result['summaries'].items()};assert expected==result['guards_pass']
    assert result['winner']==max((k for k in expected if expected[k]),key=lambda k:s.record.selection.rank(result['summaries'][k]))
    assert count==435
    receipt={'passed':True,'official_cells':count,'grid_configurations':27,'selected_seed_diagnostic_cells':30,'control_exact':True,'unchanged_native_grading':True,'selection_and_guards_checked':True,'aggregate_sha256':p.sha(p.RESULTS/f'{s.STUDY}_aggregate.json'),'verifier_sha256':p.sha(Path(__file__)),'organizer_reference_access':False}
    p.write(p.RESULTS/f'{s.STUDY}_verification.json',receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
