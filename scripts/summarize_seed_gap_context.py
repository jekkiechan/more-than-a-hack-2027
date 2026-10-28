"""Post-hoc readout of the frozen study; no fitting or candidate selection.

Rates use replicate -> match -> provider averaging. Empty strata are omitted
with provider support reported, rather than treated as zero accuracy.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import seed_gap_context as study, supervised, gap_diagnostics as gd
from experiments.eval_set.protocol import sha256
import joblib


def balanced(rows,fn):
    providers={}
    for source in sorted({r['source'] for r in rows}):
        values=[]
        for match in sorted({r['match'] for r in rows if r['source']==source}):
            rates=[fn(r) for r in rows if r['match']==match]
            rates=[r for r in rates if r is not None]
            if rates: values.append(np.mean(rates))
        if values: providers[source]=float(np.mean(values))
    return dict(value=float(np.mean(list(providers.values()))) if providers else None,
                providers=providers,n_providers=len(providers))


def diagnostic_summary(rows):
    result={}
    definitions={
        'frames':dict(team_accuracy=('correct_team','live_frames'),
            wrong_team_rate=('wrong_team','live_frames'),false_dead_rate=('false_dead','live_frames'),
            missed_dead_rate=('missed_dead','dead_frames')),
        'spells':dict(short_miss_rate=('short_missed','short_reference'),
            fragmentation_rate=('fragmented','reference_spells'),miss_rate=('missed','reference_spells')),
        'turnovers':dict(recall_1s=('matched_within_1s','reference'),
            recall_5s=('matched_within_5s','reference'),
            mean_absolute_seconds_matched=('absolute_error_seconds_sum','matched_within_5s'),
            mean_signed_seconds_matched=('signed_error_seconds_sum','matched_within_5s'))}
    for section,mappings in definitions.items():
        result[section]={}
        for stratum in rows[0]['diagnostics'][section]:
            result[section][stratum]={}
            for metric,(num,den) in mappings.items():
                def fn(r):
                    d=r['diagnostics'][section][stratum]
                    return gd.ratio(d[num],d[den])
                result[section][stratum][metric]=balanced(rows,fn)
            # Exposures are descriptive sums, not independent samples; stress
            # includes repeated evaluation of the same reference population.
            result[section][stratum]['summed_exposures']={
                k:sum(r['diagnostics'][section][stratum][k] for r in rows)
                for k in sorted(set(v for pair in mappings.values() for v in pair))}
    return result


def mechanism(base,long):
    """Post-hoc partition of long-gap live rows by newly visible bracketing.

    Counts identify associations, not causal effects of individual contacts.
    """
    rows=[]
    for source,path,view,replicate in study.cells():
        cached=joblib.load(study.OUT/f'{path.stem}_{view}_{replicate}.joblib')
        inputs,labels,cov=cached['inputs'],cached['labels'],cached['cov']
        live=np.isin(labels,('H','A')) & (cov['gap']>=10)
        both=(cov['before']>=0)&(cov['after']>=0)
        groups=dict(all_long=live,newly_bracketed=live & ~cov['bracket']['5'] & cov['bracket']['15'],
            already_bracketed=live & cov['bracket']['5'],
            still_outside_window=live & both & ~cov['bracket']['15'],
            no_two_contacts=live & ~both)
        with np.load(study.OUT/f'{path.stem}_{view}_{replicate}.npz') as saved:
            for seed in study.SEEDS:
                prediction=supervised.decode(saved[f'seed_{seed}'],supervised.model_periods(inputs),float(inputs['fps']),3)
                with np.load(study.OUT/f'{path.stem}_{view}_{replicate}_context15_{seed}.npz') as candidate:
                    q=candidate['prediction']
                for group,mask in groups.items():
                    n=int(mask.sum())
                    rows.append(dict(source=source,match=path.stem,view=view,replicate=replicate,seed=seed,
                        group=group,live_frames=n,control_correct=int((mask&(prediction==labels)).sum()),
                        context_correct=int((mask&(q==labels)).sum()),
                        corrected=int((mask&(prediction!=labels)&(q==labels)).sum()),
                        regressed=int((mask&(prediction==labels)&(q!=labels)).sum())))
    summary=[]
    for seed in study.SEEDS:
        for view in ('clean','feed_like'):
            for group in ('all_long','newly_bracketed','already_bracketed','still_outside_window','no_two_contacts'):
                selected=[r for r in rows if (r['seed'],r['view'],r['group'])==(seed,view,group)]
                summary.append(dict(seed=seed,view=view,group=group,
                    control_accuracy=balanced(selected,lambda r:gd.ratio(r['control_correct'],r['live_frames'])),
                    context_accuracy=balanced(selected,lambda r:gd.ratio(r['context_correct'],r['live_frames'])),
                    paired_accuracy_delta=balanced(selected,lambda r:gd.ratio(r['context_correct']-r['control_correct'],r['live_frames'])),
                    summed_frame_exposures=sum(r['live_frames'] for r in selected)))
    return dict(rows=rows,summary=summary,
        caveat='Post-hoc diagnostic partition; observing contact proxies inside a window is not proof that the network uses them. Empty strata omit unsupported providers.')


def main():
    base=json.loads(study.RESULT.read_text())
    path=ROOT/'experiments/results/seed_gap_context_long.json'
    long=json.loads(path.read_text()) if path.exists() else None
    allrows=base['rows']+(long['rows'] if long else [])
    output=dict(source_sha256=sha256(__file__),
        base_report_sha256=sha256(study.RESULT),
        long_report_sha256=sha256(path) if long else None,
        status='Post-hoc descriptive readout; original model declaration and evaluation unchanged',
        aggregation='Replicates, then matches, then equal providers. Empty diagnostic strata omit unsupported providers and report support. Summed exposures are repeated rows, not independent samples.',
        diagnostic_summaries=[dict(recipe=recipe,view=view,
            **diagnostic_summary([r for r in allrows if r['recipe']==recipe and r['view']==view]))
            for recipe in sorted({r['recipe'] for r in allrows}) for view in ('clean','feed_like')])
    if long: output['mechanism']=mechanism(base,long)
    study.write(ROOT/'experiments/results/seed_gap_context_readout.json',output)


if __name__=='__main__': main()
