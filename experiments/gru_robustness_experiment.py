"""Matched player-slot augmentation for the latest player-only D joint GRU.

Public labels are never loaded. Historical baseline scores are reused; all new
checkpoints are sealed before scoring the reused external development cohort.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import warnings
import joblib
import numpy as np
from experiments import (live_gap_experiment as old, factorized_had as model,
                         learned_temporal as lt, corruption_experiment as ce,
                         dead_transfer as dt, multisource_eval as ms,
                         refine_domain, refine_features, robust, light_corruption as lc,
                         supervised, tracking_corruption as tc, error_analysis, spell_diagnostics)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
DEFAULT = ROOT/'data/experiments/gru_slot_robustness_v2'
ARMS = ('control', 'slot_dropout')
SEEDS = (2026, 2027)


def fingerprints():
    return dict(old.fingerprint(), **{str(p.relative_to(ROOT)): sha256(p)
        for p in (Path(__file__), Path(model.__file__), Path(spell_diagnostics.__file__))})


def slot_view(inputs, seed):
    """Keep ball evidence native and mask players before recomputing features."""
    out = tc.inputs_only(inputs)
    out['players'] = refine_domain.drop_player_slots(out, seed)['players']
    if 'player_observed' in out:
        out['player_observed'] = np.isfinite(out['players']).all(axis=-1)
    return out


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'predeclared.json').exists():
        return check(out)
    manifest = json.loads((ROOT/'experiments/eval_set/manifest.json').read_text())
    data = {}
    for provider in manifest['sources'].values():
        for role in ('train', 'selection'):
            for entry in provider[role]:
                if sha256(ROOT/entry['path']) != entry['sha256']:
                    raise ValueError('External input hash mismatch')
                data[entry['path']] = entry['sha256']
    archived = old.DEFAULT/'report.json'
    previous = json.loads(archived.read_text())
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=fingerprints(),
        data_sha256=data, previous_report_sha256=sha256(archived),
        calibration=previous['spec']['calibration'], arms=ARMS, seeds=SEEDS, views=old.VIEWS,
        training_sources={s: old.training_sources(s) for s in dt.SPLITS},
        training_matches={s: [p.stem for t in old.training_sources(s) for p in dt.SPLITS[t]['train']] for s in dt.SPLITS},
        evaluation_matches={s: [p.stem for p in dt.SPLITS[s]['eval']] for s in dt.SPLITS},
        hypothesis='Missing player robustness lost by joint training can be recovered by restoring the legacy raw player-slot-dropout mechanism without changing architecture, labels or update budget.',
        mechanism='Control: clean/coarse-ball views plus 25% whole-window ball-gap replacement. Slot-dropout: same first view; second coarse-ball view and its gap alternative also mask 25% of each team slots per continuous block, retaining at least five per team. Recompute all features from each raw view. Exactly half the training records carry dropout; same H/A swaps and window choices.',
        training='Player-only D factorized shared-weight bidirectional GRU hidden24; 8 epochs; AdamW .001/.0001; batch64; clipping1; 5Hz; 10s core +5s context on each side; equal provider/match mass; training-only normalization; seeds2026/2027.',
        primary='Equal-provider official spell F1; exact ties macro F1; report fixed auxiliary .7 spell + .3 macro.',
        guardrails='Both seeds: clean spell >= archived current recipe +.01; each provider clean D/macro >= baseline -.02; stress spell >= baseline -.01. Also report paired augmentation-control deltas and whether clean spell loss exceeds .01.',
        budget='12 fits (2 arms x2 seeds x3 full-pipeline provider holdouts), 60 new match/view/arm/seed evaluations. Freeze all before scoring; no extra arms, early stopping, decoder/threshold tuning, seed selection or public scoring.',
        decoder_switch_cost=lt.COST, historical_development_exposure=True,
        baseline_changed=False, organizer_labels_read=False, fresh_confirmation=False)
    ce.write_json(out/'predeclared.json', spec, exclusive=True)
    return spec


def check(out):
    spec = json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256'] != fingerprints():
        raise ValueError('Frozen source changed')
    if spec['previous_report_sha256'] != sha256(old.DEFAULT/'report.json'):
        raise ValueError('Archived comparator changed')
    return spec


def prepare(out, source):
    spec = check(out)
    for i, path in enumerate(dt.SPLITS[source]['train']):
        dest = out/f'train_{path.stem}.joblib'
        receipt = dest.with_suffix('.json')
        if dest.exists():
            if not receipt.exists() or json.loads(receipt.read_text())['sha256'] != sha256(dest):
                raise ValueError('Incomplete or changed feature cache')
            continue
        started = time.monotonic()
        inputs, labels = ms.view(path)
        if sha256(path) != spec['data_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Training input changed')
        legacy = robust.masked_inputs(inputs, lt.SEED+i)
        gap = lc.ball_gap_view(inputs, spec['calibration'], ce.seed(path.stem, 'train'))
        views = dict(clean=inputs, legacy=legacy, gap=gap,
                     dropout_legacy=slot_view(legacy, lt.SEED+i),
                     dropout_gap=slot_view(gap, lt.SEED+i))
        records = {name: old.feature_view(view, labels, source, path.stem) for name,view in views.items()}
        joblib.dump(records, dest)
        ce.write_json(receipt, dict(sha256=sha256(dest), input_sha256=sha256(path),
            predeclared_sha256=sha256(out/'predeclared.json'), seconds=time.monotonic()-started), exclusive=True)
        print('PREPARED', source, path.stem, round(time.monotonic()-started, 1), flush=True)


def paired_records(cache, match, arm):
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    records = []
    for variant in (0, 1):
        original = 'clean' if variant == 0 else ('legacy' if arm == 'control' else 'dropout_legacy')
        alternate = 'dropout_gap' if arm == 'slot_dropout' and variant == 1 else 'gap'
        for swap in (False, True):
            records.append(old.mixed_record(cache[original]['joint'], cache[alternate]['joint'], match, variant, swap))
    return records


def training_records(out, source, arm):
    records = []
    for provider in old.training_sources(source):
        for path in dt.SPLITS[provider]['train']:
            cached = out/f'train_{path.stem}.joblib'
            receipt = json.loads(cached.with_suffix('.json').read_text())
            if sha256(cached) != receipt['sha256'] or receipt['predeclared_sha256'] != sha256(out/'predeclared.json'):
                raise ValueError('Feature cache changed')
            records.extend(paired_records(joblib.load(cached), path.stem, arm))
    return records


def checkpoint(out, source, arm, seed):
    return out/f'without_{source}_{arm}_{seed}.joblib'


def fit(out, source, arm, seed):
    spec = check(out)
    path = checkpoint(out, source, arm, seed)
    if path.exists():
        ce.load_model(out, path)
        return
    if (out/'checkpoints_frozen.json').exists():
        raise ValueError('Training closed')
    start = time.monotonic()
    fitted = model.fit(training_records(out, source, arm), player_only=True, seed=seed)
    ce.freeze_model(out, path, fitted, dict(heldout=source, sources=spec['training_sources'][source],
        training_matches=spec['training_matches'][source], arm=arm, seed=seed,
        seconds=time.monotonic()-start, parameters=fitted['parameters'],
        optimizer_updates=fitted['optimizer_updates'], windows_per_epoch=fitted['windows_per_epoch']))
    print('FROZEN', source, arm, seed, flush=True)


def seal(out):
    check(out)
    rows=[]
    for source in dt.SPLITS:
        for seed in SEEDS:
            paired=[]
            for arm in ARMS:
                path=checkpoint(out, source, arm, seed)
                ce.load_model(out,path)
                receipt=json.loads(path.with_suffix('.json').read_text())
                paired.append(tuple(receipt[k] for k in ('parameters','optimizer_updates','windows_per_epoch')))
                rows.append(dict(file=path.name,sha256=sha256(path),receipt_sha256=sha256(path.with_suffix('.json'))))
            if paired[0] != paired[1]:
                raise ValueError('Paired model capacity or training budget differs')
    dest=out/'checkpoints_frozen.json'
    if dest.exists():
        if json.loads(dest.read_text())['checkpoints'] != rows:
            raise ValueError('Sealed checkpoints changed')
    else:
        ce.write_json(dest,dict(created_utc=datetime.now(timezone.utc).isoformat(),checkpoints=rows),exclusive=True)


def evaluate(out, source):
    spec=check(out)
    if not (out/'checkpoints_frozen.json').exists():
        raise ValueError('Seal all checkpoints before scoring')
    seal(out)
    models={(a,s):ce.load_model(out,checkpoint(out,source,a,s)) for a in ARMS for s in SEEDS}
    dest=out/f'{source}_results.json'
    rows=json.loads(dest.read_text()) if dest.exists() else []
    for path in dt.SPLITS[source]['eval']:
        if sha256(path) != spec['data_sha256'][str(path.relative_to(ROOT))]:
            raise ValueError('Evaluation input changed')
        original,labels=ms.view(path)
        for view in old.VIEWS:
            if all(any(r['match']==path.stem and r['view']==view and r['arm']==a and r['seed']==s for r in rows) for a,s in models):
                continue
            inputs=original if view=='clean' else tc.corrupt(original,spec['calibration'],ce.seed(path.stem,view))
            started=time.monotonic()
            joint=old.feature_view(inputs,labels,source,path.stem)['joint']
            feature_seconds=time.monotonic()-started
            periods=supervised.model_periods(inputs)
            joblib.dump(dict(joint=joint, inputs=tc.inputs_only(inputs), labels=labels), out/f'eval_{path.stem}_{view}.joblib')
            for (arm,seed),fitted in models.items():
                if any(r['match']==path.stem and r['view']==view and r['arm']==arm and r['seed']==seed for r in rows):
                    continue
                started=time.monotonic()
                sampled=model.predict(joint,fitted)
                probability=np.column_stack([lt.upsample(sampled[:,k],joint,periods) for k in range(3)])
                prediction=supervised.decode(probability,periods,float(inputs['fps']),lt.COST)
                seconds=time.monotonic()-started
                analysis=error_analysis.analyze(dict(inputs,labels=labels),prediction)
                m=analysis['metrics']; short=analysis['by_gt_duration'][:2]
                count=sum(v['n_gt'] for v in short);hits=sum(v['n_matched_at_iou_0_5'] for v in short)
                false=m['n_pred_spells']-m['spell_f1'][.5]['n_matched']
                minutes=np.isin(labels,supervised.LABELS).sum()/float(inputs['fps'])/60
                row=dict(source=source,match=path.stem,view=view,arm=arm,seed=seed,
                    spell_f1=m['spell_f1_mean'],macro_f1=m['macro_f1'],d_f1=m['per_class']['D']['f1'],
                    d_recall=m['per_class']['D']['recall'],d_precision=m['per_class']['D']['precision'],
                    short_spell_recall=hits/count if count else None,short_reference_spells=count,short_matched_spells=hits,
                    false_possessions=int(false),false_possessions_per_minute=false/minutes,
                    fragmented_reference_spells=analysis['overlap_structure']['n_fragmented_gt_spells'],
                    n_gt_spells=m['n_gt_spells'],n_pred_spells=m['n_pred_spells'],graded_frames=int(np.isin(labels,supervised.LABELS).sum()),
                    feature_seconds=feature_seconds,inference_seconds=seconds,total_seconds=feature_seconds+seconds,
                    per_class=m['per_class'], spell_diagnostics=spell_diagnostics.spell_diagnostics(labels,prediction,period=inputs['period'],fps=float(inputs['fps']),timestamps=inputs.get('timestamps')),spell_thresholds={str(k):v for k,v in m['spell_f1'].items()})
                np.savez_compressed(out/f'{source}_{path.stem}_{view}_{arm}_{seed}.npz',prediction=prediction,probability=probability)
                rows.append(row);ce.write_json(dest,rows)
                print('SCORED',source,path.stem,view,arm,seed,round(row['spell_f1'],5),flush=True)


def report(out):
    from experiments.deadball_transfer_experiment import summarize
    spec=check(out);seal(out)
    rows=[r for s in dt.SPLITS for r in json.loads((out/f'{s}_results.json').read_text())]
    expected={(s,p.stem,v,a,k) for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] for v in old.VIEWS for a in ARMS for k in SEEDS}
    if len(rows)!=len(expected) or {(r['source'],r['match'],r['view'],r['arm'],r['seed']) for r in rows}!=expected:
        raise ValueError('Incomplete or duplicate evaluation')
    archived=json.loads((old.DEFAULT/'report.json').read_text())
    base=[dict(r,seed=2026) for r in archived['rows'] if r['arm']=='baseline']
    for r in rows:
        reference=next(b for b in base if (b['source'],b['match'],b['view'])==(r['source'],r['match'],r['view']))
        if reference['n_gt_spells'] != r['n_gt_spells']:
            raise ValueError('Grading population changed')
    summaries=summarize(rows+base)
    for r in summaries: r['weighted_70_30']=.7*r['spell_f1']+.3*r['macro_f1']
    baseline={r['view']:r for r in summaries if r['arm']=='baseline'}
    screens=[]
    for arm in ARMS:
        for seed in SEEDS:
            clean=next(r for r in summaries if r['arm']==arm and r['seed']==seed and r['view']=='clean')
            stress=next(r for r in summaries if r['arm']==arm and r['seed']==seed and r['view']=='corrupted')
            delta=clean['spell_f1']-baseline['clean']['spell_f1']
            provider=min(clean['per_source'][s][k]-baseline['clean']['per_source'][s][k] for s in dt.SPLITS for k in ('d_f1','macro_f1'))
            stress_delta=stress['spell_f1']-baseline['corrupted']['spell_f1']
            screens.append(dict(arm=arm,seed=seed,clean_spell_delta=delta,worst_provider_d_or_macro_delta=provider,
                stress_spell_delta=stress_delta,passes=delta>=.01 and provider>=-.02 and stress_delta>=-.01))
    result=dict(spec=spec,rows=rows,archived_comparators=base,summaries=summaries,screens=screens,
                baseline_changed=False,fresh_confirmation=False)
    ce.write_json(out/'report.json',result,exclusive=True)
    for r in summaries: print('SUMMARY',r['arm'],r['seed'],r['view'],r['spell_f1'],r['macro_f1'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','prepare','fit','seal','evaluate','report'))
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--source',choices=tuple(dt.SPLITS))
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if args.action=='freeze': freeze(args.out)
        elif args.action=='seal': seal(args.out)
        elif args.action=='report': report(args.out)
        elif args.action=='fit':
            if args.source is None or args.arm is None or args.seed is None: parser.error('fit requires source, arm, seed')
            fit(args.out,args.source,args.arm,args.seed)
        else:
            for s in ([args.source] if args.source else dt.SPLITS):
                (prepare if args.action=='prepare' else evaluate)(args.out,s)


if __name__=='__main__': main()
