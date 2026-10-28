"""One frozen dead-ball information-path ablation, with two fixed fit seeds.

All five scoring matches are exposed development data. No candidate selection,
threshold search, new organizer scoring, or automatic baseline promotion.
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
                         refine_features, supervised, tracking_corruption as tc,
                         error_analysis)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
DEFAULT = ROOT/'data/experiments/deadball_transfer_v1'
ARMS = ('factorized_all', 'factorized_player_D')
SEEDS = (2026, 2027)


def fingerprints():
    return dict(old.fingerprint(), **{str(p.relative_to(ROOT)): sha256(p)
        for p in (Path(__file__), Path(model.__file__))})


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'predeclared.json').exists():
        return check(out)
    previous = old.check(old.DEFAULT)
    for path, digest in previous['data_sha256'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError('External data changed')
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=fingerprints(), previous_spec_sha256=sha256(old.DEFAULT/'predeclared.json'),
        previous_report_sha256=sha256(old.DEFAULT/'report.json'),
        data_sha256=previous['data_sha256'], calibration=previous['calibration'],
        training_sources=previous['training_sources'], training_matches=previous['training_matches'],
        evaluation_matches=previous['evaluation_matches'],
        training_caches={p.name: sha256(p) for p in old.DEFAULT.glob('train_*.joblib')},
        arms=ARMS, seeds=SEEDS, views=old.VIEWS,
        hypothesis='Ball/team inputs permit provider-specific dead-ball shortcuts. Restrict P(D) to the existing strictly player-only feature block.',
        design='Two factorized networks with identical weights/capacity/optimizer/inputs/augmentation; only the D input path differs. Both execute two shared-weight GRU passes. Original joint GRU and baseline are archived comparators.',
        training='Unchanged joint-GRU clean/coarse and 25% whole-window ball-gap recipe; direct and H/A-swapped records; equal source/match weights; train-only normalization; eight epochs; same categorical likelihood, no D class weighting; AdamW .001/.0001, batch64, hidden24; fixed seeds2026/2027.',
        output='P(H/A)=(1-P(D))*P(H/A|live); D sigmoid and live softmax. Fixed original Viterbi cost3. D sees only player values and their missingness masks in constrained arm.',
        analysis='Report both seeds, mean/range (not a seed-selected checkpoint or ensemble), all providers, clean and two fixed stress views. Paired control isolates D information restriction; original joint comparison also changes output factorization.',
        guardrail='For both seeds: +.01 equal-provider clean spell vs current recipe; no provider clean D-F1 or macro-F1 more than .02 below current recipe; stress spell no more than .01 below current recipe. Descriptive development screen only, no promotion.',
        adaptive_exposure='Hypothesis follows the already-read five-match results. These are reused development matches; provider holdout does not undo researcher exposure. No fresh confirmation claim.',
        anti_overfit='Exactly two arms x two seeds x three provider folds; all 12 checkpoints frozen before any new scoring; fixed training duration; no early stopping, feature/threshold/decoder sweep, best-seed selection, or follow-on tuning from these scores.',
        baseline_changed=False, organizer_labels_read=False, fresh_confirmation=False)
    ce.write_json(out/'predeclared.json', spec, exclusive=True)
    return spec


def check(out):
    spec = json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256'] != fingerprints():
        raise ValueError('Frozen source changed')
    if spec['previous_report_sha256'] != sha256(old.DEFAULT/'report.json'):
        raise ValueError('Archived comparator report changed')
    return spec


def checkpoint(out, source, arm, seed):
    return out/f'without_{source}_{arm}_{seed}.joblib'


def training_records(source, spec):
    records = []
    for provider in old.training_sources(source):
        for path in dt.SPLITS[provider]['train']:
            cached = old.DEFAULT/f'train_{path.stem}.joblib'
            if sha256(cached) != spec['training_caches'][cached.name]:
                raise ValueError('Training cache changed')
            r = joblib.load(cached)
            for v, name in enumerate(('clean', 'legacy')):
                for swap in (False, True):
                    records.append(old.mixed_record(r[name]['joint'], r['gap']['joint'], path.stem, v, swap))
    return records


def fit(out, source, arm, seed):
    spec = check(out)
    path = checkpoint(out, source, arm, seed)
    if path.exists():
        ce.load_model(out, path)
        return
    if (out/'checkpoints_frozen.json').exists():
        raise ValueError('Training closed before evaluation')
    start = time.monotonic()
    fitted = model.fit(training_records(source, spec), player_only=arm == 'factorized_player_D', seed=seed)
    ce.freeze_model(out, path, fitted, dict(heldout=source, sources=spec['training_sources'][source],
        training_matches=spec['training_matches'][source], arm=arm, seed=seed,
        seconds=time.monotonic()-start, parameters=fitted['parameters'],
        optimizer_updates=fitted['optimizer_updates'], windows_per_epoch=fitted['windows_per_epoch']))
    print('FROZEN', source, arm, seed, flush=True)


def seal(out):
    check(out)
    rows=[]
    for source in dt.SPLITS:
        for arm in ARMS:
            for seed in SEEDS:
                path=checkpoint(out, source, arm, seed)
                ce.load_model(out, path)
                rows.append(dict(file=path.name, sha256=sha256(path), receipt_sha256=sha256(path.with_suffix('.json'))))
    destination=out/'checkpoints_frozen.json'
    if destination.exists():
        if json.loads(destination.read_text())['checkpoints'] != rows:
            raise ValueError('Frozen checkpoints changed')
    else:
        ce.write_json(destination, dict(created_utc=datetime.now(timezone.utc).isoformat(), checkpoints=rows), exclusive=True)


def evaluate(out, source):
    spec=check(out)
    if not (out/'checkpoints_frozen.json').exists():
        raise ValueError('Freeze all 12 checkpoints before scoring')
    seal(out)
    models={(a,s):ce.load_model(out,checkpoint(out,source,a,s)) for a in ARMS for s in SEEDS}
    destination=out/f'{source}_results.json'
    rows=json.loads(destination.read_text()) if destination.exists() else []
    for path in dt.SPLITS[source]['eval']:
        original,labels=ms.view(path)
        for view in old.VIEWS:
            if all(any(r['match']==path.stem and r['view']==view and r['arm']==a and r['seed']==s for r in rows) for a,s in models):
                continue
            inputs=original if view=='clean' else tc.corrupt(original,spec['calibration'],ce.seed(path.stem,view))
            started=time.monotonic()
            team,names=refine_features.features(inputs)
            record=ce.record(inputs,labels,source,path.stem,view)
            joint=dict(record,x=np.column_stack((team[record['indices']],record['x'])),names=names+['D_'+n for n in record['names']])
            feature_seconds=time.monotonic()-started
            periods=supervised.model_periods(inputs)
            for (arm,seed),fitted in models.items():
                if any(r['match']==path.stem and r['view']==view and r['arm']==arm and r['seed']==seed for r in rows):
                    continue
                started=time.monotonic()
                sampled=model.predict(joint,fitted)
                probability=np.column_stack([lt.upsample(sampled[:,k],record,periods) for k in range(3)])
                prediction=supervised.decode(probability,periods,float(inputs['fps']),lt.COST)
                seconds=time.monotonic()-started
                analysis=error_analysis.analyze(dict(inputs,labels=labels),prediction)
                m=analysis['metrics'];short=analysis['by_gt_duration'][:2]
                count=sum(v['n_gt'] for v in short);hits=sum(v['n_matched_at_iou_0_5'] for v in short)
                false=m['n_pred_spells']-m['spell_f1'][.5]['n_matched']
                minutes=np.isin(labels,supervised.LABELS).sum()/float(inputs['fps'])/60
                row=dict(source=source,match=path.stem,view=view,arm=arm,seed=seed,
                    spell_f1=m['spell_f1_mean'],macro_f1=m['macro_f1'],d_f1=m['per_class']['D']['f1'],
                    d_recall=m['per_class']['D']['recall'],d_precision=m['per_class']['D']['precision'],
                    short_spell_recall=hits/count if count else None,short_reference_spells=count,short_matched_spells=hits,
                    false_possessions=int(false),false_possessions_per_minute=false/minutes,
                    fragmented_reference_spells=analysis['overlap_structure']['n_fragmented_gt_spells'],
                    n_gt_spells=m['n_gt_spells'],n_pred_spells=m['n_pred_spells'],
                    feature_seconds=feature_seconds,inference_seconds=seconds,total_seconds=feature_seconds+seconds,
                    per_class=m['per_class'])
                np.savez_compressed(out/f'{source}_{path.stem}_{view}_{arm}_{seed}.npz',prediction=prediction)
                rows.append(row);ce.write_json(destination,rows)
                print('SCORED',source,path.stem,view,arm,seed,round(row['spell_f1'],5),round(row['d_f1'],5),flush=True)


def summarize(rows):
    keys=('spell_f1','macro_f1','d_f1','short_spell_recall','false_possessions_per_minute','total_seconds')
    summaries=[]
    for arm in sorted({r['arm'] for r in rows}):
        seeds=sorted({r['seed'] for r in rows if r['arm']==arm})
        for seed in seeds:
            for view in ('clean','corrupted'):
                per={}
                for source in dt.SPLITS:
                    selected=[r for r in rows if r['arm']==arm and r['seed']==seed and r['source']==source and ((r['view']=='clean')==(view=='clean'))]
                    per[source]={k:float(np.mean([r[k] for r in selected])) for k in keys}
                summaries.append(dict(arm=arm,seed=seed,view=view,per_source=per,
                    **{k:float(np.mean([p[k] for p in per.values()])) for k in keys}))
    return summaries


def report(out):
    spec=check(out);seal(out)
    rows=[r for source in dt.SPLITS for r in json.loads((out/f'{source}_results.json').read_text())]
    expected={(source,p.stem,view,a,s) for source in dt.SPLITS for p in dt.SPLITS[source]['eval'] for view in old.VIEWS for a in ARMS for s in SEEDS}
    if len(rows)!=len(expected) or {(r['source'],r['match'],r['view'],r['arm'],r['seed']) for r in rows}!=expected:
        raise ValueError('Incomplete or duplicate evaluation')
    archived=json.loads((old.DEFAULT/'report.json').read_text())
    comparators=[dict(r,seed=2026) for r in archived['rows'] if r['arm'] in ('baseline','had_gru')]
    summaries=summarize(rows+comparators)
    baseline={r['view']:r for r in summaries if r['arm']=='baseline'}
    screens=[]
    for arm in ARMS:
        for seed in SEEDS:
            clean=next(r for r in summaries if r['arm']==arm and r['seed']==seed and r['view']=='clean')
            stress=next(r for r in summaries if r['arm']==arm and r['seed']==seed and r['view']=='corrupted')
            spell=clean['spell_f1']-baseline['clean']['spell_f1']
            provider=min(clean['per_source'][s][k]-baseline['clean']['per_source'][s][k] for s in dt.SPLITS for k in ('d_f1','macro_f1'))
            stress_delta=stress['spell_f1']-baseline['corrupted']['spell_f1']
            screens.append(dict(arm=arm,seed=seed,clean_spell_delta=spell,worst_provider_d_or_macro_delta=provider,
                stress_spell_delta=stress_delta,passes=spell>=.01 and provider>=-.02 and stress_delta>=-.01))
    result=dict(spec=spec,rows=rows,archived_comparators=comparators,summaries=summaries,screens=screens,
        checkpoint_receipts={p.name:json.loads(p.read_text()) for p in out.glob('without_*.json')},
        baseline_changed=False,fresh_confirmation=False)
    ce.write_json(out/'report.json',result,exclusive=True)
    for r in summaries:
        print('SUMMARY',r['arm'],r['seed'],r['view'],*(round(r[k],5) for k in ('spell_f1','macro_f1','d_f1')),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','fit','seal','evaluate','report'))
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
            if args.source is None or args.arm is None or args.seed is None:
                parser.error('fit requires source, arm and seed')
            fit(args.out,args.source,args.arm,args.seed)
        else:
            for source in ([args.source] if args.source else dt.SPLITS): evaluate(args.out,source)


if __name__=='__main__': main()
