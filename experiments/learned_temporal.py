"""Bounded, multi-provider temporal D ablation with fixed H/A attribution.

Offline context is allowed; no input window crosses a match, period or time gap.
The public challenge labels are never loaded by development commands.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from threadpoolctl import threadpool_limits

from experiments import dead_transfer, multisource_eval, refine_domain, robust, supervised

ROOT = Path(__file__).resolve().parents[1]
RATE = 5
CORE, CONTEXT = 50, 25  # 10 s outputs with 5 s on either side: 20 s input.
EPOCHS, SEED, COST = 8, 2026, 3
MODELS = ('hgb', 'ordered_hgb', 'lgbm_008', 'lgbm_002', 'tcn', 'bigru')
FOLDS = ('pooled', 'without_sportec', 'without_metrica', 'without_skillcorner')
SPEC = {
    'models': MODELS, 'folds': FOLDS, 'rate_hz': RATE,
    'neural': {'epochs': EPOCHS, 'core_frames': CORE, 'context_each_side': CONTEXT,
               'batch_size': 64, 'hidden': 24, 'learning_rate': .001,
               'weight_decay': .0001, 'seed': SEED,
               'tcn_dilations': [1, 2, 4, 8], 'tcn_kernel': 5},
    'hgb': {'max_iter': 160, 'max_leaf_nodes': 31, 'learning_rate': .08},
    'lgbm': {'learning_rate_estimators': [[.08, 160], [.02, 640]],
             'num_leaves': 31, 'min_child_samples': 40, 'reg_lambda': 1},
    'ordered_features': 'raw collective features at +/-1,3,5 s plus future-minus-past at those lags',
    'd_inputs': 'player motion/shape only; no ball, owner, event or target fields',
    'augmentation': 'clean + fixed-seed player-slot dropout; equal source and match weights',
    'missingness': 'train-only median/std normalization plus explicit finite-value masks for neural inputs',
    'decode': {'switch_cost': COST, 'd_threshold': .5},
    'attribution': 'fixed pooled multisource_eval HGB conditional H/A for every D head and fold',
    'provider_holdout_scope': 'D head only; fixed H/A backbone has seen training matches from all providers',
    'evaluation': 'existing development matches only; no fresh confirmation or public-label selection',
    'selection': 'equal-provider mean official spell F1 on D-head provider-holdout rows; tie macro F1; baseline remains eligible',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_spec(out):
    spec = dict(SPEC, source_sha256=digest(__file__), splits={
        s: {role: [p.stem for p in paths] for role, paths in split.items()}
        for s, split in dead_transfer.SPLITS.items()})
    path = out/'predeclared.json'
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != json.loads(json.dumps(v)) for k, v in spec.items()):
            raise ValueError('Experiment spec/source changed; use a new output directory')
    else:
        path.write_text(json.dumps(dict(spec, created_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')


def sampled_indices(data):
    """Uniform five-Hz grids reset at every input discontinuity."""
    fps = float(data['fps'])
    stride = round(fps/RATE)
    if stride < 1 or not np.isclose(stride*RATE, fps):
        raise ValueError('Expected an integer multiple of the common sampling rate')
    return np.concatenate([np.arange(a, b, stride) for a, b in supervised.blocks(supervised.model_periods(data))])


def player_features(inputs):
    """A strict allowlist prevents labels or ball metadata reaching the D head."""
    clean = {k: inputs[k] for k in ('players', 'teams', 'period', 'fps', 'timestamps', 'pitch') if k in inputs}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return refine_domain.features(clean)


def prepare_record(path, source, augment=False, seed=SEED):
    inputs, keep, target = dead_transfer.prepare(path)
    periods = supervised.model_periods(inputs)
    indices = sampled_indices(inputs)
    variants = [inputs]
    if augment:
        variants.append(refine_domain.drop_player_slots(inputs, seed))
    records = []
    for i, view in enumerate(variants):
        x, names = player_features(view)
        y = np.where(keep[indices], target[indices], -1).astype(np.int8)
        records.append(dict(source=source, match=path.stem, variant=i,
                            x=x[indices], y=y, period=periods[indices], names=names,
                            indices=indices, native_n=len(periods), fps=float(inputs['fps'])))
    return records


def prepare(out):
    for source, split in dead_transfer.SPLITS.items():
        for role, paths in split.items():
            for i, path in enumerate(paths):
                cache = out/f'{path.stem}_d_features.joblib'
                if not cache.exists():
                    started = time.monotonic()
                    records = prepare_record(path, source, role == 'train', SEED+i)
                    joblib.dump(records, cache)
                    print(f'D FEATURES {path.stem} {time.monotonic()-started:.1f}s', flush=True)


def records_for(out, sources):
    return [r for s in sources for path in dead_transfer.SPLITS[s]['train']
            for r in joblib.load(out/f'{path.stem}_d_features.joblib')]


def ordered_features(x, names, period):
    """Keep direction of motion change, without consulting any labels."""
    raw = np.array([i for i, name in enumerate(names) if '_mean_' not in name])
    parts = [x]
    for seconds in (1, 3, 5):
        lag = seconds*RATE
        before, after = np.full_like(x[:, raw], np.nan), np.full_like(x[:, raw], np.nan)
        for a, b in supervised.blocks(period):
            if b-a > lag:
                before[a+lag:b] = x[a:b-lag, raw]
                after[a:b-lag] = x[a+lag:b, raw]
        parts.extend((before, after, after-before))
    return np.column_stack(parts)


def matrix(records, ordered=False):
    xs, ys, weights = [], [], []
    for r in records:
        keep = r['y'] >= 0
        x = ordered_features(r['x'], r['names'], r['period']) if ordered else r['x']
        xs.append(x[keep])
        ys.append(r['y'][keep])
        # Every source has unit mass, then every match/augmentation equal mass.
        count = sum(v['source'] == r['source'] for v in records)
        weights.append(np.full(keep.sum(), 1/(count*keep.sum())))
    w = np.concatenate(weights)
    return np.concatenate(xs), np.concatenate(ys), w/w.mean()


def make_network(kind, n_features):
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self):
            super().__init__()
            if kind == 'bigru':
                self.sequence = nn.GRU(n_features, 24, batch_first=True, bidirectional=True)
                self.output = nn.Linear(48, 1)
            elif kind == 'tcn':
                self.input = nn.Conv1d(n_features, 24, 1)
                self.layers = nn.ModuleList([nn.Conv1d(24, 24, 5, padding=2*d, dilation=d)
                                             for d in (1, 2, 4, 8)])
                self.output = nn.Conv1d(24, 1, 1)
            else:
                raise ValueError(kind)

        def forward(self, x):
            if kind == 'bigru':
                x, _ = self.sequence(x)
                return self.output(x).squeeze(-1)
            x = torch.relu(self.input(x.transpose(1, 2)))
            for layer in self.layers:
                x = x + torch.relu(layer(x))
            return self.output(x).squeeze(1)
    return Network()


def normalization(records):
    """Fit on training inputs only; never turn unobserved values into zero motion."""
    x = np.concatenate([r['x'][r['y'] >= 0] for r in records])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median = np.nanmedian(x, axis=0)
        scale = np.nanstd(x, axis=0)
    return np.nan_to_num(median), np.where(np.isfinite(scale) & (scale > .01), scale, 1).astype(np.float32)


def normalize(x, median, scale):
    mask = np.isfinite(x)
    value = np.where(mask, (x-median)/scale, 0)
    return np.column_stack((np.clip(value, -10, 10), mask)).astype(np.float32)


def chunks(record):
    """Indices for 10 s cores with 5 s context, clipped inside the same block."""
    for a, b in supervised.blocks(record['period']):
        for start in range(a, b, CORE):
            full = np.clip(np.arange(start-CONTEXT, start+CORE+CONTEXT), a, b-1)
            valid_core = min(CORE, b-start)
            yield start, valid_core, full


def fit_neural(kind, records):
    import torch
    from torch.utils.data import DataLoader, Dataset
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    median, scale = normalization(records)
    normalized = [normalize(r['x'], median, scale) for r in records]
    samples = [(i, start, size, full) for i, r in enumerate(records) for start, size, full in chunks(r)]
    masses = []
    for r in records:
        count = sum(v['source'] == r['source'] for v in records)
        masses.append(1/(count*max(1, (r['y'] >= 0).sum())))
    masses = np.asarray(masses)/np.mean(masses)

    class Windows(Dataset):
        def __len__(self):
            return len(samples)

        def __getitem__(self, item):
            i, start, size, full = samples[item]
            target = np.zeros(CORE, dtype=np.float32)
            weight = np.zeros(CORE, dtype=np.float32)
            y = records[i]['y'][start:start+size]
            target[:size] = np.maximum(y, 0)
            weight[:size] = (y >= 0)*masses[i]
            return normalized[i][full], target, weight

    loader = DataLoader(Windows(), batch_size=64, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(SEED))
    model = make_network(kind, normalized[0].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history = []
    for epoch in range(EPOCHS):
        model.train()
        numerator = denominator = 0.
        for x, y, weight in loader:
            if weight.sum() == 0:
                continue
            optimizer.zero_grad()
            logits = model(x)[:, CONTEXT:CONTEXT+CORE]
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
            loss = (losses*weight).sum()/weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            numerator += float((losses.detach()*weight).sum())
            denominator += float(weight.sum())
        history.append(numerator/denominator)
        print(f'NEURAL {kind} epoch={epoch+1} training_bce={history[-1]:.5f}', flush=True)
    return dict(kind=kind, state={k: v.detach().numpy() for k, v in model.state_dict().items()},
                median=median, scale=scale, history=history, input_features=normalized[0].shape[1])


def fit(kind, records):
    if kind in ('tcn', 'bigru'):
        return fit_neural(kind, records)
    x, y, weights = matrix(records, kind == 'ordered_hgb')
    if kind in ('hgb', 'ordered_hgb'):
        estimator = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31,
                        learning_rate=.08, min_samples_leaf=40, l2_regularization=1,
                        early_stopping=False, random_state=31)
    else:
        from lightgbm import LGBMClassifier
        slow = kind == 'lgbm_002'
        estimator = LGBMClassifier(objective='binary', n_estimators=640 if slow else 160,
                        learning_rate=.02 if slow else .08, num_leaves=31, min_child_samples=40,
                        reg_lambda=1, n_jobs=1, random_state=31, verbosity=-1)
    with threadpool_limits(limits=1):
        estimator.fit(x, y, sample_weight=weights)
    return dict(kind=kind, estimator=estimator)


def predict_sampled(record, model):
    kind = model['kind']
    if kind in ('tcn', 'bigru'):
        import torch
        torch.set_num_threads(1)
        net = make_network(kind, model['input_features'])
        net.load_state_dict({k: torch.as_tensor(v) for k, v in model['state'].items()})
        net.eval()
        x = normalize(record['x'], model['median'], model['scale'])
        windows = list(chunks(record))
        out = np.empty(len(x), dtype=np.float32)
        with torch.inference_mode():
            for offset in range(0, len(windows), 64):
                batch = windows[offset:offset+64]
                inp = torch.as_tensor(np.stack([x[full] for _, _, full in batch]))
                prob = torch.sigmoid(net(inp)).numpy()
                for i, (start, size, _) in enumerate(batch):
                    out[start:start+size] = prob[i, CONTEXT:CONTEXT+size]
        return out
    x = (ordered_features(record['x'], record['names'], record['period'])
         if kind == 'ordered_hgb' else record['x'])
    with threadpool_limits(limits=1):
        return model['estimator'].predict_proba(x)[:, 1]


def upsample(probability, record, native_period):
    """Interpolate probabilities in each continuous input block only."""
    out = np.empty(record['native_n'], dtype=float)
    for a, b in supervised.blocks(native_period):
        rows = (record['indices'] >= a) & (record['indices'] < b)
        out[a:b] = np.interp(np.arange(a, b), record['indices'][rows], probability[rows])
    return out


def summarize(inputs, labels, base, dead):
    probabilities = refine_domain.combine(base, dead, 0.)
    predicted = supervised.decode(probabilities, supervised.model_periods(inputs), float(inputs['fps']), COST)
    result = supervised.score(dict(inputs, labels=labels), predicted)
    keep = np.isin(labels, supervised.LABELS)
    truth = labels[keep] == 'D'
    raw = dead[keep] >= .5
    precision, recall, f1, _ = precision_recall_fscore_support(truth, raw, average='binary', zero_division=0)
    row = dict(macro_f1=result['macro_f1'], spell_f1=result['spell_f1_mean'],
               d_f1=result['per_class']['D']['f1'], d_precision=result['per_class']['D']['precision'],
               d_recall=result['per_class']['D']['recall'], raw_d_precision=float(precision),
               raw_d_recall=float(recall), raw_d_f1=float(f1),
               d_auroc=float(roc_auc_score(truth, dead[keep])) if len(np.unique(truth)) == 2 else None,
               n_gt_spells=result['n_gt_spells'], n_pred_spells=result['n_pred_spells'],
               graded_frames=int(keep.sum()))
    from experiments.eval_set.metrics import d_boundary_metrics
    row['d_boundaries'] = d_boundary_metrics(labels, predicted, period=inputs['period'],
                    fps=float(inputs['fps']), timestamps=inputs.get('timestamps'))
    return row, predicted


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    freeze_spec(out)
    if args.prepare:
        prepare(out)
        return
    sources = tuple(s for s in dead_transfer.SPLITS if args.fold != f'without_{s}')
    records = records_for(out, sources)
    for kind in args.models:
        model_path = out/f'{args.fold}_{kind}.joblib'
        marker = out/f'{args.fold}_{kind}_frozen.json'
        if not marker.exists():
            if model_path.exists():
                raise ValueError('Unfrozen model found; inspect interrupted run')
            start = time.monotonic()
            model = fit(kind, records)
            joblib.dump(model, model_path)
            marker.write_text(json.dumps(dict(model_sha256=digest(model_path), spec_sha256=digest(out/'predeclared.json'),
                      created_utc=datetime.now(timezone.utc).isoformat(), fit_seconds=time.monotonic()-start,
                      sources=sources, training=[r['match'] for r in records]), indent=2)+'\n')
            print(f'FROZEN {args.fold} {kind}', flush=True)
    del records
    if args.fit_only:
        return
    result_path = out/f'{args.fold}_results.json'
    rows = json.loads(result_path.read_text()) if result_path.exists() else []
    for source, split in dead_transfer.SPLITS.items():
        if args.fold != 'pooled' and source in sources:
            continue
        for path in split['eval']:
            inputs, labels = multisource_eval.view(path)
            record = joblib.load(out/f'{path.stem}_d_features.joblib')[0]
            base = np.load(out/f'{path.stem}_attribution.npz')['probabilities']
            for kind in args.models:
                if any(r['model'] == kind and r['match'] == path.stem for r in rows):
                    continue
                model_path = out/f'{args.fold}_{kind}.joblib'
                marker = json.loads((out/f'{args.fold}_{kind}_frozen.json').read_text())
                if marker['model_sha256'] != digest(model_path) or marker['spec_sha256'] != digest(out/'predeclared.json'):
                    raise ValueError('Frozen model/spec hash mismatch')
                model = joblib.load(model_path)
                p = upsample(predict_sampled(record, model), record, supervised.model_periods(inputs))
                summary, pred = summarize(inputs, labels, base, p)
                row = dict(fold=args.fold, model=kind, source=source, match=path.stem, **summary)
                rows.append(row)
                np.savez_compressed(out/f'{args.fold}_{kind}_{path.stem}_prediction.npz', dead=p, prediction=pred)
                result_path.write_text(json.dumps(rows, indent=2)+'\n')
                print(json.dumps({k: v for k, v in row.items() if k != 'd_boundaries'}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'data/experiments/learned_temporal')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--fit-only', action='store_true')
    parser.add_argument('--fold', choices=FOLDS, default='pooled')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
