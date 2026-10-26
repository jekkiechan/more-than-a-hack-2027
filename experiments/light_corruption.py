"""A fixed 25% corruption replacement, separated by team and D tasks.

The original training recipe supplies the other 75%. Team rows, team-swap
pairs, GRU windows, epochs and optimizer update counts stay fixed. Selection
uses historical external development matches only; no public scoring command.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from experiments import (corruption_experiment as ce, current_baseline,
                         learned_temporal as lt, multisource_eval,
                         refine_features, robust, supervised,
                         tracking_corruption as tc)
from experiments.eval_set.protocol import paired_source_bootstrap, sha256

ROOT = lt.ROOT
DEFAULT = ROOT/'data/experiments/light_corruption_v1'
FRACTION = .25
ARMS = {'baseline': ('baseline', 'baseline'),
        'ball_team25': ('light', 'baseline'),
        'player_D25': ('baseline', 'light'), 'paired25': ('light', 'light')}


def chosen(n, seed, fraction=FRACTION):
    """Fixed-size selection independent of labels and feature values."""
    result = np.zeros(n, dtype=bool)
    result[np.random.default_rng(seed).choice(n, round(n*fraction), replace=False)] = True
    return result


def ball_gap_view(inputs, calibration, seed):
    """Keep coordinates/player tracking native; add only the frozen ball mask."""
    original = tc.inputs_only(inputs)
    rich = tc.corrupt(original, calibration, seed)
    original['ball'][~rich['ball_observed']] = np.nan
    original['ball_observed'] = np.isfinite(original['ball']).all(axis=1)
    return original


def mix_team(clean, legacy, gap, match):
    """Replace exactly a quarter of each original view, preserving swap pairs."""
    n = len(clean)//2
    if len(clean) != 2*n or legacy.shape != clean.shape or gap.shape != clean.shape:
        raise ValueError('Expected equal direct/reversed feature views')
    result = []
    for variant, x in enumerate((clean, legacy)):
        select = chosen(n, ce.seed(match, f'light_team_{variant}'))
        select = np.r_[select, select]
        result.append(np.where(select[:, None], gap, x))
    return np.concatenate(result)


def source_hash():
    return dict(ce.sources_hash(), **{'experiments/light_corruption.py': sha256(__file__)})


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'predeclared.json').exists():
        check(out)
        return
    previous = ce.check(ce.DEFAULT)
    cached = [ce.DEFAULT/'predeclared.json']
    cached += sorted(ce.DEFAULT.glob('train_*.joblib'))
    cached += sorted(ce.DEFAULT.glob('eval_*.joblib'))
    cached += sorted(ce.DEFAULT.glob('pending_*.joblib'))
    cached += sorted(ce.DEFAULT.glob('*_results.json'))
    cached += [p for p in ce.DEFAULT.glob('*.joblib') if p.with_suffix('.json').exists()]
    if len(list(ce.DEFAULT.glob('train_*.joblib'))) != 13:
        raise ValueError('Previous 13-match training cache required')
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_hash(), fraction=FRACTION, arms=ARMS, views=ce.VIEWS,
        folds=lt.FOLDS, calibration=previous['calibration'],
        cached_inputs_sha256={str(p.relative_to(ROOT)): sha256(p) for p in cached},
        training='75% original clean/coarse augmentation recipe plus 25% targeted replacement',
        team='Long ball gaps/sampling mask only, native coordinates and players; old Markov augmentation retained',
        D='Player/coordinate portion of frozen composite corruption; old clean/slot-dropout windows retained',
        replacement='Quarter of each original team view/direct+swap pairs; quarter of each original D record windows',
        budget='Unchanged HGB rows/capacity; unchanged GRU window count, batch size, eight epochs and optimizer updates',
        neural_context='Each input window entirely from one observation variant; no feature splicing inside a window',
        normalization='Training-only core rows of the selected mixture; same finite-value masks',
        evaluation='Same clean and two composite-stress views/graded frames as first batch; 90 new plus 30 reused baseline rows',
        primary='Equal-provider clean spell F1 in D-provider holdouts; exact-tie macro F1; baseline remains eligible',
        stress='Secondary diagnostic; two seeds averaged within each match; no combined clean/stress selection score',
        scope='D provider holdouts only; team models pooled; reused development, no fresh confirmation',
        organizer='No reference labels or new public score; report archived baseline competition metrics separately',
        historical_public_scores_known=True, automatic_promotion=False)
    ce.write_json(out/'predeclared.json', spec, exclusive=True)
    print('FROZEN', out, flush=True)


def check(out):
    spec = json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256'] != source_hash():
        raise ValueError('Frozen source changed; use a new experiment directory')
    return spec


def prepare(out, source):
    spec = check(out)
    for i, path in enumerate(ce.dead_transfer.SPLITS[source]['train']):
        target = out/f'team_{path.stem}.joblib'
        if target.exists():
            continue
        started = time.monotonic()
        inputs, labels = multisource_eval.view(path)
        old = joblib.load(ce.DEFAULT/f'train_{path.stem}.joblib')
        n = len(old['team'][1])//2
        clean, y, names = old['team'][0][:n], old['team'][1], old['team'][2]
        legacy, legacy_y, legacy_names = ce.training_team(robust.masked_inputs(inputs, lt.SEED+i), labels)
        gap, gap_y, gap_names = ce.training_team(
            ball_gap_view(inputs, spec['calibration'], ce.seed(path.stem, 'train')), labels)
        if names != legacy_names or names != gap_names:
            raise ValueError('Feature schema differs')
        if not np.array_equal(y, np.r_[legacy_y, gap_y]):
            raise ValueError('Training labels/rows changed')
        x = mix_team(clean, legacy, gap, path.stem)
        joblib.dump((x, y, names), target)
        print('PREPARED TEAM', path.stem, round(time.monotonic()-started, 1), flush=True)


def fit_team(out):
    check(out)
    target = out/'team_light.joblib'
    if target.exists():
        ce.load_model(out, target)
        return
    cache = {}
    for source, split in ce.dead_transfer.SPLITS.items():
        records = [joblib.load(out/f'team_{p.stem}.joblib') for p in split['train']]
        cache[source] = (np.concatenate([r[0] for r in records]),
                         np.concatenate([r[1] for r in records]), records[0][2])
    started = time.monotonic()
    model = multisource_eval.fit(tuple(cache), cache)
    ce.freeze_model(out, target, model, dict(seconds=time.monotonic()-started,
                    training='Pooled; quarter of each legacy view replaced by ball-gap-only inputs'))
    print('FROZEN TEAM', round(time.monotonic()-started, 1), flush=True)


def d_records(fold):
    records = []
    for source, split in ce.dead_transfer.SPLITS.items():
        if fold == f'without_{source}':
            continue
        for path in split['train']:
            cached = joblib.load(ce.DEFAULT/f'train_{path.stem}.joblib')
            physical = cached['augmented'][1]
            for i, original in enumerate(cached['baseline']):
                if (not np.array_equal(original['y'], physical['y'])
                        or not np.array_equal(original['period'], physical['period'])):
                    raise ValueError('Corruption changed original labels or blocks')
                windows = list(lt.chunks(original))
                select = chosen(len(windows), ce.seed(path.stem, f'light_D_{i}'))
                mask = np.zeros(len(original['x']), bool)
                for use, (a, size, _) in zip(select, windows):
                    mask[a:a+size] = use
                records.append(dict(original, physical_x=physical['x'],
                                    replace_windows=select, replace_rows=mask))
    return records


def fit_neural(records):
    import torch
    from torch.utils.data import DataLoader, Dataset
    torch.set_num_threads(1)
    torch.manual_seed(lt.SEED)
    norm = [dict(r, x=np.where(r['replace_rows'][:, None], r['physical_x'], r['x'])) for r in records]
    median, scale = lt.normalization(norm)
    del norm
    old = [lt.normalize(r['x'], median, scale) for r in records]
    rich = [lt.normalize(r['physical_x'], median, scale) for r in records]
    samples = [(i, a, size, full, bool(use)) for i, r in enumerate(records)
               for use, (a, size, full) in zip(r['replace_windows'], lt.chunks(r))]
    masses = np.array([1/(sum(v['source'] == r['source'] for v in records)
                         *max(1, (r['y'] >= 0).sum())) for r in records])
    masses /= masses.mean()

    class Windows(Dataset):
        def __len__(self): return len(samples)

        def __getitem__(self, item):
            i, start, size, full, use = samples[item]
            target, weight = np.zeros(lt.CORE, np.float32), np.zeros(lt.CORE, np.float32)
            y = records[i]['y'][start:start+size]
            target[:size] = np.maximum(y, 0)
            weight[:size] = (y >= 0)*masses[i]
            return (rich if use else old)[i][full], target, weight

    loader = DataLoader(Windows(), batch_size=64, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(lt.SEED))
    model = lt.make_network('bigru', old[0].shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history, updates = [], 0
    for epoch in range(lt.EPOCHS):
        model.train()
        numerator = denominator = 0.
        for x, y, weight in loader:
            if not weight.sum(): continue
            optimizer.zero_grad()
            logits = model(x)[:, lt.CONTEXT:lt.CONTEXT+lt.CORE]
            losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')
            loss = (losses*weight).sum()/weight.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            updates += 1
            numerator += float((losses.detach()*weight).sum())
            denominator += float(weight.sum())
        history.append(numerator/denominator)
        print('NEURAL', epoch+1, round(history[-1], 5), flush=True)
    return dict(kind='bigru', state={k: v.detach().numpy() for k, v in model.state_dict().items()},
                median=median, scale=scale, history=history, input_features=old[0].shape[1],
                windows_per_epoch=len(samples), optimizer_updates=updates)


def fit_d(out, fold):
    check(out)
    records = d_records(fold)
    for kind in ('ordered_hgb', 'bigru'):
        target = out/f'{fold}_{kind}.joblib'
        if target.exists():
            ce.load_model(out, target)
            continue
        started = time.monotonic()
        if kind == 'bigru':
            model = fit_neural(records)
        else:
            mixed = []
            for r in records:
                old = lt.ordered_features(r['x'], r['names'], r['period'])
                rich = lt.ordered_features(r['physical_x'], r['names'], r['period'])
                mixed.append(dict(r, x=np.where(r['replace_rows'][:, None], rich, old)))
            x, y, weight = lt.matrix(mixed, ordered=False)
            estimator = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31,
                learning_rate=.08, min_samples_leaf=40, l2_regularization=1,
                early_stopping=False, random_state=31)
            with threadpool_limits(limits=1): estimator.fit(x, y, sample_weight=weight)
            model = dict(kind='ordered_hgb', estimator=estimator)
            del mixed, x
        ce.freeze_model(out, target, model, dict(fold=fold, seconds=time.monotonic()-started,
            training_matches=sorted({r['match'] for r in records}),
            original_windows=sum(len(r['replace_windows']) for r in records),
            replaced_windows=sum(int(r['replace_windows'].sum()) for r in records)))
        print('FROZEN D', fold, kind, round(time.monotonic()-started, 1), flush=True)


def team_probabilities(out, source):
    check(out)
    model = ce.load_model(out, out/'team_light.joblib')
    for path in ce.dead_transfer.SPLITS[source]['eval']:
        for view in ce.VIEWS:
            target = out/f'team_{path.stem}_{view}.npz'
            if target.exists(): continue
            cached = joblib.load(ce.DEFAULT/f'pending_{path.stem}_{view}.joblib')
            p = refine_features.probabilities_from_features(cached['matrix'], cached['names'], model)
            np.savez_compressed(target, probability=p)
            print('TEAM PROBABILITIES', path.stem, view, flush=True)


def evaluate(out, fold):
    check(out)
    baseline = current_baseline.load_models()
    original_d = {kind: baseline[kind] if fold == 'pooled' else
                  ce.load_model(ce.DEFAULT, ce.DEFAULT/f'{fold}_baseline_{kind}.joblib')
                  for kind in ('bigru', 'ordered_hgb')}
    light_d = {kind: ce.load_model(out, out/f'{fold}_{kind}.joblib') for kind in original_d}
    result = out/f'{fold}_results.json'
    rows = json.loads(result.read_text()) if result.exists() else [dict(r, evaluation_reused=True)
           for r in json.loads((ce.DEFAULT/f'{fold}_results.json').read_text()) if r['model'] == 'baseline']
    for source, split in ce.dead_transfer.SPLITS.items():
        if fold != 'pooled' and fold != f'without_{source}': continue
        for path in split['eval']:
            inputs, labels = multisource_eval.view(path)
            periods = supervised.model_periods(inputs)
            for view in ce.VIEWS:
                cached = joblib.load(ce.DEFAULT/f'eval_{path.stem}_{view}.joblib')
                teams = {'baseline': cached['baseline'],
                         'light': np.load(out/f'team_{path.stem}_{view}.npz')['probability']}
                dead = {name: .5*sum(lt.upsample(lt.predict_sampled(cached['record'], model),
                           cached['record'], periods) for model in members.values())
                        for name, members in (('baseline', original_d), ('light', light_d))}
                for arm, (team, d) in ARMS.items():
                    if any(r['match'] == path.stem and r['view'] == view and r['model'] == arm for r in rows): continue
                    summary, prediction = lt.summarize(inputs, labels, teams[team], dead[d])
                    alive = np.isin(labels, ['H', 'A'])
                    team_pred = supervised.LABELS[np.argmax(teams[team][:, :2], axis=1)]
                    keep = np.isin(labels, supervised.LABELS)
                    row = dict(fold=fold, source=source, match=path.stem, view=view, model=arm,
                        visibility=cached['visibility'], evaluation_reused=False,
                        conditional_team_accuracy=float(np.mean(team_pred[alive] == labels[alive])),
                        raw_d_brier=float(np.mean((dead[d][keep]-(labels[keep] == 'D'))**2)), **summary)
                    rows.append(row)
                    np.savez_compressed(out/f'{fold}_{arm}_{path.stem}_{view}.npz', prediction=prediction)
                    ce.write_json(result, rows)
                    print('SCORED', fold, path.stem, view, arm, round(row['spell_f1'], 5), round(row['macro_f1'], 5), flush=True)


def report(out):
    spec = check(out)
    rows = [r for fold in lt.FOLDS for r in json.loads((out/f'{fold}_results.json').read_text())]
    summaries, deltas = [], []
    for holdout in (False, True):
        for view in ('clean', 'corrupted'):
            old = ce.match_means(rows, 'baseline', view, holdout)
            for arm in ARMS:
                matches = ce.match_means(rows, arm, view, holdout)
                per = {s: {k: float(np.mean([r[k] for r in matches if r['eval_source'] == s]))
                           for k in ce.METRICS} for s in ce.dead_transfer.SPLITS}
                item = dict(evaluation='provider_holdout_D' if holdout else 'pooled', view=view,
                    model=arm, per_source=per,
                    **{k: float(np.mean([v[k] for v in per.values()])) for k in ce.METRICS})
                item['weighted_70_30'] = .7*item['spell_f1']+.3*item['macro_f1']
                summaries.append(item)
                if arm != 'baseline': deltas.append(dict(evaluation=item['evaluation'], view=view,
                            model=arm, **paired_source_bootstrap(old, matches)))
    eligible = [r for r in summaries if r['evaluation'] == 'provider_holdout_D' and r['view'] == 'clean']
    selected = max(eligible, key=lambda r: (r['spell_f1'], r['macro_f1']))
    ce.write_json(out/'selection.json', dict(rule=spec['primary'], selected=selected,
        pooled_checkpoints_sha256={p.name: sha256(p) for p in out.glob('*.joblib')
                                   if p.with_suffix('.json').exists()}), exclusive=True)
    # Selection is frozen before reading any archived competition diagnostics.
    archived = json.loads((ROOT/'experiments/results/current_baseline_public_check.json').read_text())['scores']['gru_hgb']
    competition = dict(model='Adopted GRU/HGB baseline', spell_f1=archived['spell_f1_mean'],
        macro_f1=archived['macro_f1'], dead_ball_f1=archived['per_class']['D']['f1'],
        graded_frames=archived['n_graded'], reference_possessions=archived['n_gt_spells'],
        new_candidate_scored=False, full_match_score=None,
        source='experiments/results/current_baseline_public_check.json',
        boundary='AGENTS.md: Never use repeated public-window checks as an iteration loop')
    destination = ROOT/'experiments/results/light_corruption_development.json'
    ce.write_json(destination, dict(spec=spec, summaries=summaries, paired_deltas=deltas, rows=rows,
        externally_selected=selected, recorded_competition_check=competition,
        baseline_changed=False, organizer_check_performed=False), exclusive=True)
    for r in summaries: print('SUMMARY', r['evaluation'], r['view'], r['model'], round(r['spell_f1'], 5), round(r['macro_f1'], 5))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('freeze', 'prepare', 'fit-team', 'fit-d', 'team-probs', 'evaluate', 'report'))
    p.add_argument('--out', type=Path, default=DEFAULT)
    p.add_argument('--source', choices=tuple(ce.dead_transfer.SPLITS))
    p.add_argument('--fold', choices=lt.FOLDS, default='pooled')
    args = p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if args.action == 'freeze': freeze(args.out)
        elif args.action == 'prepare':
            for source in ([args.source] if args.source else ce.dead_transfer.SPLITS): prepare(args.out, source)
        elif args.action == 'fit-team': fit_team(args.out)
        elif args.action == 'fit-d': fit_d(args.out, args.fold)
        elif args.action == 'team-probs':
            for source in ([args.source] if args.source else ce.dead_transfer.SPLITS): team_probabilities(args.out, source)
        elif args.action == 'evaluate': evaluate(args.out, args.fold)
        elif args.action == 'report': report(args.out)


if __name__ == '__main__': main()
