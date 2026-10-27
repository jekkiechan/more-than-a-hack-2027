"""Fixed full-provider-holdout experiment: live gaps and one H/A/D GRU.

No organizer scoring, training, or baseline-promotion entrypoint. Data and
frame-level predictions remain in ignored data/. Aggregate results are public.
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
from experiments import (corruption_experiment as ce, dead_transfer as dt,
                         error_analysis, had_gru, learned_temporal as lt,
                         light_corruption as lc, multisource_eval as ms,
                         refine_domain, refine_features, robust, supervised,
                         tracking_corruption as tc)
from experiments.eval_set.protocol import sha256

ROOT = lt.ROOT
DEFAULT = ROOT/'data/experiments/live_gap_had_v1'
ARMS = ('baseline', 'live_gap_pipeline', 'had_gru', 'live_gap_had_gru')
VIEWS = ('clean', 'corrupt_1', 'corrupt_2')


def training_sources(heldout):
    if heldout not in dt.SPLITS:
        raise ValueError('Expected an evaluation provider')
    return [s for s in dt.SPLITS if s != heldout]


def live_gap_view(inputs, labels, calibration, seed):
    """Training-only conditioning on external live intervals; targets unchanged.

    Physical corruption retains the old assumed error ranges. A renewal mask is
    generated on the concatenated live-time clock within each tracking block,
    then intersected with its real live runs. Thus no D/X target is converted,
    no existing gap is filled, and D/X ball observations are left native.
    """
    original = tc.inputs_only(inputs)
    out = tc.corrupt(original, calibration, seed)
    rng = np.random.default_rng(seed+1)
    fps = float(inputs['fps'])
    missing = float(calibration['ball_missing_fraction'])
    edges = np.asarray(calibration['gap_edges_seconds'])
    # Remove the old state-agnostic ball gaps. Player corruption is retained;
    # native ball XY gets independent smooth localization error (.2 m SD).
    ball = original['ball'].copy()
    for a, b in supervised.blocks(supervised.model_periods(inputs)):
        if inputs['period'][a] <= 0:
            continue
        live = np.flatnonzero(np.isin(labels[a:b], ['H', 'A']))+a
        if not len(live):
            continue
        n = b-a
        ball[a:b] += tc._smooth(n, fps, rng, 1., .2, 2)
        at = 0
        while at < len(live):
            k = rng.choice(len(edges)-1, p=calibration['gap_bin_probability'])
            duration = rng.uniform(edges[k], edges[k+1])
            at += max(1, round(rng.exponential(duration*(1-missing)/missing)*fps))
            stop = min(len(live), at+max(1, round(duration*fps)))
            ball[live[at:stop]] = np.nan
            at = stop
    active = np.isin(labels, ['H', 'A'])
    ball[~active] = original['ball'][~active]
    out['ball'] = ball
    out['ball_observed'] = np.isfinite(ball).all(axis=1)
    return out


def fingerprint():
    paths = [Path(__file__), Path(had_gru.__file__), Path(lc.__file__), Path(error_analysis.__file__)]
    return dict(ce.sources_hash(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths})


def archive_paths(source):
    fold = f'without_{source}'
    return {
        'team': ROOT/f'data/experiments/archived_full_holdout/{fold}_full_holdout_equal_sources_team.joblib',
        **{kind: ROOT/f'data/experiments/archived_d_holdout/{fold}_baseline_{kind}.joblib'
           for kind in ('bigru', 'ordered_hgb')}}


def freeze(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'predeclared.json').exists():
        return check(out)
    manifest = json.loads((ROOT/'experiments/eval_set/manifest.json').read_text())
    data_hashes = {}
    for entry in manifest['sources'].values():
        for role in ('train', 'selection'):
            for item in entry[role]:
                actual = sha256(ROOT/item['path'])
                if actual != item['sha256']:
                    raise ValueError(f"Data fingerprint differs: {item['path']}")
                data_hashes[item['path']] = actual
    archived = {}
    for source in dt.SPLITS:
        for kind, path in archive_paths(source).items():
            receipt = json.loads(path.with_suffix('.json').read_text())
            if sha256(path) != receipt['sha256']:
                raise ValueError(f'Archive checksum differs: {path}')
            expected_matches = {p.stem for s in training_sources(source) for p in dt.SPLITS[s]['train']}
            if set(receipt['training_matches']) != expected_matches:
                raise ValueError(f'Archive provider isolation differs: {path}')
            archived[f'{source}/{kind}'] = dict(path=str(path.relative_to(ROOT)),
                sha256=sha256(path), receipt_sha256=sha256(path.with_suffix('.json')),
                receipt=receipt)
    calibration_path = ROOT/'experiments/results/realistic_corruption_development.json'
    calibration = json.loads(calibration_path.read_text())['spec']['calibration']
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=fingerprint(),
        data_sha256=data_hashes, archived_baseline=archived, arms=ARMS, evaluation_views=VIEWS,
        calibration=calibration, calibration_report_sha256=sha256(calibration_path),
        training_sources={s: training_sources(s) for s in dt.SPLITS},
        training_matches={s: [p.stem for t in training_sources(s) for p in dt.SPLITS[t]['train']] for s in dt.SPLITS},
        evaluation_matches={s: [p.stem for p in dt.SPLITS[s]['eval']] for s in dt.SPLITS},
        primary='Equal-provider clean official spell F1; exact-tie macro F1; +0.01 over matched baseline screen',
        secondary='Two fixed label-free composite stress seeds averaged within match, then provider',
        replacement_fraction=.25, decoder_switch_cost=lt.COST,
        live_gap='External training labels alone locate H/A intervals; empirical aggregate gap histogram on live-time clock, clipped at dead/unknown intervals; physical error assumptions unchanged',
        pipeline='Same 75% clean/coarse team rows and D windows; replace 25% with live-gap/fragmentation/uncertainty view; capacities unchanged',
        gru='One bidirectional GRU hidden=24, 3 logits; 5 Hz, 10 s cores + 5 s context each side; 8 epochs, AdamW .001/.0001, batch 64, seed 2026; direct+swapped records',
        gru_inputs='Union of existing refine_features team geometry/contact/history and player-only D motion features; finite masks and training-only normalization; identical raw evaluation views',
        gru_training='Two original clean/coarse-ball views; quarter of whole windows replaced by baseline ball-gap or new live-gap view. Both have same source/match weights and update budget; joint loss replaces separate H/A and D losses, not a head-only ablation',
        controls='No short-spell weighting, decoder tuning, provider calibration, optimizer sweep or source weighting sweep',
        labels='Existing external converter targets preserved; X and native visibility grading fixed before corruption; no inferred labels from nearest-player geometry',
        limitations=['Five reused development matches, no independent confirmation',
                     'Training live conditioning does not measure competition live-gap distribution',
                     'Synthetic tracking errors are assumptions, not transfer proof',
                     'SkillCorner censored D/boundaries remain unassessed'],
        organizer_labels_read=False, fresh_confirmation=False, automatic_promotion=False)
    ce.write_json(out/'predeclared.json', spec, exclusive=True)
    return spec


def check(out):
    spec = json.loads((out/'predeclared.json').read_text())
    if spec['source_sha256'] != fingerprint():
        raise ValueError('Frozen sources changed; use a fresh output directory')
    return spec


def feature_view(inputs, labels, source, match):
    start = time.monotonic()
    team, names = refine_features.features(inputs)
    player, player_names = lt.player_features(inputs)
    indices = lt.sampled_indices(inputs)
    stride = max(1, round(float(inputs['fps'])/ms.TRAIN_HZ))
    rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::stride]
    y = np.full(len(labels), -1, np.int8)
    for i, label in enumerate(supervised.LABELS):
        y[labels == label] = i
    common = dict(source=source, match=match, indices=indices, native_n=len(labels),
                  fps=float(inputs['fps']), period=supervised.model_periods(inputs)[indices])
    return dict(team=(team[rows], y[rows], names),
        joint=dict(common, x=np.column_stack((team[indices], player[indices])),
                   y=y[indices], names=names+['D_'+n for n in player_names]),
        dead=dict(common, x=player[indices], y=np.where(y[indices] < 0, -1, y[indices] == 2).astype(np.int8), names=player_names),
        feature_seconds=time.monotonic()-start)


def prepare(out, source):
    spec = check(out)
    for i, path in enumerate(dt.SPLITS[source]['train']):
        target = out/f'train_{path.stem}.joblib'
        if target.exists():
            continue
        start = time.monotonic()
        inputs, labels = ms.view(path)
        views = {'clean': inputs, 'legacy': robust.masked_inputs(inputs, lt.SEED+i),
                 'gap': lc.ball_gap_view(inputs, spec['calibration'], ce.seed(path.stem, 'train')),
                 'live': live_gap_view(inputs, labels, spec['calibration'], ce.seed(path.stem, 'live_train'))}
        records = {k: feature_view(v, labels, source, path.stem) for k, v in views.items()}
        records['dropout'] = ce.record(refine_domain.drop_player_slots(inputs, lt.SEED+i), labels, source, path.stem, 1)
        records['audit'] = {k: visibility_by_state(v, labels) for k, v in views.items()}
        joblib.dump(records, target)
        print('PREPARED', path.stem, round(time.monotonic()-start, 1), flush=True)


def visibility_by_state(inputs, labels):
    result = {}
    observed = np.isfinite(inputs['ball']).all(axis=1)
    for name, mask in [('live', np.isin(labels, ['H', 'A'])), ('dead', labels == 'D')]:
        result[name] = dict(frames=int(mask.sum()), missing_fraction=float((~observed[mask]).mean()) if mask.any() else None)
    return result


def mixed_record(original, alternate, match, variant, swap=False):
    result = dict(original)
    windows = list(lt.chunks(result))
    chosen = lc.chosen(len(windows), ce.seed(match, f'live_window_{variant}'))
    rows = np.zeros(len(result['y']), bool)
    for use, (a, size, _) in zip(chosen, windows):
        rows[a:a+size] = use
    result.update(alternate=alternate['x'], physical_x=alternate['x'],
                  replace_windows=chosen, replace_rows=rows)
    if swap:
        result['x'] = supervised.swap_team_features(result['x'], result['names'])
        result['alternate'] = supervised.swap_team_features(result['alternate'], result['names'])
        result['y'] = np.where(np.isin(result['y'], [0, 1]), 1-result['y'], result['y'])
    return result


def fit(out, source, arm):
    check(out)
    sources = training_sources(source)
    kinds = ('team', 'ordered_hgb', 'bigru') if arm == 'live_gap_pipeline' else ('joint',)
    if arm not in ARMS[1:]:
        raise ValueError('Baseline is an archived fixed comparator')
    for kind in kinds:
        target = out/f'without_{source}_{arm}_{kind}.joblib'
        if target.exists():
            ce.load_model(out, target)
            continue
        start = time.monotonic()
        records, cache = [], {}
        for provider in sources:
            provider_team = []
            for path in dt.SPLITS[provider]['train']:
                r = joblib.load(out/f'train_{path.stem}.joblib')
                if kind == 'team':
                    xs, ys = [], []
                    for v, name in enumerate(('clean', 'legacy')):
                        x, y, names = r[name]['team']
                        select = lc.chosen(len(y), ce.seed(path.stem, f'light_team_{v}'))
                        x = np.where(select[:, None], r['live']['team'][0], x)
                        xs.extend((x, supervised.swap_team_features(x, names)))
                        ys.extend((y, np.where(y < 2, 1-y, y)))
                    provider_team.append((np.concatenate(xs), np.concatenate(ys), names))
                elif kind == 'joint':
                    alternate = r['live' if arm == 'live_gap_had_gru' else 'gap']['joint']
                    for v, name in enumerate(('clean', 'legacy')):
                        for swap in (False, True):
                            records.append(mixed_record(r[name]['joint'], alternate, path.stem, v, swap))
                else:
                    for v, original in enumerate((r['clean']['dead'], r['dropout'])):
                        records.append(mixed_record(original, r['live']['dead'], path.stem, v))
                del r
            if provider_team:
                cache[provider] = (np.concatenate([r[0] for r in provider_team]),
                                   np.concatenate([r[1] for r in provider_team]), provider_team[0][2])
        if kind == 'team':
            model = ms.fit(tuple(cache), cache)
        elif kind == 'joint':
            model = had_gru.fit(records)
        elif kind == 'bigru':
            model = lc.fit_neural(records)
        else:
            mixed = []
            for r in records:
                old = lt.ordered_features(r['x'], r['names'], r['period'])
                new = lt.ordered_features(r['alternate'], r['names'], r['period'])
                mixed.append(dict(r, x=np.where(r['replace_rows'][:, None], new, old)))
            x, y, weights = lt.matrix(mixed)
            estimator = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31,
                learning_rate=.08, min_samples_leaf=40, l2_regularization=1,
                early_stopping=False, random_state=31)
            with threadpool_limits(limits=1):
                estimator.fit(x, y, sample_weight=weights)
            model = dict(kind='ordered_hgb', estimator=estimator)
        ce.freeze_model(out, target, model, dict(sources=sources,
            training_matches=[p.stem for s in sources for p in dt.SPLITS[s]['train']],
            arm=arm, kind=kind, seconds=time.monotonic()-start))
        print('FROZEN', source, arm, kind, round(time.monotonic()-start, 1), flush=True)


def evaluate(out, source):
    spec = check(out)
    for kind, path in archive_paths(source).items():
        if sha256(path) != spec['archived_baseline'][f'{source}/{kind}']['sha256']:
            raise ValueError('Archived baseline changed')
    models = {'baseline': {k: joblib.load(p) for k, p in archive_paths(source).items()}}
    for arm in ARMS[1:]:
        kinds = ('team', 'bigru', 'ordered_hgb') if arm == 'live_gap_pipeline' else ('joint',)
        models[arm] = {k: ce.load_model(out, out/f'without_{source}_{arm}_{k}.joblib') for k in kinds}
    destination = out/f'{source}_results.json'
    rows = json.loads(destination.read_text()) if destination.exists() else []
    for path in dt.SPLITS[source]['eval']:
        original, labels = ms.view(path)
        for view in VIEWS:
            if all(any(r['match'] == path.stem and r['view'] == view and r['arm'] == a for r in rows) for a in ARMS):
                continue
            inputs = original if view == 'clean' else tc.corrupt(original, spec['calibration'], ce.seed(path.stem, view))
            start = time.monotonic()
            team_x, names = refine_features.features(inputs)
            team_seconds = time.monotonic()-start
            start = time.monotonic()
            record = ce.record(inputs, labels, source, path.stem, view)
            d_seconds = time.monotonic()-start
            joint = dict(record, x=np.column_stack((team_x[record['indices']], record['x'])),
                         names=names+['D_'+n for n in record['names']])
            periods = supervised.model_periods(inputs)
            for arm in ARMS:
                if any(r['match'] == path.stem and r['view'] == view and r['arm'] == arm for r in rows):
                    continue
                start = time.monotonic()
                if 'joint' in models[arm]:
                    p5 = had_gru.predict(joint, models[arm]['joint'])
                    probability = np.column_stack([lt.upsample(p5[:, k], record, periods) for k in range(3)])
                else:
                    team = refine_features.probabilities_from_features(team_x, names, models[arm]['team'])
                    dead = .5*sum(lt.upsample(lt.predict_sampled(record, models[arm][k]), record, periods)
                                  for k in ('bigru', 'ordered_hgb'))
                    probability = refine_domain.combine(team, dead, 0.)
                predicted = supervised.decode(probability, periods, float(inputs['fps']), lt.COST)
                infer_seconds = time.monotonic()-start
                analysis = error_analysis.analyze(dict(inputs, labels=labels), predicted)
                metrics = analysis['metrics']
                short = analysis['by_gt_duration'][:2]
                short_n = sum(v['n_gt'] for v in short)
                short_hits = sum(v['n_matched_at_iou_0_5'] for v in short)
                headline = metrics['spell_f1'][.5]
                false = metrics['n_pred_spells']-headline['n_matched']
                graded_minutes = np.isin(labels, supervised.LABELS).sum()/float(inputs['fps'])/60
                row = dict(source=source, match=path.stem, view=view, arm=arm,
                    spell_f1=metrics['spell_f1_mean'], macro_f1=metrics['macro_f1'],
                    d_f1=metrics['per_class']['D']['f1'], short_spell_recall=short_hits/short_n if short_n else None,
                    short_reference_spells=short_n, short_matched_spells=short_hits,
                    false_possessions=int(false), false_possessions_per_minute=false/graded_minutes,
                    fragmented_reference_spells=analysis['overlap_structure']['n_fragmented_gt_spells'],
                    n_gt_spells=metrics['n_gt_spells'], n_pred_spells=metrics['n_pred_spells'],
                    inference_seconds=infer_seconds, feature_seconds=team_seconds+d_seconds,
                    total_seconds=infer_seconds+team_seconds+d_seconds,
                    duration_bins=analysis['by_gt_duration'], per_class=metrics['per_class'],
                    visibility=visibility_by_state(inputs, labels))
                np.savez_compressed(out/f'{source}_{path.stem}_{view}_{arm}.npz', prediction=predicted)
                rows.append(row)
                ce.write_json(destination, rows)
                print('SCORED', source, path.stem, view, arm, round(row['spell_f1'], 5), flush=True)


def report(out):
    spec = check(out)
    rows = [r for s in dt.SPLITS for r in json.loads((out/f'{s}_results.json').read_text())]
    expected = {(s, p.stem, v, a) for s in dt.SPLITS for p in dt.SPLITS[s]['eval'] for v in VIEWS for a in ARMS}
    if {(r['source'], r['match'], r['view'], r['arm']) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError('Incomplete/duplicate comparison')
    keys = ('spell_f1', 'macro_f1', 'd_f1', 'short_spell_recall', 'false_possessions_per_minute', 'total_seconds')
    summaries = []
    for view in ('clean', 'corrupted'):
        for arm in ARMS:
            per = {}
            for source in dt.SPLITS:
                chosen = [r for r in rows if r['source'] == source and r['arm'] == arm
                          and ((r['view'] == 'clean') == (view == 'clean'))]
                per[source] = {k: float(np.mean([r[k] for r in chosen])) for k in keys}
            summary = dict(view=view, arm=arm, per_source=per,
                           **{k: float(np.mean([v[k] for v in per.values()])) for k in keys})
            summary['weighted_70_30'] = .7*summary['spell_f1']+.3*summary['macro_f1']
            summaries.append(summary)
    baseline = next(r for r in summaries if r['view'] == 'clean' and r['arm'] == 'baseline')
    screens = {r['arm']: dict(delta=r['spell_f1']-baseline['spell_f1'],
                            passes=r['spell_f1']-baseline['spell_f1'] >= .01)
               for r in summaries if r['view'] == 'clean' and r['arm'] != 'baseline'}
    result = dict(spec=spec, summaries=summaries, screens=screens, rows=rows,
                  baseline_changed=False, organizer_check_performed=False,
                  checkpoint_receipts={p.name: json.loads(p.read_text()) for p in out.glob('without_*.json')},
                  training_visibility={p.stem: joblib.load(p)['audit'] for p in out.glob('train_*.joblib')})
    ce.write_json(out/'report.json', result, exclusive=True)
    for r in summaries:
        print('SUMMARY', r['view'], r['arm'], round(r['spell_f1'], 5), round(r['macro_f1'], 5), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'prepare', 'fit', 'evaluate', 'report'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--source', choices=tuple(dt.SPLITS))
    parser.add_argument('--arm', choices=ARMS[1:])
    args = parser.parse_args()
    if args.action == 'fit' and not args.arm:
        parser.error('fit requires --arm')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if args.action == 'freeze':
            freeze(args.out)
        elif args.action == 'report':
            report(args.out)
        else:
            for source in ([args.source] if args.source else dt.SPLITS):
                if args.action == 'prepare': prepare(args.out, source)
                elif args.action == 'fit': fit(args.out, source, args.arm)
                else: evaluate(args.out, source)


if __name__ == '__main__':
    main()
