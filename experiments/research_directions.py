"""Bounded external-development pilot for the first three research directions.

Freeze before fitting/scoring. Public organizer labels and closed public gates
are never opened. Calibration reads tracking from organizer half one only;
half two assesses missingness transfer without possession targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import warnings
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from threadpoolctl import threadpool_limits
from evaluation import possession_metrics as pm
from experiments import (current_baseline, dead_transfer, learned_temporal as lt,
                         multisource_eval, refine_domain, refine_features,
                         supervised, tracking_research as tr)
from experiments.eval_set.protocol import sha256

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('matched', 'gap', 'gap_dedup')
SPEC = dict(version=1, training_hz=2, hgb_iterations=160, hgb_leaves=31,
            learning_rate=.08, train_seed=192026, evaluation_seeds=list(tr.SEEDS),
            arms=list(ARMS), training='50% clean, 50% state-conditioned gaps and slot defects; team symmetry',
            weighting='Equal provider, match, view and team-swap mass',
            D='Unchanged pooled GRU/HGB probabilities shared by every team arm',
            decoder_switch_cost=3, ranking='Provider/seed mean official spell F1, exact ties macro F1',
            auxiliary='0.7 spell + 0.3 macro, never used for competition ranking',
            scope='Reused pooled external development; NOT unseen-provider transfer or fresh confirmation',
            organizer='Tracking-only first-half calibration; no possession labels or public scoring',
            uncertainty='Three evaluation-view seeds; one frozen training draw, not training-seed uncertainty')


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def paths():
    return [(source, role, p) for source, split in dead_transfer.SPLITS.items()
            for role, files in split.items() for p in files]


def source_hashes():
    return {str(p.relative_to(ROOT)):sha256(p) for p in
            (Path(__file__), Path(tr.__file__), Path(refine_features.__file__),
             Path(current_baseline.__file__), Path(lt.__file__))}


def check(out):
    spec = json.loads((out/'frozen.json').read_text())
    if spec['source_sha256'] != source_hashes() or spec['spec'] != SPEC:
        raise ValueError('Source/spec changed; use a new research output directory')
    for _, _, path in paths():
        if spec['external_inputs_sha256'][str(path.relative_to(ROOT))] != sha256(path):
            raise ValueError('External input hash mismatch')
    return spec


def calibrate(out, tactical):
    if (out/'frozen.json').exists():
        raise ValueError('Already frozen; use the existing calibration or a new directory')
    out.mkdir(parents=True, exist_ok=True)
    frozen = dict(spec=SPEC, source_sha256=source_hashes(),
                  external_inputs_sha256={str(p.relative_to(ROOT)):sha256(p) for _, _, p in paths()},
                  tactical_sha256=sha256(tactical), created_utc=datetime.now(timezone.utc).isoformat(),
                  baseline_manifest_sha256=sha256(current_baseline.MANIFEST),
                  baseline_model_hashes=json.loads(current_baseline.MANIFEST.read_text())['models_sha256'])
    # Exclusive freeze precedes all data fitting and outcome measurements.
    with (out/'frozen.json').open('x') as handle:
        json.dump(frozen, handle, indent=2)
        handle.write('\n')
    inputs = current_baseline.load_inputs(tactical)
    def half(number):
        rows = np.flatnonzero(inputs['period'] == number)
        a, b = rows[0], rows[-1]+1
        return {k:(v[a:b].copy() if k in ('ball', 'players', 'period', 'timestamps', 'ball_observed', 'player_observed') else v.copy())
                for k, v in inputs.items()}
    calibration = tr.fit_missingness(half(1))
    joblib.dump(calibration, out/'calibration.joblib')
    second = half(2)
    indices, x = tr.states(second)
    period = supervised.model_periods(second)[indices]
    seen = np.isfinite(second['ball'][indices]).all(axis=1)
    keep = seen[:-1] & (period[:-1] == period[1:])
    target = (~seen[1:])[keep]
    probability = calibration['estimator'].predict_proba(x[:-1][keep])[:, 1]
    heldout = dict(n_eligible=int(keep.sum()), n_onsets=int(target.sum()),
                   onset_auroc=float(roc_auc_score(target, probability)) if len(np.unique(target)) == 2 else None,
                   log_loss=float(log_loss(target, probability, labels=[False, True])),
                   constant_log_loss=float(log_loss(target, np.full(len(target), calibration['onset_probability']), labels=[False, True])),
                   scope='Organizer half two, observable missingness only; no possession targets')
    report = dict(spec=SPEC, organizer_profile=tr.profile(inputs),
                  calibration_profile=calibration['profile'], missingness_half_two=heldout,
                  baseline_changed=False, organizer_scored=False)
    write_json(out/'calibration_report.json', report)
    print('CALIBRATED', json.dumps(heldout), flush=True)


def training_seed(match):
    import hashlib
    return SPEC['train_seed']+int(hashlib.sha256(match.encode()).hexdigest()[:6], 16)


def prepare(out, matches=None):
    check(out)
    calibration = joblib.load(out/'calibration.joblib')
    for source, role, path in paths():
        if role != 'train':
            continue
        if matches is not None and path.stem not in matches:
            continue
        target = out/f'train_{path.stem}.joblib'
        if target.exists():
            continue
        inputs, labels = multisource_eval.view(path)
        rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::max(1, round(float(inputs['fps'])/SPEC['training_hz']))]
        y = np.array([np.flatnonzero(supervised.LABELS == label)[0] for label in labels[rows]], dtype=np.int8)
        view = tr.challenge_view(inputs, calibration, training_seed(path.stem))
        cache = {arm:[] for arm in ARMS}
        schemas = {}
        for variant in (inputs, view):
            raw, raw_names = refine_features.features(variant)
            extra, extra_names = tr.gap_features(variant)
            dedup, dedup_names = tr.features(variant, gap=True, dedup=True)
            matrices = {'matched':(raw, raw_names), 'gap':(np.column_stack((raw, extra)), list(raw_names)+extra_names),
                        'gap_dedup':(dedup, dedup_names)}
            for arm, (x, names) in matrices.items():
                cache[arm].append(x[rows])
                schemas[arm] = names
        joblib.dump(dict(source=source, match=path.stem, y=y, matrix=cache, schemas=schemas), target)
        print('PREPARED', source, path.stem, len(rows), flush=True)


def fit(out, arms=ARMS):
    check(out)
    for arm in arms:
        if arm not in ARMS:
            raise ValueError('Unknown arm')
        target = out/f'{arm}.joblib'
        if target.exists():
            continue
        xs, ys, ws, names = [], [], [], None
        for source, role, path in paths():
            if role != 'train':
                continue
            cache = joblib.load(out/f'train_{path.stem}.joblib')
            schema = cache['schemas'][arm]
            if names is not None and names != schema:
                raise ValueError('Training schemas differ')
            names = schema
            count = len(dead_transfer.SPLITS[source]['train'])
            y = cache['y']
            for x in cache['matrix'][arm]:
                xs.extend((x, supervised.swap_team_features(x, names)))
                ys.extend((y, np.where(y < 2, 1-y, y)))
                ws.extend((np.full(len(y), 1/(count*len(y))),)*2)
        weights = np.concatenate(ws)
        estimator = HistGradientBoostingClassifier(max_iter=SPEC['hgb_iterations'], max_leaf_nodes=SPEC['hgb_leaves'],
            learning_rate=SPEC['learning_rate'], min_samples_leaf=40, l2_regularization=1,
            early_stopping=False, random_state=17)
        with threadpool_limits(limits=1):
            estimator.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=weights/weights.mean())
        model = dict(estimator=estimator, feature_names=names, classes=estimator.classes_.tolist())
        joblib.dump(model, target)
        print('FITTED', arm, len(weights), flush=True)
    if all((out/f'{arm}.joblib').exists() for arm in ARMS):
        write_json(out/'models_frozen.json', {arm:sha256(out/f'{arm}.joblib') for arm in ARMS})


def diagnostics(inputs, labels, prediction):
    score = supervised.score(dict(inputs, labels=labels), prediction)
    eligible = np.isin(labels, ['H', 'A']) & ~np.isfinite(inputs['ball']).all(axis=1)
    periods = supervised.model_periods(inputs)
    inserts = np.flatnonzero(np.diff(periods) != 0)+1
    truth = pm.label_codes(np.insert(labels, inserts, 'X'))
    guess = pm.label_codes(np.insert(prediction, inserts, 'X'))
    runs = pm.valid_runs(truth)
    spells = pm.possession_spells(truth, runs)
    predicted = pm.possession_spells(pm.fill_abstentions(guess, runs), runs)
    matched = pm.matched_gt_flags(pm.spell_overlaps(spells, predicted), len(spells), .5)
    short = np.array([(b-a)/float(inputs['fps']) < 3 for a,b,_ in spells])
    return dict(spell_f1=score['spell_f1_mean'], macro_f1=score['macro_f1'],
                auxiliary_70_30=.7*score['spell_f1_mean']+.3*score['macro_f1'],
                d_precision=score['per_class']['D']['precision'], d_recall=score['per_class']['D']['recall'],
                gap_live_frames=int(eligible.sum()),
                gap_live_team_accuracy=float((prediction[eligible] == labels[eligible]).mean()) if eligible.any() else None,
                transition_f1={str(seconds):score['transitions'][max(1, round(seconds*float(inputs['fps'])))]['f1'] for seconds in (.5, 1.)},
                short_spells=int(short.sum()), short_spell_recall=float(matched[short].mean()) if short.any() else None,
                n_gt_spells=score['n_gt_spells'], n_pred_spells=score['n_pred_spells'])


def evaluate(out, matches=None, report_name='development.json'):
    frozen = check(out)
    baseline_manifest = json.loads(current_baseline.MANIFEST.read_text())
    if sha256(current_baseline.MANIFEST) != frozen['baseline_manifest_sha256']:
        raise ValueError('Baseline manifest changed')
    hashes = json.loads((out/'models_frozen.json').read_text())
    for arm in ARMS:
        if hashes[arm] != sha256(out/f'{arm}.joblib'):
            raise ValueError('Pilot checkpoint changed')
    models = {arm:joblib.load(out/f'{arm}.joblib') for arm in ARMS}
    baseline = current_baseline.load_models()
    calibration = joblib.load(out/'calibration.joblib')
    if Path(report_name).name != report_name:
        raise ValueError('Report name must be a filename')
    report_path = out/report_name
    rows = json.loads(report_path.read_text())['rows'] if report_path.exists() else []
    profiles = json.loads(report_path.read_text()).get('view_profiles', []) if report_path.exists() else []
    for source, role, path in paths():
        if role != 'eval':
            continue
        if matches is not None and path.stem not in matches:
            continue
        inputs, labels = multisource_eval.view(path)
        for seed in (None, *tr.SEEDS):
            view_name = 'clean' if seed is None else f'challenge_{seed}'
            if all(any(r['arm'] == arm and r['match'] == path.stem and r['view'] == view_name for r in rows)
                   for arm in ('current', *ARMS)):
                continue
            view = inputs if seed is None else tr.challenge_view(inputs, calibration, seed)
            profiles.append(dict(source=source, match=path.stem, view=view_name, **tr.profile(view)))
            dead = current_baseline.dead_probability(view, baseline)
            raw, names = refine_features.features(view)
            extra, extra_names = tr.gap_features(view)
            dedup, dedup_names = tr.features(view)
            matrices = {'current':(raw, names, baseline['attribution']), 'matched':(raw, names, models['matched']),
                        'gap':(np.column_stack((raw, extra)), list(names)+extra_names, models['gap']),
                        'gap_dedup':(dedup, dedup_names, models['gap_dedup'])}
            for arm, (x, schema, model) in matrices.items():
                if any(r['arm'] == arm and r['match'] == path.stem and r['view'] == view_name for r in rows):
                    continue
                team = refine_features.probabilities_from_features(x, schema, model)
                probability = refine_domain.combine(team, dead, 0.)
                prediction = supervised.decode(probability, supervised.model_periods(view), float(view['fps']), 3)
                row = dict(source=source, match=path.stem, view=view_name, arm=arm, **diagnostics(view, labels, prediction))
                rows.append(row)
                write_json(report_path, dict(spec=SPEC, rows=rows, view_profiles=profiles, baseline_changed=False,
                                             organizer_scored=False, target_masks_unchanged=True))
                print('SCORED', source, path.stem, view_name, arm, round(row['spell_f1'], 4), flush=True)


def merge(out):
    """Merge independently written evaluation shards, rejecting missing cells."""
    check(out)
    rows, profiles = [], []
    for path in sorted(out.glob('development_*.json')):
        report = json.loads(path.read_text())
        if report['spec'] != SPEC:
            raise ValueError('Shard spec differs')
        rows.extend(report['rows'])
        profiles.extend(report['view_profiles'])
    expected = {(source, path.stem, view, arm) for source, role, path in paths() if role == 'eval'
                for view in ('clean', *(f'challenge_{s}' for s in tr.SEEDS)) for arm in ('current', *ARMS)}
    observed = [(r['source'], r['match'], r['view'], r['arm']) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Evaluation shards contain duplicates or missing cohort cells')
    write_json(out/'development.json', dict(spec=SPEC, rows=rows, view_profiles=profiles,
               baseline_changed=False, organizer_scored=False, target_masks_unchanged=True))
    print('MERGED', len(rows), 'evaluations', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('calibrate', 'prepare', 'fit', 'evaluate', 'merge', 'all'))
    p.add_argument('--out', type=Path, default=ROOT/'data/experiments/top_three_v1')
    p.add_argument('--tactical', type=Path, default=ROOT/'data/challenge/inputs.npz')
    p.add_argument('--matches', nargs='+', help='Prepare/evaluate only these fixed match IDs; fitting still requires every training match')
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--report-name', default='development.json', help='Distinct development_*.json filenames allow parallel evaluation')
    args = p.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for action in (('calibrate', 'prepare', 'fit', 'evaluate') if args.action == 'all' else (args.action,)):
            if action == 'calibrate':
                calibrate(args.out, args.tactical)
            elif action == 'prepare':
                prepare(args.out, args.matches)
            elif action == 'fit':
                fit(args.out, args.arms)
            elif action == 'evaluate':
                evaluate(args.out, args.matches, args.report_name)
            else:
                globals()[action](args.out)


if __name__ == '__main__':
    main()
