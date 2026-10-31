"""One bounded, fixed-update ablation of six additional training matches.

Frozen historical runners/manifests stay unchanged. The separately requested
organizer diagnostic is implemented in a separate entrypoint after the
external decision and checkpoint seal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import (combined_objective_deployment as deployment,
                         factorized_had as network, learned_temporal as lt,
                         preparation as prep, supervised, error_analysis,
                         tracking_corruption as tc)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('MTAH_DATA_DIR', ROOT / 'data')).resolve()
OUT = DATA / 'experiments/skillcorner_expansion_v1'
MANIFEST = ROOT / 'experiments/eval_set/candidates/skillcorner_expansion_v1.json'
OLD_MANIFEST = ROOT / 'experiments/eval_set/manifest.json'
OLD_CACHE = DATA / 'experiments/objective_alignment_v1'
SEEDS = (2026, 2027)
FOLDS = ('leave_out_sportec', 'leave_out_metrica', 'pooled')
VIEWS = (('clean', 'clean'), ('feed_like', 'corrupt_1'), ('feed_like', 'corrupt_2'))
SCORE_KEYS = ('spell_f1', 'macro_f1', 'd_f1', 'd_precision', 'd_recall',
              'weighted_70_30', 'short_spell_recall')


def sha(path):
    return prep.sha256(Path(path))


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Refusing to change completed receipt: {path}')
        return
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def read_manifest(path=MANIFEST):
    value = read(path)
    prep.split_plan(path, 'pooled')
    training = {i['match'] for s in value['sources'].values() for i in s['train']}
    selection = {i['match'] for s in value['sources'].values() for i in s['selection']}
    reserved = {i['match'] for i in value.get('reserved_external_tests', [])}
    if training & selection or training & reserved or selection & reserved:
        raise ValueError('Match leakage across training, selection or reserved test')
    if value['historical_tests_excluded'] != read(OLD_MANIFEST)['historical_tests_excluded']:
        raise ValueError('Historical test exclusions changed')
    if path == MANIFEST and (len(training), len(selection), len(reserved)) != (19, 7, 1):
        raise ValueError('Unexpected expansion cohort')
    return value


def training_items(fold, expanded=True):
    path = MANIFEST if expanded else OLD_MANIFEST
    read_manifest(path)
    return prep.split_plan(path, fold)['train']


def selection_items(reserved=False):
    value = read_manifest()
    if reserved:
        return value['reserved_external_tests']
    return [dict(i, source=s) for s, rows in value['sources'].items() for i in rows['selection']]


def control_path(fold, seed):
    if fold == 'pooled':
        return DATA / f'models/joint_gru_combined_ensemble_v1/joint_gru_{seed}.joblib'
    source = fold.removeprefix('leave_out_')
    return OLD_CACHE / f'without_{source}_combined_{seed}.joblib'


def candidate_path(fold, seed):
    if fold == 'leave_out_skillcorner':
        return control_path(fold, seed)
    return OUT / f'{fold}_{seed}.joblib'


def source_hashes():
    return {str(p.relative_to(ROOT)): sha(p)
            for folder in ('experiments', 'evaluation')
            for p in sorted((ROOT / folder).rglob('*.py'))}


def old_cache_info():
    declaration = read(ROOT / 'experiments/results/combined_objective_deployment_declaration.json')
    return {r['match']: r for r in declaration['training_cache_files']}


def prepare():
    from experiments import expansion_preparation as ep
    original = {i['match'] for i in training_items('pooled', expanded=False)}
    calibration = read(ROOT / 'experiments/results/realistic_corruption_development.json')['spec']['calibration']
    source_index = {i['match']: n for n, i in enumerate(read_manifest()['sources']['skillcorner']['train'])}
    for item in training_items('pooled'):
        if item['match'] not in original:
            ep.prepare_training(item, source_index[item['match']], OUT, calibration, DATA)
    # Native/augmented evaluation views contain no learned transformations.
    for item in selection_items():
        for view, replicate in VIEWS:
            old = DATA / f'experiments/seed_gap_context_v1/{item["match"]}_{view}_{replicate}.joblib'
            if not old.exists():
                ep.prepare_eval(item, view, replicate, OUT, calibration, DATA)


def freeze():
    from experiments import expansion_preparation, expansion_budget
    read_manifest()
    old_info = old_cache_info()
    inputs = {}
    for item in training_items('pooled') + selection_items() + selection_items(True):
        path = prep.data_path(item, DATA)
        if sha(path) != item['sha256']:
            raise ValueError(f'Input checksum mismatch: {item["match"]}')
        inputs[str(path.relative_to(DATA))] = item['sha256']
    caches = {}
    for item in training_items('pooled'):
        if item['match'] in old_info:
            info = old_info[item['match']]
            path = (ROOT / info['cache']).resolve()
            if sha(path) != info['cache_sha256'] or sha(ROOT / info['receipt']) != info['receipt_sha256']:
                raise ValueError('Historical training cache checksum changed')
        else:
            path = OUT / f'train_{item["match"]}.joblib'
            receipt = read(path.with_suffix('.json'))
            if sha(path) != receipt['sha256']:
                raise ValueError('New training cache checksum changed')
        caches[str(path.relative_to(DATA))] = sha(path)
        caches[str(path.with_suffix('.json').relative_to(DATA))] = sha(path.with_suffix('.json'))
    archived_evaluation = read(ROOT.parent / 'experiment_restore/mtah-data/objective_ensemble_cache_manifest.json')
    expected_eval = {i['path']: i['sha256'] for i in archived_evaluation['files']}
    evaluation = {}
    for item in selection_items():
        for view, replicate in VIEWS:
            old = DATA / f'experiments/seed_gap_context_v1/{item["match"]}_{view}_{replicate}.joblib'
            path = old if old.exists() else OUT / f'eval_{item["match"]}_{view}_{replicate}.joblib'
            relative = str(path.relative_to(DATA))
            actual = sha(path)
            if old.exists():
                if expected_eval.get(relative) != actual:
                    raise ValueError('Historical evaluation cache checksum mismatch')
            else:
                receipt = read(path.with_suffix('.json'))
                if actual != receipt['cache_sha256'] or receipt['input_sha256'] != item['sha256']:
                    raise ValueError('New evaluation cache checksum mismatch')
                evaluation[str(path.with_suffix('.json').relative_to(DATA))] = sha(path.with_suffix('.json'))
            evaluation[relative] = actual
    controls, budgets = {}, {}
    archive = read(ROOT.parent / 'experiment_restore/mtah-data/objective_alignment_cache_manifest.json')
    expected = {i['path']: i['sha256'] for i in archive['files']}
    baseline_manifest = read(ROOT / 'experiments/baseline_manifest.json')
    for fold in (*FOLDS, 'leave_out_skillcorner'):
        for seed in SEEDS:
            path = control_path(fold, seed)
            relative = str(path.relative_to(DATA))
            target = (baseline_manifest['models_sha256'][path.name] if fold == 'pooled'
                      else expected[relative])
            if sha(path) != target:
                raise ValueError('Restored baseline checkpoint checksum mismatch')
            model = joblib.load(path)
            controls[relative] = target
            budgets[f'{fold}:{seed}'] = {k: int(model[k]) for k in
                ('windows_per_epoch', 'optimizer_updates', 'parameters')}
    value = dict(study_id='skillcorner_expansion_v1', created_utc=datetime.now(timezone.utc).isoformat(),
        question='Does adding six same-source training matches improve spell recovery at the original optimizer budget?',
        arms=['restored_original_13_training_matches', 'expanded_19_training_matches'],
        seeds=list(SEEDS), fit_folds=list(FOLDS), new_fits=6,
        primary='Seven-match full-provider holdout; original five-match subset also reported',
        supplement='Pooled fits on the same seven held-out matches; normalization and every component exclude all selection matches',
        unchanged_skillcorner_fold='Both arms reuse the same without-SkillCorner checkpoints; this is parity, not an expansion gain',
        split_manifest_sha256=sha(MANIFEST), old_split_manifest_sha256=sha(OLD_MANIFEST),
        source_sha256=source_hashes(), input_sha256=inputs, cache_sha256=caches,
        evaluation_sha256=evaluation,
        control_sha256=controls, control_budgets=budgets,
        preparation='Original thirteen historical caches restored exactly. New matches use shared preparation, existing augmentation and source/match loss masses; no added weighting hyperparameter.',
        sampling='Eight epochs, batch64; unchanged control path at original window count; candidate draws that same number without replacement each epoch from the expanded set',
        normalization='Recomputed from expanded TRAIN records only; expected to change because training data change',
        capacity='Same hidden24/player-only-D GRU, combined duration+transition objective, 10s cores/5s context, AdamW .001/.0001, clip1',
        decoder='Equal native H/A/D probabilities from both seeds, then unchanged Viterbi cost3',
        views=[list(v) for v in VIEWS], calibration=read(ROOT / 'experiments/results/realistic_corruption_development.json')['spec']['calibration'],
        screen={'primary_clean_spell_gain_min': .01, 'primary_feed_like_spell_delta_min': -.01,
                'provider_clean_macro_delta_min': -.02, 'provider_clean_D_f1_delta_min': -.02,
                'pooled_clean_spell_delta': 'strictly positive', 'pooled_feed_like_spell_delta_min': -.01,
                'pooled_provider_clean_macro_and_D_delta_min': -.02},
        reserved_test='2017461: evaluate only after external screen passes and all checkpoints/decision are sealed; no fitting or selection on it',
        organizer='Explicitly requested one-time fixed pooled candidate diagnostic after external decision, whether pass or fail; seal predictions and claim a distinct exclusive gate; never select or tune from it',
        historical_organizer_diagnostic_known=True, fitting_organizer_labels=False,
        interpretation='Provider/event-derived silver development evidence; synthetic feed_like stress is not real tactical gap confirmation')
    if (OUT / 'predeclared.json').exists():
        check()
        return
    write(OUT / 'predeclared.json', value)
    write(ROOT / 'experiments/results/skillcorner_expansion_declaration.json', value)
    print('FROZEN', sha(OUT / 'predeclared.json'), flush=True)


def check():
    spec = read(OUT / 'predeclared.json')
    if (source_hashes() != spec['source_sha256'] or sha(MANIFEST) != spec['split_manifest_sha256']
            or sha(OLD_MANIFEST) != spec['old_split_manifest_sha256']):
        raise ValueError('Frozen executable source or cohort changed')
    for group in ('input_sha256', 'cache_sha256', 'control_sha256', 'evaluation_sha256'):
        for relative, expected in spec[group].items():
            if sha(DATA / relative) != expected:
                raise ValueError(f'Frozen artifact changed: {relative}')
    return spec


def load_training_records(fold):
    info = old_cache_info()
    records = []
    for item in training_items(fold):
        path = ROOT / info[item['match']]['cache'] if item['match'] in info else OUT / f'train_{item["match"]}.joblib'
        cache = joblib.load(path, mmap_mode='r')
        if cache['match'] != item['match'] or cache['source'] != item['source'] or cache['input_sha256'] != item['sha256']:
            raise ValueError('Training cache identity mismatch')
        records.extend(deployment.paired_records(cache, item['match']))
    return records


def fit(fold, seed):
    from experiments import objective_losses, expansion_budget
    spec = check()
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError('Fit is outside the declared six candidates')
    if (OUT / 'checkpoints_frozen.json').exists():
        raise ValueError('Training is sealed')
    path = candidate_path(fold, seed)
    if path.exists():
        receipt = read(path.with_suffix('.json'))
        if sha(path) != receipt['sha256']:
            raise ValueError('Existing fitted checkpoint changed')
        return
    records = load_training_records(fold)
    expected = spec['control_budgets'][f'{fold}:{seed}']
    start = time.monotonic()
    with deployment.file_backed_normalization(OUT / 'temporary') as storage:
        with expansion_budget.matched_window_budget(expected['windows_per_epoch']) as budget:
            model = objective_losses.fit(records, player_only=True, seed=seed, objective='combined')
    if model['optimizer_updates'] != expected['optimizer_updates'] or model['parameters'] != expected['parameters']:
        raise ValueError('Candidate does not match the frozen optimizer/capacity budget')
    model['dataset_expansion'] = 'skillcorner_expansion_v1'
    joblib.dump(model, path)
    receipt = dict(fold=fold, seed=seed, sha256=sha(path), declaration_sha256=sha(OUT / 'predeclared.json'),
        training_matches=[i['match'] for i in training_items(fold)],
        parameters=model['parameters'], optimizer_updates=model['optimizer_updates'],
        dataset_windows=model['windows_per_epoch'], budget=budget.as_dict(), storage=storage,
        seconds=time.monotonic()-start)
    write(path.with_suffix('.json'), receipt)
    print('FITTED', fold, seed, receipt['seconds'], flush=True)


def seal():
    check()
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            path = candidate_path(fold, seed)
            receipt = read(path.with_suffix('.json'))
            if sha(path) != receipt['sha256'] or receipt['declaration_sha256'] != sha(OUT / 'predeclared.json'):
                raise ValueError('Checkpoint or its fit declaration changed')
            rows.append(dict(fold=fold, seed=seed, path=str(path.relative_to(DATA)),
                             sha256=sha(path), receipt_sha256=sha(path.with_suffix('.json'))))
    write(OUT / 'checkpoints_frozen.json', dict(checkpoints=rows))


def predict_native(record, fitted, inputs):
    sampled = network.predict(record, fitted)
    periods = supervised.model_periods(inputs)
    probability = np.column_stack([lt.upsample(sampled[:, k], record, periods) for k in range(3)])
    if not np.isfinite(probability).all() or (probability < 0).any() or not np.allclose(probability.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid native H/A/D probability')
    return probability


def score_cell(inputs, labels, probability):
    prediction = supervised.decode(probability, supervised.model_periods(inputs), float(inputs['fps']), lt.COST)
    analysis = error_analysis.analyze(dict(inputs, labels=labels), prediction)
    m = analysis['metrics']
    short = analysis['by_gt_duration'][:2]
    n_short = sum(r['n_gt'] for r in short)
    result = dict(spell_f1=m['spell_f1_mean'], macro_f1=m['macro_f1'],
        weighted_70_30=.7*m['spell_f1_mean']+.3*m['macro_f1'],
        d_f1=m['per_class']['D']['f1'], d_precision=m['per_class']['D']['precision'],
        d_recall=m['per_class']['D']['recall'], n_gt_spells=m['n_gt_spells'], n_pred_spells=m['n_pred_spells'],
        short_spell_recall=sum(r['n_matched_at_iou_0_5'] for r in short)/n_short if n_short else None,
        short_reference_spells=n_short, spell_by_iou={str(k): v['f1'] for k, v in m['spell_f1'].items()},
        fragmented_reference_spells=analysis['overlap_structure']['n_fragmented_gt_spells'],
        graded_frames=int(np.isin(labels, ['H', 'A', 'D']).sum()))
    return result, prediction


def evaluation_cache(item, view, replicate):
    old = DATA / f'experiments/seed_gap_context_v1/{item["match"]}_{view}_{replicate}.joblib'
    path = old if old.exists() else OUT / f'eval_{item["match"]}_{view}_{replicate}.joblib'
    spec = read(OUT / 'predeclared.json')
    expected = spec['evaluation_sha256'].get(str(path.relative_to(DATA)))
    if expected is None or sha(path) != expected:
        raise ValueError('Evaluation input changed after declaration')
    cached = joblib.load(path)
    record = cached.get('record', cached.get('joint'))
    if record is None:
        raise ValueError('Evaluation cache is missing its feature record')
    return path, cached['inputs'], cached['labels'], record


def validate_result(path):
    """Replay official metrics from sealed probabilities before accepting rows."""
    path = Path(path)
    value = read(path)
    spec = read(OUT / 'predeclared.json')
    if value['declaration_sha256'] != sha(OUT / 'predeclared.json'):
        raise ValueError('Evaluation result belongs to a different declaration')
    cache = DATA / value['cache']
    expected = spec['evaluation_sha256'].get(value['cache'])
    if expected is None or value['cache_sha256'] != expected or sha(cache) != expected:
        raise ValueError('Evaluation result used a changed cache')
    probability_file = path.with_suffix('.npz')
    if sha(probability_file) != value['probability_sha256']:
        raise ValueError('Saved probabilities changed')
    cached = joblib.load(cache)
    with np.load(probability_file, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    if len(value['rows']) != 6 or {(r['arm'], r['member']) for r in value['rows']} != {
            (a, m) for a in ('control', 'candidate') for m in ('2026', '2027', 'ensemble')}:
        raise ValueError('Incomplete paired score rows')
    for arm in ('control', 'candidate'):
        if not np.array_equal(arrays[f'{arm}_ensemble'], .5*(arrays[f'{arm}_2026']+arrays[f'{arm}_2027'])):
            raise ValueError('Ensemble does not equal member probability mean')
    for row in value['rows']:
        stem = f'{row["protocol"]}_{row["match"]}_{row["view"]}_{row["replicate"]}'
        if stem != path.stem:
            raise ValueError('Score identity does not match its artifact path')
        key = f'{row["arm"]}_{row["member"]}'
        metrics, prediction = score_cell(cached['inputs'], cached['labels'], arrays[key])
        if not np.array_equal(prediction, arrays[f'{key}_prediction']):
            raise ValueError('Saved prediction differs from unchanged cost-3 decoder')
        for name, actual in metrics.items():
            if isinstance(actual, float):
                if abs(actual-row[name]) > 1e-12:
                    raise ValueError(f'Saved score differs from official replay: {name}')
            elif actual != row[name]:
                raise ValueError(f'Saved score differs from official replay: {name}')
    return value['rows']


def evaluate(protocol='primary', match=None, reserved=False):
    spec = check()
    if not (OUT / 'checkpoints_frozen.json').exists():
        raise ValueError('Seal all models before scoring')
    seal()
    if protocol not in ('primary', 'pooled'):
        raise ValueError('Unknown evaluation protocol')
    if reserved and not read(OUT / 'decision.json')['adopt_dataset']:
        raise ValueError('Reserved test requires a successful sealed external screen')
    for item in selection_items(reserved):
        if match and item['match'] != match:
            continue
        fold = f'leave_out_{item["source"]}' if protocol == 'primary' else 'pooled'
        for view, replicate in VIEWS:
            dest = OUT / f'{protocol}_{item["match"]}_{view}_{replicate}.json'
            if dest.exists():
                validate_result(dest)
                continue
            cache_path, inputs, labels, record = evaluation_cache(item, view, replicate)
            arrays, rows = {}, []
            for arm in ('control', 'candidate'):
                for seed in SEEDS:
                    path = control_path(fold, seed) if arm == 'control' else candidate_path(fold, seed)
                    if arm == 'candidate' and fold == 'leave_out_skillcorner':
                        probability = arrays[f'control_{seed}']
                    else:
                        probability = predict_native(record, joblib.load(path), inputs)
                    arrays[f'{arm}_{seed}'] = probability
                    archived = OLD_CACHE / f'{item["source"]}_{item["match"]}_{view}_{replicate}_combined_{seed}.npz'
                    if protocol == 'primary' and arm == 'control' and archived.exists():
                        with np.load(archived, allow_pickle=False) as original:
                            if not np.allclose(probability, original['probability'], rtol=0, atol=1e-6):
                                raise ValueError('Historical native-probability parity failed')
                arrays[f'{arm}_ensemble'] = .5*(arrays[f'{arm}_2026']+arrays[f'{arm}_2027'])
                for member in ('2026', '2027', 'ensemble'):
                    result, prediction = score_cell(inputs, labels, arrays[f'{arm}_{member}'])
                    arrays[f'{arm}_{member}_prediction'] = prediction
                    rows.append(dict(protocol=protocol, source=item['source'], match=item['match'],
                        view=view, replicate=replicate, arm=arm, member=member, **result))
            if len({r['n_gt_spells'] for r in rows}) != 1 or len({r['graded_frames'] for r in rows}) != 1:
                raise ValueError('Paired evaluation populations changed')
            np.savez_compressed(dest.with_suffix('.npz'), **arrays)
            write(dest, dict(rows=rows, probability_sha256=sha(dest.with_suffix('.npz')),
                cache=str(cache_path.relative_to(DATA)), cache_sha256=sha(cache_path),
                declaration_sha256=sha(OUT / 'predeclared.json')))
            print('SCORED', protocol, item['match'], view, replicate,
                  [round(r['spell_f1'], 6) for r in rows if r['member'] == 'ensemble'], flush=True)


def summarize(rows, protocol, arm, view, member='ensemble', originals_only=False):
    original = {i['match'] for i in prep.split_plan(OLD_MANIFEST)['selection']}
    chosen = [r for r in rows if (r['protocol'], r['arm'], r['view'], r['member']) == (protocol, arm, view, member)
              and (not originals_only or r['match'] in original)]
    result = {}
    providers = sorted({r['source'] for r in chosen})
    for source in providers:
        subset = [r for r in chosen if r['source'] == source]
        per_match = []
        for match in sorted({r['match'] for r in subset}):
            group = [r for r in subset if r['match'] == match]
            per_match.append({key: float(np.mean([r[key] for r in group if r[key] is not None])) for key in SCORE_KEYS})
        result[source] = {key: float(np.mean([r[key] for r in per_match])) for key in SCORE_KEYS}
    if not result:
        raise ValueError('Empty summary population')
    return dict(equal_provider={key: float(np.mean([r[key] for r in result.values()])) for key in SCORE_KEYS}, providers=result)


def report():
    seal()
    rows = []
    for protocol in ('primary', 'pooled'):
        for item in selection_items():
            for view, replicate in VIEWS:
                rows.extend(validate_result(OUT / f'{protocol}_{item["match"]}_{view}_{replicate}.json'))
    summaries = {f'{protocol}:{arm}:{view}:{member}': summarize(rows, protocol, arm, view, member)
                 for protocol in ('primary', 'pooled') for arm in ('control', 'candidate')
                 for view in ('clean', 'feed_like') for member in ('2026', '2027', 'ensemble')}
    original_five = {f'{arm}:{view}': summarize(rows, 'primary', arm, view, originals_only=True)
                     for arm in ('control', 'candidate') for view in ('clean', 'feed_like')}
    saved = read(ROOT / 'experiments/results/objective_ensemble_rows.json')['rows']
    for row in rows:
        if row['protocol'] != 'primary' or row['arm'] != 'control' or row['member'] != 'ensemble':
            continue
        previous = [r for r in saved if (r['match'], r['view'], r['replicate'], r['recipe']) ==
                    (row['match'], row['view'], row['replicate'], 'combined_objective_equal_probability_ensemble')]
        if previous:
            for key in ('spell_f1', 'macro_f1', 'd_f1', 'n_gt_spells', 'n_pred_spells'):
                if abs(row[key]-previous[0][key]) > 1e-9:
                    raise ValueError(f'Historical score parity failed: {key}')
    gates, deltas = {}, {}
    for protocol in ('primary', 'pooled'):
        for view in ('clean', 'feed_like'):
            a = summaries[f'{protocol}:control:{view}:ensemble']
            b = summaries[f'{protocol}:candidate:{view}:ensemble']
            delta = {k: b['equal_provider'][k]-a['equal_provider'][k] for k in SCORE_KEYS}
            deltas[f'{protocol}:{view}'] = delta
            threshold = .01 if protocol == 'primary' and view == 'clean' else -.01
            gates[f'{protocol}:{view}:spell'] = (delta['spell_f1'] >= threshold
                if not (protocol == 'pooled' and view == 'clean') else delta['spell_f1'] > 0)
            if view == 'clean':
                for source in a['providers']:
                    for key in ('macro_f1', 'd_f1'):
                        gates[f'{protocol}:{source}:{key}'] = b['providers'][source][key]-a['providers'][source][key] >= -.02
    result = dict(study_id='skillcorner_expansion_v1', summaries=summaries,
        original_five=original_five, deltas=deltas, guards=gates,
        adopt_dataset=all(gates.values()), source_and_model_checks='passed',
        historical_control_score_parity=True, completed_rows=len(rows), new_fits=6,
        organizer_labels_read=False, public_diagnostic_used_for_decision=False)
    write(OUT / 'decision.json', result)
    write(ROOT / 'experiments/results/skillcorner_expansion.json', result)
    write(OUT / 'external_rows.json', dict(rows=rows))
    print(json.dumps(dict(adopt_dataset=result['adopt_dataset'], deltas=deltas, guards=gates), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'freeze', 'fit', 'seal', 'evaluate', 'report'))
    parser.add_argument('--fold', choices=FOLDS)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--protocol', choices=('primary', 'pooled'), default='primary')
    parser.add_argument('--match')
    parser.add_argument('--reserved', action='store_true')
    args = parser.parse_args()
    if args.action == 'fit': fit(args.fold, args.seed)
    elif args.action == 'evaluate': evaluate(args.protocol, args.match, args.reserved)
    else: globals()[args.action]()


if __name__ == '__main__':
    main()
