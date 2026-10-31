"""Frozen, external-only abstention comparison; no fitting or organizer scoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
from evaluation import possession_metrics as pm
from experiments import (corruption_experiment as ce, current_baseline as cb,
                         learned_temporal as lt, multisource_eval, supervised,
                         tracking_corruption as tc)
from experiments.eval_set.protocol import paired_source_bootstrap, sha256

ROOT = lt.ROOT
OUT = ROOT / 'data/experiments/confidence_abstention_v1'
THRESHOLDS = (.5, .6, .7, .8)
ARMS = ['baseline'] + [f'{mode}_{int(t*100)}' for mode in ('abstain', 'hold') for t in THRESHOLDS]
REPORT = ROOT / 'experiments/results/confidence_abstention.json'
DECLARATION = ROOT / 'experiments/results/confidence_abstention_declaration.json'


def gate(probabilities, decoded, periods, threshold, mode):
    """Use probability of the Viterbi-selected class, without reference labels.

    hold fills within input time blocks, never reference-derived graded runs.
    Leading gaps use the first accepted label (offline). With no accepted label,
    hold retains the existing decoder; abstain remains entirely X.
    """
    p, decoded, periods = np.asarray(probabilities), np.asarray(decoded), np.asarray(periods)
    if p.shape != (len(decoded), 3) or periods.shape != decoded.shape:
        raise ValueError('Mismatched prediction shapes')
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError('Expected finite normalized probabilities')
    if mode not in ('abstain', 'hold') or not 0 <= threshold <= 1:
        raise ValueError('Invalid gate configuration')
    if not np.isin(decoded, supervised.LABELS).all():
        raise ValueError('Decoder must supply H/A/D')
    codes = pm.label_codes(decoded)
    out = decoded.copy()
    out[p[np.arange(len(p)), codes] < threshold] = 'X'
    if mode == 'hold':
        for a, b in supervised.blocks(periods):
            known = np.flatnonzero(out[a:b] != 'X')
            if not len(known):
                out[a:b] = decoded[a:b]
                continue
            carry = np.maximum.accumulate(np.where(out[a:b] != 'X', np.arange(b-a), -1))
            carry[carry < 0] = known[0]
            out[a:b] = out[a:b][carry]
    return out


def sources():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT/'experiments').glob('*.py'))} | {
        str(p.relative_to(ROOT)): sha256(p) for p in [ROOT/'evaluation/possession_metrics.py',
        ROOT/'experiments/eval_set/protocol.py', ROOT/'experiments/baseline_manifest.json',
        ROOT/'experiments/model_decision_policy.json']}


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if DECLARATION.exists():
        check()
        return
    manifest = json.loads((ROOT/'experiments/eval_set/manifest.json').read_text())
    inputs = {}
    for source in manifest['sources'].values():
        for entry in source['selection']:
            path = ROOT/entry['path']
            if sha256(path) != entry['sha256']:
                raise ValueError(f'Input hash mismatch: {path}')
            inputs[entry['path']] = entry['sha256']
    cb.load_models()
    model_files = sorted((ROOT/'data/models/gru_hgb_ballgap25_v1').glob('*.joblib'))
    for source in ce.dead_transfer.SPLITS:
        for kind in ('bigru', 'ordered_hgb'):
            path = ce.DEFAULT/f'without_{source}_baseline_{kind}.joblib'
            ce.load_model(ce.DEFAULT, path)
            model_files.append(path)
    old = json.loads((ce.DEFAULT/'predeclared.json').read_text())
    spec = dict(study='confidence_abstention_v1', frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        hypothesis='Suppressing low-confidence decoded labels improves spell F1 without material clean/stress regressions',
        primary_arm='abstain_60', sensitivity_thresholds=list(THRESHOLDS), arms=ARMS,
        confidence='Uncalibrated joint H/A/D probability assigned to the Viterbi-selected class; strict < threshold',
        hold_control='Same rejected frames, explicitly filled within input-only time blocks; first accepted state backfills leading gaps; no accepted state retains baseline',
        protocol='D-only provider holdout, pooled unchanged ballgap25 H/A; five previously exposed external development matches; not full-pipeline holdout or confirmation',
        splits={s: {role: [p.stem for p in paths] for role, paths in split.items()}
                for s, split in ce.dead_transfer.SPLITS.items()},
        source_semantics={s: {k: v[k] for k in ('labels', 'quality', 'shortcut')} for s, v in manifest['sources'].items()},
        grading='Original native X and SkillCorner <12 observed-player filter fixed before corruption; no target-dependent inference; preserve period/timestamp breaks',
        views=list(ce.VIEWS), seeds={p.stem: {v: ce.seed(p.stem, v) for v in ce.VIEWS[1:]}
              for split in ce.dead_transfer.SPLITS.values() for p in split['eval']},
        calibration=old['calibration'], corruption_recipe=tc.RECIPE,
        calibration_status='Reused frozen label-free organizer tracking gap profile, no new organizer access; no independent tracking-quality audit',
        training='None; restored checkpoints and normalization unchanged, original fixed training seeds',
        budget='15 prediction cells, 9 arms each = 135 evaluations; 0 fits; no added arms after scores',
        primary='Official spell F1, macro F1 for exact ties; fixed auxiliary .7 spell + .3 macro',
        aggregation='Average stress seeds within match, matches within provider, then equal provider weights',
        guardrails={'clean_spell_delta_min': 0., 'stress_spell_delta_min': 0.,
                    'macro_delta_min_each_view': -.005, 'provider_spell_delta_min_each_view': -.02},
        decision='No automatic adoption. Primary hypothesis passes only if abstain_60 improves clean spell and meets every guardrail. Sensitivity arms are exploratory.',
        diagnostics=['abstention rate', 'changed label rate', 'D precision/recall', 'spell counts',
                     'short-spell recall IoU .5', 'typed boundary F1/latency at .5 seconds',
                     'abstention versus hold temporal differences from native reference X gaps'],
        baseline_parity='Every cell must match archived ball_team25 provider-holdout spell/macro within 1e-12 before any candidate scoring',
        source_sha256=sources(), input_sha256=inputs,
        models_sha256={str(p.relative_to(ROOT)): sha256(p) for p in model_files},
        inherited_declaration_sha256=sha256(ce.DEFAULT/'predeclared.json'),
        historical_report_sha256=sha256(ROOT/'experiments/results/light_corruption_development.json'),
        private_output=str(OUT.relative_to(ROOT)), organizer_labels_read=False, organizer_scored=False,
        historical_tests_accessed=False, fresh_confirmation=False)
    ce.write_json(DECLARATION, spec, exclusive=True)
    print('FROZEN', DECLARATION, flush=True)


def check():
    spec = json.loads(DECLARATION.read_text())
    if spec['source_sha256'] != sources():
        raise ValueError('Frozen source changed')
    for key in ('input_sha256', 'models_sha256'):
        for name, digest in spec[key].items():
            if sha256(ROOT/name) != digest:
                raise ValueError(f'Frozen artifact changed: {name}')
    if sha256(ce.DEFAULT/'predeclared.json') != spec['inherited_declaration_sha256']:
        raise ValueError('Inherited corruption declaration changed')
    if sha256(ROOT/'experiments/results/light_corruption_development.json') != spec['historical_report_sha256']:
        raise ValueError('Historical baseline report changed')
    return spec


def prepare():
    spec = check()
    if (OUT/'prediction_seal.json').exists():
        verify_seal()
        return
    models = cb.load_models()
    cells = []
    for source, split in ce.dead_transfer.SPLITS.items():
        models = dict(models, **{kind: ce.load_model(ce.DEFAULT, ce.DEFAULT/f'without_{source}_baseline_{kind}.joblib')
                               for kind in ('bigru', 'ordered_hgb')})
        for path in split['eval']:
            inputs, labels = multisource_eval.view(path)
            periods = supervised.model_periods(inputs)
            for view in ce.VIEWS:
                target = OUT/f'{path.stem}_{view}.npz'
                if not target.exists():
                    tracking = inputs if view == 'clean' else tc.corrupt(inputs, spec['calibration'], ce.seed(path.stem, view))
                    probability = cb.predict_probabilities(tracking, models)
                    decoded = supervised.decode(probability, periods, float(inputs['fps']), lt.COST)
                    np.savez_compressed(target, probability=probability, baseline=decoded,
                        labels=labels, period=inputs['period'], timestamps=inputs['timestamps'],
                        model_periods=periods, fps=inputs['fps'])
                cells.append(dict(source=source, match=path.stem, view=view, file=target.name, sha256=sha256(target)))
                print('PREPARED', path.stem, view, flush=True)
    ce.write_json(OUT/'prediction_seal.json', dict(declaration_sha256=sha256(DECLARATION), cells=cells), exclusive=True)


def verify_seal():
    seal = json.loads((OUT/'prediction_seal.json').read_text())
    if seal['declaration_sha256'] != sha256(DECLARATION) or len(seal['cells']) != 15:
        raise ValueError('Prediction seal mismatch')
    for cell in seal['cells']:
        if sha256(OUT/cell['file']) != cell['sha256']:
            raise ValueError('Prediction cache changed')
    return seal


def diagnostics(data, prediction):
    # Insert ungraded separators exactly as the official external adapter does.
    inserts = np.flatnonzero(np.diff(data['model_periods']) != 0)+1
    gt = pm.label_codes(np.insert(data['labels'], inserts, 'X'))
    pred = pm.label_codes(np.insert(prediction, inserts, 'X'))
    runs = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pred, runs)
    gsp, psp = pm.possession_spells(gt, runs), pm.possession_spells(pred, runs)
    flags = pm.matched_gt_flags(pm.spell_overlaps(gsp, psp), len(gsp), .5)
    seconds = np.array([(b-a)/float(data['fps']) for a, b, _ in gsp])
    short = {}
    for name, keep in [('under_1s', seconds < 1), ('1_to_3s', (seconds >= 1) & (seconds < 3)), ('3s_plus', seconds >= 3)]:
        short[name] = dict(n=int(keep.sum()), matched=int(flags[keep].sum()),
                           recall=float(flags[keep].mean()) if keep.any() else None)
    return short


def score_cell(data, prediction):
    result = supervised.score(data, prediction)
    boundary = result['transitions'][max(1, round(.5*float(data['fps'])))].copy()
    for key in ('pairs', 'missed', 'spurious'):
        boundary.pop(key)
    for key in ('mean_latency', 'median_latency', 'mean_abs_latency', 'p90_abs_latency'):
        if boundary[key] is not None:
            boundary[key] /= float(data['fps'])
    keep = data['labels'] != 'X'
    return dict(spell_f1=result['spell_f1_mean'], macro_f1=result['macro_f1'],
        weighted_70_30=.7*result['spell_f1_mean']+.3*result['macro_f1'],
        n_graded=result['n_graded'], n_gt_spells=result['n_gt_spells'], n_pred_spells=result['n_pred_spells'],
        abstention_rate=float(np.mean(prediction[keep] == 'X')),
        changed_rate=float(np.mean(prediction[keep] != data['baseline'][keep])),
        d_precision=result['per_class']['D']['precision'], d_recall=result['per_class']['D']['recall'],
        d_f1=result['per_class']['D']['f1'], boundary_half_second=boundary,
        short_spells=diagnostics(data, prediction))


def evaluate():
    check()
    seal = verify_seal()
    old = json.loads((ROOT/'experiments/results/light_corruption_development.json').read_text())['rows']
    baseline_rows = []
    # Establish all 15 baseline cells before inspecting any new arm.
    for cell in seal['cells']:
        with np.load(OUT/cell['file'], allow_pickle=False) as f:
            data = dict(f)
        target = OUT/f"{cell['match']}_{cell['view']}_baseline.json"
        summary = json.loads(target.read_text()) if target.exists() else score_cell(data, data['baseline'])
        expected = next(r for r in old if r['fold'] == f"without_{cell['source']}" and
                        r['model'] == 'ball_team25' and r['match'] == cell['match'] and r['view'] == cell['view'])
        for key in ('spell_f1', 'macro_f1'):
            if abs(summary[key]-expected[key]) > 1e-12:
                raise ValueError(f"Baseline parity failed {cell['match']} {cell['view']} {key}: {summary[key]} != {expected[key]}")
        if not target.exists():
            ce.write_json(target, summary, exclusive=True)
        baseline_rows.append(dict(source=cell['source'], match=cell['match'], view=cell['view'], arm='baseline', **summary))
    print('BASELINE PARITY PASSED: 15/15', flush=True)
    rows = baseline_rows.copy()
    for cell in seal['cells']:
        with np.load(OUT/cell['file'], allow_pickle=False) as f:
            data = dict(f)
        for arm in ARMS[1:]:
            target = OUT/f"{cell['match']}_{cell['view']}_{arm}.json"
            if target.exists():
                summary = json.loads(target.read_text())
            else:
                mode, number = arm.split('_')
                pred = gate(data['probability'], data['baseline'], data['model_periods'], int(number)/100, mode)
                summary = score_cell(data, pred)
                ce.write_json(target, summary, exclusive=True)
            rows.append(dict(source=cell['source'], match=cell['match'], view=cell['view'], arm=arm, **summary))
        print('SCORED', cell['match'], cell['view'], flush=True)
    summarize(rows, seal)


def summarize(rows, seal):
    keys = ('spell_f1', 'macro_f1', 'weighted_70_30', 'abstention_rate', 'changed_rate', 'd_precision', 'd_recall', 'd_f1', 'n_pred_spells', 'n_gt_spells')
    summaries, paired = [], []
    for view in ('clean', 'stress'):
        means = {}
        for arm in ARMS:
            selected = [r for r in rows if r['arm'] == arm and (r['view'] == 'clean') == (view == 'clean')]
            matches = []
            for source, match in sorted({(r['source'], r['match']) for r in selected}):
                cells = [r for r in selected if r['match'] == match]
                matches.append(dict(eval_source=source, match=match, **{k: float(np.mean([r[k] for r in cells])) for k in keys}))
            means[arm] = matches
            per = {s: {k: float(np.mean([r[k] for r in matches if r['eval_source'] == s])) for k in keys} for s in ce.dead_transfer.SPLITS}
            summaries.append(dict(view=view, arm=arm, per_source=per, **{k: float(np.mean([r[k] for r in per.values()])) for k in keys}))
            if arm != 'baseline':
                paired.append(dict(view=view, arm=arm, **paired_source_bootstrap(means['baseline'], matches)))
    assert len(rows) == 135 and len({(r['match'], r['view'], r['arm']) for r in rows}) == 135
    for cell in seal['cells']:
        group = [r for r in rows if r['match'] == cell['match'] and r['view'] == cell['view']]
        assert len({(r['n_graded'], r['n_gt_spells']) for r in group}) == 1
    verdicts = {}
    for arm in ARMS[1:]:
        failures = []
        for view in ('clean', 'stress'):
            base = next(r for r in summaries if r['arm'] == 'baseline' and r['view'] == view)
            candidate = next(r for r in summaries if r['arm'] == arm and r['view'] == view)
            if candidate['spell_f1'] < base['spell_f1']:
                failures.append(f'{view}: spell decreases')
            if candidate['macro_f1']-base['macro_f1'] < -.005:
                failures.append(f'{view}: macro loses >.005')
            for s in base['per_source']:
                if candidate['per_source'][s]['spell_f1']-base['per_source'][s]['spell_f1'] < -.02:
                    failures.append(f'{view}: {s} spell loses >.02')
        verdicts[arm] = dict(guardrails_pass=not failures, failures=failures)
    ce.write_json(REPORT, dict(declaration_sha256=sha256(DECLARATION), prediction_seal_sha256=sha256(OUT/'prediction_seal.json'),
        baseline_parity_cells=15, evaluations=135, summaries=summaries, paired_deltas=paired,
        guardrails=verdicts, rows=rows, baseline_changed=False, organizer_scored=False), exclusive=True)
    for r in summaries:
        print('SUMMARY', r['view'], r['arm'], round(r['spell_f1'], 6), round(r['macro_f1'], 6), round(r['abstention_rate'], 4), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'prepare', 'evaluate'))
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        {'freeze': freeze, 'prepare': prepare, 'evaluate': evaluate}[args.action]()


if __name__ == '__main__':
    main()
