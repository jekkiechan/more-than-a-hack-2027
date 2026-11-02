"""GAR TEST no-ball bridge for the PCBAS player-only evidence.

Question: how much does removing every ball observation cost the exact adopted
package on GAR TEST, where dense silver H/A/D labels exist, and is the PCBAS
player-only pass agreement (68.80%) consistent with GAR under the same input
regime? No fitting, threshold search, organizer-label access or promotion.
See GAR_NOBALL_BRIDGE_V1.md for the frozen protocol and readout.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'gar_noball_bridge_v1'
OUT = ROOT / 'data/experiments' / STUDY
RECORDS = OUT / 'records'
DECLARATION = ROOT / 'experiments/results' / f'{STUDY}_declaration.json'
RESULT = ROOT / 'experiments/results' / f'{STUDY}.json'
PRIOR_RESULT = ROOT / 'experiments/results/gar_deployment_eval_v2.json'
VIEWS = ('clean', 'noball')
ARMS = ('original_full', 'calibrated')
MODE = 'phase_avg'
# Touch-run end codes from soccernet_dense_labels.END_CODES -> PCBAS-like classes.
ANCHOR_CLASSES = {6: 'drive', 1: 'pass', 5: 'cross', 4: 'shot'}
VISIBILITY_BINS = ((0, 5, '0-5'), (6, 11, '6-11'), (12, None, '12+'))
# PCBAS_EXPANDED_ANCHOR_V2, twelve TRAIN games. GAR clips centre on passes/crosses/shots, so
# ball carries are rarely annotated there; pass is the primary bridge class, drive secondary.
PCBAS_EQUAL_GAME = dict(drive=0.7157, pass_=0.6880, cross=0.6689, shot=0.5879)
PRIMARY_CLASS = 'pass'
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = 20261102
SOURCES = ('experiments/gar_noball_bridge_v1.py', 'experiments/gar_deployment_eval_v2.py',
           'experiments/gar_heldout_eval_v1.py', 'experiments/soccernet_dense_labels.py',
           'experiments/tests/test_gar_noball_bridge_v1.py')


def digest(path):
    from experiments.gar_deployment_eval_v2 import digest as _digest
    return _digest(path)


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def remove_ball(inputs):
    """Every ball coordinate missing and unobserved; players, time and breaks unchanged."""
    out = dict(inputs)
    out['ball'] = np.full_like(np.asarray(inputs['ball'], dtype=float), np.nan)
    out['ball_observed'] = np.zeros(len(out['ball']), dtype=bool)
    return out


def anchors_from_records(records):
    """PCBAS-like action anchors from controlled GAR touch runs, on the game-level frame axis.

    Anchor frame is the last native frame of the run (release for pass/cross/shot,
    carry end for ball carries). Uncontrolled, censored and unknown-team runs are skipped.
    """
    rows, offset = [], 0
    for record in records:
        n = int(record['native_n'])
        columns = list(record['touch_run_columns'])
        table = np.asarray(record['touch_runs'])
        if table.size:
            table = table.reshape(-1, len(columns))
            for run in table:
                run = dict(zip(columns, (int(v) for v in run)))
                if run['end_code'] not in ANCHOR_CLASSES or not run['controlled'] or run['censored']:
                    continue
                if run['team'] not in (0, 1) or not 0 <= run['end'] < n:
                    continue
                rows.append(dict(index=offset + run['end'], team=run['team'],
                                 cls=ANCHOR_CLASSES[run['end_code']]))
        offset += n
    return rows


def anchor_metrics(anchors, prediction, visible):
    """Exact decoded-team agreement; D/X count wrong. Visibility strata pool all classes."""
    prediction = np.asarray(prediction).astype(str)
    out = {}
    for name in ANCHOR_CLASSES.values():
        sel = [a for a in anchors if a['cls'] == name]
        states = [prediction[a['index']] for a in sel]
        out[name] = dict(n=len(sel), correct=int(sum(s == 'HA'[a['team']] for a, s in zip(sel, states))),
                         predicted_d=int(sum(s == 'D' for s in states)))
    strata = {}
    for low, high, label in VISIBILITY_BINS:
        sel = [a for a in anchors if visible[a['index']] >= low and (high is None or visible[a['index']] <= high)]
        strata[label] = dict(n=len(sel), correct=int(sum(prediction[a['index']] == 'HA'[a['team']] for a in sel)),
                             predicted_d=int(sum(prediction[a['index']] == 'D' for a in sel)))
    return dict(classes=out, visibility=strata)


def live_metrics(labels, prediction):
    labels = np.asarray(labels).astype(str)
    prediction = np.asarray(prediction).astype(str)
    live = np.isin(labels, ('H', 'A'))
    dead = labels == 'D'
    return dict(live_frames=int(live.sum()), live_team_correct=int((prediction[live] == labels[live]).sum()),
                live_predicted_d=int((prediction[live] == 'D').sum()),
                dead_frames=int(dead.sum()), dead_predicted_d=int((prediction[dead] == 'D').sum()))


def declare():
    if DECLARATION.exists():
        raise FileExistsError(DECLARATION)
    from experiments import gar_deployment_eval_v2 as v2
    prior = json.loads(v2.DECLARATION.read_text())
    packages = {arm: dict(manifest=str(v2.MANIFESTS[arm].relative_to(ROOT)),
                          manifest_sha256=digest(v2.MANIFESTS[arm])) for arm in ARMS}
    for arm in ARMS:
        v2.checkpoint_package(arm)  # verifies checkpoint hashes
    write_once(DECLARATION, dict(
        study=STUDY, status='frozen', created_utc=datetime.now(timezone.utc).isoformat(),
        question='On GAR TEST, how much does removing every ball observation cost the exact adopted package, '
                 'and is PCBAS player-only pass agreement consistent with GAR under the same regime?',
        games=prior['match_ids']['test'], split='test', exposure='GAR TEST was scored once by gar_deployment_eval_v2 '
        '(clean/v2). No model was selected on it; this adds a new input view only.',
        packages=packages, arms=list(ARMS), mode='phase_avg (six GAR sampling phases, frozen decoder)',
        views=dict(clean='unchanged tracking', noball='all ball coordinates NaN and ball_observed False; players, '
                   'masks, periods and timestamps unchanged'),
        parity='clean original_full and calibrated phase_avg per-game spell/macro F1 must equal the saved '
               'gar_deployment_eval_v2 TEST values within 1e-9 before any noball readout',
        metrics=['official-scorer spell_f1/macro_f1/d_f1 on silver labels', 'live-frame team agreement (silver H/A '
                 'frames; D/X wrong)', 'predicted-D rate on silver live frames', 'PCBAS-like anchor agreement at '
                 'controlled touch-run end frames: BC->drive, PA->pass, CR->cross, SH->shot',
                 'anchor agreement by visible players 0-5/6-11/12+'],
        primary='equal-game noball pass-anchor agreement of original_full, compared descriptively with PCBAS '
                f"{PCBAS_EQUAL_GAME['pass_']}; and clean-minus-noball spell F1 with paired whole-game bootstrap",
        pcbas_reference=PCBAS_EQUAL_GAME,
        decision_rule=dict(
            consistent='GAR noball pass agreement within 5 points of PCBAS: no evidence of a 2022->2023/24 cohort '
                       'penalty in player-only team attribution; PCBAS team-head training is not prioritized',
            gap='GAR noball pass agreement more than 5 points above PCBAS: cohort/visibility gap; an H/A-only '
                'PCBAS auxiliary fine-tune becomes the recommended training experiment',
            caveat='GAR and PCBAS anchors are defined by different annotation schemes; this is a bridge, not an '
                   'equivalence test'),
        bootstrap=dict(draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED, unit='whole game'),
        fitting=False, threshold_search=False, organizer_reference_access=False, promotion=False,
        source_sha256={rel: digest(ROOT / rel) for rel in SOURCES}))
    print('DECLARED', DECLARATION)


def check_seal():
    from experiments import gar_deployment_eval_v2 as v2
    d = json.loads(DECLARATION.read_text())
    for rel, sha in d['source_sha256'].items():
        if digest(ROOT / rel) != sha:
            raise ValueError('Declared source changed: ' + rel)
    for arm, entry in d['packages'].items():
        if digest(ROOT / entry['manifest']) != entry['manifest_sha256']:
            raise ValueError('Package changed: ' + arm)
    return d


def prepare(workers=3):
    check_seal()
    from experiments import soccernet_dense_labels as dense
    if not (RECORDS / 'test' / 'manifest.json').is_file():
        dense.prepare_split('test', output=RECORDS, workers=workers)


def run_game(game_id):
    from experiments import gar_deployment_eval_v2 as v2
    from experiments import gar_heldout_eval_v1 as previous
    check_seal()
    target = OUT / 'games' / f'{game_id}.json'
    if target.exists():
        return str(target)
    started = time.monotonic()
    records = previous.load_game_records('test', game_id, RECORDS)
    captured = previous.capture_inputs('test', game_id)
    inputs, labels, saved_x, names = previous.build_game(records, captured)
    sampled_fps = float(records[0]['fps'])
    anchors = anchors_from_records(records)
    for a in anchors:  # anchor team must agree with the silver label where the label is a team
        if labels[a['index']] in ('H', 'A') and labels[a['index']] != 'HA'[a['team']]:
            a['silver_conflict'] = True
    visible = np.asarray(inputs['player_observed']).sum(axis=1)
    out, predictions = {}, {}
    for view in VIEWS:
        data = inputs if view == 'clean' else remove_ball(inputs)
        phases = previous.phase_set(data, names, sampled_fps)
        if view == 'clean' and not np.array_equal(phases[0]['x'], saved_x, equal_nan=True):
            raise ValueError('Phase-0 features differ from the dense records')
        native = v2.audit.tc.inputs_only(data)
        out[view] = {}
        for arm in ARMS:
            p = np.mean([v2.native_probability(r, native, arm) for r in phases], axis=0)
            metrics, pred = previous.score_game(labels, inputs['period'], inputs['timestamps'],
                                                float(inputs['fps']), p)
            pred = np.asarray(pred).astype(str)
            out[view][arm] = dict(scorer=metrics, live=live_metrics(labels, pred),
                                  anchors=anchor_metrics(anchors, pred, visible))
            predictions[f'{view}__{arm}'] = pred.astype('S1')
        print('VIEW', game_id, view, round(time.monotonic() - started, 1), flush=True)
    path = target.with_suffix('.npz')
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels.astype('S1'), visible=visible, **predictions)
    write_once(target, dict(study=STUDY, game=game_id, declaration_sha256=digest(DECLARATION),
        native_frames=len(labels), graded_frames=int((labels != 'X').sum()),
        ball_observed_fraction_clean=float(np.mean(inputs['ball_observed'])),
        visible_player_frame_mean=float(visible.mean()), anchors=len(anchors),
        anchor_silver_conflicts=int(sum(a.get('silver_conflict', False) for a in anchors)),
        scores=out, prediction_sha256=digest(path), seconds=time.monotonic() - started))
    return str(target)


def games(workers=2):
    d = check_seal()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        for future in as_completed([pool.submit(run_game, g) for g in d['games']]):
            print('DONE', future.result(), flush=True)


def _ratio(cells, key='correct'):
    n = sum(c['n'] for c in cells)
    return (sum(c[key] for c in cells) / n) if n else float('nan')


def game_table(rows, view, arm):
    table = {}
    for r in rows:
        s = r['scores'][view][arm]
        live = s['live']
        table[r['game']] = dict(
            spell_f1=s['scorer']['spell_f1'], macro_f1=s['scorer']['macro_f1'], d_f1=s['scorer'].get('d_f1'),
            live_team_agreement=live['live_team_correct'] / live['live_frames'],
            live_predicted_d=live['live_predicted_d'] / live['live_frames'],
            dead_predicted_d=live['dead_predicted_d'] / max(live['dead_frames'], 1),
            **{f'{c}_agreement': _ratio([s['anchors']['classes'][c]]) for c in ANCHOR_CLASSES.values()},
            **{f'{c}_predicted_d': _ratio([s['anchors']['classes'][c]], 'predicted_d') for c in ANCHOR_CLASSES.values()})
    return table


def bootstrap_mean_ci(values, draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def report():
    d = check_seal()
    rows = [json.loads((OUT / 'games' / f'{g}.json').read_text()) for g in d['games']]
    prior = json.loads(PRIOR_RESULT.read_text())['splits']['test']['scores']['clean']
    parity = {}
    for arm in ARMS:
        diffs = [abs(r['scores']['clean'][arm]['scorer'][m] - prior[arm][MODE]['per_game'][r['game']][m])
                 for r in rows for m in ('spell_f1', 'macro_f1')]
        parity[arm] = max(diffs)
        if parity[arm] > 1e-9:
            raise ValueError(f'Clean parity with gar_deployment_eval_v2 failed for {arm}: {parity[arm]}')
    result = dict(study=STUDY, declaration_sha256=digest(DECLARATION), games=d['games'],
                  clean_parity_max_abs_difference=parity, views={}, deltas={}, pooled_anchors={},
                  context=dict(ball_observed_fraction_clean=float(np.mean([r['ball_observed_fraction_clean'] for r in rows])),
                               visible_player_frame_mean=float(np.mean([r['visible_player_frame_mean'] for r in rows])),
                               anchors=int(sum(r['anchors'] for r in rows)),
                               anchor_silver_conflicts=int(sum(r['anchor_silver_conflicts'] for r in rows))),
                  pcbas_equal_game_reference=PCBAS_EQUAL_GAME,
                  fitting=False, organizer_reference_access=False, promotion=False)
    for view in VIEWS:
        result['views'][view] = {}
        for arm in ARMS:
            table = game_table(rows, view, arm)
            keys = list(next(iter(table.values())))
            result['views'][view][arm] = dict(per_game=table, mean={k: float(np.nanmean([t[k] for t in table.values()])) for k in keys},
                ci95={k: bootstrap_mean_ci([t[k] for t in table.values()]) for k in ('spell_f1', 'pass_agreement', 'drive_agreement', 'live_team_agreement')})
            cells = [r['scores'][view][arm]['anchors'] for r in rows]
            result['pooled_anchors'].setdefault(view, {})[arm] = dict(
                classes={c: dict(n=sum(x['classes'][c]['n'] for x in cells), agreement=_ratio([x['classes'][c] for x in cells]),
                                 predicted_d=_ratio([x['classes'][c] for x in cells], 'predicted_d')) for c in ANCHOR_CLASSES.values()},
                visibility={b: dict(n=sum(x['visibility'][b]['n'] for x in cells), agreement=_ratio([x['visibility'][b] for x in cells]),
                                    predicted_d=_ratio([x['visibility'][b] for x in cells], 'predicted_d')) for _, _, b in VISIBILITY_BINS})
    for arm in ARMS:
        clean, noball = result['views']['clean'][arm]['per_game'], result['views']['noball'][arm]['per_game']
        result['deltas'][arm] = {k: dict(mean=float(np.mean([clean[g][k] - noball[g][k] for g in d['games']])),
                                         ci95=bootstrap_mean_ci([clean[g][k] - noball[g][k] for g in d['games']]))
                                 for k in ('spell_f1', 'macro_f1', 'live_team_agreement', 'drive_agreement', 'pass_agreement')}
    nb = result['views']['noball']['original_full']['mean']['pass_agreement']
    ref = PCBAS_EQUAL_GAME['pass_']
    result['bridge'] = dict(primary_class=PRIMARY_CLASS, gar_noball_equal_game=nb, pcbas_equal_game=ref,
                            difference_points=100 * (nb - ref),
                            outcome='consistent' if abs(nb - ref) <= 0.05 else ('gap' if nb > ref else 'gar_lower'))
    write_once(RESULT, result)
    print(json.dumps(dict(bridge=result['bridge'], deltas=result['deltas']), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('declare', 'prepare', 'games', 'game', 'report'))
    parser.add_argument('--game')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.stage == 'declare':
        declare()
    elif args.stage == 'prepare':
        prepare(args.workers)
    elif args.stage == 'games':
        games(args.workers)
    elif args.stage == 'game':
        print(run_game(args.game))
    else:
        report()


if __name__ == '__main__':
    main()
