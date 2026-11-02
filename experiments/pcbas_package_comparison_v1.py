"""Paired package comparison on unscored PCBAS reserve games (player-only regime).

Compares the adopted original_full phase-average package with the runner-up
calibrated dense package plus the same five-phase averaging, at sparse PCBAS
action anchors with all ball input missing. Half of the 36 previously unscored
TRAIN reserve games are used; the other half stay sealed. No fitting, threshold
or offset search, organizer-label access or promotion. See
PCBAS_PACKAGE_COMPARISON_V1.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'pcbas_package_comparison_v1'
SALT = STUDY
DATA = ROOT / 'data/independent/sn_pcbas_2026'
H5 = DATA / 'train_tactical_data.h5'
EVENTS = DATA / 'playbyplay_train.json'
OUT = ROOT / 'data/experiments' / STUDY
DECLARATION = ROOT / 'experiments/results' / f'{STUDY}_declaration.json'
RESULT = ROOT / 'experiments/results' / f'{STUDY}_result.json'
PRIOR_DECLARATION = ROOT / 'experiments/results/pcbas_expanded_anchor_v2_declaration.json'
MANIFESTS = {
    'original_full_phase': ROOT / 'experiments/baseline_manifest.json',
    'calibrated_phase': ROOT / 'experiments/baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json',
}
ARMS = tuple(MANIFESTS)
N_COMPARE = 18
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261103
KNOWN_VAL_GAMES = {'game_18', 'game_24', 'game_47'}
CLASSES = {1: 'drive', 2: 'pass', 3: 'cross', 5: 'shot'}
SOURCES = ('experiments/pcbas_package_comparison_v1.py', 'experiments/pcbas_event_anchor_v1.py',
           'experiments/pcbas_expanded_anchor_v2.py', 'experiments/feed_like_v2_public.py',
           'experiments/original_full_phase_baseline.py', 'experiments/calibrated_dense_baseline.py',
           'experiments/sn_dense_baseline.py', 'experiments/feed_spell_push.py',
           'experiments/tests/test_pcbas_package_comparison_v1.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order='C')).hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def rank(game_ids, salt=SALT):
    return sorted(sorted(set(game_ids)), key=lambda g: hashlib.sha256(f'{salt}:{g}'.encode()).hexdigest())


def split_reserve(reserve):
    """Label-free deterministic split of the unscored reserve into compare/sealed halves."""
    reserve = list(reserve)
    if len(set(reserve)) != len(reserve) or len(reserve) != 2 * N_COMPARE or set(reserve) & KNOWN_VAL_GAMES:
        raise ValueError('Expected 36 unscored TRAIN reserve games')
    ranked = rank(reserve)
    return ranked[:N_COMPARE], ranked[N_COMPARE:]


# ---------------------------------------------------------------- inference

def load_models():
    from experiments import current_baseline
    return {arm: (json.loads(path.read_text()), current_baseline.load_models(manifest_path=path))
            for arm, path in MANIFESTS.items()}


def member_phases(inputs, models, manifest):
    """Per-member, per-phase native probabilities: array (members, phases, frames, 3)."""
    from experiments import feed_like_v2_public as fl, sn_dense_baseline, tracking_corruption
    from experiments.feed_like_v2_public import audit, live_gap_experiment, old, sn
    sn_dense_baseline.validate_models(models, manifest['architecture'])
    clean = tracking_corruption.inputs_only(inputs)
    if np.isfinite(clean['ball']).any() or clean['ball_observed'].any():
        raise ValueError('This study requires completely missing ball input')
    placeholder = np.full(len(clean['ball']), 'X', dtype='U1')
    base = old.inference_record(live_gap_experiment.feature_view(clean, placeholder, 'competition', 'BAR-ATH')['joint'])
    records = audit.phase_records(clean, base)
    if len(records) != 5:
        raise ValueError('Expected five 5 Hz phases at 25 Hz')
    out = []
    for m, t in sn_dense_baseline.TEACHERS.items():
        out.append([old.upsample(sn.predict(dict(r, names=list(r['names'])), models[m], models[t]), r, clean)
                    for r in records])
    return np.asarray(out)


def decode_arm(arm, phases, inputs):
    """original_full: pool members then phases (deployed order). calibrated: phase-average each
    member, then the calibrated decoder normalizes members before pooling."""
    from experiments import calibrated_dense_baseline as calibration, feed_spell_push, supervised
    periods, fps = supervised.model_periods(inputs), float(inputs['fps'])
    if arm == 'original_full_phase':
        native = phases.mean(axis=0).mean(axis=0)[None]
    elif arm == 'calibrated_phase':
        native = phases.mean(axis=1)
    elif arm == 'calibrated_phase0':
        native = phases[:, 0]
    else:
        raise ValueError(arm)
    return feed_spell_push.decode(native, periods, fps, calibration.CONFIG)


def predict_half(inputs, runtime):
    out = {}
    for arm in ARMS:
        manifest, models = runtime[arm]
        phases = member_phases(inputs, models, manifest)
        out[arm] = decode_arm(arm, phases, inputs)
        if arm == 'calibrated_phase':
            out['calibrated_phase0'] = decode_arm('calibrated_phase0', phases, inputs)
    return out


# ---------------------------------------------------------------- stages

def declare():
    if DECLARATION.exists():
        raise FileExistsError(DECLARATION)
    import h5py
    prior = json.loads(PRIOR_DECLARATION.read_text())
    prior_selected = [r['game_id'] for r in prior['cohort']['selected']]
    reserve = [r['game_id'] for r in prior['cohort']['reserve']]
    compare, sealed = split_reserve(reserve)
    keys = [f'{g}_H{h}' for g in compare for h in (1, 2)]
    runtime = load_models()  # verifies all eight checkpoint hashes
    feature_sha = {}
    with h5py.File(H5, 'r') as source:
        for key in keys:
            feature_sha[key] = array_sha(source[key][:, :13])  # action-class column never read
    write_once(DECLARATION, dict(
        study=STUDY, status='frozen', created_utc=datetime.now(timezone.utc).isoformat(),
        question='Under the player-only PCBAS regime, does the runner-up calibrated dense package with five-phase '
                 'averaging agree with the acting team more often than the adopted original_full phase package?',
        arms=dict(original_full_phase='adopted package, exact deployed recipe',
                  calibrated_phase='calibrated dense checkpoints; per-member five-phase mean; frozen f062 decoder',
                  calibrated_phase0='secondary: the archived calibrated package exactly as deployed'),
        primary='paired equal-game Drive agreement difference calibrated_phase minus original_full_phase; '
                f'{BOOTSTRAPS} whole-game bootstrap draws, seed {BOOTSTRAP_SEED}',
        decision_rule='Informative only: this view has no ball. A paired 95% interval excluding zero is reported as '
                      'player-only evidence and weighed with GAR TEST and original feed_like_v2; it cannot by itself '
                      'change the adopted package.',
        cohort=dict(compare=compare, sealed=sealed, prior_scored=prior_selected + sorted(KNOWN_VAL_GAMES),
                    rule=f'sha256("{SALT}:<game>") ranking of the 36 unscored reserve games; first {N_COMPARE} compared, '
                         'rest sealed for future confirmation'),
        halves=keys, target_free_feature_sha256=feature_sha,
        input_sha256={str(H5.relative_to(ROOT)): sha(H5), str(EVENTS.relative_to(ROOT)): sha(EVENTS)},
        manifest_sha256={arm: sha(path) for arm, path in MANIFESTS.items()},
        model_sha256={arm: runtime[arm][0]['models_sha256'] for arm in ARMS},
        anchors='Frozen pcbas_event_anchor_v1.select_anchors rules; D/X wrong; replay excluded',
        adapter='Frozen pcbas_event_anchor_v1.adapt_rows: standardized 105x68, ROI visibility mask, no ball',
        parity='First 750 frames of the first compared half: original_full_phase must equal '
               'original_full_phase_baseline.predict_labels and calibrated_phase0 must equal '
               'calibrated_dense_baseline.predict_labels exactly',
        fitting=False, threshold_search=False, organizer_reference_access=False, promotion=False,
        source_sha256={rel: sha(ROOT / rel) for rel in SOURCES}))
    print('DECLARED', len(keys), 'halves;', 'compare', compare)


def verify():
    d = json.loads(DECLARATION.read_text())
    for rel, value in d['source_sha256'].items():
        if sha(ROOT / rel) != value:
            raise ValueError('Declared source changed: ' + rel)
    for arm, path in MANIFESTS.items():
        if sha(path) != d['manifest_sha256'][arm]:
            raise ValueError('Manifest changed: ' + arm)
    return d


def _load(source, key, d):
    from experiments.pcbas_event_anchor_v1 import adapt_rows
    rows = source[key][:, :13]
    if array_sha(rows) != d['target_free_feature_sha256'][key]:
        raise ValueError('Input rows changed: ' + key)
    return adapt_rows(rows, int(key[-1]))


def parity():
    d = verify()
    target = OUT / 'parity_receipt.json'
    if target.exists():
        print('ALREADY', target)
        return
    import h5py
    from experiments import calibrated_dense_baseline, original_full_phase_baseline, tracking_corruption
    runtime = load_models()
    key = d['halves'][0]
    with h5py.File(H5, 'r') as source:
        inputs, frames, _ = _load(source, key, d)
    cut = {k: (v[:750] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(frames) else v) for k, v in inputs.items()}
    ours = predict_half(cut, runtime)
    clean = tracking_corruption.inputs_only(cut)
    direct_a = original_full_phase_baseline.predict_labels(clean, runtime['original_full_phase'][1], runtime['original_full_phase'][0])
    direct_c = calibrated_dense_baseline.predict_labels(clean, runtime['calibrated_phase'][1], runtime['calibrated_phase'][0])
    ok = bool(np.array_equal(ours['original_full_phase'], direct_a) and np.array_equal(ours['calibrated_phase0'], direct_c))
    if not ok:
        raise ValueError('Parity failed')
    write_once(target, dict(declaration_sha256=sha(DECLARATION), half=key, frames=750, exact_equal=ok))
    print('PARITY OK')


def predict(key):
    d = verify()
    if not (OUT / 'parity_receipt.json').is_file():
        raise ValueError('Run parity first')
    if key not in d['halves']:
        raise ValueError('Unselected half')
    out = OUT / f'{key}.npz'
    if out.exists():
        print('ALREADY', key)
        return
    import h5py
    runtime = load_models()
    with h5py.File(H5, 'r') as source:
        inputs, frames, audit = _load(source, key, d)
    started = time.monotonic()
    preds = predict_half(inputs, runtime)
    tmp = out.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, frames=frames, visible_players=inputs['player_observed'].sum(axis=1),
                        **{arm: np.asarray(p).astype('U1') for arm, p in preds.items()})
    os.link(tmp, out)
    tmp.unlink()
    write_once(OUT / f'{key}.receipt.json', dict(half=key, declaration_sha256=sha(DECLARATION), output_sha256=sha(out),
               frames=len(frames), input_audit=audit, seconds=time.monotonic() - started,
               target_events_read=False))
    print('PREDICTED', key, len(frames), round(time.monotonic() - started, 1), flush=True)


def predict_all():
    d = verify()
    for key in d['halves']:
        predict(key)


def seal():
    d = verify()
    halves = {}
    for key in d['halves']:
        receipt = json.loads((OUT / f'{key}.receipt.json').read_text())
        if receipt['output_sha256'] != sha(OUT / f'{key}.npz'):
            raise ValueError('Output changed: ' + key)
        halves[key] = receipt['output_sha256']
    write_once(OUT / 'prediction_seal.json', dict(declaration_sha256=sha(DECLARATION), halves=halves,
                                                  target_events_read=False))
    print('SEALED', len(halves))


def paired_bootstrap(a, b, draws=BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(draws, len(a)))
    diff = (b - a)[idx].mean(axis=1)
    return dict(mean_difference=float((b - a).mean()), ci95=[float(x) for x in np.quantile(diff, [0.025, 0.975])],
                games_b_better=int((b > a).sum()), games_a_better=int((b < a).sum()), games=len(a))


def score():
    d = verify()
    seal_value = json.loads((OUT / 'prediction_seal.json').read_text())
    for key, value in seal_value['halves'].items():
        if sha(OUT / f'{key}.npz') != value:
            raise ValueError('Sealed output changed: ' + key)
    write_once(OUT / 'score_gate_claimed.json', dict(declaration_sha256=sha(DECLARATION),
                                                      prediction_seal_sha256=sha(OUT / 'prediction_seal.json')))
    if sha(EVENTS) != d['input_sha256'][str(EVENTS.relative_to(ROOT))]:
        raise ValueError('Event file changed')
    from experiments.pcbas_event_anchor_v1 import select_anchors
    truth = json.loads(EVENTS.read_text())
    arms = list(ARMS) + ['calibrated_phase0']
    per_game = {arm: {} for arm in arms}
    excluded = Counter()
    for key in d['halves']:
        game = key.rsplit('_H', 1)[0]
        with np.load(OUT / f'{key}.npz', allow_pickle=False) as f:
            frames = f['frames']
            preds = {arm: f[arm].astype(str) for arm in arms}
        anchors, ex = select_anchors(truth['events'][key], frames)
        excluded.update(ex)
        for arm in arms:
            cell = per_game[arm].setdefault(game, {c: dict(n=0, correct=0, predicted_d=0) for c in CLASSES.values()})
            for a in anchors:
                c = cell[CLASSES[a['class_id']]]
                state = preds[arm][a['index']]
                c['n'] += 1
                c['correct'] += int(state == 'HA'[a['team']])
                c['predicted_d'] += int(state == 'D')
    games = d['cohort']['compare']
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for c in CLASSES.values():
            cells = [per_game[arm][g][c] for g in games]
            summary[arm][c] = dict(
                equal_game_agreement=float(np.mean([x['correct'] / x['n'] for x in cells if x['n']])),
                pooled_agreement=sum(x['correct'] for x in cells) / sum(x['n'] for x in cells),
                pooled_predicted_d=sum(x['predicted_d'] for x in cells) / sum(x['n'] for x in cells),
                anchors=sum(x['n'] for x in cells))
    comparisons = {}
    base = 'original_full_phase'
    for other in ('calibrated_phase', 'calibrated_phase0'):
        comparisons[f'{other}_minus_{base}'] = {
            c: paired_bootstrap([per_game[base][g][c]['correct'] / per_game[base][g][c]['n'] for g in games if per_game[base][g][c]['n']],
                                [per_game[other][g][c]['correct'] / per_game[other][g][c]['n'] for g in games if per_game[base][g][c]['n']])
            for c in CLASSES.values()}
    adopted = [per_game[base][g]['drive']['correct'] / per_game[base][g]['drive']['n'] for g in games]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.asarray(adopted)[rng.integers(0, len(adopted), size=(BOOTSTRAPS, len(adopted)))].mean(axis=1)
    result = dict(study=STUDY, declaration_sha256=sha(DECLARATION), games=games, summary=summary,
                  comparisons=comparisons, per_game=per_game, excluded=dict(excluded),
                  adopted_drive_equal_game_ci95=[float(x) for x in np.quantile(boot, [0.025, 0.975])],
                  ball_missing_fraction=1.0, fitting=False, promotion=False,
                  interpretation='Player-only sparse-anchor evidence on 18 previously unscored PCBAS games; '
                                 '18 further reserve games remain sealed.')
    write_once(RESULT, result)
    print(json.dumps(dict(summary=summary, comparisons={k: v['drive'] for k, v in comparisons.items()},
                          adopted_ci=result['adopted_drive_equal_game_ci95']), indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('declare', 'parity', 'predict', 'predict-all', 'seal', 'score'))
    parser.add_argument('--half')
    args = parser.parse_args()
    {'declare': declare, 'parity': parity, 'predict-all': predict_all, 'seal': seal, 'score': score}.get(
        args.stage, lambda: predict(args.half))()


if __name__ == '__main__':
    main()
