"""QA aggregates for the dense SoccerNet labels (writes only small aggregate JSON)."""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments import soccernet_dense_labels as dl

ACTIONS_CHECKED = ('PLAYER SUCCESSFUL TACKLE', 'PASS', 'HEADER', 'CROSS', 'SHOT',
                   'THROW IN', 'FREE KICK', 'GOAL', 'HIGH PASS')
NATIVE_CODES = {0: 'H', 1: 'A', 2: 'D', -1: 'unknown'}
TACKLE_OFFSETS = (-15, 0, 15, 30, 60)       # native frames after the metadata centre


def _runs(labels: np.ndarray, value: int):
    """Maximal runs (start, stop, touches_edge) of one label value."""
    mask = labels == value
    if not mask.any():
        return []
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return [(int(a), int(b), a == 0 or b == len(labels)) for a, b in zip(edges[::2], edges[1::2])]


def _quantiles(values, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    if not len(values):
        return {}
    return {f'p{int(q * 100):02d}': float(np.quantile(values, q)) for q in qs}


def split_report(split: str, output=dl.DEFAULT_OUT, seed=0) -> dict:
    records, payloads = dl.load_dense_records(split, output, with_payload=True)
    report = {'split': split, 'games': len(payloads), 'records': len(records)}
    fps = {p['fps'] for p in payloads}
    native_total = sum(len(r['full_y']) for r in records)
    mean_fps = float(np.mean([r['native_fps'] for r in records]))
    report['native_frames'] = int(native_total)
    report['hours'] = native_total / mean_fps / 3600.0
    report['sampled_rows'] = int(sum(len(r['y']) for r in records))
    class_counts, prov_counts = Counter(), Counter()
    prov_by_class = defaultdict(Counter)
    sampled_counts = Counter()
    for r in records:
        for c, k in zip(*np.unique(r['full_y'], return_counts=True)):
            class_counts[NATIVE_CODES[int(c)]] += int(k)
        for c, k in zip(*np.unique(r['full_prov'], return_counts=True)):
            prov_counts[dl.PROVENANCE_NAMES[int(c)]] += int(k)
        for c, k in zip(*np.unique(r['y'], return_counts=True)):
            sampled_counts[NATIVE_CODES[int(c)]] += int(k)
        for cls in (0, 1, 2):
            sel = r['full_y'] == cls
            for c, k in zip(*np.unique(r['full_prov'][sel], return_counts=True)):
                prov_by_class[NATIVE_CODES[cls]][dl.PROVENANCE_NAMES[int(c)]] += int(k)
    report['label_share_native'] = {k: v / native_total for k, v in sorted(class_counts.items())}
    report['label_share_sampled'] = {k: v / sum(sampled_counts.values())
                                     for k, v in sorted(sampled_counts.items())}
    report['provenance_share_native'] = {k: v / native_total for k, v in sorted(prov_counts.items())}
    report['provenance_by_class_share'] = {c: {k: v / sum(d.values()) for k, v in sorted(d.items())}
                                           for c, d in prov_by_class.items()}
    # Switches (adjacent known H/A labels differing), and all class transitions.
    switches = 0
    transitions = Counter()
    switches_sampled = 0
    for r in records:
        fy = r['full_y']
        both = (fy[1:] >= 0) & (fy[:-1] >= 0) & (fy[1:] != fy[:-1])
        for a, b in zip(fy[:-1][both], fy[1:][both]):
            transitions[f'{NATIVE_CODES[int(a)]}->{NATIVE_CODES[int(b)]}'] += 1
        switches += int((both & (fy[1:] < 2) & (fy[:-1] < 2)).sum())
        y = r['y']
        sw = (y[1:] >= 0) & (y[:-1] >= 0) & (y[1:] != y[:-1])
        switches_sampled += int((sw & (y[1:] < 2) & (y[:-1] < 2)).sum())
    report['ha_switches_native'] = switches
    report['ha_switches_per_hour'] = switches / report['hours']
    report['ha_switches_sampled_adjacent'] = switches_sampled
    report['class_transitions_native'] = dict(transitions)
    # Spell lengths (contiguous same-label H or A runs, native frames -> seconds).
    spells, complete = [], []
    for r in records:
        f = float(r['native_fps'])
        for cls in (0, 1):
            for a, b, edge in _runs(r['full_y'], cls):
                seconds = (b - a) / f
                spells.append(seconds)
                if not edge:
                    complete.append(seconds)
    spells, complete = np.asarray(spells), np.asarray(complete)
    report['spells'] = {
        'n_all': int(len(spells)), 'n_complete': int(len(complete)),
        'all_quantiles_s': _quantiles(spells), 'complete_quantiles_s': _quantiles(complete),
        'all_share_under_3s': float((spells < 3).mean()) if len(spells) else None,
        'complete_share_under_3s': float((complete < 3).mean()) if len(complete) else None,
        'complete_share_under_1s': float((complete < 1).mean()) if len(complete) else None,
        'complete_share_over_10s': float((complete > 10).mean()) if len(complete) else None,
        'complete_mean_s': float(complete.mean()) if len(complete) else None,
    }
    # Spells under the 'literal' (in-run) rule, for sensitivity.
    lit = []
    for r in records:
        f = float(r['native_fps'])
        for cls in (0, 1):
            lit += [(b - a) / f for a, b, edge in _runs(r['full_y_literal'], cls) if not edge]
    lit = np.asarray(lit)
    report['spells_literal_rule'] = {
        'n_complete': int(len(lit)), 'complete_quantiles_s': _quantiles(lit),
        'complete_share_under_3s': float((lit < 3).mean()) if len(lit) else None,
        'ha_switches_native': int(sum(
            int(((r['full_y_literal'][1:] < 2) & (r['full_y_literal'][:-1] < 2) &
                 (r['full_y_literal'][1:] >= 0) & (r['full_y_literal'][:-1] >= 0) &
                 (r['full_y_literal'][1:] != r['full_y_literal'][:-1])).sum()) for r in records))}
    # Touch-run statistics.
    runs_all = np.concatenate([r['touch_runs'] for r in records if len(r['touch_runs'])])
    lengths = (runs_all[:, 1] - runs_all[:, 0] + 1)
    report['touch_runs'] = {
        'n': int(len(runs_all)), 'median_frames': float(np.median(lengths)),
        'unknown_team_share': float((runs_all[:, 2] < 0).mean()),
        'uncontrolled_share': float((runs_all[:, 4] == 0).mean()),
        'censored_share': float((runs_all[:, 5] == 1).mean()),
        'end_codes': {str(k): int(v) for k, v in zip(*np.unique(runs_all[:, 3], return_counts=True))},
        'out_events': int(sum(len(r['out_frames']) for r in records)),
    }
    # Agreement with sparse centre anchors (before anchor-only fill).
    agree = Counter()
    disagreements = []
    by_action = defaultdict(Counter)
    for r in records:
        known = np.flatnonzero(r['anchor_full_y'] >= 0)
        for i in known:
            dense_label = int(r['full_y'][i])
            from_anchor = int(r['full_prov'][i]) == dl.P_ANCHOR
            a = int(r['anchor_full_y'][i])
            if from_anchor:
                agree['dense_unknown_anchor_filled'] += 1
            elif dense_label == a:
                agree['agree'] += 1
            elif dense_label == 2:
                agree['dense_D_anchor_HA'] += 1
            else:
                agree['disagree_HA'] += 1
                runs = r['touch_runs']
                near = runs[(runs[:, 0] > i) & (runs[:, 0] <= i + 3) & (runs[:, 2] == a)]
                if len(near):
                    agree['disagree_HA_anchor_team_run_starts_within_3_frames'] += 1
                disagreements.append((r['segment_id'], int(i), dense_label, a,
                                      int(r['full_prov'][i])))
    total = sum(agree.values())
    labelled = agree['agree'] + agree['disagree_HA'] + agree['dense_D_anchor_HA']
    report['anchor_agreement'] = {
        'anchor_frames': total, **{k: int(v) for k, v in agree.items()},
        'agree_share_of_dense_known': agree['agree'] / labelled if labelled else None,
        'dense_unknown_share': agree['dense_unknown_anchor_filled'] / total if total else None,
        'first_disagreements(segment,local_frame,dense,anchor,prov)': disagreements[:20],
    }
    report['tackle_semantics'] = tackle_report(records, payloads)
    report['spot_check_switches'] = spot_check(records, n=20, seed=seed)
    report['label_provenance_names'] = dl.PROVENANCE_NAMES
    return report


def tackle_report(records, payloads) -> dict:
    """Does metadata TACKLE team equal the tackler (actor) or the team that keeps the ball?"""
    index = {}
    for r in records:
        gp = (str(r['match']).removeprefix('soccernet_'), int(r['game_period']))
        index.setdefault(gp, []).append(r)
    out = defaultdict(lambda: defaultdict(Counter))
    flip = defaultdict(Counter)
    for p in payloads:
        for s in p['samples']:
            if s['action'] not in ACTIONS_CHECKED or s['team'] not in ('home', 'away'):
                continue
            if s['center_frame'] is None or s['period_hint'] < 1:
                continue
            meta = 0 if s['team'] == 'home' else 1
            for r in p['records']:
                if int(r['game_period']) != s['period_hint']:
                    continue
                frames = r['full_frame_numbers']
                j = int(s['center_frame'] - frames[0])
                if 0 <= j < len(frames) and frames[j] == s['center_frame']:
                    pre, post = j - 15, j + 30
                    if pre >= 0 and post < len(frames):
                        a_, b_ = int(r['full_y'][pre]), int(r['full_y'][post])
                        if a_ in (0, 1) and b_ in (0, 1):
                            flip[s['action']][
                                f"before_{'meta' if a_ == meta else 'opp'}"
                                f"_after_{'meta' if b_ == meta else 'opp'}"] += 1
                    for off in TACKLE_OFFSETS:
                        k = j + off
                        if k >= len(frames):
                            continue
                        lab = int(r['full_y'][k])
                        key = ('unknown' if lab < 0 else 'D' if lab == 2 else
                               'same_as_meta' if lab == meta else 'opposite_of_meta')
                        out[s['action']][f'+{off}f'][key] += 1
                    break
    result = {a: {o: dict(c) for o, c in d.items()} for a, d in out.items()}
    result['flip_minus15_to_plus30'] = {a: dict(c) for a, c in flip.items()}
    return result


def spot_check(records, n=20, seed=0) -> list:
    rng = random.Random(seed)
    switches = []
    for ri, r in enumerate(records):
        fy = r['full_y']
        sw = np.flatnonzero((fy[1:] >= 0) & (fy[:-1] >= 0) & (fy[1:] != fy[:-1]) &
                            (fy[1:] < 2) & (fy[:-1] < 2)) + 1
        switches += [(ri, int(i)) for i in sw]
    picks = rng.sample(switches, min(n, len(switches)))
    names = {0: 'H', 1: 'A', 2: 'D', -1: '?'}
    codes = {v: k for k, v in dl.END_CODES.items()}
    out = []
    for ri, i in picks:
        r = records[ri]
        runs = r['touch_runs']
        near = [row for row in runs if row[1] >= i - 120 and row[0] <= i + 120]
        txt = ' | '.join(
            f"[{row[0] - i:+d}..{row[1] - i:+d}] {names[int(row[2])] if row[2] >= 0 else 'team?'}"
            f" end={codes.get(int(row[3]), '?') or 'none'}{'' if row[4] else ' UNCTRL'}"
            for row in near)
        out.append({'record': r['segment_id'], 'switch_frame_local': i,
                    'from': names[int(r['full_y'][i - 1])], 'to': names[int(r['full_y'][i])],
                    'runs_relative_to_switch': txt})
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=dl.DEFAULT_OUT)
    parser.add_argument('--report', type=Path,
                        default=dl.ROOT / 'data/experiments/sn_dense_v1/labels_report.json')
    parser.add_argument('--splits', nargs='+', default=list(dl.SPLITS))
    args = parser.parse_args()
    full = {s: split_report(s, args.output) for s in args.splits}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(full, indent=1, sort_keys=True, default=float) + '\n')
    for s, rep in full.items():
        print(s, rep['games'], rep['records'], round(rep['hours'], 2), 'h',
              rep['label_share_native'], 'switches', rep['ha_switches_native'])


if __name__ == '__main__':
    main()
