"""Aggregate diagnostics of the unchanged official possession spell objective.

No fitting, candidate selection, organizer loading, or per-frame export occurs.
Overlap counts describe error geometry, not mutually exclusive causal labels.
"""
from __future__ import annotations

import numpy as np

from evaluation import possession_metrics as pm
from experiments.metrics import prepare_tracking_codes


def _pairs(overlaps, threshold):
    """Expose the exact official IoU-descending greedy assignment for diagnostics."""
    used_gt, used_pred, pairs = set(), set(), []
    for iou, gi, pi in overlaps:
        if iou < threshold:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        pairs.append((iou, gi, pi))
    return pairs


def _summary(values):
    return dict(n=len(values), mean=float(np.mean(values)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p90_abs=float(np.percentile(np.abs(values), 90)) if len(values) else None)


def spell_diagnostics(labels, predictions, period=None, fps=25.0, timestamps=None):
    """Return JSON-safe aggregate diagnostics on the exact official scored spells.

    Duration recall bins use reference duration and the *global* assignment at
    each threshold; filtering/rematching subgroups would change the objective.
    Boundary errors exclude reference spells touching a graded-run edge because
    those endpoints may be artificial observation cuts. Signed error is prediction
    minus reference time. Diagnostics never change the official score.
    """
    _, truth, guess, inserts = prepare_tracking_codes(
        labels, predictions, period, fps, timestamps)
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(guess, runs)
    gt = pm.possession_spells(truth, runs)
    pred = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt, pred)
    score = pm.spell_scores(truth, filled, runs)
    durations = np.array([(b-a)/fps for a, b, _ in gt])
    run_starts, run_ends = {a for a, _ in runs}, {b for _, b in runs}
    complete = np.array([a not in run_starts and b not in run_ends for a, b, _ in gt], dtype=bool)
    masks = {
        'under_3_seconds': durations < 3,
        '3_to_under_10_seconds': (durations >= 3) & (durations < 10),
        '10_seconds_or_longer': durations >= 10,
        'home': np.array([k == pm.H for _, _, k in gt], dtype=bool),
        'away': np.array([k == pm.A for _, _, k in gt], dtype=bool),
        'complete_observed_spells': complete,
        'touches_graded_run_edge': ~complete,
    }
    per_iou = {}
    for threshold in pm.SPELL_IOUS:
        pairs = _pairs(overlaps, threshold)
        matched = np.zeros(len(gt), dtype=bool)
        for _, gi, _ in pairs:
            matched[gi] = True
        # Protect diagnostic parity if the pinned official assignment ever changes.
        official = pm.matched_gt_flags(overlaps, len(gt), threshold)
        if not np.array_equal(matched, official):
            raise RuntimeError('Diagnostic matching differs from official scorer')
        per_iou[str(threshold)] = {
            **score['per_iou'][threshold],
            'reference_recall_groups': {
                name: dict(n_gt=int(mask.sum()), n_matched=int(matched[mask].sum()),
                           recall=float(matched[mask].mean()) if mask.any() else None)
                for name, mask in masks.items()},
            'unmatched_with_eligible_overlap': sum(
                not matched[i] and any(i == gi and iou >= threshold for iou, gi, _ in overlaps)
                for i in range(len(gt))),
        }

    gt_overlap_counts = np.zeros(len(gt), dtype=int)
    pred_overlap_counts = np.zeros(len(pred), dtype=int)
    best = np.zeros(len(gt))
    for iou, gi, pi in overlaps:
        gt_overlap_counts[gi] += 1
        pred_overlap_counts[pi] += 1
        best[gi] = max(best[gi], iou)
    pairs = _pairs(overlaps, pm.SPELL_IOU_HEADLINE)
    complete_pairs = [(gi, pi) for _, gi, pi in pairs if complete[gi]]
    starts = [(pred[pi][0]-gt[gi][0])/fps for gi, pi in complete_pairs]
    ends = [(pred[pi][1]-gt[gi][1])/fps for gi, pi in complete_pairs]
    return {
        'spell_f1_mean': score['mean_f1'], 'n_gt_spells': len(gt), 'n_pred_spells': len(pred),
        'spell_count_error': len(pred)-len(gt), 'per_iou': per_iou,
        'graded_runs': len(runs), 'period_or_time_separators': len(inserts),
        'n_abstain_on_graded': int(np.sum((truth != pm.X) & (guess == pm.X))),
        'reference_spells_touching_graded_run_edge': int((~complete).sum()),
        'overlap_geometry': {
            'reference_without_same_team_overlap': int((gt_overlap_counts == 0).sum()),
            'reference_overlapped_by_multiple_same_team_predictions': int((gt_overlap_counts > 1).sum()),
            'predictions_overlapping_multiple_same_team_references': int((pred_overlap_counts > 1).sum()),
            'reference_best_same_team_iou_below_0_3': int((best < .3).sum()),
            'reference_best_same_team_iou_0_3_to_under_0_7': int(((best >= .3) & (best < .7)).sum()),
            'reference_best_same_team_iou_at_least_0_7': int((best >= .7).sum()),
        },
        'matched_complete_boundary_errors_seconds_at_iou_0_5': {
            'start': _summary(starts), 'end': _summary(ends),
            'excluded_run_edge_matches': len(pairs)-len(complete_pairs),
        },
    }
