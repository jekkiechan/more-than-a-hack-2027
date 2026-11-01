"""Fixed posterior-sample minimum-risk decoder for possession spell F1.

The decoder samples candidate paths and independent reference paths from the
same cost-3 Potts posterior, then returns the candidate with highest expected
official spell F1. It is an opt-in research transform; no target labels enter
inference. Exact H/A-symmetric inputs retain the reference decoder's
H-first Viterbi path as the first candidate and deterministic utility-tie rule.
"""
from __future__ import annotations

import numpy as np

from evaluation import possession_metrics as pm
from experiments import supervised


DEFAULT_CANDIDATE_SAMPLES = 16
DEFAULT_REFERENCE_SAMPLES = 32
DEFAULT_SEED = 20261101
OUTPUT_LABELS = tuple(supervised.LABELS) + ("X",)


def _blocks(period, valid_mask):
    """Return valid half-open runs, split at time/period breaks and mask holes."""
    period = np.asarray(period)
    valid = np.asarray(valid_mask, dtype=bool)
    if period.ndim != 1 or valid.ndim != 1 or len(period) != len(valid):
        raise ValueError("period and valid_mask must be 1-D arrays with equal length")
    n = len(period)
    if not n:
        return []
    split = np.zeros(n, dtype=bool)
    split[0] = True
    split[1:] = (period[1:] != period[:-1]) | (valid[1:] != valid[:-1])
    edges = np.flatnonzero(split)
    bounds = np.r_[edges, n]
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if valid[a]]


def _canonicalize(probabilities, valid_classes):
    """Orient asymmetric team evidence canonically and report whether to unflip."""
    result = probabilities.copy()
    has_h = bool(0 in valid_classes)
    has_a = bool(1 in valid_classes)
    swap_back = False
    if has_h != has_a:
        # Make the sole available team occupy canonical H, then map back.
        if has_a:
            result[:, [0, 1]] = result[:, [1, 0]]
            swap_back = True
        return result, swap_back
    if has_h and has_a:
        unequal = np.flatnonzero(result[:, 0] != result[:, 1])
        if len(unequal) and result[unequal[0], 0] < result[unequal[0], 1]:
            result[:, [0, 1]] = result[:, [1, 0]]
            swap_back = True
    return result, swap_back


def _logsumexp(values, axis):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis,
                                                   keepdims=True)), axis=axis)


def _sample_block(logp, penalty, sample_count, rng):
    """Exact forward-filtering backward-sampling for one contiguous block."""
    return _sample_block_with_uniforms(logp, penalty, rng.random((sample_count, len(logp))))


def _sample_block_with_uniforms(logp, penalty, uniforms):
    """FFBS using pre-drawn uniforms, allowing independent streams in one pass."""
    n, k = logp.shape
    sample_count = uniforms.shape[0]
    alpha = np.empty((n, k), dtype=np.float64)
    alpha[0] = logp[0]
    for t in range(1, n):
        transition = np.broadcast_to(alpha[t - 1][:, None], (k, k)).copy()
        transition -= penalty
        np.fill_diagonal(transition, alpha[t - 1])
        alpha[t] = logp[t] + _logsumexp(transition, axis=0)
        alpha[t] -= np.max(alpha[t])

    paths = np.empty((sample_count, n), dtype=np.int8)
    terminal = np.exp(alpha[-1] - _logsumexp(alpha[-1], axis=0))
    paths[:, -1] = np.searchsorted(np.cumsum(terminal), uniforms[:, -1], side="right")
    paths[:, -1] = np.minimum(paths[:, -1], k - 1)
    for t in range(n - 2, -1, -1):
        transition = np.full((sample_count, k), -penalty, dtype=np.float64)
        transition[np.arange(sample_count), paths[:, t + 1]] = 0.0
        log_weights = alpha[t][None, :] + transition
        log_weights -= _logsumexp(log_weights, axis=1)[:, None]
        cdf = np.cumsum(np.exp(log_weights), axis=1)
        states = np.sum(uniforms[:, t, None] > cdf, axis=1)
        paths[:, t] = np.minimum(states, k - 1)
    return paths


def _spells(codes, runs):
    return pm.possession_spells(codes, runs)


def _overlaps_linear(gt_spells, pred_spells):
    """Same-team overlap pairs in official order, using disjoint ordered spells."""
    pairs = []
    by_team_gt = {team: [(a, b, i) for i, (a, b, c) in enumerate(gt_spells) if c == team]
                  for team in pm.POSSESSION_LABELS}
    by_team_pred = {team: [(a, b, j) for j, (a, b, c) in enumerate(pred_spells) if c == team]
                    for team in pm.POSSESSION_LABELS}
    for team in pm.POSSESSION_LABELS:
        left, right = by_team_gt[team], by_team_pred[team]
        i = j = 0
        while i < len(left) and j < len(right):
            ga, gb, gi = left[i]
            pa, pb, pj = right[j]
            if pa < gb and ga < pb:
                intersection = min(gb, pb) - max(ga, pa)
                union = max(gb, pb) - min(ga, pa)
                pairs.append((intersection / float(union), gi, pj))
            if gb <= pb:
                i += 1
            if pb <= gb:
                j += 1
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return pairs


def _spell_f1_from_spells(gt_spells, pred_spells):
    overlaps = _overlaps_linear(gt_spells, pred_spells)
    f1s = []
    for threshold in pm.SPELL_IOUS:
        tp, _, _ = pm.match_spells(overlaps, len(gt_spells), len(pred_spells), threshold)
        precision = tp / float(len(pred_spells)) if pred_spells else 0.0
        recall = tp / float(len(gt_spells)) if gt_spells else 0.0
        f1s.append((2.0 * precision * recall / (precision + recall))
                   if precision + recall else 0.0)
    return float(np.mean(f1s))


def _viterbi(probabilities, period, fps, switch_cost):
    return supervised.decode(probabilities, period, fps, switch_cost)


def decode(probabilities, period, fps, switch_cost=3, *, valid_mask=None,
           candidate_samples=DEFAULT_CANDIDATE_SAMPLES,
           reference_samples=DEFAULT_REFERENCE_SAMPLES, seed=DEFAULT_SEED):
    """Return an MBR path using fixed posterior-sampling budgets.

    The expected utility is the official spell F1 averaged across IoU 0.3–0.7.
    Candidate selection uses whole-record counts: valid runs are scored together,
    so an independently selected block cannot alter global TP/FP/FN aggregation.
    Frames excluded by ``valid_mask`` are returned as X and split temporal runs.
    """
    probabilities = np.asarray(probabilities)
    period = np.asarray(period)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_frames, 3) for H/A/D")
    if period.ndim != 1 or len(period) != len(probabilities):
        raise ValueError("period must be a 1-D array matching probabilities")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(switch_cost) or switch_cost < 0:
        raise ValueError("fps must be positive and switch_cost must be nonnegative")
    if candidate_samples < 0 or reference_samples < 1:
        raise ValueError("candidate_samples must be nonnegative and reference_samples positive")
    if len(probabilities) == 0:
        return np.empty(0, dtype="U1")
    valid_mask = np.ones(len(probabilities), dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid_mask.shape != (len(probabilities),):
        raise ValueError("valid_mask must match probabilities")
    runs = _blocks(period, valid_mask)
    if not runs:
        return np.full(len(probabilities), "X", dtype="U1")

    original_valid_classes = np.flatnonzero(probabilities.sum(axis=0) > 0)
    if len(original_valid_classes) == 0:
        raise ValueError("probabilities must contain at least one globally available class")
    canonical, swap_back = _canonicalize(probabilities, original_valid_classes)
    valid_classes = np.flatnonzero(canonical.sum(axis=0) > 0)
    sliced = canonical[:, valid_classes]
    logp = np.log(np.maximum(sliced, 1e-7)).astype(np.float64)
    penalty = float(switch_cost) * float(fps) / 10.0
    candidate_rng, reference_rng = [np.random.default_rng(s) for s in
                                    np.random.SeedSequence(seed).spawn(2)]
    sampled_by_block = np.full(
        (candidate_samples + reference_samples, len(probabilities)), 3, dtype=np.int8)
    # Follow the baseline's model-period boundaries exactly. The label-free
    # validity mask limits returned labels and metric runs, but does not alter
    # inference context across a masked interval.
    for start, stop in supervised.blocks(period):
        uniforms = np.concatenate((
            candidate_rng.random((candidate_samples, stop - start)),
            reference_rng.random((reference_samples, stop - start))), axis=0)
        sampled_by_block[:, start:stop] = valid_classes[
            _sample_block_with_uniforms(logp[start:stop], penalty, uniforms)]

    map_labels = _viterbi(probabilities, period, fps, switch_cost)
    label_to_code = {label: index for index, label in enumerate(OUTPUT_LABELS)}
    map_codes = np.array([label_to_code[label] for label in map_labels], dtype=np.int8)
    if swap_back:
        map_codes[map_codes == 0] = 4
        map_codes[map_codes == 1] = 0
        map_codes[map_codes == 4] = 1
    map_codes[~valid_mask] = 3
    candidates = [map_codes]
    candidates.extend(sampled_by_block[i].copy() for i in range(candidate_samples))
    references = sampled_by_block[candidate_samples:]
    for path in candidates:
        path[~valid_mask] = 3
    references[:, ~valid_mask] = 3
    reference_spells = [_spells(path, runs) for path in references]
    utilities = []
    for candidate in candidates:
        candidate_spells = _spells(candidate, runs)
        utilities.append(float(np.mean([
            _spell_f1_from_spells(ref_spells, candidate_spells)
            for ref_spells in reference_spells])))
    selected = candidates[int(np.argmax(utilities))].copy()
    if swap_back:
        selected[selected == 0] = 4
        selected[selected == 1] = 0
        selected[selected == 4] = 1
    return np.asarray([OUTPUT_LABELS[int(code)] for code in selected], dtype="U1")

