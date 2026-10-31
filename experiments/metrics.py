"""Use the unchanged competition metrics on external matches without joining halves.

Unknown reference labels remain X and are excluded, never relabelled as dead ball.
Consequently a benchmark with only H/A targets cannot establish H/A/D performance.
"""

import numpy as np

from evaluation import possession_metrics as pm


def prepare_tracking_codes(labels, predictions, period=None, fps=25.0, timestamps=None):
    """Validate and preserve native gaps, inserting ungraded input-break separators.

    Shared by the official score wrapper and supplementary spell diagnostics so
    they operate on exactly the same evaluation population and interval edges.
    """
    truth = np.asarray(labels).astype(str)
    guess = np.asarray(predictions).astype(str)
    if truth.ndim != 1 or guess.shape != truth.shape or len(truth) == 0:
        raise ValueError("Expected equal, nonempty one-dimensional label arrays")
    if not np.isin(truth, pm.LABELS).all() or not np.isin(guess, pm.LABELS).all():
        raise ValueError("Labels must be H, A, D or X")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive")
    boundaries = np.zeros(len(truth) - 1, dtype=bool)
    if period is not None:
        period = np.asarray(period)
        if period.shape != truth.shape:
            raise ValueError("period must have one entry per frame")
        boundaries |= period[1:] != period[:-1]
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=float)
        if timestamps.shape != truth.shape or not np.isfinite(timestamps).all():
            raise ValueError("timestamps must be finite with one entry per frame")
        delta = np.diff(timestamps)
        boundaries |= (delta <= 0) | (delta > 1.5 / fps)

    # An ungraded separator makes the official scorer reset interval matching and
    # abstention filling. It does not give the model any possession boundaries.
    inserts = np.flatnonzero(boundaries) + 1
    gt_codes = pm.label_codes(np.insert(truth, inserts, "X"))
    pred_codes = pm.label_codes(np.insert(guess, inserts, "X"))
    return truth, gt_codes, pred_codes, inserts


def evaluate_tracking(labels, predictions, period=None, fps=25.0, timestamps=None,
                      bootstrap_n=0):
    """Score a frozen prediction; all time tolerances use the input sampling rate.

    Call serially: the upstream metric's FPS module constant is temporarily set
    for duration summaries, then restored. No upstream scoring code is changed.
    """
    truth, gt_codes, pred_codes, inserts = prepare_tracking_codes(
        labels, predictions, period, fps, timestamps)
    previous_fps = pm.FPS
    pm.FPS = float(fps)
    try:
        result = pm.evaluate(
            gt_codes, pred_codes,
            taus=(max(1, round(0.5 * fps)), max(1, round(fps))),
            bootstrap_n=bootstrap_n,
        )
    finally:
        pm.FPS = previous_fps
    present = [label for label in pm.SCORED if np.any(truth == label)]
    result.update({
        "original_n_frames": len(truth),
        "period_or_time_separators": len(inserts),
        "sampling_fps": float(fps),
        "label_coverage": float(np.mean(truth != "X")),
        "target_classes_present": present,
        "macro_f1_present_classes": float(np.mean([
            result["per_class"][label]["f1"] for label in present
        ])) if present else None,
        "three_class_benchmark": len(present) == 3,
    })
    return result
