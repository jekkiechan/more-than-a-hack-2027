"""Validation-only oracle diagnostics; never training or production predictions.

Every oracle uses reference labels deliberately to answer an analysis question.
Only aggregate metrics are saved. This script cannot open a held-out match.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from evaluation import possession_metrics as pm
from experiments import supervised
from experiments.metrics import evaluate_tracking

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external_audit/sportec"
MODEL_DIR = ROOT / "data/experiments/robust"
VALIDATION = "J03WOY"
TRAIN = {"J03WMX", "J03WN1", "J03WOH"}
OUT = ROOT / "experiments/results/headroom_validation.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def blocks(data):
    period = supervised.model_periods(data)
    return supervised.blocks(period)


def segments(labels, data):
    result = []
    for left, right in blocks(data):
        edges = np.r_[left, left + np.flatnonzero(labels[left + 1:right] != labels[left:right - 1]) + 1, right]
        result.extend((int(a), int(b), str(labels[a])) for a, b in zip(edges[:-1], edges[1:]))
    return result


def compact_score(data, prediction):
    score = evaluate_tracking(data["labels"], prediction, period=data["period"],
                              fps=float(data["fps"]), timestamps=data["timestamps"])
    keys = ("macro_f1", "spell_f1_mean", "spell_f1_at_headline", "accuracy",
            "alive_dead_f1", "team_acc_gt_alive", "per_class", "spell_f1",
            "n_gt_spells", "n_pred_spells", "share_err", "label_coverage")
    return {key: score[key] for key in keys}


def sampled_reference(labels, data, step, offset, nearest=False):
    """Oracle labels on a fixed coarse grid, expanded without model predictions."""
    prediction = np.full(len(labels), "X", dtype="U1")
    for left, right in blocks(data):
        length = right - left
        samples = np.arange(min(offset, length - 1), length, step)
        relative = (np.arange(length) - samples[0]) / step
        indices = np.floor(relative + (0.5 if nearest else 0)).astype(int)
        prediction[left:right] = labels[left + samples[np.clip(indices, 0, len(samples) - 1)]]
    return prediction


def remove_short_reference_spells(labels, data, seconds):
    """Deliberately delete short true possessions, measuring metric sensitivity.

    Iteratively replace the shortest H/A run below the threshold by its longer
    adjacent graded run (previous on a tie). Never cross X or temporal boundaries.
    Dead runs are never candidates for removal. This is not an inference method.
    """
    prediction = labels.copy()
    removed = 0
    limit = seconds * float(data["fps"])
    for left, right in blocks(data):
        while True:
            edges = np.r_[left, left + np.flatnonzero(prediction[left + 1:right] != prediction[left:right - 1]) + 1, right]
            runs = [(int(a), int(b), str(prediction[a])) for a, b in zip(edges[:-1], edges[1:])]
            candidates = []
            for index, (a, b, label) in enumerate(runs):
                if label not in ("H", "A") or b - a >= limit:
                    continue
                adjacent = [j for j in (index - 1, index + 1)
                            if 0 <= j < len(runs) and runs[j][2] != "X"]
                if adjacent:
                    neighbor = max(adjacent, key=lambda j: (runs[j][1] - runs[j][0], -j))
                    candidates.append((b - a, a, b, runs[neighbor][2]))
            if not candidates:
                break
            _, a, b, replacement = min(candidates)
            prediction[a:b] = replacement
            removed += 1
    return prediction, removed


def duration_diagnostics(data, prediction):
    truth = data["labels"]
    # Match the official gap handling, including half and timestamp separators.
    bounds = [start for start, _ in blocks(data)][1:]
    gt = pm.label_codes(np.insert(truth, bounds, "X"))
    pred = pm.label_codes(np.insert(prediction, bounds, "X"))
    valid = pm.valid_runs(gt)
    pred = pm.fill_abstentions(pred, valid)
    spells = pm.possession_spells(gt, valid)
    guessed = pm.possession_spells(pred, valid)
    overlaps = pm.spell_overlaps(spells, guessed)
    matched = pm.matched_gt_flags(overlaps, len(spells), 0.5)
    durations = np.array([(b - a) / float(data["fps"]) for a, b, _ in spells])
    bins = [(0, .2), (.2, .5), (.5, 1), (1, 2), (2, 5), (5, np.inf)]
    return {
        "n_spells": len(spells),
        "duration_quantiles_seconds": dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"),
                                               np.percentile(durations, [0, 10, 25, 50, 75, 90, 100]).tolist())),
        "bins": [{"minimum_seconds_inclusive": lo, "maximum_seconds_exclusive": hi if np.isfinite(hi) else None,
                  "spells": int(np.sum((durations >= lo) & (durations < hi))),
                  "matched_at_iou_0_5": int(np.sum(matched & (durations >= lo) & (durations < hi))),
                  "baseline_recall_at_iou_0_5": float(matched[(durations >= lo) & (durations < hi)].mean())
                      if np.any((durations >= lo) & (durations < hi)) else None}
                 for lo, hi in bins],
        "short_spell_share": {
            str(limit): {"spells": int(np.sum(durations < limit)),
                         "fraction_of_spells": float(np.mean(durations < limit)),
                         "fraction_of_live_frames": float(durations[durations < limit].sum() / durations.sum())}
            for limit in (.2, .5, 1.0)},
    }


def main():
    frozen = json.loads((MODEL_DIR / "frozen_config.json").read_text())
    if {Path(row["path"]).stem for row in frozen["training"]} != TRAIN:
        raise ValueError("The diagnostic requires the original three-match training partition")
    if frozen["split"]["validation"] != [VALIDATION]:
        raise ValueError("Unexpected validation partition")
    model_path = MODEL_DIR / "model.joblib"
    if digest(model_path) != frozen["model_sha256"]:
        raise ValueError("Frozen robust model has changed")
    for name in ("experiments/supervised.py", "experiments/metrics.py"):
        if digest(ROOT / name) != frozen["source_sha256"][name]:
            raise ValueError(f"Frozen inference or metric implementation changed: {name}")
    path = BASE / "processed" / f"{VALIDATION}.npz"
    if digest(path) != frozen["validation_data_sha256"][VALIDATION]:
        raise ValueError("Original validation data has changed")
    data = supervised.load(path)
    # Labels enter oracle diagnostics only after the model produces probabilities.
    inputs = {key: value for key, value in data.items() if key != "labels"}
    model = joblib.load(model_path)
    probabilities = supervised.predict_probabilities(inputs, model)
    baseline = supervised.decode(probabilities, supervised.model_periods(inputs),
                                 float(data["fps"]), model["config"]["switch_cost"])
    truth = np.asarray(data["labels"])
    if not np.isin(truth, ["H", "A", "D"]).all():
        raise ValueError("These specific diagnostics require fully graded H/A/D validation")
    team_probabilities = probabilities[:, :2]
    team_probabilities /= np.maximum(team_probabilities.sum(axis=1, keepdims=True), 1e-12)
    conditional_team = np.array(["H", "A"])[np.argmax(team_probabilities, axis=1)]
    perfect_status = np.where(baseline == "D", conditional_team, baseline)
    perfect_status[truth == "D"] = "D"
    perfect_team = baseline.copy()
    team_errors = np.isin(baseline, ["H", "A"]) & np.isin(truth, ["H", "A"])
    perfect_team[team_errors] = truth[team_errors]
    scenarios = {
        "frozen_robust_baseline": {"oracle_only": False, "scores": compact_score(data, baseline)},
        "oracle_perfect_dead_live_status": {
            "oracle_only": True,
            "definition": "Correct every D decision; preserve decoded H/A except predicted-D true-live frames use frozen conditional H/A argmax.",
            "scores": compact_score(data, perfect_status)},
        "oracle_perfect_team_keep_dead_live_decisions": {
            "oracle_only": True,
            "definition": "Correct H/A confusion on true-live predicted-live frames; preserve every original D/live error.",
            "scores": compact_score(data, perfect_team)},
        "oracle_perfect_all_labels": {"oracle_only": True, "scores": compact_score(data, truth)},
    }
    step = round(float(data["fps"]) / 5)
    quantization = {}
    for method, nearest in (("sample_and_hold", False), ("nearest_sample", True)):
        offsets = []
        for offset in range(step):
            prediction = sampled_reference(truth, data, step, offset, nearest)
            offsets.append({"offset_frames": offset, "scores": compact_score(data, prediction)})
        quantization[method] = {
            "oracle_only": True, "target_fps": 5, "all_fixed_grid_offsets": offsets,
            "macro_f1_range": [min(row["scores"]["macro_f1"] for row in offsets), max(row["scores"]["macro_f1"] for row in offsets)],
            "spell_f1_range": [min(row["scores"]["spell_f1_mean"] for row in offsets), max(row["scores"]["spell_f1_mean"] for row in offsets)],
            "mean_macro_f1": float(np.mean([row["scores"]["macro_f1"] for row in offsets])),
            "mean_spell_f1": float(np.mean([row["scores"]["spell_f1_mean"] for row in offsets])),
        }
    short_removal = []
    for threshold in (.2, .5, 1.0):
        prediction, removed = remove_short_reference_spells(truth, data, threshold)
        short_removal.append({"oracle_only": True, "threshold_seconds_strict": threshold,
                              "runs_removed_iteratively": removed,
                              "frames_changed": int(np.sum(prediction != truth)),
                              "scores": compact_score(data, prediction)})
    errors = {
        "true_live_predicted_dead_frames": int(np.sum((truth != "D") & (baseline == "D"))),
        "true_dead_predicted_live_frames": int(np.sum((truth == "D") & (baseline != "D"))),
        "home_away_confusion_frames": int(np.sum(team_errors & (baseline != truth))),
        "total_error_frames": int(np.sum(baseline != truth)),
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "validation_match": VALIDATION,
        "analysis_only": True, "heldout_test_labels_read": False,
        "warning": "Oracle labels are used deliberately for diagnosis only. No oracle prediction is saved, deployed, trained on, or used to select a model. These scenarios are not global performance ceilings.",
        "model_sha256": digest(model_path), "model_config": model["config"],
        "validation_npz_sha256": digest(path), "script_sha256": digest(__file__),
        "scenarios": scenarios, "error_frame_decomposition": errors,
        "reference_duration_diagnostics": duration_diagnostics(data, baseline),
        "oracle_5hz_quantization": quantization, "oracle_remove_short_reference_spells": short_removal,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    for name, row in scenarios.items():
        print(name, row["scores"]["macro_f1"], row["scores"]["spell_f1_mean"], flush=True)
    for name, row in quantization.items():
        print("5hz", name, row["macro_f1_range"], row["spell_f1_range"], flush=True)
    for row in short_removal:
        print("remove_short", row["threshold_seconds_strict"], row["scores"]["macro_f1"], row["scores"]["spell_f1_mean"], flush=True)
    print(OUT, flush=True)


if __name__ == "__main__":
    main()
