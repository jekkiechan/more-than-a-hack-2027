"""Tracking-only possession inference with confirmed, retrospective state changes.

No source ownership or ball-status inputs are consumed. SkillCorner's external
targets currently support H/A evaluation only, so dead-ball inference defaults
off; its optional geometric heuristic is not a validated stoppage detector.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.metrics import evaluate_tracking


DEFAULT_CONFIG = {
    "interpolate_s": 0.5,
    "smooth_s": 0.3,
    "control_radius_m": 4.0,
    "margin": 0.0,
    "confirmation_s": 0.3,
    "dead_ball": False,
    "stationary_s": 5.0,
    "stationary_radius_m": 0.35,
    "restart_speed_m_s": 1.5,
}


def blocks(period: np.ndarray):
    edges = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return zip(edges[:-1], edges[1:])


def rolling_mean(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.copy()
    index = np.arange(len(values))
    low, high = np.maximum(0, index-radius), np.minimum(len(values), index+radius+1)
    total = np.r_[0.0, np.cumsum(values, dtype=np.float64)]
    return (total[high]-total[low]) / (high-low)


def fill_short_gaps(ball: np.ndarray, max_frames: int) -> np.ndarray:
    result = np.asarray(ball, dtype=np.float64).copy()
    valid = np.isfinite(result).all(axis=1)
    good = np.flatnonzero(valid)
    if max_frames <= 0 or len(good) < 2:
        return result
    for left, right in zip(good[:-1], good[1:]):
        gap = right-left-1
        if 0 < gap <= max_frames:
            frac = np.arange(1, gap+1, dtype=np.float64)/(gap+1)
            result[left+1:right] = result[left] + frac[:, None]*(result[right]-result[left])
    return result


def prepare(data: dict, interpolate_s: float = 0.0) -> dict:
    """Return label-free geometry; independent of the target array, if present."""
    ball = np.asarray(data["ball"], dtype=np.float64)
    players = np.asarray(data["players"], dtype=np.float64)
    teams = np.asarray(data["teams"])
    period = np.asarray(data["period"])
    fps = float(data["fps"])
    if fps <= 0 or players.shape[:1] != ball.shape[:1]:
        raise ValueError("Invalid sampling rate or incompatible tracking arrays")
    if "timestamps" in data:
        delta = np.diff(np.asarray(data["timestamps"]))
        cuts = (period[1:] != period[:-1]) | (delta <= 0) | (delta > 1.5/fps)
        period = np.r_[0, np.cumsum(cuts)]
    ball = ball.copy()
    for start, end in blocks(period):
        ball[start:end] = fill_short_gaps(ball[start:end], round(interpolate_s*fps))
    distances = np.linalg.norm(players-ball[:, None, :], axis=2)
    distances[~np.isfinite(distances)] = np.inf
    nearest = np.full((len(ball), 2), np.inf)
    for team in (0, 1):
        if np.any(teams == team):
            nearest[:, team] = np.min(distances[:, teams == team], axis=1)
    return {"nearest": nearest, "ball": ball, "period": period, "fps": fps,
            "pitch": np.asarray(data.get("pitch", [105.0, 68.0]))}


def dead_evidence(prepared: dict, config: dict) -> np.ndarray:
    out = np.zeros(len(prepared["ball"]), dtype=bool)
    for start, end in blocks(prepared["period"]):
        xy = prepared["ball"][start:end]
        observed = np.isfinite(xy).all(axis=1)
        pitch = prepared["pitch"]
        outside = observed & ((xy[:, 0] < -0.25) | (xy[:, 0] > pitch[0]+0.25)
                              | (xy[:, 1] < -0.25) | (xy[:, 1] > pitch[1]+0.25))
        width = max(3, round(config["stationary_s"]*prepared["fps"]))
        finite_xy = np.where(observed[:, None], xy, 1e6)
        span = maximum_filter1d(finite_xy, width, axis=0, mode="nearest") - minimum_filter1d(
            finite_xy, width, axis=0, mode="nearest")
        fully_visible = minimum_filter1d(observed.astype(np.int8), width, mode="constant") > 0
        stationary = fully_visible & (np.linalg.norm(span, axis=1) < config["stationary_radius_m"])
        out[start:end] = outside | stationary
    return out


def decode(prepared: dict, config: dict) -> np.ndarray:
    nearest = prepared["nearest"]
    ball, fps, period = prepared["ball"], prepared["fps"], prepared["period"]
    labels = np.full(len(ball), "X", dtype="U1")
    radius = round(config["smooth_s"]*fps)
    confirm = max(1, round(config["confirmation_s"]*fps))
    dead = dead_evidence(prepared, config) if config["dead_ball"] else None
    for start, end in blocks(period):
        distances = nearest[start:end]
        finite = np.isfinite(distances).all(axis=1)
        close = np.min(distances, axis=1) <= config["control_radius_m"]
        support = finite & close
        signed = np.zeros(end-start)
        # Relative proximity is less sensitive to a long aerial pass than a
        # hard vote for a player several metres away from the ball.
        signed[support] = ((distances[support, 1]-distances[support, 0]) /
                           (distances[support].sum(axis=1)+1e-6))
        strength = rolling_mean(support.astype(float), radius)
        evidence = rolling_mean(signed, radius)
        average = np.divide(evidence, strength, out=np.zeros_like(evidence), where=strength > 0)
        candidate = np.full(end-start, -1, dtype=np.int8)
        candidate[(strength > 0) & (average > config["margin"])] = 0
        candidate[(strength > 0) & (average < -config["margin"])] = 1
        if config["dead_ball"]:
            xy = ball[start:end]
            ball_speed = np.zeros(len(xy))
            if len(xy) > 1:
                ball_speed[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1)*fps
            ball_speed[~np.isfinite(ball_speed)] = 0
            candidate[dead[start:end]] = 2
        state, pending, pending_start, pending_count = -1, -1, 0, 0
        result = np.full(end-start, -1, dtype=np.int8)
        for i, proposed in enumerate(candidate):
            if config["dead_ball"] and state == 2 and proposed in (0, 1):
                if ball_speed[i] < config["restart_speed_m_s"]:
                    proposed = -1
            if proposed < 0 or proposed == state:
                pending, pending_count = -1, 0
            else:
                if proposed != pending:
                    pending, pending_start, pending_count = int(proposed), i, 1
                else:
                    pending_count += 1
                if pending_count >= confirm:
                    state = pending
                    # Offline inference: once control is confirmed, locate the
                    # turnover at the beginning of its supporting evidence.
                    result[pending_start:i+1] = state
                    pending, pending_count = -1, 0
            result[i] = state
        known = np.flatnonzero(result >= 0)
        if len(known):
            result[:known[0]] = result[known[0]]
        lookup = np.array(["H", "A", "D", "X"])
        labels[start:end] = lookup[np.where(result >= 0, result, 3)]
    return labels


def predict(data: dict, config: dict | None = None) -> np.ndarray:
    config = {**DEFAULT_CONFIG, **(config or {})}
    return decode(prepare(data, config["interpolate_s"]), config)


def nearest_baseline(data: dict) -> np.ndarray:
    prepared = prepare(data)
    distances = prepared["nearest"]
    pred = np.full(len(distances), "X", dtype="U1")
    good = np.isfinite(distances).all(axis=1)
    pred[good] = np.where(distances[good, 0] <= distances[good, 1], "H", "A")
    return pred


def nearest_with_dead_baseline(data: dict, config: dict) -> np.ndarray:
    """Ablation: nearest visible team plus the same geometric dead-ball cue."""
    prediction = nearest_baseline(data)
    if config["dead_ball"]:
        prediction[dead_evidence(prepare(data, config["interpolate_s"]), config)] = "D"
    return prediction


def score(data: dict, labels: np.ndarray) -> dict:
    """Shared official metrics with period/time-boundary and sampling-rate handling."""
    truth = np.asarray(data["labels"])
    metrics = evaluate_tracking(truth, labels, period=data["period"],
                                fps=float(data["fps"]), timestamps=data.get("timestamps"))
    return {
        "accuracy": metrics["accuracy"],
        "binary_macro_f1": float(np.mean([metrics["per_class"][c]["f1"] for c in ("H", "A")])),
        "official_three_class_macro_f1": metrics["macro_f1"],
        "spell_f1": metrics["spell_f1_mean"],
        "n_graded": metrics["n_graded"],
        "coverage": float(np.mean(truth != "X")),
        "n_gt_spells": metrics["n_gt_spells"],
        "n_pred_spells": metrics["n_pred_spells"],
        "per_class": metrics["per_class"],
        "team_accuracy_on_true_live_frames": metrics["team_acc_gt_alive"],
        "no_dead_ball_targets": not bool(np.any(truth == "D")),
        "macro_f1_present_classes": metrics["macro_f1_present_classes"],
        "three_class_benchmark": metrics["three_class_benchmark"],
    }


def load_match(directory: Path, match: str) -> dict:
    with np.load(directory / (match+".npz"), allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def tune(split: dict, directory: Path, output: Path, with_dead: bool = False,
         refine: bool = False):
    base_config = dict(DEFAULT_CONFIG)
    dead_training = []
    if refine:
        base_config.update(json.loads((output/"frozen_config.json").read_text()))
    if with_dead and not refine:
        training = {name: load_match(directory, name) for name in split["train"]}
        training_geometry = {name: prepare(data) for name, data in training.items()}
        for seconds, radius in itertools.product((0.12, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0),
                                                 (0.01, 0.05, 0.2, 0.5, 1.0)):
            cfg = {**base_config, "dead_ball": True, "stationary_s": seconds,
                   "stationary_radius_m": radius}
            f1s = []
            for name, data in training.items():
                truth = data["labels"]
                mask = truth != "X"
                wanted = truth[mask] == "D"
                guessed = dead_evidence(training_geometry[name], cfg)[mask]
                tp = np.sum(wanted & guessed)
                denominator = np.sum(wanted)+np.sum(guessed)
                f1s.append(float(2*tp/denominator) if denominator else 0.0)
            dead_training.append({"config": cfg, "mean_train_dead_f1": float(np.mean(f1s))})
        dead_training.sort(key=lambda row: row["mean_train_dead_f1"], reverse=True)
        base_config = dead_training[0]["config"]
    matches = {name: load_match(directory, name) for name in split["validation"]}
    gaps = (base_config["interpolate_s"],) if refine else (0.0, 0.5, 1.0)
    cache = {(name, gap): prepare(data, gap) for name, data in matches.items()
             for gap in gaps}
    candidates = []
    if refine:
        combinations = itertools.product(gaps, (0.0, 0.3, 0.7, 1.5, 2.5),
                                         (0.3, 0.5, 0.75, 1.0, 1.5, 2.0), (0.1, 0.3, 0.7, 1.5))
    else:
        combinations = itertools.product(gaps, (0.0, 0.3, 0.7, 1.5),
                                         (2.0, 4.0, 8.0, 1000.0), (0.1, 0.3, 0.7))
    for gap, smooth, control, confirmation in combinations:
        config = {**base_config, "interpolate_s": gap, "smooth_s": smooth,
                  "control_radius_m": control, "confirmation_s": confirmation}
        scores = {name: score(data, decode(cache[name, gap], config)) for name, data in matches.items()}
        objective = float(np.mean([s["spell_f1"] for s in scores.values()]))
        macro = float(np.mean([s["macro_f1_present_classes"] for s in scores.values()]))
        candidates.append({"config": config, "validation_spell_f1": objective,
                           "validation_macro_f1": macro,
                           "matches": scores})
    # Stated objective: reach 0.90 per-frame macro F1, then maximize spell F1.
    candidates.sort(key=lambda row: (row["validation_macro_f1"] >= 0.9,
                                    row["validation_spell_f1"] if row["validation_macro_f1"] >= 0.9
                                    else row["validation_macro_f1"]), reverse=True)
    output.mkdir(parents=True, exist_ok=True)
    if dead_training:
        (output/"dead_training_search.json").write_text(json.dumps(dead_training, indent=2)+"\n")
    search_name = "validation_search_refinement.json" if refine else "validation_search.json"
    (output/search_name).write_text(json.dumps(candidates, indent=2)+"\n")
    (output/"frozen_config.json").write_text(json.dumps(candidates[0]["config"], indent=2)+"\n")
    pareto = {"selection": candidates[0],
              "best_macro": max(candidates, key=lambda row: row["validation_macro_f1"]),
              "best_spell": max(candidates, key=lambda row: row["validation_spell_f1"])}
    (output/"validation_pareto.json").write_text(json.dumps(pareto, indent=2)+"\n")
    print(json.dumps(candidates[0], indent=2))


def evaluate_split(split: dict, directory: Path, output: Path, phase: str):
    config = json.loads((output/"frozen_config.json").read_text())
    report_path = output/(phase+"_results.json")
    if phase == "test" and report_path.exists():
        raise RuntimeError("Test already evaluated; do not repeatedly tune against it")
    results = {}
    for name in split[phase]:
        data = load_match(directory, name)
        prediction = predict(data, config)
        results[name] = {"state_machine": score(data, prediction),
                         "nearest_baseline": score(data, nearest_baseline(data)),
                         "nearest_with_dead_baseline": score(data, nearest_with_dead_baseline(data, config))}
        np.save(output/(name+"_predictions.npy"), prediction)
    report = {"phase": phase, "config": config, "results": results}
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("tune", "train", "validation", "test"))
    parser.add_argument("--split", type=Path, default=ROOT/"data/external/split.json")
    parser.add_argument("--data", type=Path, default=ROOT/"data/external/processed")
    parser.add_argument("--out", type=Path, default=ROOT/"data/experiments/state_machine")
    parser.add_argument("--dead-ball", action="store_true", help="Fit geometric dead-ball thresholds on external training labels")
    parser.add_argument("--refine", action="store_true", help="Validation-only refinement around small player-ball control distances")
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    if args.action == "tune":
        tune(split, args.data, args.out, args.dead_ball, args.refine)
    else:
        evaluate_split(split, args.data, args.out, args.action)


if __name__ == "__main__":
    main()
