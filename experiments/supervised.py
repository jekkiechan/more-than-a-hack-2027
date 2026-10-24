"""External-match supervised baseline: tracking features, gradient boosting, Viterbi.

No possession/owner/ball-status metadata is used as input. Labels enter only fit
and evaluation. Whole-match train/validation/test assignments are supplied by the
dataset loader; the hackathon public ground truth is never loaded here.
"""

from __future__ import annotations

import os

# Three experiments share a small CPU worker. Set these before NumPy/sklearn.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.metrics import evaluate_tracking  # noqa: E402

LABELS = np.array(["H", "A", "D"])


def load(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def blocks(period):
    boundaries = np.r_[0, np.flatnonzero(np.diff(period) != 0) + 1, len(period)]
    return list(zip(boundaries[:-1], boundaries[1:]))


def model_periods(data):
    """Split tracking at time discontinuities as well as period boundaries."""
    period = np.asarray(data["period"])
    breaks = np.diff(period) != 0
    if "timestamps" in data:
        delta = np.diff(data["timestamps"])
        breaks |= (delta <= 0) | (delta > 1.5 / float(data["fps"]))
    return np.r_[0, np.cumsum(breaks)]


def _short_fill(ball, period, fps, max_gap_s=0.5):
    """Interpolate short internal observation gaps without crossing periods."""
    filled = np.array(ball, dtype=np.float64, copy=True)
    for start, stop in blocks(period):
        valid = np.flatnonzero(np.isfinite(ball[start:stop]).all(axis=1))
        if len(valid) < 2:
            continue
        for left, right in zip(valid[:-1], valid[1:]):
            if 1 < right - left <= round(max_gap_s * fps) + 1:
                alpha = np.arange(1, right - left) / (right - left)
                filled[start+left+1:start+right] = (
                    filled[start+left] * (1-alpha[:, None])
                    + filled[start+right] * alpha[:, None])
    return filled


def _rolling(values, width, period):
    values = np.asarray(values, dtype=np.float64)
    output = np.empty_like(values)
    for start, stop in blocks(period):
        valid = np.isfinite(values[start:stop])
        weighted = uniform_filter1d(np.where(valid, values[start:stop], 0),
                                    size=width, axis=0, mode="nearest")
        weight = uniform_filter1d(valid.astype(float), size=width,
                                 axis=0, mode="nearest")
        output[start:stop] = np.divide(weighted, weight,
                                      out=np.full_like(weighted, np.nan),
                                      where=weight > 0)
    return output


def features(data):
    """Return frame features from geometry/motion/observations, never targets.

    Positions are metres, players can be arbitrary fragmented track slots. Team
    features aggregate visible positions, so consistent player IDs are unnecessary.
    All rolling windows are period-local. Centred windows are allowed for the
    offline submission and provide short before/after context around a touch.
    """
    raw_ball = np.asarray(data["ball"], dtype=np.float64)
    players = np.asarray(data["players"], dtype=np.float64)
    teams = np.asarray(data["teams"], dtype=np.int8)
    period = model_periods(data)
    fps = float(data["fps"])
    ball = _short_fill(raw_ball, period, fps)
    observed = np.isfinite(raw_ball).all(axis=1)
    available = np.isfinite(ball).all(axis=1)
    distance = np.linalg.norm(players - ball[:, None, :], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    columns, names = [], []

    def add(name, values):
        names.append(name)
        columns.append(np.asarray(values, dtype=np.float64))

    add("ball_available", available)
    # Explicit source observation flag is permitted: it describes visibility,
    # not possession. SkillCorner also includes extrapolated coordinates.
    add("ball_observed", np.asarray(data.get("ball_observed", observed), dtype=float))
    velocity = np.full_like(ball, np.nan)
    for start, stop in blocks(period):
        if stop-start > 1:
            velocity[start:stop] = np.gradient(ball[start:stop], axis=0) * fps
    speed = np.linalg.norm(velocity, axis=1)
    add("ball_speed", np.minimum(speed, 60))
    previous_gap = np.zeros(len(ball), dtype=float)
    next_gap = np.zeros(len(ball), dtype=float)
    for start, stop in blocks(period):
        age = 0
        for i in range(start, stop):
            age = 0 if observed[i] else age + 1
            previous_gap[i] = min(age / fps, 30)
        age = 0
        for i in range(stop-1, start-1, -1):
            age = 0 if observed[i] else age + 1
            next_gap[i] = min(age / fps, 30)
    add("missing_before_s", previous_gap)
    add("missing_after_s", next_gap)
    pitch = np.asarray(data.get("pitch", [105, 68]), dtype=float)
    edge = np.minimum(np.minimum(ball[:, 0], pitch[0]-ball[:, 0]),
                      np.minimum(ball[:, 1], pitch[1]-ball[:, 1]))
    add("ball_pitch_edge_distance", np.clip(edge, -20, 35))

    # Collective motion is useful stoppage/restart evidence and does not require
    # player identities. Exclude implausible jumps from fragmented tracking slots.
    player_velocity = np.full_like(players, np.nan)
    for start, stop in blocks(period):
        if stop-start > 1:
            player_velocity[start:stop] = np.gradient(players[start:stop], axis=0) * fps
    player_speed = np.linalg.norm(player_velocity, axis=2)
    player_speed[(player_speed > 12) | ~np.isfinite(player_speed)] = np.nan
    valid_speed = np.isfinite(player_speed)
    count = valid_speed.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        motion = {
            "player_speed_median": np.nanmedian(player_speed, axis=1),
            "player_speed_q75": np.nanquantile(player_speed, 0.75, axis=1),
            "player_stationary_fraction": np.divide(
                (player_speed < 1).sum(axis=1), count,
                out=np.full(len(ball), np.nan), where=count > 0),
            "player_running_fraction": np.divide(
                (player_speed > 4).sum(axis=1), count,
                out=np.full(len(ball), np.nan), where=count > 0),
        }
    add("valid_player_speed_count", count)
    for name, values in motion.items():
        add(name, values)
        for seconds in (1., 3.):
            add(f"{name}_mean_{seconds}s", _rolling(values, 2*round(seconds*fps)+1, period))

    nearest = []
    for team, prefix in enumerate(("H", "A")):
        d = np.sort(distance[:, teams == team], axis=1)
        if d.shape[1] == 0:
            raise ValueError(f"No tracking slots for team {team}")
        near = np.minimum(d[:, 0], 100)
        near[~available] = np.nan
        nearest.append(near)
        for rank in (0, 1, 2):
            selected = np.minimum(d[:, min(rank, d.shape[1]-1)], 100)
            selected[~available] = np.nan
            add(f"{prefix}_distance{rank+1}", selected)
        for radius in (2, 5, 10):
            add(f"{prefix}_within{radius}m", np.sum(d <= radius, axis=1))
        add(f"{prefix}_visible_players", np.isfinite(players[:, teams == team]).all(axis=2).sum(axis=1))

    advantage = nearest[1] - nearest[0]
    add("home_distance_advantage", advantage)
    total = nearest[0] + nearest[1] + 1
    add("home_relative_advantage", advantage / total)
    home_signal = np.where(np.isfinite(advantage), np.sign(advantage), np.nan)
    minimum = np.minimum(nearest[0], nearest[1])
    # Last and next close approaches are contact evidence, not source ownership.
    # The classifier learns how much to trust them during passes and duels.
    for radius in (1.0, 2.0, 3.0):
        contact = np.where(minimum <= radius, np.sign(advantage), 0)
        before, after = np.zeros(len(ball)), np.zeros(len(ball))
        age_before, age_after = np.full(len(ball), 30.), np.full(len(ball), 30.)
        for start, stop in blocks(period):
            last, age = 0., 30.
            for i in range(start, stop):
                if contact[i] != 0:
                    last, age = contact[i], 0.
                before[i], age_before[i] = last, age
                age = min(30., age + 1/fps)
            last, age = 0., 30.
            for i in range(stop-1, start-1, -1):
                if contact[i] != 0:
                    last, age = contact[i], 0.
                after[i], age_after[i] = last, age
                age = min(30., age + 1/fps)
        add(f"home_contact_before_{radius}m", before)
        add(f"home_contact_after_{radius}m", after)
        add(f"contact_age_before_{radius}m", age_before)
        add(f"contact_age_after_{radius}m", age_after)
    for seconds in (0.25, 1.0, 2.5):
        width = max(1, 2 * round(seconds * fps) + 1)
        for prefix, near in zip(("H", "A"), nearest):
            add(f"{prefix}_distance_mean_{seconds}s", _rolling(near, width, period))
        add(f"home_advantage_mean_{seconds}s", _rolling(advantage, width, period))
        add(f"home_nearest_vote_{seconds}s", _rolling(home_signal, width, period))
        add(f"speed_mean_{seconds}s", _rolling(np.minimum(speed, 60), width, period))
        add(f"visible_mean_{seconds}s", _rolling(observed.astype(float), width, period))
    matrix = np.column_stack(columns).astype(np.float32)
    # Preserve NaN for HistGradientBoosting's native missing-value handling.
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, names


def swap_team_features(matrix, names):
    """Home/away augmentation, without recomputing expensive geometry."""
    output = matrix.copy()
    index = {name: i for i, name in enumerate(names)}
    for i, name in enumerate(names):
        if name.startswith("H_"):
            output[:, i] = matrix[:, index["A_" + name[2:]]]
        elif name.startswith("A_"):
            output[:, i] = matrix[:, index["H_" + name[2:]]]
        elif name.startswith("home_"):
            output[:, i] = -matrix[:, i]
    return output


def fit(train_paths, iterations=140, leaves=23, train_fps=4, two_stage=False):
    xs, ys, names, manifest = [], [], None, []
    for path in train_paths:
        data = load(path)
        x, these_names = features(data)
        if names is not None and names != these_names:
            raise ValueError("Inconsistent feature definitions")
        names = these_names
        labels = np.asarray(data["labels"])
        stride = max(1, round(float(data["fps"]) / train_fps))
        rows = np.flatnonzero(np.isin(labels, LABELS))[::stride]
        y = np.array([np.flatnonzero(LABELS == v)[0] for v in labels[rows]], dtype=np.int8)
        x = x[rows]
        xs.extend((x, swap_team_features(x, names)))
        ys.extend((y, np.where(y < 2, 1-y, y)))
        manifest.append({"path": str(path), "labelled_sampled_frames": len(rows)})
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    def classifier():
        return HistGradientBoostingClassifier(
            max_iter=iterations, max_leaf_nodes=leaves, learning_rate=0.08,
            min_samples_leaf=35, l2_regularization=1, early_stopping=False,
            random_state=17)
    with threadpool_limits(limits=1):
        if two_stage and 2 in y:
            alive_model = classifier().fit(x, (y != 2).astype(int))
            team_model = classifier().fit(x[y != 2], y[y != 2])
            models = {"alive_estimator": alive_model, "team_estimator": team_model}
            classes = [0, 1, 2]
        else:
            model = classifier().fit(x, y)
            models = {"estimator": model}
            classes = model.classes_.tolist()
    return {**models, "feature_names": names,
            "config": {"iterations": iterations, "leaves": leaves,
                       "train_fps": train_fps, "switch_cost": 0,
                       "two_stage": bool(two_stage and 2 in y)},
            "training": manifest, "classes": classes}


def predict_probabilities(data, model):
    x, names = features(data)
    if names != model["feature_names"]:
        raise ValueError("Feature schema mismatch")
    def infer(values):
        if "alive_estimator" not in model:
            return model["estimator"].predict_proba(values)
        alive = model["alive_estimator"].predict_proba(values)[:, 1]
        team = model["team_estimator"].predict_proba(values)
        return np.column_stack((team[:, 0] * alive, team[:, 1] * alive, 1-alive))
    with threadpool_limits(limits=1):
        p = infer(x)
        swapped = infer(swap_team_features(x, names))
    classes = np.asarray(model["classes"])
    probabilities = np.zeros((len(x), 3), dtype=float)
    for k, label in enumerate(classes):
        reverse = 1-label if label < 2 else label
        reverse_index = int(np.flatnonzero(classes == reverse)[0])
        probabilities[:, label] = (p[:, k] + swapped[:, reverse_index]) / 2
    return probabilities


def decode(probabilities, period, fps, switch_cost=0):
    """Potts/Viterbi smoothing, independently for each period.

    switch_cost is in 10-Hz log-likelihood units and scales with frame rate.
    Classes absent during training remain unavailable, rather than being invented.
    """
    valid_classes = np.flatnonzero(probabilities.sum(axis=0) > 0)
    probabilities = probabilities[:, valid_classes]
    if switch_cost <= 0:
        return LABELS[valid_classes[np.argmax(probabilities, axis=1)]]
    logp = np.log(np.maximum(probabilities, 1e-7))
    penalty = switch_cost * fps / 10
    result = np.empty(len(period), dtype=np.int8)
    for start, stop in blocks(period):
        back = np.zeros((stop-start, len(valid_classes)), dtype=np.int8)
        value = logp[start].copy()
        for row in range(start+1, stop):
            transition = np.broadcast_to(value[:, None], (len(value), len(value))).copy() - penalty
            np.fill_diagonal(transition, value)
            previous = np.argmax(transition, axis=0)
            back[row-start] = previous
            value = transition[previous, np.arange(len(valid_classes))] + logp[row]
        state = int(np.argmax(value))
        for row in range(stop-1, start-1, -1):
            result[row] = valid_classes[state]
            state = int(back[row-start, state])
    return LABELS[result]


def predict(data, model):
    return decode(predict_probabilities(data, model), model_periods(data),
                  float(data["fps"]), model["config"]["switch_cost"])


def score(data, predicted):
    return evaluate_tracking(data["labels"], predicted, period=data["period"],
                             fps=float(data["fps"]), timestamps=data.get("timestamps"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=ROOT/"data/external_audit/sportec/split.json")
    parser.add_argument("--processed", type=Path, default=ROOT/"data/external_audit/sportec/processed")
    parser.add_argument("--out", type=Path, default=ROOT/"data/experiments/supervised")
    parser.add_argument("--test", action="store_true",
                        help="explicitly authorize one held-out evaluation after model selection")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text())
    paths = lambda key: [args.processed / f"{name}.npz" for name in split[key]]
    validation = [(p.stem, load(p)) for p in paths("validation")]
    candidates = []
    best_key, best_model = None, None
    best_macro_key, best_macro_model = None, None
    best_spell_key, best_spell_model = None, None
    # Two multiclass capacities plus a two-stage model, with five temporal
    # penalties. Selection uses validation whole matches only, never held-out test.
    for leaves, iterations, two_stage in [(15, 100, False), (31, 160, False), (31, 160, True)]:
        print(f"Training HGB leaves={leaves} iterations={iterations} two_stage={two_stage}", flush=True)
        model = fit(paths("train"), iterations=iterations, leaves=leaves, two_stage=two_stage)
        validation_probs = [(name, data, predict_probabilities(data, model))
                            for name, data in validation]
        for switch_cost in (0, 3, 8, 16, 32):
            scores = {}
            for name, data, probabilities in validation_probs:
                predicted = decode(probabilities, model_periods(data), float(data["fps"]), switch_cost)
                scores[name] = score(data, predicted)
            mean_spell = float(np.mean([s["spell_f1_mean"] for s in scores.values()]))
            mean_macro = float(np.mean([s["macro_f1_present_classes"] for s in scores.values()]))
            candidate = {"leaves": leaves, "iterations": iterations, "two_stage": two_stage,
                         "switch_cost": switch_cost, "mean_spell_f1": mean_spell,
                         "mean_present_class_macro_f1": mean_macro, "scores": scores}
            candidates.append(candidate)
            print(json.dumps({k:v for k,v in candidate.items() if k != "scores"}), flush=True)
            # User target is high per-frame F1 as well as stable possessions:
            # maximize spellF1 subject to macroF1 >= .90 when attainable.
            key = ((1, mean_spell, mean_macro) if mean_macro >= .9
                   else (0, mean_macro, mean_spell))
            if best_key is None or key > best_key:
                best_key, best_model = key, dict(model)
                best_model["config"] = dict(model["config"], switch_cost=switch_cost)
            if best_macro_key is None or (mean_macro, mean_spell) > best_macro_key:
                best_macro_key, best_macro_model = (mean_macro, mean_spell), dict(model)
                best_macro_model["config"] = dict(model["config"], switch_cost=switch_cost)
            if best_spell_key is None or (mean_spell, mean_macro) > best_spell_key:
                best_spell_key, best_spell_model = (mean_spell, mean_macro), dict(model)
                best_spell_model["config"] = dict(model["config"], switch_cost=switch_cost)
    # Persist frozen model/config BEFORE loading any held-out labels/features.
    joblib.dump(best_model, args.out/"model.joblib")
    joblib.dump(best_macro_model, args.out/"best_macro_model.joblib")
    joblib.dump(best_spell_model, args.out/"best_spell_model.joblib")
    frozen = {"config": best_model["config"], "training": best_model["training"],
              "classes": best_model["classes"], "split": split,
              "selection": "highest validation spellF1 among macroF1 >= .90; otherwise highest macroF1",
              "best_macro_config": best_macro_model["config"],
              "best_spell_config": best_spell_model["config"]}
    (args.out/"frozen_config.json").write_text(json.dumps(frozen, indent=2)+"\n")
    (args.out/"validation.json").write_text(json.dumps(candidates, indent=2)+"\n")
    tests = {}
    if args.test:
        for path in paths("test"):
            data = load(path)
            predicted = predict(data, best_model)
            tests[path.stem] = score(data, predicted)
            np.save(args.out/f"{path.stem}_predictions.npy", predicted)
        (args.out/"test.json").write_text(json.dumps(tests, indent=2)+"\n")
    print("FROZEN", json.dumps(frozen), flush=True)
    for name, result in tests.items():
        print("TEST", name, json.dumps({key:result[key] for key in (
            "accuracy", "macro_f1", "macro_f1_present_classes", "spell_f1_mean",
            "n_gt_spells", "n_pred_spells", "label_coverage", "target_classes_present")}))


if __name__ == "__main__":
    main()
