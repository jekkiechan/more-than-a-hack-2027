"""Geometry emissions + a supervised-transition hidden Markov possession model.

Only physical tracking fields enter features. Possession labels are training targets,
and are never read by ``predict``. Match periods reset sequence inference. Unknown
external targets remain unscored; this does not turn them into dead-ball truth.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.metrics import evaluate_tracking

STATES = np.array(["H", "A", "D"])
SWAP = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])


def load_data(path):
    with np.load(path, allow_pickle=False) as raw:
        return {key: raw[key] for key in raw.files}


def sequence_blocks(data):
    n = len(data["ball"])
    period = data.get("period", np.ones(n, dtype=int))
    timestamp = data.get("timestamps", np.arange(n) / float(data["fps"]))
    delta = np.diff(timestamp)
    broken = (period[1:] != period[:-1]) | (delta <= 0) | (delta > 1.5 / float(data["fps"]))
    bounds = np.r_[0, np.flatnonzero(broken) + 1, n]
    return list(zip(bounds[:-1], bounds[1:]))


def short_interpolate(ball, max_gap):
    """Interpolate bounded short gaps only, never extrapolate ends."""
    out = ball.copy()
    good = np.isfinite(out).all(axis=1)
    indices = np.flatnonzero(good)
    if len(indices) < 2:
        return out
    for left, right in zip(indices[:-1], indices[1:]):
        if 1 < right - left <= max_gap + 1:
            alpha = np.arange(1, right - left)[:, None] / (right - left)
            out[left + 1:right] = out[left] + alpha * (out[right] - out[left])
    return out


def features(data):
    """Twenty symmetric geometry/motion features; no target-derived inputs."""
    fps = float(data["fps"])
    original_ball = np.asarray(data["ball"], dtype=float)
    players = np.asarray(data["players"], dtype=float)
    teams = np.asarray(data["teams"])
    n = len(original_ball)
    ball = original_ball.copy()
    pvelocity = np.full_like(players, np.nan)
    bvelocity = np.full_like(ball, np.nan)
    smooth_speed = np.zeros(n)
    net_ball_speed = np.zeros(n)
    observed = np.isfinite(original_ball).all(axis=1)
    visibility = np.zeros(n)
    for left, right in sequence_blocks(data):
        ball[left:right] = short_interpolate(ball[left:right], max(1, int(round(.5 * fps))))
        if right - left > 1:
            bvelocity[left + 1:right] = np.diff(ball[left:right], axis=0) * fps
            pvelocity[left + 1:right] = np.diff(players[left:right], axis=0) * fps
        lag = max(1, int(round(2 * fps)))
        if right - left > lag:
            net_ball_speed[left + lag:right] = np.linalg.norm(
                ball[left + lag:right] - ball[left:right-lag], axis=1) * fps / lag
        visibility[left:right] = uniform_filter1d(observed[left:right].astype(float),
                                                  size=max(1, int(round(2 * fps))), mode="nearest")
    speed = np.linalg.norm(bvelocity, axis=1)
    raw_player_speed = np.linalg.norm(pvelocity, axis=2)
    # Identity jumps are not physical sprinting. Exclude them from collective motion.
    raw_player_speed[(raw_player_speed > 15) | ~np.isfinite(raw_player_speed)] = np.nan
    player_speed = np.nan_to_num(raw_player_speed, nan=0)
    for left, right in sequence_blocks(data):
        smooth_speed[left:right] = uniform_filter1d(np.clip(np.nan_to_num(speed[left:right]), 0, 45),
                                                    size=max(1, int(round(fps))), mode="nearest")

    # distance and relative-speed slots interleave home/away, allowing an exact team swap.
    side_features = []
    for side in (0, 1):
        selected = teams == side
        points = players[:, selected]
        distance = np.linalg.norm(points - ball[:, None], axis=2)
        distance = np.where(np.isfinite(distance), distance, 150.0)
        if distance.shape[1] == 0:
            raise ValueError("Both teams need at least one track slot")
        nearest_idx = np.argmin(distance, axis=1)
        sorted_d = np.sort(distance, axis=1)
        near = sorted_d[:, 0]
        near2 = sorted_d[:, min(1, sorted_d.shape[1] - 1)]
        near3 = sorted_d[:, min(2, sorted_d.shape[1] - 1)]
        velocity = pvelocity[:, selected][np.arange(n), nearest_idx]
        relative = np.linalg.norm(velocity - bvelocity, axis=1)
        relative = np.nan_to_num(relative, nan=30.0, posinf=30.0)
        local_speed = player_speed[:, selected][np.arange(n), nearest_idx]
        side_features.append([np.log1p(near), np.log1p(near2), np.log1p(near3),
                              np.log1p(np.clip(relative, 0, 50)), np.log1p(local_speed)])
    values = [side_features[side][k] for k in range(5) for side in range(2)]
    pitch = np.asarray(data.get("pitch", [105.0, 68.0]), dtype=float)
    outside = np.maximum.reduce([-ball[:, 0], ball[:, 0] - pitch[0],
                                  -ball[:, 1], ball[:, 1] - pitch[1], np.zeros(n)])
    active_count = np.sum(np.isfinite(raw_player_speed), axis=1)
    mean_player_speed = player_speed.sum(axis=1) / np.maximum(active_count, 1)
    moving_fraction = np.sum(player_speed > 2.0, axis=1) / np.maximum(active_count, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median_player_speed = np.nan_to_num(np.nanmedian(raw_player_speed, axis=1))
    values.extend([np.log1p(np.clip(np.nan_to_num(speed, nan=0), 0, 45)),
                   np.log1p(smooth_speed), visibility, observed.astype(float),
                   np.log1p(np.nan_to_num(outside, nan=0)), np.log1p(mean_player_speed),
                   np.isfinite(ball).all(axis=1).astype(float),
                   np.log1p(median_player_speed), moving_fraction,
                   np.log1p(np.clip(np.nan_to_num(net_ball_speed), 0, 45))])
    out = np.stack(values, axis=1)
    assert out.shape == (n, len(SWAP))
    return np.nan_to_num(out, nan=0, posinf=10, neginf=-10)


def learn_model(train_data, components=4, seed=17):
    rng = np.random.default_rng(seed)
    arrays, targets = [], []
    transition_counts = np.ones((3, 3), dtype=float)
    durations = []
    for data in train_data:
        x = features(data)
        y = np.asarray(data["labels"])
        usable = np.isin(y, STATES)
        # Sampling limits density estimation cost; both sides appear symmetrically.
        selected = np.flatnonzero(usable)[::2]
        arrays.extend([x[selected], x[selected][:, SWAP]])
        targets.extend([y[selected], np.where(y[selected] == "H", "A", np.where(y[selected] == "A", "H", y[selected]))])
        for left, right in sequence_blocks(data):
            for a, b in zip(y[left:right - 1], y[left + 1:right]):
                if a in STATES and b in STATES:
                    i, j = int(np.flatnonzero(STATES == a)[0]), int(np.flatnonzero(STATES == b)[0])
                    transition_counts[i, j] += 1
            durations.append((right-left) / float(data["fps"]))
    x = np.concatenate(arrays)
    y = np.concatenate(targets)
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    states = STATES[np.array([np.any(y == label) for label in STATES])]
    models = []
    for state in states:
        rows = np.flatnonzero(y == state)
        if len(rows) > 25000:
            rows = rng.choice(rows, size=25000, replace=False)
        model = GaussianMixture(n_components=components, covariance_type="full", max_iter=70,
                                reg_covar=.03, random_state=seed, n_init=1)
        model.fit(x[rows])
        models.append(model)
    # Symmetrize H/A dynamics to avoid learning club identity or home advantage.
    transition_counts = (transition_counts + transition_counts[[1, 0, 2]][:, [1, 0, 2]]) / 2
    indices = [int(np.flatnonzero(STATES == s)[0]) for s in states]
    counts = transition_counts[np.ix_(indices, indices)]
    transition = counts / counts.sum(axis=1, keepdims=True)
    return {"scaler": scaler, "models": models, "states": states,
            "transition": transition, "components": components,
            "train_fps": float(train_data[0]["fps"]), "seed": seed}


def emission_log_prob(data, model):
    raw_features = features(data)
    x = model["scaler"].transform(raw_features)
    likelihood = np.stack([m.score_samples(x) for m in model["models"]], axis=1)
    # With a long ball gap, marginalize absent ball features instead of pretending
    # that a zero-speed, 150-metre-distant synthetic ball was observed. Only
    # collective player motion remains as dead/live evidence in these frames.
    missing = raw_features[:, 16] == 0
    if missing.any():
        keep = np.array([15, 17, 18])
        partial = x[missing][:, keep]
        for column, mixture in enumerate(model["models"]):
            components = []
            for k, weight in enumerate(mixture.weights_):
                covariance = mixture.covariances_[k][np.ix_(keep, keep)]
                components.append(np.log(weight) + multivariate_normal.logpdf(
                    partial, mean=mixture.means_[k, keep], cov=covariance))
            likelihood[missing, column] = logsumexp(np.stack(components, axis=1), axis=1)
    # Limit the impact of a single outlier frame on the sequence decoder.
    likelihood -= likelihood.max(axis=1, keepdims=True)
    return np.maximum(likelihood, -15.0)


def viterbi(log_emission, log_transition):
    n, nstates = log_emission.shape
    back = np.zeros((n, nstates), dtype=np.int8)
    score = log_emission[0].copy()
    for i in range(1, n):
        choices = score[:, None] + log_transition
        previous = np.argmax(choices, axis=0)
        back[i] = previous
        score = choices[previous, np.arange(nstates)] + log_emission[i]
        score -= score.max()
    path = np.empty(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(data, model, log_emission, transition_scale=1.0, dead_transition_scale=None):
    transition = model["transition"].copy()
    # Preserve per-second transition rates when running at a different frame rate.
    ratio = model["train_fps"] / float(data["fps"])
    for i in range(len(transition)):
        stay = transition[i, i] ** ratio
        others = np.arange(len(transition)) != i
        denom = transition[i, others].sum()
        transition[i, others] *= (1-stay) / denom if denom else 0
        transition[i, i] = stay
    log_transition = np.log(np.maximum(transition, 1e-9)) * transition_scale
    if dead_transition_scale is not None and "D" in model["states"]:
        dead = int(np.flatnonzero(model["states"] == "D")[0])
        for other in range(len(transition)):
            if other != dead:
                log_transition[dead, other] = np.log(max(transition[dead, other], 1e-9)) * dead_transition_scale
                log_transition[other, dead] = np.log(max(transition[other, dead], 1e-9)) * dead_transition_scale
    result = np.full(len(log_emission), "X", dtype="<U1")
    for left, right in sequence_blocks(data):
        if right > left:
            result[left:right] = model["states"][viterbi(log_emission[left:right], log_transition)]
    return result


def predict(data, config=None):
    if config is None or "model_path" not in config:
        raise ValueError("Provide the frozen config with model_path")
    model = joblib.load(config["model_path"])
    with threadpool_limits(limits=1):
        return decode(data, model, emission_log_prob(data, model), config.get("transition_scale", 1.0),
                      config.get("dead_transition_scale"))


def score(data, prediction):
    truth = np.asarray(data["labels"])
    metric = evaluate_tracking(truth, prediction, period=data["period"],
                               fps=float(data["fps"]), timestamps=data["timestamps"])
    supported = [label for label in STATES if np.any(truth == label)]
    metric["supported_classes"] = list(supported)
    metric["supported_class_macro_f1"] = float(np.mean([metric["per_class"][label]["f1"] for label in supported]))
    metric["unknown_target_fraction"] = float(np.mean(truth == "X"))
    return metric


def experiment(split_path, output, evaluate_test=False):
    split = json.loads(Path(split_path).read_text())
    base = Path(split_path).parent / "processed"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train = [load_data(base / f"{name}.npz") for name in split["train"]]
    validation = [load_data(base / f"{name}.npz") for name in split["validation"]]
    trials = []
    best = None
    with threadpool_limits(limits=1):
        for components in (2, 4, 8):
            model = learn_model(train, components=components)
            path = output / f"gmm_{components}.joblib"
            joblib.dump(model, path)
            emissions = [emission_log_prob(data, model) for data in validation]
            for scale in (0.0, .5, 1.0, 2.0, 4.0, 8.0):
                metrics = [score(data, decode(data, model, emit, scale))
                           for data, emit in zip(validation, emissions)]
                trial = {"components": components, "transition_scale": scale,
                         "validation_spell_f1": float(np.mean([m["spell_f1_mean"] for m in metrics])),
                         "validation_supported_macro_f1": float(np.mean([m["supported_class_macro_f1"] for m in metrics]))}
                trials.append(trial)
                print(json.dumps(trial), flush=True)
                macro, spell = trial["validation_supported_macro_f1"], trial["validation_spell_f1"]
                # User goal: first meet .90 frame macro F1, then improve spell F1.
                key = (macro >= .9, spell if macro >= .9 else macro,
                       macro if macro >= .9 else spell)
                if best is None or key > best[0]:
                    best = (key, {"model_path": str(path.resolve()), "transition_scale": scale,
                                  "components": components}, metrics)
    config = best[1]
    (output / "frozen_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "validation_trials.json").write_text(json.dumps(trials, indent=2) + "\n")
    # No further model selection occurs after this point.
    results = {"config": config, "split": split, "validation": best[2], "test": {}}
    for name in split["test"] if evaluate_test else []:
        data = load_data(base / f"{name}.npz")
        prediction = predict(data, config)
        metric = score(data, prediction)
        results["test"][name] = metric
        np.save(output / f"{name}_predictions.npy", prediction)
        print(json.dumps({"test_match": name, "spell_f1": metric["spell_f1_mean"],
                          "supported_macro_f1": metric["supported_class_macro_f1"],
                          "official_macro_f1": metric["macro_f1"]}), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="data/external/split.json")
    parser.add_argument("--out", default="data/experiments/temporal")
    parser.add_argument("--evaluate-test", action="store_true", help="Only after freezing the primary benchmark")
    args = parser.parse_args()
    experiment(args.split, args.out, args.evaluate_test)
