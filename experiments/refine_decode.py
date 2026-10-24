"""Two bounded validation-only decoder refinements of the frozen robust model.

Round 1 separates team-switch and dead/alive costs. Round 2 uses a training-only
boundary classifier to lower the team-switch cost near plausible turnovers.
The frame classifier stays unchanged. No held-out match is loaded by this CLI.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import maximum_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from experiments import robust, supervised
from experiments.stress import mask_ball

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external_audit/sportec/processed"
TRAIN = ("J03WMX", "J03WN1", "J03WOH")
VALIDATION = "J03WOY"
ROUND1_TEAM_COSTS = (0., 1., 3., 8.)
ROUND1_DEAD_COSTS = (1., 3., 8.)
ROUND2_TEAM_COSTS = (3., 8.)
ROUND2_POWERS = (1., 2.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probabilities_from_features(x, model):
    """Exactly the frozen robust classifier's team-symmetrized probabilities."""
    with threadpool_limits(limits=1):
        original = model["estimator"].predict_proba(x)
        swapped = model["estimator"].predict_proba(
            supervised.swap_team_features(x, model["feature_names"]))
    if model["classes"] != [0, 1, 2]:
        raise ValueError("Refinement expects the frozen three-class robust model")
    return (original + swapped[:, [1, 0, 2]]) / 2


def shift(values, frames, periods):
    """Return x[t + frames], clipped within period/time-contiguous blocks."""
    out = np.empty_like(values)
    for start, stop in supervised.blocks(periods):
        indices = np.clip(np.arange(start, stop) + frames, start, stop-1)
        out[start:stop] = values[indices]
    return out


def boundary_features(x, names, probabilities, periods, fps):
    """Team-invariant geometric and posterior context; no target fields."""
    index = {name: i for i, name in enumerate(names)}
    columns, feature_names = [], []
    def add(name, values):
        columns.append(values)
        feature_names.append(name)
    def feature(name):
        return x[:, index[name]]
    for name in ("ball_available", "ball_observed", "ball_speed",
                 "missing_before_s", "missing_after_s", "ball_pitch_edge_distance",
                 "player_speed_median", "player_running_fraction"):
        add(name, feature(name))
    for rank in (1, 2, 3):
        home, away = feature(f"H_distance{rank}"), feature(f"A_distance{rank}")
        add(f"minimum_distance{rank}", np.minimum(home, away))
        add(f"distance_difference{rank}", np.abs(home-away))
    for radius in (1., 2., 3.):
        before = feature(f"home_contact_before_{radius}m")
        after = feature(f"home_contact_after_{radius}m")
        add(f"contact_team_agreement_{radius}", before * after)
        add(f"contact_age_before_{radius}", feature(f"contact_age_before_{radius}m"))
        add(f"contact_age_after_{radius}", feature(f"contact_age_after_{radius}m"))
    logodds = np.log(np.maximum(probabilities[:, 0], 1e-7)) - np.log(np.maximum(probabilities[:, 1], 1e-7))
    add("team_confidence", np.abs(logodds))
    add("alive_probability", 1-probabilities[:, 2])
    advantage = feature("home_distance_advantage")
    for seconds in (.12, .3, .6, 1.):
        frames = max(1, round(seconds * fps))
        left, right = shift(logodds, -frames, periods), shift(logodds, frames, periods)
        add(f"posterior_agreement_{seconds}", left * right)
        add(f"posterior_change_{seconds}", np.abs(left-right))
        add(f"posterior_min_confidence_{seconds}", np.minimum(np.abs(left), np.abs(right)))
        left, right = shift(advantage, -frames, periods), shift(advantage, frames, periods)
        add(f"geometry_agreement_{seconds}", left * right)
        add(f"geometry_change_{seconds}", np.abs(left-right))
    matrix = np.column_stack(columns).astype(np.float32)
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix, feature_names


def decode(probabilities, periods, fps, config, boundary_probability=None):
    """Viterbi with symmetric H/A costs and independent D transition costs."""
    logp = np.log(np.maximum(probabilities, 1e-7))
    team_penalty = np.full(len(logp), float(config["team_cost"]) * fps / 10)
    if boundary_probability is not None:
        team_penalty *= np.maximum(.05, (1-boundary_probability) ** config["boundary_power"])
    dead_penalty = float(config["dead_cost"]) * fps / 10
    result = np.empty(len(logp), dtype=np.int8)
    for start, stop in supervised.blocks(periods):
        back = np.zeros((stop-start, 3), dtype=np.int8)
        value = logp[start].copy()
        for row in range(start+1, stop):
            costs = np.array([[0., team_penalty[row], dead_penalty],
                              [team_penalty[row], 0., dead_penalty],
                              [dead_penalty, dead_penalty, 0.]])
            transition = value[:, None] - costs
            previous = np.argmax(transition, axis=0)
            back[row-start] = previous
            value = transition[previous, np.arange(3)] + logp[row]
        state = int(np.argmax(value))
        for row in range(stop-1, start-1, -1):
            result[row] = state
            state = int(back[row-start, state])
    return supervised.LABELS[result]


def predict(data, model, config, boundary_model=None):
    """Predict from tracking + frozen robust model + frozen refinement config."""
    if "config" in config:
        config = config["config"]
    data = robust.tracking_inputs(data)
    x, names = supervised.features(data)
    probabilities = probabilities_from_features(x, model)
    period = supervised.model_periods(data)
    boundary = None
    if config.get("boundary_power", 0):
        if boundary_model is None:
            boundary_model = joblib.load(ROOT / config["boundary_model_path"])
        bx, bn = boundary_features(x, names, probabilities, period, float(data["fps"]))
        if bn != boundary_model["feature_names"]:
            raise ValueError("Boundary feature mismatch")
        with threadpool_limits(limits=1):
            boundary = boundary_model["estimator"].predict_proba(bx)[:, 1]
    return decode(probabilities, period, float(data["fps"]), config, boundary)


def compact_score(data, predictions):
    result = supervised.score(data, predictions)
    return {key: result[key] for key in (
        "macro_f1", "spell_f1_mean", "accuracy", "n_gt_spells", "n_pred_spells",
        "spell_f1", "per_class", "gt_home_share", "pred_home_share", "share_err")}


def key(trial):
    macro, spell = trial["validation"]["macro_f1"], trial["validation"]["spell_f1_mean"]
    return (1, spell, macro) if macro >= .9 else (0, macro, spell)


def fit_boundary(model, out):
    xs, ys, names, training = [], [], None, []
    for match in TRAIN:
        data = supervised.load(BASE / f"{match}.npz")
        period, fps = supervised.model_periods(data), float(data["fps"])
        inputs = robust.tracking_inputs(data)
        x, feature_names = supervised.features(inputs)
        probabilities = probabilities_from_features(x, model)
        bx, names = boundary_features(x, feature_names, probabilities, period, fps)
        labels = np.asarray(data["labels"])
        live = np.isin(labels, ["H", "A"])
        transition = np.r_[False, live[1:] & live[:-1] & (labels[1:] != labels[:-1]) & (period[1:] == period[:-1])]
        target = np.zeros(len(labels), dtype=bool)
        radius = max(1, round(.16 * fps))
        for start, stop in supervised.blocks(period):
            target[start:stop] = maximum_filter1d(transition[start:stop].astype(np.int8), 2*radius+1, mode="constant") > 0
        # Include all positive windows plus every third negative live frame.
        rows = np.flatnonzero(live & (target | (np.arange(len(labels)) % 3 == 0)))
        xs.append(bx[rows])
        ys.append(target[rows].astype(np.int8))
        training.append({"match": match, "n_sampled": len(rows),
                         "n_positive": int(target[rows].sum()), "source_sha256": digest(BASE / f"{match}.npz")})
        print("boundary_training", match, len(rows), int(target[rows].sum()), flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    weights = np.where(y == 1, .5 / np.mean(y == 1), .5 / np.mean(y == 0))
    estimator = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15,
        learning_rate=.08, min_samples_leaf=35, l2_regularization=2,
        early_stopping=False, random_state=29)
    with threadpool_limits(limits=1):
        estimator.fit(x, y, sample_weight=weights)
    result = {"estimator": estimator, "feature_names": names, "training": training,
              "target": "Within .16 seconds of a true H/A turnover, training matches only",
              "warning": "Class-balanced scores are boundary evidence, not calibrated probabilities."}
    joblib.dump(result, out / "boundary.joblib")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/refine_decode")
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen_config.json").exists():
        raise RuntimeError("Refinement already frozen; do not replace after held-out evaluation")
    model_path = ROOT / "data/experiments/robust/model.joblib"
    model = joblib.load(model_path)
    data = supervised.load(BASE / f"{VALIDATION}.npz")
    inputs = robust.tracking_inputs(data)
    x, names = supervised.features(inputs)
    probabilities = probabilities_from_features(x, model)
    period, fps = supervised.model_periods(data), float(data["fps"])
    np.save(out / "validation_probabilities.npy", probabilities)
    trials = []
    def evaluate(config, round_name, boundary=None):
        pred = decode(probabilities, period, fps, config, boundary)
        trial = {"round": round_name, "config": config, "validation": compact_score(data, pred)}
        trials.append(trial)
        print(json.dumps({"round": round_name, "config": config,
                          "macro_f1": trial["validation"]["macro_f1"],
                          "spell_f1_mean": trial["validation"]["spell_f1_mean"]}), flush=True)
        (out / "validation.json").write_text(json.dumps(trials, indent=2)+"\n")
        return trial
    baseline_config = {"team_cost": 3., "dead_cost": 3., "boundary_power": 0.}
    baseline = evaluate(baseline_config, "baseline")
    expected = supervised.decode(probabilities, period, fps, 3.)
    assert np.array_equal(decode(probabilities, period, fps, baseline_config), expected)
    for team in ROUND1_TEAM_COSTS:
        for dead in ROUND1_DEAD_COSTS:
            config = {"team_cost": team, "dead_cost": dead, "boundary_power": 0.}
            if config != baseline_config:
                evaluate(config, "1_separate_costs")
    round1_best = max(trials, key=key)
    boundary_model = fit_boundary(model, out)
    bx, bn = boundary_features(x, names, probabilities, period, fps)
    assert bn == boundary_model["feature_names"]
    with threadpool_limits(limits=1):
        boundary = boundary_model["estimator"].predict_proba(bx)[:, 1]
    np.save(out / "validation_boundary_evidence.npy", boundary)
    boundary_path = str((out / "boundary.joblib").relative_to(ROOT))
    for team in ROUND2_TEAM_COSTS:
        for power in ROUND2_POWERS:
            config = {"team_cost": team, "dead_cost": round1_best["config"]["dead_cost"],
                      "boundary_power": power, "boundary_model_path": boundary_path}
            evaluate(config, "2_adaptive_boundary", boundary)
    best = max(trials, key=key)
    config = best["config"]
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "selection": "Max clean validation spell F1 among frame macro F1 >= .9; else highest macro F1",
              "rounds": 2, "n_trials_including_baseline": len(trials),
              "training_matches": list(TRAIN), "validation_match": VALIDATION,
              "robust_model_path": str(model_path.relative_to(ROOT)),
              "robust_model_sha256": digest(model_path), "boundary_model_sha256": digest(out / "boundary.joblib"),
              "source_sha256": digest(__file__),
              "baseline_validation": baseline["validation"], "selected_validation": best["validation"],
              "round1_best": round1_best, "test_evaluated": False,
              "masked_validation_used_for_selection": False}
    (out / "frozen_config.json").write_text(json.dumps(frozen, indent=2)+"\n")
    masked = mask_ball(inputs, missing_rate=.41, mean_gap_s=1., seed=2026)
    masked_pred = predict(masked, model, config, boundary_model)
    masked_score = compact_score(data, masked_pred)
    masked_score["ball_missing_fraction"] = float(np.mean(~np.isfinite(masked["ball"]).all(axis=1)))
    (out / "validation_stress.json").write_text(json.dumps(masked_score, indent=2)+"\n")
    print("FROZEN", json.dumps({"config": config, "macro_f1": best["validation"]["macro_f1"],
          "spell_f1_mean": best["validation"]["spell_f1_mean"], "masked_macro_f1": masked_score["macro_f1"],
          "masked_spell_f1_mean": masked_score["spell_f1_mean"]}), flush=True)


if __name__ == "__main__":
    main()
