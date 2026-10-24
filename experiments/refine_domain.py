"""Player-only dead-ball head to measure dependence on frozen ball coordinates.

Fits only on original external training matches. Clean validation selects a
bounded blend/decoder grid; missing-ball validation remains supplementary.
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
import sys
import warnings

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import supervised
from experiments.export_reports import aggregates
from experiments.robust import tracking_inputs
from experiments.stress import mask_ball

WEIGHTS = (0., .25, .5, .75, 1.)
COSTS = (1, 3, 8)
CAPACITIES = ((31, 160), (63, 260))


def row_quantiles(values, qs):
    """Vectorized NumPy linear NaN-quantiles for short player rows."""
    ordered = np.sort(values, axis=1)
    count = np.isfinite(ordered).sum(axis=1)
    q = np.asarray(qs, dtype=float)[:, None]
    index = np.maximum(count-1, 0)[None, :]*q
    low = np.floor(index).astype(int)
    high = np.minimum(low+1, np.maximum(count-1, 0)[None, :])
    alpha = index-low
    rows = np.arange(len(values))[None, :]
    return ordered[rows, low]*(1-alpha) + ordered[rows, high]*alpha


def features(data):
    """Collective-player features; never inspect ball, possession or labels.

The D head is invariant to home/away names and roster-slot order. Counts are
fractions of observed players so padded roster length cannot reveal the source.
"""
    players = np.asarray(data["players"], dtype=float)
    period = supervised.model_periods(data)
    fps = float(data["fps"])
    pitch = np.asarray(data.get("pitch", [105., 68.]), dtype=float)
    velocity = np.full_like(players, np.nan)
    for start, stop in supervised.blocks(period):
        if stop - start > 1:
            velocity[start:stop] = np.gradient(players[start:stop], axis=0) * fps
    speed = np.linalg.norm(velocity, axis=2)
    bad = (speed > 12) | ~np.isfinite(speed)
    speed[bad] = np.nan
    velocity[bad] = np.nan
    visible = np.isfinite(players).all(axis=2)
    speed_count = np.isfinite(speed).sum(axis=1)
    count = visible.sum(axis=1)
    columns, names = [], []

    def add(name, values, windows=(.3, 1., 3.)):
        names.append(name)
        columns.append(values)
        for seconds in windows:
            names.append(f"{name}_mean_{seconds}s")
            columns.append(supervised._rolling(values, 2*round(seconds*fps)+1, period))

    def fraction(mask, denominator):
        return np.divide(mask.sum(axis=1), denominator,
                         out=np.full(len(players), np.nan), where=denominator > 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(speed, axis=1)
        add("speed_mean", mean_speed)
        add("speed_std", np.nanstd(speed, axis=1))
        for q, value in zip((.25, .5, .75, .9), row_quantiles(speed, (.25, .5, .75, .9))):
            add(f"speed_q{q}", value)
        for threshold in (.5, 1., 2.):
            add(f"slow_fraction_{threshold}", fraction(speed < threshold, speed_count))
        for threshold in (4., 6.):
            add(f"fast_fraction_{threshold}", fraction(speed > threshold, speed_count))
        mean_v = np.nanmean(velocity, axis=1)
        add("motion_coherence", np.linalg.norm(mean_v, axis=1) / (mean_speed + .1))
        add("valid_motion_fraction", np.divide(speed_count, count,
                                               out=np.full(len(players), np.nan), where=count > 0))
        # Pitch-scale shape, unaffected by direction of play or team names.
        for axis in (0, 1):
            lo, hi = row_quantiles(players[:, :, axis], (.1, .9))
            add(f"spread_axis{axis}", (hi-lo)/pitch[axis], windows=(1., 3.))
        center = np.nanmean(players, axis=1)
        spread = np.linalg.norm(players-center[:, None, :], axis=2)
        add("radial_spread", np.nanmedian(spread, axis=1), windows=(1., 3.))
        edge = np.minimum(np.minimum(players[:, :, 0], pitch[0]-players[:, :, 0]),
                          np.minimum(players[:, :, 1], pitch[1]-players[:, :, 1]))
        add("touchline_fraction", fraction((edge < 5) & visible, count), windows=(1., 3.))
        # Team-centroid separation is symmetric and useful during set plays.
        teams = np.asarray(data["teams"])
        means = [np.nanmean(players[:, teams == team], axis=1) for team in (0, 1)]
        add("team_centroid_distance", np.linalg.norm(means[0]-means[1], axis=1), windows=(1., 3.))
    x = np.column_stack(columns).astype(np.float32)
    x[~np.isfinite(x)] = np.nan
    return x, names


def drop_player_slots(data, seed):
    """Random missing players per period, independent of all target fields."""
    out = {key: data[key] for key in ("players", "teams", "period", "fps", "timestamps", "pitch") if key in data}
    out["players"] = np.array(data["players"], copy=True)
    rng = np.random.default_rng(seed)
    for start, stop in supervised.blocks(supervised.model_periods(data)):
        active = np.isfinite(out["players"][start:stop]).any(axis=(0, 2))
        for team in (0, 1):
            slots = np.flatnonzero(active & (data["teams"] == team))
            remove = rng.choice(slots, size=min(max(0, len(slots)-5), round(.25*len(slots))), replace=False)
            out["players"][start:stop, remove] = np.nan
    return out


def training_arrays(paths):
    xs, ys, names = [], [], None
    for i, path in enumerate(paths):
        data = supervised.load(path)
        rows = np.flatnonzero(np.isin(data["labels"], ["H", "A", "D"]))[::6]
        for variant in (data, drop_player_slots(data, 9026+i)):
            x, names = features(variant)
            xs.append(x[rows])
            ys.append((data["labels"][rows] == "D").astype(np.int8))
    x, y = np.concatenate(xs), np.concatenate(ys)
    return x, y, names


def fit_head(training, leaves, iterations):
    x, y, names = training
    estimator = HistGradientBoostingClassifier(
        max_iter=iterations, max_leaf_nodes=leaves, learning_rate=.08,
        min_samples_leaf=40, l2_regularization=1, early_stopping=False, random_state=31)
    with threadpool_limits(limits=1):
        estimator.fit(x, y)
    return {"dead_estimator": estimator, "feature_names": names,
            "config": {"leaves": leaves, "iterations": iterations,
                       "ball_dead_weight": 1., "switch_cost": 3},
            "training_rows": len(y)}


def player_dead_probability(data, model):
    x, names = features(data)
    if names != model["feature_names"]:
        raise ValueError("Player feature schema mismatch")
    with threadpool_limits(limits=1):
        return model["dead_estimator"].predict_proba(x)[:, 1]


def combine(base, player_dead, weight):
    alive = base[:, :2].sum(axis=1)
    team = np.divide(base[:, :2], alive[:, None], out=np.full_like(base[:, :2], .5),
                     where=alive[:, None] > 1e-12)
    dead = weight*base[:, 2] + (1-weight)*player_dead
    return np.column_stack((team*(1-dead[:, None]), dead))


def predict_probabilities(data, model):
    base = supervised.predict_probabilities(tracking_inputs(data), model["base_model"])
    head = player_dead_probability(data, model)
    return combine(base, head, model["config"]["ball_dead_weight"])


def predict(data, model):
    return supervised.decode(predict_probabilities(data, model), supervised.model_periods(data),
                             float(data["fps"]), model["config"]["switch_cost"])


def selection(row):
    macro, spell = row["clean"]["macro_f1"], row["clean"]["spell_f1_mean"]
    return (1, spell, macro) if macro >= .90 else (0, macro, spell)


def run():
    out = ROOT / "data/experiments/refine_domain"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "frozen_config.json").exists():
        raise RuntimeError("Output already frozen")
    split = json.loads((ROOT / "data/external_audit/sportec/split.json").read_text())
    base_path = ROOT / "data/experiments/robust/model.joblib"
    base_model = joblib.load(base_path)
    directory = ROOT / "data/external_audit/sportec/processed"
    val = supervised.load(directory / f"{split['validation'][0]}.npz")
    masked = mask_ball(tracking_inputs(val), seed=2026)
    masked["labels"] = val["labels"]
    views = {"clean": val, "masked": masked}
    base_prob = {name: supervised.predict_probabilities(tracking_inputs(data), base_model)
                 for name, data in views.items()}
    candidates, models = [], []
    training = training_arrays([directory/f"{name}.npz" for name in split["train"]])
    for leaves, iterations in CAPACITIES:
        print("FIT PLAYER HEAD", leaves, iterations, flush=True)
        model = fit_head(training, leaves, iterations)
        model["base_model"] = base_model
        models.append(model)
        head = player_dead_probability(val, model)
        # Ball masking cannot change these player-only head probabilities.
        for weight in WEIGHTS:
            probs = {name: combine(p, head, weight) for name, p in base_prob.items()}
            for cost in COSTS:
                row = {"leaves": leaves, "iterations": iterations,
                       "ball_dead_weight": weight, "switch_cost": cost}
                for name, data in views.items():
                    pred = supervised.decode(probs[name], supervised.model_periods(data), float(data["fps"]), cost)
                    row[name] = aggregates(supervised.score(data, pred))
                candidates.append(row)
                print(json.dumps({**{k:v for k,v in row.items() if k not in views},
                                  **{name:{k:row[name][k] for k in ("macro_f1", "spell_f1_mean", "alive_dead_f1")}
                                     for name in views}}), flush=True)
    best = max(candidates, key=selection)
    pure = max([row for row in candidates if row["ball_dead_weight"] == 0], key=selection)
    for label, row in (("model", best), ("player_only_model", pure)):
        model = next(m for m in models if m["config"]["leaves"] == row["leaves"])
        model = dict(model, config={k:row[k] for k in ("leaves", "iterations", "ball_dead_weight", "switch_cost")})
        joblib.dump(model, out / f"{label}.joblib")
    report = {"selection": "max clean validation spell among macro>=.90; otherwise max macro",
              "selected": best, "player_only_diagnostic": pure, "candidates": candidates}
    (out / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "split": split,
              "config": {k:best[k] for k in ("leaves", "iterations", "ball_dead_weight", "switch_cost")},
              "player_only_config": {k:pure[k] for k in ("leaves", "iterations", "ball_dead_weight", "switch_cost")},
              "test_accessed": False, "competition_gt_accessed": False,
              "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (Path(__file__), base_path, out/"model.joblib", out/"player_only_model.joblib")}}
    (out / "frozen_config.json").write_text(json.dumps(frozen, indent=2)+"\n")
    print("FROZEN", json.dumps(frozen), flush=True)


if __name__ == "__main__":
    run()
