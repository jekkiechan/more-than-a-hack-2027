"""Fit the tactical H/A expert on original TRAIN matches (whole-match/provider folds).

Rows: 5 Hz live H/A frames of each TRAIN match, clean plus one `feed_like_v2` view, each
direct and team-swapped. Equal weight per provider, then match, then view.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from experiments import feed_like_v2, supervised, tactical_ha_features as tactical, tracking_corruption

ROOT = Path(__file__).resolve().parents[1]
TRAIN = {"Sportec": ["J03WMX", "J03WN1", "J03WOH"], "Metrica": ["metrica_1", "metrica_2"],
         "SkillCorner": ["skillcorner_had_" + m for m in
                         "1874553 1886347 1899585 1925299 1927964 1953632 1959846 1986691".split()]}
VIEWS = ("clean", "v2a")
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, early_stopping=False)
SKILLCORNER_MIN_VISIBLE = 12


def graded_labels(provider, raw):
    labels = np.asarray(raw["labels"]).astype("U1").copy()
    if provider == "SkillCorner":
        labels[np.asarray(raw["player_observed"]).sum(axis=1) < SKILLCORNER_MIN_VISIBLE] = "X"
    return labels


def calibration():
    return json.loads((ROOT / "experiments/results/feed_like_v2_calibration.json").read_text())["calibration"]


def rows(provider, match, view, raw):
    labels = graded_labels(provider, raw)
    inputs = tracking_corruption.inputs_only(raw)
    if view != "clean":
        seed = int(hashlib.sha256(f"tactical_ha_v1/train/{view}/{match}".encode()).hexdigest()[:8], 16)
        inputs = feed_like_v2.corrupt(inputs, labels, calibration(), seed)
    direct, names = tactical.features(inputs)
    swapped, _ = tactical.features(tactical.swap_inputs(inputs))
    index = tactical.sample_indices(len(labels), float(inputs["fps"]), supervised.model_periods(inputs))
    index = index[np.isin(labels[index], ["H", "A"])]
    return direct[index], swapped[index], labels[index] == "H", names


def fit(tables, exclude=None, seed=2026):
    """tables: {(provider, match, view): (direct, swapped, is_home, names)}."""
    providers = [p for p in TRAIN if p != exclude]
    xs, ys, ws, names = [], [], [], None
    for provider in providers:
        for match in TRAIN[provider]:
            for view in VIEWS:
                direct, swapped, home, names = tables[provider, match, view]
                xs.append(np.concatenate([direct, swapped]))
                ys.append(np.concatenate([home, ~home]))
                ws.append(np.full(2 * len(home), 1.0 / (len(providers) * len(TRAIN[provider]) * len(VIEWS) * 2 * len(home))))
    weight = np.concatenate(ws)
    weight *= len(weight) / weight.sum()
    estimator = HistGradientBoostingClassifier(random_state=seed, **PARAMS).fit(
        np.concatenate(xs), np.concatenate(ys), sample_weight=weight)
    return {"estimator": estimator, "names": list(names), "exclude": exclude, "seed": seed,
            "params": PARAMS, "views": VIEWS, "train": {p: TRAIN[p] for p in providers}}
