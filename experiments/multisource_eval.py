"""Multi-source validation suite for the full H/A/D model.

The fork's model selection used one Sportec validation match (J03WOY), whose
frozen-ball dead periods make D nearly free. This scores the same refined model
family on three providers with the official metric, and asks whether adding
broadcast-like training sources helps on providers the model has not seen.

Evaluation splits are those fixed in ``dead_transfer.py``. Views are
challenge-like: SkillCorner positions are NaN wherever SkillCorner did not
detect the object (the challenge's convention) and SkillCorner frames with fewer
than ``dead_transfer.MIN_VISIBLE`` detected players are ungraded (X) so the
broadcast close-up shortcut is neither learned nor rewarded. D comes either from
the full model or from a player-only head trained on the same sources.

Model: ``refine_features`` physical features, clean + 41% Markov ball-masked
copies, team-swap augmentation, HGB 63 leaves / 260 iterations (the refined
capacity), Viterbi decoding at fixed switch costs. No challenge data is read.
"""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import dead_transfer, refine_domain, refine_features, robust, supervised  # noqa: E402

MIXTURES = (("sportec",), ("sportec", "metrica"), ("sportec", "skillcorner"),
            ("sportec", "metrica", "skillcorner"))
SWITCH_COSTS = (3, 8)
TRAIN_HZ = 4


def view(path):
    """Challenge-like tracking inputs and graded labels for one match."""
    data = supervised.load(path)
    labels = np.array(data["labels"], copy=True)
    inputs = robust.tracking_inputs(data)
    if path.name.startswith("skillcorner"):
        ball = np.array(inputs["ball"], copy=True)
        ball[~data["ball_observed"]] = np.nan
        players = np.array(inputs["players"], copy=True)
        players[~data["player_observed"]] = np.nan
        inputs = dict(inputs, ball=ball, players=players,
                      player_observed=data["player_observed"].copy())
        labels[data["player_observed"].sum(axis=1) < dead_transfer.MIN_VISIBLE] = "X"
    return inputs, labels


def training_rows(paths, seed):
    xs, ys, ws, names = [], [], [], None
    for i, path in enumerate(paths):
        inputs, labels = view(path)
        stride = max(1, round(float(inputs["fps"]) / TRAIN_HZ))
        rows = np.flatnonzero(np.isin(labels, supervised.LABELS))[::stride]
        y = np.array([np.flatnonzero(supervised.LABELS == v)[0] for v in labels[rows]], dtype=np.int8)
        for variant in (inputs, robust.masked_inputs(inputs, seed + i)):
            x, names = refine_features.features(variant)
            x = x[rows]
            xs.extend((x, supervised.swap_team_features(x, names)))
            ys.extend((y, np.where(y < 2, 1 - y, y)))
            ws.extend((np.ones(len(y)), np.ones(len(y))))
    return np.concatenate(xs), np.concatenate(ys), names


def fit(mixture, cache):
    xs, ys, ws = [], [], []
    for source in mixture:
        x, y, names = cache[source]
        xs.append(x)
        ys.append(y)
        ws.append(np.full(len(y), 1.0 / len(y)))
    w = np.concatenate(ws)
    w *= len(w) / w.sum()
    estimator = HistGradientBoostingClassifier(max_iter=260, max_leaf_nodes=63, learning_rate=.08,
                                               min_samples_leaf=35, l2_regularization=1,
                                               early_stopping=False, random_state=17)
    with threadpool_limits(limits=1):
        estimator.fit(np.concatenate(xs), np.concatenate(ys), sample_weight=w)
    return {"estimator": estimator, "feature_names": names, "classes": estimator.classes_.tolist()}


def fit_dead_head(mixture, dead_cache):
    x = np.concatenate([dead_cache[s][0] for s in mixture])
    y = np.concatenate([dead_cache[s][1] for s in mixture])
    w = np.concatenate([np.full(len(dead_cache[s][1]), 1.0 / len(dead_cache[s][1])) for s in mixture])
    return dead_transfer.fit(x, y, w * len(w) / w.sum())


def score(inputs, labels, probabilities, cost):
    predictions = supervised.decode(probabilities, supervised.model_periods(inputs), float(inputs["fps"]), cost)
    result = supervised.score(dict(inputs, labels=labels), predictions)
    per_class = result["per_class"]
    return {"macro_f1": result["macro_f1"], "spell_f1": result["spell_f1_mean"],
            "d_f1": per_class.get("D", {}).get("f1"), "team_acc_alive": result["team_acc_gt_alive"],
            "n_gt_spells": result["n_gt_spells"], "n_pred_spells": result["n_pred_spells"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/multisource_eval.json")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/experiments/multisource")
    args = parser.parse_args()
    started = time.monotonic()
    args.cache.mkdir(parents=True, exist_ok=True)
    cache, dead_cache = {}, {}
    for source, split in dead_transfer.SPLITS.items():
        path = args.cache / f"{source}_train.joblib"
        if path.exists():
            cache[source], dead_cache[source] = joblib.load(path)
        else:
            cache[source] = training_rows(split["train"], 2026)
            dead_cache[source] = dead_transfer.matrices(split["train"], 9026, augment=True)
            joblib.dump((cache[source], dead_cache[source]), path)
        print(f"PREPARED {source} rows={len(cache[source][1])}", flush=True)
    evaluation = []
    for source, split in dead_transfer.SPLITS.items():
        for path in split["eval"]:
            inputs, labels = view(path)
            x, names = refine_features.features(inputs)
            px, _ = refine_domain.features(inputs)
            evaluation.append((source, path.stem, inputs, labels, x, px))
    rows = json.loads(args.out.read_text())["rows"] if args.out.exists() else []
    for mixture in MIXTURES:
        if any(r["train"] == "+".join(mixture) for r in rows):
            continue
        model = fit(mixture, cache)
        head = fit_dead_head(mixture, dead_cache)
        for source, name, inputs, labels, x, px in evaluation:
            base = refine_features.probabilities_from_features(x, model["feature_names"], model)
            with threadpool_limits(limits=1):
                player_dead = head.predict_proba(px)[:, 1]
            variants = {"full_model_D": base, "player_only_D": refine_domain.combine(base, player_dead, 0.0)}
            for variant, probabilities in variants.items():
                for cost in SWITCH_COSTS:
                    row = {"train": "+".join(mixture), "eval_source": source, "match": name,
                           "unseen_source": source not in mixture, "d_head": variant, "switch_cost": cost,
                           **score(inputs, labels, probabilities, cost)}
                    rows.append(row)
                    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
                          flush=True)
        # Save after every mixture so a stopped run resumes where it left off.
        args.out.write_text(json.dumps({"mixtures": ["+".join(m) for m in MIXTURES],
                                        "switch_costs": SWITCH_COSTS, "rows": rows}, indent=2) + "\n")
    args.out.write_text(json.dumps({"mixtures": ["+".join(m) for m in MIXTURES], "switch_costs": SWITCH_COSTS,
                                    "rows": rows, "last_run_seconds": round(time.monotonic() - started, 1)},
                                   indent=2) + "\n")


if __name__ == "__main__":
    main()
