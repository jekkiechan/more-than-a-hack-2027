"""Cross-source transfer of a player-only dead-ball (D) detector.

Every open source has a provider-specific dead-ball shortcut that the challenge's
tactical feed is not known to share: Sportec freezes the ball, Metrica's sample
games blank it, and SkillCorner's broadcast cuts to close-ups (about 4 of 22
players detected during D versus 14 live). A detector that only looks at
collective player motion and shape (``refine_domain.features``) cannot use the
ball shortcuts. Evaluating it on a source it was not trained on approximates the
challenge situation: an unseen provider, camera and annotation team.

Fixed before fitting (whole matches, ascending IDs within a source):

- sportec:     train J03WMX J03WN1 J03WOH; eval J03WOY
- metrica:     train games 1 2;            eval game 3 (FIFA EPTS, the challenge format)
- skillcorner: train first 8 match IDs;    eval last 3 match IDs

SkillCorner positions are kept only where SkillCorner says the player was
detected, as in the challenge's NaN convention. Rows with fewer than
``MIN_VISIBLE`` detected players are dropped from SkillCorner training and
evaluation so the broadcast-cut shortcut cannot be learned or rewarded.

Seven training mixtures (every non-empty subset of sources) are scored on every
evaluation split. Nothing here reads challenge data. Sportec test matches
J03WPY/J03WR9/J03WQQ are never opened.
"""

from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import refine_domain, supervised  # noqa: E402

SPORTEC = ROOT / "data/external_audit/sportec/processed"
EXTERNAL = ROOT / "data/external/processed"
SKILLCORNER_IDS = (1874553, 1886347, 1899585, 1925299, 1927964, 1953632, 1959846, 1986691,
                   1996435, 1996436, 2006229)
SPLITS = {
    "sportec": {"train": [SPORTEC / f"{m}.npz" for m in ("J03WMX", "J03WN1", "J03WOH")],
                "eval": [SPORTEC / "J03WOY.npz"]},
    "metrica": {"train": [EXTERNAL / f"metrica_{g}.npz" for g in (1, 2)],
                "eval": [EXTERNAL / "metrica_3.npz"]},
    "skillcorner": {"train": [EXTERNAL / f"skillcorner_had_{m}.npz" for m in SKILLCORNER_IDS[:8]],
                    "eval": [EXTERNAL / f"skillcorner_had_{m}.npz" for m in SKILLCORNER_IDS[8:]]},
}
MIN_VISIBLE = 12
TRAIN_HZ = 4
LEAVES, ITERATIONS = 31, 160


def prepare(path):
    """Player-only inputs, graded rows and binary D target for one match."""
    data = supervised.load(path)
    inputs = {k: data[k] for k in ("players", "teams", "period", "fps", "timestamps", "pitch")}
    labels = np.asarray(data["labels"])
    keep = np.isin(labels, ["H", "A", "D"])
    if "player_observed" in data and path.name.startswith("skillcorner"):
        players = np.array(inputs["players"], copy=True)
        players[~data["player_observed"]] = np.nan
        inputs["players"] = players
        keep &= data["player_observed"].sum(axis=1) >= MIN_VISIBLE
    return inputs, keep, (labels == "D").astype(np.int8)


def matrices(paths, seed, augment):
    xs, ys = [], []
    for i, path in enumerate(paths):
        inputs, keep, y = prepare(path)
        stride = max(1, round(float(inputs["fps"]) / TRAIN_HZ)) if augment else 1
        rows = np.flatnonzero(keep)[::stride]
        views = [inputs] + ([refine_domain.drop_player_slots(inputs, seed + i)] if augment else [])
        for view in views:
            x, names = refine_domain.features(view)
            xs.append(x[rows])
            ys.append(y[rows])
    return np.concatenate(xs), np.concatenate(ys), names


def fit(x, y, weights):
    model = HistGradientBoostingClassifier(max_iter=ITERATIONS, max_leaf_nodes=LEAVES, learning_rate=.08,
                                           min_samples_leaf=40, l2_regularization=1, early_stopping=False,
                                           random_state=31)
    with threadpool_limits(limits=1):
        model.fit(x, y, sample_weight=weights)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/results/dead_transfer.json")
    args = parser.parse_args()
    started = time.monotonic()
    train, evaluation = {}, {}
    for source, split in SPLITS.items():
        train[source] = matrices(split["train"], 9026, augment=True)
        evaluation[source] = [(p.stem, *matrices([p], 0, augment=False)[:2]) for p in split["eval"]]
        print(f"PREPARED {source} train_rows={len(train[source][1])} d_share={train[source][1].mean():.3f}",
              flush=True)
    rows = []
    sources = list(SPLITS)
    for size in range(1, len(sources) + 1):
        for mixture in combinations(sources, size):
            x = np.concatenate([train[s][0] for s in mixture])
            y = np.concatenate([train[s][1] for s in mixture])
            # Each source contributes equal total weight, so 8 SkillCorner matches
            # cannot drown out 2 Metrica matches.
            w = np.concatenate([np.full(len(train[s][1]), 1.0 / len(train[s][1])) for s in mixture])
            w *= len(w) / w.sum()
            model = fit(x, y, w)
            for target, matches in evaluation.items():
                scores = []
                for name, xe, ye in matches:
                    with threadpool_limits(limits=1):
                        p = model.predict_proba(xe)[:, 1]
                    scores.append({"match": name, "frames": int(len(ye)), "d_share": float(ye.mean()),
                                   "d_f1": float(f1_score(ye, p >= .5)), "auroc": float(roc_auc_score(ye, p))})
                row = {"train": "+".join(mixture), "eval": target, "unseen_source": target not in mixture,
                       "d_f1": float(np.mean([s["d_f1"] for s in scores])),
                       "auroc": float(np.mean([s["auroc"] for s in scores])), "matches": scores}
                rows.append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                                  for k, v in row.items() if k != "matches"}), flush=True)
    args.out.write_text(json.dumps({
        "splits": {s: {k: [p.stem for p in v] for k, v in split.items()} for s, split in SPLITS.items()},
        "min_visible_skillcorner": MIN_VISIBLE, "train_hz": TRAIN_HZ, "capacity": [LEAVES, ITERATIONS],
        "features": "refine_domain.features (player-only, team- and slot-symmetric)",
        "threshold": 0.5, "rows": rows, "seconds": round(time.monotonic() - started, 1)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
