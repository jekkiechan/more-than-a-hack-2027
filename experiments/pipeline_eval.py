"""Bounded, resumable leave-one-provider-out possession pipeline experiment.

Run ``python -m experiments.pipeline_eval``. Only the existing external training
and validation matches are opened. Historical tests and competition labels are
never read. All three providers contribute equally to selection; within a
provider, its validation matches and clean/degraded conditions count equally.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from experiments import (dead_transfer, multisource_eval, possession_pipeline as pipeline,
                         refine_decode, refine_domain, refine_features, refine_weighted,
                         robust, supervised)

ROOT = Path(__file__).resolve().parents[1]
SOURCES = tuple(dead_transfer.SPLITS)
TRAIN_HZ = 2
CAPACITY = {"max_iter": 120, "max_leaf_nodes": 31, "learning_rate": .08,
            "min_samples_leaf": 35, "l2_regularization": 1, "early_stopping": False}
CANDIDATES = [
    {"name": name, "members": members, "context_weight": weight,
     "player_dead_weight": dead_weight, "team_cost": 3., "dead_cost": dead_cost}
    for name, members, weight, dead_weight, dead_cost in (
        ("baseline", ["baseline", "baseline"], 0., 0., 3.),
        ("preprocessed", ["preprocessed", "preprocessed"], 0., 0., 3.),
        ("context", ["context", "context"], 0., 0., 3.),
        ("ensemble", ["baseline", "context"], .5, 0., 3.),
        ("ensemble_half_player_D", ["baseline", "context"], .5, .5, 3.),
        ("ensemble_player_D", ["baseline", "context"], .5, 1., 3.),
        ("ensemble_half_player_D_cost8", ["baseline", "context"], .5, .5, 8.),
        ("ensemble_player_D_cost8", ["baseline", "context"], .5, 1., 8.),
    )
]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def protocol(seed):
    paths = [p for split in dead_transfer.SPLITS.values() for role in ("train", "eval") for p in split[role]]
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size < 1024]
    if missing:
        raise ValueError("Missing processed data (or unresolved Git LFS pointers): " + ", ".join(missing))
    sources = ["pipeline_eval.py", "possession_pipeline.py", "multisource_eval.py", "dead_transfer.py",
               "supervised.py", "refine_features.py", "refine_domain.py", "refine_decode.py",
               "refine_weighted.py", "robust.py", "stress.py", "metrics.py"]
    return {
        "version": 1, "seed": seed, "capacity": CAPACITY, "train_hz": TRAIN_HZ,
        "train_weight": "beta=.5 short live spells; equal total per match, then per source",
        "splits": {s: {k: [p.stem for p in paths] for k, paths in split.items()}
                   for s, split in dead_transfer.SPLITS.items()},
        "folds": {s: [other for other in SOURCES if other != s] for s in SOURCES},
        "conditions": ["clean", "degraded"], "candidates": CANDIDATES,
        "augmentation": "same clean and degraded views for every ablation, each with H/A swap",
        "selection": "max mean spell F1 over providers, matches and conditions; macro F1 breaks ties",
        "limits": "reused validation matches; one seed per run; not fresh confirmation or competition performance",
        "data_sha256": {str(p.relative_to(ROOT / "data")): robust.sha256(p) for p in paths},
        "source_sha256": {name: robust.sha256(ROOT / "experiments" / name) for name in sources},
        "official_metric_sha256": robust.sha256(ROOT / "evaluation/possession_metrics.py"),
        "versions": {name: __import__(name).__version__ for name in ("numpy", "scipy", "sklearn", "joblib")},
        "test_accessed": False, "competition_labels_accessed": False,
    }


def freeze_protocol(out, seed):
    spec = protocol(seed)
    path = out / "protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["protocol"] != spec:
            raise ValueError("Protocol, code, dependencies or data changed; use a new output directory")
        return previous
    # Caches may only be reused under a matching, complete provenance manifest.
    if any(out.iterdir()):
        raise ValueError("Nonempty output without protocol; use a fresh output directory")
    manifest = {"declared_utc": datetime.now(timezone.utc).isoformat(), "protocol": spec}
    atomic_json(path, manifest)
    return manifest


def training_match(path, seed):
    inputs, labels = multisource_eval.view(path)
    stride = max(1, round(float(inputs["fps"]) / TRAIN_HZ))
    rows = np.arange(0, len(labels), stride)
    rows = rows[np.isin(labels[rows], supervised.LABELS)]
    y = np.array([{"H": 0, "A": 1, "D": 2}[v] for v in labels[rows]], dtype=np.int8)
    weights, _ = refine_weighted.spell_weights(dict(inputs, labels=labels), .5)
    sampled = weights[rows]
    sampled = sampled / sampled.sum()
    matrices, targets, row_weights, dead_x, dead_y = {}, [], [], [], []
    names = {}
    for condition, view in (("clean", inputs), ("degraded", pipeline.degraded_inputs(inputs, seed))):
        print(f"FEATURES {path.stem} {condition}", flush=True)
        features, clean = pipeline.feature_views(view)
        for name, (x, schema) in features.items():
            names[name] = schema
            x = x[rows]
            matrices.setdefault(name, []).extend((x, supervised.swap_team_features(x, schema)))
        targets.extend((y, np.where(y < 2, 1 - y, y)))
        row_weights.extend((sampled / 4, sampled / 4))
        px, dead_names = refine_domain.features(clean)
        dead_x.append(px[rows])
        dead_y.append((y == 2).astype(np.int8))
    return {"x": {k: np.concatenate(v) for k, v in matrices.items()}, "names": names,
            "y": np.concatenate(targets), "weight": np.concatenate(row_weights),
            "dead_x": np.concatenate(dead_x), "dead_y": np.concatenate(dead_y),
            "dead_weight": np.full(2 * len(rows), 1. / (2 * len(rows))), "dead_names": dead_names}


def prepare_cache(task):
    path, seed, target = task
    if not target.exists():
        data = training_match(path, seed)
        temporary = target.with_suffix(".tmp")
        joblib.dump(data, temporary)
        temporary.replace(target)
    return path.stem


def fit(mixture, caches, seed, threads):
    paths = [(source, path) for source in mixture for path in dead_transfer.SPLITS[source]["train"]]
    y, weights, dy, dw = [], [], [], []
    matrices = {name: [] for name in ("baseline", "preprocessed", "context")}
    dx, names, dead_names = [], None, None
    for source, path in paths:
        cache = joblib.load(caches / f"{path.stem}.joblib", mmap_mode="r")
        if names is not None and (names != cache["names"] or dead_names != cache["dead_names"]):
            raise ValueError("Feature schemas differ across sources")
        names, dead_names = cache["names"], cache["dead_names"]
        scale = 1. / len(dead_transfer.SPLITS[source]["train"])
        y.append(cache["y"])
        weights.append(cache["weight"] * scale)
        dy.append(cache["dead_y"])
        dw.append(cache["dead_weight"] * scale)
        dx.append(cache["dead_x"])
        for name in matrices:
            matrices[name].append(cache["x"][name])
    y, w = np.concatenate(y), np.concatenate(weights)
    w *= len(w) / w.sum()
    members = {}
    for name, parts in matrices.items():
        print("FIT", "+".join(mixture), name, len(y), flush=True)
        estimator = HistGradientBoostingClassifier(**CAPACITY, random_state=seed)
        with threadpool_limits(limits=threads):
            estimator.fit(np.concatenate(parts), y, sample_weight=w)
        if estimator.classes_.tolist() != [0, 1, 2]:
            raise ValueError("Need all three training classes")
        members[name] = {"estimator": estimator, "feature_names": names[name], "classes": [0, 1, 2]}
    dy, dw = np.concatenate(dy), np.concatenate(dw)
    dw *= len(dw) / dw.sum()
    print("FIT", "+".join(mixture), "player_D", len(dy), flush=True)
    dead = HistGradientBoostingClassifier(**CAPACITY, random_state=seed)
    with threadpool_limits(limits=threads):
        dead.fit(np.concatenate(dx), dy, sample_weight=dw)
    if dead.classes_.tolist() != [0, 1]:
        raise ValueError("Need live and dead training classes")
    return {"members": members, "dead_estimator": dead, "dead_feature_names": dead_names,
            "training_sources": list(mixture), "seed": seed}


def evaluate(source, path, model, seed):
    inputs, labels = multisource_eval.view(path)
    results = []
    for condition, view in (("clean", inputs), ("degraded", pipeline.degraded_inputs(inputs, seed))):
        matrices, clean = pipeline.feature_views(view)
        p = {name: refine_features.probabilities_from_features(*matrices[name], member)
             for name, member in model["members"].items()}
        dx, dead_names = refine_domain.features(clean)
        if dead_names != model["dead_feature_names"]:
            raise ValueError("Dead feature schema differs")
        with threadpool_limits(limits=1):
            dead = model["dead_estimator"].predict_proba(dx)[:, 1]
        for candidate in CANDIDATES:
            left, right = candidate["members"]
            probs = pipeline.blend_probabilities(p[left], p[right], dead,
                                                 candidate["context_weight"], candidate["player_dead_weight"])
            pred = refine_decode.decode(probs, supervised.model_periods(clean), float(clean["fps"]), candidate)
            score = supervised.score(dict(clean, labels=labels), pred)
            row = {"source": source, "match": path.stem, "condition": condition,
                   "candidate": candidate["name"], "unseen_source": True,
                   "macro_f1": score["macro_f1"], "spell_f1": score["spell_f1_mean"],
                   "d_f1": score["per_class"]["D"]["f1"],
                   "team_acc_alive": score["team_acc_gt_alive"],
                   "n_gt_spells": score["n_gt_spells"], "n_pred_spells": score["n_pred_spells"]}
            results.append(row)
            print("SCORE", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
    return results


def summarize(rows):
    expected = {(source, path.stem, condition, candidate["name"])
                for source in SOURCES for path in dead_transfer.SPLITS[source]["eval"]
                for condition in ("clean", "degraded") for candidate in CANDIDATES}
    actual = [(r["source"], r["match"], r["condition"], r["candidate"]) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Selection requires every fold, condition and candidate exactly once")
    summaries = []
    for candidate in CANDIDATES:
        by_source = {}
        for source in SOURCES:
            selected = [r for r in rows if r["candidate"] == candidate["name"] and r["source"] == source]
            by_source[source] = {key: float(np.mean([r[key] for r in selected]))
                                 for key in ("spell_f1", "macro_f1", "d_f1")}
        summaries.append({"candidate": candidate["name"], "by_source": by_source,
                          **{key: float(np.mean([v[key] for v in by_source.values()]))
                             for key in ("spell_f1", "macro_f1", "d_f1")}})
    return sorted(summaries, key=lambda s: (s["spell_f1"], s["macro_f1"]), reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data/experiments/pipeline_v1")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--feature-workers", type=int, default=2)
    parser.add_argument("--fit-final", action="store_true", help="Refit the externally selected configuration on all training sources")
    args = parser.parse_args()
    if args.threads < 1 or args.feature_workers < 1:
        parser.error("Worker/thread counts must be positive")
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = freeze_protocol(args.out, args.seed)
    cache = args.out / "features"
    cache.mkdir(exist_ok=True)
    tasks = []
    for source_index, source in enumerate(SOURCES):
        for i, path in enumerate(dead_transfer.SPLITS[source]["train"]):
            target = cache / f"{path.stem}.joblib"
            tasks.append((path, 2026 + 100 * source_index + i, target))
    if args.feature_workers == 1:
        for task in tasks:
            prepare_cache(task)
    else:
        with ProcessPoolExecutor(max_workers=args.feature_workers) as pool:
            for name in pool.map(prepare_cache, tasks):
                print("CACHED", name, flush=True)
    result_path = args.out / "validation.json"
    rows = json.loads(result_path.read_text())["rows"] if result_path.exists() else []
    for source_index, source in enumerate(SOURCES):
        target = args.out / f"leave_{source}_out.joblib"
        if not target.exists():
            model = fit([s for s in SOURCES if s != source], cache, args.seed, args.threads)
            temporary = target.with_suffix(".tmp")
            joblib.dump(model, temporary)
            temporary.replace(target)
        model = joblib.load(target)
        for i, path in enumerate(dead_transfer.SPLITS[source]["eval"]):
            existing = [r for r in rows if r["source"] == source and r["match"] == path.stem]
            if len(existing) == 2 * len(CANDIDATES):
                continue
            if existing:
                raise ValueError("Partial match result; refuse ambiguous resume")
            rows.extend(evaluate(source, path, model, 5026 + 100 * source_index + i))
            atomic_json(result_path, {"manifest": manifest, "rows": rows})
    summaries = summarize(rows)
    selected = next(c for c in CANDIDATES if c["name"] == summaries[0]["candidate"])
    report = {"manifest": manifest, "rows": rows, "summary": summaries, "selected": selected,
              "last_run_seconds": round(time.monotonic() - started, 1)}
    atomic_json(result_path, report)
    atomic_json(args.out / "selected.json", selected)
    print("SELECTED", json.dumps(summaries[0]), flush=True)
    if args.fit_final and not (args.out / "model.joblib").exists():
        model = fit(SOURCES, cache, args.seed, args.threads)
        model["candidate"] = selected
        model["protocol_sha256"] = robust.sha256(args.out / "protocol.json")
        joblib.dump(model, args.out / "model.joblib")
        atomic_json(args.out / "model_manifest.json", {"model_sha256": robust.sha256(args.out / "model.joblib"),
                    "selected": selected, "protocol_sha256": model["protocol_sha256"]})


if __name__ == "__main__":
    main()
