"""Label-free target adaptation of the combined-objective GRU ensemble.

Arms (nothing else, no tuning):
  none         unchanged checkpoint (control)
  target_norm  replace the checkpoint's input centre/scale with statistics of the
               target cell's own inference features (label-free)
  self_train   fine-tune on the target cell's inputs with the unchanged
               ensemble's confident predictions as pseudo-labels, mixed with an
               equal number of labelled windows from the fold's own TRAIN matches

Adaptation functions receive records with ``y``/``full_y`` removed (``blind``).
Labels are loaded separately and only by the scoring code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import factorized_had as fh, learned_temporal as lt, supervised  # noqa: E402

STUDY_ID = "target_adaptation_v1"
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")).resolve()
OUT = DATA / "experiments" / STUDY_ID
OBJ = DATA / "experiments/objective_alignment_v1"
SEED_GAP = DATA / "experiments/seed_gap_context_v1"
DEPLOY = DATA / "models/joint_gru_combined_ensemble_v1"
RESULT = ROOT / "experiments/results/target_adaptation.json"
SEEDS = (2026, 2027)
ARMS = ("none", "target_norm", "self_train")
VIEWS = (("clean", "clean"), ("feed_like", "corrupt_1"), ("feed_like", "corrupt_2"))
PROVIDERS = {
    "sportec": ["J03WOY"],
    "metrica": ["metrica_3"],
    "skillcorner": ["skillcorner_had_1996435", "skillcorner_had_1996436",
                    "skillcorner_had_2006229"],
}
SETTINGS = dict(
    scale_floor_fraction=0.1,
    target_norm_statistic="nanmedian / nanstd (the statistics the checkpoint itself uses) over "
                          "the target's finite entries, pooled over original and home/away-swapped "
                          "features exactly as training pooled them",
    binary_rule="feature left at checkpoint stats if every finite target value is in {0,1}",
    pseudo_label_threshold=0.90,
    optimizer="Adam", lr=1e-4, epochs=2, batch_size=64, grad_clip=1.0,
    mix="equal number of fold-TRAIN windows per epoch, true labels, plain factorized NLL, "
        "training window/gap-alternate choice taken from the cached training records",
    target_windows="10 s cores / 5 s context, original and home/away-swapped copies, "
                   "frames with ensemble max probability >= 0.90 only",
    train_sample_seed="sha256(seed, match, view, replicate, epoch)",
)
DECISION_RULE = dict(
    pass_if=[
        "equal-provider feed_like spellF1 >= A0 + 0.01",
        "equal-provider clean spellF1 >= A0 - 0.005",
        "no provider's feed_like or clean spellF1 drops by more than 0.02",
    ],
    aggregation="replicates within match, matches within provider, providers equally",
    scored_unit="two-seed probability ensemble, Viterbi cost 3",
)


class AdaptationError(RuntimeError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_array(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def cells(match=None):
    for source, matches in PROVIDERS.items():
        for m in matches:
            if match is not None and m != match:
                continue
            for view, rep in VIEWS:
                yield dict(source=source, match=m, view=view, replicate=rep)


def cell_name(cell):
    return f"{cell['match']}_{cell['view']}_{cell['replicate']}"


# ---------------------------------------------------------------- data access
def blind(record):
    """Record without any target label field."""
    return {k: v for k, v in record.items() if k not in ("y", "full_y")}


def load_cell(cell):
    """Return (inputs, blind record, labels). Labels are for scoring only."""
    cache = joblib.load(SEED_GAP / f"{cell_name(cell)}.joblib")
    return cache["inputs"], blind(cache["record"]), cache["labels"]


def checkpoint_path(source, seed):
    return OBJ / f"without_{source}_combined_{seed}.joblib"


def load_checkpoint(source, seed):
    return joblib.load(checkpoint_path(source, seed))


# ------------------------------------------------------------------ A1
def target_statistics(record, model):
    """Label-free replacement (median, scale) for a checkpoint's normalization."""
    x = np.asarray(record["x"])
    pooled = np.vstack([x, supervised.swap_team_features(x, record["names"])])
    finite = np.isfinite(pooled)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(pooled, axis=0)
        scale = np.nanstd(pooled, axis=0)
    train_median = np.asarray(model["median"], dtype=np.float64)
    train_scale = np.asarray(model["scale"], dtype=np.float64)
    seen = finite.any(axis=0)
    binary = np.zeros(x.shape[1], dtype=bool)
    for j in np.flatnonzero(seen):
        v = pooled[finite[:, j], j]
        binary[j] = bool(np.all((v == 0) | (v == 1)))
    use = seen & ~binary & np.isfinite(median)
    new_median = np.where(use, median, train_median)
    sane = np.isfinite(scale) & (scale > .01)
    new_scale = np.where(sane, scale, train_scale)
    new_scale = np.maximum(new_scale, SETTINGS["scale_floor_fraction"] * train_scale)
    new_scale = np.where(use, new_scale, train_scale)
    return (new_median.astype(np.asarray(model["median"]).dtype),
            new_scale.astype(np.asarray(model["scale"]).dtype), use)


def with_normalization(model, median, scale):
    return dict(model, median=median, scale=scale)


# ------------------------------------------------------------------ inference
def sampled_probability(record, model):
    return fh.predict(blind(record), model)


def native_probability(sampled, record, inputs):
    period = supervised.model_periods(inputs)
    p = np.column_stack([lt.upsample(sampled[:, k], record, period) for k in range(3)])
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise AdaptationError("invalid native probability")
    return p


# ------------------------------------------------------------------ A2
def pseudo_targets(ensemble_sampled, threshold=None):
    threshold = SETTINGS["pseudo_label_threshold"] if threshold is None else threshold
    p = np.asarray(ensemble_sampled)
    return p.argmax(1).astype(np.int64), (p.max(1) >= threshold)


def target_windows(record, ensemble_sampled, model):
    """Normalized original+swapped target windows with pseudo-label weights."""
    label, keep = pseudo_targets(ensemble_sampled)
    swapped_label = np.where(label < 2, 1 - label, label)
    items = []
    for swap in (False, True):
        raw = (supervised.swap_team_features(record["x"], record["names"])
               if swap else record["x"])
        x = lt.normalize(raw, model["median"], model["scale"])
        y = swapped_label if swap else label
        for a, size, full in lt.chunks(record):
            w = np.zeros(lt.CORE, np.float32)
            t = np.zeros(lt.CORE, np.int64)
            w[:size] = keep[a:a + size]
            t[:size] = y[a:a + size]
            if w.sum() > 0:
                items.append((x[full], t, w))
    return items


class TrainPool:
    """Windows of a fold's labelled TRAIN records, normalized lazily."""

    def __init__(self, records):
        self.records = records
        counts = {}
        for r in records:
            counts[r["source"]] = counts.get(r["source"], 0) + 1
        masses = np.array([1 / (counts[r["source"]] * max(1, np.sum(r["y"] >= 0)))
                           for r in records])
        self.masses = masses / masses.mean()
        self.samples = [(i, a, size, full, bool(use))
                        for i, r in enumerate(records)
                        for (a, size, full), use in zip(lt.chunks(r), r["replace_windows"])]

    def draw(self, n, rng, model):
        pick = rng.choice(len(self.samples), size=min(n, len(self.samples)), replace=False)
        items = []
        for k in pick:
            i, a, size, full, use = self.samples[k]
            r = self.records[i]
            raw = np.asarray((r["alternate"] if use else r["x"]))[full]
            y = np.asarray(r["y"])[a:a + size]
            t = np.zeros(lt.CORE, np.int64)
            w = np.zeros(lt.CORE, np.float32)
            t[:size] = np.maximum(y, 0)
            w[:size] = (y >= 0) * self.masses[i]
            items.append((lt.normalize(raw, model["median"], model["scale"]), t, w))
        return items


def stable_seed(*parts):
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def self_train(model, tgt_items, pool, *, seed, tag, epochs=None):
    """Fine-tune one seed's network; returns a new checkpoint dict."""
    import torch
    epochs = SETTINGS["epochs"] if epochs is None else epochs
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    torch.manual_seed(seed)
    net = fh.network(model["input_features"], model["player_indices"], model["player_only"])
    net.load_state_dict({k: torch.as_tensor(v) for k, v in model["state"].items()})
    optimizer = torch.optim.Adam(net.parameters(), lr=SETTINGS["lr"])
    generator = torch.Generator().manual_seed(seed)
    history = []
    for epoch in range(epochs):
        rng = np.random.default_rng(stable_seed(seed, tag, epoch))
        items = list(tgt_items) + (pool.draw(len(tgt_items), rng, model) if pool is not None else [])
        order = torch.randperm(len(items), generator=generator).numpy()
        num = den = 0.
        net.train()
        for s in range(0, len(items), SETTINGS["batch_size"]):
            batch = [items[k] for k in order[s:s + SETTINGS["batch_size"]]]
            x = torch.as_tensor(np.stack([b[0] for b in batch]))
            y = torch.as_tensor(np.stack([b[1] for b in batch]))
            w = torch.as_tensor(np.stack([b[2] for b in batch]))
            if not w.sum():
                continue
            optimizer.zero_grad()
            logp = net(x)[:, lt.CONTEXT:lt.CONTEXT + lt.CORE]
            losses = torch.nn.functional.nll_loss(logp.transpose(1, 2), y, reduction="none")
            loss = (losses * w).sum() / w.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), SETTINGS["grad_clip"])
            optimizer.step()
            num += float((losses.detach() * w).sum())
            den += float(w.sum())
        history.append(num / den if den else None)
    state = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    return dict(model, state=state, adaptation_history=history)


def load_pool(source):
    """Labelled TRAIN windows of the fold that excludes ``source``."""
    from experiments import combined_objective_deployment as deployment
    receipt = json.loads(checkpoint_path(source, SEEDS[0]).with_suffix(".json").read_text())
    records = []
    for match in receipt["training_matches"]:
        cache = joblib.load(OBJ / f"train_{match}.joblib", mmap_mode="r")
        if cache["match"] != match:
            raise AdaptationError("training cache identity mismatch")
        records.extend(deployment.paired_records(cache, match))
    return TrainPool(records)


# ------------------------------------------------------------------ one cell
def run_arm(arm, inputs, record, source, tag, pool=None):
    """Return {seed: native prob} and sampled ensemble for the unadapted model."""
    models = {s: load_checkpoint(source, s) for s in SEEDS}
    if arm == "none":
        adapted = models
    elif arm == "target_norm":
        median, scale, _ = target_statistics(record, models[SEEDS[0]])
        adapted = {s: with_normalization(m, median, scale) for s, m in models.items()}
    elif arm == "self_train":
        control = np.mean([sampled_probability(record, m) for m in models.values()], axis=0)
        adapted = {}
        for s, m in models.items():
            items = target_windows(record, control, m)
            adapted[s] = self_train(m, items, pool, seed=s, tag=tag)
    else:
        raise ValueError(arm)
    return {s: native_probability(sampled_probability(record, m), record, inputs)
            for s, m in adapted.items()}


def score_probability(inputs, labels, probability, cell, recipe):
    from experiments import objective_ensemble as oe
    fps = float(np.asarray(inputs["fps"]))
    pred = supervised.decode(probability, supervised.model_periods(inputs), fps, lt.COST)
    return oe.score_prediction(labels, pred, inputs, recipe, dict(cell),
                               digest_array(probability))


def evaluate_cell(arm, cell, pool=None, force=False):
    dest = OUT / "cells" / f"{arm}__{cell_name(cell)}.json"
    if dest.exists() and not force:
        return json.loads(dest.read_text())
    start = time.monotonic()
    inputs, record, labels = load_cell(cell)
    per_seed = run_arm(arm, inputs, record, cell["source"], cell_name(cell), pool)
    ensemble = .5 * (per_seed[SEEDS[0]].astype(np.float64) + per_seed[SEEDS[1]].astype(np.float64))
    rows = {"ensemble": score_probability(inputs, labels, ensemble, cell, arm)}
    for s, p in per_seed.items():
        rows[f"seed_{s}"] = score_probability(inputs, labels, p, cell, f"{arm}_{s}")
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    (OUT / "probabilities").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "probabilities" / f"{arm}__{cell_name(cell)}.npz",
                        ensemble=ensemble.astype(np.float32),
                        **{f"seed_{s}": p.astype(np.float32) for s, p in per_seed.items()})
    out = dict(arm=arm, **cell, rows=rows, seconds=time.monotonic() - start)
    dest.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n")
    return out


# ------------------------------------------------------------------ parity
def parity(match="J03WOY"):
    saved = json.loads((ROOT / "experiments/results/objective_ensemble_rows.json").read_text())["rows"]
    report = []
    for cell in cells(match):
        if cell["replicate"] == "corrupt_2":
            continue
        inputs, record, labels = load_cell(cell)
        per_seed = run_arm("none", inputs, record, cell["source"], "parity")
        ens = .5 * (per_seed[2026].astype(np.float64) + per_seed[2027].astype(np.float64))
        row = score_probability(inputs, labels, ens, cell, "none")
        ref = next(r for r in saved if r["recipe"] == "combined_objective_equal_probability_ensemble"
                   and (r["source"], r["match"], r["view"], r["replicate"]) ==
                   (cell["source"], cell["match"], cell["view"], cell["replicate"]))
        diffs = {k: abs(row[k] - ref[k]) for k in
                 ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall", "weighted_70_30")}
        diffs.update({f"iou_{k}": abs(row["spell_by_iou"][k] - ref["spell_by_iou"][k])
                      for k in row["spell_by_iou"]})
        ok = max(diffs.values()) < 1e-6 and row["n_pred_spells"] == ref["n_pred_spells"]
        report.append(dict(cell=cell_name(cell), max_abs_diff=max(diffs.values()), ok=ok,
                           spell_f1=row["spell_f1"], saved_spell_f1=ref["spell_f1"],
                           n_pred=row["n_pred_spells"], saved_n_pred=ref["n_pred_spells"],
                           probability_sha_equal=row["probability_sha256"] == ref["probability_sha256"]))
        print("PARITY", json.dumps(report[-1]), flush=True)
    return report


# ------------------------------------------------------------------ declaration
def declare():
    path = OUT / "declaration.json"
    if path.exists():
        raise AdaptationError("declaration exists and is immutable")
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = {f"without_{s}_combined_{seed}.joblib": sha256_file(checkpoint_path(s, seed))
            for s in PROVIDERS for seed in SEEDS}
    deploy = {p.name: sha256_file(p) for p in sorted(DEPLOY.glob("*.joblib"))}
    spec = dict(
        study_id=STUDY_ID,
        question="Do label-free target adaptations improve holdout spellF1 under feed_like stress "
                 "without hurting clean holdout?",
        arms=list(ARMS), settings=SETTINGS, decision_rule=DECISION_RULE,
        matches=PROVIDERS, views=[list(v) for v in VIEWS], seeds=list(SEEDS),
        checkpoint_sha256=ckpt, deployment_model_sha256=deploy,
        evaluation_inputs="seed_gap_context_v1 cached corrupted inputs/features (feed_like = "
                          "stress_components.corrupt with FEED components and ce.seed(match, replicate))",
        source_sha256=sha256_file(Path(__file__)),
        organizer_labels_read=False, holdout_labels_used_for="final scoring only",
        a2_scope="all cells if within ~3 CPU-hours, otherwise feed_like only",
        parity_done_before_declaration="A0 clean and feed_like corrupt_1 for J03WOY")
    path.write_text(json.dumps(spec, indent=1) + "\n")
    return spec


# ------------------------------------------------------------------ report
METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall")
IOUS = ("0.3", "0.4", "0.5", "0.6", "0.7")


def _flat(row):
    f = {k: row[k] for k in METRICS}
    f.update({f"iou_{k}": row["spell_by_iou"][k] for k in IOUS})
    f["n_pred"] = row["n_pred_spells"]
    f["n_gt"] = row["n_gt_spells"]
    return f


def aggregate(results, member):
    """results: list of per-cell dicts for one arm. Returns view -> provider/equal."""
    out = {}
    for group in ("clean", "feed_like"):
        sel = [r for r in results if r["view"] == group]
        per_provider = {}
        per_match = {}
        for source, matches in PROVIDERS.items():
            match_vals = []
            for m in matches:
                reps = [_flat(r["rows"][member]) for r in sel if r["match"] == m]
                if not reps:
                    continue
                v = {k: float(np.mean([x[k] for x in reps])) for k in reps[0]}
                per_match[m] = v
                match_vals.append(v)
            if match_vals:
                per_provider[source] = {k: float(np.mean([v[k] for v in match_vals]))
                                        for k in match_vals[0]}
        if per_provider:
            equal = {k: float(np.mean([v[k] for v in per_provider.values()]))
                     for k in next(iter(per_provider.values()))}
            out[group] = dict(equal_provider=equal, per_provider=per_provider, per_match=per_match)
    return out


def report():
    arms = {}
    for arm in ARMS:
        res = [json.loads(p.read_text()) for p in sorted((OUT / "cells").glob(f"{arm}__*.json"))]
        if res:
            arms[arm] = dict(n_cells=len(res),
                             ensemble=aggregate(res, "ensemble"),
                             seeds={str(s): aggregate(res, f"seed_{s}") for s in SEEDS})
    decision = {}
    base = arms.get("none")
    for arm in ARMS[1:]:
        if arm not in arms or base is None or arms[arm]["n_cells"] != 15:
            continue
        a, b = arms[arm]["ensemble"], base["ensemble"]
        d_feed = a["feed_like"]["equal_provider"]["spell_f1"] - b["feed_like"]["equal_provider"]["spell_f1"]
        d_clean = a["clean"]["equal_provider"]["spell_f1"] - b["clean"]["equal_provider"]["spell_f1"]
        worst = min(a[g]["per_provider"][s]["spell_f1"] - b[g]["per_provider"][s]["spell_f1"]
                    for g in ("clean", "feed_like") for s in PROVIDERS)
        decision[arm] = dict(feed_like_delta=d_feed, clean_delta=d_clean, worst_provider_delta=worst,
                             passes=bool(d_feed >= .01 - 1e-12 and d_clean >= -.005 - 1e-12
                                         and worst >= -.02 - 1e-12))
    decl = json.loads((OUT / "declaration.json").read_text())
    payload = dict(study_id=STUDY_ID, declaration_sha256=sha256_file(OUT / "declaration.json"),
                   settings=SETTINGS, decision_rule=DECISION_RULE, arms=arms, decision=decision,
                   checkpoint_sha256=decl["checkpoint_sha256"],
                   organizer_labels_read=False, organizer_scored=False)
    RESULT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(json.dumps(decision, indent=1))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("parity", "declare", "run", "report"))
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--match")
    parser.add_argument("--views", choices=("all", "feed_like"), default="all")
    args = parser.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    if args.action == "parity":
        rep = parity(args.match or "J03WOY")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "parity.json").write_text(json.dumps(rep, indent=1) + "\n")
        if not all(r["ok"] for r in rep):
            raise SystemExit("PARITY FAILED")
    elif args.action == "declare":
        declare()
    elif args.action == "run":
        pool_by_source = {}
        for cell in cells(args.match):
            if args.views == "feed_like" and cell["view"] != "feed_like":
                continue
            pool = None
            if args.arm == "self_train":
                if cell["source"] not in pool_by_source:
                    pool_by_source = {cell["source"]: load_pool(cell["source"])}
                pool = pool_by_source[cell["source"]]
            r = evaluate_cell(args.arm, cell, pool)
            e = r["rows"]["ensemble"]
            print("CELL", args.arm, cell_name(cell), round(e["spell_f1"], 5), round(e["macro_f1"], 5),
                  round(r["seconds"], 1), flush=True)
    else:
        report()


if __name__ == "__main__":
    main()
