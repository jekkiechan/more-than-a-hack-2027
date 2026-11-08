"""Scoring harness for `v2_postproc_v1` cached development emissions.

Pipelines map one cell's emissions to an H/A/D prediction without labels.
Labels are used only by the official scorer afterwards. Aggregation follows the
established protocol: realizations within match, matches within provider, then
providers equally, separately for clean, feed_like_v1 and feed_like_v2.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np

from experiments import v2_spell_target as base
from experiments import v2_spell_balanced_eval as balanced

ROOT = Path(__file__).resolve().parents[1]
EMIT = ROOT / "data/experiments/v2_postproc_v1/emissions"
CE_ROOT = ROOT / "data/experiments/v2_postproc_v1"
SEEDS = (2026, 2027)
METRICS = ("spell_f1", "macro_f1", "short_spell_recall_iou_0_5_lt3s", "d_f1",
           "predicted_spell_count", "reference_spell_count")
VIEWS = ("clean", "feed_like_v1", "feed_like_v2")


def keys():
    return sorted(p.stem for p in EMIT.glob("*.npz") if not p.stem.endswith(".tmp"))


def load(key):
    with np.load(EMIT / f"{key}.npz", allow_pickle=False) as d:
        cell = {k: d[k] for k in d.files}
    cell["fps"] = float(cell["fps"])
    for ce_dir in sorted(CE_ROOT.glob("ce_emissions_stage*")):
        for ce_path in (ce_dir / f"{key}.npz", ce_dir / f"{key}__x.npz"):
            if ce_path.is_file():
                with np.load(ce_path, allow_pickle=False) as d:
                    cell.update({k: d[k] for k in d.files if k.startswith("phase__")})
    return cell


def family_mean(cell, family, seed=None):
    """Seed mean within phase, then equal phase mean (float64)."""
    seeds = SEEDS if seed is None else (seed,)
    if family.endswith("+x"):  # four continuation seeds (two extra draws per base seed)
        family, seeds = family[:-2], SEEDS + (2126, 2127)
    elif family.endswith("@x"):  # the two extra continuation seeds only
        family, seeds = family[:-2], (2126, 2127)
    phase = np.mean([cell[f"phase__{family}__s{s}"].astype(np.float64) for s in seeds], axis=0)
    return phase.mean(axis=0)


def decode_f062(prob, cell):
    return base.audit.decode(prob, cell["model_period"], cell["fps"])


def decode_fixed_d(prob, d_source_pred, cell, ha_cost=3.0):
    return balanced.decode_fixed_d(prob, d_source_pred, cell["model_period"], cell["fps"], ha_cost)


def score(cell, prediction):
    return base._score_cell(cell, prediction)


def _job(args):
    key, pipelines = args
    cell = load(key)
    return key, {name: {m: float(v) for m, v in score(cell, fn(cell)).items() if m in METRICS}
                 for name, fn in pipelines.items()}


def run(pipelines, workers=None, only_keys=None):
    workers = workers or int(os.environ.get("SCORE_WORKERS", "4"))
    jobs = [(k, pipelines) for k in (only_keys or keys())]
    with ProcessPoolExecutor(workers) as pool:
        rows = dict(pool.map(_job, jobs))
    return rows


def aggregate(rows, name):
    out = {}
    for view in VIEWS:
        groups = {}
        for key, cell in rows.items():
            provider, match, cell_view, _rep = key.split("__")
            if cell_view == view and name in cell:
                groups.setdefault(provider, {}).setdefault(match, []).append(cell[name])
        if not groups:
            continue
        out[view] = {}
        for metric in METRICS:
            per = {p: float(np.mean([np.mean([r[metric] for r in reps]) for reps in ms.values()]))
                   for p, ms in groups.items()}
            out[view][metric] = float(np.mean(list(per.values())))
            out[view][metric + "/providers"] = per
    return out


def table(rows, names, ref=None):
    lines = [f"{'pipeline':44s} {'clean':>8s} {'v1':>8s} {'v2':>8s} | {'mac_c':>7s} {'mac_v1':>7s} {'mac_v2':>7s} | {'sr_c':>6s} {'sr_v1':>6s} {'sr_v2':>6s}"]
    for name in names:
        a = aggregate(rows, name)
        g = lambda v, m: a.get(v, {}).get(m, float("nan"))
        lines.append(f"{name:44s} {g('clean','spell_f1'):8.6f} {g('feed_like_v1','spell_f1'):8.6f} {g('feed_like_v2','spell_f1'):8.6f} | "
                     f"{g('clean','macro_f1'):7.5f} {g('feed_like_v1','macro_f1'):7.5f} {g('feed_like_v2','macro_f1'):7.5f} | "
                     f"{g('clean','short_spell_recall_iou_0_5_lt3s'):6.4f} {g('feed_like_v1','short_spell_recall_iou_0_5_lt3s'):6.4f} {g('feed_like_v2','short_spell_recall_iou_0_5_lt3s'):6.4f}")
    return "\n".join(lines)
