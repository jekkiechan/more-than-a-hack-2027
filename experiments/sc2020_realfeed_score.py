"""Frontier-recipe scoring for `sc2020_realfeed_v1` candidate CE emissions.

A candidate replaces only the CE component of the approved frontier mixture
(CE/calibrated/predecessor conditional H/A at 0.5/0.375/0.125, seeds averaged
within phase, phases averaged, H/A cost 3, exact predecessor decoded D mask).
Labels are read only by the official scorer.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

from experiments import sc2020_realfeed_common as C
from experiments import frontier_postprocess_v1 as fp

SHORT = "short_spell_recall_iou_0_5_lt3s"
METRICS = ("spell_f1", "macro_f1", SHORT, "d_f1", "predicted_spell_count", "reference_spell_count")
VIEW_NAMES = ("clean", "feed_like_v1", "feed_like_v2")
EMIT = C.OUT / "emissions"


def emission_path(arm, key):
    return EMIT / arm / f"{key}.npz"


def _phases(cell):
    return sum(1 for name in cell if name.startswith("prob__ce__seed2026__phase"))


def _family(cell, family):
    return {seed: [cell[f"prob__{family}__seed{seed}__phase{ph}"] for ph in range(_phases(cell))]
            for seed in C.SEEDS}


def _arm_ce(arm, key, cell):
    """CE phases for an arm spec: 'control' or {'arm': weight, ...} seed-wise probability mixtures."""
    if arm == "control":
        return _family(cell, "ce")
    count = _phases(cell)
    out = {seed: [0.0] * count for seed in C.SEEDS}
    for name, weight in arm.items():
        if name == "control":
            source = _family(cell, "ce")
        else:
            with np.load(emission_path(name, key), allow_pickle=False) as d:
                source = {seed: [d[f"seed{seed}"][ph].astype(np.float64) for ph in range(count)] for seed in C.SEEDS}
        for seed in C.SEEDS:
            out[seed] = [acc + weight * p for acc, p in zip(out[seed], source[seed])]
    return out


def predict_cell(key, arm, weights=None, ha_cost=fp.HA_COST):
    cell = C.frontier_cell(key)
    weights = weights or fp.FAMILY_WEIGHTS
    pooled = {"ce": fp.pool_family(_arm_ce(arm, key, cell), 0.5),
              "calibrated": fp.pool_family(_family(cell, "calibrated"), 0.5),
              "adopted": fp.pool_family(_family(cell, "predecessor"), 0.5)}
    prob = fp.balanced.recipe_probability(pooled, weights["ce"], weights["calibrated"], fp.POOLING)
    pred = fp.decode_candidate(prob, cell["decoded__predecessor"], cell["model_period"], float(cell["fps"]), ha_cost)
    return cell, pred


def _job(args):
    key, arms = args
    C.configure()
    rows = {}
    for name, spec in arms.items():
        cell, pred = predict_cell(key, spec.get("ce", "control"), spec.get("weights"), spec.get("ha_cost", fp.HA_COST))
        score = fp._score_cell(dict(cell, fps=float(cell["fps"])), pred)
        rows[name] = {m: float(score[m]) for m in METRICS}
    return key, rows


def run(arms, workers=None):
    keys = [C.cell_key(*c) for c in C.cells()]
    with ProcessPoolExecutor(workers or int(os.environ.get("SCORE_WORKERS", "4"))) as pool:
        return dict(pool.map(_job, [(k, arms) for k in keys]))


def aggregate(rows, name):
    out = {}
    for view in VIEW_NAMES:
        groups = {}
        for key, cell in rows.items():
            provider, match, cell_view, _ = key.split("__")
            if cell_view == view:
                groups.setdefault(provider, {}).setdefault(match, []).append(cell[name])
        out[view] = {}
        for metric in METRICS:
            per = {p: float(np.mean([np.mean([r[metric] for r in reps]) for reps in ms.values()]))
                   for p, ms in groups.items()}
            out[view][metric] = float(np.mean(list(per.values())))
            out[view][metric + "/providers"] = per
    return out


def guards(candidate, control):
    """The frontier retention rule: every check must hold against the matched control."""
    failed = []
    for view in VIEW_NAMES:
        c, b = candidate[view], control[view]
        if view == "feed_like_v2":
            if not c["spell_f1"] > b["spell_f1"]:
                failed.append(f"{view}/spell_f1")
        elif c["spell_f1"] < b["spell_f1"]:
            failed.append(f"{view}/spell_f1")
        if c[SHORT] < b[SHORT]:
            failed.append(f"{view}/{SHORT}")
        if c["macro_f1"] < b["macro_f1"] - 0.001:
            failed.append(f"{view}/macro_f1")
    for provider, value in candidate["clean"]["macro_f1/providers"].items():
        if value < control["clean"]["macro_f1/providers"][provider] - 0.02:
            failed.append(f"clean/macro_f1/{provider}")
    return failed


def table(rows, names, control="control"):
    base = aggregate(rows, control)
    lines = [f"{'arm':34s} {'clean':>8s} {'v1':>8s} {'v2':>8s} | {'sr_c':>6s} {'sr_v1':>6s} {'sr_v2':>6s} | {'mac_v2':>7s} | v2 Met/SC/Spo | failed"]
    for name in names:
        a = aggregate(rows, name)
        pv = a["feed_like_v2"]["spell_f1/providers"]
        lines.append(f"{name:34s} {a['clean']['spell_f1']:8.6f} {a['feed_like_v1']['spell_f1']:8.6f} {a['feed_like_v2']['spell_f1']:8.6f} | "
                     f"{a['clean'][SHORT]:6.4f} {a['feed_like_v1'][SHORT]:6.4f} {a['feed_like_v2'][SHORT]:6.4f} | {a['feed_like_v2']['macro_f1']:7.5f} | "
                     f"{pv['Metrica']:.4f}/{pv['SkillCorner']:.4f}/{pv['Sportec']:.4f} | {','.join(guards(a, base)) if name != control else '-'}")
    return "\n".join(lines)
