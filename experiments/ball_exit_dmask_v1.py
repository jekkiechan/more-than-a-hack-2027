"""Label-free ball-exit correction of the frozen dead-ball mask (`ball_exit_dmask_v1`).

Every recent frontier study keeps the predecessor's decoded D mask, produced by a
player-only D path. This study edits only that mask with one physical rule: a ball
observed outside the field of play is dead. Two edits use tracking inputs alone:

* ``delay``: when a predicted D run starts while the ball is still observed inside
  the pitch and the ball is then observed leaving within ``WINDOW_S``, the run starts
  at the first outside observation instead.
* ``force``: frames with the ball observed outside the pitch are dead; the dead state
  holds until the ball is next observed inside or an existing D run begins, for at
  most ``HOLD_S``.

Nothing is fitted. It uses ball position relative to the pitch boundary only, never
ball speed or stillness, so it cannot use a provider's frozen stoppage coordinates.
H/A is then decoded inside the edited live spans with the unchanged frontier mixture.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os

import numpy as np

from experiments import sc2020_realfeed_common as C
from experiments import sc2020_realfeed_score as S
from experiments import frontier_postprocess_v1 as fp

STUDY = "ball_exit_dmask_v1"
WINDOW_S = 5.0
HOLD_S = 3.0
RULES = ("delay", "force", "both", "advance", "snap")
ROUND2 = {"advance_m0.0": ("advance", 0.0), "advance_m0.5": ("advance", 0.5),
          "snap_m0.0": ("snap", 0.0), "snap_m0.5": ("snap", 0.5)}
MARGINS = (0.0, 0.5)
ROUND = os.environ.get("BALL_EXIT_ROUND", "1")
DECLARATION = C.ROOT / ("experiments/results/ball_exit_dmask_v1_declaration.json" if ROUND == "1"
                        else "experiments/results/ball_exit_dmask_v1_round2_declaration.json")
RESULT = C.ROOT / ("experiments/results/ball_exit_dmask_v1.json" if ROUND == "1" else "experiments/results/ball_exit_dmask_v1_round2.json")
CONTROL = {"clean": 0.6701365499557103, "feed_like_v1": 0.532128306931943, "feed_like_v2": 0.5936187198896373}


def _runs(mask):
    edges = np.diff(np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def ball_state(ball, pitch, margin):
    """(inside, outside) boolean arrays; both False where the ball is unobserved."""
    ball = np.asarray(ball, dtype=np.float64)
    length, width = float(pitch[0]), float(pitch[1])
    seen = np.isfinite(ball).all(axis=1)
    x, y = np.where(seen, ball[:, 0], 0.0), np.where(seen, ball[:, 1], 0.0)
    beyond = np.maximum.reduce([-x, x - length, -y, y - width])
    return seen & (beyond <= 0.0), seen & (beyond > float(margin))


def edit_mask(dead, inside, outside, period, fps, rule):
    """Return the edited dead mask. Inputs are tracking-derived only; edits never cross a period block."""
    if rule not in RULES:
        raise ValueError(rule)
    dead = np.asarray(dead, dtype=bool)
    out = dead.copy()
    window, hold = int(round(WINDOW_S * fps)), int(round(HOLD_S * fps))
    for start, stop in fp.balanced.runtime.frozen.supervised.blocks(np.asarray(period)):
        if rule in ("delay", "both", "snap"):
            for a, b in _runs(dead[start:stop]):
                a, b = a + start, b + start
                if a == start:
                    continue
                seen_out = np.flatnonzero(outside[a:min(b, a + window)])
                if len(seen_out) and inside[a:a + seen_out[0]].any():
                    out[a:a + seen_out[0]] = False
        if rule in ("advance", "snap"):
            # Round 2: a D run also starts earlier when the ball was observed outside within HOLD_S
            # before it and not observed inside again. `snap` applies the delay edit first.
            for a, b in _runs(out[start:stop]):
                a = a + start
                if a == start:
                    continue
                low = max(start, a - hold)
                seen_in = np.flatnonzero(inside[low:a])
                after = low + seen_in[-1] + 1 if len(seen_in) else low
                seen_out = np.flatnonzero(outside[after:a])
                if len(seen_out) and not out[after + seen_out[0]:a].any():
                    out[after + seen_out[0]:a] = True
        if rule in ("force", "both"):
            i = start
            while i < stop:
                if outside[i] and not out[i]:
                    j = i
                    while j < stop and j - i < hold and not inside[j] and not out[j]:
                        out[j] = True
                        j += 1
                    i = j
                else:
                    i += 1
    return out


def arms():
    if os.environ.get("BALL_EXIT_ROUND", "1") == "2":
        return dict(ROUND2)
    return {f"{rule}_m{margin}": (rule, margin) for rule in ("delay", "force", "both") for margin in MARGINS}


def _job(key):
    C.configure()
    provider, match, view, rep = key.split("__")
    cell = C.frontier_cell(key)
    payload = C.prepared(provider, match, view, int(rep))
    inputs = payload["inputs"]
    fps = float(cell["fps"])
    pooled = {"ce": fp.pool_family(S._family(cell, "ce"), 0.5),
              "calibrated": fp.pool_family(S._family(cell, "calibrated"), 0.5),
              "adopted": fp.pool_family(S._family(cell, "predecessor"), 0.5)}
    prob = fp.balanced.recipe_probability(pooled, fp.FAMILY_WEIGHTS["ce"], fp.FAMILY_WEIGHTS["calibrated"], fp.POOLING)
    base = cell["decoded__predecessor"]
    scored = dict(cell, fps=fps)

    def score(mask_source):
        pred = fp.decode_candidate(prob, mask_source, cell["model_period"], fps)
        row = fp._score_cell(scored, pred)
        return {m: float(row[m]) for m in S.METRICS}

    rows = {"control": score(base)}
    for name, (rule, margin) in arms().items():
        inside, outside = ball_state(inputs["ball"], inputs["pitch"], margin)
        dead = edit_mask(base == "D", inside, outside, cell["model_period"], fps, rule)
        rows[name] = score(np.where(dead, "D", "H"))
        rows[name]["dead_frames_changed"] = float(np.mean(dead != (base == "D")))
    return key, rows


def guards(candidate, control):
    failed = S.guards(candidate, control)
    for provider, value in candidate["clean"]["d_f1/providers"].items():
        if value < control["clean"]["d_f1/providers"][provider] - 0.02:
            failed.append(f"clean/d_f1/{provider}")
    return failed


def main():
    C.configure()
    keys = [C.cell_key(*c) for c in C.cells()]
    with ProcessPoolExecutor(int(os.environ.get("SCORE_WORKERS", "4"))) as pool:
        rows = dict(pool.map(_job, keys))
    control = S.aggregate(rows, "control")
    for view, expected in CONTROL.items():
        if abs(control[view]["spell_f1"] - expected) > 1e-12:
            raise RuntimeError(f"Approved frontier control did not replay: {view}")
    out = {"schema": "ball_exit_dmask_v1_result", "study_id": STUDY,
           "declaration_sha256": hashlib.sha256(DECLARATION.read_bytes()).hexdigest(),
           "control_replay_exact": True, "organizer_reference_access": False, "fitting": False,
           "candidates": {}, "cells": rows}
    names = ["control"] + list(arms())
    for name in names:
        agg = S.aggregate(rows, name)
        failed = [] if name == "control" else guards(agg, control)
        out["candidates"][name] = {
            "aggregate": agg, "failed_retention_checks": failed, "retained": name != "control" and not failed,
            "meets_0_60": agg["feed_like_v2"]["spell_f1"] > 0.60,
            "spell_delta_vs_control": {v: agg[v]["spell_f1"] - control[v]["spell_f1"] for v in S.VIEW_NAMES},
            "cells_improved_v2": sum(1 for k, r in rows.items() if "feed_like_v2" in k and r[name]["spell_f1"] > r["control"]["spell_f1"]),
            "cells_worse_v2": sum(1 for k, r in rows.items() if "feed_like_v2" in k and r[name]["spell_f1"] < r["control"]["spell_f1"])}
    out["retained"] = sorted((n for n, c in out["candidates"].items() if c["retained"]),
                             key=lambda n: -out["candidates"][n]["aggregate"]["feed_like_v2"]["spell_f1"])
    out["success"] = any(out["candidates"][n]["meets_0_60"] for n in out["retained"])
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"{'arm':12s} {'clean':>8s} {'v1':>8s} {'v2':>8s} | sr c/v1/v2 | macro c/v1/v2 | dF1 c/v2 | v2 Met/SC/Spo | clean Met/SC/Spo | v2 cells +/- | failed")
    for name in names:
        c = out["candidates"][name]; a = c["aggregate"]
        pv, pc = a["feed_like_v2"]["spell_f1/providers"], a["clean"]["spell_f1/providers"]
        print(f"{name:12s} {a['clean']['spell_f1']:8.6f} {a['feed_like_v1']['spell_f1']:8.6f} {a['feed_like_v2']['spell_f1']:8.6f} | "
              f"{a['clean'][S.SHORT]:.3f}/{a['feed_like_v1'][S.SHORT]:.3f}/{a['feed_like_v2'][S.SHORT]:.3f} | "
              f"{a['clean']['macro_f1']:.4f}/{a['feed_like_v1']['macro_f1']:.4f}/{a['feed_like_v2']['macro_f1']:.4f} | "
              f"{a['clean']['d_f1']:.4f}/{a['feed_like_v2']['d_f1']:.4f} | {pv['Metrica']:.4f}/{pv['SkillCorner']:.4f}/{pv['Sportec']:.4f} | "
              f"{pc['Metrica']:.4f}/{pc['SkillCorner']:.4f}/{pc['Sportec']:.4f} | {c['cells_improved_v2']}/{c['cells_worse_v2']} | {','.join(c['failed_retention_checks']) or '-'}")
    print("retained:", out["retained"], "success:", out["success"])


if __name__ == "__main__":
    main()
