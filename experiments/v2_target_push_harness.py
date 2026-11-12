"""Fast development harness for the v2 spell-F1 target push (`v2_target_push_v1`).

Scope: the 35 exposed development cells (5 matches x clean/v1 x2/v2 x4). Labels are used only inside
the official scorer and in explicit diagnostics; candidate functions receive a label-free view
(`cell.labels` exists for diagnostics -- a candidate must not read it).

API
---
    from experiments import v2_target_push_harness as H
    cells = H.load_cells()                  # list[Cell]; builds a compact cache on first call
    res = H.evaluate(fn)                    # fn(cell) -> str label array (H/A/D/X) | bool dead mask
    res["aggregate"][view][metric]          # views clean/feed_like_v1/feed_like_v2; metrics spell_f1,
    res["aggregate"][view][metric + "/providers"]   # macro_f1, d_f1, short recall, pred/ref spell counts
    res["failed_retention_checks"]          # repo retention guards vs. the snap_m0.0 control
    H.control_replay()                      # frontier + snap_m0.0 replay, asserts 1e-9 match

A bool mask is "True = dead"; H/A is then decoded inside the live spans by the unchanged frontier
decoder (`cell.decode(dead)`). Cell fields: labels, period, model_period, timestamps, fps, ha (N,2 pooled
conditional H/A, frontier mixture), pD {ce,calibrated,predecessor} (N,), pooled {family: (N,3)},
base_dead, snap_dead (bool), ball (N,2) native metres (NaN = unobserved), pitch (2,), ball_obs,
n_players_obs (N,), cell.players() -> (N,40,2) lazily read from the prepared joblib.
Workers use fork; fn may be a closure. Full 35-cell evaluation ~ see the `elapsed` field.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from experiments import sc2020_realfeed_common as C
from experiments import sc2020_realfeed_score as S
from experiments import frontier_postprocess_v1 as fp
from experiments import ball_exit_dmask_v1 as BE

OUT = C.ROOT / "data/experiments/v2_target_push_v1"
CACHE = OUT / "cache"
FAMILIES = (("ce", "ce"), ("calibrated", "calibrated"), ("predecessor", "adopted"))
CONTROL_FRONTIER = BE.CONTROL
CONTROL_SNAP = {"clean": 0.681508, "feed_like_v1": 0.534731, "feed_like_v2": 0.598690}
SNAP_RULE = ("snap", 0.0)
VIEWS = S.VIEW_NAMES
_STATE = {}


class Cell:
    """One development cell; plain attribute container with decode/score helpers."""

    def __init__(self, key, d):
        self.key = key
        self.provider, self.match, self.view, rep = key.split("__")
        self.rep = int(rep)
        self.__dict__.update(d)
        self.fps = float(self.fps)
        self.n = len(self.labels)

    def decode(self, dead):
        """Frontier H/A decode inside live spans of `dead` (bool, True = D). Returns U1 labels."""
        mask = np.where(np.asarray(dead, bool), "D", "H")
        return fp.decode_candidate(self.prob_ha, mask, self.model_period, self.fps)

    def score(self, pred):
        row = fp._score_cell({"labels": self.labels, "period": self.period, "timestamps": self.timestamps,
                              "fps": self.fps}, pred)
        return {m: float(row[m]) for m in S.METRICS}

    def full_score(self, pred):
        return fp._score_cell({"labels": self.labels, "period": self.period, "timestamps": self.timestamps,
                               "fps": self.fps}, pred)

    def players(self):
        return C.prepared(self.provider, self.match, self.view, self.rep)["inputs"]["players"]

    def control_pred(self):
        return fp.decode_candidate(self.prob_ha, self.base_dead_str, self.model_period, self.fps)

    def pred_from(self, out):
        out = np.asarray(out)
        return self.decode(out) if out.dtype == bool else out.astype("U1")


def _build(key):
    C.configure()
    provider, match, view, rep = key.split("__")
    cell = C.frontier_cell(key)
    payload = C.prepared(provider, match, view, int(rep))
    inputs = payload["inputs"]
    fps = float(cell["fps"])
    pooled = {fam: fp.pool_family(S._family(cell, fam), 0.5) for fam, _ in FAMILIES}
    prob = fp.balanced.recipe_probability({"ce": pooled["ce"], "calibrated": pooled["calibrated"],
                                           "adopted": pooled["predecessor"]},
                                          fp.FAMILY_WEIGHTS["ce"], fp.FAMILY_WEIGHTS["calibrated"], fp.POOLING)
    base = cell["decoded__predecessor"]
    base_dead = base == "D"
    inside, outside = BE.ball_state(inputs["ball"], inputs["pitch"], 0.0)
    snap = BE.edit_mask(base_dead, inside, outside, cell["model_period"], fps, "snap")
    ball = np.asarray(inputs["ball"], np.float64)
    out = dict(labels=cell["labels"], period=cell["period"], model_period=cell["model_period"],
               timestamps=cell["timestamps"], fps=np.float64(fps), prob_ha=prob,
               pD_ce=pooled["ce"][:, 2], pD_calibrated=pooled["calibrated"][:, 2],
               pD_predecessor=pooled["predecessor"][:, 2],
               pooled_ce=pooled["ce"], pooled_calibrated=pooled["calibrated"], pooled_predecessor=pooled["predecessor"],
               base_dead=base_dead, snap_dead=snap, ball=ball, pitch=np.asarray(inputs["pitch"], np.float64),
               ball_obs=np.isfinite(ball).all(axis=1), n_players_obs=(np.asarray(inputs["player_observed"]) if "player_observed" in inputs else np.isfinite(inputs["players"]).all(-1)).sum(1).astype(np.int16),
               approved_baseline=cell["decoded__approved_baseline"])
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = CACHE / f".{key}.{os.getpid()}.npz"
    np.savez(tmp, **out)
    os.replace(tmp, CACHE / f"{key}.npz")
    return key


def _read(key):
    with np.load(CACHE / f"{key}.npz", allow_pickle=False) as d:
        data = {k: d[k] for k in d.files}
    data["pD"] = {"ce": data.pop("pD_ce"), "calibrated": data.pop("pD_calibrated"), "predecessor": data.pop("pD_predecessor")}
    data["pooled"] = {f: data.pop(f"pooled_{f}") for f in ("ce", "calibrated", "predecessor")}
    data["ha"] = data["prob_ha"]
    data["base_dead_str"] = np.where(data["base_dead"], "D", "H")
    data["fps"] = float(data["fps"])
    return Cell(key, data)


def keys():
    return [C.cell_key(*c) for c in C.cells()]


def build_cache(workers=3):
    todo = [k for k in keys() if not (CACHE / f"{k}.npz").is_file()]
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(workers) as pool:
            list(pool.map(_build, todo))


def load_cells():
    """All 35 cells (cached in process; compact on-disk cache built once)."""
    if "cells" not in _STATE:
        build_cache()
        _STATE["cells"] = [_read(k) for k in keys()]
    return _STATE["cells"]


def cell_map():
    return {c.key: c for c in load_cells()}


def _worker(i):
    cell = _STATE["cells"][i]
    fn = _STATE["fn"]
    out = fn(cell)
    pred = cell.pred_from(out)
    row = cell.score(pred)
    return cell.key, row


def _run_rows(fn, workers=4):
    cells = load_cells()
    _STATE["fn"] = fn
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        res = pool.map(_worker, range(len(cells)), chunksize=1)
    _STATE.pop("fn", None)
    return dict(res)


def _agg(rows):
    return S.aggregate({k: {"x": r} for k, r in rows.items()}, "x")


def _control_rows():
    path = OUT / "control_rows.json"
    if path.is_file():
        return json.loads(path.read_text())
    frontier = _run_rows(lambda c: c.control_pred())
    snap = _run_rows(lambda c: c.snap_dead)
    out = {"frontier": frontier, "snap_m0.0": snap}
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


def evaluate(fn, workers=4, control="snap_m0.0"):
    """Score `fn` on all 35 cells. Returns aggregate, per-cell rows, retention verdict vs `control`."""
    t = time.time()
    rows = _run_rows(fn, workers)
    agg = _agg(rows)
    ctrl = _agg(_control_rows()[control])
    failed = BE.guards(agg, ctrl)
    return {"aggregate": agg, "rows": rows, "failed_retention_checks": failed,
            "retained": not failed, "meets_0_60": agg["feed_like_v2"]["spell_f1"] > 0.60,
            "delta_vs_control": {v: agg[v]["spell_f1"] - ctrl[v]["spell_f1"] for v in VIEWS},
            "elapsed": time.time() - t}


def summarize(res):
    a = res["aggregate"]
    return " ".join(f"{v[:7]}:{a[v]['spell_f1']:.6f}" for v in VIEWS) + \
        f" | macro " + "/".join(f"{a[v]['macro_f1']:.6f}" for v in VIEWS) + \
        f" | dF1 " + "/".join(f"{a[v]['d_f1']:.4f}" for v in VIEWS) + \
        f" | short " + "/".join(f"{a[v][S.SHORT]:.3f}" for v in VIEWS) + \
        f" | fail {res['failed_retention_checks'] or '-'}"


def control_replay(tol=1e-9):
    """Replay frontier control and snap_m0.0; assert exact agreement with the frozen numbers."""
    c = _control_rows()
    out = {}
    for name, ref in (("frontier", CONTROL_FRONTIER), ("snap_m0.0", None)):
        agg = _agg(c[name])
        out[name] = agg
        if ref:
            for v, e in ref.items():
                assert abs(agg[v]["spell_f1"] - e) < tol, (name, v, agg[v]["spell_f1"], e)
    for v, e in CONTROL_SNAP.items():
        assert abs(out["snap_m0.0"][v]["spell_f1"] - e) < 5e-7, (v, out["snap_m0.0"][v]["spell_f1"], e)
    return out


if __name__ == "__main__":
    r = control_replay()
    for name, a in r.items():
        print(name)
        for v in VIEWS:
            print(f"  {v:13s} spell {a[v]['spell_f1']:.12f} macro {a[v]['macro_f1']:.6f} dF1 {a[v]['d_f1']:.6f} "
                  f"short {a[v][S.SHORT]:.4f} pred/ref {a[v]['predicted_spell_count']:.1f}/{a[v]['reference_spell_count']:.1f}")
    t = time.time()
    res = evaluate(lambda c: c.snap_dead)
    print("snap re-evaluation", summarize(res), f"{time.time()-t:.1f}s")
