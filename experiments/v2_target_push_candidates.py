"""Phase-2 candidate families A-D (+composition) for `v2_target_push_v1`, scored through the harness.

All inference-side steps are label-free and provider-agnostic. Labels are used only by the scorer, by the
nearest-player sanity diagnostic (family C) and, for family D only, as provider-held-out training targets.
Usage: python -m experiments.v2_target_push_candidates {geom|A|B|C|D|compose}
"""
from __future__ import annotations

import datetime
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np

from experiments import v2_target_push_harness as H
from experiments import sc2020_realfeed_score as S
from experiments import ball_exit_dmask_v1 as BE
from experiments import frontier_postprocess_v1 as fp
from experiments import feed_benchmark_audit_v1 as audit

OUT = H.OUT
PRED = OUT / "predictions"
GEOM = OUT / "geom"
FEAT = OUT / "featD"
REFINED = OUT / "refinedD"
VIEWS = S.VIEW_NAMES
PROVIDERS = ("Metrica", "SkillCorner", "Sportec")


# ------------------------------------------------------------------ generic builder
def _blend_pD(cell, pD):
    P = cell.pooled["predecessor"]
    cond = P[:, :2] / P[:, :2].sum(1, keepdims=True)
    return np.column_stack((cond * (1 - pD)[:, None], pD))


def refined_pD(cell):
    return np.load(REFINED / f"{cell.key}.npy").astype(np.float64)


def geom(cell):
    with np.load(GEOM / f"{cell.key}.npz") as d:
        return d["dh"], d["da"]


def build(cell, spec):
    """spec keys: dt, cost, pD in {pred, avg, cal, d1, d2}, snap (bool), ha_cost, c=(r, lam)."""
    dt, cost, src = spec.get("dt", 4), spec.get("cost", 3), spec.get("pD", "pred")
    default = (dt, cost, src) == (4, 3, "pred") and not spec.get("assert_control")
    if default:
        base = cell.base_dead
    else:
        pred_pD, cal_pD = cell.pD["predecessor"], cell.pD["calibrated"]
        if src == "pred":
            P = cell.pooled["predecessor"]
        else:
            pD = {"avg": 0.5 * (pred_pD + cal_pD), "cal": cal_pD, "d1": None, "d2": None}[src]
            if src in ("d1", "d2"):
                ref = refined_pD(cell)
                pD = ref if src == "d1" else 0.5 * ref + 0.5 * pred_pD
            P = _blend_pD(cell, pD)
        base = audit.decode(P, cell.model_period, cell.fps, dict(audit.CONFIG, dt=dt, dead=cost)) == "D"
        if spec.get("assert_control"):
            assert np.array_equal(base, cell.base_dead), cell.key
    if spec.get("snap", True):
        inside, outside = BE.ball_state(cell.ball, cell.pitch, 0.0)
        dead = BE.edit_mask(base, inside, outside, cell.model_period, cell.fps, "snap")
    else:
        dead = base
    ha = cell.ha
    if spec.get("c"):
        r, lam = spec["c"]
        dh, da = geom(cell)
        d1, d2 = np.minimum(dh, da), np.maximum(dh, da)
        ok = np.isfinite(d1) & (d1 <= r) & (d2 - d1 >= r / 2)
        e = np.where(ok, np.where(dh < da, 1.0, -1.0), 0.0)
        logit = np.log(np.maximum(ha[:, 0], 1e-12)) - np.log(np.maximum(ha[:, 1], 1e-12)) + lam * e
        home = 1 / (1 + np.exp(-logit))
        ha = np.column_stack((home, 1 - home))
    return fp.decode_candidate(ha, np.where(dead, "D", "H"), cell.model_period, cell.fps, float(spec.get("ha_cost", 3.0)))


# ------------------------------------------------------------------ scoring wrapper
def run_arm(name, spec, family):
    d = PRED / name
    d.mkdir(parents=True, exist_ok=True)

    def fn(cell):
        pred = build(cell, spec)
        np.savez_compressed(d / f"{cell.key}.npz", pred=pred.astype("U1"))
        return pred
    res = H.evaluate(fn)
    ctrl = H._agg(H._control_rows()["snap_m0.0"])
    a = res["aggregate"]
    rec = {"arm": name, "family": family, "spec": spec,
           "spell": {v: a[v]["spell_f1"] for v in VIEWS}, "macro": {v: a[v]["macro_f1"] for v in VIEWS},
           "d_f1": {v: a[v]["d_f1"] for v in VIEWS}, "short": {v: a[v][S.SHORT] for v in VIEWS},
           "delta_spell_vs_snap": {v: a[v]["spell_f1"] - ctrl[v]["spell_f1"] for v in VIEWS},
           "delta_macro_vs_snap": {v: a[v]["macro_f1"] - ctrl[v]["macro_f1"] for v in VIEWS},
           "delta_dF1_vs_snap": {v: a[v]["d_f1"] - ctrl[v]["d_f1"] for v in VIEWS},
           "delta_short_vs_snap": {v: a[v][S.SHORT] - ctrl[v][S.SHORT] for v in VIEWS},
           "v2_spell_by_provider": a["feed_like_v2"]["spell_f1/providers"],
           "v2_spell_delta_by_provider": {p: a["feed_like_v2"]["spell_f1/providers"][p] - ctrl["feed_like_v2"]["spell_f1/providers"][p] for p in PROVIDERS},
           "failed_retention_checks": res["failed_retention_checks"], "retained": res["retained"], "meets_0_60": res["meets_0_60"],
           "eligible_and_meets_0_60": res["retained"] and res["meets_0_60"]}
    if not rec["retained"]:
        shutil.rmtree(d, ignore_errors=True)
    return rec


def declare(family, arms, extra=None):
    ctrl = H._agg(H._control_rows()["snap_m0.0"])
    decl = {"family": family, "declared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
            "control": "snap_m0.0", "control_spell": {v: ctrl[v]["spell_f1"] for v in VIEWS},
            "arms": {k: v for k, v in arms.items()}, **(extra or {})}
    (OUT / f"declaration_{family}.json").write_text(json.dumps(decl, indent=1, default=float) + "\n")


def run_family(family, arms, extra=None):
    declare(family, arms, extra)
    recs = []
    for name, spec in arms.items():
        t = time.time()
        rec = run_arm(name, spec, family)
        recs.append(rec)
        print(f"{name:28s} " + " ".join(f"{rec['spell'][v]:.6f}" for v in VIEWS) + f" d_v2 {rec['delta_spell_vs_snap']['feed_like_v2']:+.5f} "
              f"fail={rec['failed_retention_checks']} ({time.time()-t:.0f}s)", flush=True)
    (OUT / f"results_{family}.json").write_text(json.dumps({"declaration": f"declaration_{family}.json", "arms": recs}, indent=1, default=float) + "\n")
    return recs


# ------------------------------------------------------------------ families
def arms_A():
    arms = {f"A_dt{dt}_c{c}": {"dt": dt, "cost": c} for dt in (3, 4, 5, 6) for c in (2, 3, 4) if (dt, c) != (4, 3)}
    arms["A_dt4_c3"] = {"dt": 4, "cost": 3, "assert_control": True}
    arms["A_pD_avg_dt4_c3"] = {"pD": "avg"}
    arms["A_pD_cal_dt4_c3"] = {"pD": "cal"}
    return arms


def arms_B():
    return {f"B_ha{h}": {"ha_cost": h} for h in (2.0, 2.5, 3.5, 4.0)}


def arms_C():
    return {f"C_r{r}_l{l}": {"c": (r, l)} for r in (1.0, 1.5, 2.5) for l in (0.5, 1.0, 2.0)}


def arms_D():
    return {"D1_refined": {"pD": "d1"}, "D2_half_refined": {"pD": "d2"}, "D3_refined_nosnap": {"pD": "d1", "snap": False}}


# ------------------------------------------------------------------ geometry cache and nearest-player sanity
def _geom_job(i):
    cell = H._STATE["cells"][i]
    players = cell.players().astype(np.float64)
    teams = np.asarray(C_prepared(cell)["inputs"]["teams"])
    d = np.linalg.norm(players - cell.ball[:, None, :], axis=2)
    d = np.where(np.isfinite(d), d, np.inf)
    dh = np.where(teams == 0, d, np.inf).min(1).astype(np.float32)
    da = np.where(teams == 1, d, np.inf).min(1).astype(np.float32)
    np.savez(GEOM / f"{cell.key}.npz", dh=dh, da=da)
    return cell.key


def C_prepared(cell):
    return H.C.prepared(cell.provider, cell.match, cell.view, cell.rep)


def build_geom():
    GEOM.mkdir(parents=True, exist_ok=True)
    H.load_cells()
    todo = [i for i, c in enumerate(H._STATE["cells"]) if not (GEOM / f"{c.key}.npz").is_file()]
    with mp.get_context("fork").Pool(3) as pool:
        pool.map(_geom_job, todo, chunksize=1)


def sanity():
    out = {}
    for prov in PROVIDERS:
        for r in (1.0,):
            n = agree = nlive = 0
            for c in H.load_cells():
                if c.view != "clean" or c.provider != prov:
                    continue
                dh, da = geom(c)
                d1 = np.minimum(dh, da)
                sel = c.ball_obs & (d1 <= r) & np.isin(c.labels, ("H", "A"))
                team = np.where(dh < da, "H", "A")
                n += int(sel.sum())
                agree += int((team[sel] == c.labels[sel]).sum())
                nlive += int(np.isin(c.labels, ("H", "A")).sum())
        out[prov] = {"frames_ball_observed_nearest_within_1m_live": n, "frac_of_live_frames": n / max(nlive, 1), "nearest_team_equals_label": agree / max(n, 1)}
    (OUT / "sanity_nearest_player.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "geom":
        build_geom()
        print(json.dumps(sanity(), indent=1))
    elif what in ("A", "B", "C", "D"):
        run_family(what, {"A": arms_A, "B": arms_B, "C": arms_C, "D": arms_D}[what]())
