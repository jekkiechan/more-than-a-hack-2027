"""Error-decomposition diagnostics versus `snap_m0.0` (phase 1 of `v2_target_push_v1`).

Development cells only. Labels are read only for diagnostics/oracle rungs here, never by a candidate.
Run: python -m experiments.v2_target_push_diagnostics  -> data/experiments/v2_target_push_v1/diagnostics.{json,md}
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time

import numpy as np

from experiments import v2_target_push_harness as H
from experiments import sc2020_realfeed_score as S
from experiments import ball_exit_dmask_v1 as BE
from experiments import metrics, feed_benchmark_audit_v1 as audit
import evaluation.possession_metrics as pm

OUT = H.OUT
VIEWS = S.VIEW_NAMES
BUCKETS = ((0, 1, "<1s"), (1, 3, "1-3s"), (3, 10, "3-10s"), (10, 1e9, ">10s"))


def bucket(sec):
    for lo, hi, name in BUCKETS:
        if lo <= sec < hi:
            return name


def cuts_of(cell):
    return metrics.prepare_tracking_codes(cell.labels, np.full(cell.n, "H"), cell.period, cell.fps, cell.timestamps)[3]


def runs_of(mask, cuts):
    """Half-open runs of a bool mask, split at evaluation separators."""
    mask = np.asarray(mask, bool)
    start_edge = mask & ~np.concatenate(([False], mask[:-1]))
    end_edge = mask & ~np.concatenate((mask[1:], [False]))
    starts = set(np.flatnonzero(start_edge).tolist())
    ends = set((np.flatnonzero(end_edge) + 1).tolist())
    for c in cuts:
        if 0 < c < len(mask) and mask[c - 1] and mask[c]:
            ends.add(int(c))
            starts.add(int(c))
    return list(zip(sorted(starts), sorted(ends)))


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def true_runs(cell, cuts):
    return runs_of(cell.labels == "D", cuts)


def agg_rows(res):
    a = res["aggregate"]
    return {v: {"spell_f1": a[v]["spell_f1"], "providers": a[v]["spell_f1/providers"],
                "macro_f1": a[v]["macro_f1"], "d_f1": a[v]["d_f1"], "short": a[v][S.SHORT],
                "pred_spells": a[v]["predicted_spell_count"], "ref_spells": a[v]["reference_spell_count"]}
            for v in VIEWS}


# ------------------------------------------------------------------ (a) oracle ladder
def oracle_starts_ends(pred_runs, tr, starts=True, ends=True):
    new = []
    cover = {}
    for i, p in enumerate(pred_runs):
        for j, t in enumerate(tr):
            if overlap(p, t):
                cover.setdefault(j, []).append(i)
    best = {}
    for i, p in enumerate(pred_runs):
        ov = [(overlap(p, t), j) for j, t in enumerate(tr) if overlap(p, t)]
        if ov:
            best[i] = max(ov)[1]
    for i, p in enumerate(pred_runs):
        s, e = p
        if i in best:
            j = best[i]
            if starts and cover[j][0] == i:
                s = min(tr[j][0], e - 1) if tr[j][0] < e else s
            if ends and cover[j][-1] == i:
                e = max(tr[j][1], s + 1) if tr[j][1] > s else e
        new.append((s, e))
    return new


def runs_to_mask(runs, n):
    m = np.zeros(n, bool)
    for a, b in runs:
        m[a:b] = True
    return m


def ladder_fn(name):
    def fn(cell):
        cuts = cuts_of(cell)
        dead = cell.snap_dead
        pr = runs_of(dead, cuts)
        tr = true_runs(cell, cuts)
        truth_dead = cell.labels == "D"
        ha = lambda mask: cell.decode(mask)
        if name == "snap_control":
            return dead
        if name == "i_oracle_D_mask":
            return truth_dead
        if name in ("ii_oracle_D_starts", "iii_oracle_D_ends", "viii_oracle_D_starts_and_ends"):
            new = oracle_starts_ends(pr, tr, starts=name != "iii_oracle_D_ends", ends=name != "ii_oracle_D_starts")
            return runs_to_mask(new, cell.n)
        false_runs = [p for p in pr if not any(overlap(p, t) for t in tr)]
        missed = [t for t in tr if not any(overlap(p, t) for p in pr)]
        if name == "iv_remove_false_D_runs":
            return runs_to_mask([p for p in pr if p not in false_runs], cell.n)
        if name == "v_add_missed_D_runs":
            return runs_to_mask(pr + missed, cell.n)
        if name == "iv_v_both":
            return runs_to_mask([p for p in pr if p not in false_runs] + missed, cell.n)
        pred = ha(dead)
        live = ~dead
        lab = cell.labels
        if name == "vi_oracle_HA_in_live":
            out = pred.copy()
            sel = live & np.isin(lab, ("H", "A"))
            out[sel] = lab[sel]
            return out
        if name in ("vii_a_one_true_team_per_live_span", "vii_b_true_team_boundaries_in_live"):
            out = pred.copy()
            for a, b in runs_of(live, cuts):
                seg = lab[a:b]
                ha_mask = np.isin(seg, ("H", "A"))
                if not ha_mask.any():
                    continue
                if name.startswith("vii_a"):
                    team = "H" if (seg == "H").sum() >= (seg == "A").sum() else "A"
                    out[a:b] = team
                else:
                    idx = np.where(ha_mask, np.arange(len(seg)), -1)
                    fwd = np.maximum.accumulate(idx)
                    idx2 = np.where(ha_mask, np.arange(len(seg)), len(seg))
                    bwd = np.minimum.accumulate(idx2[::-1])[::-1]
                    use = np.where(fwd >= 0, fwd, bwd)
                    out[a:b] = seg[use]
            return out
        raise ValueError(name)
    return fn


LADDER = ["snap_control", "i_oracle_D_mask", "ii_oracle_D_starts", "iii_oracle_D_ends", "viii_oracle_D_starts_and_ends",
          "iv_remove_false_D_runs", "v_add_missed_D_runs", "iv_v_both", "vi_oracle_HA_in_live",
          "vii_a_one_true_team_per_live_span", "vii_b_true_team_boundaries_in_live"]


def ladder():
    return {name: agg_rows(H.evaluate(ladder_fn(name))) for name in LADDER}


# ------------------------------------------------------------------ per-cell diagnostics (b), (c), (d), (e)
def _pool_cell(i):
    cell = H._STATE["cells"][i]
    return cell.key, {"b": diag_b(cell), "c": diag_c(cell), "d": diag_d(cell), "e": diag_e(cell), "g": diag_g(cell)}


def diag_b(cell):
    cuts = cuts_of(cell)
    graded = cell.labels != "X"
    tr = true_runs(cell, cuts)
    out = {}
    for mname, dead in (("base", cell.base_dead), ("snap", cell.snap_dead)):
        pr = runs_of(dead & graded, cuts)
        cutset = set(int(c) for c in cuts)
        ev, ee, pairs = [], [], []
        cand = sorted(((overlap(p, t) / (max(p[1], t[1]) - min(p[0], t[0])), i, j)
                       for i, p in enumerate(pr) for j, t in enumerate(tr) if overlap(p, t)), reverse=True)
        used_p, used_t = set(), set()
        for iou, i, j in cand:
            if iou < 0.2 or i in used_p or j in used_t:
                continue
            used_p.add(i), used_t.add(j)
            p, t = pr[i], tr[j]
            # skip pred edges created by the graded-region/period clip rather than by the mask
            start_ok = p[0] > 0 and graded[p[0] - 1] and p[0] not in cutset
            end_ok = p[1] < cell.n and graded[p[1]] and p[1] not in cutset
            if start_ok:
                ev.append((p[0] - t[0]) / cell.fps)
            if end_ok:
                ee.append((p[1] - t[1]) / cell.fps)
        fr = [p for p in pr if not any(overlap(p, t) for t in tr)]
        ms = [t for t in tr if not any(overlap(p, t) for p in pr)]
        out[mname] = {"start_err": ev, "end_err": ee, "n_pred_runs": len(pr), "n_true_runs": len(tr), "n_matched": len(used_p),
                      "false_runs": [(b - a) / cell.fps for a, b in fr], "missed_runs": [(b - a) / cell.fps for a, b in ms],
                      "true_run_dur": [(b - a) / cell.fps for a, b in tr], "pred_run_dur": [(b - a) / cell.fps for a, b in pr]}
    return out


def _greedy(overlaps, n_gt, n_pred, thr):
    gt_m, pr_m = {}, {}
    for iou, i, j in overlaps:
        if iou < thr:
            break
        if i in gt_m or j in pr_m:
            continue
        gt_m[i], pr_m[j] = j, i
    return gt_m, pr_m


def diag_c(cell, thr=0.5):
    pred = cell.pred_from(cell.snap_dead)
    _, gt, pr, ins = metrics.prepare_tracking_codes(cell.labels, pred, cell.period, cell.fps, cell.timestamps)
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pr, runs)
    ov = pm.spell_overlaps(gs, ps)
    gm, pm_ = _greedy(ov, len(gs), len(ps), thr)
    fps = cell.fps
    out = {"fn": {}, "fp": {}, "n_ref": len(gs), "n_pred": len(ps), "ref_matched": len(gm)}
    for i, (a, b, T) in enumerate(gs):
        dur = (b - a) / fps
        out.setdefault("ref_by_bucket", {}).setdefault(bucket(dur), [0, 0])
        out["ref_by_bucket"][bucket(dur)][0] += 1
        if i in gm:
            out["ref_by_bucket"][bucket(dur)][1] += 1
            continue
        seg = pr[a:b]
        other = 1 - T
        fD, fT, fO = (seg == pm.D).mean(), (seg == T).mean(), (seg == other).mean()
        overl = [(j, ps[j]) for (_, ii, j) in ov if ii == i]
        if fD >= 0.5:
            cat = "covered_by_D"
        elif fT + fO > 0 and fT / (fT + fO) < 0.5:
            cat = "wrong_team"
        else:
            n = len(overl)
            if n >= 2:
                cat = "fragmented_by_team_switch" if fO > 0 else "fragmented_by_D_gap"
            elif n == 1:
                p = overl[0][1]
                cat = "merged_with_neighbour" if (p[1] - p[0]) > (b - a) else "boundary_misalignment_only"
            else:
                cat = "other"
        out["fn"].setdefault(cat, {}).setdefault(bucket(dur), 0)
        out["fn"][cat][bucket(dur)] += 1
    for j, (a, b, T) in enumerate(ps):
        dur = (b - a) / fps
        out.setdefault("pred_by_bucket", {}).setdefault(bucket(dur), [0, 0])
        out["pred_by_bucket"][bucket(dur)][0] += 1
        if j in pm_:
            out["pred_by_bucket"][bucket(dur)][1] += 1
            continue
        seg = gt[a:b]
        other = 1 - T
        fD, fT, fO = (seg == pm.D).mean(), (seg == T).mean(), (seg == other).mean()
        overl = [(i, gs[i]) for (_, i, jj) in ov if jj == j]
        if fD >= 0.5:
            cat = "true_dead"
        elif fT + fO > 0 and fT / (fT + fO) < 0.5:
            cat = "wrong_team"
        else:
            n = len(overl)
            if n >= 2:
                cat = "spans_multiple_true_spells"
            elif n == 1:
                g = overl[0][1]
                cat = ("fragment_of_longer_true_spell" if (b - a) < (g[1] - g[0])
                       else "extends_beyond_true_spell" if (b - a) > (g[1] - g[0]) else "boundary_misalignment_only")
            else:
                cat = "other"
        out["fp"].setdefault(cat, {}).setdefault(bucket(dur), 0)
        out["fp"][cat][bucket(dur)] += 1
    return out


def ball_dist(ball, pitch):
    L, W = float(pitch[0]), float(pitch[1])
    x, y = ball[:, 0], ball[:, 1]
    return np.minimum.reduce([x, L - x, y, W - y])  # signed: negative once outside


def _speeds(ball, fps, lo, hi):
    idx = np.flatnonzero(np.isfinite(ball[lo:hi]).all(1)) + lo
    if len(idx) < 2:
        return np.array([])
    d = np.linalg.norm(np.diff(ball[idx], axis=0), axis=1)
    return d / (np.diff(idx) / fps)


def diag_d(cell):
    inside, outside = BE.ball_state(cell.ball, cell.pitch, 0.0)
    delay = BE.edit_mask(cell.base_dead, inside, outside, cell.model_period, cell.fps, "delay")
    edit = cell.base_dead & ~delay
    dist = ball_dist(cell.ball, cell.pitch)
    fps = cell.fps
    rows = []
    for a, b in runs_of(edit, cuts_of(cell)):
        k = b - a
        seg = cell.labels[a:b]
        graded = (seg != "X").sum()
        if not graded:
            continue
        live = np.isin(seg, ("H", "A")).sum() / graded
        sp_edit = _speeds(cell.ball, fps, max(0, a - 1), b + 1)
        sp_pre = _speeds(cell.ball, fps, max(0, a - int(fps)), a + 1)
        win = np.arange(max(0, a - int(fps / 2)), min(cell.n, a + int(fps / 2) + 1))
        win = win[cell.ball_obs[win]]
        d_a = float(dist[win[np.argmin(np.abs(win - a))]]) if len(win) else np.nan
        obs = np.flatnonzero(cell.ball_obs[a:b + 1]) + a  # includes the exit observation at b
        dseries = dist[obs]
        steps = np.diff(dseries)
        rows.append({"start": int(a), "n": int(k), "delay_s": k / fps, "speed_edit": float(np.median(sp_edit)) if len(sp_edit) else np.nan,
                     "speed_pre": float(np.median(sp_pre)) if len(sp_pre) else np.nan, "dist_at_start": d_a,
                     "monotone_toward": float(len(steps) > 0 and steps.max() <= 0.5), "net_approach_m": float(dseries[0] - dseries[-1]) if len(dseries) > 1 else np.nan,
                     "approach_rate_mps": float((dseries[0] - dseries[-1]) / (k / fps)) if len(dseries) > 1 else np.nan,
                     "frac_ball_obs": float(cell.ball_obs[a:b].mean()), "n_obs": int(len(obs)), "frac_live": float(live),
                     "graded": int(graded)})
    return {"edits": rows}


def diag_e(cell):
    p = {f: cell.pD[f] for f in ("ce", "calibrated", "predecessor")}
    pairs = {"ce-cal": np.abs(p["ce"] - p["calibrated"]), "ce-pred": np.abs(p["ce"] - p["predecessor"]),
             "cal-pred": np.abs(p["calibrated"] - p["predecessor"])}
    d = audit.decode(cell.pooled["predecessor"], cell.model_period, cell.fps)
    out = {k: {"max": float(v.max()), "mean": float(v.mean())} for k, v in pairs.items()}
    out["f062_decode_matches_base_mask"] = bool(np.array_equal(d == "D", cell.base_dead))
    out["mask_vs_pD_gt_0.5_agree"] = float(((p["predecessor"] > 0.5) == cell.base_dead).mean())
    pd_ = p["predecessor"]
    out["pD_mean_in_mask"] = float(pd_[cell.base_dead].mean())
    out["pD_mean_out_mask"] = float(pd_[~cell.base_dead].mean())
    best = max(((float(((pd_ > t) == cell.base_dead).mean()), t) for t in np.linspace(0.2, 0.8, 25)))
    out["best_threshold_agree"], out["best_threshold"] = best
    out["n_runs_mask"] = len(runs_of(cell.base_dead, cuts_of(cell)))
    out["n_runs_thresh0.5"] = len(runs_of(pd_ > 0.5, cuts_of(cell)))
    return out


def diag_g(cell, thr=0.5):
    """Extra: unmatched short (<3 s) predicted spells that sit over truly-dead frames (snap mask)."""
    pred = cell.pred_from(cell.snap_dead)
    _, gt, pr, ins = metrics.prepare_tracking_codes(cell.labels, pred, cell.period, cell.fps, cell.timestamps)
    omap = np.insert(np.arange(cell.n), ins, -1)
    runs = pm.valid_runs(gt)
    gs, ps = pm.possession_spells(gt, runs), pm.possession_spells(pr, runs)
    ov = pm.spell_overlaps(gs, ps)
    _, pm_ = _greedy(ov, len(gs), len(ps), thr)
    rows = []
    for j, (a, b, T) in enumerate(ps):
        if j in pm_ or (b - a) / cell.fps >= 3.0 or (gt[a:b] == pm.D).mean() < 0.5:
            continue
        left = a > 0 and pr[a - 1] == pm.D
        right = b < len(pr) and pr[b] == pm.D
        idx = omap[a:b]
        rows.append({"dur": (b - a) / cell.fps, "flank_D_both": bool(left and right), "flank_D_any": bool(left or right),
                     "pD_mean": float(cell.pD["predecessor"][idx[idx >= 0]].mean()),
                     "ball_obs": float(cell.ball_obs[idx[idx >= 0]].mean())})
    return {"short_true_dead_fp": rows}


def summarize_g(per):
    out = {}
    for view in ("clean", "feed_like_v2"):
        out[view] = {}
        for prov in ("all",) + PROVIDERS:
            rows = [r for k, c in by(per, view, None if prov == "all" else prov).items() for r in c["g"]["short_true_dead_fp"]]
            if rows:
                out[view][prov] = {"n": len(rows), "frac_flanked_by_pred_D_both_sides": float(np.mean([r["flank_D_both"] for r in rows])),
                                   "frac_flanked_any_side": float(np.mean([r["flank_D_any"] for r in rows])),
                                   "mean_pD_pred_in_span": float(np.mean([r["pD_mean"] for r in rows])),
                                   "frac_ball_observed": float(np.mean([r["ball_obs"] for r in rows])),
                                   "duration_s": qstats([r["dur"] for r in rows])}
    return out


def percall():
    H.load_cells()
    ctx = mp.get_context("fork")
    with ctx.Pool(4) as pool:
        return dict(pool.map(_pool_cell, range(len(H._STATE["cells"])), chunksize=1))


# ------------------------------------------------------------------ aggregation helpers
def by(per, view=None, provider=None):
    return {k: v for k, v in per.items()
            if (view is None or k.split("__")[2] == view) and (provider is None or k.split("__")[0] == provider)}


PROVIDERS = ("Metrica", "SkillCorner", "Sportec")


def qstats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    q = np.percentile(x, [10, 25, 50, 75, 90])
    return {"n": int(len(x)), "p10": q[0], "p25": q[1], "median": q[2], "p75": q[3], "p90": q[4], "iqr": q[3] - q[1], "mean": float(x.mean())}


def summarize_b(per):
    out = {}
    for mname in ("base", "snap"):
        out[mname] = {}
        for view in VIEWS:
            out[mname][view] = {}
            for prov in ("all",) + PROVIDERS:
                cells = by(per, view, None if prov == "all" else prov)
                get = lambda f: [x for c in cells.values() for x in c["b"][mname][f]]
                d = {"start_err_s": qstats(get("start_err")), "end_err_s": qstats(get("end_err")),
                     "n_cells": len(cells), "n_pred_runs": sum(c["b"][mname]["n_pred_runs"] for c in cells.values()),
                     "n_true_runs": sum(c["b"][mname]["n_true_runs"] for c in cells.values()),
                     "n_matched_iou0.2": sum(c["b"][mname]["n_matched"] for c in cells.values())}
                for kind in ("false_runs", "missed_runs"):
                    vals = np.array(get(kind), float)
                    d[kind] = {name: {"count": int(((vals >= lo) & (vals < hi)).sum()), "total_s": float(vals[(vals >= lo) & (vals < hi)].sum())}
                               for lo, hi, name in BUCKETS}
                    d[kind]["all"] = {"count": int(len(vals)), "total_s": float(vals.sum())}
                d["divide_by_n_cells_for_per_cell_mean"] = len(cells)
                out[mname][view][prov] = d
    return out


def summarize_c(per):
    out = {}
    for view in VIEWS:
        out[view] = {}
        for prov in ("all",) + PROVIDERS:
            cells = by(per, view, None if prov == "all" else prov)
            d = {"n_cells": len(cells), "fn": {}, "fp": {}, "ref_by_bucket": {}, "pred_by_bucket": {}}
            for c in cells.values():
                c = c["c"]
                for kind in ("fn", "fp"):
                    for cat, bs in c[kind].items():
                        for b, n in bs.items():
                            d[kind].setdefault(cat, {}).setdefault(b, 0)
                            d[kind][cat][b] += n
                for kind in ("ref_by_bucket", "pred_by_bucket"):
                    for b, (n, m) in c[kind].items():
                        cur = d[kind].setdefault(b, [0, 0])
                        cur[0] += n
                        cur[1] += m
            out[view][prov] = d
    return out


def auc(score, good):
    score, good = np.asarray(score, float), np.asarray(good, bool)
    ok = np.isfinite(score)
    score, good = score[ok], good[ok]
    pos, neg = score[good], score[~good]
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


DESC = ("delay_s", "speed_edit", "speed_pre", "dist_at_start", "monotone_toward", "net_approach_m", "approach_rate_mps", "frac_ball_obs")


def summarize_d(per):
    out = {}
    for label, views in (("all_views", VIEWS), ("v2_only", ("feed_like_v2",)), ("clean_only", ("clean",))):
        out[label] = {}
        for prov in ("all",) + PROVIDERS:
            rows = [r for k, c in per.items() if k.split("__")[2] in views and (prov == "all" or k.split("__")[0] == prov)
                    for r in c["d"]["edits"]]
            if not rows:
                continue
            fl = np.array([r["frac_live"] for r in rows])
            good = fl >= 0.5
            n = np.array([r["n"] for r in rows])
            d = {"n_edits": len(rows), "frames_reopened": int(n.sum()), "frac_good_edits": float(good.mean()),
                 "frame_weighted_frac_live": float((fl * np.array([r["graded"] for r in rows])).sum() / sum(r["graded"] for r in rows)),
                 "auc_good_vs_descriptor": {}, "quantiles_good": {}, "quantiles_bad": {}}
            for f in DESC:
                v = np.array([r[f] for r in rows], float)
                d["auc_good_vs_descriptor"][f] = auc(v, good)
                d["quantiles_good"][f] = qstats(v[good])
                d["quantiles_bad"][f] = qstats(v[~good])
            out[label][prov] = d
    return out


def summarize_e(per):
    out = {"max_abs_pD_diff_pooled": {}, "mean_abs_pD_diff_pooled": {}}
    for view in VIEWS:
        for pair in ("ce-cal", "ce-pred", "cal-pred"):
            vals = [c["e"][pair] for k, c in by(per, view).items()]
            out["max_abs_pD_diff_pooled"].setdefault(view, {})[pair] = max(v["max"] for v in vals)
            out["mean_abs_pD_diff_pooled"].setdefault(view, {})[pair] = float(np.mean([v["mean"] for v in vals]))
    es = [c["e"] for c in per.values()]
    out["f062_decode_of_pooled_predecessor_equals_decoded__predecessor_D_mask_all_cells"] = all(e["f062_decode_matches_base_mask"] for e in es)
    out["agreement_of_mask_with_pD_gt_0.5"] = {"min": min(e["mask_vs_pD_gt_0.5_agree"] for e in es), "mean": float(np.mean([e["mask_vs_pD_gt_0.5_agree"] for e in es]))}
    out["best_single_threshold_agreement"] = {"min": min(e["best_threshold_agree"] for e in es), "mean": float(np.mean([e["best_threshold_agree"] for e in es])),
                                              "thresholds_range": [min(e["best_threshold"] for e in es), max(e["best_threshold"] for e in es)]}
    out["pD_mean_in_mask_vs_out"] = [float(np.mean([e["pD_mean_in_mask"] for e in es])), float(np.mean([e["pD_mean_out_mask"] for e in es]))]
    out["runs_mask_vs_thresholded_0.5"] = [int(sum(e["n_runs_mask"] for e in es)), int(sum(e["n_runs_thresh0.5"] for e in es))]
    out["provenance"] = ("decoded__predecessor == feed_benchmark_audit_v1.decode(pooled predecessor P, model_period, fps) with CONFIG f062 = "
                         "feed_spell_push.configurations()[62]: family calibration dt=4, ht=1, ha=3, dead=3. feed_spell_push.decode applies a "
                         "dead-logit temperature (logit(P_D)/4) and H/A odds temperature 1, renormalises, then decoder_search.decode(family matrix) "
                         "runs a 3-state Viterbi on log-probabilities with switch costs [[0,3,3],[3,0,3],[3,3,0]]*(fps/10) per block of model_period.")
    return out


# ------------------------------------------------------------------ (d extra) oracle gate on delay edits, (f) v2 vs clean
def oracle_gated_snap(cell):
    per = _PER[cell.key]["d"]["edits"]
    dead = cell.base_dead.copy()
    for r in per:
        if r["frac_live"] >= 0.5:
            dead[r["start"]:r["start"] + r["n"]] = False
    inside, outside = BE.ball_state(cell.ball, cell.pitch, 0.0)
    return BE.edit_mask(dead, inside, outside, cell.model_period, cell.fps, "advance")


_PER = {}


def decomposition():
    cm = H.cell_map()

    def make(mask_from, ha_from):
        def fn(cell):
            if cell.view == "clean":
                return cell.snap_dead
            twin = cm[f"{cell.provider}__{cell.match}__clean__0"]
            src = {"self": cell, "clean": twin}
            d = src[mask_from].snap_dead if mask_from != "base_clean" else twin.base_dead
            prob_cell = src[ha_from]
            mask = np.where(d, "D", "H")
            return H.fp.decode_candidate(prob_cell.ha, mask, cell.model_period, cell.fps)
        return fn

    out = {}
    for name, (mf, hf) in {"D_v2__HA_v2 (v2 snap)": ("self", "self"), "D_clean__HA_v2": ("clean", "self"),
                           "D_v2__HA_clean": ("self", "clean"), "D_clean__HA_clean": ("clean", "clean")}.items():
        res = H.evaluate(make(mf, hf))
        out[name] = {v: res["aggregate"][v]["spell_f1"] for v in ("feed_like_v1", "feed_like_v2")}
        out[name]["providers_v2"] = res["aggregate"]["feed_like_v2"]["spell_f1/providers"]
        out[name]["providers_v1"] = res["aggregate"]["feed_like_v1"]["spell_f1/providers"]
    clean = H.evaluate(lambda c: c.snap_dead)["aggregate"]["clean"]["spell_f1"]
    return clean, out


def attribution(clean, d, view):
    key = lambda n: d[n][view]
    V, X1, X2 = key("D_v2__HA_v2 (v2 snap)"), key("D_clean__HA_v2"), key("D_v2__HA_clean")
    C = clean
    return {"clean": C, view: V, "drop": C - V, "D_clean_HA_v2": X1, "D_v2_HA_clean": X2,
            "D_effect_holding_HA_v2": X1 - V, "HA_effect_given_clean_D": C - X1,
            "D_effect_holding_HA_clean": C - X2, "HA_effect_given_v2_D": X2 - V,
            "shapley_D": ((X1 - V) + (C - X2)) / 2, "shapley_HA": ((C - X1) + (X2 - V)) / 2}


def decomposition_by_provider(clean_prov, d, view):
    out = {}
    for p in PROVIDERS:
        V, X1, X2 = (d[n]["providers_" + ("v2" if view == "feed_like_v2" else "v1")][p] for n in ("D_v2__HA_v2 (v2 snap)", "D_clean__HA_v2", "D_v2__HA_clean"))
        C = clean_prov[p]
        out[p] = {"clean": C, "view": V, "drop": C - V, "shapley_D": ((X1 - V) + (C - X2)) / 2, "shapley_HA": ((C - X1) + (X2 - V)) / 2}
    return out


def write_md(res):
    L = []
    P = lambda *a: L.append(" ".join(str(x) for x in a))
    f4 = lambda x: "-" if x is None else f"{x:.4f}"
    P("# v2_target_push_v1 phase-1 diagnostics (development cohort, vs snap_m0.0)\n")
    P("Spell F1 aggregation: realizations within match, matches within provider, providers equally. All labels used only for diagnostics.\n")
    c = res["control"]
    P("## Control replay\n")
    P("| | clean | v1 | v2 |\n|---|---:|---:|---:|")
    P("| frontier | " + " | ".join(f"{res['control']['frontier'][v]['spell_f1']:.6f}" for v in VIEWS) + " |")
    P("| snap_m0.0 | " + " | ".join(f"{c['snap_m0.0'][v]['spell_f1']:.6f}" for v in VIEWS) + " |")
    P("\n## a. Oracle ladder (spell F1; one thing replaced by truth on the snap_m0.0 prediction)\n")
    P("| rung | clean | v1 | v2 | v2 Met | v2 SC | v2 Spo | v2 short recall | v2 pred spells |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k, v in res["a_oracle_ladder"].items():
        pv = v["feed_like_v2"]["providers"]
        P(f"| {k} | " + " | ".join(f4(v[w]["spell_f1"]) for w in VIEWS) + f" | {pv['Metrica']:.4f} | {pv['SkillCorner']:.4f} | {pv['Sportec']:.4f} | {v['feed_like_v2']['short']:.3f} | {v['feed_like_v2']['pred_spells']:.0f} |")
    P("\nRung definitions: ii/iii/viii move each matched predicted D run's start / end / both to the true edge of the true D run it overlaps most (first/last predicted run touching that true run); iv removes predicted D runs overlapping no true D run; v adds true D runs overlapping no predicted D run; vi uses the true H/A label wherever the prediction is live and truth is H/A (D mask unchanged); vii_a gives each predicted live span one team (the true majority team inside it: no internal switches); vii_b uses true team boundaries inside predicted live spans (truth-D frames take the neighbouring true team).\n")
    P("## b. D-run timing (pred - true, seconds; matched = greedy one-to-one IoU >= 0.2)\n")
    b = res["b_run_timing_and_false_missed"]
    q = lambda x: "-" if not x else f"{x['median']:+.2f} [{x['p25']:+.2f},{x['p75']:+.2f}] p10/90 {x['p10']:+.1f}/{x['p90']:+.1f} (n={x['n']})"
    for m in ("base", "snap"):
        P(f"### {m} mask\n\n| view | provider | start err median [IQR] | end err median [IQR] |\n|---|---|---|---|")
        for v in ("clean", "feed_like_v2"):
            for pr in ("all",) + PROVIDERS:
                x = b[m][v][pr]
                P(f"| {v} | {pr} | {q(x['start_err_s'])} | {q(x['end_err_s'])} |")
        P("")
        P("False D runs (pred run overlapping no true D run) / missed D runs (true run overlapping no pred run): count (total seconds), summed over cells of the view.\n")
        P("| view | provider | cells | false <1s | 1-3s | 3-10s | >10s | missed <1s | 1-3s | 3-10s | >10s |\n|---|---|---:|---|---|---|---|---|---|---|---|")
        for v in ("clean", "feed_like_v2"):
            for pr in ("all",) + PROVIDERS:
                x = b[m][v][pr]
                cells = [f"{x[k][n]['count']} ({x[k][n]['total_s']:.0f})" for k in ("false_runs", "missed_runs") for n in ("<1s", "1-3s", "3-10s", ">10s")]
                P(f"| {v} | {pr} | {x['n_cells']} | " + " | ".join(cells) + " |")
        P("")
    P("## c. Unmatched (IoU 0.5) spell causes, snap_m0.0, counts by duration bucket [<1s, 1-3s, 3-10s, >10s]\n")
    c = res["c_spell_error_categories"]
    for v in ("clean", "feed_like_v2"):
        for pr in ("all",) + PROVIDERS:
            x = c[v][pr]
            P(f"### {v} / {pr} ({x['n_cells']} cells)\n")
            P("Reference spells [n, matched]: " + ", ".join(f"{k} {x['ref_by_bucket'].get(k)}" for k in ("<1s", "1-3s", "3-10s", ">10s")) + "  ")
            P("Predicted spells [n, matched]: " + ", ".join(f"{k} {x['pred_by_bucket'].get(k)}" for k in ("<1s", "1-3s", "3-10s", ">10s")) + "\n")
            for kind, title in (("fn", "False-negative reference spells"), ("fp", "False-positive predicted spells")):
                P(f"{title}\n\n| cause | <1s | 1-3s | 3-10s | >10s | total |\n|---|---:|---:|---:|---:|---:|")
                for cat, bs in sorted(x[kind].items(), key=lambda kv: -sum(kv[1].values())):
                    row = [bs.get(k, 0) for k in ("<1s", "1-3s", "3-10s", ">10s")]
                    P(f"| {cat} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
                P("")
    P("## d. Delay-edit descriptors (label-free) vs fraction of re-opened frames truly live\n")
    P("An edit = one predicted D run whose start the delay rule moved (frames [orig start, first outside observation)). good = >=50% of its graded re-opened frames are truly H/A. AUC = P(descriptor higher for a good edit than a bad one); below 0.5 means lower values are good. Pooled counts multiply the same events across clean/v1/v2 realizations.\n")
    d = res["d_delay_edit_descriptors"]
    for lab in ("all_views", "v2_only"):
        P(f"### {lab}\n\n| provider | edits | frames re-opened | good edits | frame-weighted live | " + " | ".join(f"AUC {k}" for k in DESC) + " |\n|---|---:|---:|---:|---:|" + "---:|" * len(DESC))
        for pr, x in d[lab].items():
            P(f"| {pr} | {x['n_edits']} | {x['frames_reopened']} | {x['frac_good_edits']:.3f} | {x['frame_weighted_frac_live']:.3f} | " + " | ".join(f4(x['auc_good_vs_descriptor'][k]) for k in DESC) + " |")
        P("")
    P("Quantiles (all_views, p10/p25/median/p75/p90) of descriptors for good vs bad edits:\n")
    for pr in ("SkillCorner", "Sportec", "all"):
        x = d["all_views"][pr]
        P(f"### {pr}\n\n| descriptor | good (n={int(x['frac_good_edits']*x['n_edits'])}) | bad |\n|---|---|---|")
        for k in DESC:
            fq = lambda y: "-" if not y else "/".join(f"{y[t]:.2f}" for t in ("p10", "p25", "median", "p75", "p90"))
            P(f"| {k} | {fq(x['quantiles_good'][k])} | {fq(x['quantiles_bad'][k])} |")
        P("")
    og = res["d_oracle_gate_on_delay_edits_spell"]
    P("Oracle gate (apply delay only to edits with >=50% truly-live re-opened frames, then advance; an upper bound, uses labels): spell F1 clean/v1/v2 = " + " / ".join(f"{og[v]['spell_f1']:.6f}" for v in VIEWS) + f" (snap_m0.0: " + " / ".join(f"{c0:.6f}" for c0 in [res['control']['snap_m0.0'][v]['spell_f1'] for v in VIEWS]) + "); v2 providers " + str({k: round(x, 4) for k, x in og['feed_like_v2']['providers'].items()}) + "\n")
    P("## e. P(D) across families and the origin of decoded__predecessor\n")
    e = res["e_pD_family_and_mask_provenance"]
    P("Max abs diff of pooled P(D) (prob[:,2]) per view: " + json.dumps(e["max_abs_pD_diff_pooled"]) + "  ")
    P("Mean abs diff: " + json.dumps({k: {a: round(b, 5) for a, b in v.items()} for k, v in e["mean_abs_pD_diff_pooled"].items()}) + "\n")
    for k in ("f062_decode_of_pooled_predecessor_equals_decoded__predecessor_D_mask_all_cells", "agreement_of_mask_with_pD_gt_0.5", "best_single_threshold_agreement", "pD_mean_in_mask_vs_out", "runs_mask_vs_thresholded_0.5"):
        P(f"- {k}: {json.dumps(e[k])}")
    P("\n" + e["provenance"] + "\n")
    P("## f. clean -> v2 (and v1) drop: D mask vs H/A\n")
    f = res["f_clean_vs_v2_decomposition"]
    P("| combination (spell F1) | v1 | v2 |\n|---|---:|---:|")
    for k, v in f["scores"].items():
        P(f"| {k} | {v['feed_like_v1']:.4f} | {v['feed_like_v2']:.4f} |")
    for vw in ("v2", "v1"):
        a = f[vw]
        P(f"\n{vw}: clean {a['clean']:.4f} -> {a[[k for k in a if k.startswith('feed')][0]]:.4f}, drop {a['drop']:.4f}. D-mask effect {a['D_effect_holding_HA_v2']:.4f} (holding v-view H/A) or {a['D_effect_holding_HA_clean']:.4f} (holding clean H/A); H/A effect {a['HA_effect_given_clean_D']:.4f} (given clean D) or {a['HA_effect_given_v2_D']:.4f} (given v D). Shapley: D {a['shapley_D']:.4f}, H/A {a['shapley_HA']:.4f}.")
    P("\nv2 by provider (Shapley): " + json.dumps({k: {a: round(b, 4) for a, b in v.items()} for k, v in f["v2_by_provider"].items()}) + "\n")
    P("## g. Extra: short (<3 s) unmatched predicted spells sitting on truly-dead frames\n")
    for v, x in res["g_short_true_dead_fp_islands"].items():
        for pr, y in x.items():
            P(f"- {v}/{pr}: n={y['n']}, flanked by predicted D on both sides {y['frac_flanked_by_pred_D_both_sides']:.3f}, on at least one side {y['frac_flanked_any_side']:.3f}, mean pooled P(D) in span {y['mean_pD_pred_in_span']:.3f}, ball observed {y['frac_ball_observed']:.2f}, median duration {y['duration_s']['median']:.2f}s")
    (OUT / "diagnostics.md").write_text("\n".join(L) + "\n")


def main():
    t0 = time.time()
    H.control_replay()
    res = {"control": {"snap_m0.0": agg_rows(H.evaluate(lambda c: c.snap_dead)),
                       "frontier": {v: {"spell_f1": H._agg(H._control_rows()["frontier"])[v]["spell_f1"]} for v in VIEWS}}}
    print("ladder", flush=True)
    res["a_oracle_ladder"] = ladder()
    print("per-cell", time.time() - t0, flush=True)
    per = percall()
    _PER.update(per)
    res["b_run_timing_and_false_missed"] = summarize_b(per)
    res["c_spell_error_categories"] = summarize_c(per)
    res["d_delay_edit_descriptors"] = summarize_d(per)
    gate = H.evaluate(oracle_gated_snap)
    res["d_oracle_gate_on_delay_edits_spell"] = agg_rows(gate)
    res["e_pD_family_and_mask_provenance"] = summarize_e(per)
    res["g_short_true_dead_fp_islands"] = summarize_g(per)
    clean, dec = decomposition()
    clean_prov = H.evaluate(lambda c: c.snap_dead)["aggregate"]["clean"]["spell_f1/providers"]
    res["f_clean_vs_v2_decomposition"] = {"scores": dec, "v2": attribution(clean, dec, "feed_like_v2"), "v1": attribution(clean, dec, "feed_like_v1"),
                                          "v2_by_provider": decomposition_by_provider(clean_prov, dec, "feed_like_v2")}
    res["elapsed_s"] = time.time() - t0
    (OUT / "diagnostics.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    write_md(json.loads((OUT / "diagnostics.json").read_text()))
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
