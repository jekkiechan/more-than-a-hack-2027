"""Family D round 2 (`D_round2`): v1 rows in training + shortcut ablations of the LOPO LightGBM P(D) refiner."""
from __future__ import annotations

import datetime
import json
import sys

import numpy as np

from experiments import v2_target_push_harness as H
from experiments import v2_target_push_refiner as R1
from experiments import v2_target_push_candidates as K
from experiments import sc2020_realfeed_score as S

OUT = H.OUT
REF2 = OUT / "refinedD2"
FEAT2 = OUT / "featD2"
MODELS = OUT / "models_D_round2"
PROVIDERS = R1.PROVIDERS
FEATURE_SETS = ("full", "nomiss", "nospeed", "nomiss_nospeed", "noball")
ARMS = {"R2_full_w1.0": ("full", 1.0), "R2_full_w0.5": ("full", 0.5), "R2_full_w0.25": ("full", 0.25),
        "R2_nomiss_w0.5": ("nomiss", 0.5), "R2_nospeed_w0.5": ("nospeed", 0.5),
        "R2_nomiss_nospeed_w0.5": ("nomiss_nospeed", 0.5), "R2_noball_w0.5": ("noball", 0.5)}
KEEP = ("R2_full_w0.5", "R2_nomiss_w0.5")
MISS = {"ball_observed", "ball_since_obs_s", "ball_until_obs_s", "ball_beyond_next_obs"}


def base_names():
    c = [c for c in H.load_cells() if c.provider == "SkillCorner"][0]
    return R1.features(c)[0]


def _ffill(vals, valid, blocks, const=0.0):
    out = np.full(len(vals), const, dtype=np.float64)
    for a, b in blocks:
        v, ok = vals[a:b], valid[a:b]
        if not ok.any():
            continue
        idx = np.arange(b - a)
        last = np.maximum.accumulate(np.where(ok, idx, -1))
        first = np.flatnonzero(ok)[0]
        last = np.where(last < 0, first, last)
        out[a:b] = v[last]
    return out


def ff_features(cell, speed):
    blocks = H.fp.balanced.runtime.frozen.supervised.blocks(cell.model_period)
    L, W = float(cell.pitch[0]), float(cell.pitch[1])
    x, y = np.nan_to_num(cell.ball[:, 0]), np.nan_to_num(cell.ball[:, 1])
    beyond = np.maximum.reduce([-x, x - L, -y, y - W])
    return np.column_stack((_ffill(speed, np.isfinite(speed), blocks), _ffill(beyond, cell.ball_obs, blocks))).astype(np.float32)


def feature_matrix(cell, names, fs, X1, ff):
    if fs == "full":
        cols = names
    elif fs == "nospeed":
        cols = [n for n in names if n != "ball_speed"]
    elif fs == "noball":
        cols = [n for n in names if not n.startswith("ball_")]
    else:
        cols = [n for n in names if not n.startswith("ball_")]
    M = X1[:, [names.index(n) for n in cols]]
    out_names = list(cols)
    if fs in ("nomiss", "nomiss_nospeed"):
        keep_player = M
        extra = ff[:, 1:2] if fs == "nomiss_nospeed" else ff
        M = np.column_stack((keep_player, extra))
        out_names += (["ball_beyond_last_ff"] if fs == "nomiss_nospeed" else ["ball_speed_ff", "ball_beyond_last_ff"])
    return out_names, M


def main():
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score
    declaration = {
        "family": "D_round2", "declared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "control": "snap_m0.0",
        "control_spell": {v: H._agg(H._control_rows()["snap_m0.0"])[v]["spell_f1"] for v in S.VIEW_NAMES},
        "training_rows": f"LOPO; clean + feed_like_v1 (2) + feed_like_v2 (4) cells of the two training providers, every {R1.STRIDE}th frame, seed 2026",
        "lightgbm_params": R1.PARAMS,
        "feature_sets": {"full": "round-1 features", "nomiss": "full minus ball_observed, since/until observed, beyond_next; ball speed and beyond-last forward-filled within block (back-filled at block start, 0 if never observed)",
                         "nospeed": "full minus ball_speed", "nomiss_nospeed": "nomiss minus ball speed (ball only via ffilled signed distance beyond boundary)",
                         "noball": "no ball_* features"},
        "arms": {k: {"featureset": fs, "w_refined": w, "pipeline": "f062 decode of blended P(D) -> snap_m0.0 -> frontier H/A decode"} for k, (fs, w) in ARMS.items()},
        "predictions_kept_for": "all eligible arms plus R2_full_w0.5 and R2_nomiss_w0.5"}
    (OUT / "declaration_D_round2.json").write_text(json.dumps(declaration, indent=1) + "\n")
    cells = H.load_cells()
    names = base_names()
    FEAT2.mkdir(parents=True, exist_ok=True)
    X1 = {c.key: np.load(R1.FEAT / f"{c.key}.npy") for c in cells}
    FF = {}
    for c in cells:
        FF[c.key] = ff_features(c, X1[c.key][:, names.index("ball_speed")].astype(np.float64))
    folds, imps = {}, {}
    for fs in FEATURE_SETS:
        (REF2 / fs).mkdir(parents=True, exist_ok=True)
        (MODELS / fs).mkdir(parents=True, exist_ok=True)
        M, nm = {}, None
        for c in cells:
            nm, M[c.key] = feature_matrix(c, names, fs, X1[c.key], FF[c.key])
        folds[fs], imps[fs] = {}, {}
        for prov in PROVIDERS:
            train = [c for c in cells if c.provider != prov]
            xs, ys = [], []
            for c in train:
                idx = np.arange(0, c.n, R1.STRIDE)
                idx = idx[c.labels[idx] != "X"]
                xs.append(M[c.key][idx])
                ys.append((c.labels[idx] == "D").astype(int))
            xt, yt = np.concatenate(xs), np.concatenate(ys)
            model = lgb.LGBMClassifier(**R1.PARAMS).fit(xt, yt)
            model.booster_.save_model(str(MODELS / fs / f"{prov}.txt"))
            gain = dict(zip(nm, model.booster_.feature_importance("gain")))
            tot = sum(gain.values())
            imps[fs][prov] = {k: v / tot for k, v in sorted(gain.items(), key=lambda kv: -kv[1])[:12]}
            acc = {}
            for c in [c for c in cells if c.provider == prov]:
                p = model.predict_proba(M[c.key])[:, 1]
                np.save(REF2 / fs / f"{c.key}.npy", p.astype(np.float32))
                g = c.labels != "X"
                a = acc.setdefault(c.view, ([], [], []))
                a[0].append((c.labels[g] == "D").astype(int)); a[1].append(p[g]); a[2].append(c.pD["predecessor"][g])
            folds[fs][prov] = {"n_train_rows": int(len(yt)), "heldout": {}}
            for v, (y, pr, pp) in acc.items():
                y, pr, pp = np.concatenate(y), np.concatenate(pr), np.concatenate(pp)
                e = 1e-6
                folds[fs][prov]["heldout"][v] = {"auc_refined": float(roc_auc_score(y, pr)), "auc_predecessor": float(roc_auc_score(y, pp)),
                                                 "logloss_refined": float(log_loss(y, np.clip(pr, e, 1 - e))), "logloss_predecessor": float(log_loss(y, np.clip(pp, e, 1 - e)))}
            print(fs, prov, flush=True)
    recs = []
    for name, (fs, w) in ARMS.items():
        rec = K.run_arm(name, {"r2": (fs, w)}, "D_round2", keep=name in KEEP)
        rec["featureset"], rec["w"] = fs, w
        recs.append(rec)
        print(f"{name:26s} " + " ".join(f"{rec['spell'][v]:.4f}" for v in S.VIEW_NAMES) + f" dv2 {rec['delta_spell_vs_snap']['feed_like_v2']:+.4f} fail={rec['failed_retention_checks']}", flush=True)
    elig = [r for r in recs if r["retained"]]
    best = max(elig, key=lambda r: r["spell"]["feed_like_v2"]) if elig else min(recs, key=lambda r: (len(r["failed_retention_checks"]), -r["spell"]["feed_like_v2"]))
    ctrl = H._control_rows()["snap_m0.0"]
    cells_v2 = {k: best["cell_spell"][k] - ctrl[k]["spell_f1"] for k in best["cell_spell"] if "feed_like_v2" in k}
    detail = {"arm": best["arm"], "selected_because": "best eligible by v2" if elig else "fewest failed guards then v2",
              "v2_cell_spell_deltas": cells_v2, "cells_improved": sum(d > 1e-12 for d in cells_v2.values()),
              "cells_declined": sum(d < -1e-12 for d in cells_v2.values()), "importances_top12_per_fold": imps[best["featureset"]]}
    for r in recs:
        if not r["retained"] and r["arm"] not in KEEP:
            pass
    json.dump({"declaration": "declaration_D_round2.json", "arms": recs, "eligible": [r["arm"] for r in elig], "best_arm_detail": detail,
               "fold_heldout_auc_logloss": folds, "importances_top12_all_featuresets": imps},
              open(OUT / "results_D_round2.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
