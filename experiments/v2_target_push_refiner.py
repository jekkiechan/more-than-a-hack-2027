"""Family D: provider-held-out LightGBM refiner of P(D) for `v2_target_push_v1`.

Leave-one-PROVIDER-out over the development matches. Training rows: clean + feed_like_v2 cells of the two
training providers, every 5th frame (labels D vs H/A; X dropped). Features are label-free, provider-agnostic,
contain no absolute time and no provider id. Predictions are written for all 35 cells from the fold model of
the cell's own provider (never trained on that provider).
"""
from __future__ import annotations

import datetime
import json
import multiprocessing as mp

import numpy as np

from experiments import v2_target_push_harness as H

OUT = H.OUT
FEAT = OUT / "featD"
REFINED = OUT / "refinedD"
PROVIDERS = ("Metrica", "SkillCorner", "Sportec")
SHIFTS_S = (-4.0, -2.0, -1.0, -0.4, 0.4, 1.0, 2.0, 4.0)
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, random_state=2026, n_jobs=4, verbose=-1)
STRIDE = 5


def _shift(x, k):
    idx = np.clip(np.arange(len(x)) + k, 0, len(x) - 1)
    return x[idx]


def _roll(x, w):
    w = max(1, int(round(w)))
    c = np.concatenate(([0.0], np.cumsum(x)))
    lo = np.clip(np.arange(len(x)) - w // 2, 0, len(x))
    hi = np.clip(np.arange(len(x)) + w // 2 + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def _since_until(obs, fps, cap=10.0):
    n = len(obs)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(obs, idx, -10**9))
    nxt = np.minimum.accumulate(np.where(obs, idx, 10**9)[::-1])[::-1]
    since = np.minimum((idx - last) / fps, cap)
    until = np.minimum((nxt - idx) / fps, cap)
    return since, until, np.clip(last, 0, n - 1), np.clip(nxt, 0, n - 1)


def features(cell):
    fps = cell.fps
    cols = {}
    for fam in ("predecessor", "calibrated"):
        p = cell.pD[fam].astype(np.float64)
        cols[f"{fam}_pD"] = p
        for s in SHIFTS_S:
            cols[f"{fam}_pD_shift{s:+.1f}s"] = _shift(p, int(round(s * fps)))
        cols[f"{fam}_pD_mean1s"] = _roll(p, fps)
        cols[f"{fam}_pD_mean3s"] = _roll(p, 3 * fps)
    ball, obs = cell.ball, cell.ball_obs
    cols["ball_observed"] = obs.astype(float)
    since, until, last, nxt = _since_until(obs, fps)
    cols["ball_since_obs_s"], cols["ball_until_obs_s"] = since, until
    L, W = float(cell.pitch[0]), float(cell.pitch[1])
    x, y = np.nan_to_num(ball[:, 0]), np.nan_to_num(ball[:, 1])
    beyond = np.maximum.reduce([-x, x - L, -y, y - W])
    beyond = np.where(obs.any(), beyond, np.nan)
    cols["ball_beyond_last_obs"] = np.where(obs.any(), beyond[last], np.nan)
    cols["ball_beyond_next_obs"] = np.where(obs.any(), beyond[nxt], np.nan)
    speed = np.full(cell.n, np.nan)
    both = obs[1:] & obs[:-1]
    speed[1:][both] = np.linalg.norm(ball[1:][both] - ball[:-1][both], axis=1) * fps
    cols["ball_speed"] = speed
    players = cell.players().astype(np.float64)
    pobs = np.isfinite(players).all(-1)
    cols["n_players_obs"] = pobs.sum(1).astype(float)
    lag = max(1, int(round(0.4 * fps)))
    disp = np.full(pobs.shape, np.nan)
    disp[lag:] = np.linalg.norm(players[lag:] - players[:-lag], axis=2) / (lag / fps)
    srt = np.sort(disp, axis=1)  # NaN last
    k = np.isfinite(disp).sum(1)
    with np.errstate(all="ignore"):
        mean = np.where(k > 0, np.nansum(disp, axis=1) / np.maximum(k, 1), np.nan)
        p90 = np.where(k > 0, srt[np.arange(cell.n), np.clip(np.floor(0.9 * (k - 1)).astype(int), 0, disp.shape[1] - 1)], np.nan)
        pxy = np.where(pobs[..., None], players, np.nan)
        spread = np.sqrt(np.nanvar(pxy[..., 0], axis=1) + np.nanvar(pxy[..., 1], axis=1))
    cols["player_speed_mean"], cols["player_speed_p90"], cols["player_spread"] = mean, p90, spread
    d2 = int(round(2 * fps))
    cols["player_speed_mean_d2s"] = mean - _shift(mean, -d2)
    cols["player_speed_p90_d2s"] = p90 - _shift(p90, -d2)
    names = list(cols)
    return names, np.column_stack([cols[n] for n in names]).astype(np.float32)


def _feat_job(i):
    cell = H._STATE["cells"][i]
    names, X = features(cell)
    np.save(FEAT / f"{cell.key}.npy", X)
    return names


def build_features():
    FEAT.mkdir(parents=True, exist_ok=True)
    H.load_cells()
    with mp.get_context("fork").Pool(3) as pool:
        return pool.map(_feat_job, range(len(H._STATE["cells"])), chunksize=1)[0]


def main():
    import lightgbm as lgb
    from sklearn.metrics import log_loss, roc_auc_score
    REFINED.mkdir(parents=True, exist_ok=True)
    (OUT / "declaration_D_model.json").write_text(json.dumps({
        "family": "D_model", "declared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "scheme": "leave-one-provider-out LightGBM binary classifier (D vs H/A; X dropped)", "params": PARAMS,
        "train_rows": f"clean + feed_like_v2 cells of the two training providers, every {STRIDE}th frame",
        "features": "see experiments/v2_target_push_refiner.py:features (pooled predecessor+calibrated P(D) at t, shifts +-0.4/1/2/4 s, "
                    "1 s/3 s rolling means; ball flags/ages/beyond-boundary/speed; player count/speed mean,p90,2s deltas/spread)",
        "forbidden": ["absolute time", "provider id", "labels at inference"]}, indent=1) + "\n")
    names = build_features()
    cells = H.load_cells()
    X = {c.key: np.load(FEAT / f"{c.key}.npy") for c in cells}
    report = {}
    for prov in PROVIDERS:
        train = [c for c in cells if c.provider != prov and c.view in ("clean", "feed_like_v2")]
        xs, ys = [], []
        for c in train:
            idx = np.arange(0, c.n, STRIDE)
            idx = idx[c.labels[idx] != "X"]
            xs.append(X[c.key][idx])
            ys.append((c.labels[idx] == "D").astype(int))
        xt, yt = np.concatenate(xs), np.concatenate(ys)
        model = lgb.LGBMClassifier(**PARAMS).fit(xt, yt)
        imp = dict(zip(names, model.booster_.feature_importance("gain")))
        tot = sum(imp.values())
        rep = {"n_train_rows": int(len(yt)), "train_D_frac": float(yt.mean()),
               "importance_gain_frac_top12": {k: v / tot for k, v in sorted(imp.items(), key=lambda kv: -kv[1])[:12]},
               "heldout": {}}
        ys_all, pr_all, pp_all = {}, {}, {}
        for c in [c for c in cells if c.provider == prov]:
            p = model.predict_proba(X[c.key])[:, 1]
            np.save(REFINED / f"{c.key}.npy", p.astype(np.float32))
            g = c.labels != "X"
            for v in (c.view,):
                ys_all.setdefault(v, []).append((c.labels[g] == "D").astype(int))
                pr_all.setdefault(v, []).append(p[g])
                pp_all.setdefault(v, []).append(c.pD["predecessor"][g])
        for v in ys_all:
            y, pr, pp = (np.concatenate(z[v]) for z in (ys_all, pr_all, pp_all))
            eps = 1e-6
            rep["heldout"][v] = {"n": int(len(y)), "auc_refined": float(roc_auc_score(y, pr)), "auc_predecessor": float(roc_auc_score(y, pp)),
                                 "logloss_refined": float(log_loss(y, np.clip(pr, eps, 1 - eps))),
                                 "logloss_predecessor": float(log_loss(y, np.clip(pp, eps, 1 - eps)))}
        report[prov] = rep
        print(prov, rep["heldout"], flush=True)
    (OUT / "refiner_fold_report.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
