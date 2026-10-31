"""Context-conditioned ball-gap stress view (evaluation only, no fitting of any model).

Hazard of a ball gap onset is fitted label-free on the competition feed tracking
(``data/challenge/inputs.npz``; the placeholder ``labels`` key is never read and
``evaluation/gt`` is never touched). Gaps are then simulated on external
provider-holdout matches and combined with the non-ball-gap feed_like components.

Pipeline (see CONTEXT_GAPS.md):  fit -> build views / realism -> freeze -> score.
  python -m experiments.context_gaps fit
  python -m experiments.context_gaps realism
  python -m experiments.context_gaps freeze
  python -m experiments.context_gaps score --match <stem>
  python -m experiments.context_gaps report
"""
from __future__ import annotations

import os
for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import supervised, tracking_corruption as tc  # noqa: E402

OUT = ROOT / "data/experiments/context_gaps_v1"
RESULT = ROOT / "experiments/results/context_gaps.json"
DECLARATION = OUT / "declaration.json"
COMPETITION_INPUTS = ROOT / "data/challenge/inputs.npz"
FEED_NO_BALL_GAPS = ("coordinates", "slot_gaps", "id_swaps")
FEED = ("ball_gaps",) + FEED_NO_BALL_GAPS
TARGET_MISSING = 0.40692572565263097      # realistic_corruption_development calibration
DURATION_CAP_S = 30.0                     # top gap_edge of the existing calibration
CAL_SEED = 20261029
FEATURES = ("ball_speed", "ball_acc", "dir_change", "nearest_dist", "n_within_3",
            "n_within_8", "ball_x", "ball_y", "d_touchline", "d_goalline", "nearest_speed")
DESCRIPTORS = ("mean_player_speed", "team_spread", "time_since_gap")
HAZARD_PARAMS = dict(max_depth=3, max_iter=100, learning_rate=0.05, min_samples_leaf=500,
                     l2_regularization=1.0, random_state=0)
LAG_S = 0.2


# ----------------------------------------------------------------------------- features

def _block_ids(data):
    return supervised.model_periods(data)


def context_features(data):
    """Return (X [n, len(FEATURES)] float with NaN where undefined, aux dict).

    Everything is computed from tracking only and in a centre-free convention:
    metres, origin at a pitch corner (both competition and provider inputs use it).
    Windows are in seconds so 10 Hz and 25 Hz inputs are comparable.
    """
    ball = np.asarray(data["ball"], float)
    players = np.asarray(data["players"], float)
    fps = float(data["fps"])
    pitch = np.asarray(data.get("pitch", [105., 68.]), float)
    n = len(ball)
    blk = _block_ids(data)
    lag = max(1, int(round(LAG_S * fps)))
    dt = lag / fps
    X = np.full((n, len(FEATURES)), np.nan)

    def shifted(a, k):
        out = np.full_like(a, np.nan)
        if k < len(a):
            out[k:] = a[:-k]
        return out

    b1 = shifted(ball, lag)
    b2 = shifted(ball, 2 * lag)
    same = np.zeros(n, bool)
    if 2 * lag < n:
        same[2 * lag:] = blk[2 * lag:] == blk[:-2 * lag]
    v1 = (ball - b1) / dt
    v0 = (b1 - b2) / dt
    speed = np.linalg.norm(v1, axis=1)
    acc = np.linalg.norm(v1 - v0, axis=1) / dt
    s0 = np.linalg.norm(v0, axis=1)
    cos = (v1 * v0).sum(1) / np.maximum(speed * s0, 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    ang = np.where((speed > 1.) & (s0 > 1.), ang, 0.)
    ok = same & np.isfinite(ball).all(1) & np.isfinite(b1).all(1) & np.isfinite(b2).all(1)
    pl = players
    diff = pl - ball[:, None, :]
    dist = np.linalg.norm(diff, axis=2)
    dist[~np.isfinite(dist)] = np.inf
    nearest = dist.argmin(1)
    nd = dist[np.arange(n), nearest]
    n3 = (dist < 3.).sum(1)
    n8 = (dist < 8.).sum(1)
    pl_prev = shifted(pl.reshape(n, -1), lag).reshape(pl.shape)
    pv = np.linalg.norm(pl - pl_prev, axis=2) / dt
    nsp = pv[np.arange(n), nearest]
    nsp[~np.isfinite(nd) | np.isinf(nd)] = np.nan
    nd = np.where(np.isinf(nd), np.nan, nd)
    X[:, 0] = speed
    X[:, 1] = acc
    X[:, 2] = ang
    X[:, 3] = nd
    X[:, 4] = n3
    X[:, 5] = n8
    X[:, 6] = ball[:, 0]
    X[:, 7] = ball[:, 1]
    X[:, 8] = np.minimum(ball[:, 1], pitch[1] - ball[:, 1])
    X[:, 9] = np.minimum(ball[:, 0], pitch[0] - ball[:, 0])
    X[:, 10] = nsp
    X[~ok] = np.nan
    # Descriptors NOT used by the hazard (for the realism discriminator).
    with np.errstate(all="ignore"):
        mps = np.nanmean(pv, axis=1)
        sp = np.sqrt(np.nanvar(pl[:, :, 0], axis=1) + np.nanvar(pl[:, :, 1], axis=1))
    aux = dict(mean_player_speed=mps, team_spread=sp, blocks=blk, ok=ok)
    return X, aux


def active_mask(data):
    return np.asarray(data["period"]) > 0


def _runs(mask):
    """Start (inclusive) and stop (exclusive) of True runs."""
    m = np.r_[False, mask, False]
    d = np.diff(m.astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def ball_gap_runs(data):
    """(start, stop) of missing-ball runs inside active frames, clipped at block edges."""
    ball_missing = ~np.isfinite(np.asarray(data["ball"], float)).all(1) & active_mask(data)
    blk = _block_ids(data)
    out = []
    for a, b in supervised.blocks(blk):
        if not active_mask(data)[a]:
            continue
        s, e = _runs(ball_missing[a:b])
        out.extend(zip(s + a, e + a))
    return out


def time_since_gap(data):
    """Seconds since the last missing-ball frame (capped 300 s), per frame."""
    miss = ~np.isfinite(np.asarray(data["ball"], float)).all(1)
    fps = float(data["fps"])
    n = len(miss)
    last = np.where(miss, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(last)
    return np.minimum((np.arange(n) - last) / fps, 300.)


# ----------------------------------------------------------------------------- hazard

def load_competition():
    """Competition tracking only; `labels` is never accessed."""
    z = np.load(COMPETITION_INPUTS, allow_pickle=False)
    keys = ("ball", "players", "teams", "period", "fps", "timestamps", "pitch")
    data = {k: np.asarray(z[k]) for k in keys}
    data["ball"] = np.asarray(data["ball"], float)
    data["players"] = np.asarray(data["players"], float)
    return data


def _onset_table(data):
    """Per-frame features, hazard target (ball missing at t+1) and eligibility."""
    X, aux = context_features(data)
    ball_ok = np.isfinite(np.asarray(data["ball"], float)).all(1)
    act = active_mask(data)
    blk = aux["blocks"]
    nxt_missing = np.zeros(len(ball_ok), bool)
    nxt_missing[:-1] = ~ball_ok[1:]
    same_next = np.zeros(len(ball_ok), bool)
    same_next[:-1] = blk[1:] == blk[:-1]
    y = (nxt_missing & same_next).astype(int)
    elig = act & ball_ok & same_next & aux["ok"]
    return X, y, elig, aux


def _impute_values(X, elig):
    return np.nanmedian(X[elig], axis=0)


def fit_hazard(data=None, verbose=True):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    data = data or load_competition()
    X, y, elig, aux = _onset_table(data)
    idx = np.flatnonzero(elig)
    med = _impute_values(X, elig)
    Xi = np.where(np.isfinite(X), X, med)
    half = idx[len(idx) // 2]
    first = idx[idx < half]
    second = idx[idx >= half]
    # purge a 10 s buffer around the split so lag windows cannot straddle halves
    second = second[second > half + int(10 * float(data["fps"]))]
    m1 = HistGradientBoostingClassifier(**HAZARD_PARAMS).fit(Xi[first], y[first])
    p2 = m1.predict_proba(Xi[second])[:, 1]
    auc = float(roc_auc_score(y[second], p2))
    # reverse split as a second check
    m1r = HistGradientBoostingClassifier(**HAZARD_PARAMS).fit(Xi[second], y[second])
    auc_rev = float(roc_auc_score(y[first], m1r.predict_proba(Xi[first])[:, 1]))
    # permutation importance on the second half (AUC drop, 3 repeats)
    rng = np.random.default_rng(0)
    imp = {}
    for j, name in enumerate(FEATURES):
        drops = []
        for _ in range(3):
            Xp = Xi[second].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            drops.append(auc - roc_auc_score(y[second], m1.predict_proba(Xp)[:, 1]))
        imp[name] = float(np.mean(drops))
    model = HistGradientBoostingClassifier(**HAZARD_PARAMS).fit(Xi[idx], y[idx])
    p_all = model.predict_proba(Xi[idx])[:, 1]
    # context effects: observed onset rate in feature quintiles (all eligible frames)
    effects = {}
    base = y[idx].mean()
    for j, name in enumerate(FEATURES):
        v = X[idx, j]
        v = np.where(np.isfinite(v), v, med[j])
        qs = np.unique(np.quantile(v, [0, .2, .4, .6, .8, 1.]))
        bins = np.clip(np.searchsorted(qs, v, side="right") - 1, 0, len(qs) - 2)
        rate = [float(y[idx][bins == b].mean() / base) for b in range(len(qs) - 1)]
        effects[name] = dict(edges=[float(q) for q in qs], rate_ratio_vs_mean=rate)
    # duration table conditioned on predicted hazard tercile at onset
    onset_idx = idx[y[idx] == 1]
    h_on = model.predict_proba(Xi[onset_idx])[:, 1]
    edges = np.quantile(h_on, [1 / 3, 2 / 3])
    fps = float(data["fps"])
    runs = ball_gap_runs(data)
    onset_to_dur = {}
    for s, e in runs:
        onset_to_dur[s - 1] = min((e - s) / fps, DURATION_CAP_S)
    durs = [[], [], []]
    for t, h in zip(onset_idx, h_on):
        d = onset_to_dur.get(t)
        if d is not None:
            durs[int(np.searchsorted(edges, h, side="right"))].append(d)
    active_min = act_minutes(data)
    info = dict(n_eligible=int(len(idx)), n_onsets=int(y[idx].sum()),
                onset_rate_per_eligible_frame=float(base),
                gaps_per_minute=float(len(runs) / active_min),
                auc_second_half=auc, auc_first_half_from_second=auc_rev,
                permutation_importance_auc_drop=imp, quintile_rate_ratio=effects,
                impute_median=dict(zip(FEATURES, map(float, med))),
                hazard_tercile_edges=[float(e) for e in edges],
                duration_tercile_sizes=[len(d) for d in durs],
                duration_tercile_median_s=[float(np.median(d)) if d else None for d in durs],
                train_hazard_mean=float(p_all.mean()))
    if verbose:
        print(json.dumps({k: v for k, v in info.items() if k != "quintile_rate_ratio"}, indent=1), flush=True)
    return dict(model=model, median=med, tercile_edges=edges, durations=[np.array(d) for d in durs],
                info=info, onset_hazard=h_on)


def act_minutes(data):
    return float(active_mask(data).sum() / float(data["fps"]) / 60.)


# ----------------------------------------------------------------------------- simulation

def hazard_per_frame(hz, data, X=None, aux=None):
    """Per-frame onset probability at the data's native rate for scalar=1 (25 Hz basis)."""
    if X is None:
        X, aux = context_features(data)
    ball_ok = np.isfinite(np.asarray(data["ball"], float)).all(1)
    act = active_mask(data)
    Xi = np.where(np.isfinite(X), X, hz["median"])
    h = hz["model"].predict_proba(Xi)[:, 1]
    h = np.where(ball_ok & act, h, 0.)
    return h * (25. / float(data["fps"]))


def simulate_context_gaps(data, hazard, hz, scalar, seed, hcache=None):
    """Return list of (start, stop) synthetic gaps and tercile index per gap."""
    rng = np.random.default_rng(seed)
    fps = float(data["fps"])
    blk = _block_ids(data)
    act = active_mask(data)
    p = np.clip(hazard * scalar, 0., 0.999999)
    cum_rate = -np.log1p(-p)
    gaps = []
    for a, b in supervised.blocks(blk):
        if not act[a]:
            continue
        H = np.cumsum(cum_rate[a:b])
        at = 0
        base = 0.
        while at < b - a - 1:
            target = base + rng.exponential(1.)
            if target > H[-1]:
                break
            t = int(np.searchsorted(H, target, side="left"))
            if t < at:
                t = at
            hv = hazard[a + t] * (fps / 25.)      # back to the 25 Hz basis used for terciles
            tercile = int(np.searchsorted(hz["tercile_edges"], hv, side="right"))
            pool = hz["durations"][tercile]
            d = float(rng.choice(pool))
            start = t + 1
            stop = min(b - a, start + max(1, int(round(d * fps))))
            if start >= b - a:
                break
            gaps.append((a + start, a + stop, tercile))
            at = stop
            base = H[min(stop, len(H)) - 1]
    return gaps


def calibrate_scalar(items, hz, target=TARGET_MISSING, seed=CAL_SEED):
    """items: list of (data, hazard). Bisection on log scalar so that the mean fraction of
    active frames covered by synthetic gaps equals `target` (mean across matches)."""
    def cover(s):
        fr = []
        for k, (data, h) in enumerate(items):
            gaps = simulate_context_gaps(data, h, hz, s, seed + k)
            m = gap_mask(len(h), gaps)
            fr.append(m.sum() / max(1, active_mask(data).sum()))
        return float(np.mean(fr))
    lo, hi = np.log(1e-2), np.log(1e3)
    for _ in range(18):
        mid = .5 * (lo + hi)
        if cover(np.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    s = float(np.exp(.5 * (lo + hi)))
    return s, cover(s)


def gap_mask(n, gaps):
    m = np.zeros(n, bool)
    for g in gaps:
        m[g[0]:g[1]] = True
    return m


def random_matched_gaps(data, gaps, seed):
    """Same per-block gap count and durations; onsets uniform at random (random composition)."""
    rng = np.random.default_rng(seed)
    blk = _block_ids(data)
    out = []
    for a, b in supervised.blocks(blk):
        mine = [g for g in gaps if a <= g[0] < b]
        if not mine:
            continue
        durs = [g[1] - g[0] for g in mine]
        rng.shuffle(durs)
        free = (b - a) - sum(durs)
        if free < 0:
            raise ValueError("gaps exceed block")
        cuts = np.sort(rng.integers(0, free + 1, size=len(durs)))
        # standard construction: gap i starts at a + cut_i + sum(previous durations)
        acc = 0
        for d, c in zip(durs, cuts):
            s = a + int(c) + acc
            out.append((s, s + d, -1))
            acc += d
    return out


def build_view(clean, calibration, feed_seed, gaps):
    """feed_like minus ball-gap component, then synthetic ball gaps on the observed ball."""
    from experiments import stress_components as sc
    view = sc.corrupt(clean, calibration, feed_seed, FEED_NO_BALL_GAPS)
    view["ball"] = np.array(view["ball"], copy=True)
    view["ball"][gap_mask(len(view["ball"]), gaps)] = np.nan
    view["ball_observed"] = np.isfinite(view["ball"]).all(1)
    return view


# ----------------------------------------------------------------------------- misc helpers

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hazard_hash(hz):
    from io import BytesIO
    import joblib
    buf = BytesIO()
    joblib.dump(dict(model=hz["model"], median=hz["median"], edges=hz["tercile_edges"],
                     durations=hz["durations"]), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def transition_points(labels, data):
    """Boundaries between adjacent graded frames (H/A/D) inside model-period blocks.
    Returns dict kind -> array of index i meaning boundary between frame i-1 and i."""
    lab = np.asarray(labels).astype(str)
    graded = np.isin(lab, ["H", "A", "D"])
    blk = _block_ids(data)
    i = np.arange(1, len(lab))
    valid = graded[1:] & graded[:-1] & (blk[1:] == blk[:-1]) & (lab[1:] != lab[:-1])
    a, b = lab[:-1][valid], lab[1:][valid]
    idx = i[valid]
    ha = np.isin(a, ["H", "A"]) & np.isin(b, ["H", "A"])
    ld = (np.isin(a, ["H", "A"]) & (b == "D")) | ((a == "D") & np.isin(b, ["H", "A"]))
    return dict(ha_change=idx[ha], live_dead=idx[ld])


def transition_coverage(trans, synth_mask, any_missing):
    out = {}
    for kind, idx in trans.items():
        n = len(idx)
        hs = int((synth_mask[idx] & synth_mask[idx - 1]).sum()) if n else 0
        ha = int((any_missing[idx] & any_missing[idx - 1]).sum()) if n else 0
        out[kind] = dict(n=int(n), hidden_synthetic=hs, hidden_any_missing=ha)
    return out


# ----------------------------------------------------------------------------- stages

def _cal():
    path = ROOT / "experiments/results/realistic_corruption_development.json"
    return json.loads(path.read_text())["spec"]["calibration"]


def _cells():
    from experiments import dead_transfer as dt
    for source, split in dt.SPLITS.items():
        for path in split["eval"]:
            yield source, path


def _clean(path):
    from experiments import multisource_eval as ms
    inputs, labels = ms.view(path)
    return tc.inputs_only(inputs), labels


def _seed(match, name):
    from experiments import corruption_experiment as ce
    return ce.seed(match, name)


def make_gaps(clean, hz, scalar, match, hazard=None):
    """The four synthetic gap sets for one match, deterministic by (match, replicate)."""
    hazard = hazard if hazard is not None else hazard_per_frame(hz, clean)
    out = {}
    for k in (1, 2):
        ctx = simulate_context_gaps(clean, hazard, hz, scalar, _seed(match, f"ctx_{k}"))
        out[f"ctx_{k}"] = ctx
        out[f"rand_{k}"] = random_matched_gaps(clean, ctx, _seed(match, f"rand_{k}"))
    return out


def _quant(x, q=(.1, .5, .9)):
    x = np.asarray(x, float)
    return [float(v) for v in np.quantile(x, q)] if len(x) else []


def _runs_from_mask(mask, data):
    blk = _block_ids(data)
    out = []
    for a, b in supervised.blocks(blk):
        if not active_mask(data)[a]:
            continue
        s, e = _runs(mask[a:b])
        out.extend(zip(s + a, e + a))
    return out


def _onset_rows(clean_X, aux, view_missing, runs, data):
    """Hazard features at onset (t = start-1) + non-hazard descriptors."""
    t = np.array([s - 1 for s, _ in runs if s > 0], int)
    tsg = time_since_gap(dict(ball=np.where(view_missing[:, None], np.nan, 0.), fps=data["fps"]))
    D = np.column_stack([aux["mean_player_speed"][t], aux["team_spread"][t], tsg[t]])
    return t, clean_X[t], D


def prepare():
    """Fit hazard, calibrate scalar, build gaps and run the label-free realism comparison."""
    import joblib
    from experiments import stress_components as sc
    OUT.mkdir(parents=True, exist_ok=True)
    comp = load_competition()
    hz = fit_hazard(comp, verbose=False)
    joblib.dump({k: v for k, v in hz.items() if k != "onset_hazard"}, OUT / "hazard.joblib")
    cal = _cal()
    matches = {}
    for source, path in _cells():
        clean, labels = _clean(path)
        X, aux = context_features(clean)
        matches[path.stem] = dict(clean=clean, X=X, aux=aux, hazard=hazard_per_frame(hz, clean, X, aux),
                                  source=source)
    # One scalar per match (4 calibration simulations each) so every match is stressed at the
    # feed's 0.407 coverage; a single global scalar left per-match coverage between 0.31 and 0.64.
    scalars, achieved_by = {}, {}
    for stem, m in matches.items():
        scalars[stem], achieved_by[stem] = calibrate_scalar([(m["clean"], m["hazard"])] * 4, hz)
        print("SCALAR", stem, scalars[stem], "achieved synthetic coverage", achieved_by[stem], flush=True)
    scalar, achieved = scalars, achieved_by
    # competition onsets
    Xc, yc, eligc, auxc = _onset_table(comp)
    comp_runs = ball_gap_runs(comp)
    comp_missing = ~np.isfinite(comp["ball"]).all(1)
    t_c, Xon_c, Don_c = _onset_rows(Xc, auxc, comp_missing, comp_runs, comp)
    active_min_c = act_minutes(comp)
    cdur = np.array([min((e - s) / float(comp["fps"]), DURATION_CAP_S) for s, e in comp_runs])
    real = dict(gaps_per_min=len(comp_runs) / active_min_c,
                missing_fraction=float(comp_missing[active_mask(comp)].mean()),
                duration_q10_50_90=_quant(cdur), duration_cap30_mean=float(cdur.mean()),
                n_gaps=len(comp_runs))
    fams = {"context_gaps": dict(minutes=0., miss=[], dur=[], n=0, cover=[], X=[], D=[]),
            "feed_like": dict(minutes=0., miss=[], dur=[], n=0, cover=[], X=[], D=[]),
            "random_gaps_matched": dict(minutes=0., miss=[], dur=[], n=0, cover=[], X=[], D=[])}
    per_match = {}
    for stem, m in matches.items():
        clean = m["clean"]
        fps = float(clean["fps"])
        act = active_mask(clean)
        native_missing = ~np.isfinite(clean["ball"]).all(1)
        gaps = make_gaps(clean, hz, scalars[stem], stem, m["hazard"])
        joblib.dump(gaps, OUT / f"{stem}_gaps.joblib")
        per_match[stem] = dict(native_missing_fraction=float(native_missing[act].mean()))
        for k in (1, 2):
            sets = {"context_gaps": gaps[f"ctx_{k}"], "random_gaps_matched": gaps[f"rand_{k}"]}
            # constant visible ball: NaNs in the output are exactly the renewal ball-gap intervals
            # (same RNG stream; ball values do not influence any draw)
            vis = dict(clean, ball=np.full_like(clean["ball"], 50.), ball_observed=np.ones(len(clean["ball"]), bool))
            fl = sc.corrupt(vis, cal, _seed(stem, f"corrupt_{k}"), ("ball_gaps",))
            fl_synth = np.isnan(fl["ball"]).any(1)
            sets["feed_like"] = [(s, e, -1) for s, e in _runs_from_mask(fl_synth, clean)]
            for name, g in sets.items():
                synth = gap_mask(len(native_missing), g)
                f = fams[name]
                f["minutes"] += act.sum() / fps / 60.
                f["n"] += len(g)
                f["cover"].append(float(synth[act].mean()))
                f["miss"].append(float((synth | native_missing)[act].mean()))
                f["dur"].extend([(e - s) / fps for s, e, *_ in g])
                runs = [(s, e) for s, e, *_ in g]
                t, Xo, Do = _onset_rows(m["X"], m["aux"], synth | native_missing, runs, clean)
                f["X"].append(Xo)
                f["D"].append(Do)
                per_match.setdefault(stem, {}).setdefault(name, []).append(
                    dict(replicate=k, n_gaps=len(g), coverage=float(synth[act].mean()),
                         union_missing=float((synth | native_missing)[act].mean())))
    realism = dict(real=real, scalar=scalar, scalar_calibration_coverage=achieved, per_match=per_match)
    onset_stats = {}
    key_idx = [FEATURES.index(n) for n in ("ball_speed", "ball_acc", "dir_change", "nearest_dist", "d_touchline", "d_goalline")]
    onset_stats["real_feed"] = {FEATURES[j]: _quant(Xon_c[:, j][np.isfinite(Xon_c[:, j])]) for j in key_idx}
    for name, f in fams.items():
        Xo = np.concatenate(f["X"])
        realism[name] = dict(gaps_per_min=f["n"] / f["minutes"], coverage_mean=float(np.mean(f["cover"])),
                             union_missing_mean=float(np.mean(f["miss"])),
                             duration_q10_50_90=_quant(f["dur"]), duration_cap30_mean=float(np.mean(np.minimum(f["dur"], DURATION_CAP_S))),
                             n_gaps=f["n"])
        onset_stats[name] = {FEATURES[j]: _quant(Xo[:, j][np.isfinite(Xo[:, j])]) for j in key_idx}
    realism["onset_feature_quantiles_q10_50_90"] = onset_stats
    # discriminator on non-hazard descriptors (+ all, for reference)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    def auc(real_rows, synth_rows):
        A = np.vstack([real_rows, synth_rows]).astype(float)
        yv = np.r_[np.ones(len(real_rows)), np.zeros(len(synth_rows))].astype(int)
        keep = np.isfinite(A).all(1)
        A, yv = A[keep], yv[keep]
        clf = HistGradientBoostingClassifier(max_depth=2, max_iter=60, learning_rate=.05, min_samples_leaf=20,
                                             class_weight="balanced", random_state=0)
        vals = []
        for rs in range(3):
            p = cross_val_predict(clf, A, yv, cv=StratifiedKFold(5, shuffle=True, random_state=rs),
                                  method="predict_proba")[:, 1]
            vals.append(roc_auc_score(yv, p))
        return float(np.mean(vals))
    disc = {}
    for name, f in fams.items():
        Do = np.concatenate(f["D"])
        Xo = np.concatenate(f["X"])
        disc[name] = dict(
            non_hazard_descriptors_auc=auc(Don_c, Do),
            all_hazard_and_descriptors_auc=auc(np.column_stack([Xon_c, Don_c]), np.column_stack([Xo, Do])),
            hazard_features_only_auc=auc(Xon_c, Xo),
            n_synthetic_onsets=int(len(Do)), n_real_onsets=int(len(Don_c)))
    # in-distribution reference: real onsets vs uniformly random visible competition frames
    rng = np.random.default_rng(1)
    ok = np.flatnonzero(eligc)
    pick = rng.choice(ok, size=5 * len(t_c), replace=False)
    tsgc = time_since_gap(comp)
    Dr = np.column_stack([auxc["mean_player_speed"][pick], auxc["team_spread"][pick], tsgc[pick]])
    disc["reference_real_onsets_vs_random_visible_frames_same_feed"] = dict(
        non_hazard_descriptors_auc=auc(Don_c, Dr),
        hazard_features_only_auc=auc(Xon_c, Xc[pick]))
    realism["discriminator_cv_auc"] = disc
    realism["hazard"] = hz["info"]
    realism["hazard_hash"] = hazard_hash(hz)
    OUT.joinpath("realism.json").write_text(json.dumps(realism, indent=1, default=float))
    print(json.dumps({k: v for k, v in realism.items() if k not in ("hazard", "per_match")}, indent=1, default=float))
    return realism



SEEDS = (2026, 2027)
VIEW_REPLICATES = (("clean", ("clean",)), ("feed_like", ("corrupt_1", "corrupt_2")),
                   ("random_gaps_matched", ("rand_1", "rand_2")), ("context_gaps", ("ctx_1", "ctx_2")))
DECISION_RULE = ("Equal-provider mean spellF1(context_gaps) <= spellF1(feed_like) - 0.02 AND the fraction of "
                 "true transitions (H<->A and live<->D, pooled over providers equally) hidden inside synthetic "
                 "ball gaps is higher for context_gaps than for feed_like -> feed_like is optimistic and "
                 "context_gaps is recommended as primary stress view. Otherwise feed_like placement is adequate.")
BASE = ROOT / "data/experiments/objective_alignment_v1"


def _checkpoint(source, seed):
    return BASE / f"without_{source}_combined_{seed}.joblib"


def freeze():
    if DECLARATION.exists():
        raise SystemExit("declaration already frozen")
    import joblib
    hz = joblib.load(OUT / "hazard.joblib")
    realism = json.loads((OUT / "realism.json").read_text())
    ckpt = {}
    for source, path in _cells():
        for seed in SEEDS:
            c = _checkpoint(source, seed)
            ckpt[str(c.relative_to(ROOT))] = sha256(c)
    spec = dict(
        id="context_gaps_v1", organizer_labels_or_ground_truth_read=False, scored_on_competition_match=False,
        hazard=dict(hash=realism["hazard_hash"], features=list(FEATURES), params=HAZARD_PARAMS,
                    fit_on="data/challenge/inputs.npz tracking only (ball, players, period, timestamps); labels key never read",
                    lag_seconds=LAG_S, hazard_basis="per frame at 25 Hz, converted by 25/fps",
                    duration="empirical competition durations capped at 30 s, three bins by predicted-hazard tercile",
                    auc_second_half=realism["hazard"]["auc_second_half"]),
        scalar_per_match=realism["scalar"], target_synthetic_coverage=TARGET_MISSING,
        scalar_definition=("one hazard multiplier per match; bisection on that match's clean input so that the fraction of active "
                           "frames inside synthetic gap intervals = 0.4069 (the quantity the feed_like renewal process targets; "
                           "native provider gaps are overlaid, so union missing is higher)"),
        views=dict(clean="reuse saved, verified by decode of saved member probabilities",
                   feed_like="reuse saved corrupt_1/corrupt_2, verified",
                   random_gaps_matched="rand_1, rand_2: same per-block gap count and durations as ctx_k, uniform onsets",
                   context_gaps="ctx_1, ctx_2"),
        other_components="stress_components.corrupt(clean, calibration, ce.seed(match,'corrupt_k'), coordinates+slot_gaps+id_swaps) "
                         "then synthetic ball gaps applied; ball-gap seeds ce.seed(match,'ctx_k') / ce.seed(match,'rand_k')",
        matches=[p.stem for _, p in _cells()], seeds=list(SEEDS), checkpoints_sha256=ckpt,
        gap_files_sha256={p.stem: sha256(OUT / f"{p.stem}_gaps.joblib") for _, p in _cells()},
        source_sha256=sha256(ROOT / "experiments/context_gaps.py"),
        decision_rule=DECISION_RULE, aggregation="replicates within match, matches within provider, providers equal",
        no_retraining_no_decoder_tuning=True)
    DECLARATION.write_text(json.dumps(spec, indent=1))
    print("FROZEN", sha256(DECLARATION))


def _check_frozen():
    spec = json.loads(DECLARATION.read_text())
    for rel, h in spec["checkpoints_sha256"].items():
        if sha256(ROOT / rel) != h:
            raise ValueError("checkpoint changed " + rel)
    for stem, h in spec["gap_files_sha256"].items():
        if sha256(OUT / f"{stem}_gaps.joblib") != h:
            raise ValueError("gap file changed " + stem)
    return spec


def _saved_rows():
    rows = json.loads((ROOT / "experiments/results/objective_ensemble_rows.json").read_text())["rows"]
    return {(r["match"], r["replicate"]): r for r in rows if r["recipe"] == "combined_objective_equal_probability_ensemble"}


def score(match):
    import joblib
    from experiments import (corruption_experiment as ce, gru_robustness_experiment as gr, live_gap_experiment as old,
                             learned_temporal as lt, objective_ensemble as oe, seed_gap_context as sg,
                             stress_components as sc)
    spec = _check_frozen()
    cal = _cal()
    saved = _saved_rows()
    out_path = OUT / f"{match}_scores.json"
    if out_path.exists():
        return
    (source, path), = [(s, p) for s, p in _cells() if p.stem == match]
    clean, labels = _clean(path)
    gaps = joblib.load(OUT / f"{match}_gaps.joblib")
    native_missing = ~np.isfinite(clean["ball"]).all(1)
    act = active_mask(clean)
    trans = transition_points(labels, clean)
    period = supervised.model_periods(clean)
    rows = []

    def coverage_row(synth):
        cov = transition_coverage(trans, synth, synth | native_missing)
        cov["synthetic_active_fraction"] = float(synth[act].mean())
        cov["any_missing_active_fraction"] = float((synth | native_missing)[act].mean())
        return cov

    def cell_row(view, rep, score_row, synth, parity=None):
        r = dict(source=source, match=match, view=view, replicate=rep,
                 **{k: score_row[k] for k in ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall", "n_gt_spells",
                                             "n_pred_spells", "short_spell_recall", "short_reference_spells", "spell_by_iou")},
                 coverage=coverage_row(synth))
        if parity is not None:
            r["parity_abs_diff_spell_f1"] = parity
        return r

    def member_probabilities(view_name, rep_name, record, inputs, saved_file=None):
        ps = []
        for seed in SEEDS:
            if saved_file is not None:
                with np.load(BASE / saved_file.format(seed=seed)) as z:
                    ps.append(z["probability"])
            else:
                fitted = ce.load_model(BASE, _checkpoint(source, seed))
                ps.append(sg.native_probability(record, fitted, inputs))
        return sg.average_probabilities(*ps)

    # clean and feed_like: verified reuse of saved member probabilities (decode + official score)
    for view, reps in VIEW_REPLICATES[:2]:
        for rep in reps:
            vname = "clean" if view == "clean" else "feed_like"
            f = f"{source}_{match}_{vname}_{rep}_combined_{{seed}}.npz"
            p = member_probabilities(view, rep, None, clean, saved_file=f)
            pred = supervised.decode(p, period, float(clean["fps"]), lt.COST)
            sr = oe.score_prediction(labels, pred, clean, "reuse", dict(source=source, match=match, view=view, replicate=rep), "x")
            ref = saved[(match, rep)]
            diff = abs(sr["spell_f1"] - ref["spell_f1"])
            if diff > 1e-9:
                raise ValueError(f"saved-row parity failed {match} {rep} {diff}")
            if view == "clean":
                synth = np.zeros(len(native_missing), bool)
            else:
                vis = dict(clean, ball=np.full_like(clean["ball"], 50.), ball_observed=np.ones(len(clean["ball"]), bool))
                synth = np.isnan(sc.corrupt(vis, cal, _seed(match, rep), ("ball_gaps",))["ball"]).any(1)
            rows.append(cell_row(view, rep, sr, synth, diff))
            print("REUSED", match, view, rep, round(sr["spell_f1"], 5), flush=True)
    # new views
    for view, reps in VIEW_REPLICATES[2:]:
        for rep in reps:
            k = rep.split("_")[1]
            t0 = time.monotonic()
            g = gaps[rep]
            view_in = build_view(clean, cal, _seed(match, f"corrupt_{k}"), g)
            record = old.feature_view(view_in, labels, source, match)["joint"]
            p = member_probabilities(view, rep, record, view_in)
            pred = supervised.decode(p, period, float(clean["fps"]), lt.COST)
            sr = oe.score_prediction(labels, pred, view_in, "context_gaps", dict(source=source, match=match, view=view, replicate=rep),
                                     hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest())
            synth = gap_mask(len(native_missing), g)
            rows.append(cell_row(view, rep, sr, synth))
            np.savez_compressed(OUT / f"{match}_{rep}_prob.npz", probability=p.astype(np.float32))
            print("SCORED", match, view, rep, round(sr["spell_f1"], 5), round(time.monotonic() - t0, 1), flush=True)
    out_path.write_text(json.dumps(rows, indent=1, default=float))


def report():
    spec = _check_frozen()
    rows = []
    for _, path in _cells():
        rows += json.loads((OUT / f"{path.stem}_scores.json").read_text())
    views = [v for v, _ in VIEW_REPLICATES]
    providers = ["sportec", "metrica", "skillcorner"]

    def agg(fn, view):
        per_provider = {}
        for prov in providers:
            per_match = []
            for m in sorted({r["match"] for r in rows if r["source"] == prov}):
                reps = [fn(r) for r in rows if r["match"] == m and r["view"] == view]
                per_match.append(np.mean(reps))
            per_provider[prov] = float(np.mean(per_match))
        return float(np.mean(list(per_provider.values()))), per_provider

    fields = dict(spell_f1=lambda r: r["spell_f1"], macro_f1=lambda r: r["macro_f1"], d_f1=lambda r: r["d_f1"],
                  short_spell_recall=lambda r: r["short_spell_recall"], n_pred_spells=lambda r: r["n_pred_spells"],
                  n_gt_spells=lambda r: r["n_gt_spells"])
    for t in ("0.3", "0.4", "0.5", "0.6", "0.7"):
        fields[f"spell_iou_{t}"] = (lambda tt: lambda r: r["spell_by_iou"][tt])(t)
    table = {v: {} for v in views}
    for v in views:
        for name, fn in fields.items():
            mean, per = agg(fn, v)
            table[v][name] = dict(equal_provider_mean=mean, per_provider=per)
    cov = {}
    for v in views:
        cov[v] = {}
        for kind in ("ha_change", "live_dead"):
            def rate(key):
                return lambda r: (r["coverage"][kind][key] / max(1, r["coverage"][kind]["n"]))
            m, per = agg(rate("hidden_synthetic"), v)
            m2, per2 = agg(rate("hidden_any_missing"), v)
            cov[v][kind] = dict(hidden_in_synthetic_gap=dict(equal_provider_mean=m, per_provider=per),
                                hidden_in_any_missing_ball=dict(equal_provider_mean=m2, per_provider=per2),
                                n_transitions_total={prov: int(sum(r["coverage"][kind]["n"] for r in rows
                                                                    if r["source"] == prov and r["view"] == "clean"))
                                                     for prov in providers})
        cov[v]["synthetic_active_fraction"] = agg(lambda r: r["coverage"]["synthetic_active_fraction"], v)[0]
        cov[v]["any_missing_active_fraction"] = agg(lambda r: r["coverage"]["any_missing_active_fraction"], v)[0]
        hs = lambda r: sum(r["coverage"][k]["hidden_synthetic"] for k in ("ha_change", "live_dead")) / max(
            1, sum(r["coverage"][k]["n"] for k in ("ha_change", "live_dead")))
        cov[v]["all_transitions_hidden_synthetic"] = agg(hs, v)[0]
        cov[v]["enrichment_vs_coverage"] = (cov[v]["all_transitions_hidden_synthetic"] /
                                            cov[v]["synthetic_active_fraction"]) if cov[v]["synthetic_active_fraction"] else None
    drop = table["feed_like"]["spell_f1"]["equal_provider_mean"] - table["context_gaps"]["spell_f1"]["equal_provider_mean"]
    placement = table["random_gaps_matched"]["spell_f1"]["equal_provider_mean"] - table["context_gaps"]["spell_f1"]["equal_provider_mean"]
    higher = (cov["context_gaps"]["all_transitions_hidden_synthetic"] > cov["feed_like"]["all_transitions_hidden_synthetic"])
    result = dict(
        study_id="context_gaps_v1", declaration_sha256=sha256(DECLARATION),
        organizer_labels_or_ground_truth_read=False, scored_on_competition_match=False,
        scores=table, transition_coverage=cov,
        decision=dict(rule=DECISION_RULE, feed_like_minus_context_spell_f1=drop,
                      random_matched_minus_context_spell_f1=placement,
                      context_hides_transitions_at_higher_rate_than_feed_like=bool(higher),
                      feed_like_optimistic=bool(drop >= 0.02 and higher)),
        parity_max_abs_diff_saved_rows=max(r.get("parity_abs_diff_spell_f1", 0.) for r in rows))
    realism = json.loads((OUT / "realism.json").read_text())
    realism.pop("per_match", None)
    result["realism"] = realism
    result["hazard"] = realism.pop("hazard")
    RESULT.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps(dict(scores={v: table[v]["spell_f1"]["equal_provider_mean"] for v in views},
                          decision=result["decision"]), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prepare", "freeze", "score", "report"])
    ap.add_argument("--match", default=None)
    a = ap.parse_args()
    {"prepare": prepare, "freeze": freeze, "report": report}.get(a.stage, lambda: score(a.match))()
