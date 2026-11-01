"""Feed-like benchmark audit: ceilings, challenge-matched ball gaps and test-time views.

Study `feed_benchmark_audit_v1`. No fitting, no organizer labels and no public scoring.
It re-uses archived probabilities and sealed fold checkpoints from earlier studies:

* `paired`    paired block bootstrap of candidate vs adopted vs leader on archived cells.
* `oracle`    label-oracle headroom decomposition (analysis only, not deployable).
* `decoder`   per-cell oracle choice over 54 calibrated-Viterbi settings.
* `arms`      score every archived arm and the per-cell oracle arm / equal blend ceilings.
* `challenge` label-free ball-gap profile of the organizer tracking under the adopted CSV.
* `views`     regenerate random-placement and challenge-matched (state-dependent) ball gaps
              on the five development matches and score adopted / candidate / raw-feed models.
* `tta`       team-swap and 5 Hz sampling-phase test-time views.
* `report`    write the aggregate JSON (no frame arrays) to experiments/results.

Archive locations (private, ignored) come from environment variables:
MTAH_DATA_DIR (checkout of the short-quality-diversity research archive),
MTAH_FSP_DIR (feed-spell-push archive) and MTAH_DT_DIR (dense-targeted archive).
Dev labels are used only to score development cells and to place simulated gaps in
development views; inference never receives labels. The challenge profile reads
organizer tracking inputs and the adopted label-free CSV only, never organizer labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import glob
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import joblib
import numpy as np

from experiments import feed_spell_push as decoder, spell_mbr_run as shared, metrics
from experiments import soccernet_extension as old, sn_dense_train as sn
from experiments import tracking_corruption as tc, supervised, preparation as prep
from experiments import stress_components as stc, learned_temporal as lt, refine_features
import evaluation.possession_metrics as pm

STUDY = "feed_benchmark_audit_v1"
DATA = Path(os.environ["MTAH_DATA_DIR"])
FSP = Path(os.environ.get("MTAH_FSP_DIR", str(ROOT.parent / "mtah-fsp")))
DT = Path(os.environ.get("MTAH_DT_DIR", str(ROOT.parent / "mtah-dt")))
OUT = ROOT / "data/experiments" / STUDY
RESULT = ROOT / "experiments/results" / f"{STUDY}.json"
WORKERS = int(os.environ.get("AUDIT_WORKERS", "4"))
CONFIG = next(c for c in decoder.configurations() if c["id"] == "f062")
FEED = ("feed_like_corrupt_1", "feed_like_corrupt_2")
MATCHES = (("Sportec", "J03WOY", "J03WOY"), ("Metrica", "metrica_3", "metrica_3"),
           ("SkillCorner", "1996435", "skillcorner_had_1996435"),
           ("SkillCorner", "1996436", "skillcorner_had_1996436"),
           ("SkillCorner", "2006229", "skillcorner_had_2006229"))
PROVIDER = {m[1]: m[0] for m in MATCHES}
FILE = {m[1]: m[2] for m in MATCHES}
FOLD = {"Metrica": "metrica", "SkillCorner": "skillcorner", "Sportec": "sportec"}
EXP = DATA / "experiments"
SQD = EXP / "short_quality_diversity_delivery_v1/artifacts"
CANDIDATE = SQD / "short_original_diversity_v1/predictions/original_full"
LEADER = EXP / "feed_short_head_v1/artifacts/predictions"
ADOPTED = FSP / "experiments/feed_spell_push_delivery_v1/artifacts/feed_spell_push_v1/f062"
ADOPTED_CSV = FSP / "experiments/feed_spell_push_delivery_v1/artifacts/feed_spell_public_v1/adopted_replay.csv"
EVAL_INPUTS = EXP / "seed_gap_context_v1"
VIEW_KINDS = ("random", "matched", "matched_live2x")
REPLICATES = (1, 2, 3, 4)


# ---------------------------------------------------------------- shared helpers

def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def decode(p, period, fps, config=CONFIG):
    return decoder.decode(np.asarray(p, dtype=np.float64)[None], period, fps, config)


def score(cell, prediction):
    return shared._score_prediction(cell["labels"], prediction, cell["period"],
                                    cell["timestamps"], float(cell["fps"]))


def load_cells(directory, member="ensemble"):
    """Archived {provider}__{match}__{view}__{member}.npz cells with labels and probabilities."""
    out = {}
    for f in sorted(glob.glob(f"{directory}/*__{member}.npz")):
        provider, match, view, _ = os.path.basename(f)[:-4].split("__")
        with np.load(f, allow_pickle=False) as d:
            out[(provider, match, view)] = {k: d[k] for k in d.files}
    return out


def load_adopted():
    """f062 archive stores decoded predictions only, named seedensemble__{match}__{view}__{rep}."""
    out = {}
    for f in sorted(glob.glob(f"{ADOPTED}/seedensemble__*.npz")):
        _, match, view, rep = os.path.basename(f)[:-4].split("__")
        match = match.replace("skillcorner_had_", "")
        out[(PROVIDER[match], match, "clean" if view == "clean" else f"feed_like_{rep}")] = \
            np.load(f)["prediction"]
    return out


def equal_provider(values):
    """values: {(provider, match, replicate/view): x} -> replicate mean, match mean, provider mean."""
    provs = {}
    for (provider, match, _), v in values.items():
        provs.setdefault(provider, {}).setdefault(match, []).append(v)
    per = {p: float(np.mean([np.mean(r) for r in m.values()])) for p, m in provs.items()}
    return float(np.mean(list(per.values()))), per


def summary(rows, key="spell_f1"):
    out = {}
    for name, views in (("clean", ("clean",)), ("feed_like", FEED)):
        mean, per = equal_provider({k: r[key] for k, r in rows.items() if k[2] in views})
        out[name] = {"equal_provider_mean": mean, "providers": per}
    return out


def tracking_cell(path):
    cached = joblib.load(path)
    inputs = cached["inputs"]
    cell = dict(labels=np.asarray(cached["labels"]).astype(str), period=inputs["period"],
                timestamps=inputs["timestamps"], fps=float(inputs["fps"]),
                model_period=supervised.model_periods(inputs))
    return cached, cell


# ---------------------------------------------------------------- model inference

def model_path(arm, provider, seed):
    fold = FOLD[provider]
    if arm == "cand":
        return SQD / f"short_original_diversity_v1/models/leave_out_{fold}_{seed}.joblib"
    if arm == "adop":
        return DT / f"experiments/dense_targeted_followups_v1/artifacts/control_D2/models/control_D2__{fold}__s{seed}.joblib"
    if arm == "raw":
        return EXP / f"feed_robustness_v1/artifacts/models/leave_out_{fold}_{seed}.joblib"
    raise ValueError(arm)


def teacher_path(provider, seed):
    return EXP / f"objective_alignment_v1/without_{FOLD[provider]}_combined_{seed}.joblib"


_LOADED = {}


def _load(path):
    if path not in _LOADED:
        _LOADED[path] = joblib.load(path)
    return _LOADED[path]


def predict(inputs, record, arm, provider):
    """Two-seed probability ensemble, exactly as the archived evaluators (labels never enter)."""
    record = old.inference_record(record)
    record["names"] = list(record["names"])
    native = tc.inputs_only(inputs)
    probs = [old.upsample(sn.predict(record, _load(model_path(arm, provider, s)),
                                     _load(teacher_path(provider, s))), record, native)
             for s in (2026, 2027)]
    return np.mean(probs, axis=0)


def phase_records(inputs, base):
    """Records for each 5 Hz sampling phase; phase 0 is the archived record exactly."""
    inputs = tc.inputs_only(inputs)
    team, _ = refine_features.features(inputs)
    player, _ = lt.player_features(inputs)
    period = supervised.model_periods(inputs)
    stride = round(float(inputs["fps"]) / lt.RATE)
    records = []
    for k in range(stride):
        idx = np.concatenate([np.arange(a + k, b, stride) if b - a > k else np.arange(a, b, stride)
                              for a, b in supervised.blocks(period)])
        records.append(dict(x=np.column_stack((team[idx], player[idx])), names=list(base["names"]),
                            period=period[idx], indices=idx, native_n=base["native_n"], fps=base["fps"]))
    return records


# ---------------------------------------------------------------- 1. paired bootstrap

BLOCK_SECONDS = 120


def _block_counts(cell, prediction, fps):
    _, truth, guess, _ = metrics.prepare_tracking_codes(cell["labels"], prediction, period=cell["period"],
                                                        fps=fps, timestamps=cell["timestamps"])
    runs = pm.valid_runs(truth)
    gt, pr = pm.possession_spells(truth, runs), pm.possession_spells(guess, runs)
    overlaps = pm.spell_overlaps(gt, pr)
    flags = np.array([pm.matched_gt_flags(overlaps, len(gt), t) for t in pm.SPELL_IOUS])
    block = int(BLOCK_SECONDS * fps)
    nb = len(truth) // block + 1
    tp, ngt, npr = np.zeros((5, nb)), np.zeros(nb), np.zeros(nb)
    for i, (a, _, _) in enumerate(gt):
        ngt[a // block] += 1
        tp[:, a // block] += flags[:, i]
    for a, _, _ in pr:
        npr[a // block] += 1
    return tp, ngt, npr


def run_paired(draws=2000):
    cand, lead, adop = load_cells(CANDIDATE), load_cells(LEADER), load_adopted()
    models = {"candidate": {k: c["prediction"] for k, c in cand.items()},
              "leader": {k: c["prediction"] for k, c in lead.items()}, "adopted": adop}
    counts = {(name,) + k: _block_counts(c, models[name][k], float(c["fps"]))
              for k, c in cand.items() for name in models}

    def agg(name, views, idx=None):
        values = {}
        for k in cand:
            if k[2] not in views:
                continue
            tp, ngt, npr = counts[(name,) + k]
            sel = idx[k[1]] if idx is not None else slice(None)
            values[k] = np.mean([2 * tp[i, sel].sum() / (ngt[sel].sum() + npr[sel].sum()) for i in range(5)])
        return equal_provider(values)[0]

    rng = np.random.default_rng(20261101)
    sizes = {k[1]: counts[("candidate",) + k][1].shape[0] for k in cand}
    out = {"point": {n: {"clean": agg(n, ("clean",)), "feed_like": agg(n, FEED)} for n in models}}
    for other in ("adopted", "leader"):
        for view, views in (("clean", ("clean",)), ("feed_like", FEED)):
            deltas = []
            for _ in range(draws):
                idx = {m: rng.integers(0, n, n) for m, n in sizes.items()}
                deltas.append(agg("candidate", views, idx) - agg(other, views, idx))
            deltas = np.array(deltas)
            out[f"candidate_minus_{other}_{view}"] = dict(
                delta=agg("candidate", views) - agg(other, views),
                ci90=[float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))],
                p_positive=float((deltas > 0).mean()))
    out["method"] = (f"{BLOCK_SECONDS}s blocks resampled within match; spells credited to the block of their "
                     f"reference/prediction start; {draws} draws. Ignores between-match, realization and "
                     "selection uncertainty, so intervals are lower bounds on total uncertainty.")
    write(OUT / "paired.json", out)
    return out


# ---------------------------------------------------------------- 2. oracle headroom

def _runs(x):
    b = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1, len(x)]
    return [(b[i], b[i + 1], x[b[i]]) for i in range(len(b) - 1)]


def oracles(c):
    labels, pred, prob = c["labels"], c["prediction"].copy(), c["probabilities"]
    fps = float(c["fps"])
    graded = labels != "X"
    out = {"model": pred}
    o = pred.copy()
    o[graded & (labels == "D")] = "D"
    m = graded & (labels != "D") & (pred == "D")
    o[m] = np.where(prob[m, 0] >= prob[m, 1], "H", "A")
    out["dead_live"] = o
    o = pred.copy()
    m = graded & (labels != "D") & (pred != "D")
    o[m] = labels[m]
    out["team"] = o
    for name, keep in (("short_lt1s", lambda d: d < 1), ("short_lt3s", lambda d: d < 3),
                       ("long_ge3s", lambda d: d >= 3)):
        o = pred.copy()
        for a, b, k in _runs(labels):
            if k in "HA" and keep((b - a) / fps):
                o[a:b] = k
        out[name] = o
    o = pred.copy()
    for a, b, k in _runs(labels):
        if k in "HA" and (b - a) / fps < 3:
            seg = o[a:b]
            seg[seg != "D"] = k
            o[a:b] = seg
    out["team_within_short_lt3s"] = o
    return out


def run_oracle():
    cand = load_cells(CANDIDATE)
    rows = {}
    for k, c in cand.items():
        for name, prediction in oracles(c).items():
            rows.setdefault(name, {})[k] = score(c, prediction)
    out = {n: {"spell": summary(r), "macro": summary(r, "macro_f1")} for n, r in rows.items()}
    write(OUT / "oracle.json", out)
    return out


# ---------------------------------------------------------------- 3. decoder per-cell oracle

GRID = [dict(family="calibration", dt=dt, ht=ht, ha=ha, dead=de, id=f"a{i}")
        for i, (dt, ht, ha, de) in enumerate((dt, ht, ha, de) for dt in (2, 4, 8) for ht in (.5, 1)
                                             for ha in (1.5, 3, 4.5) for de in (1.5, 3, 6))]
_CAND = None


def _decoder_job(arg):
    global _CAND
    if _CAND is None:
        _CAND = load_cells(CANDIDATE)
    k, g = arg
    c = _CAND[k]
    return k, g, score(c, decode(c["probabilities"], c["model_period"], float(c["fps"]), GRID[g]))["spell_f1"]


def run_decoder():
    keys = [k for k in load_cells(CANDIDATE) if k[2] in FEED]
    with ProcessPoolExecutor(WORKERS) as ex:
        table = {(k, g): f for k, g, f in ex.map(_decoder_job, [(k, g) for k in keys for g in range(len(GRID))], chunksize=4)}
    feed = lambda choose: equal_provider({k: table[(k, choose(k))] for k in keys})[0]
    f062 = next(i for i, c in enumerate(GRID) if (c["dt"], c["ht"], c["ha"], c["dead"]) == (4, 1, 3, 3))
    best = max(range(len(GRID)), key=lambda g: feed(lambda k: g))
    out = dict(settings=len(GRID), f062_feed=feed(lambda k: f062), best_global=GRID[best],
               best_global_feed=feed(lambda k: best),
               per_provider_oracle_feed=equal_provider({k: table[(k, max(range(len(GRID)), key=lambda g: np.mean(
                   [table[(j, g)] for j in keys if j[0] == k[0]])))] for k in keys})[0],
               per_cell_oracle_feed=feed(lambda k: max(range(len(GRID)), key=lambda g: table[(k, g)])))
    write(OUT / "decoder.json", out)
    return out


# ---------------------------------------------------------------- 4. arm ceilings

def arm_dirs():
    dirs = sorted({os.path.dirname(f) for f in glob.glob(f"{EXP}/*/artifacts/**/*__ensemble.npz", recursive=True)})
    keep = []
    for d in dirs:
        files = glob.glob(d + "/*__ensemble.npz")
        if "fresh" in d or "calibration" in d or len(files) != 15 or os.path.getsize(files[0]) < 10000:
            continue
        keep.append(d)
    return {d.replace(str(EXP) + "/", "").replace("/artifacts", "").replace("/predictions", ""): d for d in keep}


def _arm_job(item):
    name, d = item
    cells = load_cells(d)
    return name, {k: score(c, decode(c["probabilities"], c["model_period"], float(c["fps"])))["spell_f1"]
                  for k, c in cells.items() if k[2] in FEED}


def run_arms():
    arms = arm_dirs()
    with ProcessPoolExecutor(WORKERS) as ex:
        res = dict(ex.map(_arm_job, arms.items()))
    keys = list(next(iter(res.values())))
    cells = {n: load_cells(d) for n, d in arms.items()}
    cand_name = "short_quality_diversity_delivery_v1/short_original_diversity_v1/original_full"
    blend, shared_all, shared_major, wrong = {}, 0, 0, 0
    for k in keys:
        c = cells[cand_name][k]
        p = np.mean([cells[n][k]["probabilities"] for n in cells], axis=0)
        blend[k] = score(c, decode(p, c["model_period"], float(c["fps"])))["spell_f1"]
        graded = c["labels"] != "X"
        errs = np.array([cells[n][k]["prediction"] != c["labels"] for n in cells])[:, graded]
        cw = (c["prediction"] != c["labels"])[graded]
        wrong += cw.sum()
        shared_all += (cw & errs.all(0)).sum()
        shared_major += (cw & (errs.mean(0) > .5)).sum()
    out = dict(arms={n: equal_provider(r)[0] for n, r in res.items()},
               per_cell_oracle_arm_feed=equal_provider({k: max(r[k] for r in res.values()) for k in keys})[0],
               equal_probability_blend_feed=equal_provider(blend)[0],
               candidate_feed_errors_shared_by_all_arms=float(shared_all / wrong),
               candidate_feed_errors_shared_by_majority=float(shared_major / wrong))
    write(OUT / "arms.json", out)
    return out


# ---------------------------------------------------------------- 5. challenge gap profile

def _runs_mask(mask):
    b = np.flatnonzero(np.diff(np.r_[0, np.asarray(mask).astype(int), 0]))
    return list(zip(b[::2], b[1::2]))


def challenge_profile():
    """Organizer tracking + adopted label-free CSV only. The npz `labels` key is never read."""
    with np.load(DATA / "challenge/inputs.npz", allow_pickle=False) as d:
        observed, period = d["ball_observed"], d["period"]
    pred = np.array([r[1] for r in csv.reader(ADOPTED_CSV.open())][1:])
    inplay = period > 0
    dead = pred == "D"
    gaps = _runs_mask((~observed) & inplay)
    dur = np.array([(b - a) / 25 for a, b in gaps])
    cover = np.array([(pred[a:b] == "D").mean() for a, b in gaps])
    live_gaps = np.array([(b - a) / 25 for a, b in _runs_mask((~observed) & inplay & ~dead)])
    bins = {}
    for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 60), (60, np.inf)):
        s = (dur >= lo) & (dur < hi)
        if s.any():
            bins[f"{lo}-{hi}s"] = dict(n=int(s.sum()), time_share=float(dur[s].sum() / dur.sum()),
                                       predicted_dead_coverage=float(np.average(cover[s], weights=dur[s])))
    return dict(missing_fraction=float((~observed[inplay]).mean()),
                missing_given_predicted_dead=float((~observed[inplay & dead]).mean()),
                missing_given_predicted_live=float((~observed[inplay & ~dead]).mean()),
                predicted_dead_given_missing=float(dead[inplay & ~observed].mean()),
                predicted_dead_given_observed=float(dead[inplay & observed].mean()),
                gap_runs=len(gaps), gap_duration_bins=bins,
                live_gap_runs=len(live_gaps), live_gap_median_seconds=float(np.median(live_gaps)),
                live_gap_pool_seconds=np.minimum(live_gaps, 30.).tolist())


def dev_gap_profile():
    out = {}
    cand = load_cells(CANDIDATE)
    for k, c in cand.items():
        name = FILE[k[1]] + ("_clean_clean" if k[2] == "clean" else "_" + k[2])
        inputs = joblib.load(EVAL_INPUTS / f"{name}.joblib")["inputs"]
        obs = np.asarray(inputs["ball_observed"], bool)
        g = c["labels"] != "X"
        out["|".join(k)] = dict(
            missing=float((~obs[g]).mean()),
            true_dead_given_missing=float((c["labels"][g & ~obs] == "D").mean()) if (g & ~obs).any() else None,
            true_dead_given_observed=float((c["labels"][g & obs] == "D").mean()),
            predicted_dead_given_missing=float((c["prediction"][g & ~obs] == "D").mean()) if (g & ~obs).any() else None)
    return out


def run_challenge():
    out = dict(challenge=challenge_profile(), development=dev_gap_profile())
    write(OUT / "challenge.json", out)
    return out


# ---------------------------------------------------------------- 6. matched-gap views

def state_gaps(ball, labels, fps, rng, live_rate, dead_rate, pool):
    """Blank ball inside dead runs at the challenge rate and add short live gaps from the pool.

    Development labels place the simulated gaps only; inference never sees labels.
    """
    missing = ~np.isfinite(ball).all(axis=1)
    for a, b in _runs_mask(labels == "D"):
        if rng.random() < min(1., dead_rate / .85):
            u, v = a + int(rng.uniform(0, 1.5) * fps), b - int(rng.uniform(0, 1.) * fps)
            if v > u:
                ball[u:v] = np.nan
    live = (labels == "H") | (labels == "A")
    need = max(0., live_rate * live.sum() - (missing & live).sum())
    rate = need / max(1, live.sum()) / (pool.mean() * fps)
    for a, b in _runs_mask(live):
        at = a
        while True:
            at += max(1, int(rng.exponential(1 / max(rate, 1e-9))))
            if at >= b:
                break
            dur = max(1, int(rng.choice(pool) * fps))
            ball[at:min(b, at + dur)] = np.nan
            at += dur
    return ball


def _view_job(arg):
    provider, match, kind, rep, target = arg
    path = OUT / "views" / f"{match}__{kind}__{rep}.joblib"
    if path.exists():
        return str(path)
    cached = joblib.load(EVAL_INPUTS / f"{FILE[match]}_clean_clean.joblib")
    labels = np.asarray(cached["labels"]).astype(str)
    native = tc.inputs_only(cached["inputs"])
    if "ball_observed" in native:
        native["ball"][~np.asarray(native["ball_observed"], bool)] = np.nan
    if "player_observed" in native:
        native["players"][~np.asarray(native["player_observed"], bool)] = np.nan
    seed = int(hashlib.sha256(f"{STUDY}/{kind}/{rep}/{match}".encode()).hexdigest()[:8], 16)
    calibration = sn.default_ball_calibration()
    if kind == "random":
        view = stc.corrupt(native, calibration, seed, ("coordinates", "ball_gaps", "slot_gaps", "id_swaps"))
    else:
        view = stc.corrupt(native, calibration, seed, ("coordinates", "slot_gaps", "id_swaps"))
        live = target["missing_given_predicted_live"] * (2 if kind == "matched_live2x" else 1)
        rng = np.random.default_rng(seed + 1)
        pool = np.asarray(target["live_gap_pool_seconds"])
        for a, b in supervised.blocks(supervised.model_periods(view)):
            if view["period"][a] > 0:
                view["ball"][a:b] = state_gaps(view["ball"][a:b], labels[a:b], float(view["fps"]), rng,
                                               live, target["missing_given_predicted_dead"], pool)
    view["ball_observed"] = np.isfinite(view["ball"]).all(axis=1)
    view["player_observed"] = np.isfinite(view["players"]).all(axis=2)
    record = prep.build_record(view, labels, provider, FILE[match])
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(inputs=view, labels=labels, record=record), path)
    return str(path)


def _score_view_job(path):
    match, kind, rep = Path(path).name[:-7].split("__")
    provider = PROVIDER[match]
    dest = OUT / "view_scores" / f"{match}__{kind}__{rep}.json"
    if dest.exists():
        return str(dest)
    cached, cell = tracking_cell(path)
    inputs, labels = cached["inputs"], cell["labels"]
    obs, graded = np.asarray(inputs["ball_observed"], bool), labels != "X"
    out = dict(missing_given_dead=float((~obs[graded & (labels == "D")]).mean()),
               missing_given_live=float((~obs[graded & (labels != "D")]).mean()))
    records = phase_records(inputs, cached["record"])
    for arm in ("adop", "cand", "raw"):
        phases = [predict(inputs, r, arm, provider) for r in records] if arm != "raw" else \
                 [predict(inputs, cached["record"], arm, provider)]
        for mode, p in (("phase0", phases[0]), ("phase_tta", np.mean(phases, 0))):
            if arm == "raw" and mode == "phase_tta":
                continue
            r = score(cell, decode(p, cell["model_period"], cell["fps"]))
            out[f"{arm}_{mode}"] = {k: r[k] for k in ("spell_f1", "macro_f1", "d_f1", "predicted_spell_count",
                                                       "reference_spell_count")}
    dest.parent.mkdir(parents=True, exist_ok=True)
    write(dest, out)
    return str(dest)


def run_views():
    target = read(OUT / "challenge.json")["challenge"]
    jobs = [(p, m, k, r, target) for p, m, _ in MATCHES for k in VIEW_KINDS for r in REPLICATES]
    with ProcessPoolExecutor(WORKERS) as ex:
        paths = list(ex.map(_view_job, jobs))
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(_score_view_job, paths))
    out = {}
    for kind in VIEW_KINDS:
        rows = {}
        for f in glob.glob(str(OUT / "view_scores" / f"*__{kind}__*.json")):
            match, _, rep = Path(f).name[:-5].split("__")
            rows[(PROVIDER[match], match, rep)] = read(f)
        entry = dict(missing_given_dead=float(np.mean([r["missing_given_dead"] for r in rows.values()])),
                     missing_given_live=float(np.mean([r["missing_given_live"] for r in rows.values()])))
        for key in ("adop_phase0", "adop_phase_tta", "cand_phase0", "cand_phase_tta", "raw_phase0"):
            mean, per = equal_provider({k: r[key]["spell_f1"] for k, r in rows.items()})
            entry[key] = dict(spell_f1=mean, providers=per,
                              macro_f1=equal_provider({k: r[key]["macro_f1"] for k, r in rows.items()})[0],
                              per_replicate=[equal_provider({k: r[key]["spell_f1"] for k, r in rows.items()
                                                             if k[2] == str(rep)})[0] for rep in REPLICATES])
        entry["cand_minus_adop_per_replicate"] = [a - b for a, b in zip(entry["cand_phase0"]["per_replicate"],
                                                                         entry["adop_phase0"]["per_replicate"])]
        out[kind] = entry
    write(OUT / "views.json", out)
    return out


# ---------------------------------------------------------------- 7. TTA on archived cells

def _tta_job(name):
    dest = OUT / "tta" / f"{name}.npz"
    if dest.exists():
        return str(dest)
    match = next(m for m in PROVIDER if name.startswith(FILE[m]))
    provider = PROVIDER[match]
    cached = joblib.load(EVAL_INPUTS / f"{name}.joblib")
    inputs, labels = cached["inputs"], np.asarray(cached["labels"]).astype(str)
    swapped = tc.inputs_only(inputs)
    swapped["teams"] = 1 - np.asarray(swapped["teams"])
    swap_record = prep.build_record(swapped, labels, provider, FILE[match])
    records = phase_records(inputs, cached["record"])
    if not (np.array_equal(records[0]["indices"], cached["record"]["indices"])
            and np.allclose(records[0]["x"], cached["record"]["x"], equal_nan=True)):
        raise ValueError(f"Phase-0 record does not replay the archived record: {name}")
    out = {}
    for arm in ("cand", "adop"):
        phases = [predict(inputs, r, arm, provider) for r in records]
        out[f"{arm}_phase0"] = phases[0].astype(np.float32)
        out[f"{arm}_phase_tta"] = np.mean(phases, 0).astype(np.float32)
        out[f"{arm}_swap"] = predict(swapped, swap_record, arm, provider)[:, [1, 0, 2]].astype(np.float32)
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, **out)
    return str(dest)


def run_tta():
    cand = load_cells(CANDIDATE)
    names = {k: FILE[k[1]] + ("_clean_clean" if k[2] == "clean" else "_" + k[2]) for k in cand}
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(_tta_job, sorted(set(names.values()))))
    rows = {}
    for k, c in cand.items():
        with np.load(OUT / "tta" / f"{names[k]}.npz") as z:
            if np.abs(z["cand_phase0"] - c["probabilities"]).max() > 1e-6:
                raise ValueError(f"Candidate inference does not replay archived probabilities: {k}")
            for arm in ("cand", "adop"):
                variants = {"phase0": z[f"{arm}_phase0"], "phase_tta": z[f"{arm}_phase_tta"],
                            "swap": z[f"{arm}_swap"], "swap_tta": .5 * (z[f"{arm}_phase0"] + z[f"{arm}_swap"])}
                for mode, p in variants.items():
                    rows.setdefault(f"{arm}_{mode}", {})[k] = score(c, decode(p, c["model_period"], float(c["fps"])))
    adopted = load_adopted()
    mismatched = int(sum((decode(np.load(OUT / "tta" / f"{names[k]}.npz")["adop_phase0"], c["model_period"],
                                 float(c["fps"])) != adopted[k]).sum() for k, c in cand.items()))
    out = {n: {"spell": summary(r), "macro": summary(r, "macro_f1")} for n, r in rows.items()}
    out["adopted_phase0_decoded_label_mismatches_vs_archive"] = mismatched
    write(OUT / "tta.json", out)
    return out


# ---------------------------------------------------------------- report

def run_report():
    parts = {n: read(OUT / f"{n}.json") for n in ("paired", "oracle", "decoder", "arms", "challenge", "views", "tta")}
    parts["challenge"]["challenge"].pop("live_gap_pool_seconds")
    write(RESULT, dict(study=STUDY, organizer_reference_access=False, public_scoring=False, fitting=False,
                       promotion=False, frame_arrays_published=False, **parts))


STEPS = dict(paired=run_paired, oracle=run_oracle, decoder=run_decoder, arms=run_arms,
             challenge=run_challenge, views=run_views, tta=run_tta, report=run_report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=(*STEPS, "all"))
    step = parser.parse_args().step
    for name in (STEPS if step == "all" else (step,)):
        print("STEP", name, flush=True)
        STEPS[name]()
