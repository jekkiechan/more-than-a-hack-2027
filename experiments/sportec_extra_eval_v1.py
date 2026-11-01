"""Two additional held-out Sportec matches (J03WPY, J03WQQ) scored with four sealed arms.

Study `sportec_extra_eval_v1`. No fitting, no organizer labels, no public scoring, no promotion.
Every arm's `leave_out_sportec` fold excludes all Sportec data, so J03WPY and J03WQQ were never
training data. Development labels only place simulated defects and score the cells.

Arms: `adop`, `cand`, `raw` (v1 raw-feed augmented), `v2train` (v2 training views).
Modes: `phase0` (archived 5 Hz sampling phase) and `phase_tta` (mean over the five phases).
Views: `clean` (1), `v1` state-blind feed-like via feed_robustness_v1.raw_view (2),
`feed_like_v2` challenge-calibrated (4).

Steps: `sanity` (replay J03WOY and report match sizes), `j03woy` (fill J03WOY arm gaps from the
cached seed_gap_context_v1 inputs), `cells` (new matches), `aggregate`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_like_v2_benchmark as v2bench  # installs arm 'v2train' into audit.model_path
from experiments import feed_like_v2 as v2, feed_robustness_v1 as study1
from experiments import multisource_eval as ms, preparation as prep, supervised
from experiments import sn_dense_train as sn

STUDY = "sportec_extra_eval_v1"
ROOT = audit.ROOT
OUT = ROOT / "data/experiments" / STUDY
RESULT = ROOT / "experiments/results" / f"{STUDY}.json"
RAW = audit.DATA / "external_audit/sportec/processed"
PROVIDER = "Sportec"
NEW = ("J03WPY", "J03WQQ")
ARMS = ("adop", "cand", "raw", "v2train")
VIEWS = {"clean": (1,), "v1": (1, 2), "feed_like_v2": (1, 2, 3, 4)}
METRICS = ("spell_f1", "macro_f1", "d_f1", "predicted_spell_count", "reference_spell_count")
WORKERS = min(2, int(os.environ.get("AUDIT_WORKERS", "2")))
CACHED = audit.EVAL_INPUTS  # seed_gap_context_v1 for the existing J03WOY match
V2_SCORES = ROOT / "data/experiments/feed_like_v2_benchmark/scores"
RESULTS = ROOT / "experiments/results"


def seed_for(view, rep, match):
    return int(hashlib.sha256(f"{STUDY}/{view}/{rep}/{match}".encode()).hexdigest()[:8], 16)


def build_view(match, view, rep):
    """Return (inputs, labels, record, seed) for a new match view."""
    inputs, labels = ms.view(RAW / f"{match}.npz")
    labels = np.asarray(labels).astype(str)
    seed = seed_for(view, rep, match)
    if view == "v1":
        inputs = study1.raw_view(inputs, sn.default_ball_calibration(), seed)
    elif view == "feed_like_v2":
        inputs = v2.corrupt(inputs, labels, audit.read(v2bench.CALIBRATION)["calibration"], seed)
    elif view != "clean":
        raise ValueError(view)
    return inputs, labels, prep.build_record(inputs, labels, PROVIDER, match), seed


def score_arms(inputs, labels, record, arms=ARMS):
    """phase0 and phase-TTA metrics for each arm; the same code path for every cell."""
    cell = dict(labels=labels, period=inputs["period"], timestamps=inputs["timestamps"],
                fps=float(inputs["fps"]), model_period=supervised.model_periods(inputs))
    records = audit.phase_records(inputs, record)
    if not (np.array_equal(records[0]["indices"], record["indices"])
            and np.allclose(records[0]["x"], record["x"], equal_nan=True)):
        raise ValueError("Phase-0 record does not replay the base record")
    out = {}
    for arm in arms:
        phases = [audit.predict(inputs, r, arm, PROVIDER) for r in records]
        for mode, p in (("phase0", phases[0]), ("phase_tta", np.mean(phases, 0))):
            r = audit.score(cell, audit.decode(p, cell["model_period"], cell["fps"]))
            out[f"{arm}_{mode}"] = {k: r[k] for k in METRICS}
    graded = labels != "X"
    out["frames"] = int(len(labels))
    out["graded_frames"] = int(graded.sum())
    out["ball_missing_graded"] = float((~np.isfinite(inputs["ball"]).all(axis=1))[graded].mean())
    return out


def _cell_job(arg):
    match, view, rep = arg
    dest = OUT / "cells" / f"{match}__{view}__{rep}.json"
    if dest.exists():
        return str(dest)
    inputs, labels, record, seed = build_view(match, view, rep)
    out = score_arms(inputs, labels, record)
    out.update(match=match, view=view, rep=rep, seed=seed)
    audit.write(dest, out)
    print("DONE", dest.name, flush=True)
    return str(dest)


def run_cells():
    jobs = [(m, v, r) for m in NEW for v, reps in VIEWS.items() for r in reps]
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(_cell_job, jobs))


# ---------------------------------------------------------------- J03WOY (existing match)

def _woy_job(arg):
    view, rep, name = arg
    dest = OUT / "j03woy" / f"{view}__{rep}.json"
    if dest.exists():
        return str(dest)
    cached = joblib.load(CACHED / f"{name}.joblib")
    labels = np.asarray(cached["labels"]).astype(str)
    out = score_arms(cached["inputs"], labels, cached["record"])
    out.update(match="J03WOY", view=view, rep=rep, source=name)
    audit.write(dest, out)
    print("DONE", dest.name, flush=True)
    return str(dest)


def run_j03woy():
    jobs = [("clean", 1, "J03WOY_clean_clean"), ("v1", 1, "J03WOY_feed_like_corrupt_1"),
            ("v1", 2, "J03WOY_feed_like_corrupt_2")]
    with ProcessPoolExecutor(WORKERS) as ex:
        list(ex.map(_woy_job, jobs))


# ---------------------------------------------------------------- sanity

def run_sanity():
    out = {}
    cached = joblib.load(CACHED / "J03WOY_clean_clean.joblib")
    inputs, labels = ms.view(RAW / "J03WOY.npz")
    record = prep.build_record(inputs, labels, PROVIDER, "J03WOY")
    out["j03woy_record_x_equal"] = bool(np.array_equal(record["x"], cached["record"]["x"], equal_nan=True))
    out["j03woy_record_x_maxabs_diff"] = float(np.nanmax(np.abs(record["x"] - cached["record"]["x"])))
    out["j03woy_record_shape"] = list(record["x"].shape)
    out["j03woy_indices_equal"] = bool(np.array_equal(record["indices"], cached["record"]["indices"]))
    out["j03woy_labels_equal"] = bool(np.array_equal(np.asarray(labels).astype(str),
                                                     np.asarray(cached["labels"]).astype(str)))
    # cand phase0 on the cached J03WOY clean and the two feed views
    expected = audit.read(RESULTS / "feed_benchmark_audit_v1.json")["tta"]["cand_phase0"]["spell"]
    rows = {}
    for view, name in (("clean", "J03WOY_clean_clean"), ("feed_like_corrupt_1", "J03WOY_feed_like_corrupt_1"),
                       ("feed_like_corrupt_2", "J03WOY_feed_like_corrupt_2")):
        c = joblib.load(CACHED / f"{name}.joblib")
        cell = dict(labels=np.asarray(c["labels"]).astype(str), period=c["inputs"]["period"],
                    timestamps=c["inputs"]["timestamps"], fps=float(c["inputs"]["fps"]),
                    model_period=supervised.model_periods(c["inputs"]))
        p = audit.predict(c["inputs"], c["record"], "cand", PROVIDER)
        rows[view] = audit.decode(p, cell["model_period"], cell["fps"])
        rows[view] = audit.score(cell, rows[view])["spell_f1"]
    out["cand_phase0_spell_f1"] = rows
    out["cand_phase0_feed_mean"] = float(np.mean([rows["feed_like_corrupt_1"], rows["feed_like_corrupt_2"]]))
    out["expected_clean_Sportec"] = expected["clean"]["providers"]["Sportec"]
    out["expected_feed_Sportec"] = expected["feed_like"]["providers"]["Sportec"]
    out["clean_abs_diff"] = abs(rows["clean"] - out["expected_clean_Sportec"])
    out["feed_abs_diff"] = abs(out["cand_phase0_feed_mean"] - out["expected_feed_Sportec"])
    for m in NEW:
        inputs, labels = ms.view(RAW / f"{m}.npz")
        graded = np.asarray(labels).astype(str) != "X"
        out[f"{m}_frames"] = int(len(labels))
        out[f"{m}_graded_frames"] = int(graded.sum())
        out[f"{m}_minutes_at_fps"] = float(len(labels) / float(inputs["fps"]) / 60)
        out[f"{m}_fps"] = float(inputs["fps"])
        out[f"{m}_ball_native_missing"] = float((~np.isfinite(inputs["ball"]).all(axis=1)).mean())
    audit.write(OUT / "sanity.json", out)
    print(audit.json.dumps(out, indent=1, sort_keys=True), flush=True)
    return out


# ---------------------------------------------------------------- aggregation

def _mean(xs):
    return float(np.mean(xs))


def _woy_existing():
    """J03WOY numbers from existing results: {(arm, mode, view): {spell_f1, macro_f1, source}}."""
    tta = audit.read(RESULTS / "feed_benchmark_audit_v1.json")["tta"]
    out = {}
    for arm in ("cand", "adop"):
        for mode in ("phase0", "phase_tta"):
            t = tta[f"{arm}_{mode}"]
            for view, key in (("clean", "clean"), ("v1", "feed_like")):
                out[(arm, mode, view)] = dict(spell_f1=t["spell"][key]["providers"]["Sportec"],
                                              macro_f1=t["macro"][key]["providers"]["Sportec"],
                                              source="feed_benchmark_audit_v1.tta")
    rows = [audit.read(V2_SCORES / f"J03WOY__{r}.json") for r in (1, 2, 3, 4)]
    for arm in ARMS:
        for mode, suffix in (("phase0", ""), ("phase_tta", "_phase_tta")):
            key = arm + suffix
            if all(key in r for r in rows):
                out[(arm, mode, "feed_like_v2")] = dict(
                    spell_f1=_mean([r[key]["spell_f1"] for r in rows]),
                    macro_f1=_mean([r[key]["macro_f1"] for r in rows]),
                    d_f1=_mean([r[key]["d_f1"] for r in rows]),
                    source="feed_like_v2_benchmark/scores/J03WOY__{1..4}")
    return out


def _view_mean(rows, key, metric):
    return _mean([r[key][metric] for r in rows])


def run_aggregate():
    cells = [audit.read(p) for p in sorted((OUT / "cells").glob("*.json"))]
    woy = [audit.read(p) for p in sorted((OUT / "j03woy").glob("*.json"))]
    existing = _woy_existing()
    expected_cells = sum(len(r) for r in VIEWS.values()) * len(NEW)
    if len(cells) != expected_cells or len(woy) != 3:
        raise ValueError(f"Incomplete cells: {len(cells)}/{expected_cells}, J03WOY {len(woy)}/3")

    per_match, woy_recomputed, woy_vs_existing = {}, {}, {}
    out_arms = {}
    for arm in ARMS:
        for mode in ("phase0", "phase_tta"):
            key = f"{arm}_{mode}"
            for view in VIEWS:
                entry = {}
                for match in NEW:
                    rows = [c for c in cells if c["match"] == match and c["view"] == view]
                    entry[match] = {m: _view_mean(rows, key, m) for m in ("spell_f1", "macro_f1", "d_f1")}
                    entry[match]["n_realizations"] = len(rows)
                    entry[match]["spell_f1_per_realization"] = [r[key]["spell_f1"] for r in sorted(rows, key=lambda r: r["rep"])]
                    entry[match]["predicted_spell_count"] = _mean([r[key]["predicted_spell_count"] for r in rows])
                    entry[match]["reference_spell_count"] = _mean([r[key]["reference_spell_count"] for r in rows])
                mean_new = {m: _mean([entry[x][m] for x in NEW]) for m in ("spell_f1", "macro_f1", "d_f1")}
                # J03WOY: existing results first, else recomputed cached-input values (cand/adop checked)
                ex = existing.get((arm, mode, view))
                if view in ("clean", "v1"):
                    rows = [w for w in woy if w["view"] == view]
                    rec = {m: _view_mean(rows, key, m) for m in ("spell_f1", "macro_f1", "d_f1")}
                    woy_recomputed[f"{arm}_{mode}_{view}"] = rec
                    if ex:
                        woy_vs_existing[f"{arm}_{mode}_{view}"] = dict(
                            existing=ex["spell_f1"], recomputed=rec["spell_f1"], abs_diff=abs(ex["spell_f1"] - rec["spell_f1"]))
                    else:
                        ex = dict(rec, source="recomputed from seed_gap_context_v1 cached J03WOY inputs")
                if ex is None:
                    raise ValueError(f"No J03WOY value for {arm} {mode} {view}")
                three = {m: _mean([entry[x][m] for x in NEW] + [ex[m]]) for m in ("spell_f1", "macro_f1") }
                out_arms[f"{arm}__{mode}__{view}"] = dict(
                    arm=arm, mode=mode, view=view, per_new_match=entry, mean_two_new_matches=mean_new,
                    J03WOY={k: v for k, v in ex.items() if k in ("spell_f1", "macro_f1", "d_f1", "source")},
                    mean_three_sportec_matches=three)

    # paired deltas per match and view type
    deltas = {}
    for view in VIEWS:
        for mode in ("phase0", "phase_tta"):
            for a, b in (("cand", "adop"), ("raw", "adop"), ("v2train", "adop"), ("v2train", "cand")):
                for metric in ("spell_f1", "macro_f1"):
                    d = {m: out_arms[f"{a}__{mode}__{view}"]["per_new_match"][m][metric]
                         - out_arms[f"{b}__{mode}__{view}"]["per_new_match"][m][metric] for m in NEW}
                    d["J03WOY"] = out_arms[f"{a}__{mode}__{view}"]["J03WOY"][metric] - out_arms[f"{b}__{mode}__{view}"]["J03WOY"][metric]
                    d["mean_two_new"] = _mean([d[m] for m in NEW])
                    d["mean_three"] = _mean([d[m] for m in NEW] + [d["J03WOY"]])
                    deltas[f"{a}_minus_{b}__{mode}__{view}__{metric}"] = d
        for arm in ARMS:
            for metric in ("spell_f1", "macro_f1"):
                d = {m: out_arms[f"{arm}__phase_tta__{view}"]["per_new_match"][m][metric]
                     - out_arms[f"{arm}__phase0__{view}"]["per_new_match"][m][metric] for m in NEW}
                d["J03WOY"] = out_arms[f"{arm}__phase_tta__{view}"]["J03WOY"][metric] - out_arms[f"{arm}__phase0__{view}"]["J03WOY"][metric]
                d["mean_two_new"] = _mean([d[m] for m in NEW])
                d["mean_three"] = _mean([d[m] for m in NEW] + [d["J03WOY"]])
                deltas[f"tta_minus_phase0__{arm}__{view}__{metric}"] = d

    sizes = {m: {k: v for k, v in audit.read(OUT / "sanity.json").items() if k.startswith(m)} for m in NEW}
    audit.write(RESULT, dict(
        study=STUDY, organizer_reference_access=False, public_scoring=False, fitting=False, promotion=False,
        frame_arrays_published=False, matches_new=list(NEW), arms=list(ARMS), views={k: list(v) for k, v in VIEWS.items()},
        seeds="int(sha256(f'sportec_extra_eval_v1/{view}/{rep}/{match}')[:8],16)",
        sanity=audit.read(OUT / "sanity.json"), match_sizes=sizes,
        notes=("J03WOY v1 clean/v1 numbers come from feed_benchmark_audit_v1.tta (cached feed_like_corrupt views); new-match v1 views "
               "are fresh feed_robustness_v1.raw_view realizations, so v1 is not the identical generator. v2 J03WOY from "
               "feed_like_v2_benchmark scores. Where J03WOY values were missing they were computed from cached inputs."),
        scores=out_arms, deltas=deltas, j03woy_recomputed=woy_recomputed, j03woy_recomputed_vs_existing=woy_vs_existing))


STEPS = dict(sanity=run_sanity, j03woy=run_j03woy, cells=run_cells, aggregate=run_aggregate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=(*STEPS, "all"))
    step = parser.parse_args().step
    for name in (STEPS if step == "all" else (step,)):
        print("STEP", name, flush=True)
        STEPS[name]()
