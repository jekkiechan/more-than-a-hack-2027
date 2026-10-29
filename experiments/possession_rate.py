"""Label-free possession-structure diagnostic (possession_rate_v1).

Question: does the adopted model's possession structure on the competition
match look like its clean-holdout behaviour, its feed_like-stress behaviour,
or true provider possession rates?  No model, decoder or gate is changed, and
no competition ground truth is read (the competition ``labels`` key in
``inputs.npz`` is never loaded).

``structure_stats`` is the one fixed statistic set used everywhere.

Definitions (frames are counted only where ``active_mask`` is true):
  * possession: maximal run of one label in {H, A}.  A run ends at D, X, an
    inactive frame, a period change or a direct H<->A switch (which starts a
    new possession).
  * D-stoppage: maximal run of D.
  * direct switch: adjacent active frames H->A or A->H with no D between.
  * ball-visible / ball-missing attribution: an event (possession start,
    D-stoppage start or switch) belongs to the visibility of the first frame
    of the run it starts; rates divide by active minutes with that visibility.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_n, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/possession_rate_v1"
RESULT = ROOT / "experiments/results/possession_rate.json"
DECLARATION = OUT / "declaration.json"
QUANTILES = (10, 25, 50, 75, 90)
SHORT_SECONDS = 3.0
# Statistics used for the standardized-distance comparison (team-identity
# dependent shares H/A are excluded; D share kept).
VECTOR = ("poss_per_min", "d_stop_per_min", "dur_q10", "dur_q25", "dur_q50", "dur_q75",
          "dur_q90", "dur_mean", "frac_lt3s", "switch_per_min", "share_D")
CODE = {"H": 1, "A": 2, "D": 3}


def _rate(count, minutes):
    return float(count / minutes) if minutes > 0 else math.nan


def structure_stats(labels, active_mask, ball_visible_mask, fps, period=None):
    """Label-free possession-structure statistics for one frame-label sequence."""
    labels = np.asarray(labels)
    active = np.asarray(active_mask, dtype=bool)
    visible = np.asarray(ball_visible_mask, dtype=bool)
    if not (len(labels) == len(active) == len(visible)):
        raise ValueError("labels, active_mask and ball_visible_mask must have equal length")
    fps = float(fps)
    code = np.zeros(len(labels), dtype=np.int8)
    for name, value in CODE.items():
        code[labels == name] = value
    code[~active] = 0
    n_active = int(active.sum())
    minutes = n_active / fps / 60
    vis_minutes = int((active & visible).sum()) / fps / 60
    mis_minutes = int((active & ~visible).sum()) / fps / 60
    out = dict(n_active_frames=n_active, active_minutes=minutes,
               ball_visible_fraction=float((active & visible).sum() / n_active) if n_active else math.nan)
    for name in "HAD":
        out[f"share_{name}"] = float(((code == CODE[name]).sum()) / n_active) if n_active else math.nan
    # Run boundaries: label code change, or period change (adjacent periods).
    if len(code):
        change = code[1:] != code[:-1]
        if period is not None:
            change |= np.asarray(period)[1:] != np.asarray(period)[:-1]
        starts = np.r_[0, np.flatnonzero(change) + 1]
        stops = np.r_[starts[1:], len(code)]
    else:
        starts = stops = np.zeros(0, dtype=int)
    run_code = code[starts] if len(starts) else code[:0]
    is_poss = (run_code == 1) | (run_code == 2)
    is_d = run_code == 3
    durations = (stops - starts)[is_poss] / fps
    run_visible = visible[starts] if len(starts) else visible[:0]
    # A switch is a possession run whose immediately preceding run (contiguous,
    # same period) is a possession of the other team.
    prev_code = np.r_[0, run_code[:-1]] if len(run_code) else run_code
    switch = is_poss & (prev_code > 0) & (prev_code < 3) & (prev_code != run_code)
    if period is not None and len(starts):
        p = np.asarray(period)
        same_period = np.r_[False, p[starts[1:]] == p[starts[1:] - 1]]
        switch &= same_period
    out.update(n_poss=int(is_poss.sum()), n_d_stop=int(is_d.sum()), n_switch=int(switch.sum()),
               poss_per_min=_rate(is_poss.sum(), minutes), d_stop_per_min=_rate(is_d.sum(), minutes),
               switch_per_min=_rate(switch.sum(), minutes))
    for q in QUANTILES:
        out[f"dur_q{q}"] = float(np.percentile(durations, q)) if len(durations) else math.nan
    out["dur_mean"] = float(durations.mean()) if len(durations) else math.nan
    out["frac_lt3s"] = float((durations < SHORT_SECONDS).mean()) if len(durations) else math.nan
    for tag, mask, mins in (("visible", run_visible, vis_minutes), ("missing", ~run_visible, mis_minutes)):
        out[f"poss_per_min_{tag}"] = _rate((is_poss & mask).sum(), mins)
        out[f"d_stop_per_min_{tag}"] = _rate((is_d & mask).sum(), mins)
        out[f"switch_per_min_{tag}"] = _rate((switch & mask).sum(), mins)
    a, b = out["poss_per_min_missing"], out["poss_per_min_visible"]
    out["poss_ratio_missing_over_visible"] = float(a / b) if b and not math.isnan(a) and b > 0 else math.nan
    return out


def clean_json(value):
    """NaN -> None so aggregate files are strict JSON."""
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def ratio(pred, true):
    return float(pred / true) if true and math.isfinite(pred) and math.isfinite(true) and true != 0 else math.nan


def standardized_distance(vector, reference_mean, scale):
    """RMS of per-stat z-differences between a stat vector and a reference mean."""
    z = [(vector[k] - reference_mean[k]) / scale[k] for k in VECTOR
         if math.isfinite(vector[k]) and math.isfinite(reference_mean[k]) and scale[k] > 0]
    return float(math.sqrt(np.mean(np.square(z)))) if z else math.nan


# --------------------------------------------------------------------------- #
# Everything below needs the repository data and is exercised only via the CLI.
# --------------------------------------------------------------------------- #
def _lazy():
    sys.path.insert(0, str(ROOT))
    from experiments import (dead_transfer as dt, multisource_eval as ms, supervised, learned_temporal as lt,
                             stress_components as sc, corruption_experiment as ce)
    return dt, ms, supervised, lt, sc, ce


FEED = ("ball_gaps", "coordinates", "slot_gaps", "id_swaps")
TRAIN_MATCHES = None


def provider_of(stem):
    return "sportec" if stem.startswith("J03") else "metrica" if stem.startswith("metrica") else "skillcorner"


def available_matches():
    """(stem, path, role) for every external match that has H/A/D labels."""
    dt, *_ = _lazy()
    train = {p.stem for s in dt.SPLITS.values() for p in s["train"]}
    evals = {p.stem for s in dt.SPLITS.values() for p in s["eval"]}
    paths = sorted(dt.EXTERNAL.glob("metrica_*.npz")) + sorted(dt.EXTERNAL.glob("skillcorner_had_*.npz")) \
        + sorted(dt.SPORTEC.glob("J03*.npz"))
    return [(p.stem, p, "eval" if p.stem in evals else "train" if p.stem in train else "other") for p in paths]


def active_of(inputs, labels):
    return (np.asarray(inputs["period"]) > 0) & (labels != "X")


def visible_of(inputs):
    return np.isfinite(np.asarray(inputs["ball"])).all(axis=1)


def match_stats(path, role_labels=None):
    dt, ms, *_ = _lazy()
    inputs, labels = ms.view(path)
    return structure_stats(labels, active_of(inputs, labels), visible_of(inputs), float(inputs["fps"]),
                           inputs["period"]), inputs, labels


def aggregate(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])], dtype=float)
        out[k] = dict(min=float(v.min()), max=float(v.max()), mean=float(v.mean()), n=int(len(v))) if len(v) else None
    return out


KEYS = VECTOR + ("share_H", "share_A", "ball_visible_fraction", "switch_per_min_visible", "switch_per_min_missing",
                 "poss_per_min_visible", "poss_per_min_missing", "poss_ratio_missing_over_visible")


def cmd_truth():
    rows = []
    for stem, path, role in available_matches():
        s, _, _ = match_stats(path)
        rows.append(dict(match=stem, provider=provider_of(stem), role=role, **s))
        print("TRUTH", stem, role, round(s["poss_per_min"], 3), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "truth_rows.json").write_text(json.dumps(clean_json(rows), indent=1) + "\n")


def _load_probability(source, stem, view, rep, seeds=(2026, 2027)):
    base = ROOT / "data/experiments/objective_alignment_v1"
    ps = []
    for seed in seeds:
        with np.load(base / f"{source}_{stem}_{view}_{rep}_combined_{seed}.npz") as d:
            ps.append(d["probability"])
    return 0.5 * (ps[0] + ps[1])


def cmd_holdout():
    dt, ms, supervised, lt, sc, ce = _lazy()
    calibration = json.loads((ROOT / "data/experiments/objective_alignment_v1/predeclared.json").read_text())["calibration"]
    rows, probs = [], {}
    for source, split in dt.SPLITS.items():
        for path in split["eval"]:
            inputs, labels = ms.view(path)
            active = active_of(inputs, labels)
            period = inputs["period"]
            fps = float(inputs["fps"])
            true = structure_stats(labels, active, visible_of(inputs), fps, period)
            for view, reps in (("clean", ("clean",)), ("feed_like", ("corrupt_1", "corrupt_2"))):
                for rep in reps:
                    tracking = inputs if view == "clean" else sc.corrupt(inputs, calibration, ce.seed(path.stem, rep), FEED)
                    p = _load_probability(source, path.stem, view, rep)
                    pred = supervised.decode(p, supervised.model_periods(inputs), fps, lt.COST)
                    s = structure_stats(pred, active, visible_of(tracking), fps, period)
                    sc_ = supervised.score(dict(inputs, labels=labels), pred)
                    rows.append(dict(source=source, match=path.stem, view=view, replicate=rep, stats=s, true=true,
                                     spell_f1=float(sc_["spell_f1_mean"]), n_pred_spells=sc_["n_pred_spells"],
                                     n_gt_spells=sc_["n_gt_spells"]))
                    print("HOLDOUT", path.stem, view, rep, round(s["poss_per_min"], 3), "true", round(true["poss_per_min"], 3),
                          "spell", round(sc_["spell_f1_mean"], 4), flush=True)
    (OUT / "holdout_rows.json").write_text(json.dumps(clean_json(rows), indent=1) + "\n")


def cmd_costs():
    dt, ms, supervised, lt, sc, ce = _lazy()
    costs = (2, 3, 4, 6, 8)
    rows = []
    for source, split in dt.SPLITS.items():
        for path in split["eval"]:
            inputs, labels = ms.view(path)
            fps = float(inputs["fps"])
            for rep in ("corrupt_1", "corrupt_2"):
                p = _load_probability(source, path.stem, "feed_like", rep)
                for cost in costs:
                    pred = supervised.decode(p, supervised.model_periods(inputs), fps, cost)
                    m = supervised.score(dict(inputs, labels=labels), pred)
                    s = structure_stats(pred, active_of(inputs, labels), np.ones(len(labels), bool), fps, inputs["period"])
                    rows.append(dict(source=source, match=path.stem, replicate=rep, cost=cost,
                                     spell_f1=float(m["spell_f1_mean"]), poss_per_min=s["poss_per_min"],
                                     frac_lt3s=s["frac_lt3s"], n_pred_spells=m["n_pred_spells"]))
                    print("COST", path.stem, rep, cost, round(m["spell_f1_mean"], 4), flush=True)
    (OUT / "cost_holdout_rows.json").write_text(json.dumps(clean_json(rows), indent=1) + "\n")


def competition_context():
    with np.load(ROOT / "data/challenge/inputs.npz") as d:  # allowlist: never touches 'labels'
        data = {k: d[k] for k in ("ball", "period", "fps", "timestamps")}
    return data


def read_prediction(csv_path):
    arr = np.genfromtxt(csv_path, delimiter=",", skip_header=1, dtype=str)
    return arr[:, 1]


def cmd_competition(names):
    data = competition_context()
    active = np.asarray(data["period"]) > 0
    visible = visible_of(data)
    rows = {}
    for name, path in names.items():
        labels = read_prediction(path)
        if len(labels) != len(active):
            raise ValueError("prediction length mismatch")
        rows[name] = structure_stats(labels, active, visible, float(data["fps"]), data["period"])
        print("COMP", name, round(rows[name]["poss_per_min"], 3), flush=True)
    (OUT / "competition_rows.json").write_text(json.dumps(clean_json(rows), indent=1) + "\n")


def cmd_costs_competition():
    """Label-free competition possessions/min per switch cost from the saved-model
    probabilities is not available (no saved competition probabilities); see report."""
    raise SystemExit("not implemented: competition probabilities are not saved")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["truth", "holdout", "costs", "competition"])
    ap.add_argument("--adopted", default=str(ROOT / "data/experiments/current_baseline/joint_gru_combined_ensemble.csv"))
    ap.add_argument("--previous", default=str(OUT / "seed_ensemble.csv"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"truth": cmd_truth, "holdout": cmd_holdout, "costs": cmd_costs}.get(a.command, lambda: None)() \
        if a.command != "competition" else cmd_competition({"adopted_combined_ensemble": a.adopted,
                                                            "previous_seed_ensemble": a.previous})
