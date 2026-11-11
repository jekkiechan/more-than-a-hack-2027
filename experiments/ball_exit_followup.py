"""Bounded crossing-time comparison and exact Fable replay; no fitting."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import ball_exit_crossing as crossing, ball_exit_dmask_v1 as fable
from experiments import feed_like_v2, sc2020_realfeed_common as common
from experiments import sc2020_realfeed_score as score, frontier_postprocess_v1 as fp
from experiments import tracking_corruption, supervised

ROOT = Path(__file__).resolve().parents[1]
STUDY = "ball_exit_crossing_v1"
DECL = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
ARMS = ("frontier", "snap", "linear", "midpoint")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def raw_path(match, view, rep):
    name = common.base.audit.FILE[match]
    suffix = f"feed_like_corrupt_{rep}" if view == "feed_like_v1" else "clean_clean"
    return Path(common.os.environ["MTAH_DATA_DIR"]) / "experiments/seed_gap_context_v1" / f"{name}_{suffix}.joblib"


def load_inputs(match, view, rep):
    cached = joblib.load(raw_path(match, view, rep))
    native = tracking_corruption.inputs_only(cached["inputs"])
    if view == "feed_like_v2":
        calibration = json.loads(common.base.CALIBRATION.read_text())["calibration"]
        seed = int(hashlib.sha256(f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8], 16)
        native = feed_like_v2.corrupt(native, cached["labels"], calibration, seed)
    return native


def declare():
    # Restore receipt binds the raw native inputs used in Fable's study.
    expected = {Path(r["path"]).name: r["sha256"] for r in json.loads(
        (ROOT / "experiments/results/frontier_reconstruction_v1_restore.json").read_text())["assets"]
        if r["category"] == "seed_gap_context_clean_and_v1_eval_inputs"}
    facts = {}
    for provider, match, view, rep in common.cells():
        path = raw_path(match, view, rep)
        if sha(path) != expected[path.name]:
            raise ValueError(f"Restored input hash mismatch: {path.name}")
        native = common.FRONTIER / "cells" / f"{common.cell_key(provider, match, view, rep)}.npz"
        facts[common.cell_key(provider, match, view, rep)] = {
            "input": str(path), "input_sha256": sha(path),
            "native": str(native), "native_sha256": sha(native)}
    sources = [Path(__file__), Path(crossing.__file__), Path(fable.__file__),
               Path(score.__file__), Path(fp.__file__), Path(feed_like_v2.__file__),
               ROOT / "experiments/ball_exit_runtime.py", common.base.CALIBRATION]
    write_once(DECL, {
        "study": STUDY, "arms": ARMS, "question": "Does a bracketed crossing time improve snap's observed-exit boundary under missing observations?",
        "fitting": False, "organizer_reference_access": False, "gar_access": False,
        "exposure": "Same five exposed development matches; synthetic v2 is label-conditioned and not fresh evidence.",
        "authorization": "User requested promotion and further improvement of Fable's idea on 2026-11-11; this new bounded study follows the prior stop rule.",
        "methods": {"linear": "First straight-line intersection with pitch boundary, rounded up to a frame", "midpoint": "Bracket midpoint rounded up; geometry ablation"},
        "maximum_bracket_s": fable.WINDOW_S, "cohort": facts,
        "aggregation": "Mean replicas within match, matches within provider, providers equally",
        "retention": "Fable guards against snap: v2 spell strictly higher, clean/v1 spell and all short recalls not lower, macro within .001, provider clean macro/D-F1 within .02",
        "selection": "Eligible v2 spell descending, macro tie-break; otherwise retain unchanged snap. Freeze before any GAR/public check.",
        "control_tolerance": 1e-12, "budget": "Two extensions, 35 cells each; no follow-on threshold, margin, provider or time-window sweep",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources}})


def job(key, controls_only=False):
    provider, match, view, rep = key.split("__")
    facts = json.loads(DECL.read_text())["cohort"][key]
    for path, digest in ((facts["input"], facts["input_sha256"]), (facts["native"], facts["native_sha256"])):
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {key}")
    cell = common.frontier_cell(key)
    inputs = load_inputs(match, view, int(rep))
    if not np.array_equal(supervised.model_periods(inputs), cell["model_period"]):
        raise ValueError(f"Model blocks differ: {key}")
    for field in ("period", "timestamps"):
        np.testing.assert_array_equal(inputs[field], cell[field])
    inside, outside = fable.ball_state(inputs["ball"], inputs["pitch"], 0.)
    dead = cell["decoded__predecessor"] == "D"
    snap = fable.edit_mask(dead, inside, outside, cell["model_period"], float(cell["fps"]), "snap")
    masks = {"frontier": dead, "snap": snap}
    predictions, rows = {}, {}
    saved = json.loads((ROOT / "experiments/results/ball_exit_dmask_v1_round2.json").read_text())["cells"][key]
    for name in ARMS[:2] if controls_only else ARMS:
        if name in ARMS[2:]:
            masks[name] = crossing.refine_start(snap, inputs["ball"], inputs["pitch"], cell["model_period"], float(cell["fps"]), name)
        mask = masks[name]
        pred = fp.decode_candidate(cell["prob__approved_baseline"], np.where(mask, "D", "H"), cell["model_period"], float(cell["fps"]))
        predictions[name] = pred
        metrics = fp._score_cell(dict(cell, fps=float(cell["fps"])), pred)
        rows[name] = {m: float(metrics[m]) for m in score.METRICS}
        rows[name]["changed_fraction_vs_snap"] = float(np.mean(mask != snap))
        if name in ARMS[:2]:
            parent = "control" if name == "frontier" else "snap_m0.0"
            for metric in score.METRICS:
                if abs(rows[name][metric] - saved[parent][metric]) > 1e-12:
                    raise ValueError(f"Fable replay mismatch: {key}/{name}/{metric}")
    if controls_only:
        return rows
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{key}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **predictions)
    return key, rows, {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def run(workers):
    declaration = json.loads(DECL.read_text())
    if RESULT.exists() or any(OUT.glob("*.npz")):
        raise FileExistsError("Preserve the closed or partial run")
    for rel, digest in declaration["source_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError(f"Frozen source changed: {rel}")
    pilot = next(iter(declaration["cohort"]))
    write_once(OUT / "preflight.json", {"cell": pilot, "controls": job(pilot, controls_only=True),
                                       "declaration_sha256": sha(DECL)})
    with ProcessPoolExecutor(workers) as pool:
        results = list(pool.map(job, declaration["cohort"]))
    rows = {key: row for key, row, _ in results}
    aggregate = {name: score.aggregate(rows, name) for name in ARMS}
    guards = {name: fable.guards(aggregate[name], aggregate["snap"]) for name in ARMS[2:]}
    eligible = [name for name, failed in guards.items() if not failed]
    selected = max(eligible, key=lambda n: (aggregate[n]["feed_like_v2"]["spell_f1"], aggregate[n]["feed_like_v2"]["macro_f1"])) if eligible else "snap"
    write_once(RESULT, {"study": STUDY, "declaration_sha256": sha(DECL), "control_replay_exact": True,
        "aggregate": aggregate, "failed_guards": guards, "selected": selected, "cells": rows,
        "prediction_artifacts": {key: artifact for key, _, artifact in results},
        "organizer_reference_access": False, "gar_access": False})
    print(score.table(rows, ARMS, "snap"), flush=True)
    print("Selected:", selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "run"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    declare() if args.action == "declare" else run(args.workers)
