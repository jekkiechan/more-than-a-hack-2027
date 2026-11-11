"""One fixed non-overlapping composition of snap D edits and turnover H/A."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from experiments import ball_exit_complement as first
from experiments import ball_exit_followup as shared, ball_exit_dmask_v1 as fable
from experiments import supervised

ROOT = shared.ROOT
STUDY = "ball_exit_turnover_disjoint_v1"
DECL = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
CONTROL = "snap"
ARMS = (CONTROL, "fresh", "hard", "mix075")


def disjoint(snap, complement, original, period):
    """Use complement only in identical original/snap live runs, by block.

    An interval's two endpoints must match exactly; no labels, provider ID,
    scores, confidence thresholds or fitted parameters enter this choice.
    """
    snap, complement, original = map(np.asarray, (snap, complement, original))
    if snap.shape != complement.shape or snap.shape != original.shape or snap.shape != np.asarray(period).shape:
        raise ValueError("Predictions and blocks must align")
    if not np.array_equal(snap == "D", complement == "D"):
        raise ValueError("Complement must already use snap's exact D mask")
    result = snap.copy()
    for start, stop in supervised.blocks(np.asarray(period)):
        old_runs = set(fable._runs(original[start:stop] != "D"))
        for a, b in fable._runs(snap[start:stop] != "D"):
            if (a, b) in old_runs:
                result[start+a:start+b] = complement[start+a:start+b]
    return result


def declare():
    prior = first.read(first.RESULT)
    previous_declaration = first.read(first.DECL)
    source = [Path(__file__), Path(first.__file__), Path(fable.__file__), Path(supervised.__file__)]
    cells = {}
    for key, artifact in prior["prediction_artifacts"].items():
        cells[key] = {"frontier": previous_declaration["cells"][key]["frontier"],
                      "composition": {"path": str(ROOT / artifact["path"]), "sha256": artifact["sha256"]}}
    shared.write_once(DECL, {"study": STUDY, "arms": ARMS, "control": CONTROL, "cells": cells,
        "question": "Can separating the two mechanisms retain the H/A gain without disturbing live intervals whose boundary snap corrected?",
        "post_result_design": True, "motivation": "Simple H/A compositions raised v2 spell but lost clean/short recall; original-mask gains did not fully transfer after snap changed live intervals",
        "rule": "Use the already-declared snap+H/A predictions only for live intervals with exactly the same start/end in original and snap D masks within a model block; retain snap frontier everywhere else",
        "budget": "Three existing H/A endpoints, one fixed gating rule; no fitted parameters or further weight/window/provider search",
        "retention": previous_declaration["retention"], "aggregation": previous_declaration["aggregation"],
        "ranking": previous_declaration["ranking"], "exposure": previous_declaration["exposure"],
        "source_sha256": {str(p.relative_to(ROOT)): shared.sha(p) for p in source},
        "parent_result_sha256": shared.sha(first.RESULT), "fitting": False,
        "gar_access": False, "organizer_reference_access": False})


def job(key):
    facts = first.read(DECL)["cells"][key]
    for fact in facts.values():
        if shared.sha(fact["path"]) != fact["sha256"]:
            raise ValueError("Frozen input changed: " + key)
    cell = first.common.frontier_cell(key)
    rows, predictions = {}, {}
    with np.load(facts["composition"]["path"], allow_pickle=False) as saved:
        snap, original = saved[first.CONTROL], saved["original__frontier"]
        predictions[CONTROL] = snap
        for arm in ARMS[1:]:
            predictions[arm] = disjoint(snap, saved[f"snap__{arm}"], original, cell["model_period"])
    for arm, pred in predictions.items():
        rows[arm] = first.measure(cell, pred)
        rows[arm]["changed_fraction_vs_snap"] = float(np.mean(pred != predictions[CONTROL]))
    expected = first.read(first.RESULT)["cells"][key][first.CONTROL]
    for metric in first.scoring.METRICS:
        if abs(rows[CONTROL][metric] - expected[metric]) > 1e-10:
            raise ValueError("Snap replay failed")
    path = OUT / f"{key}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **predictions)
    return key, rows, {"path": str(path.relative_to(ROOT)), "sha256": shared.sha(path)}


def run(workers):
    declaration = first.read(DECL)
    if RESULT.exists() or (OUT.exists() and any(OUT.iterdir())):
        raise FileExistsError("Preserve previous run")
    if shared.sha(first.RESULT) != declaration["parent_result_sha256"]:
        raise ValueError("Parent result changed")
    for rel, digest in declaration["source_sha256"].items():
        if shared.sha(ROOT / rel) != digest:
            raise ValueError("Frozen source changed")
    # Before new candidate scoring, require independently replayed parent outputs.
    verification_path = first.RESULT.with_name(first.RESULT.stem + "_verification.json")
    verification = first.read(verification_path)
    if not verification["all_passed"] or verification["result_sha256"] != shared.sha(first.RESULT):
        raise ValueError("Independent parent replay is required")
    OUT.mkdir(parents=True)
    with ProcessPoolExecutor(workers) as pool:
        complete = list(pool.map(job, declaration["cells"]))
    rows = {key: row for key, row, _ in complete}
    aggregate = {arm: first.scoring.aggregate(rows, arm) for arm in ARMS}
    failed = {arm: fable.guards(aggregate[arm], aggregate[CONTROL]) for arm in ARMS[1:]}
    eligible = [arm for arm, checks in failed.items() if not checks]
    selected = max(eligible, key=lambda a: (aggregate[a]["feed_like_v2"]["spell_f1"], aggregate[a]["feed_like_v2"]["macro_f1"])) if eligible else CONTROL
    shared.write_once(RESULT, {"study": STUDY, "declaration_sha256": shared.sha(DECL),
        "parent_verification_sha256": shared.sha(verification_path), "aggregate": aggregate,
        "failed_guards": failed, "selected": selected, "cells": rows,
        "prediction_artifacts": {key: fact for key, _, fact in complete},
        "fitting": False, "gar_access": False, "organizer_reference_access": False})
    print(first.scoring.table(rows, ARMS, CONTROL), flush=True)
    print("Selected:", selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "run"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    declare() if args.action == "declare" else run(args.workers)
