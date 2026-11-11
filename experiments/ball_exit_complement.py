"""Factorial composition: frozen ball-exit D correction × archived turnover H/A."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from experiments import ball_exit_followup as previous
from experiments import ball_exit_dmask_v1 as fable
from experiments import sc2020_realfeed_common as common
from experiments import sc2020_realfeed_score as scoring
from experiments import frontier_postprocess_v1 as decode

ROOT = previous.ROOT
ARCHIVE = ROOT.parent / "turnover"
STUDY = "ball_exit_turnover_complement_v1"
DECL = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
TEAMS = ("frontier", "fresh", "hard", "mix075")
ARMS = tuple(f"{mask}__{team}" for mask in ("original", "snap") for team in TEAMS)
CONTROL = "snap__frontier"


def read(path):
    return json.loads(Path(path).read_text())


def probability_paths(key):
    return {"turnover": ARCHIVE / "data/experiments/turnover_supervision_v1/eval/cells" / f"{key}.npz",
            "mixture": ARCHIVE / "data/experiments/turnover_supervision_mix_v1/cells" / f"{key}.npz"}


def declare():
    # Freeze expected remote-member hashes before new scoring. Actual files are
    # checked after download, on every read; an archive cannot redefine the arms.
    receipt = read(ARCHIVE / "turnover_supervision_v1_receipt.json")
    if receipt["archive_sha256"] != "d3a02cc46e8d48218f6ffe56f7dfa30c1dc9317a880b6a6a54adbb12bb074c6d":
        raise ValueError("Unexpected turnover archive")
    members = {row["path"]: row for row in receipt["members"]}
    prior = read(previous.RESULT)
    cells = {}
    for key, fact in read(previous.DECL)["cohort"].items():
        row = {"frontier": {"path": fact["native"], "sha256": fact["native_sha256"]},
               "snap": {"path": str(ROOT / prior["prediction_artifacts"][key]["path"]),
                        "sha256": prior["prediction_artifacts"][key]["sha256"]}}
        for role, path in probability_paths(key).items():
            member = members[str(path.relative_to(ARCHIVE))]
            row[role] = {"path": str(path), "sha256": member["sha256"]}
        cells[key] = row
    sources = [Path(__file__), Path(decode.__file__), Path(scoring.__file__),
               Path(fable.__file__), ROOT / "experiments/v2_spell_balanced_eval.py",
               ROOT / "evaluation/possession_metrics.py"]
    evidence = [previous.RESULT, ROOT / "experiments/results/turnover_supervision_eval.json",
                ROOT / "experiments/results/turnover_supervision_mix_v1.json"]
    previous.write_once(DECL, {"study": STUDY, "status": "frozen_before_composition_scoring",
        "question": "Do turnover-trained H/A predictions complement Fable's unchanged snap D-mask correction?",
        "authorization": "User requested complementary improvements instead of extensions on 2026-11-11",
        "arms": ARMS, "control": CONTROL, "cells": cells,
        "team_recipes": "Existing frontier, fresh continuation, hard-turnover continuation, and previously retained .75 hard/.25 fresh CE; retain .5/.375/.125 family mix, equal seeds/phases, H/A cost 3",
        "dead_recipes": "Original predecessor mask versus unchanged frozen snap_m0.0; no threshold, margin, shift or restart changes",
        "budget": "Three new snap compositions across 35 cells; original-mask arms and snap frontier are replay controls; no new fits or weight search",
        "retention": "Fable guards versus snap: v2 spell strictly improves; clean/v1 spell and every short-recall mean not lower; macro within .001; provider clean macro/D-F1 within .02",
        "ranking": "Eligible v2 spell then macro; otherwise retain unchanged snap",
        "interaction": "(snap with new H/A minus original mask with new H/A) minus (snap frontier minus original frontier)",
        "aggregation": "Realizations within match, matches within provider, providers equally",
        "preflight": "All four original-mask arms and snap frontier must replay on every cell before any new composition is scored",
        "tolerance": 1e-10, "fitting": False, "gar_access": False, "organizer_reference_access": False,
        "exposure": "Same five exposed matches and label-conditioned synthetic v2; not independent confirmation",
        "source_sha256": {str(p.relative_to(ROOT)): previous.sha(p) for p in sources},
        "evidence_sha256": {str(p.relative_to(ROOT)): previous.sha(p) for p in evidence},
        "turnover_archive_sha256": receipt["archive_sha256"]})


def inputs(key):
    facts = read(DECL)["cells"][key]
    for fact in facts.values():
        if previous.sha(fact["path"]) != fact["sha256"]:
            raise ValueError(f"Frozen input hash changed: {key}")
    cell = common.frontier_cell(key)
    with np.load(facts["snap"]["path"], allow_pickle=False) as saved:
        snap = saved["snap"].copy()
    probabilities = {"frontier": cell["prob__approved_baseline"]}
    for role, mappings in (("turnover", {"fresh": "prob__fresh_control", "hard": "prob__hard_turnover"}),
                           ("mixture", {"mix075": "prob__mix075"})):
        with np.load(facts[role]["path"], allow_pickle=False) as saved:
            for team, field in mappings.items():
                probabilities[team] = np.array(saved[field], dtype=float)
    return cell, snap, probabilities


def predict(probability, mask, cell):
    labels = decode.decode_candidate(probability, mask, cell["model_period"], float(cell["fps"]))
    if not np.array_equal(labels == "D", mask == "D"):
        raise ValueError("Complement changed the frozen D mask")
    return labels


def measure(cell, pred):
    row = decode._score_cell(dict(cell, fps=float(cell["fps"])), pred)
    return {m: float(row[m]) for m in scoring.METRICS}


def replay(key):
    cell, snap, probabilities = inputs(key)
    pipe = key.replace("__", "|")
    older = read(ROOT / "experiments/results/turnover_supervision_eval.json")["cell_metrics"]
    mixture = read(ROOT / "experiments/results/turnover_supervision_mix_v1.json")["cell_metrics"]
    previous_rows = read(previous.RESULT)["cells"][key]
    expected = {"original__frontier": previous_rows["frontier"],
                "original__fresh": older["fresh_control"][pipe],
                "original__hard": older["hard_turnover"][pipe],
                "original__mix075": mixture["mix075"][pipe],
                CONTROL: previous_rows["snap"]}
    rows = {}
    for name, target in expected.items():
        mask, team = name.split("__")
        pred = predict(probabilities[team], snap if mask == "snap" else cell["decoded__predecessor"], cell)
        row = measure(cell, pred)
        for metric in scoring.METRICS:
            if abs(row[metric] - target[metric]) > 1e-10:
                raise ValueError(f"Control mismatch: {key}/{name}/{metric}")
        rows[name] = row
    return key, rows


def compose(key):
    cell, snap, probabilities = inputs(key)
    rows, predictions = {}, {}
    for name in ARMS:
        mask, team = name.split("__")
        pred = predict(probabilities[team], snap if mask == "snap" else cell["decoded__predecessor"], cell)
        predictions[name] = pred
        rows[name] = measure(cell, pred)
    path = OUT / f"{key}.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream, **predictions)
    return key, rows, {"path": str(path.relative_to(ROOT)), "sha256": previous.sha(path)}


def run(workers):
    declaration = read(DECL)
    if RESULT.exists() or (OUT.exists() and any(OUT.iterdir())):
        raise FileExistsError("Preserve existing full/partial run")
    for rel, digest in {**declaration["source_sha256"], **declaration["evidence_sha256"]}.items():
        if previous.sha(ROOT / rel) != digest:
            raise ValueError("Frozen dependency changed: " + rel)
    keys = list(declaration["cells"])
    with ProcessPoolExecutor(workers) as pool:
        controls = dict(pool.map(replay, keys))
    previous.write_once(OUT / "preflight.json", {"all_controls_passed": True,
        "cells": controls, "declaration_sha256": previous.sha(DECL)})
    print("All 175 controls replayed before composition scoring", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        complete = list(pool.map(compose, keys))
    rows = {key: row for key, row, _ in complete}
    aggregate = {arm: scoring.aggregate(rows, arm) for arm in ARMS}
    failed = {f"snap__{team}": fable.guards(aggregate[f"snap__{team}"], aggregate[CONTROL]) for team in TEAMS[1:]}
    eligible = [arm for arm, checks in failed.items() if not checks]
    selected = max(eligible, key=lambda a: (aggregate[a]["feed_like_v2"]["spell_f1"], aggregate[a]["feed_like_v2"]["macro_f1"])) if eligible else CONTROL
    interaction = {team: {view: aggregate[f"snap__{team}"][view]["spell_f1"] - aggregate[f"original__{team}"][view]["spell_f1"] - aggregate[CONTROL][view]["spell_f1"] + aggregate["original__frontier"][view]["spell_f1"] for view in scoring.VIEW_NAMES} for team in TEAMS[1:]}
    previous.write_once(RESULT, {"study": STUDY, "declaration_sha256": previous.sha(DECL),
        "preflight_sha256": previous.sha(OUT / "preflight.json"), "aggregate": aggregate,
        "failed_guards": failed, "selected": selected, "spell_interaction": interaction,
        "cells": rows, "prediction_artifacts": {key: fact for key, _, fact in complete},
        "fitting": False, "gar_access": False, "organizer_reference_access": False})
    print(scoring.table(rows, ARMS, CONTROL), flush=True)
    print("Selected:", selected, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "run"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    declare() if args.action == "declare" else run(args.workers)
