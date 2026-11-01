"""Bounded feed-first search on frozen, provider-excluded dense probabilities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
from pathlib import Path

import numpy as np

from experiments import feed_spell_push as inference, decoder_search_run as scoring
from experiments import spell_mbr_run as shared, supervised

ROOT = shared.ROOT
RESULTS = shared.RESULTS
OUT = ROOT / "data/experiments/feed_spell_push_v1"
DECLARATION = RESULTS / "feed_spell_push_v1_declaration.json"
AGGREGATE = RESULTS / "feed_spell_push_v1_aggregate.json"
SOURCES = ("experiments/feed_spell_push.py", "experiments/feed_spell_push_run.py",
           "experiments/decoder_refinement.py", "experiments/decoder_search.py",
           "experiments/decoder_search_run.py", "experiments/spell_mbr_run.py",
           "experiments/supervised.py", "experiments/metrics.py", "evaluation/possession_metrics.py")
CELLS = []


def freeze():
    prior = json.loads((RESULTS / "decoder_refinement_v3_declaration.json").read_text())
    shared._load_cells(prior)
    for rel, sha in prior["model_checkpoint_hashes"].items():
        if shared.sha256_file(shared._resolve_artifact(rel)) != sha:
            raise ValueError("Fold checkpoint hash mismatch")
    scoring.write_once(DECLARATION, {
        "study": "feed_spell_push_v1", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Can feed-first calibration, probability pooling or segmentation beat the previous feed-like leader?",
        "authorization": "User requests iteration to maximize feed-like spell F1, including ensemble, stacking, preprocessing and postprocessing methods",
        "arms": inference.configurations(), "ensemble_cells": len(inference.configurations())*15,
        "component_seed_budget": 90, "new_fits": 0, "organizer_reference_access": False,
        "primary": "feed_like equal-provider official spell F1; macro F1 for exact ties",
        "aggregation": "Mean stress realizations within match, matches within provider, providers equally",
        "guards": "clean aggregate spell >= adopted-.005; each provider clean macro/D F1 >= adopted-.02; each provider feed spell >= adopted-.01",
        "historical_clean_screen": "+.01 vs adopted reported separately; no promotion authorized",
        "confirmation": "Top three eligible arms on both component seeds; descriptive, no reselection",
        "selection_check": "Select on two providers using only those providers' guards and feed scores, apply to omitted provider; exposed development evidence",
        "cohort": "Five repeatedly exposed matches, three providers, clean plus two synthetic feed-like realizations",
        "reconstruction_status": shared.RECONSTRUCTION_STATUS,
        "stacking_limit": "No supervised meta learner: suitable nested base-model predictions excluding the outer provider throughout are unavailable",
        "source_sha256": {s: shared.sha256_file(ROOT / s) for s in SOURCES},
        "prediction_files": prior["prediction_files"],
        "evaluation_input_hashes": prior["evaluation_input_hashes"],
        "model_checkpoint_hashes": prior["model_checkpoint_hashes"],
        "prior_aggregate_sha256": shared.sha256_file(RESULTS / "decoder_refinement_v3_aggregate.json"),
        "control_tolerance": 1e-12, "public_used_for_selection": False,
    })


def paired_cells(declaration):
    cells = shared._load_cells(declaration)
    for c in cells:
        group = [z for z in cells if all(z[k] == c[k] for k in ("provider", "match", "view"))]
        by_member = {z["member"]: z for z in group}
        if len(by_member) != 3:
            raise ValueError("Incomplete member pairing")
        for z in group:
            for key in ("labels", "model_period", "period", "timestamps"):
                if not np.array_equal(c[key], z[key]):
                    raise ValueError("Native member alignment mismatch")
            if c["fps"] != z["fps"]:
                raise ValueError("Native fps mismatch")
        x = np.stack([by_member[m]["probabilities"] for m in ("2026", "2027")])
        if not np.array_equal(x.mean(axis=0), by_member["ensemble"]["probabilities"]):
            raise ValueError("Ensemble is not the exact native arithmetic mean")
        c["members"] = x if c["member"] == "ensemble" else c["probabilities"][None]
    return cells


def identity(c):
    return {k: c[k] for k in ("provider", "match", "view", "member")}


def eligible(s, b, providers=shared.PROVIDERS):
    clean = np.mean([s["clean"]["by_provider"][p]["spell_f1"] for p in providers])
    base = np.mean([b["clean"]["by_provider"][p]["spell_f1"] for p in providers])
    return (clean >= base-.005
            and all(s["clean"]["by_provider"][p][k] >= b["clean"]["by_provider"][p][k]-.02
                    for p in providers for k in ("macro_f1", "d_f1"))
            and all(s["feed_like"]["by_provider"][p]["spell_f1"] >=
                    b["feed_like"]["by_provider"][p]["spell_f1"]-.01 for p in providers))


def rank(s, providers=shared.PROVIDERS):
    return tuple(float(np.mean([s["feed_like"]["by_provider"][p][k] for p in providers]))
                 for k in ("spell_f1", "macro_f1"))


def evaluate(config):
    rows = []
    for c in CELLS:
        if c["member"] != "ensemble":
            continue
        y = inference.decode(c["members"], c["model_period"], c["fps"], config)
        row = identity(c) | scoring.quick_score(c, y)
        rows.append(row)
    return config, rows, scoring.summarize(rows)


def run(workers):
    global CELLS
    d = json.loads(DECLARATION.read_text())
    for rel, sha in d["source_sha256"].items():
        if shared.sha256_file(ROOT / rel) != sha:
            raise ValueError("Frozen source changed: "+rel)
    if d["arms"] != inference.configurations():
        raise ValueError("Arms changed")
    if shared.sha256_file(RESULTS / "decoder_refinement_v3_aggregate.json") != d["prior_aggregate_sha256"]:
        raise ValueError("Prior aggregate changed")
    OUT.mkdir(parents=True, exist_ok=False)
    scoring.write_once(OUT / "lock.json", {"declaration_sha256": shared.sha256_file(DECLARATION)})
    CELLS = paired_cells(d)
    baseline_rows = []
    for c in CELLS:
        y = supervised.decode(c["probabilities"], c["model_period"], c["fps"], 3)
        if not np.array_equal(y, c["record"]["prediction"]):
            raise ValueError("Baseline exact-path replay failed")
        row = identity(c) | scoring.quick_score(c, y)
        full = shared._score_prediction(c["labels"], y, c["period"], c["timestamps"], c["fps"])
        for k in ("spell_f1", "macro_f1", "d_f1", "graded_frames", "reference_spell_count"):
            if abs(row[k]-full[k]) > d["control_tolerance"]:
                raise ValueError("Official score parity failed")
        baseline_rows.append(identity(c) | full)
    baseline = scoring.summarize(baseline_rows)
    for v, expected in shared.EXPECTED_CONTROL.items():
        if abs(baseline[v]["equal_provider_mean"]["spell_f1"]-expected) > d["control_tolerance"]:
            raise ValueError("Baseline score replay failed")
    prior = json.loads((RESULTS / "decoder_refinement_v3_aggregate.json").read_text())
    # Use the complete paired execution path on a real cell before the full sweep.
    reference, rows, summary = evaluate(d["arms"][1])
    for v in ("clean", "feed_like"):
        if abs(summary[v]["equal_provider_mean"]["spell_f1"]-
               prior["arms"]["r073"][v]["equal_provider_mean"]["spell_f1"]) > d["control_tolerance"]:
            raise ValueError("Previous feed leader replay failed")
    scoring.write_once(RESULTS / "feed_spell_push_v1_control_replay.json", {
        "passed": True, "baseline_exact_paths": 45, "official_score_parity": 45,
        "paired_native_member_means": 15, "prior_feed_leader_cells": 15,
        "baseline": baseline, "previous_feed_leader": summary})
    summaries, arm_rows = {}, {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("fork")) as executor:
        for config, rows, summary in executor.map(evaluate, d["arms"]):
            for row in rows:
                before = next(r for r in baseline_rows if all(r[k] == row[k] for k in identity(row)))
                if any(row[k] != before[k] for k in ("graded_frames", "reference_spell_count")):
                    raise ValueError("Grading population changed")
            arm = config["id"]
            summaries[arm], arm_rows[arm] = summary, rows
            scoring.write_once(OUT / (arm+".json"), {"config": config, "rows": rows})
            print(arm, config["family"], summary["clean"]["equal_provider_mean"]["spell_f1"],
                  summary["feed_like"]["equal_provider_mean"]["spell_f1"], flush=True)
    ranked = sorted(summaries, key=lambda a: rank(summaries[a]), reverse=True)
    allowed = [a for a in ranked if eligible(summaries[a], baseline)]
    finalists = {}
    for arm in allowed[:3]:
        config = next(c for c in d["arms"] if c["id"] == arm)
        rows = []
        for c in CELLS:
            y = inference.decode(c["members"], c["model_period"], c["fps"], config)
            row = identity(c) | shared._score_prediction(c["labels"], y, c["period"], c["timestamps"], c["fps"])
            if c["member"] == "ensemble":
                before = next(r for r in arm_rows[arm] if all(r[k] == row[k] for k in identity(row)))
                if abs(before["spell_f1"]-row["spell_f1"]) > d["control_tolerance"]:
                    raise ValueError("Finalist official-wrapper replay failed")
            rows.append(row)
            path = OUT / arm / Path(c["relative_path"]).name
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, prediction=y)
        scoring.write_once(OUT / (arm+"_final.json"), {"rows": rows})
        finalists[arm] = {m: scoring.summarize(rows, m) for m in shared.MEMBERS}
    cross = {}
    for held in shared.PROVIDERS:
        train = [p for p in shared.PROVIDERS if p != held]
        choices = [a for a in summaries if eligible(summaries[a], baseline, train)]
        chosen = max(choices, key=lambda a: rank(summaries[a], train))
        cross[held] = {"selected": chosen, "selection_providers": train,
            "scores": {v: summaries[chosen][v]["by_provider"][held] for v in ("clean", "feed_like")}}
    scoring.write_once(AGGREGATE, {
        "study": "feed_spell_push_v1", "declaration_sha256": shared.sha256_file(DECLARATION),
        "arms": summaries, "ranked": ranked, "eligible_ranked": allowed,
        "selected_config": next(c for c in d["arms"] if c["id"] == allowed[0]),
        "finalists": finalists, "baseline": baseline,
        "baseline_seeds": {m: scoring.summarize(baseline_rows, m) for m in shared.MEMBERS},
        "previous_clean_leader": prior["arms"]["r069"], "previous_feed_leader": prior["arms"]["r073"],
        "cross_provider_selection": cross, "completed_ensemble_cells": len(summaries)*15,
        "component_seed_cells": len(finalists)*30, "organizer_labels_accessed": False,
        "private_prediction_hashes": {str(p.relative_to(ROOT)): shared.sha256_file(p) for p in OUT.rglob("*.npz")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else run(args.workers)
