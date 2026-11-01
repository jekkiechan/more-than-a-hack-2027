"""Freeze and evaluate the 118-arm refinement before any public diagnostic."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments import decoder_refinement as decoder, decoder_search_run as previous
from experiments import spell_mbr_run as shared

ROOT = previous.ROOT
RESULTS = previous.RESULTS
OUT = ROOT / "data/experiments/decoder_refinement_v3"
DECLARATION = RESULTS / "decoder_refinement_v3_declaration.json"
AGGREGATE = RESULTS / "decoder_refinement_v3_aggregate.json"
SOURCES = ("experiments/decoder_refinement.py", "experiments/decoder_refinement_run.py",
           "experiments/decoder_refinement_public.py", "experiments/decoder_search.py",
           "experiments/decoder_search_run.py", "experiments/spell_mbr_run.py",
           "experiments/metrics.py", "experiments/supervised.py", "evaluation/possession_metrics.py")


def freeze():
    prior = json.loads(previous.DECLARATION.read_text())
    for rel, digest in prior["source_sha256"].items():
        if shared.sha256_file(ROOT / rel) != digest:
            raise ValueError("Previous frozen source changed")
    shared._load_cells(prior)  # Exact array hashes, metadata, native masks, breaks and fold pairing.
    previous.write_once(DECLARATION, {
        "study": "decoder_refinement_v3", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does a focused split-cost refinement improve spell F1 over the previous development winner?",
        "arms": decoder.configurations(), "ensemble_cells": 1770, "max_component_seed_cells": 90,
        "source_sha256": {s: shared.sha256_file(ROOT / s) for s in SOURCES},
        "prior_declaration_sha256": shared.sha256_file(previous.DECLARATION),
        "prior_aggregate_sha256": shared.sha256_file(RESULTS / "decoder_search_v2_aggregate.json"),
        "prediction_files": prior["prediction_files"],
        "evaluation_input_hashes": prior["evaluation_input_hashes"],
        "model_checkpoint_hashes": prior["model_checkpoint_hashes"],
        "selection": "Clean spell then macro; aggregate stress and provider clean D/macro guards as before, plus each provider feed spell >= baseline-.01; seed checks descriptive only",
        "guards_reference": "Adopted cost 3; compare the previous H/A3 dead12 winner separately",
        "cross_provider_selection": "Select using only two providers and their guards, report omitted provider; all development matches already exposed",
        "public_diagnostic": "After external selection is complete, freeze the original H/A3 dead12 and final selected CSVs, deduplicate identical bytes, and score this fixed set once against public labels. Reuse saved adopted baseline; no selection or tuning afterward.",
        "explicit_user_request": "this looks promising. can you iterate on this? also, see if it has any affect on the public score",
        "public_labels_used_for_selection": False, "promotion_authorized": False,
        "primary": "Official equal-provider spell F1, macro for exact ties; fixed 70/30 auxiliary",
        "cohort": "Five exposed matches, 3 providers, clean and two frozen synthetic feed_like corruptions; reconstructed D2 controls with full provider exclusions",
        "reconstruction_status": shared.RECONSTRUCTION_STATUS,
        "screen_clean_gain_vs_adopted": .01, "control_parity_tolerance": 1e-12,
    })


def verify():
    d = json.loads(DECLARATION.read_text())
    for rel, digest in d["source_sha256"].items():
        if shared.sha256_file(ROOT / rel) != digest:
            raise ValueError(f"Frozen source changed: {rel}")
    if d["arms"] != decoder.configurations():
        raise ValueError("Configurations changed")
    for key, path in (("prior_declaration_sha256", previous.DECLARATION),
                      ("prior_aggregate_sha256", RESULTS / "decoder_search_v2_aggregate.json")):
        if shared.sha256_file(path) != d[key]:
            raise ValueError("Prior evidence changed")
    return d


def eligible(summary, baseline, providers=shared.PROVIDERS):
    return (previous.guards(summary, baseline, providers)
            and all(summary["feed_like"]["by_provider"][p]["spell_f1"] >=
                    baseline["feed_like"]["by_provider"][p]["spell_f1"]-.01 for p in providers))


def run():
    d = verify()
    OUT.mkdir(parents=True, exist_ok=False)
    previous.write_once(OUT / "lock.json", {"declaration_sha256": shared.sha256_file(DECLARATION)})
    cells = shared._load_cells(d)
    old_result = json.loads((RESULTS / "decoder_search_v2_aggregate.json").read_text())
    baseline = old_result["baseline"]
    old_rows = json.loads((previous.OUT / "a019_final.json").read_text())["rows"]
    control_config = dict(family="matrix", ha=3, dead=12)
    # Verify the original winner's exact 45 paths and saved scalar metrics.
    for cell in cells:
        guess = decoder.decode(cell["probabilities"], cell["model_period"], cell["fps"], control_config)
        with np.load(previous.OUT / "a019" / Path(cell["relative_path"]).name) as saved:
            if not np.array_equal(guess, saved["prediction"]):
                raise ValueError("Prior winner path replay failed")
        old_row = next(r for r in old_rows if all(r[k] == cell[k] for k in ("provider", "match", "view", "member")))
        metrics = previous.quick_score(cell, guess)
        for key in ("spell_f1", "macro_f1", "d_f1", "reference_spell_count", "graded_frames"):
            if abs(metrics[key]-old_row[key]) > d["control_parity_tolerance"]:
                raise ValueError("Previous winner scalar replay failed")
    previous.write_once(RESULTS / "decoder_refinement_v3_control_replay.json", {
        "passed": True, "exact_paths": 45, "scalar_cells": 45,
        "baseline_metrics_reused": True, "organizer_reference_read": False})
    summaries, arm_rows = {}, {}
    for config in d["arms"]:
        rows = []
        for cell in cells:
            if cell["member"] != "ensemble":
                continue
            guess = decoder.decode(cell["probabilities"], cell["model_period"], cell["fps"], config)
            row = {k: cell[k] for k in ("provider", "match", "view", "member")}
            row.update(previous.quick_score(cell, guess))
            reference = next(r for r in old_rows if all(r[k] == row[k] for k in ("provider", "match", "view", "member")))
            if any(row[k] != reference[k] for k in ("graded_frames", "reference_spell_count")):
                raise ValueError("Grading population changed")
            rows.append(row)
        arm_rows[config["id"]] = rows
        summaries[config["id"]] = previous.summarize(rows)
        previous.write_once(OUT / (config["id"]+".json"), {"config": config, "rows": rows})
        print(config["id"], config["family"], summaries[config["id"]]["clean"]["equal_provider_mean"]["spell_f1"],
              summaries[config["id"]]["feed_like"]["equal_provider_mean"]["spell_f1"], flush=True)
    ranked = sorted(summaries, key=lambda a: previous.rank(summaries[a]), reverse=True)
    allowed = [a for a in ranked if eligible(summaries[a], baseline)]
    finalists = allowed[:3]
    detailed = {}
    for arm in finalists:
        config = next(c for c in d["arms"] if c["id"] == arm)
        rows = []
        for cell in cells:
            guess = decoder.decode(cell["probabilities"], cell["model_period"], cell["fps"], config)
            row = {k: cell[k] for k in ("provider", "match", "view", "member")}
            row.update(shared._score_prediction(cell["labels"], guess, cell["period"], cell["timestamps"], cell["fps"]))
            if cell["member"] == "ensemble":
                before = next(r for r in arm_rows[arm] if all(r[k] == row[k] for k in ("provider", "match", "view", "member")))
                if abs(before["spell_f1"]-row["spell_f1"]) > 1e-12:
                    raise ValueError("Official full-wrapper finalist replay failed")
            rows.append(row)
            out = OUT / arm / Path(cell["relative_path"]).name
            out.parent.mkdir(exist_ok=True)
            np.savez_compressed(out, prediction=guess)
        previous.write_once(OUT / (arm+"_final.json"), {"rows": rows})
        detailed[arm] = {m: previous.summarize(rows, m) for m in shared.MEMBERS}
    cross = {}
    for held in shared.PROVIDERS:
        train = [p for p in shared.PROVIDERS if p != held]
        choices = [a for a in summaries if eligible(summaries[a], baseline, train)]
        chosen = max(choices, key=lambda a: previous.rank(summaries[a], train)) if choices else None
        cross[held] = {"selected": chosen, "selection_providers": train,
            "scores": {v: (summaries[chosen] if chosen else {"clean": baseline["clean"], "feed_like": baseline["feed_like"]})[v]["by_provider"][held]
                       for v in ("clean", "feed_like")}}
    previous.write_once(AGGREGATE, {
        "study": "decoder_refinement_v3", "declaration_sha256": shared.sha256_file(DECLARATION),
        "arms": summaries, "ranked": ranked, "eligible_ranked": allowed,
        "selected_config": next(c for c in d["arms"] if c["id"] == allowed[0]),
        "finalists": detailed, "baseline": baseline, "previous_winner": old_result["arms"]["a019"],
        "baseline_seeds": old_result["baseline_seeds"], "previous_winner_seeds": old_result["selected_seed_scores"]["a019"],
        "cross_provider_selection": cross, "public_used_for_selection": False,
        "completed_ensemble_cells": 1770, "component_seed_cells": len(finalists)*30,
        "private_predictions": {str(p.relative_to(ROOT)): shared.sha256_file(p) for p in OUT.rglob("*.npz")},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    args = parser.parse_args()
    freeze() if args.stage == "freeze" else run()
