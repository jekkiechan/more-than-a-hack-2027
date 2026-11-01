"""Frozen broad development search; no organizer reference access or model fits."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from evaluation import possession_metrics as pm
from experiments import decoder_search as ds, metrics, spell_mbr_run as shared, supervised

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
OUT = ROOT / "data/experiments/decoder_search_v2"
DECLARATION = RESULTS / "decoder_search_v2_declaration.json"
SOURCES = ("experiments/decoder_search.py", "experiments/decoder_search_run.py",
           "experiments/metrics.py", "experiments/supervised.py",
           "experiments/spell_mbr_run.py", "evaluation/possession_metrics.py")


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def restore(archive):
    """Restore only exact source arrays/checkpoints from the private LFS archive."""
    archive = Path(archive)
    manifest = json.loads((archive / "dense_targeted_followups_cache_manifest_v1.json").read_text())
    provenance = json.loads(shared.PROVENANCE.read_text())
    needed = (provenance["member_prediction_files"] | provenance["ensemble_prediction_files"] |
              provenance["model_checkpoint_hashes"])
    restored = {}
    for entry in manifest["files"]:
        rel = entry.get("restore_path")
        if rel not in needed:
            continue
        source = archive / entry["path"]
        if shared.sha256_file(source) != needed[rel]:
            raise ValueError(f"Source hash mismatch (or unmaterialized LFS pointer): {source}")
        dest = ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if shared.sha256_file(dest) != needed[rel]:
                raise ValueError(f"Existing restore destination differs: {dest}")
        else:
            shutil.copyfile(source, dest)
        restored[rel] = needed[rel]
    if restored != needed:
        raise ValueError(f"Incomplete restore: missing {set(needed)-set(restored)}")
    write_once(RESULTS / "decoder_search_v2_restore.json", {
        "study": "decoder_search_v2", "private_commit": "64b6eaafef0f25bd3229aae2021efc9e13a8a1c9",
        "manifest_sha256": shared.sha256_file(archive / "dense_targeted_followups_cache_manifest_v1.json"),
        "files": restored, "verified_files": len(restored),
        "reconstruction_status": shared.RECONSTRUCTION_STATUS,
        "model_fits_or_scoring": False})


def freeze():
    provenance = json.loads(shared.PROVENANCE.read_text())
    predictions = provenance["member_prediction_files"] | provenance["ensemble_prediction_files"]
    for rel, digest in (predictions | provenance["model_checkpoint_hashes"]).items():
        if shared.sha256_file(shared._resolve_artifact(rel)) != digest:
            raise ValueError(f"Input hash mismatch: {rel}")
    payload = {
        "study": "decoder_search_v2", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Can a bounded diverse postprocessing search improve official spell F1 over cost 3?",
        "authorization": "User explicitly requests creative alternatives and experiments beyond cost-3 Viterbi",
        "arms": ds.configurations(), "ensemble_cells_per_arm": 15,
        "confirmation": "Top 3 guard-passing ensemble clean arms (or fewer) evaluated on both component seeds; same exposed dev cohort, not independent confirmation",
        "selection": "Rank clean spell then macro among arms passing feed >= baseline-.01 and each provider clean macro/D F1 >= baseline-.02; separately report unconstrained maximum",
        "cross_provider_selection": "Select using only two providers and evaluate the omitted provider; guards use only selection providers. All are previously exposed development evidence.",
        "screen": "clean gain >= .01; failing the screen does not erase a positive exploratory gain",
        "aggregation": "Mean replicates within match, matches within provider, providers equally",
        "auxiliary": "0.70 spell + 0.30 macro; not used for ranking",
        "budget": "134 fixed arms, no post-result arms; 2010 ensemble cells plus at most 90 seed cells; no training",
        "expected_control": shared.EXPECTED_CONTROL, "control_tolerance": 1e-10,
        "source_sha256": {s: shared.sha256_file(ROOT / s) for s in SOURCES},
        "prediction_files": predictions, "model_checkpoint_hashes": provenance["model_checkpoint_hashes"],
        "evaluation_input_hashes": provenance["evaluation_input_hashes"],
        "provenance_sha256": shared.sha256_file(shared.PROVENANCE),
        "reconstruction_status": shared.RECONSTRUCTION_STATUS,
        "organizer_labels_used": False, "promotion_authorized": False,
    }
    write_once(DECLARATION, payload)


def quick_score(cell, prediction):
    """Official spell/confusion functions, skipping unused transition diagnostics."""
    _, gt, pred, _ = metrics.prepare_tracking_codes(cell["labels"], prediction,
        cell["period"], cell["fps"], cell["timestamps"])
    cm = pm.scored_confusion(pm.confusion(gt, pred))
    runs = pm.valid_runs(gt)
    spells = pm.spell_scores(gt, pred, runs)
    d = pm.per_class_prf(cm, 2)
    return {"spell_f1": spells["mean_f1"], "macro_f1": pm.macro_f1(cm),
            "d_f1": d[2], "d_precision": d[0], "d_recall": d[1],
            "graded_frames": int(cm.sum()), "reference_spell_count": spells["n_gt"],
            "predicted_spell_count": spells["n_pred"],
            "weighted_70_30": .7*spells["mean_f1"]+.3*pm.macro_f1(cm),
            "short_spell_recall_iou_0_5_lt3s": None,
            "short_reference_spell_count": None, "short_matched_spell_count": None}


def summarize(rows, member="ensemble"):
    return {v: shared._aggregate(rows, member, v) for v in ("clean", "feed_like")}


def guards(summary, baseline, providers=shared.PROVIDERS):
    def mean(view, key, values):
        return np.mean([values[view]["by_provider"][p][key] for p in providers])
    return (mean("feed_like", "spell_f1", summary) >= mean("feed_like", "spell_f1", baseline)-.01
            and all(summary["clean"]["by_provider"][p][k] >=
                        baseline["clean"]["by_provider"][p][k]-.02
                    for p in providers for k in ("macro_f1", "d_f1")))


def rank(summary, providers=shared.PROVIDERS):
    return tuple(float(np.mean([summary["clean"]["by_provider"][p][key] for p in providers]))
                 for key in ("spell_f1", "macro_f1"))


def run():
    declaration = json.loads(DECLARATION.read_text())
    for s, digest in declaration["source_sha256"].items():
        if shared.sha256_file(ROOT / s) != digest:
            raise ValueError(f"Frozen source changed: {s}")
    if declaration["arms"] != ds.configurations():
        raise ValueError("Arms changed")
    if declaration["provenance_sha256"] != shared.sha256_file(shared.PROVENANCE):
        raise ValueError("Provenance changed")
    OUT.mkdir(parents=True, exist_ok=False)
    write_once(OUT / "lock.json", {"declaration_sha256": shared.sha256_file(DECLARATION)})
    cells = shared._load_cells(declaration)
    ensemble = [c for c in cells if c["member"] == "ensemble"]
    baseline_rows = []
    for cell in cells:
        prediction = supervised.decode(cell["probabilities"], cell["model_period"], cell["fps"], 3)
        if not np.array_equal(prediction, cell["record"]["prediction"]):
            raise ValueError("Control path replay failed")
        row = {k: cell[k] for k in ("provider", "match", "view", "member")}
        row.update(quick_score(cell, prediction))
        official = shared._score_prediction(cell["labels"], prediction, cell["period"], cell["timestamps"], cell["fps"])
        for key in ("spell_f1", "macro_f1", "d_f1", "graded_frames", "reference_spell_count"):
            if abs(row[key]-official[key]) > 1e-12:
                raise ValueError(f"Selection scorer differs from official wrapper: {key}")
        row.update(official)
        baseline_rows.append(row)
    baseline = summarize(baseline_rows)
    for view, expected in shared.EXPECTED_CONTROL.items():
        if abs(baseline[view]["equal_provider_mean"]["spell_f1"]-expected) > declaration["control_tolerance"]:
            raise ValueError("Control aggregate replay failed")
    write_once(RESULTS / "decoder_search_v2_control_replay.json", {"passed": True, "baseline": baseline,
        "45_exact_paths": True, "45_official_score_parity": True})
    summaries, all_rows = {}, {}
    for config in declaration["arms"]:
        started = time.monotonic()
        rows = []
        for cell in ensemble:
            prediction = ds.decode(cell["probabilities"], cell["model_period"], cell["fps"], config)
            row = {k: cell[k] for k in ("provider", "match", "view", "member")}
            row.update(quick_score(cell, prediction))
            reference = next(r for r in baseline_rows if all(r[k] == row[k] for k in ("provider", "match", "view", "member")))
            for key in ("graded_frames", "reference_spell_count"):
                if row[key] != reference[key]:
                    raise ValueError("Grading population changed")
            rows.append(row)
        summaries[config["id"]] = summarize(rows)
        all_rows[config["id"]] = rows
        write_once(OUT / (config["id"]+".json"), {"config": config, "rows": rows, "seconds": time.monotonic()-started})
        s = summaries[config["id"]]
        print(config["id"], config["family"], s["clean"]["equal_provider_mean"]["spell_f1"],
              s["feed_like"]["equal_provider_mean"]["spell_f1"], flush=True)
    ranked = sorted(summaries, key=lambda a: rank(summaries[a]), reverse=True)
    eligible = [a for a in ranked if guards(summaries[a], baseline)]
    selected = eligible[:3]
    seed_scores = {}
    for arm in selected:
        config = next(c for c in declaration["arms"] if c["id"] == arm)
        rows = list(all_rows[arm])
        for cell in cells:
            prediction = ds.decode(cell["probabilities"], cell["model_period"], cell["fps"], config)
            row = {k: cell[k] for k in ("provider", "match", "view", "member")}
            row.update(shared._score_prediction(cell["labels"], prediction, cell["period"], cell["timestamps"], cell["fps"]))
            # Official full-wrapper replay checks selected ensemble selection scores too.
            if cell["member"] == "ensemble":
                old = next(r for r in rows if all(r[k] == row[k] for k in ("provider", "match", "view", "member")))
                if abs(old["spell_f1"]-row["spell_f1"]) > 1e-12:
                    raise ValueError("Final official replay failed")
                rows.remove(old)
            rows.append(row)
            path = OUT / arm / (Path(cell["relative_path"]).name)
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, prediction=prediction)
        write_once(OUT / (arm+"_final.json"), {"rows": rows})
        seed_scores[arm] = {m: summarize(rows, m) for m in shared.MEMBERS}
    cross = {}
    for held in shared.PROVIDERS:
        training = [p for p in shared.PROVIDERS if p != held]
        possible = [a for a in summaries if guards(summaries[a], baseline, training)]
        chosen = max(possible, key=lambda a: rank(summaries[a], training)) if possible else "baseline"
        cross[held] = {"chosen": chosen, "selection_providers": training,
            "held_provider": {v: (baseline if chosen == "baseline" else summaries[chosen])[v]["by_provider"][held]
                              for v in ("clean", "feed_like")}}
    write_once(RESULTS / "decoder_search_v2_aggregate.json", {
        "study": "decoder_search_v2", "baseline": baseline, "baseline_seeds": {m: summarize(baseline_rows, m) for m in shared.MEMBERS},
        "arm_count": len(summaries), "ensemble_cells": len(summaries)*15,
        "ranked_arm_ids": ranked, "eligible_ranked_arm_ids": eligible,
        "arms": summaries, "selected_seed_scores": seed_scores, "cross_provider_selection": cross,
        "organizer_labels_used": False, "promotion_performed": False,
        "declaration_sha256": shared.sha256_file(DECLARATION),
        "private_outputs": {str(p.relative_to(ROOT)): shared.sha256_file(p) for p in sorted(OUT.rglob("*.npz"))},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("restore", "freeze", "run"))
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.stage == "restore":
        restore(args.archive)
    elif args.stage == "freeze":
        freeze()
    else:
        run()


if __name__ == "__main__":
    main()
