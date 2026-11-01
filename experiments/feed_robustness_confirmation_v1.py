"""Fresh corruption realizations for the fixed training-study winner.

Preparation is declared before results. Scoring is conditional on improvement;
it cannot select a different arm, decoder or mixture weight.
"""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import argparse
import hashlib
from pathlib import Path

import joblib
import numpy as np
from experiments import feed_robustness_v1 as study
from experiments import preparation as prep, multisource_eval as ms
from experiments import soccernet_extension as old, soccernet_capacity as sc
from experiments import sn_dense_train as sn, supervised, tracking_corruption as tc
from experiments import spell_mbr_run as shared, decoder_search_run as scoring

STUDY = "feed_robustness_confirmation_v1"
OUT = study.ROOT / "data/experiments" / STUDY
DECLARATION = study.RESULTS / f"{STUDY}_declaration.json"
SEEDS = (551201, 551202)


def declare():
    paths = [Path(__file__), study.ROOT / "experiments/feed_robustness_v1.py"]
    paths += [study.ROOT / "experiments" / p for p in prep.SOURCES]
    items = old.selection_items()
    for item in items:
        if study.sha(prep.data_path(item, old.DATA)) != item["sha256"]:
            raise ValueError("Raw development input changed")
    study.write(DECLARATION, {"study": STUDY, "utc": datetime.now(timezone.utc).isoformat(),
        "condition": "Score only if the frozen feed_robustness_v1 winner improves feed spell over baseline and passes its guards; otherwise record skipped.",
        "selection": "Use the exact earlier winner. No reselection, threshold, mixture, seed or decoder tuning from these results.",
        "seeds": SEEDS, "matches": items, "calibration": sn.default_ball_calibration(),
        "components": study.COMPONENTS, "training": False, "organizer_reference_access": False,
        "evidence": "New synthetic realizations on the same previously exposed five matches; not independent-match confirmation.",
        "source_sha256": {str(p.relative_to(study.ROOT)): study.sha(p) for p in paths},
        "parent_declaration_sha256": study.sha(study.DECLARATION)})


def check():
    d = study.read(DECLARATION)
    for p, digest in d["source_sha256"].items():
        if study.sha(study.ROOT / p) != digest:
            raise ValueError("Confirmation source changed")
    if study.sha(study.DECLARATION) != d["parent_declaration_sha256"]:
        raise ValueError("Parent declaration changed")
    return d


def prepare_cell(arg):
    item, index = arg
    d = check()
    path = prep.data_path(item, old.DATA)
    if study.sha(path) != item["sha256"]:
        raise ValueError("Raw input changed")
    inputs, labels = ms.view(path)
    seed = int(hashlib.sha256(f"{STUDY}/{SEEDS[index]}/{item['match']}".encode()).hexdigest()[:8], 16)
    inputs = study.raw_view(inputs, d["calibration"], seed)
    record = prep.build_record(inputs, labels, item["source"], item["match"])
    dest = OUT / f"{item['match']}__fresh_{index+1}.joblib"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(dest)
    joblib.dump({"inputs": inputs, "labels": labels, "record": record}, dest)
    print("PREPARED", dest.name, flush=True)
    return {"path": str(dest.relative_to(study.ROOT)), "sha256": study.sha(dest),
            "match": item["match"], "provider": item["source"], "replicate": index+1, "seed": seed}


def prepare():
    d = check()
    study.write(OUT / "prepare_lock.json", {"declaration_sha256": study.sha(DECLARATION)})
    with ProcessPoolExecutor(max_workers=2) as pool:
        files = list(pool.map(prepare_cell, [(i, n) for i in d["matches"] for n in range(2)]))
    study.write(OUT / "preparation.json", {"files": files, "declaration_sha256": study.sha(DECLARATION)})


def run():
    check()
    parent = study.read(study.RESULTS / "feed_robustness_v1_aggregate.json")
    winner = parent["winner"]
    if winner == "baseline":
        study.write(study.RESULTS / f"{STUDY}_skipped.json", {"reason": "No guard-passing improvement in parent study", "scored": False})
        return
    prep_receipt = study.read(OUT / "preparation.json")
    study.write(OUT / "score_lock.json", {"parent_aggregate_sha256": study.sha(study.RESULTS / "feed_robustness_v1_aggregate.json"),
                "winner": winner, "preparation_sha256": study.sha(OUT / "preparation.json")})
    rows = {arm: [r for r in parent["rows"][arm] if r["view"] == "clean"] for arm in ("baseline", winner)}
    models = {}
    for provider in ("metrica", "skillcorner", "sportec"):
        fold = "leave_out_" + provider
        for seed in study.SEEDS:
            models[provider, seed] = (
                joblib.load(study.ROOT / f"data/experiments/dense_label_weight_v1/models/control_D2__{provider}__s{seed}.joblib"),
                joblib.load(study.OUT / "models" / f"{fold}_{seed}.joblib"),
                joblib.load(sc.initial_path(fold, seed)))
    files = {}
    for entry in prep_receipt["files"]:
        path = study.ROOT / entry["path"]
        if study.sha(path) != entry["sha256"]:
            raise ValueError("Prepared confirmation input changed")
        cached = joblib.load(path)
        inputs = tc.inputs_only(cached["inputs"])
        record = old.inference_record(cached["record"])
        c = {"provider": shared._provider_for_match(entry["match"]),
             "match": entry["match"].removeprefix("skillcorner_had_"),
             "view": f"feed_like_fresh_{entry['replicate']}", "labels": cached["labels"],
             "period": inputs["period"], "model_period": supervised.model_periods(inputs),
             "timestamps": inputs["timestamps"], "fps": float(inputs["fps"])}
        ps = {a: {} for a in rows}
        for seed in study.SEEDS:
            base, candidate, teacher = models[entry["provider"], seed]
            bp = old.upsample(sn.predict(record, base, teacher), record, inputs)
            cp = old.upsample(sn.predict(record, candidate, teacher), record, inputs)
            ps["baseline"][str(seed)] = bp
            ps[winner][str(seed)] = cp if winner == "feed_augmented" else .5*(cp+bp)
        for arm in rows:
            ps[arm]["ensemble"] = np.mean(list(ps[arm].values()), axis=0)
            for member, probability in ps[arm].items():
                c["member"] = member
                row, prediction = study.score(c, probability)
                rows[arm].append(row)
                dest = OUT / "predictions" / arm / f"{entry['match']}__fresh_{entry['replicate']}__{member}.npz"
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, probabilities=probability, prediction=prediction,
                    labels=c["labels"], period=c["period"], timestamps=c["timestamps"],
                    model_period=c["model_period"], fps=c["fps"])
                files[str(dest.relative_to(study.ROOT))] = study.sha(dest)
    summaries = {a: {m: scoring.summarize(r, m) for m in shared.MEMBERS} for a, r in rows.items()}
    study.write(study.RESULTS / f"{STUDY}_aggregate.json", {"winner_fixed_before_scoring": winner,
        "rows": rows, "summaries": summaries, "prediction_sha256": files,
        "new_scored_cells": 60, "reused_clean_cells": 30,
        "independent_match_confirmation": False, "organizer_reference_access": False})
    print({a: s["ensemble"]["feed_like"]["equal_provider_mean"]["spell_f1"] for a, s in summaries.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "prepare", "run"))
    globals()[parser.parse_args().action]()
