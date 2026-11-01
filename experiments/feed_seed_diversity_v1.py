"""Fixed four-member augmented ensemble; no seed or mixture selection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import joblib
import numpy as np
from experiments import feed_robustness_v1 as parent
from experiments import feed_robustness_confirmation_v1 as confirmation
from experiments import soccernet_capacity as sc, soccernet_extension as old
from experiments import sn_dense_train as sn, soccernet_dense_labels as dl
from experiments import spell_mbr_run as shared, decoder_search_run as scoring
from experiments import feed_spell_push_run as selection, supervised, tracking_corruption as tc

STUDY = "feed_seed_diversity_v1"
OUT = parent.ROOT / "data/experiments" / STUDY
DECLARATION = parent.RESULTS / f"{STUDY}_declaration.json"
SEEDS = (2028, 2029)


def teacher_seed(seed):
    if seed not in SEEDS:
        raise ValueError("Only the two predeclared additional seeds are allowed")
    return seed - 2


def declare():
    parent.check()
    verified = parent.read(parent.RESULTS / "feed_robustness_v1_verification.json")
    if not verified["passed"]:
        raise ValueError("Verified parent required")
    parent.write(DECLARATION, {"study": STUDY, "utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does adding two fixed augmented-model seeds improve the two-member ensemble by reducing training variance?",
        "rationale": "Parent gains feed spell in both seeds and all providers; same training data and raw defect views, no new objective.",
        "seeds": SEEDS, "teacher_seed_mapping": {str(s): teacher_seed(s) for s in SEEDS},
        "teacher_note": "Reuse the exact provider-excluded teachers and normalizers; new dense-training randomness only. These are not newly trained teacher seeds.",
        "fits": 6, "updates_per_fit": 2048, "recipe": "Same raw-feed augmented D2 recipe as parent",
        "ensemble": "Equal native probability mean of all four seeds 2026/2027/2028/2029, then fixed f062 decoder. Never select a seed or subset.",
        "primary": "Four-member original-development feed spell must exceed the existing two-member score; guards remain against the promoted baseline. .55 stretch.",
        "guards": "Parent clean aggregate >= baseline-.005; each provider clean macro/D >= baseline-.02; each provider feed spell >= baseline-.01",
        "fresh_check": "Evaluate the same fixed four-member rule on the already frozen two fresh corruption realizations, regardless of its original score. No selection or changes from the fresh check.",
        "evidence": "Exposed five-match development; new synthetic realizations are not independent matches",
        "organizer_reference_access": False, "promotion": False,
        "source_sha256": {str(Path(__file__).relative_to(parent.ROOT)): parent.sha(Path(__file__))},
        "parent_declaration_sha256": parent.sha(parent.DECLARATION),
        "parent_aggregate_sha256": parent.sha(parent.RESULTS / "feed_robustness_v1_aggregate.json"),
        "parent_model_seal_sha256": parent.sha(parent.RESULTS / "feed_robustness_v1_model_seal.json"),
        "fresh_preparation_sha256": parent.sha(confirmation.OUT / "preparation.json")})


def check():
    d = parent.read(DECLARATION)
    parent.check()
    for rel, digest in d["source_sha256"].items():
        if parent.sha(parent.ROOT / rel) != digest:
            raise ValueError("Frozen diversity source changed")
    for path, key in ((parent.DECLARATION, "parent_declaration_sha256"),
        (parent.RESULTS / "feed_robustness_v1_aggregate.json", "parent_aggregate_sha256"),
        (parent.RESULTS / "feed_robustness_v1_model_seal.json", "parent_model_seal_sha256"),
        (confirmation.OUT / "preparation.json", "fresh_preparation_sha256")):
        if parent.sha(path) != d[key]:
            raise ValueError("Frozen dependency changed")
    return d


def fit(fold, seed):
    check()
    source_seed = teacher_seed(seed)
    if fold not in parent.FOLDS:
        raise ValueError("Unknown fold")
    path = OUT / "models" / f"{fold}_{seed}.joblib"
    parent.write(path.with_suffix(".lock.json"), {"declaration_sha256": parent.sha(DECLARATION)})
    teacher = joblib.load(sc.initial_path(fold, source_seed))
    records = parent.training(fold)
    gar = dl.load_dense_records("train", output=parent.TRAIN_RECORDS)
    start = time.monotonic()
    result = sn.train(teacher, records, gar, width=96, lr=3e-4, seed=seed, steps=2048,
        fraction=.5, batch_size=32, ha_target="dense", train_d=True, require_full_native=True,
        log_every=256, verbose=True)
    if path.exists():
        raise FileExistsError(path)
    joblib.dump(result["model"], path)
    parent.write(path.with_suffix(".json"), {"fold": fold, "seed": seed, "teacher_seed": source_seed,
        "sha256": parent.sha(path), "declaration_sha256": parent.sha(DECLARATION),
        "wall_seconds": time.monotonic()-start, "config": result["config"],
        "old_matches": sorted({str(r["match"]) for r in records}), "exposures": result["exposures"]})
    print("FIT COMPLETE", fold, seed, flush=True)


def seal():
    check()
    files = {}
    for fold in parent.FOLDS:
        for seed in SEEDS:
            path = OUT / "models" / f"{fold}_{seed}.joblib"
            receipt = parent.read(path.with_suffix(".json"))
            if parent.sha(path) != receipt["sha256"] or receipt["declaration_sha256"] != parent.sha(DECLARATION):
                raise ValueError("Fit receipt differs")
            files[str(path.relative_to(parent.ROOT))] = parent.sha(path)
    parent.write(parent.RESULTS / f"{STUDY}_model_seal.json", {"files": files,
                 "declaration_sha256": parent.sha(DECLARATION)})


def load_models():
    seal_value = parent.read(parent.RESULTS / f"{STUDY}_model_seal.json")
    for rel, digest in seal_value["files"].items():
        if parent.sha(parent.ROOT / rel) != digest:
            raise ValueError("Sealed model changed")
    return {(fold, seed): (joblib.load(OUT / "models" / f"{fold}_{seed}.joblib"),
                           joblib.load(sc.initial_path(fold, teacher_seed(seed))))
            for fold in parent.FOLDS for seed in SEEDS}


def save(c, probability, rows, files, group):
    row, prediction = parent.score(c, probability)
    rows.append(row)
    key = tuple(c[k] for k in ("provider", "match", "view", "member"))
    path = OUT / "predictions" / group / ("__".join(key)+".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=probability, prediction=prediction,
        labels=c["labels"], period=c["period"], timestamps=c["timestamps"],
        model_period=c["model_period"], fps=c["fps"])
    files[str(path.relative_to(parent.ROOT))] = parent.sha(path)


def evaluate():
    check()
    models = load_models()
    parent.write(OUT / "score_lock.json", {"model_seal_sha256": parent.sha(parent.RESULTS / f"{STUDY}_model_seal.json")})
    previous = parent.read(parent.RESULTS / "feed_robustness_v1_aggregate.json")
    cells = shared._load_cells(parent.read(parent.DECLARATION))
    source = {(c["provider"], c["match"], c["view"]): c for c in cells if c["member"] == "2026"}
    rows, files = [], {}
    for (provider, match, view), c in source.items():
        fold = "leave_out_" + provider.lower()
        probabilities = []
        for seed in parent.SEEDS:
            path = parent.OUT / "predictions/feed_augmented" / ("__".join((provider, match, view, str(seed)))+".npz")
            if parent.sha(path) != previous["prediction_sha256"][str(path.relative_to(parent.ROOT))]:
                raise ValueError("Original augmented prediction changed")
            with np.load(path) as z:
                probabilities.append(z["probabilities"])
        for seed in SEEDS:
            model, teacher = models[fold, seed]
            p = parent.infer(c, model, teacher)
            probabilities.append(p)
            save(dict(c, member=str(seed)), p, rows, files, "original")
        save(dict(c, member="four_seed"), np.mean(probabilities, axis=0), rows, files, "original")
        print("SCORED", provider, match, view, flush=True)
    summaries = {m: scoring.summarize(rows, m) for m in ("2028", "2029", "four_seed")}
    baseline = previous["summaries"]["baseline"]["ensemble"]
    two = previous["summaries"]["feed_augmented"]["ensemble"]
    eligible = selection.eligible(summaries["four_seed"], baseline)
    improves = selection.rank(summaries["four_seed"]) > selection.rank(two)
    parent.write(parent.RESULTS / f"{STUDY}_aggregate.json", {"study": STUDY, "rows": rows,
        "summaries": summaries, "baseline": baseline, "two_seed": two,
        "guards_pass": eligible, "improves_two_seed": improves,
        "stretch_reached": summaries["four_seed"]["feed_like"]["equal_provider_mean"]["spell_f1"] >= .55,
        "prediction_sha256": files, "organizer_reference_access": False})
    print({v: summaries["four_seed"][v]["equal_provider_mean"]["spell_f1"] for v in ("clean", "feed_like")}, eligible, improves, flush=True)


def fresh():
    check()
    models = load_models()
    original = parent.read(parent.RESULTS / f"{STUDY}_aggregate.json")
    prior = parent.read(parent.RESULTS / "feed_robustness_confirmation_v1_aggregate.json")
    if prior["winner_fixed_before_scoring"] != "feed_augmented":
        raise ValueError("This declared continuation expects the selected raw-augmentation parent")
    preparation = parent.read(confirmation.OUT / "preparation.json")
    parent.write(OUT / "fresh_lock.json", {"declaration_sha256": parent.sha(DECLARATION),
        "original_aggregate_sha256": parent.sha(parent.RESULTS / f"{STUDY}_aggregate.json"),
        "parent_fresh_aggregate_sha256": parent.sha(parent.RESULTS / "feed_robustness_confirmation_v1_aggregate.json")})
    rows = [r for r in original["rows"] if r["view"] == "clean" and r["member"] == "four_seed"]
    files = {}
    for entry in preparation["files"]:
        path = parent.ROOT / entry["path"]
        if parent.sha(path) != entry["sha256"]:
            raise ValueError("Fresh feature cache changed")
        cached = joblib.load(path)
        inputs = tc.inputs_only(cached["inputs"])
        record = old.inference_record(cached["record"])
        c = dict(provider=shared._provider_for_match(entry["match"]), match=entry["match"].removeprefix("skillcorner_had_"),
                 view=f"feed_like_fresh_{entry['replicate']}", member="four_seed", labels=cached["labels"],
                 period=inputs["period"], model_period=supervised.model_periods(inputs),
                 timestamps=inputs["timestamps"], fps=float(inputs["fps"]))
        probabilities = []
        for seed in parent.SEEDS:
            path = confirmation.OUT / "predictions/feed_augmented" / f"{entry['match']}__fresh_{entry['replicate']}__{seed}.npz"
            if parent.sha(path) != prior["prediction_sha256"][str(path.relative_to(parent.ROOT))]:
                raise ValueError("Parent fresh probabilities changed")
            with np.load(path) as z:
                probabilities.append(z["probabilities"])
        fold = "leave_out_" + entry["provider"]
        for seed in SEEDS:
            model, teacher = models[fold, seed]
            probabilities.append(old.upsample(sn.predict(record, model, teacher), record, inputs))
        save(c, np.mean(probabilities, axis=0), rows, files, "fresh")
    summary = scoring.summarize(rows, "four_seed")
    parent.write(parent.RESULTS / f"{STUDY}_fresh.json", {"study": STUDY, "rows": rows,
        "summary": summary, "baseline": prior["summaries"]["baseline"]["ensemble"],
        "two_seed": prior["summaries"]["feed_augmented"]["ensemble"], "prediction_sha256": files,
        "new_cells": 10, "reused_clean_cells": 5, "selection_from_fresh": False,
        "independent_match_confirmation": False, "organizer_reference_access": False})
    print(summary["feed_like"]["equal_provider_mean"]["spell_f1"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("declare", "fit", "seal", "evaluate", "fresh"))
    p.add_argument("--fold", choices=parent.FOLDS)
    p.add_argument("--seed", type=int, choices=SEEDS)
    a = p.parse_args()
    fit(a.fold, a.seed) if a.action == "fit" else globals()[a.action]()
