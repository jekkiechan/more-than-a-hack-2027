"""One bounded raw-tracking augmentation experiment against calibrated D2.

Imports frozen training/inference helpers; does not alter their implementations.
No organizer input or reference is accessed. All native arrays stay private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import joblib
import numpy as np
from experiments import preparation as prep, supervised, tracking_corruption as tc
from experiments import stress_components, multisource_eval as ms
from experiments import soccernet_extension as old, soccernet_capacity as sc
from experiments import sn_dense_train as sn, soccernet_dense_labels as dl
from experiments import combined_objective_deployment as deployment
from experiments import spell_mbr_run as shared, decoder_search_run as scoring
from experiments import feed_spell_push as decoder, feed_spell_push_run as selection

STUDY = "feed_robustness_v1"
OUT = ROOT / "data/experiments" / STUDY
RESULTS = ROOT / "experiments/results"
DECLARATION = RESULTS / f"{STUDY}_declaration.json"
TRAIN_RECORDS = ROOT / "data/experiments/dense_label_weight_v1/records"
FOLDS, SEEDS = sc.FOLDS, (2026, 2027)
COMPONENTS = ("coordinates", "ball_gaps", "slot_gaps", "id_swaps")
CONFIG = next(c for c in decoder.configurations() if c["id"] == "f062")
TRAIN_SEED = 551103


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    scoring.write_once(path, value)


def sha(path):
    return prep.sha256(path)


def raw_view(inputs, calibration, seed):
    """Apply only label-free defects, preserving native observation masks."""
    native = tc.inputs_only(inputs)
    if "ball_observed" in native:
        native["ball"][~np.asarray(native["ball_observed"], dtype=bool)] = np.nan
    if "player_observed" in native:
        native["players"][~np.asarray(native["player_observed"], dtype=bool)] = np.nan
    return stress_components.corrupt(native, calibration, seed, COMPONENTS)


def replace_view(records, features):
    """Replace variant-1 direct/swap features; keep every target/window intact."""
    if len(records) != 4:
        raise ValueError("Expected four legacy paired records")
    out = [dict(r) for r in records]
    x = np.asarray(features)
    if x.shape != np.asarray(out[2]["x"]).shape:
        raise ValueError("Augmented features must align exactly")
    for i, swap in ((2, False), (3, True)):
        value = supervised.swap_team_features(x, out[i]["names"]) if swap else x
        out[i].update(x=value, alternate=value, physical_x=value)
    return out


def training(fold, augmented=True):
    records = []
    spec = old.old_spec()
    for item in prep.split_plan(fold=fold)["train"]:
        entry = spec["prepared_training_caches"][item["match"]]
        path = old.DATA / Path(entry["cache"]).relative_to("data")
        if sha(path) != entry["sha256"]:
            raise ValueError("Original training cache changed")
        cache = joblib.load(path)
        paired = deployment.paired_records(cache, item["match"])
        if augmented:
            view = joblib.load(OUT / "views" / f"{item['match']}.joblib")
            for key in ("indices", "period", "y", "names"):
                np.testing.assert_array_equal(view[key], cache["records"]["clean"][key])
            paired = replace_view(paired, view["x"])
        records.extend(paired)
    excluded = fold.removeprefix("leave_out_")
    if any(str(r["source"]) == excluded for r in records):
        raise ValueError("Held-out provider leaked into training")
    return records


def prepare():
    """TRAIN-only feature preparation. No fitting or development scoring."""
    if (OUT / "preparation.json").exists():
        raise FileExistsError("Preparation already complete")
    calibration = sn.default_ball_calibration()
    receipts = {}
    for item in prep.split_plan()["train"]:
        path = prep.data_path(item, old.DATA)
        if sha(path) != item["sha256"]:
            raise ValueError("Raw TRAIN checksum mismatch")
        inputs, labels = ms.view(path)
        seed = int(hashlib.sha256(f"{STUDY}/{TRAIN_SEED}/{item['match']}".encode()).hexdigest()[:8], 16)
        viewed = raw_view(inputs, calibration, seed)
        record = prep.build_record(viewed, labels, item["source"], item["match"])
        dest = OUT / "views" / f"{item['match']}.joblib"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise FileExistsError(dest)
        joblib.dump(record, dest)
        receipts[item["match"]] = {"raw_sha256": sha(path), "seed": seed,
            "path": str(dest.relative_to(ROOT)), "sha256": sha(dest),
            "source": item["source"], "rows": len(record["x"])}
        print("PREPARED", item["match"], len(record["x"]), flush=True)
    write(OUT / "preparation.json", {"study": STUDY, "train_only": True,
          "calibration": calibration, "components": COMPONENTS, "files": receipts})


def decode(p, period, fps):
    return decoder.decode(np.asarray(p)[None], period, fps, CONFIG)


def score(c, p):
    prediction = decode(p, c["model_period"], c["fps"])
    row = {k: c[k] for k in ("provider", "match", "view", "member")}
    row.update(shared._score_prediction(c["labels"], prediction, c["period"], c["timestamps"], c["fps"]))
    return row, prediction


def infer(c, model, teacher):
    """The feature/model path accepts no labels or truth-derived validity mask."""
    rel = shared._scalar(c["record"], "evaluation_input_path")
    cached = joblib.load(shared._resolve_artifact(rel))
    record = old.inference_record(cached["record"])
    return old.upsample(sn.predict(record, model, teacher), record, tc.inputs_only(cached["inputs"]))


def source_hashes():
    # Bind all transitive experiment implementations conservatively.
    paths = sorted((ROOT / "experiments").glob("*.py"))
    paths += [ROOT / "evaluation/possession_metrics.py", ROOT / "experiments/eval_set/manifest.json",
              ROOT / "experiments/baseline_manifest.json", ROOT / "experiments/model_decision_policy.json"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def freeze():
    upstream = read(RESULTS / "feed_spell_push_v1_declaration.json")
    inputs = dict(upstream["prediction_files"]) | upstream["evaluation_input_hashes"] | upstream["model_checkpoint_hashes"]
    preparation = read(OUT / "preparation.json")
    inputs.update({v["path"]: v["sha256"] for v in preparation["files"].values()})
    dense_manifest = TRAIN_RECORDS / "train/manifest.json"
    dm = read(dense_manifest)
    if sha(dense_manifest) != "1f905524123f63c60bd12bd42150c1d111dc779c1760f8d9b4a725077f3706d5":
        raise ValueError("Wrong dense TRAIN archive")
    inputs[str(dense_manifest.relative_to(ROOT))] = sha(dense_manifest)
    for name, digest in dm["game_files"].items():
        inputs[str((TRAIN_RECORDS / "train/games" / name).relative_to(ROOT))] = digest
    for entry in old.old_spec()["prepared_training_caches"].values():
        inputs[entry["cache"]] = entry["sha256"]
    for fold in FOLDS:
        for seed in SEEDS:
            path = sc.initial_path(fold, seed)
            expected = read(RESULTS / "dense_label_weight_v1_declaration.json")["restore"]["fold_teacher_sha256"][f"{fold}/{seed}"]
            inputs["data/" + path.relative_to(old.DATA).as_posix()] = expected
    for rel, digest in inputs.items():
        if sha(shared._resolve_artifact(rel)) != digest:
            raise ValueError("Input checksum failed: " + rel)
    for fold in FOLDS:
        training(fold)  # Exact sample/target/native alignment before freezing.
    write(DECLARATION, {"study": STUDY, "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does replacing the legacy slot/coarse training view with raw feed-like defects improve calibrated dense-GRU spell F1?",
        "baseline": "joint_gru_sn_dense_calibrated_ensemble_v1; exact archived provider-excluded D2 models with f062 decoder",
        "arms": ["baseline", "feed_augmented", "fixed_half_blend"],
        "fits": 6, "folds": list(FOLDS), "seeds": list(SEEDS), "updates": 2048,
        "training": "Unchanged sn_dense_train D2: width96, lr3e-4, D-lr1e-4, batch32, GAR fraction .5, 45 TRAIN games. Replace second original-provider view and its alternate only. Clean first view and its gap alternate retained. Same sampler, labels, objectives, normalization, H/A pairing, teacher and fixed budget.",
        "blend": "Equal arithmetic mean of baseline and candidate native probabilities, then fixed D-temperature4/cost3. Both seeds and their probability ensembles scored.",
        "train_plan": prep.split_plan(), "provider_plans": {f: prep.split_plan(fold=f) for f in FOLDS},
        "calibration": preparation["calibration"], "components": COMPONENTS,
        "primary": "Equal-provider feed_like official spell F1; macro breaks exact ties; .55 stretch",
        "guards": {"clean_spell_floor_delta": -.005, "each_provider_clean_macro_and_d_floor_delta": -.02,
                   "each_provider_feed_spell_floor_delta": -.01, "historical_clean_plus_01_screen": "report separately"},
        "stop": "Six complete fits; all sealed before scoring. No extra temperatures, mixture weights, seeds or training steps within this study.",
        "confirmation": "If a guard-passing arm improves feed spell, compare that fixed arm and baseline on two fresh corruption realizations per development match under a separately frozen label-free preparation; no reselection.",
        "evidence": "Five repeatedly exposed development matches; synthetic stress; no independent confirmation; GAR and two provider targets include silver labels.",
        "organizer_reference_access": False, "promotion": False,
        "replay_tolerances": {"saved_score": 1e-12, "runtime_probability": 2e-6, "runtime_changed_labels": 0},
        "source_sha256": source_hashes(), "input_sha256": inputs,
        "prediction_files": upstream["prediction_files"], "evaluation_input_hashes": upstream["evaluation_input_hashes"],
        "model_checkpoint_hashes": upstream["model_checkpoint_hashes"],
        "preparation_sha256": sha(OUT / "preparation.json")})


def check():
    d = read(DECLARATION)
    for rel, digest in d["source_sha256"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("Frozen source changed: " + rel)
    for rel, digest in d["input_sha256"].items():
        if sha(shared._resolve_artifact(rel)) != digest:
            raise ValueError("Frozen input changed: " + rel)
    return d


def preflight():
    d = check()
    cells = shared._load_cells(d)
    rows = [score(c, c["probabilities"])[0] for c in cells]
    summary = scoring.summarize(rows)
    prior = read(RESULTS / "feed_spell_push_v1_aggregate.json")["arms"]["f062"]
    for view in ("clean", "feed_like"):
        for key in ("spell_f1", "macro_f1", "d_f1"):
            if abs(summary[view]["equal_provider_mean"][key] - prior[view]["equal_provider_mean"][key]) > 1e-12:
                raise ValueError("Calibrated baseline replay failed")
    c = next(c for c in cells if c["provider"] == "Metrica" and c["member"] == "2026" and c["view"] == "clean")
    model_path = ROOT / "data/experiments/dense_label_weight_v1/models/control_D2__metrica__s2026.joblib"
    teacher = joblib.load(sc.initial_path("leave_out_metrica", 2026))
    p = infer(c, joblib.load(model_path), teacher)
    difference = float(np.abs(p - c["probabilities"]).max())
    changed = int(np.sum(decode(p, c["model_period"], c["fps"]) != decode(c["probabilities"], c["model_period"], c["fps"])))
    if difference > 2e-6 or changed:
        raise ValueError(f"Runtime replay failed: probability {difference}, labels {changed}")
    original = old.load_old("leave_out_metrica")
    control = training("leave_out_metrica", augmented=False)
    for a, b in zip(original, control, strict=True):
        for key in ("x", "alternate", "y", "indices", "period", "full_y", "full_period", "replace_windows"):
            np.testing.assert_array_equal(a[key], b[key])
    write(RESULTS / f"{STUDY}_control_replay.json", {"passed": True, "cells": 45,
        "runtime_probability_max_difference": difference, "changed_runtime_labels": changed,
        "training_wrapper_exact": True, "baseline": summary, "rows": rows})


def fit(fold, seed):
    check()
    if not read(RESULTS / f"{STUDY}_control_replay.json")["passed"]:
        raise ValueError("Control replay required")
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Undeclared fit")
    dest = OUT / "models" / f"{fold}_{seed}.joblib"
    write(dest.with_suffix(".lock.json"), {"declaration_sha256": sha(DECLARATION)})
    teacher = joblib.load(sc.initial_path(fold, seed))
    records = training(fold)
    gar = dl.load_dense_records("train", output=TRAIN_RECORDS)
    started = time.monotonic()
    result = sn.train(teacher, records, gar, width=96, lr=3e-4, seed=seed, steps=2048,
                      fraction=.5, batch_size=32, ha_target="dense", train_d=True,
                      require_full_native=True, log_every=256, verbose=True)
    if dest.exists():
        raise FileExistsError(dest)
    joblib.dump(result["model"], dest)
    write(dest.with_suffix(".json"), {"fold": fold, "seed": seed, "sha256": sha(dest),
        "declaration_sha256": sha(DECLARATION), "wall_seconds": time.monotonic()-started,
        "old_matches": sorted({str(r["match"]) for r in records}),
        "training_config": result["config"], "history": result.get("history", []),
        "exposures": result.get("exposures", {})})
    print("FIT COMPLETE", fold, seed, flush=True)


def seal():
    check()
    models = {}
    for fold in FOLDS:
        for seed in SEEDS:
            p = OUT / "models" / f"{fold}_{seed}.joblib"
            receipt = read(p.with_suffix(".json"))
            if sha(p) != receipt["sha256"] or receipt["declaration_sha256"] != sha(DECLARATION):
                raise ValueError("Fit seal mismatch")
            models[str(p.relative_to(ROOT))] = sha(p)
    write(RESULTS / f"{STUDY}_model_seal.json", {"models": models, "declaration_sha256": sha(DECLARATION)})


def evaluate():
    d = check()
    seal_value = read(RESULTS / f"{STUDY}_model_seal.json")
    for rel, digest in seal_value["models"].items():
        if sha(ROOT / rel) != digest:
            raise ValueError("Sealed model changed")
    write(OUT / "evaluation_lock.json", {"model_seal_sha256": sha(RESULTS / f"{STUDY}_model_seal.json")})
    cells = shared._load_cells(d)
    probabilities = {}
    for provider in shared.PROVIDERS:
        fold = "leave_out_" + provider.lower()
        for seed in SEEDS:
            model = joblib.load(OUT / "models" / f"{fold}_{seed}.joblib")
            teacher = joblib.load(sc.initial_path(fold, seed))
            for c in cells:
                if c["provider"] == provider and c["member"] == str(seed):
                    key = tuple(c[k] for k in ("provider", "match", "view", "member"))
                    probabilities[key] = infer(c, model, teacher)
                    print("INFERRED", key, flush=True)
    rows = {a: [] for a in ("baseline", "feed_augmented", "fixed_half_blend")}
    files = {}
    for c in cells:
        key = tuple(c[k] for k in ("provider", "match", "view", "member"))
        p = (np.mean([probabilities[key[:-1]+(str(s),)] for s in SEEDS], axis=0)
             if c["member"] == "ensemble" else probabilities[key])
        values = {"baseline": c["probabilities"], "feed_augmented": p,
                  "fixed_half_blend": .5*(p+c["probabilities"])}
        for arm, value in values.items():
            row, prediction = score(c, value)
            rows[arm].append(row)
            if arm != "baseline":
                dest = OUT / "predictions" / arm / ("__".join(key) + ".npz")
                dest.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(dest, probabilities=value, prediction=prediction,
                    labels=c["labels"], period=c["period"], timestamps=c["timestamps"],
                    model_period=c["model_period"], fps=c["fps"])
                files[str(dest.relative_to(ROOT))] = sha(dest)
        for a in values:
            for k in ("graded_frames", "reference_spell_count", "short_reference_spell_count"):
                if rows[a][-1][k] != rows["baseline"][-1][k]:
                    raise ValueError("Grading population changed")
    summaries = {a: {m: scoring.summarize(r, m) for m in shared.MEMBERS} for a, r in rows.items()}
    baseline = summaries["baseline"]["ensemble"]
    replay = read(RESULTS / f"{STUDY}_control_replay.json")
    if rows["baseline"] != replay["rows"]:
        raise ValueError("Baseline replay changed between preflight and evaluation")
    checks = {a: selection.eligible(v["ensemble"], baseline) for a, v in summaries.items()}
    winner = max((a for a in summaries if checks[a]), key=lambda a: selection.rank(summaries[a]["ensemble"]))
    write(RESULTS / f"{STUDY}_aggregate.json", {"study": STUDY, "rows": rows, "summaries": summaries,
        "guards_pass": checks, "winner": winner,
        "historical_clean_screen_pass": {a: v["ensemble"]["clean"]["equal_provider_mean"]["spell_f1"] >= baseline["clean"]["equal_provider_mean"]["spell_f1"] + .01 for a, v in summaries.items()},
        "stretch_reached": summaries[winner]["ensemble"]["feed_like"]["equal_provider_mean"]["spell_f1"] >= .55,
        "prediction_sha256": files, "model_seal_sha256": sha(RESULTS / f"{STUDY}_model_seal.json"),
        "organizer_reference_access": False, "independent_confirmation": False})
    for a, v in summaries.items():
        print(a, {view: v["ensemble"][view]["equal_provider_mean"]["spell_f1"] for view in ("clean", "feed_like")}, checks[a], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "preflight", "fit", "seal", "evaluate"))
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.action == "fit":
        fit(args.fold, args.seed)
    else:
        globals()[args.action]()


if __name__ == "__main__":
    main()
