"""Original-possession diversity training with four calibrated v2 input views.

This runner keeps the fixed original-provider/GAR training budget and sampler
from ``short_original_diversity_v1``. It replaces the single v1 synthetic view
in the original-provider half of each batch with four fixed v2 realizations,
selected round-robin per match. Clean paired records remain at their existing
mass. Only TRAIN tracking and labels place the simulated defects; labels never
enter the feature builder or inference path.

The runner writes only ignored data artifacts and this new experiment's result
files. It does not read organizer labels or modify a frozen runner/declaration.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

import joblib
import numpy as np

from experiments import feed_robustness_v1 as rf
from experiments import feed_like_v2 as v2
from experiments import multisource_eval as ms
from experiments import preparation as prep
from experiments import short_original_diversity as sampler_module
from experiments import short_original_diversity_v1 as short

STUDY = "v2_short_original_diversity"
ROOT = rf.ROOT
OUT = ROOT / "data/experiments" / STUDY
RESULTS = ROOT / "experiments/results"
CALIBRATION = RESULTS / "feed_like_v2_calibration.json"
DECLARATION = RESULTS / f"{STUDY}_declaration.json"
N_REALIZATIONS = 4
VIEW_SEED = 20261103
FOLDS, SEEDS = rf.FOLDS, rf.SEEDS
TRAIN_RECORDS = Path(os.environ.get(
    "SOCCERNET_GAR_TRAIN_RECORDS",
    str(ROOT / "data/experiments/dense_label_weight_v1/records")))
ARCHIVE_ROOT = Path(os.environ.get("MTAH_RF_ARCHIVE_DIR", ""))


def _archive_root() -> Path:
    candidates = [
        ARCHIVE_ROOT,
        Path(os.environ.get("MTAH_DATA_DIR", ROOT.parent / "mtah-data")) /
        "experiments/feed_robustness_v1/artifacts",
        ROOT / "data/experiments/feed_robustness_v1",
    ]
    for candidate in candidates:
        if candidate and (candidate / "views").is_dir():
            return candidate
    raise FileNotFoundError(
        "Feed robustness archive views are unavailable; set MTAH_RF_ARCHIVE_DIR "
        "to its verified artifacts directory")


def _view_path(match: str, realization: int) -> Path:
    return OUT / "views" / f"{match}__r{realization}.joblib"


def _view_seed(match: str, realization: int) -> int:
    return int(hashlib.sha256(
        f"{STUDY}/{VIEW_SEED}/{match}/{realization}".encode()
    ).hexdigest()[:8], 16)


def _validate_view_record(record: dict, cache: dict, match: str) -> None:
    clean = cache["records"]["clean"]
    for key in ("indices", "period", "y"):
        np.testing.assert_array_equal(record[key], clean[key], err_msg=f"{match}: {key}")
    if list(record["names"]) != list(clean["names"]):
        raise ValueError(f"Feature schema changed: {match}")
    if np.asarray(record["x"]).shape != np.asarray(clean["x"]).shape:
        raise ValueError(f"Feature shape changed: {match}")


def _cache_for(match: str, spec: dict) -> tuple[dict, str]:
    entry = spec["prepared_training_caches"][match]
    path = rf.old.DATA / Path(entry["cache"]).relative_to("data")
    digest = rf.sha(path)
    if digest != entry["sha256"]:
        raise ValueError(f"Prepared source cache checksum mismatch: {match}")
    return joblib.load(path), str(path)


def _v1_view_manifest() -> tuple[dict, Path]:
    archive = _archive_root()
    root = archive.parents[2]
    manifest_path = root / "feed_robustness_v1_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if rf.sha(manifest_path) != "2bf34af52ce645d9454f7da8186d5aa155e8736b51ee720a58567bac67807c4e":
        raise ValueError("Unexpected feed_robustness_v1 archive manifest")
    entries = {entry["path"]: entry for entry in manifest["files"]}
    return entries, manifest_path


def _verified_v1_view(match: str, entries: dict) -> tuple[dict, str, str]:
    archive = _archive_root()
    path = archive / "views" / f"{match}.joblib"
    rel = path.relative_to(archive.parents[2]).as_posix()
    entry = entries.get(rel)
    if entry is None:
        raise ValueError(f"v1 training view is not in verified archive manifest: {rel}")
    digest = rf.sha(path)
    if digest != entry["sha256"] or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"v1 training view hash mismatch: {rel}")
    return joblib.load(path), rel, digest


def _prepare_match(item: dict, calibration: dict, runner_sha: str) -> list[dict]:
    """Worker: build or verify all fixed v2 realizations for one TRAIN match."""
    raw_path = prep.data_path(item, rf.old.DATA)
    raw_hash = rf.sha(raw_path)
    if raw_hash != item["sha256"]:
        raise ValueError(f"Raw TRAIN checksum mismatch: {item['match']}")
    inputs, labels = ms.view(raw_path)
    cache, _ = _cache_for(item["match"], rf.old.old_spec())
    result = []
    for realization in range(N_REALIZATIONS):
        seed = _view_seed(item["match"], realization)
        dest = _view_path(item["match"], realization)
        if dest.exists():
            record = joblib.load(dest)
            receipt_path = Path(str(dest) + ".json")
            receipt = json.loads(receipt_path.read_text())
            expected = dict(raw_sha256=raw_hash, seed=seed, match=item["match"],
                            source=item["source"], realization=realization,
                            calibration_sha256=rf.sha(CALIBRATION),
                            v2_source_sha256=rf.sha(v2.__file__), sha256=rf.sha(dest),
                            rows=len(record["x"]), train_only=True)
            if (not receipt.get("runner_sha256") or any(
                    receipt.get(key) != value for key, value in expected.items())):
                raise ValueError(f"Existing v2 view receipt mismatch: {dest}")
            _validate_view_record(record, cache, item["match"])
            result.append(receipt)
            continue
        viewed = v2.corrupt(inputs, labels, calibration, seed)
        record = prep.build_record(viewed, labels, item["source"], item["match"])
        _validate_view_record(record, cache, item["match"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(record, dest)
        receipt = dict(match=item["match"], source=item["source"], realization=realization,
                       seed=seed, raw_sha256=raw_hash, sha256=rf.sha(dest),
                       rows=int(len(record["x"])), train_only=True,
                       calibration_sha256=rf.sha(CALIBRATION),
                       v2_source_sha256=rf.sha(v2.__file__), runner_sha256=runner_sha)
        Path(str(dest) + ".json").write_text(json.dumps(receipt, indent=1))
        result.append(receipt)
        print("PREPARED", item["match"], realization, flush=True)
    return result


def prepare(workers: int = 3) -> dict:
    """Build four fixed v2 feature records per TRAIN match in parallel."""
    calibration = json.loads(CALIBRATION.read_text())["calibration"]
    plan = prep.split_plan()
    if len(plan["train"]) != 13:
        raise ValueError("Expected the fixed 13-match original-provider TRAIN pool")
    runner_sha = rf.sha(__file__)
    receipts = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_prepare_match, item, calibration, runner_sha): item
                   for item in plan["train"]}
        for future in as_completed(futures):
            item = futures[future]
            receipts[item["match"]] = future.result()
    receipts = {match: sorted(values, key=lambda row: row["realization"])
                for match, values in sorted(receipts.items())}
    out = {"study": STUDY, "realizations_per_original_match": N_REALIZATIONS,
           "workers": workers, "runner_sha256": runner_sha,
           "view_runner_sha256": {
               match: {str(row["realization"]): row["runner_sha256"] for row in values}
               for match, values in receipts.items()},
           "calibration_sha256": rf.sha(CALIBRATION),
           "train_matches": receipts, "organizer_reference_access": False}
    manifest = OUT / "preparation.json"
    if manifest.exists():
        prior = json.loads(manifest.read_text())
        if prior != out:
            raise FileExistsError("Preparation manifest exists with different content")
    else:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(out, indent=1))
    return out


def _load_v2_features(match: str) -> list[np.ndarray]:
    features = []
    spec = rf.old.old_spec()
    cache, _ = _cache_for(match, spec)
    item = next(row for row in prep.split_plan()["train"] if row["match"] == match)
    raw_path = prep.data_path(item, rf.old.DATA)
    raw_hash = rf.sha(raw_path)
    preparation = json.loads((OUT / "preparation.json").read_text())
    runner_sha = preparation["view_runner_sha256"][match]
    for realization in range(N_REALIZATIONS):
        path = _view_path(match, realization)
        if not path.exists() or not Path(str(path) + ".json").exists():
            raise FileNotFoundError(f"Run prepare first; missing {path}")
        record = joblib.load(path)
        receipt = json.loads(Path(str(path) + ".json").read_text())
        expected = dict(match=match, source=item["source"], realization=realization,
                        seed=_view_seed(match, realization), raw_sha256=raw_hash,
                        sha256=rf.sha(path), rows=len(record["x"]), train_only=True,
                        calibration_sha256=rf.sha(CALIBRATION),
                        v2_source_sha256=rf.sha(v2.__file__),
                        runner_sha256=runner_sha[str(realization)])
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(f"v2 view receipt mismatch: {path}")
        _validate_view_record(record, cache, match)
        features.append(np.asarray(record["x"]))
    return features


def training(fold: str) -> list[dict]:
    """Load the v1 short-original-diversity base and attach four v2 views.

    Records, original/GAR split, native targets, window IDs and sampler inputs
    stay fixed. A private feature dispatch list is added only to the two v1
    records whose feature view is replaced by v2.
    """
    if fold not in FOLDS:
        raise ValueError(f"Unknown fold: {fold}")
    archive = _archive_root()
    v1_entries, _ = _v1_view_manifest()
    records = []
    spec = rf.old.old_spec()
    for item in prep.split_plan(fold=fold)["train"]:
        cache, _ = _cache_for(item["match"], spec)
        paired = rf.deployment.paired_records(cache, item["match"])
        view, _, _ = _verified_v1_view(item["match"], v1_entries)
        for key in ("indices", "period", "y", "names"):
            if key == "names":
                if list(view[key]) != list(cache["records"]["clean"][key]):
                    raise ValueError(f"v1 feature schema mismatch: {item['match']}")
            else:
                np.testing.assert_array_equal(view[key], cache["records"]["clean"][key])
        paired = rf.replace_view(paired, view["x"])
        v2_features = _load_v2_features(item["match"])
        for realization, features in enumerate(v2_features):
            if features.shape != np.asarray(paired[2]["x"]).shape:
                raise ValueError(f"v2 feature shape mismatch: {item['match']} r{realization}")
            if list(joblib.load(_view_path(item["match"], realization))["names"]) != list(paired[2]["names"]):
                raise ValueError(f"v2 feature names mismatch: {item['match']} r{realization}")
        for index, swap in ((2, False), (3, True)):
            oriented = [
                rf.supervised.swap_team_features(value, paired[index]["names"]) if swap else value
                for value in v2_features
            ]
            paired[index].update(x=oriented[0], alternate=oriented[0], physical_x=oriented[0])
            paired[index]["_v2_feature_views"] = oriented
            paired[index]["_v2_view_match"] = item["match"]
        records.extend(paired)
    excluded = fold.removeprefix("leave_out_")
    if any(str(r["source"]) == excluded for r in records):
        raise ValueError(f"Held-out provider leaked into training: {fold}")
    return records


def _source_files() -> list[Path]:
    return [
        Path(__file__),
        Path(rf.__file__), Path(v2.__file__), Path(ms.__file__), Path(prep.__file__),
        Path(rf.sn.__file__), Path(rf.sc.__file__), Path(rf.dl.__file__),
        Path(rf.deployment.__file__), Path(short.__file__),
        Path(sampler_module.__file__), Path(short.parent.__file__),
        ROOT / "experiments/short_label_quality.py",
        ROOT / "experiments/feed_short_sampler.py",
        ROOT / "experiments/dense_short_team_v1.py",
        ROOT / "experiments/soccernet_capacity_model.py",
        ROOT / "experiments/factorized_had.py",
        ROOT / "experiments/objective_losses.py",
        ROOT / "experiments/refine_features.py",
        ROOT / "experiments/learned_temporal.py",
        ROOT / "experiments/supervised.py",
        ROOT / "experiments/baseline_manifest.json",
        ROOT / "experiments/model_decision_policy.json",
        RESULTS / "feed_like_v2_calibration.json",
        RESULTS / "short_original_diversity_v1_declaration.json",
        RESULTS / "short_original_diversity_v1_training_audit.json",
        RESULTS / "feed_like_v2_train.json",
    ]


def _view_pipeline_files() -> list[Path]:
    return [
        Path(v2.__file__), Path(prep.__file__), Path(ms.__file__),
        ROOT / "experiments/refine_features.py",
        ROOT / "experiments/learned_temporal.py",
        ROOT / "experiments/supervised.py",
        ROOT / "experiments/tracking_corruption.py",
        ROOT / "experiments/stress_components.py",
        ROOT / "experiments/soccernet_extension.py",
        ROOT / "experiments/datasets.py",
    ]


def freeze() -> dict:
    """Seal the fixed four-view, six-fit comparison after preparation."""
    if not (OUT / "preparation.json").is_file():
        raise FileNotFoundError("Run prepare before freeze")
    preparation = json.loads((OUT / "preparation.json").read_text())
    if preparation.get("calibration_sha256") != rf.sha(CALIBRATION):
        raise ValueError("Preparation calibration hash changed")
    plan = prep.split_plan()
    if set(preparation["train_matches"]) != {row["match"] for row in plan["train"]}:
        raise ValueError("Prepared match inventory differs from the fixed TRAIN plan")
    old_spec = rf.old.old_spec()
    v1_entries, v1_manifest = _v1_view_manifest()
    source_hashes = {}
    for path in _source_files():
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)] = rf.sha(path)
    view_pipeline = {str(path.relative_to(ROOT)): rf.sha(path)
                     for path in _view_pipeline_files()}
    for relative, digest in view_pipeline.items():
        source_hashes[relative] = digest
    inputs = {str((OUT / "preparation.json").relative_to(ROOT)): rf.sha(OUT / "preparation.json"),
              str(v1_manifest): rf.sha(v1_manifest)}
    aborted_receipt = OUT / "aborted_preparation.json"
    if aborted_receipt.exists():
        inputs[str(aborted_receipt.relative_to(ROOT))] = rf.sha(aborted_receipt)
    current_runner = rf.sha(__file__)
    view_runners = sorted({digest for views in preparation["view_runner_sha256"].values()
                           for digest in views.values()})
    if len(preparation["view_runner_sha256"]) != len(plan["train"]) or any(
            len(views) != N_REALIZATIONS for views in preparation["view_runner_sha256"].values()):
        raise ValueError("Preparation view-runner inventory is incomplete")
    if set(view_runners) != {current_runner}:
        amendment = dict(
            reason="Fifty valid views were generated under the initial runner wrapper and one stale realization was rejected. The remaining missing realization and replacement were generated under the frozen wrapper. The v2 corruption and feature-builder source closure, input bytes, seeds, and calibration are hash-bound and identical across both batches.",
            view_runner_sha256=view_runners, frozen_runner_sha256=current_runner,
            calibration_sha256=rf.sha(CALIBRATION), view_pipeline_sha256=view_pipeline,
            aborted_prefix_receipt_sha256=rf.sha(OUT / "aborted_parallel_preparation.json"),
            changed_artifacts="the stale r2 artifact is preserved under aborted_parallel_prefix; 50 verified views were reused and missing/rejected views regenerated")
        amendment_path = OUT / "preparation_amendment.json"
        if amendment_path.exists():
            previous = json.loads(amendment_path.read_text())
            if previous != amendment:
                previous_bytes = amendment_path.read_bytes()
                previous_hash = hashlib.sha256(previous_bytes).hexdigest()
                attempt_path = OUT / f"preparation_amendment_attempt_{previous_hash[:12]}.json"
                if not attempt_path.exists():
                    attempt_path.write_bytes(previous_bytes)
                inputs[str(attempt_path.relative_to(ROOT))] = rf.sha(attempt_path)
                amendment_path.write_text(json.dumps(amendment, indent=1, sort_keys=True))
        else:
            amendment_path.write_text(json.dumps(amendment, indent=1, sort_keys=True))
        inputs[str(amendment_path.relative_to(ROOT))] = rf.sha(amendment_path)
    else:
        amendment = None
    for item in plan["train"]:
        raw_path = prep.data_path(item, rf.old.DATA)
        inputs[str(raw_path)] = rf.sha(raw_path)
        cache_entry = old_spec["prepared_training_caches"][item["match"]]
        cache_path = rf.old.DATA / Path(cache_entry["cache"]).relative_to("data")
        inputs[str(cache_path)] = rf.sha(cache_path)
        v1_path = _archive_root() / "views" / f"{item['match']}.joblib"
        rel = v1_path.relative_to(_archive_root().parents[2]).as_posix()
        if rel not in v1_entries or rf.sha(v1_path) != v1_entries[rel]["sha256"]:
            raise ValueError(f"v1 input view failed archive binding: {item['match']}")
        inputs[str(v1_path)] = rf.sha(v1_path)
        for realization in range(N_REALIZATIONS):
            view_path = _view_path(item["match"], realization)
            receipt_path = Path(str(view_path) + ".json")
            if not view_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(view_path)
            inputs[str(view_path.relative_to(ROOT))] = rf.sha(view_path)
            inputs[str(receipt_path.relative_to(ROOT))] = rf.sha(receipt_path)
    gar_manifest = TRAIN_RECORDS / "train/manifest.json"
    if not gar_manifest.is_file():
        raise FileNotFoundError(gar_manifest)
    inputs[str(gar_manifest)] = rf.sha(gar_manifest)
    gar_spec = json.loads(gar_manifest.read_text())
    for name, digest in gar_spec["game_files"].items():
        path = TRAIN_RECORDS / "train/games" / name
        actual = rf.sha(path)
        if actual != digest:
            raise ValueError(f"GAR TRAIN record hash mismatch: {name}")
        inputs[str(path)] = actual
    teachers = {}
    gar = rf.dl.load_dense_records("train", output=TRAIN_RECORDS)
    for fold in FOLDS:
        # Materialize records and verify all four views remain provider-excluded.
        records = training(fold)
        for seed in SEEDS:
            teacher = joblib.load(rf.sc.initial_path(fold, seed))
            sampler = short.OriginalDiversitySampler(
                records, gar, list(teacher["feature_names"]), seed)
            short.parent.base.replay(sampler, seed)
            saved = next(row["sampling"] for row in rf.read(short.AUDIT)["rows"]
                         if row["fold"] == fold and row["seed"] == seed)
            if sampler.receipt() != saved:
                raise ValueError(f"Original diversity sampler parity failed: {fold}/{seed}")
            teacher_path = rf.sc.initial_path(fold, seed)
            teachers[f"{fold}/{seed}"] = rf.sha(teacher_path)
    declaration = dict(
        study=STUDY, freeze_schema=1,
        question="Does replacing the original-provider v1 tracking augmentation with four calibrated feed_like_v2 realizations, while retaining original-possession diversity sampling, produce a guard-passing improvement on the fixed feed_like_v2 development protocol?",
        intervention="Keep the fixed original_full raw-feed-v1 base and the exact short_original_diversity_v1 sampler. Replace only the two original-provider v1 records per match with four fixed v2 feature realizations selected round-robin per sampled original window. Preserve clean record mass and v2 mass (each view equally represented in v2 draws). GAR view, labels, masks, sampling schedule, normalization, architecture, losses, optimizer and decoder remain fixed.",
        fits=6, folds=list(FOLDS), seeds=list(SEEDS), updates=2048,
        original_matches=sorted(preparation["train_matches"]),
        provider_plans={fold: prep.split_plan(fold=fold) for fold in FOLDS},
        training={"width":96,"batch_size":32,"fraction_gar":0.5,"lr_ha":3e-4,
                  "lr_d":1e-4,"dense_ha":True,"train_d":True,
                  "native_objective":True,"normalization":"reuse exact fold teacher median/scale",
                  "gar_train_records":str(TRAIN_RECORDS),
                  "sampler":"OriginalDiversitySampler; identical fixed GAR quality/diversity and original short-possession bag policy"},
        v2={"realizations_per_match":N_REALIZATIONS,"seed_base":VIEW_SEED,
            "calibration_sha256":rf.sha(CALIBRATION),"placement":"TRAIN labels only",
            "features":"tracking inputs only; no label fields passed to feature builder",
            "generator_source_sha256":view_pipeline,"preparation_amendment":amendment},
        evaluation={"protocol":"same five exposed original-provider development matches; clean plus four fixed target-independent feed_like_v2 realizations per match; average realizations within match, matches within provider, providers equally",
                   "decoder":"fixed f062 cost-3 decoder; five-phase probability averaging for all arms",
                   "arms":[
                       "new_full_phase_tta: new model's full H/A/D probabilities",
                       "new_conditional_ha_x_archived_calibrated_d: within-seed new conditional H/A crossed with archived calibrated D probabilities",
                       "new_full_plus_archived_calibrated_equal_blend: arithmetic 50/50 blend of per-seed phase-mean probabilities with archived calibrated phase-mean probabilities, then seed mean"],
                   "controls":["archived adopted original_full phase-TTA C0",
                              "archived calibrated-D phase-TTA C0"],
                   "historical_diagnostic":"feed_like_v2_train single-v2 models reported separately and not used for selection",
                   "no_candidate_specific_decoder_or_threshold_sweep":True},
        teacher_sha256=teachers,
        source_sha256=source_hashes,input_sha256=inputs,
        train_plan=plan,organizer_reference_access=False,
        primary="Equal-provider official spell F1 on feed_like_v2 with four fixed development realizations",
        target="0.60 is exploratory; no promotion or guaranteed threshold attainment",
        controls=["adopted original_full + phase TTA","historical feed_like_v2_train single-v2 result, reported separately because its sampler differs"],
        stop="Six fixed provider-fold/seed fits; no follow-on seed, rate, decoder, or v2-view-count sweep under this declaration.",
        limitations="Five exposed development matches and synthetic v2 views; no independent continuous-match confirmation.",
        promotion=False)
    if DECLARATION.exists():
        prior = json.loads(DECLARATION.read_text())
        if prior != declaration:
            raise FileExistsError("Frozen declaration already exists with different content")
    else:
        DECLARATION.parent.mkdir(parents=True, exist_ok=True)
        DECLARATION.write_text(json.dumps(declaration, indent=1, sort_keys=True))
    return declaration


def check() -> dict:
    if not DECLARATION.is_file():
        raise FileNotFoundError("Run freeze before fit")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study") != STUDY or declaration.get("fits") != 6:
        raise ValueError("Declaration identity or fit count mismatch")
    for path, digest in declaration["source_sha256"].items():
        absolute = Path(path)
        if not absolute.is_absolute():
            absolute = ROOT / absolute
        if rf.sha(absolute) != digest:
            raise ValueError(f"Frozen source changed: {path}")
    for path, digest in declaration["input_sha256"].items():
        if rf.sha(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    for fold, teacher_sha in declaration["teacher_sha256"].items():
        name, seed = fold.split("/")
        if rf.sha(rf.sc.initial_path(name, int(seed))) != teacher_sha:
            raise ValueError(f"Teacher checkpoint changed: {fold}")
    return declaration


@contextmanager
def _round_robin_v2_windows(records: list[dict]):
    """Select each fixed v2 realization equally without changing sampler draws."""
    cap = rf.sn.cap
    original = cap._prepare_window
    counters: Counter[str] = Counter()
    used: Counter[str] = Counter()

    def prepare_window(record, start, size, full, alternate, *, median, scale, swap):
        views = record.get("_v2_feature_views")
        if views is None:
            return original(record, start, size, full, alternate,
                            median=median, scale=scale, swap=swap)
        match = str(record["_v2_view_match"])
        choice = counters[match] % len(views)
        counters[match] += 1
        used[f"{match}/r{choice}"] += 1
        selected = dict(record, x=views[choice], alternate=views[choice])
        return original(selected, start, size, full, False,
                        median=median, scale=scale, swap=swap)

    cap._prepare_window = prepare_window
    try:
        yield used
    finally:
        cap._prepare_window = original


def fit(fold: str, seed: int) -> dict:
    """Run one declared full-pipeline provider-excluded fit; fitting is explicit."""
    declaration = check()
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Use one of the frozen provider folds and seeds")
    records = training(fold)
    teacher = joblib.load(rf.sc.initial_path(fold, seed))
    gar = rf.dl.load_dense_records("train", output=TRAIN_RECORDS)
    sampler = short.OriginalDiversitySampler(
        records, gar, list(teacher["feature_names"]), seed)
    dest = OUT / "models" / f"{fold}_{seed}.joblib"
    if dest.exists() or Path(str(dest) + ".json").exists():
        raise FileExistsError(f"Fit output already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with _round_robin_v2_windows(records) as view_draws:
        with short.parent.base.installed(sampler):
            result = rf.sn.train(
                teacher, records, gar, width=96, lr=3e-4, seed=seed,
                steps=2048, fraction=.5, batch_size=32, ha_target="dense",
                train_d=True, require_full_native=True, log_every=256, verbose=True)
    joblib.dump(result["model"], dest)
    receipt = dict(study=STUDY, fold=fold, seed=seed, sha256=rf.sha(dest),
                   declaration_sha256=rf.sha(DECLARATION), wall_seconds=time.monotonic()-start,
                   config=result["config"], sampler=sampler.receipt(),
                   exposures=result["exposures"], v2_window_draws=dict(view_draws),
                   v2_views_per_match=N_REALIZATIONS,
                   clean_v1_record_mass="unchanged records 0/1 from matched raw-feed control",
                   old_matches=sorted({str(r["match"]) for r in records}),
                   organizer_reference_access=False)
    Path(str(dest) + ".json").write_text(json.dumps(receipt, indent=1, default=float))
    print("FIT COMPLETE", fold, seed, "seconds", receipt["wall_seconds"], flush=True)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--workers", type=int, default=3)
    sub.add_parser("freeze")
    sub.add_parser("check")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--fold", choices=FOLDS, required=True)
    fit_parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.workers)
    elif args.action == "freeze":
        freeze()
    elif args.action == "check":
        check()
    else:
        fit(args.fold, args.seed)


if __name__ == "__main__":
    main()
