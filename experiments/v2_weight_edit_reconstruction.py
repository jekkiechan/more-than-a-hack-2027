"""Reconstruct the PR82 four-view provider-fold model recipe under a new ID.

PR82's locally verified fold checkpoints and prepared v2 TRAIN views are not in
the reachable archive.  This runner imports its immutable recipe helpers,
recreates the exact deterministic v2 feature views, and fits six new
provider-excluded checkpoints.  New hashes are recorded; byte-exact recovery is
not claimed.  No candidate scoring or weight edit is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from experiments import v2_short_original_diversity as recipe
from experiments import weight_space_edit

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_weight_edit_reconstruction_v1"
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
FIT_RECEIPT = ROOT / f"experiments/results/{STUDY}_fit_receipt.json"
DIAGNOSTICS = OUT / "diagnostics" / "train_only_directions.json"
DIRECTIONS = OUT / "diagnostics" / "train_only_directions.npz"
PR82_STUDY = "v2_short_original_diversity"


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pr82_view_seed(match: str, realization: int) -> int:
    """Preserve the PR82 TRAIN-view RNG, while the new study has its own ID."""
    return int(hashlib.sha256(
        f"{PR82_STUDY}/{recipe.VIEW_SEED}/{match}/{realization}".encode()
    ).hexdigest()[:8], 16)


def _bind_recipe_namespace() -> None:
    """Keep all generated files under this study while reusing PR82 code."""
    recipe.STUDY = STUDY
    recipe.OUT = OUT
    recipe.DECLARATION = DECLARATION
    recipe._view_seed = pr82_view_seed
    recipe.check = check
    explicit_archive = os.environ.get("MTAH_RF_ARCHIVE_DIR")
    rf_root = os.environ.get("MTAH_RF_DIR")
    if explicit_archive:
        recipe.ARCHIVE_ROOT = Path(explicit_archive)
    elif rf_root:
        recipe.ARCHIVE_ROOT = Path(rf_root) / "experiments/feed_robustness_v1/artifacts"
    elif (ROOT.parent / "mtah-rf/experiments/feed_robustness_v1/artifacts").is_dir():
        recipe.ARCHIVE_ROOT = ROOT.parent / "mtah-rf/experiments/feed_robustness_v1/artifacts"
    if os.environ.get("SOCCERNET_GAR_TRAIN_RECORDS"):
        recipe.TRAIN_RECORDS = Path(os.environ["SOCCERNET_GAR_TRAIN_RECORDS"])


def _source_files() -> dict[str, str]:
    paths = [Path(__file__), Path(weight_space_edit.__file__),
             *recipe._source_files(), *recipe._view_pipeline_files()]
    unique = {str(p.resolve()): p for p in paths}
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
            for p in sorted(unique.values(), key=lambda q: str(q))}


def _input_files() -> dict[str, str]:
    """Bind all exact inputs read by the reused preparation/training recipe."""
    inputs: dict[str, str] = {}
    preparation = OUT / "preparation.json"
    inputs[str(preparation.relative_to(ROOT))] = sha(preparation)
    for item in recipe.prep.split_plan()["train"]:
        raw = recipe.prep.data_path(item, recipe.rf.old.DATA)
        inputs[str(raw)] = sha(raw)
        entry = recipe.rf.old.old_spec()["prepared_training_caches"][item["match"]]
        cache = recipe.rf.old.DATA / Path(entry["cache"]).relative_to("data")
        inputs[str(cache)] = sha(cache)
        v1 = recipe._archive_root() / "views" / f"{item['match']}.joblib"
        inputs[str(v1)] = sha(v1)
        for rep in range(recipe.N_REALIZATIONS):
            view = recipe._view_path(item["match"], rep)
            inputs[str(view.relative_to(ROOT))] = sha(view)
            inputs[str(Path(str(view) + ".json").relative_to(ROOT))] = sha(Path(str(view) + ".json"))
    manifest = recipe.TRAIN_RECORDS / "train/manifest.json"
    inputs[str(manifest)] = sha(manifest)
    manifest_data = json.loads(manifest.read_text())
    for name in sorted(manifest_data["game_files"]):
        path = recipe.TRAIN_RECORDS / "train/games" / name
        inputs[str(path)] = sha(path)
    teachers = {}
    for fold in recipe.FOLDS:
        for seed in recipe.SEEDS:
            path = recipe.rf.sc.initial_path(fold, seed)
            teachers[f"{fold}/{seed}"] = sha(path)
    return {"files": inputs, "fold_teachers": teachers}


def prepare(workers: int = 3) -> dict:
    _bind_recipe_namespace()
    return recipe.prepare(workers=workers)


def freeze() -> dict:
    _bind_recipe_namespace()
    if DECLARATION.exists() or FIT_RECEIPT.exists():
        raise FileExistsError("This reconstruction ID is immutable once frozen")
    preparation = json.loads((OUT / "preparation.json").read_text())
    if preparation.get("realizations_per_original_match") != 4:
        raise ValueError("Expected four generated v2 views per TRAIN match")
    inputs = _input_files()
    declaration = {
        "study_id": STUDY,
        "status": "reconstruction_frozen_before_fitting",
        "question": "Can the exact PR82 four-view short-diversity recipe be reconstructed from currently reachable TRAIN inputs for a subsequent direct-weight-edit study?",
        "recipe_source": "experiments/v2_short_original_diversity.py, imported read-only; no frozen runner changes",
        "reconstruction": {
            "pr82_model_artifacts_available": False,
            "pr82_native_views_available": False,
            "view_seed_namespace_preserved": PR82_STUDY,
            "model_hashes_are_new_reconstruction": True,
            "byte_exact_restore_claim": False,
            "same_recipe_expected": "same source recipe, six provider-fold/seed fits, TRAIN data, sampler, optimizer budget and deterministic v2 view seeds; result parameters need not be byte-identical across environments",
        },
        "training": {
            "folds": list(recipe.FOLDS), "seeds": list(recipe.SEEDS),
            "fits": 6, "updates_per_fit": 2048,
            "v2_realizations_per_original_train_match": 4,
            "provider_exclusion": "all original-provider tracking and labels for the held-out provider are excluded; all GAR data are TRAIN only",
            "organizer_reference_access": False,
            "gar_valid_or_test_access": False,
            "public_scoring": False,
        },
        "inputs": inputs,
        "source_sha256": _source_files(),
        "promotion": False,
        "evaluation": False,
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(declaration, indent=1, sort_keys=True) + "\n")
    return declaration


def check() -> dict:
    _bind_recipe_namespace()
    if not DECLARATION.is_file():
        raise FileNotFoundError("Run prepare and freeze before fitting")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("training", {}).get("fits") != 6:
        raise ValueError("Reconstruction declaration identity or budget mismatch")
    current_sources = _source_files()
    if declaration.get("source_sha256") != current_sources:
        raise RuntimeError("Frozen training or reconstruction source changed")
    current_inputs = _input_files()
    if declaration.get("inputs") != current_inputs:
        raise RuntimeError("Frozen reconstruction inputs changed")
    return declaration


def _fit_one(fold: str, seed: int) -> dict:
    _bind_recipe_namespace()
    check()
    model_path = OUT / "models" / f"{fold}_{seed}.joblib"
    receipt_path = Path(str(model_path) + ".json")
    if model_path.exists() or receipt_path.exists():
        if not model_path.is_file() or not receipt_path.is_file():
            raise RuntimeError(f"Incomplete prior fit preserved for review: {model_path}")
        saved = json.loads(receipt_path.read_text())
        required = {"study": STUDY, "fold": fold, "seed": seed,
                    "sha256": sha(model_path), "declaration_sha256": sha(DECLARATION)}
        if any(saved.get(k) != v for k, v in required.items()):
            raise RuntimeError(f"Existing fit receipt does not match frozen inputs: {model_path}")
        expected_config = {"width": 96, "lr": 3e-4, "seed": seed, "steps": 2048,
                           "fraction": 0.5, "batch_size": 32, "ha_target": "dense",
                           "train_d": True, "d_lr": 1e-4, "ball_dropout": 0.0}
        if any(saved.get("config", {}).get(k) != v for k, v in expected_config.items()):
            raise RuntimeError(f"Existing fit configuration mismatch: {model_path}")
        return saved
    return recipe.fit(fold, seed)


def fit_all(workers: int = 1) -> dict:
    _bind_recipe_namespace()
    check()
    if FIT_RECEIPT.exists():
        raise FileExistsError("Fit receipt already exists; reconstruction is immutable")
    if not isinstance(workers, int) or not 1 <= workers <= 2:
        raise ValueError("Use one or two CPU fit workers")
    jobs = [(fold, seed) for fold in recipe.FOLDS for seed in recipe.SEEDS]
    rows = []
    if workers == 1:
        rows = [_fit_one(fold, seed) for fold, seed in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda_pair_fit, jobs))
    if len(rows) != 6:
        raise RuntimeError(f"Expected six completed fits, got {len(rows)}")
    receipt = {
        "study_id": STUDY,
        "declaration_sha256": sha(DECLARATION),
        "recipe_study": PR82_STUDY,
        "reconstruction": True,
        "byte_exact_restore_claim": False,
        "fits": rows,
        "all_six_fits_complete": len(rows) == 6,
        "organizer_reference_access": False,
        "public_scoring": False,
    }
    FIT_RECEIPT.write_text(json.dumps(receipt, indent=1, sort_keys=True, default=float) + "\n")
    return receipt


def lambda_pair_fit(job):
    """Top-level spawn-safe dispatcher for process-pool fit jobs."""
    return _fit_one(*job)


def pooled_training_records() -> list[dict]:
    """Build the same four-view recipe over all thirteen eligible TRAIN matches."""
    _bind_recipe_namespace()
    records: list[dict] = []
    spec = recipe.rf.old.old_spec()
    plan = recipe.prep.split_plan(fold="pooled")
    for item in plan["train"]:
        cache, _ = recipe._cache_for(item["match"], spec)
        paired = recipe.rf.deployment.paired_records(cache, item["match"])
        v1_view, _, _ = recipe._verified_v1_view(item["match"], recipe._v1_view_manifest()[0])
        paired = recipe.rf.replace_view(paired, v1_view["x"])
        views = recipe._load_v2_features(item["match"])
        for index, swap in ((2, False), (3, True)):
            oriented = [
                recipe.rf.supervised.swap_team_features(value, paired[index]["names"])
                if swap else value
                for value in views
            ]
            paired[index].update(x=oriented[0], alternate=oriented[0], physical_x=oriented[0])
            paired[index]["_v2_feature_views"] = oriented
            paired[index]["_v2_view_match"] = item["match"]
        records.extend(paired)
    trained = {str(item["match"]) for item in plan["train"]}
    included = {str(record["match"]) for record in records}
    if trained != included or len(trained) != 13:
        raise ValueError("Pooled records do not cover exactly the thirteen TRAIN matches")
    return records


def fit_pooled_member(seed: int, output_path: str | Path) -> dict:
    """Fit one pooled two-seed deployment member using the frozen v2 recipe.

    The caller must freeze a separate pooled deployment declaration before
    invoking this function. The returned model has the same dictionary schema
    as the six full-provider-fold checkpoints, and the caller should hash/save
    it beside an immutable fit receipt.
    """
    _bind_recipe_namespace()
    if seed not in recipe.SEEDS:
        raise ValueError(f"Seed must be one of {recipe.SEEDS}")
    check()
    import time
    teacher = recipe.joblib.load(recipe.rf.sc.initial_path("pooled", seed))
    records = pooled_training_records()
    gar = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    sampler = recipe.short.OriginalDiversitySampler(
        records, gar, list(teacher["feature_names"]), seed)
    start = time.monotonic()
    with recipe._round_robin_v2_windows(records) as view_draws:
        with recipe.short.parent.base.installed(sampler):
            result = recipe.rf.sn.train(
                teacher, records, gar, width=96, lr=3e-4, seed=seed,
                steps=2048, fraction=.5, batch_size=32, ha_target="dense",
                train_d=True, require_full_native=True, log_every=256, verbose=True)
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"Pooled checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    recipe.joblib.dump(result["model"], destination)
    return {
        "study_id": "v2_weight_edit_pooled_deployment_v1",
        "seed": seed,
        "path": str(destination),
        "sha256": sha(destination),
        "elapsed_seconds": time.monotonic() - start,
        "training_config": result["config"],
        "sampler": sampler.receipt(),
        "v2_window_draws": dict(view_draws),
        "training_matches": sorted({str(r["match"]) for r in records}),
        "reconstructed": True,
        "organizer_reference_access": False,
    }


def diagnose_directions() -> dict:
    """Derive per-fit rank-one failure directions from eligible TRAIN records."""
    _bind_recipe_namespace()
    check()
    if not FIT_RECEIPT.is_file():
        raise FileNotFoundError("All six reconstructed fold fits must finish first")
    if DIAGNOSTICS.exists() or DIRECTIONS.exists():
        raise FileExistsError("Train-only direction diagnostics already exist")
    fit_receipt = json.loads(FIT_RECEIPT.read_text())
    if fit_receipt.get("all_six_fits_complete") is not True:
        raise RuntimeError("Reconstruction fit receipt is incomplete")
    vectors, rows = {}, []
    for fold in recipe.FOLDS:
        records = recipe.training(fold)
        # Four records/match are ordered clean/direct, clean/swapped,
        # v2-r0/direct, v2-r0/swapped. The helper performs both orientations,
        # so pass only the direct record for each distinct fixed view.
        clean_records, v2_records = records[0::4], records[2::4]
        clean_matches = {str(r["match"]) for r in clean_records}
        if (len(clean_records) != len(clean_matches)
                or clean_matches != {str(r["match"]) for r in v2_records}):
            raise RuntimeError(f"Expected aligned clean/v2-r0 records in {fold}")
        for seed in recipe.SEEDS:
            model_path = OUT / "models" / f"{fold}_{seed}.joblib"
            receipt_path = Path(str(model_path) + ".json")
            if not model_path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(model_path)
            fit = json.loads(receipt_path.read_text())
            if fit.get("sha256") != sha(model_path) or fit.get("declaration_sha256") != sha(DECLARATION):
                raise RuntimeError(f"Fit receipt is stale or corrupt: {model_path}")
            model = recipe.joblib.load(model_path)
            teacher = recipe.joblib.load(recipe.rf.sc.initial_path(fold, seed))
            diagnosis_records = [dict(record, view_name="clean") for record in clean_records]
            diagnosis_records.extend(dict(record, view_name="v2_r0") for record in v2_records)
            diagnosis = weight_space_edit.diagnose_model(model, teacher, diagnosis_records)
            direction = recipe.np.asarray(diagnosis["direction"], dtype=recipe.np.float32)
            key = f"{fold}/{seed}"
            vectors[key] = direction
            rows.append({
                "fold": fold, "excluded_provider": fold.removeprefix("leave_out_"),
                "seed": seed, "training_matches": sorted(clean_matches),
                "training_match_count": len(clean_records),
                "activation_diagnostic": diagnosis["activation_diagnostic"],
                "view_count": diagnosis["view_count"],
                "direction_l2_norm": float(recipe.np.linalg.norm(direction)),
                "direction_sha256": diagnosis["direction_sha256"],
            })
    DIAGNOSTICS.parent.mkdir(parents=True, exist_ok=True)
    recipe.np.savez_compressed(DIRECTIONS, **vectors)
    report = {
        "study_id": STUDY,
        "diagnostic": "rank-one H/A final-projection direction from late versus correctly timed known H/A boundaries",
        "directions_sha256": sha(DIRECTIONS),
        "fit_receipt_sha256": sha(FIT_RECEIPT),
        "source_sha256": _source_files(),
        "rows": rows,
        "training_only": True,
        "evaluation_inputs_used": False,
        "organizer_reference_access": False,
        "public_scoring": False,
    }
    DIAGNOSTICS.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--workers", type=int, default=3)
    sub.add_parser("freeze")
    sub.add_parser("check")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--workers", type=int, default=1)
    sub.add_parser("diagnose")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.workers)
    elif args.action == "freeze":
        result = freeze()
    elif args.action == "check":
        result = check()
    elif args.action == "fit":
        result = fit_all(args.workers)
    else:
        result = diagnose_directions()
    print(json.dumps(result, indent=1, sort_keys=True, default=float), flush=True)


if __name__ == "__main__":
    main()
