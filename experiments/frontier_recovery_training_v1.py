"""Reconstruct the selected stage-1024 CE expert and its 35 native cells.

This creates a new, explicitly reconstructed cache for the already selected
v2 spell frontier. It reuses the frozen numerical trainers and source recipe;
it never edits the historical study files or scores a new candidate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np

from experiments import v2_iterative_train as iterative
from experiments import v2_short_original_diversity as recipe
from experiments import v2_spell_balanced_train as balanced_train
from experiments import v2_spell_balanced_eval as balanced_eval
from experiments import v2_data_capacity_eval as capacity
from experiments import v2_spell_target_runtime as model_runtime
from experiments import feed_spell_push, sn_dense_train as sn
from experiments import soccernet_extension as old

ROOT = Path(__file__).resolve().parents[1]
STUDY = "frontier_recovery_v1"
OUT = ROOT / "data/experiments" / STUDY
TRAIN_VIEW_OUT = OUT / "training_views"
TRAIN_DECL = ROOT / f"experiments/results/{STUDY}_training_declaration.json"
FIT_SEAL = OUT / "training_seal.json"
CELL_MANIFEST = OUT / "frontier_native_probability_manifest.json"
STAGE1_UPDATES = 512
STAGE2_UPDATES = 512
LR_STAGE1 = 3e-4
LR_STAGE2 = 1e-4
SEEDS = (2026, 2027)
FOLDS = iterative.FOLDS
VIEWS = (("clean", (0,)), ("feed_like_v1", (1, 2)), ("feed_like_v2", (1, 2, 3, 4)))
BASELINE_RECIPE = {
    "recipe_id": "ce0.5_cal0.375_ad0.125_arithmetic_ha3",
    "ce_weight": 0.5, "calibrated_weight": 0.375, "adopted_weight": 0.125,
    "pooling": "arithmetic", "ha_cost": 3.0, "training_arm": "base", "training_stage": 1024,
}
EXPECTED_BASELINE = {"clean": 0.6701365499557103,
                     "feed_like_v1": 0.532128306931943,
                     "feed_like_v2": 0.5936187198896373}
_TRAIN_CACHE: dict[str, list[dict]] = {}
_GAR_CACHE: list[dict] | None = None
AUDIT = model_runtime.frozen.audit
PROVIDER_LABEL = {"metrica": "Metrica", "skillcorner": "SkillCorner", "sportec": "Sportec"}


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=float)
        stream.write("\n")


def _configure() -> None:
    """Redirect mutable legacy helpers to this new study's ignored cache."""
    data = Path(os.environ.get("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))).resolve()
    sqd = Path(os.environ.get("MTAH_SQD_DIR", str(ROOT.parent / "mtah-sqd"))).resolve()
    rf_archive = Path(os.environ.get(
        "MTAH_RF_ARCHIVE_DIR", str(ROOT.parent / "mtah-rf/experiments/feed_robustness_v1/artifacts")))
    gar_root = Path(os.environ.get(
        "V2_ITERATIVE_GAR_ARCHIVE",
        str(ROOT.parent / "mtah-dt/experiments/dense_targeted_followups_v1/artifacts/train_records")))
    old.DATA = data
    recipe.rf.old.DATA = data
    recipe.rf.sc.DATA = data
    recipe.OUT = TRAIN_VIEW_OUT
    recipe.ARCHIVE_ROOT = rf_archive
    recipe.TRAIN_RECORDS = OUT / "gar_records"
    train_link = recipe.TRAIN_RECORDS / "train"
    if not train_link.exists():
        train_link.parent.mkdir(parents=True, exist_ok=True)
        train_link.symlink_to(gar_root, target_is_directory=True)
    elif train_link.resolve() != gar_root.resolve():
        raise ValueError(f"Unexpected recovery GAR link target: {train_link}")
    model_runtime.frozen._roots()
    # Explicitly use restored SQD root for warm-start folds.
    recipe.ARCHIVE_ROOT = rf_archive
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _ = sqd  # root is consumed by the shared frozen runtime's _roots().


@contextmanager
def _trainer_scope(lr: float, study: str):
    old_lr, old_study = iterative.LR, iterative.STUDY
    iterative.LR, iterative.STUDY = float(lr), study
    try:
        yield
    finally:
        iterative.LR, iterative.STUDY = old_lr, old_study


def expected_cells() -> list[str]:
    return [f"{provider}|{match}|{view}|{rep}"
            for provider, match, _ in AUDIT.MATCHES
            for view, reps in VIEWS for rep in reps]


def _source_files() -> list[Path]:
    # Numerical source closure for feature generation and the two unchanged
    # training stages. Evaluation sources are bound later in the cell manifest.
    modules = [iterative, recipe, recipe.rf, recipe.v2, recipe.prep, recipe.ms,
               recipe.short, recipe.sampler_module, recipe.short.parent,
               recipe.rf.sc, recipe.rf.sn, recipe.rf.dl, recipe.rf.deployment,
               recipe.rf.sn.cap, recipe.rf.sn.fh, recipe.rf.sn.lt,
               recipe.rf.sn.ol, recipe.rf.supervised, recipe.rf.old,
               recipe.rf.stress_components, recipe.rf.tc]
    paths = {Path(module.__file__) for module in modules if hasattr(module, "__file__")}
    paths.update({Path(__file__), Path(balanced_train.__file__),
                  ROOT / "experiments/results/v2_iterative_train_v1_declaration.json",
                  ROOT / "experiments/results/v2_iterative_train_v1_seal.json",
                  ROOT / "experiments/results/v2_spell_balanced_v1_training_declaration.json",
                  ROOT / "experiments/results/v2_spell_balanced_v1_training_seal.json",
                  ROOT / "experiments/results/short_original_diversity_v1_model_seal.json",
                  ROOT / "experiments/results/feed_like_v2_calibration.json",
                  ROOT / "experiments/eval_set/manifest.json",
                  ROOT / "experiments/baseline_manifest.json",
                  ROOT / "experiments/model_decision_policy.json"})
    return sorted(paths)


def _archive_facts() -> dict:
    """Verify source training/control artifacts against their private manifests."""
    _configure()
    spec = recipe.rf.old.old_spec()
    data_root = recipe.rf.old.DATA
    sqd_root = Path(os.environ.get("MTAH_SQD_DIR", str(ROOT.parent / "mtah-sqd"))).resolve()
    dt_root = Path(os.environ.get("MTAH_DT_DIR", str(ROOT.parent / "mtah-dt"))).resolve()
    data_manifest = data_root / "seed_gap_context_cache_manifest.json"
    teacher_manifest = data_root / "objective_alignment_cache_manifest.json"
    sqd_manifest = sqd_root / "short_quality_diversity_v1_cache_manifest.json"
    rf_archive = recipe._archive_root()
    rf_root = rf_archive.parents[2]
    rf_manifest = rf_root / "feed_robustness_v1_cache_manifest.json"
    dt_manifest = dt_root / "dense_targeted_followups_cache_manifest_v1.json"
    facts = {}

    def fact(key: str, manifest: Path, root: Path, path: Path) -> None:
        facts[key] = model_runtime.frozen._artifact_fact(
            manifest, root, str(path.relative_to(root)))

    for fold in FOLDS:
        provider = fold.removeprefix("leave_out_")
        for seed in SEEDS:
            warm = Path(os.environ.get("MTAH_SQD_DIR", str(ROOT.parent / "mtah-sqd"))).resolve() / \
                "experiments/short_quality_diversity_delivery_v1/artifacts/short_original_diversity_v1/models" / \
                f"{fold}_{seed}.joblib"
            teacher = recipe.rf.sc.initial_path(fold, seed)
            fact(f"warmstart/{provider}/{seed}", sqd_manifest, sqd_root, warm)
            fact(f"teacher/{provider}/{seed}", teacher_manifest, data_root, teacher)
            adop = model_runtime.frozen._model_path("adop", PROVIDER_LABEL[provider], seed)
            fact(f"calibrated/{provider}/{seed}", dt_manifest, dt_root, adop)
            predecessor = model_runtime.frozen._model_path("cand", PROVIDER_LABEL[provider], seed)
            fact(f"predecessor/{provider}/{seed}", sqd_manifest, sqd_root, predecessor)

    plan = recipe.prep.split_plan(fold="pooled")
    if len(plan["train"]) != 13:
        raise ValueError("Expected 13 original TRAIN matches")
    entries, v1_manifest = recipe._v1_view_manifest()
    facts["feed_robustness_v1_manifest"] = {
        "path": str(v1_manifest), "sha256": sha(v1_manifest)}
    for item in plan["train"]:
        raw = recipe.prep.data_path(item, data_root)
        if sha(raw) != item["sha256"]:
            raise ValueError(f"Original TRAIN input differs from split manifest: {item['match']}")
        facts[f"train_raw/{item['match']}"] = {"path": str(raw), "sha256": sha(raw)}
        entry = spec["prepared_training_caches"][item["match"]]
        cached_path = data_root / Path(entry["cache"]).relative_to("data")
        if sha(cached_path) != entry["sha256"]:
            raise ValueError(f"Prepared original training cache hash mismatch: {item['match']}")
        facts[f"train_features/{item['match']}"] = {"path": str(cached_path), "sha256": sha(cached_path)}
        v1_path = rf_archive / "views" / f"{item['match']}.joblib"
        rel = v1_path.relative_to(rf_root).as_posix()
        archive_entry = entries.get(rel)
        if (archive_entry is None or archive_entry.get("sha256") != sha(v1_path)
                or archive_entry.get("bytes") != v1_path.stat().st_size):
            raise ValueError(f"Historical v1 training view archive seal mismatch: {item['match']}")
        facts[f"train_v1/{item['match']}"] = {"path": str(v1_path), "sha256": sha(v1_path)}

    gar_manifest = recipe.TRAIN_RECORDS / "train/manifest.json"
    gar = json.loads(gar_manifest.read_text())
    if len(gar.get("game_files", {})) != 45:
        raise ValueError("Recovery requires exactly the 45 GAR TRAIN records; VALID/TEST are excluded")
    facts["gar_train_manifest"] = {"path": str(gar_manifest), "sha256": sha(gar_manifest)}
    for name, digest in sorted(gar["game_files"].items()):
        path = gar_manifest.parent / "games" / name
        if sha(path) != digest:
            raise ValueError(f"GAR TRAIN record hash mismatch: {name}")
        facts[f"gar_train/{name}"] = {"path": str(path), "sha256": digest}

    return facts


def _training_inputs() -> dict:
    _configure()
    facts = _archive_facts()
    for file in (TRAIN_VIEW_OUT / "preparation.json",):
        facts[f"training_preparation/{file.name}"] = {"path": str(file), "sha256": sha(file)}
    for match, receipts in json.loads((TRAIN_VIEW_OUT / "preparation.json").read_text())["train_matches"].items():
        for row in receipts:
            path = TRAIN_VIEW_OUT / "views" / f"{match}__r{row['realization']}.joblib"
            receipt = Path(str(path) + ".json")
            if sha(path) != row["sha256"] or not receipt.is_file():
                raise ValueError(f"Generated TRAIN v2 view receipt mismatch: {match}/{row['realization']}")
            receipt_data = json.loads(receipt.read_text())
            if receipt_data.get("sha256") != row["sha256"]:
                raise ValueError(f"Generated TRAIN v2 view receipt content mismatch: {match}/{row['realization']}")
            facts[f"training_v2/{match}/r{row['realization']}"] = {
                "path": str(path), "sha256": sha(path), "receipt_path": str(receipt), "receipt_sha256": sha(receipt)}
    return facts


def _training_source_hashes() -> dict:
    paths = _source_files()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Training source closure missing: " + ", ".join(missing))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def prepare_training(workers: int = 2) -> dict:
    if workers not in (1, 2):
        raise ValueError("Use one or two workers")
    _configure()
    if TRAIN_DECL.exists() or FIT_SEAL.exists():
        raise FileExistsError("Training recovery is already frozen; preserve its artifacts")
    TRAIN_VIEW_OUT.mkdir(parents=True, exist_ok=True)
    prep_receipt = recipe.prepare(workers=workers)
    old_decl = json.loads((ROOT / "experiments/results/v2_iterative_train_v1_declaration.json").read_text())
    old_inputs = old_decl.get("input_sha256", {})
    view_parity = {}
    for item in recipe.prep.split_plan()["train"]:
        for realization in range(recipe.N_REALIZATIONS):
            path = recipe._view_path(item["match"], realization)
            old_path = ROOT / "data/experiments/v2_iterative_train_v1/views" / path.name
            old_receipt = Path(str(old_path) + ".json")
            expected = old_inputs.get(str(old_path))
            if expected is None or sha(path) != expected:
                raise ValueError(f"Reconstructed TRAIN view differs from sealed bytes: {path}")
            receipt = Path(str(path) + ".json")
            expected_receipt = old_inputs.get(str(old_receipt))
            if expected_receipt is None or sha(receipt) != expected_receipt:
                raise ValueError(f"Reconstructed view receipt differs from sealed bytes: {receipt}")
            view_parity[str(path.relative_to(ROOT))] = sha(path)
    # Confirm each fit fold excludes its provider and that the 45-game TRAIN
    # manifest is complete before a recovery declaration can be sealed.
    for fold in FOLDS:
        records = recipe.training(fold)
        excluded = fold.removeprefix("leave_out_")
        if any(str(record.get("source", "")).lower() == excluded for record in records):
            raise ValueError(f"Held-out provider leaked into {fold}")
    gar = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    if len({str(row["match"]) for row in gar}) != 45:
        raise ValueError("Expected all 45 GAR TRAIN games")
    return {"preparation": prep_receipt, "historical_view_byte_parity": view_parity,
            "training_records": len(gar),
            "training_matches": 13, "folds": list(FOLDS)}


def freeze_training() -> dict:
    _configure()
    if TRAIN_DECL.exists() or FIT_SEAL.exists():
        raise FileExistsError("Training recovery is already frozen")
    if not (TRAIN_VIEW_OUT / "preparation.json").is_file():
        raise FileNotFoundError("Run prepare-training before freezing recovery")
    # Re-run the fixed preparation checks here even if views were produced by
    # the shared helper directly; declaration freeze requires historical byte parity.
    prep_check = prepare_training(workers=2)
    facts = _training_inputs()
    provider_plans = {fold: recipe.prep.split_plan(fold=fold) for fold in FOLDS}
    d = {
        "schema": "frontier_recovery_v1_training",
        "study_id": STUDY,
        "status": "frozen_training_reconstruction_before_fit",
        "reconstruction_scope": "selected current frontier CE stage1024 provider-holdout component only",
        "source_status": "reconstructed",
        "frontier_study_id": "v2_spell_frontier_v1",
        "frontier_result_sha256": sha(ROOT / "experiments/results/v2_spell_frontier_v1_round1.json"),
        "training_arm": "base",
        "training_stage": 1024,
        "provider_excluded_fits": 6,
        "folds": list(FOLDS), "seeds": list(SEEDS),
        "stages": [
            {"stage": 512, "updates": 512, "learning_rate": LR_STAGE1,
             "initialization": "exact archived original_full provider-fold checkpoint, paired teacher and normalization"},
            {"stage": 1024, "updates": 512, "learning_rate": LR_STAGE2,
             "initialization": "same-fold/seed reconstructed stage512 checkpoint; repeat seed and sampler schedule"},
        ],
        "numerical_config": {"batch_size": 32, "gar_fraction": 0.5, "optimizer": "AdamW",
                             "weight_decay": iterative.cap.WEIGHT_DECAY,
                             "gradient_clip": iterative.cap.GRADIENT_CLIP,
                             "input_dropout": iterative.cap.INPUT_DROPOUT,
                             "spell_coefficient": iterative.cap.SPELL_COEFFICIENT,
                             "live_transition_coefficient": iterative.cap.LIVE_TRANSITION_COEFFICIENT,
                             "train_d": False, "sampler": "OriginalDiversitySampler",
                             "v2_views_per_original_match": 4,
                             "torch_threads_per_fit": 1, "workers_max": 2},
        "environment": {"python": sys.version, "platform": platform.platform(),
                         "numpy": np.__version__, "joblib": joblib.__version__,
                         "torch": __import__("torch").__version__,
                         "torch_threads_per_fit": 1, "max_workers": 2},
        "provider_plans": provider_plans,
        "historical_view_byte_parity": prep_check["historical_view_byte_parity"],
        "input_sha256": facts,
        "source_sha256": _training_source_hashes(),
        "historical_fit_runtime_seconds": {
            "stage512_ce_control": [157.2886, 98.6906, 172.9, 108.2, 135.1, 165.0],
            "stage1024_continuation": [113.8118, 116.5396, 91.9229, 106.9292, 107.0763, 111.7226],
            "serial_cpu_minutes_total": 24.8},
        "historical_source_comparison": {
            "status": "numerical Python source hashes verified against historical iterative/balanced declarations by parent audit",
            "known_non_numerical_differences": ["baseline_manifest.json", "model_decision_policy.json"]},
        "gar_valid_test_access": False, "organizer_reference_access": False,
        "scoring_performed": False, "candidate_scoring": False,
        "promotion": False,
    }
    _write_once(TRAIN_DECL, d)
    return d


def _check_training_binding() -> dict:
    if not TRAIN_DECL.is_file():
        raise RuntimeError("Freeze the new training recovery declaration before fitting")
    d = json.loads(TRAIN_DECL.read_text())
    _configure()
    if d.get("source_sha256") != _training_source_hashes():
        raise ValueError("Recovery training sources changed after declaration")
    if d.get("input_sha256") != _training_inputs():
        raise ValueError("Recovery training inputs changed after declaration")
    return d


def _tensor_sha(model: dict) -> str:
    h = hashlib.sha256()
    for group in ("state", "d_state"):
        values = model.get(group, {})
        h.update(group.encode())
        for name in sorted(values):
            array = np.ascontiguousarray(values[name])
            h.update(name.encode()); h.update(array.dtype.str.encode())
            h.update(str(array.shape).encode()); h.update(array.tobytes())
    for name in ("median", "scale"):
        array = np.ascontiguousarray(model[name])
        h.update(name.encode()); h.update(array.dtype.str.encode())
        h.update(str(array.shape).encode()); h.update(array.tobytes())
    h.update("\0".join(model.get("feature_names", [])).encode())
    return h.hexdigest()


def _original_fit_rows() -> tuple[dict, dict]:
    first = json.loads((ROOT / "experiments/results/v2_iterative_train_v1_seal.json").read_text())
    second = json.loads((ROOT / "experiments/results/v2_spell_balanced_v1_training_seal.json").read_text())
    stage512 = {(r["fold"], int(r["seed"])): r for r in first["fits"] if r.get("arm") == "ce_control"}
    stage1024 = {(r["fold"], int(r["seed"])): r for r in second["fits"]}
    if len(stage512) != 6 or len(stage1024) != 6:
        raise ValueError("Historical seals must contain six CE folds at both stages")
    return stage512, stage1024


def _stage_path(stage: int, fold: str, seed: int) -> Path:
    return OUT / "models" / f"stage{stage}" / f"{fold}_{seed}.joblib"


def _get_training_records(fold: str) -> list[dict]:
    global _GAR_CACHE
    if fold not in _TRAIN_CACHE:
        _TRAIN_CACHE[fold] = recipe.training(fold)
    if _GAR_CACHE is None:
        _GAR_CACHE = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    return _TRAIN_CACHE[fold]


def _fit_stage(job: tuple[int, str, int]) -> dict:
    import torch
    torch.set_num_threads(1)
    stage, fold, seed = job
    declaration = json.loads(TRAIN_DECL.read_text())
    _configure()
    path = _stage_path(stage, fold, seed)
    receipt_path = Path(str(path) + ".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(f"Refusing to overwrite recovery checkpoint: {path}")
    teacher_path = recipe.rf.sc.initial_path(fold, seed)
    teacher = joblib.load(teacher_path)
    old_records = _get_training_records(fold)
    gar_records = _GAR_CACHE
    assert gar_records is not None
    if stage == 512:
        prior_path = Path(os.environ.get("MTAH_SQD_DIR", str(ROOT.parent / "mtah-sqd"))).resolve() / \
            "experiments/short_quality_diversity_delivery_v1/artifacts/short_original_diversity_v1/models" / \
            f"{fold}_{seed}.joblib"
        prior = joblib.load(prior_path)
        lr, trainer_study = LR_STAGE1, iterative.STUDY
        arm_name = "ce_control"
    elif stage == 1024:
        prior_path = _stage_path(512, fold, seed)
        prior = joblib.load(prior_path)
        lr, trainer_study = LR_STAGE2, balanced_train.STUDY
        arm_name = "ce_control_lr1e-4_continuation"
    else:
        raise ValueError(stage)
    if prior.get("kind") != "soccernet_capacity" or prior.get("hidden") != 96:
        raise ValueError(f"Invalid checkpoint at stage {stage}: {prior_path}")
    if prior["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError(f"Fold/seed checkpoint is paired with the wrong teacher: {fold}/{seed}")
    sampler = recipe.short.OriginalDiversitySampler(old_records, gar_records,
                                                    list(teacher["feature_names"]), seed)
    started = time.monotonic()
    with _trainer_scope(lr, trainer_study):
        with recipe._round_robin_v2_windows(old_records) as view_draws:
            with recipe.short.parent.base.installed(sampler):
                result = iterative.fit_from_checkpoints(
                    teacher, prior, old_records, gar_records, seed=seed,
                    arm="ce_control", verbose=True)
    model = result["model"]
    if stage == 1024:
        model = balanced_train._annotate_result(
            result, teacher=teacher, stage=1024, cumulative_updates=1024,
            warmstart_path=prior_path, arm=arm_name,
            schedule="same seed reset as stage512; repeated sample schedule")
    if set(model["d_state"]) != set(prior["d_state"]):
        raise ValueError("Training unexpectedly changed the frozen adopted D state")
    for name, value in prior["d_state"].items():
        np.testing.assert_array_equal(model["d_state"][name], value)
    for name in ("median", "scale"):
        np.testing.assert_array_equal(model[name], prior[name])
    if model["feature_names"] != prior["feature_names"]:
        raise ValueError("Training changed the frozen feature normalization schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(model, temp)
    os.replace(temp, path)
    historical_first, historical_second = _original_fit_rows()
    expected = (historical_first if stage == 512 else historical_second)[(fold, seed)]
    receipt = {
        "study_id": STUDY, "stage": stage, "fold": fold, "seed": seed,
        "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "tensor_sha256": _tensor_sha(model), "original_sealed_sha256": expected["sha256"],
        "original_file_sha_matches": sha(path) == expected["sha256"],
        "source_status": "reconstructed", "declaration_sha256": sha(TRAIN_DECL),
        "teacher_path": str(teacher_path), "teacher_sha256": sha(teacher_path),
        "warmstart_path": str(prior_path), "warmstart_sha256": sha(prior_path),
        "wall_seconds": time.monotonic() - started, "config": result["config"],
        "sampling": sampler.receipt(), "view_draws": dict(view_draws),
        "exposures": result["exposures"], "loss_intervals": result["loss_intervals"],
        "unchanged_d_state": True, "organizer_reference_access": False,
        "scoring_performed": False,
    }
    _write_once(receipt_path, receipt)
    print(f"RECOVERED FIT stage={stage} fold={fold} seed={seed} {receipt['wall_seconds']:.1f}s", flush=True)
    return receipt


def fit(workers: int = 2) -> dict:
    if workers not in (1, 2):
        raise ValueError("Training is limited to one or two workers")
    _check_training_binding()
    if FIT_SEAL.exists():
        raise FileExistsError(f"Recovery fit seal already exists: {FIT_SEAL}")
    jobs = [(512, fold, seed) for fold in FOLDS for seed in SEEDS]
    stages = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fit_stage, job): job for job in jobs}
        for idx, future in enumerate(as_completed(futures), 1):
            row = future.result()
            stages[(row["stage"], row["fold"], row["seed"])] = row
            print(f"STAGE512 {idx}/{len(jobs)}", flush=True)
    jobs = [(1024, fold, seed) for fold in FOLDS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fit_stage, job): job for job in jobs}
        for idx, future in enumerate(as_completed(futures), 1):
            row = future.result()
            stages[(row["stage"], row["fold"], row["seed"])] = row
            print(f"STAGE1024 {idx}/{len(jobs)}", flush=True)
    fit_rows = [stages[key] for key in sorted(stages)]
    if len(fit_rows) != 12:
        raise RuntimeError("Expected both training stages for all six fold/seed pairs")
    _write_once(FIT_SEAL, {
        "study_id": STUDY, "declaration_sha256": sha(TRAIN_DECL),
        "fits": fit_rows, "all_provider_folds_complete": True,
        "training_stages": [512, 1024], "new_fit_count": 12,
        "source_status": "reconstructed", "organizer_reference_access": False,
        "scoring_performed": False, "candidate_scoring": False,
    })
    return json.loads(FIT_SEAL.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-training", "freeze-training", "fit"))
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    _configure()
    if args.action == "prepare-training":
        print(json.dumps(prepare_training(args.workers), indent=2))
    elif args.action == "freeze-training":
        print(json.dumps(freeze_training(), indent=2))
    else:
        print(json.dumps(fit(args.workers), indent=2))


if __name__ == "__main__":
    main()
