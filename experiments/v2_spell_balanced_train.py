"""Continuation training for the v2 spell-balanced development experiment.

The numerical loop and objectives are reused from the completed iterative CE
control. This module binds new declarations, receipts and output paths for the
second 512-update stage and the later pooled TRAIN reproduction.
"""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from experiments import v2_iterative_train as iterative
from experiments import v2_iterative_train_run as prior_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_spell_balanced_v1"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
TRAIN_DECL = ROOT / f"experiments/results/{STUDY}_training_declaration.json"
TRAIN_SEAL = ROOT / f"experiments/results/{STUDY}_training_seal.json"
POOL_DECL = ROOT / f"experiments/results/{STUDY}_pooled_declaration.json"
POOL_SEAL = ROOT / f"experiments/results/{STUDY}_pooled_seal.json"
OUT = ROOT / "data/experiments" / STUDY / "train"
FOLDS, SEEDS = iterative.FOLDS, iterative.SEEDS
STAGE_UPDATES = 512
LR_STAGE_512 = 3e-4
LR_STAGE_1024 = 1e-4


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


@contextmanager
def _trainer_scope(lr: float):
    """Override only trainer metadata/rate; restore globals even on failure."""
    old_lr, old_study = iterative.LR, iterative.STUDY
    iterative.LR, iterative.STUDY = float(lr), STUDY
    try:
        yield
    finally:
        iterative.LR, iterative.STUDY = old_lr, old_study


def _bind_prior_recipe():
    """Use the exact prepared source/view bundle from the sealed CE study."""
    prior_run.bind()
    return prior_run.recipe


def prior_model_path(fold: str, seed: int) -> Path:
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Undeclared provider fold or seed")
    return prior_run.model_path("ce_control", fold, seed)


def output_path(fold: str, seed: int) -> Path:
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Undeclared provider fold or seed")
    return OUT / "models" / f"{fold}_{seed}.joblib"


def _pooled_output_path(seed: int, stage: int) -> Path:
    if seed not in SEEDS or stage not in (512, 1024):
        raise ValueError("Pooled outputs support only seeds 2026/2027 and stages 512/1024")
    return OUT / "models" / "pooled" / f"stage{stage}_{seed}.joblib"


def _pooled_adopted_paths() -> tuple[Path, Path]:
    manifest = json.loads((ROOT / "experiments/baseline_manifest.json").read_text())
    directory = ROOT / manifest["model_directory"]
    return directory / "dense_{seed}.joblib", directory / "teacher_{seed}.joblib"


def _source_facts() -> dict[str, str]:
    recipe = _bind_prior_recipe()
    paths = {Path(__file__), Path(iterative.__file__), Path(prior_run.__file__),
             ROOT / "experiments/tests/test_v2_spell_balanced_train.py",
             *recipe._source_files(), *recipe._view_pipeline_files(),
             PLAN, ROOT / "experiments/results/v2_iterative_v1_plan.json",
             ROOT / "experiments/results/v2_iterative_train_v1_declaration.json",
             ROOT / "experiments/results/v2_iterative_train_v1_seal.json",
             ROOT / "experiments/baseline_manifest.json",
             ROOT / "experiments/model_decision_policy.json"}
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def _input_facts() -> dict[str, str]:
    recipe = _bind_prior_recipe()
    prior_declaration = json.loads(prior_run.DECL.read_text())
    if prior_declaration.get("input_sha256") != prior_run.input_facts():
        raise ValueError("The sealed CE training inputs differ from their declaration")
    inputs = dict(prior_declaration["input_sha256"])
    paths = [prior_run.DECL, prior_run.SEAL]
    for fold in FOLDS:
        for seed in SEEDS:
            model = prior_model_path(fold, seed)
            paths.extend((model, Path(str(model) + ".json"),
                          recipe.rf.sc.initial_path(fold, seed)))
    baseline_manifest = json.loads((ROOT / "experiments/baseline_manifest.json").read_text())
    dense_pattern, teacher_pattern = _pooled_adopted_paths()
    for seed in SEEDS:
        for path in (Path(str(dense_pattern).format(seed=seed)),
                     Path(str(teacher_pattern).format(seed=seed))):
            expected = (baseline_manifest["models_sha256"][path.name])
            if sha(path) != expected:
                raise ValueError(f"Restored pooled baseline hash mismatch: {path}")
            paths.append(path)
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        inputs[str(path)] = sha(path)
    return inputs


def _check_prior_seal() -> dict:
    _bind_prior_recipe()
    prior_run.check(full=True)
    declaration = json.loads(prior_run.DECL.read_text())
    seal = json.loads(prior_run.SEAL.read_text())
    if (seal.get("study_id") != prior_run.STUDY or
            seal.get("declaration_sha256") != sha(prior_run.DECL) or
            seal.get("all_12_complete") is not True):
        raise ValueError("Prior CE-control fit seal is incomplete or inconsistent")
    by_arm_fold_seed = {(row["arm"], row["fold"], int(row["seed"])): row
                        for row in seal["fits"]}
    for fold in FOLDS:
        for seed in SEEDS:
            row = by_arm_fold_seed.get(("ce_control", fold, seed))
            model = prior_model_path(fold, seed)
            receipt = Path(str(model) + ".json")
            if row is None or not model.is_file() or not receipt.is_file():
                raise FileNotFoundError(f"Missing sealed CE warm start: {fold}/{seed}")
            sidecar = json.loads(receipt.read_text())
            if (row != sidecar or sha(model) != row.get("sha256") or
                    row.get("declaration_sha256") != sha(prior_run.DECL)):
                raise ValueError(f"CE warm-start seal/receipt mismatch: {fold}/{seed}")
    return declaration


def freeze() -> dict:
    """Bind the six provider-excluded stage-1024 fits without running them."""
    if TRAIN_DECL.exists() or TRAIN_SEAL.exists():
        raise FileExistsError("Stage-1024 training declaration already exists")
    if not PLAN.is_file():
        raise FileNotFoundError(PLAN)
    plan = json.loads(PLAN.read_text())
    if plan.get("study_id") != STUDY:
        raise ValueError("Unexpected spell-balanced plan")
    prior_declaration = _check_prior_seal()
    recipe = _bind_prior_recipe()
    cohorts = {fold: recipe.prep.split_plan(fold=fold)["train"] for fold in FOLDS}
    for fold, rows in cohorts.items():
        excluded = fold.removeprefix("leave_out_").lower()
        if any(str(row["source"]).lower() == excluded for row in rows):
            raise ValueError(f"Held-out provider leaked into {fold} continuation")
    inputs, sources = _input_facts(), _source_facts()
    declaration = dict(
        schema="v2_spell_balanced_training_v1",
        study_id=STUDY,
        plan_sha256=sha(PLAN),
        previous_training_declaration_sha256=sha(prior_run.DECL),
        previous_training_seal_sha256=sha(prior_run.SEAL),
        previous_training_input_sha256=prior_declaration["input_sha256"],
        question="Does a second 512-update continuation of the matched CE control at a lower learning rate improve spell retention while keeping macro regression within the frozen slight-tradeoff limit?",
        arms=["ce_control_lr1e-4_continuation"],
        folds=list(FOLDS), seeds=list(SEEDS),
        budget=dict(fits=6, updates_per_fit=STAGE_UPDATES, cumulative_updates=1024,
                    learning_rate=LR_STAGE_1024, batch_size=iterative.BATCH_SIZE,
                    gar_fraction=iterative.FRACTION_GAR, workers_max=2),
        initialization="Exact sealed v2_iterative_train_v1 ce_control fold/seed checkpoint after 512 updates; original paired teacher, normalization and d_state retained",
        continuation_schedule="Reset each fixed seed at stage start. This intentionally repeats the prior 512-step draw schedule; it does not claim fresh-window exposure.",
        fixed_objective="Existing shared duration-weighted H/A CE plus live-transition objective; no boundary-margin term. Existing distinct-possession sampler and same four round-robin v2 TRAIN views.",
        train_cohorts=cohorts,
        exclusions="Exclude the evaluated provider from every original TRAIN record, teacher, checkpoint and normalization; GAR TRAIN only.",
        input_sha256=inputs, source_sha256=sources,
        organizer_reference_access=False, gar_valid_test_access=False,
        public_or_GAR_scoring=False, promotion=False)
    write_once(TRAIN_DECL, declaration)
    return declaration


def check(full: bool = True) -> dict:
    _bind_prior_recipe()
    if not TRAIN_DECL.is_file():
        raise FileNotFoundError("Freeze stage-1024 training declaration before fitting")
    declaration = json.loads(TRAIN_DECL.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("source_sha256") != _source_facts():
        raise ValueError("Spell-balanced trainer source changed after freeze")
    if full and declaration.get("input_sha256") != _input_facts():
        raise ValueError("Spell-balanced training inputs changed after freeze")
    return declaration


def _annotate_result(result: dict, *, teacher: dict, stage: int,
                     cumulative_updates: int, warmstart_path: Path,
                     arm: str, schedule: str) -> dict:
    model = result["model"]
    if model["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("Trainer export changed the canonical teacher hash")
    provenance = dict(study=STUDY, arm=arm, stage=stage,
                      stage_updates=STAGE_UPDATES, cumulative_updates=cumulative_updates,
                      warm_start_path=str(warmstart_path), warm_start_file_sha256=sha(warmstart_path),
                      continuation_schedule=schedule)
    result["config"].update(provenance)
    model["training"].update(provenance)
    model["step"] = int(cumulative_updates)
    return model


def _load_checkpoint(path: Path) -> dict:
    checkpoint = joblib.load(path)
    if checkpoint.get("kind") != "soccernet_capacity" or checkpoint.get("hidden") != 96:
        raise ValueError(f"Unexpected warm-start checkpoint format: {path}")
    return checkpoint


def fit_fold(fold: str, seed: int, *, verbose: bool = True) -> dict:
    """Fit one of the six frozen provider-excluded lower-LR continuations."""
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Use one of the frozen folds and seeds")
    declaration = check(full=True)
    model_path = output_path(fold, seed)
    receipt_path = Path(str(model_path) + ".json")
    if model_path.exists() or receipt_path.exists():
        raise FileExistsError(model_path)
    recipe = _bind_prior_recipe()
    prior_path = prior_model_path(fold, seed)
    teacher = joblib.load(recipe.rf.sc.initial_path(fold, seed))
    prior_model = _load_checkpoint(prior_path)
    if prior_model["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError(f"Warm-start teacher does not match: {fold}/{seed}")
    old_records = recipe.training(fold)
    gar_records = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    sampler = recipe.short.OriginalDiversitySampler(
        old_records, gar_records, list(teacher["feature_names"]), seed)
    start = time.monotonic()
    # fit_from_checkpoints consumes the warm-start object itself; this local
    # path-based adapter keeps the receipt tied to the already sealed file.
    with _trainer_scope(LR_STAGE_1024):
        with recipe._round_robin_v2_windows(old_records) as view_draws:
            with recipe.short.parent.base.installed(sampler):
                result = iterative.fit_from_checkpoints(
                    teacher, prior_model, old_records, gar_records,
                    seed=seed, arm="ce_control", verbose=verbose)
    model = result["model"]
    _annotate_result(result, teacher=teacher, stage=1024, cumulative_updates=1024,
        warmstart_path=prior_path.relative_to(ROOT), arm="ce_control_lr1e-4_continuation",
        schedule="seed reset; repeats stage-1 sample schedule")
    if set(model["d_state"]) != set(prior_model["d_state"]):
        raise ValueError("D state keys changed during continuation")
    for key, value in prior_model["d_state"].items():
        np.testing.assert_array_equal(model["d_state"][key], value)
    for field in ("median", "scale", "feature_names"):
        if field == "feature_names":
            if model[field] != prior_model[field]:
                raise ValueError("Normalization schema changed during continuation")
        else:
            np.testing.assert_array_equal(model[field], prior_model[field])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    receipt = dict(study_id=STUDY, stage=1024, stage_updates=STAGE_UPDATES,
        cumulative_updates=1024, fold=fold, seed=seed,
        path=str(model_path.relative_to(ROOT)), sha256=sha(model_path),
        declaration_sha256=sha(TRAIN_DECL), wall_seconds=time.monotonic()-start,
        config=result["config"], warmstart_path=str(prior_path.relative_to(ROOT)),
        warmstart_sha256=sha(prior_path), teacher_sha256=sha(recipe.rf.sc.initial_path(fold, seed)),
        sampling=sampler.receipt(), view_draws=dict(view_draws), exposures=result["exposures"],
        loss_intervals=result["loss_intervals"], unchanged_d_state=True,
        old_matches=sorted({str(row["match"]) for row in old_records}),
        gar_games=sorted({str(row["match"]) for row in gar_records}),
        organizer_reference_access=False, gar_valid_test_access=False)
    write_once(receipt_path, receipt)
    return receipt


def seal_folds() -> dict:
    check(full=True)
    if TRAIN_SEAL.exists():
        raise FileExistsError(TRAIN_SEAL)
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            path = output_path(fold, seed)
            receipt = json.loads(Path(str(path) + ".json").read_text())
            if (receipt.get("declaration_sha256") != sha(TRAIN_DECL) or
                    receipt.get("sha256") != sha(path) or receipt.get("stage") != 1024 or
                    receipt.get("fold") != fold or int(receipt.get("seed", -1)) != seed or
                    receipt.get("unchanged_d_state") is not True):
                raise ValueError(f"Continuation fit receipt mismatch: {fold}/{seed}")
            rows.append(receipt)
    if len(rows) != 6:
        raise ValueError("Expected exactly six provider-fold continuation receipts")
    seal = dict(study_id=STUDY, declaration_sha256=sha(TRAIN_DECL),
                fits=rows, all_six_complete=True,
                repeated_seed_schedule=True, cumulative_updates=1024,
                organizer_reference_access=False, promotion=False)
    write_once(TRAIN_SEAL, seal)
    return seal


def pooled_training() -> list[dict]:
    """Load the same 13 original TRAIN matches with the fixed four v2 views."""
    recipe = _bind_prior_recipe()
    spec = recipe.rf.old.old_spec()
    entries, _ = recipe._v1_view_manifest()
    records = []
    plan = recipe.prep.split_plan(fold="pooled")
    for item in plan["train"]:
        match = item["match"]
        cache, _ = recipe._cache_for(match, spec)
        paired = recipe.rf.deployment.paired_records(cache, match)
        view, _, _ = recipe._verified_v1_view(match, entries)
        for key in ("indices", "period", "y"):
            np.testing.assert_array_equal(view[key], cache["records"]["clean"][key])
        if list(view["names"]) != list(cache["records"]["clean"]["names"]):
            raise ValueError(f"v1 view schema mismatch: {match}")
        paired = recipe.rf.replace_view(paired, view["x"])
        views = recipe._load_v2_features(match)
        for feature in views:
            if feature.shape != np.asarray(paired[2]["x"]).shape:
                raise ValueError(f"v2 view shape mismatch: {match}")
        for index, swap in ((2, False), (3, True)):
            oriented = [recipe.rf.supervised.swap_team_features(x, paired[index]["names"])
                        if swap else x for x in views]
            paired[index].update(x=oriented[0], alternate=oriented[0], physical_x=oriented[0],
                                 _v2_feature_views=oriented, _v2_view_match=match)
        records.extend(paired)
    expected = {row["match"] for row in plan["train"]}
    got = {str(row["match"]) for row in records}
    if got != expected or len(got) != 13:
        raise ValueError("Pooled v2 TRAIN record inventory must contain exactly 13 matches")
    return records


def _pool_teacher_and_adopted(seed: int) -> tuple[dict, dict, Path, Path]:
    if seed not in SEEDS:
        raise ValueError("Use one of the two frozen seeds")
    dense_pattern, teacher_pattern = _pooled_adopted_paths()
    dense_path = Path(str(dense_pattern).format(seed=seed))
    teacher_path = Path(str(teacher_pattern).format(seed=seed))
    manifest = json.loads((ROOT / "experiments/baseline_manifest.json").read_text())
    if sha(dense_path) != manifest["models_sha256"][dense_path.name]:
        raise ValueError(f"Pooled adopted model hash mismatch: {dense_path}")
    if sha(teacher_path) != manifest["models_sha256"][teacher_path.name]:
        raise ValueError(f"Pooled teacher hash mismatch: {teacher_path}")
    adopted, teacher = joblib.load(dense_path), joblib.load(teacher_path)
    if adopted["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("Pooled adopted model is paired with another teacher")
    return teacher, adopted, teacher_path, dense_path


def freeze_pooled(selection_path: Path) -> dict:
    """Bind pooled TRAIN stages to the final development selection, no fitting."""
    if POOL_DECL.exists() or POOL_SEAL.exists():
        raise FileExistsError("Pooled training declaration already exists")
    check(full=True)
    if not TRAIN_SEAL.is_file():
        raise FileNotFoundError("Seal all six provider-fold continuation fits first")
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    if selection.get("study_id") != STUDY or selection.get("final_selection") is not True:
        raise ValueError("Pooled training needs the final frozen development selection")
    stage = int(selection["training_stage"])
    if stage not in (512, 1024):
        raise ValueError("Selected pooled reproduction stage must be 512 or 1024")
    inputs = {str(selection_path): sha(selection_path),
              str(TRAIN_DECL): sha(TRAIN_DECL), str(TRAIN_SEAL): sha(TRAIN_SEAL),
              str(PLAN): sha(PLAN)}
    paths = []
    for seed in SEEDS:
        teacher, adopted, teacher_path, dense_path = _pool_teacher_and_adopted(seed)
        paths.extend((teacher_path, dense_path))
    for path in paths:
        inputs[str(path)] = sha(path)
    pool_decl = dict(schema="v2_spell_balanced_pooled_v1", study_id=STUDY,
        selection_path=str(selection_path), selection_sha256=sha(selection_path),
        selected_training_recipe=selection.get("recipe"), training_stage=stage,
        seed_resets_repeat_the_same_512_draw_schedule=True,
        maximum_new_fits=2 if stage == 512 else 4,
        stages=[512] if stage == 512 else [512, 1024],
        stage_recipe={"512": {"updates": 512, "lr": LR_STAGE_512},
                      "1024": {"updates": 512, "lr": LR_STAGE_1024}},
        initialization="Exact adopted pooled original_full H/A and d_state; paired adopted pooled teacher and normalization",
        data="All 13 original TRAIN matches plus 45 GAR TRAIN games, same distinct-possession sampler and four fixed v2 views",
        pooled_train_matches=sorted(row["match"] for row in _bind_prior_recipe().prep.split_plan(fold="pooled")["train"]),
        input_sha256=inputs, source_sha256=_source_facts(),
        organizer_reference_access=False, gar_valid_test_access=False,
        public_or_GAR_scoring=False, promotion=False)
    write_once(POOL_DECL, pool_decl)
    return pool_decl


def check_pooled(full: bool = True) -> dict:
    _bind_prior_recipe()
    declaration = json.loads(POOL_DECL.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("source_sha256") != _source_facts():
        raise ValueError("Pooled training source changed after freeze")
    if full:
        for raw_path, digest in declaration["input_sha256"].items():
            if sha(raw_path) != digest:
                raise ValueError(f"Pooled training input changed: {raw_path}")
    return declaration


def fit_pooled_stage(seed: int, stage: int, *, verbose: bool = True) -> dict:
    """Fit one pooled stage after final development selection and pool freeze."""
    if seed not in SEEDS or stage not in (512, 1024):
        raise ValueError("Use a frozen seed and cumulative stage 512 or 1024")
    pool_declaration = check_pooled(full=True)
    selected_stage = int(pool_declaration["training_stage"])
    if stage > selected_stage:
        raise ValueError("Cannot fit a pooled stage beyond the selected development stage")
    if stage == 1024 and selected_stage != 1024:
        raise ValueError("Stage 1024 is authorized only when that stage was selected")
    path = _pooled_output_path(seed, stage)
    receipt_path = Path(str(path) + ".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    recipe = _bind_prior_recipe()
    teacher, adopted, teacher_path, adopted_path = _pool_teacher_and_adopted(seed)
    if stage == 512:
        prior_path = adopted_path
        cumulative, lr = 512, LR_STAGE_512
        prior = adopted
    else:
        prior_path = _pooled_output_path(seed, 512)
        stage512_receipt = json.loads(Path(str(prior_path) + ".json").read_text())
        if (stage512_receipt.get("study_id") != STUDY or
                stage512_receipt.get("stage") != 512 or
                stage512_receipt.get("pooled_declaration_sha256") != sha(POOL_DECL) or
                stage512_receipt.get("sha256") != sha(prior_path)):
            raise ValueError(f"Stage-512 pooled source receipt is invalid: {seed}")
        prior = _load_checkpoint(prior_path)
        cumulative, lr = 1024, LR_STAGE_1024
    old_records = pooled_training()
    gar_records = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    if len({str(row["match"]) for row in gar_records}) != 45:
        raise ValueError("Pooled fit must use all 45 GAR TRAIN games")
    sampler = recipe.short.OriginalDiversitySampler(
        old_records, gar_records, list(teacher["feature_names"]), seed)
    start = time.monotonic()
    with _trainer_scope(lr):
        with recipe._round_robin_v2_windows(old_records) as view_draws:
            with recipe.short.parent.base.installed(sampler):
                result = iterative.fit_from_checkpoints(
                    teacher, prior, old_records, gar_records,
                    seed=seed, arm="ce_control", verbose=verbose)
    model = _annotate_result(result, teacher=teacher, stage=cumulative,
        cumulative_updates=cumulative, warmstart_path=prior_path,
        arm="pooled_ce_control",
        schedule="same seed reset as stage 512; repeated sample schedule, no fresh-window claim")
    for field in ("median", "scale", "feature_names"):
        if field == "feature_names":
            if model[field] != adopted[field]:
                raise ValueError("Pooled model feature schema changed")
        else:
            np.testing.assert_array_equal(model[field], adopted[field])
    if set(model["d_state"]) != set(adopted["d_state"]):
        raise ValueError("Pooled D state keys changed")
    for name, value in adopted["d_state"].items():
        np.testing.assert_array_equal(model["d_state"][name], value)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    receipt = dict(study_id=STUDY, stage=cumulative, stage_updates=STAGE_UPDATES,
        cumulative_updates=cumulative, seed=seed, path=str(path.relative_to(ROOT)),
        sha256=sha(path), pooled_declaration_sha256=sha(POOL_DECL),
        wall_seconds=time.monotonic()-start, config=result["config"],
        warmstart_path=str(prior_path), warmstart_sha256=sha(prior_path),
        teacher_path=str(teacher_path), teacher_sha256=sha(teacher_path),
        adopted_d_state_unchanged=True, sampling=sampler.receipt(),
        view_draws=dict(view_draws), exposures=result["exposures"],
        loss_intervals=result["loss_intervals"],
        old_matches=sorted({str(row["match"]) for row in old_records}),
        gar_games=sorted({str(row["match"]) for row in gar_records}),
        organizer_reference_access=False, gar_valid_test_access=False)
    write_once(receipt_path, receipt)
    return receipt


def seal_pooled() -> dict:
    declaration = check_pooled(full=True)
    if POOL_SEAL.exists():
        raise FileExistsError(POOL_SEAL)
    fits = []
    for seed in SEEDS:
        for stage in declaration["stages"]:
            path = _pooled_output_path(seed, stage)
            receipt = json.loads(Path(str(path) + ".json").read_text())
            if (receipt.get("pooled_declaration_sha256") != sha(POOL_DECL) or
                    receipt.get("sha256") != sha(path)):
                raise ValueError(f"Pooled stage receipt mismatch: {seed}/{stage}")
            fits.append(receipt)
    expected = 2 if declaration["training_stage"] == 512 else 4
    if len(fits) != expected:
        raise ValueError(f"Expected {expected} pooled stage fits; found {len(fits)}")
    result = dict(study_id=STUDY, pooled_declaration_sha256=sha(POOL_DECL),
        fits=fits, all_authorized_stages_complete=True,
        repeated_seed_schedule=True, organizer_reference_access=False, promotion=False)
    write_once(POOL_SEAL, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit-fold", "seal-folds",
                                           "freeze-pooled", "fit-pooled", "seal-pooled"))
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--stage", type=int, choices=(512, 1024))
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "fit-fold":
        if args.fold is None or args.seed is None:
            parser.error("fit-fold requires --fold and --seed")
        fit_fold(args.fold, args.seed)
    elif args.action == "seal-folds":
        seal_folds()
    elif args.action == "freeze-pooled":
        if args.selection is None:
            parser.error("freeze-pooled requires --selection")
        freeze_pooled(args.selection)
    elif args.action == "fit-pooled":
        if args.seed is None or args.stage is None:
            parser.error("fit-pooled requires --seed and --stage")
        fit_pooled_stage(args.seed, args.stage)
    else:
        seal_pooled()


if __name__ == "__main__":
    main()
