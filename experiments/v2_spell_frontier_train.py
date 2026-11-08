"""Paired repeat-versus-fresh-draw continuation for the v2 spell frontier study."""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
from pathlib import Path
import random as std_random
import time
from types import SimpleNamespace

import joblib
import numpy as np

from experiments import v2_iterative_train as iterative
from experiments import v2_spell_balanced_train as parent_train
from experiments import v2_iterative_train_run as prior_run
from experiments import soccernet_capacity_model as cap

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_spell_frontier_v1"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
DECL = ROOT / f"experiments/results/{STUDY}_training_declaration.json"
SEAL = ROOT / f"experiments/results/{STUDY}_training_seal.json"
POOL_DECL = ROOT / f"experiments/results/{STUDY}_pooled_declaration.json"
POOL_SEAL = ROOT / f"experiments/results/{STUDY}_pooled_seal.json"
OUT = ROOT / "data/experiments" / STUDY / "train"
FOLDS, SEEDS = iterative.FOLDS, iterative.SEEDS
ARMS = ("repeat", "fresh")
UPDATES = 512
LR = 1e-4
FRESH_OFFSET = 104729


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


@contextmanager
def rng_scope(offset: int):
    """Offset only the trainer's local Python RNG constructor, restoring it reliably."""
    if offset not in (0, FRESH_OFFSET):
        raise ValueError("Only repeat (0) and fresh (104729) offsets are declared")
    old_random, old_lr, old_study = iterative.random, iterative.LR, iterative.STUDY
    iterative.random = SimpleNamespace(Random=lambda seed: std_random.Random(int(seed) + offset))
    iterative.LR, iterative.STUDY = LR, STUDY
    try:
        yield
    finally:
        iterative.random, iterative.LR, iterative.STUDY = old_random, old_lr, old_study


def prior_path(fold, seed):
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Undeclared fold/seed")
    return parent_train.output_path(fold, seed)


def output_path(arm, fold, seed):
    if fold not in FOLDS or seed not in SEEDS or arm not in ARMS:
        raise ValueError("Undeclared fit")
    return OUT / "models" / arm / f"{fold}_{seed}.joblib"


def pooled_output_path(arm, seed):
    if arm not in ("repeat", "fresh") or seed not in SEEDS:
        raise ValueError("Pooled new-arm outputs require repeat/fresh and a declared seed")
    return OUT / "models" / "pooled" / arm / f"stage1536_{seed}.joblib"


def selected_pooled_model_path(selection, seed):
    """Resolve selected deployment weights, reusing base weights without refitting."""
    arm, stage = selection["training_arm"], int(selection["training_stage"])
    if arm == "base" and stage == 1024:
        return parent_train._pooled_output_path(seed, 1024)
    if arm in ARMS and stage == 1536:
        return pooled_output_path(arm, seed)
    raise ValueError("Selection must be base/stage1024 or repeat|fresh/stage1536")


def pooled_warmstart(seed):
    """Load the balanced continuation checkpoint, never the adopted base model."""
    if seed not in SEEDS:
        raise ValueError("Undeclared seed")
    path = parent_train._pooled_output_path(seed, 1024)
    model = joblib.load(path)
    if (model.get("training", {}).get("study") != parent_train.STUDY or
            model.get("training", {}).get("stage") != 1024):
        raise ValueError("Pooled warm start is not the sealed balanced stage-1024 checkpoint")
    receipt = Path(str(path) + ".json")
    if not receipt.is_file() or json.loads(receipt.read_text()).get("sha256") != sha(path):
        raise ValueError("Pooled warm start receipt/hash mismatch")
    return model, path


@contextmanager
def capture_windows(recipe, old_records, gar_records, sampler):
    """Capture exact record/window identifiers after the existing sampler hook."""
    rows = []
    with recipe.short.parent.base.installed(sampler):
        old_draw, new_draw = cap._draw_old, cap._draw_new

        def observed_old(rng, groups, records):
            draw = old_draw(rng, groups, records)
            ri, start, size, _full, alternate = draw
            record = records[ri]
            rows.append(("old", str(record.get("source", "")), str(record["match"]),
                         int(start), int(size), bool(alternate)))
            return draw

        def observed_new(rng, groups):
            draw = new_draw(rng, groups)
            ri, start, size, _full, _alternate = draw
            rows.append(("new", "", str(gar_records[ri]["match"]), int(start), int(size), False))
            return draw

        cap._draw_old, cap._draw_new = observed_old, observed_new
        try:
            yield rows
        finally:
            cap._draw_old, cap._draw_new = old_draw, new_draw


def make_sampler(recipe, old, gar, names, seed, offset):
    """Offset only the distinct-possession stream; keep quality/base streams paired."""
    sampler = recipe.short.OriginalDiversitySampler(old, gar, names, seed)
    sampler.original_rng = std_random.Random(seed + offset + 1106644)
    return sampler


def _sources():
    recipe = prior_run.bind()
    paths = {Path(__file__), Path(__file__).with_name("v2_spell_balanced_train.py"),
             Path(iterative.__file__), Path(prior_run.__file__),
             ROOT / "experiments/tests/test_v2_spell_frontier_train.py",
             PLAN, ROOT / "experiments/results/v2_spell_balanced_v1_training_declaration.json",
             ROOT / "experiments/results/v2_spell_balanced_v1_training_seal.json",
             ROOT / "experiments/results/v2_spell_balanced_v1_pooled_declaration.json",
             ROOT / "experiments/results/v2_spell_balanced_v1_pooled_seal.json",
             ROOT / "experiments/baseline_manifest.json",
             *recipe._source_files(), *recipe._view_pipeline_files()}
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def _input_facts():
    prior_run.bind()
    parent_train.check(full=True)
    parent_train.check_pooled(full=True)
    facts = {}
    for p in (parent_train.TRAIN_DECL, parent_train.TRAIN_SEAL,
              parent_train.POOL_DECL, parent_train.POOL_SEAL):
        facts[str(p.relative_to(ROOT))] = sha(p)
    for fold in FOLDS:
        for seed in SEEDS:
            p = prior_path(fold, seed)
            facts[str(p.relative_to(ROOT))] = sha(p)
            facts[str(Path(str(p) + ".json").relative_to(ROOT))] = sha(Path(str(p) + ".json"))
    for fold in FOLDS:
        for seed in SEEDS:
            p = prior_run.recipe.rf.sc.initial_path(fold, seed)
            facts[str(p)] = sha(p)
    return facts


def freeze():
    if DECL.exists() or SEAL.exists():
        raise FileExistsError("Preserve the frozen frontier study artifacts")
    plan = json.loads(PLAN.read_text())
    if plan.get("study_id") != STUDY:
        raise ValueError("Unexpected frontier plan")
    parent_train.check(full=True)
    parent_train.check_pooled(full=True)
    pool_seal = json.loads(parent_train.POOL_SEAL.read_text())
    if pool_seal.get("all_authorized_stages_complete") is not True:
        raise ValueError("Selected stage-1024 pooled source is not sealed")
    source_facts = _sources()
    declaration = {
        "schema": "v2_spell_frontier_training_v1", "study_id": STUDY,
        "plan_sha256": sha(PLAN), "source_sha256": source_facts,
        "input_sha256": _input_facts(), "folds": list(FOLDS), "seeds": list(SEEDS),
        "arms": list(ARMS), "updates_per_arm": UPDATES,
        "cumulative_continuation_updates": 1536,
        "learning_rate": LR, "batch_size": iterative.BATCH_SIZE,
        "gar_fraction": iterative.FRACTION_GAR, "max_workers": 2,
        "warm_start": "sealed v2_spell_balanced_v1 stage-1024 CE checkpoint, same fold/seed",
        "offsets": {"repeat": 0, "fresh": FRESH_OFFSET},
        "freshness_scope": "Offset only the distinct-possession sampler original_rng and the trainer-local draw RNG. Sampler base/quality RNG and Torch/dropout seed remain unchanged.",
        "loss": "Unchanged dense duration-weighted H/A CE plus live-transition objective",
        "d_state_normalization_teacher": "Copied unchanged from paired stage-1024 checkpoint/teacher",
        "selection_claim": "Matched continuation with repeated versus deterministic offset draw schedule; no new labels/data/views.",
        "organizer_reference_access": False, "gar_valid_test_access": False,
        "public_scoring": False, "promotion": False}
    write_once(DECL, declaration)
    return declaration


def check(full=True):
    prior_run.bind()
    declaration = json.loads(DECL.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("source_sha256") != _sources():
        raise ValueError("Frontier training source changed after freeze")
    if full and declaration.get("input_sha256") != _input_facts():
        raise ValueError("Frontier inputs changed after freeze")
    return declaration


def fit_one(fold, seed, arm, *, verbose=True):
    if arm not in ARMS or fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Undeclared fold, seed or arm")
    declaration = check(full=True)
    path = output_path(arm, fold, seed)
    receipt_path = Path(str(path) + ".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    recipe = prior_run.bind()
    teacher_path = recipe.rf.sc.initial_path(fold, seed)
    teacher, prior = joblib.load(teacher_path), joblib.load(prior_path(fold, seed))
    if prior["training"].get("study") != parent_train.STUDY or prior["training"].get("stage") != 1024:
        raise ValueError("Warm start is not the sealed stage-1024 fold checkpoint")
    if prior["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("Warm-start teacher mismatch")
    old = recipe.training(fold)
    gar = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    offset = 0 if arm == "repeat" else FRESH_OFFSET
    sampler = make_sampler(recipe, old, gar, list(teacher["feature_names"]), seed, offset)
    start = time.monotonic()
    with recipe._round_robin_v2_windows(old) as views:
        with capture_windows(recipe, old, gar, sampler) as sampled_windows:
            with rng_scope(offset):
                result = iterative.fit_from_checkpoints(
                    teacher, prior, old, gar, seed=seed, arm="ce_control", verbose=verbose)
    model = result["model"]
    if model["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("Trainer export changed teacher identity")
    if set(model["d_state"]) != set(prior["d_state"]):
        raise ValueError("D state keys changed")
    for key, value in prior["d_state"].items():
        np.testing.assert_array_equal(model["d_state"][key], value)
    for key in ("median", "scale"):
        np.testing.assert_array_equal(model[key], prior[key])
    if model["feature_names"] != prior["feature_names"]:
        raise ValueError("Normalization schema changed")
    provenance = {"study": STUDY, "arm": arm, "stage": 1536,
        "stage_updates": UPDATES, "cumulative_updates": 1536,
        "draw_seed_offset": offset, "warm_start_path": str(prior_path(fold, seed).relative_to(ROOT)),
        "warm_start_sha256": sha(prior_path(fold, seed))}
    model["training"].update(provenance)
    result["config"].update(provenance)
    model["step"] = 1536
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    unique_windows = sorted({tuple(row) for row in sampled_windows})
    draw_bytes = json.dumps(sampled_windows, separators=(",", ":")).encode()
    receipt = {"study_id": STUDY, **provenance, "fold": fold, "seed": seed,
        "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "declaration_sha256": sha(DECL), "teacher_sha256": sha(teacher_path),
        "wall_seconds": time.monotonic()-start, "config": result["config"],
        "sampling": sampler.receipt(), "view_draws": dict(views),
        "exposures": result["exposures"], "loss_intervals": result["loss_intervals"],
        "sampled_window_ids": [list(row) for row in unique_windows],
        "draw_window_sequence_sha256": hashlib.sha256(draw_bytes).hexdigest(),
        "draw_window_count": len(sampled_windows),
        "d_state_unchanged": True, "organizer_reference_access": False}
    write_once(receipt_path, receipt)
    return receipt


def fit_all(workers=2):
    if workers not in (1, 2):
        raise ValueError("Use one or two workers")
    check(full=True)
    jobs = [(f, s, arm) for f in FOLDS for s in SEEDS for arm in ARMS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_one, *job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print("FIT", row["fold"], row["seed"], row["arm"], row["sha256"], flush=True)
    return rows


def seal():
    declaration = check(full=True)
    fits = []
    for fold in FOLDS:
        for seed in SEEDS:
            for arm in ARMS:
                path = output_path(arm, fold, seed)
                receipt = json.loads(Path(str(path) + ".json").read_text())
                if (receipt.get("declaration_sha256") != sha(DECL) or
                        receipt.get("sha256") != sha(path) or receipt.get("d_state_unchanged") is not True):
                    raise ValueError(f"Unsealed fit receipt: {fold}/{seed}/{arm}")
                fits.append(receipt)
    if len(fits) != 12:
        raise ValueError("Expected all 12 paired fits")
    by_fit = {(row["fold"], int(row["seed"]), row["arm"]): row for row in fits}
    paired_overlap = {}
    for fold in FOLDS:
        for seed in SEEDS:
            repeat = by_fit[(fold, seed, "repeat")]
            fresh = by_fit[(fold, seed, "fresh")]
            repeat_ids = {tuple(row) for row in repeat["sampled_window_ids"]}
            fresh_ids = {tuple(row) for row in fresh["sampled_window_ids"]}
            paired_overlap[f"{fold}/{seed}"] = {
                "repeat_unique_windows": len(repeat_ids),
                "fresh_unique_windows": len(fresh_ids),
                "overlap_unique_windows": len(repeat_ids & fresh_ids),
                "different_sequence_digest": repeat["draw_window_sequence_sha256"] != fresh["draw_window_sequence_sha256"],
                "different_window_inventory": repeat_ids != fresh_ids}
    if not all(row["different_sequence_digest"] for row in paired_overlap.values()):
        raise ValueError("Fresh schedule did not produce different sampled windows")
    result = {"study_id": STUDY, "declaration_sha256": sha(DECL), "fits": fits,
              "paired_draw_window_overlap": paired_overlap,
              "all_12_complete": True, "organizer_reference_access": False,
              "public_scoring": False, "promotion": False}
    write_once(SEAL, result)
    return result


def freeze_pooled(selection_path):
    """Freeze only the selected pooled continuation; base selection adds no fit."""
    if POOL_DECL.exists() or POOL_SEAL.exists():
        raise FileExistsError("Preserve the existing pooled frontier declaration")
    check(full=True)
    selection_path = Path(selection_path)
    selection = json.loads(selection_path.read_text())
    if selection.get("study_id") != STUDY or selection.get("final_selection") is not True:
        raise ValueError("Pooled reproduction requires the final frozen selection")
    arm, stage = selection["training_arm"], int(selection["training_stage"])
    if (arm, stage) not in (("base", 1024), ("repeat", 1536), ("fresh", 1536)):
        raise ValueError("Selection must be base/1024 or repeat|fresh/1536")
    selected_hashes = {str(selection_path): sha(selection_path), str(DECL): sha(DECL),
                       str(SEAL): sha(SEAL), str(parent_train.POOL_DECL): sha(parent_train.POOL_DECL),
                       str(parent_train.POOL_SEAL): sha(parent_train.POOL_SEAL)}
    teachers = {}
    for seed in SEEDS:
        teacher, adopted, teacher_path, model_path = parent_train._pool_teacher_and_adopted(seed)
        teachers[str(seed)] = {"teacher_path": str(teacher_path), "teacher_sha256": sha(teacher_path),
                               "adopted_path": str(model_path), "adopted_sha256": sha(model_path)}
        selected_hashes[str(teacher_path)] = sha(teacher_path)
        stage1024 = parent_train._pooled_output_path(seed, 1024)
        selected_hashes[str(stage1024)] = sha(stage1024)
    if arm != "base":
        if not SEAL.is_file():
            raise FileNotFoundError("Seal the 12 fold fits before pooled reproduction")
        fit_seal = json.loads(SEAL.read_text())
        if (fit_seal.get("all_12_complete") is not True or
                fit_seal.get("declaration_sha256") != sha(DECL)):
            raise ValueError("Frontier fold-fit seal is incomplete or mismatched")
        fit_rows = {(row["fold"], int(row["seed"]), row["arm"]): row
                    for row in fit_seal.get("fits", [])}
        for fold in FOLDS:
            for seed in SEEDS:
                row = fit_rows.get((fold, seed, arm))
                if row is None:
                    raise ValueError(f"Selected fold checkpoint is absent: {fold}/{seed}/{arm}")
                model_path = ROOT / row["path"]
                if sha(model_path) != row.get("sha256"):
                    raise ValueError(f"Selected fold checkpoint changed: {fold}/{seed}/{arm}")
                selected_hashes[str(model_path)] = sha(model_path)
    new_fits = 0 if arm == "base" else 2
    declaration = {"schema": "v2_spell_frontier_pooled_v1", "study_id": STUDY,
        "selection_path": str(selection_path), "selection_sha256": sha(selection_path),
        "training_arm": arm, "training_stage": stage,
        "new_fit_required": arm != "base", "new_fit_count": new_fits,
        "warmstart": "sealed balanced pooled stage1024 checkpoint by seed" if arm != "base" else "reuse sealed balanced pooled stage1024 checkpoints",
        "teachers_and_adopted_models": teachers, "input_sha256": selected_hashes,
        "source_sha256": _sources(), "organizer_reference_access": False,
        "gar_valid_test_access": False, "public_scoring": False, "promotion": False}
    write_once(POOL_DECL, declaration)
    return declaration


def check_pooled(full=True):
    declaration = json.loads(POOL_DECL.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("source_sha256") != _sources():
        raise ValueError("Pooled frontier source changed after freeze")
    if full:
        for path, expected in declaration["input_sha256"].items():
            if sha(path) != expected:
                raise ValueError(f"Pooled frontier input changed: {path}")
    return declaration


def fit_pooled_one(seed, *, verbose=True):
    declaration = check_pooled(full=True)
    arm, stage = declaration["training_arm"], int(declaration["training_stage"])
    if arm == "base":
        raise ValueError("Base selection reuses balanced pooled models and requires no fit")
    if seed not in SEEDS:
        raise ValueError("Undeclared pooled seed")
    path = pooled_output_path(arm, seed)
    receipt_path = Path(str(path) + ".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    recipe = prior_run.bind()
    teacher, adopted, teacher_path, _adopted_path = parent_train._pool_teacher_and_adopted(seed)
    warmstart, warmstart_path = pooled_warmstart(seed)
    if warmstart["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("Balanced pooled warm-start teacher mismatch")
    old = parent_train.pooled_training()
    gar = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    if len({str(row["match"]) for row in old}) != 13 or len({str(row["match"]) for row in gar}) != 45:
        raise ValueError("Pooled frontier fit requires all 13 original and 45 GAR TRAIN matches")
    offset = 0 if arm == "repeat" else FRESH_OFFSET
    sampler = make_sampler(recipe, old, gar, list(teacher["feature_names"]), seed, offset)
    start = time.monotonic()
    with recipe._round_robin_v2_windows(old) as views:
        with capture_windows(recipe, old, gar, sampler) as sampled_windows:
            with rng_scope(offset):
                result = iterative.fit_from_checkpoints(teacher, warmstart, old, gar,
                    seed=seed, arm="ce_control", verbose=verbose)
    model = result["model"]
    if model["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("Pooled frontier export changed teacher identity")
    for key in ("median", "scale"):
        np.testing.assert_array_equal(model[key], warmstart[key])
    if model["feature_names"] != warmstart["feature_names"] or set(model["d_state"]) != set(warmstart["d_state"]):
        raise ValueError("Pooled frontier normalization or D schema changed")
    for key, value in warmstart["d_state"].items():
        np.testing.assert_array_equal(model["d_state"][key], value)
    provenance = {"study": STUDY, "arm": arm, "stage": 1536, "stage_updates": UPDATES,
        "cumulative_updates": 1536, "draw_seed_offset": offset,
        "warm_start_path": str(warmstart_path.relative_to(ROOT)),
        "warm_start_sha256": sha(warmstart_path)}
    model["training"].update(provenance)
    result["config"].update(provenance)
    model["step"] = 1536
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    unique_windows = sorted({tuple(row) for row in sampled_windows})
    receipt = {"study_id": STUDY, **provenance, "seed": seed,
        "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "declaration_sha256": sha(POOL_DECL), "teacher_sha256": sha(teacher_path),
        "base_pooled_stage1024_sha256": sha(warmstart_path),
        "wall_seconds": time.monotonic()-start, "config": result["config"],
        "sampling": sampler.receipt(), "view_draws": dict(views),
        "sampled_window_ids": [list(row) for row in unique_windows],
        "draw_window_sequence_sha256": hashlib.sha256(json.dumps(sampled_windows,
            separators=(",", ":")).encode()).hexdigest(),
        "draw_window_count": len(sampled_windows), "d_state_unchanged": True,
        "organizer_reference_access": False}
    write_once(receipt_path, receipt)
    return receipt


def seal_pooled():
    declaration = check_pooled(full=True)
    arm = declaration["training_arm"]
    if arm == "base":
        models = {str(seed): {"path": str(selected_pooled_model_path(
            {"training_arm": "base", "training_stage": 1024}, seed).relative_to(ROOT)),
            "sha256": sha(selected_pooled_model_path(
                {"training_arm": "base", "training_stage": 1024}, seed))}
            for seed in SEEDS}
        fits = []
    else:
        fits = []
        for seed in SEEDS:
            path = pooled_output_path(arm, seed)
            receipt = json.loads(Path(str(path) + ".json").read_text())
            if (receipt.get("declaration_sha256") != sha(POOL_DECL) or
                    receipt.get("sha256") != sha(path) or receipt.get("d_state_unchanged") is not True):
                raise ValueError(f"Invalid pooled frontier fit receipt: {seed}")
            fits.append(receipt)
        if len(fits) != 2:
            raise ValueError("Expected exactly two selected-arm pooled fits")
        models = {str(seed): {"path": str(pooled_output_path(arm, seed).relative_to(ROOT)),
                              "sha256": sha(pooled_output_path(arm, seed))} for seed in SEEDS}
    result = {"study_id": STUDY, "pooled_declaration_sha256": sha(POOL_DECL),
        "training_arm": arm, "training_stage": declaration["training_stage"],
        "models": models, "fits": fits, "new_fit_count": len(fits),
        "all_selected_pooled_models_ready": True,
        "organizer_reference_access": False, "public_scoring": False, "promotion": False}
    write_once(POOL_SEAL, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit-all", "seal",
                                           "freeze-pooled", "fit-pooled", "seal-pooled"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--selection", type=Path)
    args = parser.parse_args()
    if args.action == "freeze": freeze()
    elif args.action == "fit-all": fit_all(args.workers)
    elif args.action == "seal": seal()
    elif args.action == "freeze-pooled":
        if args.selection is None: parser.error("freeze-pooled requires --selection")
        freeze_pooled(args.selection)
    elif args.action == "fit-pooled":
        if args.seed is None: parser.error("fit-pooled requires --seed")
        fit_pooled_one(args.seed)
    else: seal_pooled()


if __name__ == "__main__":
    main()
