"""Reconstruct the missing pooled CE spell-frontier continuation weights.

This study reuses the frozen numerical fitter and pooled TRAIN record loader,
but owns new declarations, receipts, and output paths.  It does not amend the
historical v2_spell_balanced_v1 declaration or seal.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import joblib
import numpy as np

from experiments import v2_iterative_train as iterative
from experiments import v2_spell_balanced_train as balanced
from experiments import v2_iterative_runtime as iterative_runtime

ROOT = Path(__file__).resolve().parents[1]
STUDY = "frontier_pooled_reconstruction_v1"
OUT = ROOT / "data/experiments" / STUDY
RESULTS = ROOT / "experiments/results"
DECL = RESULTS / f"{STUDY}_declaration.json"
SEAL = RESULTS / f"{STUDY}_seal.json"
HIST_DECL = RESULTS / "v2_spell_balanced_v1_pooled_declaration.json"
HIST_SEAL = RESULTS / "v2_spell_balanced_v1_pooled_seal.json"
HIST_SELECTION = RESULTS / "v2_spell_balanced_v1_selection.json"
TRAIN_DECL = RESULTS / "v2_spell_balanced_v1_training_declaration.json"
TRAIN_SEAL = RESULTS / "v2_spell_balanced_v1_training_seal.json"
PLAN = RESULTS / f"{STUDY}_plan.json"
FRONTIER_MANIFEST = ROOT / "experiments/baselines/joint_gru_sn_dense_spell_frontier_ensemble_v1.json"
FRONTIER_POLICY = ROOT / "experiments/baselines/model_decision_policy_joint_gru_sn_dense_spell_frontier_ensemble_v1.json"
SEEDS = (2026, 2027)
STAGES = (512, 1024)
STAGE_UPDATES = 512
LR = {512: 3e-4, 1024: 1e-4}
LOSS_TOLERANCE = 1e-7
VERIFICATION = RESULTS / f"{STUDY}_verification.json"
ORIGINAL_PREDECLARED_SHA256 = "1f53e94696d62e6988b4c4dea40d4de58f7deede4809f2fbfcd157e2ab53739d"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def original_checkpoint_sha(declaration: dict, seed: int, stage: int) -> str:
    """Resolve stage keys after JSON round-tripping turns them into strings."""
    return declaration["original_checkpoint_sha256"][str(seed)][str(stage)]


def output_path(seed: int, stage: int) -> Path:
    if seed not in SEEDS or stage not in STAGES:
        raise ValueError("Only the two sealed seeds and cumulative stages are allowed")
    return OUT / "models" / f"stage{stage}_{seed}.joblib"


def _recipe():
    """Bind the shared loader to this study's private TRAIN-record adapter."""
    recipe = balanced.prior_run.recipe
    recipe.rf.old.DATA = Path(os.environ.get(
        "MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))).resolve()
    recipe.OUT = OUT
    recipe.ARCHIVE_ROOT = Path(os.environ.get(
        "MTAH_RF_ARCHIVE_DIR",
        str(ROOT.parent / "mtah-rf/experiments/feed_robustness_v1/artifacts")))
    recipe.TRAIN_RECORDS = OUT / "gar"
    target = Path(os.environ.get(
        "V2_ITERATIVE_GAR_ARCHIVE",
        str(ROOT.parent / "mtah-dt/experiments/dense_targeted_followups_v1/artifacts/train_records")))
    link = recipe.TRAIN_RECORDS / "train"
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError("Study GAR link resolves to a different archive")
    elif link.exists():
        if link.resolve() != target.resolve():
            raise RuntimeError("Study GAR path is occupied by an unexpected source")
    elif target.is_dir():
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target, target_is_directory=True)
    return recipe


@contextmanager
def _bound_recipe():
    """Apply the path adapter only within this call, then restore module state."""
    recipe = balanced.prior_run.recipe
    old = (recipe.OUT, recipe.ARCHIVE_ROOT, recipe.TRAIN_RECORDS,
           recipe.rf.old.DATA, balanced._bind_prior_recipe)
    try:
        bound = _recipe()
        balanced._bind_prior_recipe = _recipe
        yield bound
    finally:
        recipe.OUT, recipe.ARCHIVE_ROOT, recipe.TRAIN_RECORDS, recipe.rf.old.DATA = old[:4]
        balanced._bind_prior_recipe = old[4]


def prepare() -> dict:
    """Generate fixed v2 training views inside this study's isolated output."""
    with _bound_recipe() as recipe:
        if (OUT / "preparation.json").exists():
            raise FileExistsError("Reconstruction preparation is already present")
        with iterative_runtime.installed():
            return recipe.prepare(workers=1)


@contextmanager
def _trainer_scope(lr: float):
    old_lr, old_study = iterative.LR, iterative.STUDY
    iterative.LR, iterative.STUDY = float(lr), STUDY
    try:
        yield
    finally:
        iterative.LR, iterative.STUDY = old_lr, old_study


def _warmstart(seed: int) -> tuple[dict, dict, Path, Path]:
    if seed not in SEEDS:
        raise ValueError("Use a declared pooled seed")
    manifest = json.loads(FRONTIER_MANIFEST.read_text())
    model_dir = ROOT / manifest["model_directory"]
    teacher_path = model_dir / f"teacher_adopted_{seed}.joblib"
    adopted_path = model_dir / f"adopted_{seed}.joblib"
    for path in (teacher_path, adopted_path):
        expected = manifest.get("models_sha256", {}).get(path.name)
        if not expected or sha(path) != expected:
            raise ValueError(f"Spell-frontier model does not match its manifest: {path}")
    teacher, adopted = joblib.load(teacher_path), joblib.load(adopted_path)
    if adopted["source"]["checkpoint_sha256"] != iterative.cap._checkpoint_sha(teacher):
        raise ValueError(f"Adopted model/teacher pairing differs for seed {seed}")
    return teacher, adopted, teacher_path, adopted_path


def _data_inputs() -> set[Path]:
    """Enumerate the exact TRAIN inputs consumed by the pooled loader/fitter."""
    with _bound_recipe() as recipe:
        return _data_inputs_bound(recipe)


def _data_inputs_bound(recipe) -> set[Path]:
    spec = recipe.rf.old.old_spec()
    historical_inputs = json.loads(TRAIN_DECL.read_text())["input_sha256"]
    data_dir = recipe.rf.old.DATA
    predeclared = data_dir / "experiments/objective_alignment_v1/predeclared.json"
    prepared_manifest = data_dir / "experiments/objective_alignment_v1/prepared_training_manifest.json"
    cache_manifest = data_dir / "objective_alignment_cache_manifest.json"
    if sha(predeclared) != ORIGINAL_PREDECLARED_SHA256:
        raise ValueError("Original objective-alignment declaration hash mismatch")
    original_spec = json.loads(predeclared.read_text())
    prepared_spec = json.loads(prepared_manifest.read_text())
    if (sha(prepared_manifest) != original_spec["prepared_training_manifest_sha256"] or
            prepared_spec.get("preparation_declaration_sha256") !=
            original_spec["preparation_declaration_sha256"]):
        raise ValueError("Original prepared-cache manifest does not match its declaration")
    cache_manifest_data = json.loads(cache_manifest.read_text())
    cache_manifest_entries = {entry["path"]: entry for entry in cache_manifest_data["files"]}
    rows = recipe.prep.split_plan(fold="pooled")["train"]
    if len(rows) != 13:
        raise ValueError(f"Expected 13 original TRAIN matches, found {len(rows)}")
    archive = recipe._archive_root()
    entries, v1_manifest = recipe._v1_view_manifest()
    archive_root = archive.parents[2]
    paths: set[Path] = {recipe.OUT / "preparation.json", recipe.CALIBRATION,
                        v1_manifest, archive_root / "feed_robustness_v1_cache_manifest.json",
                        predeclared, prepared_manifest, cache_manifest}
    for item in rows:
        match = item["match"]
        raw_path = recipe.prep.data_path(item, data_dir)
        if sha(raw_path) != item["sha256"]:
            raise ValueError(f"Original raw TRAIN input mismatch: {match}")
        if original_spec["data_sha256"].get(item["path"]) != item["sha256"]:
            raise ValueError(f"Raw source is not pinned by the original declaration: {match}")
        paths.add(raw_path)
        cache_info = spec["prepared_training_caches"][match]
        declared_cache = original_spec["prepared_training_caches"][match]
        if cache_info != declared_cache:
            raise ValueError(f"Prepared-cache recipe metadata changed: {match}")
        cache_path = data_dir / Path(cache_info["cache"]).relative_to("data")
        cache_receipt = data_dir / Path(cache_info["receipt"]).relative_to("data")
        if (sha(cache_path) != cache_info["sha256"] or
                cache_path.stat().st_size != cache_info["bytes"] or
                sha(cache_receipt) != cache_info["receipt_sha256"] or
                json.loads(cache_receipt.read_text()).get("sha256") != cache_info["sha256"]):
            raise ValueError(f"Prepared cache or receipt does not match the original declaration: {match}")
        cache_rel = Path(cache_info["cache"]).relative_to("data").as_posix()
        receipt_rel = Path(cache_info["receipt"]).relative_to("data").as_posix()
        manifest_cache = cache_manifest_entries.get(cache_rel)
        manifest_receipt = cache_manifest_entries.get(receipt_rel)
        if (manifest_cache is None or manifest_receipt is None or
                manifest_cache.get("sha256") != cache_info["sha256"] or
                manifest_cache.get("bytes") != cache_info["bytes"] or
                manifest_receipt.get("sha256") != cache_info["receipt_sha256"]):
            raise ValueError(f"Cache manifest does not seal the original prepared cache: {match}")
        paths.update((cache_path, cache_receipt))
        if (match not in prepared_spec["caches"] or
                prepared_spec["caches"][match] != cache_info):
            raise ValueError(f"Prepared manifest does not list original cache: {match}")
        v1_path = archive / "views" / f"{match}.joblib"
        v1_entry = entries.get(v1_path.relative_to(archive_root).as_posix())
        if (v1_entry is None or sha(v1_path) != v1_entry.get("sha256") or
                v1_path.stat().st_size != v1_entry.get("bytes")):
            raise ValueError(f"Original v1 view does not match its archive manifest: {match}")
        paths.add(v1_path)
        rel = v1_path.relative_to(archive_root).as_posix()
        if rel not in entries:
            raise ValueError(f"Unmanifested original TRAIN view: {rel}")
        for realization in range(recipe.N_REALIZATIONS):
            view = recipe._view_path(match, realization)
            original_view = ROOT / "data/experiments/v2_iterative_train_v1/views" / view.name
            original_receipt = Path(str(original_view) + ".json")
            expected_view_sha = historical_inputs.get(str(original_view))
            expected_receipt_sha = historical_inputs.get(str(original_receipt))
            if (not expected_view_sha or not expected_receipt_sha or
                    sha(view) != expected_view_sha or sha(Path(str(view) + ".json")) != expected_receipt_sha):
                raise ValueError(f"Regenerated v2 view differs from historical sealed bytes: {match}/r{realization}")
            paths.update((view, Path(str(view) + ".json")))
    manifest = recipe.TRAIN_RECORDS / "train/manifest.json"
    paths.add(manifest)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    game_files = json.loads(manifest.read_text()).get("game_files", {})
    if len(game_files) != 45:
        raise ValueError(f"Expected 45 sealed GAR TRAIN games, found {len(game_files)}")
    for filename in game_files:
        game_path = manifest.parent / "games" / filename
        expected = game_files[filename]
        legacy_path = ROOT / "data/experiments/v2_iterative_train_v1/gar/train/games" / filename
        if (sha(game_path) != expected or historical_inputs.get(str(legacy_path)) != expected):
            raise ValueError(f"GAR TRAIN game is not the historical sealed file: {filename}")
        paths.add(game_path)
    legacy_manifest = ROOT / "data/experiments/v2_iterative_train_v1/gar/train/manifest.json"
    if historical_inputs.get(str(legacy_manifest)) != sha(manifest):
        raise ValueError("GAR TRAIN manifest differs from the historical declaration")
    return paths


def _source_paths() -> set[Path]:
    with _bound_recipe() as recipe:
        hist = json.loads(HIST_DECL.read_text())
        paths = {ROOT / relative for relative in hist["source_sha256"]
                 if relative not in ("experiments/baseline_manifest.json",
                                     "experiments/model_decision_policy.json")}
        paths.update((Path(__file__), Path(iterative.__file__), Path(balanced.__file__),
                      Path(balanced.prior_run.__file__), Path(iterative_runtime.__file__),
                      HIST_DECL, HIST_SEAL, HIST_SELECTION, TRAIN_DECL, TRAIN_SEAL, PLAN,
                      FRONTIER_MANIFEST, FRONTIER_POLICY,
                      ROOT / "experiments/tests/test_frontier_pooled_reconstruction.py"))
        paths.update(path for path in recipe._source_files()
                     if path.name not in ("baseline_manifest.json", "model_decision_policy.json"))
        paths.update(recipe._view_pipeline_files())
        return paths


def _input_paths() -> set[Path]:
    paths = _data_inputs()
    paths.update((HIST_DECL, HIST_SEAL, HIST_SELECTION, TRAIN_DECL, TRAIN_SEAL,
                  PLAN, FRONTIER_MANIFEST, FRONTIER_POLICY))
    for seed in SEEDS:
        _, _, teacher_path, adopted_path = _warmstart(seed)
        paths.update((teacher_path, adopted_path))
    return paths


def _preparation_receipt_comparison() -> dict[str, Any]:
    historical_inputs = json.loads(TRAIN_DECL.read_text())["input_sha256"]
    historical_path = ROOT / "data/experiments/v2_iterative_train_v1/preparation.json"
    expected_sha = historical_inputs.get(str(historical_path))
    current_path = OUT / "preparation.json"
    current = json.loads(current_path.read_text())
    normalized = dict(current)
    # The historical runner prepared views with its default of two workers;
    # this reconstruction used one worker to keep the preparation process small.
    original_workers = normalized.get("workers")
    normalized["workers"] = 2
    # The frozen preparation helper wrote compact JSON with indentation and no
    # trailing newline. Recreate that serialization only for the comparison;
    # the actual one-worker receipt remains untouched and hash-bound.
    normalized_sha = hashlib.sha256(json.dumps(normalized, indent=1).encode()).hexdigest()
    if not expected_sha or original_workers != 1 or normalized_sha != expected_sha:
        raise ValueError("Preparation receipt differs beyond the declared worker-count metadata")
    return dict(path=str(current_path.relative_to(ROOT)), sha256=sha(current_path),
        historical_sha256=expected_sha, normalized_sha256=normalized_sha,
        exact_after_normalizing_worker_count=True,
        metadata_difference=dict(field="workers", reconstructed=original_workers,
                                 historical=2))


def _facts(paths: set[Path]) -> dict[str, str]:
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError("Missing reconstruction inputs:\n" + "\n".join(missing))
    return {str(path): sha(path) for path in sorted(paths)}


def freeze() -> dict:
    if DECL.exists() or SEAL.exists():
        raise FileExistsError("Reconstruction study is already frozen or sealed")
    plan = json.loads(PLAN.read_text())
    if (plan.get("study") != STUDY or
            plan.get("status") != "declared_before_new_preparation_fits_or_scoring"):
        raise ValueError("Unexpected or late reconstruction plan")
    historical = json.loads(HIST_DECL.read_text())
    seal = json.loads(HIST_SEAL.read_text())
    selected = json.loads(HIST_SELECTION.read_text())
    if (historical.get("study_id") != "v2_spell_balanced_v1" or
            not seal.get("all_authorized_stages_complete") or
            seal.get("pooled_declaration_sha256") != sha(HIST_DECL) or
            sha(HIST_SEAL) != plan["source_recipe"]["seal_sha256"] or
            sha(HIST_DECL) != plan["source_recipe"]["declaration_sha256"]):
        raise ValueError("Historical pooled declaration/seal is inconsistent")
    if selected.get("final_selection") is not True or selected.get("training_stage") != 1024:
        raise ValueError("Historical final selection does not authorize stage 1024")
    if (sha(HIST_SELECTION) != historical.get("selection_sha256") or
            selected.get("recipe") != historical.get("selected_training_recipe")):
        raise ValueError("Historical pooled selection hash/recipe binding differs")
    if historical.get("selected_training_recipe", {}).get("id") != "ce0.5_cal0.25_ad0.25_arithmetic_ha3":
        raise ValueError("Unexpected historical pooled recipe")
    mutable_pointers = {"experiments/baseline_manifest.json",
                        "experiments/model_decision_policy.json"}
    for relative, expected_sha in historical["source_sha256"].items():
        if relative in mutable_pointers:
            continue
        source_path = ROOT / relative
        if not source_path.is_file() or sha(source_path) != expected_sha:
            raise ValueError(f"Historical numerical source changed: {relative}")
    data = _facts(_input_paths())
    sources = _facts(_source_paths())
    prep_comparison = _preparation_receipt_comparison()
    expected = {"2026": {512: "d95bb9e08e76690b09632481cfda5f9c422272305b0d5d5eb699c3b2a2dd6071",
                          1024: "d9691336672b13191bf2ce47715fc904f3bc6c14b8477b2f6ff2425d9f78d9c2"},
                "2027": {512: "b302362bdfd9c8395285bdce32e70e260136c51608d4e52d0030ecb5e449a7f7",
                          1024: "4592c10dc06c947077f6d8569833c6d229d3a5620350740edd26fada6d83b0fa"}}
    declaration = dict(
        schema="frontier_pooled_reconstruction_v1", study_id=STUDY,
        plan_path=str(PLAN.relative_to(ROOT)), plan_sha256=sha(PLAN),
        question="Can the unavailable pooled CE spell-frontier package weights be reconstructed from the frozen training recipe and byte-identical adopted initialization?",
        status="reconstructed_comparator; original checkpoint bytes unavailable",
        historical_study_id="v2_spell_balanced_v1",
        historical_pooled_declaration_sha256=sha(HIST_DECL),
        historical_pooled_seal_sha256=sha(HIST_SEAL),
        historical_selection_sha256=sha(HIST_SELECTION),
        initialization="Byte-identical adopted pooled H/A weights and paired teacher from joint_gru_sn_dense_spell_frontier_ensemble_v1; stage 1024 starts from this study's stage-512 reconstruction.",
        seeds=list(SEEDS), stages=list(STAGES), updates_per_stage=STAGE_UPDATES,
        stage_recipe={str(stage): {"updates": STAGE_UPDATES, "lr": LR[stage]} for stage in STAGES},
        maximum_fits=plan["training"]["maximum_fits"],
        execution=plan["training"]["execution"],
        training=dict(batch_size=iterative.BATCH_SIZE, gar_fraction=iterative.FRACTION_GAR,
                      old_matches=13, gar_train_games=45,
                      view_realizations=plan["training"]["v2_realizations"],
                      sampling="Frozen OriginalDiversitySampler; four fixed v2 TRAIN views round-robin; same seed reset repeats the 512 draw schedule"),
        budget=dict(fits=4, sequential=True, cumulative_updates_per_seed=1024),
        original_checkpoint_sha256=expected,
        model_initialization_sha256={str(seed): {"adopted": sha(_warmstart(seed)[3]),
                                                  "teacher": sha(_warmstart(seed)[2])}
                                     for seed in SEEDS},
        input_sha256=data, source_sha256=sources,
        preparation_receipt_comparison=prep_comparison,
        training_loss_absolute_tolerance=LOSS_TOLERANCE,
        organizer_reference_access=False, gar_valid_test_access=False,
        public_or_gar_scoring=False, promotion=False)
    write_once(DECL, declaration)
    return declaration


def check() -> dict:
    declaration = json.loads(DECL.read_text())
    if declaration.get("study_id") != STUDY:
        raise ValueError("Unexpected reconstruction declaration")
    if declaration.get("source_sha256") != _facts(_source_paths()):
        raise ValueError("Reconstruction source changed after freeze")
    if declaration.get("input_sha256") != _facts(_input_paths()):
        raise ValueError("Reconstruction input changed after freeze")
    return declaration


def fit_one(seed: int, stage: int, *, verbose: bool = True) -> dict:
    with _bound_recipe() as recipe:
        return _fit_one_bound(recipe, seed, stage, verbose=verbose)


def _fit_one_bound(recipe, seed: int, stage: int, *, verbose: bool) -> dict:
    declaration = check()
    if seed not in SEEDS or stage not in STAGES:
        raise ValueError("Use a declared seed and cumulative stage")
    if stage == 1024:
        prior_path = output_path(seed, 512)
        prior_receipt_path = Path(str(prior_path) + ".json")
        if not prior_path.is_file() or not prior_receipt_path.is_file():
            raise FileNotFoundError(f"Stage-512 reconstruction required first: {seed}")
        prior_receipt = json.loads(prior_receipt_path.read_text())
        if (prior_receipt.get("seed") != seed or prior_receipt.get("stage") != 512 or
                prior_receipt.get("declaration_sha256") != sha(DECL) or
                prior_receipt.get("sha256") != sha(prior_path)):
            raise ValueError("Stage-512 reconstructed warmstart receipt is invalid")
        prior = joblib.load(prior_path)
    else:
        _, prior, _, prior_path = _warmstart(seed)
    path = output_path(seed, stage)
    receipt_path = Path(str(path) + ".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    teacher, adopted, teacher_path, _ = _warmstart(seed)
    old_records = balanced.pooled_training()
    gar_records = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    if len({str(row["match"]) for row in old_records}) != 13:
        raise ValueError("Pooled fit must contain all 13 original TRAIN matches")
    if len({str(row["match"]) for row in gar_records}) != 45:
        raise ValueError("Pooled fit must contain all 45 GAR TRAIN games")
    sampler = recipe.short.OriginalDiversitySampler(
        old_records, gar_records, list(teacher["feature_names"]), seed)
    start = time.monotonic()
    with _trainer_scope(LR[stage]):
        with recipe._round_robin_v2_windows(old_records) as view_draws:
            with recipe.short.parent.base.installed(sampler):
                result = iterative.fit_from_checkpoints(
                    teacher, prior, old_records, gar_records,
                    seed=seed, arm="ce_control", verbose=verbose)
    model = result["model"]
    for field in ("median", "scale", "feature_names"):
        if field == "feature_names":
            if model[field] != adopted[field]:
                raise ValueError("Reconstructed feature schema differs from initialization")
        else:
            np.testing.assert_array_equal(model[field], adopted[field])
    if set(model["d_state"]) != set(adopted["d_state"]):
        raise ValueError("Reconstructed D state keys differ from initialization")
    for name, value in adopted["d_state"].items():
        np.testing.assert_array_equal(model["d_state"][name], value)
    model["step"] = stage
    model["training"].update(study=STUDY, stage=stage, stage_updates=STAGE_UPDATES,
                              cumulative_updates=stage,
                              warm_start_path=str(prior_path),
                              warm_start_file_sha256=sha(prior_path),
                              reconstruction=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    receipt = dict(study_id=STUDY, stage=stage, stage_updates=STAGE_UPDATES,
        cumulative_updates=stage, seed=seed, path=str(path.relative_to(ROOT)),
        sha256=sha(path), declaration_sha256=sha(DECL),
        original_expected_sha256=original_checkpoint_sha(declaration, seed, stage),
        wall_seconds=time.monotonic()-start, config=result["config"],
        warmstart_path=str(prior_path), warmstart_sha256=sha(prior_path),
        teacher_path=str(teacher_path), teacher_sha256=sha(teacher_path),
        adopted_d_state_unchanged=True, sampling=sampler.receipt(),
        view_draws=dict(view_draws), exposures=result["exposures"],
        loss_intervals=result["loss_intervals"],
        old_matches=sorted({str(row["match"]) for row in old_records}),
        gar_games=sorted({str(row["match"]) for row in gar_records}),
        organizer_reference_access=False, gar_valid_test_access=False)
    expected = _historical_fit(seed, stage)
    reconstructed_losses = receipt["loss_intervals"]
    original_losses = expected["loss_intervals"]
    if len(reconstructed_losses) != len(original_losses):
        max_loss_delta = None
    else:
        max_loss_delta = max(abs(float(a["mean_loss"]) - float(b["mean_loss"]))
                             for a, b in zip(reconstructed_losses, original_losses))
    receipt["verification"] = dict(
        sampling_exact=(receipt["sampling"] == expected["sampling"]),
        view_draws_exact=(receipt["view_draws"] == expected["view_draws"]),
        exposures_exact=(receipt["exposures"] == expected["exposures"]),
        loss_interval_count_exact=(len(reconstructed_losses) == len(original_losses)),
        max_loss_interval_abs_delta=max_loss_delta,
        loss_trace_within_tolerance=(max_loss_delta is not None and
                                     max_loss_delta <= LOSS_TOLERANCE),
        normalization_and_d_state_exact=(receipt["adopted_d_state_unchanged"] is True),
        original_checkpoint_file_sha_match=(receipt["sha256"] ==
                                             receipt["original_expected_sha256"]))
    write_once(receipt_path, receipt)
    return receipt


def _historical_fit(seed: int, stage: int) -> dict:
    seal = json.loads(HIST_SEAL.read_text())
    rows = [row for row in seal["fits"]
            if int(row["seed"]) == seed and int(row["stage"]) == stage]
    if len(rows) != 1:
        raise ValueError(f"Expected one historical pooled fit for {seed}/{stage}")
    return rows[0]


def seal() -> dict:
    declaration = check()
    if SEAL.exists():
        raise FileExistsError(SEAL)
    fits = []
    for seed in SEEDS:
        for stage in STAGES:
            path = output_path(seed, stage)
            receipt = json.loads(Path(str(path) + ".json").read_text())
            if (receipt.get("declaration_sha256") != sha(DECL) or
                    receipt.get("sha256") != sha(path) or
                    receipt.get("original_expected_sha256") !=
                    original_checkpoint_sha(declaration, seed, stage)):
                raise ValueError(f"Reconstructed fit receipt mismatch: {seed}/{stage}")
            verification = receipt.get("verification", {})
            if not all(verification.get(key) is True for key in (
                    "sampling_exact", "view_draws_exact", "exposures_exact",
                    "loss_interval_count_exact", "loss_trace_within_tolerance",
                    "normalization_and_d_state_exact")):
                raise ValueError(f"Reconstruction parity guard failed: {seed}/{stage}")
            fits.append(receipt)
    models = {row["path"]: row["sha256"] for row in fits}
    result = dict(study_id=STUDY, declaration_sha256=sha(DECL), fits=fits,
        models_sha256=models,
        all_four_fits_complete=True, original_byte_identity=False,
        reconstruction_status="reconstructed comparator; numerical and artifact parity must be reported separately",
        organizer_reference_access=False, promotion=False)
    write_once(SEAL, result)
    return result


def verify() -> dict:
    """Persist fit diagnostics, model reload checks, and the seal binding."""
    declaration = check()
    if not SEAL.is_file():
        seal()
    sealed = json.loads(SEAL.read_text())
    if sealed.get("declaration_sha256") != sha(DECL):
        raise ValueError("Pooled reconstruction seal is bound to a different declaration")
    if len(sealed.get("fits", [])) != 4:
        raise ValueError("Pooled reconstruction seal must contain four fits")
    if set(sealed.get("models_sha256", {})) != {row["path"] for row in sealed["fits"]}:
        raise ValueError("Seal model hash map differs from its fit rows")
    fits = []
    for row in sealed["fits"]:
        path = ROOT / row["path"]
        if sha(path) != row["sha256"]:
            raise ValueError(f"Reconstructed checkpoint hash changed: {path}")
        if sealed["models_sha256"].get(row["path"]) != row["sha256"]:
            raise ValueError(f"Seal artifact hash map mismatch: {row['path']}")
        sidecar = json.loads(Path(str(path) + ".json").read_text())
        if sidecar != row:
            raise ValueError(f"Fit receipt differs from the sealed row: {row['path']}")
        model = joblib.load(path)
        _, adopted, _, _ = _warmstart(int(row["seed"]))
        normalizers_exact = (model["feature_names"] == adopted["feature_names"] and
            np.array_equal(model["median"], adopted["median"]) and
            np.array_equal(model["scale"], adopted["scale"]))
        d_state_exact = (set(model["d_state"]) == set(adopted["d_state"]) and all(
            np.array_equal(model["d_state"][key], adopted["d_state"][key])
            for key in adopted["d_state"]))
        receipt = json.loads(Path(str(path) + ".json").read_text())
        fits.append(dict(seed=row["seed"], stage=row["stage"], path=row["path"],
            sha256=row["sha256"],
            original_expected_sha256=original_checkpoint_sha(
                declaration, int(row["seed"]), int(row["stage"])),
            original_file_sha_match=(row["sha256"] == receipt["original_expected_sha256"]),
            normalizers_exact=normalizers_exact, d_state_exact=d_state_exact,
            parity=receipt.get("verification", {})))
    all_passed = len(fits) == 4 and all(
        row["normalizers_exact"] and row["d_state_exact"] and
        all(row["parity"].get(key) is True for key in (
            "sampling_exact", "view_draws_exact", "exposures_exact",
            "loss_interval_count_exact", "loss_trace_within_tolerance"))
        for row in fits)
    expected_prediction_sha = json.loads(PLAN.read_text())[
        "reconstruction_verification"]["full_prediction_parity"].split(
            "SHA256 must equal saved")[-1].split(" before")[0]
    result = dict(schema="frontier_pooled_reconstruction_verification_v1",
        study_id=STUDY, seal_sha256=sha(SEAL), declaration_sha256=sha(DECL),
        all_passed=all_passed, fit_count=len(fits), models_sha256=sealed["models_sha256"],
        fits=fits, prediction_csv_parity="pending reference-free deployment replay",
        prediction_parity_required_sha256=expected_prediction_sha,
        original_checkpoint_byte_identity=all(row["original_file_sha_match"] for row in fits),
        reconstructed_comparator=True, promotion=False)
    write_once(VERIFICATION, result)
    if not all_passed:
        raise ValueError(f"Reconstruction diagnostics failed; details preserved in {VERIFICATION}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "freeze", "fit-one", "verify", "seal"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--stage", type=int, choices=STAGES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        freeze()
    elif args.action == "fit-one":
        if args.seed is None or args.stage is None:
            parser.error("fit-one requires --seed and --stage")
        fit_one(args.seed, args.stage, verbose=not args.quiet)
    elif args.action == "verify":
        verify()
    else:
        seal()


if __name__ == "__main__":
    main()
