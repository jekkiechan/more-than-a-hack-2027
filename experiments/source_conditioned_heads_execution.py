"""Finite C/H/HD source-conditioned dense-supervision comparison.

All model/data artifacts stay in ignored MTAH_DATA_DIR. This runner writes only
aggregate evidence into experiments/results; it never reads organizer labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import (factorized_had as fh, learned_temporal as lt,
                         sn_dense_run as dense_run, sn_dense_train as sn,
                         source_conditioned_heads as sch,
                         source_conditioned_train as sct,
                         soccernet_capacity as sc, soccernet_extension as old,
                         soccernet_capacity_model as cap, supervised)

DATA = sc.DATA
OUT = DATA / "experiments/source_conditioned_heads_v1"
MODELS, FITS, SCORES, PREDICTIONS = (OUT / p for p in ("models", "fits", "scores", "predictions"))
DECLARATION = ROOT / "experiments/results/source_conditioned_heads_declaration.json"
AGGREGATES = ROOT / "experiments/results/source_conditioned_heads_scores.json"
REPORT = ROOT / "experiments/SOURCE_CONDITIONED_HEADS_REPORT.md"
MODEL_SEAL = OUT / "models_frozen.json"
C_MODEL_SEAL = OUT / "control_models_frozen.json"
SCORE_SEAL = OUT / "scores_frozen.json"
FOLDS = tuple(sc.FOLDS)
SEEDS = (2026, 2027)
ARMS = ("C", "H", "HD")
TOLERANCE = 2e-6


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"Refusing to overwrite frozen file: {path}")
        return
    with path.open("x") as handle:
        handle.write(encoded)


def read(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def _verify_hash(path: Path, expected: str, label: str) -> str:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} missing: {path}")
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {path}")
    return actual


def teacher_path(fold: str, seed: int) -> Path:
    return sc.initial_path(fold, seed)


def tag(arm: str, fold: str, seed: int) -> str:
    return f"{arm}__{fold.removeprefix('leave_out_')}__{seed}"


def model_path(arm: str, fold: str, seed: int) -> Path:
    return MODELS / f"{tag(arm, fold, seed)}.joblib"


def receipt_path(arm: str, fold: str, seed: int) -> Path:
    return FITS / f"{tag(arm, fold, seed)}.json"


def prediction_path(arm: str, fold: str, seed: int) -> Path:
    return PREDICTIONS / f"{tag(arm, fold, seed)}.npz"


def _expected_teacher_hashes() -> dict[str, str]:
    frozen = read(ROOT / "experiments/results/sn_dense_v1_declaration.json")
    return {str(teacher_path(fold, seed).relative_to(DATA)): frozen["data_sha256"][
            str(teacher_path(fold, seed).relative_to(DATA))]
            for fold in FOLDS for seed in SEEDS}


def verify_inputs() -> dict:
    """Check the exact fold teachers, target caches, GAR records and eval cells."""
    expected = _expected_teacher_hashes()
    actual = {}
    for relative, digest in expected.items():
        path = DATA / relative
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Fold teacher restore/hash mismatch: {relative}")
        actual[relative] = digest
    manifest = old.old_spec()
    training_paths = {}
    for entry in manifest["prepared_training_caches"].values():
        for field in ("cache", "receipt"):
            path = DATA / Path(entry[field]).relative_to("data")
            if not path.is_file():
                raise FileNotFoundError(path)
            expected_sha = entry["sha256" if field == "cache" else "receipt_sha256"]
            if sha(path) != expected_sha:
                raise ValueError(f"Prepared target cache mismatch: {path}")
            training_paths[str(path.relative_to(DATA))] = expected_sha
    selection = {}
    for item in old.selection_items():
        for view, rep in old.VIEWS:
            path = old.eval_path(item, view, rep)
            if not path.is_file():
                raise FileNotFoundError(path)
            selection[str(path.relative_to(DATA))] = sha(path)
    gar_manifest = dense_run.dl.DEFAULT_OUT / "train" / "manifest.json"
    if not gar_manifest.is_file():
        raise FileNotFoundError(f"Dense SoccerNet TRAIN records missing: {gar_manifest}")
    gar = read(gar_manifest)
    if len(gar["games"]) != 45:
        raise ValueError("Expected exactly 45 pinned SoccerNet TRAIN fixtures")
    game_ids = sorted(map(str, gar["game_records"]))
    tracking = dense_run.dl.tracking_root(dense_run.dl.DEFAULT_RAW)
    expected_games = {str(sample["game_id"])
                      for sample in dense_run.dl.read_split_samples(tracking, "train")}
    if set(game_ids) != expected_games:
        raise ValueError("Dense GAR TRAIN fixture IDs differ from the frozen source declaration")
    game_files = {}
    for name, digest in sorted(gar["game_files"].items()):
        path = gar_manifest.parent / "games" / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Dense GAR prepared game hash mismatch: {name}")
        game_files[name] = digest
    raw_receipt = DATA / "independent/soccernet_gar/acquisition_manifest.json"
    raw = read(raw_receipt)
    pinned = read(ROOT / "experiments/results/soccernet_capacity_declaration.json")[
        "raw_source_sha256"]
    raw_files = {entry["path"]: entry for entry in raw["files"]}
    if (raw.get("dataset") != pinned["dataset"] or raw.get("branch") != pinned["branch"]
            or raw.get("revision") != pinned["revision"]
            or set(raw_files) != set(pinned["file_sha256"])):
        raise ValueError("SoccerNet source restore receipt identity differs from the pinned dataset")
    for path, expected_sha in pinned["file_sha256"].items():
        entry = raw_files[path]
        if entry["sha256"] != expected_sha:
            raise ValueError(f"Raw SoccerNet receipt hash differs: {path}")
        _verify_hash(DATA / "independent/soccernet_gar" / path, expected_sha,
                     "Pinned SoccerNet TRAIN file")
    dense_decl = read(ROOT / "experiments/results/sn_dense_v1_declaration.json")
    expected_dense_manifest = dense_decl["data_sha256"]["dense_records_train_manifest"]
    actual_dense_manifest = sha(gar_manifest)
    if actual_dense_manifest != expected_dense_manifest:
        raise ValueError("Rebuilt GAR dense TRAIN manifest differs from the pinned D2 input; "
                         "diagnose the prepared records before any fit")
    target_root = DATA / "experiments/objective_alignment_v1"
    target_spec = target_root / "predeclared.json"
    target_manifest = target_root / "prepared_training_manifest.json"
    target_parity = target_root / "training_parity.json"
    checkpoint_manifest = target_root / "checkpoints_frozen.json"
    for path in (target_spec, target_manifest, target_parity, checkpoint_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    parity = read(target_parity)
    target_manifest_sha = sha(target_manifest)
    if parity["prepared_training_manifest_sha256"] != target_manifest_sha:
        raise ValueError("Prepared target training manifest differs from its parity receipt")
    checkpoint_frozen = read(checkpoint_manifest)
    cp_entries = {entry["file"]: entry for entry in checkpoint_frozen["new_checkpoints"]}
    teacher_receipts = {}
    for fold in FOLDS:
        plan = sc.prep.split_plan(fold=fold)
        for seed in SEEDS:
            checkpoint = teacher_path(fold, seed)
            receipt = checkpoint.with_suffix(".json")
            entry = cp_entries.get(checkpoint.name)
            if entry is None:
                raise ValueError(f"Teacher absent from frozen checkpoint manifest: {checkpoint.name}")
            model_sha = _verify_hash(checkpoint, entry["sha256"], "Fold teacher")
            _verify_hash(receipt, entry["receipt_sha256"], "Fold teacher receipt")
            provenance = read(receipt)
            if (provenance.get("sha256") != model_sha
                    or provenance.get("heldout") != fold.removeprefix("leave_out_")
                    or provenance.get("seed") != seed
                    or set(provenance.get("training_matches", [])) != {i["match"] for i in plan["train"]}
                    or set(provenance.get("sources", [])) != {i["source"] for i in plan["train"]}):
                raise ValueError(f"Fold teacher provenance identity differs: {fold}/{seed}")
            teacher_receipts[str(receipt.relative_to(DATA))] = entry["receipt_sha256"]
    return dict(fold_teachers=actual, prepared_target_caches=dict(sorted(training_paths.items())),
        teacher_provenance_receipts=teacher_receipts,
        selection_feature_caches=dict(sorted(selection.items())),
        gar_dense_train_manifest_sha256=sha(gar_manifest), gar_games=game_ids,
        gar_game_files=game_files, gar_source_receipt_sha256=sha(raw_receipt),
        raw_pinned_file_sha256=pinned["file_sha256"],
        target_predeclared_sha256=sha(target_spec),
        target_prepared_training_manifest_sha256=target_manifest_sha,
        target_training_parity_sha256=sha(target_parity),
        fold_checkpoint_manifest_sha256=sha(checkpoint_manifest))


def _source_hashes() -> dict[str, str]:
    names = ("experiments/source_conditioned_heads.py",
        "experiments/source_conditioned_train.py", "experiments/source_conditioned_heads_execution.py",
        "experiments/sn_dense_train.py", "experiments/sn_dense_run.py",
        "experiments/soccernet_capacity.py", "experiments/soccernet_capacity_model.py",
        "experiments/factorized_had.py", "experiments/learned_temporal.py",
        "experiments/soccernet_extension.py", "experiments/objective_losses.py",
        "experiments/soccernet_dense_labels.py", "experiments/supervised.py",
        "experiments/tests/test_source_conditioned_heads.py",
        "experiments/SOURCE_CONDITIONED_HEADS.md",
        "experiments/objective_ensemble.py", "experiments/metrics.py",
        "experiments/spell_diagnostics.py", "evaluation/possession_metrics.py",
        "experiments/preparation.py", "experiments/eval_set/manifest.json",
        "experiments/provider_touch_calibration_v2.py",
        "experiments/results/provider_touch_sportec_train_nested_v2_declaration.json",
        "experiments/results/provider_touch_sportec_train_nested_v2_summary.json",
        "experiments/results/sn_dense_v1_declaration.json",
        "experiments/results/baseline_sn_dense_promotion.json",
        "experiments/results/sn_dense_v1_progress.md",
        "experiments/SUPERVISION_ALIGNMENT_PLAN.md",
        "experiments/baseline_manifest.json", "experiments/model_decision_policy.json")
    return {name: sha(ROOT / name) for name in names}


def freeze() -> dict:
    """Write the execution declaration only after every private input verifies."""
    if DECLARATION.exists():
        raise FileExistsError(f"Declaration already frozen: {DECLARATION}")
    inputs = verify_inputs()
    baseline_decl = read(ROOT / "experiments/results/sn_dense_v1_declaration.json")
    touch_decl = read(ROOT / "experiments/results/provider_touch_sportec_train_nested_v2_declaration.json")
    touch_result = read(ROOT / "experiments/results/provider_touch_sportec_train_nested_v2_summary.json")
    if (touch_decl.get("study_id") != "provider_touch_sportec_train_nested_v2"
            or len(touch_result.get("outer_folds", [])) != 3
            or any(row.get("selected_config", {}).get("id") != "contest_hold_tail_0s"
                   for row in touch_result["outer_folds"])):
        raise ValueError("Training-only touch-calibration decision is missing or changed")
    historical = read(ROOT / "experiments/results/baseline_sn_dense_promotion.json")
    if (historical["evidence"]["declaration_sha256"] != sha(
                ROOT / "experiments/results/sn_dense_v1_declaration.json")
            or historical["evidence"]["progress_sha256"] != sha(
                ROOT / "experiments/results/sn_dense_v1_progress.md")):
        raise ValueError("Historical D2 aggregate does not bind the declared progress/declaration")
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    value = dict(study_id="source_conditioned_heads_v1", date="2026-10-31",
        question="Does source-specific target/GAR output calibration improve H/A/D full-provider transfer under the frozen D2 training budget?",
        status="frozen before fitting/scoring", planning_base="f9a741a",
        source_git_commit=source_commit,
        plan_sha256=sha(ROOT / "experiments/SUPERVISION_ALIGNMENT_PLAN.md"),
        baseline=dict(package="joint_gru_sn_dense_ensemble_v1",
            manifest_sha256=sha(ROOT / "experiments/baseline_manifest.json"),
            D2_declaration_sha256=sha(ROOT / "experiments/results/sn_dense_v1_declaration.json")),
        matrix=dict(arms=list(ARMS), folds=list(FOLDS), seeds=list(SEEDS),
            fits=len(ARMS) * len(FOLDS) * len(SEEDS), pooled_fits=0,
            C="shared target/GAR HA and D projections via frozen sn_dense_train.train",
            H="separate target/GAR HA projections; shared player-only D projection",
            HD="separate target/GAR HA and player-only D projections"),
        matched_recipe=dict(steps=2048, batch_size=32, fraction=0.5, ha_width=96,
            d_width=24, ha_learning_rate=0.0003, d_learning_rate=0.0001,
            input_dropout=0.1, weight_decay=0.0001, gradient_clip=1.0,
            seed_stream="2026/2027; separate Python source-window RNG; duplicated source heads do not consume Torch RNG draws",
            sampling="exact D2 draw order, source/match/game/window groups and 50/50 old/GAR batches",
            loss="D2 H/A CE + complete-spell factors + 0.05 group-balanced live-transition losses + separate all-known-label D BCE; source-weighted means",
            normalization="frozen fold-specific teacher median/scale; held-out provider excluded from every learned target component",
            inference="5 seconds context, 10 second cores, period/time breaks, direct+swapped team views; conditional H/A and D averaged separately before combination",
            decoder="cost 3; official evaluation code"),
        initialization=dict(source="same fold/seed combined-objective teacher as D2",
            ha="cap._copy_teacher_function on the already random-initialized width-96 HA encoder; cloned target projection to GAR",
            d="separate factorized_had player-only masked GRU; teacher third output row copied to scalar target D head; cloned to GAR for HD",
            extra_projection_parameters=dict(H=386, HD=435),
            capacity_confounds="extra source projections are explicit; no parameter-matched controls in this finite matrix"),
        control_replay=dict(protocol="initial C parameters on one actual saved clean selection record per fold/seed before scoring",
            max_abs_probability_tolerance=TOLERANCE, persisted_replay_receipts=True),
        trained_control_reconstruction=dict(
            expected_metrics_source="experiments/results/baseline_sn_dense_promotion.json",
            expected_metrics_source_sha256=sha(ROOT / "experiments/results/baseline_sn_dense_promotion.json"),
            historical_progress="experiments/results/sn_dense_v1_progress.md",
            historical_progress_sha256=sha(ROOT / "experiments/results/sn_dense_v1_progress.md"),
            expected_values_rounding="published four-decimal D2 full-provider development ensemble spell F1",
            absolute_tolerance=0.000051),
        evaluation=dict(matches="all five selection matches, each scored by its fully held-out provider fold",
            views=[dict(view=v, replicate=r) for v, r in old.VIEWS],
            outputs_per_arm=15, seed_ensemble="average native probabilities across seeds within each fold/cell before cost-3 decoding",
            primary="unrounded equal-provider spell F1; macro F1 exact-tie break",
            guards="feed_like spell >= C - .01; each provider clean macro and D-F1 >= C - .02; +.01 clean screen shown separately",
            organizer="no organizer labels, public scoring, promotion or selection"),
        prior_exposure="all five selection matches have prior research exposure; reused development evidence only",
        label_rule_arm=dict(L_eligible=False,
            decision="Omit L from this full-pipeline matrix before candidate scoring",
            reason="Sportec-only schema and nested TRAIN CV do not establish transfer to GAR/Metrica/SkillCorner; the leave_out_sportec fold has no eligible Sportec calibration matches.",
            calibration_declaration_sha256=sha(ROOT / "experiments/results/provider_touch_sportec_train_nested_v2_declaration.json"),
            calibration_result_sha256=sha(ROOT / "experiments/results/provider_touch_sportec_train_nested_v2_summary.json")),
        data=inputs, source_sha256=_source_hashes(),
        execution_stages=["fit six C controls", "seal C models", "score C and require historical aggregate replay",
            "fit fixed H and HD controls only after C passes", "seal all 18 models before H/HD scoring",
            "score H/HD", "seal all 18 score receipts before aggregation"],
        implementation="C calls frozen sn_dense_train.train; H/HD call source_conditioned_train.train; no frozen runner or baseline edits")
    write_once(DECLARATION, value)
    return value


def check_frozen() -> dict:
    declaration = read(DECLARATION)
    if declaration["source_sha256"] != _source_hashes():
        raise ValueError("Experiment source changed after declaration freeze")
    if declaration["plan_sha256"] != sha(ROOT / "experiments/SUPERVISION_ALIGNMENT_PLAN.md"):
        raise ValueError("Experiment plan changed after declaration freeze")
    data = declaration["data"]
    for relative, digest in data["fold_teachers"].items():
        _verify_hash(DATA / relative, digest, "Fold teacher")
    for relative, digest in data["teacher_provenance_receipts"].items():
        _verify_hash(DATA / relative, digest, "Fold teacher provenance")
    for group in ("prepared_target_caches", "selection_feature_caches"):
        for relative, digest in data[group].items():
            _verify_hash(DATA / relative, digest, group)
    manifest = dense_run.dl.DEFAULT_OUT / "train" / "manifest.json"
    _verify_hash(manifest, data["gar_dense_train_manifest_sha256"], "Dense GAR TRAIN manifest")
    for name, digest in data["gar_game_files"].items():
        _verify_hash(manifest.parent / "games" / name, digest, "Dense GAR game record")
    target_root = DATA / "experiments/objective_alignment_v1"
    for relative, key in (("predeclared.json", "target_predeclared_sha256"),
                          ("prepared_training_manifest.json", "target_prepared_training_manifest_sha256"),
                          ("training_parity.json", "target_training_parity_sha256"),
                          ("checkpoints_frozen.json", "fold_checkpoint_manifest_sha256")):
        _verify_hash(target_root / relative, data[key], f"Target source manifest: {relative}")
    raw_receipt = DATA / "independent/soccernet_gar/acquisition_manifest.json"
    _verify_hash(raw_receipt, data["gar_source_receipt_sha256"], "Pinned SoccerNet acquisition receipt")
    # Raw TRAIN bytes were all SHA-verified against the immutable source pin at freeze.
    # Fitting consumes only the separately sealed dense game records above.
    return declaration


def _initial_capacity(teacher: dict, seed: int) -> dict:
    import torch
    torch.manual_seed(seed)
    model = cap.network(int(teacher["input_features"]), 96, 0.1)
    cap._copy_teacher_function(teacher, model)
    return dict(kind="soccernet_capacity", hidden=96, input_features=int(teacher["input_features"]),
        state={k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()},
        median=np.array(teacher["median"], copy=True), scale=np.array(teacher["scale"], copy=True),
        feature_names=list(teacher["feature_names"]), training={"input_dropout": 0.1})


def replay() -> dict:
    """Assert C initialization/inference parity on actual cached selection features."""
    declaration = check_frozen()
    if any(receipt_path(a, f, s).exists() for a in ARMS for f in FOLDS for s in SEEDS):
        raise ValueError("Control replay must precede all fit receipts")
    import torch
    torch.set_num_threads(1)
    records = []
    for fold in FOLDS:
        for seed in SEEDS:
            teacher_file = teacher_path(fold, seed)
            _verify_hash(teacher_file, declaration["data"]["fold_teachers"][
                str(teacher_file.relative_to(DATA))], "Fold teacher")
            teacher = joblib.load(teacher_file)
            item = next(i for i in old.selection_items() if "leave_out_" + i["source"] == fold)
            cache_file = old.eval_path(item, "clean", "clean")
            _verify_hash(cache_file, declaration["data"]["selection_feature_caches"][
                str(cache_file.relative_to(DATA))], "Selection cache")
            cached = joblib.load(cache_file)
            record = old.inference_record(cached["record"])
            baseline_model = _initial_capacity(teacher, seed)
            torch.manual_seed(seed)
            candidate_net = sch.network(int(teacher["input_features"]), 96,
                list(teacher["player_indices"]), d_hidden=24, input_dropout=0.1,
                separate_ha=False, separate_d=False)
            sch.initialize_from_teacher(candidate_net, teacher)
            candidate_model = dict(kind="source_conditioned_dense", arm="C",
                state=sch.export_state(candidate_net), hidden=96, d_hidden=24, input_dropout=0.1,
                separate_ha=False, separate_d=False, median=teacher["median"], scale=teacher["scale"],
                feature_names=teacher["feature_names"], input_features=teacher["input_features"],
                player_indices=teacher["player_indices"])
            expected = sn.predict(record, baseline_model, teacher)
            actual = sct.predict(record, candidate_model)
            worst = float(np.max(np.abs(expected - actual)))
            if worst > TOLERANCE:
                raise ValueError(f"C target inference parity failed for {fold}/{seed}: {worst:.9g} > {TOLERANCE}")
            records.append(dict(fold=fold, seed=seed, match=item["match"],
                cache_sha256=sha(cache_file), max_abs_probability_difference=worst,
                tolerance=TOLERANCE))
    receipt = dict(study_id=declaration["study_id"], declaration_sha256=sha(DECLARATION),
        replay_rows=records, passed=True, organizer_labels_read=False, scored_metrics=False)
    write_once(OUT / "control_replay.json", receipt)
    return receipt


def load_train(fold: str):
    teacher = joblib.load(teacher_path(fold, 2026))
    records = dense_run.load_train(anchor_mode=False)
    old_records = old.load_old(fold)
    return teacher, old_records, records


def fit_one(arm: str, fold: str, seed: int) -> dict:
    declaration = check_frozen()
    if MODEL_SEAL.exists():
        raise ValueError("No fits are allowed after the full model seal")
    if arm == "C" and C_MODEL_SEAL.exists():
        raise ValueError("No C control fits are allowed after the C model seal")
    if arm != "C":
        trained_control = OUT / "training_control_replay.json"
        if not trained_control.is_file():
            raise ValueError("H/HD fits require a passing C reconstructed-control gate")
        gate = read(trained_control)
        expected_scores = {f"{fold}__{seed}": sha(SCORES / f"{tag('C', fold, seed)}.json")
                           for fold in FOLDS for seed in SEEDS}
        _check_control_model_seal()
        if (not gate.get("passed") or gate.get("study_id") != declaration["study_id"]
                or gate.get("declaration_sha256") != sha(DECLARATION)
                or gate.get("model_seal_sha256") != sha(C_MODEL_SEAL)
                or gate.get("control_score_receipts") != expected_scores):
            raise ValueError("C reconstructed-control receipt failed before H/HD fit")
    replay_receipt = read(OUT / "control_replay.json")
    if not replay_receipt.get("passed") or sha(OUT / "control_replay.json") == "":
        raise ValueError("A passing actual-data control replay is required before fitting")
    mp, rp = model_path(arm, fold, seed), receipt_path(arm, fold, seed)
    if rp.exists() or mp.exists():
        raise FileExistsError(f"Fit artifacts already exist: {tag(arm, fold, seed)}")
    teacher_file = teacher_path(fold, seed)
    _verify_hash(teacher_file, declaration["data"]["fold_teachers"][
        str(teacher_file.relative_to(DATA))], "Fold teacher")
    teacher = joblib.load(teacher_file)
    old_records = old.load_old(fold)
    gar_records = dense_run.load_train(anchor_mode=False)
    def label_counts(records):
        y = np.concatenate([np.asarray(record["y"], dtype=np.int64) for record in records])
        return dict(ha_frames=int(np.count_nonzero((y == 0) | (y == 1))),
                    d_loss_valid_frames=int(np.count_nonzero(y >= 0)),
                    dead_class_frames=int(np.count_nonzero(y == 2)),
                    unknown_frames=int(np.count_nonzero(y < 0)), total_frames=int(y.size))
    source_label_counts = dict(target=label_counts(old_records), soccernet=label_counts(gar_records))
    start = time.monotonic()
    if arm == "C":
        result = sn.train(teacher, old_records, gar_records, width=96, lr=3e-4,
            seed=seed, steps=2048, fraction=0.5, batch_size=32, ha_target="dense",
            train_d=True, d_lr=1e-4, input_dropout=0.1, require_full_native=True,
            log_every=256, verbose=True)
    else:
        result = sct.train(teacher, old_records, gar_records, arm=arm, seed=seed,
            steps=2048, fraction=0.5, batch_size=32, width=96, d_width=24,
            lr=3e-4, d_lr=1e-4, input_dropout=0.1, require_full_native=True,
            log_every=256, verbose=True)
    model = result["model"]
    MODELS.mkdir(parents=True, exist_ok=True); FITS.mkdir(parents=True, exist_ok=True)
    temp = mp.with_suffix(".joblib.tmp")
    joblib.dump(model, temp, compress=0)
    os.replace(temp, mp)
    receipt = dict(study_id="source_conditioned_heads_v1", arm=arm, fold=fold, seed=seed,
        declaration_sha256=sha(DECLARATION), control_replay_sha256=sha(OUT / "control_replay.json"),
        model_path=str(mp.relative_to(DATA)), model_sha256=sha(mp),
        steps_run=int(model.get("step", 2048)), config=model.get("training", {}),
        wall_seconds=time.monotonic() - start,
        fit_logs=model.get("metadata", {}), fitting_performed=True,
        available_label_counts=source_label_counts,
        effective_sampled_label_counts=model.get("metadata", {}).get("effective_label_counts"),
        organizer_labels_read=False, scored=False)
    write_once(rp, receipt)
    return receipt


def _predict(arm: str, model: dict, teacher: dict, record: dict) -> np.ndarray:
    return sn.predict(record, model, teacher) if arm == "C" else sct.predict(record, model)


def _expected_keys(fold: str) -> set[str]:
    return {f"{item['match']}__{view}__{rep}"
            for item in old.selection_items() if "leave_out_" + item["source"] == fold
            for view, rep in old.VIEWS}


def seal_control_models() -> dict:
    """Freeze the six fixed C controls before the early reconstruction gate."""
    declaration = check_frozen()
    if C_MODEL_SEAL.exists():
        raise FileExistsError(C_MODEL_SEAL)
    entries = {}
    for fold in FOLDS:
        for seed in SEEDS:
            arm = "C"
            fit_file, model_file = receipt_path(arm, fold, seed), model_path(arm, fold, seed)
            if not fit_file.is_file():
                raise FileNotFoundError(f"Incomplete C control fits: {fit_file}")
            fit = read(fit_file)
            if (fit.get("study_id") != declaration["study_id"] or fit.get("arm") != arm
                    or fit.get("fold") != fold or fit.get("seed") != seed
                    or fit.get("declaration_sha256") != sha(DECLARATION)
                    or fit.get("model_path") != str(model_file.relative_to(DATA))
                    or fit.get("steps_run") != 2048 or not fit.get("fitting_performed")
                    or fit.get("scored") is not False):
                raise ValueError(f"C fit identity mismatch: {fit_file}")
            entries[tag(arm, fold, seed)] = dict(model_path=str(model_file.relative_to(DATA)),
                model_sha256=_verify_hash(model_file, fit["model_sha256"], "C model"),
                fit_receipt_path=str(fit_file.relative_to(DATA)),
                fit_receipt_sha256=sha(fit_file))
    if len(entries) != 6:
        raise ValueError(f"Expected 6 fitted C controls, got {len(entries)}")
    seal = dict(study_id=declaration["study_id"], declaration_sha256=sha(DECLARATION),
        models=entries, model_count=6, staged_control_seal=True, sealed_before_control_scoring=True)
    write_once(C_MODEL_SEAL, seal)
    return seal


def _check_control_model_seal() -> dict:
    seal = read(C_MODEL_SEAL)
    declaration = check_frozen()
    if (seal.get("study_id") != declaration["study_id"]
            or seal.get("declaration_sha256") != sha(DECLARATION)
            or len(seal.get("models", {})) != 6):
        raise ValueError("Six-model C control seal is invalid")
    for fold in FOLDS:
        for seed in SEEDS:
            key = tag("C", fold, seed)
            entry = seal["models"].get(key)
            if not entry:
                raise ValueError(f"Sealed C model missing: {key}")
            _verify_hash(DATA / entry["model_path"], entry["model_sha256"], "Sealed C model")
            _verify_hash(DATA / entry["fit_receipt_path"], entry["fit_receipt_sha256"],
                         "Sealed C fit receipt")
    return seal


def seal_models() -> dict:
    """Freeze identities and hashes for the complete 18-model matrix."""
    declaration = check_frozen()
    if MODEL_SEAL.exists():
        raise FileExistsError(MODEL_SEAL)
    control_replay = OUT / "training_control_replay.json"
    if not control_replay.is_file() or not read(control_replay).get("passed"):
        raise ValueError("Full model seal requires the passing C reconstructed-control gate")
    entries = {}
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                fit_receipt_file = receipt_path(arm, fold, seed)
                model_file = model_path(arm, fold, seed)
                if not fit_receipt_file.is_file():
                    raise FileNotFoundError(f"Incomplete fit matrix: {fit_receipt_file}")
                fit = read(fit_receipt_file)
                if (fit.get("study_id") != declaration["study_id"]
                        or fit.get("arm") != arm or fit.get("fold") != fold or fit.get("seed") != seed
                        or fit.get("declaration_sha256") != sha(DECLARATION)
                        or fit.get("model_path") != str(model_file.relative_to(DATA))
                        or fit.get("steps_run") != 2048 or not fit.get("fitting_performed")
                        or fit.get("scored") is not False):
                    raise ValueError(f"Fit identity mismatch before model seal: {fit_receipt_file}")
                model_sha = _verify_hash(model_file, fit["model_sha256"], "Model seal input")
                entries[tag(arm, fold, seed)] = dict(model_path=str(model_file.relative_to(DATA)),
                    model_sha256=model_sha, fit_receipt_path=str(fit_receipt_file.relative_to(DATA)),
                    fit_receipt_sha256=sha(fit_receipt_file))
    if len(entries) != 18:
        raise ValueError(f"Expected 18 fitted models, got {len(entries)}")
    control_seal = _check_control_model_seal()
    for key, entry in control_seal["models"].items():
        if entries[key] != entry:
            raise ValueError(f"C full-matrix model identity differs from early control seal: {key}")
    seal = dict(study_id=declaration["study_id"], declaration_sha256=sha(DECLARATION),
        models=entries, model_count=len(entries), c_control_seal_sha256=sha(C_MODEL_SEAL),
        sealed_before_candidate_scoring=True)
    write_once(MODEL_SEAL, seal)
    return seal


def _check_model_seal() -> dict:
    seal = read(MODEL_SEAL)
    declaration = check_frozen()
    if (seal.get("study_id") != declaration["study_id"]
            or seal.get("declaration_sha256") != sha(DECLARATION)
            or seal.get("c_control_seal_sha256") != sha(C_MODEL_SEAL)
            or len(seal.get("models", {})) != 18):
        raise ValueError("Full C/H/HD model seal is invalid")
    control = _check_control_model_seal()
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                key = tag(arm, fold, seed)
                entry = seal["models"].get(key)
                if not entry:
                    raise ValueError(f"Sealed model missing: {key}")
                _verify_hash(DATA / entry["model_path"], entry["model_sha256"], "Sealed model")
                _verify_hash(DATA / entry["fit_receipt_path"], entry["fit_receipt_sha256"],
                             "Sealed fit receipt")
                if arm == "C" and entry != control["models"][key]:
                    raise ValueError(f"C model differs from early control seal: {key}")
    return seal


def score_one(arm: str, fold: str, seed: int) -> dict:
    declaration = check_frozen()
    seal = _check_control_model_seal() if arm == "C" else _check_model_seal()
    if SCORE_SEAL.exists():
        raise ValueError("No scoring is allowed after the full score seal")
    if arm != "C":
        control = OUT / "training_control_replay.json"
        if not control.is_file():
            raise ValueError("C trained-control reconstruction must pass before H/HD scoring")
        control_receipt = read(control)
        expected_control_scores = {f"{fold}__{seed}": sha(SCORES / f"{tag('C', fold, seed)}.json")
                                   for fold in FOLDS for seed in SEEDS}
        if (not control_receipt.get("passed")
                or control_receipt.get("study_id") != declaration["study_id"]
                or control_receipt.get("declaration_sha256") != sha(DECLARATION)
                or control_receipt.get("model_seal_sha256") != sha(C_MODEL_SEAL)
                or control_receipt.get("control_score_receipts") != expected_control_scores):
            raise ValueError("C trained-control replay receipt identity/hash check failed")
    if not receipt_path(arm, fold, seed).is_file():
        raise FileNotFoundError(f"Missing completed fit: {tag(arm, fold, seed)}")
    fit_receipt = read(receipt_path(arm, fold, seed))
    model_file = model_path(arm, fold, seed)
    if (fit_receipt.get("study_id") != declaration["study_id"]
            or fit_receipt.get("arm") != arm or fit_receipt.get("fold") != fold
            or fit_receipt.get("seed") != seed
            or fit_receipt.get("declaration_sha256") != sha(DECLARATION)
            or fit_receipt.get("model_path") != str(model_file.relative_to(DATA))
            or fit_receipt.get("steps_run") != 2048
            or fit_receipt.get("fitting_performed") is not True
            or fit_receipt.get("scored") is not False):
        raise ValueError("Fit receipt identity/declaration mismatch before score")
    sealed = seal["models"][tag(arm, fold, seed)]
    _verify_hash(model_file, sealed["model_sha256"], "Fitted model")
    teacher_file = teacher_path(fold, seed)
    _verify_hash(teacher_file, declaration["data"]["fold_teachers"][
        str(teacher_file.relative_to(DATA))], "Fold teacher")
    model = joblib.load(model_file)
    teacher = joblib.load(teacher_file)
    rows, arrays, keys = [], {}, []
    for item in old.selection_items():
        if "leave_out_" + item["source"] != fold:
            continue
        for view, rep in old.VIEWS:
            cache_file = old.eval_path(item, view, rep)
            _verify_hash(cache_file, declaration["data"]["selection_feature_caches"][
                str(cache_file.relative_to(DATA))], "Selection cache")
            cached = joblib.load(cache_file)
            record = old.inference_record(cached["record"])
            probability = _predict(arm, model, teacher, record)
            native = old.upsample(probability, record, cached["inputs"])
            row, prediction = old.metric_row(native, cached["inputs"], cached["labels"], item,
                                              view, rep, arm, str(seed))
            diag = old.scoring.spell_diagnostics.spell_diagnostics(
                cached["labels"], prediction, period=cached["inputs"]["period"],
                fps=float(cached["inputs"]["fps"]), timestamps=cached["inputs"]["timestamps"])
            row["predictions_overlapping_multiple_same_team_references"] = int(
                diag["overlap_geometry"]["predictions_overlapping_multiple_same_team_references"])
            rows.append(row)
            key = f"{item['match']}__{view}__{rep}"
            arrays[key] = native
            keys.append(key)
    expected = sum("leave_out_" + i["source"] == fold for i in old.selection_items()) * len(old.VIEWS)
    actual_keys = {f"{r['match']}__{r['view']}__{r['replicate']}" for r in rows}
    if len(rows) != expected or set(keys) != _expected_keys(fold) or actual_keys != _expected_keys(fold):
        raise ValueError(f"Expected {expected} view cells for {fold}, found {len(rows)}")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    pp = prediction_path(arm, fold, seed)
    temp = pp.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, keys=np.asarray(keys), **arrays)
    os.replace(temp, pp)
    value = dict(study_id="source_conditioned_heads_v1", arm=arm, fold=fold,
        seed=seed, declaration_sha256=sha(DECLARATION), fit_sha256=sha(receipt_path(arm, fold, seed)),
        model_sha256=sha(model_file), prediction_sha256=sha(pp), keys=keys,
        rows=rows, scoring_performed=True,
        organizer_labels_read=False)
    SCORES.mkdir(parents=True, exist_ok=True)
    write_once(SCORES / f"{tag(arm, fold, seed)}.json", value)
    return value


def verify_training_control() -> dict:
    """Gate H/HD scores on the reconstructed C ensemble matching D2 evidence."""
    declaration = check_frozen()
    _check_control_model_seal()
    rows = []
    for fold in FOLDS:
        per_seed = {}
        for seed in SEEDS:
            path = SCORES / f"{tag('C', fold, seed)}.json"
            if not path.is_file():
                raise FileNotFoundError(f"C control score missing: {path}")
            score = read(path)
            _verify_score_receipt(score, "C", fold, seed, declaration)
            per_seed[seed] = _load_native("C", fold, seed)
        for item in old.selection_items():
            if "leave_out_" + item["source"] != fold:
                continue
            for view, rep in old.VIEWS:
                key = f"{item['match']}__{view}__{rep}"
                avg = (per_seed[2026][key] + per_seed[2027][key]) * 0.5
                cached = joblib.load(old.eval_path(item, view, rep))
                row, prediction = old.metric_row(avg, cached["inputs"], cached["labels"], item,
                                                  view, rep, "C", "ensemble")
                diag = old.scoring.spell_diagnostics.spell_diagnostics(
                    cached["labels"], prediction, period=cached["inputs"]["period"],
                    fps=float(cached["inputs"]["fps"]), timestamps=cached["inputs"]["timestamps"])
                row["predictions_overlapping_multiple_same_team_references"] = int(
                    diag["overlap_geometry"]["predictions_overlapping_multiple_same_team_references"])
                rows.append(row)
    if len(rows) != 15 or len({f"{r['match']}__{r['view']}__{r['replicate']}" for r in rows}) != 15:
        raise ValueError("C trained-control replay requires exactly 15 unique ensemble cells")
    metrics = {view: sc._aggregate_provider(rows, "C", view, "ensemble")
               for view in ("clean", "feed_like")}
    promotion = read(ROOT / "experiments/results/baseline_sn_dense_promotion.json")
    evidence = promotion["evidence"]["development_full_provider_holdout"]
    expected = dict(clean_spell=float(evidence["two_seed_ensemble_clean_spell"]),
                    feed_like_spell=float(evidence["two_seed_ensemble_feed_like_spell"]))
    actual = dict(clean_spell=metrics["clean"]["equal_provider_mean"]["spell_f1"],
                  feed_like_spell=metrics["feed_like"]["equal_provider_mean"]["spell_f1"])
    differences = {key: abs(actual[key] - expected[key]) for key in expected}
    # Historical aggregate is published to four decimals; allow half a rounding unit.
    passed = all(value <= 0.000051 for value in differences.values())
    receipt = dict(study_id=declaration["study_id"], declaration_sha256=sha(DECLARATION),
        model_seal_sha256=sha(C_MODEL_SEAL), control_score_receipts={
            f"{fold}__{seed}": sha(SCORES / f"{tag('C', fold, seed)}.json")
            for fold in FOLDS for seed in SEEDS},
        ensemble_rows=rows, metrics=metrics, historical_d2_rounded=expected,
        actual=actual, absolute_differences=differences,
        tolerance=0.000051, expected_metrics_source_sha256=sha(
            ROOT / "experiments/results/baseline_sn_dense_promotion.json"), passed=passed,
        interpretation="Exact D2 data and training recipe reconstructed; published aggregate replay within four-decimal rounding tolerance",
        organizer_labels_read=False)
    write_once(OUT / "training_control_replay.json", receipt)
    if not passed:
        raise ValueError(f"Reconstructed C control does not replay published D2 aggregate: {differences}")
    return receipt


def _verify_score_receipt(score: dict, arm: str, fold: str, seed: int,
                          declaration: dict | None = None) -> None:
    declaration = declaration or check_frozen()
    fit_file = receipt_path(arm, fold, seed)
    model_file = model_path(arm, fold, seed)
    pred_file = prediction_path(arm, fold, seed)
    if (score.get("study_id") != declaration["study_id"] or score.get("arm") != arm
            or score.get("fold") != fold or score.get("seed") != seed
            or score.get("declaration_sha256") != sha(DECLARATION)
            or score.get("fit_sha256") != sha(fit_file)
            or score.get("model_sha256") != sha(model_file)
            or score.get("prediction_sha256") != sha(pred_file)):
        raise ValueError(f"Score receipt identity/hash mismatch: {arm}/{fold}/{seed}")
    expected = _expected_keys(fold)
    if set(score.get("keys", [])) != expected:
        raise ValueError(f"Score receipt cell keys mismatch: {arm}/{fold}/{seed}")
    rows = score.get("rows", [])
    row_keys = {f"{r['match']}__{r['view']}__{r['replicate']}" for r in rows}
    if len(rows) != len(expected) or row_keys != expected:
        raise ValueError(f"Score receipt row keys mismatch: {arm}/{fold}/{seed}")
    with np.load(pred_file, allow_pickle=False) as archive:
        pred_keys = set(map(str, archive["keys"]))
        array_keys = set(archive.files) - {"keys"}
        native = {str(k): np.asarray(archive[str(k)]) for k in archive["keys"]}
    if pred_keys != expected or array_keys != expected:
        raise ValueError(f"Prediction archive key mismatch: {arm}/{fold}/{seed}")
    recomputed_rows = []
    for item in old.selection_items():
        if "leave_out_" + item["source"] != fold:
            continue
        for view, rep in old.VIEWS:
            key = f"{item['match']}__{view}__{rep}"
            cache_file = old.eval_path(item, view, rep)
            cached = joblib.load(cache_file)
            row, prediction = old.metric_row(native[key], cached["inputs"], cached["labels"],
                                              item, view, rep, arm, str(seed))
            diag = old.scoring.spell_diagnostics.spell_diagnostics(
                cached["labels"], prediction, period=cached["inputs"]["period"],
                fps=float(cached["inputs"]["fps"]), timestamps=cached["inputs"]["timestamps"])
            row["predictions_overlapping_multiple_same_team_references"] = int(
                diag["overlap_geometry"]["predictions_overlapping_multiple_same_team_references"])
            recomputed_rows.append(row)
    if json.dumps(recomputed_rows, sort_keys=True, allow_nan=False) != \
            json.dumps(rows, sort_keys=True, allow_nan=False):
        raise ValueError(f"Per-seed score rows do not recompute from sealed predictions: {arm}/{fold}/{seed}")


def seal_scores() -> dict:
    """Freeze all 18 score/prediction receipts before aggregate recomputation."""
    declaration = check_frozen()
    _check_model_seal()
    control_file = OUT / "training_control_replay.json"
    if not control_file.is_file() or not read(control_file).get("passed"):
        raise ValueError("A passing reconstructed C control is required before score sealing")
    if SCORE_SEAL.exists():
        raise FileExistsError(SCORE_SEAL)
    entries = {}
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                score_file = SCORES / f"{tag(arm, fold, seed)}.json"
                if not score_file.is_file():
                    raise FileNotFoundError(f"Incomplete score matrix: {score_file}")
                score = read(score_file)
                _verify_score_receipt(score, arm, fold, seed, declaration)
                entries[tag(arm, fold, seed)] = dict(score_receipt_path=str(score_file.relative_to(DATA)),
                    score_receipt_sha256=sha(score_file),
                    prediction_path=str(prediction_path(arm, fold, seed).relative_to(DATA)),
                    prediction_sha256=sha(prediction_path(arm, fold, seed)))
    if len(entries) != 18:
        raise ValueError(f"Expected 18 score receipts, found {len(entries)}")
    receipt = dict(study_id=declaration["study_id"], declaration_sha256=sha(DECLARATION),
        model_seal_sha256=sha(MODEL_SEAL), scores=entries, score_count=len(entries),
        sealed_before_aggregate=True)
    write_once(SCORE_SEAL, receipt)
    return receipt


def _check_score_seal() -> dict:
    seal = read(SCORE_SEAL)
    declaration = check_frozen()
    if (seal.get("study_id") != declaration["study_id"]
            or seal.get("declaration_sha256") != sha(DECLARATION)
            or seal.get("model_seal_sha256") != sha(MODEL_SEAL)
            or len(seal.get("scores", {})) != 18):
        raise ValueError("Full score seal is invalid")
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                key = tag(arm, fold, seed)
                entry = seal["scores"].get(key)
                if not entry:
                    raise ValueError(f"Sealed score missing: {key}")
                score_file = DATA / entry["score_receipt_path"]
                prediction_file = DATA / entry["prediction_path"]
                _verify_hash(score_file, entry["score_receipt_sha256"], "Sealed score receipt")
                _verify_hash(prediction_file, entry["prediction_sha256"], "Sealed prediction")
    return seal


def queue(workers: int, stage: str) -> None:
    if workers < 1 or workers > 3:
        raise ValueError("Use 1-3 workers to stay within the declared host resources")
    if stage not in ("fit_control", "score_control", "fit_candidates", "score_candidates"):
        raise ValueError(stage)
    if stage.startswith("fit") and not (OUT / "control_replay.json").is_file():
        raise ValueError("Run `replay` before fitting")
    if stage == "score_control":
        _check_control_model_seal()
    if stage == "score_candidates":
        _check_model_seal()
    if stage == "score_candidates":
        control = OUT / "training_control_replay.json"
        if not control.is_file() or not read(control).get("passed"):
            raise ValueError("C trained-control reconstruction must pass before candidate scoring")
    arms = ("C",) if stage in ("fit_control", "score_control") else (
        ("H", "HD") if stage in ("fit_candidates", "score_candidates") else ARMS)
    specs = [(arm, fold, seed) for arm in arms for fold in FOLDS for seed in SEEDS]
    if stage.startswith("fit"):
        specs = [s for s in specs if not receipt_path(*s).exists()]
    else:
        specs = [s for s in specs if not (SCORES / f"{tag(*s)}.json").exists()]
    processes = []
    logdir = OUT / "logs"; logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    for index, spec in enumerate(specs):
        arm, fold, seed = spec
        while len(processes) >= workers:
            p, log = processes.pop(0)
            code = p.wait(); log.close()
            if code: raise RuntimeError(f"Worker failed: {p.args}")
        name = tag(arm, fold, seed)
        log = (logdir / f"{stage}_{name}.log").open("ab")
        command = "fit" if stage.startswith("fit") else "score"
        p = subprocess.Popen([sys.executable, "-m", "experiments.source_conditioned_heads_execution",
            command, "--arm", arm, "--fold", fold, "--seed", str(seed)], cwd=ROOT,
            env=env, stdout=log, stderr=subprocess.STDOUT)
        processes.append((p, log))
        print("STARTED", stage, name, flush=True)
    for p, log in processes:
        code = p.wait(); log.close()
        if code: raise RuntimeError(f"Worker failed: {p.args}")
    if stage == "score_control":
        verify_training_control()


def _load_native(arm, fold, seed):
    path = prediction_path(arm, fold, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {str(k): np.asarray(archive[str(k)]) for k in archive["keys"]}


def aggregate() -> dict:
    declaration = check_frozen()
    _check_model_seal()
    _check_score_seal()
    initial_replay = read(OUT / "control_replay.json")
    if (initial_replay.get("study_id") != declaration["study_id"]
            or initial_replay.get("declaration_sha256") != sha(DECLARATION)
            or not initial_replay.get("passed")
            or len(initial_replay.get("replay_rows", [])) != 6
            or any(row.get("max_abs_probability_difference", float("inf")) > TOLERANCE
                   for row in initial_replay["replay_rows"])):
        raise ValueError("Initial C inference replay receipt is invalid")
    trained_replay_path = OUT / "training_control_replay.json"
    if not trained_replay_path.is_file():
        raise FileNotFoundError("Trained C control replay receipt missing")
    trained_replay = read(trained_replay_path)
    if (trained_replay.get("study_id") != declaration["study_id"]
            or trained_replay.get("declaration_sha256") != sha(DECLARATION)
            or trained_replay.get("model_seal_sha256") != sha(C_MODEL_SEAL)
            or not trained_replay.get("passed")
            or trained_replay.get("control_score_receipts") != {
                f"{fold}__{seed}": sha(SCORES / f"{tag('C', fold, seed)}.json")
                for fold in FOLDS for seed in SEEDS}):
        raise ValueError("Trained C control replay receipt is invalid")
    results = {"study_id": declaration["study_id"], "declaration_sha256": sha(DECLARATION),
               "arms": {}, "organizer_labels_read": False}
    for arm in ARMS:
        seed_rows, ensemble_rows = [], []
        for fold in FOLDS:
            per_seed = {}
            for seed in SEEDS:
                score = read(SCORES / f"{tag(arm, fold, seed)}.json")
                _verify_score_receipt(score, arm, fold, seed, declaration)
                per_seed[seed] = score["rows"]
            native = {s: _load_native(arm, fold, s) for s in SEEDS}
            if set(native[SEEDS[0]]) != set(native[SEEDS[1]]):
                raise ValueError("Seed ensemble cell keys differ")
            for seed in SEEDS:
                seed_rows.extend(per_seed[seed])
            for item in old.selection_items():
                if "leave_out_" + item["source"] != fold:
                    continue
                for view, rep in old.VIEWS:
                    key = f"{item['match']}__{view}__{rep}"
                    avg = (native[2026][key] + native[2027][key]) * 0.5
                    cached = joblib.load(old.eval_path(item, view, rep))
                    row, _ = old.metric_row(avg, cached["inputs"], cached["labels"], item,
                                             view, rep, arm, "ensemble")
                    ensemble_rows.append(row)
        expected_total = sum(len(_expected_keys(fold)) for fold in FOLDS)
        ensemble_keys = {f"{r['match']}__{r['view']}__{r['replicate']}" for r in ensemble_rows}
        if len(ensemble_rows) != expected_total or len(ensemble_keys) != expected_total or expected_total != 15:
            raise ValueError(f"Expected exactly 15 ensemble cells for {arm}; got {len(ensemble_rows)}")
        def by_view(rows, member="ensemble"):
            return {view: sc._aggregate_provider(rows, arm, view, member)
                    for view in ("clean", "feed_like")}
        # The seed table is descriptive; ensemble rows are the declared ranking scope.
        results["arms"][arm] = dict(seed_member_rows=seed_rows,
            ensemble_rows=ensemble_rows, ensemble=by_view(ensemble_rows),
            member_2026=by_view([r for f in FOLDS for r in read(SCORES /
                f"{tag(arm, f, 2026)}.json")["rows"]], "2026"),
            member_2027=by_view([r for f in FOLDS for r in read(SCORES /
                f"{tag(arm, f, 2027)}.json")["rows"]], "2027"))
        merge_split = {}
        for view in ("clean", "feed_like"):
            view_rows = [r for r in ensemble_rows if r["view"] == view]
            merge_split[view] = dict(
                fragmentation=sum(int(r["fragmented_reference_spells"]) for r in view_rows),
                merge_count=sum(int(r.get("predictions_overlapping_multiple_same_team_references", 0))
                                for r in view_rows),
                complete_boundary_timing_by_cell={
                    f"{r['match']}__{r['view']}__{r['replicate']}":
                        r["complete_spell_boundary_timing_iou_0_5"] for r in view_rows})
        results["arms"][arm]["split_merge_boundary_summary"] = merge_split
    baseline = results["arms"]["C"]["ensemble"]
    comparisons = {}
    for arm in ("H", "HD"):
        candidate = results["arms"][arm]["ensemble"]
        checks = sc.guard_checks(candidate, baseline)
        clean_gain = candidate["clean"]["equal_provider_mean"]["spell_f1"] - \
                     baseline["clean"]["equal_provider_mean"]["spell_f1"]
        comparisons[arm] = dict(guard_checks=checks, guard_pass=all(checks.values()),
            clean_spell_gain_vs_C=clean_gain,
            plus_0_01_clean_screen_pass=clean_gain >= 0.01)
    results["comparisons_vs_C"] = comparisons
    write_once(AGGREGATES, results)
    REPORT.write_text(_markdown_report(results))
    return results


def _markdown_report(value: dict) -> str:
    lines = ["# Source-conditioned output head comparison", "",
        "18 fits: C/H/HD × three full-provider folds × seeds 2026/2027. Seed ensembles average native probabilities before cost-3 decoding. All matches are exposed development data.", "",
        "| Arm | Clean spell | Feed-like spell | Clean macro | Clean D-F1 | Feed guard | Clean +.01 screen |",
        "|---|---:|---:|---:|---:|---|---|"]
    baseline = value["arms"]["C"]["ensemble"]
    for arm in ARMS:
        row = value["arms"][arm]["ensemble"]
        clean = row["clean"]["equal_provider_mean"]
        feed = row["feed_like"]["equal_provider_mean"]
        comp = value.get("comparisons_vs_C", {}).get(arm, {})
        lines.append(f"| {arm} | {clean['spell_f1']:.6f} | {feed['spell_f1']:.6f} | "
            f"{clean['macro_f1']:.6f} | {clean['d_f1']:.6f} | "
            f"{str(comp.get('guard_checks', {}).get('feed_like_spell', 'baseline'))} | "
            f"{str(comp.get('plus_0_01_clean_screen_pass', 'baseline'))} |")
    lines.extend(["", "The declared practical guard compares feed-like spell F1 and each provider's clean macro/D-F1 against C. The +.01 clean spell threshold is reported separately. No organizer score, promotion, or independent broadcast confirmation is included.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    sub.add_parser("replay")
    f = sub.add_parser("fit")
    s = sub.add_parser("score")
    for p in (f, s):
        p.add_argument("--arm", choices=ARMS, required=True)
        p.add_argument("--fold", choices=FOLDS, required=True)
        p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    q = sub.add_parser("queue"); q.add_argument("--workers", type=int, default=2)
    q.add_argument("--stage", choices=("fit_control", "score_control", "fit_candidates", "score_candidates"), required=True)
    sub.add_parser("seal-control-models")
    sub.add_parser("seal-models")
    sub.add_parser("seal-scores")
    sub.add_parser("training-control")
    sub.add_parser("aggregate")
    args = parser.parse_args()
    if args.cmd == "freeze": print(json.dumps(freeze(), indent=2))
    elif args.cmd == "replay": print(json.dumps(replay(), indent=2))
    elif args.cmd == "fit": print(json.dumps(fit_one(args.arm, args.fold, args.seed), indent=2))
    elif args.cmd == "score": print(json.dumps(score_one(args.arm, args.fold, args.seed), indent=2))
    elif args.cmd == "queue": queue(args.workers, args.stage)
    elif args.cmd == "seal-control-models": print(json.dumps(seal_control_models(), indent=2))
    elif args.cmd == "seal-models": print(json.dumps(seal_models(), indent=2))
    elif args.cmd == "seal-scores": print(json.dumps(seal_scores(), indent=2))
    elif args.cmd == "training-control": print(json.dumps(verify_training_control(), indent=2))
    elif args.cmd == "aggregate": print(json.dumps(aggregate(), indent=2))


if __name__ == "__main__":
    main()
