"""Bounded TRAIN-only turnover-window continuation from reconstructed CE."""
from __future__ import annotations

from contextlib import contextmanager
import argparse
import hashlib
import json
from pathlib import Path
import random as std_random
import subprocess
import sys
import time
from types import SimpleNamespace

import joblib
import numpy as np

from experiments import frontier_recovery_training_v1 as recovery
from experiments import turnover_event_sampler as event_sampler
from experiments import v2_iterative_train as iterative
from experiments import v2_spell_frontier_train as frontier
from experiments import v2_extra_data_train as extra

ROOT = Path(__file__).resolve().parents[1]
STUDY = "turnover_supervision_v1"
PLAN = ROOT / "experiments/TURNOVER_SUPERVISION_V1_PLAN.md"
OUT = ROOT / "data/experiments" / STUDY / "train"
TRAIN_DECL = ROOT / "experiments/results/turnover_supervision_train_declaration.json"
FIT_SEAL = OUT / "training_seal.json"
ARMS = ("fresh_control", "uniform_turnover", "hard_turnover")
FOLDS, SEEDS = iterative.FOLDS, iterative.SEEDS
UPDATES, LR = 256, 1e-4
BASE_STAGE, FRESH_OFFSET, TURNOVER_RATE = 1024, extra.FRESH_OFFSET, .25


def sha(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, default=float)
        stream.write("\n")


def output_path(arm, fold, seed):
    if arm not in ARMS or fold not in FOLDS or seed not in SEEDS:
        raise ValueError("undeclared turnover fit")
    return OUT / "models" / arm / f"{fold}_{seed}.joblib"


def _source_paths():
    recipe = recovery.recipe
    return {Path(__file__), Path(event_sampler.__file__),
            ROOT / "experiments/tests/test_turnover_supervision_train.py",
            ROOT / "experiments/tests/test_turnover_sampler_schedule.py",
            PLAN, Path(recovery.__file__), Path(iterative.__file__),
            Path(frontier.__file__), Path(extra.__file__),
            *recovery._source_files(),
            *recipe._source_files(), *recipe._view_pipeline_files()}


def _source_facts():
    return {str(path.relative_to(ROOT)): sha(path)
            for path in sorted(_source_paths())}


def _input_facts():
    recovery._configure()
    # Verify the reconstruction's exact source and actual data binding.
    recovery._check_training_binding()
    facts = {"recovery_training_declaration": sha(recovery.TRAIN_DECL),
             "recovery_preparation": sha(recovery.TRAIN_VIEW_OUT / "preparation.json"),
             "recovery_source_sha256": recovery._training_source_hashes(),
             "recovery_input_sha256": recovery._training_inputs()}
    for fold in FOLDS:
        for seed in SEEDS:
            path = recovery._stage_path(BASE_STAGE, fold, seed)
            receipt_path = Path(str(path) + ".json")
            receipt = json.loads(receipt_path.read_text())
            if (receipt.get("sha256") != sha(path) or
                    int(receipt.get("stage", -1)) != BASE_STAGE or
                    receipt.get("fold") != fold or int(receipt.get("seed", -1)) != seed or
                    receipt.get("declaration_sha256") != sha(recovery.TRAIN_DECL)):
                raise ValueError(f"reconstructed stage-1024 checkpoint receipt mismatch: {fold}/{seed}")
            facts[f"stage1024/{fold}/{seed}"] = {
                "path": str(path), "sha256": sha(path),
                "receipt_path": str(receipt_path), "receipt_sha256": sha(receipt_path)}
            teacher_path = recovery.recipe.rf.sc.initial_path(fold, seed)
            facts[f"teacher/{fold}/{seed}"] = {"path": str(teacher_path), "sha256": sha(teacher_path)}
    expected_matches = {str(x["match"]) for x in recovery.recipe.prep.split_plan(fold="pooled")["train"]}
    if len(expected_matches) != 13:
        raise ValueError("expected the fixed 13 original-provider TRAIN matches")
    return facts


def freeze():
    """Write the immutable training declaration; deliberately separate from fitting."""
    if TRAIN_DECL.exists() or FIT_SEAL.exists():
        raise FileExistsError("preserve existing turnover training declaration/seal")
    recovery._configure()
    inputs = _input_facts()
    cohorts = {}
    for fold in FOLDS:
        items = recovery.recipe.prep.split_plan(fold=fold)["train"]
        excluded = fold.removeprefix("leave_out_")
        if any(str(item.get("source", "")).lower() == excluded for item in items):
            raise ValueError(f"held-out provider leaked into {fold}")
        cohorts[fold] = sorted({str(item["match"]) for item in items})
    plan_hash = sha(PLAN)
    declaration = {
        "schema": "turnover_supervision_train_v1", "study_id": STUDY,
        "plan_path": str(PLAN.relative_to(ROOT)), "plan_sha256": plan_hash,
        "source_sha256": _source_facts(), "input_sha256": inputs,
        "arms": list(ARMS), "folds": list(FOLDS), "seeds": list(SEEDS),
        "budget": {"fits": 18, "updates_per_fit": UPDATES,
                   "cumulative_stage": BASE_STAGE + UPDATES, "lr": LR,
                   "batch_size": iterative.BATCH_SIZE,
                   "original_windows_per_batch": 16, "gar_windows_per_batch": 16,
                   "workers": 1},
        "cohorts": cohorts,
        "sampling": {"base_sampler": "OriginalDiversitySampler, invoked on every draw",
                     "original_short_replacement_rate": .25,
                     "turnover_override_rate": TURNOVER_RATE,
                     "turnover_eligibility": "native adjacent H/A edge; stable same-team H/A for a full second on each side; same native period; both contexts within existing selected-record core",
                     "turnover_duration": "all eligible edge events independent of adjacent possession duration",
                     "hardness": "min(mean pre-edge and post-edge correct-team conditional probability over clean native one-second contexts); lower is harder; bottom half within physical match, stable identity tie-break",
                     "hardness_source": "matched fold/seed reconstructed stage-1024 CE checkpoint on one clean unswapped original record per physical TRAIN match",
                     "gar_quality_and_views": "unchanged existing quality sampler and four v2 realizations; same record/view and GAR schedule across arms",
                     "freshness_offset": FRESH_OFFSET},
        "objective": "existing dense duration-weighted H/A CE plus live-transition loss; GAR fraction .5; D state frozen",
        "training_matches_only": True, "gar_test_or_valid_access": False,
        "organizer_reference_access": False, "public_scoring": False, "promotion": False}
    write_once(TRAIN_DECL, declaration)
    return declaration


def check(full=True):
    recovery._configure()
    declaration = json.loads(TRAIN_DECL.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("source_sha256") != _source_facts():
        raise ValueError("turnover training source changed after freeze")
    if declaration.get("plan_sha256") != sha(PLAN):
        raise ValueError("predeclared turnover plan changed after freeze")
    if full and declaration.get("input_sha256") != _input_facts():
        raise ValueError("turnover training inputs changed after freeze")
    return declaration


def _clean_records(records):
    """Pick one direct clean feature representation per physical match."""
    result = {}
    for record in records:
        match = str(record["match"])
        if match in result:
            continue
        # The frozen recovery recipe emits its clean direct/swap pair first.
        indices = np.asarray(record["indices"], dtype=np.int64)
        if not np.array_equal(np.asarray(record["y"]), np.asarray(record["full_y"])[indices]):
            raise ValueError(f"first record is not direct clean labels for {match}")
        result[match] = record
    return result


def _conditional_native(record, model, teacher):
    """Run the paired-swap invariant CE predictor and upsample to native rows."""
    from experiments import learned_temporal as lt
    # Fix D to zero so this measures the CE expert's conditional H/A signal,
    # independent of the copied frozen D component.
    baseline = np.zeros((len(record["x"]), 3), dtype=np.float32)
    baseline[:, :2] = .5
    conditional = iterative.cap.predict(record, model, teacher,
        original_probability=baseline)[:, :2]
    full_period = np.asarray(record["full_period"])
    info = {"indices": np.asarray(record["indices"]), "native_n": len(full_period)}
    return np.column_stack([lt.upsample(conditional[:, k], info, full_period)
                            for k in range(2)])


def hardness_scores(records, checkpoint, teacher):
    """Score each physical match once using the symmetric clean checkpoint."""
    scores = {}
    clean = _clean_records(records)
    for match, record in sorted(clean.items()):
        probabilities = _conditional_native(record, checkpoint, teacher)
        for event in event_sampler.eligible_events(record):
            edge = event["edge"]
            before = probabilities[edge + 1 - int(round(float(record.get("native_fps", record["fps"])))):edge + 1]
            after = probabilities[edge + 1:edge + 1 + int(round(float(record.get("native_fps", record["fps"]))))]
            pre_correct = float(np.mean(before[:, event["left"]]))
            post_correct = float(np.mean(after[:, event["right"]]))
            scores[event["identity"]] = {"pre_correct": pre_correct,
                                          "post_correct": post_correct,
                                          "hardness": min(pre_correct, post_correct)}
    return scores


@contextmanager
def _training_scope():
    old_random, old_lr, old_fraction, old_study = (
        iterative.random, iterative.LR, iterative.FRACTION_GAR, iterative.STUDY)
    iterative.random = SimpleNamespace(
        Random=lambda seed: std_random.Random(int(seed) + FRESH_OFFSET))
    iterative.LR, iterative.FRACTION_GAR, iterative.STUDY = LR, .5, STUDY
    try:
        yield
    finally:
        iterative.random, iterative.LR = old_random, old_lr
        iterative.FRACTION_GAR, iterative.STUDY = old_fraction, old_study


def fit_one(fold, seed, arm, *, verbose=True):
    check(full=False)
    path = output_path(arm, fold, seed)
    receipt_path = Path(str(path) + ".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite turnover fit: {path}")
    teacher_path = recovery.recipe.rf.sc.initial_path(fold, seed)
    input_facts = json.loads(TRAIN_DECL.read_text())["input_sha256"]
    warm_key, teacher_key = f"stage1024/{fold}/{seed}", f"teacher/{fold}/{seed}"
    warm_fact, teacher_fact = input_facts[warm_key], input_facts[teacher_key]
    if sha(warm_fact["path"]) != warm_fact["sha256"] or sha(teacher_fact["path"]) != teacher_fact["sha256"]:
        raise ValueError("fold/seed checkpoint input changed after training freeze")
    if sha(warm_fact["receipt_path"]) != warm_fact["receipt_sha256"]:
        raise ValueError("reconstructed checkpoint receipt changed after training freeze")
    if sha(teacher_path) != teacher_fact["sha256"]:
        raise ValueError("resolved teacher differs from frozen fold/seed teacher")
    teacher = joblib.load(teacher_path)
    warm_path = recovery._stage_path(BASE_STAGE, fold, seed)
    if str(warm_path) != warm_fact["path"]:
        raise ValueError("resolved CE checkpoint differs from frozen input path")
    checkpoint = joblib.load(warm_path)
    if checkpoint.get("training", {}).get("stage") != BASE_STAGE:
        raise ValueError("warm start is not reconstructed CE stage 1024")
    if checkpoint.get("source", {}).get("checkpoint_sha256") != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("warm start teacher mismatch")
    old_records, gar_records = recovery._get_training_records(fold), recovery._GAR_CACHE
    if gar_records is None or len({str(x["match"]) for x in gar_records}) != 45:
        raise ValueError("expected exactly 45 GAR TRAIN games")
    hardness = None
    hardness_file = None
    if arm == "hard_turnover":
        hardness_file = OUT / "private" / f"hardness_{fold}_{seed}.json"
        expected_inputs = json.loads(TRAIN_DECL.read_text())["input_sha256"]
        if hardness_file.exists():
            saved = json.loads(hardness_file.read_text())
            if (saved.get("warm_start_sha256") != sha(warm_path) or
                    saved.get("input_sha256") != expected_inputs or
                    saved.get("teacher_sha256") != sha(teacher_path)):
                raise ValueError("hardness catalog is bound to different checkpoint or inputs")
            hardness = {tuple(row["identity"]): row for row in saved["events"]}
        else:
            raw = hardness_scores(old_records, checkpoint, teacher)
            catalog = {"study_id": STUDY, "fold": fold, "seed": seed,
                       "warm_start_path": str(warm_path),
                       "warm_start_sha256": sha(warm_path),
                       "teacher_path": str(teacher_path),
                       "teacher_sha256": sha(teacher_path),
                       "input_sha256": expected_inputs,
                       "events": [{"identity": list(identity), **score}
                                  for identity, score in sorted(raw.items())]}
            write_once(hardness_file, catalog)
            hardness = {tuple(row["identity"]): row for row in catalog["events"]}
    sampler = event_sampler.TurnoverSampler(old_records, gar_records,
        list(teacher["feature_names"]), seed, arm=arm, hardness=hardness,
        replacement_rate=TURNOVER_RATE)
    sampler.original_rng = std_random.Random(seed + FRESH_OFFSET + 1106644)
    initial = iterative.warmstart_state(teacher, checkpoint)
    view_ctx = recovery.recipe._round_robin_v2_windows(old_records)
    started = time.monotonic()
    with frontier.capture_windows(recovery.recipe, old_records, gar_records, sampler) as sampled:
        with view_ctx as view_draws:
            with _training_scope():
                result = iterative._fit_loop(initial, old_records, gar_records,
                    seed=seed, arm="ce_control", verbose=verbose, steps=UPDATES)
    model = result["model"]
    for key, value in checkpoint.get("d_state", {}).items():
        np.testing.assert_array_equal(model["d_state"][key], value)
    if set(model["d_state"]) != set(checkpoint["d_state"]):
        raise ValueError("frozen D state changed")
    for key in ("median", "scale"):
        np.testing.assert_array_equal(model[key], checkpoint[key])
    if model["feature_names"] != checkpoint["feature_names"]:
        raise ValueError("normalization schema changed")
    provenance = {"study": STUDY, "arm": arm, "fold": fold, "seed": seed,
                  "stage": BASE_STAGE + UPDATES, "stage_updates": UPDATES,
                  "cumulative_updates": BASE_STAGE + UPDATES,
                  "warm_start_stage": BASE_STAGE, "warm_start_sha256": sha(warm_path),
                  "draw_seed_offset": FRESH_OFFSET,
                  "turnover_rate": TURNOVER_RATE if arm != "fresh_control" else 0.0}
    model["training"].update(provenance)
    model["step"] = BASE_STAGE + UPDATES
    result["config"].update(provenance)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    windows = [list(x) for x in sampled]
    receipt = {**provenance, "model_path": str(path.relative_to(ROOT)),
        "model_sha256": sha(path), "declaration_sha256": sha(TRAIN_DECL),
        "warm_start_path": str(warm_path), "teacher_path": str(teacher_path),
        "teacher_sha256": sha(teacher_path), "config": result["config"],
        "sampling": sampler.receipt(), "view_draws": dict(view_draws),
        "exposures": result["exposures"], "loss_intervals": result.get("loss_intervals", []),
        "sampled_window_ids": windows,
        "draw_sequence_sha256": hashlib.sha256(json.dumps(windows, separators=(",", ":")).encode()).hexdigest(),
        "draw_count": len(windows), "wall_seconds": time.monotonic() - started,
        "d_state_unchanged": True, "normalization_unchanged": True,
        "organizer_reference_access": False}
    if arm == "hard_turnover":
        receipt["hardness_catalog_path"] = str(hardness_file.relative_to(ROOT))
        receipt["hardness_catalog_sha256"] = sha(hardness_file)
    receipt["old_record_index"] = [
        {"index": ri, "source": str(record["source"]), "match": str(record["match"]),
         "v2_view_match": str(record.get("_v2_view_match", "")),
         "v2_view_count": len(record.get("_v2_feature_views", [])),
         "original_clean_or_gap_view": int(ri % 4) if len(old_records) % 4 == 0 else None}
        for ri, record in enumerate(old_records)]
    write_once(receipt_path, receipt)
    return receipt


def _fit_subprocess(fold, seed, arm):
    command = [sys.executable, "-m", "experiments.turnover_supervision_train", "fit-one",
               "--fold", fold, "--seed", str(seed), "--arm", arm]
    subprocess.run(command, check=True, cwd=ROOT)
    path = output_path(arm, fold, seed)
    return json.loads(Path(str(path) + ".json").read_text())


def fit_all():
    """Run each declared cell in a fresh serial process to release TRAIN caches."""
    declaration = check(full=True)
    if FIT_SEAL.exists():
        raise FileExistsError("preserve existing turnover fit seal")
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            for arm in ARMS:
                rows.append(_fit_subprocess(fold, seed, arm))
    by_key = {(r["fold"], r["seed"], r["arm"]): r for r in rows}
    parity = {}
    for fold in FOLDS:
        for seed in SEEDS:
            base = by_key[(fold, seed, "fresh_control")]
            for arm in ARMS[1:]:
                row = by_key[(fold, seed, arm)]
                for key in ("view_draws", "exposures"):
                    if row[key] != base[key]:
                        raise ValueError(f"matched {key} differs for {fold}/{seed}/{arm}")
                parity[f"{fold}/{seed}/{arm}"] = {
                    "same_record_view_and_GAR_exposure": True,
                    "different_window_sequence": row["draw_sequence_sha256"] != base["draw_sequence_sha256"]}
    if not all(row["different_window_sequence"] for row in parity.values()):
        raise ValueError("turnover arms did not alter sampled window sequences")
    return {"study_id": STUDY, "fits": rows, "fit_count": len(rows),
            "all_18_complete": len(rows) == 18, "matched_schedule_parity": parity}


def seal():
    check(full=True)
    if FIT_SEAL.exists():
        raise FileExistsError("preserve existing turnover fit seal")
    rows = []
    for fold in FOLDS:
        for seed in SEEDS:
            for arm in ARMS:
                path = output_path(arm, fold, seed)
                receipt_path = Path(str(path) + ".json")
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("model_sha256") != sha(path) or receipt.get("declaration_sha256") != sha(TRAIN_DECL):
                    raise ValueError(f"fit receipt mismatch: {fold}/{seed}/{arm}")
                rows.append(receipt)
    by_key = {(r["fold"], int(r["seed"]), r["arm"]): r for r in rows}
    parity = {}
    for fold in FOLDS:
        for seed in SEEDS:
            base = by_key[(fold, seed, "fresh_control")]
            for arm in ARMS[1:]:
                row = by_key[(fold, seed, arm)]
                for key in ("view_draws", "exposures"):
                    if row[key] != base[key]:
                        raise ValueError(f"matched {key} differs: {fold}/{seed}/{arm}")
                parity[f"{fold}/{seed}/{arm}"] = {
                    "same_record_view_and_GAR_exposure": True,
                    "different_window_sequence": row["draw_sequence_sha256"] != base["draw_sequence_sha256"]}
    if not all(value["different_window_sequence"] for value in parity.values()):
        raise ValueError("turnover arms did not alter sampled window sequences")
    result = {"study_id": STUDY, "declaration_sha256": sha(TRAIN_DECL),
              "fits": rows, "fit_count": len(rows), "all_18_complete": len(rows) == 18,
              "matched_schedule_parity": parity, "organizer_reference_access": False,
              "promotion": False}
    write_once(FIT_SEAL, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit-one", "fit-all", "seal"))
    parser.add_argument("--fold", choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "fit-one":
        if args.fold is None or args.seed is None or args.arm is None:
            parser.error("fit-one requires --fold, --seed and --arm")
        fit_one(args.fold, args.seed, args.arm)
    elif args.action == "fit-all":
        fit_all()
    else:
        seal()
