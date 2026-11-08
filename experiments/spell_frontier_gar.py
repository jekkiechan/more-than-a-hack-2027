"""Paired GAR evaluation of the frozen spell-frontier blend.

The prior GAR deployment view is reused as the paired comparator. TEST is
reported separately from the already exposed VALID split. This runner does
not fit, tune, promote, or access the organizer reference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import joblib
import numpy as np

from experiments import current_baseline as baseline
from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_like_v2 as v2
from experiments import supervised
from experiments import gar_deployment_eval_v2 as prior
from experiments import gar_heldout_eval_v1 as heldout
from experiments import soccernet_dense_labels as dense
from experiments import spell_frontier_baseline as package
from experiments import v2_iterative_runtime

ROOT = Path(__file__).resolve().parents[1]
STUDY = "spell_frontier_gar_v1"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
ABORTED_PREFLIGHT = ROOT / "experiments/results/spell_frontier_gar_v1_aborted_predeclaration.json"
OUT = ROOT / "data/experiments" / STUDY
GAMES = OUT / "games"
RECORDS = prior.RECORDS
MANIFEST = ROOT / "data/models/joint_gru_sn_dense_spell_frontier_ensemble_v1/baseline_manifest.json"
SPLITS = ("test", "valid")
VIEWS = ("clean", "v2_1", "v2_2", "v2_3", "v2_4")
REPS = (1, 2, 3, 4)
SEEDS = (2026, 2027)
WORKERS = min(2, max(1, int(os.environ.get("SPELL_FRONTIER_GAR_WORKERS", "1"))))
MATRIX_KEYS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
               "predicted_spell_count", "reference_spell_count", "short_spell_recall_iou_0_5_lt3s")
_MODEL_CACHE = {}
_RAW_FACTS_CACHE = None


def digest(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path | str) -> dict:
    return json.loads(Path(path).read_text())


def write_once(path: Path | str, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def model_package():
    if "manifest" in _MODEL_CACHE:
        return _MODEL_CACHE["manifest"], _MODEL_CACHE["models"]
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Stage the frozen spell-frontier manifest first: {MANIFEST}")
    manifest = read(MANIFEST)
    models = baseline.load_models(manifest_path=MANIFEST)
    package.validate(models, manifest)
    ids = set(_game_ids()["test"] + _game_ids()["valid"])
    for role, model in models.items():
        training = model.get("training") or {}
        seen = {str(game).removeprefix("soccernet_") for game in training.get("soccernet_games", [])}
        if seen & ids:
            raise ValueError(f"GAR evaluation game leaked into trained head {role}")
    if manifest.get("training", {}).get("ce_stage") != 1024:
        raise ValueError("Frontier package does not bind the selected pooled stage")
    _MODEL_CACHE.update(manifest=manifest, models=models)
    return manifest, models


def _raw_file_facts() -> list[dict]:
    global _RAW_FACTS_CACHE
    if _RAW_FACTS_CACHE is not None:
        return _RAW_FACTS_CACHE
    parent = read(prior.DECLARATION)
    tracking = dense.tracking_root(dense.DEFAULT_RAW)
    facts = []
    for row in parent["raw_files"]:
        path = tracking / row["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        actual = digest(path)
        if size != int(row["bytes"]) or actual != row["sha256"]:
            raise ValueError(f"GAR raw input differs from prior declaration: {row['path']}")
        facts.append({"path": row["path"], "bytes": size, "sha256": actual})
    _RAW_FACTS_CACHE = facts
    return facts


def _source_hashes() -> dict[str, str]:
    paths = (Path(__file__), Path(package.__file__), Path(prior.__file__), Path(heldout.__file__),
             Path(dense.__file__), Path(audit.__file__), Path(v2.__file__), Path(baseline.__file__),
             Path(v2_iterative_runtime.__file__),
             ROOT / "experiments/results/gar_deployment_eval_v2_declaration.json",
             ROOT / "experiments/results/gar_deployment_eval_v2.json",
             ROOT / "experiments/results/gar_heldout_eval_v1_download_receipt.json",
             ROOT / "experiments/results/gar_deployment_eval_v2_verification.json",
             PLAN, ABORTED_PREFLIGHT,
             ROOT / "experiments/tests/test_spell_frontier_gar.py")
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def _prior_game_artifacts(split: str, game: str) -> dict:
    inventory = read(ROOT / "experiments/results/gar_deployment_eval_v2_artifacts.json")
    rows = {row["name"]: row for row in inventory["native_predictions"]}
    name = f"{split}__{game}.npz"
    if name not in rows:
        raise ValueError(f"Frozen GAR artifact inventory lacks {name}")
    report = read(prior.RESULT)
    scores = report["splits"][split]["scores"]
    clean = scores["clean"]["original_full"]["phase_avg"]["per_game"][game]
    v2 = scores["v2"]["original_full"]["phase_avg"]["per_game"][game]
    return {"inventory": rows[name], "clean": clean, "v2": v2}


def _game_ids() -> dict[str, list[str]]:
    parent = read(prior.DECLARATION)
    ids = parent["match_ids"]
    if (len(ids.get("test", [])) != 10 or len(ids.get("valid", [])) != 9 or
            set(ids["test"]) & set(ids["valid"])):
        raise ValueError("Prior GAR declaration must provide disjoint TEST10 and VALID9 IDs")
    return {split: list(ids[split]) for split in SPLITS}


def declare() -> dict:
    if DECLARATION.exists():
        raise FileExistsError(f"Preserve existing {STUDY} declaration")
    parent = read(prior.DECLARATION)
    parent_result = read(prior.RESULT)
    if (parent.get("study") != prior.STUDY or parent_result.get("study") != prior.STUDY or
            parent_result.get("declaration_sha256") != digest(prior.DECLARATION)):
        raise ValueError("Frozen GAR deployment artifacts are not aligned")
    if parent.get("organizer_reference_access") is not False or parent_result.get("organizer_reference_access") is not False:
        raise ValueError("GAR comparator artifacts claim organizer reference access")
    manifest, _models = model_package()
    ids = _game_ids()
    raw_files = _raw_file_facts()
    record_manifests = {}
    for split in SPLITS:
        path = RECORDS / split / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"Prepare prior GAR records before freezing: {path}")
        record_manifests[split] = {"path": str(path.relative_to(ROOT)), "sha256": digest(path)}
        record_manifest = read(path)
        if set(record_manifest["games"]) != set(ids[split]):
            raise ValueError(f"Prepared records differ from prior GAR game IDs: {split}")
    prior_games = {f"{split}/{game}": _prior_game_artifacts(split, game)
                   for split in SPLITS for game in ids[split]}
    calibration = heldout.CALIBRATION
    calibration_sha = digest(calibration)
    if calibration_sha != parent.get("calibration_sha256"):
        raise ValueError("GAR v2 calibration differs from prior declaration")
    payload = {"study": STUDY, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Paired, frozen deployment check of the spell-frontier pooled blend on previously used GAR TEST/VALID data.",
        "selection_or_tuning": False, "fitting": False, "public_scoring": False,
        "organizer_reference_access": False, "promotion": False,
        "exposure": {"test": "Repeated GAR TEST exposure; descriptive paired check, not fresh confirmation.",
                     "valid": "Previously exposed GAR VALID; separate development diagnostic."},
        "game_ids": ids, "splits": {"test": {"games": 10}, "valid": {"games": 9}},
        "views": list(VIEWS), "corruption_repetitions": list(REPS),
        "corruption_seed": "sha256(f'gar_deployment_eval_v2/{rep}/{game_id}')[:8], decimal big-endian",
        "aggregation": "Within each game: equal four v2 realizations, then equal games; TEST and VALID separate.",
        "package": {"manifest_path": str(MANIFEST.relative_to(ROOT)),
                    "manifest_sha256": digest(MANIFEST), "architecture": manifest["architecture"],
                    "model_sha256": manifest["models_sha256"], "roles": [f"{r}_{s}" for r in package.ROLES for s in package.SEEDS]},
        "parent_declaration_sha256": digest(prior.DECLARATION), "parent_result_sha256": digest(prior.RESULT),
        "frozen_plan_sha256": digest(PLAN),
        "aborted_predeclaration_preflight_sha256": digest(ABORTED_PREFLIGHT),
        "prior_control_reconstruction": {
            "prediction_payload_available_locally": False,
            "available_prior_evidence": "Frozen per-game score report plus verified 19-file prediction inventory; NPZ payloads were absent locally.",
            "replay": "Reconstruct original_full phase_avg through the sealed adopted two-seed package on the same clean and four seeded views; require clean per-game and four-view mean per-game metrics to match the frozen report exactly.",
            "saved_npz_hashes_are_inventory_only": True},
        "prior_game_artifacts": {key: value for key, value in prior_games.items()},
        "prior_artifact_inventory_sha256": digest(ROOT / "experiments/results/gar_deployment_eval_v2_artifacts.json"),
        "prepared_record_manifests": record_manifests, "raw_files": raw_files,
        "calibration_sha256": calibration_sha, "source_sha256": _source_hashes(),
        "runtime_inference_only": True, "prior_test_exposure": True}
    write_once(DECLARATION, payload)
    return payload


def check_declaration() -> dict:
    declaration = read(DECLARATION)
    if declaration.get("study") != STUDY or declaration.get("source_sha256") != _source_hashes():
        raise ValueError("Spell-frontier GAR source changed after freeze")
    if declaration.get("parent_declaration_sha256") != digest(prior.DECLARATION) or declaration.get("parent_result_sha256") != digest(prior.RESULT):
        raise ValueError("Prior GAR comparator changed after freeze")
    if _raw_file_facts() != declaration.get("raw_files"):
        raise ValueError("Restored GAR tracking inputs changed after freeze")
    if declaration.get("prior_artifact_inventory_sha256") != digest(ROOT / "experiments/results/gar_deployment_eval_v2_artifacts.json"):
        raise ValueError("Prior GAR artifact inventory changed")
    manifest, models = model_package()
    if (digest(MANIFEST) != declaration["package"]["manifest_sha256"] or
            manifest["models_sha256"] != declaration["package"]["model_sha256"]):
        raise ValueError("Frozen frontier package changed")
    for split, fact in declaration["prepared_record_manifests"].items():
        if digest(ROOT / fact["path"]) != fact["sha256"]:
            raise ValueError(f"Prepared GAR records changed: {split}")
    return declaration


def _models():
    if "models" not in _MODEL_CACHE:
        _MODEL_CACHE["manifest"], _MODEL_CACHE["models"] = model_package()
    return _MODEL_CACHE["manifest"], _MODEL_CACHE["models"]


def _corrupt_view(inputs, labels, calibration, game_id, rep):
    return heldout.feed_view(inputs, labels, calibration, f"{prior.STUDY}/{game_id}", rep)


def _score(labels, inputs, prediction):
    cell = {"labels": labels, "period": inputs["period"],
            "timestamps": inputs["timestamps"], "fps": float(inputs["fps"])}
    row = audit.score(cell, prediction)
    return {k: row[k] for k in MATRIX_KEYS if k in row}


def _run_game(task):
    split, game_id = task
    check_declaration()
    output = GAMES / f"{split}__{game_id}.json"
    if output.exists():
        saved = read(output)
        pred_path = ROOT / saved["prediction_path"]
        if (saved.get("study") != STUDY or saved.get("declaration_sha256") != digest(DECLARATION) or
                not pred_path.is_file() or digest(pred_path) != saved.get("prediction_sha256")):
            raise ValueError(f"Existing GAR game output failed its frozen receipt: {split}/{game_id}")
        return str(output)
    started = time.monotonic()
    declaration = read(DECLARATION)
    prior_fact = declaration["prior_game_artifacts"][f"{split}/{game_id}"]
    records = heldout.load_game_records(split, game_id, RECORDS)
    captured = heldout.capture_inputs(split, game_id)
    inputs, labels, saved_x, names = heldout.build_game(records, captured)
    if len(labels) != int(prior_fact["inventory"]["native_frames"]):
        raise ValueError(f"Paired frame count differs from frozen GAR artifact inventory: {split}/{game_id}")
    if int((labels != "X").sum()) != int(prior_fact["inventory"]["graded_frames"]):
        raise ValueError(f"Paired graded-frame count differs from frozen GAR artifact inventory: {split}/{game_id}")
    sampled_fps = float(records[0]["fps"])
    calibration = read(heldout.CALIBRATION)["calibration"]
    manifest, models = _models()
    package.validate(models, manifest)
    view_rows, predictions, baseline_predictions, corruption_seeds = {}, {}, {}, {}
    reconstructed_control_metrics = {}
    for view in VIEWS:
        if view == "clean":
            view_inputs = inputs
        else:
            view_inputs, seed = _corrupt_view(inputs, labels, calibration, game_id, int(view[-1]))
            corruption_seeds[view] = seed
        with v2_iterative_runtime.installed():
            phases = heldout.phase_set(view_inputs, names, sampled_fps)
        if len(phases) != round(float(view_inputs["fps"]) / 5.0):
            raise ValueError(f"Unexpected GAR native phase count: {split}/{game_id}/{view}")
        if view == "clean" and not np.array_equal(phases[0]["x"], saved_x, equal_nan=True):
            raise ValueError(f"Phase-0 features differ from prior GAR dense records: {split}/{game_id}")
        components = package.components_from_records(view_inputs, phases, models, manifest)
        prediction = np.asarray(package.decode_components(view_inputs, components)).astype("U1")
        previous_prediction = np.asarray(audit.decode(components["adopted"],
            supervised.model_periods(view_inputs), float(view_inputs["fps"]), audit.CONFIG)).astype("U1")
        if prediction.shape != previous_prediction.shape or not np.array_equal(prediction == "D", previous_prediction == "D"):
            raise ValueError(f"Spell-frontier changed the exact adopted D mask: {split}/{game_id}/{view}")
        metrics = _score(labels, view_inputs, prediction)
        previous_metrics = _score(labels, view_inputs, previous_prediction)
        if metrics["reference_spell_count"] != previous_metrics["reference_spell_count"]:
            raise ValueError(f"Reference counts changed: {split}/{game_id}/{view}")
        reconstructed_control_metrics[view] = previous_metrics
        view_rows[view] = {"spell_frontier": metrics, "original_full_phase_avg": previous_metrics,
            "delta": {key: float(metrics[key]) - float(previous_metrics[key])
                      for key in MATRIX_KEYS if key in metrics and key in previous_metrics}}
        predictions[view] = prediction.astype("S1")
        baseline_predictions[view] = previous_prediction.astype("S1")
    for metric in ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall", "reference_spell_count"):
        if not np.isclose(reconstructed_control_metrics["clean"][metric], prior_fact["clean"][metric], atol=1e-12, rtol=0):
            raise ValueError(f"Reconstructed adopted clean control differs from prior report: {split}/{game_id}/{metric}")
        v2_mean = float(np.mean([reconstructed_control_metrics[view][metric] for view in VIEWS[1:]]))
        if not np.isclose(v2_mean, prior_fact["v2"][metric], atol=1e-12, rtol=0):
            raise ValueError(f"Reconstructed adopted v2 control differs from prior report: {split}/{game_id}/{metric}")
    path = GAMES / f"{split}__{game_id}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, labels=labels.astype("S1"), period=inputs["period"],
        timestamps=inputs["timestamps"], fps=inputs["fps"],
        **{f"pred__{view}": pred for view, pred in predictions.items()},
        **{f"adopted_control__{view}": pred for view, pred in baseline_predictions.items()})
    sidecar = {"study": STUDY, "split": split, "game": game_id,
        "declaration_sha256": digest(DECLARATION), "records": len(records),
        "native_frames": int(len(labels)), "graded_frames": int((labels != "X").sum()),
        "phase_count": len(phases), "views": view_rows,
        "corruption_seeds": corruption_seeds, "paired_D_mask_equal": True,
        "prediction_path": str(path.relative_to(ROOT)), "prediction_sha256": digest(path),
        "baseline_reconstructed_from_frozen_adopted_package": True,
        "baseline_paired_reference_counts_equal": True,
        "seconds": time.monotonic() - started, "organizer_reference_access": False,
        "public_scoring": False, "fitting": False, "tuning": False, "promotion": False}
    write_once(output, sidecar)
    return str(output)


def run_games(workers: int = WORKERS) -> None:
    if workers not in (1, 2):
        raise ValueError("GAR inference is limited to one or two workers")
    declaration = check_declaration()
    GAMES.mkdir(parents=True, exist_ok=True)
    jobs = [(split, game) for split in SPLITS for game in declaration["game_ids"][split]]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(_run_game, task) for task in jobs]):
            print("DONE", future.result(), flush=True)


def _aggregate_game(rows: list[dict], split: str) -> dict:
    if len(rows) != (10 if split == "test" else 9):
        raise ValueError(f"Unexpected {split.upper()} game count")
    metric_keys = list(MATRIX_KEYS)
    by_view = {}
    for view in VIEWS:
        per_game = {row["game"]: row["views"][view] for row in rows}
        frontier_means = {key: float(np.mean([per_game[g]["spell_frontier"][key] for g in per_game]))
                          for key in metric_keys}
        baseline_means = {key: float(np.mean([per_game[g]["original_full_phase_avg"][key] for g in per_game]))
                          for key in metric_keys}
        delta_means = {key: float(np.mean([per_game[g]["delta"][key] for g in per_game]))
                       for key in metric_keys}
        by_view[view] = {"spell_frontier_equal_game": frontier_means,
            "original_full_phase_avg_equal_game": baseline_means,
            "paired_delta_equal_game": delta_means, "per_game": per_game}
    v2_per_game = {row["game"]: {key: float(np.mean([row["views"][view]["spell_frontier"][key]
                       for view in VIEWS[1:]])) for key in metric_keys} for row in rows}
    prior_v2_per_game = {row["game"]: {key: float(np.mean([row["views"][view]["original_full_phase_avg"][key]
                       for view in VIEWS[1:]])) for key in metric_keys} for row in rows}
    delta_v2_per_game = {game: {key: v2_per_game[game][key] - prior_v2_per_game[game][key]
                                for key in metric_keys} for game in v2_per_game}
    by_view["v2_mean_over_four_reps"] = {"spell_frontier_equal_game": {
        key: float(np.mean([row[key] for row in v2_per_game.values()])) for key in metric_keys},
        "original_full_phase_avg_equal_game": {
        key: float(np.mean([row[key] for row in prior_v2_per_game.values()])) for key in metric_keys},
        "paired_delta_equal_game": {
        key: float(np.mean([row[key] for row in delta_v2_per_game.values()])) for key in metric_keys},
        "per_game": {game: {"spell_frontier": v2_per_game[game],
                            "original_full_phase_avg": prior_v2_per_game[game],
                            "delta": delta_v2_per_game[game]} for game in v2_per_game}}
    return by_view


def report() -> dict:
    declaration = check_declaration()
    result = {"study": STUDY, "declaration_sha256": digest(DECLARATION), "splits": {},
        "organizer_reference_access": False, "public_scoring": False,
        "fitting": False, "tuning": False, "promotion": False,
        "test_exposure": "Repeated exposure; paired descriptive deployment check only.",
        "valid_exposure": "Previously selection-exposed; separate development diagnostic only."}
    for split in SPLITS:
        rows = [read(GAMES / f"{split}__{game}.json") for game in declaration["game_ids"][split]]
        for row in rows:
            if row.get("declaration_sha256") != digest(DECLARATION):
                raise ValueError("Mixed study declarations in game outputs")
            pred_path = ROOT / row["prediction_path"]
            if digest(pred_path) != row["prediction_sha256"]:
                raise ValueError(f"Prediction cache changed: {row['game']}")
        result["splits"][split] = {"games": len(rows),
            "records": sum(row["records"] for row in rows),
            "graded_frames": sum(row["graded_frames"] for row in rows),
            "scores": _aggregate_game(rows, split)}
    write_once(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "check", "games", "report"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args()
    action = args.action
    if action == "declare":
        declare()
    elif action == "check":
        check_declaration()
    elif action == "games":
        run_games(args.workers)
    else:
        report()


if __name__ == "__main__":
    main()
