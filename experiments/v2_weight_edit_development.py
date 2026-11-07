"""Bounded development evaluation of direct H/A recurrent weight edits.

This new runner consumes the reconstructed four-view model family and its
training-only directions. It compares six fixed direct tensor edits with the
unmodified 50/50 new/calibrated blend at the already-selected H/A cost 1.5.
The archived original_full phase package is replayed as a separate deployed
control. Public and GAR scoring belong to a separate frozen deployment study.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import calibrated_dense_baseline as calibration
from experiments import feed_spell_push, original_full_phase_baseline
from experiments import v2_spell_target_runtime as upstream
from experiments import v2_training_eval as v2_eval
from experiments import v2_weight_edit_reconstruction as reconstruction
from experiments import weight_space_edit

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_weight_edit_development_v1"
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
CONTROL_RESULT = ROOT / f"experiments/results/{STUDY}_control_replay.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
WINNER_RECEIPT = ROOT / f"experiments/results/{STUDY}_winner.json"
SELECTED_CHOICE = ROOT / "data/experiments/v2_weight_edit_selected_choice.json"
V1_RESULT = ROOT / f"experiments/results/{STUDY}_v1.json"
DIRECTION_REPORT = reconstruction.DIAGNOSTICS
DIRECTIONS = reconstruction.DIRECTIONS
TRAIN_DIRECTION_REPORT = OUT / "diagnostics" / "direction_fit_status.json"
TRAIN_DIRECTIONS = OUT / "diagnostics" / "direction_vectors.npz"
CONTROL_PREFLIGHT = ROOT / f"experiments/results/{STUDY}_control_preflight.json"
BASELINE_SCORECARD = ROOT / "experiments/results/original_full_phase_average_scorecard_v2.json"
BASELINE_GUARD = ROOT / "experiments/results/phase_average_guard_v1.json"
CELL_OUT = OUT / "cell_artifacts"
SEEDS = (2026, 2027)
VIEWS = (("clean", (0,)), ("feed_like_v2", (1, 2, 3, 4)))
EDIT_ARMS = {
    "blend_unedited_ha15": ["none", 0.0],
    "projection_a025": ["projection", 0.25],
    "projection_a050": ["projection", 0.50],
    "projection_a100": ["projection", 1.00],
    "update_bias_m025": ["update_bias", -0.25],
    "update_bias_m050": ["update_bias", -0.50],
    "update_bias_p025": ["update_bias", 0.25],
}
METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
           "predicted_spell_count", "reference_spell_count",
           "short_spell_recall_iou_0_5_lt3s")


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text())


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def _fit_path(provider: str, seed: int) -> Path:
    fold = f"leave_out_{upstream.frozen.audit.FOLD[provider]}"
    return reconstruction.OUT / "models" / f"{fold}_{seed}.joblib"


def _teacher_path(provider: str, seed: int) -> Path:
    return upstream.frozen._teacher_path(provider, seed)


def _provider_for_fold(fold: str) -> str:
    suffix = fold.removeprefix("leave_out_")
    matches = [provider for provider, name in upstream.frozen.audit.FOLD.items()
               if name == suffix]
    if len(matches) != 1:
        raise ValueError(f"Cannot map provider fold to unique provider: {fold}")
    return matches[0]


def _source_hashes() -> dict[str, str]:
    paths = {
        Path(__file__), Path(reconstruction.__file__), Path(weight_space_edit.__file__),
        Path(v2_eval.__file__), Path(upstream.__file__), Path(upstream.frozen.__file__),
        Path(upstream.frozen.audit.__file__), Path(upstream.frozen.sn.__file__),
        Path(upstream.frozen.old.__file__), Path(original_full_phase_baseline.__file__),
        ROOT / "experiments/feed_like_v2.py", ROOT / "experiments/feed_spell_push.py",
        ROOT / "experiments/metrics.py", ROOT / "evaluation/possession_metrics.py",
        ROOT / "experiments/results/v2_calibration_transfer.json",
        BASELINE_SCORECARD,
        BASELINE_GUARD,
    }
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def _verify_training_artifacts() -> dict:
    reconstruction.check()
    if not reconstruction.FIT_RECEIPT.is_file():
        raise FileNotFoundError("Complete reconstructed six-fit receipt first")
    fits = _json(reconstruction.FIT_RECEIPT)
    if fits.get("all_six_fits_complete") is not True or len(fits.get("fits", [])) != 6:
        raise RuntimeError("Expected six complete provider-excluded reconstructed fits")
    model_facts = {}
    for fold in reconstruction.recipe.FOLDS:
        for seed in SEEDS:
            path = reconstruction.OUT / "models" / f"{fold}_{seed}.joblib"
            receipt = Path(str(path) + ".json")
            if not path.is_file() or not receipt.is_file():
                raise FileNotFoundError(path)
            record = _json(receipt)
            if (record.get("sha256") != sha(path)
                    or record.get("declaration_sha256") != sha(reconstruction.DECLARATION)):
                raise RuntimeError(f"Reconstruction fit receipt mismatch: {path}")
            model_facts[f"{fold}/{seed}"] = {"path": str(path.relative_to(ROOT)),
                                             "sha256": sha(path),
                                             "receipt_sha256": sha(receipt)}
    return {"fits": fits, "models": model_facts}


def _direction_artifacts() -> dict:
    """Build the frozen per-fit direction from TRAIN only; omit all if incomplete."""
    if not TRAIN_DIRECTION_REPORT.is_file():
        facts = _verify_training_artifacts()
        vectors, rows, failures = {}, [], []
        for fold in reconstruction.recipe.FOLDS:
            records = reconstruction.recipe.training(fold)
            clean_records, v2_records = records[0::4], records[2::4]
            if len(clean_records) != len(v2_records):
                raise RuntimeError(f"Clean/v2 training record counts differ: {fold}")
            for seed in SEEDS:
                key = f"{fold}/{seed}"
                model_path = reconstruction.OUT / "models" / f"{fold}_{seed}.joblib"
                teacher_path = _teacher_path(_provider_for_fold(fold), seed)
                model = joblib.load(model_path)
                teacher = joblib.load(teacher_path)
                diagnostic_records = []
                diagnostic_records.extend(dict(record, view_name="clean") for record in clean_records)
                diagnostic_records.extend(dict(record, view_name="v2_r0") for record in v2_records)
                try:
                    diagnosis = weight_space_edit.diagnose_model(model, teacher, diagnostic_records)
                    direction = np.asarray(diagnosis["direction"], dtype=np.float32)
                    vectors[key] = direction
                    rows.append({"fold": fold, "seed": seed,
                                 "model_sha256": sha(model_path),
                                 "teacher_sha256": sha(teacher_path),
                                 "direction_sha256": diagnosis["direction_sha256"],
                                 "training_matches": sorted({str(r["match"]) for r in clean_records}),
                                 "activation_diagnostic": diagnosis["activation_diagnostic"]})
                except ValueError as exc:
                    reason = str(exc)
                    allowed = ("lacks a delayed or correctly timed transition group" in reason
                               or "no training match contains both delayed and control transitions" in reason
                               or "eligible TRAIN cohort lacks a delayed or correctly timed transition group" in reason)
                    if not allowed:
                        raise
                    failures.append({"fold": fold, "seed": seed, "reason": reason,
                                     "model_sha256": sha(model_path),
                                     "teacher_sha256": sha(teacher_path)})
                    rows.append({"fold": fold, "seed": seed, "direction_available": False,
                                 "reason": reason, "model_sha256": sha(model_path),
                                 "teacher_sha256": sha(teacher_path)})
        available = not failures and len(vectors) == 6
        if available:
            TRAIN_DIRECTIONS.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(TRAIN_DIRECTIONS, **vectors)
        report = {
            "study_id": STUDY,
            "protocol": "clean and first deterministic v2 TRAIN view, direct records with native and swapped orientations; equal-match delayed-versus-correct hidden means; no GAR or development direction data",
            "rows": rows, "failures": failures,
            "projection_edit_eligible_all_six": available,
            "directions_path": str(TRAIN_DIRECTIONS.relative_to(ROOT)) if available else None,
            "directions_sha256": sha(TRAIN_DIRECTIONS) if available else None,
            "fit_receipt_sha256": sha(reconstruction.FIT_RECEIPT),
            "training_only": True, "evaluation_inputs_used": False,
            "organizer_reference_access": False,
        }
        _write_once(TRAIN_DIRECTION_REPORT, report)
    report = _json(TRAIN_DIRECTION_REPORT)
    if report.get("training_only") is not True or report.get("evaluation_inputs_used") is not False:
        raise RuntimeError("Direction diagnostic must be TRAIN-only")
    if report.get("projection_edit_eligible_all_six"):
        if not TRAIN_DIRECTIONS.is_file() or report.get("directions_sha256") != sha(TRAIN_DIRECTIONS):
            raise RuntimeError("Train-only direction vectors do not match their receipt")
        with np.load(TRAIN_DIRECTIONS, allow_pickle=False) as d:
            if set(d.files) != {f"{fold}/{seed}" for fold in reconstruction.recipe.FOLDS for seed in SEEDS}:
                raise RuntimeError("Expected one direction for all six folds and seeds")
    elif TRAIN_DIRECTIONS.exists():
        raise RuntimeError("Uniform projection skip must not leave partial directions")
    return report


def _active_arms() -> dict:
    report = _direction_artifacts()
    if report["projection_edit_eligible_all_six"]:
        return EDIT_ARMS
    return {name: spec for name, spec in EDIT_ARMS.items() if not name.startswith("projection_")}


def freeze() -> dict:
    """Bind all candidate edits and development cells before inference/scoring."""
    if DECLARATION.exists() or CONTROL_RESULT.exists() or RESULT.exists():
        raise FileExistsError("This development study ID is immutable once used")
    facts = _verify_training_artifacts()
    direction_report = _direction_artifacts()
    arms = _active_arms()
    audit = upstream.frozen.audit
    transfer = _json(ROOT / "experiments/results/v2_calibration_transfer.json")
    if transfer.get("highest_eligible_arm") != "equal_50_50_new_full_plus_archived_calibrated":
        raise RuntimeError("Expected the saved 50/50 model-family winner")
    selected_config = dict(transfer["selected_config"]["config"])
    expected_config = {"family": "calibration", "dt": 4, "dead": 3, "ha": 1.5, "ht": 1.0}
    if any(selected_config.get(k) != v for k, v in expected_config.items()):
        raise RuntimeError(f"Saved selected calibration changed: {selected_config}")
    declaration = {
        "study_id": STUDY,
        "status": "fixed_edits_frozen_before_development_scores",
        "question": "Does a train-diagnosed direct edit of the four-view model's H/A representation improve official spell F1 beyond the fixed 50/50 new/calibrated v2-development winner while passing the existing clean/provider guardrails?",
        "arms": arms,
        "edit_definition": {
            "projection": "For each provider-fold/seed, W' = W - alpha*(P_contrast Wv)*v.T on the conditional H/A final output matrix only. v is the normalized equal-match mean contrast between one-sample-late and correctly-timed stable known H/A boundaries from original-provider TRAIN clean and first fixed v2 views, native+swapped orientations. D state, all other tensors, normalization and teacher are unchanged.",
            "update_bias": "Add the fixed delta to the GRU update-gate input-bias slices [H:2H] in forward/reverse directions, both conditional H/A provider-fold states only. D state/output, recurrent bias and all other tensors are unchanged.",
            "direction_artifact": {"available_for_all_six": direction_report["projection_edit_eligible_all_six"],
                                  "path": str(TRAIN_DIRECTIONS.relative_to(ROOT)) if direction_report["projection_edit_eligible_all_six"] else None,
                                  "sha256": sha(TRAIN_DIRECTIONS) if direction_report["projection_edit_eligible_all_six"] else None,
                                  "diagnostic_path": str(TRAIN_DIRECTION_REPORT.relative_to(ROOT)),
                                  "diagnostic_sha256": sha(TRAIN_DIRECTION_REPORT),
                                  "uniform_skip_rule": "if any fold/seed lacks delayed/correct groups or a common contributing match, omit all projection arms and all direction arrays; retain gate edits"},
            "no_optimizer_or_backprop_for_edits": True,
        },
        "candidate_pipeline": "Each edited new four-view full model is blended 50/50 per seed and native phase with the corresponding archived calibrated model; equal seed mean, equal phase mean, then the saved dt=4/dead=3/HA=1.5/HT=1 calibration decoder.",
        "controls": {
            "blend_unedited_ha15": "Reconstructed new four-view full model + archived calibrated model at equal probability weights and the fixed HA=1.5 decoder; matched control for all edits.",
            "deployed_original_full": "Archived original_full two-seed probability pool + five native phases, decoded with unchanged f062 (HA=3).",
            "historical_reference": {"clean": 0.667555533, "v2": 0.583571208,
                                     "eligible_blend_clean": 0.665355, "eligible_blend_v2": 0.5927491913246999},
        },
        "cohort": {
            "matches": [{"provider": p, "match": m} for p, m, _ in audit.MATCHES],
            "primary_views": ["clean", "feed_like_v2"], "feed_realizations": [1, 2, 3, 4],
            "v2_seed_rule": "feed_like_v2_benchmark/{rep}/{match} SHA-256-derived seed, unchanged from the frozen target protocol",
            "aggregation": "replicates within match, matches within provider, providers equally",
            "full_provider_exclusion": "all fitted components and normalizations omit the evaluated provider",
            "v1": "scored only after the primary winner is frozen; two historical replicas retained as a regression diagnostic",
        },
        "decoder": {"blend": selected_config, "deployed_original_full": "f062"},
        "guards": {
            "clean_spell_floor": 0.662555533,
            "clean_provider_macro_and_d_floor": "each provider no worse than original_full baseline - 0.02",
            "v2_spell_floor": 0.573571208,
            "ranking": "highest unrounded v2 spell among arms passing every fixed guard; macro breaks exact ties",
            "no_guard_relaxation": True,
        },
        "budget": {"reconstructed_provider_fold_fits": 6,
            "candidate_arms": list(arms),
                   "primary_cells": 25, "v1_cells_after_selection": 10},
        "provenance": {
            "reconstruction_declaration_sha256": sha(reconstruction.DECLARATION),
            "reconstruction_fit_receipt_sha256": sha(reconstruction.FIT_RECEIPT),
            "model_facts": facts["models"],
            "direction_artifact_sha256": sha(TRAIN_DIRECTIONS) if direction_report["projection_edit_eligible_all_six"] else None,
            "direction_report_sha256": sha(TRAIN_DIRECTION_REPORT),
            "active_arms": arms,
            "source_sha256": _source_hashes(),
        },
        "organizer_reference_access": False,
        "public_scoring": False,
        "GAR_TEST_scoring": False,
        "promotion": False,
        "limitations": "Five exposed development matches and synthetic model-conditioned feed_like_v2 views; reconstructed checkpoints are new fits, not byte-exact restored PR82 artifacts.",
    }
    _write_once(DECLARATION, declaration)
    return declaration


def _require_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze and review the edit declaration before scoring")
    declaration = _json(DECLARATION)
    # Rebind the frozen preparation/cache wrapper to this study's namespace.
    # Its helper implementation is reused, but no prepared data or emissions
    # are written under the older v2 runtime study.
    upstream.frozen.STUDY = STUDY
    upstream.frozen.OUT = OUT
    upstream.frozen.DECLARATION = DECLARATION
    if declaration.get("study_id") != STUDY or declaration.get("arms") != _active_arms():
        raise RuntimeError("Development declaration does not match the fixed arm list")
    if declaration["provenance"]["source_sha256"] != _source_hashes():
        raise RuntimeError("Development source changed after the declaration")
    facts = _verify_training_artifacts()
    if declaration["provenance"]["model_facts"] != facts["models"]:
        raise RuntimeError("Provider-fold model bytes changed after the declaration")
    direction_report = _direction_artifacts()
    if declaration["provenance"]["direction_report_sha256"] != sha(TRAIN_DIRECTION_REPORT):
        raise RuntimeError("Training direction report changed after the declaration")
    if direction_report["projection_edit_eligible_all_six"]:
        if declaration["provenance"]["direction_artifact_sha256"] != sha(TRAIN_DIRECTIONS):
            raise RuntimeError("Training direction artifact changed after the declaration")
    return declaration


def _edited_model(fold: str, seed: int, arm: str):
    base_path = reconstruction.OUT / "models" / f"{fold}_{seed}.joblib"
    model = joblib.load(base_path)
    kind, strength = EDIT_ARMS[arm]
    if kind == "none":
        return model
    edited = dict(model)
    state = model["state"]
    if kind == "projection":
        with np.load(TRAIN_DIRECTIONS, allow_pickle=False) as d:
            direction = np.asarray(d[f"{fold}/{seed}"])
        edited["state"] = weight_space_edit.edit_conditional_ha_state_projection(
            state, direction, alpha=strength)
    elif kind == "update_bias":
        edited["state"] = weight_space_edit.shift_gru_update_bias(
            state, hidden_size=int(model["hidden"]), delta=strength)
    else:
        raise ValueError(f"Unknown edit type: {kind}")
    return edited


def _new_prediction(view: dict, phase_record: dict, provider: str, match: str,
                    view_name: str, rep: int, phase: int, seed: int,
                    model: dict, d_cache: dict) -> np.ndarray:
    fold = f"leave_out_{upstream.frozen.audit.FOLD[provider]}"
    teacher_path = _teacher_path(provider, seed)
    teacher = joblib.load(teacher_path)
    record = upstream.frozen.audit.old.inference_record(phase_record)
    record["names"] = list(record["names"])
    cache_key = (fold, seed, match, view_name, rep, phase)
    # The original model D path is independent of every H/A edit. Compute once
    # per record/model and pass it to the shared conditional H/A inference path.
    if cache_key not in d_cache:
        from experiments import factorized_had as fh
        feature = {key: record[key] for key in ("x", "names", "period")}
        if "timestamps" in record:
            feature["timestamps"] = record["timestamps"]
            feature["fps"] = record["fps"]
        d_teacher = dict(teacher, state=model["d_state"])
        d_cache[cache_key] = fh.predict(feature, d_teacher)
    probability = upstream.frozen.sn.cap.predict(
        record, model, teacher, original_probability=d_cache[cache_key])
    return upstream.frozen.audit.old.upsample(
        probability, record, upstream.frozen.tc.inputs_only(view))


def _cached_calibrated(view, phase_record, provider, match, view_name, rep, phase, seed):
    return upstream.frozen.predict_seed(view, phase_record, "adop", provider, seed)


def _decode(probability: np.ndarray, period: np.ndarray, fps: float, *, deployed=False):
    config = calibration.CONFIG if deployed else {
        "family": "calibration", "dt": 4, "dead": 3, "ha": 1.5, "ht": 1.0,
    }
    return feed_spell_push.decode(probability[None], period, fps, config)


def _score_cell(provider: str, match: str, view_name: str, rep: int,
                arms: list[str], prepared: tuple | None = None) -> dict:
    _require_declaration()
    if prepared is None:
        view, labels, _, phases, period = upstream._prepare_cell(
            provider, match, view_name, rep)
    else:
        view, labels, _, phases, period = prepared
    cell = {"labels": labels, "period": view["period"], "timestamps": view["timestamps"],
            "fps": float(view["fps"])}
    phase_probs = {arm: [] for arm in arms}
    seed_phase_probs = {arm: {seed: [] for seed in SEEDS} for arm in arms}
    deployed_original = []
    pooled_by_arm = {arm: [] for arm in arms}
    for phase, phase_record in enumerate(phases):
        d_cache = {}
        calibrated_cache = {}
        for arm in arms:
            seed_members = []
            for seed in SEEDS:
                fold = f"leave_out_{upstream.frozen.audit.FOLD[provider]}"
                model = _edited_model(fold, seed, arm)
                new = _new_prediction(view, phase_record, provider, match, view_name,
                                      rep, phase, seed, model, d_cache)
                calibrated_key = (seed, phase)
                if calibrated_key not in calibrated_cache:
                    calibrated_cache[calibrated_key] = _cached_calibrated(
                        view, phase_record, provider, match, view_name, rep, phase, seed)
                calibrated = calibrated_cache[calibrated_key]
                seed_value = (new + calibrated) * 0.5
                seed_phase_probs[arm][seed].append(seed_value)
                seed_members.append(seed_value)
            phase_value = np.mean(seed_members, axis=0)
            phase_probs[arm].append(phase_value)
        original_seed_probs = [upstream.frozen.predict_seed(
            view, phase_record, "cand", provider, seed) for seed in SEEDS]
        deployed_original.append(np.mean(original_seed_probs, axis=0))
    result = {}
    seed_metric_rows = {}
    for arm in arms:
        pooled = np.mean(phase_probs[arm], axis=0)
        pooled_by_arm[arm] = pooled
        pred = _decode(pooled, period, cell["fps"])
        result[arm] = upstream.frozen._score_cell(cell, pred)
        seed_metric_rows[arm] = {}
        for seed in SEEDS:
            seed_pooled = np.mean(seed_phase_probs[arm][seed], axis=0)
            seed_pred = _decode(seed_pooled, period, cell["fps"])
            seed_metric_rows[arm][str(seed)] = upstream.frozen._score_cell(cell, seed_pred)
    old_prob = np.mean(deployed_original, axis=0)
    old_pred = _decode(old_prob, period, cell["fps"], deployed=True)
    result["deployed_original_full"] = upstream.frozen._score_cell(cell, old_pred)
    result["_artifact"] = {
        "labels": np.asarray(labels).astype("U1"), "period": np.asarray(period),
        "timestamps": np.asarray(view["timestamps"]), "fps": np.asarray(float(view["fps"])),
        "probabilities": {arm: value for arm, value in pooled_by_arm.items()},
        "seed_probabilities": {f"{arm}__{seed}": np.mean(seed_phase_probs[arm][seed], axis=0)
                               for arm in arms for seed in SEEDS},
        "predictions": {arm: _decode(value, period, cell["fps"]) for arm, value in pooled_by_arm.items()},
        "deployed_original_probabilities": old_prob, "deployed_original_predictions": old_pred,
        "input_sha256": _cell_input_hash(view, labels),
        "graded_frames": np.asarray(int(np.sum(np.asarray(labels) != "X")), dtype=np.int64),
        "frame_count": np.asarray(len(labels), dtype=np.int64),
    }
    result["_seed_metrics"] = seed_metric_rows
    return result


def _cell_input_hash(view: dict, labels: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, value in [("labels", labels), *[(k, view[k]) for k in sorted(view)]]:
        array = np.ascontiguousarray(value)
        h.update(name.encode())
        h.update(str(array.dtype).encode())
        h.update(str(array.shape).encode())
        if array.dtype.kind in "USO":
            h.update(json.dumps(array.tolist(), separators=(",", ":"), default=str).encode())
        else:
            h.update(array.view(np.uint8))
    return h.hexdigest()


def _cell_key(key: tuple) -> str:
    return "__".join(str(part).replace("/", "_") for part in key)


def _cell_paths(stage: str, key: tuple) -> tuple[Path, Path]:
    stem = _cell_key(key)
    root = CELL_OUT / stage
    return root / f"{stem}.npz", root / f"{stem}.json"


def _load_cell(stage: str, key: tuple) -> dict | None:
    array_path, receipt_path = _cell_paths(stage, key)
    if not array_path.exists() and not receipt_path.exists():
        return None
    if not array_path.is_file() or not receipt_path.is_file():
        raise RuntimeError(f"Incomplete immutable cell artifact pair: {array_path}")
    receipt = _json(receipt_path)
    if (receipt.get("study_id") != STUDY
            or receipt.get("declaration_sha256") != sha(DECLARATION)
            or receipt.get("arrays_sha256") != sha(array_path)
            or receipt.get("key") != list(key)):
        raise RuntimeError(f"Cell artifact receipt mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as saved:
        metrics = json.loads(str(saved["metrics_json"].item()))
    return metrics


def _save_cell(stage: str, key: tuple, value: dict) -> dict:
    array_path, receipt_path = _cell_paths(stage, key)
    if array_path.exists() or receipt_path.exists():
        cached = _load_cell(stage, key)
        if cached is None:
            raise RuntimeError("Existing cell artifact could not be loaded")
        return cached
    artifact = value.pop("_artifact", None)
    if artifact is None:
        # Unit-test/mocked scorers may exercise orchestration without model
        # arrays. Real scoring always supplies the replay arrays.
        return value
    arrays = {"labels": artifact["labels"], "period": artifact["period"],
              "timestamps": artifact["timestamps"], "fps": artifact["fps"],
              "indices": np.arange(len(artifact["labels"]), dtype=np.int64),
              "reference_denominator": np.asarray(len(artifact["labels"]), dtype=np.int64),
              "graded_frames": artifact["graded_frames"],
              "frame_count": artifact["frame_count"],
              "deployed_original_probabilities": artifact["deployed_original_probabilities"],
              "deployed_original_predictions": artifact["deployed_original_predictions"],
              "metrics_json": np.asarray(json.dumps(value, sort_keys=True, separators=(",", ":"))),
              "input_sha256": np.asarray(artifact["input_sha256"]),
              "declaration_sha256": np.asarray(sha(DECLARATION))}
    for arm, probability in artifact["probabilities"].items():
        arrays[f"probabilities__{arm}"] = probability
        arrays[f"predictions__{arm}"] = artifact["predictions"][arm]
    for name, probability in artifact["seed_probabilities"].items():
        arrays[f"seed_probabilities__{name}"] = probability
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **arrays)
    _write_once(receipt_path, {"study_id": STUDY, "stage": stage, "key": list(key),
                              "declaration_sha256": sha(DECLARATION),
                              "arrays_sha256": sha(array_path),
                              "input_sha256": artifact["input_sha256"],
                              "arms": sorted(artifact["probabilities"]),
        "independent_replay_arrays": ["labels", "period", "timestamps", "fps", "indices",
                                       "reference_denominator",
                                       "graded_frames", "frame_count",
                                       "probabilities__*", "predictions__*",
                                       "seed_probabilities__*"]})
    return value


def _score_cell_task(task: tuple) -> tuple[str, dict]:
    stage, provider, match, view_name, rep, arms, prepared = task
    key = (provider, match, view_name, rep)
    cached = _load_cell(stage, key)
    if cached is not None:
        return _cell_key(key), cached
    result = _score_cell(provider, match, view_name, rep, arms, prepared=prepared)
    return _cell_key(key), _save_cell(stage, key, result)


def _aggregate(rows: dict, arm: str, view: str) -> dict:
    audit = upstream.frozen.audit
    selected = {(provider, match, str(rep)): row[arm]
                for (provider, match, view_name, rep), row in rows.items()
                if view_name == view}
    scores = {}
    for metric in METRICS:
        values = {key: value[metric] for key, value in selected.items()}
        mean, providers = audit.equal_provider(values)
        scores[metric] = {"equal_provider_mean": mean, "providers": providers}
    return scores


def _aggregate_seed(rows: dict, arm: str, view: str, seed: int) -> dict:
    audit = upstream.frozen.audit
    selected = {(provider, match, str(rep)): row["_seed_metrics"][arm][str(seed)]
                for (provider, match, view_name, rep), row in rows.items()
                if view_name == view and "_seed_metrics" in row}
    if not selected:
        return {}
    scores = {}
    for metric in METRICS:
        values = {key: value[metric] for key, value in selected.items()}
        mean, providers = audit.equal_provider(values)
        scores[metric] = {"equal_provider_mean": mean, "providers": providers}
    return scores


def _evaluate_guards(aggregates: dict, archived_control: dict) -> dict:
    """Apply the fixed common clean/provider/v2 guardrails to every blend arm."""
    results = {}
    clean_control = _baseline_clean_provider_guards()
    if not clean_control["macro_f1"].get("providers"):
        # Keep the pure guard helper testable with a supplied archived-control
        # aggregate; production uses the separately hash-bound saved reference.
        clean_control = archived_control["clean"]
    v2_control = archived_control["feed_like_v2"]
    for arm, views in aggregates.items():
        if arm == "deployed_original_full":
            continue
        checks = {
            "clean_spell_floor": views["clean"]["spell_f1"]["equal_provider_mean"] >= 0.662555533,
            "v2_spell_floor": views["feed_like_v2"]["spell_f1"]["equal_provider_mean"] >= 0.573571208,
        }
        provider_checks = {}
        for provider in ("Sportec", "Metrica", "SkillCorner"):
            provider_checks[provider] = {
                "macro_f1": views["clean"]["macro_f1"]["providers"][provider]
                >= clean_control["macro_f1"]["providers"][provider] - 0.02,
                "d_f1": views["clean"]["d_f1"]["providers"][provider]
                >= clean_control["d_f1"]["providers"][provider] - 0.02,
            }
        checks["clean_provider_floors"] = all(
            metric for values in provider_checks.values() for metric in values.values())
        results[arm] = {"eligible": all(checks.values()), "checks": checks,
                        "providers": provider_checks}
    return results


def _baseline_clean_provider_guards() -> dict:
    guard = _json(BASELINE_GUARD)
    # These saved phase-0 rows are the adopted original_full provider guard
    # reference, not values re-estimated from the reconstructed candidate run.
    source = guard["summaries"]["ensemble"]["phase0"]["clean"]
    return {metric: {"providers": source[metric]["providers"]}
            for metric in ("macro_f1", "d_f1")}


def _archived_scorecard_targets() -> dict:
    scorecard = _json(BASELINE_SCORECARD)
    aggregates = scorecard["original_five_match_aggregates"]
    return {
        "clean": {"spell_f1": float(aggregates["clean"]["spell"]),
                  "macro_f1": float(aggregates["clean"]["macro"])},
        "feed_like_v2": {"spell_f1": float(aggregates["v2"]["spell"]),
                         "macro_f1": float(aggregates["v2"]["macro"])},
    }


def _select_winner(aggregates: dict, guard_results: dict) -> str | None:
    """Choose maximum unrounded v2 spell; macro breaks exact ties."""
    eligible = [arm for arm, result in guard_results.items() if result["eligible"]]
    if not eligible:
        return None
    return max(eligible, key=lambda arm: (
        aggregates[arm]["feed_like_v2"]["spell_f1"]["equal_provider_mean"],
        aggregates[arm]["feed_like_v2"]["macro_f1"]["equal_provider_mean"]))


def _rank_arms(aggregates: dict) -> list[str]:
    arms = [arm for arm in aggregates if arm != "deployed_original_full"]
    return sorted(arms, key=lambda arm: (
        aggregates[arm]["feed_like_v2"]["spell_f1"]["equal_provider_mean"],
        aggregates[arm]["feed_like_v2"]["macro_f1"]["equal_provider_mean"]), reverse=True)


def control_replay(workers: int = 3) -> dict:
    _require_declaration()
    if CONTROL_RESULT.exists():
        raise FileExistsError("Control replay already exists; preserve it")
    if not CONTROL_PREFLIGHT.is_file():
        raise RuntimeError("Run and review the representative controls-only preflight first")
    preflight = _json(CONTROL_PREFLIGHT)
    if (preflight.get("declaration_sha256") != sha(DECLARATION)
            or preflight.get("controls_only") is not True
            or preflight.get("candidate_edit_scores") is not False):
        raise RuntimeError("Representative preflight is not bound to this declaration")
    rows = {}
    jobs = []
    for provider, match, _ in upstream.frozen.audit.MATCHES:
        for view, reps in VIEWS:
            for rep in reps:
                key = (provider, match, view, rep)
                cached = _load_cell("control", key)
                if cached is not None:
                    rows[key] = cached
                else:
                    jobs.append(("control", provider, match, view, rep,
                                 ["blend_unedited_ha15"], None))
    workers = min(max(1, workers), len(jobs))
    if workers:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_score_cell_task, job) for job in jobs]
            for future in as_completed(futures):
                name, result = future.result()
                parts = name.split("__")
                key = (parts[0], parts[1], parts[2], int(parts[3]))
                rows[key] = result
                print("CONTROL_CELL", name, flush=True)
    # First replay the archived deployment control over the full cohort. A
    # mismatch blocks all edit candidates; do not adjust the tolerance.
    expected = _archived_scorecard_targets()
    archived = {view: _aggregate(rows, "deployed_original_full", view) for view, _ in VIEWS}
    replay_checks = {}
    for view, metrics in expected.items():
        name = "clean" if view == "clean" else "feed_like_v2"
        for metric, target in metrics.items():
            actual = archived[name][metric]["equal_provider_mean"]
            replay_checks[f"{view}/{metric}"] = {"expected": target, "actual": actual,
                                                 "abs_error": abs(actual-target)}
    if any(item["abs_error"] > 1e-8 for item in replay_checks.values()):
        raise RuntimeError(f"Archived original_full control replay failed: {replay_checks}")
    blend = {view: _aggregate(rows, "blend_unedited_ha15", view) for view, _ in VIEWS}
    output = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
        "controls_only": True, "cell_count": len(rows), "rows": {
        "|".join(map(str, key)): score for key, score in rows.items()},
        "archived_original_full_replay": archived,
        "archived_original_full_replay_checks": replay_checks,
        "archived_original_full_replay_passed": True,
        "reconstructed_unedited_blend": blend,
        "saved_baseline_scorecard_sha256": sha(BASELINE_SCORECARD),
        "saved_baseline_guard_sha256": sha(BASELINE_GUARD),
        "saved_baseline_provider_spell": _json(BASELINE_SCORECARD)["original_provider_spell"],
        "saved_baseline_provider_guards": _baseline_clean_provider_guards(),
        "historical_eligible_blend": {"clean": 0.665355,
                                      "feed_like_v2": 0.5927491913246999},
        "reconstruction_fit_receipt_sha256": sha(reconstruction.FIT_RECEIPT),
        "no_candidate_edit_scores": True, "organizer_reference_access": False,
        "public_scoring": False}
    # Require all clean and v2 cells before any edit arm scores are permitted.
    if len(rows) != 25:
        raise RuntimeError(f"Control replay is incomplete: {len(rows)}/25 cells")
    _write_once(CONTROL_RESULT, output)
    return output


def control_preflight() -> dict:
    """Run one controls-only cell to catch inference/cache issues cheaply."""
    _require_declaration()
    if CONTROL_PREFLIGHT.exists():
        raise FileExistsError("Representative control preflight already exists")
    provider, match, view, rep = upstream.frozen.audit.MATCHES[0][0], \
        upstream.frozen.audit.MATCHES[0][1], "clean", 0
    key = (provider, match, view, rep)
    result = _score_cell_task(("control_preflight", provider, match, view, rep,
                               ["blend_unedited_ha15"], None))[1]
    check_path, _ = _cell_paths("control_preflight", key)
    output = {"study_id": STUDY, "representative_cell": list(key),
              "artifact_path": str(check_path.relative_to(ROOT)),
              "artifact_sha256": sha(check_path), "metrics": result,
              "controls_only": True, "candidate_edit_scores": False,
              "declaration_sha256": sha(DECLARATION),
              "organizer_reference_access": False, "public_scoring": False}
    _write_once(CONTROL_PREFLIGHT, output)
    return output


def run_primary(workers: int = 3) -> dict:
    declaration = _require_declaration()
    if not CONTROL_RESULT.is_file():
        raise RuntimeError("Complete the controls-only 25-cell replay before edit scoring")
    control = _json(CONTROL_RESULT)
    if (control.get("archived_original_full_replay_passed") is not True
            or control.get("declaration_sha256") != sha(DECLARATION)
            or control.get("cell_count") != 25):
        raise RuntimeError("The complete archived-control replay has not passed for this declaration")
    if RESULT.exists():
        raise FileExistsError("Primary result already exists; preserve it")
    cell_rows = {}
    active_arms = _active_arms()
    audit = upstream.frozen.audit
    jobs = []
    for provider, match, _ in audit.MATCHES:
        for view, reps in VIEWS:
            for rep in reps:
                key = (provider, match, view, rep)
                cached = _load_cell("candidate", key)
                if cached is not None:
                    cell_rows[key] = cached
                else:
                    jobs.append(("candidate", provider, match, view, rep,
                                 list(active_arms), None))
    with ProcessPoolExecutor(max_workers=min(max(1, workers), len(jobs) or 1)) as pool:
        futures = [pool.submit(_score_cell_task, job) for job in jobs]
        for future in as_completed(futures):
            name, result = future.result()
            parts = name.split("__")
            key = (parts[0], parts[1], parts[2], int(parts[3]))
            cell_rows[key] = result
            print("SCORED", key, flush=True)
    if len(cell_rows) != 25:
        raise RuntimeError(f"Incomplete primary cohort: {len(cell_rows)}/25 cells")
    aggregate = {arm: {view: _aggregate(cell_rows, arm, view) for view, _ in VIEWS}
                 for arm in active_arms}
    seed_aggregate = {arm: {view: {str(seed): _aggregate_seed(cell_rows, arm, view, seed)
                                   for seed in SEEDS} for view, _ in VIEWS}
                      for arm in active_arms}
    aggregate["deployed_original_full"] = {
        view: _aggregate(cell_rows, "deployed_original_full", view) for view, _ in VIEWS}
    guards = _evaluate_guards(aggregate, aggregate["deployed_original_full"])
    winner = _select_winner(aggregate, guards)
    ranking = _rank_arms(aggregate)
    result = {
        "study_id": STUDY, "declaration_sha256": sha(DECLARATION),
        "control_replay_sha256": sha(CONTROL_RESULT), "cell_count": len(cell_rows),
        "arms": aggregate,
        "component_seed_diagnostics": seed_aggregate,
        "guard_results": guards,
        "eligible_winner": winner,
        "numerical_v2_ranking": [{"rank": i + 1, "arm": arm,
                                  "spell_f1": aggregate[arm]["feed_like_v2"]["spell_f1"]["equal_provider_mean"],
                                  "macro_f1": aggregate[arm]["feed_like_v2"]["macro_f1"]["equal_provider_mean"],
                                  "eligible": guards[arm]["eligible"]}
                                 for i, arm in enumerate(ranking)],
        "selection_rule": "best eligible v2 spell; macro breaks exact ties; no selection if no arm passes every common guard",
        "cells": {"|".join(map(str, key)): value for key, value in cell_rows.items()},
        "reconstruction_is_byte_exact_restore": False,
        "organizer_reference_access": False, "public_scoring": False,
        "promotion": False,
    }
    _write_once(RESULT, result)
    return result


def freeze_winner() -> dict:
    """Freeze the primary winner before any v1 regression scoring."""
    if WINNER_RECEIPT.exists():
        raise FileExistsError("Winner receipt already exists; preserve it")
    if not RESULT.is_file():
        raise FileNotFoundError("Complete clean/v2 primary scoring before selecting a winner")
    result = _json(RESULT)
    winner = result.get("eligible_winner")
    eligible = winner is not None
    if winner is None:
        candidate_arms = [arm for arm in result["arms"] if arm != "deployed_original_full"]
        winner = max(candidate_arms, key=lambda arm: (
            result["arms"][arm]["feed_like_v2"]["spell_f1"]["equal_provider_mean"],
            result["arms"][arm]["feed_like_v2"]["macro_f1"]["equal_provider_mean"]))
    if eligible and not result["guard_results"][winner]["eligible"]:
        raise RuntimeError("Selected winner does not pass all fixed guards")
    kind, strength = EDIT_ARMS[winner]
    model_facts = _verify_training_artifacts()["models"]
    deployment_edit = {"kind": kind, "strength": float(strength),
                       "alpha": float(strength) if kind == "projection" else None,
                       "delta": float(strength) if kind == "update_bias" else None}
    if kind == "projection":
        deployment_edit["direction_path"] = str(TRAIN_DIRECTIONS.relative_to(ROOT))
        deployment_edit["direction_sha256"] = sha(TRAIN_DIRECTIONS)
    else:
        deployment_edit["direction_path"] = None
        deployment_edit["direction_sha256"] = None
    value = {"study_id": STUDY, "primary_result_sha256": sha(RESULT),
             "winner": winner, "eligible": eligible,
             "selection_note": "If no arm passes all guards, the numerical v2 leader is frozen for reporting only and remains ineligible.",
             "guards": result["guard_results"], "v1_scoring_may_start": True,
             "public_or_gar_scoring": False}
    _write_once(WINNER_RECEIPT, value)
    choice = {
        "study_id": STUDY,
        "development_report_path": str(RESULT.relative_to(ROOT)),
        "development_report_sha256": sha(RESULT),
        "winner_id": winner, "eligible": eligible,
        "edit": deployment_edit,
        "decoder_config": {"family": "calibration", "dt": 4, "dead": 3,
                           "ha": 1.5, "ht": 1.0},
        "blend": {"new_component": 0.5, "calibrated_component": 0.5,
                  "seeds": list(SEEDS), "phase_pooling": "mean probabilities before decoder"},
        "training_fit_receipt_sha256": sha(reconstruction.FIT_RECEIPT),
        "training_models": model_facts,
        "direction_report_sha256": sha(TRAIN_DIRECTION_REPORT),
        "selected_model_provenance_sha256": hashlib.sha256(json.dumps(
            {"model_facts": model_facts, "edit": deployment_edit,
             "fit_receipt_sha256": sha(reconstruction.FIT_RECEIPT)},
            sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "winner_receipt_path": str(WINNER_RECEIPT.relative_to(ROOT)),
        "winner_receipt_sha256": sha(WINNER_RECEIPT),
        "frozen_before_v1_gar_public": True,
        "selection_uses_v1_gar_public": False,
    }
    _write_once(SELECTED_CHOICE, choice)
    return value


def score_v1(workers: int = 3) -> dict:
    """Score the frozen primary winner on historical v1 replicas after selection."""
    if V1_RESULT.exists():
        raise FileExistsError("V1 readout already exists; preserve it")
    if not WINNER_RECEIPT.is_file():
        raise RuntimeError("Freeze the primary winner before v1 scoring")
    winner_receipt = _json(WINNER_RECEIPT)
    primary = _json(RESULT)
    winner = winner_receipt["winner"]
    if winner_receipt["primary_result_sha256"] != sha(RESULT):
        raise RuntimeError("Primary result changed after winner freeze")
    audit = upstream.frozen.audit
    arms = [winner, "blend_unedited_ha15", "deployed_original_full"]
    rows, jobs = {}, []
    for provider, match, _ in audit.MATCHES:
        for rep in (1, 2):
            filename = audit.FILE[match]
            cached = joblib.load(audit.EVAL_INPUTS / f"{filename}_feed_like_corrupt_{rep}.joblib")
            view, labels, record = cached["inputs"], np.asarray(cached["labels"]).astype(str), cached["record"]
            phases = audit.phase_records(view, record)
            period = audit.supervised.model_periods(view)
            prepared = (view, labels, record, phases, period)
            key = (provider, match, "feed_like_v1", rep)
            cached_result = _load_cell("v1", key)
            if cached_result is not None:
                rows[key] = cached_result
            else:
                jobs.append(("v1", provider, match, "feed_like_v1", rep, arms, prepared))
    with ProcessPoolExecutor(max_workers=min(max(1, workers), len(jobs) or 1)) as pool:
        futures = [pool.submit(_score_cell_task, job) for job in jobs]
        for future in as_completed(futures):
            name, result = future.result()
            parts = name.split("__")
            key = (parts[0], parts[1], parts[2], int(parts[3]))
            rows[key] = result
            print("SCORED_V1", key, flush=True)
    if len(rows) != 10:
        raise RuntimeError(f"Incomplete v1 cohort: {len(rows)}/10 cells")
    aggregates = {arm: _aggregate(rows, arm, "feed_like_v1") for arm in arms}
    seed_aggregates = {arm: {str(seed): _aggregate_seed(rows, arm, "feed_like_v1", seed)
                             for seed in SEEDS} for arm in arms}
    result = {
        "study_id": STUDY, "winner_receipt_sha256": sha(WINNER_RECEIPT),
        "primary_result_sha256": sha(RESULT), "winner": winner,
        "winner_eligible": winner_receipt["eligible"], "cells": {
            "|".join(map(str, key)): value for key, value in rows.items()},
        "arms": aggregates, "component_seed_diagnostics": seed_aggregates,
        "replicates": [1, 2],
        "selection_used_v1": False, "organizer_reference_access": False,
        "public_scoring": False, "promotion": False,
    }
    _write_once(V1_RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "control-preflight", "control", "score-primary", "freeze-winner", "score-v1"))
    parser.add_argument("--workers", type=int, default=3)
    action = parser.parse_args().action
    functions = {"freeze": freeze, "control-preflight": control_preflight}
    result = functions[action]() if action in functions else {
        "control": control_replay, "score-primary": run_primary,
        "freeze-winner": freeze_winner, "score-v1": score_v1}[action](workers=args.workers)
    print(json.dumps(result, indent=1, sort_keys=True, default=float), flush=True)


if __name__ == "__main__":
    main()
