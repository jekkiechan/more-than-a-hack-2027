"""Evaluate turnover-supervised CE experts inside the approved frontier ensemble.

The saved baseline-only replay is a hard gate: it validates all 35 cached
frontier cells against the approved result before any candidate inference or
candidate scoring can begin. Candidate scoring replaces only the CE family.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import joblib
import numpy as np

from experiments import frontier_input_prepare_v1 as input_preparation
from experiments import frontier_postprocess_readout as readout
from experiments import frontier_postprocess_v1 as postprocess
from experiments import frontier_recovery_training_v1 as recovery_training
from experiments import frontier_recovery_v1 as recovery
from experiments import sn_dense_train as sn
from experiments import soccernet_extension as old
from experiments import turnover_supervision_train as training
from experiments import v2_spell_balanced_eval as balanced

ROOT = Path(__file__).resolve().parents[1]
STUDY = "turnover_supervision_v1"
ARMS = ("fresh_control", "uniform_turnover", "hard_turnover")
SEEDS = (2026, 2027)
FAMILY_WEIGHTS = {"ce": .5, "calibrated": .375, "predecessor": .125}
BASELINE_RECIPE_ID = "ce0.5_cal0.375_ad0.125_arithmetic_ha3"
HA_COST = 3.0
TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-8
EXPECTED_CELLS = frozenset(postprocess.frontier.expected_cells())
VIEWS = {"clean": [0], "feed_like_v1": [1, 2],
         "feed_like_v2": [1, 2, 3, 4]}

NATIVE_MANIFEST = ROOT / "data/experiments/frontier_recovery_v1/frontier_native_probability_manifest.json"
TRAIN_DECL = training.TRAIN_DECL
FIT_SEAL = training.FIT_SEAL
OUT = ROOT / "data/experiments/turnover_supervision_v1/eval"
DECLARATION = ROOT / "experiments/results/turnover_supervision_eval_declaration.json"
BASELINE_REPLAY = ROOT / "experiments/results/turnover_supervision_eval_baseline_replay.json"
RESULT = ROOT / "experiments/results/turnover_supervision_eval.json"
SMOKE_RECEIPT = OUT / "baseline_phase_smoke.json"


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _cell_parts(key: str) -> tuple[str, str, str, int]:
    fields = key.split("|")
    if len(fields) != 4 or not fields[3].isdigit():
        raise ValueError(f"Invalid cell key: {key}")
    return fields[0], fields[1], fields[2], int(fields[3])


def _validate_cell_inventory(actual) -> None:
    if len(actual) != 35 or set(actual) != EXPECTED_CELLS:
        raise ValueError("Expected the exact frozen 35-cell cohort")


def expected_fit_keys() -> set[tuple[str, str, int]]:
    folds = tuple(recovery_training.FOLDS)
    return {(arm, fold, seed) for arm in ARMS for fold in folds for seed in SEEDS}


def _fit_rows(seal: Mapping) -> dict[tuple[str, str, int], dict]:
    rows = seal.get("fits")
    if not isinstance(rows, list):
        raise ValueError("Training fit seal has no fit inventory")
    index = {}
    for row in rows:
        key = (str(row.get("arm")), str(row.get("fold")), int(row.get("seed", -1)))
        if key in index:
            raise ValueError(f"Duplicate training fit in seal: {key}")
        index[key] = row
    if set(index) != expected_fit_keys() or len(index) != 18:
        raise ValueError("Training fit seal must contain exactly 3 arms × 3 folds × 2 seeds")
    return index


def _frontier_hashes() -> dict[str, str]:
    artifacts = {
        "declaration": postprocess.frontier.DECLARATION,
        "result": postprocess.frontier.RESULT,
        "replay": postprocess.frontier.REPLAY,
        "selection": postprocess.FRONTIER_SELECTION,
    }
    return {f"{name}_sha256": sha256(path) for name, path in artifacts.items()}


def _source_hashes() -> dict[str, str]:
    paths = {
        Path(__file__), Path(training.__file__), Path(input_preparation.__file__),
        Path(recovery.__file__), Path(recovery_training.__file__),
        Path(postprocess.__file__), Path(readout.__file__), Path(balanced.__file__),
        Path(sn.__file__), Path(old.__file__), Path(postprocess.frontier.__file__),
        Path(postprocess.boundary.__file__), Path(postprocess.disagreement.__file__),
        Path(postprocess.balanced.controls.__file__),
        Path(postprocess.balanced.controls.audit.__file__),
        Path(postprocess.balanced.controls.audit.decoder.__file__),
        Path(postprocess.balanced.controls.audit.metrics.__file__),
        ROOT / "evaluation/possession_metrics.py",
        ROOT / "experiments/soccernet_capacity_model.py",
        ROOT / "experiments/learned_temporal.py",
        ROOT / "experiments/refine_features.py",
        ROOT / "experiments/TURNOVER_SUPERVISION_V1_PLAN.md",
        ROOT / "experiments/baseline_manifest.json",
        ROOT / "experiments/model_decision_policy.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_round1_declaration.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_round1.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_replay.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_selection.json",
        ROOT / "experiments/turnover_supervision_readout.py",
        ROOT / "experiments/tests/test_turnover_supervision_eval.py",
        ROOT / "experiments/tests/test_turnover_supervision_readout.py",
        NATIVE_MANIFEST, input_preparation.DECLARATION, input_preparation.MANIFEST,
        TRAIN_DECL, FIT_SEAL, SMOKE_RECEIPT,
    }
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Evaluation source closure is incomplete: {missing}")
    return {str(path.resolve().relative_to(ROOT)): sha256(path)
            for path in sorted(paths)}


def _validate_native_inventory(manifest: Mapping) -> None:
    _validate_cell_inventory(manifest.get("cells", {}))
    if (manifest.get("schema") != "frontier_postprocess_v1_cache_manifest" or
            manifest.get("frontier_study_id") != postprocess.frontier.STUDY or
            manifest.get("expected_cells") is None or
            set(manifest.get("expected_cells", ())) != EXPECTED_CELLS or
            set(manifest.get("cells", {})) != EXPECTED_CELLS or len(EXPECTED_CELLS) != 35):
        raise ValueError("Approved frontier probability manifest does not contain the exact 35-cell cohort")
    postprocess._validate_native_manifest(manifest)
    if manifest.get("baseline_recipe", recovery.BASELINE_RECIPE) != recovery.BASELINE_RECIPE:
        raise ValueError("Native probability manifest does not bind the approved .5/.375/.125 recipe")


def _native_inputs() -> tuple[dict, dict]:
    if not NATIVE_MANIFEST.is_file():
        raise FileNotFoundError(f"Missing approved frontier native manifest: {NATIVE_MANIFEST}")
    manifest = json.loads(NATIVE_MANIFEST.read_text())
    _validate_native_inventory(manifest)
    return manifest, manifest["cells"]


def _training_inputs() -> tuple[dict, dict, dict]:
    if not TRAIN_DECL.is_file() or not FIT_SEAL.is_file():
        raise FileNotFoundError("Freeze evaluation only after all 18 declared fits are sealed")
    declaration = training.check(full=False)
    seal = json.loads(FIT_SEAL.read_text())
    if declaration.get("study_id") != STUDY or seal.get("study_id") != STUDY:
        raise ValueError("Turnover training declaration/seal belongs to another study")
    if declaration.get("arms") != list(ARMS) and declaration.get("arm_ids") != list(ARMS):
        raise ValueError("Turnover training arms differ from the eval protocol")
    if (seal.get("all_18_complete") is not True or seal.get("fit_count") != 18 or
            seal.get("declaration_sha256") != sha256(TRAIN_DECL)):
        raise ValueError("Turnover training fit seal is not complete")
    recovery_training._configure()
    rows = _fit_rows(seal)
    facts = {}
    for (arm, fold, seed), row in rows.items():
        rel = row.get("model_path") or row.get("path")
        expected_path = OUT.parent / "train" / "models" / arm / f"{fold}_{seed}.joblib"
        path = (ROOT / rel).resolve() if rel else expected_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing turnover model: {arm}/{fold}/{seed}")
        digest = sha256(path)
        expected_digest = row.get("model_sha256") or row.get("sha256")
        if expected_digest and digest != expected_digest:
            raise ValueError(f"Training seal model hash mismatch: {arm}/{fold}/{seed}")
        receipt_path = Path(str(path) + ".json")
        if not receipt_path.is_file() or json.loads(receipt_path.read_text()) != row:
            raise ValueError(f"Training fit sidecar differs from seal: {arm}/{fold}/{seed}")
        facts[f"{arm}|{fold}|{seed}"] = {"path": str(path.relative_to(ROOT)),
            "sha256": digest, "receipt_sha256": sha256(receipt_path)}
    return declaration, seal, facts


def freeze() -> dict:
    """Write a one-time declaration after training artifacts are sealed."""
    if any(path.exists() for path in (DECLARATION, BASELINE_REPLAY, RESULT)):
        raise FileExistsError("Preserve existing turnover evaluation evidence")
    native, cells = _native_inputs()
    train_decl, seal, model_facts = _training_inputs()
    hashes = _source_hashes()
    if "experiments/TURNOVER_SUPERVISION_V1_PLAN.md" not in hashes:
        raise ValueError("The frozen turnover plan must be present and hashed")
    prepared_manifest = json.loads(input_preparation.MANIFEST.read_text())
    if set(prepared_manifest.get("cells", {})) != EXPECTED_CELLS:
        raise ValueError("Prepared native-input inventory differs from the exact 35-cell cohort")
    cache_facts = {}
    input_facts = {}
    frontier_result = json.loads(postprocess.frontier.RESULT.read_text())
    for key, fact in cells.items():
        path = ROOT / fact["path"]
        if not path.is_file() or sha256(path) != fact["sha256"]:
            raise ValueError(f"Missing or changed approved frontier cache: {key}")
        saved_identity = fact.get("identity", {})
        if saved_identity.get("cell") != key:
            raise ValueError(f"Native cache identity does not bind {key}")
        cache_facts[key] = {"path": fact["path"], "sha256": fact["sha256"],
                            "phase_count": fact["phase_count"], "identity": saved_identity}
        input_facts[key] = {
            "raw_input_sha256": saved_identity.get("input_sha256"),
            "prepared_input_cache_sha256": saved_identity.get("prepared_input_cache_sha256"),
            "prepared_input_manifest_sha256": sha256(input_preparation.MANIFEST),
            "prepared_input_declaration_sha256": sha256(input_preparation.DECLARATION),
            "prepared_input_cache_receipt_sha256": prepared_manifest["cells"][key]["cache_sha256"],
            "phase_count": fact["phase_count"],
        }
    declaration = {
        "schema": "turnover_supervision_eval_declaration_v1",
        "study_id": STUDY, "status": "frozen_before_candidate_scoring",
        "question": "Does turnover-conditioned CE supervision improve the approved frontier ensemble on the same 35-cell development cohort?",
        "cohort": {"cell_count": 35, "cells": sorted(EXPECTED_CELLS),
                   "views": VIEWS, "providers": sorted({p for p, _, _ in recovery.AUDIT.MATCHES})},
        "training": {"declaration": {"path": str(TRAIN_DECL.relative_to(ROOT)), "sha256": sha256(TRAIN_DECL)},
            "seal": {"path": str(FIT_SEAL.relative_to(ROOT)), "sha256": sha256(FIT_SEAL)},
            "models": model_facts, "arm_ids": list(ARMS), "fit_count": 18},
        "native_manifest": {"path": str(NATIVE_MANIFEST.relative_to(ROOT)), "sha256": sha256(NATIVE_MANIFEST)},
        "baseline_phase_smoke": {"path": str(SMOKE_RECEIPT.relative_to(ROOT)),
            "sha256": sha256(SMOKE_RECEIPT), "max_abs_difference": 0.0,
            "candidate_scoring": False, "label_scoring": False},
        "native_cells": cache_facts, "inputs": input_facts,
        "baseline": {"recipe_id": BASELINE_RECIPE_ID, "family_weights": FAMILY_WEIGHTS,
            "pooling": "within each family: average seeds within native phase, then equal mean across phases; arithmetic family mixture",
            "ha_cost": HA_COST, "decoded_d_source": "saved decoded__predecessor from approved recovery cache",
            "saved_frontier_result_sha256": sha256(postprocess.frontier.RESULT),
            "saved_frontier_selection_sha256": sha256(postprocess.FRONTIER_SELECTION)},
        "candidate_arms": {arm: {"role": "retention_candidate",
            "change": "replace only the CE family with this arm's two-seed mean"} for arm in ARMS},
        "seed_diagnostics": {f"{arm}_seed{seed}": {
            "role": "descriptive_seed_spread_only", "retention_candidate": False,
            "change": "replace only CE with this seed's equal native-phase mean; calibrated and predecessor remain the full two-seed phase pools"}
            for arm in ARMS for seed in SEEDS},
        "scoring": {"baseline_replay_required_before_candidate_inference": True,
            "per_cell_metric_tolerance": TOLERANCE, "aggregate_tolerance": AGGREGATE_TOLERANCE,
            "official_metrics": list(readout.balanced.METRICS),
            "guards": postprocess.GUARDS,
            "paired_control": "compare uniform_turnover and hard_turnover against fresh_control on identical cells and equal-provider aggregates"},
        "artifacts": {"probabilities_and_predictions": "data/experiments/turnover_supervision_v1/eval/cells/{cell}.npz",
            "cell_scores": "experiments/results/turnover_supervision_eval.json"},
        "source_hashes": hashes,
        "organizer_reference_access": False, "gar_test_or_selection": False,
        "promotion": False, "candidate_scoring_performed": False,
    }
    _write_once(DECLARATION, declaration)
    return declaration


def require_declaration() -> tuple[dict, dict, dict]:
    if not DECLARATION.is_file():
        raise FileNotFoundError("Freeze turnover evaluation before replay or candidate scoring")
    dec = json.loads(DECLARATION.read_text())
    if dec.get("study_id") != STUDY or dec.get("status") != "frozen_before_candidate_scoring":
        raise ValueError("Unsupported turnover evaluation declaration")
    if dec.get("source_hashes") != _source_hashes():
        raise ValueError("Evaluation source closure changed after freeze")
    train_decl, seal, models = _training_inputs()
    native, cells = _native_inputs()
    prepared_manifest = json.loads(input_preparation.MANIFEST.read_text())
    if set(dec.get("inputs", {})) != EXPECTED_CELLS or set(prepared_manifest.get("cells", {})) != EXPECTED_CELLS:
        raise ValueError("Evaluation declaration no longer binds all prepared inputs")
    for key, fact in cells.items():
        declared_input = dec["inputs"][key]
        identity = fact.get("identity", {})
        prepared_fact = prepared_manifest["cells"][key]
        if (declared_input.get("raw_input_sha256") != identity.get("input_sha256") or
                declared_input.get("prepared_input_cache_sha256") != identity.get("prepared_input_cache_sha256") or
                declared_input.get("prepared_input_cache_receipt_sha256") != prepared_fact.get("cache_sha256") or
                declared_input.get("prepared_input_manifest_sha256") != sha256(input_preparation.MANIFEST) or
                declared_input.get("prepared_input_declaration_sha256") != sha256(input_preparation.DECLARATION)):
            raise ValueError(f"Prepared input receipt changed after evaluation freeze: {key}")
    if (dec["training"]["declaration"]["sha256"] != sha256(TRAIN_DECL) or
            dec["training"]["seal"]["sha256"] != sha256(FIT_SEAL) or
            dec["training"]["models"] != models or
            dec["native_manifest"]["sha256"] != sha256(NATIVE_MANIFEST) or
            set(dec["native_cells"]) != EXPECTED_CELLS or
            set(cells) != EXPECTED_CELLS or
            dec.get("baseline_phase_smoke", {}).get("sha256") != sha256(SMOKE_RECEIPT)):
        raise ValueError("Evaluation declaration no longer binds model/input inventory")
    return dec, native, cells


def _load_cell(key: str, fact: Mapping, frontier_hashes: Mapping) -> dict:
    return readout._read_cache(ROOT, key, fact, "reconstructed", frontier_hashes)


def replay_baseline() -> dict:
    """Replay all 35 approved outputs and seal the no-candidate-scoring gate."""
    dec, native, facts = require_declaration()
    if BASELINE_REPLAY.exists():
        raise FileExistsError("Preserve the existing baseline replay")
    frontier_hashes = _frontier_hashes()
    saved = json.loads(postprocess.frontier.RESULT.read_text())
    if set(saved.get("cell_metrics", {})) != EXPECTED_CELLS:
        raise ValueError("Approved frontier result lacks all 35 cell scores")
    rows, prediction_hashes, cache_hashes = {}, {}, {}
    for key in sorted(EXPECTED_CELLS):
        fact = facts[key]
        cell = _load_cell(key, fact, frontier_hashes)
        probability = readout._probability_for_weight(cell, .5)
        if not np.allclose(probability, cell["baseline_probability"], atol=TOLERANCE, rtol=0):
            raise ValueError(f"Approved ensemble probability replay failed: {key}")
        prediction = readout._decode(probability, cell["decoded_predecessor"], cell, HA_COST)
        if not np.array_equal(prediction, cell["baseline_prediction"]):
            raise ValueError(f"Approved ensemble prediction replay failed: {key}")
        readout._assert_dx_mask(prediction, cell["decoded_predecessor"], key)
        score = readout._official_score(cell, prediction)
        expected = saved["cell_metrics"][key].get(BASELINE_RECIPE_ID)
        if expected is None:
            raise ValueError(f"Approved result lacks baseline metric row: {key}")
        readout._compare_row(score, expected, f"approved_frontier/{key}")
        rows[key] = score
        prediction_hashes[key] = hashlib.sha256(prediction.tobytes()).hexdigest()
        cache_hashes[key] = fact["sha256"]
    aggregate = readout._aggregate(rows, VIEWS)
    for view, metrics in postprocess.BASELINE_METRICS.items():
        for metric, expected in metrics.items():
            actual = aggregate[view][metric]["equal_provider"]
            if abs(actual - expected) > AGGREGATE_TOLERANCE:
                raise ValueError(f"Approved baseline aggregate replay failed: {view}/{metric}")
    receipt = {"schema": "turnover_supervision_eval_baseline_replay_v1",
        "study_id": STUDY, "passed": True, "candidate_scoring_performed": False,
        "declaration_sha256": sha256(DECLARATION),
        "native_manifest_sha256": sha256(NATIVE_MANIFEST),
        "frontier_artifacts": frontier_hashes, "source_hashes": dec["source_hashes"],
        "cells": len(rows), "cell_metrics": rows, "aggregate": aggregate,
        "prediction_sha256": prediction_hashes, "native_cache_sha256": cache_hashes,
        "tolerance": {"per_cell": TOLERANCE, "aggregate": AGGREGATE_TOLERANCE}}
    _write_once(BASELINE_REPLAY, receipt)
    return receipt


def _verify_baseline_replay(dec: Mapping, facts: Mapping, receipt: Mapping) -> None:
    """Recheck the entire gate receipt before any candidate inference starts."""
    frontier_hashes = _frontier_hashes()
    expected_cache_hashes = {key: facts[key]["sha256"] for key in EXPECTED_CELLS}
    if (receipt.get("schema") != "turnover_supervision_eval_baseline_replay_v1" or
            receipt.get("study_id") != STUDY or receipt.get("passed") is not True or
            receipt.get("candidate_scoring_performed") is not False or
            receipt.get("declaration_sha256") != sha256(DECLARATION) or
            receipt.get("native_manifest_sha256") != sha256(NATIVE_MANIFEST) or
            receipt.get("source_hashes") != dec["source_hashes"] or
            receipt.get("frontier_artifacts") != frontier_hashes or
            receipt.get("native_cache_sha256") != expected_cache_hashes or
            receipt.get("tolerance") != {"per_cell": TOLERANCE,
                                          "aggregate": AGGREGATE_TOLERANCE} or
            set(receipt.get("cell_metrics", {})) != EXPECTED_CELLS or
            set(receipt.get("prediction_sha256", {})) != EXPECTED_CELLS):
        raise ValueError("Baseline replay receipt is incomplete or differs from the frozen inputs")
    saved = json.loads(postprocess.frontier.RESULT.read_text())
    rows, prediction_hashes = {}, {}
    for key in sorted(EXPECTED_CELLS):
        cell = _load_cell(key, facts[key], frontier_hashes)
        probability = readout._probability_for_weight(cell, .5)
        if not np.allclose(probability, cell["baseline_probability"], atol=TOLERANCE, rtol=0):
            raise ValueError(f"Baseline probability changed after replay: {key}")
        prediction = readout._decode(probability, cell["decoded_predecessor"], cell, HA_COST)
        if not np.array_equal(prediction, cell["baseline_prediction"]):
            raise ValueError(f"Baseline prediction changed after replay: {key}")
        score = readout._official_score(cell, prediction)
        readout._compare_row(score, receipt["cell_metrics"][key], f"replay_receipt/{key}")
        readout._compare_row(score, saved["cell_metrics"][key][BASELINE_RECIPE_ID],
                             f"approved_frontier/{key}")
        digest = hashlib.sha256(prediction.tobytes()).hexdigest()
        if receipt["prediction_sha256"][key] != digest:
            raise ValueError(f"Baseline prediction hash changed after replay: {key}")
        rows[key], prediction_hashes[key] = score, digest
    aggregate = readout._aggregate(rows, VIEWS)
    readout._compare_aggregate(aggregate, receipt["aggregate"], "replay_receipt_aggregate")


def _candidate_probability(phases: Mapping[int, list[np.ndarray]], *, seed: int | None = None) -> np.ndarray:
    """Pool CE in the declared seed/phase order; seed=None means full two-seed mean."""
    if set(phases) != set(SEEDS) or len(phases[SEEDS[0]]) != len(phases[SEEDS[1]]):
        raise ValueError("Candidate CE requires paired seeds and matching native phases")
    phase_count = len(phases[SEEDS[0]])
    if phase_count not in (2, 5):
        raise ValueError("Unexpected native phase count")
    if seed is not None and seed not in SEEDS:
        raise ValueError("Unknown seed diagnostic")
    pooled = []
    for phase in range(phase_count):
        left = balanced.validate_probabilities(phases[2026][phase], "CE seed 2026")
        right = balanced.validate_probabilities(phases[2027][phase], "CE seed 2027")
        if left.shape != right.shape:
            raise ValueError("Paired CE probability shapes differ")
        pooled.append(left if seed == 2026 else right if seed == 2027 else (left + right) / 2)
    return balanced.validate_probabilities(np.mean(pooled, axis=0), "pooled turnover CE")


def candidate_mixture(ce: np.ndarray, calibrated: np.ndarray,
                      predecessor: np.ndarray) -> np.ndarray:
    """Apply the current approved family weights to already pooled emissions."""
    return balanced.recipe_probability({"ce": ce, "calibrated": calibrated,
        "adopted": predecessor}, .5, .375, "arithmetic")


def _predict_phase(phase_record: Mapping, model: dict, teacher: dict,
                   inputs: Mapping) -> np.ndarray:
    """Run CE inference on a target-free phase record and restore native rows."""
    record = old.inference_record(dict(phase_record))
    record["names"] = list(record["names"])
    raw = sn.predict(record, model, teacher)
    return old.upsample(raw, record, dict(inputs))


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def _score_cell(key: str, native_fact: Mapping,
                model_facts: Mapping) -> tuple[str, dict]:
    recovery_training._configure()
    prepared = input_preparation.load(key)
    prepared_manifest = json.loads(input_preparation.MANIFEST.read_text())
    prepared["cache_sha256"] = prepared_manifest["cells"][key]["cache_sha256"]
    baseline_cell = _load_cell(key, native_fact, _frontier_hashes())
    provider, match, view, rep = _cell_parts(key)
    fold = f"leave_out_{recovery.AUDIT.FOLD[provider]}"
    inputs, labels, phases, model_period = (prepared["inputs"], prepared["labels"],
        prepared["phases"], prepared["model_period"])
    if not np.array_equal(labels, baseline_cell["labels"]):
        raise ValueError(f"Prepared labels differ from baseline cache: {key}")
    for field in ("period", "timestamps", "model_period"):
        value = model_period if field == "model_period" else inputs[field]
        if not np.array_equal(np.asarray(value), np.asarray(baseline_cell[field])):
            raise ValueError(f"Prepared native metadata differs from baseline cache: {key}/{field}")
    if len(phases) != native_fact["phase_count"]:
        raise ValueError(f"Prepared phase count differs from baseline cache: {key}")

    teachers = {seed: joblib.load(recovery_training.recipe.rf.sc.initial_path(fold, seed))
                for seed in SEEDS}
    all_outputs: dict[str, np.ndarray] = {}
    diagnostics: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    base_components = baseline_cell["components"]
    calibrated = readout._pool_family(base_components["calibrated"], .5)
    predecessor = readout._pool_family(base_components["predecessor"], .5)
    for arm in ARMS:
        by_seed = {seed: [] for seed in SEEDS}
        for seed in SEEDS:
            fit = model_facts[f"{arm}|{fold}|{seed}"]
            model = joblib.load(ROOT / fit["path"])
            hashes[f"{arm}_seed{seed}_model_sha256"] = fit["sha256"]
            for phase in phases:
                by_seed[seed].append(_predict_phase(phase, model, teachers[seed], inputs))
        pooled_ce = _candidate_probability(by_seed)
        probability = candidate_mixture(pooled_ce, calibrated, predecessor)
        prediction = readout._decode(probability, baseline_cell["decoded_predecessor"],
                                     baseline_cell, HA_COST)
        all_outputs[f"prob__{arm}"] = probability.astype(np.float64)
        all_outputs[f"pred__{arm}"] = prediction.astype("U1")
        all_outputs[f"ce__{arm}"] = pooled_ce.astype(np.float64)
        for seed in SEEDS:
            for phase_idx, phase_probability in enumerate(by_seed[seed]):
                all_outputs[f"prob__ce__{arm}__seed{seed}__phase{phase_idx}"] = np.asarray(
                    phase_probability, dtype=np.float64)
        for seed in SEEDS:
            seed_ce = _candidate_probability(by_seed, seed=seed)
            seed_probability = candidate_mixture(seed_ce, calibrated, predecessor)
            seed_prediction = readout._decode(seed_probability,
                baseline_cell["decoded_predecessor"], baseline_cell, HA_COST)
            diag = f"{arm}_seed{seed}"
            diagnostics[f"prob__{diag}"] = seed_probability.astype(np.float64)
            diagnostics[f"pred__{diag}"] = seed_prediction.astype("U1")
    identity = {"cell": key, "provider": provider, "match": match, "view": view,
        "rep": rep, "native_cache_sha256": native_fact["sha256"],
        "prepared_input_cache_sha256": prepared["cache_sha256"],
        "training_declaration_sha256": sha256(TRAIN_DECL),
        "training_seal_sha256": sha256(FIT_SEAL), **hashes}
    arrays = {**all_outputs, **diagnostics, **{
        f"identity__{name}": np.asarray(str(value)) for name, value in identity.items()}}
    path = OUT / "cells" / f"{key.replace('|', '__')}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Preserve existing candidate cache: {path}")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)
    return key, {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                 "identity": identity}


def score(workers: int = 2) -> dict:
    """Infer and score declared candidates after a successful full baseline replay."""
    if workers not in (1, 2):
        raise ValueError("Use one or two inference workers")
    dec, native, facts = require_declaration()
    if RESULT.exists():
        raise FileExistsError("Preserve the existing turnover evaluation result")
    if not BASELINE_REPLAY.is_file():
        raise ValueError("A complete baseline-only replay is required before candidate scoring")
    replay = json.loads(BASELINE_REPLAY.read_text())
    if (replay.get("passed") is not True or replay.get("candidate_scoring_performed") is not False or
            replay.get("declaration_sha256") != sha256(DECLARATION) or
            set(replay.get("cell_metrics", {})) != EXPECTED_CELLS):
        raise ValueError("Baseline replay receipt is missing, incomplete, or bound to another declaration")
    _verify_baseline_replay(dec, facts, replay)
    _, _, model_facts = _training_inputs()
    frontier_hashes = _frontier_hashes()
    input_manifest = json.loads(input_preparation.MANIFEST.read_text())
    if set(input_manifest.get("cells", {})) != EXPECTED_CELLS:
        raise ValueError("Prepared input inventory changed after evaluation freeze")
    # Baseline replay is deliberately complete before this candidate inference loop.
    tasks = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        for key in sorted(EXPECTED_CELLS):
            tasks[pool.submit(_score_cell, key, facts[key], model_facts)] = key
        output_facts = {}
        for index, future in enumerate(as_completed(tasks), 1):
            key = tasks.pop(future)
            output_facts[key] = future.result()[1]
            print(f"SCORED TURNOVER CELL {index}/35 {key}", flush=True)
    cell_metrics = {arm: {} for arm in ARMS}
    seed_metrics = {f"{arm}_seed{seed}": {} for arm in ARMS for seed in SEEDS}
    prediction_hashes = {name: {} for name in [*ARMS,
        *(f"{arm}_seed{seed}" for arm in ARMS for seed in SEEDS)]}
    for key in sorted(EXPECTED_CELLS):
        path = ROOT / output_facts[key]["path"]
        baseline_cell = _load_cell(key, facts[key], frontier_hashes)
        with np.load(path, allow_pickle=False) as archive:
            for name in prediction_hashes:
                prediction = readout._read_labels(archive[f"pred__{name}"], name)
                if prediction.shape != baseline_cell["labels"].shape:
                    raise ValueError(f"Candidate prediction shape mismatch: {key}/{name}")
                readout._assert_dx_mask(prediction, baseline_cell["baseline_prediction"], f"{key}/{name}")
                row = readout._official_score(baseline_cell, prediction)
                readout._compare_row({k: row[k] for k in ("graded_frames", "reference_spell_count",
                    "short_reference_spell_count")}, {k: replay["cell_metrics"][key][k]
                    for k in ("graded_frames", "reference_spell_count", "short_reference_spell_count")},
                    f"grading_invariant/{key}/{name}")
                if name in ARMS:
                    cell_metrics[name][key] = row
                else:
                    seed_metrics[name][key] = row
                prediction_hashes[name][key] = hashlib.sha256(prediction.tobytes()).hexdigest()
    aggregates = {arm: readout._aggregate(cell_metrics[arm], VIEWS) for arm in ARMS}
    seed_aggregates = {arm: readout._aggregate(seed_metrics[arm], VIEWS) for arm in seed_metrics}
    baseline_aggregate = replay["aggregate"]
    guards = {arm: readout._independent_guards(aggregates[arm], baseline_aggregate) for arm in ARMS}
    paired_fresh_control = {}
    for arm in ("uniform_turnover", "hard_turnover"):
        paired_fresh_control[arm] = {
            view: {metric: aggregates[arm][view][metric]["equal_provider"] -
                   aggregates["fresh_control"][view][metric]["equal_provider"]
                   for metric in readout.balanced.METRICS}
            for view in VIEWS}
    result = {"schema": "turnover_supervision_eval_result_v1", "study_id": STUDY,
        "passed": True, "declaration_sha256": sha256(DECLARATION),
        "baseline_replay_sha256": sha256(BASELINE_REPLAY),
        "native_manifest_sha256": sha256(NATIVE_MANIFEST),
        "candidate_arms": list(ARMS), "seed_diagnostic_arms": list(seed_metrics),
        "cells": output_facts, "cell_metrics": cell_metrics,
        "seed_diagnostic_metrics": seed_metrics,
        "aggregate": aggregates, "seed_diagnostic_aggregate": seed_aggregates,
        "baseline_aggregate": baseline_aggregate, "current_baseline_guards": guards,
        "paired_fresh_control_deltas": paired_fresh_control,
        "prediction_sha256": prediction_hashes,
        "organizer_reference_access": False, "promotion": False}
    _write_once(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay", "score"))
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    if args.action == "freeze":
        value = freeze()
    elif args.action == "replay":
        value = replay_baseline()
    else:
        value = score(args.workers)
    print(json.dumps(value, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
