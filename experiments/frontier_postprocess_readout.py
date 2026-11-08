"""Independent replay of the approved frontier postprocessing experiment.

The verifier reads sealed probabilities and saved predictions only. It rebuilds
each declared arm, scores predictions with the official spell-MBR metric
implementation, and recomputes match/provider aggregates and retention guards.
It performs no model inference or fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import boundary_relocation_v1 as boundary
from experiments import frontier_disagreement_v1 as disagreement
from experiments import frontier_postprocess_v1 as runner
from experiments import spell_mbr_run as official
from experiments import v2_spell_balanced_eval as balanced

ROOT = runner.ROOT
SEEDS = (2026, 2027)
WEIGHTS = (0.25, 0.50, 0.75)
BOUNDARY_ARMS = ("shift_0.2", "shift_0.4", "shift_0.8")
HA_COST_ARMS = ("ha_cost_2", "ha_cost_4")
GATE_ARMS = ("disagreement_lambda_0.5", "disagreement_lambda_1.0")
TOL = 1e-10
FAMILY_WEIGHTS = {"ce": .5, "calibrated": .375, "predecessor": .125}
BASELINE_RECIPE_ID = "ce0.5_cal0.375_ad0.125_arithmetic_ha3"
BASELINE_METRICS = {
    "clean": {"spell_f1": 0.6701365499557103, "macro_f1": 0.8718643812171453},
    "feed_like_v1": {"spell_f1": 0.532128306931943, "macro_f1": 0.8261543004095224},
    "feed_like_v2": {"spell_f1": 0.5936187198896373, "macro_f1": 0.8522037287846299},
}
VIEW_REPLICATES = {"clean": [0], "feed_like_v1": [1, 2],
                   "feed_like_v2": [1, 2, 3, 4]}

PREDICTION_KEYS = {
    "seed2026_0.25": "pred__seed2026_weight_0.25",
    "seed2026_0.50": None,
    "seed2026_0.75": "pred__seed2026_weight_0.75",
    "boundary_shift_0.2": "pred__boundary_shift_0.2",
    "boundary_shift_0.4": "pred__boundary_shift_0.4",
    "boundary_shift_0.8": "pred__boundary_shift_0.8",
    "ha_cost_2": "pred__ha_cost_2",
    "ha_cost_4": "pred__ha_cost_4",
    "disagreement_lambda_0.5": "pred__disagreement_lambda_0.5",
    "disagreement_lambda_1.0": "pred__disagreement_lambda_1.0",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_paths(declaration: Mapping, root: Path) -> dict[str, Path]:
    artifacts = declaration.get("frontier_artifacts", {})
    roles = {"declaration", "result", "replay", "selection"}
    if set(artifacts) != roles:
        raise ValueError("Declaration must bind four selected frontier artifacts")
    return {f"{role}_sha256": _resolve(root, facts["path"])
            for role, facts in artifacts.items()}


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _check_source_hashes(inventory: Mapping[str, str], root: Path) -> None:
    if not inventory:
        raise ValueError("Source hash inventory is empty")
    for relative, expected in inventory.items():
        path = _resolve(root, relative)
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed frozen source: {relative}")


def _frontier_artifacts(manifest: Mapping, declaration: Mapping, root: Path) -> dict[str, str]:
    paths = _artifact_paths(declaration, root)
    actual = {name: sha256(path) for name, path in paths.items() if path.is_file()}
    if len(actual) != 4:
        raise ValueError("Selected frontier artifact is missing")
    for role, fact in declaration["frontier_artifacts"].items():
        if fact.get("sha256") != actual.get(f"{role}_sha256"):
            raise ValueError(f"Selected frontier artifact hash changed: {role}")
    manifest_facts = manifest.get("frontier_artifacts", {})
    if manifest_facts and manifest_facts != declaration["frontier_artifacts"]:
        raise ValueError("Native manifest and declaration frontier receipts differ")
    return actual


def _validate_manifest(manifest: Mapping, declaration: Mapping,
                       manifest_path: Path, root: Path) -> dict[str, str]:
    if manifest.get("schema") != "frontier_postprocess_v1_cache_manifest":
        raise ValueError("Unsupported frontier postprocess cache manifest")
    status = manifest.get("source_status")
    if status not in ("exact_restored", "reconstructed"):
        raise ValueError("Cache provenance must be exact_restored or reconstructed")
    if status == "reconstructed" and not isinstance(manifest.get("reconstruction_provenance"), dict):
        raise ValueError("Reconstructed source needs a structured provenance receipt")
    if (manifest.get("frontier_study_id") != runner.frontier.STUDY or
            declaration.get("frontier_study_id") != runner.frontier.STUDY):
        raise ValueError("Cache is not bound to the selected frontier study")
    recipe = {"recipe_id": BASELINE_RECIPE_ID, "ce_weight": .5,
              "calibrated_weight": .375, "adopted_weight": .125,
              "pooling": "arithmetic", "ha_cost": 3.0,
              "training_arm": "base", "training_stage": 1024}
    if manifest.get("baseline_recipe") != recipe or manifest.get("family_weights") != FAMILY_WEIGHTS:
        raise ValueError("Manifest does not freeze the approved frontier mixture")
    expected_tolerances = {"per_cell_metric": TOL, "aggregate_metric": 1e-8}
    if (manifest.get("arm_ids") != list(runner.ARM_IDS) or
            manifest.get("arm_settings") != runner.arm_settings() or
            manifest.get("guards") != runner.GUARDS or
            manifest.get("ranking") != runner.RANKING or
            manifest.get("comparison_tolerances") != expected_tolerances):
        raise ValueError("Evaluation manifest differs from the frozen arm/guard protocol")
    if manifest.get("seeds") != list(SEEDS):
        raise ValueError("Manifest must bind paired seeds 2026 and 2027")
    expected = set(runner.EXPECTED_CELLS)
    if (set(manifest.get("expected_cells", ())) != expected or
            len(manifest.get("expected_cells", ())) != 35 or
            set(manifest.get("cells", {})) != expected):
        raise ValueError("Manifest must enumerate the exact 35-cell frontier cohort")
    if manifest.get("view_replicates", VIEW_REPLICATES) != VIEW_REPLICATES:
        raise ValueError("Manifest view/replicate map differs from the frozen cohort")
    if manifest.get("baseline_metrics") != BASELINE_METRICS:
        raise ValueError("Manifest baseline metrics differ from the selected frontier result")
    if not isinstance(manifest.get("source_hashes"), dict) or not manifest["source_hashes"]:
        raise ValueError("Manifest must bind source code and declarations")
    required_sources = {"experiments/frontier_postprocess_v1.py",
        "experiments/frontier_disagreement_v1.py", "experiments/v2_spell_balanced_eval.py",
        "experiments/v2_spell_frontier_eval.py", "experiments/boundary_relocation_v1.py",
        "experiments/spell_mbr_run.py", "experiments/frontier_postprocess_readout.py",
        "evaluation/possession_metrics.py"}
    if not required_sources.issubset(manifest["source_hashes"]):
        raise ValueError(f"Source inventory omits required verifier inputs: {sorted(required_sources - set(manifest['source_hashes']))}")
    for key in expected:
        fact = manifest["cells"][key]
        if not fact.get("path") or not isinstance(fact.get("sha256"), str):
            raise ValueError(f"Cache path/hash missing for {key}")
        if fact.get("phase_count") not in (2, 5):
            raise ValueError(f"Unsupported native phase count for {key}")
        identity = fact.get("identity")
        if (not isinstance(identity, dict) or identity.get("cell") != key or
                identity.get("source_status") != status):
            raise ValueError(f"Cell provenance identity mismatch for {key}")
    native_ref = manifest.get("frozen_native_manifest")
    if not isinstance(native_ref, dict) or not native_ref.get("path") or not native_ref.get("sha256"):
        raise ValueError("Evaluation manifest must bind the immutable native cache manifest")
    native_path = _resolve(root, native_ref["path"])
    if not native_path.is_relative_to(root) or not native_path.is_file() or sha256(native_path) != native_ref["sha256"]:
        raise ValueError("Bound native probability manifest is missing or changed")
    native = json.loads(native_path.read_text())
    for field in ("cells", "expected_cells", "source_status", "frontier_artifacts",
                  "baseline_recipe", "family_weights", "seeds", "baseline_metrics"):
        if manifest.get(field) != native.get(field):
            raise ValueError(f"Evaluation manifest differs from immutable native manifest: {field}")
    if manifest.get("preparation_source_hashes") != native.get("source_hashes"):
        raise ValueError("Native preparation source inventory differs from evaluation manifest")
    _check_source_hashes(native.get("source_hashes", {}), root)
    if (declaration.get("study_id") != runner.STUDY or
            declaration.get("status") != "frozen_before_candidate_scoring" or
            declaration.get("source_status") != status or
            declaration.get("evaluation_manifest_sha256") != sha256(manifest_path) or
            declaration.get("native_manifest") != manifest.get("frozen_native_manifest")):
        raise ValueError("Frozen declaration does not bind this evaluation/native manifest")
    if (declaration.get("source_hashes") != manifest.get("source_hashes") or
            declaration.get("frontier_artifacts") != manifest.get("frontier_artifacts") or
            declaration.get("baseline_recipe") != recipe or
            declaration.get("family_weights") != FAMILY_WEIGHTS or
            declaration.get("reconstruction_provenance") != manifest.get("reconstruction_provenance") or
            declaration.get("arm_ids") != list(runner.ARM_IDS) or
            declaration.get("arm_settings") != runner.arm_settings() or
            declaration.get("guards") != runner.GUARDS or
            declaration.get("ranking") != runner.RANKING or
            declaration.get("comparison_tolerances") != manifest.get("comparison_tolerances") or
            declaration.get("expected_baseline_metrics") != BASELINE_METRICS):
        raise ValueError("Declaration source closure, arms, guards, or tolerances changed")
    cohort = declaration.get("cohort", {})
    if (cohort.get("cell_count") != 35 or set(cohort.get("cells", ())) != set(expected) or
            cohort.get("seeds") != list(SEEDS) or
            declaration.get("organizer_reference_access") is not False or
            declaration.get("promotion") is not False):
        raise ValueError("Declaration cohort or no-public/no-promotion contract is invalid")
    frontier_hashes = _frontier_artifacts(manifest, declaration, root)
    paths = _artifact_paths(declaration, root)
    declaration = json.loads(paths["declaration_sha256"].read_text())
    replay = json.loads(paths["replay_sha256"].read_text())
    selection = json.loads(paths["selection_sha256"].read_text())
    if declaration.get("study_id") != runner.frontier.STUDY:
        raise ValueError("Selected frontier declaration belongs to another study")
    if replay.get("study_id") != runner.frontier.STUDY or replay.get("passed") is not True or replay.get("candidate_scoring") is not False:
        raise ValueError("Selected frontier control replay is missing or failed")
    if (selection.get("pipeline_id") != "base__" + BASELINE_RECIPE_ID or
            selection.get("recipe", {}).get("recipe_id") != BASELINE_RECIPE_ID or
            selection.get("training_arm") != "base" or selection.get("training_stage") != 1024):
        raise ValueError("Selected frontier artifact does not identify the approved baseline")
    _check_source_hashes(manifest["source_hashes"], root)
    return frontier_hashes


def _read_labels(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind in "iu":
        if not np.isin(array, (0, 1, 2, 3)).all():
            raise ValueError(f"{name} contains an invalid encoded state")
        array = np.asarray(("H", "A", "D", "X"), dtype="U1")[array]
    else:
        array = array.astype("U1", copy=False)
    if not np.isin(array, ("H", "A", "D", "X")).all():
        raise ValueError(f"{name} contains an invalid state")
    return array


def _cell_key_parts(key: str) -> tuple[str, str, str, int]:
    fields = key.split("|")
    if len(fields) != 4:
        raise ValueError(f"Invalid frozen cell key: {key}")
    provider, match, view, rep = fields
    if not rep.isdigit():
        raise ValueError(f"Invalid cell replicate: {key}")
    return provider, match, view, int(rep)


def _read_cache(root: Path, key: str, fact: Mapping,
                source_status: str, frontier_hashes: Mapping[str, str]) -> dict:
    path = _resolve(root, fact["path"])
    if not path.is_relative_to(root) or not path.is_file() or sha256(path) != fact["sha256"]:
        raise ValueError(f"Missing, unsafe, or changed probability cache: {key}")
    phase_count = fact.get("phase_count")
    if phase_count not in (2, 5):
        raise ValueError(f"Unsupported phase count: {key}")
    with np.load(path, allow_pickle=False) as saved:
        identity = {name.removeprefix("identity__"): str(saved[name].item())
                    for name in saved.files if name.startswith("identity__")}
        expected_identity = fact.get("identity", {})
        if not expected_identity or identity != {name: str(value)
                                                 for name, value in expected_identity.items()}:
            raise ValueError(f"Cache identity differs from manifest: {key}")
        for name, value in identity.items():
            if name.endswith("sha256") and (len(value) != 64 or
                    any(char not in "0123456789abcdef" for char in value.lower())):
                raise ValueError(f"Invalid cache identity digest: {key}/{name}")
        must_bind = {
            "cell": key, "source_status": source_status,
            "frontier_study_id": runner.frontier.STUDY,
            "frontier_declaration_sha256": frontier_hashes["declaration_sha256"],
            "frontier_result_sha256": frontier_hashes["result_sha256"],
            "frontier_replay_sha256": frontier_hashes["replay_sha256"],
            "selection_sha256": frontier_hashes["selection_sha256"],
        }
        if any(identity.get(name) != str(value) for name, value in must_bind.items()):
            raise ValueError(f"Cache does not bind current frontier identity: {key}")
        components = {family: {seed: [] for seed in SEEDS}
                      for family in ("ce", "calibrated", "predecessor")}
        for family, seeds in components.items():
            for seed in seeds:
                for phase in range(phase_count):
                    name = f"prob__{family}__seed{seed}__phase{phase}"
                    if name not in saved.files:
                        raise ValueError(f"Missing paired phase probability {name}: {key}")
                    components[family][seed].append(
                        balanced.validate_probabilities(saved[name], name))
        cell = {name: saved[name].copy() for name in
                ("labels", "period", "timestamps", "model_period")}
        cell["fps"] = float(saved["fps"])
        cell["baseline_probability"] = np.asarray(saved["prob__approved_baseline"], dtype=np.float64).copy()
        cell["baseline_prediction"] = _read_labels(saved["decoded__approved_baseline"],
                                                   "approved frontier prediction")
        cell["predecessor_prediction"] = _read_labels(saved["decoded__predecessor"],
                                                       "decoded predecessor")
    n = len(cell["labels"])
    if (n == 0 or any(np.asarray(cell[name]).shape != (n,) for name in
                      ("period", "timestamps", "model_period", "baseline_prediction",
                       "predecessor_prediction")) or
            cell["baseline_probability"].shape != (n, 2) or
            not np.isfinite(cell["baseline_probability"]).all() or
            (cell["baseline_probability"] < 0).any() or
            not np.allclose(cell["baseline_probability"].sum(axis=1), 1., atol=1e-6, rtol=0) or
            not np.isin(np.asarray(cell["labels"]).astype(str), ("H", "A", "D", "X")).all() or
            not np.isfinite(np.asarray(cell["timestamps"], dtype=float)).all() or
            not math.isfinite(cell["fps"]) or cell["fps"] <= 0):
        raise ValueError(f"Labels, masks, timebase and probabilities do not align: {key}")
    if identity.get("phase_count") is not None and int(identity["phase_count"]) != phase_count:
        raise ValueError(f"Native phase count differs from cache identity: {key}")
    for family, by_seed in components.items():
        for seed, phases in by_seed.items():
            if len(phases) != phase_count or any(p.shape != (n, 3) for p in phases):
                raise ValueError(f"Phase probability alignment failure: {key}/{family}/{seed}")
    delta = np.diff(np.asarray(cell["timestamps"], dtype=float))
    base_period = np.asarray(cell["period"])
    model_period = np.asarray(cell["model_period"])
    breaks = (base_period[1:] != base_period[:-1]) | (delta <= 0) | (delta > 1.5 / cell["fps"])
    if np.any(breaks & (model_period[1:] == model_period[:-1])):
        raise ValueError(f"Model period failed to preserve a native time break: {key}")
    # Independently rebuild each family's paired-seed mean using the current
    # control weight and verify the archived predecessor's decoded mask.
    adopted = _pool_family(components["predecessor"], .5)
    predecessor = balanced.controls.audit.decode(adopted, cell["model_period"],
        cell["fps"], balanced.controls.audit.CONFIG)
    if not np.array_equal(predecessor, cell["predecessor_prediction"]):
        raise ValueError(f"Saved predecessor prediction does not replay: {key}")
    cell.update({"components": components, "identity": identity,
                 "decoded_predecessor": predecessor})
    return cell


def _pool_family(phases: Mapping[int, list[np.ndarray]], weight_2026: float) -> np.ndarray:
    if set(phases) != set(SEEDS) or len(phases[2026]) != len(phases[2027]):
        raise ValueError("Incomplete paired seed phase inventory")
    if len(phases[2026]) not in (2, 5):
        raise ValueError("Unexpected native phase count")
    per_phase = []
    for left, right in zip(phases[2026], phases[2027]):
        left = balanced.validate_probabilities(left)
        right = balanced.validate_probabilities(right)
        if left.shape != right.shape:
            raise ValueError("Paired phase shapes differ")
        per_phase.append(weight_2026 * left + (1.0 - weight_2026) * right)
    return balanced.validate_probabilities(np.mean(per_phase, axis=0), "independent phase pool")


def _probability_for_weight(cell: Mapping, weight: float) -> np.ndarray:
    experts = {family: _pool_family(cell["components"][family], weight)
               for family in ("ce", "calibrated", "predecessor")}
    mixture = {"ce": experts["ce"], "calibrated": experts["calibrated"],
               "adopted": experts["predecessor"]}
    return balanced.recipe_probability(mixture, .5, .375, "arithmetic")


def _decode(probability: np.ndarray, baseline: np.ndarray, cell: Mapping,
            ha_cost: float = 3.0) -> np.ndarray:
    return balanced.decode_fixed_d(probability, baseline, cell["model_period"],
                                   cell["fps"], ha_cost)


def _official_score(cell: Mapping, prediction: np.ndarray) -> dict:
    raw = official._score_prediction(cell["labels"], prediction, cell["period"],
                                     cell["timestamps"], float(cell["fps"]))
    return {
        "spell_f1": raw["spell_f1"], "macro_f1": raw["macro_f1"],
        "weighted_70_30": raw["weighted_70_30"], "d_f1": raw["d_f1"],
        "d_precision": raw["d_precision"], "d_recall": raw["d_recall"],
        "spell_by_iou": raw["spell_by_iou"],
        "reference_spell_count": raw["reference_spell_count"],
        "predicted_spell_count": raw["predicted_spell_count"],
        "short_reference_spell_count": raw["short_reference_spell_count"],
        "short_matched_spell_count": raw["short_matched_spell_count"],
        "short_spell_recall_iou_0_5_lt3s": raw["short_spell_recall_iou_0_5_lt3s"],
        "graded_frames": raw["graded_frames"],
    }


def _compare_row(actual: Mapping, expected: Mapping, where: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"Metric inventory mismatch at {where}: {set(actual) ^ set(expected)}")
    for name, left in actual.items():
        right = expected[name]
        path = f"{where}/{name}"
        if isinstance(left, Mapping):
            if not isinstance(right, Mapping):
                raise ValueError(f"Metric shape mismatch at {path}")
            _compare_row(left, right, path)
        elif isinstance(left, (int, np.integer)) and not isinstance(left, bool):
            if int(left) != int(right):
                raise ValueError(f"Count mismatch at {path}: {left} != {right}")
        elif left is None:
            if right is not None:
                raise ValueError(f"Null metric mismatch at {path}")
        elif isinstance(left, (float, np.floating)):
            if not math.isfinite(float(left)) or not math.isfinite(float(right)) or abs(float(left)-float(right)) > TOL:
                raise ValueError(f"Metric mismatch at {path}: {left} != {right}")
        elif left != right:
            raise ValueError(f"Metric mismatch at {path}: {left} != {right}")


def _aggregate(rows: Mapping[str, Mapping], views: Mapping[str, list[int]]) -> dict:
    """Average replicates within match, matches within provider, providers equally."""
    grouped_cells = {key: _cell_key_parts(key) for key in rows}
    result = {}
    metric_names = tuple(balanced.METRICS)
    for view, reps in views.items():
        result[view] = {}
        for metric in metric_names:
            by_provider: dict[str, dict[str, list[float]]] = {}
            cell_count = 0
            for key, (provider, match, cell_view, rep) in grouped_cells.items():
                if cell_view != view or rep not in reps:
                    continue
                cell_count += 1
                value = rows[key][metric]
                if value is not None:
                    by_provider.setdefault(provider, {}).setdefault(match, []).append(float(value))
            provider_means = {}
            for provider, matches in by_provider.items():
                match_means = [float(np.mean(values)) for values in matches.values() if values]
                if match_means:
                    provider_means[provider] = float(np.mean(match_means))
            result[view][metric] = {
                "equal_provider": float(np.mean(list(provider_means.values()))) if provider_means else None,
                "providers": dict(sorted(provider_means.items())),
                "cells": cell_count,
            }
    return result


def _compare_aggregate(actual: Mapping, expected: Mapping, where: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"Aggregate view inventory mismatch at {where}")
    for view, metrics in expected.items():
        if set(actual[view]) != set(metrics):
            raise ValueError(f"Aggregate metric inventory mismatch at {where}/{view}")
        for metric, row in metrics.items():
            observed = actual[view][metric]
            _compare_row(observed, row, f"{where}/{view}/{metric}")


def _run_sequence(prediction: np.ndarray) -> tuple[str, ...]:
    return tuple(prediction[np.r_[True, prediction[1:] != prediction[:-1]]].tolist())


def _assert_dx_mask(prediction: np.ndarray, baseline: np.ndarray, where: str) -> None:
    for state in ("D", "X"):
        if not np.array_equal(prediction == state, baseline == state):
            raise ValueError(f"Arm changed exact {state} mask: {where}")


def _prediction_path(root: Path, pred_dir: Path, fact: Mapping, key: str) -> Path:
    path = _resolve(root, fact["path"])
    if not path.is_relative_to(pred_dir) or not path.is_file() or sha256(path) != fact.get("sha256"):
        raise ValueError(f"Missing, unsafe, or changed prediction cache: {key}")
    return path


def _independent_guards(candidate: Mapping, baseline: Mapping) -> dict:
    checks = {}
    for view in ("clean", "feed_like_v1", "feed_like_v2"):
        minimum_spell = baseline[view]["spell_f1"]["equal_provider"]
        actual_spell = candidate[view]["spell_f1"]["equal_provider"]
        checks[f"{view}/spell_f1"] = {"actual": actual_spell, "minimum": minimum_spell,
            "passed": (actual_spell > minimum_spell if view == "feed_like_v2" else actual_spell >= minimum_spell)}
        minimum_short = baseline[view]["short_spell_recall_iou_0_5_lt3s"]["equal_provider"]
        actual_short = candidate[view]["short_spell_recall_iou_0_5_lt3s"]["equal_provider"]
        checks[f"{view}/short_spell_recall_iou_0_5_lt3s"] = {
            "actual": actual_short, "minimum": minimum_short, "passed": actual_short >= minimum_short}
        minimum_macro = baseline[view]["macro_f1"]["equal_provider"] - .001
        actual_macro = candidate[view]["macro_f1"]["equal_provider"]
        checks[f"{view}/macro_f1"] = {"actual": actual_macro, "minimum": minimum_macro,
                                       "passed": actual_macro >= minimum_macro}
    for metric in ("macro_f1", "d_f1"):
        for provider, baseline_value in baseline["clean"][metric]["providers"].items():
            actual = candidate["clean"][metric]["providers"][provider]
            minimum = baseline_value - .02
            checks[f"clean/{provider}/{metric}"] = {"actual": actual, "minimum": minimum,
                                                      "passed": actual >= minimum}
    return {"checks": checks, "passed": all(row["passed"] for row in checks.values())}


def _result_rows(result: Mapping) -> dict[str, dict[str, Mapping]]:
    rows = {"seed2026_0.50": result["baseline_rows"]}
    seed_rows = result["seed_cell_metrics"]
    for weight in (.25, .50, .75):
        name = f"seed2026_{weight:.2f}"
        if name != "seed2026_0.50":
            rows[name] = seed_rows[name]
    for strength in BOUNDARY_ARMS:
        rows[f"boundary_{strength}"] = result["boundary"]["cell_metrics"][strength]
    for arm in HA_COST_ARMS:
        rows[arm] = result["ha_cost"]["cell_metrics"][arm]
    for arm in GATE_ARMS:
        rows[arm] = result["disagreement"]["cell_metrics"][arm]
    return rows


def _result_aggregates(result: Mapping) -> dict[str, Mapping]:
    aggregates = {"seed2026_0.50": result["seed_aggregate"]["seed2026_0.50"]}
    for weight in (.25, .75):
        arm = f"seed2026_{weight:.2f}"
        aggregates[arm] = result["seed_aggregate"][arm]
    for strength in BOUNDARY_ARMS:
        aggregates[f"boundary_{strength}"] = result["boundary"]["aggregate"][strength]
    for arm in HA_COST_ARMS:
        aggregates[arm] = result["ha_cost"]["aggregate"][arm]
    for arm in GATE_ARMS:
        aggregates[arm] = result["disagreement"]["aggregate"][arm]
    return aggregates


def _rebuild_prediction(arm: str, cell: Mapping) -> np.ndarray:
    baseline = cell["baseline_prediction"]
    if arm == "seed2026_0.50":
        return baseline.copy()
    if arm.startswith("seed2026_"):
        weight = float(arm.removeprefix("seed2026_"))
        return _decode(_probability_for_weight(cell, weight), cell["decoded_predecessor"], cell)
    equal_prob = _probability_for_weight(cell, .5)
    if arm.startswith("boundary_"):
        strength = arm.removeprefix("boundary_")
        # The boundary routine consumes H/A/D and preserves frozen predecessor
        # D/X rows. Supply all three channels in its native convention.
        return boundary.relocate(np.column_stack((equal_prob, np.zeros(len(equal_prob)))),
            baseline, cell["model_period"], cell["timestamps"], cell["fps"], strength)
    if arm.startswith("ha_cost_"):
        return _decode(equal_prob, cell["decoded_predecessor"], cell,
                       ha_cost=float(arm.removeprefix("ha_cost_")))
    if arm.startswith("disagreement_lambda_"):
        pooled = {family: _pool_family(cell["components"][family], .5)
                  for family in ("ce", "calibrated", "predecessor")}
        pooled["adopted"] = pooled["predecessor"]
        return disagreement.decode(pooled, cell["decoded_predecessor"], cell["model_period"],
                                   cell["fps"], float(arm.removeprefix("disagreement_lambda_")))
    raise ValueError(f"Unknown arm: {arm}")


def verify(manifest_path: Path, declaration_path: Path, preflight_path: Path, result_path: Path,
           predictions_dir: Path, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve()
    declaration_path = Path(declaration_path).resolve()
    preflight_path = Path(preflight_path).resolve()
    result_path = Path(result_path).resolve()
    pred_dir = Path(predictions_dir).resolve()
    manifest = json.loads(manifest_path.read_text())
    declaration = json.loads(declaration_path.read_text())
    frontier_hashes = _validate_manifest(manifest, declaration, manifest_path, root)
    expected_cells = set(runner.EXPECTED_CELLS)
    if set(manifest["expected_cells"]) != expected_cells or set(manifest["cells"]) != expected_cells:
        raise ValueError("Manifest does not cover the exact frontier cohort")
    preflight = json.loads(preflight_path.read_text())
    if (preflight.get("study_id") != runner.STUDY or preflight.get("passed") is not True or
            preflight.get("candidate_scoring_performed") is not False or
            preflight.get("manifest_sha256") != sha256(manifest_path) or
            preflight.get("declaration_sha256") != sha256(declaration_path) or
            preflight.get("source_status") != manifest["source_status"] or
            preflight.get("source_hashes") != manifest["source_hashes"]):
        raise ValueError("Baseline preflight does not bind this manifest")
    result = json.loads(result_path.read_text())
    if (result.get("study_id") != runner.STUDY or
            result.get("source_status") != manifest["source_status"] or
            result.get("manifest_sha256") != sha256(manifest_path) or
            result.get("evaluation_manifest_sha256") != sha256(manifest_path) or
            result.get("declaration_sha256") != sha256(declaration_path) or
            result.get("preflight_sha256") != sha256(preflight_path) or
            result.get("source_hashes") != manifest["source_hashes"] or
            result.get("arm_ids") != list(runner.ARM_IDS) or
            result.get("frontier_artifacts") != declaration.get("frontier_artifacts") or
            result.get("baseline_replay", {}).get("passed") is not True or
            result.get("baseline_replay", {}).get("manifest_sha256") != sha256(manifest_path)):
        raise ValueError("Run result does not bind the verified control replay")
    cache_facts = manifest["cells"]
    if set(result.get("prediction_inventory", {})) != expected_cells:
        raise ValueError("Prediction inventory does not cover the frozen cohort")
    if set(preflight.get("baseline_rows", {})) != expected_cells:
        raise ValueError("Preflight baseline rows do not cover all cells")
    if set(result.get("baseline_rows", {})) != expected_cells:
        raise ValueError("Run baseline rows do not cover all cells")
    if set(result.get("seed_cell_metrics", {})) != {"seed2026_0.25", "seed2026_0.50", "seed2026_0.75"}:
        raise ValueError("Run result does not contain the complete seed-weight inventory")
    saved_frontier_result = json.loads(_artifact_paths(declaration, root)["result_sha256"].read_text())
    if saved_frontier_result.get("study_id") != runner.frontier.STUDY:
        raise ValueError("Selected frontier result belongs to another study")
    saved_cells = saved_frontier_result.get("cell_metrics", {})
    if set(saved_cells) != expected_cells:
        raise ValueError("Immutable frontier round-one result is missing a frozen cohort cell")

    loaded = {}
    independently_scored: dict[str, dict] = {arm: {} for arm in PREDICTION_KEYS}
    prediction_counts: dict[str, dict] = {arm: {} for arm in PREDICTION_KEYS}
    prediction_hashes = {}
    cache_hashes = {}
    boundary_order_checks = 0
    for key in sorted(expected_cells):
        cell = _read_cache(root, key, cache_facts[key], manifest["source_status"], frontier_hashes)
        loaded[key] = cell
        cache_hashes[key] = cache_facts[key]["sha256"]
        baseline = cell["baseline_prediction"]
        equal_probability = _probability_for_weight(cell, .5)
        if not np.allclose(equal_probability, cell["baseline_probability"], atol=1e-10, rtol=0):
            raise ValueError(f"Current .50 recipe probability does not replay approved baseline: {key}")
        decoded_baseline = _decode(equal_probability, cell["decoded_predecessor"], cell)
        if not np.array_equal(decoded_baseline, baseline):
            raise ValueError(f"Current .50 recipe prediction does not replay approved baseline: {key}")
        _assert_dx_mask(baseline, cell["decoded_predecessor"], f"approved baseline/{key}")
        baseline_score = _official_score(cell, baseline)
        _compare_row(baseline_score, preflight["baseline_rows"][key], f"preflight/{key}")
        _compare_row(baseline_score, result["baseline_rows"][key], f"result_baseline/{key}")
        # The immutable round-one result contains full-precision metrics for the
        # chosen recipe for every development cell.
        saved = saved_cells[key].get(runner.BASELINE_RECIPE_ID)
        if saved is None:
            raise ValueError(f"Saved frontier result lacks selected recipe row: {key}")
        _compare_row(baseline_score, saved, f"frontier_round1/{key}")
        _compare_row(baseline_score, result["seed_cell_metrics"]["seed2026_0.50"][key],
                     f"result_seed_control/{key}")
        if baseline_score["reference_spell_count"] != preflight["baseline_rows"][key]["reference_spell_count"] or baseline_score["graded_frames"] != preflight["baseline_rows"][key]["graded_frames"]:
            raise ValueError(f"Control grading population differs from preflight: {key}")

        pred_fact = result["prediction_inventory"][key]
        pred_path = _prediction_path(root, pred_dir, pred_fact, key)
        prediction_hashes[key] = pred_fact["sha256"]
        with np.load(pred_path, allow_pickle=False) as saved_predictions:
            required_names = {"baseline_prediction"} | {
                value for value in PREDICTION_KEYS.values() if value is not None}
            if not required_names.issubset(saved_predictions.files):
                raise ValueError(f"Prediction file is missing one or more arm arrays: {key}")
            saved_baseline = _read_labels(saved_predictions["baseline_prediction"], "baseline_prediction")
            if not np.array_equal(saved_baseline, baseline):
                raise ValueError(f"Saved baseline prediction differs from native cache: {key}")
            for arm in PREDICTION_KEYS:
                npz_key = PREDICTION_KEYS[arm]
                actual_prediction = baseline.copy() if npz_key is None else _read_labels(saved_predictions[npz_key], npz_key)
                if actual_prediction.shape != baseline.shape:
                    raise ValueError(f"Prediction length mismatch: {key}/{arm}")
                expected_prediction = _rebuild_prediction(arm, cell)
                if not np.array_equal(actual_prediction, expected_prediction):
                    raise ValueError(f"Saved prediction cannot be independently reconstructed: {key}/{arm}")
                if arm != "seed2026_0.50":
                    _assert_dx_mask(actual_prediction, baseline, f"{key}/{arm}")
                if arm.startswith("boundary_"):
                    if _run_sequence(actual_prediction) != _run_sequence(baseline):
                        raise ValueError(f"Boundary arm changed raw run order/count: {key}/{arm}")
                    boundary_order_checks += 1
                metrics = _official_score(cell, actual_prediction)
                _compare_row(metrics, result_rows_for_cell(result, arm)[key], f"result/{arm}/{key}")
                if (metrics["graded_frames"] != baseline_score["graded_frames"] or
                        metrics["reference_spell_count"] != baseline_score["reference_spell_count"] or
                        metrics["short_reference_spell_count"] != baseline_score["short_reference_spell_count"]):
                    raise ValueError(f"Candidate changed grading/reference population: {key}/{arm}")
                for metric in ("d_f1", "d_precision", "d_recall"):
                    if metrics[metric] != baseline_score[metric]:
                        raise ValueError(f"Candidate changed fixed-mask {metric}: {key}/{arm}")
                independently_scored[arm][key] = metrics
                prediction_counts[arm][key] = {
                    **{state: int(np.count_nonzero(actual_prediction == state))
                       for state in ("H", "A", "D", "X")},
                    "predicted_spell_count": metrics["predicted_spell_count"],
                    "reference_spell_count": metrics["reference_spell_count"],
                    "graded_frames": metrics["graded_frames"],
                }

    views = VIEW_REPLICATES
    aggregates = {arm: _aggregate(rows, views) for arm, rows in independently_scored.items()}
    saved_aggregates = _result_aggregates(result)
    for arm, aggregate in aggregates.items():
        _compare_aggregate(aggregate, saved_aggregates[arm], f"aggregate/{arm}")
    baseline_aggregate = aggregates["seed2026_0.50"]
    _compare_aggregate(baseline_aggregate, preflight.get("aggregate", {}), "preflight_aggregate")
    guards = {arm: _independent_guards(aggregates[arm], baseline_aggregate)
              for arm in PREDICTION_KEYS if arm != "seed2026_0.50"}
    sort_key = lambda arm: (
        -aggregates[arm]["feed_like_v2"]["spell_f1"]["equal_provider"],
        -aggregates[arm]["feed_like_v2"]["macro_f1"]["equal_provider"], arm)
    ranking = sorted((arm for arm, guard in guards.items() if guard["passed"]), key=sort_key)
    all_candidate_ranking = sorted(guards, key=sort_key)
    return {
        "study_id": runner.STUDY, "passed": True,
        "source_status": manifest["source_status"],
        "evidence_scope": manifest.get("evidence_scope"),
        "cells_verified": len(expected_cells), "arms_verified": len(PREDICTION_KEYS),
        "candidate_arms_verified": len(PREDICTION_KEYS) - 1,
        "scored_predictions": len(expected_cells) * len(PREDICTION_KEYS),
        "boundary_raw_run_order_checks": boundary_order_checks,
        "hashes": {
            "cache_manifest": sha256(manifest_path), "declaration": sha256(declaration_path),
            "baseline_preflight": sha256(preflight_path),
            "run_result": sha256(result_path), "verifier_source": sha256(Path(__file__).resolve()),
            "frontier_artifacts": frontier_hashes,
            "source_files": manifest["source_hashes"],
            "probability_caches": cache_hashes,
            "prediction_files": prediction_hashes,
        },
        "aggregate": aggregates, "baseline_aggregate": baseline_aggregate,
        "guards": guards, "eligible_ranking": ranking,
        "all_candidate_ranking": all_candidate_ranking,
        "prediction_counts_by_arm": prediction_counts,
        "organizer_reference_access": False, "promotion": False,
    }


def result_rows_for_cell(result: Mapping, arm: str) -> Mapping[str, Mapping]:
    if arm == "seed2026_0.50":
        return result["baseline_rows"]
    if arm.startswith("seed2026_"):
        return result["seed_cell_metrics"][arm]
    if arm.startswith("boundary_"):
        return result["boundary"]["cell_metrics"][arm.removeprefix("boundary_")]
    if arm.startswith("ha_cost_"):
        return result["ha_cost"]["cell_metrics"][arm]
    return result["disagreement"]["cell_metrics"][arm]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.manifest, args.declaration, args.preflight, args.result,
                    args.predictions_dir, args.root)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
