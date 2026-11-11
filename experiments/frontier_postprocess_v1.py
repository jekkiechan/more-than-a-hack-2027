"""Bounded seed-weight and boundary diagnostics for the approved frontier.

This runner consumes hash-bound native per-phase probability caches for the
selected stage-1024 recipe. It only scores candidate arms after replaying the
saved .593619 frontier baseline over the complete 35-cell cohort.
"""
from __future__ import annotations

import hashlib
import inspect
import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import boundary_relocation_v1 as boundary
from experiments import frontier_disagreement_v1 as disagreement
from experiments import v2_spell_balanced_eval as balanced
from experiments import v2_spell_frontier_eval as frontier
from experiments import v2_spell_frontier_round2 as frontier_r2

ROOT = Path(__file__).resolve().parents[1]
STUDY = "frontier_postprocess_v1"
FRONTIER_SELECTION = ROOT / "experiments/results/v2_spell_frontier_v1_selection.json"
SEEDS = (2026, 2027)
SEED2026_WEIGHTS = (0.25, 0.5, 0.75)
HA_COSTS = (2.0, 4.0)
BOUNDARY_STRENGTHS = ("shift_0.2", "shift_0.4", "shift_0.8")
DISAGREEMENT_LAMBDAS = (.5, 1.0)
ARM_IDS = ("seed2026_0.50", "seed2026_0.25", "seed2026_0.75",
           "boundary_shift_0.2", "boundary_shift_0.4", "boundary_shift_0.8",
           "ha_cost_2", "ha_cost_4", "disagreement_lambda_0.5",
           "disagreement_lambda_1.0")
BASELINE_METRICS = {
    "clean": {"spell_f1": 0.6701365499557103, "macro_f1": 0.8718643812171453},
    "feed_like_v1": {"spell_f1": 0.532128306931943, "macro_f1": 0.8261543004095224},
    "feed_like_v2": {"spell_f1": 0.5936187198896373, "macro_f1": 0.8522037287846299},
}
GUARDS = {
    "spell_f1": {"clean": {"comparison": ">=", "baseline": BASELINE_METRICS["clean"]["spell_f1"]},
        "feed_like_v1": {"comparison": ">=", "baseline": BASELINE_METRICS["feed_like_v1"]["spell_f1"]},
        "feed_like_v2": {"comparison": ">", "baseline": BASELINE_METRICS["feed_like_v2"]["spell_f1"]}},
    "short_spell_recall_iou_0_5_lt3s": {"views": ["clean", "feed_like_v1", "feed_like_v2"],
        "comparison": ">=", "baseline_metric": "short_spell_recall_iou_0_5_lt3s",
        "baseline_source": "same-view equal-provider current control"},
    "macro_f1": {"views": ["clean", "feed_like_v1", "feed_like_v2"],
        "comparison": ">=", "baseline_delta": -0.001},
    "provider_clean_macro_f1_and_d_f1": {"comparison": ">=", "baseline_delta": -0.02,
        "scope": "each provider, clean view"},
    "fixed_masks": {"preserve": ["D", "X"], "from": "approved baseline", "exact": True},
    "grading_population": {"invariant": ["graded_frames", "reference_spell_count",
        "short_reference_spell_count"], "exact": True},
}
RANKING = "eligible arms by unrounded feed_like_v2 spell_f1 descending, macro_f1 descending, arm_id ascending"
PER_CELL_TOLERANCE = 1e-10
AGGREGATE_TOLERANCE = 1e-8
FAMILY_WEIGHTS = {"ce": 0.5, "calibrated": 0.375, "predecessor": 0.125}
POOLING = "arithmetic"
HA_COST = 3.0
BASELINE_RECIPE_ID = "ce0.5_cal0.375_ad0.125_arithmetic_ha3"
METRICS = balanced.METRICS
EXPECTED_CELLS = frontier.expected_cells()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths() -> dict[str, Path]:
    modules = [balanced, frontier, boundary, disagreement]
    pending = list(modules)
    seen = set()
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        modules.append(module)
        pending.extend(value for value in vars(module).values()
                       if inspect.ismodule(value) and value.__name__.startswith("experiments"))
    paths = {Path(module.__file__).resolve() for module in modules
             if getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(ROOT)}
    paths.update({Path(__file__).resolve(), ROOT / "evaluation/possession_metrics.py",
        ROOT / "experiments/baseline_manifest.json", ROOT / "experiments/model_decision_policy.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_selection.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_round1.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_replay.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_round1_declaration.json",
        ROOT / "experiments/frontier_postprocess_readout.py"})
    return {str(path.relative_to(ROOT)): path for path in sorted(paths)}


def source_hashes() -> dict[str, str]:
    return {relative: sha256(path) for relative, path in _source_paths().items()}


def verify_source_hashes(inventory: Mapping[str, str]) -> None:
    if not inventory:
        raise ValueError("Source hash inventory is empty")
    for relative, expected in inventory.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or changed frozen source: {relative}")


def _validate_native_manifest(manifest: Mapping) -> None:
    if manifest.get("schema") != "frontier_postprocess_v1_cache_manifest":
        raise ValueError("Unsupported frontier postprocess cache manifest")
    if manifest.get("source_status") not in ("exact_restored", "reconstructed"):
        raise ValueError("Cache source must be explicitly marked exact_restored or reconstructed")
    if manifest["source_status"] == "reconstructed" and not manifest.get("reconstruction_provenance"):
        raise ValueError("Reconstructed cache needs explicit provenance and limitations")
    if manifest.get("frontier_study_id") != frontier.STUDY:
        raise ValueError("Cache is not bound to the selected v2 spell frontier")
    artifact_paths = {"declaration": frontier.DECLARATION, "result": frontier.RESULT,
                      "replay": frontier.REPLAY, "selection": FRONTIER_SELECTION}
    expected_artifacts = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                          for name, path in artifact_paths.items()}
    if manifest.get("frontier_artifacts") != expected_artifacts:
        raise ValueError("Manifest must bind the selected frontier declaration/result/replay/selection")
    recipe = manifest.get("baseline_recipe")
    expected_recipe = {"recipe_id": BASELINE_RECIPE_ID, "ce_weight": .5,
        "calibrated_weight": .375, "adopted_weight": .125,
        "pooling": "arithmetic", "ha_cost": 3.0, "training_arm": "base", "training_stage": 1024}
    if recipe != expected_recipe:
        raise ValueError("Manifest does not bind the approved .593619 frontier recipe")
    current_selection = json.loads(FRONTIER_SELECTION.read_text())
    recipe_keys = ("recipe_id", "ce_weight", "calibrated_weight", "adopted_weight", "pooling", "ha_cost")
    if {k: current_selection.get("recipe", {}).get(k) for k in recipe_keys} != {
            k: expected_recipe[k] for k in recipe_keys}:
        raise ValueError("Current frontier selection file does not match the approved recipe")
    if (current_selection.get("recipe", {}).get("id") != BASELINE_RECIPE_ID or
            current_selection.get("recipe", {}).get("training_stage") != 1024 or
            current_selection.get("training_arm") != "base"):
        raise ValueError("Current frontier selection is not the approved stage-1024 base arm")
    if manifest.get("family_weights") != FAMILY_WEIGHTS:
        raise ValueError("Postprocessing arms must keep the approved family weights fixed")
    if manifest.get("seeds") != list(SEEDS):
        raise ValueError("Both paired baseline seeds are required")
    inventory = manifest.get("cells")
    expected = manifest.get("expected_cells")
    if not isinstance(inventory, dict) or set(inventory) != EXPECTED_CELLS:
        raise ValueError("Cache inventory must cover all 35 frozen frontier cells")
    if not isinstance(expected, list) or set(expected) != EXPECTED_CELLS or len(expected) != 35:
        raise ValueError("Expected cell inventory must be the exact frontier 35-cell cohort")
    source = manifest.get("source_hashes")
    if not isinstance(source, dict) or not source:
        raise ValueError("Manifest must bind runner, decoder, scorer, and frontier artifact hashes")
    if any(not isinstance(value, str) or len(value) != 64 for value in source.values()):
        raise ValueError("Source inventory must contain SHA-256 digests")
    if manifest.get("baseline_metrics") != BASELINE_METRICS:
        raise ValueError("Manifest baseline replay metrics differ from the approved frontier result")
    for key, fact in inventory.items():
        if not isinstance(fact, dict) or not fact.get("path") or not fact.get("sha256"):
            raise ValueError(f"Missing cache path/hash for {key}")
        if len(fact["sha256"]) != 64:
            raise ValueError(f"Invalid cache SHA-256: {key}")
        if fact.get("phase_count") not in (2, 5):
            raise ValueError(f"Unsupported native phase count for {key}")
        identity = fact.get("identity")
        if not isinstance(identity, dict) or identity.get("cell") != key:
            raise ValueError(f"Cache identity does not bind cell {key}")
        if identity.get("source_status") != manifest["source_status"]:
            raise ValueError(f"Cache status differs from manifest for {key}")
        for field, artifact in (("frontier_declaration_sha256", "declaration"),
                                ("frontier_result_sha256", "result"),
                                ("frontier_replay_sha256", "replay"),
                                ("selection_sha256", "selection")):
            if identity.get(field) != manifest["frontier_artifacts"][artifact]["sha256"]:
                raise ValueError(f"Cache identity source receipt differs for {key}/{field}")


def arm_settings() -> dict:
    return {
        "seed2026_0.50": {"role": "baseline_control", "seed2026_weight": .5,
                           "seed2027_weight": .5, "family_weights": FAMILY_WEIGHTS,
                           "ha_cost": 3.0},
        "seed2026_0.25": {"role": "seed_weight", "seed2026_weight": .25,
                           "seed2027_weight": .75, "family_weights": FAMILY_WEIGHTS,
                           "ha_cost": 3.0},
        "seed2026_0.75": {"role": "seed_weight", "seed2026_weight": .75,
                           "seed2027_weight": .25, "family_weights": FAMILY_WEIGHTS,
                           "ha_cost": 3.0},
        **{f"boundary_{strength}": {"role": "boundary_median", "strength": strength,
            "seed2026_weight": .5, "seed2027_weight": .5, "family_weights": FAMILY_WEIGHTS}
           for strength in BOUNDARY_STRENGTHS},
        **{f"ha_cost_{cost:g}": {"role": "ha_switch_cost", "ha_cost": cost,
            "seed2026_weight": .5, "seed2027_weight": .5, "family_weights": FAMILY_WEIGHTS}
           for cost in HA_COSTS},
        **{f"disagreement_lambda_{lam:.1f}": {"role": "cross_expert_disagreement_gate",
            "lambda": lam, "ha_cost": 3.0, "seed2026_weight": .5,
            "seed2027_weight": .5, "family_weights": FAMILY_WEIGHTS}
           for lam in DISAGREEMENT_LAMBDAS},
    }


def freeze(native_manifest_path: Path, evaluation_manifest_path: Path,
           declaration_path: Path) -> tuple[dict, dict]:
    """Freeze an evaluation manifest/declaration after native cache preparation."""
    if evaluation_manifest_path.exists() or declaration_path.exists():
        raise FileExistsError("Refusing to overwrite the evaluation manifest or declaration")
    native = json.loads(native_manifest_path.read_text())
    _validate_native_manifest(native)
    verify_source_hashes(native["source_hashes"])
    for key, fact in native["cells"].items():
        cache_path = _cache_path(fact["path"])
        if not cache_path.is_file() or sha256(cache_path) != fact["sha256"]:
            raise ValueError(f"Native cache is missing or changed before freeze: {key}")
    closure = source_hashes()
    verify_source_hashes(closure)
    native_ref = {"path": str(native_manifest_path.resolve().relative_to(ROOT)),
                  "sha256": sha256(native_manifest_path)}
    arms = arm_settings()
    if tuple(arms) != ARM_IDS:
        raise RuntimeError("Frozen arm inventory/order differs from the reviewed ten arms")
    evaluation = {**native,
        "frozen_native_manifest": native_ref,
        "preparation_source_hashes": native["source_hashes"],
        "source_hashes": closure,
        "arm_ids": list(ARM_IDS), "arm_settings": arms,
        "guards": GUARDS, "ranking": RANKING,
        "comparison_tolerances": {"per_cell_metric": PER_CELL_TOLERANCE,
                                   "aggregate_metric": AGGREGATE_TOLERANCE},
        "baseline_metrics": BASELINE_METRICS,
        "candidate_scoring_performed_at_freeze": False}
    evaluation_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_manifest_path.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n")
    declaration = {"study_id": STUDY,
        "status": "frozen_before_candidate_scoring",
        "source_status": native["source_status"],
        "reconstruction_provenance": native.get("reconstruction_provenance"),
        "native_manifest": native_ref,
        "evaluation_manifest_sha256": sha256(evaluation_manifest_path),
        "frontier_study_id": frontier.STUDY,
        "frontier_artifacts": native["frontier_artifacts"],
        "baseline_recipe": native["baseline_recipe"],
        "family_weights": FAMILY_WEIGHTS, "arm_ids": list(ARM_IDS),
        "arm_settings": arms, "guards": GUARDS, "ranking": RANKING,
        "comparison_tolerances": evaluation["comparison_tolerances"],
        "source_hashes": closure,
        "expected_baseline_metrics": BASELINE_METRICS,
        "cohort": {"cell_count": 35, "cells": sorted(EXPECTED_CELLS),
                   "seeds": list(SEEDS)},
        "organizer_reference_access": False, "promotion": False}
    declaration_path.parent.mkdir(parents=True, exist_ok=True)
    declaration_path.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    return evaluation, declaration


def validate_manifest(manifest: Mapping) -> None:
    """Validate a frozen evaluation manifest and its complete live source closure."""
    _validate_native_manifest(manifest)
    if manifest.get("arm_ids") != list(ARM_IDS) or manifest.get("arm_settings") != arm_settings():
        raise ValueError("Evaluation manifest arm inventory/settings are not frozen")
    if (manifest.get("guards") != GUARDS or manifest.get("ranking") != RANKING or
            manifest.get("comparison_tolerances") != {"per_cell_metric": PER_CELL_TOLERANCE,
                "aggregate_metric": AGGREGATE_TOLERANCE}):
        raise ValueError("Evaluation manifest guards/ranking/tolerances differ from the frozen protocol")
    current_sources = source_hashes()
    if manifest.get("source_hashes") != current_sources:
        raise ValueError("Frozen source inventory differs from the complete current closure")
    verify_source_hashes(current_sources)
    native_ref = manifest.get("frozen_native_manifest")
    if not isinstance(native_ref, dict):
        raise ValueError("Evaluation manifest must bind the immutable native cache manifest")
    native_path = _cache_path(native_ref.get("path", ""))
    if not native_path.is_file() or sha256(native_path) != native_ref.get("sha256"):
        raise ValueError("Bound native cache manifest is missing or changed")
    native = json.loads(native_path.read_text())
    _validate_native_manifest(native)
    for key in ("cells", "expected_cells", "source_status", "frontier_artifacts",
                "baseline_recipe", "family_weights", "seeds", "baseline_metrics"):
        if manifest.get(key) != native.get(key):
            raise ValueError(f"Evaluation manifest differs from its native input on {key}")
    if manifest.get("preparation_source_hashes") != native.get("source_hashes"):
        raise ValueError("Preparation source closure changed in evaluation manifest")


def _cache_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Cache path escapes the repository")
    if {part.lower() for part in Path(relative).parts}.intersection({"test", "tests", "organizer", "public", "gt"}):
        raise ValueError("Test/organizer reference paths cannot be experiment inputs")
    return path


def _read_cell(key: str, fact: Mapping, source_status: str) -> dict:
    path = _cache_path(fact["path"])
    if not path.is_file() or sha256(path) != fact["sha256"]:
        raise ValueError(f"Missing or changed native frontier cache: {key}")
    with np.load(path, allow_pickle=False) as saved:
        identity = {name.removeprefix("identity__"): str(saved[name].item())
                    for name in saved.files if name.startswith("identity__")}
        expected_identity = fact["identity"]
        required = ("cell", "frontier_study_id", "frontier_result_sha256",
                    "frontier_declaration_sha256", "frontier_replay_sha256",
                    "selection_sha256", "source_status")
        if any(name not in identity for name in required):
            raise ValueError(f"Incomplete frontier cache identity: {key}")
        if any(identity.get(name) != str(expected_identity.get(name)) for name in required):
            raise ValueError(f"Frontier cache identity mismatch: {key}")
        if identity != {name: str(value) for name, value in expected_identity.items()}:
            raise ValueError(f"Frontier cache identity inventory differs: {key}")
        components = {family: {seed: [] for seed in SEEDS}
                      for family in ("ce", "calibrated", "predecessor")}
        for family in components:
            for seed in SEEDS:
                for phase in range(fact["phase_count"]):
                    name = f"prob__{family}__seed{seed}__phase{phase}"
                    if name not in saved.files:
                        raise ValueError(f"Missing paired family/seed/phase array {name}: {key}")
                    components[family][seed].append(balanced.validate_probabilities(saved[name], name))
        cell = {name: saved[name].copy() for name in
                ("labels", "period", "timestamps", "model_period")}
        cell["fps"] = float(saved["fps"])
        cell["baseline_probability"] = validate_conditional_ha(
            saved["prob__approved_baseline"], "approved baseline probability")
        cell["baseline_prediction"] = saved["decoded__approved_baseline"].astype("U1", copy=True)
        cell["predecessor_prediction"] = saved["decoded__predecessor"].astype("U1", copy=True)
    n = len(cell["labels"])
    if (any(np.asarray(cell[name]).shape != (n,) for name in
            ("period", "timestamps", "model_period")) or
            cell["baseline_prediction"].shape != (n,) or
            cell["predecessor_prediction"].shape != (n,) or
            cell["baseline_probability"].shape != (n, 2) or
            not np.isin(cell["labels"].astype(str), ("H", "A", "D", "X")).all() or
            not np.isin(cell["baseline_prediction"], ("H", "A", "D", "X")).all()):
        raise ValueError(f"Native masks, labels, time breaks and baseline do not align: {key}")
    if not math.isfinite(cell["fps"]) or cell["fps"] <= 0 or not np.isfinite(cell["timestamps"]).all():
        raise ValueError(f"Invalid timebase metadata: {key}")
    if any(any(p.shape != (n, 3) for p in phases)
           for family in components.values() for phases in family.values()):
        raise ValueError(f"Family probabilities do not align: {key}")
    delta = np.diff(np.asarray(cell["timestamps"], dtype=float))
    breaks = ((np.diff(np.asarray(cell["period"])) != 0) | (delta <= 0) |
              (delta > 1.5 / cell["fps"]))
    if np.any(breaks & (np.diff(np.asarray(cell["model_period"])) == 0)):
        raise ValueError(f"Model period does not preserve native time breaks: {key}")
    # The approved frontier runner uses the adopted control's decoded mask.
    pooled_adopted = pool_family(components["predecessor"], .5)
    decoded_predecessor = balanced.controls.audit.decode(pooled_adopted,
        cell["model_period"], cell["fps"], balanced.controls.audit.CONFIG)
    if not np.array_equal(decoded_predecessor, cell["predecessor_prediction"]):
        raise ValueError(f"Saved predecessor decode differs from the frozen adopted control: {key}")
    provider, match, view, rep = key.split("|")
    cell.update({"provider": provider, "match": match, "view": view, "rep": int(rep)})
    return {**cell, "components": components, "identity": identity,
            "decoded_predecessor": decoded_predecessor}


def pool_family(phases: Mapping[int, list[np.ndarray]], weight_2026: float) -> np.ndarray:
    weight = float(weight_2026)
    if not math.isfinite(weight) or not 0 <= weight <= 1 or set(phases) != set(SEEDS):
        raise ValueError("Invalid seed weight or incomplete paired seeds")
    counts = {len(phases[seed]) for seed in SEEDS}
    if counts not in ({2}, {5}):
        raise ValueError("Expected paired phase inventories of length two or five")
    per_phase = []
    for p26, p27 in zip(phases[2026], phases[2027]):
        p26 = balanced.validate_probabilities(p26)
        p27 = balanced.validate_probabilities(p27)
        if p26.shape != p27.shape:
            raise ValueError("Paired seed probabilities are misaligned")
        per_phase.append(weight * p26 + (1 - weight) * p27)
    return balanced.validate_probabilities(np.mean(per_phase, axis=0), "phase-pooled family")


def validate_conditional_ha(probability: np.ndarray, name: str = "conditional H/A") -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 2 or len(p) == 0 or not np.isfinite(p).all() or
            (p < 0).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0)):
        raise ValueError(f"Invalid conditional H/A probability array: {name}")
    return p


def probability_for_weight(cell: Mapping, weight_2026: float) -> np.ndarray:
    pooled = family_probabilities(cell, weight_2026)
    components = {"ce": pooled["ce"], "calibrated": pooled["calibrated"],
                  "adopted": pooled["predecessor"]}
    return validate_conditional_ha(balanced.recipe_probability(components, FAMILY_WEIGHTS["ce"],
        FAMILY_WEIGHTS["calibrated"], POOLING))


def family_probabilities(cell: Mapping, weight_2026: float) -> dict[str, np.ndarray]:
    return {family: pool_family(cell["components"][family], weight_2026)
            for family in ("ce", "calibrated", "predecessor")}


def decode_candidate(probability: np.ndarray, baseline: np.ndarray,
                     model_period: np.ndarray, fps: float, ha_cost: float = HA_COST) -> np.ndarray:
    # Preserve the original frontier's exact D/X behavior and decoder call.
    return balanced.decode_fixed_d(probability, baseline, model_period, fps, ha_cost)


def load_complete(manifest: Mapping) -> dict:
    validate_manifest(manifest)
    verify_source_hashes(manifest["source_hashes"])
    return {key: _read_cell(key, manifest["cells"][key], manifest["source_status"])
            for key in sorted(EXPECTED_CELLS)}


def _score_cell(cell: Mapping, prediction: np.ndarray) -> dict:
    return balanced.controls.audit.score(cell, prediction)


def _assert_population(candidate: Mapping, baseline: Mapping, key: str) -> None:
    for metric in ("graded_frames", "reference_spell_count", "short_reference_spell_count"):
        if candidate[metric] != baseline[metric]:
            raise RuntimeError(f"Candidate changed grading population {metric}: {key}")


def _assert_fixed_masks(candidate: np.ndarray, baseline: np.ndarray, key: str) -> None:
    fixed = np.isin(baseline, ("D", "X"))
    if not np.array_equal(candidate[fixed], baseline[fixed]):
        raise RuntimeError(f"Candidate changed approved baseline D/X labels: {key}")


def _assert_metric_row(actual: Mapping, expected: Mapping, key: str) -> None:
    if set(actual) != set(expected):
        raise ValueError(f"Frontier baseline metric inventory differs for {key}")
    for name, actual_value in actual.items():
        expected_value = expected[name]
        if isinstance(actual_value, Mapping):
            if not isinstance(expected_value, Mapping):
                raise ValueError(f"Metric type differs for {key}/{name}")
            _assert_metric_row(actual_value, expected_value, f"{key}/{name}")
        elif isinstance(actual_value, (int, np.integer)) and not isinstance(actual_value, bool):
            if int(actual_value) != int(expected_value):
                raise ValueError(f"Frontier baseline count differs for {key}/{name}")
        elif isinstance(actual_value, (float, np.floating)):
            if (not math.isfinite(float(expected_value)) or
                    abs(float(actual_value)-float(expected_value)) > PER_CELL_TOLERANCE):
                raise ValueError(f"Frontier baseline metric differs for {key}/{name}")
        else:
            if actual_value != expected_value:
                raise ValueError(f"Frontier baseline field differs for {key}/{name}")


def _verify_declaration(manifest_path: Path, declaration_path: Path,
                        manifest: Mapping) -> dict:
    declaration = json.loads(declaration_path.read_text())
    if (declaration.get("study_id") != STUDY or
            declaration.get("status") != "frozen_before_candidate_scoring" or
            declaration.get("evaluation_manifest_sha256") != sha256(manifest_path) or
            declaration.get("arm_ids") != list(ARM_IDS) or
            declaration.get("arm_settings") != arm_settings() or
            declaration.get("guards") != GUARDS or declaration.get("ranking") != RANKING or
            declaration.get("comparison_tolerances") != manifest.get("comparison_tolerances") or
            declaration.get("source_hashes") != manifest.get("source_hashes") or
            declaration.get("native_manifest") != manifest.get("frozen_native_manifest")):
        raise ValueError("Frozen declaration does not bind this exact evaluation manifest and protocol")
    verify_source_hashes(declaration["source_hashes"])
    return declaration


def preflight(manifest_path: Path, declaration_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    _verify_declaration(manifest_path, declaration_path, manifest)
    loaded = load_complete(manifest)
    baseline_rows, identity = {}, {}
    result = json.loads(frontier.RESULT.read_text())
    saved_cells = result.get("cell_metrics", {})
    if set(saved_cells) != EXPECTED_CELLS:
        raise ValueError("Saved frontier result does not cover the exact 35-cell cohort")
    for key, cell in loaded.items():
        recipe_probability = probability_for_weight(cell, .5)
        if not np.allclose(recipe_probability, cell["baseline_probability"], atol=1e-8, rtol=0):
            raise ValueError(f"Saved .50 recipe probability differs from frontier baseline: {key}")
        prediction = decode_candidate(recipe_probability, cell["decoded_predecessor"],
            cell["model_period"], cell["fps"])
        if not np.array_equal(prediction, cell["baseline_prediction"]):
            raise ValueError(f"Saved .50 decoded output differs from frontier baseline: {key}")
        baseline_rows[key] = _score_cell(cell, prediction)
        expected_row = saved_cells[key].get(BASELINE_RECIPE_ID)
        if not isinstance(expected_row, dict):
            raise ValueError(f"Saved frontier result misses selected recipe metrics: {key}")
        _assert_metric_row(baseline_rows[key], expected_row, key)
        identity[key] = {"prediction_sha256": hashlib.sha256(
            cell["baseline_prediction"].tobytes()).hexdigest(),
            "d_mask_sha256": hashlib.sha256((cell["decoded_predecessor"] == "D").tobytes()).hexdigest(),
            "cell_count": len(cell["labels"])}
    aggregate = balanced.aggregate_cell_metrics(baseline_rows)
    for view, expected in BASELINE_METRICS.items():
        for metric, value in expected.items():
            actual = aggregate[view][metric]["equal_provider"]
            if abs(actual - value) > 1e-8:
                raise ValueError(f"Approved frontier baseline replay failed {view}/{metric}: {actual} != {value}")
    return {"study_id": STUDY, "manifest_sha256": sha256(manifest_path),
        "declaration_sha256": sha256(declaration_path),
        "source_status": manifest["source_status"], "source_hashes": manifest["source_hashes"],
        "passed": True, "baseline_rows": baseline_rows, "aggregate": aggregate,
        "cell_identity": identity, "tolerance": 1e-8,
        "candidate_scoring_performed": False}


def run(manifest_path: Path, declaration_path: Path, preflight_path: Path,
        prediction_dir: Path | None = None) -> dict:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    _verify_declaration(manifest_path, declaration_path, manifest)
    pre = json.loads(preflight_path.read_text())
    if (pre.get("study_id") != STUDY or pre.get("passed") is not True or
            pre.get("candidate_scoring_performed") is not False or
            pre.get("manifest_sha256") != sha256(manifest_path) or
            pre.get("declaration_sha256") != sha256(declaration_path) or
            pre.get("source_status") != manifest.get("source_status") or
            pre.get("source_hashes") != manifest.get("source_hashes")):
        raise ValueError("A passed baseline-only replay for this exact manifest is required")
    loaded = load_complete(manifest)
    # Recompute the all-cell control replay and compare it with the frozen receipt
    # before scoring the first candidate.
    current = preflight(manifest_path, declaration_path)
    if current != pre:
        raise ValueError("Frontier control replay differs from its preflight receipt")
    baseline_rows = current["baseline_rows"]
    rows, predictions = {}, {}
    for weight in SEED2026_WEIGHTS:
        rows[weight], predictions[weight] = {}, {}
        for key, cell in loaded.items():
            if weight == .5:
                pred, metrics = cell["baseline_prediction"], baseline_rows[key]
            else:
                probability = probability_for_weight(cell, weight)
                pred = decode_candidate(probability, cell["decoded_predecessor"],
                                        cell["model_period"], cell["fps"])
                metrics = _score_cell(cell, pred)
                _assert_population(metrics, baseline_rows[key], key)
                _assert_fixed_masks(pred, cell["baseline_prediction"], key)
            rows[weight][key], predictions[weight][key] = metrics, pred
    seed_aggregate = {f"seed2026_{weight:.2f}": balanced.aggregate_cell_metrics(cell_rows)
                      for weight, cell_rows in rows.items()}
    boundary_rows, boundary_predictions = {}, {}
    equal_probabilities = {key: probability_for_weight(cell, .5) for key, cell in loaded.items()}
    for strength in BOUNDARY_STRENGTHS:
        boundary_rows[strength], boundary_predictions[strength] = {}, {}
        for key, cell in loaded.items():
            boundary_probability = np.column_stack((equal_probabilities[key],
                                                     np.zeros(len(equal_probabilities[key]))))
            pred = boundary.relocate(boundary_probability, cell["baseline_prediction"],
                cell["model_period"], cell["timestamps"], cell["fps"], strength)
            _assert_fixed_masks(pred, cell["baseline_prediction"], f"{key}/{strength}")
            boundary_rows[strength][key] = _score_cell(cell, pred)
            _assert_population(boundary_rows[strength][key], baseline_rows[key], key)
            boundary_predictions[strength][key] = pred
    boundary_aggregate = {strength: balanced.aggregate_cell_metrics(cell_rows)
                          for strength, cell_rows in boundary_rows.items()}
    ha_cost_rows, ha_cost_predictions = {}, {}
    for cost in HA_COSTS:
        arm = f"ha_cost_{cost:g}"
        ha_cost_rows[arm], ha_cost_predictions[arm] = {}, {}
        for key, cell in loaded.items():
            pred = decode_candidate(equal_probabilities[key], cell["decoded_predecessor"],
                                    cell["model_period"], cell["fps"], ha_cost=cost)
            _assert_fixed_masks(pred, cell["baseline_prediction"], f"{key}/{cost}")
            metrics = _score_cell(cell, pred)
            _assert_population(metrics, baseline_rows[key], key)
            ha_cost_rows[arm][key] = metrics
            ha_cost_predictions[arm][key] = pred
    ha_cost_aggregate = {arm: balanced.aggregate_cell_metrics(cell_rows)
                         for arm, cell_rows in ha_cost_rows.items()}
    disagreement_rows, disagreement_predictions = {}, {}
    for lam in DISAGREEMENT_LAMBDAS:
        arm = f"disagreement_lambda_{lam:.1f}"
        disagreement_rows[arm], disagreement_predictions[arm] = {}, {}
        for key, cell in loaded.items():
            pooled = family_probabilities(cell, .5)
            experts = {"ce": pooled["ce"], "calibrated": pooled["calibrated"],
                       "adopted": pooled["predecessor"]}
            pred = disagreement.decode(experts, cell["decoded_predecessor"],
                cell["model_period"], cell["fps"], lam)
            _assert_fixed_masks(pred, cell["baseline_prediction"], f"{key}/{lam}")
            metrics = _score_cell(cell, pred)
            _assert_population(metrics, baseline_rows[key], key)
            disagreement_rows[arm][key] = metrics
            disagreement_predictions[arm][key] = pred
    disagreement_aggregate = {arm: balanced.aggregate_cell_metrics(cell_rows)
                              for arm, cell_rows in disagreement_rows.items()}
    prediction_inventory = {}
    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        if any(prediction_dir.iterdir()):
            raise FileExistsError("Prediction directory must be empty")
        for key in sorted(EXPECTED_CELLS):
            arrays = {f"pred__seed2026_weight_{weight:.2f}": predictions[weight][key]
                      for weight in SEED2026_WEIGHTS}
            arrays.update({f"pred__boundary_{strength}": boundary_predictions[strength][key]
                           for strength in BOUNDARY_STRENGTHS})
            arrays.update({f"pred__{arm}": predictions_by_cell[key]
                           for arm, predictions_by_cell in ha_cost_predictions.items()})
            arrays.update({f"pred__{arm}": predictions_by_cell[key]
                           for arm, predictions_by_cell in disagreement_predictions.items()})
            arrays["baseline_prediction"] = loaded[key]["baseline_prediction"]
            path = prediction_dir / (key.replace("|", "__") + ".npz")
            np.savez_compressed(path, **arrays)
            prediction_inventory[key] = {"path": str(path), "sha256": sha256(path)}
    return {"study_id": STUDY, "source_status": manifest["source_status"],
        "arm_ids": list(ARM_IDS),
        "evidence_scope": manifest.get("evidence_scope"),
        "baseline_recipe": manifest["baseline_recipe"],
        "manifest_sha256": sha256(manifest_path),
        "evaluation_manifest_sha256": sha256(manifest_path),
        "declaration_sha256": sha256(declaration_path),
        "preflight_sha256": sha256(preflight_path),
        "frontier_artifacts": manifest["frontier_artifacts"],
        "source_hashes": manifest["source_hashes"],
        "seed2026_weights": list(SEED2026_WEIGHTS), "family_weights": FAMILY_WEIGHTS,
        "pooling_order": "paired seed mixture within each native phase, then equal phase mean, then exact balanced.recipe_probability",
        "baseline_replay": {"passed": True, "manifest_sha256": current["manifest_sha256"]},
        "baseline_rows": baseline_rows, "seed_cell_metrics": {
            f"seed2026_{weight:.2f}": rows[weight] for weight in SEED2026_WEIGHTS},
        "seed_aggregate": seed_aggregate,
        "ha_cost": {"arms": list(HA_COSTS), "cell_metrics": ha_cost_rows,
                    "aggregate": ha_cost_aggregate},
        "disagreement": {"lambdas": list(DISAGREEMENT_LAMBDAS),
                         "cell_metrics": disagreement_rows,
                         "aggregate": disagreement_aggregate},
        "boundary": {"arms": list(BOUNDARY_STRENGTHS), "cell_metrics": boundary_rows,
                     "aggregate": boundary_aggregate},
        "prediction_inventory": prediction_inventory,
        "official_metrics": list(METRICS), "organizer_reference_access": False,
        "promotion": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "preflight", "run"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--native-manifest", type=Path)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.native_manifest is None or args.declaration is None:
            parser.error("freeze requires --native-manifest and --declaration")
        freeze(args.native_manifest, args.manifest, args.declaration)
        return
    if args.declaration is None or args.output is None:
        parser.error("preflight and run require --declaration and --output")
    if args.action == "preflight":
        result = preflight(args.manifest, args.declaration)
    else:
        if args.preflight_result is None or args.predictions_dir is None:
            parser.error("run requires --preflight-result and --predictions-dir")
        result = run(args.manifest, args.declaration, args.preflight_result, args.predictions_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
