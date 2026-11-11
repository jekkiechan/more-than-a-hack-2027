"""Bounded seed-weight comparison over sealed native probability caches.

The runner accepts an explicit manifest so it can consume either the exact
frontier caches or a clearly named historical control. It does not declare or
score a cohort until the cache source and scope have been confirmed.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import v2_spell_balanced_eval as scoring
from experiments import boundary_relocation_v1 as boundary
from experiments import complementary_pilot_prepare as preparation

ROOT = Path(__file__).resolve().parents[1]
STUDY = "seed_mix_v1"
SEEDS = (2026, 2027)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)  # weight assigned to seed 2026
HA_COST = 3.0
POOLING_ORDER = "seed_weighted_mean_within_phase_then_equal_phase_mean"
METRICS = scoring.METRICS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_weights(weights=WEIGHTS) -> tuple[float, ...]:
    values = tuple(float(value) for value in weights)
    if (not values or any(not math.isfinite(value) or value < 0 or value > 1
                          for value in values) or len(set(values)) != len(values)):
        raise ValueError("Seed weights must be unique finite values in [0, 1]")
    return values


def _safe_cache_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Cache path escapes the repository root")
    parts = {part.lower() for part in Path(relative).parts}
    if parts.intersection({"test", "tests", "organizer", "public", "gt"}):
        raise ValueError("Test or organizer/reference paths are not valid experiment inputs")
    return path


def _source_paths() -> dict[str, Path]:
    pending = [preparation, scoring, boundary]
    modules = []
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
             if getattr(module, "__file__", None) and
             Path(module.__file__).resolve().is_relative_to(ROOT)}
    paths.update({Path(__file__).resolve(), ROOT / "evaluation/possession_metrics.py",
        ROOT / "experiments/baseline_manifest.json", ROOT / "experiments/model_decision_policy.json",
        ROOT / "experiments/baselines/joint_gru_sn_dense_original_diversity_phase_ensemble_v1.json"})
    return {str(path.relative_to(ROOT)): path for path in sorted(paths)
            if path.is_relative_to(ROOT)}


def _source_hashes() -> dict[str, str]:
    return {rel: sha256(path) for rel, path in _source_paths().items()}


def verify_source_hashes(hashes: Mapping[str, str], root: Path = ROOT) -> None:
    if not hashes:
        raise ValueError("Frozen source hash inventory is empty")
    for relative, expected in hashes.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Frozen source file is missing or unsafe: {relative}")
        if sha256(path) != expected:
            raise ValueError(f"Frozen source changed: {relative}")


def validate_manifest(manifest: Mapping) -> None:
    if manifest.get("schema") != "seed_mix_v1_cache_manifest":
        raise ValueError("Unsupported seed-mix cache manifest")
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("kind") not in ("current_frontier", "historical_control"):
        raise ValueError("Manifest must name current_frontier or historical_control provenance")
    if not source.get("study_id") or not source.get("source_sha256"):
        raise ValueError("Cache source study and source hash are required")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Manifest must bind runner and metric source hashes")
    if any(not isinstance(value, str) or len(value) != 64 for value in source_hashes.values()):
        raise ValueError("Every source hash must be a SHA-256 hex string")
    scope = manifest.get("evidence_scope")
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("Evidence scope must name the exact model/cache context")
    if manifest.get("pooling_order") != POOLING_ORDER:
        raise ValueError("Phase pooling order is fixed: equal phases within seed, then seed weighting")
    if tuple(manifest.get("seeds", ())) != SEEDS:
        raise ValueError("Manifest must contain the paired seeds 2026 and 2027 in order")
    cells = manifest.get("cells")
    expected = manifest.get("expected_cells")
    cohort = manifest.get("cohort")
    if not isinstance(cells, dict) or not isinstance(expected, list) or not expected:
        raise ValueError("Manifest must enumerate a nonempty expected cell cohort")
    if not isinstance(cohort, dict) or not isinstance(cohort.get("matches"), list) or not cohort["matches"]:
        raise ValueError("Manifest must freeze the complete match cohort")
    views = cohort.get("view_replicates")
    if not isinstance(views, dict) or not views:
        raise ValueError("Manifest must freeze every view and realization index")
    if set(cohort.get("view_groups", {})) != {"clean", "feed_like_v1", "feed_like_v2"}:
        raise ValueError("Metric cohort must group clean, v1 and v2 views")
    if len(cohort["matches"]) != 5:
        raise ValueError("Historical original_full pilot requires all five declared matches")
    expected_match_pairs = sorted((provider, match) for provider, match, _ in preparation.audit.MATCHES)
    actual_match_pairs = sorted(tuple(pair) for pair in cohort["matches"])
    if actual_match_pairs != expected_match_pairs:
        raise ValueError("Historical pilot match/provider cohort differs from the frozen five matches")
    fixed_views = {"clean": [0], "feed_like_v1": [1, 2],
                   "feed_like_v2": [1, 2, 3, 4]}
    if views != fixed_views:
        raise ValueError("Historical pilot views and corruption realizations are fixed")
    product = {f"{provider}|{match}|{view}|{rep}"
               for provider, match in (tuple(item) for item in cohort["matches"])
               for view, reps in views.items() for rep in reps}
    if product != set(expected):
        raise ValueError("Expected cells are not the complete match × view × realization product")
    if len(expected) != 35:
        raise ValueError("Historical pilot cohort must contain exactly 35 cells")
    if len(expected) != len(set(expected)) or set(cells) != set(expected):
        raise ValueError("Cache inventory must cover every expected cell exactly once")
    for key in expected:
        if not isinstance(key, str) or len(key.split("|")) != 4:
            raise ValueError(f"Invalid provider|match|view|rep cell key: {key!r}")
        provider, match, view, rep = key.split("|")
        if not provider or not match or view not in ("clean", "feed_like_v1", "feed_like_v2"):
            raise ValueError(f"Invalid cohort cell: {key}")
        if not rep.isdigit():
            raise ValueError(f"Invalid realization index: {key}")
        fact = cells[key]
        if not isinstance(fact, dict) or not fact.get("path") or not fact.get("sha256"):
            raise ValueError(f"Cache path/hash missing for {key}")
        identity = fact.get("identity")
        if (not isinstance(identity, dict) or identity.get("cell") != key or
                identity.get("source_study_id") != source["study_id"]):
            raise ValueError(f"Cache identity must bind its cell: {key}")
        for hash_name in ("input_sha256", "calibration_sha256", "model_manifest_sha256"):
            if not isinstance(identity.get(hash_name), str) or len(identity[hash_name]) != 64:
                raise ValueError(f"Cache identity misses {hash_name}: {key}")
        if not isinstance(fact.get("phase_count"), int) or fact["phase_count"] < 1:
            raise ValueError(f"Invalid phase count: {key}")


def _phase_arrays(saved, prefix: str, seed: int, count: int) -> list[np.ndarray]:
    arrays = []
    for phase in range(count):
        name = f"prob__{prefix}__seed{seed}__phase{phase}"
        if name not in saved.files:
            raise ValueError(f"Missing paired seed phase array: {name}")
        arrays.append(scoring.validate_probabilities(saved[name], name))
    if len({array.shape for array in arrays}) != 1:
        raise ValueError(f"Misaligned phases for {prefix}/seed{seed}")
    return arrays


def decode_candidate(probability: np.ndarray, predecessor: np.ndarray,
                     model_period: np.ndarray, fps: float) -> np.ndarray:
    """Apply the fixed-D HA decoder and retain exact predecessor D/X rows."""
    baseline = np.asarray(predecessor).astype("U1", copy=False)
    proxy = baseline.copy()
    # Shared fixed-D decoding treats any non-D row as live. Substitute X with
    # D during decoding so unknown/native-invalid rows cannot bridge HA spans.
    proxy[baseline == "X"] = "D"
    prediction = scoring.decode_fixed_d(probability, proxy, model_period, fps, HA_COST).copy()
    prediction[baseline == "X"] = "X"
    frozen = np.isin(baseline, ("D", "X"))
    if not np.array_equal(prediction[frozen], baseline[frozen]):
        raise RuntimeError("Candidate decoding changed a frozen D/X row")
    return prediction


def load_cell(root: Path, key: str, fact: Mapping, source_study_id: str) -> dict:
    path = _safe_cache_path(root, fact["path"])
    if not path.is_file() or sha256(path) != fact["sha256"]:
        raise ValueError(f"Missing or changed native probability cache: {key}")
    phase_count = fact["phase_count"]
    with np.load(path, allow_pickle=False) as saved:
        identity = {name.removeprefix("identity__"): str(saved[name].item())
                    for name in saved.files if name.startswith("identity__")}
        expected_identity = fact["identity"]
        required_identity = ("cell", "source_study_id", "input_sha256",
                             "calibration_sha256", "model_manifest_sha256")
        if any(name not in identity for name in required_identity):
            raise ValueError(f"Native cache identity is incomplete: {key}")
        if any(identity.get(name) != str(expected_identity.get(name)) for name in required_identity):
            raise ValueError(f"Native cache identity mismatch: {key}")
        if any(identity.get(name) != str(value)
               for name, value in expected_identity.items() if name in identity):
            raise ValueError(f"Native cache identity mismatch: {key}")
        if expected_identity.get("cell") != key or expected_identity.get("source_study_id") != source_study_id:
            raise ValueError(f"Frozen manifest identity mismatch: {key}")
        arrays = {seed: _phase_arrays(saved, "candidate", seed, phase_count)
                  for seed in SEEDS}
        if "prob__historical_ensemble" not in saved.files:
            raise ValueError(f"Historical equal-seed native ensemble is missing: {key}")
        historical_ensemble = scoring.validate_probabilities(
            saved["prob__historical_ensemble"], "historical ensemble")
        if "decoded__historical_f062" not in saved.files:
            raise ValueError(f"Historical f062 decoded control is missing: {key}")
        predecessor = saved["decoded__historical_f062"].astype("U1", copy=True)
        cell = {name: saved[name].copy() for name in ("labels", "period", "timestamps", "model_period")}
        cell["fps"] = float(saved["fps"])
    n = len(cell["labels"])
    if (any(np.asarray(cell[name]).shape != (n,) for name in ("period", "timestamps", "model_period")) or
            np.asarray(cell["timestamps"]).dtype.kind not in "fi" or
            not np.isfinite(cell["timestamps"]).all() or
            not np.isin(cell["labels"].astype(str), ("H", "A", "D", "X")).all()):
        raise ValueError(f"Labels, native masks, timestamps and breaks must align: {key}")
    # model_period must preserve both native period boundaries and irregular-time breaks.
    delta = np.diff(np.asarray(cell["timestamps"], dtype=float))
    base_period = np.asarray(cell["period"])
    model_period = np.asarray(cell["model_period"])
    required_breaks = (base_period[1:] != base_period[:-1]) | (delta <= 0) | (delta > 1.5 / cell["fps"])
    if np.any(required_breaks & (model_period[1:] == model_period[:-1])):
        raise ValueError(f"Model period failed to preserve a native time break: {key}")
    if predecessor.shape != (n,) or not set(np.unique(predecessor)).issubset({"H", "A", "D", "X"}):
        raise ValueError(f"Invalid historical f062 predecessor prediction: {key}")
    if any(len(phases) != phase_count or any(p.shape != (n, 3) for p in phases)
           for phases in arrays.values()):
        raise ValueError(f"Per-seed probabilities and cell metadata must align: {key}")
    if historical_ensemble.shape != (n, 3):
        raise ValueError(f"Historical equal-seed ensemble does not align: {key}")
    if not math.isfinite(cell["fps"]) or cell["fps"] <= 0:
        raise ValueError(f"Invalid frame rate: {key}")
    provider, match, view, rep = key.split("|")
    cell.update({"provider": provider, "match": match, "view": view, "rep": int(rep)})
    return {**cell, "predecessor": predecessor, "candidate_phases": arrays,
            "historical_ensemble": historical_ensemble}


def freeze(preparation_manifest: Path, manifest_out: Path, declaration_out: Path,
           root: Path = ROOT) -> tuple[dict, dict]:
    """Bind restored preparation artifacts and freeze the exact 35-cell study."""
    if manifest_out.exists() or declaration_out.exists():
        raise FileExistsError("Refusing to overwrite the frozen manifest or declaration")
    prepared = json.loads(preparation_manifest.read_text())
    if prepared.get("scoring_performed") is not False:
        raise ValueError("Preparation receipt must be prediction-only and unscored")
    prepared_sources = prepared.get("source_hashes")
    if not isinstance(prepared_sources, dict) or not prepared_sources:
        raise ValueError("Preparation receipt must bind its inference sources")
    verify_source_hashes(prepared_sources, root)
    source_id = "original_full_phase_ensemble_v1"
    normalized_cells = {}
    match_pairs = set()
    view_replicates = {"clean": [0], "feed_like_v1": [1, 2], "feed_like_v2": [1, 2, 3, 4]}
    expected = []
    for raw_fact in prepared.get("cells", {}).values():
        identity = raw_fact["identity"]
        provider, match, raw_view = identity["provider"], identity["match"], identity["view"]
        if raw_view == "clean":
            family, rep = "clean", 0
        elif raw_view.startswith("feed_like_v1"):
            family = "feed_like_v1"
            rep = int(identity["rep"]) if "rep" in identity else int(raw_view.rsplit("r", 1)[-1])
        elif raw_view.startswith("feed_like_v2"):
            family = "feed_like_v2"
            rep = int(identity["rep"]) if "rep" in identity else int(raw_view.rsplit("r", 1)[-1])
        else:
            raise ValueError(f"Unrecognized preparation view: {raw_view}")
        key = f"{provider}|{match}|{family}|{rep}"
        if key in normalized_cells:
            raise ValueError(f"Duplicate normalized cache cell: {key}")
        match_pairs.add((provider, match))
        expected.append(key)
        fact_identity = dict(identity, cell=key, source_study_id=source_id,
            input_sha256=identity.get("input_sha256", identity.get("source_input_sha256")),
            calibration_sha256=identity.get("calibration_sha256"),
            model_manifest_sha256=identity.get("model_manifest_sha256"))
        normalized_cells[key] = {"path": raw_fact["path"], "sha256": raw_fact["sha256"],
                                 "identity": fact_identity, "phase_count": raw_fact["phase_count"]}
    for key, fact in normalized_cells.items():
        cache_path = _safe_cache_path(root, fact["path"])
        if not cache_path.is_file() or sha256(cache_path) != fact["sha256"]:
            raise ValueError(f"Preparation cache is missing or changed: {key}")
    source_hashes = _source_hashes()
    manifest = {"schema": "seed_mix_v1_cache_manifest",
        "source": {"kind": "historical_control", "study_id": source_id,
                   "source_sha256": sha256(preparation_manifest)},
        "evidence_scope": "historical original_full phase ensemble; five exposed matches; clean, two v1 and four v2 realizations",
        "pooling_order": POOLING_ORDER, "seeds": list(SEEDS), "weights": list(WEIGHTS),
        "source_hashes": source_hashes,
        "cohort": {"matches": [list(pair) for pair in sorted(match_pairs)],
            "view_replicates": view_replicates,
            "view_groups": {"clean": ["clean"], "feed_like_v1": ["feed_like_v1"],
                            "feed_like_v2": ["feed_like_v2"]}},
        "expected_cells": sorted(expected), "cells": normalized_cells,
        "expected_control_spell_f1": {"clean": .6675555328266306,
            "feed_like_v1": .5315755112553304, "feed_like_v2": .5835712076382747},
        "expected_control_macro_f1": {"clean": .8712713689886137,
            "feed_like_v1": .826852823195384, "feed_like_v2": .8499746497420088}}
    validate_manifest(manifest)
    verify_source_hashes(source_hashes, root)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    declaration = {"study_id": STUDY,
        "question": "Does fixed complementary weighting of original_full seeds 2026/2027 improve official spell F1 on the historical phase baseline?",
        "evidence_status": "historical predecessor pilot only; no efficacy claim about the current frontier model",
        "arms": [{"weight_seed2026": weight, "weight_seed2027": 1.0-weight} for weight in WEIGHTS],
        "boundary_arms": ["shift_0.2", "shift_0.4", "shift_0.8"],
        "boundary_input": "only the equal-seed .50 prediction; independent fixed diagnostic arms",
        "cohort": manifest["cohort"], "metrics": list(METRICS),
        "gates": {"clean_v1_v2_spell": "at least historical control; v2 must strictly improve",
                  "short_spell_recall": "at least historical control in clean/v1/v2",
                  "macro_f1": "no worse than control - .001 in clean/v1/v2",
                  "provider_clean_macro_d_f1": "no worse than control - .02"},
        "fixed_decoder": {"historical_predecessor": "f062", "ha_cost": HA_COST,
                          "decoded_d_mask_and_x_rows": "preserved exactly"},
        "manifest_sha256": sha256(manifest_out), "source_hashes": source_hashes,
        "preparation_receipt": {"path": str(preparation_manifest.resolve().relative_to(root.resolve())),
                                 "sha256": sha256(preparation_manifest)},
        "expected_control_spell_f1": manifest["expected_control_spell_f1"],
        "expected_control_macro_f1": manifest["expected_control_macro_f1"],
        "organizer_reference_access": False, "promotion": False}
    declaration_out.parent.mkdir(parents=True, exist_ok=True)
    declaration_out.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    return manifest, declaration


def seed_mixture(probabilities: Mapping[int, np.ndarray], weight_2026: float) -> np.ndarray:
    weight = float(weight_2026)
    if not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Seed weight must be finite and in [0, 1]")
    if set(probabilities) != set(SEEDS):
        raise ValueError("Both declared seeds are required")
    first, second = (scoring.validate_probabilities(probabilities[s], f"seed{s}") for s in SEEDS)
    if first.shape != second.shape:
        raise ValueError("Seed probability arrays must align")
    result = weight * first + (1 - weight) * second
    return scoring.validate_probabilities(result, "seed mixture")


def pool_phases(phases: Mapping[int, list[np.ndarray]], weight_2026: float) -> np.ndarray:
    """Weight paired seeds within each phase, then average the phase outputs."""
    if set(phases) != set(SEEDS) or len(phases[SEEDS[0]]) != len(phases[SEEDS[1]]):
        raise ValueError("Paired seed phase inventories are incomplete")
    if not phases[SEEDS[0]]:
        raise ValueError("At least one phase is required")
    per_phase = []
    for left, right in zip(phases[SEEDS[0]], phases[SEEDS[1]]):
        per_phase.append(seed_mixture({SEEDS[0]: left, SEEDS[1]: right}, weight_2026))
    if len({array.shape for array in per_phase}) != 1:
        raise ValueError("Pooled phase arrays do not align")
    return scoring.validate_probabilities(np.mean(per_phase, axis=0), "phase pooled seed mixture")


def _load_complete_cells(manifest: Mapping, root: Path) -> dict:
    return {key: load_cell(root, key, manifest["cells"][key], manifest["source"]["study_id"])
            for key in manifest["expected_cells"]}


def _preflight_cells(manifest: Mapping, loaded: Mapping) -> tuple[dict, dict]:
    control_rows = {}
    inventory = {}
    for key, cell in loaded.items():
        baseline = cell["predecessor"]
        control_probability = pool_phases(cell["candidate_phases"], 0.5)
        if not np.allclose(control_probability, cell["historical_ensemble"], atol=1e-12, rtol=0):
            raise ValueError(f"Equal-seed phase pooling differs from saved native ensemble: {key}")
        control_prediction = decode_candidate(control_probability, baseline,
            cell["model_period"], cell["fps"])
        if not np.array_equal(control_prediction, baseline):
            raise ValueError(f"Equal-seed mixture does not reproduce exact historical f062 control: {key}")
        control_rows[key] = scoring.controls.audit.score(cell, baseline)
        inventory[key] = {"cell_count": len(cell["labels"]),
                          "historical_prediction_sha256": hashlib.sha256(baseline.tobytes()).hexdigest(),
                          "d_mask_sha256": hashlib.sha256((baseline == "D").tobytes()).hexdigest()}
    if set(loaded) != set(manifest["expected_cells"]):
        raise RuntimeError("Preflight did not load the complete frozen cohort")
    aggregate = aggregate_cells(control_rows, manifest)
    for metric_key, metric_name in (("expected_control_spell_f1", "spell_f1"),
                                    ("expected_control_macro_f1", "macro_f1")):
        for view, expected_value in manifest[metric_key].items():
            actual = aggregate.get(view, {}).get(metric_name, {}).get("equal_provider")
            if actual is None or abs(actual - float(expected_value)) > 1e-8:
                raise ValueError(f"Historical control {metric_name} parity failed for {view}: {actual} != {expected_value}")
    return control_rows, {"passed": True,
        "spell_f1": {view: metrics["spell_f1"]["equal_provider"] for view, metrics in aggregate.items()},
        "aggregate": aggregate, "tolerance": 1e-8,
        "baseline_rows": control_rows, "cell_inventory": inventory}


def aggregate_cells(rows: Mapping[str, dict], manifest: Mapping) -> dict:
    """Use official cell scores and average views within match/provider equally."""
    groups = manifest["cohort"]["view_groups"]
    result = {}
    for name, members in groups.items():
        result[name] = {}
        for metric in METRICS:
            values = {}
            for key, row in rows.items():
                provider, match, view, rep = key.split("|")
                if view in members:
                    values[(provider, match, int(rep))] = float(row[metric])
            mean, providers = scoring.controls.capacity.equal_provider(values)
            result[name][metric] = {"equal_provider": mean, "providers": providers,
                                    "cells": len(values)}
    return result


def _gates(candidate: Mapping, control: Mapping) -> dict:
    checks = {}
    for view in ("clean", "feed_like_v1", "feed_like_v2"):
        for metric, floor in (("spell_f1", 0.0),
                              ("short_spell_recall_iou_0_5_lt3s", 0.0)):
            actual = candidate[view][metric]["equal_provider"]
            baseline = control[view][metric]["equal_provider"]
            passed = actual >= baseline + floor
            checks[f"{view}/{metric}"] = {"actual": actual, "minimum": baseline, "passed": passed}
        metric = "macro_f1"
        actual = candidate[view][metric]["equal_provider"]
        baseline = control[view][metric]["equal_provider"]
        checks[f"{view}/{metric}"] = {"actual": actual, "minimum": baseline - .001,
                                      "passed": actual >= baseline - .001}
    for metric in ("macro_f1", "d_f1"):
        for provider, baseline in control["clean"][metric]["providers"].items():
            actual = candidate["clean"][metric]["providers"][provider]
            checks[f"clean/{provider}/{metric}"] = {"actual": actual,
                "minimum": baseline - .02, "passed": actual >= baseline - .02}
    checks["feed_like_v2/spell_f1_strict_improvement"] = {
        "actual": candidate["feed_like_v2"]["spell_f1"]["equal_provider"],
        "minimum": control["feed_like_v2"]["spell_f1"]["equal_provider"],
        "passed": candidate["feed_like_v2"]["spell_f1"]["equal_provider"] >
                  control["feed_like_v2"]["spell_f1"]["equal_provider"]}
    return {"checks": checks, "passed": all(item["passed"] for item in checks.values()),
            "interpretation": "historical pilot plausibility screen; no adoption implication"}


def _verify_frozen_pair(manifest_path: Path, declaration_path: Path,
                       root: Path) -> tuple[dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest)
    declaration = json.loads(declaration_path.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Declaration does not bind this exact cache manifest")
    if declaration.get("source_hashes") != manifest.get("source_hashes"):
        raise ValueError("Manifest and declaration source inventories differ")
    verify_source_hashes(manifest["source_hashes"], root)
    prep_fact = declaration.get("preparation_receipt", {})
    prep_path = (root / prep_fact.get("path", "")).resolve()
    if (not prep_path.is_relative_to(root.resolve()) or not prep_path.is_file() or
            sha256(prep_path) != prep_fact.get("sha256")):
        raise ValueError("Bound native-probability preparation receipt is missing or changed")
    return manifest, declaration


def preflight(manifest_path: Path, declaration_path: Path, root: Path = ROOT) -> dict:
    manifest, declaration = _verify_frozen_pair(manifest_path, declaration_path, root)
    loaded = _load_complete_cells(manifest, root)
    _rows, control = _preflight_cells(manifest, loaded)
    return {"study_id": STUDY, "manifest_sha256": sha256(manifest_path),
            "declaration_sha256": sha256(declaration_path),
            "source_hashes": manifest["source_hashes"], "source": manifest["source"],
            "evidence_scope": manifest["evidence_scope"], "control_preflight": control}


def run(manifest_path: Path, declaration_path: Path, preflight_path: Path, root: Path = ROOT,
        prediction_dir: Path | None = None) -> dict:
    manifest, declaration = _verify_frozen_pair(manifest_path, declaration_path, root)
    frozen_preflight = json.loads(preflight_path.read_text())
    if (frozen_preflight.get("study_id") != STUDY or
            frozen_preflight.get("manifest_sha256") != sha256(manifest_path) or
            frozen_preflight.get("declaration_sha256") != sha256(declaration_path) or
            frozen_preflight.get("source_hashes") != manifest["source_hashes"] or
            frozen_preflight.get("control_preflight", {}).get("passed") is not True):
        raise ValueError("A passed preflight for this exact frozen manifest is required")
    weights = validate_weights(manifest.get("weights", WEIGHTS))
    if weights != WEIGHTS:
        raise ValueError("The five fixed seed weights are part of the frozen study")
    # Reload and recheck every bound cache/control before candidate scoring.
    loaded = _load_complete_cells(manifest, root)
    control_rows, control = _preflight_cells(manifest, loaded)
    old_control = frozen_preflight["control_preflight"]
    if control != old_control:
        raise ValueError("Current inputs no longer reproduce the immutable preflight receipt")
    inventory = control["cell_inventory"]
    rows = {weight: {} for weight in weights}
    for key, cell in loaded.items():
        adopted_prediction = cell["predecessor"]
        for weight in weights:
            mixture = pool_phases(cell["candidate_phases"], weight)
            prediction = decode_candidate(mixture, adopted_prediction,
                cell["model_period"], cell["fps"])
            rows[weight][key] = scoring.controls.audit.score(cell, prediction)
            if rows[weight][key]["reference_spell_count"] != control_rows[key]["reference_spell_count"]:
                raise RuntimeError(f"Reference spell count changed for {key}")
            if weight == 0.5 and rows[weight][key] != control_rows[key]:
                raise RuntimeError(f"Equal-seed recipe does not reproduce historical control metrics: {key}")
    expected = set(manifest["expected_cells"])
    if any(set(cell_rows) != expected for cell_rows in rows.values()):
        raise RuntimeError("Scored cells do not match the complete declared cohort")
    aggregate = {format(weight, ".2f"): aggregate_cells(cell_rows, manifest)
                 for weight, cell_rows in rows.items()}
    candidate_ids = [format(weight, ".2f") for weight in weights]
    gates = {recipe_id: _gates(aggregate[recipe_id], control["aggregate"])
             for recipe_id in candidate_ids}
    boundary_rows = {strength: {} for strength in ("shift_0.2", "shift_0.4", "shift_0.8")}
    boundary_predictions = {}
    for key, cell in loaded.items():
        pooled = pool_phases(cell["candidate_phases"], .5)
        for strength in boundary.STRENGTHS:
            if strength == "baseline":
                continue
            pred = boundary.relocate(pooled, cell["predecessor"], cell["model_period"],
                                     cell["timestamps"], cell["fps"], strength)
            # The boundary readout is constrained to keep predecessor D/X spans.
            fixed = cell["predecessor"]
            if not np.array_equal(pred[np.isin(fixed, ("D", "X"))],
                                  fixed[np.isin(fixed, ("D", "X"))]):
                raise RuntimeError("Boundary relocation changed a frozen D/X row")
            boundary_rows[strength][key] = scoring.controls.audit.score(cell, pred)
            boundary_predictions.setdefault(key, {})[strength] = pred
    boundary_aggregate = {arm: aggregate_cells(cell_rows, manifest)
                          for arm, cell_rows in boundary_rows.items()}
    baseline_aggregate = control["aggregate"]
    boundary_gates = {arm: _gates(values, baseline_aggregate)
                      for arm, values in boundary_aggregate.items()}
    prediction_inventory = {}
    if prediction_dir is not None:
        if not prediction_dir.exists():
            prediction_dir.mkdir(parents=True)
        if any(prediction_dir.iterdir()):
            raise FileExistsError("Prediction directory must be empty for this frozen run")
        for key, cell in loaded.items():
            arrays = {f"pred__seed2026_weight_{weight:.2f}": decode_candidate(
                pool_phases(cell["candidate_phases"], weight), cell["predecessor"],
                cell["model_period"], cell["fps"]) for weight in weights}
            arrays.update({f"pred__boundary_{arm}": pred
                           for arm, pred in boundary_predictions[key].items()})
            arrays["baseline_prediction"] = cell["predecessor"]
            out_path = prediction_dir / (key.replace("|", "__") + ".npz")
            np.savez_compressed(out_path, **arrays)
            prediction_inventory[key] = {"path": str(out_path), "sha256": sha256(out_path)}
    return {"study_id": STUDY, "source": manifest["source"],
            "evidence_scope": manifest["evidence_scope"], "weights_seed2026": list(weights),
            "phase_pooling_order": POOLING_ORDER, "fixed_ha_cost": HA_COST,
            "control_preflight": {"passed": True, "spell_f1": control["spell_f1"],
                "tolerance": control["tolerance"],
                "preflight_sha256": sha256(preflight_path)},
            "baseline_rows": control_rows,
            "cell_metrics": {format(weight, ".2f"): cell_rows for weight, cell_rows in rows.items()},
            "aggregate": aggregate, "gates": gates,
            "boundary": {"cell_metrics": boundary_rows, "aggregate": boundary_aggregate,
                         "gates": boundary_gates},
            "baseline_aggregate": baseline_aggregate, "cell_inventory": inventory,
            "prediction_inventory": prediction_inventory,
            "official_metrics": list(METRICS), "organizer_reference_access": False,
            "promotion": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "preflight", "run"))
    parser.add_argument("--preparation-manifest", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--declaration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-result", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        if args.preparation_manifest is None or args.declaration is None:
            parser.error("freeze requires --preparation-manifest and --declaration")
        freeze(args.preparation_manifest, args.output, args.declaration)
        return
    if args.manifest is None or args.declaration is None:
        parser.error("preflight/run require --manifest and --declaration")
    if args.action == "preflight":
        result = preflight(args.manifest, args.declaration)
    else:
        if args.preflight_result is None:
            parser.error("run requires --preflight-result from a passed control preflight")
        if args.predictions_dir is None:
            parser.error("run requires --predictions-dir for independent prediction replay")
        result = run(args.manifest, args.declaration, args.preflight_result,
                     prediction_dir=args.predictions_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
