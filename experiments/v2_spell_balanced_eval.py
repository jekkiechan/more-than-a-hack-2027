"""Matched fixed-D probability-mixture search for balanced spell improvement.

The 36 pooled recipes are scored only after a controls-only replay reproduces
both the archived adopted baseline and the previously saved CE/adopted anchor.
Per-seed diagnostics run afterward for a frozen, result-selected shortlist and
never affect ranking or eligibility.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np
import joblib

from experiments import v2_iterative_eval as controls
from experiments import v2_iterative_followup as prior
from experiments import v2_spell_target_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_spell_balanced_v1"
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / f"experiments/results/{STUDY}_round1_declaration.json"
CONTROL_REPLAY = ROOT / f"experiments/results/{STUDY}_control_replay.json"
RESULT = ROOT / f"experiments/results/{STUDY}_round1.json"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
R2_DECLARATION = ROOT / "experiments/results/v2_iterative_v1_round2_declaration.json"
R2_RESULT = ROOT / "experiments/results/v2_iterative_v1_round2.json"
TRAIN_SEAL = ROOT / "experiments/results/v2_iterative_train_v1_seal.json"
SEEDS = (2026, 2027)
VIEWS = controls.VIEWS
METRICS = controls.METRICS
QUALITY = controls.QUALITY_METRICS
TOLERANCE = 1e-10
MACRO_ALLOWANCE = 0.001
ANCHOR = {"ce_weight": 0.5, "calibrated_weight": 0.0, "pooling": "arithmetic", "ha_cost": 3.0}
CE_WEIGHTS = (0.25, 0.5, 0.75)
CALIBRATED_WEIGHTS = (0.0, 0.125, 0.25)
POOLINGS = ("arithmetic", "log")
HA_COSTS = (1.5, 3.0)
WORKERS = min(3, max(1, int(os.environ.get("V2_SPELL_BALANCED_WORKERS", "3"))))
TORCH_THREADS = max(1, int(os.environ.get("V2_SPELL_BALANCED_TORCH_THREADS", "1")))
HISTORICAL_MAIN = controls.HISTORICAL_MAIN
HISTORICAL_V1 = controls.HISTORICAL_V1


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def _float_token(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def recipe_key(recipe: Mapping) -> str:
    adopted_weight = 1.0 - float(recipe["ce_weight"]) - float(recipe["calibrated_weight"])
    return (f"ce{_float_token(recipe['ce_weight'])}_cal{_float_token(recipe['calibrated_weight'])}"
            f"_ad{_float_token(adopted_weight)}_{recipe['pooling']}_ha{_float_token(recipe['ha_cost'])}")


def recipes() -> list[dict]:
    result = []
    for ce_weight in CE_WEIGHTS:
        for calibrated_weight in CALIBRATED_WEIGHTS:
            adopted_weight = 1.0 - ce_weight - calibrated_weight
            if adopted_weight < -1e-15:
                continue
            for pooling in POOLINGS:
                for ha_cost in HA_COSTS:
                    result.append({"ce_weight": ce_weight,
                                   "calibrated_weight": calibrated_weight,
                                   "adopted_weight": max(0.0, adopted_weight),
                                   "pooling": pooling, "ha_cost": ha_cost,
                                   "recipe_id": recipe_key({"ce_weight": ce_weight,
                                       "calibrated_weight": calibrated_weight,
                                       "pooling": pooling, "ha_cost": ha_cost})})
    return result


RECIPES = recipes()
ARMS = tuple(row["recipe_id"] for row in RECIPES)


def _expected_cells() -> list[tuple[str, str, str, int]]:
    return [(provider, match, view, rep)
            for provider, match, _filename in runtime.frozen.audit.MATCHES
            for view, reps in VIEWS for rep in reps]


def _cell_key(provider: str, match: str, view: str, rep: int) -> str:
    return f"{provider}|{match}|{view}|{int(rep)}"


def _source_hashes() -> dict[str, str]:
    frozen = runtime.frozen
    paths = (Path(__file__), Path(controls.__file__), Path(prior.__file__),
             Path(frozen.__file__), Path(frozen.audit.__file__),
             Path(frozen.audit.decoder.__file__), Path(frozen.audit.metrics.__file__),
             Path(frozen.supervised.__file__), Path(frozen.old.__file__),
             Path(frozen.tc.__file__), Path(frozen.v2.__file__), Path(frozen.sn.__file__),
             Path(frozen.sn.cap.__file__), Path(frozen.sn.fh.__file__), Path(frozen.sn.lt.__file__),
             Path(ROOT / "experiments/refine_features.py"),
             Path(ROOT / "experiments/learned_temporal.py"),
             Path(ROOT / "experiments/soccernet_capacity_model.py"),
             Path(ROOT / "evaluation/possession_metrics.py"), PLAN,
             R2_DECLARATION, R2_RESULT, TRAIN_SEAL,
             HISTORICAL_MAIN, HISTORICAL_V1,
             ROOT / "experiments/baseline_manifest.json",
             ROOT / "experiments/model_decision_policy.json",
             frozen.CALIBRATION)
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
            for path in paths}


def _upstream_facts() -> tuple[dict, dict, dict]:
    controls.runtime.frozen._roots()
    if not controls.DECLARATION.is_file() or not controls.REPLAY.is_file():
        raise RuntimeError("The verified iterative control replay is required")
    if not R2_DECLARATION.is_file() or not R2_RESULT.is_file() or not TRAIN_SEAL.is_file():
        raise RuntimeError("The sealed stage-512 CE continuation study is required")
    controls._require_frozen()
    replay = json.loads(controls.REPLAY.read_text())
    if replay.get("passed") is not True or replay.get("candidate_scoring") is not False:
        raise RuntimeError("Adopted control replay failed or included candidate scoring")
    r2 = json.loads(R2_RESULT.read_text())
    if (r2.get("study_id") != "v2_iterative_v1_round2" or
            r2.get("declaration_sha256") != sha(R2_DECLARATION) or
            r2.get("round1_result_sha256") != sha(controls.ROUND1_RESULT)):
        raise RuntimeError("Stage-512 result does not bind the expected iterative study")
    expected = {_cell_key(*cell) for cell in _expected_cells()}
    if len(expected) != 35 or set(replay.get("emission_cache", {})) != expected:
        raise RuntimeError("Expected exactly the frozen 35-cell development cohort")
    control_facts = {}
    for key, fact in replay["emission_cache"].items():
        path = ROOT / fact["path"]
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise RuntimeError(f"Adopted/calibrated probability cache changed: {key}")
        prepared_path = ROOT / fact["prepared_path"]
        if not prepared_path.is_file() or sha(prepared_path) != fact["prepared_sha256"]:
            raise RuntimeError(f"Frozen phase/preparation cache changed: {key}")
        control_facts[key] = {"path": fact["path"], "sha256": fact["sha256"],
                              "prepared_path": fact["prepared_path"],
                              "prepared_sha256": fact["prepared_sha256"],
                              "identity": fact["identity"]}
    candidate_facts = {}
    for key, fact in r2["candidate_emissions"].items():
        if not key.startswith("ce_control/"):
            continue
        cell = key.removeprefix("ce_control/")
        if cell not in expected:
            raise RuntimeError(f"Unexpected stage-512 cell key: {cell}")
        path = ROOT / fact["path"]
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise RuntimeError(f"Stage-512 CE emission changed: {cell}")
        candidate_facts[cell] = {"path": fact["path"], "sha256": fact["sha256"],
                                 "identity": fact["identity"]}
    if set(candidate_facts) != expected:
        raise RuntimeError("Stage-512 CE emissions do not cover exactly 35 cells")
    inputs = controls._require_frozen()["provenance"]["archived_inputs"]
    if set(inputs) != expected:
        raise RuntimeError("Frozen native inputs do not cover exactly 35 cells")
    return control_facts, candidate_facts, {"replay": replay, "r2": r2}


def freeze() -> dict:
    """Bind the fixed grid and all upstream artifacts before replay/scoring."""
    if any(path.exists() for path in (DECLARATION, CONTROL_REPLAY, RESULT)):
        raise RuntimeError("Round-one output exists; preserve this study ID")
    if not PLAN.is_file():
        raise FileNotFoundError(f"Parent declaration is missing: {PLAN}")
    control_facts, candidate_facts, upstream = _upstream_facts()
    baseline_expected = upstream["replay"]["adopted_aggregate"]
    anchor_expected = upstream["r2"]["aggregate"]["ce_control_conditional_blend_fixed_d_ha3"]
    value = {
        "study_id": STUDY, "round": 1,
        "status": "frozen_before_controls_replay_and_recipe_scoring",
        "question": "Can fixed mixtures of stage-512 CE, calibrated and adopted conditional H/A experts improve spell F1 across all views while limiting macro regression to 0.1 percentage point?",
        "recipes": RECIPES, "recipe_count": len(RECIPES),
        "cohort": {"cells": 35, "views": [name for name, _ in VIEWS],
                   "clean_replicates": [0], "feed_like_v1_replicates": [1, 2],
                   "feed_like_v2_replicates": [1, 2, 3, 4],
                   "seeds": list(SEEDS), "provider_weighting": "replicates within match, matches within provider, providers equally"},
        "mixture": {"components": ["ce_control stage512", "calibrated", "adopted original_full"],
                    "weights": "CE + calibrated + adopted = 1; adopted is residual nonnegative weight",
                    "poolings": {"arithmetic": "weighted arithmetic mean of each expert's conditional H/A probabilities",
                                 "log": "weighted log geometric mean of conditional H/A probabilities, normalized"},
                    "d_policy": "predecode adopted f062 output once per cell; preserve this exact D mask for every recipe"},
        "hard_eligibility": {"spell_f1": "strictly above adopted aggregate in clean, v1 and v2",
                             "macro_f1": f"at least adopted minus {MACRO_ALLOWANCE:.3f} in each view",
                             "short_spell_recall_iou_0_5_lt3s": "at least adopted in each view",
                             "d_metrics": "D-F1, D precision and D recall exactly equal adopted due frozen D mask",
                             "reference_spell_count": "invariant in every cell"},
        "aspiration": {"feed_like_v2_spell_f1_strictly_above": 0.60,
                       "required_for_incremental_eligibility": False},
        "selection": {"rank": "eligible recipes by v2 spell F1 descending, macro F1 descending, recipe_id ascending",
                      "seed_diagnostic_shortlist": "top two eligible pooled recipes plus the fixed CE=.5/calibrated=0/adopted=.5/arithmetic/HA3 anchor; deduplicate, max three; diagnostics never affect ranking",
                      "maximum_shortlist": 3},
        "controls_replay": {"required_before_new_recipe_scoring": True,
                            "baseline_expected": baseline_expected,
                            "anchor_recipe_id": recipe_key(ANCHOR),
                            "anchor_expected": anchor_expected,
                            "metrics": list(METRICS), "tolerance": TOLERANCE,
                            "checks_per_control": 24},
        "upstream": {"control_declaration": sha(controls.DECLARATION),
                     "control_replay": sha(controls.REPLAY),
                     "iterative_round2_declaration": sha(R2_DECLARATION),
                     "iterative_round2_result": sha(R2_RESULT),
                     "iterative_training_seal": sha(TRAIN_SEAL),
                     "adopted_calibrated_emissions": control_facts,
                     "stage512_ce_emissions": candidate_facts},
        "source_sha256": _source_hashes(),
        "organizer_reference_access": False, "public_or_GAR_scoring": False,
        "promotion": False,
    }
    if len(RECIPES) != 36 or len(set(ARMS)) != 36 or recipe_key(ANCHOR) not in ARMS:
        raise RuntimeError("Declared grid identity error")
    _write_once(DECLARATION, value)
    return value


def load_frozen_emissions(cell_key: str) -> dict:
    """Load pooled and per-seed native probabilities for one declared cell."""
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze the balanced evaluator before loading emissions")
    declaration = json.loads(DECLARATION.read_text())
    control_fact = declaration["upstream"]["adopted_calibrated_emissions"].get(cell_key)
    ce_fact = declaration["upstream"]["stage512_ce_emissions"].get(cell_key)
    if control_fact is None or ce_fact is None:
        raise KeyError(f"Cell is outside frozen cohort: {cell_key}")
    for fact in (control_fact, ce_fact):
        if sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Frozen source emission changed: {fact['path']}")
    control_path = ROOT / control_fact["path"]
    ce_path = ROOT / ce_fact["path"]
    with np.load(control_path, allow_pickle=False) as c, np.load(ce_path, allow_pickle=False) as ce:
        values = {"adopted": c["prob__adopted"].copy(),
                  "calibrated": c["prob__calibrated"].copy(),
                  "ce": ce["prob__pooled"].copy(),
                  "seed": {seed: {"adopted": c[f"prob__cand_seed{seed}"].copy(),
                                  "calibrated": c[f"prob__adop_seed{seed}"].copy(),
                                  "ce": ce[f"prob__seed{seed}"].copy()}
                           for seed in SEEDS},
                  "labels": c["labels"].copy(), "period": c["period"].copy(),
                  "timestamps": c["timestamps"].copy(),
                  "model_period": c["model_period"].copy(), "fps": float(c["fps"])}
    lengths = {value.shape[0] for name, value in values.items()
               if isinstance(value, np.ndarray) and value.ndim > 0 and name not in ("timestamps",)}
    if len(lengths) != 1 or next(iter(lengths)) != len(values["labels"]):
        raise RuntimeError(f"Frozen component arrays do not align for {cell_key}")
    for name in ("adopted", "calibrated", "ce"):
        validate_probabilities(values[name], name)
    for seed in SEEDS:
        for name, probability in values["seed"][seed].items():
            validate_probabilities(probability, f"{name}/seed{seed}")
    return values


def validate_probabilities(probability: np.ndarray, name: str = "probability") -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if (p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or
            (p < 0).any() or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise ValueError(f"Invalid H/A/D probability matrix: {name}")
    return p


def conditional_ha(probability: np.ndarray) -> np.ndarray:
    p = validate_probabilities(probability)
    ha = p[:, :2]
    mass = ha.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("Every frame must have positive H/A probability mass")
    return ha / mass


def recipe_probability(components: Mapping[str, np.ndarray], ce_weight: float,
                       calibrated_weight: float, pooling: str) -> np.ndarray:
    """Return weighted conditional H/A expert probabilities, without labels."""
    if pooling not in POOLINGS:
        raise ValueError(f"Unknown pooling rule {pooling!r}")
    weights = {"ce": float(ce_weight), "calibrated": float(calibrated_weight),
               "adopted": 1.0 - float(ce_weight) - float(calibrated_weight)}
    if (any(not np.isfinite(w) or w < -1e-15 for w in weights.values()) or
            abs(sum(weights.values()) - 1.0) > 1e-12):
        raise ValueError("Mixture weights must be finite, nonnegative and sum to one")
    selected = [(conditional_ha(components[name]), max(0.0, weight))
                for name, weight in weights.items() if weight > 0.0]
    shapes = {p.shape for p, _ in selected}
    if len(shapes) != 1:
        raise ValueError("Expert conditional probability shapes differ")
    if pooling == "arithmetic":
        result = sum(weight * probability for probability, weight in selected)
    else:
        logits = sum(weight * np.log(np.clip(probability, 1e-12, 1.0))
                     for probability, weight in selected)
        logits -= logits.max(axis=1, keepdims=True)
        result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    return result


def _decoder(ha_cost: float) -> dict:
    return controls.capacity._decoder(float(ha_cost))


def decode_fixed_d(probability: np.ndarray, adopted_decoded: np.ndarray,
                   period: np.ndarray, fps: float, ha_cost: float) -> np.ndarray:
    """Decode conditional H/A inside the already frozen adopted live spans."""
    p = np.asarray(probability, dtype=np.float64)
    adopted = np.asarray(adopted_decoded).astype("U1", copy=False)
    period = np.asarray(period)
    if p.ndim != 2 or p.shape[1] not in (2, 3) or adopted.shape != (len(p),) or period.shape != (len(p),):
        raise ValueError("Probability, adopted mask and model period must align")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Candidate probabilities must be finite and nonnegative")
    ha = p[:, :2].copy()
    mass = ha.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("Candidate H/A mass must be positive in every row")
    ha /= mass
    conditional = np.column_stack((ha, np.zeros(len(ha), dtype=np.float64)))
    live = adopted != "D"
    result = adopted.copy()
    for start, stop in runtime.frozen.supervised.blocks(period):
        i = start
        while i < stop:
            if not live[i]:
                i += 1
                continue
            j = i + 1
            while j < stop and live[j]:
                j += 1
            result[i:j] = controls.audit.decode(conditional[i:j], period[i:j], fps,
                                                _decoder(ha_cost))
            i = j
    if not np.array_equal(result == "D", adopted == "D"):
        raise RuntimeError("Fixed-D decoder changed the predecoded adopted mask")
    return result


def _load_control_arrays(fact: dict) -> dict:
    path = ROOT / fact["path"]
    with np.load(path, allow_pickle=False) as saved:
        identity = {k.removeprefix("identity__"): str(saved[k].item())
                    for k in saved.files if k.startswith("identity__")}
        if identity != {k: str(v) for k, v in fact["identity"].items()}:
            raise RuntimeError(f"Control emission identity mismatch: {path}")
        return {k.removeprefix("prob__"): saved[k].copy()
                for k in saved.files if k.startswith("prob__")} | {
                    "labels": saved["labels"].copy(), "period": saved["period"].copy(),
                    "timestamps": saved["timestamps"].copy(),
                    "model_period": saved["model_period"].copy(), "fps": float(saved["fps"])}


def _load_components_for_cell(key: str, declaration: dict | None = None) -> dict:
    if declaration is None:
        declaration = json.loads(DECLARATION.read_text())
    control_fact = declaration["upstream"]["adopted_calibrated_emissions"][key]
    ce_fact = declaration["upstream"]["stage512_ce_emissions"][key]
    control = _load_control_arrays(control_fact)
    ce_path = ROOT / ce_fact["path"]
    with np.load(ce_path, allow_pickle=False) as ce:
        identity = {name.removeprefix("identity__"): str(ce[name].item())
                    for name in ce.files if name.startswith("identity__")}
        expected_identity = {name: str(value) for name, value in ce_fact["identity"].items()}
        if identity != expected_identity:
            raise RuntimeError(f"Stage-512 CE emission identity mismatch: {ce_path}")
        control.update({"ce": ce["prob__pooled"].copy(),
                        "seed_ce": {seed: ce[f"prob__seed{seed}"].copy() for seed in SEEDS}})
    control["components"] = {"adopted": control["adopted"],
                              "calibrated": control["calibrated"], "ce": control["ce"]}
    control["seed_components"] = {
        seed: {"adopted": control[f"cand_seed{seed}"],
               "calibrated": control[f"adop_seed{seed}"],
               "ce": control["seed_ce"][seed]} for seed in SEEDS}
    return control


def _score_prediction(cell: dict, prediction: np.ndarray) -> dict:
    return controls.audit.score({"labels": cell["labels"], "period": cell["period"],
                                 "timestamps": cell["timestamps"], "fps": cell["fps"]}, prediction)


def _metrics_for_recipe(cell: dict, recipe: dict, adopted_decoded: np.ndarray,
                        components: Mapping[str, np.ndarray]) -> tuple[dict, np.ndarray]:
    probability = recipe_probability(components, recipe["ce_weight"],
                                     recipe["calibrated_weight"], recipe["pooling"])
    prediction = decode_fixed_d(probability, adopted_decoded, cell["model_period"],
                                cell["fps"], recipe["ha_cost"])
    return _score_prediction(cell, prediction), prediction


def replay_controls() -> dict:
    """Recompute adopted and known-anchor metrics before any grid-arm scoring."""
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze the round-one declaration before replay")
    declaration = require_declaration()
    if CONTROL_REPLAY.exists() or RESULT.exists():
        raise RuntimeError("Control replay or round-one output already exists")
    baseline_rows, anchor_rows, inventory = {}, {}, {}
    control_facts = declaration["upstream"]["adopted_calibrated_emissions"]
    for key in sorted(control_facts):
        cell = _load_control_arrays(control_facts[key])
        adopted_prediction = controls.audit.decode(cell["adopted"], cell["model_period"],
                                                   cell["fps"], controls.audit.CONFIG)
        baseline_rows[key] = controls.audit.score({"labels": cell["labels"], "period": cell["period"],
            "timestamps": cell["timestamps"], "fps": cell["fps"]}, adopted_prediction)
        components = _load_components_for_cell(key, declaration)
        anchor_recipe = dict(ANCHOR, adopted_weight=0.5, recipe_id=recipe_key(ANCHOR))
        anchor_rows[key], anchor_prediction = _metrics_for_recipe(
            components, anchor_recipe, adopted_prediction, components["components"])
        if anchor_rows[key]["reference_spell_count"] != baseline_rows[key]["reference_spell_count"]:
            raise RuntimeError(f"Reference spell count changed in anchor replay: {key}")
        inventory[key] = {"adopted_d_mask_sha256": hashlib.sha256(
            adopted_prediction.tobytes()).hexdigest(),
            "anchor_prediction_sha256": hashlib.sha256(anchor_prediction.tobytes()).hexdigest(),
            "cell_count": len(adopted_prediction)}
    adopted_aggregate = aggregate_cell_metrics(baseline_rows)
    anchor_aggregate = aggregate_cell_metrics(anchor_rows)
    expected_adopted = declaration["controls_replay"]["baseline_expected"]
    expected_anchor = declaration["controls_replay"]["anchor_expected"]
    checks = {}
    for label, actual, expected in (("adopted", adopted_aggregate, expected_adopted),
                                    ("anchor", anchor_aggregate, expected_anchor)):
        for view, _ in VIEWS:
            for metric in METRICS:
                value = float(actual[view][metric]["equal_provider"])
                target = float(expected[view][metric]["equal_provider"])
                checks[f"{label}/{view}/{metric}"] = {"actual": value, "expected": target,
                    "absolute_error": abs(value - target),
                    "passed": abs(value - target) <= TOLERANCE}
    receipt = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
               "controls_only": True, "candidate_scoring": False,
               "cell_count": len(baseline_rows), "adopted": adopted_aggregate,
               "anchor": anchor_aggregate, "cell_counts": _cell_counts(baseline_rows),
               "reference_spell_counts": {
                   key: row["reference_spell_count"] for key, row in baseline_rows.items()},
               "cell_hashes": inventory, "checks": checks,
               "tolerance": TOLERANCE, "passed": all(row["passed"] for row in checks.values())}
    _write_once(CONTROL_REPLAY, receipt)
    if not receipt["passed"]:
        raise RuntimeError(f"Baseline/anchor replay failed 1e-10 checks: {checks}")
    return receipt


def aggregate_cell_metrics(rows: Mapping[str, dict]) -> dict:
    """Aggregate 35 saved cell scores with canonical equal-provider ordering."""
    result = {}
    for view, reps in VIEWS:
        result[view] = {}
        for metric in METRICS:
            values = {}
            for key in sorted(rows):
                provider, match, cell_view, rep = key.split("|")
                if cell_view == view and int(rep) in reps:
                    values[(provider, match, int(rep))] = float(rows[key][metric])
            mean, providers = controls.capacity.equal_provider(values)
            result[view][metric] = {"equal_provider": mean, "providers": providers,
                                    "cells": len(values)}
    return result


def _cell_counts(rows: Mapping[str, dict]) -> dict:
    return {f"{view}/{rep}": sum(1 for key in rows
                                 if key.split("|")[2:] == [view, str(rep)])
            for view, reps in VIEWS for rep in reps}


def eligibility(candidate: Mapping[str, Mapping[str, float]],
                baseline: Mapping[str, Mapping[str, float]]) -> dict:
    """Apply frozen incremental eligibility; v2 >.60 remains an aspiration."""
    checks = {}
    for view, _ in VIEWS:
        spell, baseline_spell = float(candidate[view]["spell_f1"]), float(baseline[view]["spell_f1"])
        macro, baseline_macro = float(candidate[view]["macro_f1"]), float(baseline[view]["macro_f1"])
        short = float(candidate[view]["short_spell_recall_iou_0_5_lt3s"])
        baseline_short = float(baseline[view]["short_spell_recall_iou_0_5_lt3s"])
        for name, actual, minimum, comparator in (
                ("spell_f1", spell, baseline_spell, "strictly_greater"),
                ("macro_f1", macro, baseline_macro - MACRO_ALLOWANCE, "greater_or_equal"),
                ("short_spell_recall_iou_0_5_lt3s", short, baseline_short, "greater_or_equal")):
            passed = actual > minimum if comparator == "strictly_greater" else actual >= minimum
            checks[f"{view}/{name}"] = {"actual": actual, "minimum": minimum,
                                         "comparison": comparator, "passed": passed}
    return {"checks": checks, "eligible": all(row["passed"] for row in checks.values())}


def select_seed_shortlist(eligibilities: Mapping[str, dict], pooled: Mapping[str, dict]) -> list[str]:
    """Return top two eligible pooled recipes plus anchor, without seed selection."""
    eligible_ids = [recipe for recipe, gates in eligibilities.items() if gates["eligible"]]
    ranked = sorted(eligible_ids, key=lambda recipe: (
        -float(pooled[recipe]["feed_like_v2"]["spell_f1"]["equal_provider"]),
        -float(pooled[recipe]["feed_like_v2"]["macro_f1"]["equal_provider"]), recipe))
    selected = ranked[:2]
    anchor_id = recipe_key(ANCHOR)
    if anchor_id not in selected:
        selected.append(anchor_id)
    # Preserve rank order, then place the fixed anchor if it was not already selected.
    return list(dict.fromkeys(selected))[:3]


def _score_cell_job(job):
    key, fact, expected_reference_count = job
    components = _load_components_for_cell(key)
    adopted_decoded = controls.audit.decode(components["adopted"], components["model_period"],
                                            components["fps"], controls.audit.CONFIG)
    rows, predictions = {}, {}
    for recipe in RECIPES:
        metrics, prediction = _metrics_for_recipe(components, recipe, adopted_decoded,
                                                   components["components"])
        if metrics["reference_spell_count"] != expected_reference_count:
            raise RuntimeError(f"Reference spell count changed: {key}/{recipe['recipe_id']}")
        rows[recipe["recipe_id"]] = metrics
        predictions[recipe["recipe_id"]] = _encode_prediction(prediction)
    cache = OUT / "predictions" / f"{key.replace('|', '__')}.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    identity = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "cell": key,
                "control_sha256": fact["sha256"], "ce_sha256": fact["ce_sha256"]}
    arrays = {f"pred__{arm}": pred for arm, pred in predictions.items()}
    arrays.update({f"identity__{name}": np.asarray(str(value)) for name, value in identity.items()})
    np.savez_compressed(cache, **arrays)
    return key, rows, {"path": str(cache.relative_to(ROOT)), "sha256": sha(cache), "identity": identity}


def _encode_prediction(prediction: np.ndarray) -> np.ndarray:
    values = np.asarray(prediction).astype("U1")
    mapping = {"H": 0, "A": 1, "D": 2, "X": 3}
    if not np.isin(values, tuple(mapping)).all():
        raise ValueError("Unexpected official prediction labels")
    return np.asarray([mapping[value] for value in values], dtype=np.int8)


def _decode_prediction(encoded: np.ndarray) -> np.ndarray:
    labels = np.asarray(encoded)
    if not np.isin(labels, (0, 1, 2, 3)).all():
        raise ValueError("Invalid saved prediction codes")
    return np.asarray(["H", "A", "D", "X"], dtype="U1")[labels]


def _worker_init():
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except ImportError:
        pass


def _seed_diagnostic_job(job):
    key, recipes_for_cell = job
    components = _load_components_for_cell(key)
    adopted_decoded = controls.audit.decode(components["adopted"], components["model_period"],
                                            components["fps"], controls.audit.CONFIG)
    result, predictions = {}, {}
    cell = {"labels": components["labels"], "period": components["period"],
            "timestamps": components["timestamps"], "fps": components["fps"]}
    for recipe in recipes_for_cell:
        for seed in SEEDS:
            probability = recipe_probability(components["seed_components"][seed],
                recipe["ce_weight"], recipe["calibrated_weight"], recipe["pooling"])
            prediction = decode_fixed_d(probability, adopted_decoded, components["model_period"],
                                        components["fps"], recipe["ha_cost"])
            metrics = controls.audit.score(cell, prediction)
            result[f"{recipe['recipe_id']}|{seed}"] = metrics
            predictions[f"{recipe['recipe_id']}|{seed}"] = _encode_prediction(prediction)
    path = OUT / "seed_predictions" / f"{key.replace('|', '__')}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    identity = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "cell": key,
                "recipes": [r["recipe_id"] for r in recipes_for_cell]}
    arrays = {f"pred__{name}": pred for name, pred in predictions.items()}
    arrays.update({f"identity__{name}": np.asarray(str(value)) for name, value in identity.items()})
    np.savez_compressed(path, **arrays)
    return key, result, {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "identity": identity}


def require_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze balanced evaluator before use")
    declaration = json.loads(DECLARATION.read_text())
    if declaration["source_sha256"] != _source_hashes():
        raise RuntimeError("Balanced evaluator source changed after freeze")
    controls_facts, ce_facts, upstream = _upstream_facts()
    if (controls_facts != declaration["upstream"]["adopted_calibrated_emissions"] or
            ce_facts != declaration["upstream"]["stage512_ce_emissions"]):
        raise RuntimeError("A frozen input emission changed after declaration")
    if (sha(controls.REPLAY) != declaration["upstream"]["control_replay"] or
            sha(R2_RESULT) != declaration["upstream"]["iterative_round2_result"] or
            sha(R2_DECLARATION) != declaration["upstream"]["iterative_round2_declaration"] or
            sha(TRAIN_SEAL) != declaration["upstream"]["iterative_training_seal"]):
        raise RuntimeError("Upstream replay or training identity changed")
    if len(declaration["recipes"]) != 36 or [r["recipe_id"] for r in declaration["recipes"]] != list(ARMS):
        raise RuntimeError("Frozen recipe grid changed")
    return declaration


def run_round1() -> dict:
    """Score all 36 pooled recipes after the archived replay passes."""
    declaration = require_declaration()
    if not CONTROL_REPLAY.is_file():
        raise RuntimeError("Replay adopted baseline and anchor before recipe scoring")
    replay = json.loads(CONTROL_REPLAY.read_text())
    if replay.get("passed") is not True or replay.get("candidate_scoring") is not False:
        raise RuntimeError("Controls-only replay did not pass")
    if RESULT.exists():
        raise RuntimeError("Round-one result already exists; scoring is one-shot")
    control_facts = declaration["upstream"]["adopted_calibrated_emissions"]
    ce_facts = declaration["upstream"]["stage512_ce_emissions"]
    reference_counts = replay["reference_spell_counts"]
    jobs = [(key, {"path": fact["path"], "sha256": fact["sha256"],
                   "ce_path": ce_facts[key]["path"], "ce_sha256": ce_facts[key]["sha256"]},
             reference_counts[key]) for key, fact in sorted(control_facts.items())]
    cell_rows, pred_inventory = {}, {}
    workers = min(WORKERS, len(jobs))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_score_cell_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            key, metrics, prediction_fact = future.result()
            cell_rows[key] = metrics
            pred_inventory[key] = prediction_fact
            print(f"BALANCED R1 {done}/{len(jobs)} {key}", flush=True)
    if set(cell_rows) != set(control_facts) or any(set(cell_rows[k]) != set(ARMS) for k in cell_rows):
        raise RuntimeError("Pooled scoring did not produce all 36 arms on all 35 cells")
    aggregated = {arm: aggregate_cell_metrics({key: row[arm] for key, row in cell_rows.items()})
                  for arm in ARMS}
    baseline = replay["adopted"]
    gates = {}
    for arm in ARMS:
        candidate_values = {view: {metric: aggregated[arm][view][metric]["equal_provider"]
                                   for metric in METRICS} for view, _ in VIEWS}
        baseline_values = {view: {metric: baseline[view][metric]["equal_provider"]
                                  for metric in METRICS} for view, _ in VIEWS}
        gates[arm] = eligibility(candidate_values, baseline_values)
        gates[arm]["d_metrics_unchanged"] = all(
            candidate_values[view][metric] == baseline_values[view][metric]
            for view, _ in VIEWS for metric in ("d_f1", "d_precision", "d_recall"))
        gates[arm]["reference_spell_counts_invariant"] = all(
            cell_rows[key][arm]["reference_spell_count"] == replay["reference_spell_counts"][key]
            for key in cell_rows)
        gates[arm]["eligible"] = (gates[arm]["eligible"] and gates[arm]["d_metrics_unchanged"] and
                                   gates[arm]["reference_spell_counts_invariant"])
    shortlist = select_seed_shortlist(gates, aggregated)
    seed_rows, seed_inventory = {}, {}
    seed_jobs = [(key, [row for row in RECIPES if row["recipe_id"] in shortlist])
                 for key in sorted(cell_rows)]
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(seed_jobs)), initializer=_worker_init) as pool:
        futures = {pool.submit(_seed_diagnostic_job, job): job for job in seed_jobs}
        for future in as_completed(futures):
            key, metrics, fact = future.result()
            seed_rows[key] = metrics
            seed_inventory[key] = fact
    seed_aggregate = aggregate_seed_metrics(seed_rows, shortlist)
    result = {"study_id": STUDY, "round": 1,
              "declaration_sha256": sha(DECLARATION),
              "control_replay_sha256": sha(CONTROL_REPLAY),
              "recipes": RECIPES, "recipe_ids": list(ARMS),
              "cohort": {"cell_count": len(cell_rows), "cell_counts": replay["cell_counts"],
                         "reference_spell_counts": replay["reference_spell_counts"]},
              "aggregate": aggregated, "eligibility": gates,
              "v2_target_aspiration": {arm: {
                  "actual": aggregated[arm]["feed_like_v2"]["spell_f1"]["equal_provider"],
                  "target_strictly_above": 0.60,
                  "passed": aggregated[arm]["feed_like_v2"]["spell_f1"]["equal_provider"] > 0.60}
                  for arm in ARMS},
              "eligible_ranking": sorted((arm for arm in ARMS if gates[arm]["eligible"]),
                  key=lambda arm: (-aggregated[arm]["feed_like_v2"]["spell_f1"]["equal_provider"],
                                   -aggregated[arm]["feed_like_v2"]["macro_f1"]["equal_provider"], arm)),
              "selected_for_round2_seed_diagnostics": shortlist,
              "selected_for_round2": [{**next(row for row in RECIPES
                  if row["recipe_id"] == arm), "training_stage": 512} for arm in shortlist],
              "seed_diagnostics": seed_aggregate, "seed_cells": seed_rows,
              "cell_metrics": cell_rows,
              "candidate_prediction_cache": pred_inventory,
              "seed_prediction_cache": seed_inventory,
              "organizer_reference_access": False, "public_or_GAR_scoring": False,
              "promotion": False}
    _write_once(RESULT, result)
    return result


def aggregate_seed_metrics(seed_rows: Mapping[str, dict], shortlist: list[str]) -> dict:
    """Aggregate exact-seed diagnostics after pooled ranking is frozen."""
    output = {}
    for arm in shortlist:
        output[arm] = {}
        for seed in SEEDS:
            rows = {key: values[f"{arm}|{seed}"] for key, values in seed_rows.items()}
            output[arm][str(seed)] = aggregate_cell_metrics(rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay-controls", "run-round1"))
    action = parser.parse_args().action
    {"freeze": freeze, "replay-controls": replay_controls,
     "run-round1": run_round1}[action]()


if __name__ == "__main__":
    main()
