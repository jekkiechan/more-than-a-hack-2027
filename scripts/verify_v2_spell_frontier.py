"""Independent saved-output audit for the reference-free spell-frontier study."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from experiments import v2_spell_balanced_eval as balanced
from experiments import v2_spell_frontier_eval as frontier

ROOT = frontier.ROOT
ATOL = 1e-12
AUDIT_PLAN = ROOT / "experiments/results/v2_spell_frontier_v1_audit_plan.json"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _close(a, b, label: str) -> None:
    if isinstance(a, dict) or isinstance(b, dict):
        if not isinstance(a, dict) or not isinstance(b, dict) or set(a) != set(b):
            raise AssertionError(f"{label}: nested key mismatch")
        for key in a:
            _close(a[key], b[key], f"{label}/{key}")
    elif isinstance(a, (str, bool)) or isinstance(b, (str, bool)) or a is None or b is None:
        if a != b:
            raise AssertionError(f"{label}: {a!r} != {b!r}")
    elif isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
        if int(a) != int(b):
            raise AssertionError(f"{label}: {a} != {b}")
    elif not np.isclose(float(a), float(b), rtol=0, atol=ATOL):
        raise AssertionError(f"{label}: {a} != {b}")


def _decode(encoded: np.ndarray) -> np.ndarray:
    if not np.isin(encoded, (0, 1, 2, 3)).all():
        raise AssertionError("Saved prediction has invalid class codes")
    return np.asarray(("H", "A", "D", "X"), dtype="U1")[encoded]


def _mix(components: dict, recipe: dict, seed: int | None = None) -> np.ndarray:
    """Compute conditional HA probabilities independently of the evaluator."""
    if seed is None:
        values = components
    else:
        values = {"ce": components["seed_ce"][seed],
                  "calibrated": components["seed_calibrated"][seed],
                  "adopted": components["seed_adopted"][seed]}
    weights = {"ce": float(recipe["ce_weight"]),
               "calibrated": float(recipe["calibrated_weight"]),
               "adopted": float(recipe["adopted_weight"])}
    members = []
    for name, weight in weights.items():
        if weight <= 0:
            continue
        p = np.asarray(values[name], dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3:
            raise AssertionError(f"Unexpected {name} emissions shape {p.shape}")
        ha = p[:, :2].copy()
        denom = ha.sum(axis=1, keepdims=True)
        if not np.isfinite(ha).all() or np.any(denom <= 0):
            raise AssertionError(f"Invalid {name} conditional HA emissions")
        ha /= denom
        members.append((ha, weight))
    if not members:
        raise AssertionError("Recipe has no positive expert weights")
    if recipe["pooling"] == "arithmetic":
        mixed = sum(weight * probs for probs, weight in members)
    elif recipe["pooling"] == "log":
        logits = sum(weight * np.log(np.clip(probs, 1e-12, 1.0)) for probs, weight in members)
        logits -= logits.max(axis=1, keepdims=True)
        mixed = np.exp(logits)
    else:
        raise AssertionError(f"Unknown pooling rule: {recipe['pooling']}")
    mixed /= mixed.sum(axis=1, keepdims=True)
    return mixed


def _aggregate(cell_metrics: dict, recipe_ids: list[str]) -> dict:
    result = {}
    for rid in recipe_ids:
        result[rid] = {}
        for view, repetitions in balanced.VIEWS:
            result[rid][view] = {}
            for metric in balanced.METRICS:
                grouped = {}
                for cell, rows in cell_metrics.items():
                    provider, match, actual_view, repetition = cell.split("|")
                    if actual_view == view and int(repetition) in repetitions:
                        grouped.setdefault(provider, {}).setdefault(match, []).append(
                            float(rows[rid][metric]))
                if not grouped:
                    raise AssertionError(f"No cells available for aggregation: {rid}/{view}/{metric}")
                providers = {provider: float(np.mean([np.mean(reps) for reps in matches.values()]))
                             for provider, matches in grouped.items()}
                result[rid][view][metric] = {
                    "equal_provider": float(np.mean(list(providers.values()))),
                    "providers": providers,
                    "cells": sum(len(reps) for matches in grouped.values() for reps in matches.values()),
                }
    return result


def _seed_aggregate(seed_cells: dict, recipe_ids: list[str]) -> dict:
    result = {}
    for rid in recipe_ids:
        result[rid] = {}
        for seed in frontier.SEEDS:
            rows = {cell: {rid: metrics[f"{rid}|{seed}"]}
                    for cell, metrics in seed_cells.items()}
            result[rid][str(seed)] = _aggregate(rows, [rid])[rid]
    return result


def _source_hashes_match(declaration: dict, module) -> None:
    if declaration.get("source_sha256") != module._source_hashes():
        raise AssertionError("Frozen frontier source inventory changed")


def freeze_audit_plan() -> dict:
    """Bind the verifier/tests and both frontier declarations before audit output exists."""
    if AUDIT_PLAN.exists():
        raise FileExistsError(AUDIT_PLAN)
    from experiments import v2_spell_frontier_round2 as round2
    plan_path = ROOT / "experiments/results/v2_spell_frontier_v1_plan.json"
    r1_decl, r2_decl = frontier.DECLARATION, round2.DECLARATION
    if not plan_path.is_file() or not r1_decl.is_file() or not r2_decl.is_file():
        raise FileNotFoundError("Parent plan and both frontier declarations are required")
    tests = (ROOT / "experiments/tests/test_v2_spell_frontier_public.py",
             ROOT / "experiments/tests/test_verify_v2_spell_frontier.py",
             ROOT / "experiments/tests/test_v2_spell_frontier_eval.py",
             ROOT / "experiments/tests/test_v2_spell_frontier_round2.py")
    if any(not path.is_file() for path in tests):
        raise FileNotFoundError("All declared frontier audit tests must exist before the audit plan freezes")
    source_files = (Path(__file__).resolve(), *tests, plan_path, r1_decl, r2_decl)
    bindings = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    value = {"study_id": frontier.STUDY, "status": "frozen_before_independent_audits",
        "parent_plan_sha256": sha(plan_path),
        "round1_declaration_sha256": sha(r1_decl), "round2_declaration_sha256": sha(r2_decl),
        "verifier_and_test_sha256": bindings,
        "round1_scope": "Recompute official scores for all30 saved R1 predictions on all35 cells; replay probabilities/decoder only for the frozen shortlist and both seeds; independently reduce aggregates, gates and ranking; verify current-leader replay.",
        "round2_scope": "Recompute native phase pooling and replay every selected pipeline, both seeds and pooled ensemble on all35 cells; independently reduce all scorecards, gates and selection diagnostics.",
        "comparison": "Prediction arrays exact; metric scalars absolute tolerance 1e-12 and zero relative tolerance; counts and identities exact.",
        "maximum_workers": 1, "organizer_reference_access": False,
        "gar_valid_test_access": False, "public_scoring": False, "promotion": False}
    _write_once(AUDIT_PLAN, value)
    return value


def _check_audit_plan(round2_module) -> dict:
    if not AUDIT_PLAN.is_file():
        raise FileNotFoundError("Freeze the independent audit plan before verifying results")
    plan = json.loads(AUDIT_PLAN.read_text())
    if (plan.get("study_id") != frontier.STUDY or
            plan.get("parent_plan_sha256") != sha(ROOT / "experiments/results/v2_spell_frontier_v1_plan.json") or
            plan.get("round1_declaration_sha256") != sha(frontier.DECLARATION) or
            plan.get("round2_declaration_sha256") != sha(round2_module.DECLARATION)):
        raise AssertionError("Frontier audit plan declarations or parent plan changed")
    for name, expected in plan.get("verifier_and_test_sha256", {}).items():
        path = ROOT / name
        if not path.is_file() or sha(path) != expected:
            raise AssertionError(f"Frontier audit verifier/test binding changed: {name}")
    if plan.get("maximum_workers") != 1:
        raise AssertionError("Audit concurrency differs from the predeclared one-worker limit")
    return plan


def _read_npz(path: Path, expected_sha: str, *, expected_names: set[str]) -> dict:
    if not path.is_file() or sha(path) != expected_sha:
        raise AssertionError(f"Prediction cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_names:
            raise AssertionError(f"Prediction cache field inventory mismatch: {path}")
        return {name: archive[name].copy() for name in archive.files}


def _load_components(cell: str, declaration: dict) -> dict:
    """Load stage-1024 and fixed-control arrays from separately bound source caches."""
    control_fact = declaration["inputs"]["control"][cell]
    prediction_fact = declaration["inputs"]["stage1024_probability_cache"][cell]
    cpath, ppath = ROOT / control_fact["path"], ROOT / prediction_fact["path"]
    if sha(cpath) != control_fact["sha256"] or sha(ppath) != prediction_fact["sha256"]:
        raise AssertionError(f"Frozen frontier emissions changed: {cell}")
    with np.load(cpath, allow_pickle=False) as c, np.load(ppath, allow_pickle=False) as p:
        ci = {name.removeprefix("identity__"): str(c[name].item())
              for name in c.files if name.startswith("identity__")}
        pi = {name.removeprefix("identity__"): str(p[name].item())
              for name in p.files if name.startswith("identity__")}
        if ci != {name: str(value) for name, value in control_fact["identity"].items()}:
            raise AssertionError(f"Frontier control cache identity mismatch: {cell}")
        if pi != {name: str(value) for name, value in prediction_fact["identity"].items()} or (
                pi.get("cell") != cell or pi.get("study_id") != "v2_spell_balanced_v1" or
                pi.get("declaration_sha256") != sha(frontier.balanced_r2.DECLARATION)):
            raise AssertionError(f"Stage-1024 cache identity mismatch: {cell}")
        native_phase_count = int(round(float(c["fps"]) / 5.0))
        if (native_phase_count not in (2, 5) or
                int(pi.get("phase_count", -1)) != native_phase_count or
                not np.isclose(float(c["fps"]) / native_phase_count, 5.0, rtol=0, atol=1e-12)):
            raise AssertionError(f"Stage-1024 phase count mismatch: {cell}")
        phase = {seed: [np.asarray(p[f"phase_ce_seed{seed}_{i}"])
                        for i in range(native_phase_count)] for seed in frontier.SEEDS}
        for seed in frontier.SEEDS:
            seed_pooled = np.mean(phase[seed], axis=0)
            if not np.allclose(seed_pooled, p[f"seed_ce_{seed}"], rtol=0, atol=ATOL):
                raise AssertionError(f"Stage-1024 per-seed native phase mean mismatch: {cell}/{seed}")
        paired_ce = np.mean([np.mean([phase[seed][i] for seed in frontier.SEEDS], axis=0)
                             for i in range(native_phase_count)], axis=0)
        if not np.allclose(paired_ce, p["pooled_ce"], rtol=0, atol=ATOL):
            raise AssertionError(f"Stage-1024 paired seed/phase pool mismatch: {cell}")
        values = {"adopted": c["prob__adopted"].copy(),
                  "calibrated": c["prob__calibrated"].copy(),
                  "seed_adopted": {seed: c[f"prob__cand_seed{seed}"].copy() for seed in frontier.SEEDS},
                  "seed_calibrated": {seed: c[f"prob__adop_seed{seed}"].copy() for seed in frontier.SEEDS},
                  "ce": p["pooled_ce"].copy(),
                  "seed_ce": {seed: p[f"seed_ce_{seed}"].copy() for seed in frontier.SEEDS},
                  "labels": c["labels"].copy(), "period": c["period"].copy(),
                  "timestamps": c["timestamps"].copy(), "model_period": c["model_period"].copy(),
                  "fps": float(c["fps"])}
    probability_shapes = [np.asarray(values[name]).shape for name in ("adopted", "calibrated", "ce")]
    row_shapes = [np.asarray(values[name]).shape for name in ("labels", "period")]
    if (any(len(shape) != 2 or shape[1] != 3 for shape in probability_shapes) or
            len({shape[0] for shape in probability_shapes}) != 1 or
            any(len(shape) != 1 or shape[0] != probability_shapes[0][0] for shape in row_shapes) or
            probability_shapes[0][0] != len(values["timestamps"])):
        raise AssertionError(f"Stage-1024 emissions are not frame-aligned: {cell}")
    for seed in frontier.SEEDS:
        for group in ("seed_adopted", "seed_calibrated", "seed_ce"):
            if values[group][seed].shape != values["adopted"].shape:
                raise AssertionError(f"Seed emission shape mismatch: {cell}/{group}/{seed}")
    return values


def _score(cell: dict, prediction: np.ndarray) -> dict:
    return balanced.controls.audit.score({name: cell[name] for name in
        ("labels", "period", "timestamps", "fps")}, prediction)


def _replay_prediction(cell: dict, recipe: dict, adopted_decoded: np.ndarray,
                       seed: int | None = None) -> np.ndarray:
    probability = _mix(cell, recipe, seed=seed)
    return balanced.decode_fixed_d(probability, adopted_decoded, cell["model_period"],
                                   cell["fps"], float(recipe.get("ha_cost", 3.0)))


def verify_round1() -> dict:
    from experiments import v2_spell_frontier_round2 as round2
    _check_audit_plan(round2)
    declaration = json.loads(frontier.DECLARATION.read_text())
    replay = json.loads(frontier.REPLAY.read_text())
    result = json.loads(frontier.RESULT.read_text())
    _source_hashes_match(declaration, frontier)
    if (result.get("study_id") != frontier.STUDY or result.get("round") != 1 or
            result.get("declaration_sha256") != sha(frontier.DECLARATION) or
            result.get("replay_sha256") != sha(frontier.REPLAY) or
            replay.get("passed") is not True or replay.get("candidate_scoring") is not False):
        raise AssertionError("Frontier round-one provenance or leader replay mismatch")
    cells = frontier.expected_cells()
    recipe_ids = [row["recipe_id"] for row in declaration["recipe_grid"]]
    recipes = {row["recipe_id"]: row for row in declaration["recipe_grid"]}
    if (len(cells) != 35 or len(recipe_ids) != 30 or len(set(recipe_ids)) != 30 or
            result.get("recipe_ids") != recipe_ids or result.get("recipes") != declaration["recipe_grid"]):
        raise AssertionError("Frontier round-one recipe/cell inventory mismatch")
    if (set(result.get("prediction_cache", {})) != cells or
            set(result.get("seed_prediction_cache", {})) != cells or
            set(result.get("cell_metrics", {})) != cells or set(result.get("seed_cells", {})) != cells):
        raise AssertionError("Frontier round-one cell inventory mismatch")
    seed_shortlist = result["seed_shortlist"]
    if len(seed_shortlist) > 3 or not set(seed_shortlist).issubset(recipe_ids):
        raise AssertionError("Frontier round-one seed shortlist is invalid")
    cell_metrics, seed_cells = {}, {}
    leader_rows = {rid: {} for rid in frontier.LEADER_IDS}
    recipe_order = result["recipe_ids"]
    for cell_key in sorted(cells):
        control = _load_components(cell_key, declaration)
        adopted = balanced.runtime.frozen.audit.decode(control["adopted"], control["model_period"],
            control["fps"], balanced.runtime.frozen.audit.CONFIG)
        fact = result["prediction_cache"][cell_key]
        path = ROOT / fact["path"]
        pred_cache = _read_npz(path, fact["sha256"], expected_names={
            *(f"pred__{rid}" for rid in recipe_ids), "identity__study_id",
            "identity__declaration_sha256", "identity__cell", "identity__recipe_order"})
        identity = {name.removeprefix("identity__"): str(pred_cache[name].item())
                    for name in pred_cache if name.startswith("identity__")}
        if (identity.get("study_id") != frontier.STUDY or identity.get("cell") != cell_key or
                identity.get("declaration_sha256") != sha(frontier.DECLARATION) or
                json.loads(identity.get("recipe_order", "[]")) != recipe_order):
            raise AssertionError(f"R1 prediction cache identity mismatch: {cell_key}")
        rows = {}
        for rid in recipe_ids:
            stored = _decode(pred_cache[f"pred__{rid}"])
            if rid in seed_shortlist:
                prediction = _replay_prediction(control, recipes[rid], adopted)
                if not np.array_equal(prediction, stored):
                    raise AssertionError(f"R1 recipe prediction replay mismatch: {cell_key}/{rid}")
            else:
                prediction = stored
            if not np.array_equal(stored == "D", adopted == "D"):
                raise AssertionError(f"R1 saved fixed-D mask changed: {cell_key}/{rid}")
            rows[rid] = _score(control, prediction)
            if rid in leader_rows:
                leader_rows[rid][cell_key] = rows[rid]
        cell_metrics[cell_key] = rows

        seed_fact = result["seed_prediction_cache"][cell_key]
        seed_path = ROOT / seed_fact["path"]
        seed_names = {f"pred__{rid}|{seed}" for rid in seed_shortlist for seed in frontier.SEEDS}
        seed_names |= {"identity__study_id", "identity__declaration_sha256", "identity__cell",
                       "identity__recipe_order", "identity__seed_order"}
        seed_cache = _read_npz(seed_path, seed_fact["sha256"], expected_names=seed_names)
        seed_identity = {name.removeprefix("identity__"): str(seed_cache[name].item())
                         for name in seed_cache if name.startswith("identity__")}
        if (seed_identity.get("study_id") != frontier.STUDY or seed_identity.get("cell") != cell_key or
                seed_identity.get("declaration_sha256") != sha(frontier.DECLARATION) or
                json.loads(seed_identity.get("recipe_order", "[]")) != seed_shortlist or
                json.loads(seed_identity.get("seed_order", "[]")) != list(frontier.SEEDS)):
            raise AssertionError(f"R1 seed cache identity mismatch: {cell_key}")
        seed_cells[cell_key] = {}
        for rid in seed_shortlist:
            for seed in frontier.SEEDS:
                prediction = _replay_prediction(control, recipes[rid], adopted, seed=seed)
                stored = _decode(seed_cache[f"pred__{rid}|{seed}"])
                if not np.array_equal(prediction, stored):
                    raise AssertionError(f"R1 seed prediction replay mismatch: {cell_key}/{rid}/{seed}")
                seed_cells[cell_key][f"{rid}|{seed}"] = _score(control, prediction)
    _close(cell_metrics, result["cell_metrics"], "frontier_r1/cell_metrics")
    _close(seed_cells, result["seed_cells"], "frontier_r1/seed_cells")
    aggregate = _aggregate(cell_metrics, recipe_ids)
    _close(aggregate, result["aggregate"], "frontier_r1/aggregate")
    seed_aggregate = _seed_aggregate(seed_cells, seed_shortlist)
    _close(seed_aggregate, result["seed_diagnostics"], "frontier_r1/seed_diagnostics")

    balanced_r2_result = json.loads(frontier.CONTROL_ROUND2.read_text())
    replay_aggregate, replay_checks = {}, {}
    for rid in frontier.LEADER_IDS:
        if rid not in balanced_r2_result["aggregate"]:
            raise AssertionError(f"Current balanced pipeline lacks declared leader {rid}")
        replay_aggregate[rid] = balanced.aggregate_cell_metrics(leader_rows[rid])
        for view, _ in balanced.VIEWS:
            for metric in balanced.METRICS:
                actual = replay_aggregate[rid][view][metric]["equal_provider"]
                expected = balanced_r2_result["aggregate"][rid][view][metric]["equal_provider"]
                replay_checks[f"{rid}/{view}/{metric}"] = {
                    "actual": actual, "expected": expected,
                    "absolute_error": abs(actual - expected),
                    "passed": abs(actual - expected) <= frontier.TOLERANCE}
    if not all(row["passed"] for row in replay_checks.values()):
        raise AssertionError("Independent current-leader replay exceeded declared tolerance")
    _close(replay_aggregate, replay["aggregate"], "frontier_r1/leader_replay_aggregate")
    _close(replay_checks, replay["checks"], "frontier_r1/leader_replay_checks")

    control_replay = json.loads(frontier.CONTROL_REPLAY.read_text())
    gates = {}
    references = control_replay["reference_spell_counts"]
    baseline = control_replay["adopted"]
    for rid in recipe_ids:
        candidate = {view: {metric: aggregate[rid][view][metric]["equal_provider"]
                            for metric in balanced.METRICS} for view, _ in balanced.VIEWS}
        target = {view: {metric: baseline[view][metric]["equal_provider"]
                         for metric in balanced.METRICS} for view, _ in balanced.VIEWS}
        gate = balanced.eligibility(candidate, target)
        gate["d_metrics_unchanged"] = all(candidate[v][m] == target[v][m]
            for v, _ in balanced.VIEWS for m in ("d_f1", "d_precision", "d_recall"))
        gate["reference_spell_counts_invariant"] = all(
            cell_metrics[key][rid]["reference_spell_count"] == references[key] for key in cells)
        gate["beats_current_best"] = {"actual": aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
            "minimum_strictly_greater": frontier.CURRENT_BEST_V2,
            "passed": aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"] > frontier.CURRENT_BEST_V2}
        gate["eligible"] = (gate["eligible"] and gate["d_metrics_unchanged"] and
            gate["reference_spell_counts_invariant"])
        gates[rid] = gate
    _close(gates, result["eligibility"], "frontier_r1/eligibility")
    ranking = sorted(recipe_ids, key=lambda rid: (-aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
        -aggregate[rid]["feed_like_v2"]["macro_f1"]["equal_provider"], rid))
    expected_ranking = [rid for rid in ranking if gates[rid]["eligible"]]
    if result["eligible_ranking"] != expected_ranking:
        raise AssertionError("R1 eligible ranking changed")
    if result["seed_shortlist"] != frontier._seed_shortlist(gates, aggregate):
        raise AssertionError("R1 seed shortlist changed")

    return {"study_id": frontier.STUDY, "round": 1, "passed": True,
        "cell_count": len(cells), "recipe_count": len(recipe_ids),
        "all_saved_predictions_rescored": True, "all_seed_predictions_replayed": True,
        "mixture_decoder_replayed_recipe_ids": seed_shortlist,
        "round1_result_sha256": sha(frontier.RESULT),
        "round1_declaration_sha256": sha(frontier.DECLARATION),
        "aggregate_recomputed": aggregate, "eligibility_recomputed": gates,
        "seed_aggregate_recomputed": seed_aggregate,
        "organizer_reference_access": False, "gar_valid_test_access": False}


def verify_round2() -> dict:
    from experiments import v2_spell_frontier_round2 as round2
    _check_audit_plan(round2)
    declaration = json.loads(round2.DECLARATION.read_text())
    result = json.loads(round2.RESULT.read_text())
    _source_hashes_match(declaration, round2)
    if (result.get("study_id") != frontier.STUDY or result.get("round") != 2 or
            result.get("declaration_sha256") != sha(round2.DECLARATION) or
            result.get("pipelines") != declaration.get("pipelines") or
            result.get("pipeline_ids") != declaration.get("pipeline_ids") or
            result.get("training_stage") != 1536):
        raise AssertionError("Frontier round-two provenance or pipeline inventory mismatch")
    if (set(result.get("cell_metrics", {})) != frontier.expected_cells() or
            set(result.get("seed_cells", {})) != frontier.expected_cells() or
            set(result.get("prediction_cache", {})) != frontier.expected_cells()):
        raise AssertionError("Frontier round-two cell inventory mismatch")
    pipelines = declaration["pipelines"]
    pipeline_ids = declaration["pipeline_ids"]
    if [row["pipeline_id"] for row in pipelines] != pipeline_ids:
        raise AssertionError("R2 pipeline ordering differs from frozen declaration")
    recipe_by_pipeline = {row["pipeline_id"]: row["recipe"] for row in pipelines}
    if set(recipe_by_pipeline) != set(pipeline_ids) or len(pipeline_ids) > 6:
        raise AssertionError("R2 pipeline declaration is invalid")

    cell_metrics, seed_cells = {}, {}
    phase_counts = {}
    for key in sorted(frontier.expected_cells()):
        fact = declaration["controls"][key]
        prepared, control = round2._cell_inputs(key, {**fact, "cell": key})
        fps = float(control["fps"])
        phase_count = int(round(fps / 5.0))
        if phase_count not in (2, 5) or not np.isclose(fps / phase_count, 5.0, rtol=0, atol=1e-12):
            raise AssertionError(f"Unexpected native phase count: {key}/{fps}")
        if len(prepared.get("phases", [])) != phase_count:
            raise AssertionError(f"Prepared phase inventory mismatch: {key}")
        phase_counts[key] = phase_count

        cache_fact = result["prediction_cache"][key]
        path = ROOT / cache_fact["path"]
        if not path.is_file() or sha(path) != cache_fact["sha256"]:
            raise AssertionError(f"R2 cache hash mismatch: {key}")
        expected_names = {f"pooled__{pid}" for pid in pipeline_ids}
        expected_names |= {f"pred__{pid}" for pid in pipeline_ids}
        expected_names |= {f"seedpred__{pid}|{seed}" for pid in pipeline_ids for seed in round2.SEEDS}
        expected_names |= {f"phase__{arm}__{seed}__{phase}" for arm in round2.ARMS
                           for seed in round2.SEEDS for phase in range(phase_count)}
        expected_names |= {"identity__study_id", "identity__declaration_sha256", "identity__cell",
                           "identity__pipeline_order", "identity__seed_order"}
        cache = _read_npz(path, cache_fact["sha256"], expected_names=expected_names)
        identity = {name.removeprefix("identity__"): str(cache[name].item())
                    for name in cache if name.startswith("identity__")}
        fact_identity = cache_fact["identity"]
        if (identity.get("study_id") != frontier.STUDY or identity.get("cell") != key or
                identity.get("declaration_sha256") != sha(round2.DECLARATION) or
                json.loads(identity.get("pipeline_order", "[]")) != pipeline_ids or
                json.loads(identity.get("seed_order", "[]")) != list(round2.SEEDS) or
                {name: json.loads(identity[name]) if name in ("pipeline_order", "seed_order") else identity[name]
                 for name in identity} != fact_identity):
            raise AssertionError(f"R2 cache identity mismatch: {key}")
        adopted_decoded = balanced.runtime.frozen.audit.decode(control["adopted"],
            control["model_period"], control["fps"], balanced.runtime.frozen.audit.CONFIG)
        cell = {name: control[name] for name in ("labels", "period", "timestamps", "fps")}
        rows, seed_rows = {}, {}
        for arm in round2.ARMS:
            phase_by_seed = {seed: [np.asarray(cache[f"phase__{arm}__{seed}__{phase}"])
                                    for phase in range(phase_count)] for seed in round2.SEEDS}
            for seed in round2.SEEDS:
                if any(array.shape != control["adopted"].shape for array in phase_by_seed[seed]):
                    raise AssertionError(f"R2 native phase shape mismatch: {key}/{arm}/{seed}")
                for phase, array in enumerate(phase_by_seed[seed]):
                    if (not np.isfinite(array).all() or np.any(array < 0) or
                            not np.allclose(array.sum(axis=1), 1.0, rtol=0, atol=1e-6)):
                        raise AssertionError(f"R2 phase probabilities invalid: {key}/{arm}/{seed}/{phase}")
            pooled_ce = np.mean([np.mean([phase_by_seed[seed][phase] for seed in round2.SEEDS], axis=0)
                                 for phase in range(phase_count)], axis=0)
            if not np.isfinite(pooled_ce).all() or np.any(pooled_ce < 0) or not np.allclose(
                    pooled_ce.sum(axis=1), 1.0, rtol=0, atol=1e-6):
                raise AssertionError(f"R2 pooled H/A/D probabilities are invalid: {key}/{arm}")
            per_seed_ce = {seed: np.mean(phase_by_seed[seed], axis=0)
                           for seed in round2.SEEDS}
            for pipeline in pipelines:
                pid = pipeline["pipeline_id"]
                if pipeline["training_arm"] != arm:
                    continue
                recipe = recipe_by_pipeline[pid]
                mix_components = {"ce": pooled_ce, "calibrated": control["calibrated"],
                                  "adopted": control["adopted"]}
                if not np.allclose(cache[f"pooled__{pid}"], pooled_ce, rtol=0, atol=ATOL):
                    raise AssertionError(f"R2 saved pooled CE differs from native phase means: {key}/{pid}")
                probability = _mix(mix_components, recipe)
                prediction = balanced.decode_fixed_d(probability, adopted_decoded,
                    control["model_period"], control["fps"], float(recipe["ha_cost"]))
                saved = _decode(cache[f"pred__{pid}"])
                if not np.array_equal(prediction, saved):
                    raise AssertionError(f"R2 pooled prediction replay mismatch: {key}/{pid}")
                if not np.array_equal(saved == "D", adopted_decoded == "D"):
                    raise AssertionError(f"R2 fixed-D mask changed: {key}/{pid}")
                rows[pid] = _score(control, prediction)
                for seed in round2.SEEDS:
                    seed_components = {"ce": per_seed_ce[seed],
                        "calibrated": control[f"adop_seed{seed}"],
                        "adopted": control[f"cand_seed{seed}"]}
                    seed_prob = _mix(seed_components, recipe)
                    seed_prediction = balanced.decode_fixed_d(seed_prob, adopted_decoded,
                        control["model_period"], control["fps"], float(recipe["ha_cost"]))
                    seed_name = f"seedpred__{pid}|{seed}"
                    if not np.array_equal(seed_prediction, _decode(cache[seed_name])):
                        raise AssertionError(f"R2 seed prediction replay mismatch: {key}/{pid}/{seed}")
                    seed_rows[f"{pid}|{seed}"] = _score(control, seed_prediction)
        cell_metrics[key], seed_cells[key] = rows, seed_rows

    _close(cell_metrics, result["cell_metrics"], "frontier_r2/cell_metrics")
    _close(seed_cells, result["seed_cells"], "frontier_r2/seed_cells")
    aggregate = _aggregate(cell_metrics, pipeline_ids)
    _close(aggregate, result["aggregate"], "frontier_r2/aggregate")
    seed_aggregate = _seed_aggregate(seed_cells, pipeline_ids)
    _close(seed_aggregate, result["seed_diagnostics"], "frontier_r2/seed_diagnostics")
    control_replay = json.loads(frontier.CONTROL_REPLAY.read_text())
    refs = control_replay["reference_spell_counts"]
    if result.get("cohort", {}).get("reference_spell_counts") != refs:
        raise AssertionError("R2 cohort reference-count inventory changed")
    baseline = control_replay["adopted"]
    gates = {}
    for pid in pipeline_ids:
        candidate = {view: {metric: aggregate[pid][view][metric]["equal_provider"]
                            for metric in balanced.METRICS} for view, _ in balanced.VIEWS}
        target = {view: {metric: baseline[view][metric]["equal_provider"]
                         for metric in balanced.METRICS} for view, _ in balanced.VIEWS}
        gate = balanced.eligibility(candidate, target)
        gate["d_metrics_unchanged"] = all(candidate[v][m] == target[v][m]
            for v, _ in balanced.VIEWS for m in ("d_f1", "d_precision", "d_recall"))
        gate["reference_spell_counts_invariant"] = all(
            cell_metrics[key][pid]["reference_spell_count"] == refs[key]
            for key in frontier.expected_cells())
        gate["beats_current_best"] = {"actual": aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"],
            "minimum_strictly_greater": frontier.CURRENT_BEST_V2,
            "passed": aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"] > frontier.CURRENT_BEST_V2}
        gate["eligible"] = gate["eligible"] and gate["d_metrics_unchanged"] and gate["reference_spell_counts_invariant"]
        gates[pid] = gate
    _close(gates, result["eligibility"], "frontier_r2/eligibility")
    ranking = sorted(pipeline_ids, key=lambda pid: (
        -aggregate[pid]["feed_like_v2"]["spell_f1"]["equal_provider"],
        -aggregate[pid]["feed_like_v2"]["macro_f1"]["equal_provider"], pid))
    if result.get("eligible_ranking") != [pid for pid in ranking if gates[pid]["eligible"]]:
        raise AssertionError("R2 eligible pipeline ranking changed")
    return {"study_id": frontier.STUDY, "round": 2, "passed": True,
        "cell_count": len(cell_metrics), "recipe_count": len(pipeline_ids),
        "pipeline_ids": pipeline_ids, "all_saved_predictions_rescored": True,
        "native_phase_pooling_recomputed": True,
        "phase_counts_by_cell": phase_counts,
        "round2_result_sha256": sha(round2.RESULT),
        "round2_declaration_sha256": sha(round2.DECLARATION),
        "aggregate_recomputed": aggregate, "eligibility_recomputed": gates,
        "seed_aggregate_recomputed": seed_aggregate,
        "organizer_reference_access": False, "gar_valid_test_access": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-plan", "verify"), nargs="?", default="verify")
    parser.add_argument("--round", choices=("1", "2"), default="1")
    parser.add_argument("--output", type=Path,
        default=ROOT / "data/experiments/v2_spell_frontier_v1/independent_round1.json")
    args = parser.parse_args()
    if args.action == "freeze-plan":
        plan = freeze_audit_plan()
        print(json.dumps({"frozen": True, "path": str(AUDIT_PLAN),
                          "audit_plan_sha256": sha(AUDIT_PLAN)}, indent=2))
        return
    receipt = verify_round1() if args.round == "1" else verify_round2()
    receipt["verifier_source_sha256"] = sha(Path(__file__).resolve())
    receipt["audit_plan_sha256"] = sha(AUDIT_PLAN)
    _write_once(args.output, receipt)
    print(json.dumps({"passed": True, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
