"""Evaluate declared extra-data continuation arms on the frozen 35-cell cohort."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import v2_spell_balanced_eval as balanced
from experiments import v2_spell_balanced_round2 as balanced_r2
from experiments import v2_spell_balanced_train as balanced_train
from experiments import v2_spell_frontier_eval as frontier

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_extra_data_v1"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
REPLAY = ROOT / f"experiments/results/{STUDY}_control_replay.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
PREFLIGHT_DECL = ROOT / f"experiments/results/{STUDY}_preflight_declaration.json"
PREFLIGHT_RESULT = ROOT / f"experiments/results/{STUDY}_preflight.json"
OUT = ROOT / "data/experiments" / STUDY
FRONTIER_DECL = ROOT / "experiments/results/v2_spell_frontier_v1_round2_declaration.json"
FRONTIER_R1_DECL = ROOT / "experiments/results/v2_spell_frontier_v1_round1_declaration.json"
FRONTIER_RESULT = ROOT / "experiments/results/v2_spell_frontier_v1_round1.json"
FRONTIER_R2_RESULT = ROOT / "experiments/results/v2_spell_frontier_v1_round2.json"
FRONTIER_SELECTION = ROOT / "experiments/results/v2_spell_frontier_v1_selection.json"
FRONTIER_CONTROL_REPLAY = ROOT / "experiments/results/v2_spell_frontier_v1_replay.json"
TRAIN_DECL = ROOT / f"experiments/results/{STUDY}_training_declaration.json"
TRAIN_SEAL = ROOT / f"experiments/results/{STUDY}_training_seal.json"
CONTROL_RECIPE_ID = "ce0.5_cal0.375_ad0.125_arithmetic_ha3"
CONTROL_EXPECTED = {"clean": .6701365499557103,
                    "feed_like_v1": .532128306931943,
                    "feed_like_v2": .5936187198896373}
TOLERANCE = 1e-12
MACRO_ALLOWANCE = .001
SEEDS = (2026, 2027)
VIEWS = balanced.VIEWS
METRICS = balanced.METRICS
WORKERS = min(2, max(1, int(os.environ.get("V2_EXTRA_DATA_WORKERS", "2"))))
TORCH_THREADS = 1


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as f:
        json.dump(data, f, indent=1, sort_keys=True, default=float)
        f.write("\n")


def _source_hashes():
    frozen = balanced.runtime.frozen
    return {str(p.relative_to(ROOT)): sha(p) for p in (
        Path(__file__), Path(balanced.__file__), Path(frontier.__file__),
        Path(balanced_r2.__file__),
        Path(balanced_train.__file__),
        Path(frozen.__file__), Path(frozen.audit.__file__), Path(frozen.audit.decoder.__file__),
        Path(frozen.audit.metrics.__file__), Path(frozen.supervised.__file__),
        Path(frozen.old.__file__), Path(frozen.tc.__file__), Path(frozen.sn.__file__),
        Path(frozen.sn.cap.__file__), Path(frozen.sn.fh.__file__), Path(frozen.sn.lt.__file__),
        ROOT / "experiments/refine_features.py", ROOT / "experiments/learned_temporal.py",
        ROOT / "experiments/soccernet_capacity_model.py", ROOT / "evaluation/possession_metrics.py",
        ROOT / "experiments/v2_extra_data_train.py",
        PLAN, FRONTIER_DECL, FRONTIER_R1_DECL, FRONTIER_RESULT, FRONTIER_R2_RESULT,
        frontier.DECLARATION, balanced_r2.DECLARATION, balanced_r2.RESULT,
        FRONTIER_SELECTION, FRONTIER_CONTROL_REPLAY,
        ROOT / "experiments/tests/test_v2_extra_data_eval.py")}


def _expected_cells():
    return frontier.expected_cells()


def _preflight_source_hashes():
    values = _source_hashes()
    values.pop("experiments/v2_extra_data_train.py", None)
    return values


def _load_training_seal(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    seal = json.loads(path.read_text())
    fits = seal.get("fits", [])
    if seal.get("all_18_complete") is not True:
        raise RuntimeError("Extra-data fit seal is not complete")
    if seal.get("study_id") != STUDY:
        raise RuntimeError("Training seal belongs to another study")
    training_decl = json.loads(TRAIN_DECL.read_text())
    if seal.get("declaration_sha256") != sha(TRAIN_DECL):
        raise RuntimeError("Training seal does not bind the frozen training declaration")
    arms = sorted({row["arm"] for row in fits})
    declared_arms = sorted(json.loads(PLAN.read_text())["arms"])
    if arms != declared_arms:
        raise RuntimeError("Fit seal arms differ from the parent plan")
    expected = {(arm, fold, seed) for arm in arms for fold in balanced.runtime.frozen.audit.FOLD.values()
                for seed in SEEDS}
    # Normalize provider names from FOLD values into canonical trainer fold names.
    expected = {(arm, f"leave_out_{fold}", seed) for arm in arms
                for fold in set(balanced.runtime.frozen.audit.FOLD.values()) for seed in SEEDS}
    found = {(row.get("arm"), row.get("fold"), int(row.get("seed", -1))) for row in fits}
    if len(arms) != 3 or found != expected or len(fits) != 18:
        raise RuntimeError("Expected exactly three arms × 3 provider folds × 2 paired seeds")
    inventory = {}
    declaration_sha = seal.get("declaration_sha256")
    if not declaration_sha:
        raise RuntimeError("Fit seal is missing its declaration binding")
    for row in fits:
        if int(row.get("stage", -1)) != 1280 or int(row.get("stage_updates", -1)) != 256:
            raise RuntimeError("Extra-data fits must be stage 1280 with exactly 256 continuation updates")
        path = ROOT / row["path"]
        receipt_path = Path(str(path) + ".json")
        if not path.is_file() or sha(path) != row.get("sha256"):
            raise RuntimeError(f"Sealed checkpoint changed: {row.get('arm')}/{row.get('fold')}/{row.get('seed')}")
        if row.get("declaration_sha256") != declaration_sha or row.get("d_state_unchanged") is not True:
            raise RuntimeError("Fit row does not bind the training declaration")
        if not receipt_path.is_file() or json.loads(receipt_path.read_text()) != row:
            raise RuntimeError("Fit sidecar differs from sealed row")
        provider = next((provider for provider, fold in balanced.runtime.frozen.audit.FOLD.items()
                         if row["fold"] == f"leave_out_{fold}"), None)
        if provider is None:
            raise RuntimeError(f"Unknown whole-provider fold: {row['fold']}")
        teacher_path = balanced.runtime.frozen._teacher_path(provider, int(row["seed"]))
        if row.get("teacher_sha256") != sha(teacher_path):
            raise RuntimeError(f"Paired teacher hash mismatch: {row['fold']}/{row['seed']}")
        key = f"{row['arm']}/{row['fold']}/{int(row['seed'])}"
        inventory[key] = {"path": str(path.relative_to(ROOT)), "sha256": row["sha256"],
                          "receipt_path": str(receipt_path.relative_to(ROOT)) if receipt_path.is_file() else None,
                          "receipt_sha256": sha(receipt_path) if receipt_path.is_file() else None,
                          "arm": row["arm"], "fold": row["fold"], "seed": int(row["seed"]),
                          "teacher_sha256": row.get("teacher_sha256")}
    return arms, inventory, seal


def _control_facts():
    declaration = json.loads(FRONTIER_DECL.read_text())
    controls = declaration["controls"]
    if set(controls) != _expected_cells():
        raise RuntimeError("Frontier declaration is missing members of the frozen 35-cell cohort")
    for key, fact in controls.items():
        for path_key, hash_key in (("path", "sha256"), ("prepared_path", "prepared_sha256")):
            path = ROOT / fact[path_key]
            if not path.is_file() or sha(path) != fact[hash_key]:
                raise RuntimeError(f"Frozen frontier input/cache changed: {key}/{path_key}")
    return controls


def _preflight_model_facts():
    balanced_train.check(full=True)
    seal = json.loads(balanced_train.TRAIN_SEAL.read_text())
    if seal.get("all_six_complete") is not True:
        raise RuntimeError("Balanced stage1024 fold checkpoints are not sealed")
    files = {}
    for provider, fold_name in balanced.runtime.frozen.audit.FOLD.items():
        fold = f"leave_out_{fold_name}"
        for seed in SEEDS:
            model = balanced_train.output_path(fold, seed)
            rows = [r for r in seal["fits"] if r.get("fold") == fold and int(r.get("seed", -1)) == seed]
            if len(rows) != 1 or not model.is_file() or sha(model) != rows[0].get("sha256"):
                raise RuntimeError(f"Missing balanced stage1024 checkpoint: {provider}/{seed}")
            teacher = balanced.runtime.frozen._teacher_path(provider, seed)
        files[f"model/{provider}/{seed}"] = {"path": str(model.relative_to(ROOT)), "sha256": sha(model),
                "teacher_path": str(teacher), "teacher_sha256": sha(teacher),
                "fold": fold}
    return files


def freeze_preflight():
    if PREFLIGHT_DECL.exists() or PREFLIGHT_RESULT.exists():
        raise RuntimeError("Preflight artifacts already exist; preserve them")
    if not PLAN.is_file() or json.loads(PLAN.read_text()).get("study_id") != STUDY:
        raise RuntimeError("Parent plan is missing or mismatched")
    controls = _control_facts()
    model_facts = _preflight_model_facts()
    frontier_result = json.loads(FRONTIER_RESULT.read_text())
    value = {"study_id": STUDY, "status": "frozen_before_extra_data_fits_and_candidate_scoring",
        "control_recipe": {"recipe_id": CONTROL_RECIPE_ID, "ce_weight": .5,
            "calibrated_weight": .375, "adopted_weight": .125,
            "pooling": "arithmetic", "ha_cost": 3.0},
        "cohort": sorted(controls), "controls": controls,
        "stage1024_models": model_facts,
        "expected_aggregate": frontier_result["aggregate"][CONTROL_RECIPE_ID],
        "published_spell_targets": CONTROL_EXPECTED, "tolerance": TOLERANCE,
        "representative_cell": "Metrica|metrica_3|clean|0",
        "source_sha256": _preflight_source_hashes(), "public_reference_access": False,
        "candidate_scoring": False, "promotion": False}
    _write_once(PREFLIGHT_DECL, value)
    return value


def require_preflight():
    declaration = json.loads(PREFLIGHT_DECL.read_text())
    if declaration.get("source_sha256") != _preflight_source_hashes():
        raise RuntimeError("Preflight source changed after freeze")
    if declaration.get("controls") != _control_facts() or declaration.get("stage1024_models") != _preflight_model_facts():
        raise RuntimeError("Preflight inputs or stage1024 checkpoints changed")
    return declaration


def _infer_stage1024(prepared, provider, seed):
    frozen = balanced.runtime.frozen
    fold = f"leave_out_{frozen.audit.FOLD[provider]}"
    model = joblib.load(balanced_train.output_path(fold, seed))
    teacher = joblib.load(frozen._teacher_path(provider, seed))
    outputs = []
    for record in prepared["phases"]:
        infer = frozen.old.inference_record(record)
        infer["names"] = list(infer["names"])
        sampled = frozen.sn.predict(infer, model, teacher)
        outputs.append(frozen.old.upsample(sampled, infer, frozen.tc.inputs_only(prepared["view"])))
    return outputs


def run_preflight():
    declaration = require_preflight()
    if PREFLIGHT_RESULT.exists():
        raise RuntimeError("Preflight result already exists")
    # Verify the true phase inference path for both paired seeds on one frozen cell.
    representative = declaration["representative_cell"]
    fact = declaration["controls"][representative]
    prepared_path = ROOT / fact["prepared_path"]
    prepared = joblib.load(prepared_path)
    phase_checks = {}
    cache_fact = json.loads(balanced_r2.RESULT.read_text())["prediction_cache"][representative]
    with np.load(ROOT / cache_fact["path"], allow_pickle=False) as cache:
        for seed in SEEDS:
            inferred = _infer_stage1024(prepared, "Metrica", seed)
            expected = [cache[f"phase_ce_seed{seed}_{i}"] for i in range(int(cache["identity__phase_count"].item()))]
            if len(inferred) != len(expected):
                raise RuntimeError("Representative native phase count changed")
            for i, (actual, target) in enumerate(zip(inferred, expected)):
                error = float(np.max(np.abs(np.asarray(actual, dtype=np.float64) - target)))
                phase_checks[f"{seed}/{i}"] = {"max_abs_error": error, "passed": error <= TOLERANCE}
    # Recompute the fixed frontier control over the full cached cohort before any new fit is scored.
    cell_rows = {}
    for key, cell_fact in sorted(declaration["controls"].items()):
        control = frontier.load_pooled_components(key)
        decoded = balanced.runtime.frozen.audit.decode(control["adopted"], control["model_period"],
            control["fps"], balanced.runtime.frozen.audit.CONFIG)
        probability = balanced.recipe_probability(control, .5, .375, "arithmetic")
        prediction = balanced.decode_fixed_d(probability, decoded, control["model_period"],
            control["fps"], 3.0)
        cell = {k: control[k] for k in ("labels", "period", "timestamps", "fps")}
        cell_rows[key] = balanced.controls.audit.score(cell, prediction)
    aggregate = balanced.aggregate_cell_metrics(cell_rows)
    checks = {}
    expected = declaration["expected_aggregate"]
    for view, _ in VIEWS:
        for metric in METRICS:
            actual = aggregate[view][metric]["equal_provider"]
            target = expected[view][metric]["equal_provider"]
            checks[f"{view}/{metric}"] = {"actual": actual, "expected": target,
                "absolute_error": abs(actual-target), "passed": abs(actual-target) <= TOLERANCE}
    for view, target in CONTROL_EXPECTED.items():
        actual = aggregate[view]["spell_f1"]["equal_provider"]
        checks[f"published/{view}/spell_f1"] = {"actual": actual, "expected": target,
            "absolute_error": abs(actual-target), "passed": abs(actual-target) <= TOLERANCE}
    passed = all(check["passed"] for check in checks.values()) and all(x["passed"] for x in phase_checks.values())
    receipt = {"study_id": STUDY, "declaration_sha256": sha(PREFLIGHT_DECL),
        "control_recipe": CONTROL_RECIPE_ID, "representative_phase_checks": phase_checks,
        "cell_metrics": cell_rows, "aggregate": aggregate, "checks": checks,
        "tolerance": TOLERANCE, "passed": passed, "candidate_scoring": False,
        "public_reference_access": False}
    _write_once(PREFLIGHT_RESULT, receipt)
    if not passed:
        raise RuntimeError("Frozen stage1024 inference or full-cohort control replay failed")
    return receipt


def freeze(fit_seal_path: Path):
    fit_seal_path = Path(fit_seal_path)
    if not fit_seal_path.is_absolute():
        fit_seal_path = ROOT / fit_seal_path
    if any(path.exists() for path in (DECLARATION, REPLAY, RESULT)):
        raise RuntimeError("Extra-data study artifacts already exist")
    plan = json.loads(PLAN.read_text())
    if plan.get("study_id") != STUDY:
        raise RuntimeError("Parent declaration is missing or mismatched")
    if not FRONTIER_CONTROL_REPLAY.is_file() or json.loads(FRONTIER_CONTROL_REPLAY.read_text()).get("passed") is not True:
        raise RuntimeError("Exact frontier adopted/control replay is required")
    if not PREFLIGHT_RESULT.is_file() or json.loads(PREFLIGHT_RESULT.read_text()).get("passed") is not True:
        raise RuntimeError("Run the frozen source/cache preflight before candidate fit freeze")
    if fit_seal_path.resolve() != TRAIN_SEAL.resolve():
        raise RuntimeError("Use the parent-declared extra-data training seal path")
    arms, fits, seal = _load_training_seal(fit_seal_path)
    controls = _control_facts()
    frontier_result = json.loads(FRONTIER_RESULT.read_text())
    selection = json.loads(FRONTIER_SELECTION.read_text())
    value = {"study_id": STUDY, "status": "frozen_before_extra_data_candidate_scoring",
        "arms": arms, "fit_seal_path": str(fit_seal_path.relative_to(ROOT)),
        "fit_seal_sha256": sha(fit_seal_path), "fit_declaration_sha256": seal["declaration_sha256"],
        "fit_inventory": fits, "cohort": {"cells": 35, "seeds": list(SEEDS),
        "views": [view for view, _ in VIEWS]},
        "recipe": {"recipe_id": CONTROL_RECIPE_ID, "ce_weight": .5,
            "calibrated_weight": .375, "adopted_weight": .125,
            "pooling": "arithmetic", "ha_cost": 3.0},
        "controls": controls,
        "frontier_control_expected": CONTROL_EXPECTED,
        "frontier_result_sha256": sha(FRONTIER_RESULT),
        "frontier_selection_sha256": sha(FRONTIER_SELECTION),
        "frontier_control_replay_sha256": sha(FRONTIER_CONTROL_REPLAY),
        "preflight_declaration_sha256": sha(PREFLIGHT_DECL),
        "preflight_result_sha256": sha(PREFLIGHT_RESULT),
        "tolerances": {"frontier_control_replay": TOLERANCE, "macro_allowance": MACRO_ALLOWANCE},
        "criteria": {"spell": "at least frontier control each view; strictly above in v2",
            "macro": "at least frontier control - .001 each view",
            "short_spell_recall_iou_0_5_lt3s": "at least frontier control each view",
            "fixed_d_and_reference_counts": "exactly unchanged"},
        "source_sha256": _source_hashes(), "organizer_reference_access": False,
        "public_scoring": False, "promotion": False}
    _write_once(DECLARATION, value)
    return value


def require_declaration():
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze extra-data evaluation before use")
    declaration = json.loads(DECLARATION.read_text())
    if declaration["source_sha256"] != _source_hashes():
        raise RuntimeError("Extra-data source changed after freeze")
    seal = ROOT / declaration["fit_seal_path"]
    arms, inventory, _ = _load_training_seal(seal)
    if sha(seal) != declaration["fit_seal_sha256"] or inventory != declaration["fit_inventory"]:
        raise RuntimeError("Training fit seal or checkpoints changed")
    if (sha(PREFLIGHT_DECL) != declaration["preflight_declaration_sha256"] or
            sha(PREFLIGHT_RESULT) != declaration["preflight_result_sha256"] or
            json.loads(PREFLIGHT_RESULT.read_text()).get("passed") is not True):
        raise RuntimeError("Frozen control preflight changed or failed")
    if _control_facts() != declaration["controls"]:
        raise RuntimeError("Frozen 35-cell controls changed")
    return declaration


def _load_cell(key, fact):
    control = balanced._load_control_arrays(fact)
    prepared_path = ROOT / fact["prepared_path"]
    prepared = joblib.load(prepared_path)
    if prepared.get("identity") != fact["identity"]:
        raise RuntimeError(f"Prepared phase cache identity mismatch: {key}")
    return prepared, control


def _pool(phases_by_seed):
    phase_counts = {len(rows) for rows in phases_by_seed.values()}
    if set(phases_by_seed) != set(SEEDS) or len(phase_counts) != 1:
        raise ValueError("Need paired seeds with matching phase counts")
    count = next(iter(phase_counts))
    pooled = np.mean([np.mean([phases_by_seed[seed][phase] for seed in SEEDS], axis=0)
                      for phase in range(count)], axis=0)
    per_seed = {seed: np.mean(phases_by_seed[seed], axis=0) for seed in SEEDS}
    return pooled, per_seed


def replay_frontier_control():
    declaration = require_declaration()
    if REPLAY.exists() or RESULT.exists():
        raise RuntimeError("Extra-data replay/result already exists")
    preflight = json.loads(PREFLIGHT_RESULT.read_text())
    if (preflight.get("passed") is not True or
            sha(PREFLIGHT_RESULT) != declaration["preflight_result_sha256"] or
            sha(PREFLIGHT_DECL) != declaration["preflight_declaration_sha256"]):
        raise RuntimeError("Pre-fit control replay changed or failed")
    receipt = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
        "preflight_result_sha256": sha(PREFLIGHT_RESULT), "candidate_scoring": False,
        "aggregate": preflight["aggregate"], "cell_metrics": preflight["cell_metrics"],
        "checks": preflight["checks"], "passed": True,
        "reference_spell_counts": {key: row["reference_spell_count"]
                                    for key, row in preflight["cell_metrics"].items()},
        "tolerance": TOLERANCE}
    _write_once(REPLAY, receipt)
    if not receipt["passed"]:
        raise RuntimeError("Frontier control replay failed the pre-candidate 1e-12 guard")
    return receipt


def _teacher(provider, seed, expected_hash):
    path = balanced.runtime.frozen._teacher_path(provider, seed)
    if sha(path) != expected_hash:
        raise RuntimeError(f"Teacher hash differs from the fit receipt: {provider}/{seed}")
    return joblib.load(path)


def _candidate_phase_outputs(prepared, provider, arm, seed, fit_fact, teacher_hash):
    fold = f"leave_out_{balanced.runtime.frozen.audit.FOLD[provider]}"
    model_path = ROOT / fit_fact[f"{arm}/{fold}/{seed}"]["path"]
    model = joblib.load(model_path)
    teacher = _teacher(provider, seed, teacher_hash)
    frozen = balanced.runtime.frozen
    output = []
    for record in prepared["phases"]:
        infer = frozen.old.inference_record(record)
        infer["names"] = list(infer["names"])
        sampled = frozen.sn.predict(infer, model, teacher)
        output.append(frozen.old.upsample(sampled, infer, frozen.tc.inputs_only(prepared["view"])))
    return output


def _cell_job(job):
    key, fact, arms, fit_facts, recipe = job
    prepared, control = _load_cell(key, fact)
    provider = key.split("|", 1)[0]
    decoded = balanced.runtime.frozen.audit.decode(control["adopted"], control["model_period"],
        control["fps"], balanced.runtime.frozen.audit.CONFIG)
    scorecell = {k: control[k] for k in ("labels", "period", "timestamps", "fps")}
    results, seed_rows, probabilities, predictions, seed_predictions = {}, {}, {}, {}, {}
    for arm in arms:
        phase_by_seed = {seed: _candidate_phase_outputs(prepared, provider, arm, seed, fit_facts,
            fit_facts[f"{arm}/leave_out_{balanced.runtime.frozen.audit.FOLD[provider]}/{seed}"]["teacher_sha256"])
            for seed in SEEDS}
        pooled, per_seed = _pool(phase_by_seed)
        mix = {"ce": pooled, "calibrated": control["calibrated"], "adopted": control["adopted"]}
        probability = balanced.recipe_probability(mix, recipe["ce_weight"], recipe["calibrated_weight"], recipe["pooling"])
        prediction = balanced.decode_fixed_d(probability, decoded, control["model_period"], control["fps"], recipe["ha_cost"])
        results[arm] = balanced.controls.audit.score(scorecell, prediction)
        probabilities[f"pooled__{arm}"] = pooled
        predictions[f"pred__{arm}"] = balanced._encode_prediction(prediction)
        for seed in SEEDS:
            seed_mix = {"ce": per_seed[seed], "calibrated": control[f"adop_seed{seed}"],
                        "adopted": control[f"cand_seed{seed}"]}
            seed_probability = balanced.recipe_probability(seed_mix, recipe["ce_weight"], recipe["calibrated_weight"], recipe["pooling"])
            seed_prediction = balanced.decode_fixed_d(seed_probability, decoded, control["model_period"], control["fps"], recipe["ha_cost"])
            seed_rows[f"{arm}|{seed}"] = balanced.controls.audit.score(scorecell, seed_prediction)
            seed_predictions[f"seedpred__{arm}|{seed}"] = balanced._encode_prediction(seed_prediction)
        for seed in SEEDS:
            for phase, p in enumerate(phase_by_seed[seed]):
                probabilities[f"phase__{arm}__{seed}__{phase}"] = p
    return key, results, seed_rows, probabilities, predictions, seed_predictions


def _worker_init():
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except ImportError:
        pass


def run():
    declaration = require_declaration()
    replay = json.loads(REPLAY.read_text()) if REPLAY.is_file() else {}
    if replay.get("passed") is not True:
        raise RuntimeError("Replay frontier control before candidate scoring")
    if RESULT.exists():
        raise RuntimeError("Extra-data scoring is one-shot")
    fits, arms, recipe = declaration["fit_inventory"], declaration["arms"], declaration["recipe"]
    jobs = [(key, fact, arms, fits, recipe) for key, fact in sorted(declaration["controls"].items())]
    cell_rows, seed_rows, cache = {}, {}, {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs)), initializer=_worker_init) as pool:
        futures = [pool.submit(_cell_job, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            key, rows, seeds, probs, preds, seedpreds = future.result()
            cell_rows[key], seed_rows[key] = rows, seeds
            cache[key] = (probs, preds, seedpreds)
            print(f"EXTRA-DATA {n}/{len(jobs)} {key}", flush=True)
    if set(cell_rows) != _expected_cells() or any(set(row) != set(arms) for row in cell_rows.values()):
        raise RuntimeError("Extra-data evaluation did not cover all arm/cell pairs")
    aggregate = {arm: balanced.aggregate_cell_metrics({key: row[arm] for key, row in cell_rows.items()}) for arm in arms}
    baseline = replay["aggregate"]
    refs = replay["reference_spell_counts"]
    gates = {}
    for arm in arms:
        values = {view: {m: aggregate[arm][view][m]["equal_provider"] for m in METRICS} for view, _ in VIEWS}
        control = {view: {m: baseline[view][m]["equal_provider"] for m in METRICS} for view, _ in VIEWS}
        checks = {}
        for view, _ in VIEWS:
            for metric, cmp in (("spell_f1", "ge"), ("macro_f1", "macro"),
                                ("short_spell_recall_iou_0_5_lt3s", "ge")):
                floor = control[view][metric] - MACRO_ALLOWANCE if cmp == "macro" else control[view][metric]
                actual = values[view][metric]
                passed = actual >= floor and (view != "feed_like_v2" or metric != "spell_f1" or actual > floor)
                checks[f"{view}/{metric}"] = {"actual": actual, "minimum": floor, "passed": passed}
        d_same = all(values[v][m] == control[v][m] for v, _ in VIEWS for m in ("d_f1", "d_precision", "d_recall"))
        refs_same = all(cell_rows[k][arm]["reference_spell_count"] == refs[k] for k in cell_rows)
        gates[arm] = {"checks": checks, "d_metrics_unchanged": d_same,
                      "reference_spell_counts_invariant": refs_same,
                      "eligible": all(x["passed"] for x in checks.values()) and d_same and refs_same}
    historical = json.loads(balanced.CONTROL_REPLAY.read_text())["adopted"]
    historical_gates = {}
    for arm in arms:
        values = {view: {m: aggregate[arm][view][m]["equal_provider"] for m in METRICS} for view, _ in VIEWS}
        base = {view: {m: historical[view][m]["equal_provider"] for m in METRICS} for view, _ in VIEWS}
        historical_gates[arm] = balanced.eligibility(values, base)
    inventories = {}
    for key, (probs, preds, seedpreds) in cache.items():
        path = OUT / "predictions" / f"{key.replace('|', '__')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        identity = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "cell": key,
                    "arm_order": arms, "seed_order": list(SEEDS)}
        np.savez_compressed(path, **probs, **preds, **seedpreds,
            **{f"identity__{k}": np.asarray(json.dumps(v) if isinstance(v, list) else str(v)) for k, v in identity.items()})
        inventories[key] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "identity": identity}
    seed_aggregate = {arm: {str(seed): balanced.aggregate_cell_metrics({
        key: {m: row[f"{arm}|{seed}"][m] for m in METRICS} for key, row in seed_rows.items()}) for seed in SEEDS} for arm in arms}
    result = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "arms": arms,
        "recipe": declaration["recipe"], "aggregate": aggregate, "eligibility": gates,
        "historical_baseline_gates": historical_gates, "cell_metrics": cell_rows,
        "seed_cells": seed_rows, "seed_diagnostics": seed_aggregate,
        "prediction_cache": inventories, "control_replay_sha256": sha(REPLAY),
        "cohort": {"cell_count": len(cell_rows), "reference_spell_counts": refs},
        "organizer_reference_access": False, "public_scoring": False, "promotion": False}
    _write_once(RESULT, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight-freeze", "preflight-run", "freeze", "replay-control", "run"))
    parser.add_argument("--fit-seal", type=Path)
    args = parser.parse_args()
    if args.action == "preflight-freeze":
        freeze_preflight()
    elif args.action == "preflight-run":
        run_preflight()
    elif args.action == "freeze":
        if args.fit_seal is None:
            parser.error("freeze requires --fit-seal")
        freeze(args.fit_seal)
    elif args.action == "replay-control":
        replay_frontier_control()
    else:
        run()


if __name__ == "__main__":
    main()
