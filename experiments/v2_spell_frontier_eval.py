"""Fit-free search over cached stage-1024, calibrated and adopted probabilities."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import v2_spell_balanced_eval as balanced
from experiments import v2_spell_balanced_round2 as balanced_r2

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_spell_frontier_v1"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
DECLARATION = ROOT / f"experiments/results/{STUDY}_round1_declaration.json"
REPLAY = ROOT / f"experiments/results/{STUDY}_replay.json"
RESULT = ROOT / f"experiments/results/{STUDY}_round1.json"
OUT = ROOT / "data/experiments" / STUDY
CONTROL_DECL = balanced.DECLARATION
CONTROL_REPLAY = balanced.CONTROL_REPLAY
CONTROL_ROUND2 = balanced_r2.RESULT
SELECTION = ROOT / "experiments/results/v2_spell_balanced_v1_selection.json"
TOLERANCE = 1e-10
MACRO_ALLOWANCE = .001
CURRENT_BEST_V2 = .5911741181681092
SEEDS = (2026, 2027)
VIEWS = balanced.VIEWS
METRICS = balanced.METRICS
QUALITY = balanced.QUALITY
CE_WEIGHTS = (.25, .375, .5, .625, .75)
CAL_WEIGHTS = (.25, .375, .5, .625, .75)
POOLINGS = ("arithmetic", "log")
LEADER = {"ce_weight": .5, "calibrated_weight": .25, "pooling": "arithmetic", "ha_cost": 3.0}
WORKERS = min(2, max(1, int(os.environ.get("V2_SPELL_FRONTIER_WORKERS", "2"))))


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=1, sort_keys=True, default=float)
        f.write("\n")


def _float_token(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def recipe_id(recipe: Mapping) -> str:
    adopted = 1.0 - float(recipe["ce_weight"]) - float(recipe["calibrated_weight"])
    return (f"ce{_float_token(recipe['ce_weight'])}_cal{_float_token(recipe['calibrated_weight'])}"
            f"_ad{_float_token(adopted)}_{recipe['pooling']}_ha3")


def recipe_grid() -> list[dict]:
    grid = []
    for ce in CE_WEIGHTS:
        for cal in CAL_WEIGHTS:
            adopted = 1.0 - ce - cal
            if adopted < -1e-12:
                continue
            for pooling in POOLINGS:
                item = {"ce_weight": ce, "calibrated_weight": cal,
                        "adopted_weight": max(0.0, adopted), "pooling": pooling,
                        "ha_cost": 3.0}
                item["recipe_id"] = recipe_id(item)
                grid.append(item)
    return grid


RECIPES = recipe_grid()
ARM_IDS = tuple(row["recipe_id"] for row in RECIPES)
LEADER_IDS = tuple(recipe_id({**LEADER, "pooling": pool}) for pool in POOLINGS)


def expected_cells() -> set[str]:
    return {balanced._cell_key(*cell) for cell in balanced._expected_cells()}


def _source_hashes() -> dict[str, str]:
    frozen = balanced.runtime.frozen
    files = (Path(__file__), Path(balanced.__file__), Path(balanced_r2.__file__),
             Path(frozen.__file__), Path(frozen.audit.__file__),
             Path(frozen.audit.decoder.__file__), Path(frozen.audit.metrics.__file__),
             Path(frozen.supervised.__file__), Path(frozen.old.__file__),
             Path(frozen.tc.__file__), Path(frozen.sn.__file__),
             Path(frozen.sn.cap.__file__), Path(frozen.sn.fh.__file__), Path(frozen.sn.lt.__file__),
             ROOT / "experiments/refine_features.py", ROOT / "experiments/learned_temporal.py",
             ROOT / "experiments/soccernet_capacity_model.py",
             ROOT / "evaluation/possession_metrics.py", PLAN, CONTROL_DECL,
             CONTROL_REPLAY, CONTROL_ROUND2, SELECTION,
             ROOT / "experiments/tests/test_v2_spell_frontier_eval.py",
             ROOT / "experiments/baseline_manifest.json",
             ROOT / "experiments/model_decision_policy.json")
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def _inputs() -> tuple[dict, dict, dict]:
    cdecl = json.loads(CONTROL_DECL.read_text())
    replay = json.loads(CONTROL_REPLAY.read_text())
    r2 = json.loads(CONTROL_ROUND2.read_text())
    if (r2.get("study_id") != "v2_spell_balanced_v1" or
            r2.get("declaration_sha256") != sha(balanced_r2.DECLARATION)):
        raise RuntimeError("Balanced stage-1024 result does not bind its declaration")
    if replay.get("passed") is not True or replay.get("candidate_scoring") is not False:
        raise RuntimeError("Balanced adopted control replay did not pass")
    cells = expected_cells()
    control = cdecl["upstream"]["adopted_calibrated_emissions"]
    pred = r2.get("prediction_cache", {})
    if set(control) != cells or set(pred) != cells or len(cells) != 35:
        raise RuntimeError("Expected 35 balanced stage-1024 probability caches")
    for key, fact in control.items():
        p = ROOT / fact["path"]
        if not p.is_file() or sha(p) != fact["sha256"]:
            raise RuntimeError(f"Control probabilities changed: {key}")
    for key, fact in pred.items():
        p = ROOT / fact["path"]
        if not p.is_file() or sha(p) != fact["sha256"]:
            raise RuntimeError(f"Stage-1024 prediction cache changed: {key}")
    return control, pred, {"replay": replay, "round2": r2, "declaration": cdecl}


def freeze() -> dict:
    if any(p.exists() for p in (DECLARATION, REPLAY, RESULT)):
        raise RuntimeError("Frontier round-one artifacts already exist")
    plan = json.loads(PLAN.read_text())
    if plan.get("study_id") != STUDY:
        raise RuntimeError("Parent frontier plan is absent or mismatched")
    controls, predictions, upstream = _inputs()
    selection = json.loads(SELECTION.read_text())
    if selection.get("study_id") != "v2_spell_balanced_v1":
        raise RuntimeError("Fallback selection does not match the balanced study")
    selected_v2 = float(selection["aggregate"]["feed_like_v2"]["spell_f1"]["equal_provider"])
    if selected_v2 != CURRENT_BEST_V2:
        raise RuntimeError("Frozen current-best fallback metric differs from the declared frontier floor")
    value = {
        "study_id": STUDY, "round": 1,
        "status": "frozen_before_control_replay_and_frontier_scoring",
        "question": "Can a denser stage-1024 conditional H/A mixture improve spell retention across all views?",
        "recipe_grid": RECIPES, "recipe_count": len(RECIPES),
        "cohort": {"cells": 35, "views": [v for v, _ in VIEWS], "seeds": list(SEEDS)},
        "eligibility": {"spell_f1": "strictly above adopted in clean/v1/v2",
                        "macro_f1": "at least adopted - 0.001 in each view",
                        "short_spell_recall_iou_0_5_lt3s": "at least adopted in each view",
                        "d_metrics": "exactly equal adopted with fixed D mask",
                        "reference_spell_count": "identical in every cell"},
        "selection": "eligible by v2 spell F1 descending, then macro F1 descending, then recipe_id ascending",
        "seed_diagnostics": "top two eligible pooled recipes plus arithmetic current leader; max three, diagnostics never rerank",
        "leader_replay": {"recipe_ids": list(LEADER_IDS), "metrics": list(METRICS),
                          "tolerance": TOLERANCE, "required_before_grid_scoring": True},
        "fallback": {"selection_recipe": selection["recipe"], "training_stage": selection["training_stage"],
                     "v2_spell_f1": selected_v2, "comparison_only": True},
        "inputs": {"control": controls, "stage1024_probability_cache": predictions},
        "upstream": {"control_declaration_sha256": sha(CONTROL_DECL),
                     "control_replay_sha256": sha(CONTROL_REPLAY),
                     "balanced_round2_sha256": sha(CONTROL_ROUND2),
                     "selection_sha256": sha(SELECTION)},
        "source_sha256": _source_hashes(),
        "organizer_reference_access": False, "public_scoring": False, "promotion": False}
    if len(RECIPES) != 30 or len(set(ARM_IDS)) != 30:
        raise RuntimeError("Frontier grid must contain 30 unique recipes")
    _write_once(DECLARATION, value)
    return value


def require_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze frontier evaluator first")
    dec = json.loads(DECLARATION.read_text())
    if dec.get("source_sha256") != _source_hashes():
        raise RuntimeError("Frontier source changed after freeze")
    controls, predictions, _ = _inputs()
    if controls != dec["inputs"]["control"] or predictions != dec["inputs"]["stage1024_probability_cache"]:
        raise RuntimeError("Frozen probability cache inventory changed")
    if dec.get("recipe_grid") != RECIPES:
        raise RuntimeError("Frontier recipe grid changed")
    return dec


def _load_cell(key: str, control_fact: dict, prediction_fact: dict) -> dict:
    cpath = ROOT / control_fact["path"]
    ppath = ROOT / prediction_fact["path"]
    with np.load(cpath, allow_pickle=False) as c, np.load(ppath, allow_pickle=False) as p:
        identity = {k.removeprefix("identity__"): str(p[k].item())
                    for k in p.files if k.startswith("identity__")}
        if (identity.get("cell") != key or identity.get("study_id") != "v2_spell_balanced_v1" or
                identity.get("declaration_sha256") != sha(balanced_r2.DECLARATION) or
                int(identity.get("phase_count", -1)) not in (2, 5)):
            raise RuntimeError(f"Stage-1024 probability cache identity mismatch: {key}")
        control_identity = {k.removeprefix("identity__"): str(c[k].item())
                            for k in c.files if k.startswith("identity__")}
        if control_identity != {k: str(v) for k, v in control_fact["identity"].items()}:
            raise RuntimeError(f"Adopted/calibrated cache identity mismatch: {key}")
        values = {"adopted": c["prob__adopted"].copy(),
                "calibrated": c["prob__calibrated"].copy(),
                "seed_adopted": {s: c[f"prob__cand_seed{s}"].copy() for s in SEEDS},
                "seed_calibrated": {s: c[f"prob__adop_seed{s}"].copy() for s in SEEDS},
                "ce": p["pooled_ce"].copy(),
                "seed_ce": {s: p[f"seed_ce_{s}"].copy() for s in SEEDS},
                "labels": c["labels"].copy(), "period": c["period"].copy(),
                "timestamps": c["timestamps"].copy(),
                "model_period": c["model_period"].copy(), "fps": float(c["fps"])}
        lengths = {np.asarray(values[name]).shape[0] for name in ("adopted", "calibrated", "ce", "labels", "period")}
        if len(lengths) != 1 or next(iter(lengths)) != len(values["timestamps"]):
            raise RuntimeError(f"Cached phase probabilities do not align: {key}")
        for seed in SEEDS:
            if any(values[group][seed].shape != values["adopted"].shape
                   for group in ("seed_adopted", "seed_calibrated", "seed_ce")):
                raise RuntimeError(f"Seed cache shape mismatch: {key}/{seed}")
        return values


def load_pooled_components(cell_key: str) -> dict:
    """Load frozen stage-1024 pooled and paired-seed probabilities for one cell."""
    declaration = require_declaration()
    return _load_cell(cell_key, declaration["inputs"]["control"][cell_key],
                      declaration["inputs"]["stage1024_probability_cache"][cell_key])


def _score_recipe(cell: dict, recipe: dict, adopted_decoded: np.ndarray) -> tuple[dict, np.ndarray]:
    components = {"ce": cell["ce"], "calibrated": cell["calibrated"], "adopted": cell["adopted"]}
    prob = balanced.recipe_probability(components, recipe["ce_weight"],
        recipe["calibrated_weight"], recipe["pooling"])
    prediction = balanced.decode_fixed_d(prob, adopted_decoded, cell["model_period"], cell["fps"], 3.0)
    metrics = balanced.controls.audit.score({k: cell[k] for k in ("labels", "period", "timestamps", "fps")}, prediction)
    return metrics, prediction


def _worker_job(job: tuple):
    key, control_fact, pred_fact, recipes = job
    cell = _load_cell(key, control_fact, pred_fact)
    adopted_decoded = balanced.runtime.frozen.audit.decode(cell["adopted"], cell["model_period"],
        cell["fps"], balanced.runtime.frozen.audit.CONFIG)
    rows, preds = {}, {}
    for recipe in recipes:
        metrics, prediction = _score_recipe(cell, recipe, adopted_decoded)
        rows[recipe["recipe_id"]] = metrics
        preds[recipe["recipe_id"]] = balanced._encode_prediction(prediction)
    return key, rows, preds


def _aggregates(cell_rows: Mapping[str, dict], recipe_ids: list[str]) -> dict:
    return {rid: balanced.aggregate_cell_metrics({key: row[rid] for key, row in cell_rows.items()})
            for rid in recipe_ids}


def _seed_shortlist(gates: Mapping[str, dict], aggregate: Mapping[str, dict]) -> list[str]:
    ids = [rid for rid, gate in gates.items() if gate["eligible"]]
    ids.sort(key=lambda rid: (-aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
                              -aggregate[rid]["feed_like_v2"]["macro_f1"]["equal_provider"], rid))
    selected = ids[:2]
    leader = LEADER_IDS[0]
    if leader not in selected:
        selected.append(leader)
    return list(dict.fromkeys(selected))[:3]


def replay_leaders() -> dict:
    declaration = require_declaration()
    if REPLAY.exists() or RESULT.exists():
        raise RuntimeError("Frontier replay/scoring already exists")
    saved = json.loads(CONTROL_ROUND2.read_text())
    leader_rows = {rid: {} for rid in LEADER_IDS}
    for key, cf in sorted(declaration["inputs"]["control"].items()):
        cell = _load_cell(key, cf, declaration["inputs"]["stage1024_probability_cache"][key])
        adopted = balanced.runtime.frozen.audit.decode(cell["adopted"], cell["model_period"], cell["fps"],
                                                        balanced.runtime.frozen.audit.CONFIG)
        for rid in LEADER_IDS:
            recipe = next(r for r in RECIPES if r["recipe_id"] == rid)
            if rid not in saved["aggregate"]:
                continue
            metrics, _ = _score_recipe(cell, recipe, adopted)
            leader_rows[rid][key] = metrics
    checks, aggregates = {}, {}
    for rid in LEADER_IDS:
        if rid not in saved["aggregate"]:
            raise RuntimeError(f"Prior balanced run did not save leader replay arm {rid}")
        aggregates[rid] = balanced.aggregate_cell_metrics(leader_rows[rid])
        for view, _ in VIEWS:
            for metric in METRICS:
                actual = aggregates[rid][view][metric]["equal_provider"]
                expected = saved["aggregate"][rid][view][metric]["equal_provider"]
                checks[f"{rid}/{view}/{metric}"] = {"actual": actual, "expected": expected,
                    "absolute_error": abs(actual - expected), "passed": abs(actual - expected) <= TOLERANCE}
    receipt = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "controls_only": True,
               "candidate_scoring": False, "aggregate": aggregates, "checks": checks,
               "tolerance": TOLERANCE, "passed": all(x["passed"] for x in checks.values())}
    _write_once(REPLAY, receipt)
    if not receipt["passed"]:
        raise RuntimeError("Saved leader probability replay did not match to 1e-10")
    return receipt


def run_round1() -> dict:
    declaration = require_declaration()
    if not REPLAY.is_file() or json.loads(REPLAY.read_text()).get("passed") is not True:
        raise RuntimeError("Run exact leader replay before frontier candidates")
    if RESULT.exists():
        raise RuntimeError("Frontier scoring is one-shot")
    jobs = [(key, declaration["inputs"]["control"][key],
             declaration["inputs"]["stage1024_probability_cache"][key], RECIPES)
            for key in sorted(expected_cells())]
    cell_rows, pooled_predictions = {}, {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs))) as pool:
        futures = [pool.submit(_worker_job, job) for job in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            key, rows, preds = fut.result()
            cell_rows[key] = rows
            pooled_predictions[key] = preds
            print(f"FRONTIER R1 {n}/{len(jobs)} {key}", flush=True)
    if set(cell_rows) != expected_cells() or any(set(rows) != set(ARM_IDS) for rows in cell_rows.values()):
        raise RuntimeError("Frontier grid coverage mismatch")
    aggregate = _aggregates(cell_rows, list(ARM_IDS))
    baseline = json.loads(CONTROL_REPLAY.read_text())["adopted"]
    gates = {}
    refs = json.loads(CONTROL_REPLAY.read_text())["reference_spell_counts"]
    for rid in ARM_IDS:
        candidate = {view: {m: aggregate[rid][view][m]["equal_provider"] for m in METRICS}
                     for view, _ in VIEWS}
        target = {view: {m: baseline[view][m]["equal_provider"] for m in METRICS} for view, _ in VIEWS}
        gate = balanced.eligibility(candidate, target)
        gate["d_metrics_unchanged"] = all(candidate[v][m] == target[v][m]
            for v, _ in VIEWS for m in ("d_f1", "d_precision", "d_recall"))
        gate["reference_spell_counts_invariant"] = all(
            cell_rows[key][rid]["reference_spell_count"] == refs[key] for key in cell_rows)
        v2 = aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"]
        gate["beats_current_best"] = {"actual": v2, "minimum_strictly_greater": CURRENT_BEST_V2,
                                       "passed": v2 > CURRENT_BEST_V2}
        gate["eligible"] = (gate["eligible"] and gate["d_metrics_unchanged"] and
                            gate["reference_spell_counts_invariant"])
        gates[rid] = gate
    shortlist = _seed_shortlist(gates, aggregate)
    seed_rows, seed_predictions = {}, {}
    for key, cf in sorted(declaration["inputs"]["control"].items()):
        cell = _load_cell(key, cf, declaration["inputs"]["stage1024_probability_cache"][key])
        adopted = balanced.runtime.frozen.audit.decode(cell["adopted"], cell["model_period"], cell["fps"],
                                                        balanced.runtime.frozen.audit.CONFIG)
        seed_rows[key], seed_predictions[key] = {}, {}
        for rid in shortlist:
            recipe = next(r for r in RECIPES if r["recipe_id"] == rid)
            for seed in SEEDS:
                seed_cell = {**cell, "ce": cell["seed_ce"][seed],
                    "calibrated": cell["seed_calibrated"][seed], "adopted": cell["seed_adopted"][seed]}
                metrics, prediction = _score_recipe(seed_cell, recipe, adopted)
                seed_rows[key][f"{rid}|{seed}"] = metrics
                seed_predictions[key][f"{rid}|{seed}"] = balanced._encode_prediction(prediction)
    seed_aggregate = balanced.aggregate_seed_metrics(seed_rows, shortlist)
    pred_cache, seed_cache = {}, {}
    declaration_hash = sha(DECLARATION)
    for key in sorted(cell_rows):
        out = OUT / "predictions" / f"{key.replace('|', '__')}.npz"
        out.parent.mkdir(parents=True, exist_ok=True)
        pooled_identity = {"study_id": STUDY, "declaration_sha256": declaration_hash,
                           "cell": key, "recipe_order": list(ARM_IDS)}
        np.savez_compressed(out,
            **{f"pred__{rid}": pooled_predictions[key][rid] for rid in ARM_IDS},
            **{f"identity__{name}": np.asarray(json.dumps(value) if isinstance(value, (list, dict))
                                                  else str(value))
               for name, value in pooled_identity.items()})
        pred_cache[key] = {"path": str(out.relative_to(ROOT)), "sha256": sha(out),
                           "identity": pooled_identity}
        seed_path = OUT / "seed_predictions" / f"{key.replace('|', '__')}.npz"
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        seed_identity = {"study_id": STUDY, "declaration_sha256": declaration_hash,
                         "cell": key, "recipe_order": list(shortlist), "seed_order": list(SEEDS)}
        np.savez_compressed(seed_path,
            **{f"pred__{rid}|{seed}": seed_predictions[key][f"{rid}|{seed}"]
               for rid in shortlist for seed in SEEDS},
            **{f"identity__{name}": np.asarray(json.dumps(value) if isinstance(value, (list, dict))
                                                  else str(value))
               for name, value in seed_identity.items()})
        seed_cache[key] = {"path": str(seed_path.relative_to(ROOT)), "sha256": sha(seed_path),
                           "identity": seed_identity}
    ordered = sorted(ARM_IDS, key=lambda rid: (-aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
        -aggregate[rid]["feed_like_v2"]["macro_f1"]["equal_provider"], rid))
    selection = json.loads(SELECTION.read_text())
    result = {"study_id": STUDY, "round": 1, "declaration_sha256": sha(DECLARATION),
        "replay_sha256": sha(REPLAY), "recipes": RECIPES, "recipe_ids": list(ARM_IDS),
        "cohort": {"cell_count": len(cell_rows), "cell_counts": json.loads(CONTROL_REPLAY.read_text())["cell_counts"],
                   "reference_spell_counts": refs},
        "aggregate": aggregate, "eligibility": gates,
        "eligible_ranking": [rid for rid in ordered if gates[rid]["eligible"]],
        "seed_shortlist": shortlist, "seed_diagnostics": seed_aggregate, "seed_cells": seed_rows,
        "cell_metrics": cell_rows, "prediction_cache": pred_cache,
        "seed_prediction_cache": seed_cache,
        "fallback_comparison": {"selection_recipe": selection["recipe"], "training_stage": selection["training_stage"],
                                "v2_spell_f1": CURRENT_BEST_V2},
        "organizer_reference_access": False, "public_scoring": False, "promotion": False}
    _write_once(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay-leaders", "run-round1"))
    action = parser.parse_args().action
    {"freeze": freeze, "replay-leaders": replay_leaders, "run-round1": run_round1}[action]()


if __name__ == "__main__":
    main()
