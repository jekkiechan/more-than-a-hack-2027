"""Fixed, cached probability mixtures of the sealed extra-data arms.

The four recipes are declared before replay or scoring. This module performs no
model inference or training; it reuses only the native probability caches from
the completed extra-data study and the frozen fixed-D decoder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_extra_data_eval as extra
from experiments import v2_spell_balanced_eval as balanced

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_extra_data_mix_v1"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
REPLAY = ROOT / f"experiments/results/{STUDY}_control_replay.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
PRIOR_DECL = extra.DECLARATION
PRIOR_RESULT = extra.RESULT
PRIOR_REPLAY = extra.REPLAY
TOLERANCE = 1e-12
SEEDS = extra.SEEDS
VIEWS, METRICS = balanced.VIEWS, balanced.METRICS
QUALITY_FRACTIONS = (.25, .5, .75)
BASE_RECIPE = {"ce_weight": .5, "calibrated_weight": .375,
               "adopted_weight": .125, "pooling": "arithmetic", "ha_cost": 3.0}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def recipes():
    result = []
    for quality in QUALITY_FRACTIONS:
        result.append({"recipe_id": f"fresh_q{int(quality * 100)}",
                       "components": {"gar_quality50": quality, "fresh_control": 1-quality}})
    result.append({"recipe_id": "weight_q50",
                   "components": {"gar_quality50": .5, "gar_weight625": .5}})
    return result


RECIPES = recipes()
CONTROL_RECIPES = {
    "fresh_control": {"recipe_id": "fresh_control", "components": {"fresh_control": 1.0}},
    "gar_quality50": {"recipe_id": "gar_quality50", "components": {"gar_quality50": 1.0}},
    "gar_weight625": {"recipe_id": "gar_weight625", "components": {"gar_weight625": 1.0}},
}


def _expected_cells():
    return set(extra._expected_cells())


def _validate_plan(plan):
    declared = plan.get("recipes", {})
    expected = {row["recipe_id"]: row["components"] for row in RECIPES}
    if declared != expected or plan.get("budget", {}).get("candidate_recipes") != 4:
        raise RuntimeError("Parent plan differs from the fixed four-recipe grid")
    outer = plan.get("outer_recipe", {})
    if any((float(outer.get(key, -1)) != value if isinstance(value, float)
            else outer.get(key) != value) for key, value in BASE_RECIPE.items()):
        raise RuntimeError("Parent plan differs from the fixed outer recipe")


def _source_hashes():
    frozen = balanced.runtime.frozen
    paths = [Path(__file__), Path(extra.__file__), Path(balanced.__file__),
             Path(frozen.__file__), Path(frozen.audit.__file__),
             Path(frozen.audit.decoder.__file__), Path(frozen.audit.metrics.__file__),
             Path(frozen.supervised.__file__), Path(frozen.old.__file__),
             Path(frozen.tc.__file__), Path(frozen.sn.__file__),
             Path(frozen.sn.cap.__file__), Path(frozen.sn.fh.__file__), Path(frozen.sn.lt.__file__),
             ROOT / "experiments/refine_features.py", ROOT / "experiments/learned_temporal.py",
             ROOT / "experiments/soccernet_capacity_model.py", ROOT / "evaluation/possession_metrics.py",
             ROOT / "experiments/tests/test_v2_extra_data_mix_eval.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def _input_facts():
    prior = json.loads(PRIOR_RESULT.read_text())
    decl = json.loads(PRIOR_DECL.read_text())
    plan = json.loads(PLAN.read_text())
    if plan.get("parent_result_sha256") != sha(PRIOR_RESULT):
        raise RuntimeError("Parent plan does not bind the completed extra-data result")
    if prior.get("study_id") != extra.STUDY or prior.get("declaration_sha256") != sha(PRIOR_DECL):
        raise RuntimeError("Prior extra-data result does not bind its declaration")
    if prior.get("control_replay_sha256") != sha(PRIOR_REPLAY) or json.loads(PRIOR_REPLAY.read_text()).get("passed") is not True:
        raise RuntimeError("Prior extra-data control replay is absent or failed")
    if set(prior.get("prediction_cache", {})) != _expected_cells():
        raise RuntimeError("Prior extra-data cache does not cover the exact 35-cell cohort")
    facts = {}
    for cell, item in prior["prediction_cache"].items():
        path = ROOT / item["path"]
        if not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError(f"Prior probability cache changed: {cell}")
        facts[cell] = {"path": item["path"], "sha256": item["sha256"], "identity": item["identity"]}
    controls = decl["controls"]
    if set(controls) != _expected_cells():
        raise RuntimeError("Prior frozen controls do not cover 35 cells")
    for cell, fact in controls.items():
        path = ROOT / fact["path"]
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise RuntimeError(f"Prior native control changed: {cell}")
    return facts, controls, prior


def freeze():
    if any(p.exists() for p in (DECLARATION, REPLAY, RESULT)):
        raise RuntimeError("Study outputs already exist; preserve this study ID")
    plan = json.loads(PLAN.read_text())
    if plan.get("study_id") != STUDY:
        raise RuntimeError("Parent plan is missing or mismatched")
    _validate_plan(plan)
    input_facts, control_facts, prior = _input_facts()
    value = {"study_id": STUDY, "status": "frozen_before_control_replay_and_candidate_scoring",
        "plan_sha256": sha(PLAN), "recipes": RECIPES, "recipe_count": 4,
        "controls_to_replay": list(CONTROL_RECIPES), "control_tolerance": TOLERANCE,
        "base_recipe": BASE_RECIPE,
        "cohort": {"cells": 35, "seeds": list(SEEDS), "views": [v for v, _ in VIEWS],
                   "aggregation": "realizations within match, matches within provider, equal providers"},
        "criteria": {"spell": "at least frontier control on clean/v1/v2; v2 strictly greater",
                     "macro": "at least frontier control minus .001 in each view",
                     "short_spell_recall_iou_0_5_lt3s": "at least frontier control in each view",
                     "fixed_d_and_reference_counts": "exactly unchanged"},
        "frontier_control": extra.CONTROL_EXPECTED,
        "upstream": {"declaration_sha256": sha(PRIOR_DECL), "result_sha256": sha(PRIOR_RESULT),
                     "replay_sha256": sha(PRIOR_REPLAY), "prediction_cache": input_facts,
                     "control_inputs": control_facts},
        "source_sha256": _source_hashes(), "organizer_reference_access": False,
        "public_scoring": False, "training_or_inference": False, "promotion": False}
    if len(RECIPES) != 4 or len({r["recipe_id"] for r in RECIPES}) != 4:
        raise RuntimeError("Recipe inventory mismatch")
    _write_once(DECLARATION, value)
    return value


def require_declaration():
    declaration = json.loads(DECLARATION.read_text())
    if declaration["source_sha256"] != _source_hashes():
        raise RuntimeError("Mixture evaluator source changed after freeze")
    facts, controls, _ = _input_facts()
    if facts != declaration["upstream"]["prediction_cache"] or controls != declaration["upstream"]["control_inputs"]:
        raise RuntimeError("Frozen probability cache or control inventory changed")
    if sha(PLAN) != declaration["plan_sha256"]:
        raise RuntimeError("Parent plan changed after freeze")
    _validate_plan(json.loads(PLAN.read_text()))
    if (sha(PRIOR_DECL) != declaration["upstream"]["declaration_sha256"] or
            sha(PRIOR_RESULT) != declaration["upstream"]["result_sha256"] or
            sha(PRIOR_REPLAY) != declaration["upstream"]["replay_sha256"]):
        raise RuntimeError("Bound prior study outputs changed")
    return declaration


def _load_cache(cell, fact):
    path = ROOT / fact["path"]
    with np.load(path, allow_pickle=False) as src:
        identity = {k.removeprefix("identity__"): str(src[k].item())
                    for k in src.files if k.startswith("identity__")}
        expected = {k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in fact["identity"].items()}
        if identity != expected:
            raise RuntimeError(f"Cache identity mismatch: {cell}")
        return {k: src[k].copy() for k in src.files if not k.startswith("identity__")}


def _combine(cache, weights, suffix=""):
    arrays = [float(weight) * cache[f"pooled__{arm}{suffix}"] for arm, weight in weights.items()]
    if not arrays:
        raise ValueError("Empty mixture")
    return np.sum(arrays, axis=0)


def _score_cell(cell, cache, control, recipe, decoded):
    components = recipe.get("components", {"fresh_control": 1.0})
    ce = _combine(cache, components)
    probability = balanced.recipe_probability({"ce": ce, "calibrated": control["calibrated"],
        "adopted": control["adopted"]}, .5, .375, "arithmetic")
    pred = balanced.decode_fixed_d(probability, decoded, control["model_period"], control["fps"], 3.0)
    scorecell = {k: control[k] for k in ("labels", "period", "timestamps", "fps")}
    metrics = balanced.controls.audit.score(scorecell, pred)
    seed_rows, seed_predictions, seed_probabilities = {}, {}, {}
    for seed in SEEDS:
        per_seed = {}
        phase_counts = set()
        for arm in components:
            phase_keys = sorted((key for key in cache if key.startswith(f"phase__{arm}__{seed}__")),
                                key=lambda key: int(key.rsplit("__", 1)[1]))
            if not phase_keys:
                raise RuntimeError(f"Missing native phase probabilities for {arm}/{seed}")
            phase_counts.add(len(phase_keys))
            rows = [cache[key] for key in phase_keys]
            per_seed[arm] = np.mean(rows, axis=0)
        if len(phase_counts) != 1:
            raise RuntimeError(f"Mixture arms have different native phase counts for seed {seed}")
        expected_count = round(float(control["fps"]) / 5.0)
        if phase_counts != {expected_count}:
            raise RuntimeError(f"Cached native phase count differs from FPS for seed {seed}")
        seed_ce = sum(weight * per_seed[arm] for arm, weight in components.items())
        seed_mix = balanced.recipe_probability({"ce": seed_ce,
            "calibrated": control[f"adop_seed{seed}"], "adopted": control[f"cand_seed{seed}"]},
            .5, .375, "arithmetic")
        seed_pred = balanced.decode_fixed_d(seed_mix, decoded, control["model_period"], control["fps"], 3.0)
        seed_rows[str(seed)] = balanced.controls.audit.score(scorecell, seed_pred)
        seed_predictions[f"seedpred__{recipe['recipe_id']}__{seed}"] = balanced._encode_prediction(seed_pred)
        seed_probabilities[f"seedprob__{recipe['recipe_id']}__{seed}"] = seed_mix
    if not np.array_equal(pred == "D", decoded == "D"):
        raise RuntimeError(f"Fixed-D mask changed: {cell}/{recipe['recipe_id']}")
    return metrics, probability, pred, seed_rows, seed_probabilities, seed_predictions


def replay_controls():
    declaration = require_declaration()
    if REPLAY.exists() or RESULT.exists():
        raise RuntimeError("Control replay/result already exists")
    prior = json.loads(PRIOR_RESULT.read_text())
    rows, checks = {}, {}
    for cell, fact in sorted(declaration["upstream"]["prediction_cache"].items()):
        cache = _load_cache(cell, fact)
        control = balanced._load_control_arrays(declaration["upstream"]["control_inputs"][cell])
        decoded = balanced.runtime.frozen.audit.decode(control["adopted"], control["model_period"],
            control["fps"], balanced.runtime.frozen.audit.CONFIG)
        rows[cell] = {}
        for recipe_id, recipe in CONTROL_RECIPES.items():
            scores, _, pred, seed_scores, _, seed_preds = _score_cell(cell, cache, control, recipe, decoded)
            target = prior["cell_metrics"][cell][recipe_id]
            errs = {metric: abs(float(scores[metric]) - float(target[metric])) for metric in METRICS}
            pred_equal = np.array_equal(balanced._encode_prediction(pred), cache[f"pred__{recipe_id}"])
            seed_pred_equal = all(np.array_equal(seed_preds[f"seedpred__{recipe_id}__{seed}"],
                cache[f"seedpred__{recipe_id}|{seed}"]) for seed in SEEDS)
            key = f"{cell}/{recipe_id}"
            checks[key] = {"max_abs_error": max(errs.values()), "passed": max(errs.values()) <= TOLERANCE,
                           "metrics": errs, "prediction_equal": bool(pred_equal),
                           "seed_predictions_equal": bool(seed_pred_equal),
                           "d_mask_equal": bool(np.array_equal(pred == "D", decoded == "D"))}
            rows[cell][recipe_id] = scores
        print(f"MIX CONTROL {len(rows)}/35 {cell}", flush=True)
    receipt = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "checks": checks,
        "cell_metrics": rows, "checks_count": len(checks), "tolerance": TOLERANCE,
        "passed": len(checks) == 105 and all(x["passed"] and x["prediction_equal"] and
            x["seed_predictions_equal"] and x["d_mask_equal"] for x in checks.values()),
        "candidate_scoring": False, "public_scoring": False}
    _write_once(REPLAY, receipt)
    if not receipt["passed"]:
        raise RuntimeError("Prior arms did not replay exactly; no candidates scored")
    return receipt


def run():
    declaration = require_declaration()
    replay = json.loads(REPLAY.read_text())
    if replay.get("passed") is not True or replay.get("declaration_sha256") != sha(DECLARATION):
        raise RuntimeError("Complete controls replay before mixture scoring")
    if RESULT.exists():
        raise RuntimeError("Mixture scoring is one-shot")
    cell_rows, seed_rows, predcache = {}, {}, {}
    for cell, fact in sorted(declaration["upstream"]["prediction_cache"].items()):
        cached = _load_cache(cell, fact)
        control = balanced._load_control_arrays(declaration["upstream"]["control_inputs"][cell])
        decoded = balanced.runtime.frozen.audit.decode(control["adopted"], control["model_period"],
            control["fps"], balanced.runtime.frozen.audit.CONFIG)
        rows, seeds, arrays = {}, {}, {}
        for recipe in RECIPES:
            metrics, prob, pred, seed_metric, seed_prob, seed_pred = _score_cell(cell, cached, control, recipe, decoded)
            rid = recipe["recipe_id"]
            rows[rid], seeds[rid] = metrics, seed_metric
            arrays[f"prob__{rid}"] = prob
            arrays[f"pred__{rid}"] = balanced._encode_prediction(pred)
            arrays.update(seed_prob); arrays.update(seed_pred)
        cell_rows[cell], seed_rows[cell] = rows, seeds
        predcache[cell] = arrays
        print(f"MIX CANDIDATES {len(cell_rows)}/35 {cell}", flush=True)
    aggregate = {r["recipe_id"]: balanced.aggregate_cell_metrics({c: rows[r["recipe_id"]]
        for c, rows in cell_rows.items()}) for r in RECIPES}
    frontier = json.loads(PRIOR_REPLAY.read_text())["aggregate"]
    eligibility = {}
    for recipe in RECIPES:
        rid = recipe["recipe_id"]
        values = {v: {m: aggregate[rid][v][m]["equal_provider"] for m in METRICS} for v, _ in VIEWS}
        baseline = {v: {m: frontier[v][m]["equal_provider"] for m in METRICS} for v, _ in VIEWS}
        checks = {}
        for view, _ in VIEWS:
            for metric, rule in (("spell_f1", "ge"), ("macro_f1", "macro"),
                                 ("short_spell_recall_iou_0_5_lt3s", "ge")):
                floor = baseline[view][metric] - .001 if rule == "macro" else baseline[view][metric]
                actual = values[view][metric]
                passed = actual >= floor and (view != "feed_like_v2" or metric != "spell_f1" or actual > floor)
                checks[f"{view}/{metric}"] = {"actual": actual, "minimum": floor, "passed": passed}
        gate = {"checks": checks, "eligible": all(row["passed"] for row in checks.values())}
        refs = all(cell_rows[c][rid]["reference_spell_count"] ==
                   json.loads(PRIOR_REPLAY.read_text())["reference_spell_counts"][c] for c in cell_rows)
        replay_cells = json.loads(REPLAY.read_text())["cell_metrics"]
        d_same = all(cell_rows[c][rid][m] == replay_cells[c]["fresh_control"][m]
                     for c in cell_rows for m in ("d_f1", "d_precision", "d_recall"))
        gate.update({"d_metrics_unchanged": d_same, "reference_spell_counts_invariant": refs,
                     "eligible": gate["eligible"] and d_same and refs})
        eligibility[rid] = gate
    seed_aggregate = {r["recipe_id"]: {str(seed): balanced.aggregate_cell_metrics({c:
        {m: rows[r["recipe_id"]][str(seed)][m] for m in METRICS} for c, rows in seed_rows.items()})
        for seed in SEEDS} for r in RECIPES}
    inventory = {}
    for cell, arrays in predcache.items():
        path = OUT / "predictions" / f"{cell.replace('|', '__')}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays,
            identity__study_id=np.asarray(STUDY), identity__declaration_sha256=np.asarray(sha(DECLARATION)),
            identity__cell=np.asarray(cell), identity__recipe_ids=np.asarray(json.dumps([r["recipe_id"] for r in RECIPES])))
        inventory[cell] = {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
    ranking = sorted((r["recipe_id"] for r in RECIPES if eligibility[r["recipe_id"]]["eligible"]),
        key=lambda rid: (-aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
                         -aggregate[rid]["feed_like_v2"]["macro_f1"]["equal_provider"], rid))
    result = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "recipes": RECIPES,
        "aggregate": aggregate, "eligibility": eligibility, "eligible_ranking": ranking,
        "cell_metrics": cell_rows, "seed_cells": seed_rows, "seed_diagnostics": seed_aggregate,
        "prediction_cache": inventory, "control_replay_sha256": sha(REPLAY),
        "cohort": {"cell_count": len(cell_rows)}, "public_scoring": False,
        "organizer_reference_access": False, "promotion": False}
    _write_once(RESULT, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay-controls", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "replay-controls": replay_controls, "run": run}[args.action]()


if __name__ == "__main__":
    main()
