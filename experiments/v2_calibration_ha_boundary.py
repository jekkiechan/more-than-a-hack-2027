"""One fixed H/A-cost boundary refinement for the v2 spell target.

This runner is conditional on the already-frozen three-arm calibration transfer
remaining below .60. It reuses only the 160 typed and sealed per-seed/phase
probability caches; it never fits a model or infers a missing cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_calibration_target as calibration
from experiments import v2_calibration_transfer as transfer
from experiments import v2_calibration_transfer_runtime as transfer_runtime
from experiments import v2_training_eval_runtime_v3 as main_runtime
from experiments import v2_spell_target_runtime as upstream


ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_calibration_ha_boundary_v1"
PLAN = ROOT / "experiments/results/v2_calibration_ha_boundary_plan.json"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
PREFLIGHT = ROOT / f"experiments/results/{STUDY}_preflight.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
HA_COSTS = (0.5, 0.75, 1.0, 1.5, 2.0)
RECIPES = (
    "archived_calibrated_phase",
    "new_full_phase_tta",
    "new_conditional_ha_x_archived_calibrated_d",
    "equal_50_50_new_full_plus_archived_calibrated",
)
SEEDS = tuple(transfer.SEEDS)
VIEWS = tuple(transfer.VIEWS)
AUDIT = upstream.frozen.audit
BASE_CONFIG = dict(calibration.ARMS[0]["config"])
BASE_CONFIG.update(dt=4, ht=1.0, dead=3)
_MAIN_RUNTIME = {}


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arms() -> list[dict]:
    return [{"id": f"{recipe}_ha{ha:g}", "recipe": recipe,
             "config": dict(BASE_CONFIG, ha=ha)}
            for recipe in RECIPES for ha in HA_COSTS]


ARMS = _arms()
ARM_IDS = tuple(arm["id"] for arm in ARMS)
CONTROL_RECIPE = {
    "archived_calibrated_at_ha_1.5": ("archived_calibrated_phase", 1.5),
    "new_blend_at_f062_ha_3": ("equal_50_50_new_full_plus_archived_calibrated", 3.0),
}


def _source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__), Path(calibration.__file__), Path(transfer.__file__),
        Path(transfer_runtime.__file__), Path(main_runtime.__file__),
        Path(main_runtime.runtime_v2.__file__), Path(transfer.base.__file__),
        Path(transfer.upstream.frozen.__file__), Path(transfer.upstream.frozen.audit.__file__),
        ROOT / "experiments/feed_spell_push.py",
        ROOT / "experiments/decoder_refinement.py",
        ROOT / "experiments/decoder_search.py",
        ROOT / "experiments/spell_mbr_run.py", ROOT / "experiments/metrics.py",
        ROOT / "evaluation/possession_metrics.py",
    )
    return {str(path.relative_to(ROOT)): _sha(path) for path in paths}


def _bindings() -> dict:
    """Validate transfer, main cache correction, runtime and model/cache seals."""
    transfer_runtime._verify_amendment()
    _, transfer_seal = transfer._require_seal()
    main_decl, main_seal = main_runtime._verify()
    _MAIN_RUNTIME["declaration"] = main_decl
    _MAIN_RUNTIME["seal"] = main_seal
    if not transfer.RESULT.is_file():
        raise FileNotFoundError("Completed fixed calibration-transfer result is required")
    transfer_result = json.loads(transfer.RESULT.read_text())
    if (transfer_result.get("study_id") != transfer.STUDY or
            transfer_result.get("declaration_sha256") != _sha(transfer.DECLARATION) or
            transfer_result.get("transfer_seal_sha256") != _sha(transfer.SEAL)):
        raise RuntimeError("Transfer result is not bound to its frozen declaration/seal")
    transfer_values = [row["feed_like_v2"]["equal_provider"]["spell_f1"]
                       for row in transfer_result["candidate_aggregate"].values()]
    if not transfer_values or max(transfer_values) >= 0.60:
        raise RuntimeError("Transfer reached .60; the conditional H/A refinement must not run")
    if transfer_result.get("candidate_arms") != list(transfer.ARMS):
        raise RuntimeError("Transfer result arm inventory changed")
    if transfer_seal.get("cache_facts", {}).get("emission_count") != 160:
        raise RuntimeError("Expected exactly 160 sealed per-seed/phase emission files")
    inventory = main_decl.get("typed_candidate_cache_inventory", {})
    if len(inventory) != 160:
        raise RuntimeError("Main v3 runtime does not bind all 160 candidate emission caches")
    return {
        "transfer_result": transfer_result,
        "transfer_result_sha256": _sha(transfer.RESULT),
        "transfer_declaration_sha256": _sha(transfer.DECLARATION),
        "transfer_seal_sha256": _sha(transfer.SEAL),
        "transfer_runtime_amendment_sha256": _sha(transfer_runtime.AMENDMENT),
        "main_evaluation_result_sha256": _sha(transfer.base.RESULT),
        "main_evaluation_declaration_sha256": _sha(transfer.base.DECLARATION),
        "main_evaluation_seal_sha256": _sha(transfer.base.SEAL),
        "main_runtime_declaration_sha256": _sha(main_runtime.DECLARATION),
        "main_runtime_typed_cache_inventory": inventory,
        "main_model_hashes": {key: value["sha256"] for key, value in main_seal["models"].items()},
        "upstream_runtime_declaration_sha256": _sha(upstream.frozen.DECLARATION),
        "upstream_runtime_preflight_sha256": _sha(upstream.frozen.PREFLIGHT),
        "calibration_result_sha256": _sha(calibration.RESULT),
        "calibration_declaration_sha256": _sha(calibration.DECLARATION),
    }


def _validate_all_emissions(bindings: dict) -> dict:
    """Validate each existing typed emission, including archived C equality."""
    _, main_seal = main_runtime._verify()
    base = transfer.base
    expected, identities = main_runtime._expected_emission_inventory(main_seal)
    if len(expected) != 160 or len(identities) != 160:
        raise RuntimeError(f"Expected 160 emissions, got {len(expected)}")
    facts = {}
    actual = {str(path.relative_to(ROOT)) for path in (base.OUT / "emissions").glob("*.npz")}
    if actual != set(expected):
        raise RuntimeError("Candidate emission inventory differs from the sealed 160 files")
    for relative, (path, identity) in identities.items():
        expected_sha = bindings["main_runtime_typed_cache_inventory"].get(relative)
        if expected_sha is None or not path.is_file() or _sha(path) != expected_sha:
            raise RuntimeError(f"Missing or changed typed candidate cache: {path}")
        new_full, calibrated = main_runtime._validate_cache(
            path, identity, main_seal["archived_calibrated_emissions"])
        facts[relative] = {"sha256": expected_sha,
                           "new_full_sha256": base._array_sha(new_full),
                           "calibrated_sha256": base._array_sha(calibrated),
                           "frames": int(len(new_full))}
    return facts


def freeze() -> dict:
    if DECLARATION.exists() or PREFLIGHT.exists() or RESULT.exists():
        raise RuntimeError("Boundary refinement declaration or output already exists")
    if not PLAN.is_file():
        raise FileNotFoundError("Frozen single-axis plan is required")
    plan = json.loads(PLAN.read_text())
    if plan.get("study_id") != STUDY or plan.get("trigger", {}).get("transfer_result_present_at_plan_freeze"):
        raise RuntimeError("Boundary plan was not frozen before transfer results")
    bindings = _bindings()
    cache_facts = _validate_all_emissions(bindings)
    expected_controls = plan["preflight_controls"]
    if ({row["id"] for row in expected_controls} != set(CONTROL_RECIPE) or
            len(expected_controls) != 2):
        raise RuntimeError("Expected exactly two fixed diagnostic controls")
    declaration = {
        "study_id": STUDY,
        "status": "frozen_fixed_HA_cost_boundary_after_transfer_below_target_before_refinement_scores",
        "plan": {"path": str(PLAN.relative_to(ROOT)), "sha256": _sha(PLAN), "contents": plan},
        "arms": ARMS,
        "budget": {"recipes": 4, "HA_costs": list(HA_COSTS), "arms": 20,
                   "cells_per_arm": 25, "ensemble_score_cells": 500,
                   "preflight_control_cells": 50, "selected_seed_diagnostic_cells": 50,
                   "fits": 0, "new_emissions": 0},
        "conditional_trigger": {
            "transfer_result_sha256": bindings["transfer_result_sha256"],
            "maximum_transfer_v2_spell": max(
                row["feed_like_v2"]["equal_provider"]["spell_f1"]
                for row in bindings["transfer_result"]["candidate_aggregate"].values()),
            "condition_passed": True, "threshold": 0.60,
        },
        "preflight_controls": expected_controls,
        "guards": plan["guards"],
        "selection": {"primary": "equal-provider v2 spell F1; macro exact-tie break",
                      "report": "best numerical and best eligible separately",
                      "seed_diagnostics": "best eligible if present, otherwise best numerical; no reselection"},
        "provenance": {"source_sha256": _source_hashes(), **{
            key: value for key, value in bindings.items() if key != "transfer_result"},
            "emission_array_facts": cache_facts,
            "baseline_manifest_sha256": _sha(ROOT / "experiments/baseline_manifest.json"),
            "decision_policy_sha256": _sha(ROOT / "experiments/model_decision_policy.json"),
            "upstream_c0_preflight_sha256": _sha(upstream.frozen.PREFLIGHT),
        },
        "organizer_reference_access": False, "public_scoring": False, "promotion": False,
    }
    DECLARATION.write_text(json.dumps(declaration, indent=1, sort_keys=True) + "\n")
    return declaration


def _verify_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze fixed H/A boundary settings before scoring")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("arms") != ARMS:
        raise RuntimeError("Boundary declaration arm list changed")
    if declaration["provenance"]["source_sha256"] != _source_hashes():
        raise RuntimeError("Boundary source changed after freeze")
    current = _bindings()
    for key, value in current.items():
        if key == "transfer_result":
            continue
        if declaration["provenance"].get(key) != value:
            raise RuntimeError(f"Boundary input binding changed: {key}")
    cache_facts = _validate_all_emissions(current)
    if cache_facts != declaration["provenance"]["emission_array_facts"]:
        raise RuntimeError("Boundary candidate emission facts changed")
    if declaration["plan"]["sha256"] != _sha(PLAN):
        raise RuntimeError("Fixed boundary plan changed")
    return declaration


def _load_cell_probabilities(provider: str, match: str, view: str, rep: int):
    view_data, labels, _, phases, model_period = upstream.frozen._prepare_cell(
        provider, match, view, rep)
    if "seal" not in _MAIN_RUNTIME:
        correction, main_seal = main_runtime._verify()
        _MAIN_RUNTIME["declaration"] = correction
        _MAIN_RUNTIME["seal"] = main_seal
    else:
        correction, main_seal = _MAIN_RUNTIME["declaration"], _MAIN_RUNTIME["seal"]
    base = transfer.base
    fold = AUDIT.FOLD[provider]
    base_decl_sha = _sha(base.DECLARATION)
    outputs = {recipe: [] for recipe in RECIPES}
    seed_outputs = {recipe: {seed: [] for seed in SEEDS} for recipe in RECIPES}
    for phase, phase_record in enumerate(phases):
        for seed in SEEDS:
            model_sha = main_seal["models"][f"leave_out_{fold}/{seed}"]["sha256"]
            identity = main_runtime._identity(
                view_data, phase_record, provider, match, view, rep, seed, phase,
                base_decl_sha, model_sha)
            path = base._emission_path(provider, match, view, rep, seed, phase)
            new_full, calibrated = main_runtime._validate_cache(
                path, identity, main_seal["archived_calibrated_emissions"])
            members = {
                "archived_calibrated_phase": calibrated,
                "new_full_phase_tta": new_full,
                "new_conditional_ha_x_archived_calibrated_d":
                    base.cross_conditional_ha(new_full, calibrated[:, 2]),
                "equal_50_50_new_full_plus_archived_calibrated": 0.5 * (new_full + calibrated),
            }
            for recipe, probability in members.items():
                outputs[recipe].append((phase, seed, probability))
                seed_outputs[recipe][seed].append((phase, probability))
    cell = {"labels": labels, "period": view_data["period"],
            "timestamps": view_data["timestamps"], "fps": float(view_data["fps"]),
            "model_period": model_period}
    return cell, outputs, seed_outputs, len(phases)


def _phase_ensemble(members: list, phase_count: int, seed: int | None = None) -> np.ndarray:
    phase_means = []
    for phase in range(phase_count):
        phase_values = [prob for p, s, prob in members
                        if p == phase and (seed is None or s == seed)]
        expected = 2 if seed is None else 1
        if len(phase_values) != expected:
            raise RuntimeError(f"Missing phase/seed probability: phase={phase} seed={seed}")
        phase_means.append(np.mean(phase_values, axis=0))
    return np.mean(phase_means, axis=0)


def _score_probability(cell: dict, probability: np.ndarray, config: dict) -> dict:
    prediction = upstream.frozen.audit.decode(probability, cell["model_period"],
                                              cell["fps"], config)
    return upstream.frozen._score_cell(cell, prediction)


def _score_job(job: tuple[str, str, str, int]) -> tuple[str, dict]:
    provider, match, view, rep = job
    key = f"{provider}__{match}__{view}__{rep}"
    cell, outputs, _, phase_count = _load_cell_probabilities(provider, match, view, rep)
    scores = {}
    for arm in ARMS:
        probability = _phase_ensemble(outputs[arm["recipe"]], phase_count)
        scores[arm["id"]] = _score_probability(cell, probability, arm["config"])
    return key, scores


def preflight() -> dict:
    declaration = _verify_declaration()
    if PREFLIGHT.exists() or RESULT.exists():
        raise RuntimeError("Boundary preflight/result already exists")
    control_scores = {control["id"]: {} for control in declaration["preflight_controls"]}
    for provider, match, _ in AUDIT.MATCHES:
        for view, reps in VIEWS:
            for rep in reps:
                key = f"{provider}__{match}__{view}__{rep}"
                cell, outputs, _, phase_count = _load_cell_probabilities(provider, match, view, rep)
                for control in declaration["preflight_controls"]:
                    recipe, ha = CONTROL_RECIPE[control["id"]]
                    config = dict(BASE_CONFIG, ha=ha)
                    probability = _phase_ensemble(outputs[recipe], phase_count)
                    control_scores[control["id"]][key] = _score_probability(cell, probability, config)
        print(f"BOUNDARY PREFLIGHT {provider}/{match}", flush=True)
    checks = {}
    for control in declaration["preflight_controls"]:
        values = {(key.split("__")[0], key.split("__")[1], key.rsplit("__", 1)[-1]):
                  row["spell_f1"] for key, row in control_scores[control["id"]].items()
                  if "feed_like_v2" in key}
        mean = AUDIT.equal_provider(values)[0]
        expected = control["expected_feed_like_v2_spell"]
        checks[control["id"]] = {"actual": mean, "expected": expected,
                                  "abs_error": abs(mean - expected),
                                  "tolerance": 1e-12}
    passed = all(row["abs_error"] <= row["tolerance"] for row in checks.values())
    receipt = {"study_id": STUDY, "declaration_sha256": _sha(DECLARATION),
               "preflight_passed": passed, "controls_only": True,
               "candidate_grid_scores_written": False, "checks": checks,
               "control_cells": control_scores,
               "transfer_result_sha256": declaration["conditional_trigger"]["transfer_result_sha256"]}
    PREFLIGHT.write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    if not passed:
        raise RuntimeError(f"Boundary diagnostic preflight failed: {checks}")
    return receipt


def _aggregate(rows: dict) -> dict:
    result = {}
    for view in ("clean", "feed_like_v2"):
        result[view] = {}
        for arm in ARM_IDS:
            providers = {}
            for key, cells in rows.items():
                provider, match, cell_view, _rep = key.split("__")
                if cell_view == view:
                    providers.setdefault(provider, {}).setdefault(match, []).append(cells[arm])
            arm_result = {}
            for metric in ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                           "predicted_spell_count", "reference_spell_count",
                           "short_spell_recall_iou_0_5_lt3s"):
                per_provider = {provider: float(np.mean([
                    np.mean([row[metric] for row in match_rows])
                    for match_rows in matches.values()]))
                    for provider, matches in providers.items()}
                arm_result[metric] = {"equal_provider_mean": float(np.mean(list(per_provider.values()))),
                                      "providers": per_provider}
            arm_result["auxiliary_70_30"] = {
                "equal_provider_mean": 0.7 * arm_result["spell_f1"]["equal_provider_mean"] +
                                       0.3 * arm_result["macro_f1"]["equal_provider_mean"],
                "providers": {p: 0.7 * arm_result["spell_f1"]["providers"][p] +
                             0.3 * arm_result["macro_f1"]["providers"][p]
                             for p in providers},
            }
            result[view][arm] = arm_result
    return result


def _guardrails(aggregate: dict) -> dict:
    baseline = upstream.frozen.PREFLIGHT
    controls = json.loads(baseline.read_text())["controls"]
    ref_cells = {tuple(key.split("__")): row["cand"]["C0"]
                 for key, row in controls.items()}
    baseline_summary = {}
    for view, reps in VIEWS:
        selected = {(p, m, str(r)): score for (p, m, cell_view, r), score in ref_cells.items()
                    if cell_view == view and int(r) in reps}
        metrics, providers = {}, {}
        for metric in ("spell_f1", "macro_f1", "d_f1"):
            mean, per = AUDIT.equal_provider({key: row[metric] for key, row in selected.items()})
            metrics[metric], providers[metric] = mean, per
        baseline_summary[view] = {"equal_provider": metrics, "providers": providers}
    output = {}
    for arm in ARM_IDS:
        clean, stress = aggregate["clean"][arm], aggregate["feed_like_v2"][arm]
        ref_clean, ref_stress = baseline_summary["clean"], baseline_summary["feed_like_v2"]
        checks = {
            "clean_equal_provider_spell_ge_minus_005":
                clean["spell_f1"]["equal_provider_mean"] >=
                ref_clean["equal_provider"]["spell_f1"] - 0.005,
            "clean_provider_macro_ge_minus_002": all(
                clean["macro_f1"]["providers"][p] >= ref_clean["providers"]["macro_f1"][p] - 0.02
                for p in ref_clean["providers"]["macro_f1"]),
            "clean_provider_d_f1_ge_minus_002": all(
                clean["d_f1"]["providers"][p] >= ref_clean["providers"]["d_f1"][p] - 0.02
                for p in ref_clean["providers"]["d_f1"]),
            "v2_equal_provider_spell_ge_minus_001":
                stress["spell_f1"]["equal_provider_mean"] >=
                ref_stress["equal_provider"]["spell_f1"] - 0.01,
        }
        output[arm] = {"eligible": all(checks.values()), "checks": checks}
    return {"baseline": baseline_summary, "arms": output}


def _seed_diagnostics(selected: str) -> dict:
    arm = next(row for row in ARMS if row["id"] == selected)
    rows = {str(seed): {} for seed in SEEDS}
    for provider, match, _ in AUDIT.MATCHES:
        for view, reps in VIEWS:
            for rep in reps:
                key = f"{provider}__{match}__{view}__{rep}"
                cell, _, seed_outputs, phase_count = _load_cell_probabilities(
                    provider, match, view, rep)
                for seed in SEEDS:
                    probability = _phase_ensemble(seed_outputs[arm["recipe"]],
                                                  phase_count, seed=seed)
                    rows[str(seed)][key] = _score_probability(cell, probability, arm["config"])
                    print(f"SELECTED SEED {selected}/{seed}: {key}", flush=True)
    summary = {}
    for seed, cells in rows.items():
        summary[seed] = {}
        for view in ("clean", "feed_like_v2"):
            providers = {}
            for key, score in cells.items():
                provider, match, cell_view, _rep = key.split("__")
                if cell_view == view:
                    providers.setdefault(provider, {}).setdefault(match, []).append(score)
            view_summary = {}
            for metric in ("spell_f1", "macro_f1"):
                per_provider = {provider: float(np.mean([
                    np.mean([row[metric] for row in match_rows])
                    for match_rows in matches.values()]))
                    for provider, matches in providers.items()}
                view_summary[metric] = {"equal_provider_mean": float(np.mean(list(per_provider.values()))),
                                        "providers": per_provider}
            summary[seed][view] = {**view_summary, "cells": sum(
                1 for key in cells if key.split("__")[2] == view)}
    return {"selected_arm": selected, "selection_use": "descriptive only; no reselection",
            "cells_per_seed": 25, "scores": rows, "aggregate": summary}


def run() -> dict:
    declaration = _verify_declaration()
    if not PREFLIGHT.is_file():
        raise RuntimeError("Both frozen diagnostic controls must pass before candidate scores")
    preflight_receipt = json.loads(PREFLIGHT.read_text())
    if (preflight_receipt.get("preflight_passed") is not True or
            preflight_receipt.get("declaration_sha256") != _sha(DECLARATION)):
        raise RuntimeError("Boundary preflight is failed or uses a different declaration")
    if RESULT.exists():
        raise RuntimeError("Boundary result already exists")
    rows = {}
    jobs = [(p, m, view, rep) for p, m, _ in AUDIT.MATCHES
            for view, reps in VIEWS for rep in reps]
    for i, job in enumerate(jobs, 1):
        key, values = _score_job(job)
        rows[key] = values
        print(f"BOUNDARY {i}/{len(jobs)}: {key}", flush=True)
    aggregate = _aggregate(rows)
    guards = _guardrails(aggregate)
    ranked = sorted(ARM_IDS, key=lambda arm: (
        aggregate["feed_like_v2"][arm]["spell_f1"]["equal_provider_mean"],
        aggregate["feed_like_v2"][arm]["macro_f1"]["equal_provider_mean"]), reverse=True)
    eligible = [arm for arm in ranked if guards["arms"][arm]["eligible"]]
    selected = eligible[0] if eligible else (ranked[0] if ranked else None)
    seed_diagnostics = _seed_diagnostics(selected) if selected is not None else None
    result = {"study_id": STUDY, "declaration_sha256": _sha(DECLARATION),
              "preflight_sha256": _sha(PREFLIGHT), "arms": list(ARM_IDS),
              "cells": rows, "aggregate": aggregate, "guardrails": guards,
              "ranked_numerical": ranked, "ranked_eligible": eligible,
              "best_numerical": ranked[0] if ranked else None,
              "best_eligible": eligible[0] if eligible else None,
              "selected_for_seed_diagnostic": selected,
              "selected_seed_diagnostics": seed_diagnostics,
              "target_spell_f1": 0.60,
              "organizer_reference_access": False, "public_scoring": False,
              "promotion": False}
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "preflight", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "preflight": preflight, "run": run}[action]()
