"""Post-selection component-seed diagnostic for the frozen v2 calibration grid.

The readout deterministically reports the best eligible grid arm and the best
numerical arm (duplicates collapse), then evaluates each selected setting on
the two existing component seeds across the same 25 development cells. It adds
no decoder settings and cannot change the grid selection.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_spell_push as decoder
from experiments import v2_calibration_target as grid
from experiments import v2_spell_target as cache_study
from experiments import v2_spell_target_runtime as upstream


ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_calibration_seed_readout"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
SEEDS = tuple(cache_study.SEEDS)


def _sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select(grid_result: dict) -> list[str]:
    numerical = grid_result["best_numerical_arm_before_guards"]
    eligible = grid_result["selected_candidate"]
    selected = []
    for arm in (eligible, numerical):
        if arm is not None and arm not in selected:
            selected.append(arm)
    return selected


def _load_grid_result() -> tuple[dict, dict]:
    if not grid.RESULT.is_file() or not grid.DECLARATION.is_file():
        raise FileNotFoundError("Completed frozen v2 calibration grid is required")
    result = json.loads(grid.RESULT.read_text())
    declaration = json.loads(grid.DECLARATION.read_text())
    if result.get("declaration_sha256") != _sha(grid.DECLARATION):
        raise RuntimeError("Grid result does not bind the frozen declaration")
    if result.get("arms") != list(grid.ARM_IDS) or declaration.get("arms") != [
            {"id": arm["id"], "family": arm["family"], "config": arm["config"]}
            for arm in grid.ARMS]:
        raise RuntimeError("Grid arm inventory changed")
    if set(result.get("cells", {})) != {
            f"{provider}__{match}__{view}__{rep}"
            for provider, match, _ in audit.MATCHES
            for view, reps in (("clean", (0,)), ("feed_like_v2", (1, 2, 3, 4)))
            for rep in reps}:
        raise RuntimeError("Grid result does not contain the exact 25-cell cohort")
    return result, declaration


def _saved_cell_rows() -> dict:
    rows = {}
    decl_sha = _sha(grid.DECLARATION)
    for provider, match, _ in audit.MATCHES:
        for view, reps in (("clean", (0,)), ("feed_like_v2", (1, 2, 3, 4))):
            for rep in reps:
                key = f"{provider}__{match}__{view}__{rep}"
                path = grid._cell_path(key)
                receipt_path = path.with_suffix(".receipt.json")
                receipt = json.loads(receipt_path.read_text())
                if receipt.get("declaration_sha256") != decl_sha or receipt.get("sha256") != _sha(path):
                    raise RuntimeError(f"Grid cell receipt mismatch: {path}")
                payload = json.loads(path.read_text())
                if payload.get("declaration_sha256") != decl_sha or payload.get("arms") != list(grid.ARM_IDS):
                    raise RuntimeError(f"Grid cell declaration mismatch: {path}")
                rows[key] = payload["scores"]
    return rows


def _independent_summary(rows: dict) -> dict:
    """Independently reduce saved cell scores, balancing matches and providers."""
    summary = {}
    for view_name in ("clean", "feed_like_v2"):
        summary[view_name] = {}
        for arm in grid.ARM_IDS:
            providers = {}
            for key, cell in rows.items():
                provider, match, view, _rep = key.split("__")
                if view == view_name:
                    providers.setdefault(provider, {}).setdefault(match, []).append(cell[arm])
            arm_summary = {}
            for metric in ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                           "predicted_spell_count", "reference_spell_count",
                           "short_spell_recall_iou_0_5_lt3s"):
                per_provider = {
                    provider: float(np.mean([np.mean([cell[metric] for cell in match_cells])
                                             for match_cells in matches.values()]))
                    for provider, matches in providers.items()
                }
                arm_summary[metric] = {
                    "equal_provider_mean": float(np.mean(list(per_provider.values()))),
                    "providers": per_provider,
                }
            arm_summary["auxiliary_70_30"] = {
                "equal_provider_mean": (0.7 * arm_summary["spell_f1"]["equal_provider_mean"] +
                                         0.3 * arm_summary["macro_f1"]["equal_provider_mean"]),
                "providers": {provider: 0.7 * arm_summary["spell_f1"]["providers"][provider] +
                             0.3 * arm_summary["macro_f1"]["providers"][provider]
                             for provider in providers},
            }
            summary[view_name][arm] = arm_summary
    return summary


def _ranking(summary: dict, arms: list[str]) -> list[str]:
    return sorted(arms, key=lambda arm: (
        summary["feed_like_v2"][arm]["spell_f1"]["equal_provider_mean"],
        summary["feed_like_v2"][arm]["macro_f1"]["equal_provider_mean"]), reverse=True)


def freeze() -> dict:
    """Seal the deterministic selection identities before seed scoring."""
    if DECLARATION.exists() or RESULT.exists():
        raise RuntimeError("Readout declaration/result already exists")
    grid_result, grid_declaration = _load_grid_result()
    grid._verify_declaration()
    saved = _saved_cell_rows()
    recomputed = _independent_summary(saved)
    for view in recomputed:
        for arm in grid.ARM_IDS:
            for metric, values in recomputed[view][arm].items():
                if metric == "auxiliary_70_30":
                    if abs(values["equal_provider_mean"] -
                           grid_result["aggregate"][view][arm][metric]["equal_provider_mean"]) > 1e-12:
                        raise RuntimeError(f"Independent 70/30 aggregate mismatch: {view}/{arm}")
                    continue
                if abs(values["equal_provider_mean"] -
                       grid_result["aggregate"][view][arm][metric]["equal_provider_mean"]) > 1e-12:
                    raise RuntimeError(f"Independent aggregate mismatch: {view}/{arm}/{metric}")
                for provider, value in values["providers"].items():
                    expected = grid_result["aggregate"][view][arm][metric]["providers"][provider]
                    if abs(value - expected) > 1e-12:
                        raise RuntimeError(f"Independent provider aggregate mismatch: {view}/{arm}/{metric}/{provider}")
    selected = _select(grid_result)
    by_id = {arm["id"]: arm for arm in grid.ARMS}
    declaration = {
        "study_id": STUDY,
        "status": "deterministic_post_selection_component_seed_diagnostic",
        "selection": {
            "rule": "best eligible and best numerical grid arm by v2 spell F1, macro F1 only as exact tie-break; duplicates collapse",
            "best_eligible": grid_result["selected_candidate"],
            "best_numerical": grid_result["best_numerical_arm_before_guards"],
            "selected_arm_ids": selected,
            "selected_configurations": [by_id[arm] for arm in selected],
        },
        "scope": {
            "arms": len(selected), "seeds": list(SEEDS), "cells_per_arm_seed": 25,
            "maximum_diagnostic_cells": 100 if len(selected) == 2 else 50,
            "cohort": "same five clean and twenty feed_like_v2 cells as the completed grid",
            "phase_rule": "fixed configuration applied separately to each seed's phase-mean emissions",
            "no_reselection": True, "no_new_settings": True,
        },
        "aggregate_recheck": {
            "method": "independent equal-provider reduction from all 25 saved cell files for each of the 109 grid arms",
            "arms_checked": len(grid.ARM_IDS), "cells_checked": len(saved),
            "tolerance": 1e-12, "passed": True,
        },
        "bindings": {
            "grid_declaration_sha256": _sha(grid.DECLARATION),
            "grid_result_sha256": _sha(grid.RESULT),
            "runtime_declaration_sha256": _sha(cache_study.DECLARATION),
            "runtime_preflight_sha256": _sha(cache_study.PREFLIGHT),
            "calibrator_source_sha256": _sha(Path(grid.__file__)),
            "source_sha256": _sha(Path(__file__)),
            "selected_input_cache_sha256": grid_declaration["provenance"]["emission_files"],
        },
        "organizer_reference_access": False, "public_scoring": False,
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(declaration, indent=1, sort_keys=True) + "\n")
    return declaration


def _verify_frozen() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze deterministic seed-readout identities first")
    declaration = json.loads(DECLARATION.read_text())
    bindings = declaration["bindings"]
    if bindings["source_sha256"] != _sha(Path(__file__)):
        raise RuntimeError("Readout source changed after freeze")
    if bindings["calibrator_source_sha256"] != _sha(Path(grid.__file__)):
        raise RuntimeError("Calibration grid source changed after readout freeze")
    if bindings["grid_declaration_sha256"] != _sha(grid.DECLARATION):
        raise RuntimeError("Grid declaration changed after readout freeze")
    if bindings["grid_result_sha256"] != _sha(grid.RESULT):
        raise RuntimeError("Grid result changed after readout freeze")
    if bindings["runtime_declaration_sha256"] != _sha(cache_study.DECLARATION) or \
            bindings["runtime_preflight_sha256"] != _sha(cache_study.PREFLIGHT):
        raise RuntimeError("Upstream runtime bindings changed after readout freeze")
    current, _ = _load_grid_result()
    grid._verify_declaration()
    if _select(current) != declaration["selection"]["selected_arm_ids"]:
        raise RuntimeError("Deterministic selected settings changed")
    return declaration


def _seed_cell_scores(view_data: dict, labels: np.ndarray, model_period: np.ndarray,
                      phases: list, per_seed: dict, arm: dict, seed: int) -> dict:
    config = arm["config"]
    family = config["family"]
    if family == "blend50":
        probability = np.mean([
            np.mean([per_seed[(name, phase)][seed] for phase in range(len(phases))], axis=0)
            for name in cache_study.FAMILIES
        ], axis=0)
    else:
        probability = np.mean([per_seed[(family, phase)][seed]
                               for phase in range(len(phases))], axis=0)
    cell = {"labels": labels, "period": view_data["period"],
            "timestamps": view_data["timestamps"], "fps": float(view_data["fps"])}
    prediction = decoder.decode(probability[None, :, :], model_period,
                                cell["fps"], config["config"])
    return audit.score(cell, prediction)


def _summarize_seed(cells: dict) -> dict:
    output = {}
    for view_name in ("clean", "feed_like_v2"):
        by_provider = {}
        for key, score in cells.items():
            provider, match, view, _rep = key.split("__")
            if view == view_name:
                by_provider.setdefault(provider, {}).setdefault(match, []).append(score)
        metrics = {}
        for metric in ("spell_f1", "macro_f1"):
            per_provider = {
                provider: float(np.mean([np.mean([row[metric] for row in match_rows])
                                         for match_rows in matches.values()]))
                for provider, matches in by_provider.items()
            }
            metrics[metric] = {
                "equal_provider_mean": float(np.mean(list(per_provider.values()))),
                "providers": per_provider,
            }
        output[view_name] = {**metrics, "cells": sum(
            1 for key in cells if key.split("__")[2] == view_name)}
    return output


def run() -> dict:
    declaration = _verify_frozen()
    if RESULT.exists():
        raise RuntimeError("Seed-readout result already exists")
    configurations = {arm["id"]: arm for arm in declaration["selection"]["selected_configurations"]}
    rows = {arm_id: {str(seed): {} for seed in SEEDS}
            for arm_id in declaration["selection"]["selected_arm_ids"]}
    for provider, match, _ in audit.MATCHES:
        for view, reps in (("clean", (0,)), ("feed_like_v2", (1, 2, 3, 4))):
            for rep in reps:
                key = f"{provider}__{match}__{view}__{rep}"
                view_data, labels, _, phases, model_period = upstream._prepare_cell(
                    provider, match, view, rep)
                _, per_seed = cache_study._phase_probabilities(
                    view_data, provider, match, view, rep, phases)
                for arm_id in declaration["selection"]["selected_arm_ids"]:
                    arm = configurations[arm_id]
                    for seed in SEEDS:
                        rows[arm_id][str(seed)][key] = _seed_cell_scores(
                            view_data, labels, model_period, phases, per_seed, arm, seed)
                        print(f"SEED DIAGNOSTIC {arm_id}/{seed}: {key}", flush=True)
    summary = {}
    for arm_id, seed_rows in rows.items():
        summary[arm_id] = {}
        for seed, cells in seed_rows.items():
            summary[arm_id][seed] = _summarize_seed(cells)
    result = {
        "study_id": STUDY, "declaration_sha256": _sha(DECLARATION),
        "grid_declaration_sha256": _sha(grid.DECLARATION),
        "grid_result_sha256": _sha(grid.RESULT),
        "selected_arm_ids": declaration["selection"]["selected_arm_ids"],
        "best_eligible": declaration["selection"]["best_eligible"],
        "best_numerical": declaration["selection"]["best_numerical"],
        "seed_scores": rows, "aggregate": summary,
        "all_109_arm_aggregate_recheck": declaration["aggregate_recheck"],
        "selection_reused_without_reselection": True,
        "organizer_reference_access": False, "public_scoring": False,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "run": run}[action]()
