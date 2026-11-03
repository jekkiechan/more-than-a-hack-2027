"""Bounded v2-only calibration of the frozen f062 decoder.

Consumes declaration-bound five/ two-phase emissions from the successful
``v2_spell_target_runtime`` C0 preflight. It performs no model inference and
changes only D-transition temperature/cost (``dt``/``dead``) and H/A
temperature/cost (``ht``/``ha``); every other f062 setting stays fixed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_spell_push as decoder
from experiments import v2_spell_target_runtime as upstream
from experiments import v2_spell_target as cache_study


STUDY = "v2_calibration_target"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / "experiments/results" / f"{STUDY}_declaration.json"
RESULT = ROOT / "experiments/results" / f"{STUDY}.json"
TEMP_D = (2, 4, 8)
SWITCH_COST = (1.5, 3, 6)
HA_COST = (1.5, 3, 6)
HA_TEMP = (0.65, 1.0)
FAMILIES = ("adop", "cand")
BASE_CONFIG = copy.deepcopy(audit.CONFIG)


def configurations() -> list[dict]:
    """Return 108 fixed family/grid arms and the fixed 50/50 blend control."""
    arms = []
    for family in FAMILIES:
        for dt in TEMP_D:
            for dead in SWITCH_COST:
                for ha in HA_COST:
                    for ht in HA_TEMP:
                        config = copy.deepcopy(BASE_CONFIG)
                        config.update(dt=dt, dead=dead, ha=ha, ht=ht)
                        arms.append({"id": f"{family}_dt{dt:g}_dead{dead:g}_ha{ha:g}_ht{ht:g}",
                                     "family": family, "config": config})
    arms.append({"id": "blend50_f062", "family": "blend50",
                 "config": copy.deepcopy(BASE_CONFIG)})
    return arms


ARMS = configurations()
ARM_IDS = tuple(a["id"] for a in ARMS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    names = (
        "experiments/v2_calibration_target.py",
        "experiments/v2_spell_target_runtime.py",
        "experiments/v2_spell_target.py",
        "experiments/feed_benchmark_audit_v1.py",
        "experiments/feed_like_v2.py",
        "experiments/feed_spell_push.py",
        "experiments/decoder_refinement.py",
        "experiments/decoder_search.py",
        "experiments/preparation.py",
        "experiments/supervised.py",
        "experiments/spell_mbr_run.py",
        "experiments/metrics.py",
        "experiments/tracking_corruption.py",
        "experiments/stress_components.py",
        "evaluation/possession_metrics.py",
    )
    return {name: _sha256(ROOT / name) for name in names}


def _runtime_ready() -> tuple[dict, dict]:
    if not cache_study.DECLARATION.is_file() or not cache_study.PREFLIGHT.is_file():
        raise RuntimeError("The runtime declaration and successful C0 preflight are required")
    declaration = json.loads(cache_study.DECLARATION.read_text())
    receipt = json.loads(cache_study.PREFLIGHT.read_text())
    declaration = cache_study._require_declaration()
    if (receipt.get("preflight_passed") is not True or
            receipt.get("declaration_sha256") != _sha256(cache_study.DECLARATION)):
        raise RuntimeError("The runtime C0 preflight failed or uses another declaration")
    return declaration, receipt


def _emission_facts() -> dict[str, dict]:
    """Bind all existing cache files; this function never creates emissions."""
    _runtime_ready()
    cache_study._roots()
    facts = {}
    for provider, match, _ in audit.MATCHES:
        for view, reps in (("clean", (0,)), ("feed_like_v2", (1, 2, 3, 4))):
            for rep in reps:
                view_data, _, _, phases, _ = upstream._prepare_cell(provider, match, view, rep)
                paths = [cache_study._cache_path(provider, match, view, rep, family,
                                                 phase, seed)
                         for family in FAMILIES for phase in range(len(phases))
                         for seed in cache_study.SEEDS]
                missing = [path for path in paths if not path.is_file()]
                if missing:
                    raise FileNotFoundError(
                        f"Preflight did not produce required emission: {missing[0]}")
                # Every path is confirmed present first, so this can only read and
                # validate caches; it cannot trigger inference.
                cache_study._phase_probabilities(view_data, provider, match,
                                                 view, rep, phases)
                for family in FAMILIES:
                    for phase in range(len(phases)):
                        for seed in cache_study.SEEDS:
                            path = cache_study._cache_path(provider, match, view, rep,
                                                           family, phase, seed)
                            # Validate archive/cache contract without invoking inference.
                            with np.load(path, allow_pickle=False) as stored:
                                probabilities = stored["probabilities"]
                                if (str(stored["declaration_sha256"].item()) !=
                                        _sha256(cache_study.DECLARATION)):
                                    raise RuntimeError(f"Emission declaration mismatch: {path}")
                                if (probabilities.ndim != 2 or probabilities.shape[1] != 3 or
                                        not np.isfinite(probabilities).all() or
                                        (probabilities < 0).any() or
                                        not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)):
                                    raise RuntimeError(f"Invalid probability cache: {path}")
                            facts[str(path.relative_to(ROOT))] = {
                                "sha256": _sha256(path), "rows": int(len(probabilities)),
                                "provider": provider, "match": match, "view": view,
                                "replicate": rep, "family": family, "phase": phase,
                                "seed": seed,
                            }
    return facts


def freeze() -> dict:
    """Freeze fixed candidate grid and provenance before calibration scoring."""
    if DECLARATION.exists() or RESULT.exists():
        raise RuntimeError("Study declaration or result already exists")
    runtime_declaration, runtime_preflight = _runtime_ready()
    facts = _emission_facts()
    historical_scorecard = json.loads(cache_study.SCORECARD.read_text())
    phase_guard = json.loads((ROOT / "experiments/results/phase_average_guard_v1.json").read_text())
    phase0_clean = historical_scorecard["arms"]["cand"]["clean"]["spell"]
    declaration = {
        "study_id": STUDY,
        "status": "frozen_v2_decoder_calibration_before_candidate_scores",
        "question": "Does a bounded D and H/A decoder calibration grid improve original-provider feed_like_v2 spell F1 over fixed f062 while preserving clean and provider guards?",
        "method": {
            "baseline_decoder": "frozen feed_spell_push f062",
            "parameters_varied": {"dt": list(TEMP_D), "dead": list(SWITCH_COST),
                                  "ha": list(HA_COST), "ht": list(HA_TEMP)},
            "parameter_meaning": {"dt": "D-state temperature", "dead": "D-transition cost",
                                  "ha": "H/A transition cost", "ht": "H/A temperature"},
            "fixed_fields": {k: v for k, v in BASE_CONFIG.items()
                             if k not in ("dt", "dead", "ha", "ht")},
            "emissions": "two seed probabilities averaged within each native phase, then phase probabilities averaged equally; blend control averages the two family phase means 50/50",
            "blend_control": "equal 50/50 mean of calibrated dense and original_full phase-mean emissions decoded with unmodified f062",
            "learned_parameters": 0,
        },
        "arms": [{"id": arm["id"], "family": arm["family"], "config": arm["config"]}
                 for arm in ARMS],
        "cohort": {
            "matches": [{"provider": p, "match": m} for p, m, _ in audit.MATCHES],
            "cells": "5 clean cells and 20 feed_like_v2 cells (four fixed realizations per match)",
            "aggregation": "realizations within match, matches within provider, then providers equally",
            "native_phase_counts": "read from each verified phase cache; SkillCorner uses two 10 Hz phases, 25 Hz providers use five phases",
            "folds": "archived full-provider holdout models in the upstream runtime declaration",
        },
        "budget": {"candidate_arms": 109, "candidate_arm_cells": 2725,
                   "candidate_score_writes": 2725, "fits": 0, "re_inference": 0,
                   "worker_default": 2, "worker_override": "V2_CALIBRATION_WORKERS"},
        "ranking": "official spell F1, macro F1 only for exact spell ties",
        "auxiliary": "0.7*spell_f1 + 0.3*macro_f1, not the ranking metric",
        "guards": {
            "clean_spell_vs_original_full_phase": -0.005,
            "each_provider_clean_macro_f1_vs_original_full_phase": -0.02,
            "each_provider_clean_d_f1_vs_original_full_phase": -0.02,
            "v2_spell_vs_original_full_phase": -0.01,
            "phase0_original_full_plus_0_01_diagnostic": {
                "report_separately_not_an_eligibility_guard": True,
                "reference_scorecard_sha256": _sha256(cache_study.SCORECARD),
                "reference_arm": "cand phase0",
                "reference_clean_spell": phase0_clean,
                "threshold": phase0_clean + 0.01,
            },
            "new_clean_plus_0_01_screen_vs_current_phase_baseline": {
                "report_separately_not_an_eligibility_guard": True,
                "reference_arm": "cand_dt4_dead3_ha3_ht1", "delta": 0.01,
            },
            "prior_historical_clean_screen": {
                "source": "phase_average_guard_v1",
                "source_sha256": _sha256(ROOT / "experiments/results/phase_average_guard_v1.json"),
                "status": "failed before phase-average adoption; preserve existing result",
                "delta": phase_guard["historical_clean_screen_delta"],
                "passes": phase_guard["historical_clean_screen_passes"],
                "recomputed": False,
                "policy_sha256": _sha256(ROOT / "experiments/model_decision_policy.json"),
            },
            "reference_arm": "cand_dt4_dead3_ha3_ht1 (unmodified f062 from original_full phase emissions)",
        },
        "upstream": {
            "study_id": upstream.STUDY,
            "declaration_sha256": _sha256(cache_study.DECLARATION),
            "preflight_sha256": _sha256(cache_study.PREFLIGHT),
            "preflight_passed": runtime_preflight["preflight_passed"],
            "declaration": runtime_declaration,
        },
        "provenance": {
            "source_sha256": _source_hashes(),
            "calibration_sha256": _sha256(cache_study.CALIBRATION),
            "baseline_manifest_sha256": _sha256(ROOT / "experiments/baseline_manifest.json"),
            "decision_policy_sha256": _sha256(ROOT / "experiments/model_decision_policy.json"),
            "emission_files": facts,
            "organizer_reference_access": False,
            "public_scoring": False,
        },
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(declaration, indent=1, sort_keys=True) + "\n")
    return declaration


def _verify_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError(f"Calibration scoring blocked until frozen: {DECLARATION}")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or tuple(a["id"] for a in declaration["arms"]) != ARM_IDS:
        raise RuntimeError("Frozen calibration declaration does not match the runner")
    if declaration["provenance"]["source_sha256"] != _source_hashes():
        raise RuntimeError("Calibration source changed after freeze")
    if declaration["provenance"]["calibration_sha256"] != _sha256(cache_study.CALIBRATION):
        raise RuntimeError("feed_like_v2 calibration changed after freeze")
    if declaration["provenance"]["baseline_manifest_sha256"] != _sha256(ROOT / "experiments/baseline_manifest.json"):
        raise RuntimeError("Active baseline manifest changed after freeze")
    if declaration["provenance"]["decision_policy_sha256"] != _sha256(ROOT / "experiments/model_decision_policy.json"):
        raise RuntimeError("Decision policy changed after freeze")
    if declaration["guards"]["prior_historical_clean_screen"]["policy_sha256"] != _sha256(
            ROOT / "experiments/model_decision_policy.json"):
        raise RuntimeError("Decision policy changed after freeze")
    prior_screen = declaration["guards"]["prior_historical_clean_screen"]
    if prior_screen["source_sha256"] != _sha256(ROOT / "experiments/results/phase_average_guard_v1.json"):
        raise RuntimeError("Prior historical clean-screen receipt changed after freeze")
    phase0 = declaration["guards"]["phase0_original_full_plus_0_01_diagnostic"]
    if phase0["reference_scorecard_sha256"] != _sha256(cache_study.SCORECARD):
        raise RuntimeError("Historical phase0 scorecard changed after freeze")
    _, receipt = _runtime_ready()
    if (_sha256(cache_study.DECLARATION) != declaration["upstream"]["declaration_sha256"] or
            _sha256(cache_study.PREFLIGHT) != declaration["upstream"]["preflight_sha256"] or
            receipt.get("preflight_passed") is not True):
        raise RuntimeError("Upstream runtime declaration/preflight changed")
    if _emission_facts() != declaration["provenance"]["emission_files"]:
        raise RuntimeError("Verified cached emissions changed after freeze")
    return declaration


def _cell_inputs(provider: str, match: str, view: str, rep: int):
    view_data, labels, record, phases, period = upstream._prepare_cell(
        provider, match, view, rep)
    for family in FAMILIES:
        for phase in range(len(phases)):
            for seed in cache_study.SEEDS:
                emission = cache_study._cache_path(provider, match, view, rep,
                                                   family, phase, seed)
                if not emission.is_file():
                    raise FileNotFoundError(f"Refusing to infer missing frozen emission: {emission}")
    pooled, _ = cache_study._phase_probabilities(view_data, provider, match, view, rep, phases)
    probabilities = {
        family: np.mean([pooled[(family, phase)] for phase in range(len(phases))], axis=0)
        for family in FAMILIES
    }
    cell = {"labels": labels, "period": view_data["period"],
            "timestamps": view_data["timestamps"], "fps": float(view_data["fps"])}
    return cell, probabilities, period


def _cell_scores(provider: str, match: str, view: str, rep: int) -> dict:
    cell, probabilities, model_period = _cell_inputs(provider, match, view, rep)
    output = {}
    for arm in ARMS:
        if arm["family"] == "blend50":
            probability = (probabilities["adop"] + probabilities["cand"]) / 2
        else:
            probability = probabilities[arm["family"]]
        prediction = decoder.decode(probability[None, :, :], model_period,
                                    cell["fps"], arm["config"])
        output[arm["id"]] = audit.score(cell, prediction)
    return output


def _summarize(rows: dict) -> dict:
    summary = {}
    for view in ("clean", "feed_like_v2"):
        summary[view] = {}
        for arm_id in ARM_IDS:
            values = {}
            for key, row in rows.items():
                provider, match, cell_view, _rep = key.split("__")
                if cell_view == view:
                    values.setdefault(provider, {}).setdefault(match, []).append(row[arm_id])
            metrics = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                       "predicted_spell_count", "reference_spell_count",
                       "short_spell_recall_iou_0_5_lt3s")
            arm_summary = {}
            for metric in metrics:
                per_provider = {provider: float(np.mean([
                    np.mean([cell[metric] for cell in match_cells])
                    for match_cells in matches.values()]))
                    for provider, matches in values.items()}
                arm_summary[metric] = {
                    "equal_provider_mean": float(np.mean(list(per_provider.values()))),
                    "providers": per_provider,
                }
            arm_summary["auxiliary_70_30"] = {
                "equal_provider_mean": (0.7 * arm_summary["spell_f1"]["equal_provider_mean"] +
                                         0.3 * arm_summary["macro_f1"]["equal_provider_mean"]),
                "providers": {provider: 0.7 * arm_summary["spell_f1"]["providers"][provider] +
                             0.3 * arm_summary["macro_f1"]["providers"][provider]
                             for provider in arm_summary["spell_f1"]["providers"]},
            }
            summary[view][arm_id] = arm_summary
    return summary


def _eligibility(summary: dict, phase0_screen_floor: float) -> dict:
    baseline = "cand_dt4_dead3_ha3_ht1"
    reference = summary["clean"][baseline]
    v2_reference = summary["feed_like_v2"][baseline]["spell_f1"]["equal_provider_mean"]
    out = {}
    for arm_id in ARM_IDS:
        clean = summary["clean"][arm_id]
        v2_spell = summary["feed_like_v2"][arm_id]["spell_f1"]["equal_provider_mean"]
        checks = {
            "clean_spell_delta_ge_minus_005":
                clean["spell_f1"]["equal_provider_mean"] - reference["spell_f1"]["equal_provider_mean"] >= -0.005,
            "provider_clean_macro_ge_minus_002": all(
                clean["macro_f1"]["providers"].get(p, float("-inf")) -
                reference["macro_f1"]["providers"][p] >= -0.02
                for p in reference["macro_f1"]["providers"]),
            "provider_clean_d_f1_ge_minus_002": all(
                clean["d_f1"]["providers"].get(p, float("-inf")) -
                reference["d_f1"]["providers"][p] >= -0.02
                for p in reference["d_f1"]["providers"]),
            "v2_spell_delta_ge_minus_001": v2_spell - v2_reference >= -0.01,
        }
        out[arm_id] = {
            "eligible": all(checks.values()), "checks": checks,
            "clean_spell_delta_vs_current_phase_baseline":
                clean["spell_f1"]["equal_provider_mean"] -
                reference["spell_f1"]["equal_provider_mean"],
            "clean_spell_delta_vs_phase0_original_full":
                clean["spell_f1"]["equal_provider_mean"] - phase0_screen_floor + 0.01,
            "new_clean_plus_001_screen_vs_current_phase_baseline_pass":
                clean["spell_f1"]["equal_provider_mean"] >=
                reference["spell_f1"]["equal_provider_mean"] + 0.01,
            "phase0_original_full_plus_001_diagnostic_pass":
                clean["spell_f1"]["equal_provider_mean"] >= phase0_screen_floor,
            "prior_historical_clean_screen_preserved": {
                "status": "failed before phase-average adoption; original receipt preserved",
                "recomputed": False,
            },
        }
    return out


def _cell_path(key: str) -> Path:
    return OUT / "cells" / f"{key}.json"


def _load_cell(key: str, declaration_sha: str) -> dict | None:
    path = _cell_path(key)
    if not path.exists():
        return None
    receipt_path = path.with_suffix(".receipt.json")
    if not receipt_path.exists():
        raise RuntimeError(f"Incomplete calibration cell result: {path}")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("declaration_sha256") != declaration_sha or
            receipt.get("sha256") != _sha256(path)):
        raise RuntimeError(f"Calibration cell receipt mismatch: {path}")
    row = json.loads(path.read_text())
    if (row.get("declaration_sha256") != declaration_sha or
            row.get("arms") != list(ARM_IDS)):
        raise RuntimeError(f"Calibration cell uses a different declaration: {path}")
    return row["scores"]


def _score_job(job: tuple[str, str, str, int, str]) -> tuple[str, dict]:
    provider, match, view, rep, declaration_sha = job
    key = f"{provider}__{match}__{view}__{rep}"
    cached = _load_cell(key, declaration_sha)
    if cached is not None:
        return key, cached
    scores = _cell_scores(provider, match, view, rep)
    path = _cell_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"study_id": STUDY, "declaration_sha256": declaration_sha,
               "arms": list(ARM_IDS), "scores": scores}
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)
    receipt_path = path.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps({"declaration_sha256": declaration_sha,
                                        "sha256": _sha256(path)},
                                       indent=1, sort_keys=True) + "\n")
    return key, scores


def run() -> dict:
    declaration = _verify_declaration()
    if RESULT.exists():
        raise RuntimeError("Result already exists; calibration study output is immutable")
    rows = {}
    jobs = [(provider, match, view, rep)
            for provider, match, _ in audit.MATCHES
            for view, reps in (("clean", (0,)), ("feed_like_v2", (1, 2, 3, 4)))
            for rep in reps]
    declaration_sha = _sha256(DECLARATION)
    workers = int(os.environ.get("V2_CALIBRATION_WORKERS", "2"))
    if workers < 1:
        raise ValueError("V2_CALIBRATION_WORKERS must be positive")
    work = [(provider, match, view, rep, declaration_sha)
            for provider, match, view, rep in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_score_job, item) for item in work]
        for done, future in enumerate(as_completed(futures), 1):
            key, scores = future.result()
            rows[key] = scores
            print(f"CALIBRATION {done}/{len(work)}: {key}", flush=True)
    summary = _summarize(rows)
    phase0_floor = declaration["guards"]["phase0_original_full_plus_0_01_diagnostic"]["threshold"]
    eligibility = _eligibility(summary, phase0_floor)
    eligible = [arm_id for arm_id in ARM_IDS if eligibility[arm_id]["eligible"]]
    # Ranking is fixed: spell first, macro only as an exact spell tie-break.
    ranked = sorted(eligible, key=lambda arm_id: (
        summary["feed_like_v2"][arm_id]["spell_f1"]["equal_provider_mean"],
        summary["feed_like_v2"][arm_id]["macro_f1"]["equal_provider_mean"]), reverse=True)
    all_ranked = sorted(ARM_IDS, key=lambda arm_id: (
        summary["feed_like_v2"][arm_id]["spell_f1"]["equal_provider_mean"],
        summary["feed_like_v2"][arm_id]["macro_f1"]["equal_provider_mean"]), reverse=True)
    result = {
        "study_id": STUDY,
        "declaration_sha256": _sha256(DECLARATION),
        "arms": list(ARM_IDS), "cells": rows, "aggregate": summary,
        "eligibility": eligibility,
        "ranked_eligible_by_v2_spell_then_macro": ranked,
        "selected_candidate": ranked[0] if ranked else None,
        "best_numerical_arm_before_guards": all_ranked[0] if all_ranked else None,
        "ranked_all_arms_by_v2_spell_then_macro": all_ranked,
        "workers": workers,
        "selection_is_provisional_development_only": True,
        "organizer_reference_access": False, "public_scoring": False,
        "external_confirmation": False,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("freeze", "run"))
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.step]()
