"""Frozen, matched evaluation helpers for the v2 iterative-data study.

This module deliberately keeps control preparation and candidate scoring in
separate commands.  Raw labels are consumed only by the frozen corruption
constructor and official scoring functions; candidate probability transforms
never receive labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import joblib

from experiments import feed_benchmark_audit_v1 as audit
from experiments import v2_data_capacity_eval as capacity
from experiments import v2_spell_target_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_iterative_v1"
OUT = ROOT / "data/experiments" / STUDY
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
REPLAY = ROOT / f"experiments/results/{STUDY}_control_replay.json"
ROUND1_DECLARATION = ROOT / f"experiments/results/{STUDY}_round1_declaration.json"
ROUND1_RESULT = ROOT / f"experiments/results/{STUDY}_round1.json"
SEEDS = tuple(runtime.frozen.SEEDS)
VIEWS = capacity.VIEWS
METRICS = capacity.METRICS
QUALITY_METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                   "short_spell_recall_iou_0_5_lt3s")
ARMS = ("joint_probability_blend_ha3", "fixed_d_conditional_ha3",
        "fixed_d_conditional_ha1_5", "fixed_d_logit_ha3")
WORKERS = max(1, int(os.environ.get("V2_ITERATIVE_WORKERS", "2")))
TORCH_THREADS = max(1, int(os.environ.get("V2_ITERATIVE_TORCH_THREADS", "1")))
HISTORICAL_MAIN = ROOT / "experiments/results/v2_weight_edit_development_v1.json"
HISTORICAL_V1 = ROOT / "experiments/results/v2_weight_edit_development_v1_v1.json"


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


def cell_key(provider: str, match: str, view: str, rep: int) -> str:
    return f"{provider}|{match}|{view}|{int(rep)}"


def _views():
    return [(p, m, view, int(rep)) for p, m, _ in runtime.frozen.audit.MATCHES
            for view, reps in VIEWS for rep in reps]


def _input_and_source_facts():
    data_root, sqd_root, dt_root = runtime.frozen._roots()
    data_manifest = data_root / "seed_gap_context_cache_manifest.json"
    teacher_manifest = data_root / "objective_alignment_cache_manifest.json"
    sqd_manifest = sqd_root / "short_quality_diversity_v1_cache_manifest.json"
    dt_manifest = dt_root / "dense_targeted_followups_cache_manifest_v1.json"
    inputs = {}
    for provider, match, view, rep in _views():
        filename = runtime.frozen.audit.FILE[match]
        suffix = ("clean_clean.joblib" if view in ("clean", "feed_like_v2")
                  else f"feed_like_corrupt_{rep}.joblib")
        path = runtime.frozen.audit.EVAL_INPUTS / f"{filename}_{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen archived raw input is missing: {path}")
        inputs[cell_key(provider, match, view, rep)] = {
            "path": str(path), "sha256": sha(path),
            "archive_fact": runtime.frozen._artifact_fact(
                data_manifest, data_root, str(path.relative_to(data_root)))}
    controls = {}
    for provider in runtime.frozen.audit.FOLD:
        for seed in SEEDS:
            for family in ("adop", "cand"):
                p = runtime.frozen._model_path(family, provider, seed)
                if family == "adop":
                    manifest, root = dt_manifest, dt_root
                else:
                    manifest, root = sqd_manifest, sqd_root
                controls[f"{family}/{provider}/{seed}"] = {
                    "path": str(p), "sha256": sha(p),
                    "archive_fact": runtime.frozen._artifact_fact(
                        manifest, root, str(p.relative_to(root)))}
            p = runtime.frozen._teacher_path(provider, seed)
            controls[f"teacher/{provider}/{seed}"] = {
                "path": str(p), "sha256": sha(p),
                "archive_fact": runtime.frozen._artifact_fact(
                    teacher_manifest, data_root, str(p.relative_to(data_root)))}
    sources = [Path(__file__), Path(capacity.__file__), Path(runtime.frozen.__file__),
               Path(runtime.frozen.audit.__file__), Path(runtime.frozen.v2.__file__),
               Path(runtime.frozen.prep.__file__), Path(runtime.frozen.tc.__file__),
               Path(runtime.frozen.audit.decoder.__file__), Path(runtime.frozen.sn.__file__),
               Path(runtime.frozen.old.__file__), Path(runtime.frozen.supervised.__file__),
               Path(runtime.frozen.audit.metrics.__file__),
               Path(runtime.frozen.sn.cap.__file__),
               Path(runtime.frozen.sn.fh.__file__),
               Path(runtime.frozen.sn.lt.__file__),
               Path(ROOT / "experiments/refine_features.py"),
               Path(ROOT / "experiments/learned_temporal.py"),
               Path(ROOT / "evaluation/possession_metrics.py"),
               ROOT / f"experiments/results/{STUDY}_plan.json",
               HISTORICAL_MAIN, HISTORICAL_V1,
               ROOT / "experiments/baseline_manifest.json",
               ROOT / "experiments/model_decision_policy.json",
               runtime.frozen.CALIBRATION]
    source_hashes = {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p)
                     for p in sources}
    return inputs, controls, source_hashes


def freeze() -> dict:
    """Declare the 35 fixed cells and all inputs/control artifacts before inference."""
    runtime.frozen._roots()
    if any(p.exists() for p in (DECLARATION, REPLAY, ROUND1_DECLARATION, ROUND1_RESULT)):
        raise RuntimeError("Study output already exists; preserve this study ID")
    if not HISTORICAL_MAIN.is_file() or not HISTORICAL_V1.is_file():
        raise FileNotFoundError("Saved adopted-control expectations are required")
    inputs, controls, sources = _input_and_source_facts()
    declaration = {
        "study_id": STUDY,
        "status": "frozen_before_control_inference",
        "question": "Can added iterative training data raise matched feed_like_v2 spell F1 above 0.60 without regressions in the six clean/v1/v2 quality metrics?",
        "cohort": {"matches": [{"provider": p, "match": m} for p, m, _ in runtime.frozen.audit.MATCHES],
                   "views": [name for name, _ in VIEWS], "clean_cells": 5,
                   "feed_like_v1_replicates": [1, 2], "feed_like_v2_replicates": [1, 2, 3, 4],
                   "cell_count": len(_views()), "seeds": list(SEEDS),
                   "aggregation": "replicates within match, matches within provider, providers equally"},
        "controls": {"adopted": "archived original_full (cand), f062 decode",
                     "calibrated": "archived calibrated dense (adop), fixed decoder declared per arm",
                     "control_replay_before_candidates": True,
                     "adopted_metric_tolerance": 1e-8,
                     "fixed_d_parity": "reported as a diagnostic; no equality assertion"},
        "round1_arms": list(ARMS),
        "success": {"metrics": list(QUALITY_METRICS), "views": [x[0] for x in VIEWS],
                    "adopted_comparison": "zero tolerance, candidate >= adopted in every aggregate metric/view",
                    "v2_spell_f1_strictly_greater_than": 0.60,
                    "reference_spell_count_invariant": True,
                    "diagnostics": ["predicted_spell_count", "provider_metrics", "seed_metrics",
                                    "macro_tie_ordering", "weighted_70_30"]},
        "provenance": {"archived_inputs": inputs, "control_models_and_teachers": controls,
                       "source_sha256": sources,
                       "calibration_sha256": sha(runtime.frozen.CALIBRATION),
                       "baseline_manifest_sha256": sha(ROOT / "experiments/baseline_manifest.json"),
                       "decision_policy_sha256": sha(ROOT / "experiments/model_decision_policy.json")},
        "organizer_reference_access": False, "public_or_GAR_scoring": False, "promotion": False,
    }
    _write_once(DECLARATION, declaration)
    return declaration


def _require_frozen() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze evaluator declaration before inference")
    d = json.loads(DECLARATION.read_text())
    if d.get("study_id") != STUDY or d["provenance"]["source_sha256"] != _input_and_source_facts()[2]:
        raise RuntimeError("Study source or declaration identity changed after freeze")
    inputs, controls, _ = _input_and_source_facts()
    if d["provenance"]["archived_inputs"] != inputs or d["provenance"]["control_models_and_teachers"] != controls:
        raise RuntimeError("A declared input or control artifact changed")
    return d


def _phase_control_job(job):
    provider, match, view_name, rep = job
    path, view, labels, phases, period = capacity._raw_cache(provider, match, view_name, rep)
    per_family, pooled_family = {}, {}
    for family in ("adop", "cand"):
        phase_seed = {seed: [runtime.frozen.predict_seed(view, phase_record, family, provider, seed)
                             for phase_record in phases] for seed in SEEDS}
        per_family[family] = {seed: np.mean(phase_seed[seed], axis=0) for seed in SEEDS}
        # Match deployment arithmetic exactly: average component seeds within each
        # native phase, then average those phase means.
        phase_means = [np.mean([phase_seed[seed][phase] for seed in SEEDS], axis=0)
                       for phase in range(len(phases))]
        pooled_family[family] = np.mean(phase_means, axis=0)
    return {"provider": provider, "match": match, "view": view_name, "replicate": rep,
            "input_sha256": sha(path), "labels": labels, "view_data": view,
            "period": period, "phases": phases, "phase_count": len(phases),
            "probabilities": {f"{fam}_seed{seed}": probs
                              for fam, seeded in per_family.items() for seed, probs in seeded.items()},
            "adopted": pooled_family["cand"],
            "calibrated": pooled_family["adop"]}


def _save_control(row: dict, declaration_sha: str) -> dict:
    key = cell_key(row["provider"], row["match"], row["view"], row["replicate"])
    path = OUT / "controls" / (key.replace("|", "__") + ".npz")
    identity = {"study_id": STUDY, "declaration_sha256": declaration_sha, "cell": key,
                "input_sha256": row["input_sha256"], "phase_count": row["phase_count"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"prob__{k}": v for k, v in row["probabilities"].items()}
    arrays.update({"prob__adopted": row["adopted"], "prob__calibrated": row["calibrated"],
                   "labels": row["labels"], "timestamps": row["view_data"]["timestamps"],
                   "period": row["view_data"]["period"], "model_period": row["period"],
                   "fps": np.asarray(float(row["view_data"]["fps"]))})
    arrays.update({f"identity__{k}": np.asarray(str(v)) for k, v in identity.items()})
    np.savez_compressed(path, **arrays)
    prepared_path = OUT / "prepared" / (key.replace("|", "__") + ".joblib")
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_value = {"view": row["view_data"], "labels": row["labels"],
                      "phases": row["phases"], "period": row["period"],
                      "identity": identity}
    joblib.dump(prepared_value, prepared_path, compress=0)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path), "identity": identity,
            "prepared_path": str(prepared_path.relative_to(ROOT)),
            "prepared_sha256": sha(prepared_path)}


def _load_control(fact: dict) -> dict:
    path = ROOT / fact["path"]
    if not path.is_file() or sha(path) != fact["sha256"]:
        raise RuntimeError(f"Control emission changed or missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        actual = {k.removeprefix("identity__"): str(saved[k].item())
                  for k in saved.files if k.startswith("identity__")}
        if actual != {k: str(v) for k, v in fact["identity"].items()}:
            raise RuntimeError(f"Control emission identity mismatch: {path}")
        return {k.removeprefix("prob__"): saved[k].copy()
                for k in saved.files if k.startswith("prob__")} | {
                    "labels": saved["labels"].copy(), "period": saved["period"].copy(),
                    "timestamps": saved["timestamps"].copy(),
                    "model_period": saved["model_period"].copy(), "fps": float(saved["fps"])}


def _historical_adopted() -> dict:
    main = json.loads(HISTORICAL_MAIN.read_text())["arms"]["deployed_original_full"]
    v1 = json.loads(HISTORICAL_V1.read_text())["arms"]["deployed_original_full"]
    result = {view: main[view] for view in ("clean", "feed_like_v2")}
    result["feed_like_v1"] = v1
    return {view: {metric: float(result[view][metric]["equal_provider_mean"])
                   for metric in METRICS} for view, _ in VIEWS}


def _aggregate(rows: Mapping[str, dict], family: str) -> dict:
    result = {}
    for view, reps in VIEWS:
        result[view] = {}
        for metric in METRICS:
            vals = {}
            for key, row in rows.items():
                p, m, v, rep = key.split("|")
                if v == view and int(rep) in reps:
                    vals[(p, m, int(rep))] = float(row[family][metric])
            mean, providers = capacity.equal_provider(vals)
            result[view][metric] = {"equal_provider": mean, "providers": providers}
    return result


def prepare_controls() -> dict:
    """Emit frozen controls and replay adopted original_full against saved metrics."""
    declaration = _require_frozen()
    if REPLAY.exists() or ROUND1_DECLARATION.exists() or ROUND1_RESULT.exists():
        raise RuntimeError("Control replay or later-stage output already exists")
    jobs = _views()
    workers = min(WORKERS, len(jobs))
    rows, inventory = {}, {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_phase_control_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            row = future.result()
            key = cell_key(row["provider"], row["match"], row["view"], row["replicate"])
            inventory[key] = _save_control(row, sha(DECLARATION))
            cell = {"labels": row["labels"], "period": row["view_data"]["period"],
                    "timestamps": row["view_data"]["timestamps"], "fps": float(row["view_data"]["fps"])}
            row["adopted_metrics"] = audit.score(cell, audit.decode(row["adopted"], row["period"], cell["fps"], audit.CONFIG))
            row["calibrated_metrics"] = audit.score(cell, audit.decode(row["calibrated"], row["period"], cell["fps"], audit.CONFIG))
            rows[key] = row
            print(f"CONTROL {done}/{len(jobs)} {key}", flush=True)
    aggregate = _aggregate(rows, "adopted_metrics")
    expected = _historical_adopted()
    checks = {f"{v}/{m}": {"actual": aggregate[v][m]["equal_provider"],
                           "expected": expected[v][m],
                           "absolute_error": abs(aggregate[v][m]["equal_provider"] - expected[v][m]),
                           "passed": abs(aggregate[v][m]["equal_provider"] - expected[v][m]) <= 1e-8}
              for v, _ in VIEWS for m in METRICS}
    # Fixed-D H/A replay is diagnostic: its D mask is adopted, while H/A is decoded
    # conditionally. The official joint decoder need not produce identical labels.
    fixed_d = {}
    for key, row in rows.items():
        cell = {"labels": row["labels"], "period": row["view_data"]["period"],
                "timestamps": row["view_data"]["timestamps"], "fps": float(row["view_data"]["fps"])}
        pred = capacity._adopted_dead_prediction(row["adopted"], row["adopted"], row["period"],
                                                  cell["fps"], capacity._decoder(3.0))
        fixed_d[key] = audit.score(cell, pred)
    passed = all(x["passed"] for x in checks.values())
    result = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
              "controls_only": True, "candidate_scoring": False, "passed": passed,
              "tolerance": 1e-8, "adopted_replay_checks": checks,
              "adopted_aggregate": aggregate,
              "calibrated_aggregate": _aggregate(rows, "calibrated_metrics"),
              "fixed_d_adopted_metrics": fixed_d, "emission_cache": inventory,
              "cell_count": len(rows), "cell_counts": _counts(rows),
              "reference_spell_counts": _reference_counts(rows),
              "control_model_hashes": declaration["provenance"]["control_models_and_teachers"],
              "workers": workers, "torch_threads_per_worker": TORCH_THREADS}
    _write_once(REPLAY, result)
    if not passed:
        raise RuntimeError(f"Adopted control replay did not match historical metrics: {checks}")
    return result


def _worker_init():
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except ImportError:
        pass


def _counts(rows):
    return {f"{view}/{rep}": sum(1 for _ in rows if _split_key(_)[2:] == (view, rep))
            for view, reps in VIEWS for rep in reps}


def _split_key(key):
    p, m, view, rep = key.split("|")
    return p, m, view, int(rep)


def _reference_counts(rows):
    result = {}
    for view, reps in VIEWS:
        for rep in reps:
            result[f"{view}/{rep}"] = int(sum(row["adopted_metrics"]["reference_spell_count"]
                for key, row in rows.items() if _split_key(key)[2:] == (view, rep)))
    return result


def freeze_round1(candidate_models: Mapping[str, str | Path] | None = None) -> dict:
    """Freeze candidate checkpoints and four fixed blend definitions before scoring."""
    _require_frozen()
    if not REPLAY.is_file():
        raise RuntimeError("Replay controls successfully before freezing round one")
    replay = json.loads(REPLAY.read_text())
    if replay.get("passed") is not True or replay.get("candidate_scoring") is not False:
        raise RuntimeError("Control replay did not pass")
    if ROUND1_DECLARATION.exists() or ROUND1_RESULT.exists():
        raise RuntimeError("Round-one outputs already exist")
    if candidate_models:
        raise ValueError("Round one compares archived controls only; candidate models belong to a later round")
    declaration = {"study_id": STUDY, "round": 1, "status": "frozen_before_candidate_scoring",
                   "parent_declaration_sha256": sha(DECLARATION), "control_replay_sha256": sha(REPLAY),
                   "arms": list(ARMS),
                   "methods": {
                       ARMS[0]: "equal categorical probability blend of candidate and adopted original_full, then f062 HA=3 decode",
                       ARMS[1]: "equal conditional H/A probability blend under frozen adopted D mask, HA=3",
                       ARMS[2]: "same conditional blend and fixed adopted D mask, HA=1.5",
                       ARMS[3]: "equal conditional H/A logit blend under frozen adopted D mask, HA=3"},
                   "baseline_policy": {"hard_metrics": list(QUALITY_METRICS),
                                       "zero_tolerance_vs_adopted": True,
                                       "strict_v2_spell_floor": 0.60,
                                       "counts_invariant": True,
                                       "ranking": "spell_f1, exact ties broken by macro_f1",
                                       "weighted_70_30": "diagnostic only"},
                   "candidate_scoring_started": False, "organizer_reference_access": False,
                   "public_or_GAR_scoring": False, "promotion": False}
    _write_once(ROUND1_DECLARATION, declaration)
    return declaration


def _score_saved_probability(control: dict, probability: np.ndarray, config: dict) -> dict:
    cell = {"labels": control["labels"], "period": control["period"],
            "timestamps": control["timestamps"], "fps": control["fps"]}
    pred = audit.decode(probability, control["model_period"], cell["fps"], config)
    return audit.score(cell, pred)


def _score_saved_fixed_d(control: dict, probability: np.ndarray, adopted: np.ndarray,
                         ha_cost: float) -> dict:
    cell = {"labels": control["labels"], "period": control["period"],
            "timestamps": control["timestamps"], "fps": control["fps"]}
    pred = fixed_d_prediction(probability, adopted, control["model_period"],
                              cell["fps"], ha_cost)
    return audit.score(cell, pred)


def _aggregate_arm(rows: Mapping[str, dict], arm: str, metric: str) -> tuple[float, dict]:
    result = {}
    for view, reps in VIEWS:
        values = {}
        for key, row in rows.items():
            p, m, v, rep = _split_key(key)
            if v == view and rep in reps:
                values[(p, m, rep)] = float(row[arm][metric])
        mean, providers = capacity.equal_provider(values)
        result[view] = {"equal_provider": mean, "providers": providers}
    return result


def run_round1() -> dict:
    """Score only the four frozen adopted/calibrated control blends."""
    _require_frozen()
    if not ROUND1_DECLARATION.is_file() or not REPLAY.is_file():
        raise RuntimeError("Freeze round one and pass the control replay before scoring")
    frozen = json.loads(ROUND1_DECLARATION.read_text())
    replay = json.loads(REPLAY.read_text())
    if (frozen.get("control_replay_sha256") != sha(REPLAY) or
            replay.get("passed") is not True or frozen.get("arms") != list(ARMS)):
        raise RuntimeError("Round-one declaration or controls replay changed")
    if ROUND1_RESULT.exists():
        raise RuntimeError("Round-one result already exists; scoring is one-shot")
    rows = {}
    seeds = {}
    for key, fact in replay["emission_cache"].items():
        control = _load_control(fact)
        adopted = control["adopted"]
        calibrated = control["calibrated"]
        p_joint = categorical_blend(calibrated, adopted)
        p_cond = conditional_probability_blend(calibrated, adopted)
        p_logit = conditional_logit_blend(calibrated, adopted)
        scored = {
            ARMS[0]: _score_saved_probability(control, p_joint, capacity._decoder(3.0)),
            ARMS[1]: _score_saved_fixed_d(control, p_cond, adopted, 3.0),
            ARMS[2]: _score_saved_fixed_d(control, p_cond, adopted, 1.5),
            ARMS[3]: _score_saved_fixed_d(control, p_logit, adopted, 3.0),
            "adopted_original_full": _score_saved_probability(control, adopted, audit.CONFIG),
            "calibrated": _score_saved_probability(control, calibrated, audit.CONFIG),
        }
        expected_reference = scored["adopted_original_full"]["reference_spell_count"]
        if any(metrics["reference_spell_count"] != expected_reference
               for arm, metrics in scored.items() if arm in ARMS):
            raise RuntimeError(f"Reference spell count changed within matched cell {key}")
        rows[key] = scored
        p, m, view, rep = _split_key(key)
        for seed in SEEDS:
            seed_adopted = control[f"cand_seed{seed}"]
            seed_calibrated = control[f"adop_seed{seed}"]
            seed_key = f"{p}|{m}|{view}|{rep}|{seed}"
            seeds[seed_key] = {
                arm: metrics for arm, metrics in (
                    (ARMS[0], _score_saved_probability(control,
                        categorical_blend(seed_calibrated, seed_adopted), capacity._decoder(3.0))),
                    (ARMS[1], _score_saved_fixed_d(control,
                        conditional_probability_blend(seed_calibrated, seed_adopted), seed_adopted, 3.0)),
                    (ARMS[2], _score_saved_fixed_d(control,
                        conditional_probability_blend(seed_calibrated, seed_adopted), seed_adopted, 1.5)),
                    (ARMS[3], _score_saved_fixed_d(control,
                        conditional_logit_blend(seed_calibrated, seed_adopted), seed_adopted, 3.0))) }
    aggregate, gates, diagnostics = {}, {}, {}
    adopted_aggregate = replay["adopted_aggregate"]
    for arm in (*ARMS, "adopted_original_full", "calibrated"):
        aggregate[arm] = {}
        for metric in METRICS:
            aggregate[arm][metric] = _aggregate_arm(rows, arm, metric)
        quality = {view: {metric: aggregate[arm][metric][view]["equal_provider"]
                          for metric in METRICS} for view, _ in VIEWS}
        if arm in ARMS:
            adopted_quality = {view: {metric: adopted_aggregate[view][metric]["equal_provider"]
                                      for metric in METRICS} for view, _ in VIEWS}
            gates[arm] = hard_success(quality, adopted_quality)
        diagnostics[arm] = {view: {"weighted_70_30":
                            .7 * aggregate[arm]["spell_f1"][view]["equal_provider"] +
                            .3 * aggregate[arm]["macro_f1"][view]["equal_provider"],
                            "provider_metrics": {metric: aggregate[arm][metric][view]["providers"]
                                                 for metric in METRICS}}
                            for view, _ in VIEWS}
    ranking = {}
    for view, _ in VIEWS:
        ranking[view] = sorted(ARMS, key=lambda arm: (
            -aggregate[arm]["spell_f1"][view]["equal_provider"],
            -aggregate[arm]["macro_f1"][view]["equal_provider"], arm))
    seed_summary = _aggregate_seed_scores(seeds)
    result = {"study_id": STUDY, "round": 1,
              "declaration_sha256": sha(DECLARATION),
              "round1_declaration_sha256": sha(ROUND1_DECLARATION),
              "control_replay_sha256": sha(REPLAY), "arms": list(ARMS),
              "aggregate": aggregate, "strict_success_gates": gates,
              "diagnostics": diagnostics, "macro_tie_ordered_ranking": ranking,
              "seed_diagnostics": seed_summary,
              "matches": rows, "reference_spell_counts": replay["reference_spell_counts"],
              "adopted_reference_spell_counts_invariant": True,
              "organizer_reference_access": False, "public_or_GAR_scoring": False,
              "promotion": False}
    _write_once(ROUND1_RESULT, result)
    return result


def _aggregate_seed_scores(seed_rows: Mapping[str, dict]) -> dict:
    result = {}
    for arm in ARMS:
        result[arm] = {}
        for seed in SEEDS:
            result[arm][str(seed)] = {}
            for view, reps in VIEWS:
                result[arm][str(seed)][view] = {}
                for metric in QUALITY_METRICS:
                    values = {}
                    for key, row in seed_rows.items():
                        p, m, v, rep, seed_text = key.split("|")
                        if int(seed_text) == seed and v == view and int(rep) in reps:
                            values[(p, m, int(rep))] = float(row[arm][metric])
                    mean, providers = capacity.equal_provider(values)
                    result[arm][str(seed)][view][metric] = {
                        "equal_provider": mean, "providers": providers, "cells": len(values)}
    return result


def categorical_blend(candidate: np.ndarray, adopted: np.ndarray, weight: float = .5) -> np.ndarray:
    return _normalize((1.0 - weight) * np.asarray(adopted, dtype=np.float64) +
                      weight * np.asarray(candidate, dtype=np.float64))


def _conditional(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    mass = p[:, :2].sum(axis=1)
    if (mass <= 0).any():
        raise ValueError("H/A probability mass must be positive")
    return p[:, :2] / mass[:, None]


def conditional_probability_blend(candidate: np.ndarray, adopted: np.ndarray,
                                  weight: float = .5) -> np.ndarray:
    ha = _normalize((1.0 - weight) * _conditional(adopted) + weight * _conditional(candidate))
    return np.column_stack((ha, np.zeros(len(ha), dtype=np.float64)))


def conditional_logit_blend(candidate: np.ndarray, adopted: np.ndarray,
                            weight: float = .5) -> np.ndarray:
    a, b = _conditional(adopted), _conditional(candidate)
    logits = ((1.0 - weight) * np.log(np.clip(a, 1e-12, 1.0)) +
              weight * np.log(np.clip(b, 1e-12, 1.0)))
    logits -= logits.max(axis=1, keepdims=True)
    ha = _normalize(np.exp(logits))
    return np.column_stack((ha, np.zeros(len(ha), dtype=np.float64)))


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] not in (2, 3) or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Expected a finite, nonnegative N by 2/3 probability matrix")
    mass = p.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise ValueError("Probability rows must have positive mass")
    return p / mass


def fixed_d_prediction(probability: np.ndarray, adopted_probability: np.ndarray,
                       period: np.ndarray, fps: float, ha_cost: float) -> np.ndarray:
    return capacity._adopted_dead_prediction(probability, adopted_probability, period, fps,
                                               capacity._decoder(ha_cost))


def evaluate_candidate_cell(labels: np.ndarray, view: Mapping, period: np.ndarray,
                            adopted_probability: np.ndarray,
                            candidate_probability: np.ndarray) -> dict:
    """Score fixed round-one transforms; labels enter only official scoring."""
    adopted = _normalize(adopted_probability)
    candidate = _normalize(candidate_probability)
    if adopted.shape != candidate.shape or adopted.shape[0] != len(labels):
        raise ValueError("Control and candidate probability shapes must match the cell")
    cell = {"labels": labels, "period": view["period"], "timestamps": view["timestamps"],
            "fps": float(view["fps"])}
    p_joint = categorical_blend(candidate, adopted)
    p_cond = conditional_probability_blend(candidate, adopted)
    p_logit = conditional_logit_blend(candidate, adopted)
    predictions = {
        ARMS[0]: audit.decode(p_joint, period, cell["fps"], capacity._decoder(3.0)),
        ARMS[1]: fixed_d_prediction(p_cond, adopted, period, cell["fps"], 3.0),
        ARMS[2]: fixed_d_prediction(p_cond, adopted, period, cell["fps"], 1.5),
        ARMS[3]: fixed_d_prediction(p_logit, adopted, period, cell["fps"], 3.0),
    }
    return {arm: audit.score(cell, pred) for arm, pred in predictions.items()}


def hard_success(candidate: Mapping, adopted: Mapping, spell_floor: float = .60) -> dict:
    checks = {}
    for view, _ in VIEWS:
        for metric in QUALITY_METRICS:
            value, baseline = float(candidate[view][metric]), float(adopted[view][metric])
            checks[f"{view}/{metric}_vs_adopted"] = {"actual": value, "minimum": baseline,
                                                       "passed": value >= baseline}
    v2 = float(candidate["feed_like_v2"]["spell_f1"])
    checks["feed_like_v2/spell_f1_strictly_above_0_60"] = {
        "actual": v2, "minimum_exclusive": spell_floor, "passed": v2 > spell_floor}
    for view, _ in VIEWS:
        if "reference_spell_count" in candidate[view] and "reference_spell_count" in adopted[view]:
            actual = float(candidate[view]["reference_spell_count"])
            expected = float(adopted[view]["reference_spell_count"])
            checks[f"{view}/reference_spell_count_invariant"] = {
                "actual": actual, "expected": expected, "passed": actual == expected}
    return {"checks": checks, "all_passed": all(x["passed"] for x in checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "prepare-controls", "freeze-round1", "run-round1"))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "prepare-controls":
        prepare_controls()
    elif args.action == "freeze-round1":
        freeze_round1()
    else:
        run_round1()


if __name__ == "__main__":
    main()
