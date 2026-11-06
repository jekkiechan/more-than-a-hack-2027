"""Round-two evaluation for the paired v2 iterative H/A continuation fits.

This is a separate append-only source from the frozen control and round-one
evaluator. It consumes only the already prepared development cells, archived
original teachers, the declared round-two checkpoint seal, and official score
helpers. Organizer and GAR evaluation labels are not available here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np

from experiments import v2_iterative_eval as base
from experiments import v2_iterative_train as trainer
from experiments import v2_iterative_train_run as training_run

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_iterative_v1_round2"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
SEEDS = tuple(trainer.SEEDS)
TRAIN_ARMS = tuple(trainer.ARMS)
ARMS = ("ce_control_fixed_d_ha3", "boundary_margin_fixed_d_ha3",
        "ce_control_conditional_blend_fixed_d_ha3",
        "boundary_margin_conditional_blend_fixed_d_ha3")
METRICS = base.METRICS
QUALITY = base.QUALITY_METRICS
WORKERS = max(1, int(os.environ.get("V2_ITERATIVE_WORKERS", "2")))
TORCH_THREADS = max(1, int(os.environ.get("V2_ITERATIVE_TORCH_THREADS", "1")))


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def _equal_provider(values: dict) -> tuple[float, dict]:
    """Use canonical cell order so worker completion order cannot change a tie."""
    ordered = {key: values[key] for key in sorted(values)}
    return base.capacity.equal_provider(ordered)


def _source_hashes() -> dict[str, str]:
    frozen = base.runtime.frozen
    paths = [Path(__file__), Path(base.__file__), Path(training_run.__file__),
             Path(trainer.__file__), Path(frozen.__file__), Path(frozen.audit.__file__),
             Path(frozen.sn.__file__), Path(frozen.old.__file__),
             Path(frozen.supervised.__file__), Path(frozen.audit.decoder.__file__),
             Path(frozen.audit.metrics.__file__), Path(frozen.sn.cap.__file__),
             Path(frozen.sn.fh.__file__), Path(frozen.sn.lt.__file__),
             Path(ROOT / "evaluation/possession_metrics.py"),
             Path(ROOT / "experiments/soccernet_capacity_model.py"),
             Path(ROOT / "experiments/soccernet_extension.py"),
             ROOT / f"experiments/results/{base.STUDY}_plan.json"]
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
            for path in paths}


def _fit_facts() -> tuple[dict, dict]:
    if not training_run.SEAL.is_file() or not training_run.DECL.is_file():
        raise FileNotFoundError("Training declaration and completed fit seal are required")
    declaration = training_run.check(full=False)
    seal = json.loads(training_run.SEAL.read_text())
    if (seal.get("study_id") != training_run.STUDY or
            seal.get("declaration_sha256") != sha(training_run.DECL) or
            seal.get("all_12_complete") is not True):
        raise RuntimeError("Iterative training seal is incomplete or inconsistent")
    fits = {}
    for arm in TRAIN_ARMS:
        for fold in trainer.FOLDS:
            for seed in SEEDS:
                model = training_run.model_path(arm, fold, seed)
                receipt = Path(str(model) + ".json")
                model_rows = [row for row in seal.get("fits", [])
                              if (row.get("arm"), row.get("fold"), row.get("seed")) == (arm, fold, seed)]
                if len(model_rows) != 1 or not model.is_file() or not receipt.is_file():
                    raise RuntimeError(f"Missing or duplicate fit receipt: {arm}/{fold}/{seed}")
                fact = model_rows[0]
                receipt_value = json.loads(receipt.read_text())
                if (fact.get("sha256") != sha(model) or receipt_value != fact):
                    raise RuntimeError(f"Fit changed after training seal: {arm}/{fold}/{seed}")
                fits[f"{arm}/{fold}/{seed}"] = {
                    "path": str(model.relative_to(ROOT)), "sha256": sha(model),
                    "receipt": str(receipt.relative_to(ROOT)), "receipt_sha256": sha(receipt)}
    return declaration, fits


def _control_facts() -> dict:
    if not base.REPLAY.is_file() or not base.DECLARATION.is_file():
        raise RuntimeError("Frozen control replay is required")
    base._require_frozen()
    replay = json.loads(base.REPLAY.read_text())
    if replay.get("passed") is not True or replay.get("candidate_scoring") is not False:
        raise RuntimeError("Adopted control parity did not pass")
    expected = {base.cell_key(provider, match, view, rep)
                for provider, match, view, rep in base._views()}
    if len(expected) != 35 or set(replay["emission_cache"]) != expected:
        raise RuntimeError("Round two requires the exact 35-cell control cohort")
    facts = {}
    for key, fact in replay["emission_cache"].items():
        control_path = ROOT / fact["path"]
        prepared_path = ROOT / fact["prepared_path"]
        if (not control_path.is_file() or sha(control_path) != fact["sha256"] or
                not prepared_path.is_file() or sha(prepared_path) != fact["prepared_sha256"]):
            raise RuntimeError(f"Frozen control or prepared phase cache changed: {key}")
        facts[key] = {"control_path": fact["path"], "control_sha256": fact["sha256"],
                      "prepared_path": fact["prepared_path"],
                      "prepared_sha256": fact["prepared_sha256"],
                      "identity": fact["identity"]}
    return facts


def freeze() -> dict:
    """Bind the completed trainer, controls, inputs and round-two protocol."""
    base._require_frozen()
    if not base.ROUND1_RESULT.is_file():
        raise RuntimeError("Run the parity-passed round one before freezing round two")
    if DECLARATION.exists() or RESULT.exists():
        raise RuntimeError("Round-two output already exists; preserve the study ID")
    train_declaration, fits = _fit_facts()
    controls = _control_facts()
    round1_hash = sha(base.ROUND1_RESULT)
    declaration = {
        "study_id": STUDY, "round": 2, "status": "frozen_before_candidate_scoring",
        "question": "Does warm-start H/A continuation with boundary-local loss improve the fixed adopted-D H/A pipeline on matched clean/v1/v2 cells?",
        "arms": list(ARMS),
        "training": {"declaration": str(training_run.DECL.relative_to(ROOT)),
                     "declaration_sha256": sha(training_run.DECL),
                     "seal": str(training_run.SEAL.relative_to(ROOT)),
                     "seal_sha256": sha(training_run.SEAL), "fits": fits,
                     "training_source_sha256": train_declaration["source_sha256"]},
        "controls": controls,
        "parent": {"declaration_sha256": sha(base.DECLARATION),
                   "control_replay_sha256": sha(base.REPLAY),
                   "round1_declaration_sha256": sha(base.ROUND1_DECLARATION),
                   "round1_result_sha256": round1_hash},
        "methods": {
            "ce_control_fixed_d_ha3": "ce_control continuation probabilities; preserve the frozen adopted original_full D mask; decode conditional H/A at ha=3",
            "boundary_margin_fixed_d_ha3": "boundary_margin continuation probabilities; preserve frozen adopted D mask; decode conditional H/A at ha=3",
            "ce_control_conditional_blend_fixed_d_ha3": "uniform conditional H/A probability mean of ce_control continuation and adopted original_full; fixed adopted D mask; ha=3",
            "boundary_margin_conditional_blend_fixed_d_ha3": "uniform conditional H/A probability mean of boundary_margin continuation and adopted original_full; fixed adopted D mask; ha=3"},
        "aggregation": "average paired seed probabilities within each phase, then average native phases; report each seed separately",
        "hard_success": {"metrics": list(QUALITY), "views": [v for v, _ in base.VIEWS],
                         "no_regression_vs_adopted_zero_tolerance": True,
                         "feed_like_v2_spell_f1_strictly_above": 0.60,
                         "reference_spell_counts_invariant": True},
        "diagnostics": ["per_provider metrics", "per_seed metrics", "predicted spell counts",
                        "macro tie ordering", "0.7 spell + 0.3 macro"],
        "source_sha256": _source_hashes(), "worker_default": WORKERS,
        "torch_threads_per_worker": TORCH_THREADS,
        "organizer_reference_access": False, "public_or_GAR_scoring": False,
        "promotion": False,
    }
    _write_once(DECLARATION, declaration)
    return declaration


def _load_prepared(fact: dict) -> dict:
    path = ROOT / fact["prepared_path"]
    if sha(path) != fact["prepared_sha256"]:
        raise RuntimeError(f"Prepared phase records changed: {path}")
    prepared = joblib.load(path)
    if prepared.get("identity") != fact["identity"]:
        raise RuntimeError(f"Prepared phase identity mismatch: {path}")
    return prepared


def _predict_seed(view: dict, record: dict, provider: str, arm: str, seed: int) -> np.ndarray:
    fold = f"leave_out_{base.runtime.frozen.audit.FOLD[provider]}"
    key = f"{arm}/{fold}/{seed}"
    model_fact = _FITS[key]
    model = _MODELS.get(key)
    if model is None:
        model = joblib.load(ROOT / model_fact["path"])
        _MODELS[key] = model
    teacher_key = (provider, seed)
    teacher = _TEACHERS.get(teacher_key)
    if teacher is None:
        teacher = joblib.load(base.runtime.frozen._teacher_path(provider, seed))
        _TEACHERS[teacher_key] = teacher
    infer_record = base.runtime.frozen.old.inference_record(record)
    infer_record["names"] = list(infer_record["names"])
    sampled = base.runtime.frozen.sn.predict(infer_record, model, teacher)
    return base.runtime.frozen.old.upsample(
        sampled, infer_record, base.runtime.frozen.tc.inputs_only(view))


def _control_arrays(fact: dict) -> dict:
    with np.load(ROOT / fact["control_path"], allow_pickle=False) as saved:
        return {k.removeprefix("prob__"): saved[k].copy()
                for k in saved.files if k.startswith("prob__")} | {
                    "labels": saved["labels"].copy(), "period": saved["period"].copy(),
                    "timestamps": saved["timestamps"].copy(),
                    "model_period": saved["model_period"].copy(), "fps": float(saved["fps"])}


def _score_arm(control: dict, candidate: np.ndarray, arm: str) -> tuple[dict, np.ndarray]:
    adopted = control["adopted"]
    if "conditional_blend" in arm:
        candidate = base.conditional_probability_blend(candidate, adopted)
    prediction = base.fixed_d_prediction(candidate, adopted, control["model_period"],
                                          control["fps"], 3.0)
    cell = {"labels": control["labels"], "period": control["period"],
            "timestamps": control["timestamps"], "fps": control["fps"]}
    return base.audit.score(cell, prediction), prediction


def _cell_job(job):
    key, control_fact, arm = job
    prepared = _load_prepared(control_fact)
    control = _control_arrays(control_fact)
    view = prepared["view"]
    phases = prepared["phases"]
    candidate_by_seed, phase_by_seed = {}, {}
    for train_arm in TRAIN_ARMS:
        if arm.startswith(train_arm):
            selected_train_arm = train_arm
            break
    else:
        raise ValueError(f"Round-two arm has no training recipe: {arm}")
    provider = prepared["identity"]["cell"].split("|")[0]
    for seed in SEEDS:
        phase_by_seed[seed] = [_predict_seed(view, record, provider, selected_train_arm, seed)
                               for record in phases]
        candidate_by_seed[seed] = np.mean(phase_by_seed[seed], axis=0)
    # Deployment-compatible pool: average seeds inside each native phase, then phases.
    all_phase_seed = [np.mean([phase_by_seed[seed][phase] for seed in SEEDS], axis=0)
                      for phase in range(len(phases))]
    pooled_candidate = np.mean(all_phase_seed, axis=0)
    result = {}
    predictions = {}
    for out_arm in (ARMS[0], ARMS[1], ARMS[2], ARMS[3]):
        if out_arm.startswith(selected_train_arm):
            result[out_arm], predictions[out_arm] = _score_arm(control, pooled_candidate, out_arm)
    cell = {"labels": control["labels"], "period": control["period"],
            "timestamps": control["timestamps"], "fps": control["fps"]}
    adopted_prediction = base.audit.decode(control["adopted"], control["model_period"],
                                           control["fps"], base.audit.CONFIG)
    adopted_metrics = base.audit.score(cell, adopted_prediction)
    reference_count = adopted_metrics["reference_spell_count"]
    if any(metrics["reference_spell_count"] != reference_count for metrics in result.values()):
        raise RuntimeError(f"Round-two arm changed the reference spell count in {key}")
    seed_metrics, seed_predictions = {}, {}
    for seed, probability in candidate_by_seed.items():
        seed_arm = f"{selected_train_arm}_fixed_d_ha3"
        metrics, pred = _score_arm(control, probability, seed_arm)
        seed_metrics[str(seed)], seed_predictions[str(seed)] = metrics, pred
        seed_blend = f"{selected_train_arm}_conditional_blend_fixed_d_ha3"
        blend_metrics, blend_pred = _score_arm(control, probability, seed_blend)
        seed_metrics[str(seed) + "_blend"], seed_predictions[str(seed) + "_blend"] = blend_metrics, blend_pred
    for seed_name, metrics in seed_metrics.items():
        if metrics["reference_spell_count"] != reference_count:
            raise RuntimeError(f"Seed diagnostic changed reference spell count in {key}/{seed_name}")
    return key, selected_train_arm, result, predictions, candidate_by_seed, pooled_candidate, seed_metrics, seed_predictions


def _worker_init(fits=None):
    global _FITS
    _FITS = fits or {}
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except ImportError:
        pass


def _aggregate_seed_scores(seed_rows: dict) -> dict:
    bucket = {}
    for key, row in seed_rows.items():
        provider, match, view, rep, train_arm, seed_label = key.split("|")
        rep = int(rep)
        if seed_label.endswith("_blend"):
            seed, variant = seed_label.removesuffix("_blend"), "conditional_blend"
        else:
            seed, variant = seed_label, "fixed_d"
        target = bucket.setdefault(train_arm, {}).setdefault(variant, {}).setdefault(seed, {}).setdefault(view, {})
        for metric in QUALITY:
            target.setdefault(metric, {})[(provider, match, rep)] = float(row[metric])
    result = {}
    for train_arm, variants in bucket.items():
        result[train_arm] = {}
        for variant, seeds in variants.items():
            result[train_arm][variant] = {}
            for seed, views in seeds.items():
                result[train_arm][variant][seed] = {}
                for view, metrics in views.items():
                    result[train_arm][variant][seed][view] = {}
                    allowed_reps = dict(base.VIEWS)[view]
                    for metric, values in metrics.items():
                        selected = {key: value for key, value in values.items() if key[2] in allowed_reps}
                        mean, providers = _equal_provider(selected)
                        result[train_arm][variant][seed][view][metric] = {
                            "equal_provider": mean, "providers": providers, "cells": len(selected)}
    return result


def run() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze round-two declaration before candidate scoring")
    declaration = json.loads(DECLARATION.read_text())
    if declaration["source_sha256"] != _source_hashes():
        raise RuntimeError("Round-two source changed after freeze")
    if sha(training_run.SEAL) != declaration["training"]["seal_sha256"]:
        raise RuntimeError("Training seal changed after round-two freeze")
    _FIT_DECL, live_fits = _fit_facts()
    if live_fits != declaration["training"]["fits"]:
        raise RuntimeError("A round-two model or receipt changed after freeze")
    controls = _control_facts()
    if controls != declaration["controls"]:
        raise RuntimeError("Frozen round-two control/prepared inputs changed")
    if RESULT.exists():
        raise RuntimeError("Round-two result already exists; scoring is one-shot")
    rows, seed_rows, emissions = {}, {}, {}
    jobs = [(key, fact, arm) for key, fact in controls.items() for arm in TRAIN_ARMS]
    workers = min(WORKERS, len(jobs))
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(declaration["training"]["fits"],)) as pool:
        futures = {pool.submit(_cell_job, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            key, train_arm, scored, predictions, per_seed, pooled, seed_metrics, seed_predictions = future.result()
            cell = rows.setdefault(key, {arm: {} for arm in ARMS})
            for arm, metrics in scored.items():
                cell[arm] = metrics
            for seed_key, metrics in seed_metrics.items():
                seed_rows[f"{key}|{train_arm}|{seed_key}"] = metrics
            path = OUT / "emissions" / f"{key.replace('|', '__')}__{train_arm}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            control_fact = controls[key]
            identity = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
                        "cell": key, "training_arm": train_arm,
                        "control_sha256": control_fact["control_sha256"]}
            arrays = {f"prob__seed{seed}": probability for seed, probability in per_seed.items()}
            arrays["prob__pooled"] = pooled
            arrays.update({f"pred__{arm}": pred for arm, pred in predictions.items()})
            arrays.update({f"pred__seed__{seed}": pred for seed, pred in seed_predictions.items()})
            arrays.update({f"identity__{name}": np.asarray(str(value)) for name, value in identity.items()})
            np.savez_compressed(path, **arrays)
            emissions[f"{train_arm}/{key}"] = {"path": str(path.relative_to(ROOT)),
                                                "sha256": sha(path), "identity": identity}
            print(f"ROUND2 {done}/{len(jobs)} {train_arm}/{key}", flush=True)
    if (len(rows) != len(controls) or
            any(set(cell) != set(ARMS) or any(not cell[arm] for arm in ARMS)
                for cell in rows.values())):
        raise RuntimeError("Round-two result does not contain all four arms for all 35 cells")
    if len(seed_rows) != len(controls) * len(TRAIN_ARMS) * len(SEEDS) * 2:
        raise RuntimeError("Round-two seed diagnostics are incomplete")
    aggregate = {}
    adopted = json.loads(base.REPLAY.read_text())["adopted_aggregate"]
    gates = {}
    diagnostics = {}
    for arm in ARMS:
        aggregate[arm] = {}
        for view, reps in base.VIEWS:
            aggregate[arm][view] = {}
            for metric in METRICS:
                values = {}
                for key, arm_rows in rows.items():
                    provider, match, cell_view, rep = base._split_key(key)
                    if cell_view == view and rep in reps and arm in arm_rows:
                        values[(provider, match, rep)] = float(arm_rows[arm][metric])
                mean, providers = _equal_provider(values)
                aggregate[arm][view][metric] = {"equal_provider": mean, "providers": providers}
        candidate = {view: {metric: aggregate[arm][view][metric]["equal_provider"] for metric in METRICS}
                     for view, _ in base.VIEWS}
        adopted_values = {view: {metric: adopted[view][metric]["equal_provider"] for metric in METRICS}
                          for view, _ in base.VIEWS}
        gates[arm] = base.hard_success(candidate, adopted_values)
        diagnostics[arm] = {view: {
            "weighted_70_30": .7 * aggregate[arm][view]["spell_f1"]["equal_provider"] +
                              .3 * aggregate[arm][view]["macro_f1"]["equal_provider"],
            "provider_metrics": {metric: aggregate[arm][view][metric]["providers"] for metric in METRICS}}
            for view, _ in base.VIEWS}
    rankings = {view: sorted(ARMS, key=lambda arm: (
        -aggregate[arm][view]["spell_f1"]["equal_provider"],
        -aggregate[arm][view]["macro_f1"]["equal_provider"], arm)) for view, _ in base.VIEWS}
    seed_aggregate = _aggregate_seed_scores(seed_rows)
    result = {"study_id": STUDY, "round": 2,
              "declaration_sha256": sha(DECLARATION),
              "round1_result_sha256": declaration["parent"]["round1_result_sha256"],
              "aggregate": aggregate, "strict_success_gates": gates,
              "diagnostics": diagnostics, "macro_tie_ordered_ranking": rankings,
              "matches": rows, "seed_metrics": seed_rows,
              "seed_aggregate": seed_aggregate, "candidate_emissions": emissions,
              "organizer_reference_access": False, "public_or_GAR_scoring": False,
              "promotion": False}
    _write_once(RESULT, result)
    return result


_FITS = {}
_MODELS = {}
_TEACHERS = {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "run": run}[action]()


if __name__ == "__main__":
    main()
