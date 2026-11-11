"""Evaluate the frozen spell-balanced shortlist on stage-1024 checkpoints.

This is deliberately separate from the frozen round-one source.  It scores
only the recipes selected by the pooled stage-512 development results and
keeps seed diagnostics out of selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import v2_spell_balanced_eval as r1
from experiments import v2_spell_balanced_train as trainer

ROOT = r1.ROOT
STUDY = r1.STUDY
DECLARATION = ROOT / f"experiments/results/{STUDY}_round2_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}_round2.json"
OUT = ROOT / "data/experiments" / STUDY / "round2"
SEEDS = tuple(trainer.SEEDS)
VIEWS = r1.VIEWS
METRICS = r1.METRICS
WORKERS = min(3, max(1, int(os.environ.get("V2_SPELL_BALANCED_WORKERS", "3"))))
TORCH_THREADS = max(1, int(os.environ.get("V2_SPELL_BALANCED_TORCH_THREADS", "1")))
_MODELS: dict[str, dict] = {}
_TEACHERS: dict[tuple[str, int], dict] = {}
_TRAINING_FACTS: dict[str, dict] = {}


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


def _source_hashes() -> dict[str, str]:
    frozen = r1.runtime.frozen
    paths = (Path(__file__), Path(r1.__file__), Path(trainer.__file__),
             Path(trainer.iterative.__file__), Path(trainer.prior_run.__file__),
             Path(frozen.__file__), Path(frozen.audit.__file__),
             Path(frozen.audit.decoder.__file__), Path(frozen.audit.metrics.__file__),
             Path(frozen.sn.__file__), Path(frozen.sn.cap.__file__),
             Path(frozen.sn.fh.__file__), Path(frozen.sn.lt.__file__),
             Path(frozen.old.__file__), Path(frozen.supervised.__file__),
             Path(frozen.tc.__file__), Path(frozen.v2.__file__),
             ROOT / "experiments/refine_features.py",
             ROOT / "experiments/learned_temporal.py",
             ROOT / "experiments/soccernet_capacity_model.py",
             ROOT / "evaluation/possession_metrics.py",
             r1.PLAN, r1.DECLARATION, r1.CONTROL_REPLAY, r1.RESULT,
             ROOT / "experiments/tests/test_v2_spell_balanced_eval.py",
             ROOT / "experiments/tests/test_v2_spell_balanced_round2.py",
             trainer.TRAIN_DECL, trainer.TRAIN_SEAL,
             ROOT / "experiments/baseline_manifest.json",
             ROOT / "experiments/model_decision_policy.json")
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def _expected_cells() -> set[str]:
    return {r1._cell_key(*cell) for cell in r1._expected_cells()}


def _training_facts() -> dict:
    trainer.check(full=True)
    seal = json.loads(trainer.TRAIN_SEAL.read_text())
    if (seal.get("study_id") != STUDY or
            seal.get("declaration_sha256") != sha(trainer.TRAIN_DECL) or
            seal.get("all_six_complete") is not True):
        raise RuntimeError("Stage-1024 continuation seal is incomplete or inconsistent")
    facts = {}
    for fold in trainer.FOLDS:
        for seed in SEEDS:
            model = trainer.output_path(fold, seed)
            receipt = Path(str(model) + ".json")
            rows = [row for row in seal.get("fits", [])
                    if row.get("fold") == fold and int(row.get("seed", -1)) == seed]
            if len(rows) != 1 or not model.is_file() or not receipt.is_file():
                raise RuntimeError(f"Missing or duplicate stage-1024 receipt: {fold}/{seed}")
            row = rows[0]
            if json.loads(receipt.read_text()) != row or row.get("sha256") != sha(model):
                raise RuntimeError(f"Stage-1024 fit changed after sealing: {fold}/{seed}")
            facts[f"{fold}/{seed}"] = {
                "path": str(model.relative_to(ROOT)), "sha256": sha(model),
                "receipt": str(receipt.relative_to(ROOT)), "receipt_sha256": sha(receipt),
                "fold": fold, "seed": seed}
    if len(facts) != 6:
        raise RuntimeError("Expected exactly six stage-1024 checkpoints")
    return facts


def _control_facts() -> dict:
    declaration = r1.require_declaration()
    replay = json.loads(r1.CONTROL_REPLAY.read_text())
    if (replay.get("passed") is not True or replay.get("candidate_scoring") is not False or
            set(replay.get("reference_spell_counts", {})) != _expected_cells()):
        raise RuntimeError("Round-one controls replay is not complete")
    facts = {}
    for key, fact in declaration["upstream"]["adopted_calibrated_emissions"].items():
        prepared = ROOT / fact["prepared_path"]
        emission = ROOT / fact["path"]
        if (not prepared.is_file() or sha(prepared) != fact["prepared_sha256"] or
                not emission.is_file() or sha(emission) != fact["sha256"]):
            raise RuntimeError(f"Frozen phase/control artifact changed: {key}")
        facts[key] = {"path": fact["path"], "sha256": fact["sha256"],
                      "prepared_path": fact["prepared_path"],
                      "prepared_sha256": fact["prepared_sha256"],
                      "identity": fact["identity"]}
    if set(facts) != _expected_cells():
        raise RuntimeError("Expected exactly 35 prepared matched cells")
    return facts


def _shortlist() -> list[dict]:
    result = json.loads(r1.RESULT.read_text())
    if result.get("declaration_sha256") != sha(r1.DECLARATION):
        raise RuntimeError("Round-one result does not bind its frozen declaration")
    rows = result.get("selected_for_round2", [])
    if not rows or len(rows) > 3 or len({r["recipe_id"] for r in rows}) != len(rows):
        raise RuntimeError("Round-one frozen shortlist must contain one to three unique recipes")
    if any(int(row.get("training_stage", -1)) != 512 for row in rows):
        raise RuntimeError("Round-two shortlist must refer to stage-512 recipes")
    allowed = {row["recipe_id"]: row for row in r1.RECIPES}
    for row in rows:
        expected = allowed.get(row["recipe_id"])
        if expected is None or any(row.get(name) != expected[name] for name in
                                   ("ce_weight", "calibrated_weight", "adopted_weight", "pooling", "ha_cost")):
            raise RuntimeError("Round-two recipe tuple differs from the frozen R1 grid")
    expected_shortlist = r1.select_seed_shortlist(result["eligibility"], result["aggregate"])
    if [row["recipe_id"] for row in rows] != expected_shortlist:
        raise RuntimeError("Round-two shortlist differs from the frozen R1 pooled ranking")
    if not any(row["recipe_id"] == r1.recipe_key(r1.ANCHOR) for row in rows):
        raise RuntimeError("Round-two shortlist omitted the fixed anchor")
    return rows


def freeze() -> dict:
    """Freeze training, cohort and exact R1 shortlist before stage-1024 scoring."""
    if DECLARATION.exists() or RESULT.exists():
        raise RuntimeError("Round-two output exists; preserve this study ID")
    if not r1.RESULT.is_file() or not r1.CONTROL_REPLAY.is_file():
        raise RuntimeError("Round one and controls-only replay must finish first")
    recipes = [{**{key: value for key, value in row.items() if key != "training_stage"},
                "training_stage": 1024} for row in _shortlist()]
    train_facts = _training_facts()
    controls = _control_facts()
    value = {
        "study_id": STUDY, "round": 2,
        "status": "frozen_before_stage1024_candidate_scoring",
        "parent": {"round1_declaration_sha256": sha(r1.DECLARATION),
                   "control_replay_sha256": sha(r1.CONTROL_REPLAY),
                   "round1_result_sha256": sha(r1.RESULT)},
        "training": {"declaration": str(trainer.TRAIN_DECL.relative_to(ROOT)),
                     "declaration_sha256": sha(trainer.TRAIN_DECL),
                     "seal": str(trainer.TRAIN_SEAL.relative_to(ROOT)),
                     "seal_sha256": sha(trainer.TRAIN_SEAL), "fits": train_facts},
        "training_stage": 1024,
        "recipes": recipes, "recipe_ids": [row["recipe_id"] for row in recipes],
        "cohort": {"cells": 35, "views": [view for view, _ in VIEWS],
                   "seeds": list(SEEDS), "provider_weighting": "equal provider"},
        "aggregation": "mean paired seeds within native phase, then mean native phases; seed diagnostics average phases within seed",
        "d_policy": "use predecoded adopted original_full D mask, unchanged for every recipe",
        "source_sha256": _source_hashes(), "controls": controls,
        "organizer_reference_access": False, "public_or_GAR_scoring": False,
        "promotion": False}
    _write_once(DECLARATION, value)
    return value


def require_declaration() -> dict:
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze round two before use")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("source_sha256") != _source_hashes():
        raise RuntimeError("Round-two source changed after declaration")
    if declaration.get("parent") != {
            "round1_declaration_sha256": sha(r1.DECLARATION),
            "control_replay_sha256": sha(r1.CONTROL_REPLAY),
            "round1_result_sha256": sha(r1.RESULT)}:
        raise RuntimeError("Round-one parent artifacts changed after round-two freeze")
    if declaration.get("training", {}).get("seal_sha256") != sha(trainer.TRAIN_SEAL):
        raise RuntimeError("Stage-1024 training seal changed after round-two freeze")
    if declaration.get("controls") != _control_facts():
        raise RuntimeError("Prepared cell controls changed after round-two freeze")
    _training_facts()
    if declaration.get("recipes") != _shortlist():
        raise RuntimeError("Round-two recipes differ from the frozen round-one shortlist")
    return declaration


def _load_control(key: str, fact: dict) -> dict:
    with np.load(ROOT / fact["path"], allow_pickle=False) as saved:
        return {name.removeprefix("prob__"): saved[name].copy()
                for name in saved.files if name.startswith("prob__")} | {
            "labels": saved["labels"].copy(), "period": saved["period"].copy(),
            "timestamps": saved["timestamps"].copy(),
            "model_period": saved["model_period"].copy(), "fps": float(saved["fps"])}


def _load_prepared(fact: dict) -> dict:
    path = ROOT / fact["prepared_path"]
    prepared = joblib.load(path)
    if prepared.get("identity") != fact["identity"]:
        raise RuntimeError(f"Prepared identity mismatch: {path}")
    return prepared


def _stage_model(provider: str, seed: int) -> dict:
    fold = f"leave_out_{r1.runtime.frozen.audit.FOLD[provider]}"
    key = f"{fold}/{seed}"
    if key not in _MODELS:
        fact = _TRAINING_FACTS.get(key)
        if fact is None:
            raise RuntimeError("Stage-1024 checkpoint facts were not initialized")
        _MODELS[key] = joblib.load(ROOT / fact["path"])
    return _MODELS[key]


def _predict_phases(prepared: dict, provider: str, seed: int) -> list[np.ndarray]:
    model = _stage_model(provider, seed)
    teacher_key = (provider, seed)
    if teacher_key not in _TEACHERS:
        _TEACHERS[teacher_key] = joblib.load(r1.runtime.frozen._teacher_path(provider, seed))
    teacher = _TEACHERS[teacher_key]
    view = prepared["view"]
    predictions = []
    for record in prepared["phases"]:
        infer_record = r1.runtime.frozen.old.inference_record(record)
        infer_record["names"] = list(infer_record["names"])
        sampled = r1.runtime.frozen.sn.predict(infer_record, model, teacher)
        predictions.append(r1.runtime.frozen.old.upsample(
            sampled, infer_record, r1.runtime.frozen.tc.inputs_only(view)))
    return predictions


def _mean_seed_phase(phase_outputs: dict[int, list[np.ndarray]]) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if set(phase_outputs) != set(SEEDS) or not phase_outputs[SEEDS[0]]:
        raise ValueError("Expected both paired seeds and at least one native phase")
    phase_count = len(phase_outputs[SEEDS[0]])
    if any(len(phase_outputs[seed]) != phase_count for seed in SEEDS):
        raise ValueError("Paired seeds must have identical native phase inventory")
    for phase in range(phase_count):
        shapes = {np.asarray(phase_outputs[seed][phase]).shape for seed in SEEDS}
        if len(shapes) != 1:
            raise ValueError("Paired seed probability shapes differ")
    pooled = np.mean([np.mean([phase_outputs[seed][i] for seed in SEEDS], axis=0)
                      for i in range(phase_count)], axis=0)
    seed_means = {seed: np.mean(phase_outputs[seed], axis=0) for seed in SEEDS}
    return pooled, seed_means


def _cell_job(job: tuple[str, dict, list[dict], str]):
    key, fact, recipes, declaration_hash = job
    cell_control = _load_control(key, fact)
    prepared = _load_prepared(fact)
    provider = key.split("|", 1)[0]
    phase_outputs = {seed: _predict_phases(prepared, provider, seed) for seed in SEEDS}
    pooled_ce, seed_ce = _mean_seed_phase(phase_outputs)
    adopted = cell_control["adopted"]
    adopted_decoded = r1.runtime.frozen.audit.decode(adopted, cell_control["model_period"],
        cell_control["fps"], r1.runtime.frozen.audit.CONFIG)
    cell = {name: cell_control[name] for name in ("labels", "period", "timestamps", "fps")}
    result, seed_rows, predictions, seed_predictions = {}, {}, {}, {}
    for recipe in recipes:
        rid = recipe["recipe_id"]
        comps = {"ce": pooled_ce, "adopted": adopted,
                 "calibrated": cell_control["calibrated"]}
        p = r1.recipe_probability(comps, recipe["ce_weight"],
                                  recipe["calibrated_weight"], recipe["pooling"])
        pred = r1.decode_fixed_d(p, adopted_decoded, cell_control["model_period"],
                                 cell_control["fps"], recipe["ha_cost"])
        metrics = r1.controls.audit.score(cell, pred)
        result[rid] = metrics
        predictions[rid] = r1._encode_prediction(pred)
        for seed in SEEDS:
            seed_comps = {"ce": seed_ce[seed], "adopted": cell_control[f"cand_seed{seed}"],
                          "calibrated": cell_control[f"adop_seed{seed}"]}
            seed_p = r1.recipe_probability(seed_comps, recipe["ce_weight"],
                recipe["calibrated_weight"], recipe["pooling"])
            seed_pred = r1.decode_fixed_d(seed_p, adopted_decoded,
                cell_control["model_period"], cell_control["fps"], recipe["ha_cost"])
            seed_rows[f"{rid}|{seed}"] = r1.controls.audit.score(cell, seed_pred)
            seed_predictions[f"{rid}|{seed}"] = r1._encode_prediction(seed_pred)
    reference = r1.controls.audit.score(cell, adopted_decoded)["reference_spell_count"]
    if any(row["reference_spell_count"] != reference for row in result.values()):
        raise RuntimeError(f"Reference spell count changed for {key}")
    out = OUT / "predictions" / f"{key.replace('|', '__')}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"pred__{rid}": pred for rid, pred in predictions.items()}
    payload.update({f"seedpred__{name}": pred for name, pred in seed_predictions.items()})
    payload.update({"pooled_ce": pooled_ce})
    payload.update({f"seed_ce_{seed}": seed_ce[seed] for seed in SEEDS})
    payload.update({f"phase_ce_seed{seed}_{phase}": probability
                    for seed in SEEDS for phase, probability in enumerate(phase_outputs[seed])})
    identity = {"study_id": STUDY, "declaration_sha256": declaration_hash, "cell": key,
                "phase_count": len(prepared["phases"])}
    payload.update({f"identity__{name}": np.asarray(str(value)) for name, value in identity.items()})
    np.savez_compressed(out, **payload)
    return key, result, seed_rows, reference, {"path": str(out.relative_to(ROOT)),
        "sha256": sha(out), "identity": identity}


def _worker_init() -> None:
    try:
        import torch
        torch.set_num_threads(TORCH_THREADS)
    except ImportError:
        pass


def run_round2() -> dict:
    declaration = require_declaration()
    global _TRAINING_FACTS
    _TRAINING_FACTS = declaration["training"]["fits"]
    if RESULT.exists():
        raise RuntimeError("Round-two result already exists; scoring is one-shot")
    jobs = [(key, fact, declaration["recipes"], sha(DECLARATION))
            for key, fact in sorted(declaration["controls"].items())]
    cell_rows, seed_rows, references, inventory = {}, {}, {}, {}
    with ProcessPoolExecutor(max_workers=min(WORKERS, len(jobs)), initializer=_worker_init) as pool:
        futures = [pool.submit(_cell_job, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            key, metrics, seeds, reference, fact = future.result()
            cell_rows[key], seed_rows[key], references[key], inventory[key] = metrics, seeds, reference, fact
            print(f"BALANCED R2 {index}/{len(jobs)} {key}", flush=True)
    recipe_ids = declaration["recipe_ids"]
    if set(cell_rows) != _expected_cells() or any(set(rows) != set(recipe_ids) for rows in cell_rows.values()):
        raise RuntimeError("Round-two scorer did not cover every shortlist recipe and matched cell")
    aggregate = {rid: r1.aggregate_cell_metrics({key: row[rid] for key, row in cell_rows.items()})
                 for rid in recipe_ids}
    baseline = json.loads(r1.CONTROL_REPLAY.read_text())
    baseline_counts = baseline["reference_spell_counts"]
    eligibility = {}
    for rid in recipe_ids:
        candidate = {view: {metric: aggregate[rid][view][metric]["equal_provider"] for metric in METRICS}
                     for view, _ in VIEWS}
        base_metrics = {view: {metric: baseline["adopted"][view][metric]["equal_provider"]
                               for metric in METRICS} for view, _ in VIEWS}
        gates = r1.eligibility(candidate, base_metrics)
        gates["d_metrics_unchanged"] = all(candidate[view][metric] == base_metrics[view][metric]
            for view, _ in VIEWS for metric in ("d_f1", "d_precision", "d_recall"))
        gates["reference_spell_counts_invariant"] = all(references[key] == baseline_counts[key]
                                                         for key in references)
        gates["eligible"] = gates["eligible"] and gates["d_metrics_unchanged"] and gates["reference_spell_counts_invariant"]
        eligibility[rid] = gates
    seed_diagnostics = r1.aggregate_seed_metrics(seed_rows, recipe_ids)
    result = {"study_id": STUDY, "round": 2,
        "training_stage": 1024,
        "declaration_sha256": sha(DECLARATION),
        "round1_result_sha256": sha(r1.RESULT), "training_seal_sha256": sha(trainer.TRAIN_SEAL),
        "recipes": declaration["recipes"], "recipe_ids": recipe_ids,
        "cohort": {"cell_count": len(cell_rows), "reference_spell_counts": references},
        "aggregate": aggregate, "eligibility": eligibility,
        "v2_target_aspiration": {rid: {"actual": aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"],
            "target_strictly_above": .60,
            "passed": aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"] > .60} for rid in recipe_ids},
        "seed_diagnostics": seed_diagnostics, "seed_cells": seed_rows,
        "cell_metrics": cell_rows, "prediction_cache": inventory,
        "organizer_reference_access": False, "public_or_GAR_scoring": False, "promotion": False}
    _write_once(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run-round2"))
    action = parser.parse_args().action
    (freeze if action == "freeze" else run_round2)()


if __name__ == "__main__":
    main()
