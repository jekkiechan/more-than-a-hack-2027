"""Independent saved-output audit for v2_spell_balanced_v1 development rounds."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_spell_balanced_eval as study
from experiments import v2_spell_balanced_round2 as round2_study

ROOT = study.ROOT
R1_RESULT = study.RESULT
R1_DECL = study.DECLARATION
CONTROL_REPLAY = study.CONTROL_REPLAY
R2_RESULT = round2_study.RESULT
R2_DECL = round2_study.DECLARATION
R2_EXECUTION_RECEIPT = ROOT / "experiments/results/v2_spell_balanced_v1_round2_execution_receipt.json"
R2_EXECUTION_AMENDMENT = ROOT / "experiments/results/v2_spell_balanced_v1_round2_execution_amendment.json"
ATOL = 1e-12
AUDIT_PLAN = ROOT / "experiments/results/v2_spell_balanced_v1_audit_plan.json"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def _decode_saved(encoded: np.ndarray) -> np.ndarray:
    if not np.isin(encoded, (0, 1, 2, 3)).all():
        raise AssertionError("Prediction cache has invalid class codes")
    return np.asarray(("H", "A", "D", "X"), dtype="U1")[encoded]


def _mix(components: dict, recipe: dict, *, seed=None) -> np.ndarray:
    """Independently form the conditional H/A mixture from expert probabilities."""
    values = components if seed is None else components["seed"][seed]
    weights = {"ce": float(recipe["ce_weight"]),
               "calibrated": float(recipe["calibrated_weight"]),
               "adopted": float(recipe["adopted_weight"])}
    members = []
    for name, weight in weights.items():
        if weight <= 0:
            continue
        p = np.asarray(values[name], dtype=np.float64)
        ha = p[:, :2].copy()
        ha /= ha.sum(axis=1, keepdims=True)
        members.append((ha, weight))
    if recipe["pooling"] == "arithmetic":
        result = sum(weight * ha for ha, weight in members)
    else:
        logits = sum(weight * np.log(np.clip(ha, 1e-12, 1.0)) for ha, weight in members)
        logits -= logits.max(axis=1, keepdims=True)
        result = np.exp(logits)
    return result / result.sum(axis=1, keepdims=True)


def _equal_provider(rows: dict[tuple[str, str, int], float]) -> tuple[float, dict]:
    grouped = {}
    for (provider, match, _rep), value in sorted(rows.items()):
        grouped.setdefault(provider, {}).setdefault(match, []).append(float(value))
    providers = {p: float(np.mean([np.mean(reps) for reps in matches.values()]))
                 for p, matches in grouped.items()}
    return float(np.mean(list(providers.values()))), providers


def aggregate_cell_metrics(cell_metrics: dict[str, dict[str, dict]], recipe_ids: list[str]) -> dict:
    result = {}
    for recipe_id in recipe_ids:
        result[recipe_id] = {}
        for view, reps in study.VIEWS:
            result[recipe_id][view] = {}
            for metric in study.METRICS:
                values = {}
                for key in sorted(cell_metrics):
                    provider, match, cell_view, rep = key.split("|")
                    if cell_view == view and int(rep) in reps:
                        values[(provider, match, int(rep))] = cell_metrics[key][recipe_id][metric]
                mean, providers = _equal_provider(values)
                result[recipe_id][view][metric] = {"equal_provider": mean,
                    "providers": providers, "cells": len(values)}
    return result


def aggregate_seed_metrics(seed_rows: dict[str, dict], recipe_ids: list[str]) -> dict:
    result = {}
    for recipe_id in recipe_ids:
        result[recipe_id] = {}
        for seed in study.SEEDS:
            rows = {key: values[f"{recipe_id}|{seed}"] for key, values in seed_rows.items()}
            result[recipe_id][str(seed)] = aggregate_cell_metrics(
                {key: {recipe_id: row} for key, row in rows.items()}, [recipe_id])[recipe_id]
    return result


def _seed_aggregate(seed_rows: dict[str, dict], recipe_ids: list[str]) -> dict:
    """Independent seed reducer; seed_rows is cell -> recipe|seed -> metrics."""
    result = {}
    for rid in recipe_ids:
        result[rid] = {}
        for seed in study.SEEDS:
            one_seed = {cell: {rid: rows[f"{rid}|{seed}"]}
                        for cell, rows in seed_rows.items()}
            result[rid][str(seed)] = aggregate_cell_metrics(one_seed, [rid])[rid]
    return result


def _score_rows(rows_by_cell: dict, recipe_ids: list[str]) -> dict:
    """Independent reducer for round-two native cell score rows."""
    return aggregate_cell_metrics(rows_by_cell, recipe_ids)


def _phase_count(fps: float) -> int:
    count = int(round(float(fps) / 5.0))
    if count < 1 or not np.isclose(float(fps) / count, 5.0, rtol=0, atol=1e-12):
        raise AssertionError(f"Unexpected native phase count for fps={fps}")
    return count


def _write_once(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(contents)


def verify_round1() -> dict:
    declaration, result = json.loads(R1_DECL.read_text()), json.loads(R1_RESULT.read_text())
    replay = json.loads(CONTROL_REPLAY.read_text())
    if declaration.get("source_sha256") != study._source_hashes():
        raise AssertionError("Round-one frozen source inventory changed")
    if (result.get("declaration_sha256") != sha(R1_DECL) or
            result.get("control_replay_sha256") != sha(CONTROL_REPLAY) or
            replay.get("passed") is not True or replay.get("candidate_scoring") is not False):
        raise AssertionError("Round-one provenance/control replay mismatch")
    recipes = {row["recipe_id"]: row for row in result["recipes"]}
    ids = result["recipe_ids"]
    shortlist = result["selected_for_round2_seed_diagnostics"]
    expected_cells = set(declaration["upstream"]["adopted_calibrated_emissions"])
    if (len(expected_cells) != 35 or len(ids) != 36 or len(set(ids)) != len(ids) or
            set(recipes) != set(ids) or len(shortlist) > 3 or
            not set(shortlist).issubset(ids)):
        raise AssertionError("Unexpected R1 recipe or cell inventory")
    if set(result["candidate_prediction_cache"]) != expected_cells:
        raise AssertionError("R1 prediction cache inventory mismatch")

    cell_metrics = {}
    seed_metrics = {}
    control_facts = declaration["upstream"]["adopted_calibrated_emissions"]
    for key in sorted(expected_cells):
        components = study.load_frozen_emissions(key)
        fact = result["candidate_prediction_cache"][key]
        path = ROOT / fact["path"]
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise AssertionError(f"R1 prediction cache hash mismatch: {key}")
        with np.load(path, allow_pickle=False) as cache:
            expected_names = {f"pred__{recipe_id}" for recipe_id in ids} | {
                "identity__study_id", "identity__declaration_sha256", "identity__cell",
                "identity__control_sha256", "identity__ce_sha256"}
            if set(cache.files) != expected_names:
                raise AssertionError(f"R1 prediction cache inventory mismatch: {key}")
            identity = {name.removeprefix("identity__"): str(cache[name].item())
                        for name in cache.files if name.startswith("identity__")}
            if identity != {name: str(value) for name, value in fact["identity"].items()}:
                raise AssertionError(f"R1 prediction cache identity mismatch: {key}")
            if identity.get("cell") != key:
                raise AssertionError(f"R1 prediction cache cell mismatch: {key}")
            cell = {name: components[name] for name in ("labels", "period", "timestamps", "fps")}
            adopted_decoded = None
            rows = {}
            for recipe_id in ids:
                stored = _decode_saved(cache[f"pred__{recipe_id}"])
                rows[recipe_id] = study.controls.audit.score(cell, stored)
                if rows[recipe_id]["reference_spell_count"] != replay["reference_spell_counts"][key]:
                    raise AssertionError(f"R1 reference count changed: {key}/{recipe_id}")
            # Replay only the frozen result-selected shortlist through mixture and decoder.
            adopted_decoded = study.controls.audit.decode(components["adopted"],
                components["model_period"], components["fps"], study.controls.audit.CONFIG)
            for recipe_id in ids:
                if not np.array_equal(_decode_saved(cache[f"pred__{recipe_id}"]) == "D",
                                      adopted_decoded == "D"):
                    raise AssertionError(f"R1 saved prediction D mask differs from adopted fixed D: {key}/{recipe_id}")
            for recipe_id in shortlist:
                recipe = recipes[recipe_id]
                probability = _mix(components, recipe)
                prediction = study.decode_fixed_d(probability, adopted_decoded,
                    components["model_period"], components["fps"], recipe["ha_cost"])
                stored = _decode_saved(cache[f"pred__{recipe_id}"])
                if not np.array_equal(prediction, stored):
                    raise AssertionError(f"R1 selected recipe prediction replay differs: {key}/{recipe_id}")
            cell_metrics[key] = rows

        seed_fact = result["seed_prediction_cache"][key]
        seed_path = ROOT / seed_fact["path"]
        if not seed_path.is_file() or sha(seed_path) != seed_fact["sha256"]:
            raise AssertionError(f"R1 seed prediction cache hash mismatch: {key}")
        with np.load(seed_path, allow_pickle=False) as cache:
            expected_seed_names = {f"pred__{recipe_id}|{seed}"
                for recipe_id in result["selected_for_round2_seed_diagnostics"]
                for seed in study.SEEDS} | {"identity__study_id", "identity__declaration_sha256",
                "identity__cell", "identity__recipes"}
            if set(cache.files) != expected_seed_names:
                raise AssertionError(f"R1 seed cache inventory mismatch: {key}")
            seed_identity = {name.removeprefix("identity__"): ast.literal_eval(str(cache[name].item()))
                             if name == "identity__recipes" else str(cache[name].item())
                             for name in cache.files if name.startswith("identity__")}
            if seed_identity.get("cell") != key or seed_identity.get("study_id") != study.STUDY:
                raise AssertionError(f"R1 seed cache identity mismatch: {key}")
            if seed_identity.get("recipes") != shortlist:
                raise AssertionError(f"R1 seed cache shortlist identity mismatch: {key}")
            if seed_identity.get("declaration_sha256") != sha(R1_DECL):
                raise AssertionError(f"R1 seed cache declaration mismatch: {key}")
            for seed in study.SEEDS:
                seed_rows_by_cell = seed_metrics.setdefault(key, {})
                for recipe_id in result["selected_for_round2_seed_diagnostics"]:
                    recipe = recipes[recipe_id]
                    probability = _mix(components, recipe, seed=seed)
                    prediction = study.decode_fixed_d(probability, adopted_decoded,
                        components["model_period"], components["fps"], recipe["ha_cost"])
                    stored = _decode_saved(cache[f"pred__{recipe_id}|{seed}"])
                    if not np.array_equal(prediction, stored):
                        raise AssertionError(f"R1 seed prediction replay differs: {key}/{recipe_id}/{seed}")
                    cell = {name: components[name] for name in ("labels", "period", "timestamps", "fps")}
                    seed_rows_by_cell[f"{recipe_id}|{seed}"] = study.controls.audit.score(cell, prediction)
                    expected_seed_cell = result["seed_cells"][key][f"{recipe_id}|{seed}"]
                    if seed_rows_by_cell[f"{recipe_id}|{seed}"] != expected_seed_cell:
                        _close(seed_rows_by_cell[f"{recipe_id}|{seed}"],
                               expected_seed_cell,
                               f"round1/seed_cell/{key}/{recipe_id}/{seed}")

    aggregate = aggregate_cell_metrics(cell_metrics, ids)
    _close(aggregate, result["aggregate"], "round1/aggregate")
    adopted = replay["adopted"]
    expected_gates = {}
    for recipe_id in ids:
        candidate = {view: {metric: aggregate[recipe_id][view][metric]["equal_provider"]
                            for metric in study.METRICS} for view, _ in study.VIEWS}
        baseline = {view: {metric: adopted[view][metric]["equal_provider"]
                           for metric in study.METRICS} for view, _ in study.VIEWS}
        gate = study.eligibility(candidate, baseline)
        gate["d_metrics_unchanged"] = all(candidate[view][metric] == baseline[view][metric]
            for view, _ in study.VIEWS for metric in ("d_f1", "d_precision", "d_recall"))
        gate["reference_spell_counts_invariant"] = all(
            cell_metrics[key][recipe_id]["reference_spell_count"] == replay["reference_spell_counts"][key]
            for key in expected_cells)
        gate["eligible"] = gate["eligible"] and gate["d_metrics_unchanged"] and gate["reference_spell_counts_invariant"]
        expected_gates[recipe_id] = gate
    _close(expected_gates, result["eligibility"], "round1/eligibility")
    ranking = sorted((recipe_id for recipe_id in ids if expected_gates[recipe_id]["eligible"]),
        key=lambda recipe_id: (-aggregate[recipe_id]["feed_like_v2"]["spell_f1"]["equal_provider"],
                               -aggregate[recipe_id]["feed_like_v2"]["macro_f1"]["equal_provider"], recipe_id))
    if ranking != result["eligible_ranking"]:
        raise AssertionError("Round-one eligible ranking mismatch")
    if shortlist != study.select_seed_shortlist(expected_gates, aggregate):
        raise AssertionError("Round-one seed diagnostic shortlist mismatch")
    for key in sorted(expected_cells):
        for recipe_id in ids:
            if cell_metrics[key][recipe_id] != result["cell_metrics"][key][recipe_id]:
                _close(cell_metrics[key][recipe_id], result["cell_metrics"][key][recipe_id],
                       f"round1/cell/{key}/{recipe_id}")
    seed_aggregate = _seed_aggregate(seed_metrics, shortlist)
    _close(seed_aggregate, result["seed_diagnostics"], "round1/seed_diagnostics")
    return {"study_id": study.STUDY, "round": 1, "passed": True,
            "cell_count": len(cell_metrics), "recipe_count": len(ids),
            "seed_cell_count": len(seed_metrics),
            "all_candidate_scores_recomputed": True,
            "mixture_decoder_replayed_recipe_ids": shortlist,
            "round1_result_sha256": sha(R1_RESULT),
            "round1_declaration_sha256": sha(R1_DECL),
            "aggregate_recomputed": aggregate,
            "seed_aggregate_recomputed": seed_aggregate,
            "organizer_reference_access": False, "gar_valid_test_access": False}


def verify_round2() -> dict:
    declaration, result = json.loads(R2_DECL.read_text()), json.loads(R2_RESULT.read_text())
    if (result.get("study_id") != study.STUDY or result.get("round") != 2 or
            result.get("declaration_sha256") != sha(R2_DECL) or
            result.get("round1_result_sha256") != sha(R1_RESULT)):
        raise AssertionError("Round-two provenance or parent binding mismatch")
    if declaration.get("source_sha256") != round2_study._source_hashes():
        raise AssertionError("Frozen round-two source inventory changed")
    amendment = json.loads(R2_EXECUTION_AMENDMENT.read_text())
    execution_receipt = json.loads(R2_EXECUTION_RECEIPT.read_text())
    execution = __import__("experiments.v2_spell_balanced_execution", fromlist=["check"])
    if amendment != execution.check():
        raise AssertionError("Round-two execution amendment no longer matches declared source/input facts")
    if (execution_receipt.get("completed") is not True or
            execution_receipt.get("amendment_sha256") != sha(R2_EXECUTION_AMENDMENT) or
            execution_receipt.get("result_sha256") != sha(R2_RESULT) or
            execution_receipt.get("source_and_input_sha256") != execution._facts() or
            execution_receipt.get("organizer_reference_access") is not False):
        raise AssertionError("Round-two execution receipt is incomplete or does not bind saved result")
    cells = round2_study._expected_cells()
    recipe_rows = declaration["recipes"]
    recipe_ids = [row["recipe_id"] for row in recipe_rows]
    if (len(cells) != 35 or set(result["cell_metrics"]) != cells or
            set(result["prediction_cache"]) != cells or
            set(result["seed_cells"]) != cells or set(result["cohort"]["reference_spell_counts"]) != cells):
        raise AssertionError("Round-two saved cell inventory mismatch")
    if result.get("recipes") != recipe_rows or result.get("recipe_ids") != recipe_ids:
        raise AssertionError("Round-two recipe shortlist changed")
    cell_metrics, seed_rows, references = {}, {}, {}
    for key in sorted(cells):
        components = study.load_frozen_emissions(key)
        fact = result["prediction_cache"][key]
        path = ROOT / fact["path"]
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise AssertionError(f"Round-two prediction cache hash mismatch: {key}")
        with np.load(path, allow_pickle=False) as cache:
            identity = {name.removeprefix("identity__"): str(cache[name].item())
                        for name in cache.files if name.startswith("identity__")}
            if identity != {name: str(value) for name, value in fact["identity"].items()}:
                raise AssertionError(f"Round-two cache identity mismatch: {key}")
            if identity.get("cell") != key or identity.get("declaration_sha256") != sha(R2_DECL):
                raise AssertionError(f"Round-two cache identity does not bind declaration: {key}")
            phase_count = _phase_count(components["fps"])
            if int(fact["identity"].get("phase_count", -1)) != phase_count:
                raise AssertionError(f"Round-two native phase count identity mismatch: {key}")
            expected_cache_names = {"pooled_ce", *(f"seed_ce_{seed}" for seed in study.SEEDS)}
            expected_cache_names |= {f"phase_ce_seed{seed}_{phase}" for seed in study.SEEDS
                                     for phase in range(phase_count)}
            expected_cache_names |= {f"pred__{rid}" for rid in recipe_ids}
            expected_cache_names |= {f"seedpred__{rid}|{seed}" for rid in recipe_ids for seed in study.SEEDS}
            expected_cache_names |= {f"identity__{name}" for name in fact["identity"]}
            if set(cache.files) != expected_cache_names:
                raise AssertionError(f"Round-two cache inventory mismatch: {key}")
            phases = {seed: [np.asarray(cache[f"phase_ce_seed{seed}_{phase}"])
                             for phase in range(phase_count)]
                      for seed in study.SEEDS}
            # Match the native order: pair seeds within each phase, then mean phases.
            phase_means = np.stack([np.mean(np.stack([phases[seed][phase] for seed in study.SEEDS]), axis=0)
                                    for phase in range(phase_count)])
            independently_pooled = np.mean(phase_means, axis=0)
            if not np.allclose(np.asarray(cache["pooled_ce"]), independently_pooled, rtol=0, atol=ATOL):
                raise AssertionError(f"Round-two pooled CE does not replay seed/phase average: {key}")
            seed_ce = {seed: np.mean(np.stack(phases[seed]), axis=0) for seed in study.SEEDS}
            for seed in study.SEEDS:
                if not np.allclose(seed_ce[seed], cache[f"seed_ce_{seed}"], rtol=0, atol=ATOL):
                    raise AssertionError(f"Round-two seed CE phase mean mismatch: {key}/{seed}")
            adopted_decoded = study.controls.audit.decode(components["adopted"],
                components["model_period"], components["fps"], study.controls.audit.CONFIG)
            cell = {name: components[name] for name in ("labels", "period", "timestamps", "fps")}
            pooled_rows = {}
            for recipe in recipe_rows:
                rid = recipe["recipe_id"]
                mixture = _mix({"ce": cache["pooled_ce"], "adopted": components["adopted"],
                                "calibrated": components["calibrated"]}, recipe)
                prediction = study.decode_fixed_d(mixture, adopted_decoded,
                    components["model_period"], components["fps"], recipe["ha_cost"])
                stored_pooled = _decode_saved(cache[f"pred__{rid}"])
                if not np.array_equal(prediction, stored_pooled):
                    raise AssertionError(f"Round-two pooled prediction replay mismatch: {key}/{rid}")
                pooled_rows[rid] = study.controls.audit.score(cell, prediction)
                if pooled_rows[rid] != result["cell_metrics"][key][rid]:
                    _close(pooled_rows[rid], result["cell_metrics"][key][rid], f"round2/cell/{key}/{rid}")
            seed_rows[key] = {}
            for seed in study.SEEDS:
                for recipe in recipe_rows:
                    rid = recipe["recipe_id"]
                    mixture = _mix({"ce": seed_ce[seed],
                                    "adopted": components["seed"][seed]["adopted"],
                                    "calibrated": components["seed"][seed]["calibrated"]}, recipe)
                    prediction = study.decode_fixed_d(mixture, adopted_decoded,
                        components["model_period"], components["fps"], recipe["ha_cost"])
                    saved_name = f"seedpred__{rid}|{seed}"
                    if not np.array_equal(prediction, _decode_saved(cache[saved_name])):
                        raise AssertionError(f"Round-two seed prediction replay mismatch: {key}/{rid}/{seed}")
                    actual = study.controls.audit.score(cell, prediction)
                    _close(actual, result["seed_cells"][key][f"{rid}|{seed}"],
                           f"round2/seed/{key}/{rid}/{seed}")
                    seed_rows[key][f"{rid}|{seed}"] = actual
            cell_metrics[key] = pooled_rows
            references[key] = int(study.controls.audit.score(cell, adopted_decoded)["reference_spell_count"])

    aggregate = _score_rows(cell_metrics, recipe_ids)
    _close(aggregate, result["aggregate"], "round2/aggregate")
    _close(cell_metrics, result["cell_metrics"], "round2/cell_metrics")
    _close(seed_rows, result["seed_cells"], "round2/seed_cells")
    seed_aggregate = _seed_aggregate(seed_rows, recipe_ids)
    _close(seed_aggregate, result["seed_diagnostics"], "round2/seed_diagnostics")
    baseline = json.loads(CONTROL_REPLAY.read_text())
    eligibility = {}
    for rid in recipe_ids:
        candidate = {view: {metric: aggregate[rid][view][metric]["equal_provider"] for metric in study.METRICS}
                     for view, _ in study.VIEWS}
        adopted = {view: {metric: baseline["adopted"][view][metric]["equal_provider"] for metric in study.METRICS}
                   for view, _ in study.VIEWS}
        gate = study.eligibility(candidate, adopted)
        gate["d_metrics_unchanged"] = all(candidate[view][metric] == adopted[view][metric]
            for view, _ in study.VIEWS for metric in ("d_f1", "d_precision", "d_recall"))
        gate["reference_spell_counts_invariant"] = all(
            references[key] == baseline["reference_spell_counts"][key] for key in cells)
        gate["eligible"] = gate["eligible"] and gate["d_metrics_unchanged"] and gate["reference_spell_counts_invariant"]
        eligibility[rid] = gate
    _close(eligibility, result["eligibility"], "round2/eligibility")
    return {"study_id": study.STUDY, "round": 2, "passed": True,
            "cell_count": len(cells), "recipe_count": len(recipe_ids),
            "all_candidate_scores_recomputed": True,
            "phase_pooling_recomputed": True,
            "mixture_decoder_replayed_recipe_ids": recipe_ids,
            "round2_result_sha256": sha(R2_RESULT), "round2_declaration_sha256": sha(R2_DECL),
            "round2_execution_amendment_sha256": sha(R2_EXECUTION_AMENDMENT),
            "round2_execution_receipt_sha256": sha(R2_EXECUTION_RECEIPT),
            "aggregate_recomputed": aggregate, "seed_aggregate_recomputed": seed_aggregate,
            "eligibility_recomputed": eligibility,
            "organizer_reference_access": False, "gar_valid_test_access": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=("1", "2"), default="1")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/experiments/v2_spell_balanced_v1/independent_round1.json")
    args = parser.parse_args()
    receipt = verify_round1() if args.round == "1" else verify_round2()
    receipt["verifier_source_sha256"] = sha(Path(__file__).resolve())
    receipt["audit_plan_sha256"] = sha(AUDIT_PLAN)
    _write_once(args.output, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
