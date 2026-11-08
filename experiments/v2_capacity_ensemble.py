"""Uniform three-recipe probability ensembles on the frozen v2 capacity cells.

This bounded follow-up reads only previously sealed per-recipe probability
arrays. It performs no model inference, fitting, or weight search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import v2_data_capacity_eval as base
from experiments import v2_data_capacity_eval_execution_v3 as execution


ROOT = base.ROOT
STUDY = "v2_capacity_ensemble_v1"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
OUT = ROOT / "data/experiments" / STUDY
VIEWS = base.VIEWS
RECIPES = base.RECIPES
METRICS = base.METRICS
ARMS = ("uniform_standalone_ha3", "uniform_pr82_blend_ha1_5",
        "uniform_adopted_blend_ha3", "uniform_adopted_dead_new_ha3")
PROBABILITY_KEYS = {
    "uniform_standalone_ha3": "standalone_ha3",
    "uniform_pr82_blend_ha1_5": "pr82_blend_ha1_5",
    "uniform_adopted_blend_ha3": "adopted_blend_ha3",
    "uniform_adopted_dead_new_ha3": "adopted_dead_new_ha3",
}
CORE_REPLAY = execution.REPLAY
V3_AMENDMENT = execution.AMENDMENT
V2_RECOVERY_AMENDMENT = ROOT / "experiments/results/v2_data_capacity_eval_recovery_v2_amendment.json"
BASE_VERIFICATION = ROOT / "experiments/results/v2_data_capacity_eval_verification.json"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def uniform_probability(probabilities):
    """Equal-weight mean of aligned categorical probability arrays."""
    arrays = [np.asarray(value, dtype=np.float64) for value in probabilities]
    if len(arrays) != 3 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Expected three aligned recipe probability arrays")
    if (arrays[0].ndim != 2 or arrays[0].shape[1] != 3 or
            any(not np.isfinite(a).all() or (a < 0).any() for a in arrays) or
            any(not np.allclose(a.sum(axis=1), 1., atol=1e-5, rtol=0) for a in arrays)):
        raise ValueError("Expected finite normalized H/A/D probability arrays")
    result = (arrays[0] + arrays[1] + arrays[2]) / 3.0
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-5, rtol=0):
        raise RuntimeError("Uniform ensemble is not normalized")
    return result


def _write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def _input_bindings():
    if not base.RESULT.is_file() or not base.SEAL.is_file() or not CORE_REPLAY.is_file():
        raise FileNotFoundError("Completed base result, model seal, and v3 control replay are required")
    if (not V3_AMENDMENT.is_file() or not V2_RECOVERY_AMENDMENT.is_file()
            or not BASE_VERIFICATION.is_file()):
        raise FileNotFoundError("Execution/recovery amendments and independent base audit are required")
    result = json.loads(base.RESULT.read_text())
    seal = json.loads(base.SEAL.read_text())
    replay = json.loads(CORE_REPLAY.read_text())
    if (result.get("evaluation_seal_sha256") != sha(base.SEAL) or
            result.get("control_replay_sha256") != sha(CORE_REPLAY) or
            seal.get("control_replay_sha256") != sha(CORE_REPLAY) or
            replay.get("passed") is not True):
        raise RuntimeError("Base result, seal, or control replay does not form a verified chain")
    verification = json.loads(BASE_VERIFICATION.read_text())
    if (verification.get("study_id") != base.STUDY or
            verification.get("result_sha256") != sha(base.RESULT) or
            verification.get("official_scalar_cells_replayed") != 105 or
            verification.get("inference_repeated") is not False or
            verification.get("recovery_v3_amendment_verified") is not True):
        raise RuntimeError("Independent verification of the complete base evaluation is required")
    expected_cells = {f"{p}|{m}|{v}|{r}" for p, m, v, r in base._views()}
    if set(replay.get("emission_cache", {})) != expected_cells:
        raise RuntimeError("Base control inventory is not the fixed 35-cell cohort")
    if set(result.get("candidate_emissions", {})) != {
            f"{recipe}/{key}" for recipe in RECIPES for key in expected_cells}:
        raise RuntimeError("Base candidate emission inventory is not the fixed 105 arrays")
    for fact in replay["emission_cache"].values():
        if sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Changed control emission {fact['path']}")
    for fact in result["candidate_emissions"].values():
        if sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Changed candidate emission {fact['path']}")
    return result, seal, replay


def _source_hashes():
    paths = (Path(__file__), Path(base.__file__), Path(execution.__file__),
             Path(base.audit.__file__), Path(base.audit.decoder.__file__),
             Path(ROOT / "evaluation/possession_metrics.py"))
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def freeze():
    if DECLARATION.exists() or RESULT.exists():
        raise RuntimeError("Follow-up declaration/result already exists")
    base_result, seal, replay = _input_bindings()
    candidate_facts = base_result["candidate_emissions"]
    control_facts = replay["emission_cache"]
    declaration = {
        "study_id": STUDY,
        "status": "frozen_uniform_ensemble_before_scoring",
        "question": "Does uniform probability averaging across the three fixed new capacity/data recipes improve the exposed development result without strict metric regressions?",
        "arms": list(ARMS),
        "methods": {
            ARMS[0]: {"probabilities": [f"{r}/standalone_ha3" for r in RECIPES],
                      "decoder": "f062 H/A cost 3"},
            ARMS[1]: {"probabilities": [f"{r}/pr82_blend_ha1_5" for r in RECIPES],
                      "decoder": "selected dt4/dead3/HA1.5/HT1"},
            ARMS[2]: {"probabilities": [f"{r}/adopted_blend_ha3" for r in RECIPES],
                      "decoder": "f062 H/A cost 3"},
            ARMS[3]: {"probabilities": [f"{r}/adopted_dead_new_ha3" for r in RECIPES],
                      "decoder": "fixed adopted original_full D mask; conditional HA3 inside predicted-live spans"},
        },
        "cohort": {"cells": sorted(control_facts), "views": [v for v, _ in VIEWS],
                   "aggregation": "replicates within match, matches within provider, providers equally with canonical key ordering",
                   "exposure": "same five exposed original-provider development matches; this is further development, not fresh confirmation"},
        "gates": {"strict": "clean/v1/v2 spell, macro, D-F1, D-precision, D-recall and short-spell recall >= adopted controls; v2 spell additionally >= 0.5927491913246999; zero tolerance",
                  "metrics": list(METRICS), "pr82_v2_spell_f1": base.PR82_V2_FLOOR},
        "budget": {"arms": 4, "new_fits": 0, "new_inference": 0, "new_seeds": 0,
                   "new_decoder_settings": 0, "cells_per_arm": 35,
                   "candidate_npz_count": 105, "control_npz_count": 35},
        "provenance": {
            "base_result": {"path": str(base.RESULT.relative_to(ROOT)), "sha256": sha(base.RESULT)},
            "base_seal": {"path": str(base.SEAL.relative_to(ROOT)), "sha256": sha(base.SEAL)},
            "core_control_replay": {"path": str(CORE_REPLAY.relative_to(ROOT)), "sha256": sha(CORE_REPLAY)},
            "v3_execution_amendment": {"path": str(V3_AMENDMENT.relative_to(ROOT)), "sha256": sha(V3_AMENDMENT)},
            "v2_recovery_amendment": {"path": str(V2_RECOVERY_AMENDMENT.relative_to(ROOT)), "sha256": sha(V2_RECOVERY_AMENDMENT)},
            "base_verification": {"path": str(BASE_VERIFICATION.relative_to(ROOT)), "sha256": sha(BASE_VERIFICATION)},
            "candidate_emissions": candidate_facts, "control_emissions": control_facts,
            "source_sha256": _source_hashes(),
        },
        "organizer_reference_access": False, "public_or_GAR_scoring": False,
        "promotion": False,
    }
    _write_once(DECLARATION, declaration)
    return declaration


def _validate_declaration():
    if not DECLARATION.is_file():
        raise RuntimeError("Freeze the uniform ensemble declaration before scoring")
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study_id") != STUDY or declaration.get("arms") != list(ARMS):
        raise RuntimeError("Uniform ensemble declaration identity changed")
    if declaration["provenance"]["source_sha256"] != _source_hashes():
        raise RuntimeError("Uniform ensemble source changed")
    _input_bindings()
    for binding in ("base_result", "base_seal", "core_control_replay",
                    "v3_execution_amendment", "v2_recovery_amendment", "base_verification"):
        fact = declaration["provenance"][binding]
        if sha(ROOT / fact["path"]) != fact["sha256"]:
            raise RuntimeError(f"Prior evidence changed: {binding}")
    return declaration


def _npz_arrays(path, expected_fact):
    if sha(ROOT / expected_fact["path"]) != expected_fact["sha256"]:
        raise RuntimeError(f"Sealed source NPZ changed: {expected_fact['path']}")
    with np.load(ROOT / expected_fact["path"], allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _score_cell(key, arm, declaration):
    replay = json.loads(CORE_REPLAY.read_text())
    control_fact = replay["emission_cache"][key]
    control = _npz_arrays(control_fact, control_fact)
    recipe_probs = []
    for recipe in RECIPES:
        fact = declaration["provenance"]["candidate_emissions"][f"{recipe}/{key}"]
        arrays = _npz_arrays(fact, fact)
        recipe_probs.append(arrays[f"prob__{PROBABILITY_KEYS[arm]}"])
    mean = uniform_probability(recipe_probs)
    labels, timestamps, period, model_period, fps = (control["labels"], control["timestamps"],
                                                       control["period"], control["model_period"],
                                                       float(control["fps"].item()))
    view = {"period": period, "timestamps": timestamps, "fps": fps}
    if arm == "uniform_pr82_blend_ha1_5":
        config = base.DECODERS["pr82_blend_ha1_5"]
        prediction = base.audit.decode(mean, model_period, fps, config)
    elif arm == "uniform_adopted_dead_new_ha3":
        adopted = control["prob__cand"]
        prediction = base._adopted_dead_prediction(
            mean, adopted, model_period, fps, base.DECODERS["adopted_dead_new_ha3"])
        adopted_prediction = base.audit.decode(adopted, model_period, fps, base.audit.CONFIG)
        if not np.array_equal(prediction == "D", adopted_prediction == "D"):
            raise RuntimeError("Uniform adopted-dead ensemble changed the frozen D mask")
    else:
        decoder = "adopted_blend_ha3" if arm == "uniform_adopted_blend_ha3" else "standalone_ha3"
        prediction = base.audit.decode(mean, model_period, fps, base.DECODERS[decoder])
    metrics = base.audit.score({"labels": labels, "period": period,
                                "timestamps": timestamps, "fps": fps}, prediction)
    expected_references = replay["matches"][key]["metrics"]["cand"]["reference_spell_count"]
    if (int(metrics["reference_spell_count"]) != int(expected_references) or
            len(labels) != len(timestamps) or len(labels) != len(period) or
            len(labels) != len(model_period)):
        raise RuntimeError(f"Reference/frame-count invariant failed for {key}/{arm}")
    return {"metrics": {k: float(v) if isinstance(v, np.generic) else v
                        for k, v in metrics.items()},
            "frame_count": int(len(labels)),
            "reference_spell_count": int(metrics["reference_spell_count"]),
            "probability": mean, "labels": labels, "timestamps": timestamps,
            "period": period, "model_period": model_period, "fps": fps}


def run():
    declaration = _validate_declaration()
    if RESULT.exists():
        raise RuntimeError("Ensemble result already exists")
    cells = {}
    for provider, match, view, rep in base._views():
        key = f"{provider}|{match}|{view}|{rep}"
        cells[key] = {"provider": provider, "match": match, "view": view,
                      "replicate": rep, "arms": {}}
        cell_values = {}
        for arm in ARMS:
            row = _score_cell(key, arm, declaration)
            cells[key]["arms"][arm] = {"metrics": row["metrics"],
                                         "frame_count": row["frame_count"],
                                         "reference_spell_count": row["reference_spell_count"]}
            cell_values[arm] = row
        first = cell_values[ARMS[0]]
        cache_path = OUT / "emissions" / f"{key.replace('|', '__')}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        identity = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION), "cell_key": key}
        np.savez_compressed(cache_path, labels=first["labels"], timestamps=first["timestamps"],
                            period=first["period"], model_period=first["model_period"],
                            fps=np.asarray(first["fps"]),
                            **{f"prob__{arm}": cell_values[arm]["probability"] for arm in ARMS},
                            **{f"identity__{k}": np.asarray(str(v)) for k, v in identity.items()})
        cells[key]["emission"] = {"path": str(cache_path.relative_to(ROOT)),
                                  "sha256": sha(cache_path), "identity": identity}
    aggregate = {}
    for view, reps in VIEWS:
        aggregate[view] = {}
        for arm in ARMS:
            aggregate[view][arm] = {}
            for metric in METRICS:
                values = {}
                for key, row in cells.items():
                    if row["view"] == view and row["replicate"] in reps:
                        values[(row["provider"], row["match"], row["replicate"])] = row["arms"][arm]["metrics"][metric]
                mean, providers = base.equal_provider(dict(sorted(values.items())))
                aggregate[view][arm][metric] = {"equal_provider": mean, "providers": providers}
    replay = json.loads(CORE_REPLAY.read_text())
    adopted = {view: {metric: replay["aggregate"][view]["cand"][metric]["equal_provider"]
                      for metric in METRICS} for view, _ in VIEWS}
    gates = {}
    for arm in ARMS:
        candidate = {view: {metric: aggregate[view][arm][metric]["equal_provider"]
                            for metric in METRICS} for view, _ in VIEWS}
        gates[arm] = base.hard_gate(candidate, adopted, base.PR82_V2_FLOOR)
    seed_diagnostics = json.loads(base.RESULT.read_text())["component_seed_diagnostics"]
    out = {"study_id": STUDY, "declaration_sha256": sha(DECLARATION),
           "arms": list(ARMS), "cells": cells, "aggregate": aggregate,
           "strict_success_gates": gates,
           "component_member_diagnostics": {"source": "reused prior per-seed standalone metrics; recipes remain separate members",
                                             "values": seed_diagnostics},
           "provenance": declaration["provenance"],
           "no_new_inference_or_fits": True, "organizer_reference_access": False,
           "public_or_GAR_scoring": False, "promotion": False}
    _write_once(RESULT, out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "run": run}[action]()


if __name__ == "__main__":
    main()
