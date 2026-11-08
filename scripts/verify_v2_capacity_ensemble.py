"""Independently audit the fixed three-recipe v2 probability ensembles.

Uses saved native probabilities/labels and the official scalar evaluator. It
runs no model inference, fits no model, and cannot change the original study.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_capacity_ensemble_v1"
DECLARATION = ROOT / f"experiments/results/{STUDY}_declaration.json"
RESULT = ROOT / f"experiments/results/{STUDY}.json"
VERIFICATION = ROOT / f"experiments/results/{STUDY}_verification.json"
BASE_RESULT = ROOT / "experiments/results/v2_data_capacity_eval.json"
BASE_VERIFICATION = ROOT / "experiments/results/v2_data_capacity_eval_verification.json"
BASE_DECLARATION = ROOT / "experiments/results/v2_data_capacity_eval_declaration.json"
BASE_REPLAY = ROOT / "experiments/results/v2_data_capacity_eval_control_replay_v3.json"
BASE_SEAL = ROOT / "data/experiments/v2_data_capacity_eval/evaluation_seal.json"
BASE_V3_AMENDMENT = ROOT / "experiments/results/v2_data_capacity_eval_execution_v3_amendment.json"
BASE_V2_RECOVERY_AMENDMENT = ROOT / "experiments/results/v2_data_capacity_eval_recovery_v2_amendment.json"
METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
           "predicted_spell_count", "reference_spell_count",
           "short_spell_recall_iou_0_5_lt3s")
STRICT_METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                  "short_spell_recall_iou_0_5_lt3s")
VIEWS = ("clean", "feed_like_v1", "feed_like_v2")
PR82_V2_SPELL = 0.5927491913246999
ARMS = ("uniform_standalone_ha3", "uniform_pr82_blend_ha1_5",
        "uniform_adopted_blend_ha3", "uniform_adopted_dead_new_ha3")
PROBABILITY_KEYS = {
    "uniform_standalone_ha3": "standalone_ha3",
    "uniform_pr82_blend_ha1_5": "pr82_blend_ha1_5",
    "uniform_adopted_blend_ha3": "adopted_blend_ha3",
    "uniform_adopted_dead_new_ha3": "adopted_dead_new_ha3",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def equal_provider(values: dict[tuple[str, str, int], float]) -> tuple[float, dict[str, float]]:
    """Canonical replica→match→provider means, matching the frozen policy."""
    grouped: dict[str, dict[str, list[float]]] = {}
    for (provider, match, _rep), value in sorted(values.items()):
        grouped.setdefault(provider, {}).setdefault(match, []).append(float(value))
    if not grouped:
        raise ValueError("Cannot aggregate an empty cohort")
    providers = {provider: float(np.mean([np.mean(matches[match]) for match in sorted(matches)]))
                 for provider, matches in sorted(grouped.items())}
    return float(np.mean([providers[p] for p in sorted(providers)])), providers


def uniform_probability(probabilities: list[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(p, dtype=np.float64) for p in probabilities]
    if len(arrays) != 3 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Expected three aligned recipe probability arrays")
    if (arrays[0].ndim != 2 or arrays[0].shape[1] != 3 or
            any(not np.isfinite(a).all() or (a < 0).any() for a in arrays) or
            any(not np.allclose(a.sum(axis=1), 1., atol=1e-5, rtol=0) for a in arrays)):
        raise ValueError("Expected finite normalized H/A/D probabilities")
    return (arrays[0] + arrays[1] + arrays[2]) / 3.0


def verify_dead_mask(candidate_mask, adopted_mask) -> None:
    if not np.array_equal(np.asarray(candidate_mask, dtype=bool),
                          np.asarray(adopted_mask, dtype=bool)):
        raise ValueError("Uniform protected ensemble changed the adopted D mask")


def evaluate_gates(candidate: dict, adopted: dict) -> dict:
    checks = {}
    for view in VIEWS:
        for metric in STRICT_METRICS:
            actual, minimum = float(candidate[view][metric]), float(adopted[view][metric])
            checks[f"{view}/{metric}_vs_adopted"] = {
                "actual": actual, "minimum": minimum, "passed": actual >= minimum}
    actual = float(candidate["feed_like_v2"]["spell_f1"])
    checks["feed_like_v2/spell_f1_vs_pr82"] = {
        "actual": actual, "minimum": PR82_V2_SPELL, "passed": actual >= PR82_V2_SPELL}
    return {"checks": checks, "all_passed": all(c["passed"] for c in checks.values())}


def _base_modules():
    from experiments import v2_capacity_ensemble as study
    from experiments import v2_data_capacity_eval as base
    from experiments import v2_data_capacity_eval_execution_v3 as execution
    return study, base, execution


def _verify_prior_chain(declaration: dict, base_result: dict, base_verification: dict,
                        base_replay: dict) -> dict:
    study, base, execution = _base_modules()
    provenance = declaration["provenance"]
    expected_bindings = {
        "base_result": (BASE_RESULT, sha(BASE_RESULT)),
        "base_seal": (BASE_SEAL, sha(BASE_SEAL)),
        "core_control_replay": (BASE_REPLAY, sha(BASE_REPLAY)),
        "v3_execution_amendment": (BASE_V3_AMENDMENT, sha(BASE_V3_AMENDMENT)),
        "v2_recovery_amendment": (BASE_V2_RECOVERY_AMENDMENT, sha(BASE_V2_RECOVERY_AMENDMENT)),
        "base_verification": (BASE_VERIFICATION, sha(BASE_VERIFICATION)),
    }
    for name, (path, digest) in expected_bindings.items():
        fact = provenance[name]
        if fact != {"path": str(path.relative_to(ROOT)), "sha256": digest}:
            raise ValueError(f"Prior study binding mismatch: {name}")
    if provenance["source_sha256"] != study._source_hashes():
        raise ValueError("Ensemble source closure changed")
    if base_result.get("organizer_reference_access") is not False or base_result.get("promotion") is not False:
        raise ValueError("Base study provenance flags changed")
    # Preserve the first stage's negative decision exactly: 12 candidate pipelines, zero strict passes.
    gates = base_verification.get("gates_recomputed_with_zero_tolerance", {})
    if len(gates) != 12 or any(row.get("all_passed") for row in gates.values()):
        raise ValueError("Original capacity study strict-gate outcome is not the preserved 0/12 negative")
    if base_verification.get("result_sha256") != sha(BASE_RESULT):
        raise ValueError("Base study independent verification no longer binds its result")
    if (base_verification.get("study_id") != base.STUDY or
            base_verification.get("official_scalar_cells_replayed") != 105 or
            base_verification.get("inference_repeated") is not False or
            base_verification.get("recovery_v3_amendment_verified") is not True):
        raise ValueError("Base study lacks its required independent 105-cell native audit")
    return {"recipes": list(base.RECIPES), "base": base, "study": study,
            "execution": execution, "base_result": base_result, "base_replay": base_replay}


def _load_npz(fact: dict) -> dict:
    path = ROOT / fact["path"]
    if not path.is_file() or sha(path) != fact["sha256"]:
        raise ValueError(f"Sealed ensemble input missing or changed: {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _score_native(base, arm: str, probability: np.ndarray, arrays: dict) -> tuple[dict, np.ndarray]:
    labels, period = arrays["labels"].astype(str), arrays["period"]
    model_period = arrays["model_period"]
    fps = float(arrays["fps"].item())
    if arm == "uniform_pr82_blend_ha1_5":
        prediction = base.audit.decode(probability, model_period, fps,
                                       base.DECODERS["pr82_blend_ha1_5"])
    elif arm == "uniform_adopted_dead_new_ha3":
        prediction = base._adopted_dead_prediction(
            probability, arrays["prob__cand"], model_period, fps,
            base.DECODERS["adopted_dead_new_ha3"])
        adopted_prediction = base.audit.decode(arrays["prob__cand"], model_period, fps, base.audit.CONFIG)
        verify_dead_mask(prediction == "D", adopted_prediction == "D")
    else:
        decoder = "adopted_blend_ha3" if arm == "uniform_adopted_blend_ha3" else "standalone_ha3"
        prediction = base.audit.decode(probability, model_period, fps, base.DECODERS[decoder])
    cell = {"labels": labels, "period": period,
            "timestamps": arrays["timestamps"], "fps": fps}
    return base.audit.score(cell, prediction), prediction


def _validate_saved_counts(cell_arm: dict, replayed_metrics: dict, frame_count: int,
                          baseline_reference_count: int) -> None:
    """Check the runner's declared count fields against replay and native grid."""
    if int(cell_arm["frame_count"]) != int(frame_count):
        raise ValueError("Saved frame count differs from native array length")
    if (int(cell_arm["reference_spell_count"]) != int(baseline_reference_count) or
            int(replayed_metrics["reference_spell_count"]) != int(baseline_reference_count)):
        raise ValueError("Reference spell count changed")
    if int(cell_arm["metrics"]["predicted_spell_count"]) != int(
            replayed_metrics["predicted_spell_count"]):
        raise ValueError("Saved predicted spell count differs from scalar replay")


def _adopted_baseline(control_replay: dict) -> dict:
    """Read the adopted candidate baseline from the sealed control replay."""
    return {view: {metric: control_replay["aggregate"][view]["cand"][metric]["equal_provider"]
                   for metric in METRICS} for view in VIEWS}


def verify(declaration: dict, result: dict, base_result: dict, base_verification: dict) -> dict:
    study_id = declaration.get("study_id")
    if study_id != STUDY or result.get("study_id") != STUDY:
        raise ValueError("Wrong ensemble study identity")
    if result.get("declaration_sha256") != sha(DECLARATION):
        raise ValueError("Ensemble result does not bind its frozen declaration")
    if declaration.get("status") != "frozen_uniform_ensemble_before_scoring":
        raise ValueError("Ensemble declaration is not the frozen pre-score declaration")
    if result.get("arms") != list(ARMS) or declaration.get("arms") != list(ARMS):
        raise ValueError("Ensemble arm inventory differs from four fixed methods")
    if (result.get("no_new_inference_or_fits") is not True or
            result.get("organizer_reference_access") is not False or
            result.get("public_or_GAR_scoring") is not False or result.get("promotion") is not False):
        raise ValueError("Ensemble result provenance flags are invalid")
    base_replay = json.loads(BASE_REPLAY.read_text())
    chain = _verify_prior_chain(declaration, base_result, base_verification, base_replay)
    base = chain["base"]
    recipes = chain["recipes"]
    expected_cells = {f"{p}|{m}|{v}|{r}" for p, m, v, r in base._views()}
    if set(result.get("cells", {})) != expected_cells:
        raise ValueError("Ensemble result cell inventory differs from the frozen 35 cells")
    budget = declaration.get("budget", {})
    if (budget.get("arms") != 4 or budget.get("new_fits") != 0 or
            budget.get("new_inference") != 0 or budget.get("new_seeds") != 0 or
            budget.get("cells_per_arm") != 35 or budget.get("candidate_npz_count") != 105 or
            budget.get("control_npz_count") != 35):
        raise ValueError("Ensemble declaration budget differs from the fixed follow-up")
    if declaration.get("cohort", {}).get("cells") != sorted(expected_cells):
        raise ValueError("Ensemble declaration cohort differs from the fixed 35 cells")
    facts = declaration["provenance"]
    if result.get("provenance") != facts:
        raise ValueError("Ensemble result provenance differs from its declaration")
    if facts["candidate_emissions"] != base_result["candidate_emissions"]:
        raise ValueError("Declaration changed the sealed base candidate emission inventory")
    if facts["control_emissions"] != base_replay["emission_cache"]:
        raise ValueError("Declaration changed the sealed base control emission inventory")
    if set(facts["control_emissions"]) != expected_cells:
        raise ValueError("Declaration control emission inventory mismatch")
    candidate_facts = facts["candidate_emissions"]
    if set(candidate_facts) != {f"{recipe}/{key}" for recipe in recipes for key in expected_cells}:
        raise ValueError("Declaration base candidate emission inventory mismatch")
    recomputed_values = {view: {arm: {metric: {} for metric in METRICS} for arm in ARMS}
                         for view in VIEWS}
    replayed = 0
    for key in sorted(expected_cells):
        provider, match, view, rep = key.split("|")
        rep = int(rep)
        row = result["cells"][key]
        control_fact = facts["control_emissions"][key]
        control_arrays = _load_npz(control_fact)
        if (row.get("provider") != provider or row.get("match") != match or
                row.get("view") != view or int(row.get("replicate", -1)) != rep):
            raise ValueError(f"Ensemble cell identity mismatch: {key}")
        for arm in ARMS:
            cell_arm = row["arms"][arm]
            fact = row["emission"]
            identity = fact.get("identity", {})
            if identity != {"study_id": STUDY,
                            "declaration_sha256": sha(DECLARATION),
                            "cell_key": key}:
                raise ValueError(f"Ensemble emission identity mismatch: {key}")
            arrays = _load_npz(fact)
            native_identity = {field.removeprefix("identity__"): str(arrays[field].item())
                               for field in arrays if field.startswith("identity__")}
            if native_identity != identity:
                raise ValueError(f"Ensemble native identity mismatch: {key}")
            expected_array_names = {"labels", "timestamps", "period", "model_period", "fps",
                                    *(f"prob__{name}" for name in ARMS),
                                    "identity__study_id", "identity__declaration_sha256",
                                    "identity__cell_key"}
            if set(arrays) != expected_array_names:
                raise ValueError(f"Ensemble native array inventory mismatch: {key}")
            for field in ("labels", "timestamps", "period", "model_period", "fps"):
                if not np.array_equal(arrays[field], control_arrays[field]):
                    raise ValueError(f"Ensemble native grid differs from controls: {key}/{arm}/{field}")
            recipe_arrays = []
            for recipe in recipes:
                source_fact = candidate_facts[f"{recipe}/{key}"]
                source = _load_npz(source_fact)
                for field in ("labels", "timestamps", "period", "model_period", "fps"):
                    if not np.array_equal(source[field], control_arrays[field]):
                        raise ValueError(f"Base recipe cell grid differs from controls: {recipe}/{key}/{field}")
                recipe_arrays.append(source[f"prob__{PROBABILITY_KEYS[arm]}"])
            expected_probability = uniform_probability(recipe_arrays)
            if not np.array_equal(arrays[f"prob__{arm}"], expected_probability):
                raise ValueError(f"Saved probability is not the fixed uniform mean: {key}/{arm}")
            metrics, _prediction = _score_native(base, arm, expected_probability, control_arrays)
            baseline_count = base_result["matches"][key]["control"]["metrics"]["cand"][
                "reference_spell_count"]
            _validate_saved_counts(cell_arm, metrics, len(control_arrays["labels"]), baseline_count)
            saved_metrics = cell_arm["metrics"]
            for metric in METRICS:
                actual, saved = metrics[metric], saved_metrics[metric]
                if abs(float(actual) - float(saved)) > 1e-12:
                    raise ValueError(f"Official scalar replay mismatch {key}/{arm}/{metric}")
                if metric == "reference_spell_count":
                    if actual != baseline_count:
                        raise ValueError(f"Reference spell count changed: {key}/{arm}")
                recomputed_values[view][arm][metric][(provider, match, rep)] = float(actual)
            replayed += 1
    aggregates = {view: {} for view in VIEWS}
    for view in VIEWS:
        for arm in ARMS:
            aggregates[view][arm] = {}
            for metric in METRICS:
                mean, providers = equal_provider(recomputed_values[view][arm][metric])
                actual = result["aggregate"][view][arm][metric]
                if abs(float(actual["equal_provider"]) - mean) > 1e-12:
                    raise ValueError(f"Ensemble aggregate mismatch: {view}/{arm}/{metric}")
                if set(actual["providers"]) != set(providers) or any(
                        abs(float(actual["providers"][p]) - value) > 1e-12 for p, value in providers.items()):
                    raise ValueError(f"Ensemble provider aggregate mismatch: {view}/{arm}/{metric}")
                aggregates[view][arm][metric] = {"equal_provider": mean, "providers": providers}
    adopted = _adopted_baseline(base_replay)
    gates = {}
    for arm in ARMS:
        candidate = {view: {metric: aggregates[view][arm][metric]["equal_provider"]
                            for metric in METRICS} for view in VIEWS}
        gates[arm] = evaluate_gates(candidate, adopted)
        if len(gates[arm]["checks"]) != 19:
            raise AssertionError("Strict gate inventory must contain the frozen 19 checks")
        if result["strict_success_gates"][arm] != gates[arm]:
            raise ValueError(f"Ensemble strict gate mismatch: {arm}")
    member_diag = result.get("component_member_diagnostics", {}).get("values")
    if member_diag != base_result.get("component_seed_diagnostics"):
        raise ValueError("Per-seed component diagnostics must be preserved unchanged and separately")
    return {"study_id": STUDY, "result_sha256": sha(RESULT),
            "declaration_sha256": sha(DECLARATION),
            "prior_base_result_sha256": sha(BASE_RESULT),
            "prior_base_verification_sha256": sha(BASE_VERIFICATION),
            "native_ensemble_cells_replayed": replayed,
            "uniform_probabilities_recomputed": True,
            "strict_gates_recomputed": gates,
            "original_stage_preserved_as_0_of_12_strict_passes": True,
            "per_seed_values_not_ensembled_or_claimed": True,
            "inference_repeated": False, "organizer_or_GAR_scoring": False}


def main() -> dict:
    declaration = json.loads(DECLARATION.read_text())
    result = json.loads(RESULT.read_text())
    base_result = json.loads(BASE_RESULT.read_text())
    base_verification = json.loads(BASE_VERIFICATION.read_text())
    receipt = verify(declaration, result, base_result, base_verification)
    with VERIFICATION.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
