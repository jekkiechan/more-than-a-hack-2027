"""Independent saved-output audit for the v2 iterative possession study.

This verifier reads frozen emissions, control caches, fit seals and results. It
does not fit models, generate corruption views, access public labels, or score
GAR validation/test data. Metrics are recomputed with the official challenge
decoder/evaluator from saved probability or prediction arrays.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import feed_benchmark_audit_v1 as official
from experiments import v2_data_capacity_eval as capacity
from experiments import v2_iterative_eval as round1_eval
from experiments import v2_iterative_round3 as round3_eval
from experiments import v2_iterative_train_run as train_run

ROOT = Path(__file__).resolve().parents[1]
ROUND2_RESULT = ROOT / "experiments/results/v2_iterative_v1_round2.json"
ROUND2_DECL = ROOT / "experiments/results/v2_iterative_v1_round2_declaration.json"
ROUND3_RESULT = ROOT / "experiments/results/v2_iterative_v1_round3.json"
ROUND3_DECL = ROOT / "experiments/results/v2_iterative_v1_round3_declaration.json"
TRAIN_DECL = ROOT / "experiments/results/v2_iterative_train_v1_declaration.json"
TRAIN_SEAL = ROOT / "experiments/results/v2_iterative_train_v1_seal.json"
VIEWS = capacity.VIEWS
METRICS = capacity.METRICS
QUALITY = round1_eval.QUALITY_METRICS
ATOL = 1e-12


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _assert_close(actual, expected, label: str, *, atol: float = ATOL) -> None:
    if isinstance(actual, dict) or isinstance(expected, dict):
        if not isinstance(actual, dict) or not isinstance(expected, dict) or set(actual) != set(expected):
            raise AssertionError(f"{label}: nested metric keys differ")
        for name in actual:
            _assert_close(actual[name], expected[name], f"{label}/{name}", atol=atol)
        return
    if isinstance(actual, (int, np.integer)) or isinstance(expected, (int, np.integer)):
        if actual != expected:
            raise AssertionError(f"{label}: {actual!r} != {expected!r}")
        return
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=atol):
        raise AssertionError(f"{label}: {actual!r} != {expected!r} (atol={atol})")


def _key_parts(key: str):
    provider, match, view, rep = key.split("|")
    return provider, match, view, int(rep)


def _expected_keys():
    return {round1_eval.cell_key(provider, match, view, rep)
            for provider, match, _ in round1_eval.runtime.frozen.audit.MATCHES
            for view, reps in VIEWS for rep in reps}


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {name: saved[name].copy() for name in saved.files}


def _check_probabilities(probability: np.ndarray, n: int, label: str) -> None:
    if probability.shape != (n, 3):
        raise AssertionError(f"{label}: expected {(n, 3)}, got {probability.shape}")
    if not np.isfinite(probability).all() or (probability < 0).any():
        raise AssertionError(f"{label}: invalid probability values")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise AssertionError(f"{label}: probability rows do not sum to one")


def _normalize(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or (p < 0).any():
        raise AssertionError("Expected a finite, nonnegative N by 3 probability matrix")
    mass = p.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise AssertionError("Probability rows must have positive mass")
    return p / mass


def _conditional(probability: np.ndarray) -> np.ndarray:
    p = _normalize(probability)
    mass = p[:, :2].sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise AssertionError("H/A probability mass must be positive")
    return p[:, :2] / mass


def _normalize_ha(probability: np.ndarray) -> np.ndarray:
    p = np.asarray(probability, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise AssertionError("Expected a finite, nonnegative N by 2 H/A matrix")
    mass = p.sum(axis=1, keepdims=True)
    if (mass <= 0).any():
        raise AssertionError("H/A rows must have positive mass")
    return p / mass


def _round1_transforms(adopted: np.ndarray, calibrated: np.ndarray) -> dict[str, np.ndarray]:
    a, c = _normalize(adopted), _normalize(calibrated)
    joint = _normalize((a + c) * 0.5)
    cond = _normalize_ha((_conditional(a) + _conditional(c)) * 0.5)
    cond3 = np.column_stack((cond, np.zeros(len(cond))))
    logit = 0.5 * np.log(np.clip(_conditional(a), 1e-12, 1.0)) + \
        0.5 * np.log(np.clip(_conditional(c), 1e-12, 1.0))
    logit -= logit.max(axis=1, keepdims=True)
    logit = _normalize(np.column_stack((np.exp(logit), np.zeros(len(logit)))))
    return {
        "joint_probability_blend_ha3": joint,
        "fixed_d_conditional_ha3": cond3,
        "fixed_d_conditional_ha1_5": cond3,
        "fixed_d_logit_ha3": logit,
    }


def _score_fixed_d(probability, adopted, period, fps, cost):
    return capacity._adopted_dead_prediction(probability, adopted, period, fps,
                                               capacity._decoder(cost))


def _score_cell(control: dict, prediction: np.ndarray) -> dict:
    cell = {"labels": control["labels"], "period": control["period"],
            "timestamps": control["timestamps"], "fps": control["fps"]}
    return official.score(cell, prediction)


def _decode_fixed_metrics(control, probability, adopted, cost):
    prediction = _score_fixed_d(probability, adopted, control["model_period"],
                                control["fps"], cost)
    return _score_cell(control, prediction)


def _read_control(fact: dict, key: str, declaration_sha: str) -> dict:
    path = ROOT / fact["path"]
    if not path.is_file() or sha(path) != fact["sha256"]:
        raise AssertionError(f"Control emission hash mismatch: {path}")
    saved = _load_npz(path)
    identity = {k.removeprefix("identity__"): str(v.item())
                for k, v in saved.items() if k.startswith("identity__")}
    expected = {k: str(v) for k, v in fact["identity"].items()}
    if identity != expected or identity.get("cell") != key or identity.get("declaration_sha256") != declaration_sha:
        raise AssertionError(f"Control identity mismatch for {key}")
    control = {k.removeprefix("prob__"): v for k, v in saved.items()
               if k.startswith("prob__")}
    control.update({k: saved[k] for k in ("labels", "period", "timestamps", "model_period")})
    control["fps"] = float(saved["fps"])
    n = len(control["labels"])
    for name in ("period", "timestamps", "model_period"):
        if len(control[name]) != n:
            raise AssertionError(f"Control {name} length mismatch: {key}")
    for name in ("adopted", "calibrated", "adop_seed2026", "adop_seed2027",
                 "cand_seed2026", "cand_seed2027"):
        _check_probabilities(control[name], n, f"{key}/{name}")
    for prefix in ("adop", "cand"):
        pooled = np.mean([control[f"{prefix}_seed{seed}"] for seed in (2026, 2027)], axis=0)
        if not np.allclose(pooled, control["calibrated" if prefix == "adop" else "adopted"],
                           rtol=0.0, atol=1e-7):
            raise AssertionError(f"Seed pool mismatch for {key}/{prefix}")
    return control


def _mean(values):
    grouped = {}
    for (provider, match, rep), value in sorted(values.items()):
        grouped.setdefault(provider, {}).setdefault(match, []).append(float(value))
    if set(grouped) != {p for p, _, _ in round1_eval.runtime.frozen.audit.MATCHES}:
        raise AssertionError("Aggregate is missing a provider")
    provider_means = {p: float(np.mean([np.mean(reps) for reps in matches.values()]))
                      for p, matches in grouped.items()}
    return float(np.mean(list(provider_means.values()))), provider_means


def _aggregate(cell_metrics: dict[str, dict[str, dict]]) -> dict:
    out = {}
    for view, reps in VIEWS:
        out[view] = {}
        for metric in METRICS:
            values = {}
            for key, arms in cell_metrics.items():
                provider, match, cell_view, rep = _key_parts(key)
                if cell_view == view and rep in reps:
                    for arm, metrics in arms.items():
                        values.setdefault(arm, {})[(provider, match, rep)] = metrics[metric]
            for arm, bycell in values.items():
                mean, providers = _mean(bycell)
                out[view].setdefault(arm, {})[metric] = {"equal_provider": mean,
                                                         "providers": providers}
    # Return result[arm][metric][view] to match the frozen result document.
    arms = {arm for view in out.values() for arm in view for metric in view[arm]}
    return {arm: {metric: {view: out[view][arm][metric] for view, _ in VIEWS}
                  for metric in METRICS} for arm in arms}


def _compare_metrics(actual: dict, expected: dict, label: str) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"{label}: metric keys differ")
    for name in actual:
        _assert_close(actual[name], expected[name], f"{label}/{name}")


def _check_counts(keys: set[str], expected_keys: set[str]) -> None:
    if keys != expected_keys:
        raise AssertionError(f"Development cell inventory mismatch: missing={sorted(expected_keys-keys)}, extra={sorted(keys-expected_keys)}")
    expected = {f"{view}/{rep}": 5 for view, reps in VIEWS for rep in reps}
    actual = {f"{view}/{rep}": sum(_key_parts(k)[2:] == (view, rep) for k in keys)
              for view, reps in VIEWS for rep in reps}
    if actual != expected:
        raise AssertionError(f"Cell count mismatch: {actual}")


def _verify_hash_map(entries: dict[str, str], label: str) -> None:
    for raw_path, expected in entries.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected:
            raise AssertionError(f"{label} hash mismatch: {path}")


def _round2_seed_key(cell: str, train_arm: str, seed_label: str) -> str:
    return f"{cell}|{train_arm}|{seed_label}"


def _expected_round2_seed_keys(cells: set[str]) -> set[str]:
    labels = [label for seed in train_run.trainer.SEEDS
              for label in (str(seed), f"{seed}_blend")]
    return {_round2_seed_key(cell, arm, label)
            for cell in cells for arm in train_run.trainer.ARMS for label in labels}


def _fit_receipt_matches(seal_row: dict, sidecar: dict) -> bool:
    """The fit seal embeds each original sidecar receipt, sans a second hash."""
    return seal_row == sidecar


def _check_reference_count(control_count: int, candidate_metrics: dict, label: str) -> None:
    if int(candidate_metrics["reference_spell_count"]) != int(control_count):
        raise AssertionError(f"Reference spell count changed: {label}")


def _seed_metrics_by_cell(cell: str, train_arm: str, seed_label: str,
                          metrics: dict, target: dict[str, dict]) -> None:
    target.setdefault(cell, {})[f"{train_arm}|{seed_label}"] = metrics


def _round2_seed_aggregate(seed_cell_metrics: dict[str, dict[str, dict]]) -> dict:
    result = {}
    for train_arm in train_run.trainer.ARMS:
        result[train_arm] = {"fixed_d": {}, "conditional_blend": {}}
        for seed in train_run.trainer.SEEDS:
            for variant, label in (("fixed_d", str(seed)), ("conditional_blend", f"{seed}_blend")):
                rows = {}
                for cell, arms in seed_cell_metrics.items():
                    seed_key = f"{train_arm}|{label}"
                    if seed_key not in arms:
                        continue
                    provider, match, view, rep = _key_parts(cell)
                    for metric in QUALITY:
                        rows.setdefault((view, metric), {})[(provider, match, rep)] = arms[seed_key][metric]
                result[train_arm][variant][str(seed)] = {}
                for view, _ in VIEWS:
                    result[train_arm][variant][str(seed)][view] = {}
                    for metric in QUALITY:
                        values = {k: v for k, v in rows.get((view, metric), {}).items()
                                  if k[2] in dict(VIEWS)[view]}
                        mean, providers = _mean(values)
                        result[train_arm][variant][str(seed)][view][metric] = {
                            "equal_provider": mean, "providers": providers, "cells": len(values)}
    return result


def verify_round1() -> dict:
    declaration = json.loads(round1_eval.DECLARATION.read_text())
    replay = json.loads(round1_eval.REPLAY.read_text())
    round_decl = json.loads(round1_eval.ROUND1_DECLARATION.read_text())
    result = json.loads(round1_eval.ROUND1_RESULT.read_text())
    decl_sha, replay_sha, round_decl_sha = sha(round1_eval.DECLARATION), sha(round1_eval.REPLAY), sha(round1_eval.ROUND1_DECLARATION)
    if (replay.get("declaration_sha256") != decl_sha or not replay.get("passed") or
            replay.get("controls_only") is not True or replay.get("candidate_scoring") is not False):
        raise AssertionError("Frozen control replay is not bound or passing")
    if (round_decl.get("parent_declaration_sha256") != decl_sha or
            round_decl.get("control_replay_sha256") != replay_sha or
            round_decl.get("arms") != list(round1_eval.ARMS) or
            round_decl.get("candidate_scoring_started") is not False):
        raise AssertionError("Round-one declaration does not bind its frozen parents")
    if (result.get("study_id") != round1_eval.STUDY or result.get("round") != 1 or
            result.get("declaration_sha256") != decl_sha or
            result.get("round1_declaration_sha256") != round_decl_sha or
            result.get("control_replay_sha256") != replay_sha):
        raise AssertionError("Round-one result provenance hashes mismatch")
    expected_arms = set(round1_eval.ARMS) | {"adopted_original_full", "calibrated"}
    if result.get("arms") != list(round1_eval.ARMS):
        raise AssertionError("Round-one declared arm inventory mismatch")
    if set(result.get("matches", {})) != set(replay.get("emission_cache", {})):
        raise AssertionError("Round-one saved match inventory mismatch")
    if set(result.get("aggregate", {})) != expected_arms:
        raise AssertionError("Round-one aggregate arm inventory mismatch")
    if set(result.get("seed_diagnostics", {})) != set(round1_eval.ARMS):
        raise AssertionError("Round-one seed diagnostic arm inventory mismatch")
    if result.get("organizer_reference_access") is not False or result.get("public_or_GAR_scoring") is not False or result.get("promotion") is not False:
        raise AssertionError("Round-one result contains an invalid scoring-access or promotion declaration")
    if declaration.get("cohort", {}).get("cell_count") != len(_expected_keys()):
        raise AssertionError("Frozen cohort declaration has an unexpected cell count")
    _verify_hash_map(declaration["provenance"]["source_sha256"], "round-one source")
    input_facts = declaration["provenance"]["archived_inputs"]
    for key, fact in input_facts.items():
        path = Path(fact["path"])
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise AssertionError(f"Round-one archived input hash mismatch: {key}")
    for key, fact in declaration["provenance"]["control_models_and_teachers"].items():
        path = Path(fact["path"])
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise AssertionError(f"Round-one control model hash mismatch: {key}")
    keys = set(replay.get("emission_cache", {}))
    _check_counts(keys, _expected_keys())
    reconstructed = {}
    seed_rows = {}
    reference_by_view_rep = {}
    controls = {}
    for key, fact in replay["emission_cache"].items():
        control = _read_control(fact, key, decl_sha)
        controls[key] = control
        transforms = _round1_transforms(control["adopted"], control["calibrated"])
        decoded = {}
        for arm, probability in transforms.items():
            if "fixed_d_" in arm:
                cost = 1.5 if "1_5" in arm else 3.0
                prediction = _score_fixed_d(probability, control["adopted"], control["model_period"], control["fps"], cost)
            else:
                prediction = official.decode(probability, control["model_period"], control["fps"], official.CONFIG)
            decoded[arm] = prediction
        decoded["adopted_original_full"] = official.decode(control["adopted"], control["model_period"], control["fps"], official.CONFIG)
        decoded["calibrated"] = official.decode(control["calibrated"], control["model_period"], control["fps"], official.CONFIG)
        metrics = {arm: _score_cell(control, pred) for arm, pred in decoded.items()}
        expected_row = result["matches"][key]
        if set(expected_row) != set(metrics):
            raise AssertionError(f"Round-one saved arm inventory mismatch for {key}")
        for arm, row_metrics in metrics.items():
            _compare_metrics(row_metrics, expected_row[arm], f"round1/{key}/{arm}")
        ref = int(metrics["adopted_original_full"]["reference_spell_count"])
        if any(int(metrics[a]["reference_spell_count"]) != ref for a in transforms):
            raise AssertionError(f"Round-one reference spell count changed within {key}")
        provider, match, view, rep = _key_parts(key)
        reference_by_view_rep.setdefault(f"{view}/{rep}", 0)
        reference_by_view_rep[f"{view}/{rep}"] += ref
        reconstructed[key] = metrics
        for seed in (2026, 2027):
            seed_adopted = control[f"cand_seed{seed}"]
            seed_calibrated = control[f"adop_seed{seed}"]
            seed_probs = _round1_transforms(seed_adopted, seed_calibrated)
            seed_metrics = {}
            for arm, probability in seed_probs.items():
                if "fixed_d_" in arm:
                    cost = 1.5 if "1_5" in arm else 3.0
                    pred = _score_fixed_d(probability, seed_adopted, control["model_period"], control["fps"], cost)
                else:
                    pred = official.decode(probability, control["model_period"], control["fps"], official.CONFIG)
                seed_metrics[arm] = _score_cell(control, pred)
            seed_rows[f"{key}|{seed}"] = seed_metrics
    expected_counts = replay.get("reference_spell_counts", {})
    if expected_counts != reference_by_view_rep:
        raise AssertionError("Round-one reference spell totals differ from the saved control receipt")
    if result.get("reference_spell_counts") != reference_by_view_rep or result.get("adopted_reference_spell_counts_invariant") is not True:
        raise AssertionError("Round-one saved reference spell receipt differs from recomputed controls")
    aggregate = _aggregate(reconstructed)
    for arm, metrics in aggregate.items():
        saved = result["aggregate"][arm]
        for metric in METRICS:
            for view, _ in VIEWS:
                _assert_close(metrics[metric][view]["equal_provider"],
                              saved[metric][view]["equal_provider"],
                              f"round1/aggregate/{arm}/{metric}/{view}")
                for provider, value in metrics[metric][view]["providers"].items():
                    _assert_close(value, saved[metric][view]["providers"][provider],
                                  f"round1/providers/{arm}/{metric}/{view}/{provider}")
    seed_aggregate = {}
    for arm in round1_eval.ARMS:
        seed_aggregate[arm] = {}
        for seed in (2026, 2027):
            seed_aggregate[arm][str(seed)] = {}
            for view, reps in VIEWS:
                seed_aggregate[arm][str(seed)][view] = {}
                for metric in QUALITY:
                    values = {}
                    for tagged_key, rows in seed_rows.items():
                        cell_key, seed_text = tagged_key.rsplit("|", 1)
                        provider, match, cell_view, rep = _key_parts(cell_key)
                        if int(seed_text) == seed and cell_view == view and rep in reps:
                            values[(provider, match, rep)] = rows[arm][metric]
                    mean, providers = _mean(values)
                    seed_aggregate[arm][str(seed)][view][metric] = {
                        "equal_provider": mean, "providers": providers, "cells": len(values)}
    for arm in round1_eval.ARMS:
        if seed_aggregate[arm] != result["seed_diagnostics"][arm]:
            raise AssertionError(f"Round-one seed aggregate mismatch {arm}")
    baseline = replay["adopted_aggregate"]
    gates = {}
    for arm in round1_eval.ARMS:
        checks = {}
        for view, _ in VIEWS:
            for metric in QUALITY:
                actual = aggregate[arm][metric][view]["equal_provider"]
                floor = baseline[view][metric]["equal_provider"]
                checks[f"{view}/{metric}_vs_adopted"] = {"actual": actual, "minimum": floor,
                                                            "passed": actual >= floor}
        value = aggregate[arm]["spell_f1"]["feed_like_v2"]["equal_provider"]
        checks["feed_like_v2/spell_f1_strictly_above_0_60"] = {
            "actual": value, "minimum_exclusive": .60, "passed": value > .60}
        for view, _ in VIEWS:
            actual_count = aggregate[arm]["reference_spell_count"][view]["equal_provider"]
            expected_count = baseline[view]["reference_spell_count"]["equal_provider"]
            checks[f"{view}/reference_spell_count_invariant"] = {
                "actual": actual_count, "expected": expected_count,
                "passed": actual_count == expected_count}
        gates[arm] = {"checks": checks, "all_passed": all(c["passed"] for c in checks.values())}
        if gates[arm] != result["strict_success_gates"][arm]:
            raise AssertionError(f"Round-one strict gate mismatch: {arm}")
    receipt = {"study_id": "v2_iterative_v1", "round": 1, "passed": True,
               "control_replay_sha256": replay_sha, "round1_result_sha256": sha(round1_eval.ROUND1_RESULT),
               "cell_count": len(reconstructed), "reference_spell_counts": reference_by_view_rep,
               "official_rescore": True, "strict_gates_recomputed": gates,
               "aggregate_recomputed": aggregate,
               "seed_aggregate_recomputed": seed_aggregate}
    return receipt


def _verify_training_seal() -> dict:
    d = json.loads(TRAIN_DECL.read_text())
    seal = json.loads(TRAIN_SEAL.read_text())
    if seal.get("study_id") != train_run.STUDY or seal.get("declaration_sha256") != sha(TRAIN_DECL) or not seal.get("all_12_complete"):
        raise AssertionError("Training seal identity/inventory is invalid")
    _verify_hash_map(d["source_sha256"], "training source")
    _verify_hash_map(d["input_sha256"], "training input")
    fit_rows = {(row["arm"], row["fold"], int(row["seed"])): row for row in seal["fits"]}
    if len(fit_rows) != 12:
        raise AssertionError("Training seal does not contain 12 unique fits")
    schedules = {}
    d_state_checks = {}
    for arm in train_run.trainer.ARMS:
        for fold in train_run.trainer.FOLDS:
            for seed in train_run.trainer.SEEDS:
                key = (arm, fold, int(seed))
                row = fit_rows[key]
                model_path = train_run.model_path(arm, fold, seed)
                receipt_path = Path(str(model_path) + ".json")
                if sha(model_path) != row["sha256"]:
                    raise AssertionError(f"Training artifact hash mismatch {key}")
                receipt = json.loads(receipt_path.read_text())
                if not _fit_receipt_matches(row, receipt):
                    raise AssertionError(f"Fit sidecar differs from the seal row {key}")
                if receipt.get("sha256") != sha(model_path) or receipt.get("declaration_sha256") != sha(TRAIN_DECL):
                    raise AssertionError(f"Training fit receipt mismatch {key}")
                prior_path = train_run.prior_path(fold, seed)
                if sha(prior_path) != receipt.get("warmstart_sha256"):
                    raise AssertionError(f"Warm-start hash mismatch {key}")
                model, prior = joblib.load(model_path), joblib.load(prior_path)
                if set(model.get("d_state", {})) != set(prior.get("d_state", {})):
                    raise AssertionError(f"D-state keys changed {key}")
                for name, value in prior["d_state"].items():
                    if not np.array_equal(value, model["d_state"][name]):
                        raise AssertionError(f"D-state changed {key}/{name}")
                if row.get("unchanged_d_state") is not True or receipt.get("unchanged_d_state") is not True:
                    raise AssertionError(f"D-state preservation not recorded {key}")
                d_state_checks[f"{arm}/{fold}/{seed}"] = sha(model_path)
                pair_key = (fold, int(seed))
                signatures = (receipt["sampling"], receipt["view_draws"], receipt["exposures"],
                              receipt["old_matches"], receipt["gar_games"])
                if pair_key in schedules and schedules[pair_key] != signatures:
                    raise AssertionError(f"Paired arm sampling differs: {pair_key}")
                schedules[pair_key] = signatures
    return {"fit_count": len(fit_rows), "d_state_invariance": d_state_checks,
            "paired_sampling_equal": True, "declaration_sha256": sha(TRAIN_DECL),
            "seal_sha256": sha(TRAIN_SEAL)}


def verify_round2() -> dict:
    if not ROUND2_DECL.is_file() or not ROUND2_RESULT.is_file():
        raise FileNotFoundError("Round-two declaration/result not available yet")
    declaration = json.loads(ROUND2_DECL.read_text())
    result = json.loads(ROUND2_RESULT.read_text())
    if declaration.get("study_id") != "v2_iterative_v1_round2" or result.get("declaration_sha256") != sha(ROUND2_DECL):
        raise AssertionError("Round-two result/declaration identity mismatch")
    _verify_hash_map(declaration["source_sha256"], "round-two source")
    parent_result_sha = sha(round1_eval.ROUND1_RESULT)
    if declaration["parent"]["round1_result_sha256"] != parent_result_sha or result.get("round1_result_sha256") != parent_result_sha:
        raise AssertionError("Round-two result does not bind round one")
    train_receipt = _verify_training_seal()
    controls = declaration["controls"]
    for key, fact in controls.items():
        for path_key, hash_key in (("control_path", "control_sha256"),
                                   ("prepared_path", "prepared_sha256")):
            path = ROOT / fact[path_key]
            if not path.is_file() or sha(path) != fact[hash_key]:
                raise AssertionError(f"Round-two frozen control hash mismatch: {key}/{path_key}")
    keys = {key for key, fact in controls.items()}
    _check_counts(keys, _expected_keys())
    emissions = result.get("candidate_emissions", {})
    expected_emission_keys = {f"{arm}/{key}" for arm in train_run.trainer.ARMS for key in keys}
    if set(emissions) != expected_emission_keys:
        raise AssertionError("Round-two candidate emission inventory mismatch")
    if set(result.get("seed_metrics", {})) != _expected_round2_seed_keys(keys):
        raise AssertionError("Round-two seed metric inventory mismatch")
    cell_metrics = {key: {} for key in keys}
    seed_cell_metrics = {}
    ref_counts = {}
    for inventory_key, fact in emissions.items():
        train_arm, key = inventory_key.split("/", 1)
        path = ROOT / fact["path"]
        if not path.is_file() or sha(path) != fact["sha256"]:
            raise AssertionError(f"Round-two emission hash mismatch {path}")
        saved = _load_npz(path)
        identity = {k.removeprefix("identity__"): str(v.item()) for k, v in saved.items()
                    if k.startswith("identity__")}
        if identity != {k: str(v) for k, v in fact["identity"].items()}:
            raise AssertionError(f"Round-two emission identity mismatch {inventory_key}")
        control = _read_control({"path": declaration["controls"][key]["control_path"],
                                 "sha256": declaration["controls"][key]["control_sha256"],
                                 "identity": declaration["controls"][key]["identity"]}, key,
                                sha(round1_eval.DECLARATION))
        n = len(control["labels"])
        pooled = saved["prob__pooled"]
        seeds = {seed: saved[f"prob__seed{seed}"] for seed in train_run.trainer.SEEDS}
        for seed, prob in seeds.items(): _check_probabilities(prob, n, f"{inventory_key}/seed{seed}")
        _check_probabilities(pooled, n, f"{inventory_key}/pooled")
        if not np.allclose(pooled, np.mean(list(seeds.values()), axis=0), rtol=0.0, atol=1e-7):
            raise AssertionError(f"Seed pooling mismatch {inventory_key}")
        cost = 3.0
        predictions = {
            f"{train_arm}_fixed_d_ha3": _score_fixed_d(pooled, control["adopted"], control["model_period"], control["fps"], cost),
            f"{train_arm}_conditional_blend_fixed_d_ha3": _score_fixed_d(
                _normalize_ha((_conditional(pooled) + _conditional(control["adopted"])) * .5),
                control["adopted"], control["model_period"], control["fps"], cost),
        }
        for arm, pred in predictions.items():
            if f"pred__{arm}" not in saved or not np.array_equal(pred, saved[f"pred__{arm}"]):
                raise AssertionError(f"Saved pooled prediction does not replay {inventory_key}/{arm}")
            metrics = _score_cell(control, pred)
            _compare_metrics(metrics, result["matches"][key][arm], f"round2/{key}/{arm}")
            cell_metrics[key][arm] = metrics
        reference = int(_score_cell(control, official.decode(control["adopted"], control["model_period"], control["fps"], official.CONFIG))["reference_spell_count"])
        for arm, pred in predictions.items():
            _check_reference_count(reference, _score_cell(control, pred), f"{inventory_key}/{arm}")
        provider, match, view, rep = _key_parts(key)
        if train_arm == train_run.trainer.ARMS[0]:
            ref_counts.setdefault(f"{view}/{rep}", 0)
            ref_counts[f"{view}/{rep}"] += reference
        for seed in train_run.trainer.SEEDS:
            for suffix, probability in ((str(seed), seeds[seed]), (str(seed)+"_blend",
                    _normalize_ha((_conditional(seeds[seed]) + _conditional(control["adopted"])) * .5))):
                arm = f"{train_arm}_fixed_d_ha3" if suffix.isnumeric() else f"{train_arm}_conditional_blend_fixed_d_ha3"
                pred = _score_fixed_d(probability, control["adopted"], control["model_period"], control["fps"], 3.0)
                stored_name = f"pred__seed__{seed}" if suffix.isnumeric() else f"pred__seed__{seed}_blend"
                if stored_name not in saved or not np.array_equal(pred, saved[stored_name]):
                    raise AssertionError(f"Saved seed prediction does not replay {inventory_key}/{suffix}")
                metrics = _score_cell(control, pred)
                seed_label = str(seed) if suffix.isnumeric() else f"{seed}_blend"
                saved_seed = result["seed_metrics"][_round2_seed_key(key, train_arm, seed_label)]
                _compare_metrics(metrics, saved_seed, f"round2/seed/{key}/{seed}/{suffix}")
                _seed_metrics_by_cell(key, train_arm, seed_label, metrics, seed_cell_metrics)
    if set(result["matches"]) != keys:
        raise AssertionError("Round-two cell result inventory mismatch")
    if ref_counts != json.loads(round1_eval.REPLAY.read_text())["reference_spell_counts"]:
        raise AssertionError("Round-two reference spell totals differ from frozen controls")
    aggregate = _aggregate(cell_metrics)
    for arm in result["aggregate"]:
        for metric in METRICS:
            for view, _ in VIEWS:
                expected = result["aggregate"][arm][view][metric]["equal_provider"]
                _assert_close(aggregate[arm][metric][view]["equal_provider"], expected,
                              f"round2/aggregate/{arm}/{metric}/{view}")
                for provider, value in aggregate[arm][metric][view]["providers"].items():
                    _assert_close(value, result["aggregate"][arm][view][metric]["providers"][provider],
                                  f"round2/provider/{arm}/{metric}/{view}/{provider}")
    adopted = json.loads(round1_eval.REPLAY.read_text())["adopted_aggregate"]
    gates = {}
    for arm in result["aggregate"]:
        checks = {}
        for view, _ in VIEWS:
            for metric in QUALITY:
                actual = aggregate[arm][metric][view]["equal_provider"]
                minimum = adopted[view][metric]["equal_provider"]
                checks[f"{view}/{metric}_vs_adopted"] = {"actual": actual, "minimum": minimum,
                                                            "passed": actual >= minimum}
        v2 = aggregate[arm]["spell_f1"]["feed_like_v2"]["equal_provider"]
        checks["feed_like_v2/spell_f1_strictly_above_0_60"] = {
            "actual": v2, "minimum_exclusive": .60, "passed": v2 > .60}
        for view, _ in VIEWS:
            actual_count = aggregate[arm]["reference_spell_count"][view]["equal_provider"]
            expected_count = adopted[view]["reference_spell_count"]["equal_provider"]
            checks[f"{view}/reference_spell_count_invariant"] = {
                "actual": actual_count, "expected": expected_count,
                "passed": actual_count == expected_count}
        gates[arm] = {"checks": checks, "all_passed": all(x["passed"] for x in checks.values())}
        if gates[arm] != result["strict_success_gates"][arm]:
            raise AssertionError(f"Round-two hard gate mismatch {arm}")
    seed_aggregate = _round2_seed_aggregate(seed_cell_metrics)
    if seed_aggregate != result["seed_aggregate"]:
        raise AssertionError("Round-two saved seed aggregate differs from the independent rescore")
    return {"study_id": "v2_iterative_v1_round2", "round": 2, "passed": True,
            "round2_result_sha256": sha(ROUND2_RESULT), "round2_declaration_sha256": sha(ROUND2_DECL),
            "training_audit": train_receipt, "cell_count": len(keys),
            "reference_spell_counts": ref_counts, "aggregate_recomputed": aggregate,
            "strict_gates_recomputed": gates,
            "seed_aggregate_recomputed": seed_aggregate,
            "seed_cell_count": len(result["seed_metrics"])}


def _round3_combinations(trained: np.ndarray, calibrated: np.ndarray,
                        adopted: np.ndarray) -> dict[str, np.ndarray]:
    """Rebuild round-three conditional H/A experts without calling its runner."""
    experts = [_conditional(x) for x in (trained, calibrated, adopted)]
    if any(x.shape != experts[0].shape for x in experts):
        raise AssertionError("Round-three experts are not aligned")
    result = {}
    for arm in round3_eval.ARMS:
        members = experts if arm.startswith("three_") else experts[:2]
        if arm.endswith("_log"):
            logits = np.mean([np.log(np.clip(x, 1e-12, 1.0)) for x in members], axis=0)
            logits -= logits.max(axis=1, keepdims=True)
            ha = np.exp(logits)
        else:
            ha = np.mean(members, axis=0)
        ha /= ha.sum(axis=1, keepdims=True)
        result[arm] = np.column_stack((ha, np.zeros(len(ha))))
    return result


def _round3_aggregate(cell_metrics: dict[str, dict[str, dict]]) -> dict:
    out = {}
    for arm in round3_eval.ARMS:
        out[arm] = {}
        for view, reps in VIEWS:
            out[arm][view] = {}
            for metric in METRICS:
                values = {}
                for key, arms in cell_metrics.items():
                    provider, match, cell_view, rep = _key_parts(key)
                    if cell_view == view and rep in reps:
                        values[(provider, match, rep)] = arms[arm][metric]
                mean, providers = _mean(values)
                out[arm][view][metric] = {"equal_provider": mean, "providers": providers}
    return out


def _round3_seed_aggregate(seed_metrics: dict[str, dict]) -> dict:
    result = {}
    for arm in round3_eval.ARMS:
        result[arm] = {}
        for seed in train_run.trainer.SEEDS:
            result[arm][str(seed)] = {}
            for view, reps in VIEWS:
                result[arm][str(seed)][view] = {}
                for metric in QUALITY:
                    values = {}
                    for tagged_key, arms in seed_metrics.items():
                        cell, seed_text = tagged_key.rsplit("|", 1)
                        provider, match, cell_view, rep = _key_parts(cell)
                        if int(seed_text) == int(seed) and cell_view == view and rep in reps:
                            values[(provider, match, rep)] = arms[arm][metric]
                    mean, providers = _mean(values)
                    result[arm][str(seed)][view][metric] = {
                        "equal_provider": mean, "providers": providers}
    return result


def _round3_compare_aggregate(actual: dict, expected: dict, label: str,
                              metrics=METRICS) -> None:
    if set(actual) != set(expected):
        raise AssertionError(f"{label}: arm/seed inventory mismatch")
    for arm in actual:
        if set(actual[arm]) != set(expected[arm]):
            raise AssertionError(f"{label}/{arm}: view inventory mismatch")
        for view in actual[arm]:
            if set(actual[arm][view]) != set(expected[arm][view]):
                raise AssertionError(f"{label}/{arm}/{view}: metric inventory mismatch")
            for metric in metrics:
                left, right = actual[arm][view][metric], expected[arm][view][metric]
                _assert_close(left["equal_provider"], right["equal_provider"],
                              f"{label}/{arm}/{view}/{metric}")
                if set(left["providers"]) != set(right["providers"]):
                    raise AssertionError(f"{label}/{arm}/{view}/{metric}: provider inventory mismatch")
                for provider, value in left["providers"].items():
                    _assert_close(value, right["providers"][provider],
                                  f"{label}/{arm}/{view}/{metric}/{provider}")


def _read_round3_prediction_artifact(fact: dict, expected_names: set[str],
                                     label: str) -> dict[str, np.ndarray]:
    path = ROOT / fact["path"]
    if not path.is_file() or sha(path) != fact["sha256"]:
        raise AssertionError(f"Round-three prediction artifact hash mismatch: {label}")
    saved = _load_npz(path)
    if set(saved) != expected_names:
        raise AssertionError(f"Round-three prediction inventory mismatch: {label}")
    return saved


def _round3_gates(aggregate: dict, adopted: dict) -> dict:
    gates = {}
    for arm in round3_eval.ARMS:
        checks = {}
        for view, _ in VIEWS:
            for metric in QUALITY:
                actual = aggregate[arm][view][metric]["equal_provider"]
                minimum = adopted[view][metric]
                checks[f"{view}/{metric}_vs_adopted"] = {
                    "actual": actual, "minimum": minimum, "passed": actual >= minimum}
        actual = aggregate[arm]["feed_like_v2"]["spell_f1"]["equal_provider"]
        checks["feed_like_v2/spell_f1_strictly_above_0_60"] = {
            "actual": actual, "minimum_exclusive": .60, "passed": actual > .60}
        for view, _ in VIEWS:
            actual = aggregate[arm][view]["reference_spell_count"]["equal_provider"]
            expected = adopted[view]["reference_spell_count"]
            checks[f"{view}/reference_spell_count_invariant"] = {
                "actual": actual, "expected": expected, "passed": actual == expected}
        gates[arm] = {"checks": checks, "all_passed": all(row["passed"] for row in checks.values())}
    return gates


def verify_round3() -> dict:
    if not ROUND3_DECL.is_file() or not ROUND3_RESULT.is_file():
        raise FileNotFoundError("Round-three declaration/result not available yet")
    declaration = json.loads(ROUND3_DECL.read_text())
    result = json.loads(ROUND3_RESULT.read_text())
    if (declaration.get("study_id") != "v2_iterative_v1_round3" or
            declaration.get("round") != 3 or
            result.get("study_id") != declaration["study_id"] or
            result.get("declaration_sha256") != sha(ROUND3_DECL)):
        raise AssertionError("Round-three result/declaration identity mismatch")
    _verify_hash_map(declaration["source_sha256"], "round-three source")
    r2_path = ROUND2_RESULT
    replay_path = round1_eval.REPLAY
    if (sha(r2_path) != declaration["previous_result_sha256"] or
            sha(replay_path) != declaration["control_replay_sha256"]):
        raise AssertionError("Round-three parent evidence hash mismatch")
    r2 = json.loads(r2_path.read_text())
    replay = json.loads(replay_path.read_text())
    if not replay.get("passed"):
        raise AssertionError("Round-three controls are not bound to a passing replay")

    expected_selected = max(train_run.trainer.ARMS, key=lambda arm: (
        r2["aggregate"][f"{arm}_fixed_d_ha3"]["feed_like_v2"]["spell_f1"]["equal_provider"],
        r2["aggregate"][f"{arm}_fixed_d_ha3"]["feed_like_v2"]["macro_f1"]["equal_provider"]))
    selected = declaration.get("selected_training_arm")
    if selected != expected_selected or result.get("selected_training_arm") != selected:
        raise AssertionError("Round-three selected training arm does not match the frozen round-two rule")
    if declaration.get("selected_arm_retention") != r2["strict_success_gates"][f"{selected}_fixed_d_ha3"]:
        raise AssertionError("Round-three selected-arm retention receipt differs from round two")
    for payload, label in ((declaration, "declaration"), (result, "result")):
        if payload.get("organizer_reference_access") is not False or payload.get("public_or_GAR_scoring") is not False:
            raise AssertionError(f"Round-three {label} contains an invalid scoring-access declaration")
        if payload.get("promotion") is not False:
            raise AssertionError(f"Round-three {label} unexpectedly records promotion")
    if declaration.get("arms") != list(round3_eval.ARMS):
        raise AssertionError("Round-three arm definition changed")
    if declaration.get("controls") != replay.get("emission_cache"):
        raise AssertionError("Round-three controls differ from the frozen replay emissions")
    selected_emissions = {key: r2["candidate_emissions"][f"{selected}/{key}"]
                          for key in replay["emission_cache"]}
    if declaration.get("candidate_emissions") != selected_emissions:
        raise AssertionError("Round-three selected emissions differ from round two")

    keys = set(replay["emission_cache"])
    _check_counts(keys, _expected_keys())
    if set(declaration["candidate_emissions"]) != keys or set(declaration["controls"]) != keys:
        raise AssertionError("Round-three declaration cell inventory mismatch")
    if set(result.get("matches", {})) != keys or set(result.get("predictions", {})) != keys:
        raise AssertionError("Round-three result cell or prediction inventory mismatch")
    prediction_paths = [fact["path"] for fact in result["predictions"].values()]
    if len(set(prediction_paths)) != len(prediction_paths):
        raise AssertionError("Round-three prediction artifacts do not have unique paths")
    expected_seed_keys = {f"{key}|{seed}" for key in keys for seed in train_run.trainer.SEEDS}
    if set(result.get("seed_metrics", {})) != expected_seed_keys:
        raise AssertionError("Round-three seed metric inventory mismatch")

    cell_metrics, seed_metrics = {}, {}
    reference_counts = {}
    for key in sorted(keys):
        control = _read_control(declaration["controls"][key], key,
                                sha(round1_eval.DECLARATION))
        emission_fact = declaration["candidate_emissions"][key]
        emission_path = ROOT / emission_fact["path"]
        if not emission_path.is_file() or sha(emission_path) != emission_fact["sha256"]:
            raise AssertionError(f"Round-three selected emission hash mismatch: {key}")
        with np.load(emission_path, allow_pickle=False) as source:
            probabilities = {name: source[name].copy() for name in
                             ("prob__pooled", "prob__seed2026", "prob__seed2027")}
        n = len(control["labels"])
        for name, probability in probabilities.items():
            _check_probabilities(probability, n, f"round3/{key}/{name}")
        groups = {"ensemble": _round3_combinations(probabilities["prob__pooled"],
                                                        control["calibrated"], control["adopted"])}
        for seed in train_run.trainer.SEEDS:
            groups[str(seed)] = _round3_combinations(probabilities[f"prob__seed{seed}"],
                control[f"adop_seed{seed}"], control[f"cand_seed{seed}"])

        cell = {name: control[name] for name in ("labels", "period", "timestamps", "fps")}
        adopted_pred = official.decode(control["adopted"], control["model_period"],
                                       control["fps"], official.CONFIG)
        reference = _score_cell(control, adopted_pred)["reference_spell_count"]
        expected_prediction_names = {f"pred__{group}__{arm}" for group, group_probs in groups.items()
                                     for arm in group_probs}
        saved_predictions = _read_round3_prediction_artifact(
            result["predictions"][key], expected_prediction_names, key)

        pooled_metrics = {}
        for group, group_probs in groups.items():
            group_metrics = {}
            for arm, probability in group_probs.items():
                prediction = _score_fixed_d(probability, control["adopted"],
                    control["model_period"], control["fps"], 3.0)
                stored = saved_predictions[f"pred__{group}__{arm}"]
                if not np.array_equal(prediction, stored):
                    raise AssertionError(f"Round-three saved prediction does not replay: {key}/{group}/{arm}")
                metrics = _score_cell(control, prediction)
                _check_reference_count(reference, metrics, f"{key}/{group}/{arm}")
                _compare_metrics(metrics,
                    (result["matches"][key] if group == "ensemble" else
                     result["seed_metrics"][f"{key}|{group}"])[arm],
                    f"round3/{key}/{group}/{arm}")
                group_metrics[arm] = metrics
            if group == "ensemble":
                pooled_metrics = group_metrics
            else:
                seed_metrics[f"{key}|{group}"] = group_metrics
        cell_metrics[key] = pooled_metrics
        provider, match, view, rep = _key_parts(key)
        reference_counts.setdefault(f"{view}/{rep}", 0)
        reference_counts[f"{view}/{rep}"] += int(reference)

    if reference_counts != replay.get("reference_spell_counts"):
        raise AssertionError("Round-three reference spell totals differ from frozen controls")
    aggregate = _round3_aggregate(cell_metrics)
    _round3_compare_aggregate(aggregate, result["aggregate"], "round3/aggregate")

    adopted = {view: {metric: replay["adopted_aggregate"][view][metric]["equal_provider"]
                      for metric in METRICS} for view, _ in VIEWS}
    gates = _round3_gates(aggregate, adopted)
    if gates != result.get("strict_success_gates"):
        raise AssertionError("Round-three hard gates differ from the independent recomputation")

    seed_aggregate = _round3_seed_aggregate(seed_metrics)
    _round3_compare_aggregate(seed_aggregate, result["seed_aggregate"],
                              "round3/seed_aggregate", metrics=QUALITY)
    return {"study_id": "v2_iterative_v1_round3", "round": 3, "passed": True,
            "round3_result_sha256": sha(ROUND3_RESULT),
            "round3_declaration_sha256": sha(ROUND3_DECL), "selected_training_arm": selected,
            "cell_count": len(keys), "reference_spell_counts": reference_counts,
            "aggregate_recomputed": aggregate, "strict_gates_recomputed": gates,
            "seed_aggregate_recomputed": seed_aggregate,
            "prediction_artifact_count": len(result["predictions"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "data/experiments/v2_iterative_v1/independent_verification.json")
    args = parser.parse_args()
    receipts = []
    if args.round in ("1", "all"):
        receipts.append(verify_round1())
    if args.round in ("2", "all") and ROUND2_RESULT.is_file():
        receipts.append(verify_round2())
    elif args.round == "2":
        raise FileNotFoundError(ROUND2_RESULT)
    if args.round in ("3", "all") and ROUND3_RESULT.is_file():
        receipts.append(verify_round3())
    elif args.round == "3":
        raise FileNotFoundError(ROUND3_RESULT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "v2_iterative_independent_verification_v1", "rounds": receipts,
               "passed": all(row["passed"] for row in receipts),
               "organizer_reference_access": False, "public_or_GAR_scoring": False,
               "new_training": False}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": payload["passed"], "rounds": [r["round"] for r in receipts],
                      "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
