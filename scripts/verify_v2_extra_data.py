#!/usr/bin/env python3
"""Independent saved-output audit for v2_extra_data_v1; never reads public labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import v2_extra_data_eval as study  # noqa: E402
from experiments import v2_spell_balanced_eval as shared  # noqa: E402

DECL = study.DECLARATION
RESULT = study.RESULT
OUT = ROOT / "experiments/results/v2_extra_data_v1_verification.json"
TOL = 1e-12


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _conditional(prob):
    x = np.asarray(prob, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Invalid probability matrix")
    ha = x[:, :2]
    total = ha.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError("Zero H/A mass")
    return ha / total


def _blend(ce, calibrated, adopted, recipe):
    w = (recipe["ce_weight"], recipe["calibrated_weight"], recipe["adopted_weight"])
    result = (w[0] * _conditional(ce) + w[1] * _conditional(calibrated) +
              w[2] * _conditional(adopted))
    return result / result.sum(axis=1, keepdims=True)


def _fixed_d_decode(probability, adopted, model_period, fps):
    frozen = shared.runtime.frozen
    adopted = np.asarray(adopted).astype("U1", copy=False)
    candidate = adopted.copy()
    p = np.asarray(probability, dtype=np.float64)
    conditional = np.column_stack((p, np.zeros(len(p), dtype=np.float64)))
    live = adopted != "D"
    for start, stop in frozen.supervised.blocks(model_period):
        i = start
        while i < stop:
            if not live[i]:
                i += 1
                continue
            j = i + 1
            while j < stop and live[j]:
                j += 1
            candidate[i:j] = frozen.audit.decode(conditional[i:j], model_period[i:j], fps,
                                                   shared._decoder(3.0))
            i = j
    if not np.array_equal(candidate == "D", adopted == "D"):
        raise AssertionError("D mask changed")
    return candidate


def _load_control(key, fact):
    path = ROOT / fact["path"]
    if sha(path) != fact["sha256"]:
        raise ValueError(f"Control cache hash mismatch: {key}")
    with np.load(path, allow_pickle=False) as z:
        identity = {k.removeprefix("identity__"): str(z[k].item())
                    for k in z.files if k.startswith("identity__")}
        wanted = {k: str(v) for k, v in fact["identity"].items()}
        if identity != wanted:
            raise ValueError(f"Control identity mismatch: {key}")
        return {name.removeprefix("prob__"): z[name].copy()
                for name in z.files if name.startswith("prob__")} | {
            "labels": z["labels"].copy(), "period": z["period"].copy(),
            "timestamps": z["timestamps"].copy(), "model_period": z["model_period"].copy(),
            "fps": float(z["fps"])}


def _score(cell, prediction):
    return shared.controls.audit.score({k: cell[k] for k in ("labels", "period", "timestamps", "fps")}, prediction)


def _audit_cell(key, declaration, result):
    fact = declaration["controls"][key]
    control = _load_control(key, fact)
    cache_fact = result["prediction_cache"][key]
    path = ROOT / cache_fact["path"]
    if sha(path) != cache_fact["sha256"]:
        raise ValueError(f"Prediction-cache hash mismatch: {key}")
    with np.load(path, allow_pickle=False) as saved:
        identity = {name.removeprefix("identity__"): str(saved[name].item())
                    for name in saved.files if name.startswith("identity__")}
        if identity != {k: str(v) if not isinstance(v, list) else json.dumps(v)
                        for k, v in cache_fact["identity"].items()}:
            raise ValueError(f"Prediction-cache identity mismatch: {key}")
        phase_prefix = f"phase__{declaration['arms'][0]}__{study.SEEDS[0]}__"
        phase_ids = sorted(int(name.removeprefix(phase_prefix)) for name in saved.files
                           if name.startswith(phase_prefix))
        if phase_ids not in (list(range(2)), list(range(5))):
            raise ValueError(f"Unexpected native phase IDs for {key}: {phase_ids}")
        phase_count = len(phase_ids)
        rows, seed_rows = {}, {}
        adopted_decoded = shared.runtime.frozen.audit.decode(control["adopted"], control["model_period"],
            control["fps"], shared.runtime.frozen.audit.CONFIG)
        for arm in declaration["arms"]:
            phases = {seed: [] for seed in study.SEEDS}
            for seed in study.SEEDS:
                for phase in range(phase_count):
                    phases[seed].append(saved[f"phase__{arm}__{seed}__{phase}"].copy())
            pooled = np.mean([np.mean([phases[s][phase] for s in study.SEEDS], axis=0)
                              for phase in range(len(phases[study.SEEDS[0]]))], axis=0)
            saved_pooled = saved[f"pooled__{arm}"]
            if not np.allclose(pooled, saved_pooled, atol=TOL, rtol=0):
                raise AssertionError(f"Seed/phase pooled probabilities differ: {key}/{arm}")
            p = _blend(pooled, control["calibrated"], control["adopted"], declaration["recipe"])
            pred = _fixed_d_decode(p, adopted_decoded, control["model_period"], control["fps"])
            code = {"H": 0, "A": 1, "D": 2, "X": 3}
            encoded = np.asarray([code[x] for x in pred], dtype=np.int8)
            if not np.array_equal(encoded, saved[f"pred__{arm}"]):
                raise AssertionError(f"Saved ensemble prediction differs: {key}/{arm}")
            rows[arm] = _score(control, pred)
            for seed in study.SEEDS:
                seed_ce = np.mean(phases[seed], axis=0)
                seed_p = _blend(seed_ce, control[f"adop_seed{seed}"],
                                control[f"cand_seed{seed}"], declaration["recipe"])
                seed_pred = _fixed_d_decode(seed_p, adopted_decoded, control["model_period"], control["fps"])
                seed_encoded = np.asarray([code[x] for x in seed_pred], dtype=np.int8)
                if not np.array_equal(seed_encoded, saved[f"seedpred__{arm}|{seed}"]):
                    raise AssertionError(f"Saved seed prediction differs: {key}/{arm}/{seed}")
                seed_rows[f"{arm}|{seed}"] = _score(control, seed_pred)
    if any(rows[arm]["reference_spell_count"] != rows[declaration["arms"][0]]["reference_spell_count"]
           for arm in declaration["arms"]):
        raise AssertionError(f"Reference spell count changed: {key}")
    return rows, seed_rows


def _aggregate(rows):
    """Reduce reps within match, matches within provider, then providers equally."""
    result = {}
    for view, reps in study.VIEWS:
        result[view] = {}
        for metric in study.METRICS:
            grouped = {}
            for key, values in rows.items():
                provider, match, cell_view, rep = key.split("|")
                if cell_view == view and int(rep) in reps:
                    grouped.setdefault(provider, {}).setdefault(match, []).append(float(values[metric]))
            by_provider = {provider: float(np.mean([np.mean(match_rows) for match_rows in matches.values()]))
                           for provider, matches in grouped.items()}
            result[view][metric] = {"equal_provider": float(np.mean(list(by_provider.values()))),
                "providers": by_provider,
                "cells": sum(len(v) for matches in grouped.values() for v in matches.values())}
    return result


def verify():
    declaration, result = json.loads(DECL.read_text()), json.loads(RESULT.read_text())
    if result.get("declaration_sha256") != sha(DECL):
        raise ValueError("Result does not bind the frozen declaration")
    for name, expected in declaration["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Bound evaluator source changed: {name}")
    fit_seal = ROOT / declaration["fit_seal_path"]
    if sha(fit_seal) != declaration["fit_seal_sha256"]:
        raise ValueError("Training seal changed")
    for name, fact in declaration["fit_inventory"].items():
        model = ROOT / fact["path"]
        if sha(model) != fact["sha256"]:
            raise ValueError(f"Fit checkpoint changed: {name}")
    cells = sorted(declaration["controls"])
    if cells != sorted(study._expected_cells()) or len(cells) != 35:
        raise ValueError("Cohort mismatch")
    cell_rows, seed_rows = {}, {}
    for key in cells:
        cell_rows[key], seed_rows[key] = _audit_cell(key, declaration, result)
        for arm in declaration["arms"]:
            for metric in study.METRICS:
                if abs(cell_rows[key][arm][metric] - result["cell_metrics"][key][arm][metric]) > TOL:
                    raise AssertionError(f"Cell score mismatch: {key}/{arm}/{metric}")
            for seed in study.SEEDS:
                saved = result["seed_cells"][key][f"{arm}|{seed}"]
                rebuilt = seed_rows[key][f"{arm}|{seed}"]
                for metric in study.METRICS:
                    if abs(rebuilt[metric] - saved[metric]) > TOL:
                        raise AssertionError(f"Seed score mismatch: {key}/{arm}/{seed}/{metric}")
    arms = declaration["arms"]
    aggregates = {arm: _aggregate({key: cell_rows[key][arm] for key in cells}) for arm in arms}
    seed_aggregates = {arm: {str(seed): _aggregate({
        key: {m: seed_rows[key][f"{arm}|{seed}"][m] for m in study.METRICS} for key in cells})
        for seed in study.SEEDS} for arm in arms}
    checks = {}
    for arm in arms:
        for view, _ in study.VIEWS:
            for metric in study.METRICS:
                actual = aggregates[arm][view][metric]["equal_provider"]
                recorded = result["aggregate"][arm][view][metric]["equal_provider"]
                checks[f"{arm}/{view}/{metric}"] = abs(actual-recorded) <= TOL
                checks[f"{arm}/{view}/{metric}/providers"] = (
                    aggregates[arm][view][metric]["providers"] == result["aggregate"][arm][view][metric]["providers"])
    if not all(checks.values()):
        raise AssertionError(f"Saved aggregate mismatch: {[k for k,v in checks.items() if not v][:5]}")
    baseline = json.loads(study.REPLAY.read_text())["aggregate"]
    gates = {}
    for arm in arms:
        checks_arm = {}
        for view, _ in study.VIEWS:
            for metric, kind in (("spell_f1", "ge"), ("macro_f1", "macro"),
                                 ("short_spell_recall_iou_0_5_lt3s", "ge")):
                target = baseline[view][metric]["equal_provider"] - (study.MACRO_ALLOWANCE if kind == "macro" else 0)
                actual = aggregates[arm][view][metric]["equal_provider"]
                strict = view == "feed_like_v2" and metric == "spell_f1"
                passed = actual > target if strict else actual >= target
                checks_arm[f"{view}/{metric}"] = {"actual": actual, "minimum": target,
                    "comparison": "strictly_greater" if strict else "greater_or_equal", "passed": passed}
        d_same = all(aggregates[arm][v][m]["equal_provider"] == baseline[v][m]["equal_provider"]
                     for v, _ in study.VIEWS for m in ("d_f1", "d_precision", "d_recall"))
        refs_same = all(cell_rows[k][arm]["reference_spell_count"] == result["cohort"]["reference_spell_counts"][k]
                        for k in cells)
        gates[arm] = {"checks": checks_arm, "d_metrics_unchanged": d_same,
                      "reference_spell_counts_invariant": refs_same,
                      "eligible": all(checks_arm.values()) and d_same and refs_same}
        if gates[arm] != result["eligibility"][arm]:
            raise AssertionError(f"Eligibility mismatch: {arm}")
    ranking = sorted(arms, key=lambda arm: (-aggregates[arm]["feed_like_v2"]["spell_f1"]["equal_provider"],
        -aggregates[arm]["feed_like_v2"]["macro_f1"]["equal_provider"], arm))
    ranking = [arm for arm in ranking if gates[arm]["eligible"]]
    if "eligible_ranking" in result and ranking != result["eligible_ranking"]:
        raise AssertionError("Eligible ranking mismatch")
    receipt = {"study_id": study.STUDY, "passed": True, "cell_count": len(cells),
        "arms": arms, "declaration_sha256": sha(DECL), "result_sha256": sha(RESULT),
        "aggregate": aggregates, "seed_diagnostics": seed_aggregates,
        "eligibility": gates, "eligible_ranking": ranking,
        "scored_public_reference": False, "promotion": False}
    with OUT.open("x") as f:
        json.dump(receipt, f, indent=1, sort_keys=True)
        f.write("\n")
    return receipt


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
