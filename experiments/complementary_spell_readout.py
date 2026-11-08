"""Independent, prediction-file based audit for ``seed_mix_v1``.

This readout deliberately trusts neither the runner's saved cell metrics nor
its aggregates: it verifies all bound files, reloads labels/masks, scores each
saved prediction with the official spell-MBR metric implementation, and
reconstructs the match-then-provider means. It never runs inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import seed_mix_v1 as study
from experiments import spell_mbr_run as official

ARMS = tuple(f"{w:.2f}" for w in study.WEIGHTS) + (
    "boundary/shift_0.2", "boundary/shift_0.4", "boundary/shift_0.8")
TOL = 1e-10


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _close(actual, expected, where: str, tol: float = TOL) -> None:
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if int(actual) != int(expected):
            raise ValueError(f"Independent metric mismatch at {where}: {actual} != {expected}")
    elif expected is None:
        if actual is not None:
            raise ValueError(f"Independent metric mismatch at {where}: {actual} != null")
    elif not math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tol):
        raise ValueError(f"Independent metric mismatch at {where}: {actual} != {expected}")


def _prediction_labels(encoded: np.ndarray) -> np.ndarray:
    # seed_mix writes unicode labels directly; accept the conventional compact
    # 0..3 representation as well for forward-compatible saved readouts.
    a = np.asarray(encoded)
    if a.ndim != 1:
        raise ValueError("Saved prediction must be one-dimensional")
    if a.dtype.kind in "iu":
        if not np.isin(a, (0, 1, 2, 3)).all():
            raise ValueError("Saved prediction contains an invalid encoded label")
        return np.asarray(("H", "A", "D", "X"), dtype="U1")[a]
    a = a.astype("U1", copy=False)
    if not np.isin(a, ("H", "A", "D", "X")).all():
        raise ValueError("Saved prediction contains an invalid label")
    return a


def _score_cell(cell: Mapping, prediction: np.ndarray) -> dict:
    raw = official._score_prediction(cell["labels"], prediction, cell["period"],
                                     cell["timestamps"], float(cell["fps"]))
    return {
        "spell_f1": raw["spell_f1"], "macro_f1": raw["macro_f1"],
        "d_f1": raw["d_f1"], "d_precision": raw["d_precision"],
        "d_recall": raw["d_recall"], "predicted_spell_count": raw["predicted_spell_count"],
        "reference_spell_count": raw["reference_spell_count"],
        "short_spell_recall_iou_0_5_lt3s": raw["short_spell_recall_iou_0_5_lt3s"],
        "graded_frames": raw["graded_frames"],
        "short_reference_spell_count": raw["short_reference_spell_count"],
        "short_matched_spell_count": raw["short_matched_spell_count"],
    }


def _aggregate(rows: Mapping[str, dict], manifest: Mapping) -> dict:
    out = {}
    groups = manifest["cohort"]["view_groups"]
    for group, views in groups.items():
        out[group] = {}
        for metric in study.METRICS:
            by_cell = {}
            for key, row in rows.items():
                provider, match, view, rep = key.split("|")
                if view in views:
                    by_cell[(provider, match, int(rep))] = row[metric]
            per_provider = {}
            for (provider, match, _rep), value in by_cell.items():
                per_provider.setdefault(provider, {}).setdefault(match, []).append(value)
            provider_means = {}
            for provider, matches in per_provider.items():
                # Replicates are first averaged within their match; matches are
                # then averaged equally within provider.
                match_means = []
                for values in matches.values():
                    present = [float(v) for v in values if v is not None]
                    match_means.append(float(np.mean(present)) if present else None)
                present = [v for v in match_means if v is not None]
                if present:
                    provider_means[provider] = float(np.mean(present))
            out[group][metric] = {
                "equal_provider": float(np.mean(list(provider_means.values()))) if provider_means else None,
                "providers": dict(sorted(provider_means.items())),
                "cells": len(by_cell),
            }
    return out


def _compare_metric_tree(actual: Mapping, expected: Mapping, where: str) -> None:
    for metric in study.METRICS:
        _close(actual[metric], expected[metric], f"{where}/{metric}")
    for count in ("graded_frames", "short_reference_spell_count", "short_matched_spell_count"):
        _close(actual[count], expected[count], f"{where}/{count}")


def _guard_report(candidate: Mapping, control: Mapping) -> dict:
    checks = {}
    for view in ("clean", "feed_like_v1", "feed_like_v2"):
        for metric in ("spell_f1", "short_spell_recall_iou_0_5_lt3s"):
            actual = candidate[view][metric]["equal_provider"]
            minimum = control[view][metric]["equal_provider"]
            checks[f"{view}/{metric}"] = {"actual": actual, "minimum": minimum, "passed": actual >= minimum}
        actual = candidate[view]["macro_f1"]["equal_provider"]
        minimum = control[view]["macro_f1"]["equal_provider"] - .001
        checks[f"{view}/macro_f1"] = {"actual": actual, "minimum": minimum, "passed": actual >= minimum}
    for metric in ("macro_f1", "d_f1"):
        for provider, baseline in control["clean"][metric]["providers"].items():
            actual = candidate["clean"][metric]["providers"][provider]
            minimum = baseline - .02
            checks[f"clean/{provider}/{metric}"] = {"actual": actual, "minimum": minimum,
                                                       "passed": actual >= minimum}
    actual = candidate["feed_like_v2"]["spell_f1"]["equal_provider"]
    minimum = control["feed_like_v2"]["spell_f1"]["equal_provider"]
    checks["feed_like_v2/spell_f1_strict_improvement"] = {
        "actual": actual, "minimum": minimum, "passed": actual > minimum}
    return {"checks": checks, "passed": all(c["passed"] for c in checks.values())}


def _run_guard(result_gate: Mapping, independent: Mapping, control: Mapping, arm: str) -> None:
    recomputed = _guard_report(independent, control)
    saved = result_gate[arm]
    if saved.get("passed") != recomputed["passed"] or saved.get("checks", {}).keys() != recomputed["checks"].keys():
        raise ValueError(f"Saved guard inventory/status differs from independent guards: {arm}")
    for name, check in recomputed["checks"].items():
        saved_check = saved["checks"][name]
        if saved_check.get("passed") != check["passed"]:
            raise ValueError(f"Saved guard result differs: {arm}/{name}")
        _close(saved_check["actual"], check["actual"], f"guard/{arm}/{name}/actual")
        _close(saved_check["minimum"], check["minimum"], f"guard/{arm}/{name}/minimum")


def _collapsed_order(y: np.ndarray) -> tuple[str, ...]:
    # The boundary relocator can move boundaries but must preserve the ordered
    # run labels and every non-H/A run.
    return tuple(y[np.r_[True, y[1:] != y[:-1]]].tolist())


def _prediction_file(root: Path, pred_dir: Path, fact: Mapping, key: str) -> Path:
    path = _resolve(root, fact["path"])
    if not path.is_relative_to(pred_dir.resolve()):
        raise ValueError(f"Prediction path escapes the declared prediction directory: {key}")
    if not path.is_file() or _sha(path) != fact["sha256"]:
        raise ValueError(f"Missing or changed saved prediction: {key}")
    return path


def verify(manifest_path: Path, declaration_path: Path, preflight_path: Path,
           result_path: Path, predictions_dir: Path, root: Path = study.ROOT) -> dict:
    """Verify a frozen run entirely from receipts, native caches, and predictions."""
    root = Path(root).resolve()
    manifest_path, declaration_path = Path(manifest_path).resolve(), Path(declaration_path).resolve()
    preflight_path, result_path = Path(preflight_path).resolve(), Path(result_path).resolve()
    pred_dir = Path(predictions_dir).resolve()
    manifest = json.loads(manifest_path.read_text())
    study.validate_manifest(manifest)
    declaration = json.loads(declaration_path.read_text())
    if declaration.get("study_id") != study.STUDY or declaration.get("manifest_sha256") != _sha(manifest_path):
        raise ValueError("Declaration does not bind the supplied manifest")
    if declaration.get("source_hashes") != manifest.get("source_hashes"):
        raise ValueError("Manifest/declaration source inventories differ")
    study.verify_source_hashes(manifest["source_hashes"], root)
    preflight = json.loads(preflight_path.read_text())
    if (preflight.get("study_id") != study.STUDY or
            preflight.get("manifest_sha256") != _sha(manifest_path) or
            preflight.get("declaration_sha256") != _sha(declaration_path) or
            preflight.get("source_hashes") != manifest["source_hashes"] or
            preflight.get("control_preflight", {}).get("passed") is not True):
        raise ValueError("Passed preflight does not bind these exact inputs")
    result = json.loads(result_path.read_text())
    if (result.get("study_id") != study.STUDY or
            result.get("source") != manifest.get("source") or
            result.get("evidence_scope") != manifest.get("evidence_scope") or
            result.get("control_preflight", {}).get("preflight_sha256") != _sha(preflight_path) or
            result.get("control_preflight", {}).get("passed") is not True):
        raise ValueError("Run result does not bind the passed preflight")
    expected = set(manifest["expected_cells"])
    if set(result.get("baseline_rows", {})) != expected or set(result.get("cell_inventory", {})) != expected:
        raise ValueError("Baseline or cell inventory is not the complete frozen cohort")
    if set(result.get("cell_metrics", {})) != {f"{w:.2f}" for w in study.WEIGHTS}:
        raise ValueError("Result does not contain all five fixed seed-weight arms")
    if set(result.get("boundary", {}).get("cell_metrics", {})) != {"shift_0.2", "shift_0.4", "shift_0.8"}:
        raise ValueError("Result does not contain all three boundary arms")
    inventory = result.get("prediction_inventory", {})
    if set(inventory) != expected:
        raise ValueError("Saved prediction inventory does not cover every cohort cell")

    independent: dict[str, dict] = {arm: {} for arm in ARMS}
    control_rows = {}
    boundary_count_order = 0
    for key in manifest["expected_cells"]:
        fact = manifest["cells"][key]
        cache_path = study._safe_cache_path(root, fact["path"])
        if not cache_path.is_file() or _sha(cache_path) != fact["sha256"]:
            raise ValueError(f"Missing or changed native probability cache: {key}")
        cell = study.load_cell(root, key, fact, manifest["source"]["study_id"])
        p_fact = inventory[key]
        pred_path = _prediction_file(root, pred_dir, p_fact, key)
        with np.load(pred_path, allow_pickle=False) as saved:
            expected_names = {f"pred__seed2026_weight_{w:.2f}" for w in study.WEIGHTS}
            expected_names |= {"pred__boundary_shift_0.2", "pred__boundary_shift_0.4",
                               "pred__boundary_shift_0.8", "baseline_prediction"}
            if not expected_names.issubset(saved.files):
                raise ValueError(f"Prediction archive is incomplete: {key}")
            baseline = _prediction_labels(saved["baseline_prediction"])
            if not np.array_equal(baseline, cell["predecessor"]):
                raise ValueError(f"Saved baseline prediction differs from hashed cache: {key}")
            if not np.array_equal(baseline[np.isin(baseline, ("D", "X"))], cell["predecessor"][np.isin(cell["predecessor"], ("D", "X"))]):
                raise ValueError(f"Baseline D/X identity mismatch: {key}")
            control_rows[key] = _score_cell(cell, baseline)
            for weight in study.WEIGHTS:
                name = f"pred__seed2026_weight_{weight:.2f}"
                prediction = _prediction_labels(saved[name])
                if prediction.shape != baseline.shape:
                    raise ValueError(f"Prediction length mismatch: {key}/{name}")
                if not np.array_equal(prediction == "D", baseline == "D") or not np.array_equal(prediction == "X", baseline == "X"):
                    raise ValueError(f"Seed-weight decoder changed an immutable D/X row: {key}/{name}")
                arm = f"{weight:.2f}"
                independent[arm][key] = _score_cell(cell, prediction)
                _compare_metric_tree(independent[arm][key], result["cell_metrics"][arm][key], f"{arm}/{key}")
                for count in ("reference_spell_count", "graded_frames"):
                    _close(independent[arm][key][count], control_rows[key][count], f"{arm}/{key}/{count}_baseline_parity")
            for strength in ("shift_0.2", "shift_0.4", "shift_0.8"):
                name = f"pred__boundary_{strength}"
                prediction = _prediction_labels(saved[name])
                if prediction.shape != baseline.shape:
                    raise ValueError(f"Boundary prediction length mismatch: {key}/{strength}")
                fixed = np.isin(baseline, ("D", "X"))
                if not np.array_equal(prediction == "D", baseline == "D") or not np.array_equal(prediction == "X", baseline == "X"):
                    raise ValueError(f"Boundary arm changed an immutable D/X row: {key}/{strength}")
                if _collapsed_order(prediction) != _collapsed_order(baseline):
                    raise ValueError(f"Boundary arm changed raw run count/order: {key}/{strength}")
                boundary_count_order += 1
                arm = f"boundary/{strength}"
                independent[arm][key] = _score_cell(cell, prediction)
                _compare_metric_tree(independent[arm][key], result["boundary"]["cell_metrics"][strength][key], f"{arm}/{key}")
                for count in ("reference_spell_count", "graded_frames"):
                    _close(independent[arm][key][count], control_rows[key][count], f"{arm}/{key}/{count}_baseline_parity")
        for metric in study.METRICS:
            _close(control_rows[key][metric], result["baseline_rows"][key][metric], f"baseline/{key}/{metric}")
        recorded = result["cell_inventory"][key]
        if recorded.get("cell_count") != len(baseline):
            raise ValueError(f"Recorded raw frame count mismatch: {key}")
        if recorded.get("historical_prediction_sha256") != hashlib.sha256(baseline.tobytes()).hexdigest():
            raise ValueError(f"Recorded historical prediction hash mismatch: {key}")
        if recorded.get("d_mask_sha256") != hashlib.sha256((baseline == "D").tobytes()).hexdigest():
            raise ValueError(f"Recorded historical D-mask hash mismatch: {key}")

    baseline_aggregate = _aggregate(control_rows, manifest)
    for group, metrics in baseline_aggregate.items():
        for metric, row in metrics.items():
            _close(result["baseline_aggregate"][group][metric]["equal_provider"], row["equal_provider"],
                   f"baseline_aggregate/{group}/{metric}")
            if result["baseline_aggregate"][group][metric]["providers"].keys() != row["providers"].keys():
                raise ValueError(f"Baseline provider inventory mismatch: {group}/{metric}")
            for provider, value in row["providers"].items():
                _close(result["baseline_aggregate"][group][metric]["providers"][provider], value,
                       f"baseline_aggregate/{group}/{metric}/{provider}")
    aggregates = {}
    gate_passes = {}
    for arm in ARMS:
        rows = independent[arm]
        if set(rows) != expected:
            raise ValueError(f"Independent arm cell coverage mismatch: {arm}")
        aggregate = _aggregate(rows, manifest)
        aggregates[arm] = aggregate
        if arm.startswith("boundary/"):
            label = arm.split("/", 1)[1]
            saved_aggregate = result["boundary"]["aggregate"][label]
            saved_gates = result["boundary"]["gates"]
        else:
            label = arm
            saved_aggregate = result["aggregate"][label]
            saved_gates = result["gates"]
        for group, metrics in aggregate.items():
            for metric, row in metrics.items():
                saved = saved_aggregate[group][metric]
                _close(saved["equal_provider"], row["equal_provider"], f"aggregate/{arm}/{group}/{metric}")
                if saved["providers"].keys() != row["providers"].keys():
                    raise ValueError(f"Provider aggregate inventory mismatch: {arm}/{group}/{metric}")
                for provider, value in row["providers"].items():
                    _close(saved["providers"][provider], value, f"aggregate/{arm}/{group}/{metric}/{provider}")
        _run_guard(saved_gates, aggregate, baseline_aggregate, label)
        gate_passes[arm] = _guard_report(aggregate, baseline_aggregate)["passed"]

    # Pick the declared v2 winner by v2 spell F1, then macro F1, then lower
    # weight for deterministic ties. LOO chooses on the other two providers.
    weights = [f"{w:.2f}" for w in study.WEIGHTS]
    v2_winner = max(weights, key=lambda w: (
        aggregates[w]["feed_like_v2"]["spell_f1"]["equal_provider"],
        aggregates[w]["feed_like_v2"]["macro_f1"]["equal_provider"], -float(w)))
    loo = {}
    providers = sorted(aggregates[weights[0]]["feed_like_v2"]["spell_f1"]["providers"])
    for heldout in providers:
        def train_score(w):
            vals = aggregates[w]["feed_like_v2"]["spell_f1"]["providers"]
            return float(np.mean([v for p, v in vals.items() if p != heldout]))
        def train_macro(w):
            vals = aggregates[w]["feed_like_v2"]["macro_f1"]["providers"]
            return float(np.mean([v for p, v in vals.items() if p != heldout]))
        chosen = max(weights, key=lambda w: (train_score(w), train_macro(w), -float(w)))
        loo[heldout] = {"selected_weight_seed2026": float(chosen),
            "other_provider_mean_spell_f1": train_score(chosen),
            "other_provider_mean_macro_f1": train_macro(chosen),
            "heldout_spell_f1": aggregates[chosen]["feed_like_v2"]["spell_f1"]["providers"][heldout]}

    return {
        "study_id": study.STUDY, "passed": True,
        "input_sha256": {"manifest": _sha(manifest_path), "declaration": _sha(declaration_path),
                         "preflight": _sha(preflight_path), "result": _sha(result_path),
                         "verifier": _sha(Path(__file__).resolve())},
        "source": manifest["source"], "evidence_scope": manifest["evidence_scope"],
        "cells_verified": len(expected), "arms_verified": len(ARMS),
        "scored_predictions": len(expected) * len(ARMS),
        "boundary_count_and_order_checks": boundary_count_order,
        "aggregate": aggregates,
        "baseline_aggregate": baseline_aggregate,
        "guards": {arm: {"passed": gate_passes[arm],
                          "checks": _guard_report(aggregates[arm], baseline_aggregate)["checks"]}
                   for arm in ARMS},
        "v2_weight_winner_seed2026": float(v2_winner),
        "v2_winner_spell_f1": aggregates[v2_winner]["feed_like_v2"]["spell_f1"]["equal_provider"],
        "leave_one_provider_out_seed_weight": loo,
        "organizer_reference_access": False, "promotion": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=study.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.manifest, args.declaration, args.preflight, args.result,
                    args.predictions_dir, args.root)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
