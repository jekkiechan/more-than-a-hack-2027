"""Independent replay and scoring for the fixed R2 gate-then-boundary composition.

This verifier uses sealed R1 probabilities and labels, and saved R1/R2 predictions.
It reconstructs gate then boundary using the shared primitive only; it never calls
the R2 runner's composition function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import boundary_relocation_v1 as boundary
from experiments import frontier_postprocess_readout as r1readout
from experiments import frontier_postprocess_round2 as runner
from experiments import frontier_postprocess_v1 as r1

ROOT = r1.ROOT
GATE_ARM = "disagreement_lambda_1.0"
BOUNDARY_ARM = "boundary_shift_0.8"
COMBINED_ARM = "gate_lambda_1_then_boundary_shift_0.8"
TOL = 1e-10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _check_inventory(inventory: Mapping[str, str], root: Path) -> None:
    if not inventory:
        raise ValueError("Empty bound source/evidence hash inventory")
    for name, digest in inventory.items():
        path = _resolve(root, name)
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed bound source/evidence: {name}")


def _validate_binding(manifest_path: Path, declaration_path: Path,
                      result_path: Path, pred_dir: Path, root: Path) -> tuple[dict, dict, dict, dict]:
    manifest = json.loads(manifest_path.read_text())
    declaration = json.loads(declaration_path.read_text())
    result = json.loads(result_path.read_text())
    if (manifest.get("schema") != "frontier_postprocess_round2_manifest" or
            manifest.get("study_id") != runner.STUDY or
            declaration.get("study_id") != runner.STUDY or
            declaration.get("status") != "frozen_before_candidate_scoring" or
            declaration.get("evaluation_manifest_sha256") != sha256(manifest_path) or
            declaration.get("source_hashes") != manifest.get("source_hashes") or
            declaration.get("composition") != manifest.get("composition") or
            declaration.get("guards") != manifest.get("guards") or
            manifest.get("guards") != r1.GUARDS or manifest.get("ranking") != r1.RANKING or
            manifest.get("source_status") not in ("exact_restored", "reconstructed") or
            manifest.get("evidence_scope") is None or
            manifest.get("composition") != {"order": [GATE_ARM, BOUNDARY_ARM],
                "gate_lambda": 1.0, "boundary_strength": "shift_0.8",
                "arm_id": COMBINED_ARM,
                "probability_source": "equal-seed frontier H/A probabilities padded with zero D"}):
        raise ValueError("R2 declaration does not bind the fixed composition")
    expected_cells = set(r1.EXPECTED_CELLS)
    if (manifest.get("cohort") != {"cell_count": 35, "cells": sorted(expected_cells)} or
            declaration.get("cohort") != manifest["cohort"] or
            manifest.get("r1_evaluation_manifest_sha256") != manifest.get("r1_artifacts", {}).get("evaluation_manifest.json", {}).get("sha256") or
            manifest.get("organizer_reference_access") is not False or
            manifest.get("promotion") is not False):
        raise ValueError("R2 cohort or no-public/no-promotion contract is invalid")
    _check_inventory(manifest.get("source_hashes", {}), root)
    if (result.get("study_id") != runner.STUDY or
            result.get("source_status") != manifest["source_status"] or
            result.get("evidence_scope") != manifest["evidence_scope"] or
            result.get("manifest_sha256") != sha256(manifest_path) or
            result.get("declaration_sha256") != sha256(declaration_path) or
            result.get("source_hashes") != manifest["source_hashes"] or
            result.get("arm_id") != COMBINED_ARM or
            result.get("composition") != manifest["composition"] or
            result.get("r1_result_sha256") is None or
            result.get("r1_readout_sha256") is None or
            result.get("baseline_replay", {}).get("passed") is not True):
        raise ValueError("R2 result does not bind the frozen declaration and R1 control replay")
    if set(result.get("cell_metrics", {})) != expected_cells or set(result.get("baseline_rows", {})) != expected_cells:
        raise ValueError("R2 result is missing cell metrics or matched controls")
    if set(result.get("prediction_inventory", {})) != expected_cells:
        raise ValueError("R2 prediction inventory must contain all 35 cells")
    r1facts = manifest.get("r1_artifacts", {})
    required = {"evaluation_manifest.json", "declaration.json", "preflight.json", "results.json",
                "frontier_postprocess_v1_readout.json", "native_manifest.json"}
    if set(r1facts) != required:
        raise ValueError("R2 must bind all R1 receipts, including the native probability manifest")
    for name, fact in r1facts.items():
        path = _resolve(root, fact["path"])
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != fact.get("sha256"):
            raise ValueError(f"Bound R1 artifact missing or changed: {name}")
    readout_path = _resolve(root, r1facts["frontier_postprocess_v1_readout.json"]["path"])
    r1readout_result = json.loads(readout_path.read_text())
    r1result_path = _resolve(root, r1facts["results.json"]["path"])
    r1manifest_path = _resolve(root, r1facts["evaluation_manifest.json"]["path"])
    r1declaration_path = _resolve(root, r1facts["declaration.json"]["path"])
    hashes = r1readout_result.get("hashes", {})
    if (r1readout_result.get("study_id") != r1.STUDY or r1readout_result.get("passed") is not True or
            r1readout_result.get("scored_predictions") != 350 or
            hashes.get("cache_manifest") != sha256(r1manifest_path) or
            hashes.get("declaration") != sha256(r1declaration_path) or
            hashes.get("run_result") != sha256(r1result_path) or
            hashes.get("verifier_source") != sha256(Path(r1readout.__file__).resolve()) or
            result.get("r1_result_sha256") != sha256(r1result_path) or
            result.get("r1_readout_sha256") != sha256(readout_path)):
        raise ValueError("Independent passed R1 readout does not bind R2 inputs")
    if (result["baseline_replay"].get("preflight_sha256") !=
            sha256(_resolve(root, r1facts["preflight.json"]["path"]))):
        raise ValueError("R2 control replay receipt differs from frozen R1 preflight")
    if set(hashes.get("prediction_files", {})) != expected_cells or set(hashes.get("probability_caches", {})) != expected_cells:
        raise ValueError("R1 independent readout does not bind complete prediction/cache inventories")
    if result.get("official_metrics") != list(r1.METRICS):
        raise ValueError("R2 metric inventory differs from official R1 metrics")
    native_fact = r1facts["native_manifest.json"]
    native = json.loads(_resolve(root, native_fact["path"]).read_text())
    native_ref = r1manifest_path and json.loads(r1manifest_path.read_text()).get("frozen_native_manifest", {})
    if (native_fact["sha256"] != native_ref.get("sha256") or
            native_fact["path"] != native_ref.get("path") or
            sha256(_resolve(root, native_fact["path"])) != native_fact["sha256"] or
            set(hashes["probability_caches"]) != set(native.get("cells", {}))):
        raise ValueError("R2 native manifest receipt differs from R1 evaluation manifest")
    return manifest, declaration, result, {"r1_readout": r1readout_result,
        "r1_result": json.loads(r1result_path.read_text()), "r1_manifest": json.loads(r1manifest_path.read_text()),
        "r1_manifest_path": r1manifest_path, "r1_result_path": r1result_path,
        "r1_declaration_path": r1declaration_path, "r1_hashes": hashes, "pred_dir": pred_dir}


def _compare(a: Mapping, b: Mapping, where: str) -> None:
    if set(a) != set(b):
        raise ValueError(f"Metric inventory mismatch at {where}")
    for key, x in a.items():
        y = b[key]
        if isinstance(x, Mapping):
            if not isinstance(y, Mapping): raise ValueError(f"Metric structure mismatch: {where}/{key}")
            _compare(x, y, f"{where}/{key}")
        elif isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            if int(x) != int(y): raise ValueError(f"Count mismatch at {where}/{key}")
        elif isinstance(x, (float, np.floating)):
            if not math.isfinite(float(x)) or not math.isfinite(float(y)) or abs(float(x)-float(y)) > TOL:
                raise ValueError(f"Metric mismatch at {where}/{key}")
        elif x != y: raise ValueError(f"Metric mismatch at {where}/{key}")


def _prediction(path: Path, name: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as f:
        if name not in f.files: raise ValueError(f"Prediction archive lacks {name}")
        return r1readout._read_labels(f[name], name)


def _compose_independent(gated: np.ndarray, cell: Mapping, ha: np.ndarray) -> np.ndarray:
    probabilities = r1.validate_conditional_ha(ha)
    return boundary.relocate(np.column_stack((probabilities, np.zeros(len(probabilities)))),
        gated, cell["model_period"], cell["timestamps"], cell["fps"], "shift_0.8")


def _block_sequences(labels: np.ndarray, model_period: np.ndarray) -> tuple[tuple[str, ...], ...]:
    cuts = np.r_[0, np.flatnonzero(model_period[1:] != model_period[:-1]) + 1, len(labels)]
    return tuple(r1readout._run_sequence(labels[a:b]) for a, b in zip(cuts[:-1], cuts[1:]))


def verify(manifest_path: Path, declaration_path: Path, result_path: Path,
           predictions_dir: Path, output_path: Path | None = None, root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    manifest_path, declaration_path, result_path = (Path(x).resolve() for x in
        (manifest_path, declaration_path, result_path))
    pred_dir = Path(predictions_dir).resolve()
    manifest, declaration, result, r1info = _validate_binding(manifest_path,
        declaration_path, result_path, pred_dir, root)
    r1_manifest_path = r1info["r1_manifest_path"]
    r1decl = json.loads(r1info["r1_declaration_path"].read_text())
    r1manifest = r1info["r1_manifest"]
    frontier_hashes = r1readout._validate_manifest(r1manifest, r1decl, r1_manifest_path, root)
    r1_result = r1info["r1_result"]
    cells, rows, gate_rows, boundary_rows, pred_hashes, replayed = {}, {}, {}, {}, {}, {}
    gate_order_checks = 0
    for key in sorted(r1.EXPECTED_CELLS):
        fact = r1manifest["cells"][key]
        if r1info["r1_hashes"]["probability_caches"].get(key) != fact.get("sha256"):
            raise ValueError(f"R1 independent readout does not bind native cache: {key}")
        cell = r1readout._read_cache(root, key, fact, r1manifest["source_status"], frontier_hashes)
        cells[key] = cell
        base = cell["baseline_prediction"]
        ha = r1readout._probability_for_weight(cell, .5)
        if not np.allclose(ha, cell["baseline_probability"], atol=TOL, rtol=0):
            raise ValueError(f"Equal-seed H/A probabilities changed: {key}")
        baseline_score = r1readout._official_score(cell, base)
        _compare(baseline_score, r1_result["baseline_rows"][key], f"R1 control/{key}")
        fact = r1_result["prediction_inventory"][key]
        r1pred_path = _resolve(root, fact["path"])
        if (not r1pred_path.is_file() or sha256(r1pred_path) != fact["sha256"] or
                fact["sha256"] != r1info["r1_hashes"]["prediction_files"][key]):
            raise ValueError(f"R1 candidate predictions missing or changed: {key}")
        with np.load(r1pred_path, allow_pickle=False) as saved:
            r1base = r1readout._read_labels(saved["baseline_prediction"], "baseline_prediction")
            r1gate = r1readout._read_labels(saved["pred__disagreement_lambda_1.0"], "gate prediction")
            r1boundary = r1readout._read_labels(saved["pred__boundary_shift_0.8"], "boundary prediction")
        if not np.array_equal(r1base, base): raise ValueError(f"R1 baseline prediction differs: {key}")
        gate = r1readout._rebuild_prediction(GATE_ARM, cell)
        bound = r1readout._rebuild_prediction("boundary_shift_0.8", cell)
        if not np.array_equal(gate, r1gate) or not np.array_equal(bound, r1boundary):
            raise ValueError(f"R1 gate/boundary standalone arm failed independent replay: {key}")
        gate_score = r1readout._official_score(cell, gate)
        boundary_score = r1readout._official_score(cell, bound)
        _compare(gate_score, r1_result["disagreement"]["cell_metrics"][GATE_ARM][key], f"R1 gate/{key}")
        _compare(boundary_score, r1_result["boundary"]["cell_metrics"]["shift_0.8"][key], f"R1 boundary/{key}")
        _compare(gate_score, result["component_metrics"]["gate"]["cell_metrics"][key], f"R2 gate control/{key}")
        _compare(boundary_score, result["component_metrics"]["boundary_control"]["cell_metrics"][key], f"R2 boundary control/{key}")
        gate_rows[key], boundary_rows[key] = gate_score, boundary_score
        combined = _compose_independent(gate, cell, ha)
        if _block_sequences(combined, cell["model_period"]) != _block_sequences(gate, cell["model_period"]):
            raise ValueError(f"Boundary composition changed gated run order/count: {key}")
        gate_order_checks += 1
        pfact = result["prediction_inventory"][key]
        ppath = _resolve(root, pfact["path"])
        if (not ppath.is_relative_to(pred_dir) or not ppath.is_file() or sha256(ppath) != pfact.get("sha256")):
            raise ValueError(f"R2 prediction missing, unsafe, or changed: {key}")
        pred_hashes[key] = pfact["sha256"]
        actual = _prediction(ppath, "prediction")
        saved_base = _prediction(ppath, "baseline_prediction")
        saved_gate = _prediction(ppath, "gate_prediction")
        saved_boundary = _prediction(ppath, "boundary_control_prediction")
        if not np.array_equal(saved_base, base) or not np.array_equal(saved_gate, gate) or not np.array_equal(saved_boundary, bound):
            raise ValueError(f"R2 controls or standalone predictions differ from independently replayed R1: {key}")
        if not np.array_equal(actual, combined): raise ValueError(f"R2 composition does not reconstruct: {key}")
        r1readout._assert_dx_mask(actual, base, f"R2/{key}")
        _compare(baseline_score, result["baseline_rows"][key], f"R2 baseline/{key}")
        replayed[key] = (cell, base, actual, baseline_score)
    # Complete all 35 control and standalone replays before scoring any R2 arm.
    for key in sorted(replayed):
        cell, base, actual, baseline_score = replayed[key]
        score = r1readout._official_score(cell, actual)
        _compare(score, result["cell_metrics"][key], f"R2 result/{key}")
        for metric in ("graded_frames", "reference_spell_count", "short_reference_spell_count"):
            if score[metric] != baseline_score[metric]: raise ValueError(f"R2 changed grading population {metric}: {key}")
        for metric in ("d_f1", "d_precision", "d_recall"):
            if score[metric] != baseline_score[metric]: raise ValueError(f"R2 changed fixed-mask metric {metric}: {key}")
        rows[key] = score
    aggregate = r1readout._aggregate(rows, r1readout.VIEW_REPLICATES)
    _compare(aggregate, result["aggregate"], "R2 aggregate")
    gate_aggregate = r1readout._aggregate(gate_rows, r1readout.VIEW_REPLICATES)
    boundary_aggregate = r1readout._aggregate(boundary_rows, r1readout.VIEW_REPLICATES)
    _compare(gate_aggregate, result["component_metrics"]["gate"]["aggregate"], "R2 gate aggregate")
    _compare(boundary_aggregate, result["component_metrics"]["boundary_control"]["aggregate"], "R2 boundary aggregate")
    _compare(gate_aggregate, r1info["r1_readout"]["aggregate"][GATE_ARM], "R1 gate aggregate")
    _compare(boundary_aggregate, r1info["r1_readout"]["aggregate"]["boundary_shift_0.8"], "R1 boundary aggregate")
    baseline_aggregate = r1readout._aggregate(r1_result["baseline_rows"], r1readout.VIEW_REPLICATES)
    guards = r1readout._independent_guards(aggregate, baseline_aggregate)
    out = {"study_id": runner.STUDY, "passed": True, "source_status": r1manifest["source_status"],
        "cells_verified": 35, "candidate_arms_verified": 1, "candidate_predictions_scored": 35,
        "control_predictions_replayed": 105, "scored_predictions": 140,
        "boundary_raw_run_order_checks": gate_order_checks,
        "aggregate": aggregate, "baseline_aggregate": baseline_aggregate, "guards": guards,
        "eligible": guards["passed"], "decision": "eligible" if guards["passed"] else "ineligible",
        "ranking": [COMBINED_ARM] if guards["passed"] else [],
        "hashes": {"manifest": sha256(manifest_path), "declaration": sha256(declaration_path),
            "result": sha256(result_path), "verifier_source": sha256(Path(__file__).resolve()),
            "r1_artifacts": {k: v["sha256"] for k, v in manifest["r1_artifacts"].items()},
            "prediction_files": pred_hashes, "source_files": manifest["source_hashes"]},
        "organizer_reference_access": False, "promotion": False}
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x") as stream:
            json.dump(out, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.manifest, args.declaration, args.result, args.predictions_dir, args.output)


if __name__ == "__main__":
    main()
