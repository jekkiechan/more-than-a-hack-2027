"""One fixed composition of the selected R1 gate and boundary decoder."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from experiments import boundary_relocation_v1 as boundary
from experiments import frontier_postprocess_v1 as r1
from experiments import v2_spell_balanced_eval as balanced

ROOT = r1.ROOT
STUDY = "frontier_postprocess_round2"
R1_DIR = ROOT / "data/experiments/frontier_postprocess_v1"
R1_MANIFEST = R1_DIR / "evaluation_manifest.json"
R1_DECLARATION = R1_DIR / "declaration.json"
R1_PREFLIGHT = R1_DIR / "preflight.json"
R1_RESULT = R1_DIR / "results.json"
R1_READOUT = ROOT / "experiments/results/frontier_postprocess_v1_readout.json"
VERIFIER_SOURCE = ROOT / "experiments/frontier_postprocess_round2_readout.py"
GATE_ARM = "disagreement_lambda_1.0"
BOUNDARY_ARM = "boundary_shift_0.8"
COMBINED_ARM = "gate_lambda_1_then_boundary_shift_0.8"
SOURCE_STATUS = "reconstructed"
EVIDENCE_SCOPE = "same five exposed development matches; reconstructed current frontier comparator; one fixed postprocessing composition"
GUARDS = r1.GUARDS
RANKING = r1.RANKING


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _r1_inputs() -> dict:
    manifest = json.loads(R1_MANIFEST.read_text())
    declaration = json.loads(R1_DECLARATION.read_text())
    preflight = json.loads(R1_PREFLIGHT.read_text())
    result = json.loads(R1_RESULT.read_text())
    if not R1_READOUT.is_file():
        raise FileNotFoundError(f"Independent R1 readout is required: {R1_READOUT}")
    readout = json.loads(R1_READOUT.read_text())
    if (result.get("study_id") != r1.STUDY or
            result.get("baseline_replay", {}).get("passed") is not True or
            result.get("manifest_sha256") != sha256(R1_MANIFEST) or
            result.get("declaration_sha256") != sha256(R1_DECLARATION) or
            result.get("preflight_sha256") != sha256(R1_PREFLIGHT) or
            preflight.get("passed") is not True or preflight.get("candidate_scoring_performed") is not False):
        raise ValueError("R1 result/preflight is not a completed, passed study")
    if set(result.get("prediction_inventory", {})) != r1.EXPECTED_CELLS:
        raise ValueError("R1 saved predictions do not cover all 35 cells")
    return {"manifest": manifest, "declaration": declaration, "preflight": preflight,
            "result": result, "readout": readout}


def source_closure() -> dict[str, str]:
    paths = {R1_MANIFEST, R1_DECLARATION, R1_PREFLIGHT, R1_RESULT, R1_READOUT,
             VERIFIER_SOURCE, Path(__file__).resolve()}
    paths.update((ROOT / name).resolve() for name in r1.source_hashes())
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def _check_closure(closure: Mapping[str, str]) -> None:
    if not closure:
        raise ValueError("Empty source closure")
    for name, digest in closure.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Missing or changed frozen source/evidence: {name}")


def freeze(manifest_path: Path, declaration_path: Path) -> tuple[dict, dict]:
    """Freeze this single composition against the completed independent R1 evidence."""
    if manifest_path.exists() or declaration_path.exists():
        raise FileExistsError("Refusing to overwrite the round-two manifest/declaration")
    inp = _r1_inputs()
    if not VERIFIER_SOURCE.is_file():
        raise FileNotFoundError(f"Independent round-two verifier must exist before freeze: {VERIFIER_SOURCE}")
    # Validate complete native inputs and the exact R1 control replay before sealing.
    r1.validate_manifest(inp["manifest"])
    r1._verify_declaration(R1_MANIFEST, R1_DECLARATION, inp["manifest"])
    readout_hashes = inp["readout"].get("hashes", {})
    native_ref = inp["manifest"].get("frozen_native_manifest", {})
    native_path = (ROOT / native_ref.get("path", "")).resolve()
    if (inp["readout"].get("study_id") != r1.STUDY or
            inp["readout"].get("passed") is not True or
            inp["readout"].get("scored_predictions") != 350 or
            readout_hashes.get("run_result") != sha256(R1_RESULT) or
            readout_hashes.get("cache_manifest") != sha256(R1_MANIFEST) or
            readout_hashes.get("declaration") != sha256(R1_DECLARATION) or
            readout_hashes.get("baseline_preflight") != sha256(R1_PREFLIGHT) or
            readout_hashes.get("verifier_source") != sha256(ROOT / "experiments/frontier_postprocess_readout.py") or
            not native_path.is_file() or sha256(native_path) != native_ref.get("sha256")):
        raise ValueError("Independent R1 readout does not validate the current R1 result")
    result = inp["result"]
    manifest = inp["manifest"]
    expected_predictions = {key: facts["sha256"] for key, facts in result["prediction_inventory"].items()}
    if readout_hashes.get("prediction_files") != expected_predictions:
        raise ValueError("Independent R1 readout does not bind the complete R1 prediction inventory")
    expected_caches = {key: facts["sha256"] for key, facts in manifest["cells"].items()}
    if (readout_hashes.get("probability_caches") != expected_caches or
            readout_hashes.get("source_files") != result.get("source_hashes")):
        raise ValueError("Independent R1 readout does not bind all native caches and runner sources")
    closure = source_closure()
    _check_closure(closure)
    facts = {p.name: {"path": str(p.relative_to(ROOT)), "sha256": sha256(p)}
             for p in (R1_MANIFEST, R1_DECLARATION, R1_PREFLIGHT, R1_RESULT, R1_READOUT)}
    facts["native_manifest.json"] = {"path": native_ref["path"], "sha256": native_ref["sha256"]}
    manifest = {"schema": "frontier_postprocess_round2_manifest", "study_id": STUDY,
        "source_hashes": closure, "r1_artifacts": facts,
        "source_status": SOURCE_STATUS,
        "evidence_scope": EVIDENCE_SCOPE,
        "r1_evaluation_manifest_sha256": sha256(R1_MANIFEST),
        "cohort": {"cell_count": 35, "cells": sorted(r1.EXPECTED_CELLS)},
        "composition": {"order": [GATE_ARM, BOUNDARY_ARM], "gate_lambda": 1.0,
            "boundary_strength": "shift_0.8", "arm_id": COMBINED_ARM,
            "probability_source": "equal-seed frontier H/A probabilities padded with zero D"},
        "guards": GUARDS, "ranking": RANKING,
        "organizer_reference_access": False, "promotion": False,
        "candidate_scoring_performed_at_freeze": False}
    declaration = {**manifest, "status": "frozen_before_candidate_scoring",
        "evaluation_manifest_sha256": None}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    declaration_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    declaration["evaluation_manifest_sha256"] = sha256(manifest_path)
    declaration_path.write_text(json.dumps(declaration, indent=2, sort_keys=True) + "\n")
    return manifest, declaration


def _validate_frozen(manifest_path: Path, declaration_path: Path) -> tuple[dict, dict]:
    manifest, declaration = json.loads(manifest_path.read_text()), json.loads(declaration_path.read_text())
    expected_composition = {"order": [GATE_ARM, BOUNDARY_ARM], "gate_lambda": 1.0,
        "boundary_strength": "shift_0.8", "arm_id": COMBINED_ARM,
        "probability_source": "equal-seed frontier H/A probabilities padded with zero D"}
    protocol_keys = ("schema", "study_id", "source_hashes", "r1_artifacts", "cohort",
                     "composition", "guards", "ranking", "organizer_reference_access", "promotion",
                     "candidate_scoring_performed_at_freeze", "r1_evaluation_manifest_sha256",
                     "source_status", "evidence_scope")
    if (manifest.get("schema") != "frontier_postprocess_round2_manifest" or
            manifest.get("study_id") != STUDY or declaration.get("status") != "frozen_before_candidate_scoring" or
            declaration.get("evaluation_manifest_sha256") != sha256(manifest_path) or
            any(declaration.get(k) != manifest.get(k) for k in protocol_keys) or
            manifest.get("composition") != expected_composition or
            manifest.get("guards") != GUARDS or manifest.get("ranking") != RANKING or
            manifest.get("cohort") != {"cell_count": 35, "cells": sorted(r1.EXPECTED_CELLS)} or
            manifest.get("organizer_reference_access") is not False or manifest.get("promotion") is not False or
            manifest.get("candidate_scoring_performed_at_freeze") is not False):
        raise ValueError("Round-two freeze receipt does not match this protocol")
    _check_closure(manifest["source_hashes"])
    if (manifest.get("source_status") != SOURCE_STATUS or
            manifest.get("evidence_scope") != EVIDENCE_SCOPE or
            manifest.get("source_hashes") != source_closure()):
        raise ValueError("Round-two source/evidence scope differs from the frozen R1 evidence")
    for fact in manifest["r1_artifacts"].values():
        path = (ROOT / fact["path"]).resolve()
        if not path.is_file() or sha256(path) != fact["sha256"]:
            raise ValueError("A bound R1 artifact changed")
    return manifest, declaration


def compose(gated_labels: np.ndarray, equal_seed_ha: np.ndarray, model_period: np.ndarray,
            timestamps: np.ndarray, fps: float) -> np.ndarray:
    """Apply the frozen boundary operation to gate-decoded labels."""
    p = r1.validate_conditional_ha(equal_seed_ha)
    padded = np.column_stack((p, np.zeros(len(p), dtype=float)))
    return boundary.relocate(padded, gated_labels, model_period, timestamps, fps, "shift_0.8")


def _run_order_and_count(labels: np.ndarray) -> tuple[str, ...]:
    y = np.asarray(labels).astype("U1", copy=False)
    return tuple(y[np.r_[True, y[1:] != y[:-1]]].tolist())


def _verify_all_controls(keys, verifier) -> tuple:
    """Return only after every baseline/component control has replayed successfully."""
    ordered = tuple(keys)
    for key in ordered:
        verifier(key)
    return ordered


def run(manifest_path: Path, declaration_path: Path, prediction_dir: Path) -> dict:
    manifest, _ = _validate_frozen(manifest_path, declaration_path)
    prediction_dir = prediction_dir.resolve()
    if not prediction_dir.is_relative_to(ROOT / "data/experiments"):
        raise ValueError("Round-two prediction artifacts must stay under data/experiments")
    inp = _r1_inputs()
    loaded = r1.load_complete(inp["manifest"])
    # Re-run baseline-only preflight and demand byte-for-byte receipt parity before candidates.
    replay = r1.preflight(R1_MANIFEST, R1_DECLARATION)
    if replay != inp["preflight"]:
        raise ValueError("R1 baseline control replay differs from its frozen preflight")
    result = inp["result"]
    if prediction_dir.exists() and any(prediction_dir.iterdir()):
        raise FileExistsError("Round-two prediction directory must be empty")
    prediction_dir.mkdir(parents=True, exist_ok=True)
    rows, predictions, gate_predictions, boundary_predictions = {}, {}, {}, {}
    gate_rows, boundary_rows, equal_probabilities = {}, {}, {}
    def verify_control(key):
        cell = loaded[key]
        cached = result["prediction_inventory"][key]
        cache_path = (ROOT / cached["path"]).resolve()
        if not cache_path.is_file() or sha256(cache_path) != cached["sha256"]:
            raise ValueError(f"R1 candidate prediction changed: {key}")
        with np.load(cache_path, allow_pickle=False) as saved:
            gate_saved = saved["pred__disagreement_lambda_1.0"].astype("U1", copy=True)
            boundary_saved = saved["pred__boundary_shift_0.8"].astype("U1", copy=True)
            baseline_saved = saved["baseline_prediction"].astype("U1", copy=True)
        if not np.array_equal(baseline_saved, cell["baseline_prediction"]):
            raise ValueError(f"R1 saved control prediction differs from native replay: {key}")
        equal_ha = r1.probability_for_weight(cell, .5)
        pooled = r1.family_probabilities(cell, .5)
        experts = {"ce": pooled["ce"], "calibrated": pooled["calibrated"],
                   "adopted": pooled["predecessor"]}
        gate = r1.disagreement.decode(experts, cell["decoded_predecessor"],
            cell["model_period"], cell["fps"], 1.0)
        gate_row = r1._score_cell(cell, gate)
        r1._assert_metric_row(gate_row,
            result["disagreement"]["cell_metrics"][GATE_ARM][key], f"{key}/{GATE_ARM}")
        if not np.array_equal(gate, gate_saved):
            raise ValueError(f"R1 gate prediction replay differs: {key}")
        boundary_from_control = compose(cell["baseline_prediction"], equal_ha,
            cell["model_period"], cell["timestamps"], cell["fps"])
        boundary_row = r1._score_cell(cell, boundary_from_control)
        r1._assert_metric_row(boundary_row,
            result["boundary"]["cell_metrics"]["shift_0.8"][key], f"{key}/{BOUNDARY_ARM}")
        if not np.array_equal(boundary_from_control, boundary_saved):
            raise ValueError(f"R1 boundary prediction replay differs: {key}")
        gate_predictions[key], boundary_predictions[key] = gate, boundary_from_control
        gate_rows[key], boundary_rows[key] = gate_row, boundary_row
        equal_probabilities[key] = equal_ha
    verified_cells = _verify_all_controls(sorted(r1.EXPECTED_CELLS), verify_control)
    # Score composition only after all 35 control and standalone candidate replays pass.
    for key in verified_cells:
        cell = loaded[key]
        gate = gate_predictions[key]
        equal_ha = equal_probabilities[key]
        combined = compose(gate, equal_ha, cell["model_period"], cell["timestamps"], cell["fps"])
        if _run_order_and_count(combined) != _run_order_and_count(gate):
            raise RuntimeError(f"Boundary composition changed gated run order/count: {key}")
        combined_row = r1._score_cell(cell, combined)
        r1._assert_population(combined_row, replay["baseline_rows"][key], key)
        r1._assert_fixed_masks(combined, cell["baseline_prediction"], key)
        rows[key], predictions[key] = combined_row, combined
    aggregate = balanced.aggregate_cell_metrics(rows)
    gate_aggregate = balanced.aggregate_cell_metrics(gate_rows)
    boundary_aggregate = balanced.aggregate_cell_metrics(boundary_rows)
    inventory = {}
    for key in sorted(r1.EXPECTED_CELLS):
        path = prediction_dir / (key.replace("|", "__") + ".npz")
        np.savez_compressed(path, prediction=predictions[key],
            baseline_prediction=loaded[key]["baseline_prediction"],
            gate_prediction=gate_predictions[key],
            boundary_control_prediction=boundary_predictions[key])
        inventory[key] = {"path": str(path.resolve().relative_to(ROOT)), "sha256": sha256(path)}
    return {"study_id": STUDY, "manifest_sha256": sha256(manifest_path),
        "declaration_sha256": sha256(declaration_path), "source_hashes": manifest["source_hashes"],
        "source_status": SOURCE_STATUS, "evidence_scope": EVIDENCE_SCOPE,
        "r1_result_sha256": sha256(R1_RESULT), "r1_readout_sha256": sha256(R1_READOUT),
        "arm_id": COMBINED_ARM, "composition": manifest["composition"],
        "baseline_replay": {"passed": True, "preflight_sha256": sha256(R1_PREFLIGHT)},
        "baseline_rows": replay["baseline_rows"], "cell_metrics": rows,
        "aggregate": aggregate, "component_metrics": {
            "gate": {"cell_metrics": gate_rows, "aggregate": gate_aggregate},
            "boundary_control": {"cell_metrics": boundary_rows,
                                 "aggregate": boundary_aggregate}},
        "prediction_inventory": inventory,
        "official_metrics": list(r1.METRICS), "organizer_reference_access": False,
        "promotion": False}


def _read_r1_prediction(result: Mapping, key: str, arm: str) -> np.ndarray:
    fact = result["prediction_inventory"][key]
    path = (ROOT / fact["path"]).resolve()
    with np.load(path, allow_pickle=False) as saved:
        name = {GATE_ARM: "pred__disagreement_lambda_1.0",
                BOUNDARY_ARM: "pred__boundary_shift_0.8"}[arm]
        return saved[name].astype("U1", copy=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions-dir", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.manifest, args.declaration)
        return
    if args.output is None or args.predictions_dir is None:
        parser.error("run requires --output and --predictions-dir")
    result = run(args.manifest, args.declaration, args.predictions_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
