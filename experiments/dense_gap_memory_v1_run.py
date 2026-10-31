"""Matched inference/evaluation runner for dense_gap_memory_v1.

The control producer supplies one cell with native per-seed probabilities and
the same cached inputs/labels used to score those controls. This runner checks
the reconstructed control rows against that producer's saved rows before it
scores the candidate. It never fits a model or reads organizer labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import soccernet_capacity as capacity
from experiments import supervised
from experiments.dense_gap_memory_v1 import apply_memory, ARMS


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/dense_gap_memory_v1"
SEEDS = (2026, 2027)


def _array_sha(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_producer_provenance(receipt_path: Path, prediction_root: Path,
                               member_scores_path: Path, ensemble_scores_path: Path,
                               expected_arm: str = "control_D2") -> str:
    """Preflight the producer's sealed fold/file manifest once before replay."""
    receipt_path, prediction_root = Path(receipt_path), Path(prediction_root)
    receipt_sha = _file_sha(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    expected_folds = {source: "leave_out_" + source
                      for source in ("sportec", "metrica", "skillcorner")}
    if receipt.get("source_study_id") != "dense_label_weight_v1":
        raise ValueError("Producer provenance names the wrong source study")
    if receipt.get("source_arm") != expected_arm or receipt.get("passed") is not True:
        raise ValueError("Producer provenance did not pass for the declared control arm")
    if not receipt.get("reconstruction_status"):
        raise ValueError("Producer provenance lacks reconstruction status")
    if receipt.get("fold_mapping") != expected_folds:
        raise ValueError("Producer provenance has an unexpected provider-fold mapping")
    if receipt.get("member_count") != 30 or receipt.get("ensemble_count") != 15:
        raise ValueError("Producer provenance does not cover the 30+15 control artifacts")
    if (_file_sha(member_scores_path) != receipt.get("member_scores_sha256") or
            _file_sha(ensemble_scores_path) != receipt.get("ensemble_scores_sha256")):
        raise ValueError("Producer control score files differ from the provenance receipt")
    entries = receipt.get("files")
    if not isinstance(entries, list) or len(entries) != 45:
        raise ValueError("Producer provenance must enumerate 45 prediction files")
    seen = set()
    member_counts = {"2026": 0, "2027": 0, "ensemble": 0}
    for entry in entries:
        rel = Path(entry["path"])
        path = rel if rel.is_absolute() else ROOT / rel
        try:
            path.resolve().relative_to(prediction_root.resolve())
        except ValueError as exc:
            raise ValueError("Provenance prediction path escapes the declared root") from exc
        if not path.is_file() or _file_sha(path) != entry.get("sha256"):
            raise ValueError(f"Producer prediction file missing or changed: {path}")
        fold = entry.get("fold")
        member = str(entry.get("member"))
        if fold not in expected_folds.values():
            raise ValueError(f"Unexpected provider fold in provenance: {fold}")
        if member not in ("2026", "2027", "ensemble"):
            raise ValueError(f"Unexpected ensemble member in provenance: {member}")
        if expected_arm not in path.parts:
            raise ValueError("Prediction path does not identify the declared control arm")
        if path in seen:
            raise ValueError(f"Duplicate prediction file in provenance: {path}")
        seen.add(path)
        member_counts[member] += 1
    if member_counts != {"2026": 15, "2027": 15, "ensemble": 15}:
        raise ValueError(f"Unexpected control prediction member counts: {member_counts}")
    for fold in expected_folds.values():
        fold_entries = [entry for entry in entries if entry.get("fold") == fold]
        if not fold_entries:
            raise ValueError(f"No provenance prediction files for {fold}")
    return receipt_sha


def _metric_parity(actual: dict, saved: dict, atol: float = 1e-12) -> None:
    """Recheck all common official metrics and IoU rows before candidate scoring."""
    capacity._metric_parity(actual, saved)


def replay_control_cell(*, item: dict, view: str, replicate: str, inputs: dict,
                        labels: np.ndarray, probabilities: dict[int, np.ndarray],
                        saved_control_rows: dict[int, dict],
                        saved_control_predictions: dict[int, np.ndarray]) -> dict:
    """Validate one reconstructed control cell without scoring a candidate."""
    labels = np.asarray(labels)
    if set(probabilities) != set(SEEDS) or set(saved_control_rows) != {2026, 2027, "ensemble"}:
        raise ValueError("Both control seeds and ensemble evidence are required")
    if set(saved_control_predictions) != {2026, 2027, "ensemble"}:
        raise ValueError("Saved control predictions for both seeds and ensemble are required")
    period = supervised.model_periods(inputs)
    native = {}
    rows = {}
    for seed in SEEDS:
        p = np.asarray(probabilities[seed])
        if p.shape != (len(labels), 3):
            raise ValueError(f"Seed {seed} control probabilities must be native H/A/D")
        if (not np.issubdtype(p.dtype, np.number) or not np.isfinite(p).all() or
                (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5)):
            raise ValueError(f"Seed {seed} control probabilities are invalid")
        saved_probability_hash = saved_control_rows[seed].get(
            "probabilities_sha256", saved_control_rows[seed].get("probability_sha256"))
        if saved_probability_hash is not None and saved_probability_hash != _array_sha(p):
            raise ValueError(f"Seed {seed} control probability hash replay failed")
        row, prediction = capacity._score_probability(
            p, inputs, labels, item, view, replicate, "baseline", str(seed))
        _metric_parity(row, saved_control_rows[seed])
        if not np.array_equal(prediction, saved_control_predictions[seed]):
            raise ValueError(f"Seed {seed} reconstructed control prediction replay failed")
        native[seed], rows[seed] = p, row
    native["ensemble"] = .5 * (native[2026] + native[2027])
    saved_ensemble_hash = saved_control_rows["ensemble"].get(
        "probabilities_sha256", saved_control_rows["ensemble"].get("probability_sha256"))
    if saved_ensemble_hash is not None and saved_ensemble_hash != _array_sha(native["ensemble"]):
        raise ValueError("Reconstructed ensemble control probability hash replay failed")
    row, prediction = capacity._score_probability(
        native["ensemble"], inputs, labels, item, view, replicate,
        "baseline", "ensemble")
    _metric_parity(row, saved_control_rows["ensemble"])
    if not np.array_equal(prediction, saved_control_predictions["ensemble"]):
        raise ValueError("Reconstructed ensemble control prediction replay failed")
    rows["ensemble"] = row
    return {
        "control_replay_passed": True,
        "source": item["source"],
        "match": item["match"],
        "fold": item.get("fold", "leave_out_" + item["source"]),
        "view": view,
        "replicate": replicate,
        "probability_sha256": {str(seed): _array_sha(native[seed]) for seed in SEEDS} | {
            "ensemble": _array_sha(native["ensemble"])},
        "prediction_sha256": {str(seed): _array_sha(saved_control_predictions[seed])
                              for seed in SEEDS} | {
            "ensemble": _array_sha(saved_control_predictions["ensemble"])},
        "input_sha256": {
            "labels": _array_sha(np.asarray(labels)),
            "period": _array_sha(np.asarray(inputs["period"])),
            "timestamps": _array_sha(np.asarray(inputs["timestamps"])),
            "ball_observed": _array_sha(np.asarray(inputs["ball_observed"])),
            "fps": _array_sha(np.asarray(inputs["fps"])),
        },
        "rows": {str(key): value for key, value in rows.items()},
    }


def replay_cell_path(source: str, match: str, view: str, replicate: str) -> Path:
    return OUT / "control_replay" / f"{source}_{match}_{view}_{replicate}.json"


def save_replay_cell(result: dict) -> Path:
    path = replay_cell_path(result["source"], result["match"],
                            result["view"], result["replicate"])
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(capacity._jsonable(result), indent=2, sort_keys=True,
                         allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite different control replay receipt: {path}")
    else:
        path.write_text(payload)
    return path


def verify_replay_inputs(cell: dict, receipt: dict) -> None:
    """Check restored controls and metadata against a completed replay gate."""
    item = cell["item"]
    expected_identity = (item["source"], item["match"], cell["view"], cell["replicate"])
    actual_identity = (receipt["source"], receipt["match"], receipt["view"], receipt["replicate"])
    if not receipt.get("control_replay_passed") or actual_identity != expected_identity:
        raise ValueError("Saved control replay receipt identity is invalid")
    hashes = {str(seed): _array_sha(np.asarray(cell["probabilities"][seed]))
              for seed in SEEDS}
    hashes["ensemble"] = _array_sha(.5 * (np.asarray(cell["probabilities"][2026]) +
                                           np.asarray(cell["probabilities"][2027])))
    if hashes != receipt["probability_sha256"]:
        raise ValueError("Control probabilities changed after the replay gate")
    inputs, labels = cell["inputs"], np.asarray(cell["labels"])
    input_hashes = {
        "labels": _array_sha(labels),
        "period": _array_sha(np.asarray(inputs["period"])),
        "timestamps": _array_sha(np.asarray(inputs["timestamps"])),
        "ball_observed": _array_sha(np.asarray(inputs["ball_observed"])),
        "fps": _array_sha(np.asarray(inputs["fps"])),
    }
    if input_hashes != receipt["input_sha256"]:
        raise ValueError("Holdout inputs changed after the control replay gate")
    prediction_hashes = {str(seed): _array_sha(np.asarray(cell["saved_control_predictions"][seed]))
                         for seed in SEEDS}
    prediction_hashes["ensemble"] = _array_sha(
        np.asarray(cell["saved_control_predictions"]["ensemble"]))
    if prediction_hashes != receipt["prediction_sha256"]:
        raise ValueError("Saved control predictions changed after the replay gate")


def score_cell(*, item: dict, view: str, replicate: str, inputs: dict,
               labels: np.ndarray, probabilities: dict[int, np.ndarray],
               saved_control_rows: dict[int, dict],
               saved_control_predictions: dict[int, np.ndarray],
               validated_control_replay: dict | None = None) -> dict:
    """Replay controls, then score memory and ensemble on a single external cell.

    `probabilities` are native H/A/D outputs from the separately audited
    reconstructed controls. `saved_control_rows` and predictions come from that
    producer's control evaluation and are required for exact replay parity.
    The evaluation metadata must identify the held-out provider and its match.
    """
    source, match = item["source"], item["match"]
    if not match or "leave_out_" + source not in str(item.get("fold", "leave_out_" + source)):
        raise ValueError("Cell is not identified as a full-provider holdout")
    if view not in ("clean", "feed_like"):
        raise ValueError("Unexpected evaluation view")
    if view == "clean" and replicate != "clean":
        raise ValueError("Clean cell replicate must be named clean")
    if view == "feed_like" and replicate == "clean":
        raise ValueError("Feed-like cell requires a corruption replicate")
    labels = np.asarray(labels)
    observed = np.asarray(inputs["ball_observed"])
    if observed.dtype.kind != "b":
        raise ValueError("Native ball_observed input must be a boolean mask")
    timestamps = np.asarray(inputs["timestamps"], dtype=np.float64)
    if any(x.shape != (len(labels),) for x in (observed, timestamps)):
        raise ValueError("Native inference metadata does not match labels")
    if validated_control_replay is None or not validated_control_replay.get("control_replay_passed"):
        raise ValueError("Candidate scoring requires a completed control replay receipt")
    replay_identity = (validated_control_replay["source"], validated_control_replay["match"],
                       validated_control_replay["view"], validated_control_replay["replicate"])
    if replay_identity != (source, match, view, replicate):
        raise ValueError("Control replay receipt identity does not match candidate cell")
    native = {seed: np.asarray(probabilities[seed]) for seed in SEEDS}
    native["ensemble"] = .5 * (native[2026] + native[2027])
    hashes = {str(seed): _array_sha(native[seed]) for seed in SEEDS}
    hashes["ensemble"] = _array_sha(native["ensemble"])
    if hashes != validated_control_replay["probability_sha256"]:
        raise ValueError("Control probabilities changed after the replay gate")
    baseline_rows = {2026: validated_control_replay["rows"]["2026"],
                     2027: validated_control_replay["rows"]["2027"],
                     "ensemble": validated_control_replay["rows"]["ensemble"]}
    period = supervised.model_periods(inputs)

    # Apply independently to each model, then average their adjusted probabilities.
    adjusted = {seed: apply_memory(native[seed], observed, timestamps, period)
                for seed in SEEDS}
    adjusted["ensemble"] = .5 * (adjusted[2026] + adjusted[2027])

    rows = []
    for member in ARMS:
        seed_key = {"seed_2026": 2026, "seed_2027": 2027,
                    "equal_probability_ensemble": "ensemble"}[member]
        p = adjusted[seed_key]
        row, _ = capacity._score_probability(
            p, inputs, labels, item, view, replicate, "dense_gap_memory_v1", member)
        baseline_member = (2026 if member == "seed_2026" else
                           2027 if member == "seed_2027" else "ensemble")
        for metric in ("spell_f1", "macro_f1", "d_f1"):
            row[f"baseline_{metric}"] = baseline_rows[baseline_member][metric]
        row["probability_sha256"] = _array_sha(p)
        rows.append(row)
    return {
        "study": "dense_gap_memory_v1",
        "source": source,
        "match": match,
        "fold": item.get("fold", "leave_out_" + source),
        "view": view,
        "replicate": replicate,
        "control_replay_passed": True,
        "control_probability_sha256": {str(seed): _array_sha(native[seed]) for seed in SEEDS},
        "rows": rows,
    }


def save_cell(result: dict) -> Path:
    """Write one aggregate cell receipt immutably; no frame arrays are saved."""
    OUT.mkdir(parents=True, exist_ok=True)
    name = "_".join((result["source"], result["match"], result["view"], result["replicate"]))
    path = OUT / f"{name}.json"
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite different sealed cell result: {path}")
    else:
        path.write_text(payload)
    return path


def summarize_cells(paths) -> list[dict]:
    """Equal-match, then equal-provider summaries with paired control deltas."""
    cells = [json.loads(Path(path).read_text()) for path in paths]
    summaries = []
    for view in ("clean", "feed_like"):
        for member in ARMS:
            per_provider = {}
            for cell in cells:
                if cell["view"] != view:
                    continue
                row = next(r for r in cell["rows"] if r["member"] == member)
                per_provider.setdefault(cell["source"], {}).setdefault(cell["match"], []).append(row)
            providers = {}
            for source, matches in per_provider.items():
                metric_values = {}
                for metric in ("spell_f1", "macro_f1", "d_f1"):
                    per_match = [float(np.mean([r[metric] for r in replicates]))
                                 for replicates in matches.values()]
                    per_match_baseline = [float(np.mean([r[f"baseline_{metric}"] for r in replicates]))
                                          for replicates in matches.values()]
                    metric_values[metric] = float(np.mean(per_match))
                    metric_values[f"baseline_{metric}"] = float(np.mean(per_match_baseline))
                providers[source] = metric_values
            if not providers:
                raise ValueError(f"No scored rows for {view} {member}")
            mean = {key: float(np.mean([value[key] for value in providers.values()]))
                    for key in next(iter(providers.values()))}
            for metric in ("spell_f1", "macro_f1", "d_f1"):
                mean[f"delta_{metric}"] = mean[metric] - mean[f"baseline_{metric}"]
            summaries.append({"view": view, "member": member,
                              "equal_provider": mean, "per_provider": providers})
    return summaries


def run_study(cell_factory, producer_provenance_sha256: str) -> dict:
    """Score the complete declared five-match/three-view cohort from cell data.

    `cell_factory` must be a zero-argument callable returning a fresh iterator.
    Each entry is a dict with `item`, `view`, `replicate`, `inputs`,
    `labels`, `probabilities`, `saved_control_rows`, and
    `saved_control_predictions`, as consumed by `score_cell`.
    """
    expected = {(item["source"], item["match"], view, replicate)
                for item in capacity.old.selection_items()
                for view, replicate in capacity.VIEWS}
    if not callable(cell_factory):
        raise TypeError("cell_factory must return a fresh iterator for each pass")
    if len(producer_provenance_sha256) != 64:
        raise ValueError("A verified producer provenance SHA-256 is required")
    gate_path = OUT / "control_replay_gate.json"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        if gate.get("study") != "dense_gap_memory_v1" or gate.get("status") != "PASS":
            raise ValueError("Existing control replay gate is invalid")
        if gate.get("producer_provenance_sha256") != producer_provenance_sha256:
            raise ValueError("Producer provenance changed after the control replay gate")
        replayed = set()
        replay_paths = []
        for cell in cell_factory():
            item = cell["item"]
            key = (item["source"], item["match"], cell["view"], cell["replicate"])
            if key not in expected or key in replayed:
                raise ValueError(f"Unexpected or duplicate holdout control cell: {key}")
            receipt_path = replay_cell_path(*key)
            rel = str(receipt_path.relative_to(ROOT))
            if gate.get("control_receipts", {}).get(rel) != _file_sha(receipt_path):
                raise ValueError(f"Control receipt missing or changed: {receipt_path}")
            verify_replay_inputs(cell, json.loads(receipt_path.read_text()))
            replayed.add(key); replay_paths.append(receipt_path)
        if (replayed != expected or gate.get("cell_count") != len(expected) or
                gate.get("member_rows") != 3 * len(expected)):
            raise ValueError("Existing control replay gate does not cover the exact cohort")
    else:
        # First pass validates and saves the entire control cohort. No candidate
        # is scored until every seed, ensemble, prediction and metric replays.
        replayed = set()
        replay_paths = []
        for cell in cell_factory():
            item = cell["item"]
            key = (item["source"], item["match"], cell["view"], cell["replicate"])
            if key not in expected or key in replayed:
                raise ValueError(f"Unexpected or duplicate holdout control cell: {key}")
            replay = replay_control_cell(**cell)
            replay_paths.append(save_replay_cell(replay))
            replayed.add(key)
        if replayed != expected:
            raise ValueError(f"Incomplete control replay; missing {len(expected - replayed)} cells")
        gate = {
            "study": "dense_gap_memory_v1",
            "status": "PASS",
            "cell_count": len(replay_paths),
            "member_rows": len(replay_paths) * 3,
            "producer_provenance_sha256": producer_provenance_sha256,
            "control_receipts": {str(p.relative_to(ROOT)): _file_sha(p)
                                 for p in sorted(replay_paths)},
        }
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_payload = json.dumps(gate, indent=2, sort_keys=True, allow_nan=False) + "\n"
        gate_path.write_text(gate_payload)

    # Second pass applies the candidate using sealed control receipts, without
    # scoring controls a second time.
    seen = set()
    files = []
    for cell in cell_factory():
        item = cell["item"]
        receipt_path = replay_cell_path(item["source"], item["match"],
                                        cell["view"], cell["replicate"])
        expected_sha = gate["control_receipts"].get(str(receipt_path.relative_to(ROOT)))
        if not expected_sha or _file_sha(receipt_path) != expected_sha:
            raise ValueError(f"Missing or changed control replay receipt: {receipt_path}")
        validated_replay = json.loads(receipt_path.read_text())
        verify_replay_inputs(cell, validated_replay)
        result = score_cell(**cell, validated_control_replay=validated_replay)
        key = (result["source"], result["match"], result["view"], result["replicate"])
        if key not in expected or key in seen:
            raise ValueError(f"Unexpected or duplicate holdout cell: {key}")
        seen.add(key)
        files.append(save_cell(result))
    if seen != expected:
        raise ValueError(f"Incomplete candidate cohort; missing {len(expected - seen)} cells")
    aggregate = {
        "study": "dense_gap_memory_v1",
        "cell_count": len(files),
        "controls_replayed": True,
        "producer_provenance_sha256": producer_provenance_sha256,
        "organizer_labels_read": False,
        "summaries": summarize_cells(files),
        "cells": [str(p.relative_to(ROOT)) for p in files],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "aggregate.json"
    payload = json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if report_path.exists() and report_path.read_text() != payload:
        raise ValueError(f"Refusing to overwrite different aggregate: {report_path}")
    if not report_path.exists():
        report_path.write_text(payload)
    return aggregate


def _scalar(archive, key: str):
    if key not in archive.files:
        raise ValueError(f"Control prediction is missing metadata field {key}")
    value = archive[key]
    if value.ndim == 0:
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    raise ValueError(f"Metadata field {key} must be scalar")


def load_control_cells(prediction_root: Path, scores_path: Path,
                       ensemble_scores_path: Path | None = None,
                       arm: str = "control_D2"):
    """Load and validate control NPZ cells from dense_label_weight_v1 outputs.

    Expected files follow the control producer's declared schema:
    `{arm}/{fold}/seed{seed}__{match}__{view}__{replicate}.npz` with native
    probabilities, decoded prediction, labels, and inference metadata.
    """
    prediction_root, scores_path = Path(prediction_root), Path(scores_path)
    report_paths = [scores_path] + ([ensemble_scores_path] if ensemble_scores_path else [])
    rows = []
    for score_path in report_paths:
        report = json.loads(Path(score_path).read_text())
        rows.extend(report.get("rows", report.get("scores", [])))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Control score report has no rows")
    score_index = {}
    for row in rows:
        key = (row.get("source", row.get("provider")), row.get("match"),
               row.get("view"), row.get("replicate"),
               row.get("arm", row.get("recipe")), str(row.get("member", row.get("seed"))))
        if key in score_index:
            raise ValueError(f"Duplicate control score row: {key}")
        score_index[key] = row

    def cells():
        for item in capacity.old.selection_items():
            fold = "leave_out_" + item["source"]
            for view, replicate in capacity.VIEWS:
                probabilities, inputs, labels = {}, None, None
                predictions = {}
                for seed in SEEDS:
                    path = prediction_root / arm / fold / (
                        f"seed{seed}__{item['match']}__{view}__{replicate}.npz")
                    with np.load(path, allow_pickle=False) as archive:
                        identity = {
                            "source": str(_scalar(archive, "source")),
                            "match": str(_scalar(archive, "match")),
                            "view": str(_scalar(archive, "view")),
                            "replicate": str(_scalar(archive, "replicate")),
                            "seed": int(_scalar(archive, "seed")),
                            "arm": str(_scalar(archive, "arm")),
                        }
                        if (identity["source"], identity["match"], identity["view"],
                                identity["replicate"], identity["seed"], identity["arm"]) != (
                                item["source"], item["match"], view, replicate, seed, arm):
                            raise ValueError(f"Control identity mismatch in {path}")
                        current_labels = archive["labels"].astype(str)
                        current_inputs = {key: archive[key] for key in
                                          ("period", "fps", "timestamps", "ball_observed")}
                        if "model_period" in archive.files and not np.array_equal(
                                archive["model_period"], supervised.model_periods(current_inputs)):
                            raise ValueError(f"Saved model period differs from original inputs in {path}")
                        p, prediction = archive["probabilities"], archive["prediction"]
                    if labels is not None and not np.array_equal(labels, current_labels):
                        raise ValueError("Control seed label arrays differ")
                    if inputs is not None:
                        for name in current_inputs:
                            if not np.array_equal(inputs[name], current_inputs[name]):
                                raise ValueError(f"Control seed input metadata differs: {name}")
                    labels, inputs = current_labels, current_inputs
                    probabilities[seed], predictions[seed] = p, prediction
                ensemble_probability = .5 * (probabilities[2026] + probabilities[2027])
                predictions["ensemble"] = supervised.decode(
                    ensemble_probability, supervised.model_periods(inputs),
                    float(inputs["fps"]), capacity.lt.COST)
                control_rows = {}
                for member in ("2026", "2027", "ensemble"):
                    key = (item["source"], item["match"], view, replicate,
                           arm, member)
                    if key not in score_index:
                        raise ValueError(f"Missing control metric row: {key}")
                    control_rows[2026 if member == "2026" else
                                 2027 if member == "2027" else "ensemble"] = score_index[key]
                    saved_hash = score_index[key].get("prediction_sha256")
                    if saved_hash is not None and saved_hash != _array_sha(predictions[
                            2026 if member == "2026" else
                            2027 if member == "2027" else "ensemble"]):
                        raise ValueError(f"Control decoded prediction hash mismatch: {key}")
                yield dict(item=dict(item, fold=fold), view=view, replicate=replicate,
                    inputs=inputs, labels=labels, probabilities=probabilities,
                    saved_control_rows=control_rows,
                    saved_control_predictions=predictions)
    return cells()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", type=Path, required=True,
        help="dense_label_weight_v1/predictions directory")
    parser.add_argument("--member-scores", type=Path, required=True,
        help="saved control member score rows")
    parser.add_argument("--ensemble-scores", type=Path, required=True,
        help="saved control ensemble score rows")
    parser.add_argument("--provenance", type=Path, required=True,
        help="sealed dense_label_weight_v1 control prediction provenance receipt")
    parser.add_argument("--arm", default="control_D2",
        help="reconstructed current dense control arm")
    args = parser.parse_args(argv)
    provenance_sha = verify_producer_provenance(args.provenance,
        args.prediction_root, args.member_scores, args.ensemble_scores,
        expected_arm=args.arm)
    aggregate = run_study(lambda: load_control_cells(args.prediction_root,
        args.member_scores, args.ensemble_scores, arm=args.arm), provenance_sha)
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
