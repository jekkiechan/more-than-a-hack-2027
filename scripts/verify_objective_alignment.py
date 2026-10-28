"""Independently verify the sealed objective-alignment study artifacts.

This verifier is read-only. Run it after all 18 new checkpoints are sealed and
all declared evaluation cells are complete. It audits prepared training records
against raw labels and sealed controls, checks model/budget parity, validates
saved predictions and control references, and recomputes aggregation/screens
without calling the study runner's aggregation function.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import (corruption_experiment as ce,
                         dead_transfer as dt,
                         factorized_had,
                         gru_robustness_experiment as gr,
                         learned_temporal as lt,
                         multisource_eval as ms,
                         objective_alignment as oa,
                         objective_alignment_prep as prep,
                         objective_losses,
                         supervised,
                         metrics)
from experiments.eval_set.protocol import sha256


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _same(left, right, where: str) -> None:
    """Recursive equality with exact NaN-aware comparison for feature arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        _require(a.shape == b.shape, f"Shape mismatch at {where}: {a.shape} != {b.shape}")
        if a.dtype.kind in "fc" or b.dtype.kind in "fc":
            equal = np.array_equal(a, b, equal_nan=True)
        else:
            equal = np.array_equal(a, b)
        _require(equal, f"Array mismatch at {where}")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        _require(set(left) == set(right), f"Key mismatch at {where}")
        for key in left:
            _same(left[key], right[key], f"{where}.{key}")
        return
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        _require(len(left) == len(right), f"Length mismatch at {where}")
        for index, (a, b) in enumerate(zip(left, right)):
            _same(a, b, f"{where}[{index}]")
        return
    _require(left == right, f"Value mismatch at {where}: {left!r} != {right!r}")


def _paired_records_from_prepared(metadata: dict, match: str) -> list[dict]:
    wrapped = {name: dict(joint=record) for name, record in metadata["records"].items()}
    return gr.paired_records(wrapped, match, "slot_dropout")


def _audit_training_records(out: Path, spec: dict) -> dict:
    prepared_manifest_path = out / "prepared_training_manifest.json"
    _require(prepared_manifest_path.exists(), "Prepared training manifest is missing")
    inventory = json.loads(prepared_manifest_path.read_text())
    _require(inventory.get("preparation_declaration_sha256")
             == sha256(out / "preparation_declaration.json"),
             "Prepared training manifest uses another declaration")
    _require(inventory["count"] == 13, "Expected 13 frozen training caches")
    matches = {path.stem for split in dt.SPLITS.values() for path in split["train"]}
    eval_matches = {path.stem for split in dt.SPLITS.values() for path in split["eval"]}
    _require(len(matches) == 13 and len(eval_matches) == 5 and not matches & eval_matches,
             "Training/evaluation match manifest is incomplete or overlapping")

    inspected_views = 0
    constructed_records = 0
    windows_by_match = {}
    records_by_match = {}
    train_ids = {source: {path.stem for path in split["train"]}
                 for source, split in dt.SPLITS.items()}
    eval_ids = {source: {path.stem for path in split["eval"]}
                for source, split in dt.SPLITS.items()}

    for source, split in dt.SPLITS.items():
        for path in split["train"]:
            cache_entry = inventory["caches"].get(path.stem)
            _require(cache_entry is not None, f"Prepared cache absent from manifest: {path.stem}")
            cache_path = oa.ROOT / cache_entry["cache"]
            receipt_path = oa.ROOT / cache_entry["receipt"]
            _require(cache_path.exists() and cache_path.stat().st_size == cache_entry["bytes"]
                     and sha256(cache_path) == cache_entry["sha256"],
                     f"Prepared training cache changed: {path.stem}")
            _require(receipt_path.exists()
                     and sha256(receipt_path) == cache_entry["receipt_sha256"],
                     f"Prepared training cache receipt changed: {path.stem}")
            metadata = prep.load_training_cache(path.stem, out)
            _require(metadata["match"] == path.stem and metadata["source"] == source,
                     f"Wrong regenerated cache identity: {path.stem}")
            expected_input = spec["data_sha256"][str(path.relative_to(oa.ROOT))]
            _same(metadata["input_sha256"], expected_input, f"{path.stem}.input_sha256")
            inputs, labels = ms.view(path)
            raw_y = np.full(len(labels), -1, dtype=np.int8)
            for label_id, label in enumerate(supervised.LABELS):
                raw_y[labels == label] = label_id
            _same(metadata["full_y"], raw_y, f"{path.stem}.full_resolution_labels")
            _same(metadata["full_period"], supervised.model_periods(inputs),
                  f"{path.stem}.full_resolution_periods")
            _same(metadata["fps"], float(inputs["fps"]), f"{path.stem}.fps")
            _require(path.stem in train_ids[source] and path.stem not in eval_ids[source],
                     f"Unexpected match role: {path.stem}")
            _same(set(metadata["records"]), set(prep.VIEWS), f"{path.stem}.views")

            full_y = np.asarray(metadata["full_y"], dtype=np.int8)
            full_period = np.asarray(metadata["full_period"])
            _require(len(full_y) > 0 and len(full_period) == len(full_y),
                     f"Invalid full-resolution training labels: {path.stem}")
            _require(np.isin(full_y, (-1, 0, 1, 2)).all(),
                     f"Unexpected full-resolution labels: {path.stem}")
            for view, base_record in metadata["records"].items():
                _require(base_record["source"] == source and base_record["match"] == path.stem,
                         f"Prepared feature identity differs: {path.stem}/{view}")
                indices = np.asarray(base_record["indices"], dtype=np.int64)
                y = np.asarray(base_record["y"], dtype=np.int8)
                _require(len(indices) == len(y) and len(indices) > 0
                         and np.all(np.diff(indices) > 0)
                         and indices.min() >= 0 and indices.max() < len(full_y),
                         f"Invalid sample-to-native mapping: {path.stem}/{view}")
                _require(base_record["x"].shape[0] == len(y),
                         f"Feature/label row count differs: {path.stem}/{view}")
                _require(np.array_equal(base_record["period"], full_period[indices]),
                         f"Sampled/full time blocks differ: {path.stem}/{view}")
                _same(y, full_y[indices], f"{path.stem}.{view}.sampled_labels")
                inspected_views += 1

            pairs = _paired_records_from_prepared(metadata, path.stem)
            expected_originals = ("clean", "clean", "dropout_legacy", "dropout_legacy")
            expected_alternates = ("gap", "gap", "dropout_gap", "dropout_gap")
            _require(len(pairs) == 4, f"Expected four paired records for {path.stem}")
            match_windows = 0
            for index, paired in enumerate(pairs):
                view_index = index - (index % 2)
                original = metadata["records"][expected_originals[view_index]]
                alternate = metadata["records"][expected_alternates[view_index]]
                team_swap = bool(index % 2)
                expected_x = (supervised.swap_team_features(original["x"], original["names"])
                              if team_swap else original["x"])
                expected_alternate = (
                    supervised.swap_team_features(alternate["x"], alternate["names"])
                    if team_swap else alternate["x"])
                expected_y = np.where((original["y"] < 2) & (original["y"] >= 0),
                                      1-original["y"], original["y"])
                _same(paired["x"], expected_x, f"{path.stem}.paired[{index}].x")
                _same(paired["alternate"], expected_alternate,
                      f"{path.stem}.paired[{index}].alternate")
                _same(paired["y"], expected_y, f"{path.stem}.paired[{index}].y")
                for key in ("period", "indices"):
                    _same(paired[key], original[key], f"{path.stem}.paired[{index}].{key}")
                chunks = list(lt.chunks(paired))
                _require(len(paired["replace_windows"]) == len(chunks),
                         f"Window selection count differs: {path.stem}/{index}")
                expected_rows = np.zeros(len(paired["y"]), dtype=bool)
                for use, (start, size, _full) in zip(paired["replace_windows"], chunks):
                    expected_rows[start:start+size] = bool(use)
                _same(paired["replace_rows"], expected_rows,
                      f"{path.stem}.paired[{index}].replace_rows")
                factors = objective_losses.spell_duration_factors(
                    full_y, full_period, paired["indices"], paired["y"], metadata["fps"])
                _require(np.isfinite(factors).all(),
                         f"Nonfinite train-only spell factors: {path.stem}/{index}")
                match_windows += len(chunks)
                constructed_records += 1
            windows_by_match[path.stem] = match_windows
            records_by_match[path.stem] = len(pairs)
            del metadata, pairs, inputs, labels

    windows_by_fold = {}
    for heldout in dt.SPLITS:
        providers = [source for source in dt.SPLITS if source != heldout]
        _same(spec["training_sources"][heldout], providers,
              f"declaration.training_sources.{heldout}")
        expected_matches = [path.stem for source in providers for path in dt.SPLITS[source]["train"]]
        _same(spec["training_matches"][heldout], expected_matches,
              f"declaration.training_matches.{heldout}")
        _require(not (set(expected_matches) & eval_matches),
                 f"Evaluation match entered training fold {heldout}")
        windows_by_fold[heldout] = dict(
            records=sum(records_by_match[match] for match in expected_matches),
            windows_per_epoch=sum(windows_by_match[match] for match in expected_matches))
    return dict(matches=len(matches), evaluation_matches=len(eval_matches),
                prepared_views_inspected=inspected_views,
                paired_records_constructed=constructed_records,
                windows_by_fold=windows_by_fold)


def _audit_models(out: Path, spec: dict, training_audit: dict) -> dict:
    manifest_path = out / "checkpoints_frozen.json"
    _require(manifest_path.exists(), "Candidate checkpoint manifest is missing")
    manifest = json.loads(manifest_path.read_text())
    expected_files = {
        f"without_{source}_{arm}_{seed}.joblib"
        for source in dt.SPLITS for arm in oa.NEW_ARMS for seed in oa.SEEDS
    }
    rows = manifest.get("new_checkpoints", [])
    _require(len(rows) == 18 and {row["file"] for row in rows} == expected_files,
             "Sealed manifest does not contain exactly 18 declared candidates")
    _require(manifest.get("total_models") == 24 and manifest.get("new_fits") == 18,
             "Unexpected fit budget in checkpoint manifest")
    _same(manifest.get("reused_control_checkpoints"),
          spec["training_control"]["checkpoints"],
          "checkpoint_manifest.reused_control_checkpoints")

    verified = []
    parity = json.loads((out / "training_parity.json").read_text())
    _require(parity.get("passed") is True and parity.get("fitting_performed") is False,
             "Pre-fit training parity receipt is missing or claims a fit")
    parity_rows = {(row["heldout"], row["seed"]): row for row in parity.get("rows", [])}
    _require(len(parity_rows) == len(dt.SPLITS) * len(oa.SEEDS),
             "Pre-fit training parity rows are incomplete or duplicated")
    objective_map = {"spell_sqrt": "spell_sqrt", "transition": "transition",
                     "combined": "combined"}
    declared_controls = spec["training_control"]["checkpoints"]
    frozen_control = json.loads((gr.DEFAULT / "checkpoints_frozen.json").read_text())
    control_rows = {row["file"]: row for row in frozen_control["checkpoints"]}
    for source in dt.SPLITS:
        for seed in oa.SEEDS:
            control_path = gr.checkpoint(gr.DEFAULT, source, "slot_dropout", seed)
            control_receipt = control_path.with_suffix(".json")
            control = ce.load_model(gr.DEFAULT, control_path)
            parity_row = parity_rows.get((source, seed))
            _require(parity_row is not None and all(parity_row["checks"].values()),
                     f"Pre-fit control parity failed: {source}/{seed}")
            _same(parity_row["windows_per_epoch"],
                  training_audit["windows_by_fold"][source]["windows_per_epoch"],
                  f"training_parity/{source}/{seed}/windows_per_epoch")
            for key in ("median", "scale"):
                digest = hashlib.sha256(np.ascontiguousarray(control[key]).tobytes()).hexdigest()
                _same(parity_row["normalization_sha256"][key], digest,
                      f"training_parity/{source}/{seed}/normalization_sha256/{key}")
            row = control_rows.get(control_path.name)
            _require(row is not None and row["sha256"] == sha256(control_path)
                     and row["receipt_sha256"] == sha256(control_receipt),
                     f"Restored control is not sealed: {source}/{seed}")
            declared = declared_controls.get(f"{source}/{seed}")
            _require(declared is not None, f"Control absent from declaration: {source}/{seed}")
            _same((declared["sha256"], declared["receipt_sha256"], declared["parameters"],
                   declared["optimizer_updates"], declared["windows_per_epoch"]),
                  (sha256(control_path), sha256(control_receipt), control["parameters"],
                   control["optimizer_updates"], control["windows_per_epoch"]),
                  f"declared_control/{source}/{seed}")

            for arm in oa.NEW_ARMS:
                path = oa.checkpoint(out, source, arm, seed)
                receipt_path = path.with_suffix(".json")
                manifest_row = next(item for item in rows if item["file"] == path.name)
                _require(manifest_row["sha256"] == sha256(path)
                         and manifest_row["receipt_sha256"] == sha256(receipt_path),
                         f"Candidate checkpoint or receipt changed: {path.name}")
                model = ce.load_model(out, path)
                receipt = json.loads(receipt_path.read_text())
                expected_objective = objective_map[arm]
                expected_loss = objective_losses.OBJECTIVES[expected_objective]
                _same((receipt.get("heldout"), receipt.get("arm"), receipt.get("seed"),
                       receipt.get("objective")),
                      (source, arm, seed, expected_objective), f"{path.name}.receipt")
                _same(receipt.get("sources"), spec["training_sources"][source],
                      f"{path.name}.receipt.sources")
                _same(receipt.get("training_matches"), spec["training_matches"][source],
                      f"{path.name}.receipt.training_matches")
                _same((model.get("kind"), model.get("player_only"), model.get("seed"),
                       model.get("objective")),
                      ("factorized_had", True, seed, expected_objective), f"{path.name}.model")
                _same(model.get("loss_coefficients"),
                      dict(spell=expected_loss[0], transition=expected_loss[1]),
                      f"{path.name}.loss_coefficients")
                for key in ("parameters", "optimizer_updates", "windows_per_epoch"):
                    _same(model[key], control[key], f"{path.name}.{key}")
                    _same(model[key], parity_row[key], f"{path.name}.parity.{key}")
                _same(model["windows_per_epoch"],
                      training_audit["windows_by_fold"][source]["windows_per_epoch"],
                      f"{path.name}.recomputed_windows")
                for key in ("median", "scale"):
                    _same(model[key], control[key], f"{path.name}.{key}")
                for key in ("feature_names", "player_indices", "player_only"):
                    _same(model[key], control[key], f"{path.name}.{key}")
                _same({key: value.shape for key, value in model["state"].items()},
                      {key: value.shape for key, value in control["state"].items()},
                      f"{path.name}.network_shapes")
                _require(len(model.get("history", [])) == lt.EPOCHS,
                         f"Candidate history length changed: {path.name}")
                _require(all(np.isfinite(value) for value in model["history"]),
                         f"Candidate loss history is nonfinite: {path.name}")
                _require(not any(key in model for key in ("full_y", "full_period", "indices")),
                         f"Training targets were retained in model: {path.name}")
                verified.append(dict(file=path.name, objective=expected_objective,
                                     parameters=int(model["parameters"]),
                                     updates=int(model["optimizer_updates"]),
                                     windows=int(model["windows_per_epoch"])))
    return dict(candidates=len(verified), models=verified,
                checkpoint_manifest_sha256=sha256(manifest_path))


def _expected_cells() -> set[tuple]:
    return {
        (source, path.stem, view, replicate, arm, seed)
        for source in dt.SPLITS for path in dt.SPLITS[source]["eval"]
        for view in oa.VIEWS
        for replicate in (("clean",) if view == "clean" else oa.REPLICATES)
        for arm in oa.ARMS for seed in oa.SEEDS
    }


def _verify_saved_rows(out: Path, spec: dict, report: dict) -> tuple[list[dict], dict]:
    result_rows = []
    for source in dt.SPLITS:
        path = out / f"{source}_results.json"
        _require(path.exists(), f"Missing evaluation results: {path}")
        result_rows.extend(json.loads(path.read_text()))
    expected = _expected_cells()
    keys = [(row["source"], row["match"], row["view"], row["replicate"],
             row["arm"], row["seed"]) for row in result_rows]
    _require(len(keys) == len(expected) and set(keys) == expected,
             f"Result cells incomplete or duplicated: {len(keys)}/{len(expected)}")
    report_rows = report.get("rows", [])
    report_keys = [(row["source"], row["match"], row["view"], row["replicate"],
                    row["arm"], row["seed"]) for row in report_rows]
    _require(len(report_keys) == len(expected) and set(report_keys) == expected,
             "Aggregate report rows are incomplete or duplicated")

    by_key = {key: row for key, row in zip(keys, result_rows)}
    report_by_key = {key: row for key, row in zip(report_keys, report_rows)}
    for key, row in by_key.items():
        _same(row, report_by_key[key], f"report_row/{key}")
    expected_parity = {
        (source, path.stem, view, replicate, seed)
        for source in dt.SPLITS for path in dt.SPLITS[source]["eval"]
        for view in oa.VIEWS
        for replicate in (("clean",) if view == "clean" else oa.REPLICATES)
        for seed in oa.SEEDS
    }
    parity_rows = []
    for source in dt.SPLITS:
        parity_path = out / f"{source}_control_parity.json"
        _require(parity_path.exists(), f"Missing control parity receipt: {parity_path.name}")
        current = json.loads(parity_path.read_text())
        _require(all(row.get("passed") is True for row in current),
                 f"Failed control parity receipt: {parity_path.name}")
        parity_rows.extend(current)
    parity_keys = {(row["source"], row["match"], row["view"], row["replicate"], row["seed"])
                   for row in parity_rows}
    _require(len(parity_rows) == len(expected_parity) and parity_keys == expected_parity,
             "Control parity receipts are incomplete or duplicated")
    report_parity = [row for provider_rows in report.get("control_parity", [])
                     for row in provider_rows]
    _require(len(report_parity) == len(expected_parity)
             and {(row["source"], row["match"], row["view"], row["replicate"], row["seed"])
                  for row in report_parity} == expected_parity
             and all(row.get("passed") is True for row in report_parity),
             "Report control parity receipts are incomplete or failed")
    _require(report["declaration_sha256"] == sha256(out / "predeclared.json"),
             "Report is bound to a different declaration")
    _require(report["checkpoint_manifest_sha256"] == sha256(out / "checkpoints_frozen.json"),
             "Report is bound to a different checkpoint manifest")
    cache_manifest = json.loads(oa.EVAL_CACHE_MANIFEST.read_text())
    archive_entries = {entry["path"]: entry for entry in cache_manifest["files"]}
    archive_sha = sha256(oa.EVAL_CACHE_MANIFEST)
    _same(archive_sha, oa.EVAL_CACHE_MANIFEST_SHA256,
          "evaluation_cache.pinned_archive_manifest_sha256")
    _same(archive_sha, spec["evaluation_feature_cache"]["archive_manifest_sha256"],
          "evaluation_cache.archive_manifest_sha256")
    archived_prior_key = "experiments/seed_gap_context_v1/predeclared.json"
    archived_prior_entry = archive_entries.get(archived_prior_key)
    _require(archived_prior_entry is not None,
             "Pinned evaluation cache manifest omits its predeclaration")
    prior_path = oa.EVAL_CACHE / "predeclared.json"
    _require(prior_path.exists()
             and prior_path.stat().st_size == archived_prior_entry["bytes"]
             and sha256(prior_path) == archived_prior_entry["sha256"],
             "Archived evaluation predeclaration differs from the pinned manifest")
    stress_rows = json.loads((oa.ROOT / "experiments/results/stress_component_ablation.json").read_text())["rows"]

    metric_fields = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                     "n_gt_spells", "n_pred_spells", "graded_frames")
    audited_predictions = 0
    for source in dt.SPLITS:
        for match_path in dt.SPLITS[source]["eval"]:
            match = match_path.stem
            for view in oa.VIEWS:
                replicates = ("clean",) if view == "clean" else oa.REPLICATES
                for replicate in replicates:
                    cache_key = f"{source}/{match}/{view}/{replicate}"
                    cache_row = spec["evaluation_feature_cache"]["files"].get(cache_key)
                    _require(cache_row is not None, f"Missing evaluation cache declaration: {cache_key}")
                    cache_path = oa.ROOT / cache_row["path"]
                    _require(cache_path.exists() and sha256(cache_path) == cache_row["sha256"],
                             f"Frozen evaluation cache changed: {cache_key}")
                    cached = joblib.load(cache_path)
                    inputs, labels, record = cached["inputs"], cached["labels"], cached["record"]
                    _require(record["source"] == source and record["match"] == match,
                             f"Evaluation cache identity differs: {cache_key}")
                    archived_npz_path = oa.EVAL_CACHE / f"{match}_{view}_{replicate}.npz"
                    archive_key = f"experiments/seed_gap_context_v1/{archived_npz_path.name}"
                    archive_entry = archive_entries.get(archive_key)
                    _require(archive_entry is not None,
                             f"Archived baseline probabilities absent: {archive_key}")
                    _require(archived_npz_path.exists()
                             and archived_npz_path.stat().st_size == archive_entry["bytes"]
                             and sha256(archived_npz_path) == archive_entry["sha256"],
                             f"Archived baseline probability cache changed: {archive_key}")
                    periods = supervised.model_periods(inputs)
                    graded_count = int(np.isin(labels, supervised.LABELS).sum())
                    for arm in oa.ARMS:
                        for seed in oa.SEEDS:
                            key = (source, match, view, replicate, arm, seed)
                            row = by_key[key]
                            npz_path = out / (f"{source}_{match}_{view}_{replicate}_"
                                              f"{arm}_{seed}.npz")
                            _require(npz_path.exists(), f"Missing saved prediction: {npz_path.name}")
                            with np.load(npz_path) as saved:
                                prediction = saved["prediction"]
                                probability = saved["probability"]
                            _require(probability.shape == (len(labels), 3)
                                     and prediction.shape == (len(labels),),
                                     f"Prediction shape differs: {npz_path.name}")
                            _require(np.isfinite(probability).all()
                                     and np.all(probability >= 0)
                                     and np.allclose(probability.sum(axis=1), 1.0, atol=1e-5),
                                     f"Invalid categorical probabilities: {npz_path.name}")
                            digest = hashlib.sha256(
                                np.ascontiguousarray(probability).tobytes()).hexdigest()
                            _same(row["probability_sha256"], digest,
                                  f"{npz_path.name}.probability_sha256")
                            decoded = supervised.decode(probability, periods,
                                                        float(inputs["fps"]), lt.COST)
                            _same(prediction, decoded, f"{npz_path.name}.decoded_prediction")
                            scored = metrics.evaluate_tracking(
                                labels, prediction, period=inputs.get("period"),
                                fps=float(inputs["fps"]), timestamps=inputs.get("timestamps"))
                            _same(int(row["graded_frames"]), graded_count,
                                  f"{key}.graded_frames")
                            _same(int(row["n_gt_spells"]), int(scored["n_gt_spells"]),
                                  f"{key}.n_gt_spells")
                            for field in metric_fields[:5]:
                                _require(abs(float(row[field]) - float(scored[field])) <= 1e-9,
                                         f"Saved prediction score mismatch {key}/{field}")
                            expected_weighted = .7 * scored["spell_f1_mean"] + .3 * scored["macro_f1"]
                            _require(abs(float(row["weighted_70_30"]) - expected_weighted) <= 1e-9,
                                     f"Saved prediction weighted summary mismatch: {key}")
                            for iou, values in scored["spell_f1"].items():
                                actual_iou = row["spell_by_iou"][str(iou)]
                                for field in ("f1", "precision", "recall", "n_matched",
                                              "n_spurious", "n_missed"):
                                    _require(abs(float(actual_iou[field]) - float(values[field])) <= 1e-9,
                                             f"Saved prediction IoU score mismatch {key}/{iou}/{field}")
                            if arm == "control":
                                ref = next((item for item in stress_rows
                                            if (item["source"], item["match"], item["view"],
                                                item["replicate"], item["recipe"])
                                            == (source, match, view, replicate,
                                                f"slot_dropout_{seed}")), None)
                                _require(ref is not None, f"Missing archived score reference: {key}")
                                for field in metric_fields[:-1]:
                                    _require(abs(float(row[field]) - float(ref[field])) <= 1e-9,
                                             f"Control parity differs for {key}/{field}")
                                for iou in ("0.3", "0.4", "0.5", "0.6", "0.7"):
                                    _require(abs(float(row["spell_by_iou"][iou]["f1"])
                                                 - float(ref["spell_thresholds"][iou])) <= 1e-9,
                                             f"Control IoU parity differs for {key}/{iou}")
                                with np.load(archived_npz_path) as baseline:
                                    baseline_probability = baseline[f"seed_{seed}"]
                                _require(np.allclose(probability, baseline_probability,
                                                     atol=1e-6, rtol=0),
                                         f"Control probability parity differs: {key}")
                                if view == "clean":
                                    native = gr.DEFAULT / f"{source}_{match}_clean_slot_dropout_{seed}.npz"
                                    with np.load(native) as baseline:
                                        native_probability = baseline["probability"]
                                    _require(np.allclose(probability, native_probability,
                                                         atol=1e-6, rtol=0),
                                             f"Restored native control parity differs: {key}")
                            audited_predictions += 1

                    # Every arm/seed must retain one fixed grading population.
                    baseline_population = by_key[(source, match, view, replicates[0],
                                                  "control", oa.SEEDS[0])]
                    for replicate_id in replicates:
                        for arm in oa.ARMS:
                            for seed in oa.SEEDS:
                                row = by_key[(source, match, view, replicate_id, arm, seed)]
                                for field in ("graded_frames", "n_gt_spells"):
                                    _same(row[field], baseline_population[field],
                                          f"population/{source}/{match}/{view}/{replicate_id}/{arm}/{seed}/{field}")

    return result_rows, dict(cells=len(result_rows), saved_predictions_audited=audited_predictions)


def _mean_defined(values):
    selected = [float(value) for value in values if value is not None]
    return float(np.mean(selected)) if selected else None


def _independent_aggregate(rows: list[dict]) -> list[dict]:
    metric_keys = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
                   "d_recall", "short_spell_recall", "false_possessions_per_minute",
                   "n_gt_spells", "n_pred_spells")
    iou_keys = ("0.3", "0.4", "0.5", "0.6", "0.7")
    result = []
    for arm in oa.ARMS:
        for seed in oa.SEEDS:
            for view in oa.VIEWS:
                provider_summaries = {}
                for source in dt.SPLITS:
                    match_summaries = []
                    for path in dt.SPLITS[source]["eval"]:
                        selected = [row for row in rows if
                                    (row["source"], row["match"], row["view"], row["arm"], row["seed"])
                                    == (source, path.stem, view, arm, seed)]
                        expected_n = 1 if view == "clean" else len(oa.REPLICATES)
                        _require(len(selected) == expected_n,
                                 f"Aggregate has {len(selected)} cells for {source}/{path.stem}/{view}/{arm}/{seed}")
                        match_summaries.append({
                            **{key: _mean_defined([row[key] for row in selected])
                               for key in metric_keys},
                            "spell_by_iou": {
                                iou: _mean_defined([row["spell_by_iou"][iou]["f1"]
                                                    for row in selected])
                                for iou in iou_keys
                            },
                        })
                    provider_summaries[source] = {
                        **{key: _mean_defined([row[key] for row in match_summaries])
                           for key in metric_keys},
                        "spell_by_iou": {
                            iou: _mean_defined([row["spell_by_iou"][iou]
                                                for row in match_summaries])
                            for iou in iou_keys
                        },
                        "matches": len(match_summaries),
                    }
                result.append(dict(
                    arm=arm, seed=seed, view=view,
                    per_provider=provider_summaries,
                    **{key: _mean_defined([provider_summaries[source][key]
                                           for source in dt.SPLITS])
                       for key in metric_keys},
                    spell_by_iou={iou: _mean_defined([
                        provider_summaries[source]["spell_by_iou"][iou]
                        for source in dt.SPLITS]) for iou in iou_keys},
                ))
    return result


def _audit_aggregates(rows: list[dict], report: dict) -> dict:
    recomputed = _independent_aggregate(rows)
    stored = report.get("summaries", [])
    expected_keys = {(row["arm"], row["seed"], row["view"]) for row in recomputed}
    _require(len(stored) == len(recomputed)
             and {(row["arm"], row["seed"], row["view"]) for row in stored} == expected_keys,
             "Aggregate summaries are incomplete or duplicated")
    by_key = {(row["arm"], row["seed"], row["view"]): row for row in stored}
    metric_keys = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
                   "d_recall", "short_spell_recall", "false_possessions_per_minute",
                   "n_gt_spells", "n_pred_spells")
    for expected in recomputed:
        key = (expected["arm"], expected["seed"], expected["view"])
        actual = by_key[key]
        for metric in metric_keys:
            left, right = expected[metric], actual[metric]
            if left is None or right is None:
                _same(left, right, f"summary/{key}/{metric}")
            else:
                _require(abs(float(left) - float(right)) <= 1e-12,
                         f"Aggregate mismatch {key}/{metric}: {left} != {right}")
        for source in dt.SPLITS:
            _same(expected["per_provider"][source]["matches"],
                  actual["per_provider"][source]["matches"], f"summary/{key}/{source}/matches")
            for metric in metric_keys:
                left = expected["per_provider"][source][metric]
                right = actual["per_provider"][source][metric]
                if left is None or right is None:
                    _same(left, right, f"summary/{key}/{source}/{metric}")
                else:
                    _require(abs(float(left) - float(right)) <= 1e-12,
                             f"Provider aggregate mismatch {key}/{source}/{metric}")
            for iou in ("0.3", "0.4", "0.5", "0.6", "0.7"):
                left = expected["per_provider"][source]["spell_by_iou"][iou]
                right = actual["per_provider"][source]["spell_by_iou"][iou]
                if left is None or right is None:
                    _same(left, right, f"summary/{key}/{source}/spell_by_iou/{iou}")
                else:
                    _require(abs(float(left) - float(right)) <= 1e-12,
                             f"Provider aggregate IoU mismatch {key}/{source}/{iou}")
        for iou in ("0.3", "0.4", "0.5", "0.6", "0.7"):
            left, right = expected["spell_by_iou"][iou], actual["spell_by_iou"][iou]
            if left is None or right is None:
                _same(left, right, f"summary/{key}/spell_by_iou/{iou}")
            else:
                _require(abs(float(left) - float(right)) <= 1e-12,
                         f"Aggregate IoU mismatch {key}/{iou}")

    controls = {(row["seed"], row["view"]): row for row in recomputed
                if row["arm"] == "control"}
    expected_screens = []
    for arm in oa.NEW_ARMS:
        for seed in oa.SEEDS:
            clean = next(row for row in recomputed
                         if (row["arm"], row["seed"], row["view"]) == (arm, seed, "clean"))
            stress = next(row for row in recomputed
                          if (row["arm"], row["seed"], row["view"]) == (arm, seed, "feed_like"))
            base_clean = controls[(seed, "clean")]
            base_stress = controls[(seed, "feed_like")]
            worst_provider = min(
                clean["per_provider"][source][key] - base_clean["per_provider"][source][key]
                for source in dt.SPLITS for key in ("d_f1", "macro_f1"))
            clean_delta = clean["spell_f1"] - base_clean["spell_f1"]
            stress_delta = stress["spell_f1"] - base_stress["spell_f1"]
            expected_screens.append(dict(
                arm=arm, seed=seed,
                clean_spell_delta=clean_delta,
                feed_like_spell_delta=stress_delta,
                worst_provider_clean_d_or_macro_delta=worst_provider,
                passes_positive_screen=clean_delta >= .01,
                passes_guardrails=stress_delta >= -.01 and worst_provider >= -.02,
                passes_all=clean_delta >= .01 and stress_delta >= -.01 and worst_provider >= -.02,
            ))
    stored_screens = report.get("screens", [])
    _require(len(stored_screens) == len(expected_screens), "Screen rows are incomplete")
    screen_map = {(row["arm"], row["seed"]): row for row in stored_screens}
    for expected in expected_screens:
        actual = screen_map.get((expected["arm"], expected["seed"]))
        _require(actual is not None, f"Missing screen {expected['arm']}/{expected['seed']}")
        for key, value in expected.items():
            if isinstance(value, bool):
                _same(actual[key], value, f"screen/{expected['arm']}/{expected['seed']}/{key}")
            else:
                _require(abs(float(actual[key]) - float(value)) <= 1e-12,
                         f"Screen mismatch {expected['arm']}/{expected['seed']}/{key}")
    return dict(summaries=len(recomputed), screens=len(expected_screens))


def verify(out: Path = oa.DEFAULT) -> dict:
    out = Path(out)
    spec = oa.check(out)
    manifest_path = out / "checkpoints_frozen.json"
    report_path = out / "report.json"
    _require(manifest_path.exists(), "Seal all 18 candidate checkpoints before verification")
    _require(report_path.exists(), "Complete all declared evaluation cells before verification")
    report = json.loads(report_path.read_text())
    _same(spec["arms"], list(oa.ARMS), "declaration.arms")
    _same(spec["new_arms"], list(oa.NEW_ARMS), "declaration.new_arms")
    _same(spec["seeds"], list(oa.SEEDS), "declaration.seeds")
    _same(spec["objectives"], oa.OBJECTIVE_BY_ARM, "declaration.objectives")
    _same(report.get("declaration"), spec, "report.declaration")
    _same(report.get("declaration_sha256"), sha256(out / "predeclared.json"),
          "report.declaration_sha256")
    training = _audit_training_records(out, spec)
    models = _audit_models(out, spec, training)
    rows, cells = _verify_saved_rows(out, spec, report)
    aggregates = _audit_aggregates(rows, report)
    result_copy = oa.RESULT
    _require(result_copy.exists() and sha256(result_copy) == sha256(report_path),
             "Published aggregate copy differs from the private study report")
    result = dict(
        verified=True,
        declaration_sha256=sha256(out / "predeclared.json"),
        candidate_manifest_sha256=sha256(manifest_path),
        report_sha256=sha256(report_path),
        training=training,
        models=models,
        evaluation=cells,
        aggregation=aggregates,
        verifier_source_sha256=sha256(Path(__file__)),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=oa.DEFAULT)
    args = parser.parse_args()
    result = verify(args.out)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
