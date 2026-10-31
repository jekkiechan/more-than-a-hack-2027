"""Frozen evaluation for the dense short-team attribution experiment.

The runner reuses the pre-existing D2 control probabilities and evaluates the
six prefit candidate models on the same 15 provider-holdout cells per seed.
It has no fitting path and does not load organizer labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np

from experiments import dense_short_team_v1 as study
from experiments import dense_error_audit_v1 as error_audit
from experiments import dense_label_weight_v1 as prior
from experiments import objective_ensemble as scoring
from experiments import soccernet_extension as old
from experiments import sn_dense_train as sn

ROOT = study.ROOT
RESULTS = ROOT / "experiments/results"
OUT = study.OUT
SEEDS = study.SEEDS
FOLDS = study.FOLDS
ARMS = ("control_D2", "short_team_2x")
VIEWS = old.VIEWS
EXPECTED_CELLS_PER_MEMBER = 15
CONTROL_TOLERANCE = study.CONTROL_TOLERANCE
DECLARATION = RESULTS / "dense_short_team_v1_declaration.json"
SEAL = RESULTS / "dense_short_team_v1_evaluation_seal.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _logical_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return "@DATA/" + path.relative_to(prior.old.DATA.resolve()).as_posix()


def _resolve_logical_path(value: str) -> Path:
    if value.startswith("@DATA/"):
        return prior.old.DATA / value.removeprefix("@DATA/")
    return ROOT / value


def write_once(path: Path, value: dict) -> None:
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"Refusing to overwrite frozen artifact: {path}")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _selection() -> list[dict]:
    items = old.selection_items()
    if len(items) != 5 or len({(i["source"], i["match"]) for i in items}) != 5:
        raise ValueError("Expected the frozen five-match provider selection cohort")
    return items


def _evaluation_input(item: dict, view: str, replicate: str):
    path = prior.old.eval_path(item, view, replicate)
    expected = old.old_spec()["evaluation_feature_cache"]["files"]
    rel = str(path.relative_to(prior.old.DATA))
    entry = next((v for v in expected.values() if v.get("path") == f"data/{rel}"), None)
    if entry is None or sha256(path) != entry["sha256"]:
        raise ValueError(f"Evaluation input is not bound by the frozen D2 manifest: {path}")
    cached = joblib.load(path)
    return path, cached


def validate_saved_control_replay() -> dict:
    """Read-only validation of every sealed D2 member and ensemble archive."""
    gate = prior.require_control_gate()
    for view, expected_score in study.EXPECTED_CONTROL.items():
        observed = gate["values"][view]["observed"]
        if abs(observed - expected_score) > CONTROL_TOLERANCE:
            raise ValueError(f"D2 replay exceeds declared tolerance for {view}")
    member_doc = json.loads((RESULTS / "dense_label_weight_v1_member_scores_control_D2.json").read_text())
    ensemble_doc = json.loads((RESULTS / "dense_label_weight_v1_ensemble_scores_control_D2.json").read_text())
    historical_audit = RESULTS / "dense_label_weight_v1_control_prediction_audit.json"
    historical_audit_hash = sha256(historical_audit)
    prior.audit_control_prediction_provenance(member_doc, ensemble_doc)
    if sha256(historical_audit) != historical_audit_hash:
        raise ValueError("Historical control audit receipt changed during validation")
    member_hashes = member_doc["prediction_files"]
    ensemble_hashes = ensemble_doc["prediction_files"]
    for item in _selection():
        fold = "leave_out_" + item["source"]
        for view, replicate in VIEWS:
            _, cached = _evaluation_input(item, view, replicate)
            inputs = cached["inputs"]
            labels = np.asarray(cached["labels"], dtype="U1")
            probabilities = {}
            for seed in SEEDS:
                path = _archive_path(item, view, replicate, str(seed), "control_D2")
                rel = str(path.relative_to(ROOT))
                if not path.is_file() or member_hashes.get(rel) != sha256(path):
                    raise ValueError(f"D2 seed array differs from its frozen receipt: {rel}")
                archive = _archive(path)
                reference = {"labels": labels, "period": inputs["period"],
                    "fps": np.asarray(float(inputs["fps"])),
                    "timestamps": inputs["timestamps"], "ball_observed": inputs["ball_observed"]}
                assert_native_arrays_equal(reference, archive)
                decoded = prior.supervised.decode(archive["probabilities"],
                    prior.supervised.model_periods(inputs), float(inputs["fps"]), prior.lt.COST)
                if not np.array_equal(archive["prediction"], decoded):
                    raise ValueError(f"D2 seed prediction decoder replay failed: {rel}")
                probabilities[seed] = archive["probabilities"]
            ensemble_path = _archive_path(item, view, replicate, "seedensemble", "control_D2")
            ensemble_rel = str(ensemble_path.relative_to(ROOT))
            if not ensemble_path.is_file() or ensemble_hashes.get(ensemble_rel) != sha256(ensemble_path):
                raise ValueError(f"D2 ensemble differs from its frozen receipt: {ensemble_rel}")
            ensemble = _archive(ensemble_path)
            assert_native_arrays_equal(reference, ensemble)
            expected_probability = (probabilities[2026] + probabilities[2027]) / 2.0
            if not np.array_equal(ensemble["probabilities"], expected_probability):
                raise ValueError(f"D2 ensemble is not the exact native probability mean: {ensemble_rel}")
            decoded = prior.supervised.decode(expected_probability,
                prior.supervised.model_periods(inputs), float(inputs["fps"]), prior.lt.COST)
            if not np.array_equal(ensemble["prediction"], decoded):
                raise ValueError(f"D2 ensemble decoder replay failed: {ensemble_rel}")
    return {"passed": True, "clean_feed_replay_tolerance": CONTROL_TOLERANCE,
            "member_count": 30, "ensemble_count": 15,
            "control_audit_receipt_sha256": historical_audit_hash}


def freeze_inputs() -> dict:
    """Seal all model/source/input hashes before any prediction or scoring."""
    declaration = json.loads(DECLARATION.read_text())
    if declaration.get("study") != study.STUDY_ID:
        raise ValueError("Declaration does not identify this study")
    validate_saved_control_replay()
    member_doc = json.loads((RESULTS / "dense_label_weight_v1_member_scores_control_D2.json").read_text())
    ensemble_doc = json.loads((RESULTS / "dense_label_weight_v1_ensemble_scores_control_D2.json").read_text())
    # Bind the declaration and all inference/scoring code before model use.
    source_paths = (Path(__file__), ROOT / "experiments/dense_short_team_v1.py",
                    ROOT / "experiments/dense_label_weight_v1.py",
                    ROOT / "experiments/dense_error_audit_v1.py",
                    ROOT / "experiments/objective_ensemble.py",
                    ROOT / "experiments/soccernet_extension.py",
                    ROOT / "experiments/sn_dense_train.py",
                    ROOT / "experiments/supervised.py",
                    ROOT / "experiments/model_decision_policy.json",
                    ROOT / "evaluation/possession_metrics.py")
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}
    models, fit_receipts = {}, {}
    for fold in FOLDS:
        for seed in SEEDS:
            candidate = study.candidate_model_path(fold, seed)
            fit = study.candidate_fit_receipt_path(fold, seed)
            control = study.control_model_path(fold, seed)
            if not candidate.is_file() or not fit.is_file() or not control.is_file():
                raise FileNotFoundError(f"Missing fit or control artifact for {fold}/{seed}")
            fit_receipt = json.loads(fit.read_text())
            candidate_hash = sha256(candidate)
            if fit_receipt.get("model_sha256") != candidate_hash:
                raise ValueError(f"Candidate checkpoint differs from fit receipt: {candidate}")
            expected_fit = {"study": study.STUDY_ID, "arm": "short_team_2x",
                            "fold": fold, "heldout": fold.removeprefix("leave_out_"),
                            "seed": seed, "updates": 2048}
            for key, expected in expected_fit.items():
                if fit_receipt.get(key) != expected:
                    raise ValueError(f"Fit receipt {key} differs from declaration: {fit}")
            if fit_receipt.get("declaration_sha256") != sha256(DECLARATION):
                raise ValueError(f"Fit receipt does not bind this declaration: {fit}")
            teacher_path = prior.control_cache(fold, seed)
            if fit_receipt.get("teacher_sha256") != sha256(teacher_path):
                raise ValueError(f"Fit receipt teacher hash differs: {fit}")
            for rel, expected in fit_receipt.get("source_sha256", {}).items():
                source_path = ROOT / rel
                if not source_path.is_file() or sha256(source_path) != expected:
                    raise ValueError(f"Training source differs from fit receipt: {rel}")
            models[str(candidate.relative_to(ROOT))] = candidate_hash
            models[str(control.relative_to(ROOT))] = sha256(control)
            fit_receipts[str(fit.relative_to(ROOT))] = sha256(fit)
    inputs = {}
    for item in _selection():
        for view, replicate in VIEWS:
            path, _ = _evaluation_input(item, view, replicate)
            inputs[_logical_path(path)] = sha256(path)
    training_inputs = {}
    for fold in FOLDS:
        for seed in SEEDS:
            fit_path = study.candidate_fit_receipt_path(fold, seed)
            fit_receipt = json.loads(fit_path.read_text())
            declared_training_inputs = fit_receipt.get("train_input_sha256")
            if not isinstance(declared_training_inputs, dict) or not declared_training_inputs:
                raise ValueError(f"Fit receipt lacks training input hashes: {fit_path}")
            for raw_path, expected in declared_training_inputs.items():
                path = Path(raw_path)
                if not path.is_absolute():
                    path = ROOT / path
                if not path.is_file() or sha256(path) != expected:
                    raise ValueError(f"Training input differs from fit receipt: {raw_path}")
                training_inputs[_logical_path(path)] = expected
    controls = {}
    score_file_hashes = {**member_doc["prediction_files"], **ensemble_doc["prediction_files"]}
    for item in _selection():
        for view, replicate in VIEWS:
            for seed in (*map(str, SEEDS), "seedensemble"):
                path = _archive_path(item, view, replicate, seed, "control_D2")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing saved D2 prediction: {path}")
                rel = str(path.relative_to(ROOT))
                actual = sha256(path)
                if score_file_hashes.get(rel) != actual:
                    raise ValueError(f"D2 prediction differs from prior score receipt: {rel}")
                controls[rel] = actual
    teachers = {}
    for fold in FOLDS:
        for seed in SEEDS:
            path = prior.control_cache(fold, seed)
            teachers[_logical_path(path)] = sha256(path)
    value = {"schema": 1, "study": study.STUDY_ID,
             "declaration_sha256": sha256(DECLARATION), "source_sha256": sources,
             "model_sha256": models, "fit_receipt_sha256": fit_receipts,
             "evaluation_input_sha256": inputs, "training_input_sha256": training_inputs,
             "control_prediction_sha256": controls, "teacher_sha256": teachers,
             "cohort_cells": EXPECTED_CELLS_PER_MEMBER,
             "score_started": False}
    write_once(SEAL, value)
    return value


def verify_seal() -> dict:
    if not SEAL.is_file():
        raise RuntimeError("Freeze inputs before evaluation")
    seal = json.loads(SEAL.read_text())
    if seal.get("score_started") is not False:
        raise ValueError("Invalid evaluation seal")
    if sha256(DECLARATION) != seal["declaration_sha256"]:
        raise ValueError("Frozen declaration changed after sealing")
    for rel, expected in {**seal["source_sha256"], **seal["model_sha256"],
                          **seal["fit_receipt_sha256"],
                          **seal["evaluation_input_sha256"], **seal["training_input_sha256"],
                          **seal["control_prediction_sha256"], **seal["teacher_sha256"]}.items():
        if sha256(_resolve_logical_path(rel)) != expected:
            raise ValueError(f"Frozen input changed after seal: {rel}")
    return seal


def _archive(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in z.files}


def assert_native_arrays_equal(reference: dict, candidate: dict) -> None:
    """Ensure paired predictions preserve labels, native masks, and time."""
    for key in ("labels", "period", "timestamps", "ball_observed", "fps"):
        if key not in reference or key not in candidate:
            raise ValueError(f"Paired archive is missing native array {key}")
        if not np.array_equal(np.asarray(reference[key]), np.asarray(candidate[key])):
            raise ValueError(f"Paired archives changed native {key}")


def validate_complete_cell_cohort(rows: list[dict], expected: set[tuple]) -> None:
    for arm in ARMS:
        for member in ("2026", "2027", "ensemble"):
            selected = [r for r in rows if r["arm"] == arm and r["member"] == member]
            observed = {(r["source"], r["match"], r["view"], r["replicate"]) for r in selected}
            if len(selected) != EXPECTED_CELLS_PER_MEMBER or observed != expected:
                raise ValueError(f"Incomplete or mismatched cohort for {arm}/{member}")
    keys = [(r["source"], r["match"], r["view"], r["replicate"], r["arm"], r["member"])
            for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate paired score cell")


def short_spell_metrics(labels, prediction, inputs) -> dict:
    """Recall for complete <=1s spells using the official IoU=.5 assignment."""
    from evaluation import possession_metrics as pm
    from experiments.metrics import prepare_tracking_codes
    fps = float(inputs["fps"])
    _, truth, guess, _ = prepare_tracking_codes(labels, prediction, inputs["period"],
                                                 fps, inputs["timestamps"])
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(guess, runs)
    refs = pm.possession_spells(truth, runs)
    preds = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(refs, preds)
    matched = pm.matched_gt_flags(overlaps, len(refs), .5)
    run_starts, run_ends = {a for a, _ in runs}, {b for _, b in runs}
    selected = [i for i, (start, stop, _) in enumerate(refs)
                if (stop - start) / fps <= 1.0 and start not in run_starts and stop not in run_ends]
    return {"complete_le_1s_reference_spells": len(selected),
            "complete_le_1s_matched_spells": int(matched[selected].sum()),
            "complete_le_1s_spell_recall": (float(matched[selected].mean()) if selected else None)}


def public_error_aggregate(detail: dict) -> dict:
    """Retain aggregate counts/rates while dropping match and segment detail."""
    providers = {}
    for provider, value in detail["providers"].items():
        segments = value["segment_details"]
        bins = {}
        for name in ("short_le_1s", "medium_1_to_5s", "long_gt_5s"):
            chunks = [s["gt_spell_length_bins"][name] for s in segments]
            count = sum(x["gt_spells"] for x in chunks)
            alive = sum(x["gt_alive_frames"] for x in chunks)
            wrong = sum(x["wrong_team_frames"] for x in chunks)
            false_d = sum(x["false_d_frames"] for x in chunks)
            bins[name] = {"gt_spells": count, "gt_alive_frames": alive,
                          "wrong_team_frames": wrong, "false_d_frames": false_d,
                          "wrong_team_rate": wrong / alive if alive else None,
                          "false_d_rate": false_d / alive if alive else None,
                          "false_d_spells": sum(x["false_d_spells"] for x in chunks)}
        scored = sum(x["scored_frames"] for x in segments)
        wrong = value["wrong_team_frames"]
        false_d = value["false_d_frames"]
        providers[provider] = {"matches": value["matches"], "segments": value["segments"],
            "wrong_team_frames": wrong, "wrong_team_rate": wrong / scored if scored else None,
            "false_d_frames": false_d, "false_d_rate": false_d / scored if scored else None,
            "short_spell_bins": bins}
    return {"providers": providers, "interpretation": detail["interpretation"]}


def aggregate_error_cells(frames) -> dict:
    """Audit each realization separately, then average replicate→match→provider."""
    results = {}
    for arm in sorted(frames["arm"].unique()):
        results[arm] = {}
        for view in ("clean", "feed_like"):
            selected = frames[(frames["arm"] == arm) & (frames["view"] == view)]
            per_cell = []
            for (provider, match, replicate), group in selected.groupby(
                    ["provider", "match_id", "replicate"], sort=True):
                report = error_audit.audit(group.drop(columns=["arm", "view", "replicate"]))
                summary = report["providers"][str(provider)]
                seg = summary["segment_details"]
                shorts = [s["gt_spell_length_bins"]["short_le_1s"] for s in seg]
                short_alive = sum(s["gt_alive_frames"] for s in shorts)
                short_wrong = sum(s["wrong_team_frames"] for s in shorts)
                short_false_d = sum(s["false_d_frames"] for s in shorts)
                cell = dict(provider=str(provider), match=str(match), replicate=str(replicate),
                            wrong_team_frames=summary["wrong_team_frames"],
                            false_d_frames=summary["false_d_frames"],
                            scored_frames=sum(s["scored_frames"] for s in seg),
                            short={"gt_spells": sum(s["gt_spells"] for s in shorts),
                                   "gt_alive_frames": short_alive,
                                   "wrong_team_frames": short_wrong,
                                   "false_d_frames": short_false_d,
                                   "false_d_spells": sum(s["false_d_spells"] for s in shorts),
                                   "wrong_team_rate": short_wrong / short_alive if short_alive else None,
                                   "false_d_rate": short_false_d / short_alive if short_alive else None})
                per_cell.append(cell)
            provider_stats = {}
            for provider in sorted({c["provider"] for c in per_cell}):
                cells = [c for c in per_cell if c["provider"] == provider]
                by_match = {}
                for cell in cells:
                    by_match.setdefault(cell["match"], []).append(cell)
                match_values = []
                for match_cells in by_match.values():
                    # Corruption replicates receive equal weight within match.
                    match_values.append({
                        "wrong_team_rate": float(np.mean([c["wrong_team_frames"] / max(1, c["scored_frames"])
                                                           for c in match_cells])),
                        "false_d_rate": float(np.mean([c["false_d_frames"] / max(1, c["scored_frames"])
                                                        for c in match_cells])),
                        "short_wrong_team_rate": float(np.mean([
                            c["short"]["wrong_team_rate"] for c in match_cells
                            if c["short"]["wrong_team_rate"] is not None])) if any(
                                c["short"]["wrong_team_rate"] is not None for c in match_cells) else None,
                        "short_false_d_rate": float(np.mean([
                            c["short"]["false_d_rate"] for c in match_cells
                            if c["short"]["false_d_rate"] is not None])) if any(
                                c["short"]["false_d_rate"] is not None for c in match_cells) else None,
                    })
                def mean_defined(field):
                    values = [v[field] for v in match_values if v[field] is not None]
                    return float(np.mean(values)) if values else None
                provider_stats[provider] = {
                    "matches": len(match_values), "evaluation_cells": len(cells),
                    "wrong_team_frames": sum(c["wrong_team_frames"] for c in cells),
                    "false_d_frames": sum(c["false_d_frames"] for c in cells),
                    "scored_frames": sum(c["scored_frames"] for c in cells),
                    "wrong_team_rate": mean_defined("wrong_team_rate"),
                    "false_d_rate": mean_defined("false_d_rate"),
                    "le_1s_wrong_team_rate": mean_defined("short_wrong_team_rate"),
                    "le_1s_false_d_rate": mean_defined("short_false_d_rate"),
                }
            results[arm][view] = {"per_provider": provider_stats,
                "equal_provider_mean": {key: float(np.mean([p[key] for p in provider_stats.values()
                    if p[key] is not None])) if any(p[key] is not None for p in provider_stats.values()) else None
                    for key in ("wrong_team_rate", "false_d_rate", "le_1s_wrong_team_rate", "le_1s_false_d_rate")},
                "aggregate_counts": {key: sum(c[key] for c in per_cell) for key in
                    ("wrong_team_frames", "false_d_frames", "scored_frames")},
                "weighting": "Replicates averaged within source-match, matches within provider, then providers equally; raw counts are pooled across paired cells."}
    return results


def diagnose_saved_control_ensembles():
    """Exercise final per-cell diagnostics on the 15 frozen control ensembles."""
    frames = []
    for item in _selection():
        fold = "leave_out_" + item["source"]
        for view, replicate in VIEWS:
            input_path, cached = _evaluation_input(item, view, replicate)
            archive_path = _archive_path(item, view, replicate, "seedensemble", "control_D2")
            if not archive_path.is_file():
                raise FileNotFoundError(f"Missing saved control ensemble: {archive_path}")
            archive = _archive(archive_path)
            inputs, labels = cached["inputs"], np.asarray(cached["labels"], dtype="U1")
            if sha256(input_path) != archive.get("evaluation_input_sha256", ""):
                # Legacy control archives store a JSON map, validated by the provenance receipt.
                stored_hashes = json.loads(str(np.asarray(archive["evaluation_input_hashes_json"]).item()))
                expected = f"data/{input_path.relative_to(prior.old.DATA).as_posix()}"
                if stored_hashes.get(expected) != sha256(input_path):
                    raise ValueError(f"Control ensemble input hash mismatch: {archive_path}")
            reference = {"labels": labels, "period": inputs["period"],
                "fps": np.asarray(float(inputs["fps"])), "timestamps": inputs["timestamps"],
                "ball_observed": inputs["ball_observed"]}
            assert_native_arrays_equal(reference, archive)
            frame = error_audit.frame_table(item["source"], item["match"], labels,
                archive["prediction"], inputs["ball_observed"], inputs["period"],
                inputs["timestamps"], float(inputs["fps"]))
            frame["arm"] = "control_D2"
            frame["view"] = view
            frame["replicate"] = replicate
            frames.append(frame)
    return aggregate_error_cells(__import__("pandas").concat(frames, ignore_index=True))


def _archive_path(item, view, replicate, member, arm):
    fold = "leave_out_" + item["source"]
    member_name = member if str(member).startswith("seed") else f"seed{member}"
    filename = f"{member_name}__{item['match']}__{view}__{replicate}.npz"
    if arm == "short_team_2x":
        return study.candidate_prediction_dir(fold) / filename
    return study.CONTROL_OUT / "predictions" / "control_D2" / fold / filename


def _save_prediction(path: Path, probability: np.ndarray, prediction: np.ndarray,
                     labels: np.ndarray, inputs: dict, item: dict, view: str,
                     replicate: str, fold: str, seed: str, arm: str,
                     model_hash: str, input_hash: str) -> None:
    if path.exists():
        old_archive = _archive(path)
        if (not np.array_equal(old_archive["prediction"], prediction) or
                not np.array_equal(old_archive["probabilities"], probability)):
            raise ValueError(f"Refusing to overwrite changed prediction: {path}")
        reference = {"labels": np.asarray(labels, dtype="U1"),
            "period": np.asarray(inputs["period"]), "timestamps": np.asarray(inputs["timestamps"], dtype=float),
            "ball_observed": np.asarray(inputs["ball_observed"], dtype=bool),
            "fps": np.asarray(float(inputs["fps"]))}
        assert_native_arrays_equal(reference, old_archive)
        for key, expected in {"source": item["source"], "match": item["match"],
                              "view": view, "replicate": replicate, "fold": fold,
                              "seed": seed, "arm": arm, "model_sha256": model_hash,
                              "evaluation_input_sha256": input_hash}.items():
            if key not in old_archive or str(np.asarray(old_archive[key]).reshape(()).item()) != str(expected):
                raise ValueError(f"Saved prediction metadata changed for {key}: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, probabilities=np.asarray(probability, dtype=np.float64),
        prediction=np.asarray(prediction, dtype="U1"), labels=np.asarray(labels, dtype="U1"),
        period=np.asarray(inputs["period"]), fps=np.asarray(float(inputs["fps"])),
        timestamps=np.asarray(inputs["timestamps"], dtype=float),
        ball_observed=np.asarray(inputs["ball_observed"], dtype=bool),
        source=np.asarray(item["source"]), match=np.asarray(item["match"]),
        view=np.asarray(view), replicate=np.asarray(replicate), seed=np.asarray(seed),
        arm=np.asarray(arm), fold=np.asarray(fold), model_sha256=np.asarray(model_hash),
        evaluation_input_sha256=np.asarray(input_hash))


def evaluate() -> dict:
    """Generate paired native predictions and official score rows.

    Call only after the coordinator approves the frozen declaration and seal.
    """
    seal = verify_seal()
    control_gate = json.loads((RESULTS / "dense_label_weight_v1_control_replay.json").read_text())
    if control_gate.get("passed") is not True:
        raise RuntimeError("Candidate scoring blocked until exact D2 control replay passes")
    rows, files, audit_frames = [], {}, []
    items = _selection()
    model_objects = {}
    for fold in FOLDS:
        for seed in SEEDS:
            cp = study.candidate_model_path(fold, seed)
            tp = study.control_model_path(fold, seed)
            model_objects[(fold, seed, "short_team_2x")] = joblib.load(cp)
            model_objects[(fold, seed, "control_D2")] = joblib.load(tp)
    control_probs = {}
    for item in items:
        fold = "leave_out_" + item["source"]
        for view, replicate in VIEWS:
            input_path, cached = _evaluation_input(item, view, replicate)
            inputs, labels = cached["inputs"], np.asarray(cached["labels"], dtype="U1")
            record = prior.old.inference_record(cached["record"])
            input_hash = sha256(input_path)
            member_probs = {}
            for arm in ARMS:
                for seed in SEEDS:
                    model_path = (study.candidate_model_path(fold, seed) if arm == "short_team_2x"
                                  else study.control_model_path(fold, seed))
                    model_hash = sha256(model_path)
                    model = model_objects[(fold, seed, arm)]
                    teacher_path = prior.control_cache(fold, seed)
                    teacher = joblib.load(teacher_path)
                    if arm == "short_team_2x":
                        sampled = sn.predict(record, model, teacher)
                    else:
                        # Reuse the D2 prediction saved by the previous study; no control refit.
                        control_path = _archive_path(item, view, replicate, str(seed), arm)
                        archive = _archive(control_path)
                        if sha256(control_path) != seal["control_prediction_sha256"].get(
                                str(control_path.relative_to(ROOT))):
                            raise ValueError(f"D2 prediction changed after seal: {control_path}")
                        probability = archive["probabilities"]
                        for key in ("labels", "period", "timestamps", "ball_observed", "fps"):
                            expected_array = labels if key == "labels" else inputs[key]
                            if not np.array_equal(archive[key], expected_array):
                                raise ValueError(f"Saved D2 archive changed paired native {key}")
                        decoded = prior.supervised.decode(probability,
                            prior.supervised.model_periods(inputs), float(inputs["fps"]), prior.lt.COST)
                        if not np.array_equal(archive["prediction"], decoded):
                            raise ValueError("Saved D2 prediction does not replay the official decoder")
                        member_probs[(arm, seed)] = probability
                        continue
                    probability = prior.old.upsample(sampled, record, inputs)
                    scoring._verify_probability(probability, len(labels),
                                               f"{arm}/{fold}/{seed}/{item['match']}/{view}")
                    prediction = prior.supervised.decode(probability,
                        prior.supervised.model_periods(inputs), float(inputs["fps"]), prior.lt.COST)
                    member_probs[(arm, seed)] = probability
                    output = _archive_path(item, view, replicate, str(seed), arm)
                    _save_prediction(output, probability, prediction, labels, inputs, item,
                                     view, replicate, fold, str(seed), arm, model_hash, input_hash)
            for arm in ARMS:
                for member in ("2026", "2027", "ensemble"):
                    if member == "ensemble":
                        probability = (member_probs[(arm, 2026)] + member_probs[(arm, 2027)]) / 2.0
                        seed_arg = "seedensemble"
                    else:
                        probability = member_probs[(arm, int(member))]
                        seed_arg = member
                    prediction = prior.supervised.decode(probability,
                        prior.supervised.model_periods(inputs), float(inputs["fps"]), prior.lt.COST)
                    model_paths = [study.candidate_model_path(fold, seed) if arm == "short_team_2x"
                                   else study.control_model_path(fold, seed)
                                   for seed in SEEDS] if member == "ensemble" else [
                                       study.candidate_model_path(fold, int(member)) if arm == "short_team_2x"
                                       else study.control_model_path(fold, int(member))]
                    model_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in model_paths}
                    model_hash = (hashlib.sha256(json.dumps(model_hashes, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest() if member == "ensemble"
                        else next(iter(model_hashes.values())))
                    output = _archive_path(item, view, replicate, seed_arg, arm)
                    if arm == "short_team_2x" and member == "ensemble":
                        _save_prediction(output, probability, prediction, labels, inputs, item,
                                         view, replicate, fold, seed_arg, arm,
                                         model_hash, input_hash)
                    elif arm == "control_D2" and member == "ensemble":
                        saved = _archive(output)
                        if (not np.array_equal(saved["probabilities"], probability) or
                                not np.array_equal(saved["prediction"], prediction)):
                            raise ValueError("Saved D2 ensemble is not the native member probability mean")
                        assert_native_arrays_equal(_archive(_archive_path(item, view, replicate,
                            "2026", arm)), saved)
                    cell = dict(source=item["source"], match=item["match"], view=view,
                                replicate=replicate, arm=arm, member=member)
                    row = scoring.score_prediction(labels, prediction, inputs, arm, cell,
                        scoring.digest_array(probability), sha256(output) if output.exists() else None)
                    row.update(fold=fold, model_sha256=model_hash,
                               evaluation_input_sha256=input_hash,
                               probabilities_sha256=scoring.digest_array(probability))
                    if member == "ensemble":
                        row.update(short_spell_metrics(labels, prediction, inputs))
                    rows.append(row)
                    files[str(output.relative_to(ROOT))] = sha256(output) if output.exists() else "reused_control"
                    if member == "ensemble":
                        frame = error_audit.frame_table(item["source"], item["match"], labels,
                            prediction, inputs["ball_observed"], inputs["period"],
                            inputs["timestamps"], float(inputs["fps"]))
                        frame["arm"] = arm
                        frame["view"] = view
                        frame["replicate"] = replicate
                        audit_frames.append(frame)
    expected_cohort = {(i["source"], i["match"], view, replicate)
                       for i in items for view, replicate in VIEWS}
    validate_complete_cell_cohort(rows, expected_cohort)
    # Equal-provider official readout and member seed spread.
    import pandas as pd
    private_frames = pd.concat(audit_frames, ignore_index=True)
    private_path = OUT / "private" / "short_team_audit_frames.csv"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    private_payload = private_frames.to_csv(index=False)
    if private_path.exists():
        if private_path.read_text() != private_payload:
            raise ValueError(f"Refusing to overwrite private frame audit: {private_path}")
    else:
        fd = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(private_payload)
            stream.flush()
            os.fsync(stream.fileno())
    audit = aggregate_error_cells(private_frames)
    summaries = {}
    for arm in ARMS:
        summaries[arm] = {}
        for view in ("clean", "feed_like"):
            selected = [r for r in rows if r["arm"] == arm and r["member"] == "ensemble" and r["view"] == view]
            summaries[arm][view] = prior.sc._aggregate_provider(selected, arm, view, "ensemble")
    seed_spread = {arm: {seed: {view: prior.sc._aggregate_provider(
        [r for r in rows if r["arm"] == arm and r["member"] == seed and r["view"] == view], arm, view, seed)
        for view in ("clean", "feed_like")} for seed in ("2026", "2027")} for arm in ARMS}
    seed_ranges = {arm: {view: {metric: max(seed_spread[arm][seed][view]["equal_provider_mean"][metric]
                                              for seed in ("2026", "2027")) -
                                      min(seed_spread[arm][seed][view]["equal_provider_mean"][metric]
                                              for seed in ("2026", "2027"))
                                      for metric in ("spell_f1", "macro_f1", "d_f1")}
                               for view in ("clean", "feed_like")} for arm in ARMS}
    guards = {}
    for view in ("clean", "feed_like"):
        candidate = summaries["short_team_2x"][view]
        control = summaries["control_D2"][view]
        cand_mean, ctrl_mean = candidate["equal_provider_mean"], control["equal_provider_mean"]
        providers = {}
        for provider in ("metrica", "skillcorner", "sportec"):
            cp, bp = candidate["per_provider"][provider], control["per_provider"][provider]
            providers[provider] = {"macro_f1_delta": cp["macro_f1"] - bp["macro_f1"],
                                   "d_f1_delta": cp["d_f1"] - bp["d_f1"]}
        guards[view] = {"spell_f1_delta": cand_mean["spell_f1"] - ctrl_mean["spell_f1"],
            "clean_spell_gain_at_least_0_01": (cand_mean["spell_f1"] - ctrl_mean["spell_f1"] >= .01)
                                                if view == "clean" else None,
            "stress_no_worse_than_0_01": (cand_mean["spell_f1"] - ctrl_mean["spell_f1"] >= -.01)
                                             if view == "feed_like" else None,
            "provider_macro_d_guard": {p: ({
                "macro_pass": d["macro_f1_delta"] >= -.02,
                "d_f1_pass": d["d_f1_delta"] >= -.02, **d} if view == "clean" else d)
                for p, d in providers.items()}}
    guards["all_pass"] = (
        guards["clean"]["clean_spell_gain_at_least_0_01"] and
        guards["feed_like"]["stress_no_worse_than_0_01"] and
        all(x["macro_pass"] and x["d_f1_pass"] for x in
            guards["clean"]["provider_macro_d_guard"].values()))
    result = {"schema": 1, "study": study.STUDY_ID, "seal_sha256": sha256(SEAL),
        "rows": rows, "prediction_files": files, "summaries": summaries,
        "seed_summaries": seed_spread, "seed_ranges": seed_ranges,
        "diagnostics": audit, "guards": guards,
        "public_or_organizer_labels_accessed": False, "private_frame_path": str(private_path.relative_to(ROOT)),
        "paired_cells_per_member": EXPECTED_CELLS_PER_MEMBER}
    write_once(RESULTS / "dense_short_team_v1_readout.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "evaluate"))
    args = parser.parse_args(argv)
    value = freeze_inputs() if args.command == "freeze" else evaluate()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
