"""Read-only opportunity audit of the saved combined-objective ensemble.

Scores the fixed external full-provider-holdout predictions with the unchanged
official scorer and summarizes aggregate frame/spell failure geometry. It does
not fit, infer, tune, read organizer assets, or emit per-frame outputs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from evaluation import possession_metrics as pm
from experiments import gap_diagnostics
from experiments.eval_set.protocol import sha256
from experiments.metrics import evaluate_tracking, prepare_tracking_codes
from experiments.spell_diagnostics import spell_diagnostics
from experiments import supervised

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/experiments/objective_ensemble_v1"
TRACKING = ROOT / "data/experiments/seed_gap_context_v1"
DECLARATION = ROOT / "experiments/results/spell_opportunity_audit_declaration_v2.json"
RESULT = ROOT / "experiments/results/spell_opportunity_audit.json"
COHORT = {
    "sportec": ("J03WOY",),
    "metrica": ("metrica_3",),
    "skillcorner": (
        "skillcorner_had_1996435",
        "skillcorner_had_1996436",
        "skillcorner_had_2006229",
    ),
}
MATCH_SOURCE = {match: source for source, matches in COHORT.items() for match in matches}
VIEWS = {"clean": ("clean",), "feed_like": ("corrupt_1", "corrupt_2")}
LABELS = ("H", "A", "D", "X")


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _cache_paths() -> tuple[list[Path], list[Path], list[Path]]:
    """Return the five raw inputs, fifteen joblibs, and fifteen candidate arrays."""
    objective_declaration = _json(PREDICTIONS / "declaration.json")
    raw = [ROOT / row["path"] for row in objective_declaration["evaluation_sources"].values()]
    joblibs = [
        TRACKING / f"{match}_{view}_{replicate}.joblib"
        for matches in COHORT.values()
        for match in matches
        for view, replicates in VIEWS.items()
        for replicate in replicates
    ]
    predictions = [
        PREDICTIONS / f"candidate_{match}_{view}_{replicate}.npz"
        for matches in COHORT.values()
        for match in matches
        for view, replicates in VIEWS.items()
        for replicate in replicates
    ]
    return raw, joblibs, predictions


def freeze_inputs() -> dict:
    """Seal the exact current evidence before any audit scoring is run."""
    if DECLARATION.exists():
        raise FileExistsError(f"Refusing to replace frozen receipt: {DECLARATION}")
    objective_declaration = _json(PREDICTIONS / "declaration.json")
    runs_path = PREDICTIONS / "runs.json"
    verification_path = PREDICTIONS / "verification.json"
    runs = _json(runs_path)
    verification = _json(verification_path)
    raw, joblibs, candidate_files = _cache_paths()

    if objective_declaration.get("study_id") != "objective_ensemble_v1":
        raise ValueError("Unexpected saved candidate study")
    if objective_declaration.get("cohort") != [
        {"match": match, "source": source}
        for source, matches in COHORT.items()
        for match in matches
    ]:
        raise ValueError("Saved candidate cohort differs from the fixed evaluation cohort")
    if objective_declaration.get("recipes", {}).get("candidate") != "combined_objective_equal_probability_ensemble":
        raise ValueError("Saved candidate is not the combined-objective probability ensemble")
    if not verification.get("checks", {}).get("candidate_ensemble_decode_prediction_digests_reproduced"):
        raise ValueError("Saved candidate decode parity receipt is missing")
    if not verification.get("checks", {}).get("baseline_probability_and_decode_digests_reproduced"):
        raise ValueError("Saved matched baseline parity receipt is missing")
    if not runs.get("baseline_parity", {}).get("all_probability_and_score_rows_reproduced"):
        raise ValueError("Saved paired comparison does not establish baseline parity")

    raw_records = objective_declaration["evaluation_sources"]
    eval_manifest = _json(ROOT / "experiments/eval_set/manifest.json")
    eval_records = {
        match["match"]: {"source": source, "path": match["path"], "sha256": match["sha256"]}
        for source, item in eval_manifest["sources"].items()
        for match in item["selection"]
    }
    seed_gap_inputs = _json(ROOT / "experiments/results/seed_gap_context.json")["spec"]["inputs_sha256"]
    raw_receipts = []
    for match, source in MATCH_SOURCE.items():
        record = raw_records.get(match)
        manifest_record = eval_records.get(match)
        seed_input = next((
            {"path": path, "sha256": digest}
            for path, digest in seed_gap_inputs.items() if Path(path).stem == match
        ), None)
        if (not record or record.get("source") != source or not manifest_record or
                {key: record[key] for key in ("source", "path", "sha256")} != manifest_record or
                seed_input != {"path": record["path"], "sha256": record["sha256"]}):
            raise ValueError(f"Missing source record for {match}")
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"External source artifact hash mismatch: {path}")
        raw_receipts.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})

    verified_archive = {row["path"]: row for row in objective_declaration["verified_archive_files"]}
    joblib_receipts = []
    for path in joblibs:
        rel = str(path.relative_to(ROOT / "data"))
        cache_record = verified_archive.get(rel)
        if cache_record is None:
            raise ValueError(f"No archived verification receipt for input snapshot: {rel}")
        if not path.is_file() or path.stat().st_size != cache_record["bytes"] or sha256(path) != cache_record["sha256"]:
            raise ValueError(f"Input snapshot hash mismatch: {path}")
        joblib_receipts.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})

    expected_candidate_rows = [
        row for row in runs["rows"] if row.get("recipe") == "combined_objective_equal_probability_ensemble"
    ]
    candidate_by_path = {row["probability_file"]: row for row in expected_candidate_rows}
    if len(expected_candidate_rows) != 15 or len(candidate_by_path) != 15:
        raise ValueError("Expected exactly one saved candidate output for each of fifteen cells")
    candidate_receipts = []
    for path in candidate_files:
        row = candidate_by_path.get(path.name)
        if row is None:
            raise ValueError(f"Candidate output is not in the saved paired report: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        expected_hash = row.get("probability_file_sha256")
        if expected_hash and sha256(path) != expected_hash:
            raise ValueError(f"Candidate archive hash mismatch: {path}")
        candidate_receipts.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size})

    source_paths = [
        Path(__file__), ROOT / "evaluation/possession_metrics.py", ROOT / "experiments/metrics.py",
        ROOT / "experiments/spell_diagnostics.py", ROOT / "experiments/gap_diagnostics.py",
        ROOT / "experiments/possession_geometry.py", ROOT / "experiments/supervised.py",
        ROOT / "experiments/model_decision_policy.json", ROOT / "experiments/baseline_manifest.json",
        ROOT / "experiments/eval_set/manifest.json", ROOT / "experiments/seed_gap_context.py",
    ]
    result = {
        "schema_version": 1,
        "study_id": "spell_opportunity_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Which observed failure mechanism has the largest supported spell-F1 headroom in the saved combined-objective two-seed ensemble?",
        "scope": "Post-hoc diagnostic of the five historically exposed full-provider-holdout external matches; clean and two fixed feed-like replicates.",
        "prior_attempt": {
            "status": "aborted before result write",
            "reason": "The first implementation looked for n_gt under a per-IoU diagnostic object; the official diagnostics expose that count at the top level.",
            "metric_values_reviewed": False,
            "result_artifact_written": False,
            "original_receipt_retained": False,
            "corrective_action": "Freeze this corrected source and inputs under declaration v2 before rerunning.",
        },
        "candidate": "combined_objective_equal_probability_ensemble",
        "matched_control": "current_baseline_equal_probability_ensemble from the sealed objective_ensemble_v1 rows, used only for parity and matched-cell context",
        "official_metric": "Unchanged evaluate_tracking scorer; global IoU-descending greedy H/A assignment at .3/.4/.5/.6/.7.",
        "cohort": [{"source": source, "matches": list(matches)} for source, matches in COHORT.items()],
        "views": {key: list(value) for key, value in VIEWS.items()},
        "oracle": "none",
        "organizer_assets_read": False,
        "model_fitting_or_model_inference": False,
        "saved_baseline_decoder_replayed_for_parity": True,
        "tuning_or_selection": False,
        "aggregate_only": True,
        "active_baseline_manifest_sha256": sha256(ROOT / "experiments/baseline_manifest.json"),
        "active_policy_sha256": sha256(ROOT / "experiments/model_decision_policy.json"),
        "objective_comparison_receipts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PREDICTIONS / "declaration.json", runs_path, verification_path, PREDICTIONS / "report.json",
                         ROOT / "experiments/results/seed_gap_context.json")
        },
        "external_input_receipts": raw_receipts,
        "matched_tracking_snapshot_receipts": joblib_receipts,
        "candidate_output_receipts": candidate_receipts,
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def _assert_frozen(declaration: dict) -> None:
    for path, expected in declaration["source_sha256"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Audit source changed after freeze: {path}")
    for group in ("external_input_receipts", "matched_tracking_snapshot_receipts", "candidate_output_receipts"):
        for row in declaration[group]:
            path = ROOT / row["path"]
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"Frozen input changed: {path}")
    for path, expected in declaration["objective_comparison_receipts"].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f"Saved comparison receipt changed: {path}")


def _frame_errors(labels: np.ndarray, prediction: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(str)
    prediction = np.asarray(prediction).astype(str)
    if labels.shape != prediction.shape or labels.ndim != 1:
        raise ValueError("Expected aligned one-dimensional labels and predictions")
    if not np.isin(labels, LABELS).all() or not np.isin(prediction, LABELS).all():
        raise ValueError("Frame labels must be H/A/D/X")
    cm = pm.confusion(pm.label_codes(labels), pm.label_codes(prediction))
    live = np.isin(labels, ("H", "A"))
    dead = labels == "D"
    wrong = live & np.isin(prediction, ("H", "A")) & (labels != prediction)
    false_d = live & (prediction == "D")
    abstain_live = live & (prediction == "X")
    missed_d = dead & (prediction != "D")
    return {
        "confusion_rows_truth_cols_prediction": {
            truth: {guess: int(cm[i, j]) for j, guess in enumerate(LABELS)}
            for i, truth in enumerate(LABELS[:3])
        },
        "live": {
            "frames": int(live.sum()),
            "correct_team": int((live & (labels == prediction)).sum()),
            "wrong_team": int(wrong.sum()),
            "false_D": int(false_d.sum()),
            "abstained": int(abstain_live.sum()),
            "wrong_team_rate": float(wrong.sum() / live.sum()) if live.any() else None,
            "false_D_rate": float(false_d.sum() / live.sum()) if live.any() else None,
            "abstained_rate": float(abstain_live.sum() / live.sum()) if live.any() else None,
        },
        "dead": {
            "frames": int(dead.sum()),
            "correct_D": int((dead & (prediction == "D")).sum()),
            "missed_D": int(missed_d.sum()),
            "missed_to_H": int((dead & (prediction == "H")).sum()),
            "missed_to_A": int((dead & (prediction == "A")).sum()),
            "missed_to_X": int((dead & (prediction == "X")).sum()),
            "missed_D_rate": float(missed_d.sum() / dead.sum()) if dead.any() else None,
        },
    }


def _unmatched_spell_geometry(labels: np.ndarray, prediction: np.ndarray,
                              period: np.ndarray, timestamps: np.ndarray,
                              fps: float) -> dict:
    """Describe unmatched reference spans after the official global IoU=.5 assignment."""
    _, truth, guess, _ = prepare_tracking_codes(labels, prediction, period, fps, timestamps)
    runs = pm.valid_runs(truth)
    filled = pm.fill_abstentions(guess, runs)
    gt_spells = pm.possession_spells(truth, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(gt_spells), pm.SPELL_IOU_HEADLINE)
    overlap_detail: list[tuple[bool, bool]] = []
    dominant: dict[str, int] = {"wrong_team": 0, "false_D": 0, "abstained": 0, "same_team_or_boundary": 0, "mixed_tie": 0}
    for gi, (start, stop, team) in enumerate(gt_spells):
        same_team_overlap = False
        other_team_overlap = False
        for pstart, pstop, pteam in pred_spells:
            if max(start, pstart) < min(stop, pstop):
                if pteam == team:
                    same_team_overlap = True
                else:
                    other_team_overlap = True
        overlap_detail.append((same_team_overlap, other_team_overlap))
        if matched[gi]:
            continue
        window_truth = truth[start:stop]
        window_guess = guess[start:stop]
        live = (window_truth == pm.H) | (window_truth == pm.A)
        errors = {
            "wrong_team": int((((window_truth == pm.H) & (window_guess == pm.A)) |
                               ((window_truth == pm.A) & (window_guess == pm.H))).sum()),
            "false_D": int((live & (window_guess == pm.D)).sum()),
            "abstained": int((live & (window_guess == pm.X)).sum()),
        }
        maximum = max(errors.values(), default=0)
        winners = [name for name, count in errors.items() if count == maximum and count > 0]
        category = winners[0] if len(winners) == 1 else "mixed_tie" if winners else "same_team_or_boundary"
        dominant[category] += 1
    return {
        "unmatched_reference_spells_at_iou_0_5": int((~matched).sum()),
        "unmatched_with_any_same_team_alive_overlap": int(sum((not matched[i]) and same for i, (same, _) in enumerate(overlap_detail))),
        "unmatched_with_other_team_overlap": int(sum((not matched[i]) and other for i, (_, other) in enumerate(overlap_detail))),
        "unmatched_without_same_team_alive_overlap": int(sum((not matched[i]) and not same for i, (same, _) in enumerate(overlap_detail))),
        "unmatched_dominant_frame_error_within_reference_span": dominant,
        "interpretation": "Overlap and dominant-frame-error descriptions are descriptive and may overlap; they are not exclusive causal labels.",
    }


def _score_parity(score: dict, saved: dict) -> None:
    fields = {
        "spell_f1_mean": "spell_f1",
        "macro_f1": "macro_f1",
        "n_gt_spells": "n_gt_spells",
        "n_pred_spells": "n_pred_spells",
    }
    for actual, expected in fields.items():
        if actual in ("n_gt_spells", "n_pred_spells"):
            if int(score[actual]) != int(saved[expected]):
                raise ValueError(f"Saved score parity failed: {actual}")
        elif not np.isclose(score[actual], saved[expected], atol=1e-12, rtol=0):
            raise ValueError(f"Saved score parity failed: {actual}")
    for key, label in (("d_f1", "f1"), ("d_precision", "precision"), ("d_recall", "recall")):
        if not np.isclose(score["per_class"]["D"][label], saved[key], atol=1e-12, rtol=0):
            raise ValueError(f"Saved score parity failed: D {label}")
    for threshold in pm.SPELL_IOUS:
        if not np.isclose(score["spell_f1"][threshold]["f1"], saved["spell_by_iou"][str(threshold)], atol=1e-12, rtol=0):
            raise ValueError(f"Saved score parity failed: IoU {threshold}")


def _one_cell(match: str, view: str, replicate: str, objective_runs: list[dict]) -> dict:
    source = MATCH_SOURCE[match]
    bundle = joblib.load(TRACKING / f"{match}_{view}_{replicate}.joblib")
    inputs, labels = bundle["inputs"], np.asarray(bundle["labels"]).astype(str)
    pred_path = PREDICTIONS / f"candidate_{match}_{view}_{replicate}.npz"
    with np.load(pred_path, allow_pickle=False) as archive:
        prediction = np.asarray(archive["prediction"]).astype(str)
        probability = np.asarray(archive["probability"], dtype=np.float64)
    if labels.shape != prediction.shape or len(prediction) != len(inputs["period"]):
        raise ValueError(f"Prediction/input length mismatch: {match}/{replicate}")
    if probability.shape != (len(labels), 3) or not np.isfinite(probability).all():
        raise ValueError(f"Saved categorical probability array is malformed: {match}/{replicate}")
    if not np.isin(labels, ("H", "A", "D", "X")).all() or not np.isin(prediction, ("H", "A", "D", "X")).all():
        raise ValueError(f"Invalid saved labels: {match}/{replicate}")
    candidate_rows = [r for r in objective_runs if r.get("match") == match and r.get("view") == view and
                      r.get("replicate") == replicate and r.get("recipe") == "combined_objective_equal_probability_ensemble"]
    baseline_rows = [r for r in objective_runs if r.get("match") == match and r.get("view") == view and
                     r.get("replicate") == replicate and r.get("recipe") == "current_baseline_equal_probability_ensemble"]
    if len(candidate_rows) != 1 or len(baseline_rows) != 1:
        raise ValueError(f"Missing/duplicate paired score row: {match}/{replicate}")
    candidate_row, baseline_row = candidate_rows[0], baseline_rows[0]
    if candidate_row["probability_file"] != pred_path.name or candidate_row["prediction_sha256"] != digest_array(prediction):
        raise ValueError(f"Saved combined-objective prediction receipt mismatch: {match}/{replicate}")
    if candidate_row["probability_sha256"] != digest_array(probability):
        raise ValueError(f"Saved combined-objective probability receipt mismatch: {match}/{replicate}")
    # Recreate only the saved old-baseline decode from frozen probabilities. This
    # proves the clean/feed-like cells and grading inputs match the paired report.
    baseline_archive = ROOT / "data" / baseline_row["probability_file"]
    with np.load(baseline_archive, allow_pickle=False) as archive:
        if "equal_probability_ensemble" not in archive:
            raise ValueError(f"Missing saved paired baseline probability: {baseline_archive}")
        baseline_probability = np.asarray(archive["equal_probability_ensemble"], dtype=np.float64)
    if digest_array(baseline_probability) != baseline_row["probability_sha256"]:
        raise ValueError(f"Saved baseline probability receipt mismatch: {match}/{replicate}")
    baseline_prediction = supervised.decode(
        baseline_probability, supervised.model_periods(inputs), float(inputs["fps"]), 3
    ).astype(str)
    if digest_array(baseline_prediction) != baseline_row["prediction_sha256"]:
        raise ValueError(f"Saved baseline prediction parity failed: {match}/{replicate}")

    fps = float(inputs["fps"])
    score = evaluate_tracking(labels, prediction, inputs["period"], fps, inputs["timestamps"])
    _score_parity(score, candidate_row)
    spells = spell_diagnostics(labels, prediction, inputs["period"], fps, inputs["timestamps"])
    if not np.isclose(spells["spell_f1_mean"], score["spell_f1_mean"], atol=1e-12, rtol=0):
        raise ValueError(f"Spell diagnostic parity failed: {match}/{replicate}")
    geometry = _unmatched_spell_geometry(labels, prediction, inputs["period"], inputs["timestamps"], fps)
    tracking = gap_diagnostics.analyze(inputs, labels, prediction)
    edge = spells["per_iou"][str(pm.SPELL_IOU_HEADLINE)]["reference_recall_groups"]["touches_graded_run_edge"]
    return {
        "source": source,
        "match": match,
        "view": view,
        "replicate": replicate,
        "graded_frames": int(np.isin(labels, ("H", "A", "D")).sum()),
        "baseline_prediction_sha256": baseline_row["prediction_sha256"],
        "candidate_prediction_sha256": candidate_row["prediction_sha256"],
        "candidate_probability_sha256": candidate_row["probability_sha256"],
        "candidate_file_sha256": sha256(pred_path),
        "score": {
            "spell_f1": float(score["spell_f1_mean"]),
            "macro_f1": float(score["macro_f1"]),
            "d_f1": float(score["per_class"]["D"]["f1"]),
            "d_precision": float(score["per_class"]["D"]["precision"]),
            "d_recall": float(score["per_class"]["D"]["recall"]),
            "n_gt_spells": int(score["n_gt_spells"]),
            "n_pred_spells": int(score["n_pred_spells"]),
            "per_iou_f1": {str(t): float(score["spell_f1"][t]["f1"]) for t in pm.SPELL_IOUS},
            "short_spell_recall_under_3s_at_iou_0_5": spells["per_iou"][str(pm.SPELL_IOU_HEADLINE)]["reference_recall_groups"]["under_3_seconds"]["recall"],
        },
        "frame_errors": _frame_errors(labels, prediction),
        "spells": {
            "n_gt_spells": int(spells["n_gt_spells"]),
            "n_pred_spells": int(spells["n_pred_spells"]),
            "spell_count_error": int(spells["spell_count_error"]),
            "n_short_reference_spells_under_3s": int(spells["per_iou"]["0.5"]["reference_recall_groups"]["under_3_seconds"]["n_gt"]),
            "n_short_matched_at_iou_0_5": int(spells["per_iou"]["0.5"]["reference_recall_groups"]["under_3_seconds"]["n_matched"]),
            "iou_0_5_unmatched_reference_spells": int(spells["n_gt_spells"] - spells["per_iou"]["0.5"]["n_matched"]),
            "miss_geometry": geometry,
            "overlap_geometry": spells["overlap_geometry"],
            "complete_spell_boundary_errors_at_iou_0_5": spells["matched_complete_boundary_errors_seconds_at_iou_0_5"],
            "graded_run_edge": {
                "n_reference_spells": int(edge["n_gt"]),
                "n_matched_at_iou_0_5": int(edge["n_matched"]),
                "recall_at_iou_0_5": edge["recall"],
                "fraction_of_reference_spells": float(edge["n_gt"] / spells["n_gt_spells"]) if spells["n_gt_spells"] else None,
                "interpretation": "Boundary endpoint is censored by a graded-run edge; headline scoring still includes the observed span.",
            },
        },
        "gap_contact": tracking,
        "official_scorer_parity": True,
        "matched_saved_baseline_parity": True,
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _balanced_summary(rows: list[dict], view: str) -> dict:
    selected = [row for row in rows if row["view"] == view]
    keys = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall", "short_spell_recall_under_3s_at_iou_0_5")
    provider = {}
    for source, matches in COHORT.items():
        source_rows = [row for row in selected if row["source"] == source]
        match_rows = {}
        for match in matches:
            cells = [row for row in source_rows if row["match"] == match]
            if not cells:
                raise ValueError(f"Missing cohort match in summary: {source}/{match}/{view}")
            match_rows[match] = {key: _mean([float(cell["score"][key]) for cell in cells]) for key in keys}
        provider[source] = {
            key: _mean([value[key] for value in match_rows.values() if value[key] is not None])
            for key in keys
        }
        provider[source]["matches"] = match_rows
    equal = {key: _mean([provider[source][key] for source in COHORT if provider[source][key] is not None]) for key in keys}
    equal["per_provider"] = {source: {key: provider[source][key] for key in keys} for source in COHORT}

    def summarize_frame_strata() -> dict:
        section = "frames"
        fields = ("wrong_team_rate", "false_D_rate", "abstained_live_rate", "missed_D_rate")
        names = sorted({name for row in selected for name in row["gap_contact"][section]})
        output = {}
        for name in names:
            per_source = {}
            for source, matches in COHORT.items():
                match_values = []
                for match in matches:
                    cells = [row for row in selected if row["source"] == source and row["match"] == match]
                    rates_by_field = {field: [] for field in fields}
                    support = []
                    for cell in cells:
                        counts = cell["gap_contact"][section][name]
                        support.append(int(counts.get("live_frames", 0)))
                        for field in rates_by_field:
                            numerator_name = {"wrong_team_rate": "wrong_team", "false_D_rate": "false_dead",
                                              "abstained_live_rate": "abstained_live", "missed_D_rate": "missed_dead"}[field]
                            denom_name = "dead_frames" if field == "missed_D_rate" else "live_frames"
                            rates_by_field[field].append(_ratio(counts.get(numerator_name, 0), counts.get(denom_name, 0)))
                    value = {field: _mean([x for x in rates_by_field[field] if x is not None]) for field in fields}
                    value["mean_support_per_replicate"] = _mean(support)
                    match_values.append(value)
                per_source[source] = {
                    field: _mean([value[field] for value in match_values if value[field] is not None])
                    for field in fields
                }
                per_source[source]["matches"] = {match: match_values[i] for i, match in enumerate(matches)}
            output[name] = {
                "equal_provider_mean": {
                    field: _mean([per_source[source][field] for source in COHORT if per_source[source][field] is not None])
                    for field in fields
                },
                "per_provider": per_source,
            }
        return output

    def summarize_spell_strata() -> dict:
        section = "spells"
        names = sorted({name for row in selected for name in row["gap_contact"][section]})
        output = {}
        for name in names:
            per_source = {}
            for source, matches in COHORT.items():
                per_match = {}
                for match in matches:
                    cells = [row for row in selected if row["source"] == source and row["match"] == match]
                    cell_values = []
                    for cell in cells:
                        counts = cell["gap_contact"][section][name]
                        n_ref = int(counts["reference_spells"])
                        n_short = int(counts["short_reference"])
                        cell_values.append({
                            "reference_spells": n_ref,
                            "missed_rate": _ratio(counts["missed"], n_ref),
                            "fragmented_rate": _ratio(counts["fragmented"], n_ref),
                            "short_recall": _ratio(n_short - counts["short_missed"], n_short),
                        })
                    per_match[match] = {
                        field: _mean([value[field] for value in cell_values if value[field] is not None])
                        for field in ("missed_rate", "fragmented_rate", "short_recall")
                    }
                    per_match[match]["mean_reference_spells_per_replicate"] = _mean(
                        [value["reference_spells"] for value in cell_values]
                    )
                per_source[source] = {
                    field: _mean([per_match[match][field] for match in matches if per_match[match][field] is not None])
                    for field in ("missed_rate", "fragmented_rate", "short_recall")
                }
                per_source[source]["matches"] = per_match
            output[name] = {
                "equal_provider_mean": {
                    field: _mean([per_source[source][field] for source in COHORT if per_source[source][field] is not None])
                    for field in ("missed_rate", "fragmented_rate", "short_recall")
                },
                "per_provider": per_source,
            }
        return output

    gap_contact = {"frames": summarize_frame_strata(), "spells": summarize_spell_strata()}
    return {"equal_provider_mean": equal, "gap_contact": gap_contact}


def run_audit(declaration: dict) -> dict:
    _assert_frozen(declaration)
    objective_runs = _json(PREDICTIONS / "runs.json")["rows"]
    rows = []
    for source, matches in COHORT.items():
        for match in matches:
            for view, replicates in VIEWS.items():
                for replicate in replicates:
                    rows.append(_one_cell(match, view, replicate, objective_runs))
    if len(rows) != 15 or not all(row["official_scorer_parity"] and row["matched_saved_baseline_parity"] for row in rows):
        raise ValueError("Incomplete or unmatched saved prediction audit")
    summaries = {view: _balanced_summary(rows, view) for view in VIEWS}
    return {
        "schema_version": 1,
        "study_id": declaration["study_id"],
        "declaration_sha256": sha256(DECLARATION),
        "official_scorer_sha256": sha256(ROOT / "evaluation/possession_metrics.py"),
        "organizer_assets_read": False,
        "model_fitting_or_model_inference": False,
        "saved_baseline_decoder_replayed_for_parity": True,
        "tuning_or_selection": False,
        "oracle": "none",
        "scope": declaration["scope"],
        "aggregation": "Average feed-like replicates within match, matches within provider, then equal provider mean; frame/spell category rates are descriptive.",
        "completed_cells": len(rows),
        "parity": {
            "combined_candidate_score_rows_reproduced": len(rows),
            "saved_previous_baseline_decode_rows_reproduced": len(rows),
            "reference_spell_counts_unchanged": True,
            "official_scorer_used": True,
        },
        "summaries": summaries,
        "rows": rows,
        "limitations": [
            "All five external matches were previously exposed development data; no independent confirmation is available.",
            "Metrica and SkillCorner labels are event-derived silver references; provider label semantics differ.",
            "Gap/contact contact is a tracking-derived geometry proxy, not possession truth or a causal variable.",
            "Error-category repairability and model response are not estimated; no oracle correction or expected gain is reported.",
            "Graded-run-edge spells remain in official scoring, but their cut endpoint is censored for timing interpretation.",
            "An initial implementation attempt aborted before writing results; the corrected audit uses this separately frozen v2 declaration.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="write the immutable pre-run input/source receipt")
    mode.add_argument("--run", action="store_true", help="run the read-only audit using the frozen receipt")
    args = parser.parse_args()
    if args.freeze:
        receipt = freeze_inputs()
        print(json.dumps({"declaration": str(DECLARATION.relative_to(ROOT)), "input_files": sum(len(receipt[key]) for key in (
            "external_input_receipts", "matched_tracking_snapshot_receipts", "candidate_output_receipts"))}, indent=2))
        return
    if not DECLARATION.is_file():
        raise FileNotFoundError(f"Freeze the audit inputs first with --freeze: {DECLARATION}")
    if RESULT.exists():
        raise FileExistsError(f"Refusing to overwrite completed diagnostic: {RESULT}")
    result = run_audit(_json(DECLARATION))
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
