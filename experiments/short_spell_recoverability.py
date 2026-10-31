"""Read-only emission-versus-decoder audit for short possession misses.

The raw per-frame argmax is a diagnostic counterfactual, not a deployable
prediction. The saved cost-3 path remains the candidate path and the official
global spell matcher defines every miss cohort. No model is fitted or inferred.
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
from experiments.eval_set.protocol import sha256
from experiments.metrics import evaluate_tracking, prepare_tracking_codes
from experiments import supervised

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "data/experiments/objective_ensemble_v1"
TRACKING = ROOT / "data/experiments/seed_gap_context_v1"
PRIOR_DECLARATION = ROOT / "experiments/results/spell_opportunity_audit_declaration_v2.json"
PRIOR_RESULT = ROOT / "experiments/results/spell_opportunity_audit.json"
ABORTED_DECLARATION = ROOT / "experiments/results/short_spell_recoverability_declaration.json"
DECLARATION = ROOT / "experiments/results/short_spell_recoverability_v2_declaration.json"
RESULT = ROOT / "experiments/results/short_spell_recoverability_v2.json"

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
DURATION_BINS = (
    ("under_0_5s", 0.0, 0.5),
    ("0_5_to_under_1s", 0.5, 1.0),
    ("1_to_under_2s", 1.0, 2.0),
    ("2_to_under_3s", 2.0, 3.0),
    ("3s_or_longer", 3.0, float("inf")),
)
MECHANISMS = (
    "viterbi_suppressed_correct_emission",
    "correct_emission_but_timing_or_duration_insufficient",
    "no_sufficient_local_correct_team_emission",
    "global_assignment_conflict",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _source_paths() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "evaluation/possession_metrics.py",
        ROOT / "experiments/metrics.py",
        ROOT / "experiments/supervised.py",
        ROOT / "experiments/eval_set/manifest.json",
        ROOT / "experiments/baseline_manifest.json",
        ROOT / "experiments/model_decision_policy.json",
        PREDICTIONS / "declaration.json",
        PREDICTIONS / "runs.json",
        PREDICTIONS / "verification.json",
        PREDICTIONS / "report.json",
        PRIOR_DECLARATION,
        PRIOR_RESULT,
    ]


def _cell_paths() -> tuple[list[Path], list[Path]]:
    joblibs, predictions = [], []
    for matches in COHORT.values():
        for match in matches:
            for reps in VIEWS.values():
                for replicate in reps:
                    joblibs.append(TRACKING / f"{match}_{'clean' if replicate == 'clean' else 'feed_like'}_{replicate}.joblib")
                    predictions.append(PREDICTIONS / f"candidate_{match}_{'clean' if replicate == 'clean' else 'feed_like'}_{replicate}.npz")
    return joblibs, predictions


def freeze_inputs() -> dict:
    """Write an immutable receipt before the first metric is computed."""
    if DECLARATION.exists() or RESULT.exists():
        raise FileExistsError("Refusing to replace a frozen declaration or completed audit")
    objective = _json(PREDICTIONS / "declaration.json")
    runs = _json(PREDICTIONS / "runs.json")
    verification = _json(PREDICTIONS / "verification.json")
    prior = _json(PRIOR_DECLARATION)
    prior_result = _json(PRIOR_RESULT)
    if objective.get("study_id") != "objective_ensemble_v1":
        raise ValueError("Unexpected saved candidate study")
    expected_cohort = [{"match": match, "source": source}
                       for source, matches in COHORT.items() for match in matches]
    if objective.get("cohort") != expected_cohort:
        raise ValueError("Saved candidate cohort differs from the fixed external cohort")
    if objective.get("recipes", {}).get("candidate") != "combined_objective_equal_probability_ensemble":
        raise ValueError("Saved outputs are not the adopted combined-objective ensemble")
    if not verification.get("checks", {}).get("candidate_ensemble_decode_prediction_digests_reproduced"):
        raise ValueError("Candidate probability/decode receipt is incomplete")
    if prior_result.get("completed_cells") != 15 or prior_result.get("parity", {}).get("official_scorer_used") is not True:
        raise ValueError("Prior matched baseline audit did not verify the 15 fixed cells")
    if len(prior.get("matched_tracking_snapshot_receipts", [])) != 15 or len(prior.get("candidate_output_receipts", [])) != 15:
        raise ValueError("Prior frozen receipt does not cover all tracking snapshots and predictions")

    verified = {row["path"]: row for row in objective["verified_archive_files"]}
    joblibs, predictions = _cell_paths()
    snapshot_receipts = []
    for path in joblibs:
        record = verified.get(str(path.relative_to(ROOT / "data")))
        if record is None or not path.is_file():
            raise ValueError(f"Missing verified tracking/label snapshot: {path}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Tracking/label snapshot differs from its archive receipt: {path}")
        snapshot_receipts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                  "sha256": sha256(path)})

    candidate_rows = [row for row in runs["rows"]
                      if row.get("recipe") == "combined_objective_equal_probability_ensemble"]
    by_file = {row["probability_file"]: row for row in candidate_rows}
    if len(by_file) != 15:
        raise ValueError("Expected exactly fifteen frozen candidate cells")
    candidate_receipts = []
    for path in predictions:
        row = by_file.get(path.name)
        if row is None or not path.is_file() or sha256(path) != row["probability_file_sha256"]:
            raise ValueError(f"Candidate archive is absent or differs from the saved score receipt: {path}")
        candidate_receipts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                                   "sha256": sha256(path)})

    raw_records = objective["evaluation_sources"]
    raw_inputs = []
    for match, record in raw_records.items():
        if match not in MATCH_SOURCE:
            continue
        path = ROOT / record["path"]
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"External source input differs from the saved candidate receipt: {path}")
        raw_inputs.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                           "sha256": sha256(path)})

    config = {
        "cohort": expected_cohort,
        "views": {view: list(reps) for view, reps in VIEWS.items()},
        "official_iou": pm.SPELL_IOU_HEADLINE,
        "official_matcher": "unchanged IoU-descending greedy one-to-one assignment over all same-team spell pairs before subgroup filtering",
        "duration_bins_seconds": [[name, lo, hi if np.isfinite(hi) else None]
                                   for name, lo, hi in DURATION_BINS],
        "primary_cohort": "officially unmatched, complete observed reference spells shorter than 3 seconds",
        "edge_rule": "complete iff neither interval endpoint touches a valid graded-run boundary",
        "visibility_rule": "visible_ball iff every native frame in the reference span has a finite observed ball coordinate; any other span is has_ball_gap",
        "emission_rule": "native saved H/A/D probability argmax, fixed class order H/A/D; no threshold or fit",
        "decoder_rule": "saved adopted cost-3 path; no decode replay or parameter change",
        "mechanism_partition": {
            MECHANISMS[0]: "at least one correct-team raw argmax frame inside the reference span and zero correct-team cost-3 frames inside it",
            MECHANISMS[1]: "some correct-team raw argmax, cost-3 overlap, or raw same-team overlap exists, but neither decoder suppression nor global conflict applies",
            MECHANISMS[2]: "no correct-team raw argmax frame or same-team raw/cost-3 overlap inside the reference span",
            MECHANISMS[3]: "cost-3 has a same-team candidate spell with IoU >= 0.5, but official global one-to-one assignment leaves the reference unmatched",
        },
        "raw_argmax_status": "diagnostic counterfactual only; never a candidate for adoption",
        "probability_sum_tolerance": 2e-7,
    }
    declaration = {
        "schema_version": 1,
        "study_id": "short_spell_recoverability_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Among globally matched cost-3 misses, how often do native per-frame emissions contain the correct team and how often did the decoder or interval timing remove it?",
        "scope": "Read-only diagnostic on five previously exposed full-provider-holdout development matches, one clean and two fixed feed-like views.",
        "candidate": "joint_gru_combined_ensemble_v1 archived probabilities and cost-3 predictions",
        "organizer_assets_read": False,
        "model_fitting_or_inference": False,
        "threshold_sweeps_or_candidate_selection": False,
        "reference_informed_prediction": False,
        "official_scorer": "evaluation.possession_metrics; existing evaluate_tracking wrapper and unchanged global matcher",
        "active_baseline_manifest_sha256": sha256(ROOT / "experiments/baseline_manifest.json"),
        "active_policy_sha256": sha256(ROOT / "experiments/model_decision_policy.json"),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()},
        "input_receipts": {
            "external_tracking_sources": raw_inputs,
            "tracking_and_label_snapshots": snapshot_receipts,
            "saved_candidate_probabilities_and_paths": candidate_receipts,
        },
        "prior_audit_receipts": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (PRIOR_DECLARATION, PRIOR_RESULT, PREDICTIONS / "declaration.json",
                         PREDICTIONS / "runs.json", PREDICTIONS / "verification.json",
                         PREDICTIONS / "report.json")
        },
        "previous_attempt": {
            "declaration": str(ABORTED_DECLARATION.relative_to(ROOT)),
            "sha256": sha256(ABORTED_DECLARATION),
            "status": "aborted during input validation after float32 probability-sum rounding slightly exceeded 1e-7; no result artifact was written or reviewed",
        },
        "configuration": config,
        "limitations": [
            "Every evaluation match has prior development exposure; results are descriptive and do not establish transfer.",
            "Metrica and SkillCorner references are event-derived silver labels; boundary censoring can affect spell timing.",
            "The raw argmax path is a diagnostic counterfactual and is not a deployable alternative.",
            "No reference-informed repair or oracle prediction is computed.",
        ],
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(declaration, indent=2, allow_nan=False) + "\n")
    return declaration


def _assert_frozen(declaration: dict) -> None:
    if declaration.get("study_id") != "short_spell_recoverability_v2":
        raise ValueError("Unexpected frozen audit declaration")
    for path, digest in declaration["source_sha256"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Audit source changed after freeze: {path}")
    for group in declaration["input_receipts"].values():
        for row in group:
            path = ROOT / row["path"]
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError(f"Frozen audit input changed: {path}")
    for path, digest in declaration["prior_audit_receipts"].items():
        if sha256(ROOT / path) != digest:
            raise ValueError(f"Saved candidate or matched audit receipt changed: {path}")
    if sha256(ROOT / declaration["previous_attempt"]["declaration"]) != declaration["previous_attempt"]["sha256"]:
        raise ValueError("Aborted preliminary declaration changed after the corrected freeze")
    if sha256(ROOT / "experiments/baseline_manifest.json") != declaration["active_baseline_manifest_sha256"]:
        raise ValueError("Active baseline manifest changed after freeze")
    if sha256(ROOT / "experiments/model_decision_policy.json") != declaration["active_policy_sha256"]:
        raise ValueError("Active reporting policy changed after freeze")


def _duration_bin(seconds: float) -> str:
    for name, lower, upper in DURATION_BINS:
        if lower <= seconds < upper:
            return name
    raise ValueError(f"Invalid spell duration: {seconds}")


def _original_span(start: int, stop: int, inserts: np.ndarray) -> tuple[int, int]:
    """Map a non-separator half-open span from scorer coordinates to native frames."""
    original_start = start - int(np.searchsorted(inserts, start, side="right"))
    original_stop = stop - int(np.searchsorted(inserts, stop, side="left"))
    return original_start, original_stop


def _nearest_emission_distance(team: int, start: int, stop: int,
                               run_start: int, run_stop: int, raw_codes: np.ndarray,
                               fps: float) -> float | None:
    indices = np.flatnonzero(raw_codes[run_start:run_stop] == team) + run_start
    if not len(indices):
        return None
    if np.any((indices >= start) & (indices < stop)):
        return 0.0
    frame_distance = np.where(indices < start, start - indices, indices - stop + 1).min()
    return float(frame_distance / fps)


def _classify_miss(*, candidate_matched: bool, candidate_best_iou: float,
                   raw_correct_frames: int, decoded_correct_frames: int,
                   raw_best_iou: float) -> str:
    if candidate_matched:
        raise ValueError("Only official candidate misses can be classified")
    if candidate_best_iou >= pm.SPELL_IOU_HEADLINE:
        return "global_assignment_conflict"
    if raw_correct_frames > 0 and decoded_correct_frames == 0:
        return "viterbi_suppressed_correct_emission"
    if raw_correct_frames > 0 or decoded_correct_frames > 0 or raw_best_iou > 0:
        return "correct_emission_but_timing_or_duration_insufficient"
    return "no_sufficient_local_correct_team_emission"


def _best_iou_by_gt(overlaps: list[tuple[float, int, int]], n_gt: int) -> np.ndarray:
    best = np.zeros(n_gt, dtype=float)
    for iou, gi, _ in overlaps:
        best[gi] = max(best[gi], float(iou))
    return best


def _score_parity(score: dict, saved: dict) -> None:
    for key in ("spell_f1", "macro_f1", "n_gt_spells", "n_pred_spells"):
        actual = score["spell_f1_mean"] if key == "spell_f1" else score[key]
        expected = saved[key]
        if key in ("n_gt_spells", "n_pred_spells"):
            if int(actual) != int(expected):
                raise ValueError(f"Candidate score parity failed for {key}")
        elif not np.isclose(actual, expected, atol=1e-12, rtol=0):
            raise ValueError(f"Candidate score parity failed for {key}")


def _cell(match: str, view: str, replicate: str, saved_rows: list[dict]) -> dict:
    source = MATCH_SOURCE[match]
    tracking_name = "clean" if replicate == "clean" else "feed_like"
    bundle = joblib.load(TRACKING / f"{match}_{tracking_name}_{replicate}.joblib")
    inputs, labels = bundle["inputs"], np.asarray(bundle["labels"]).astype(str)
    pred_path = PREDICTIONS / f"candidate_{match}_{tracking_name}_{replicate}.npz"
    with np.load(pred_path, allow_pickle=False) as archive:
        candidate = np.asarray(archive["prediction"]).astype(str)
        probability = np.asarray(archive["probability"], dtype=np.float64)
    n = len(labels)
    if labels.shape != candidate.shape or probability.shape != (n, 3):
        raise ValueError(f"Saved prediction and input lengths disagree for {match}/{replicate}")
    if not np.isfinite(probability).all() or np.any(probability < 0):
        raise ValueError(f"Malformed saved emission probabilities for {match}/{replicate}")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=2e-7, rtol=0):
        raise ValueError(f"Saved probabilities do not sum to one for {match}/{replicate}")
    saved = [r for r in saved_rows if r.get("match") == match and r.get("replicate") == replicate
             and r.get("view") == view and r.get("recipe") == "combined_objective_equal_probability_ensemble"]
    if len(saved) != 1:
        raise ValueError(f"Missing or duplicate paired candidate score row: {match}/{view}/{replicate}")
    saved = saved[0]
    raw = supervised.LABELS[np.argmax(probability, axis=1)].astype(str)
    fps = float(inputs["fps"])
    _, truth_codes, candidate_codes, inserts = prepare_tracking_codes(
        labels, candidate, inputs["period"], fps, inputs["timestamps"])
    _, raw_truth_codes, raw_codes, raw_inserts = prepare_tracking_codes(
        labels, raw, inputs["period"], fps, inputs["timestamps"])
    if not np.array_equal(inserts, raw_inserts) or not np.array_equal(truth_codes, raw_truth_codes):
        raise ValueError("Official matcher inputs changed between candidate and argmax diagnostics")
    runs = pm.valid_runs(truth_codes)
    candidate_filled = pm.fill_abstentions(candidate_codes, runs)
    raw_filled = pm.fill_abstentions(raw_codes, runs)
    gt_spells = pm.possession_spells(truth_codes, runs)
    candidate_spells = pm.possession_spells(candidate_filled, runs)
    raw_spells = pm.possession_spells(raw_filled, runs)
    candidate_overlaps = pm.spell_overlaps(gt_spells, candidate_spells)
    raw_overlaps = pm.spell_overlaps(gt_spells, raw_spells)
    candidate_matched = pm.matched_gt_flags(candidate_overlaps, len(gt_spells), pm.SPELL_IOU_HEADLINE)
    raw_matched = pm.matched_gt_flags(raw_overlaps, len(gt_spells), pm.SPELL_IOU_HEADLINE)
    candidate_best = _best_iou_by_gt(candidate_overlaps, len(gt_spells))
    raw_best = _best_iou_by_gt(raw_overlaps, len(gt_spells))

    # The saved cost-3 output is rescored only to establish parity with the paired
    # aggregate. Raw argmax is scored as an unselected, diagnostic path.
    candidate_score = evaluate_tracking(labels, candidate, inputs["period"], fps, inputs["timestamps"])
    _score_parity(candidate_score, saved)
    raw_score = evaluate_tracking(labels, raw, inputs["period"], fps, inputs["timestamps"])

    starts = {a for a, _ in runs}
    stops = {b for _, b in runs}
    duration_table = {
        name: {"missed": 0, "complete": 0, "graded_edge": 0,
               "visible_ball": 0, "has_ball_gap": 0,
               "complete_visible_ball": 0, "complete_has_ball_gap": 0,
               "graded_edge_visible_ball": 0, "graded_edge_has_ball_gap": 0}
        for name, _, _ in DURATION_BINS
    }
    mechanism = {name: 0 for name in MECHANISMS}
    evidence = {name: {key: [] for key in (
        "raw_correct_fraction", "decoded_correct_fraction", "p_true_mean",
        "p_true_max", "p_true_margin_mean", "candidate_best_iou",
        "raw_argmax_best_iou", "nearest_raw_emission_seconds")}
        for name in MECHANISMS}
    missed_short_complete = 0
    matched_short_complete = 0
    raw_matched_short_complete = 0
    short_complete_visibility = {"visible_ball": 0, "has_ball_gap": 0}
    short_edge_counts = {"complete": 0, "graded_edge": 0}
    nearest_within_1s = 0
    ball_observed = np.isfinite(inputs["ball"]).all(axis=1)
    if "ball_observed" in inputs:
        ball_observed &= np.asarray(inputs["ball_observed"], dtype=bool)

    for gi, (start, stop, team) in enumerate(gt_spells):
        seconds = (stop-start)/fps
        duration_name = _duration_bin(seconds)
        complete = start not in starts and stop not in stops
        boundary = "complete" if complete else "graded_edge"
        if seconds < 3 and complete:
            if raw_matched[gi]:
                raw_matched_short_complete += 1
        if candidate_matched[gi]:
            if seconds < 3 and complete:
                matched_short_complete += 1
            continue
        duration_table[duration_name]["missed"] += 1
        duration_table[duration_name][boundary] += 1
        original_start, original_stop = _original_span(start, stop, inserts)
        if original_start < 0 or original_stop > n or original_stop <= original_start:
            raise ValueError("Scorer-to-native span mapping failed")
        visible = bool(ball_observed[original_start:original_stop].all())
        visibility = "visible_ball" if visible else "has_ball_gap"
        duration_table[duration_name][visibility] += 1
        duration_table[duration_name][f"{boundary}_{visibility}"] += 1
        if seconds >= 3:
            continue
        short_edge_counts[boundary] += 1
        if not complete:
            continue
        short_complete_visibility[visibility] += 1
        missed_short_complete += 1
        local_truth = probability[original_start:original_stop, team]
        local_other = np.max(np.delete(probability[original_start:original_stop], team, axis=1), axis=1)
        augmented_length = stop-start
        native_indices = np.arange(original_start, original_stop)
        raw_correct_frames = int((raw[native_indices] == supervised.LABELS[team]).sum())
        decoded_correct_frames = int((candidate[native_indices] == supervised.LABELS[team]).sum())
        candidate_iou = float(candidate_best[gi])
        raw_iou = float(raw_best[gi])
        category = _classify_miss(
            candidate_matched=False, candidate_best_iou=candidate_iou,
            raw_correct_frames=raw_correct_frames, decoded_correct_frames=decoded_correct_frames,
            raw_best_iou=raw_iou)
        mechanism[category] += 1
        raw_fraction = raw_correct_frames / augmented_length
        decoded_fraction = decoded_correct_frames / augmented_length
        # Locate the nearest correct-team emission in this graded run to show
        # timing displacement without selecting a repair window.
        containing = next((a for a, b in runs if a <= start and stop <= b), None)
        run_stop = next((b for a, b in runs if a <= start and stop <= b), None)
        nearest = _nearest_emission_distance(team, start, stop, containing, run_stop, raw_codes, fps) if containing is not None else None
        if nearest is not None and nearest <= 1.0:
            nearest_within_1s += 1
        values = evidence[category]
        values["raw_correct_fraction"].append(raw_fraction)
        values["decoded_correct_fraction"].append(decoded_fraction)
        values["p_true_mean"].append(float(np.mean(local_truth)))
        values["p_true_max"].append(float(np.max(local_truth)))
        values["p_true_margin_mean"].append(float(np.mean(local_truth-local_other)))
        values["candidate_best_iou"].append(candidate_iou)
        values["raw_argmax_best_iou"].append(raw_iou)
        if nearest is not None:
            values["nearest_raw_emission_seconds"].append(nearest)

    if matched_short_complete + missed_short_complete == 0:
        short_recall = None
    else:
        short_recall = matched_short_complete / (matched_short_complete + missed_short_complete)
    return {
        "source": source, "match": match, "view": view, "replicate": replicate,
        "candidate_spell_f1": float(candidate_score["spell_f1_mean"]),
        "raw_argmax_spell_f1_diagnostic": float(raw_score["spell_f1_mean"]),
        "reference_spells": len(gt_spells), "candidate_matched_spells": int(candidate_matched.sum()),
        "misses_by_duration_and_observation": duration_table,
        "short_complete": {
            "references": matched_short_complete + missed_short_complete,
            "matched": matched_short_complete, "missed": missed_short_complete,
            "recall_iou_0_5": short_recall,
            "raw_argmax_matched": raw_matched_short_complete,
            "raw_argmax_recall_iou_0_5": (raw_matched_short_complete /
                (matched_short_complete + missed_short_complete)
                if matched_short_complete + missed_short_complete else None),
            "missed_by_visibility": short_complete_visibility,
            "missed_touching_graded_edge": short_edge_counts["graded_edge"],
            "mechanism_counts": mechanism,
            "nearest_correct_raw_emission_within_1s": nearest_within_1s,
        "mechanism_evidence_mean": {
                category: {key: (float(np.mean(values)) if values else None)
                           for key, values in rows.items()}
                for category, rows in evidence.items()
            },
        },
        "official_global_matcher": True,
        "candidate_score_parity": True,
    }


def _average_nested(values: list):
    """Average numeric leaves recursively; call once within match, then provider."""
    values = [value for value in values if value is not None]
    if not values:
        return None
    if isinstance(values[0], dict):
        keys = values[0].keys()
        return {key: _average_nested([value.get(key) for value in values]) for key in keys}
    if isinstance(values[0], (int, float, np.integer, np.floating)):
        return float(np.mean(values))
    return values[0]


def _summarize(cells: list[dict]) -> dict:
    """Average feed-like replicates within match, then matches within provider."""
    summaries = {}
    for view in VIEWS:
        summaries[view] = {}
        for source, matches in COHORT.items():
            per_match = {}
            for match in matches:
                match_cells = [cell for cell in cells if cell["view"] == view and cell["match"] == match]
                if not match_cells:
                    raise ValueError(f"Missing matched cell: {source}/{match}/{view}")
                per_match[match] = _average_nested(match_cells)
            summaries[view][source] = {
                "matches": per_match,
                "equal_match_mean": _average_nested(list(per_match.values())),
            }
    return summaries


def run_audit(declaration: dict) -> dict:
    _assert_frozen(declaration)
    rows = _json(PREDICTIONS / "runs.json")["rows"]
    cells = []
    for source, matches in COHORT.items():
        for match in matches:
            for view, replicates in VIEWS.items():
                for replicate in replicates:
                    cells.append(_cell(match, view, replicate, rows))
    if len(cells) != 15 or not all(c["official_global_matcher"] and c["candidate_score_parity"] for c in cells):
        raise ValueError("Incomplete saved cohort or candidate scorer parity")
    return {
        "schema_version": 1,
        "study_id": declaration["study_id"],
        "declaration_sha256": sha256(DECLARATION),
        "official_scorer_sha256": sha256(ROOT / "evaluation/possession_metrics.py"),
        "completed_cells": len(cells),
        "organizer_assets_read": False,
        "model_fitting_or_inference": False,
        "threshold_sweeps_or_candidate_selection": False,
        "reference_informed_prediction": False,
        "aggregation": "Average feed-like replicates within match, matches within provider, then report each provider; counts/rates are descriptive cell-level averages.",
        "official_matcher": "Global IoU-descending greedy one-to-one assignment at IoU 0.5 over all candidate/reference spells, with subgroup filtering afterward.",
        "summaries": _summarize(cells),
        "cells": cells,
        "interpretation": {
            "mechanism_partition": "Exclusive diagnostics for globally unmatched complete under-three-second references. The four buckets separate cost-3 suppression, surviving-but-insufficient local evidence, no local team evidence and official assignment conflicts.",
            "raw_argmax": "A retrospective diagnostic counterfactual over saved native probabilities. It is not a trained or selected prediction and does not estimate deployable gain.",
            "oracle": "No reference-informed prediction, timing repair or oracle upper bound was computed.",
            "evidence": "Per-reference probabilities and timing are summarized within aggregate mechanism buckets; no reference rows or frame predictions are emitted.",
        },
        "limitations": declaration["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true", help="seal fixed sources, inputs and diagnostic rules")
    action.add_argument("--run", action="store_true", help="run the read-only audit using the frozen receipt")
    args = parser.parse_args()
    if args.freeze:
        declaration = freeze_inputs()
        print(json.dumps({"declaration": str(DECLARATION.relative_to(ROOT)),
                          "tracking_snapshots": len(declaration["input_receipts"]["tracking_and_label_snapshots"]),
                          "candidate_archives": len(declaration["input_receipts"]["saved_candidate_probabilities_and_paths"])}, indent=2))
        return
    if not DECLARATION.is_file():
        raise FileNotFoundError("Freeze this audit first with --freeze")
    if RESULT.exists():
        raise FileExistsError(f"Refusing to overwrite completed audit: {RESULT}")
    result = run_audit(_json(DECLARATION))
    RESULT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"result": str(RESULT.relative_to(ROOT)), "cells": result["completed_cells"]}, indent=2))


if __name__ == "__main__":
    main()
