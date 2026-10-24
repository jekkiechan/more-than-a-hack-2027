"""Post-hoc possession-spell diagnostics for saved, frozen predictions.

This program never trains a model or invokes inference. It publishes aggregate
statistics only. The default inputs are the one previously evaluated external
Sportec test match and the model selected on validation before test access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from experiments.metrics import evaluate_tracking


ROOT = Path(__file__).resolve().parents[1]
DURATION_BINS = (("<1 s", 0, 1), ("1–<3 s", 1, 3),
                 ("3–<10 s", 3, 10), (">=10 s", 10, np.inf))


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None}
    return {"n": int(values.size), "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p10": float(np.percentile(values, 10)),
            "p90": float(np.percentile(values, 90))}


def gap_aware_codes(data, predictions):
    """Use exactly the separators applied by experiments.metrics.evaluate_tracking."""
    truth = np.asarray(data["labels"]).astype(str)
    predictions = np.asarray(predictions).astype(str)
    fps = float(data["fps"])
    # Shared scorer validates labels, dimensions, sampling rate and timestamps.
    metrics = evaluate_tracking(truth, predictions, period=data["period"], fps=fps,
                                timestamps=data["timestamps"])
    delta = np.diff(data["timestamps"])
    boundaries = ((data["period"][1:] != data["period"][:-1]) |
                  (delta <= 0) | (delta > 1.5 / fps))
    inserts = np.flatnonzero(boundaries) + 1
    gt_codes = pm.label_codes(np.insert(truth, inserts, "X"))
    pred_codes = pm.label_codes(np.insert(predictions, inserts, "X"))
    runs = pm.valid_runs(gt_codes)
    filled = pm.fill_abstentions(pred_codes, runs)
    return gt_codes, pred_codes, filled, runs, metrics


def analyze(data, predictions):
    gt_codes, pred_codes, filled, runs, metrics = gap_aware_codes(data, predictions)
    fps = float(data["fps"])
    gt_spells = pm.possession_spells(gt_codes, runs)
    pred_spells = pm.possession_spells(filled, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    matched = pm.matched_gt_flags(overlaps, len(gt_spells), 0.5)
    tp, fp, fn = pm.match_spells(overlaps, len(gt_spells), len(pred_spells), 0.5)
    official = pm.spell_scores(gt_codes, filled, runs)
    assert tp == int(matched.sum())
    assert tp == metrics["spell_f1"][0.5]["n_matched"]
    assert len(gt_spells) == metrics["n_gt_spells"]
    assert len(pred_spells) == metrics["n_pred_spells"]
    assert np.isclose(official["mean_f1"], metrics["spell_f1_mean"], atol=1e-14)

    duration = np.array([(stop - start) / fps for start, stop, _ in gt_spells])
    pred_duration = np.array([(stop - start) / fps for start, stop, _ in pred_spells])
    gt_degree = np.zeros(len(gt_spells), dtype=int)
    pred_degree = np.zeros(len(pred_spells), dtype=int)
    best_iou = np.zeros(len(gt_spells), dtype=float)
    for iou, gt_idx, pred_idx in overlaps:
        gt_degree[gt_idx] += 1
        pred_degree[pred_idx] += 1
        best_iou[gt_idx] = max(best_iou[gt_idx], iou)
    touches_merge = np.zeros(len(gt_spells), dtype=bool)
    for _, gt_idx, pred_idx in overlaps:
        touches_merge[gt_idx] |= pred_degree[pred_idx] >= 2
    missed = ~matched
    fragmented = gt_degree >= 2

    bins = []
    for name, lower, upper in DURATION_BINS:
        mask = (duration >= lower) & (duration < upper)
        count = int(mask.sum())
        misses = int((mask & missed).sum())
        bins.append({
            "duration": name, "min_seconds_inclusive": lower,
            "max_seconds_exclusive": upper if np.isfinite(upper) else None,
            "n_gt": count, "n_matched_at_iou_0_5": int((mask & matched).sum()),
            "n_missed_at_iou_0_5": misses,
            "recall_at_iou_0_5": ratio((mask & matched).sum(), count),
            "share_of_all_gt_spells": ratio(count, len(gt_spells)),
            "share_of_all_missed_spells": ratio(misses, fn),
            "share_of_gt_alive_time": ratio(duration[mask].sum(), duration.sum()),
            "best_same_team_iou": distribution(best_iou[mask]),
            "n_with_no_same_team_overlap": int((mask & (gt_degree == 0)).sum()),
            "n_fragmented": int((mask & fragmented).sum()),
            "n_touching_merged_prediction": int((mask & touches_merge).sum()),
        })

    # A no-overlap spell has zero correctly attributed team frames, even after
    # the official within-valid-run abstention fill used for temporal scoring.
    opposite_only = dead_only = mixed = 0
    for idx, (start, stop, team) in enumerate(gt_spells):
        if gt_degree[idx]:
            continue
        section = filled[start:stop]
        if np.all(section == (pm.A if team == pm.H else pm.H)):
            opposite_only += 1
        elif np.all(section == pm.D):
            dead_only += 1
        else:
            mixed += 1

    cm = np.asarray(metrics["confusion"])
    frame_errors = int(cm[:3].sum() - np.trace(cm[:3, :3]))
    swaps = int(cm[pm.H, pm.A] + cm[pm.A, pm.H])
    summary_keys = (
        "macro_f1", "accuracy", "spell_f1_mean", "spell_f1_at_headline",
        "n_gt_spells", "n_pred_spells", "spell_count_error", "fragmentation",
        "n_gt_segments", "n_pred_segments", "mean_spell_duration_s",
        "gt_mean_spell_duration_s", "team_acc_gt_alive", "team_acc_both_alive",
        "alive_dead_f1", "gt_home_share", "pred_home_share", "share_err",
        "per_class", "confusion", "original_n_frames", "period_or_time_separators",
        "sampling_fps", "label_coverage", "three_class_benchmark", "spell_f1",
    )
    return {
        "purpose": "Post-hoc diagnosis of frozen predictions; no fitting, tuning or model changes.",
        "matching": {
            "implementation": "Unchanged evaluation.possession_metrics spell_overlaps, matched_gt_flags and match_spells",
            "rule": "Same-team temporal IoU; greedy descending IoU, ties by GT then prediction index; one-to-one.",
            "headline_iou": 0.5,
            "gaps": "Insert X at period/time discontinuities, then use official valid_runs and fill_abstentions.",
            "duration": "Half-open spell length in frames divided by native fps; dead-ball spells excluded.",
        },
        "metrics": {key: metrics[key] for key in summary_keys},
        "by_gt_duration": bins,
        "missed_spell_categories_at_iou_0_5": {
            "n_missed": fn,
            "no_same_team_overlap": int((missed & (gt_degree == 0)).sum()),
            "same_team_overlap_but_best_iou_below_0_5": int((missed & (gt_degree > 0) & (best_iou < 0.5)).sum()),
            "qualifying_iou_but_unmatched_after_one_to_one_assignment": int((missed & (best_iou >= 0.5)).sum()),
            "zero_overlap_spell_prediction": {"entirely_opposite_team": opposite_only,
                                              "entirely_dead": dead_only,
                                              "mixed_or_abstained": mixed},
        },
        "overlap_structure": {
            "definition": "An edge is any positive-duration same-team overlap, including one frame. These are overlapping diagnostics, not mutually exclusive causes.",
            "merge_definition": "One predicted spell overlaps at least two separate GT spells of its team.",
            "fragmentation_definition": "One GT spell overlaps at least two separate predicted spells of its team.",
            "n_merged_predictions": int((pred_degree >= 2).sum()),
            "n_gt_spells_touching_merged_prediction": int(touches_merge.sum()),
            "n_missed_gt_spells_touching_merged_prediction": int((missed & touches_merge).sum()),
            "n_fragmented_gt_spells": int(fragmented.sum()),
            "n_missed_fragmented_gt_spells": int((missed & fragmented).sum()),
            "n_gt_spells_both_fragmented_and_touching_merge": int((fragmented & touches_merge).sum()),
            "n_predictions_without_same_team_overlap": int((pred_degree == 0).sum()),
            "gt_same_team_prediction_overlap_degree": {str(v): int((gt_degree == v).sum()) for v in np.unique(gt_degree)},
            "pred_same_team_gt_overlap_degree": {str(v): int((pred_degree == v).sum()) for v in np.unique(pred_degree)},
            "gt_spell_duration_seconds": distribution(duration),
            "prediction_spell_duration_seconds": distribution(pred_duration),
        },
        "frame_errors": {"total": frame_errors, "home_away_swaps": swaps,
                         "home_away_share_of_errors": ratio(swaps, frame_errors),
                         "other_errors_including_alive_dead": frame_errors - swaps},
        "limitations": [
            "One held-out external match; these are observed errors, not an accuracy ceiling or a new model-selection benchmark.",
            "Sportec's original source freezes ball XY while dead, a dataset-specific shortcut that strongly favors dead-ball detection.",
            "No challenge labels were read; these results do not establish hackathon-match accuracy.",
            "Overlap merge/fragmentation counts depend on any positive overlap and may co-occur; they are descriptive, not proof of causes.",
        ],
    }


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_report(result, destination):
    m = result["metrics"]
    misses = result["missed_spell_categories_at_iou_0_5"]
    structure = result["overlap_structure"]
    provenance = result["provenance"]
    short_spells = result["by_gt_duration"][:2]
    short_miss_share = sum(row["n_missed_at_iou_0_5"] for row in short_spells) / misses["n_missed"]
    short_time_share = sum(row["share_of_gt_alive_time"] for row in short_spells)
    lines = [
        "# Possession errors on the frozen external test match", "",
        f"The frozen `{provenance['model']}` model scored frame macro-F1 **{m['macro_f1']:.3f}** "
        f"and mean possession-spell F1 **{m['spell_f1_mean']:.3f}** on Sportec `{provenance['match']}` "
        f"under the `{provenance['condition']}` condition. "
        "This is a post-hoc diagnosis of its saved predictions; no model was changed. "
        "It describes the remaining errors, not the best attainable accuracy.", "",
        f"At IoU ≥0.5, **{m['spell_f1'][0.5]['n_matched']} / {m['n_gt_spells']}** true possessions matched "
        f"({m['spell_f1'][0.5]['recall']:.1%} recall), with **{m['spell_f1'][0.5]['n_spurious']}** unmatched predictions "
        f"({m['spell_f1'][0.5]['precision']:.1%} precision).", "",
        "| True possession duration | True spells | Matched at IoU ≥0.5 | Recall | Share of all misses | Share of live time |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["by_gt_duration"]:
        lines.append(f"| {row['duration']} | {row['n_gt']} | {row['n_matched_at_iou_0_5']} | "
                     f"{row['recall_at_iou_0_5']:.1%} | {row['share_of_all_missed_spells']:.1%} | "
                     f"{row['share_of_gt_alive_time']:.1%} |")
    lines += ["",
        f"Possessions shorter than three seconds account for **{short_miss_share:.1%}** of missed spells "
        f"but only **{short_time_share:.1%}** of live time. This explains how frame accuracy and "
        "possession-spell accuracy can diverge so sharply.", "",
        f"Of the **{misses['n_missed']}** missed spells, **{misses['no_same_team_overlap']}** had no same-team "
        f"overlap at all and **{misses['same_team_overlap_but_best_iou_below_0_5']}** had some correctly attributed "
        f"frames but no prediction reaching IoU 0.5. "
        f"**{misses['qualifying_iou_but_unmatched_after_one_to_one_assignment']}** additional misses arose only from "
        "the one-to-one matching constraint.", "",
        f"The model produced **{m['n_pred_spells']}** possessions versus **{m['n_gt_spells']}** in the reference "
        f"(mean duration **{m['mean_spell_duration_s']:.2f}s** versus **{m['gt_mean_spell_duration_s']:.2f}s**). "
        f"**{structure['n_merged_predictions']}** predictions overlap multiple true spells of the same team; "
        f"**{structure['n_fragmented_gt_spells']}** true spells overlap multiple predictions of their team. "
        f"Of the missed true spells, **{structure['n_missed_gt_spells_touching_merged_prediction']}** touch a merged "
        f"prediction and **{structure['n_missed_fragmented_gt_spells']}** are fragmented. "
        "These overlap-based counts can intersect and use any positive overlap, including one frame; "
        "they should not be added or interpreted as proven causes.", "",
        "| Matching threshold | Spell F1 | Matched true possessions |",
        "|---|---:|---:|",
    ]
    for threshold, row in m["spell_f1"].items():
        lines.append(f"| IoU ≥{threshold:.1f} | {row['f1']:.3f} | {row['n_matched']} |")
    lines += ["",
        f"**{result['frame_errors']['home_away_share_of_errors']:.1%}** of all wrong frames are home/away swaps. "
        f"The dead-ball F1 is **{m['per_class']['D']['f1']:.3f}**, but Sportec's original source freezes ball XY "
        "during dead play. That is a strong dataset-specific shortcut; the result does not establish accuracy "
        "on the hackathon's noisier, partially missing tracking.", "",
        f"Whole-match home possession share is **{m['pred_home_share']:.2%}** predicted versus "
        f"**{m['gt_home_share']:.2%}** reference (absolute error **{100*m['share_err']:.2f} percentage points**). "
        "An accurate aggregate can coexist with missing many short possessions.", "",
    ]
    if "conditional_bootstrap" in result:
        boot = result["conditional_bootstrap"]
        lines += [f"The existing 1,000-resample official 5th–95th percentile bootstrap intervals are "
                  f"**{boot['macro_f1'][0]:.3f}–{boot['macro_f1'][1]:.3f}** for frame macro-F1 and "
                  f"**{boot['spell_f1_mean'][0]:.3f}–{boot['spell_f1_mean'][1]:.3f}** for mean spell F1. "
                  "They are conditional on this one match, with a fixed model; they do not measure "
                  "generalization across matches, leagues or tracking providers.", ""]
    lines += [
        "Matching uses the unchanged official gap-aware scorer, including period/time separators, "
        "within-valid-run abstention filling, same-team overlaps and greedy one-to-one IoU matching. "
        "Only aggregate statistics are published. No public challenge ground truth was accessed.", "",
        "Reproduce from the frozen local artifacts:", "", "```bash",
        "python -m experiments.error_analysis", "```", "",
        "Full aggregate results: [results/error_analysis.json](results/error_analysis.json).", "",
    ]
    Path(destination).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/external_audit/sportec/processed/J03WPY.npz")
    parser.add_argument("--predictions", type=Path, default=ROOT / "data/experiments/benchmark/J03WPY_clean_robust.npy")
    parser.add_argument("--benchmark-results", type=Path, default=ROOT / "data/experiments/benchmark/test.json")
    parser.add_argument("--match", default="J03WPY")
    parser.add_argument("--model", default="robust")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/error_analysis.json")
    parser.add_argument("--report", type=Path, default=ROOT / "experiments/ERROR_ANALYSIS.md")
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as archive:
        # No tracking features are required: this only analyzes saved predictions.
        data = {key: archive[key] for key in ("labels", "period", "timestamps", "fps")}
    predictions = np.load(args.predictions, allow_pickle=False)
    result = analyze(data, predictions)
    result["provenance"] = {
        "match": args.match, "model": args.model, "condition": args.condition,
        "data_sha256": digest(args.data), "predictions_sha256": digest(args.predictions),
        "official_metrics_sha256": digest(ROOT / "evaluation/possession_metrics.py"),
        "analysis_code_sha256": digest(__file__),
    }
    if args.benchmark_results.exists():
        benchmark = json.loads(args.benchmark_results.read_text())
        previous = benchmark["matches"][args.match][args.condition][args.model]
        # Catch a mismatched artifact or accidental change in scorer semantics.
        for key in ("macro_f1", "spell_f1_mean", "n_gt_spells", "n_pred_spells"):
            if not np.isclose(result["metrics"][key], previous[key], atol=1e-14, rtol=0):
                raise ValueError(f"Saved benchmark does not match {key}")
        result["provenance"]["selected_before_test"] = benchmark["selected_before_test"]
        result["provenance"]["frozen_manifest_sha256"] = benchmark["manifest_sha256"]
        result["provenance"]["benchmark_results_sha256"] = digest(args.benchmark_results)
        result["conditional_bootstrap"] = {
            "source": "Previously computed official benchmark, not re-fitted or re-tuned",
            "n_resamples": previous["bootstrap_n"],
            "percentiles": list(pm.BOOTSTRAP_PERCENTILES),
            "macro_f1": [previous["bootstrap_low"], previous["bootstrap_high"]],
            "spell_f1_mean": [previous["spell_bootstrap_low"], previous["spell_bootstrap_high"]],
            "limitation": "Conditional on this match and fixed predictions; not an interval over new matches.",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_report(result, args.report)
    print(json.dumps({"output": str(args.output), "report": str(args.report),
                      "macro_f1": result["metrics"]["macro_f1"],
                      "spell_f1_mean": result["metrics"]["spell_f1_mean"]}))


if __name__ == "__main__":
    main()
