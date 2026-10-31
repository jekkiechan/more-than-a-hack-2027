"""Error audit for saved dense-ensemble external predictions.

Input is a CSV with provider, match_id, segment_id, frame, gt, pred, ball_missing
and fps columns. segment_id identifies contiguous native-time runs, so spells
and transitions never cross a half/time break. This module only reads
predictions; it does not fit, decode or score organizer references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = {"H": 0, "A": 1, "D": 2, "X": 3}


def _bool_values(values: pd.Series) -> np.ndarray:
    if values.isna().any():
        raise ValueError("ball_missing cannot contain null values")
    normalized = values.map(lambda x: str(x).strip().lower())
    mapping = {"true": True, "1": True, "1.0": True, "yes": True, "y": True,
               "false": False, "0": False, "0.0": False, "no": False, "n": False}
    bad = sorted(set(normalized) - set(mapping))
    if bad:
        raise ValueError("ball_missing must contain boolean or 0/1 values")
    return normalized.map(mapping).to_numpy(dtype=bool)


def _runs(labels: np.ndarray):
    if len(labels) == 0:
        return []
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    stops = np.r_[starts[1:], len(labels)]
    return [(int(s), int(e), int(labels[s])) for s, e in zip(starts, stops)]


def _audit_segment(gt: np.ndarray, pred: np.ndarray, missing: np.ndarray, fps: float) -> dict:
    scored = gt != LABELS["X"]
    alive_gt = (gt == 0) | (gt == 1)
    alive_pred = (pred == 0) | (pred == 1)
    both_alive = scored & alive_gt & alive_pred
    wrong_team = both_alive & (gt != pred)
    false_d = scored & alive_gt & (pred == LABELS["D"])
    missed_dead = scored & (gt == LABELS["D"]) & alive_pred
    runs_gt = [(s, e, c) for s, e, c in _runs(gt) if c in (0, 1)]
    pred_scored = pred.copy()
    pred_scored[gt == LABELS["X"]] = LABELS["X"]
    runs_pred = [(s, e, c) for s, e, c in _runs(pred_scored) if c in (0, 1)]
    # Same-team overlap graph. A GT spell touching multiple predictions is split;
    # a prediction touching multiple GT spells is a merge.
    overlaps = [[max(0, min(ge, pe) - max(gs, ps)) if gc == pc else 0
                 for ps, pe, pc in runs_pred] for gs, ge, gc in runs_gt]
    split_gt = sum(sum(v > 0 for v in row) > 1 for row in overlaps)
    merge_pred = sum(sum(overlaps[i][j] > 0 for i in range(len(runs_gt))) > 1
                     for j in range(len(runs_pred)))
    # Pair every scored class boundary, including alive/dead changes.
    gt_all, pred_all = _runs(gt), _runs(pred_scored)
    gt_edges = [(gt_all[i - 1][2], gt_all[i][2], gt_all[i][0])
                for i in range(1, len(gt_all))
                if gt_all[i - 1][2] != 3 and gt_all[i][2] != 3]
    pr_edges = [(pred_all[i - 1][2], pred_all[i][2], pred_all[i][0])
                for i in range(1, len(pred_all))
                if pred_all[i - 1][2] != 3 and pred_all[i][2] != 3]
    shifts = []
    used = set()
    max_boundary_distance = max(1, int(round(float(fps) * 1.0)))
    for left, right, frame in gt_edges:
        choices = [(abs(frame - pf), j, pf) for j, (pl, pr, pf) in enumerate(pr_edges)
                   if j not in used and (pl, pr) == (left, right) and
                   abs(frame - pf) <= max_boundary_distance]
        if choices:
            delta, j, pf = min(choices)
            used.add(j)
            shifts.append(int(pf - frame))
    def rate(mask):
        return float(mask.sum() / scored.sum()) if scored.any() else None
    gt_lengths = [(e - s) / float(fps) for s, e, _ in runs_gt]
    pred_lengths = [(e - s) / float(fps) for s, e, _ in runs_pred]
    length_breakdown = {}
    for name, lower_s, upper_s in (("short_le_1s", 0.0, 1.0), ("medium_1_to_5s", 1.0, 5.0),
                                   ("long_gt_5s", 5.0, None)):
        selected = [(s, e) for s, e, _ in runs_gt
                    if (e - s) / float(fps) > lower_s or lower_s == 0.0]
        selected = [(s, e) for s, e in selected
                    if upper_s is None or (e - s) / float(fps) <= upper_s]
        active = np.zeros(len(gt), dtype=bool)
        for s, e in selected:
            active[s:e] = True
        count = len(selected)
        length_breakdown[name] = {
            "gt_spells": count,
            "wrong_team_rate": float((wrong_team & active).sum() / max(1, active.sum())) if count else None,
            "false_d_rate": float((false_d & active).sum() / max(1, active.sum())) if count else None,
            "false_d_spells": int(sum(bool(false_d[s:e].any()) for s, e in selected)),
        }
    missing_scored = scored & missing
    observed_scored = scored & ~missing
    return {
        "frames": int(len(gt)), "scored_frames": int(scored.sum()),
        "fps": float(fps),
        "wrong_team_frames": int(wrong_team.sum()), "wrong_team_rate_scored": rate(wrong_team),
        "false_d_frames": int(false_d.sum()), "false_d_rate_scored": rate(false_d),
        "missed_dead_frames": int(missed_dead.sum()), "missed_dead_rate_scored": rate(missed_dead),
        "split_ground_truth_spells": int(split_gt), "merge_predicted_spells": int(merge_pred),
        "gt_alive_spells": len(runs_gt), "pred_alive_spells": len(runs_pred),
        "gt_spell_length_median_seconds": float(np.median(gt_lengths)) if gt_lengths else None,
        "pred_spell_length_median_seconds": float(np.median(pred_lengths)) if pred_lengths else None,
        "gt_spell_length_bins": length_breakdown,
        "matched_transition_count": len(shifts),
        "unmatched_gt_transition_count": len(gt_edges) - len(shifts),
        "unmatched_pred_transition_count": len(pr_edges) - len(used),
        "transition_match_tolerance_seconds": 1.0,
        "transition_shift_median_frames": float(np.median(shifts)) if shifts else None,
        "transition_shift_abs_median_frames": float(np.median(np.abs(shifts))) if shifts else None,
        "transition_shift_median_seconds": float(np.median(shifts) / float(fps)) if shifts else None,
        "transition_shift_abs_median_seconds": (float(np.median(np.abs(shifts)) / float(fps))
                                                 if shifts else None),
        "scored_frames_ball_missing": int(missing_scored.sum()),
        "wrong_team_rate_ball_missing": (float((wrong_team & missing).sum() / missing_scored.sum())
                                           if missing_scored.any() else None),
        "wrong_team_rate_ball_observed": (float((wrong_team & ~missing).sum() / observed_scored.sum())
                                            if observed_scored.any() else None),
        "false_d_rate_ball_missing": (float((false_d & missing).sum() / missing_scored.sum())
                                      if missing_scored.any() else None),
        "false_d_rate_ball_observed": (float((false_d & ~missing).sum() / observed_scored.sum())
                                       if observed_scored.any() else None),
        "wrong_team_frames_ball_missing": int((wrong_team & missing).sum()),
        "wrong_team_frames_ball_observed": int((wrong_team & ~missing).sum()),
        "false_d_frames_ball_missing": int((false_d & missing).sum()),
        "false_d_frames_ball_observed": int((false_d & ~missing).sum()),
        "scored_frames_ball_observed": int(observed_scored.sum()),
    }


def frame_table(provider: str, match_id: str, gt, pred, ball_observed,
                period, timestamps, fps: float) -> pd.DataFrame:
    """Build the runner's table from frozen native-rate label/prediction arrays."""
    arrays = [np.asarray(list(x) if isinstance(x, str) else x)
              for x in (gt, pred, ball_observed, period, timestamps)]
    lengths = {len(x) for x in arrays}
    if len(lengths) != 1:
        raise ValueError("ground truth, predictions and timeline arrays must have equal length")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    n = len(arrays[0])
    gt, pred, observed, period, timestamps = arrays
    breaks = np.zeros(n, dtype=bool)
    if n:
        breaks[0] = True
        breaks[1:] = ((period[1:] != period[:-1]) |
                      (np.diff(timestamps) > 1.5 / float(fps)) |
                      (np.diff(timestamps) <= 0))
    segment = np.cumsum(breaks) - 1
    return pd.DataFrame({"provider": provider, "match_id": match_id,
        "segment_id": segment, "frame": np.arange(n, dtype=np.int64), "fps": float(fps),
                         "gt": gt, "pred": pred, "ball_missing": ~observed.astype(bool)})


def audit(frame: pd.DataFrame) -> dict:
    required = {"provider", "match_id", "segment_id", "frame", "gt", "pred", "ball_missing", "fps"}
    absent = required - set(frame.columns)
    if absent:
        raise ValueError("missing columns: " + ", ".join(sorted(absent)))
    f = frame.copy()
    for col in ("gt", "pred"):
        f[col] = f[col].astype(str).str.strip().str.upper()
        if not f[col].isin(LABELS).all():
            raise ValueError(f"{col} contains labels outside H/A/D/X")
    f["fps"] = pd.to_numeric(f["fps"], errors="coerce")
    if f["fps"].isna().any() or not np.isfinite(f["fps"].to_numpy()).all() or (f["fps"] <= 0).any():
        raise ValueError("fps must contain positive finite values")
    key = ["provider", "match_id", "segment_id"]
    if f.duplicated(key + ["frame"]).any():
        raise ValueError("duplicate frame within provider/match/segment")
    groups = []
    for identity, g in f.groupby(key, sort=True):
        g = g.sort_values("frame")
        if not np.allclose(g["fps"].to_numpy(), float(g["fps"].iloc[0]), rtol=0, atol=1e-9):
            raise ValueError(f"fps must be constant within segment {identity}")
        frames = g["frame"].to_numpy()
        if not np.issubdtype(frames.dtype, np.integer) or (len(frames) > 1 and np.any(np.diff(frames) != 1)):
            raise ValueError(f"frames must be contiguous integers in {identity}")
        row = dict(zip(key, identity))
        row.update(_audit_segment(g["gt"].map(LABELS).to_numpy(), g["pred"].map(LABELS).to_numpy(),
                                  _bool_values(g["ball_missing"]), float(g["fps"].iloc[0])))
        groups.append(row)
    providers = {}
    for provider, g in f.groupby("provider", sort=True):
        summaries = [_audit_segment(x.sort_values("frame")["gt"].map(LABELS).to_numpy(),
                                    x.sort_values("frame")["pred"].map(LABELS).to_numpy(),
                                    _bool_values(x.sort_values("frame")["ball_missing"]),
                                    float(x["fps"].iloc[0]))
                     for _, x in g.groupby(key[1:], sort=True)]
        providers[str(provider)] = {"matches": int(g["match_id"].nunique()),
                                    "segments": len(summaries),
                                    "wrong_team_frames": sum(x["wrong_team_frames"] for x in summaries),
                                    "false_d_frames": sum(x["false_d_frames"] for x in summaries),
                                    "split_ground_truth_spells": sum(x["split_ground_truth_spells"] for x in summaries),
                                    "merge_predicted_spells": sum(x["merge_predicted_spells"] for x in summaries),
                                    "segment_details": summaries}
    return {"schema": 1, "input_rows": len(f), "match_segments": groups, "providers": providers,
            "interpretation": "Descriptive diagnostics on supplied external predictions; no organizer labels or pooled deployment scores are implied."}


def audit_archive(path: Path) -> dict:
    """Audit one frozen prediction NPZ emitted by dense_label_weight_v1."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        required = {"prediction", "labels", "period", "fps", "timestamps", "ball_observed",
                    "source", "match", "view", "replicate", "seed", "arm", "fold"}
        absent = required - set(archive.files)
        if absent:
            raise ValueError(f"prediction archive missing arrays: {', '.join(sorted(absent))}")
        scalar = lambda name: str(np.asarray(archive[name]).reshape(()).item())
        frame = frame_table(scalar("source"), scalar("match"), archive["labels"],
                            archive["prediction"], archive["ball_observed"], archive["period"],
                            archive["timestamps"], float(np.asarray(archive["fps"]).reshape(())))
        details = {name: scalar(name) for name in
                   ("source", "match", "view", "replicate", "seed", "arm", "fold")}
        details["archive"] = path.name
        details["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {**details, **audit(frame)}


def summarize_archives(paths) -> dict:
    """Summarize matched archive realizations equally by provider, then match.

    Raw totals are retained separately. For rates, seeds are averaged within a
    provider-match first, matches within provider next, and providers equally
    last. Views and corruption replicates remain separate cells.
    """
    reports = [audit_archive(Path(p)) for p in paths]
    cells = {}
    rate_fields = ("wrong_team_rate_scored", "false_d_rate_scored", "missed_dead_rate_scored",
                   "wrong_team_rate_ball_missing", "wrong_team_rate_ball_observed",
                   "false_d_rate_ball_missing", "false_d_rate_ball_observed")
    for r in reports:
        cell = (r["arm"], r["view"], r["replicate"])
        provider, match, seed = r["source"], r["match"], r["seed"]
        c = cells.setdefault(cell, {"raw_counts": {}, "match_realizations": {}})
        totals = c["raw_counts"]
        for field in ("frames", "scored_frames", "wrong_team_frames", "false_d_frames",
                      "missed_dead_frames", "split_ground_truth_spells", "merge_predicted_spells",
                      "gt_alive_spells", "pred_alive_spells", "matched_transition_count",
                      "unmatched_gt_transition_count", "unmatched_pred_transition_count",
                      "scored_frames_ball_missing", "scored_frames_ball_observed",
                      "wrong_team_frames_ball_missing", "wrong_team_frames_ball_observed",
                      "false_d_frames_ball_missing", "false_d_frames_ball_observed"):
            totals[field] = totals.get(field, 0) + sum(s[field] for s in r["match_segments"])
        for (provider2, match2), segments in _group_segments(r).items():
            scored = sum(s["scored_frames"] for s in segments)
            missing = sum(s["scored_frames_ball_missing"] for s in segments)
            observed = sum(s["scored_frames_ball_observed"] for s in segments)
            gt_spells = sum(s["gt_alive_spells"] for s in segments)
            pred_spells = sum(s["pred_alive_spells"] for s in segments)
            def ratio(num, den): return float(num / den) if den else None
            val = {
                "provider": provider2, "match": match2, "seed": seed,
                "wrong_team_rate_scored": ratio(sum(s["wrong_team_frames"] for s in segments), scored),
                "false_d_rate_scored": ratio(sum(s["false_d_frames"] for s in segments), scored),
                "missed_dead_rate_scored": ratio(sum(s["missed_dead_frames"] for s in segments), scored),
                "wrong_team_rate_ball_missing": ratio(sum(s["wrong_team_frames_ball_missing"] for s in segments), missing),
                "wrong_team_rate_ball_observed": ratio(sum(s["wrong_team_frames_ball_observed"] for s in segments), observed),
                "false_d_rate_ball_missing": ratio(sum(s["false_d_frames_ball_missing"] for s in segments), missing),
                "false_d_rate_ball_observed": ratio(sum(s["false_d_frames_ball_observed"] for s in segments), observed),
                "split_per_gt_spell": ratio(sum(s["split_ground_truth_spells"] for s in segments), gt_spells),
                "merge_per_pred_spell": ratio(sum(s["merge_predicted_spells"] for s in segments), pred_spells),
                "transition_abs_error_seconds": _mean_defined(
                    [s["transition_shift_abs_median_seconds"] for s in segments]),
            }
            c["match_realizations"].setdefault((provider2, match2), []).append(val)
    result_cells = []
    for (arm, view, replicate), value in sorted(cells.items()):
        by_provider = {}
        for (provider, match), realizations in value["match_realizations"].items():
            p = by_provider.setdefault(provider, {})
            for field in rate_fields + ("split_per_gt_spell", "merge_per_pred_spell",
                                        "transition_abs_error_seconds"):
                means = [_mean_defined([r[field] for r in realizations])]
                p.setdefault(field, []).append(means[0])
        provider_summary = {provider: {"matches": len(next(iter(metrics.values()), [])),
                                       **{k: _mean_defined(v) for k, v in metrics.items()}}
                            for provider, metrics in sorted(by_provider.items())}
        balanced = {field: _mean_defined([entry.get(field) for entry in provider_summary.values()])
                    for field in rate_fields + ("split_per_gt_spell", "merge_per_pred_spell",
                                                "transition_abs_error_seconds")}
        result_cells.append({"arm": arm, "view": view, "replicate": replicate,
                             "provider_balanced_match_rates": balanced,
                             "provider_rates": provider_summary,
                             "raw_counts_across_archives": value["raw_counts"]})
    return {"schema": 1, "archive_count": len(reports), "cells": result_cells,
            "aggregation": "Rates: mean seeds within each match, mean matches within each provider, equal mean across providers. Raw frame/event totals are separate and are not used for provider-balanced conclusions.",
            "interpretation": "Descriptive non-mutually-exclusive error categories. Rates do not estimate counterfactual F1 loss."}


def _group_segments(report):
    return {(report["source"], report["match"]): report["match_segments"]}


def _mean_defined(values):
    kept = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(kept)) if kept else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = (summarize_archives(sorted(args.input.glob("*.npz"))) if args.input.is_dir()
              else audit_archive(args.input) if args.input.suffix.lower() == ".npz"
              else audit(pd.read_csv(args.input)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
