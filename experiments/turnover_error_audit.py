"""Audit current-frontier H-to-A turnover errors from the approved native cache.

This is a diagnostic only: it never fits models, changes the D mask, or uses
organizer labels. Run with ``python -m experiments.turnover_error_audit``.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/experiments/frontier_recovery_v1/frontier_native_probability_manifest.json"
RESULT = ROOT / "experiments/results/turnover_error_audit.json"
DETAILS = ROOT / "data/experiments/turnover_error_audit/events.json"
SCHEMA = "turnover_error_audit_v1"
TOLERANCE_SECONDS = 1.0
DURATION_BINS = ((0.0, 3.0, "<=3s"), (3.0, 10.0, "3-10s"), (10.0, float("inf"), ">10s"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def duration_bin(seconds: float) -> str:
    for low, high, name in DURATION_BINS:
        if low < seconds <= high or (low == 0 and seconds == 0):
            return name
    raise ValueError(f"duration outside bins: {seconds}")


def segments(labels: np.ndarray, period: np.ndarray, model_period: np.ndarray,
             timestamps: np.ndarray, fps: float):
    """Yield contiguous valid native/model-time blocks, breaking timestamp gaps."""
    n = len(labels)
    if any(len(x) != n for x in (period, model_period, timestamps)):
        raise ValueError("native arrays must have equal lengths")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if n == 0:
        return
    valid = (period > 0) & np.isfinite(timestamps)
    joins = np.zeros(n, dtype=bool)
    joins[1:] = (valid[1:] & valid[:-1] & (period[1:] == period[:-1])
                 & (model_period[1:] == model_period[:-1])
                 & np.isclose(np.diff(timestamps), 1.0 / fps, atol=1e-6, rtol=0))
    starts = np.flatnonzero(valid & ~joins)
    stops = np.flatnonzero(valid & ~np.r_[joins[1:], False]) + 1
    yield from zip(starts.tolist(), stops.tolist())


def one_to_one_matches(truth_times, predicted_times, tolerance=TOLERANCE_SECONDS):
    """Maximum-cardinality monotone matching for equal-width time windows."""
    truth = sorted(float(x) for x in truth_times)
    pred = sorted(float(x) for x in predicted_times)
    i = j = 0
    pairs = []
    while i < len(truth) and j < len(pred):
        if pred[j] < truth[i] - tolerance - 1e-12:
            j += 1
        elif truth[i] < pred[j] - tolerance - 1e-12:
            i += 1
        else:
            pairs.append((truth[i], pred[j]))
            i += 1
            j += 1
    return pairs


def _read_cell(path: Path):
    # Access only the seven authorized arrays; npz members remain lazy otherwise.
    with np.load(path, allow_pickle=False) as z:
        labels = z["labels"].astype("U1", copy=True)
        period = z["period"].copy()
        model_period = z["model_period"].copy()
        timestamps = z["timestamps"].copy()
        fps = float(z["fps"])
        probability = z["prob__approved_baseline"].copy()
        decoded = z["decoded__approved_baseline"].astype("U1", copy=True)
    if probability.shape != (len(labels), 2) or len(decoded) != len(labels):
        raise ValueError(f"misaligned approved baseline arrays in {path}")
    return labels, period, model_period, timestamps, fps, probability, decoded


def audit_cell(key: str, metadata: dict, arrays):
    labels, period, model_period, ts, fps, probs, decoded = arrays
    blocks = list(segments(labels, period, model_period, ts, fps))
    truth_all, truth_eligible, predictions = [], [], []
    for block_id, (start, stop) in enumerate(blocks):
        for i in range(start, stop - 1):
            direction = (labels[i], labels[i + 1])
            if direction in (("H", "A"), ("A", "H")):
                left = i
                while left > start and labels[left - 1] == direction[0]: left -= 1
                right = i + 1
                while right + 1 < stop and labels[right + 1] == direction[1]: right += 1
                pre = (i - left + 1) / fps
                post = (right - i) / fps
                event = {"time": float(ts[i + 1]), "edge_index": i, "pre_s": pre,
                         "post_s": post, "pre_bin": duration_bin(pre),
                         "post_bin": duration_bin(post), "eligible": pre >= 1.0 and post >= 1.0,
                         "direction": f"{direction[0]}->{direction[1]}",
                         "_block_id": block_id, "_block_start": start, "_block_stop": stop}
                truth_all.append(event)
                if event["eligible"]: truth_eligible.append(event)
            pred_direction = (decoded[i], decoded[i + 1])
            if pred_direction in (("H", "A"), ("A", "H")):
                predictions.append({"time": float(ts[i + 1]), "direction": f"{pred_direction[0]}->{pred_direction[1]}",
                                    "block_id": block_id})
    # Match independently by native/model-time block and direction.
    matches = {}
    for block_id, _ in enumerate(blocks):
        for direction in ("H->A", "A->H"):
            truths = [e for e in truth_eligible if e["_block_id"] == block_id and e["direction"] == direction]
            preds = [p["time"] for p in predictions if p["block_id"] == block_id and p["direction"] == direction]
            matches.update({(block_id, direction, t): p for t, p in
                            one_to_one_matches([e["time"] for e in truths], preds)})
    events = []
    for e in truth_eligible:
        t = e["time"]
        match_key = (e["_block_id"], e["direction"], t)
        is_matched = match_key in matches
        pre_col, post_col = ((0, 1) if e["direction"] == "H->A" else (1, 0))
        block = np.arange(e["_block_start"], e["_block_stop"])
        pre_mask = (ts[block] >= t - 1.0 - 1e-9) & (ts[block] < t - 1e-9)
        post_mask = (ts[block] >= t - 1e-9) & (ts[block] < t + 1.0 - 1e-9)
        pre_prob = float(np.mean(probs[block[pre_mask], pre_col])) if pre_mask.any() else None
        post_prob = float(np.mean(probs[block[post_mask], post_col])) if post_mask.any() else None
        evidence = ("insufficient" if pre_prob is None or post_prob is None else
                    "both_correct_over_0.5" if pre_prob > .5 and post_prob > .5 else "reversed_or_weak")
        near_lo, near_hi = t - 1.0 - 1e-9, t + 1.0 + 1e-9
        near_local = np.flatnonzero((ts[block] >= near_lo) & (ts[block] <= near_hi))
        near = block[near_local]
        d_any = bool(len(near) and np.any(decoded[near] == "D"))
        d_boundary = bool((decoded[e["edge_index"]] == "D") or (decoded[e["edge_index"] + 1] == "D"))
        reason = None if is_matched else ("baseline_D_at_boundary" if d_boundary else
                 "baseline_D_within_1s" if d_any else "no_D_within_1s")
        event = {**{k: v for k, v in e.items() if not k.startswith("_")}, "matched": is_matched,
                 "matched_prediction_time": matches.get(match_key), "miss_reason": reason,
                 "baseline_D_at_boundary": d_boundary, "baseline_D_within_1s": d_any,
                 "mean_correct_team_probability_pre_1s": pre_prob,
                 "mean_correct_team_probability_post_1s": post_prob,
                 "posterior_evidence": evidence}
        events.append(event)
    misses = [e for e in events if not e["matched"]]
    miss_cross = Counter(f"{e['miss_reason']}|{e['posterior_evidence']}" for e in misses)
    pre_bins, post_bins, duration_pairs = Counter(), Counter(), Counter()
    for e in truth_all:
        pre_bins[e["pre_bin"]] += 1; post_bins[e["post_bin"]] += 1
        duration_pairs[f"{e['pre_bin']}|{e['post_bin']}"] += 1
    return ({"cell": key, "provider": metadata["identity"]["provider"],
             "match": metadata["identity"]["match"], "view": metadata["identity"]["view"],
             "rep": int(metadata["identity"]["rep"]), "all_direct_HA_edges": len(truth_all),
             "eligible_HA_edges": len(truth_eligible), "predicted_HA_edges": len(predictions),
             "recovered_one_to_one_within_1s": len(matches), "misses": len(misses),
             "misses_baseline_D_at_boundary": sum(e["baseline_D_at_boundary"] for e in misses),
             "misses_baseline_D_within_1s": sum(e["baseline_D_within_1s"] for e in misses),
             "miss_reason_counts": dict(Counter(e["miss_reason"] for e in misses)),
             "miss_posterior_evidence_counts": dict(Counter(e["posterior_evidence"] for e in misses)),
             "misses_by_D_and_posterior": dict(miss_cross),
             "posterior_evidence_counts_all_eligible": dict(Counter(e["posterior_evidence"] for e in events)),
             "pre_duration_bins_all_edges": dict(pre_bins), "post_duration_bins_all_edges": dict(post_bins),
             "pre_post_duration_bins_all_edges": dict(duration_pairs)}, events)


def run():
    manifest_hash = sha256(MANIFEST)
    manifest = json.loads(MANIFEST.read_text())
    if len(manifest["cells"]) != 35:
        raise ValueError("approved manifest must contain 35 cells")
    rows, details = [], []
    for key, entry in manifest["cells"].items():
        path = ROOT / entry["path"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"native cache hash mismatch: {key}")
        row, events = audit_cell(key, entry, _read_cell(path))
        rows.append(row)
        details.extend({"cell": key, **event} for event in events)
    def aggregate(group_keys):
        groups = defaultdict(list)
        for r in rows: groups[tuple(r[k] for k in group_keys)].append(r)
        out = []
        additive = ("all_direct_HA_edges", "eligible_HA_edges", "predicted_HA_edges",
                    "recovered_one_to_one_within_1s", "misses", "misses_baseline_D_at_boundary",
                    "misses_baseline_D_within_1s")
        for key, items in sorted(groups.items()):
            item = dict(zip(group_keys, key)); item["cells"] = len(items)
            for field in additive: item[field] = sum(x[field] for x in items)
            for field in ("miss_reason_counts", "posterior_evidence_counts_all_eligible", "miss_posterior_evidence_counts",
                          "misses_by_D_and_posterior", "pre_duration_bins_all_edges",
                          "post_duration_bins_all_edges", "pre_post_duration_bins_all_edges"):
                c = Counter()
                for x in items: c.update(x[field])
                item[field] = dict(c)
            out.append(item)
        return out
    report = {"schema": SCHEMA, "diagnostic_only": True,
      "scope_correction": {"supersedes": "experiments/results/turnover_error_audit_scope_error.json",
                           "reason": "initial exploratory run counted only H-to-A edges; corrected run includes both directions and block-specific matching"},
      "question": "Can H/A emission supervision address current-frontier direct H-to-A or A-to-H turnover misses while the D mask remains fixed?",
      "input": {"manifest": "data/experiments/frontier_recovery_v1/frontier_native_probability_manifest.json",
                "manifest_sha256": manifest_hash, "cell_count": len(rows),
                "baseline_feed_like_v2_spell_f1": manifest["baseline_metrics"]["feed_like_v2"]["spell_f1"],
                "arrays_loaded": ["labels", "period", "model_period", "timestamps", "fps",
                                  "prob__approved_baseline", "decoded__approved_baseline"]},
      "method": {"truth": "direct adjacent H-to-A and A-to-H edges in valid period>0/model_period/timestamp-contiguous blocks",
                 "eligibility": "at least 1s contiguous on the starting team immediately before and opposite team immediately after each H/A edge",
                 "matching": "directional predicted H-to-A or A-to-H edge, matched independently within each native/model-time block, one-to-one maximum-cardinality chronological matching, within +/-1s",
                 "duration_bins": ["<=3s", "3-10s", ">10s"], "d_mask": "fixed approved baseline decoder output",
                 "sampling_unit_note": "35 cache cells cover 5 matches with repeated clean/v1/v2 views; edge counts are descriptive and repeated across views, not independent event samples.",
                 "interpretation_limit": "Weak posterior evidence is diagnostic association, not causal proof that a training change will recover the event.",
                 "historical_claude_turnover_count_difference": None,
                 "historical_note": "FRONTIER_POSTPROCESS_V1.md mentions Claude turnover-count diagnostics qualitatively; no comparable numeric count was available in this cache."},
      "summary": {"cells": rows, "by_provider_view": aggregate(["provider", "view"]),
                  "by_provider": aggregate(["provider"]), "overall": aggregate([])[0]},
      "source_freeze": {"diagnostic_source_sha256": sha256(Path(__file__)),
                        "native_manifest_sha256": manifest_hash,
                        "native_cell_hashes_verified": True},
      "private_event_details": "data/experiments/turnover_error_audit/events.json"}
    RESULT.parent.mkdir(parents=True, exist_ok=True); DETAILS.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    DETAILS.write_text(json.dumps({"schema": SCHEMA, "events": details}, indent=2) + "\n")
    print(json.dumps({"aggregate": str(RESULT), "details": str(DETAILS),
                      "overall": report["summary"]["overall"]}, indent=2))


if __name__ == "__main__":
    run()
