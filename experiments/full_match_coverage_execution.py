"""Audit observed SoccerNet-GAR TRAIN clip coverage and stoppage censoring.

This reads the versioned dense-record cache and emits aggregate coverage only.
It never creates labels, joins across match/time blocks, or reads organizer data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_RECEIPT = ROOT / "experiments/results/full_match_coverage_source_restore.json"
DEFAULT_DENSE_ROOT = ROOT / "data/experiments/sn_dense_v1/records"
DEFAULT_OUTPUT = ROOT / "experiments/results/full_match_coverage_execution.json"


def _contiguous_runs(points):
    """Return half-open runs of sorted integer points, coalescing only adjacency."""
    values = sorted(set(int(x) for x in points))
    if not values:
        return []
    runs = []
    start = prev = values[0]
    for value in values[1:]:
        if value != prev + 1:
            runs.append((start, prev + 1))
            start = value
        prev = value
    runs.append((start, prev + 1))
    return runs


def stoppage_episodes(out_frames, touch_starts, n_frames):
    """Pair observed OUT runs with the next observed touch in the same record.

    Episodes without a later touch are right-censored at this record boundary.
    The function deliberately does not join adjacent records or infer missing
    OUT events. Duration is measured from the first OUT frame to restart touch.
    """
    outs = _contiguous_runs(out_frames)
    touches = sorted(set(int(x) for x in touch_starts))
    episodes = []
    out_index = 0
    touch_index = 0
    while out_index < len(outs):
        start, end = outs[out_index]
        while touch_index < len(touches) and touches[touch_index] <= end - 1:
            touch_index += 1
        if touch_index == len(touches):
            status = ("both_record_edges_censored" if start == 0
                      else "right_censored_record_end")
            episodes.append({
                "out_start": start,
                "restart": None,
                "status": status,
                "duration_frames_observed": max(0, int(n_frames) - start),
            })
            break
        restart = touches[touch_index]
        status = ("left_censored_record_start" if start == 0
                  else "complete_within_record")
        episodes.append({
            "out_start": start,
            "restart": restart,
            "status": status,
            "duration_frames_observed": restart - start,
        })
        touch_index += 1
        out_index += 1
        while out_index < len(outs) and outs[out_index][0] < restart:
            out_index += 1
    return episodes


def _percentiles(values):
    values = [float(v) for v in values]
    if not values:
        return {"median": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p90": float(np.percentile(ordered, 90)),
        "max": max(ordered),
    }


def summarize_records(records):
    """Aggregate coverage by half; preserve block counts and record-local labels."""
    record_lengths_s = []
    period_intervals = defaultdict(list)
    period_fps = {}
    period_records = defaultdict(int)
    time_blocks = set()
    period_block_ids = defaultdict(set)
    stoppage_status = defaultdict(int)
    stoppage_seconds = []
    bounded_stoppage_seconds = []
    endpoint_pair_seconds = []
    dead_seconds = 0.0
    observed_seconds = 0.0
    native_frames_observed = 0
    games = set()

    for rec in records:
        frames = np.asarray(rec["full_frame_numbers"], dtype=np.int64)
        if len(frames) == 0:
            continue
        fps = float(rec["native_fps"])
        if not fps > 0:
            raise ValueError("native_fps must be positive")
        match = str(rec["match"])
        game_period = int(rec["game_period"])
        blocks = np.asarray(rec["full_period"], dtype=np.int64)
        if len(blocks) != len(frames) or len(np.unique(blocks)) != 1:
            raise ValueError("record crosses or lacks a full_period time block")
        block = int(blocks[0])
        period_key = (match, game_period)
        block_key = (match, game_period, block)
        start, end_exclusive = int(frames.min()), int(frames.max()) + 1
        if len(np.unique(frames)) != len(frames):
            raise ValueError("duplicate native frame within one record")
        if len(frames) > 1 and np.any(np.diff(frames) != 1):
            raise ValueError("record contains an internal frame gap or reversal")
        period_intervals[period_key].append((start, end_exclusive))
        period_fps[period_key] = fps
        period_records[period_key] += 1
        time_blocks.add(block_key)
        period_block_ids[period_key].add(block)
        games.add(match)
        n = len(frames)
        observed_seconds += n / fps
        native_frames_observed += n
        record_lengths_s.append(n / fps)

        out_frames = np.asarray(rec.get("out_frames", []), dtype=np.int64)
        touch_runs = np.asarray(rec.get("touch_runs", np.empty((0, 6))), dtype=np.int64)
        touch_starts = touch_runs[:, 0].tolist() if touch_runs.size else []
        episodes = stoppage_episodes(out_frames, touch_starts, n)
        for episode in episodes:
            status = episode["status"]
            stoppage_status[status] += 1
            duration = episode["duration_frames_observed"] / fps
            stoppage_seconds.append(duration)
            if episode["status"] == "complete_within_record":
                bounded_stoppage_seconds.append(duration)
                endpoint_pair_seconds.append(duration)
            elif episode["status"] == "left_censored_record_start":
                endpoint_pair_seconds.append(duration)
        provenance = np.asarray(rec.get("full_prov", []), dtype=np.int8)
        if len(provenance) == n:
            # Dense-label provenance code 3 means OUT-derived dead-ball state.
            dead_seconds += float(np.count_nonzero(provenance == 3)) / fps

    group_coverage = []
    gaps_s = []
    envelope_seconds = 0.0
    covered_union_seconds = 0.0
    for (match, period), intervals in sorted(period_intervals.items()):
        intervals.sort()
        union_frames = 0
        first = intervals[0][0]
        last = intervals[0][1]
        for start, end in intervals:
            if start > last:
                gap = start - last
                gaps_s.append(gap / period_fps[(match, period)])
                union_frames += last - first
                first, last = start, end
            elif end > last:
                last = end
        union_frames += last - first
        span_frames = max(b for _, b in intervals) - min(a for a, _ in intervals)
        fps = period_fps[(match, period)]
        covered = union_frames / fps
        envelope = span_frames / fps
        covered_union_seconds += covered
        envelope_seconds += envelope
        group_coverage.append({
            "match": match,
            "game_period": period,
            "records": period_records[(match, period)],
            "time_blocks": len(period_block_ids[(match, period)]),
            "observed_seconds_union": round(covered, 3),
            "first_to_last_clip_span_seconds": round(envelope, 3),
            "coverage_fraction_within_observed_envelope": round(covered / envelope, 6) if envelope else None,
        })

    n_stoppages = sum(stoppage_status.values())
    n_complete = stoppage_status["complete_within_record"]
    n_endpoint_pairs = n_complete + stoppage_status["left_censored_record_start"]
    return {
        "games": len(games),
        "matches": sorted(games),
        "records": len(record_lengths_s),
        "native_frames_observed": native_frames_observed,
        "observed_record_hours": round(observed_seconds / 3600, 4),
        "record_duration_seconds": _percentiles(record_lengths_s),
        "time_blocks_with_observed_records": len(time_blocks),
        "union_covered_hours": round(covered_union_seconds / 3600, 4),
        "first_to_last_clip_envelope_hours": round(envelope_seconds / 3600, 4),
        "coverage_fraction_within_summed_observed_envelopes": round(covered_union_seconds / envelope_seconds, 6) if envelope_seconds else None,
        "unobserved_gaps_inside_observed_envelope": {
            "count": len(gaps_s),
            "seconds": _percentiles(gaps_s),
        },
        "out_to_touch_episodes": {
            "observed_out_runs": n_stoppages,
            "complete_within_same_record_and_bounded": n_complete,
            "out_to_touch_endpoint_pairs_observed": n_endpoint_pairs,
            "left_censored_at_record_start": stoppage_status["left_censored_record_start"],
            "right_censored_at_record_end": stoppage_status["right_censored_record_end"],
            "both_record_edges_censored": stoppage_status["both_record_edges_censored"],
            "left_censored_stoppages_not_identifiable_from_clips": True,
            "bounded_complete_fraction_of_observed_out_runs": round(n_complete / n_stoppages, 6) if n_stoppages else None,
            "bounded_complete_duration_seconds_total": round(sum(bounded_stoppage_seconds), 3),
            "observed_endpoint_pair_duration_seconds_total": round(sum(endpoint_pair_seconds), 3),
            "observed_duration_seconds": _percentiles(stoppage_seconds),
        },
        "dead_provenance_hours": round(dead_seconds / 3600, 4),
        "time_blocks": group_coverage,
        "interpretation_limits": [
            "Coverage fractions use only the frame-number envelope from the first to last observed clip in each match half, not the full match duration.",
            "Gaps between records may reflect clip sampling, invalid frames, or time breaks; the report counts full_period blocks separately and does not imply state continuity across gaps.",
            "An OUT-to-touch pair is complete only when both events occur in one contiguous record; this is clip-level continuity, not full-match stoppage coverage.",
            "A stoppage whose OUT occurs before a clip begins cannot be identified from that clip's event stream.",
            "OUT-to-touch events are silver action metadata; no missing stoppage or restart is inferred from event anchors.",
        ],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_audit(dense_root=DEFAULT_DENSE_ROOT, output=DEFAULT_OUTPUT):
    dense_root = Path(dense_root)
    manifest_path = dense_root / "train" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    source_receipt_path = DEFAULT_SOURCE_RECEIPT
    source_receipt = json.loads(source_receipt_path.read_text())
    acquisition_manifest_path = ROOT / "data/independent/soccernet_gar/acquisition_manifest.json"
    game_summaries = []
    compact_records = []
    for filename, expected in sorted(manifest["game_files"].items()):
        path = dense_root / "train" / "games" / filename
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"dense record hash mismatch: {path}")
        payload = joblib.load(path)
        records = payload["records"]
        game_summary = summarize_records(records)
        game_summaries.append({"game_file": filename, "sha256": actual, **game_summary})
        audit_keys = ("full_frame_numbers", "native_fps", "match", "game_period",
                      "full_period", "out_frames", "touch_runs", "full_prov")
        compact_records.extend({key: rec[key] for key in audit_keys} for rec in records)
        del payload, records
    aggregate = summarize_records(compact_records)
    del compact_records
    report = {
        "schema": 1,
        "study": "full_match_coverage_execution_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_restore_receipt": str(source_receipt_path.relative_to(ROOT)),
        "source_restore_receipt_sha256": _sha256(source_receipt_path),
        "source_restore_status": source_receipt["status"],
        "raw_source_revision": source_receipt["revision"],
        "raw_source_train_bytes": source_receipt["total_bytes"],
        "source_clip_count": manifest["n_samples"],
        "dense_game_count": len(manifest["games"]),
        "dense_record_count": sum(manifest["game_records"].values()),
        "local_acquisition_manifest_sha256": _sha256(acquisition_manifest_path),
        "dense_recipe_source_sha256": manifest.get("source_sha256", {}),
        "dense_label_module_sha256": _sha256(ROOT / "experiments/soccernet_dense_labels.py"),
        "coverage_audit_module_sha256": _sha256(Path(__file__)),
        "dense_record_manifest_sha256": _sha256(manifest_path),
        "dense_record_manifest": str(manifest_path.relative_to(ROOT)),
        "split": "train only",
        "uses_valid_or_test": False,
        "uses_organizer_labels": False,
        "aggregate": aggregate,
        "per_game": game_summaries,
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-root", type=Path, default=DEFAULT_DENSE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run_audit(args.dense_root, args.output)
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
