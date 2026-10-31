"""Fixture and event-type balancing weights for sparse SoccerNet segments.

This consumes the merged continuous-segment schema from
``soccernet_segments.py``. It assigns weights only to already mapped sparse
event-team anchors. It never propagates labels, creates negatives, or fills
unknown rows.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np

ELIGIBLE_ACTIONS = frozenset(("PASS", "CROSS", "SHOT"))


def _scalar(value):
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError("Expected scalar segment identity")
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _game_id(record):
    match = str(_scalar(record["match"]))
    if not match.startswith("soccernet_"):
        raise ValueError("Segment match must have soccernet_<game_id> identity")
    return match.removeprefix("soccernet_")


def _digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_anchor_weights(records, training_fixtures, *, max_weight=5.0):
    """Build row weights for mapped anchors in merged SoccerNet TRAIN segments.

    ``records`` must be the records from ``soccernet_segments.load_records``
    after selection by the caller's whole-fixture TRAIN plan. Every supplied
    record is checked against ``training_fixtures``; an excluded/evaluation
    fixture causes an error. The record's ``anchor_frame_mapping`` supplies
    sparse mapped events with ``sampled_index``, ``action``, ``team``,
    ``sample_id`` and ``status`` from the preparation sidecar.

    For each fixture, each anchor receives
    ``n_fixture / (n_present_actions * n_fixture_action)``. This equalizes the
    event-type contribution within each fixture while retaining that fixture's
    overall anchor mass. Weights are normalized to global mean 1, then capped
    at ``max_weight``. Multiple actions mapped to one row are averaged before
    capping. A missing event type remains absent. Invalid, unmapped, and
    unknown anchors receive zero weight; a mapped target/team conflict fails
    closed. Returns
    ``(weights, provenance)`` where weights is one float32 vector per record,
    aligned to ``record['y']``.
    """
    if not records:
        raise ValueError("No SoccerNet segment records supplied")
    allowed = frozenset(str(value) for value in training_fixtures)
    if not allowed:
        raise ValueError("Explicit non-empty whole-fixture TRAIN allowlist is required")
    if not np.isfinite(max_weight) or max_weight < 1:
        raise ValueError("max_weight must be finite and at least 1")

    weights = []
    anchors = []
    row_anchors = Counter()
    masked = Counter()
    seen = set()
    records_meta = []
    for record_index, record in enumerate(records):
        required = ("y", "match", "source", "anchor_frame_mapping")
        if any(key not in record for key in required):
            raise ValueError("Segment record is missing sparse-label identity fields")
        if str(_scalar(record["source"])).lower() != "soccernet":
            raise ValueError("Segment source must be soccernet")
        game = _game_id(record)
        if game not in allowed:
            raise ValueError(f"Fixture {game!r} is outside whole-fixture TRAIN allowlist")
        y = np.asarray(record["y"])
        if y.ndim != 1 or not np.isin(y, (-1, 0, 1)).all():
            raise ValueError("Merged SoccerNet targets must remain sparse H/A or unknown")
        weights.append(np.zeros(len(y), dtype=np.float32))
        mapped_indices = set()
        records_meta.append({"record_index": record_index, "game_id": game,
                             "known_rows": int(np.isin(y, (0, 1)).sum())})
        mappings = record["anchor_frame_mapping"]
        if not isinstance(mappings, (list, tuple)):
            raise ValueError("anchor_frame_mapping must be a list")
        for event in mappings:
            if not isinstance(event, dict):
                masked["malformed_mapping"] += 1
                continue
            if event.get("status") != "mapped":
                masked[str(event.get("status") or "unknown_status")] += 1
                continue
            action = str(event.get("action", "")).upper()
            team = str(event.get("team", "")).lower()
            if action not in ELIGIBLE_ACTIONS:
                masked["unknown_or_ineligible_action"] += 1
                continue
            if team not in ("home", "away"):
                masked["unknown_team"] += 1
                continue
            try:
                raw_index = event["sampled_index"]
                index = int(raw_index)
                anchor_team = 0 if team == "home" else 1
            except (KeyError, TypeError, ValueError):
                masked["malformed_mapping"] += 1
                continue
            if isinstance(raw_index, (float, np.floating)) and raw_index != index:
                raise ValueError("Mapped anchor sampled_index must be an integer")
            if index < 0 or index >= len(y):
                masked["index_out_of_range"] += 1
                continue
            mapped_indices.add(index)
            if int(y[index]) != anchor_team:
                masked["target_metadata_conflict"] += 1
                raise ValueError("Mapped anchor team conflicts with the segment target")
            sample_id = str(event.get("sample_id", ""))
            identity = (game, sample_id)
            if not sample_id or identity in seen:
                masked["missing_or_duplicate_anchor_identity"] += 1
                continue
            seen.add(identity)
            row_anchors[(record_index, index)] += 1
            anchors.append({"record_index": record_index, "index": index,
                            "game_id": game, "action": action,
                            "team": team, "sample_id": sample_id})
        if set(np.flatnonzero(np.isin(y, (0, 1))).tolist()) != mapped_indices:
            raise ValueError("Segment has dense or unmatched known targets; sparse targets must match mapped anchors")

    if not anchors:
        raise ValueError("No valid mapped SoccerNet anchors remain after masking")
    counts = Counter((a["game_id"], a["action"]) for a in anchors)
    fixture_counts = Counter(a["game_id"] for a in anchors)
    actions_by_fixture = {game: sum(1 for candidate, _ in counts if candidate == game)
                          for game in fixture_counts}
    anchor_weights = [
        fixture_counts[a["game_id"]] /
        (actions_by_fixture[a["game_id"]] * counts[(a["game_id"], a["action"])])
        for a in anchors
    ]
    row_values = Counter()
    for anchor, weight in zip(anchors, anchor_weights):
        row_values[(anchor["record_index"], anchor["index"])] += weight
    for (record_index, index), value in row_values.items():
        weights[record_index][index] = np.float32(value / row_anchors[(record_index, index)])
    all_positive = np.concatenate([row[row > 0] for row in weights])
    scale = float(all_positive.mean())
    for row in weights:
        row /= scale
        np.minimum(row, float(max_weight), out=row)

    canonical = sorted((a["game_id"], a["action"], a["sample_id"], a["team"],
                        a["record_index"], a["index"]) for a in anchors)
    provenance = {
        "mechanism": "within_fixture_event_balance_weights_v1",
        "weight_application": "mapped event-team anchor rows only; all other rows weight 0",
        "target_sources": ["anchor_frame_mapping", "y"],
        "training_fixtures": sorted(allowed),
        "eligible_fixtures": sorted({a["game_id"] for a in anchors}),
        "anchor_count": len(anchors),
        "counts_by_fixture_event": {
            f"{game}/{action}": count
            for (game, action), count in sorted(counts.items())},
        "max_weight": float(max_weight),
        "actual_weight_min": float(min(float(row[row > 0].min()) for row in weights if np.any(row > 0))),
        "actual_weight_max": float(max(float(row.max()) for row in weights)),
        "global_mean_before_cap": scale,
        "masked_counts": dict(sorted(masked.items())),
        "anchor_index_sha256": _digest(canonical),
        "record_summary": records_meta,
        "team_symmetric": True,
    }
    return weights, provenance

