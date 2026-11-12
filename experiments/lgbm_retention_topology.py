"""Topology-conserving boundary edits for the fixed LightGBM R2 refiner.

The refiner may move an existing dead-ball run's edges, but this adapter never
adds or removes a dead-ball run. It uses no targets or model probabilities.
"""
from __future__ import annotations

import numpy as np


_LABELS = {"H", "A", "D", "X"}
_MODES = {"both", "start", "end"}


def _runs(labels, value):
    """Half-open (start, stop) runs of value in a one-dimensional array."""
    mask = np.asarray(labels) == value
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist()))


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1])


def _ha_spell_sequence(labels):
    """Ordered H/A spells, keeping repeated teams separate across D runs."""
    seq = []
    previous = None
    for label in labels:
        label = str(label)
        if label in ("H", "A"):
            if label != previous:
                seq.append(label)
            previous = label
        else:
            previous = None
    return seq


def _blocks(model_period):
    period = np.asarray(model_period)
    if period.ndim != 1:
        raise ValueError("model_period must be one-dimensional")
    if not len(period):
        return []
    edges = np.r_[0, np.flatnonzero(period[1:] != period[:-1]) + 1, len(period)]
    return list(zip(edges[:-1].tolist(), edges[1:].tolist()))


def topology_repair(baseline_labels, refined_labels, model_period, mode="both"):
    """Move boundaries of locally one-to-one overlapping D runs.

    ``mode`` chooses which refined endpoints to apply. Unmatched or ambiguous
    runs remain at their baseline locations. Newly-live D prefixes/suffixes
    inherit the immediately adjacent baseline H/A label; absent anchors reject
    that endpoint. A proposed edit that changes the ordered H/A spell sequence
    causes a whole-block fallback.

    Returns ``(labels, diagnostics)``. Diagnostics contain only JSON-safe
    values; no input is mutated.
    """
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}")
    base = np.asarray(baseline_labels)
    refined = np.asarray(refined_labels)
    period = np.asarray(model_period)
    if base.ndim != 1 or refined.ndim != 1 or period.ndim != 1:
        raise ValueError("labels and model_period must be one-dimensional")
    if not (len(base) == len(refined) == len(period)):
        raise ValueError("baseline, refined and model_period lengths must match")
    base_s = base.astype(str)
    refined_s = refined.astype(str)
    if not set(np.unique(base_s)).issubset(_LABELS) or not set(np.unique(refined_s)).issubset(_LABELS):
        raise ValueError("labels must contain only H, A, D or X")

    result = np.array(base, copy=True)
    diagnostic = {
        "mode": mode,
        "blocks": [],
        "proposed_runs": [],
        "accepted_runs": [],
        "rejected_runs": [],
        "fallback_blocks": [],
        "changed_frames": 0,
        "d_runs_before": len(_runs(base_s, "D")),
        "d_runs_after": None,
    }

    for block_index, (lo, hi) in enumerate(_blocks(period)):
        b = base_s[lo:hi]
        r = refined_s[lo:hi]
        block_info = {"block": block_index, "start": lo, "stop": hi}
        if "X" in b:
            # An unknown/unguarded area is a hard barrier for this proposal.
            block_info.update(status="fallback", reason="baseline_contains_X")
            diagnostic["fallback_blocks"].append(block_index)
            diagnostic["blocks"].append(block_info)
            continue

        bruns = _runs(b, "D")
        rruns = _runs(r, "D")
        b_to_r = {i: [j for j, rr in enumerate(rruns) if _overlap(br, rr)]
                  for i, br in enumerate(bruns)}
        r_to_b = {j: [i for i, br in enumerate(bruns) if _overlap(br, rr)]
                  for j, rr in enumerate(rruns)}
        proposals = []
        for bi, br in enumerate(bruns):
            if len(b_to_r[bi]) != 1:
                if b_to_r[bi]:
                    diagnostic["rejected_runs"].append({
                        "block": block_index, "baseline_run": [lo + br[0], lo + br[1]],
                        "reason": "ambiguous_refined_overlap",
                    })
                continue
            ri = b_to_r[bi][0]
            if len(r_to_b[ri]) != 1:
                diagnostic["rejected_runs"].append({
                    "block": block_index, "baseline_run": [lo + br[0], lo + br[1]],
                    "refined_run": [lo + rruns[ri][0], lo + rruns[ri][1]],
                    "reason": "ambiguous_baseline_overlap",
                })
                continue
            rr = rruns[ri]
            new_start = rr[0] if mode in ("both", "start") else br[0]
            new_stop = rr[1] if mode in ("both", "end") else br[1]
            item = {
                "block": block_index,
                "baseline_run": [lo + br[0], lo + br[1]],
                "refined_run": [lo + rr[0], lo + rr[1]],
                "proposed_run": [lo + new_start, lo + new_stop],
                "baseline_run_index": bi,
                "local_proposed": (new_start, new_stop),
                "local_baseline": br,
                "rejected_endpoints": [],
            }
            # A D-to-live edge needs a team label from the adjacent baseline
            # live spell. If absent, preserve that baseline edge.
            if new_start > br[0] and (br[0] == 0 or b[br[0] - 1] not in ("H", "A")):
                item["local_proposed"] = (br[0], item["local_proposed"][1])
                item["rejected_endpoints"].append("start_no_preceding_HA_anchor")
            if new_stop < br[1] and (br[1] >= len(b) or b[br[1]] not in ("H", "A")):
                item["local_proposed"] = (item["local_proposed"][0], br[1])
                item["rejected_endpoints"].append("end_no_following_HA_anchor")
            ns, ne = item["local_proposed"]
            item["proposed_run"] = [lo + ns, lo + ne]
            diagnostic["proposed_runs"].append({k: v for k, v in item.items()
                                                if not k.startswith("local_")})
            proposals.append(item)

        # Reject invalid spans, runs that cross/touch another baseline D run,
        # and mutually colliding proposals. Keep all other baseline runs.
        invalid = set()
        for i, item in enumerate(proposals):
            ns, ne = item["local_proposed"]
            bi = item["baseline_run_index"]
            br = item["local_baseline"]
            if ns < 0 or ne > len(b) or ns >= ne:
                invalid.add(i)
                item["reject_reason"] = "invalid_output_run"
                continue
            for other_i, other in enumerate(bruns):
                if other_i == bi:
                    continue
                # Touching runs would merge into one output D run.
                if ns <= other[1] and ne >= other[0]:
                    invalid.add(i)
                    item["reject_reason"] = "crosses_or_touches_other_baseline_D_run"
                    break
        for i, first in enumerate(proposals):
            for j in range(i + 1, len(proposals)):
                second = proposals[j]
                a, bnd = first["local_proposed"]
                c, d = second["local_proposed"]
                if a <= d and bnd >= c:
                    invalid.update((i, j))
                    first["reject_reason"] = second["reject_reason"] = "overlapping_or_touching_output_runs"

        proposed_block = np.array(b, copy=True)
        accepted = []
        for i, item in enumerate(proposals):
            br = item["local_baseline"]
            if i in invalid:
                diagnostic["rejected_runs"].append({
                    "block": block_index, "baseline_run": item["baseline_run"],
                    "refined_run": item["refined_run"],
                    "proposed_run": item["proposed_run"],
                    "reason": item.get("reject_reason", "invalid_output_run"),
                })
                continue
            ns, ne = item["local_proposed"]
            # Remove this baseline D span, then place its proposed span.
            proposed_block[br[0]:br[1]] = b[br[0]:br[1]]
            proposed_block[ns:ne] = "D"
            # Restore H/A states for portions contracted out of D.
            if ns > br[0]:
                proposed_block[br[0]:ns] = b[br[0] - 1]
            if ne < br[1]:
                proposed_block[ne:br[1]] = b[br[1]]
            accepted.append(item)

        # Ensure every D edit remains a one-for-one run move, and that no
        # H/A spell was inserted, deleted, reordered, or split.
        if (_ha_spell_sequence(proposed_block) != _ha_spell_sequence(b)
                or len(_runs(proposed_block, "D")) != len(bruns)):
            diagnostic["fallback_blocks"].append(block_index)
            diagnostic["rejected_runs"].append({
                "block": block_index, "reason": "HA_sequence_or_D_run_count_changed",
            })
            block_info.update(status="fallback", reason="HA_sequence_or_D_run_count_changed")
        else:
            result[lo:hi] = proposed_block.astype(result.dtype, copy=False)
            block_info.update(status="applied", proposed=len(proposals), accepted=len(accepted))
            for item in accepted:
                diagnostic["accepted_runs"].append({
                    "block": block_index, "baseline_run": item["baseline_run"],
                    "refined_run": item["refined_run"], "output_run": item["proposed_run"],
                    "rejected_endpoints": item["rejected_endpoints"],
                })
        diagnostic["blocks"].append(block_info)

    diagnostic["changed_frames"] = int(np.count_nonzero(result.astype(str) != base_s))
    diagnostic["d_runs_after"] = len(_runs(result.astype(str), "D"))
    return result, diagnostic
