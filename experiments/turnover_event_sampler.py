"""TRAIN-only native H/A turnover events for within-record window sampling.

Events are label-continuous H/A changes with a full stable second on both
sides.  They only nominate existing windows from the same feature record;
targets, the original short-span sampler, and GAR selection remain untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random

import numpy as np

from experiments import short_original_diversity

H, A = 0, 1


def _array(record, full_name, fallback):
    return np.asarray(record.get(full_name, record[fallback]))


def eligible_events(record: dict) -> list[dict]:
    """List adjacent H/A turnovers with uninterrupted stable 1s each side.

    `period == 0` is valid. Native array adjacency is authoritative; where
    full timestamps are present they must also advance at native frame rate.
    """
    y = np.asarray(record["full_y"] if "full_y" in record else record["y"], dtype=np.int64)
    period = np.asarray(record["full_period"] if "full_period" in record else record["period"])
    if y.ndim != 1 or period.shape != y.shape:
        raise ValueError("full labels and periods must be aligned one-dimensional arrays")
    if not np.isin(y, (-1, 0, 1, 2)).all():
        raise ValueError("unexpected H/A/D/unknown label encoding")
    fps = float(record["native_fps"] if "native_fps" in record else record["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("native fps must be finite and positive")
    n = int(round(fps))
    if n < 1 or not np.isclose(n, fps, atol=1e-6):
        raise ValueError("one-second native contexts require integer native fps")
    timestamps = record.get("full_timestamps")
    if timestamps is not None:
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.shape != y.shape or not np.isfinite(timestamps).all():
            raise ValueError("full timestamps must align with native labels")
    indices = np.asarray(record.get("indices", np.arange(len(y))), dtype=np.int64)
    if indices.ndim != 1 or len(indices) == 0 and len(y):
        raise ValueError("sample indices must be a nonempty one-dimensional array")
    if np.any(np.diff(indices) <= 0):
        raise ValueError("sample indices must increase strictly")
    match = str(record.get("match", ""))
    events = []
    for edge in range(len(y) - 1):
        left, right = int(y[edge]), int(y[edge + 1])
        if left not in (H, A) or right not in (H, A) or left == right:
            continue
        lo, hi = edge + 1 - n, edge + 1 + n
        if lo < 0 or hi > len(y):
            continue
        if not np.all(period[lo:hi] == period[edge]):
            continue
        if not (np.all(y[lo:edge + 1] == left) and
                np.all(y[edge + 1:hi] == right)):
            continue
        if timestamps is not None:
            ts = timestamps[lo:hi]
            if len(ts) > 1 and not np.allclose(np.diff(ts), 1.0 / fps,
                                                rtol=0, atol=max(1e-6, 0.01/fps)):
                continue
        # Use full-native row position as an event coordinate. The same event
        # in paired/swapped views therefore receives the same identity.
        events.append({"identity": (match, int(period[edge]), int(edge)),
                       "edge": int(edge), "left": left, "right": right,
                       "context_start": int(lo), "context_stop": int(hi)})
    return events


def _core_covers(window, indices, event):
    _ri, start, size, _full, _alternate = window
    core = indices[start:start + size]
    if len(core) != size:
        return False
    return int(core[0]) <= event["context_start"] and int(core[-1]) >= event["context_stop"] - 1


def event_catalog(records: list[dict], groups: dict, *, hardness=None) -> tuple[dict, dict]:
    """Map eligible physical events to existing same-record core windows.

    `hardness`, when supplied, maps physical event identity to a scalar; lower
    values denote less reliable correct-HA signal. Catalog entries retain every
    event, while candidates exist only if its entire two-sided context is in a
    real trainer core. Paired records share one physical identity.
    """
    by_record: dict[int, dict[tuple, list]] = defaultdict(dict)
    windows_by_record = defaultdict(list)
    for candidates in groups.values():
        for window in candidates:
            windows_by_record[int(window[0])].append(window)
    physical: dict[tuple, dict] = {}
    for ri, record in enumerate(records):
        indices = np.asarray(record["indices"], dtype=np.int64)
        for event in eligible_events(record):
            identity = event["identity"]
            choices = [window for window in windows_by_record[ri]
                       if _core_covers(window, indices, event)]
            if not choices:
                continue
            item = dict(event)
            if hardness is not None:
                if identity not in hardness:
                    raise ValueError(f"missing frozen hardness for eligible event: {identity}")
                value = hardness[identity]
                if isinstance(value, dict):
                    value = min(float(value["pre_correct"]), float(value["post_correct"]))
                item["hardness"] = float(value)
                if not np.isfinite(item["hardness"]) or not 0 <= item["hardness"] <= 1:
                    raise ValueError("hardness must be a finite correct-HA probability")
            prior = physical.setdefault(identity, item)
            if "hardness" in item and prior["hardness"] != item["hardness"]:
                raise ValueError("paired records disagree on event hardness")
            by_record[ri][identity] = choices
    return dict(by_record), physical


def rank_hardest_half(physical: dict, *, per_match: bool = True) -> dict:
    """Select the lower half by min(pre, post) correct-HA confidence.

    Ties resolve by native identity, making selection deterministic. Input
    hardness may be a float or a mapping with `pre_correct`/`post_correct`.
    """
    strata = defaultdict(list)
    for identity, item in physical.items():
        value = item["hardness"]
        if isinstance(value, dict):
            value = min(float(value["pre_correct"]), float(value["post_correct"]))
        strata[str(identity[0]) if per_match else "all"].append((float(value), identity))
    selected = set()
    for rows in strata.values():
        rows.sort(key=lambda pair: (pair[0], pair[1]))
        selected.update(identity for _score, identity in rows[:(len(rows) + 1) // 2])
    return selected


class TurnoverSampler(short_original_diversity.OriginalDiversitySampler):
    """Preserve base/GAR streams while substituting only original windows."""
    def __init__(self, old_records, gar_records, names, seed, *, arm,
                 hardness=None, replacement_rate=.25):
        if arm not in ("fresh_control", "uniform_turnover", "hard_turnover"):
            raise ValueError("undeclared turnover arm")
        if not 0 <= float(replacement_rate) <= 1:
            raise ValueError("replacement rate must be in [0,1]")
        super().__init__(old_records, gar_records, names, seed)
        self.arm = arm
        self.replacement_rate = float(replacement_rate)
        self.request_rng = random.Random(int(seed) + 2301047)
        self.event_choice_rng = random.Random(int(seed) + 2301048)
        self.event_catalog, self.physical_events = event_catalog(
            old_records, self.groups["old"], hardness=hardness)
        self.all_event_catalog = self.event_catalog
        self.record_matches = {ri: str(record["match"])
                               for ri, record in enumerate(old_records)}
        if arm == "hard_turnover":
            if hardness is None:
                raise ValueError("hard_turnover requires frozen TRAIN hardness")
            selected = rank_hardest_half(self.physical_events)
            self.selected_event_ids = selected
            self.event_catalog = {ri: {identity: choices for identity, choices in entries.items()
                                       if identity in selected}
                                  for ri, entries in self.event_catalog.items()}
        else:
            self.selected_event_ids = set(self.physical_events)
        self.match_events = defaultdict(set)
        for ri, entries in self.event_catalog.items():
            for identity in entries:
                self.match_events[self.record_matches[ri]].add(identity)
        per_record_ids = defaultdict(list)
        for ri, match in self.record_matches.items():
            per_record_ids[match].append(set(self.event_catalog.get(ri, {})))
        for match, values in per_record_ids.items():
            if values and any(value != values[0] for value in values[1:]):
                raise ValueError(f"paired records have different selected event catalogs: {match}")
        self.event_bags = {}
        self.event_targets = Counter()
        self.nominated_physical_events = set()
        self.actual_counts = Counter()
        self.actual_exposures = Counter()
        self.actual_window_ids = []
        self.events_encountered = set()
        self.event_containing_draws = 0
        self.proposed_old_counts = Counter()
        self.proposed_old_exposures = Counter()
        self.request_mask = bytearray()
        self.counts["old"].update(turnover_requests=0, turnover_replacements=0,
                                   eligible_turnover_events=sum(map(len, self.event_catalog.values())))

    def replace(self, group, window):
        # Call the original diversity wrapper for every draw. This preserves
        # its .25 original short-span stream and the unchanged GAR schedule.
        if group != "old":
            return super().replace(group, window)
        before_counts, before_exposures = self.counts["old"].copy(), self.exposures.copy()
        selected = super().replace(group, window)
        requested = bool(self.request_rng.random() < self.replacement_rate)
        self.request_mask.append(int(requested))
        base_alternate = bool(selected[4])
        for key, value in (self.counts["old"] - before_counts).items():
            self.proposed_old_counts[key] += value
        for key, value in (self.exposures - before_exposures).items():
            self.proposed_old_exposures[key] += value
        # The parent sampler's count is a proposed window. Receipts separately
        # retain its RNG/sampler counters and report actual delivered windows.
        self.counts["old"] = before_counts
        self.exposures = before_exposures
        ri = int(selected[0])
        if requested:
            self.actual_counts["turnover_requests"] += 1
            entries = self.event_catalog.get(ri, {}) if self.arm != "fresh_control" else {}
            if entries:
                match = self.record_matches[ri]
                bag = self.event_bags.get(match)
                if not bag:
                    bag = sorted(self.match_events[match])
                    self.event_choice_rng.shuffle(bag)
                    self.event_bags[match] = bag
                event_id = bag.pop()
                if event_id not in entries:
                    raise ValueError("selected physical event is absent from paired record")
                self.event_targets[event_id] += 1
                self.nominated_physical_events.add(event_id)
                candidate = self.event_choice_rng.choice(entries[event_id])
                # Change only window geometry. Preserve the original draw's
                # clean/gap feature choice for matched arm schedules.
                selected = (candidate[0], candidate[1], candidate[2],
                            candidate[3], base_alternate)
                self.actual_counts["turnover_replacements"] += 1
        if selected[0] != ri:
            raise AssertionError("turnover replacement crossed a selected record")
        _record, start, size, _full, _alternate = selected
        record = self.records["old"][ri]
        labels = np.asarray(record["y"])[start:start + size]
        known = np.isin(labels, (0, 1))
        eligible = self.eligible["old"][ri][start:start + size] & known
        self.actual_counts["draws"] += 1
        self.actual_window_ids.append([int(ri), int(start), int(size), bool(selected[4])])
        self.actual_counts["known_ha_rows"] += int(known.sum())
        self.actual_counts["short_ha_rows"] += int(eligible.sum())
        self.actual_counts["short_containing_draws"] += int(eligible.any())
        self.actual_exposures["old/" + str(record["source"]) + "/" + str(record["match"])] += 1
        window_events = set()
        for event_id, candidates in self.all_event_catalog.get(ri, {}).items():
            if any((candidate[0], candidate[1], candidate[2]) ==
                   (selected[0], selected[1], selected[2])
                   for candidate in candidates):
                window_events.add(event_id)
        self.events_encountered.update(window_events)
        self.event_containing_draws += int(bool(window_events))
        return selected

    def receipt(self):
        result = super().receipt()
        result["counts"]["old"] = dict(self.actual_counts)
        result["exposures"] = {**result["exposures"], **dict(sorted(self.actual_exposures.items()))}
        result.update(base_proposal={"old_counts": dict(self.proposed_old_counts),
                                     "old_exposures": dict(sorted(self.proposed_old_exposures.items()))},
                      turnover_arm=self.arm,
                      turnover_replacement_rate=self.replacement_rate,
                      turnover_eligible_events=len(self.selected_event_ids),
                      turnover_selected_event_ids=[list(x) for x in sorted(self.selected_event_ids)],
                      turnover_nominated_events=len(self.nominated_physical_events),
                      turnover_nominated_event_ids=[list(x) for x in sorted(self.nominated_physical_events)],
                      turnover_events_encountered=[list(x) for x in sorted(self.events_encountered)],
                      turnover_event_containing_draws=self.event_containing_draws,
                      actual_old_window_ids=self.actual_window_ids,
                      turnover_request_count=int(sum(self.request_mask)),
                      turnover_request_mask_sha256=hashlib.sha256(bytes(self.request_mask)).hexdigest(),
                      turnover_catalog_events=sum(map(len, self.event_catalog.values())))
        return result
