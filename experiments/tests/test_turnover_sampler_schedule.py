"""Synthetic contracts for turnover replacement and paired training schedules."""
from __future__ import annotations

import random
import unittest

import numpy as np

from experiments import soccernet_capacity_model as cap
from experiments import learned_temporal as lt
from experiments import turnover_event_sampler as turnover
from experiments.tests.test_sn_dense_train import NAMES, old_record
from experiments.tests.test_short_label_quality import record as quality_record


def _record(match: str, *, swapped: bool = False, seed: int = 1) -> dict:
    """A 20-second 5 Hz record with four stable, window-contained turnovers."""
    record = old_record(match, seed, n=200)
    sampled = np.empty(200, dtype=np.int8)
    sampled[:40] = 0
    sampled[40:80] = 1
    sampled[80:120] = 0
    sampled[120:160] = 1
    sampled[160:] = 0
    full = np.repeat(sampled, 5)
    if swapped:
        sampled = np.where(np.isin(sampled, (0, 1)), 1 - sampled, sampled).astype(np.int8)
        full = np.where(np.isin(full, (0, 1)), 1 - full, full).astype(np.int8)
        h_idx, a_idx = NAMES.index("H_distance1"), NAMES.index("A_distance1")
        record["x"][:, [h_idx, a_idx]] = record["x"][:, [a_idx, h_idx]]
        record["alternate"][:, [h_idx, a_idx]] = record["alternate"][:, [a_idx, h_idx]]
    record["y"] = sampled
    record["full_y"] = full
    # Alternate feature choice is set on one core so the event override must
    # preserve whichever choice the base draw selected.
    n_windows = len(list(lt.chunks(dict(period=record["period"]))))
    record["replace_windows"] = np.zeros(n_windows, dtype=bool)
    record["replace_windows"][-1] = True
    record["alternate"] = record["alternate"].copy() + np.float32(0.125)
    return record


def _physical_ids(records):
    return {event["identity"] for record in records
            for event in turnover.eligible_events(record)}


class TurnoverSamplerScheduleTests(unittest.TestCase):
    def setUp(self):
        self.old = [_record("same-match", seed=1),
                    _record("same-match", swapped=True, seed=1),
                    _record("same-match", swapped=True, seed=1),
                    _record("same-match", swapped=True, seed=1)]
        self.gar = [quality_record("gar-a", 310), quality_record("gar-b", 320)]
        self.ids = _physical_ids(self.old)
        self.assertEqual(len(self.ids), 4)

    def _hardness(self):
        # Distinct scores make the bottom-half ranking unambiguous.
        scores = (0.12, 0.34, 0.76, 0.91)
        return {event_id: score for event_id, score in zip(sorted(self.ids), scores)}

    def _sampler(self, arm, *, rate=0.25):
        return turnover.TurnoverSampler(self.old, self.gar, NAMES, 73,
            arm=arm, hardness=self._hardness() if arm == "hard_turnover" else None,
            replacement_rate=rate)

    def test_three_arms_keep_record_orientation_alternate_and_gar_schedule(self):
        samplers = {arm: self._sampler(arm) for arm in
                    ("fresh_control", "uniform_turnover", "hard_turnover")}
        old_rng = {arm: random.Random(991) for arm in samplers}
        gar_rng = {arm: random.Random(992) for arm in samplers}
        proposed = {arm: [] for arm in samplers}
        actual_old = {arm: [] for arm in samplers}
        gar_windows = {arm: [] for arm in samplers}

        for _ in range(128):
            for arm, sampler in samplers.items():
                proposal = cap._draw_old(old_rng[arm], sampler.groups["old"], self.old)
                proposed[arm].append((proposal[0], proposal[4]))
                result = sampler.replace("old", proposal)
                actual_old[arm].append((result[0], result[4]))
                gar = cap._draw_new(gar_rng[arm], sampler.groups["new"])
                actual_gar = sampler.replace("new", gar)
                gar_windows[arm].append((actual_gar[0], actual_gar[1],
                                         actual_gar[2], actual_gar[4]))

        for arm in ("uniform_turnover", "hard_turnover"):
            self.assertEqual(proposed[arm], proposed["fresh_control"])
            self.assertEqual(gar_windows[arm], gar_windows["fresh_control"])
            self.assertEqual(old_rng[arm].getstate(), old_rng["fresh_control"].getstate())
            self.assertEqual(gar_rng[arm].getstate(), gar_rng["fresh_control"].getstate())
            self.assertEqual(samplers[arm].request_mask, samplers["fresh_control"].request_mask)
            self.assertEqual(samplers[arm].actual_exposures,
                             samplers["fresh_control"].actual_exposures)
            # Event replacement can move geometry, but it must retain the
            # selected paired record and clean/alternate feature choice.
            self.assertEqual([x[0:2] for x in actual_old[arm]],
                             [x[0:2] for x in actual_old["fresh_control"]])
        self.assertTrue(any(samplers["uniform_turnover"].actual_window_ids[i][1:3] !=
                            samplers["fresh_control"].actual_window_ids[i][1:3]
                            for i in range(len(samplers["fresh_control"].actual_window_ids))))

    def test_hard_arm_mines_bottom_half_of_unique_physical_events(self):
        sampler = self._sampler("hard_turnover", rate=1.0)
        selected = turnover.rank_hardest_half(sampler.physical_events)
        self.assertEqual(len(sampler.physical_events), 4)
        self.assertEqual(sampler.selected_event_ids, selected)
        self.assertEqual(len(selected), 2)
        self.assertTrue(selected.issubset(self.ids))
        self.assertEqual(sampler.receipt()["turnover_eligible_events"], 2)

        # Repeated requests traverse one physical-event bag shared across all
        # paired orientations, so a single cycle nominates each event once.
        windows = [w for choices in sampler.groups["old"].values() for w in choices]
        for i in range(2):
            sampler.replace("old", windows[i % len(windows)])
        self.assertEqual(set(sampler.nominated_physical_events), selected)
        self.assertEqual(len(sampler.event_targets), 2)
        self.assertEqual(sum(sampler.event_targets.values()), 2)

    def test_uniform_catalog_and_bags_count_each_physical_event_once(self):
        sampler = self._sampler("uniform_turnover", rate=1.0)
        self.assertEqual(len(sampler.physical_events), 4)
        self.assertEqual(sampler.receipt()["turnover_eligible_events"], 4)
        # All four paired records expose the same four identities; the sampler
        # cycles one match-level bag rather than four per-orientation bags.
        windows = [w for choices in sampler.groups["old"].values() for w in choices]
        for i in range(4):
            sampler.replace("old", windows[i % len(windows)])
        self.assertEqual(set(sampler.nominated_physical_events), self.ids)
        self.assertEqual(len(sampler.event_targets), 4)
        self.assertEqual(sum(sampler.event_targets.values()), 4)


if __name__ == "__main__":
    unittest.main()
