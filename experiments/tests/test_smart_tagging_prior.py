import unittest

import numpy as np

from experiments.smart_tagging_prior import (
    apply_smart_tagging_prior,
    classify_smart_tagging_event,
)


class SmartTaggingPriorTests(unittest.TestCase):
    def setUp(self):
        self.n = 100
        self.p = np.tile([0.40, 0.30, 0.30], (self.n, 1)).astype(np.float64)
        self.period = np.ones(self.n, dtype=np.int8)
        self.observed = np.ones(self.n, dtype=bool)
        self.observed[10:90] = False
        self.mapping = {"Home Club": "H", "Away Club": "A"}

    def event(self, code="Build Up", team="Home Club", start=10, end=90, **tags):
        return {"code": code, "start_frame": start, "end_frame": end,
                "tags": {"Team": team, **tags}}

    def apply_(self, p=None, events=None, **kwargs):
        return apply_smart_tagging_prior(
            self.p if p is None else p,
            [] if events is None else events,
            self.mapping,
            25.0,
            self.period,
            self.observed,
            **kwargs,
        )

    def test_empty_and_unusable_tags_are_exact_noops(self):
        for events in ([], [self.event(team="mystery")],
                       [self.event(code="Goal")],
                       [self.event(code="SET PIECES", Type="Kick Off (Start)")],
                       [self.event(start=12, end=12)],
                       [self.event(start=-10, end=-2)]):
            got = self.apply_(events=events)
            self.assertTrue(np.array_equal(got, self.p))
            self.assertEqual(got.dtype, self.p.dtype)

    def test_no_ball_mask_defaults_to_noop_when_missing_only(self):
        got = apply_smart_tagging_prior(self.p, [self.event()], self.mapping,
                                        25.0, self.period, None)
        self.assertTrue(np.array_equal(got, self.p))

    def test_attacking_phase_changes_only_conditional_ha_odds_on_missing_ball(self):
        got = self.apply_(events=[self.event()])
        self.assertGreater(got[50, 0] / got[50, 1], self.p[50, 0] / self.p[50, 1])
        self.assertTrue(np.array_equal(got[:, 2], self.p[:, 2]))
        self.assertTrue(np.array_equal(got[:10], self.p[:10]))
        self.assertTrue(np.array_equal(got[90:], self.p[90:]))
        self.assertTrue(np.allclose(got[20:80, :2].sum(axis=1),
                                    self.p[20:80, :2].sum(axis=1), rtol=0, atol=1e-15))
        self.assertTrue(np.array_equal(got[10:90][~self.observed[10:90], 2],
                                       self.p[10:90][~self.observed[10:90], 2]))
        self.assertTrue(np.allclose(got.sum(axis=1), 1.0))

    def test_edge_taper_and_observed_ball_gate(self):
        got = self.apply_(events=[self.event()])
        self.assertAlmostEqual(got[10, 0], self.p[10, 0])  # interval edge
        self.assertGreater(got[11, 0], self.p[11, 0])
        self.assertLess(got[11, 0] - self.p[11, 0], got[35, 0] - self.p[35, 0])
        self.assertGreater(got[35, 0], self.p[35, 0])
        self.observed[:] = True
        self.assertTrue(np.array_equal(self.apply_(events=[self.event()]), self.p))

    def test_defending_team_maps_to_opposing_possessor(self):
        got = self.apply_(events=[self.event(code="Defending in Middle Third", team="Home Club")])
        self.assertGreater(got[50, 1] / got[50, 0], self.p[50, 1] / self.p[50, 0])
        self.assertTrue(np.array_equal(got[:, 2], self.p[:, 2]))

    def test_duplicate_and_same_side_overlaps_are_invariant(self):
        one = self.apply_(events=[self.event()])
        duplicate = self.apply_(events=[self.event(), self.event()])
        overlap = self.apply_(events=[self.event(), self.event(start=20, end=80)])
        self.assertTrue(np.array_equal(one, duplicate))
        self.assertTrue(np.array_equal(one, overlap))

    def test_opposing_tags_conflict_to_neutral(self):
        got = self.apply_(events=[self.event(), self.event(team="Away Club")])
        self.assertTrue(np.array_equal(got, self.p))

    def test_team_swap_symmetry(self):
        events = [self.event(), self.event(code="Defending in Middle Third", team="Away Club")]
        original = self.apply_(events=events)
        swapped_p = self.p[:, [1, 0, 2]]
        swapped_mapping = {"Home Club": "A", "Away Club": "H"}
        swapped = apply_smart_tagging_prior(swapped_p, events, swapped_mapping,
                                            25.0, self.period, self.observed)
        self.assertTrue(np.allclose(original[:, [1, 0, 2]], swapped, rtol=0, atol=1e-15))

    def test_half_break_and_nonpositive_period_rows_block_evidence(self):
        period = np.ones(self.n, dtype=np.int8)
        period[50:] = 2
        event = self.event(start=10, end=90, Half="1st Half")
        got = apply_smart_tagging_prior(self.p, [event], self.mapping, 25.0,
                                        period, self.observed)
        self.assertGreater(got[35, 0], self.p[35, 0])
        self.assertTrue(np.array_equal(got[50:], self.p[50:]))
        period[25:30] = 0
        got = apply_smart_tagging_prior(self.p, [self.event()], self.mapping,
                                        25.0, period, self.observed)
        self.assertTrue(np.array_equal(got[25:30], self.p[25:30]))

    def test_all_live_mass_and_probability_bounds_are_preserved(self):
        p = np.tile([0.0, 0.7, 0.3], (self.n, 1))
        got = self.apply_(p=p, events=[self.event()])
        self.assertTrue(np.isfinite(got).all())
        self.assertTrue((got >= 0).all())
        self.assertTrue(np.allclose(got.sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(got[:, 2], p[:, 2]))
        self.assertTrue(np.array_equal(got[50, :2].sum(), p[50, :2].sum()))

    def test_float32_dead_probability_is_bitwise_identical(self):
        p = self.p.astype(np.float32)
        got = self.apply_(p=p, events=[self.event()])
        self.assertEqual(got.dtype, np.float32)
        self.assertEqual(got[:, 2].tobytes(), p[:, 2].tobytes())

    def test_exact_official_taxonomy_and_neutral_codes(self):
        attack = ["BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD",
                  "BALL IN THE BOX", "CREATING CHANCES", "LONG BALL",
                  "ATTACKING TRANSITION"]
        defense = ["DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD",
                   "DEFENDING IN ATTACKING THIRD", "DEFENSIVE TRANSITION"]
        for code in attack:
            self.assertEqual(classify_smart_tagging_event(self.event(code=code), self.mapping),
                             ("attacking", "H"))
        for code in defense:
            self.assertEqual(classify_smart_tagging_event(self.event(code=code), self.mapping),
                             ("defending", "H"))
        for code in ["PLAYERS IN THE BOX", "GOALS", "SET PIECES", "unknown"]:
            self.assertEqual(classify_smart_tagging_event(
                self.event(code=code, Type="Build Up"), self.mapping), ("ignored", None))

    def test_zero_team_probability_stays_zero_and_timestamps_reset_taper(self):
        p = np.tile([0.0, 0.7, 0.3], (self.n, 1))
        got = self.apply_(p=p, events=[self.event()])
        self.assertTrue(np.array_equal(got[:, 0], p[:, 0]))
        self.assertTrue(np.array_equal(got[:, 2], p[:, 2]))

        times = np.arange(self.n, dtype=np.float64) / 25.0
        times[50:] += 10.0
        got = apply_smart_tagging_prior(self.p, [self.event()], self.mapping,
                                        25.0, self.period, self.observed,
                                        timestamps=times)
        self.assertEqual(got[49, 0], self.p[49, 0])
        self.assertEqual(got[50, 0], self.p[50, 0])
        self.assertGreater(got[60, 0], self.p[60, 0])

    def test_bad_shapes_and_team_mapping_fail_clearly(self):
        with self.assertRaises(ValueError):
            self.apply_(p=np.ones((self.n, 2)))
        with self.assertRaises(ValueError):
            apply_smart_tagging_prior(self.p, [], self.mapping, 25.0,
                                      self.period[:-1], self.observed)
        with self.assertRaises(ValueError):
            apply_smart_tagging_prior(self.p, [self.event()],
                                      {"Home Club": "away"}, 25.0,
                                      self.period, self.observed)


if __name__ == "__main__":
    unittest.main()
