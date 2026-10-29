"""Unit tests for the label-free possession structure statistics."""
import math
import unittest
import numpy as np
from experiments.possession_rate import structure_stats, standardized_distance, VECTOR


def seq(s):
    return np.array(list(s))


class StructureStatsTests(unittest.TestCase):
    def stats(self, s, fps=1., visible=None, active=None, period=None):
        n = len(s)
        return structure_stats(seq(s), np.ones(n, bool) if active is None else np.asarray(active, bool),
                               np.ones(n, bool) if visible is None else np.asarray(visible, bool), fps, period)

    def test_counts_and_rates(self):
        # HH HH | D D | AAA | H : 3 possessions, 1 D stoppage, 1 direct switch (A->H)
        r = self.stats("HHHHDDAAAH", fps=1 / 6)  # 10 frames * 6 s = 1 minute
        self.assertAlmostEqual(r["active_minutes"], 1.0)
        self.assertEqual((r["n_poss"], r["n_d_stop"], r["n_switch"]), (3, 1, 1))
        self.assertAlmostEqual(r["poss_per_min"], 3.0)
        self.assertAlmostEqual(r["switch_per_min"], 1.0)
        self.assertAlmostEqual(r["share_D"], .2)
        self.assertAlmostEqual(r["share_H"] + r["share_A"] + r["share_D"], 1.0)

    def test_direct_switch_splits_possession_and_d_breaks_it(self):
        r = self.stats("HHAAHH")
        self.assertEqual((r["n_poss"], r["n_switch"]), (3, 2))
        r = self.stats("HHDHH")
        self.assertEqual((r["n_poss"], r["n_switch"], r["n_d_stop"]), (2, 0, 1))

    def test_durations_and_short_fraction(self):
        r = self.stats("HAAAHHHHHH", fps=1.)  # durations 1,3,6
        self.assertAlmostEqual(r["dur_mean"], 10 / 3)
        self.assertAlmostEqual(r["dur_q50"], 3.0)
        self.assertAlmostEqual(r["frac_lt3s"], 1 / 3)  # 3 s is not < 3 s

    def test_inactive_frames_break_runs_and_are_not_counted(self):
        active = [1, 1, 0, 0, 1, 1]
        r = self.stats("HHHHHH", active=active)
        self.assertEqual(r["n_poss"], 2)
        self.assertEqual(r["n_active_frames"], 4)
        self.assertEqual(r["n_switch"], 0)

    def test_x_labels_are_not_possessions(self):
        r = self.stats("HHXXAA")
        self.assertEqual((r["n_poss"], r["n_switch"]), (2, 0))

    def test_period_change_breaks_run_and_is_not_a_switch(self):
        r = self.stats("HHAA", period=[1, 1, 2, 2])
        self.assertEqual((r["n_poss"], r["n_switch"]), (2, 0))
        r = self.stats("HHHH", period=[1, 1, 2, 2])
        self.assertEqual(r["n_poss"], 2)

    def test_ball_visibility_split(self):
        # ball visible first 4 frames, missing last 4; possession starts decide attribution
        s = "HHAAHHAA"
        vis = [1, 1, 1, 1, 0, 0, 0, 0]
        r = self.stats(s, fps=1 / 6, visible=vis)  # 8 frames * 6 s = 48 s; 24 s each
        self.assertAlmostEqual(r["ball_visible_fraction"], .5)
        # runs: HH(v) AA(v) HH(m) AA(m)
        self.assertAlmostEqual(r["poss_per_min_visible"], 2 / .4)
        self.assertAlmostEqual(r["poss_per_min_missing"], 2 / .4)
        self.assertAlmostEqual(r["poss_ratio_missing_over_visible"], 1.0)
        # switches: AA(v), HH(m), AA(m) -> 1 visible, 2 missing
        self.assertAlmostEqual(r["switch_per_min_missing"] / r["switch_per_min_visible"], 2.0)

    def test_all_ball_visible_has_nan_missing_rate(self):
        r = self.stats("HHAA")
        self.assertTrue(math.isnan(r["poss_per_min_missing"]))
        self.assertTrue(math.isnan(r["poss_ratio_missing_over_visible"]))

    def test_empty_active_returns_nans_without_error(self):
        r = self.stats("HHAA", active=[0, 0, 0, 0])
        self.assertEqual(r["n_poss"], 0)
        self.assertTrue(math.isnan(r["poss_per_min"]))

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            structure_stats(seq("HH"), np.ones(3, bool), np.ones(2, bool), 1.)

    def test_standardized_distance(self):
        a = {k: 1.0 for k in VECTOR}
        b = {k: 3.0 for k in VECTOR}
        scale = {k: 2.0 for k in VECTOR}
        self.assertAlmostEqual(standardized_distance(a, b, scale), 1.0)
        self.assertAlmostEqual(standardized_distance(a, a, scale), 0.0)


if __name__ == "__main__":
    unittest.main()
