"""Synthetic contracts for the feed benchmark audit; no archive or organizer data is opened."""
import unittest

import numpy as np

from experiments import feed_benchmark_audit_v1 as study


class StateGapContracts(unittest.TestCase):
    def labels(self):
        # 20 s dead, 40 s home, 20 s dead, 40 s away at 25 Hz.
        return np.array(list("D" * 500 + "H" * 1000 + "D" * 500 + "A" * 1000))

    def test_dead_runs_blank_and_live_rate_is_bounded(self):
        labels = self.labels()
        ball = np.zeros((len(labels), 2))
        out = study.state_gaps(ball.copy(), labels, 25., np.random.default_rng(0),
                               live_rate=.1, dead_rate=1., pool=np.array([1., 2.]))
        missing = ~np.isfinite(out).all(axis=1)
        dead, live = labels == "D", labels != "D"
        self.assertGreater(missing[dead].mean(), .8)
        self.assertLess(missing[live].mean(), .25)
        self.assertGreater(missing[live].mean(), 0.)

    def test_no_live_gaps_when_rate_already_met(self):
        labels = self.labels()
        ball = np.zeros((len(labels), 2))
        ball[600:900] = np.nan  # native live gap already covers the requested 10%
        out = study.state_gaps(ball.copy(), labels, 25., np.random.default_rng(1),
                               live_rate=.1, dead_rate=0., pool=np.array([1.]))
        missing = ~np.isfinite(out).all(axis=1)
        np.testing.assert_array_equal(missing, ~np.isfinite(ball).all(axis=1))

    def test_equal_provider_mean_averages_replicates_then_matches(self):
        values = {("A", "m1", "r1"): 1., ("A", "m1", "r2"): 0., ("A", "m2", "r1"): 1.,
                  ("B", "m3", "r1"): 0.}
        mean, per = study.equal_provider(values)
        self.assertAlmostEqual(per["A"], .75)
        self.assertAlmostEqual(mean, .375)

    def test_decoder_grid_contains_promoted_setting(self):
        self.assertTrue(any((c["dt"], c["ht"], c["ha"], c["dead"]) == (4, 1, 3, 3) for c in study.GRID))
        self.assertEqual(len(study.GRID), 54)


if __name__ == "__main__":
    unittest.main()
