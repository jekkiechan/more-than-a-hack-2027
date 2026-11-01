"""Synthetic contracts for the challenge-calibrated feed simulation."""
import unittest

import numpy as np

from experiments import feed_like_v2 as v2


def synthetic(n=25*240, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)/25
    players = np.stack([np.column_stack((52.5+30*np.sin(t/7+i), 34+20*np.cos(t/9+i))) for i in range(22)], 1)
    ball = np.column_stack((52.5+40*np.sin(t/5), 34+25*np.cos(t/6)))
    labels = np.array(["H"]*n)
    labels[(t % 60) > 45] = "D"
    labels[(t % 60) < 20] = "A"
    data = dict(ball=ball, players=players + rng.normal(0, .01, players.shape), teams=np.r_[[0]*11, [1]*11],
                period=np.ones(n, int), fps=np.asarray(25.), timestamps=t, pitch=np.array([105., 68.]))
    return data, labels


class FeedLikeV2Contracts(unittest.TestCase):
    def calibration(self):
        data, labels = synthetic()
        real = {k: np.array(v, copy=True) for k, v in data.items()}
        dead = labels == "D"
        real["ball"][dead] = np.nan  # a feed whose ball loss is in dead play only
        real["players"][dead, :3] = np.nan
        return v2.calibrate(real, labels)

    def test_calibration_recovers_state_dependent_ball_loss(self):
        cal = self.calibration()
        self.assertGreater(cal["dead_long"]["p_any"], .9)
        self.assertEqual(cal["isolated_live"]["rate_per_s"], 0.)
        profile = v2.profile({**synthetic()[0]}, synthetic()[1])
        self.assertEqual(profile["ball_missing"], 0.)

    def test_corrupt_is_deterministic_tracking_only_and_state_dependent(self):
        cal = self.calibration()
        data, labels = synthetic(seed=1)
        before = labels.copy()
        a = v2.corrupt(data, labels, cal, 7)
        b = v2.corrupt(data, labels, cal, 7)
        np.testing.assert_array_equal(np.isnan(a["ball"]), np.isnan(b["ball"]))
        np.testing.assert_array_equal(labels, before)
        self.assertNotIn("labels", a)
        miss = ~a["ball_observed"]
        self.assertGreater(miss[labels == "D"].mean(), .5)
        self.assertLess(miss[labels != "D"].mean(), .2)
        self.assertTrue(np.isfinite(data["ball"]).all())  # input not mutated

    def test_dropouts_top_up_to_target_only(self):
        cal = self.calibration()
        data, labels = synthetic(seed=2)
        players = np.array(data["players"], copy=True)
        native = v2.native_dropout_rate(players, labels, 25.)
        target = v2.target_dropout_rate(cal, players, labels, data["pitch"])
        self.assertEqual(native["dead"], 0.)
        self.assertGreater(target["dead"], 0.)
        # A view already losing players faster than the target receives no extra dropouts.
        heavy = players.copy()
        heavy[::50] = np.nan
        out = v2._player_dropouts(heavy.copy(), labels, 25., np.random.default_rng(0), cal, data["pitch"])
        np.testing.assert_array_equal(np.isnan(out), np.isnan(heavy))

    def test_crossing_swaps_keep_player_set_and_add_no_gaps(self):
        data, _ = synthetic(seed=3)
        players = np.array(data["players"], copy=True)
        players[:, 1] = players[:, 0] + .5  # two same-team tracks always within the radius
        out = v2._crossing_swaps(players.copy(), data["teams"], 25., np.random.default_rng(0), interval=(5., 6.))
        self.assertEqual(np.isnan(out).sum(), 0)
        np.testing.assert_allclose(np.sort(out.reshape(len(out), -1), axis=1),
                                   np.sort(players.reshape(len(players), -1), axis=1))


if __name__ == "__main__":
    unittest.main()
