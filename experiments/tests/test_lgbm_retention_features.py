import unittest

import numpy as np

from experiments import lgbm_retention_features as retention


def fixture(n=20):
    t = np.arange(n, dtype=np.float64) / 10
    ball = np.column_stack((10 + t, 10 + t))
    observed = np.ones(n, dtype=bool)
    observed[[2, 3, 12, 13]] = False
    ball[2:4] = (1000, 1000)
    ball[12:14] = (2000, 2000)
    players = np.zeros((n, 4, 2), dtype=np.float64)
    players[:, :, 0] = np.arange(4)[None, :] + t[:, None]
    players[:, :, 1] = np.arange(4)[None, :] * 2
    inputs = {
        "ball": ball,
        "ball_observed": observed,
        "players": players,
        "pitch": np.array([105.0, 68.0]),
        "fps": 10.0,
        "period": np.r_[np.zeros(n // 2), np.ones(n - n // 2)],
        "timestamps": t,
    }
    p = np.tile([0.4, 0.4, 0.2], (n, 1))
    components = {"adopted": p.copy(), "calibrated": p.copy(), "ce": p.copy()}
    return inputs, components


class RetentionFeaturesTests(unittest.TestCase):
    def test_nomiss_removes_flags_and_observation_ages(self):
        inputs, components = fixture()
        names, matrix = retention.features(inputs, components, "nomiss")
        self.assertNotIn("ball_observed", names)
        self.assertNotIn("ball_since_obs_s", names)
        self.assertNotIn("ball_until_obs_s", names)
        self.assertNotIn("ball_beyond_next_obs", names)
        self.assertIn("ball_speed_ff", names)
        self.assertIn("ball_beyond_last_ff", names)
        self.assertEqual(matrix.shape, (len(inputs["ball"]), len(names)))

    def test_nomiss_forward_fill_respects_native_mask_and_resets(self):
        inputs, components = fixture()
        names, matrix = retention.features(inputs, components, "nomiss")
        beyond = matrix[:, names.index("ball_beyond_last_ff")]
        self.assertLess(beyond[2], 100.0)
        self.assertLess(beyond[3], 100.0)
        self.assertLess(beyond[10], 100.0)
        self.assertLess(beyond[12], 100.0)
        self.assertLess(beyond[13], 100.0)

    def test_nomiss_features_do_not_leak_across_blocks(self):
        inputs, components = fixture()
        names, original = retention.features(inputs, components, "nomiss")
        changed_inputs = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                          for k, v in inputs.items()}
        changed_components = {k: np.array(v, copy=True) for k, v in components.items()}
        boundary = len(inputs["ball"]) // 2
        changed_inputs["ball"][boundary:] += 500
        changed_inputs["players"][boundary:] += 500
        names2, changed = retention.features(changed_inputs, changed_components, "nomiss")
        self.assertEqual(names, names2)
        np.testing.assert_equal(original[:boundary], changed[:boundary])

    def test_view_weights_equalize_aggregate_training_mass(self):
        counts = {"clean": 5 * 100, "feed_like_v1": 10 * 100, "feed_like_v2": 20 * 100}
        masses = {view: count * retention.view_weight(view) for view, count in counts.items()}
        self.assertAlmostEqual(masses["clean"], masses["feed_like_v1"])
        self.assertAlmostEqual(masses["clean"], masses["feed_like_v2"])
        self.assertEqual(retention.view_weight("clean"), 1.0)
        self.assertEqual(retention.view_weight("feed_like_v1"), 0.5)
        self.assertEqual(retention.view_weight("feed_like_v2"), 0.25)
        with self.assertRaises(ValueError):
            retention.view_weight("unknown")


if __name__ == "__main__":
    unittest.main()
