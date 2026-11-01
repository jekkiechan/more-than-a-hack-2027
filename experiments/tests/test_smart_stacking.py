import unittest

import numpy as np

from experiments import smart_stacking as stack


class StackingContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.p = rng.dirichlet([2, 2, 1], size=(4, 100))
        self.period = np.repeat([1, 2], 50)
        self.labels = np.resize(np.array(["H", "A", "D", "X"]), 100)

    def cells(self):
        return [dict(provider=p, outer_provider="held", match=p, view=v,
                     probabilities=self.p, model_period=self.period, fps=25,
                     labels=self.labels) for p in ("one", "two")
                for v in ("clean", "feed_like_corrupt_1", "feed_like_corrupt_2")]

    def test_meta_target_provider_excluded(self):
        cells = self.cells()
        cells[0]["provider"] = "held"
        with self.assertRaisesRegex(ValueError, "leaked"):
            stack.fit(cells, "held", [0, 1])

    def test_other_outer_oof_arrays_rejected(self):
        cells = self.cells()
        cells[0]["outer_provider"] = "one"
        with self.assertRaisesRegex(ValueError, "checkpoint folds"):
            stack.fit(cells, "held", [0, 1])

    def test_temporal_context_respects_time_breaks(self):
        original, _ = stack.features(self.p, self.period, 25, True)
        changed = self.p.copy()
        changed[:, 50:] = [1, 0, 0]
        new, _ = stack.features(changed, self.period, 25, True)
        np.testing.assert_array_equal(original[:50], new[:50])

    def test_team_swap_symmetry_including_all_dead(self):
        p = self.p.copy()
        p[:, 0] = [0, 0, 1]
        h1, d1 = stack.features(p, self.period, 25, True)
        h2, d2 = stack.features(p[:, :, [1, 0, 2]], self.period, 25, True)
        np.testing.assert_allclose(h1, -h2, rtol=0, atol=2e-10)
        np.testing.assert_array_equal(d1, d2)
        model = stack.fit(self.cells(), "held", [0, 1, 2, 3], True)
        a = stack.predict(p, self.period, 25, model)
        b = stack.predict(p[:, :, [1, 0, 2]], self.period, 25, model)
        np.testing.assert_allclose(a, b[:, [1, 0, 2]], rtol=0, atol=2e-10)

    def test_dead_head_has_no_team_or_ball_feature(self):
        p = self.p.copy()
        p[:, :, :2] = p[:, :, [1, 0]]
        _, before = stack.features(self.p, self.period, 25, True)
        _, after = stack.features(p, self.period, 25, True)
        np.testing.assert_array_equal(before, after)

    def test_targets_absent_from_inference_and_normalization(self):
        model = stack.fit(self.cells(), "held", [0, 1, 2, 3], True)
        before = stack.predict(self.p, self.period, 25, model)
        self.labels[:] = "X"
        after = stack.predict(self.p, self.period, 25, model)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(before.sum(axis=1), 1., atol=1e-14)
        self.assertTrue(np.isfinite(before).all())
        self.assertTrue((before >= 0).all())

    def test_fixed_d_preservation_and_component_cross(self):
        model = stack.fit(self.cells(), "held", [0, 1, 2, 3], True)
        p = stack.predict(self.p, self.period, 25, model, dead_expert=1)
        np.testing.assert_allclose(p[:, 2], self.p[1, :, 2], rtol=0, atol=2e-16)
        crossed = stack.cross(self.p[0], self.p[1])
        np.testing.assert_allclose(crossed[:, 2], self.p[1, :, 2], atol=2e-16)

    def test_short_weights_preserve_unknowns_and_time_blocks(self):
        labels = np.array(list("HHHXXHHH"))
        periods = np.array([1, 1, 1, 1, 1, 2, 2, 2])
        w = stack.short_weights(labels, periods, 25)
        np.testing.assert_array_equal(w[3:5], 1.)
        np.testing.assert_array_equal(w[:3], w[-3:])
        self.assertTrue(np.all((w >= 1) & (w <= 4)))

    def test_bad_probabilities_rejected(self):
        for value in (np.full_like(self.p, np.nan), -self.p, np.zeros_like(self.p)):
            with self.assertRaises(ValueError):
                stack.features(value, self.period, 25, True)


if __name__ == "__main__":
    unittest.main()
