"""Exact-parity checks for the opt-in postprocessing runtime candidate."""
import unittest

import numpy as np

from experiments import postprocessing_runtime as runtime
from experiments import supervised


class PostprocessingRuntimeTests(unittest.TestCase):
    def assert_matches_reference(self, probabilities, period, fps, cost):
        actual = runtime.decode(probabilities, period, fps, cost)
        expected = supervised.decode(probabilities, period, fps, cost)
        np.testing.assert_array_equal(actual, expected)

    def test_random_probabilities_match_at_several_frame_rates_and_costs(self):
        rng = np.random.default_rng(401)
        for dtype in (np.float32, np.float64):
            for fps in (5., 10., 25., 50.):
                probabilities = rng.uniform(.01, 1., size=(113, 3)).astype(dtype)
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                period = np.zeros(len(probabilities), dtype=np.int64)
                period[31:72] = 2
                period[72:] = 3
                for cost in (0, 3, 30):
                    with self.subTest(dtype=dtype, fps=fps, cost=cost):
                        self.assert_matches_reference(probabilities, period, fps, cost)

    def test_ties_zero_rows_and_one_to_three_available_classes_match(self):
        for nclasses in (1, 2, 3):
            probabilities = np.full((9, 3), .25)
            probabilities[:, nclasses:] = 0
            probabilities[4] = 0
            period = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
            with self.subTest(nclasses=nclasses):
                for cost in (0, 3):
                    self.assert_matches_reference(probabilities, period, 25., cost)

    def test_tie_predecessor_order_matches_for_asymmetric_state_values(self):
        # Construct tied stay/switch alternatives with each state index winning
        # the final predecessor tie in turn.
        for values in ([.5, .5, .01], [.5, .01, .5], [.01, .5, .5]):
            probabilities = np.asarray([values, values, values], dtype=float)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            self.assert_matches_reference(probabilities, np.zeros(3), 10., 0.)
            self.assert_matches_reference(probabilities, np.zeros(3), 10., 3.)

    def test_near_tied_float32_emissions_preserve_rounding_and_transition_ties(self):
        rng = np.random.default_rng(19)
        costs = (0., 1e-7, 1e-6, 1e-5, .0001, .01, 1., 3.)
        for trial in range(300):
            n = int(rng.integers(2, 30))
            probabilities = np.full((n, 3), np.float32(1 / 3), dtype=np.float32)
            probabilities += (rng.integers(-4, 5, size=(n, 3)).astype(np.float32)
                              * np.float32(1e-7))
            period = np.zeros(n, dtype=np.int32)
            if n > 4:
                period[n // 2:] = 1
            cost = costs[trial % len(costs)]
            with self.subTest(trial=trial, cost=cost):
                self.assert_matches_reference(probabilities, period, 25., cost)

    def test_masks_and_each_period_are_decoded_independently(self):
        probabilities = np.array([
            [.95, .03, .02], [.95, .03, .02], [.02, .96, .02],
            [.02, .96, .02], [.02, .02, .96], [.02, .02, .96],
        ])
        probabilities[:, 2] = 0
        period = np.array([1, 1, 1, 2, 2, 2])
        self.assert_matches_reference(probabilities, period, 25., 3.)

    def test_deterministic_edge_probe_has_expected_case_count(self):
        self.assertEqual(runtime.parity_case_count(seed=991), 2880)


if __name__ == "__main__":
    unittest.main()
