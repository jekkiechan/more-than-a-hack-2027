import itertools
import unittest

import numpy as np

from experiments import frontier_disagreement_v1 as study
from experiments import v2_spell_balanced_eval as balanced


class FrontierDisagreementTests(unittest.TestCase):
    def test_zero_lambda_replays_fixed_d_cost3_and_preserves_x(self):
        rng = np.random.default_rng(503)
        for fps in (10, 25):
            n = 61
            experts = [rng.dirichlet([1.4, 1.1], n) for _ in range(3)]
            mixture = sum(w * p for w, p in zip(study.WEIGHTS, experts))
            predecessor = np.array(["H", "A", "D"] * 20 + ["H"])
            period = np.r_[np.zeros(31, int), np.ones(30, int)]
            expected = balanced.decode_fixed_d(mixture, predecessor, period, fps, 3.0)
            actual = study.decode(experts, predecessor, period, fps, 0.0)
            np.testing.assert_array_equal(actual, expected)
        y = np.array(["D", "X", "D"])
        experts = [np.tile([.8, .2], (3, 1)) for _ in range(3)]
        np.testing.assert_array_equal(study.decode(experts, y, np.zeros(3, int), 25, .5), y)

    def test_gate_formula_bounds_and_swap_equivariance(self):
        rng = np.random.default_rng(19)
        experts = [rng.dirichlet([.7, 1.1], 50) for _ in range(3)]
        g = study.disagreement(experts)
        self.assertEqual(g.shape, (50,))
        self.assertTrue(np.isfinite(g).all())
        self.assertTrue(((g >= 0) & (g <= 1)).all())
        y = np.array(["H"] * 50)
        p = np.zeros(50, int)
        result = study.decode(experts, y, p, 25, .75)
        swapped_experts = [array[:, ::-1] for array in experts]
        swapped = study.decode(swapped_experts, np.where(y == "H", "A", "H"), p, 25, .75)
        expect = np.where(result == "H", "A", "H")
        np.testing.assert_array_equal(swapped, expect)

    def test_mapping_order_does_not_change_weighted_gate(self):
        rng = np.random.default_rng(309)
        arrays = {"ce": rng.dirichlet([.4, 1.7], 30),
                  "calibrated": rng.dirichlet([1.2, .7], 30),
                  "adopted": rng.dirichlet([.8, 1.1], 30)}
        normal = study.disagreement(arrays)
        reverse = study.disagreement({key: arrays[key] for key in
                                      ("adopted", "calibrated", "ce")})
        np.testing.assert_array_equal(normal, reverse)

    def test_zero_lambda_replays_extreme_conditional_probabilities(self):
        experts = [np.array([[1., 1e-15], [1e-15, 1.], [1., 0.], [0., 1.]])
                   for _ in range(3)]
        predecessor = np.array(["H", "A", "A", "H"])
        period = np.zeros(4, int)
        expected = balanced.decode_fixed_d(experts[0], predecessor, period, 25, 3.0)
        actual = study.decode(experts, predecessor, period, 25, 0.0)
        np.testing.assert_array_equal(actual, expected)

    def test_dynamic_path_matches_brute_force_objective(self):
        rng = np.random.default_rng(83)
        for n in range(2, 8):
            experts = [rng.dirichlet([.8, 1.3], n) for _ in range(3)]
            y = np.array(["H"] * n)
            period = np.zeros(n, int)
            lam = .5
            out = study.decode(experts, y, period, 10, lam)
            mixture = sum(w * p for w, p in zip(study.WEIGHTS, experts))
            gate = study._edge_gate(study.disagreement(experts), y, period, 10)
            logp = np.log(np.maximum(mixture, 1e-7))
            edge = 1 + lam * gate
            candidates = list(itertools.product(range(2), repeat=n))
            scores = [sum(logp[t, z[t]] for t in range(n)) - sum(
                3 * edge[t] * (z[t] != z[t - 1]) for t in range(1, n))
                for z in candidates]
            best = max(scores)
            actual = tuple(0 if label == "H" else 1 for label in out)
            score = sum(logp[t, actual[t]] for t in range(n)) - sum(
                3 * edge[t] * (actual[t] != actual[t - 1]) for t in range(1, n))
            self.assertAlmostEqual(score, best, places=11)

    def test_smoothing_pads_edges_and_resets_at_d_x_and_period_breaks(self):
        values = np.array([0., 1., 0., 1.])
        np.testing.assert_allclose(study._smooth_segment(values, 1), [1/3, 1/3, 2/3, 2/3])
        y = np.array(["H"] * 4 + ["D"] + ["A"] * 4 + ["X"] + ["H"] * 4)
        period = np.r_[np.zeros(10, int), np.ones(4, int)]
        gate = np.arange(len(y), dtype=float) / 20
        edge = study._edge_gate(gate, y, period, 10)
        self.assertEqual(edge[0], 0.)
        self.assertEqual(edge[4], 0.)  # D frame / either side
        self.assertEqual(edge[5], 0.)
        self.assertEqual(edge[9], 0.)  # X frame / either side
        self.assertEqual(edge[10], 0.)  # model-period restart

    def test_lambda_zero_and_d_x_only_spans_are_noop(self):
        experts = [np.tile([.9, .1], (8, 1)), np.tile([.8, .2], (8, 1)),
                   np.tile([.7, .3], (8, 1))]
        y = np.array(list("DDXXDDXX"))
        result = study.decode(experts, y, np.zeros(8, int), 25, 1.0)
        np.testing.assert_array_equal(result, y)


if __name__ == "__main__":
    unittest.main()
