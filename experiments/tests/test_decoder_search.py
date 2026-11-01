import itertools
import unittest

import numpy as np

from experiments import decoder_search as ds, supervised


class DecoderContracts(unittest.TestCase):
    def test_matrix_replays_reference(self):
        rng = np.random.default_rng(218)
        for k in (1, 2, 3):
            for fps in (10, 25):
                p = rng.random((103, 3))
                p[:, k:] = 0
                p /= p.sum(axis=1, keepdims=True)
                period = np.r_[np.zeros(51), np.ones(52)]
                for cost in (0, .5, 3, 15):
                    config = dict(family="matrix", ha=cost, dead=cost)
                    np.testing.assert_array_equal(ds.decode(p, period, fps, config),
                                                  supervised.decode(p, period, fps, cost))

    def test_minimum_duration_exact_enumeration(self):
        rng = np.random.default_rng(97)
        for n in (1, 4, 7):
            p = rng.random((n, 3)) + .01
            lp = np.log(p / p.sum(axis=1, keepdims=True))
            cost = ds._costs(.7, .4, 10)
            minimum = np.array([3, 3, 1])
            path = ds._minimum_duration(lp, cost, minimum)
            def objective(candidate):
                edges = np.r_[0, np.flatnonzero(np.diff(candidate))+1, n]
                for u, v in zip(edges[1:-2], edges[2:-1]):
                    if v-u < minimum[candidate[u]]:
                        return -np.inf
                return sum(lp[t, state] for t, state in enumerate(candidate)) - sum(
                    cost[candidate[t-1], candidate[t]] for t in range(1, n))
            best = max(objective(np.array(c)) for c in itertools.product(range(3), repeat=n))
            self.assertAlmostEqual(objective(path), best, places=11)

    def test_team_swap_breaks_absent_states_and_no_mutation(self):
        rng = np.random.default_rng(18)
        p = rng.random((65, 3))
        p /= p.sum(axis=1, keepdims=True)
        period = np.r_[np.zeros(31), np.ones(34)]
        original = p.copy()
        for config in ds.configurations():
            result = ds.decode(p, period, 10, config)
            swapped = ds.decode(p[:, [1, 0, 2]], period, 10, config)
            expected = np.where(result == "H", "A", np.where(result == "A", "H", "D"))
            np.testing.assert_array_equal(swapped, expected, err_msg=str(config))
            split = np.r_[ds.decode(p[:31], period[:31], 10, config),
                          ds.decode(p[31:], period[31:], 10, config)]
            np.testing.assert_array_equal(result, split, err_msg=str(config))
            no_d = p.copy()
            no_d[:, 2] = 0
            self.assertNotIn("D", ds.decode(no_d, period, 10, config), str(config))
        np.testing.assert_array_equal(p, original)

    def test_invalid_input(self):
        for p in (np.zeros((2, 3)), np.full((2, 3), np.nan), -np.ones((2, 3))):
            with self.assertRaises(ValueError):
                ds.decode(p, np.zeros(2), 25, dict(family="matrix", ha=3, dead=3))


if __name__ == "__main__":
    unittest.main()
