import unittest
import numpy as np

from experiments.boundary_relocation_v1 import relocate, DISPLACEMENT_PENALTY


class BoundaryRelocationTests(unittest.TestCase):
    def sample(self):
        y = np.array(list("HHHHHAAAAAHHHHHDDXXAAAHHH"))
        p = np.full((len(y), 3), .05)
        p[:, 0] = np.where(y == "H", .9, .1)
        p[:, 1] = np.where(y == "A", .9, .1)
        p[:, 2] = np.where(y == "D", .9, .05)
        p /= p.sum(axis=1, keepdims=True)
        return y, p

    def test_baseline_noop_and_invariants(self):
        y, p = self.sample(); n = len(y)
        t = np.arange(n) / 25
        period = np.ones(n, int)
        self.assertTrue(np.array_equal(relocate(p, y, period, t, 25), y))
        out = relocate(p, y, period, t, 25, "shift_0.4")
        self.assertTrue(np.array_equal(out[np.isin(y, ["D", "X"])], y[np.isin(y, ["D", "X"])]))
        self.assertEqual(np.sum(out[1:] != out[:-1]), np.sum(y[1:] != y[:-1]))

    def test_gap_prevents_boundary_crossing_and_swap_equivariance(self):
        y, p = self.sample(); n = len(y); t = np.arange(n)/25
        period = np.ones(n, int); period[12:] = 2
        out = relocate(p, y, period, t, 25, "shift_0.8")
        self.assertEqual(out[12], y[12])
        swap = p[:, [1, 0, 2]]
        sy = np.where(y == "H", "A", np.where(y == "A", "H", y))
        swapped = relocate(swap, sy, period, t, 25, "shift_0.8")
        expect = np.where(out == "H", "A", np.where(out == "A", "H", out))
        np.testing.assert_array_equal(swapped, expect)

    def test_invalid_strength(self):
        y, p = self.sample(); n = len(y)
        with self.assertRaises(ValueError):
            relocate(p, y, np.ones(n), np.arange(n)/25, 25, "wide")

    def test_posterior_median_matches_brute_force_and_differs_from_map(self):
        # One H/A boundary with deliberately asymmetric posterior mass.
        y = np.array(list("HHHHHHHHHH"))
        y[5:] = "A"
        p = np.zeros((10, 3)); p[:, 2] = .05
        p[:3, 0], p[:3, 1] = .9, .05
        p[3:6, 0], p[3:6, 1] = .55, .40
        p[6:, 0], p[6:, 1] = .25, .70
        p /= p.sum(axis=1, keepdims=True)
        t = np.arange(10)/10
        out = relocate(p, y, np.ones(10), t, 10, "shift_0.4")
        # Independently enumerate the posterior over the exact allowed splits.
        candidates = np.arange(1, 9)
        scores = []
        for c in candidates:
            scores.append(np.log(p[:c, 0]).sum() + np.log(p[c:, 1]).sum()
                          - DISPLACEMENT_PENALTY*abs(c-5))
        w = np.exp(scores-np.max(scores)); w /= w.sum()
        expected = candidates[np.searchsorted(np.cumsum(w), .5, side="left")]
        actual = np.flatnonzero(out == "A")[0]
        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, candidates[np.argmax(scores)])

    def test_randomized_run_invariants_and_shift_caps(self):
        rng = np.random.default_rng(17)
        for _ in range(40):
            runs = rng.integers(2, 7)
            states = np.array(["H" if i % 2 == 0 else "A" for i in range(runs)])
            lengths = rng.integers(2, 15, size=runs)
            y = np.repeat(states, lengths)
            y = np.r_[y, ["D", "D", "X", "X"], np.array(list("AAHHH"))]
            n = len(y); p = rng.random((n, 3)); p /= p.sum(axis=1, keepdims=True)
            t = np.arange(n)/25; period = np.ones(n, int)
            for strength, cap in (("shift_0.2", 5), ("shift_0.4", 10), ("shift_0.8", 20)):
                out = relocate(p, y, period, t, 25, strength)
                fixed = np.isin(y, ["D", "X"])
                np.testing.assert_array_equal(out[fixed], y[fixed])
                # Each original H/A chain keeps its exact sequence of runs.
                self.assertEqual(list(_runs(out[:sum(lengths)])), list(_runs(y[:sum(lengths)])))
                old_edges = np.flatnonzero(y[1:sum(lengths)] != y[:sum(lengths)-1])+1
                new_edges = np.flatnonzero(out[1:sum(lengths)] != out[:sum(lengths)-1])+1
                np.testing.assert_array_equal(np.abs(new_edges-old_edges) <= cap, np.ones(len(old_edges), bool))

    def test_timestamp_reset_is_a_block_boundary(self):
        y, p = self.sample(); n = len(y)
        t = np.r_[np.arange(12)/25, np.arange(n-12)/25]
        period = np.r_[np.ones(12, int), np.full(n-12, 2)]
        out = relocate(p, y, period, t, 25, "shift_0.4")
        self.assertEqual(out[11], y[11]); self.assertEqual(out[12], y[12])


if __name__ == "__main__": unittest.main()


def _runs(labels):
    return tuple(labels[np.r_[True, labels[1:] != labels[:-1]]])
