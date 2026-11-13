import unittest

import numpy as np

from experiments import lgbm_boundary_gate_runtime as gate


def mask(text):
    return np.array([c == "D" for c in text])


class GateEditsTest(unittest.TestCase):
    def setUp(self):
        self.base = mask("LLLLDDDDDDLLLL")
        self.period = np.zeros(len(self.base), dtype=int)

    def apply(self, candidate, pd, base=None, period=None, threshold=0.6):
        base = self.base if base is None else base
        return gate.gate_edits(base, mask(candidate), np.full(len(base), pd) if np.isscalar(pd) else pd,
                               self.period if period is None else period, threshold)

    def test_confident_boundary_shifts_are_applied(self):
        # D run ends earlier and starts later; refiner says live.
        out = self.apply("LLLLLDDDDLLLLL", 0.1)
        np.testing.assert_array_equal(out, mask("LLLLLDDDDLLLLL"))
        # D run grows at both ends; refiner says dead.
        out = self.apply("LLLDDDDDDDDLLL", 0.9)
        np.testing.assert_array_equal(out, mask("LLLDDDDDDDDLLL"))

    def test_unconfident_edits_are_rejected(self):
        np.testing.assert_array_equal(self.apply("LLLLLDDDDLLLLL", 0.5), self.base)
        np.testing.assert_array_equal(self.apply("LLLDDDDDDDDLLL", 0.5), self.base)

    def test_threshold_is_inclusive(self):
        np.testing.assert_array_equal(self.apply("LLLDDDDDDDLLLL", 0.6), mask("LLLDDDDDDDLLLL"))
        np.testing.assert_array_equal(self.apply("LLLLLDDDDDLLLL", 0.4), mask("LLLLLDDDDDLLLL"))

    def test_run_creation_deletion_and_islands_are_rejected(self):
        for candidate, pd in (("LLLLLLLLLLLLLL", 0.0),   # delete the D run
                              ("LLLLDDLLDDLLLL", 0.0),   # live island splits the D run
                              ("LDDLDDDDDDLLLL", 1.0),   # new D island in a live stretch
                              ("DDDDDDDDDDLLLL", 1.0)):  # fill a whole live stretch
            np.testing.assert_array_equal(self.apply(candidate, pd), self.base)

    def test_single_edit_per_boundary_preserves_d_run_count(self):
        runs = lambda m: int(m[0]) + int(np.sum(m[1:] & ~m[:-1]))
        for candidate, pd in (("LLLLLLDDDDLLLL", 0.1), ("LLDDDDDDDDLLLL", 0.9),
                              ("LLLLDDDDLLLLLL", 0.1), ("LLLLDDDDDDDDLL", 0.9)):
            out = self.apply(candidate, pd)
            np.testing.assert_array_equal(out, mask(candidate))
            self.assertEqual(runs(out), runs(self.base))

    def test_opposite_confident_edits_at_one_boundary_can_split_a_run(self):
        # Documented limitation of the evaluated rule: a confident extension before a
        # run start together with a confident trim after it leaves a separate D island.
        pd = np.array([.1, .1, .9, .9, .1, .1, .9, .9, .9, .9, .1, .1, .1, .1])
        out = self.apply("LLDDLLDDDDLLLL", pd)
        np.testing.assert_array_equal(out, mask("LLDDLLDDDDLLLL"))

    def test_block_edge_live_stretch_can_gain_an_edge_run(self):
        # Documented limitation of the evaluated rule: at a block edge a confident
        # edit touching the edge is treated as a boundary shift.
        base = mask("LLLLDDDD")
        out = gate.gate_edits(base, mask("DDLLDDDD"), np.ones(8), np.zeros(8, dtype=int), 0.6)
        np.testing.assert_array_equal(out, mask("DDLLDDDD"))

    def test_period_break_isolates_runs(self):
        base = mask("DDDDDDDD")
        period = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        # Removing the first block's whole run is a deletion, not a boundary shift.
        out = gate.gate_edits(base, mask("LLLLDDDD"), np.zeros(8), period, 0.6)
        np.testing.assert_array_equal(out, base)

    def test_inputs_are_not_mutated_and_validated(self):
        base, cand = self.base.copy(), mask("LLLLLDDDDLLLLL")
        gate.gate_edits(base, cand, np.full(14, 0.1), self.period)
        np.testing.assert_array_equal(base, self.base)
        with self.assertRaises(ValueError):
            gate.gate_edits(base, cand[:-1], np.full(14, 0.1), self.period)
        with self.assertRaises(ValueError):
            gate.gate_edits(base, cand, np.full(14, 0.1), self.period, threshold=0.4)

    def test_identical_masks_return_baseline(self):
        np.testing.assert_array_equal(self.apply("LLLLDDDDDDLLLL", 0.5), self.base)


if __name__ == "__main__":
    unittest.main()
