import unittest
import numpy as np
from experiments.ball_exit_disjoint import disjoint


class DisjointContracts(unittest.TestCase):
    def test_changed_boundary_interval_uses_snap_unchanged_interval_uses_complement(self):
        original = np.array(list("HHHDDDHHHDDD"))
        snap = np.array(list("HHHHDDHHHDDD"))
        other = np.where(snap == "H", "A", snap)
        result = disjoint(snap, other, original, np.zeros(len(snap)))
        np.testing.assert_array_equal(result, list("HHHHDDAAADDD"))
        np.testing.assert_array_equal(result == "D", snap == "D")

    def test_time_breaks_are_separate_even_if_live_on_both_sides(self):
        original = np.array(list("HHHDDDHHHHHH"))
        snap = np.array(list("HHHHDDHHHHHH"))
        other = np.where(snap == "H", "A", snap)
        blocks = np.r_[np.zeros(3), np.ones(9)]
        np.testing.assert_array_equal(disjoint(snap, other, original, blocks), list("AAAH DDAAAAAA".replace(" ", "")))

    def test_all_intervals_unchanged_returns_complement(self):
        original = np.array(list("HHHDDDHHHDDD"))
        other = np.where(original == "H", "A", original)
        np.testing.assert_array_equal(disjoint(original, other, original, np.zeros(12)), other)

    def test_changed_dead_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            disjoint(np.array(list("HHD")), np.array(list("HHH")), np.array(list("HHD")), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
