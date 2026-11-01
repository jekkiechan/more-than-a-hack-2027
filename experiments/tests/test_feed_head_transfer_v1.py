import unittest
import numpy as np
from experiments.feed_head_transfer_v1 import cross


class ComponentCrossContracts(unittest.TestCase):
    def test_preserves_conditional_team_and_other_dead_probability(self):
        a = np.array([[.6, .2, .2], [0., 0., 1.]])
        b = np.array([[.05, .05, .9], [.2, .4, .4]])
        p = cross(a, b)
        np.testing.assert_allclose(p, [[.075, .025, .9], [.3, .3, .4]])
        np.testing.assert_allclose(p.sum(axis=1), 1)
        np.testing.assert_allclose(cross(a, a), a)
        np.testing.assert_allclose(cross(a[:, [1, 0, 2]], b[:, [1, 0, 2]]), p[:, [1, 0, 2]])

    def test_rejects_invalid_or_unaligned_inputs(self):
        with self.assertRaises(ValueError):
            cross(np.zeros((3, 3)), np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            cross(np.ones((3, 3)), np.ones((3, 3)))


if __name__ == '__main__':
    unittest.main()
