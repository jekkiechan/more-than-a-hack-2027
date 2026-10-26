"""Leakage and boundary contracts for the temporal D ablation."""
import importlib.util
import unittest
import numpy as np
from experiments import learned_temporal as lt


class TemporalContracts(unittest.TestCase):
    def test_direction_and_discontinuity(self):
        x = np.arange(40, dtype=np.float32)[:, None]
        period = np.repeat([0, 1], 20)
        ordered = lt.ordered_features(x, ['speed'], period)
        # First three added columns are before, after, after-before at 1 s.
        self.assertEqual(ordered[10, 3], 10)
        self.assertTrue(np.isnan(ordered[20, 1]))
        self.assertTrue(np.isnan(ordered[19, 2]))
        reversed_x = np.concatenate([x[:20][::-1], x[20:][::-1]])
        reverse = lt.ordered_features(reversed_x, ['speed'], period)
        self.assertEqual(reverse[10, 3], -10)

    def test_chunks_and_upsampling_do_not_join_blocks(self):
        r = dict(period=np.repeat([0, 1], 8), native_n=80,
                 indices=np.r_[np.arange(0, 40, 5), np.arange(40, 80, 5)])
        for start, size, full in lt.chunks(r):
            self.assertEqual(len(np.unique(r['period'][full])), 1)
            self.assertLessEqual(start+size, len(r['period']))
        p = lt.upsample(np.repeat([0., 1.], 8), r, np.repeat([0, 1], 40))
        np.testing.assert_array_equal(p, np.repeat([0., 1.], 40))

    def test_missing_has_separate_mask(self):
        x = np.array([[0., np.nan], [1., 2.]], dtype=np.float32)
        z = lt.normalize(x, np.zeros(2), np.ones(2))
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(z[0, 2], 1)
        self.assertEqual(z[0, 3], 0)

    def test_features_ignore_ball_and_labels(self):
        t = np.arange(50)/5
        players = np.stack([np.column_stack((20+t+i, 20+t*.1)) for i in range(4)], axis=1)
        inputs = dict(players=players, teams=np.array([0, 0, 1, 1]), period=np.ones(50),
                      timestamps=t, fps=5., pitch=np.array([105, 68]))
        a, names = lt.player_features(dict(inputs, ball=np.zeros((50, 2)), labels=np.full(50, 'H')))
        b, other = lt.player_features(dict(inputs, ball=np.full((50, 2), np.nan), labels=np.full(50, 'D')))
        self.assertEqual(names, other)
        np.testing.assert_array_equal(a, b)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch not installed')
    def test_neural_shape_and_context(self):
        import torch
        torch.set_num_threads(1)
        for kind in ('tcn', 'bigru'):
            torch.manual_seed(1)
            net = lt.make_network(kind, 4).eval()
            x = torch.zeros(2, 100, 4)
            out = net(x)
            self.assertEqual(tuple(out.shape), (2, 100))
            # Future input must be able to influence earlier outputs.
            x[:, 51:55] = 1
            self.assertGreater(float((net(x)[:, 50]-out[:, 50]).abs().max().detach()), 0)


if __name__ == '__main__':
    unittest.main()
