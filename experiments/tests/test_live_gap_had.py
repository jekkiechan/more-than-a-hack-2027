"""Leakage, temporal isolation and three-class prediction contracts."""
import unittest
from unittest.mock import patch
import numpy as np
from experiments import had_gru, learned_temporal as lt, live_gap_experiment as exp, supervised
from experiments.tests.test_tracking_corruption import synthetic


class LiveGapContracts(unittest.TestCase):
    def test_training_corruption_preserves_targets_unknowns_and_native_gaps(self):
        inputs = synthetic()
        original = {k: v.copy() for k, v in inputs.items()}
        labels = np.resize(np.repeat(['H', 'A', 'D', 'X'], 500), 4000)
        calibration = dict(ball_missing_fraction=.41, gap_edges_seconds=[5., 10., 30.],
                           gap_bin_probability=[.5, .5])
        view = exp.live_gap_view(inputs, labels, calibration, 42)
        again = exp.live_gap_view(inputs, labels, calibration, 42)
        np.testing.assert_array_equal(view['ball'], again['ball'])
        inactive = np.isin(labels, ['D', 'X'])
        np.testing.assert_array_equal(view['ball'][inactive], inputs['ball'][inactive])
        self.assertTrue(np.isnan(view['ball'][50:80]).all())
        self.assertGreater(np.isnan(view['ball']).sum(), np.isnan(inputs['ball']).sum())
        self.assertNotIn('labels', view)
        for k in original:
            np.testing.assert_array_equal(inputs[k], original[k])

    def test_provider_excluded_from_all_training_and_no_eval_matches(self):
        evaluation = {p.stem for v in exp.dt.SPLITS.values() for p in v['eval']}
        for source in exp.dt.SPLITS:
            providers = exp.training_sources(source)
            self.assertNotIn(source, providers)
            matches = {p.stem for s in providers for p in exp.dt.SPLITS[s]['train']}
            self.assertFalse(matches & evaluation)

    def test_features_target_independent_and_team_equivariant(self):
        inputs = synthetic(150)
        labels = np.full(150, 'H')
        a = exp.feature_view(dict(inputs, labels=labels, owner=np.zeros(150)), labels, 's', 'm')
        b = exp.feature_view(dict(inputs, labels=np.full(150, 'D'), owner=np.ones(150)), labels, 's', 'm')
        np.testing.assert_array_equal(a['joint']['x'], b['joint']['x'])
        c = exp.feature_view(dict(inputs, teams=1-inputs['teams']), labels, 's', 'm')
        np.testing.assert_allclose(c['joint']['x'],
            supervised.swap_team_features(a['joint']['x'], a['joint']['names']), equal_nan=True, atol=1e-5)

    def test_windows_never_cross_blocks_and_swap_keeps_unknowns(self):
        x = np.arange(800, dtype=np.float32).reshape(400, 2)
        r = dict(x=x, y=np.resize([-1, 0, 1, 2], 400).astype(np.int8),
                 names=['H_distance', 'A_distance'], period=np.repeat([1, 2], 200))
        mixed = exp.mixed_record(r, dict(r, x=x+100), 'one', 0, swap=True)
        self.assertEqual(mixed['replace_windows'].sum(), 2)
        np.testing.assert_array_equal(mixed['y'][r['y'] == -1], -1)
        for _, _, full in lt.chunks(mixed):
            self.assertEqual(len(np.unique(r['period'][full])), 1)

    def test_joint_fit_unknown_mask_probabilities_and_swap(self):
        r = dict(source='s', match='m', x=np.sin(np.arange(800).reshape(400, 2)).astype(np.float32),
                 y=np.resize([-1, 0, 1, 2], 400).astype(np.int8),
                 names=['H_distance', 'A_distance'], period=np.repeat([1, 2], 200))
        records = [exp.mixed_record(r, r, 'm', 0, swap=swap) for swap in (False, True)]
        with patch.object(lt, 'EPOCHS', 1):
            model = had_gru.fit(records)
        self.assertEqual(model['optimizer_updates'], 1)
        self.assertEqual(model['windows_per_epoch'], 16)
        p = had_gru.predict(r, model)
        swapped = had_gru.predict(dict(r, x=r['x'][:, ::-1]), model)
        self.assertEqual(p.shape, (400, 3))
        np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-6)
        np.testing.assert_allclose(swapped, p[:, [1, 0, 2]], atol=1e-7)
        changed = had_gru.predict(dict(r, y=np.full(400, 2)), model)
        np.testing.assert_array_equal(p, changed)


if __name__ == '__main__':
    unittest.main()
