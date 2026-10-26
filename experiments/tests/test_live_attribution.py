"""Contracts for a live-only team head with a separately frozen D model."""
import unittest

import numpy as np

from experiments import live_attribution as la, refine_domain, refine_features


class TeamEstimator:
    def predict_proba(self, x):
        home = 1/(1+np.exp(x[:, 1]-x[:, 0]))
        return np.column_stack((home, 1-home))


class LiveAttributionTests(unittest.TestCase):
    def test_sample_order_matches_clean_swap_masked_swap(self):
        inputs = dict(period=np.ones(10), timestamps=np.arange(10)/4, fps=4.)
        labels = np.array(list('HHADXXAAHD'))
        rows, y, w = la.training_metadata(inputs, labels)
        np.testing.assert_array_equal(rows, [0, 1, 2, 3, 6, 7, 8, 9])
        direct = np.array([0, 0, 1, 2, 1, 1, 0, 2])
        np.testing.assert_array_equal(y, np.concatenate((direct, np.where(direct < 2, 1-direct, direct))*2))
        for block in np.split(w, 4):
            np.testing.assert_array_equal(block, w[:len(rows)])

    def test_training_weights_reset_at_unknown_half_and_time_gap(self):
        labels = np.array(list('HHXHHHHHH'))
        inputs = dict(period=np.array([1]*5+[2]*4),
                      timestamps=np.array([0., 1., 2., 3., 4., 5., 6., 10., 11.]), fps=1.)
        _, _, weights = la.training_metadata(inputs, labels)
        # All live runs are two seconds after separating X, half, and gap.
        np.testing.assert_allclose(weights, np.sqrt(2.5))

    def test_each_provider_has_equal_mass_and_no_dead_training_target(self):
        cache = {'one': dict(x=np.zeros((2, 2)), y=np.array([0, 1]),
                             w=np.array([1., 4.]), names=['H_d', 'A_d']),
                 'two': dict(x=np.zeros((3, 2)), y=np.array([0, 1, 1]),
                             w=np.array([2., 1., 3.]), names=['H_d', 'A_d'])}
        for weighted in (False, True):
            _, y, w, _ = la.pooled_arrays(cache, weighted)
            self.assertEqual(set(y), {0, 1})
            self.assertAlmostEqual(w[:2].sum(), w[2:].sum())
            self.assertAlmostEqual(w.mean(), 1.)

    def test_binary_head_keeps_team_swap_symmetry_and_D_marginal(self):
        names = ['H_d', 'A_d']
        model = dict(estimator=TeamEstimator(), feature_names=names, classes=[0, 1])
        x = np.array([[1., 3.], [4., 2.], [0., 0.]])
        team = refine_features.probabilities_from_features(x, names, model)
        reverse = refine_features.probabilities_from_features(x[:, ::-1], names, model)
        np.testing.assert_allclose(team[:, :2], reverse[:, [1, 0]])
        np.testing.assert_allclose(team.sum(axis=1), 1)
        np.testing.assert_array_equal(team[:, 2], 0)
        dead = np.array([.1, .8, 1.])
        combined = refine_domain.combine(team, dead, 0.)
        np.testing.assert_array_equal(combined[:, 2], dead)
        np.testing.assert_allclose(combined.sum(axis=1), 1)

    def test_partial_or_duplicate_scorecard_cannot_be_ranked(self):
        for rows in ([], [dict(condition='clean', model='baseline', source='sportec', match='J03WOY')]*2):
            with self.assertRaises(ValueError):
                la.make_summary(rows)


if __name__ == '__main__':
    unittest.main()
