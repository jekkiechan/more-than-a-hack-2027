"""Promotion ordering and separation of PathCRF H/A from the temporal D head."""
import unittest
from unittest.mock import patch
import numpy as np
from experiments.current_baseline import official_better
from experiments.pathcrf_ensemble import active_phases, team_variants
from experiments.refine_domain import combine


class BaselinePathCRFTests(unittest.TestCase):
    def test_empty_periods_are_excluded_without_joining_halves(self):
        data = dict(players=np.zeros((12, 22, 2)), teams=np.repeat([0, 1], 11),
                    period=np.array([0, 0, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2]),
                    timestamps=np.arange(12)/5, fps=5., pitch=np.array([105, 68]))
        def phases(block):
            n = len(block['period'])
            self.assertTrue((block['period'] > 0).all())
            yield 0, n, np.zeros((n, 26, 8)), np.zeros(26)
        with patch('experiments.pathcrf_ensemble.pathcrf_eval.feature_phases', side_effect=phases):
            result = list(active_phases(data))
        self.assertEqual([(a, b) for a, b, _, _ in result], [(2, 6), (8, 12)])

    def test_promotion_uses_spell_first_and_macro_only_for_exact_ties(self):
        baseline = dict(spell_f1=.6, macro_f1=.8)
        self.assertFalse(official_better(dict(spell_f1=.59, macro_f1=.99), baseline))
        self.assertTrue(official_better(dict(spell_f1=.61, macro_f1=.7), baseline))
        self.assertTrue(official_better(dict(spell_f1=.6, macro_f1=.81), baseline))
        self.assertFalse(official_better(baseline, baseline))

    def test_pathcrf_changes_team_only_before_sequence_decoding(self):
        base = np.array([[.6, .2, .2], [.1, .4, .5]])
        path = np.array([[0., 1.], [1., 0.]])
        variants = team_variants(base, path)
        dead = np.array([.3, .7])
        for values in variants.values():
            probabilities = combine(values, dead, 0.)
            np.testing.assert_allclose(probabilities[:, 2], dead)
            np.testing.assert_allclose(probabilities.sum(axis=1), 1.)
        expected_live = np.array([[.375, .625], [.6, .4]])
        blend = variants['gru_hgb_team_blend']
        np.testing.assert_allclose(blend[:, :2], expected_live)


if __name__ == '__main__':
    unittest.main()
