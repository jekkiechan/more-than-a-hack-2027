"""Probability ensembling contracts, without data or optional neural packages."""
import unittest
import numpy as np
from experiments.temporal_ensemble import blend


class EnsembleTests(unittest.TestCase):
    def test_probabilities_preserve_confidence(self):
        # Both frames have a split hard vote; their confidence is different.
        p = {'gru': np.array([.99, .51]), 'tree': np.array([.49, .01])}
        result = blend(p, {'gru': .5, 'tree': .5})
        np.testing.assert_allclose(result, [.74, .26])
        np.testing.assert_array_equal(p['gru'], [.99, .51])

    def test_invalid_or_misaligned_members_fail_closed(self):
        for second in (np.array([.5]), np.array([.5, np.nan]), np.array([.5, 1.1])):
            with self.assertRaises(ValueError):
                blend({'a': np.array([.2, .7]), 'b': second}, {'a': .5, 'b': .5})
        for weights in ({'a': .7, 'b': .7}, {'a': 1.1, 'b': -.1}, {'a': np.nan, 'b': .5}):
            with self.assertRaises(ValueError):
                blend({'a': np.array([.2]), 'b': np.array([.7])}, weights)
        with self.assertRaises(KeyError):
            blend({'a': np.array([.2])}, {'a': .5, 'missing': .5})


if __name__ == '__main__':
    unittest.main()
