import unittest
import numpy as np
from experiments.feed_confirmation_runtime_v1 import normalized_inference


class FreshMetadataBoundary(unittest.TestCase):
    def test_array_and_list_names_are_equivalent_without_target_leakage(self):
        r = dict(x=np.array([[1., np.nan], [2., 4.]]), names=np.array(['a', 'b']),
                 period=np.array([1, 2]), indices=np.array([0, 5]), native_n=10, fps=25.,
                 y=np.array([0, 1]), full_y=np.zeros(10), labels=['H', 'A'])
        a = normalized_inference(r)
        b = normalized_inference(dict(r, names=['a', 'b'], y=np.array([2, -1])))
        self.assertEqual(a['names'], ['a', 'b'])
        self.assertIsInstance(a['names'], list)
        self.assertEqual(set(a), {'x','names','period','indices','native_n','fps'})
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])
        self.assertIs(a['x'], r['x'])
        self.assertIsInstance(r['names'], np.ndarray)


if __name__ == '__main__':
    unittest.main()
