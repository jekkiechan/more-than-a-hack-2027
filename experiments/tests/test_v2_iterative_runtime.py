import unittest
from unittest.mock import patch

import numpy as np

from experiments import v2_iterative_runtime as runtime


class ExactNanquantileTests(unittest.TestCase):
    def assert_exact(self, values):
        for q in (0.5, 0.9):
            with self.subTest(q=q):
                expected = runtime._ORIGINAL_NANQUANTILE(values, q, axis=1)
                actual = runtime.nanquantile(values, q, axis=1)
                np.testing.assert_array_equal(actual, expected)

    def test_random_float64_rows_with_many_nans_are_bit_exact(self):
        rng = np.random.default_rng(20261106)
        values = rng.normal(size=(257, 43)).astype(np.float64)
        values[rng.random(values.shape) < 0.72] = np.nan
        values[0] = np.nan
        values[1] = 3.25
        values[2, ::2] = np.nan
        self.assert_exact(values)

    def test_singletons_all_nan_and_extreme_finite_values(self):
        largest = np.finfo(np.float64).max
        values = np.array([
            [np.nan, np.nan, np.nan, np.nan],
            [4.5, np.nan, np.nan, np.nan],
            [-largest, largest, np.nan, np.nan],
            [-largest, -largest, largest, largest],
            [np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0), np.nan, 0.0],
        ], dtype=np.float64)
        self.assert_exact(values)

    def test_unsupported_calls_delegate_to_original(self):
        calls = []
        original = runtime._ORIGINAL_NANQUANTILE

        def sentinel(*args, **kwargs):
            calls.append((args, kwargs))
            return "delegated"

        data = np.ones((3, 4), dtype=np.float64)
        with patch.object(runtime, "_ORIGINAL_NANQUANTILE", sentinel):
            self.assertEqual(runtime.nanquantile(data, 0.1, axis=1), "delegated")
            self.assertEqual(runtime.nanquantile(data.astype(np.float32), 0.5, axis=1), "delegated")
            self.assertEqual(runtime.nanquantile(data, 0.5, axis=0), "delegated")
            self.assertEqual(runtime.nanquantile(data, 0.5, axis=1, method="nearest"), "delegated")
            self.assertEqual(runtime.nanquantile(data, 0.5, axis=1, keepdims=True), "delegated")
            self.assertEqual(runtime.nanquantile(data, 0.5, axis=1, weights=np.ones_like(data)), "delegated")
        self.assertEqual(len(calls), 6)
        self.assertIs(np.nanquantile, original)

    def test_infinities_fall_back(self):
        values = np.array([[0.0, np.inf, 1.0]], dtype=np.float64)
        with patch.object(runtime, "_ORIGINAL_NANQUANTILE", return_value="original") as fallback:
            self.assertEqual(runtime.nanquantile(values, 0.5, axis=1), "original")
            fallback.assert_called_once()

    def test_installed_context_restores_numpy_function(self):
        original = np.nanquantile
        data = np.array([[0.0, np.nan, 2.0]], dtype=np.float64)
        with runtime.installed():
            self.assertIsNot(np.nanquantile, original)
            np.testing.assert_array_equal(np.nanquantile(data, 0.9, axis=1),
                                          original(data, 0.9, axis=1))
        self.assertIs(np.nanquantile, original)

    def test_refine_feature_matrix_is_bitwise_unchanged(self):
        from experiments import refine_features
        from experiments.tests.test_contracts import tracking

        raw = tracking(n=800)
        expected, expected_names = refine_features.features(raw)
        with runtime.installed():
            actual, actual_names = refine_features.features(raw)
        self.assertEqual(actual_names, expected_names)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
