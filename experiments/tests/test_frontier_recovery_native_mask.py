"""Native-period mask must preserve first-block supervision."""
import unittest

import numpy as np

from experiments.frontier_recovery_v1 import _graded_native_mask


class NativeGradedMaskTests(unittest.TestCase):
    def test_first_valid_native_period_rows_are_kept_even_at_model_period_zero(self):
        inputs = {
            "period": np.array([1, 1, 2, 0, 0]),
            # The first valid block is encoded as zero in model-period IDs.
            "model_period": np.array([0, 0, 1, -1, -1]),
        }
        np.testing.assert_array_equal(
            _graded_native_mask(inputs), np.array([True, True, True, False, False]))

    def test_only_native_nonpositive_padding_is_excluded(self):
        inputs = {
            "period": np.array([0, -1, 1]),
            "model_period": np.array([0, 0, -1]),
        }
        np.testing.assert_array_equal(
            _graded_native_mask(inputs), np.array([False, False, True]))


if __name__ == "__main__":
    unittest.main()
