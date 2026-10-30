import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import soccernet_followup_replay as replay


class FollowupWeightReplayTests(unittest.TestCase):
    def test_ensemble_uses_native_seed_mean(self):
        first = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
        second = np.asarray([[0.4, 0.2, 0.4]], dtype=np.float64)
        result = replay._ensemble_probability(first, second)
        expected = (first + second) * 0.5
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, first.dtype)

    def test_saved_probability_and_decoding_must_match_exactly(self):
        probability = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]], dtype=np.float32)
        prediction = np.asarray([0, 1], dtype=np.int8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.npz"
            np.savez(path, probability=probability, prediction=prediction)
            replay._verify_array(path, probability, prediction)
            with self.assertRaisesRegex(ValueError, "probability differs"):
                replay._verify_array(path, probability + np.float32(1e-7), prediction)
            with self.assertRaisesRegex(ValueError, "prediction differs"):
                replay._verify_array(path, probability, prediction[::-1])


if __name__ == "__main__":
    unittest.main()
