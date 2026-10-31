import unittest

import numpy as np

from experiments import dense_boundary_v1 as boundary
from experiments import supervised


class DenseBoundaryTests(unittest.TestCase):
    def test_cost_three_agrees_when_switch_evidence_is_weak(self):
        p = np.array([[.51, .49, 0], [.49, .51, 0], [.51, .49, 0], [.49, .51, 0]])
        period = np.ones(len(p), dtype=int)
        np.testing.assert_array_equal(boundary.decode(p, period, 10), boundary.baseline_decode(p, period, 10))

    def test_confident_short_possession_survives(self):
        p = np.array([[.99, .01, 0], [.99, .01, 0], [.01, .99, 0], [.99, .01, 0], [.99, .01, 0]])
        period = np.ones(len(p), dtype=int)
        labels = boundary.decode(p, period, 10)
        self.assertEqual(labels.tolist(), ["H", "H", "A", "H", "H"])

    def test_period_breaks_are_independent(self):
        p = np.array([[.01, .99, 0], [.99, .01, 0]])
        labels = boundary.decode(p, np.array([1, 2]), 10)
        self.assertEqual(labels.tolist(), ["A", "H"])

    def test_invalid_probability_contract(self):
        with self.assertRaises(ValueError):
            boundary.decode(np.array([[np.nan, 0, 0]]), np.array([1]), 10)

    def test_paired_scoring_preserves_unknown_targets_and_reports_d_guards(self):
        p = np.array([[.99, .005, .005], [.99, .005, .005],
                      [.005, .99, .005], [.005, .005, .99],
                      [.005, .005, .99], [.005, .005, .99]])
        record = {"probabilities": p, "labels": np.array(["H", "H", "A", "X", "D", "D"]),
                  "period": np.ones(6, dtype=int), "model_period": np.zeros(6, dtype=int),
                  "fps": np.array(10.)}
        row = boundary.score_record(record)
        for arm in ("cost3", "confidence_boundary"):
            self.assertIn("d_precision", row["scores"][arm])
            self.assertIn("d_recall", row["scores"][arm])
            self.assertIn("spell_f1", row["scores"][arm])
            self.assertEqual(row["scores"][arm]["label_coverage"], 5 / 6)

    def test_saved_decoder_period_must_match_raw_period_and_time(self):
        p = np.tile([.7, .2, .1], (3, 1))
        record = {"probabilities": p, "labels": np.array(["H", "H", "H"]),
                  "period": np.ones(3, dtype=int), "model_period": np.ones(3, dtype=int),
                  "timestamps": np.arange(3, dtype=float) / 10, "fps": np.array(10.)}
        with self.assertRaisesRegex(ValueError, "model_period"):
            boundary.score_record(record)


if __name__ == "__main__":
    unittest.main()
