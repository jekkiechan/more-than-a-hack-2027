from __future__ import annotations

import unittest

import numpy as np

from experiments import tracking_corruption
from experiments import v2_spell_target_runtime as runtime
from experiments.v2_spell_target import phase_aggregate


class PhaseAggregateTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = np.full((5, 4, 3), 1 / 3, dtype=float)
        self.predictions = np.asarray([
            ["H", "A", "A", "D"],
            ["H", "A", "A", "D"],
            ["A", "H", "A", "D"],
            ["A", "H", "D", "D"],
            ["D", "D", "D", "H"],
        ])
        self.fallback = np.asarray(["D", "H", "D", "A"])

    def test_mean_decode_is_exact_fallback_copy(self):
        result = phase_aggregate(self.probabilities, self.predictions, self.fallback, "mean_decode")
        np.testing.assert_array_equal(result, self.fallback)
        self.assertIsNot(result, self.fallback)

    def test_plurality_uses_fallback_only_on_ties(self):
        result = phase_aggregate(self.probabilities, self.predictions, self.fallback, "phase_vote")
        np.testing.assert_array_equal(result, ["D", "H", "A", "D"])

    def test_confidence_vote_is_per_frame_and_tie_falls_back(self):
        p = self.probabilities.copy()
        # Two confident H votes beat two low-confidence A votes at frame 0.
        p[0, 0] = [0.99, 0.005, 0.005]
        p[1, 0] = [0.34, 0.33, 0.33]
        p[2, 0] = [0.34, 0.33, 0.33]
        p[2, 0] = [0.34, 0.33, 0.33]
        p[3, 0] = [0.34, 0.33, 0.33]
        result = phase_aggregate(p, self.predictions, self.fallback, "confidence_vote")
        self.assertEqual(result[0], "H")
        self.assertEqual(result[1], "H")  # tied vote weights fall back to f062

    def test_five_phase_consensus_requires_four_votes(self):
        predictions = self.predictions.copy()
        predictions[:, 0] = ["H", "H", "H", "H", "A"]
        predictions[:, 1] = ["A", "A", "A", "H", "D"]
        result = phase_aggregate(self.probabilities, predictions, self.fallback, "consensus_gate")
        np.testing.assert_array_equal(result[:2], ["H", "H"])

    def test_two_phase_consensus_requires_both_phases(self):
        probabilities = self.probabilities[:2]
        predictions = self.predictions[:2].copy()
        predictions[:, 0] = ["A", "A"]
        predictions[:, 1] = ["H", "A"]
        result = phase_aggregate(probabilities, predictions, self.fallback, "consensus_gate")
        np.testing.assert_array_equal(result[:2], ["A", "H"])

    def test_rejects_misaligned_or_invalid_phase_inputs(self):
        with self.assertRaises(ValueError):
            phase_aggregate(self.probabilities[:4], self.predictions[:4], self.fallback, "phase_vote")
        bad = self.probabilities.copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            phase_aggregate(bad, self.predictions, self.fallback, "phase_vote")

    def test_runtime_correction_binds_new_id_and_restores_inference_alias(self):
        self.assertIs(runtime.frozen.tc, tracking_corruption)
        self.assertEqual(runtime.frozen.STUDY, "v2_spell_target_runtime")
        self.assertNotEqual(runtime.frozen.DECLARATION.name, "v2_spell_target_declaration.json")
        self.assertIn("experiments/v2_spell_target_runtime.py", runtime.frozen._source_hashes())


if __name__ == "__main__":
    unittest.main()
