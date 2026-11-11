from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments import frontier_postprocess_round2 as study


class FrontierPostprocessRound2Contracts(unittest.TestCase):
    def test_fixed_composition_passes_gated_labels_and_equal_seed_probability(self):
        gated = np.array(["H", "H", "A", "A"])
        probability = np.array([[.8, .2], [.7, .3], [.2, .8], [.1, .9]])
        period = np.array([1, 1, 1, 1])
        timestamps = np.arange(4, dtype=float) / 25
        expected = np.column_stack((probability, np.zeros(4)))
        moved = np.array(["H", "A", "A", "A"])
        with patch.object(study.boundary, "relocate", return_value=moved) as relocate:
            result = study.compose(gated, probability, period, timestamps, 25.)
        args = relocate.call_args.args
        np.testing.assert_array_equal(args[0], expected)
        np.testing.assert_array_equal(args[1], gated)
        np.testing.assert_array_equal(args[2], period)
        np.testing.assert_array_equal(args[3], timestamps)
        self.assertEqual(args[4:], (25., "shift_0.8"))
        np.testing.assert_array_equal(result, moved)

    def test_composition_rejects_invalid_conditional_probabilities(self):
        with self.assertRaisesRegex(ValueError, "Invalid conditional H/A"):
            study.compose(np.array(["H", "A"]), np.array([[.7, .7], [.2, .8]]),
                np.array([1, 1]), np.array([0., .04]), 25.)

    def test_real_composition_keeps_dx_and_gated_run_order_across_timebreaks(self):
        labels = np.array(["H", "H", "H", "A", "A", "A", "D", "H", "H", "X"])
        # Favor an earlier H/A boundary in the first block; the timestamp gap starts a new block.
        probability = np.array([[.9, .1], [.9, .1], [.8, .2], [.1, .9], [.1, .9], [.1, .9],
                                [.5, .5], [.1, .9], [.1, .9], [.5, .5]])
        period = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2])
        timestamps = np.array([0, .04, .08, .12, .16, .20, .24, 1., 1.04, 1.08])
        moved = study.compose(labels, probability, period, timestamps, 25.)
        np.testing.assert_array_equal(moved[np.isin(labels, ("D", "X"))],
                                      labels[np.isin(labels, ("D", "X"))])
        self.assertEqual(study._run_order_and_count(moved), study._run_order_and_count(labels))
        self.assertEqual(moved[7], labels[7])  # the independent second segment stays intact

    def test_failed_last_component_replay_blocks_combined_scoring(self):
        combined_score = Mock()
        seen = []

        def replay(key):
            seen.append(key)
            if key == "cell-c":
                raise ValueError("component prediction mismatch")

        with self.assertRaisesRegex(ValueError, "component prediction mismatch"):
            verified = study._verify_all_controls(("cell-a", "cell-b", "cell-c"), replay)
            for key in verified:
                combined_score(key)
        self.assertEqual(seen, ["cell-a", "cell-b", "cell-c"])
        combined_score.assert_not_called()

    def test_study_contract_names_fixed_arm_and_order(self):
        self.assertEqual(study.STUDY, "frontier_postprocess_round2")
        self.assertEqual(study.COMBINED_ARM, "gate_lambda_1_then_boundary_shift_0.8")
        self.assertEqual(study.GATE_ARM, "disagreement_lambda_1.0")
        self.assertEqual(study.BOUNDARY_ARM, "boundary_shift_0.8")


if __name__ == "__main__":
    unittest.main()
