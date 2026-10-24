"""Protocol checks that guard against inflated cross-match evaluation scores."""

import unittest
import numpy as np

from experiments.metrics import evaluate_tracking


class EvaluationProtocolTests(unittest.TestCase):
    def test_identical_team_across_halves_is_two_possessions(self):
        result = evaluate_tracking(list("HHHH"), list("HHHH"), [1, 1, 2, 2], fps=10)
        self.assertEqual(result["n_gt_spells"], 2)
        self.assertEqual(result["n_pred_spells"], 2)
        self.assertEqual(result["spell_f1_mean"], 1)
        self.assertAlmostEqual(result["gt_mean_spell_duration_s"], 0.2)

    def test_missing_dead_targets_are_not_a_three_class_success(self):
        result = evaluate_tracking(list("HHXAA"), list("HHDAA"))
        self.assertEqual(result["n_graded"], 4)
        self.assertAlmostEqual(result["macro_f1"], 2 / 3)
        self.assertEqual(result["macro_f1_present_classes"], 1)
        self.assertFalse(result["three_class_benchmark"])

    def test_abstention_on_known_frame_is_an_accuracy_error(self):
        result = evaluate_tracking(list("HAD"), list("HXD"))
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertEqual(result["n_abstain_on_graded"], 1)

    def test_time_gap_separates_spells(self):
        result = evaluate_tracking(list("HHHH"), list("HHHH"), fps=10,
                                   timestamps=[0, 0.1, 5, 5.1])
        self.assertEqual(result["n_gt_spells"], 2)

    def test_three_class_perfect_prediction(self):
        result = evaluate_tracking(list("HHDDDAAA"), list("HHDDDAAA"), fps=25)
        self.assertTrue(result["three_class_benchmark"])
        self.assertEqual(result["macro_f1"], 1)
        self.assertEqual(result["spell_f1_mean"], 1)

    def test_invalid_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_tracking(np.array(["H", "A"]), np.array(["H"]))


if __name__ == "__main__":
    unittest.main()
