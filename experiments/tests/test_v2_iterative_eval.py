import unittest

import numpy as np

from experiments import v2_iterative_eval as evaluation


class IterativeEvaluationProtocolTests(unittest.TestCase):
    def test_cell_cohort_is_35_matched_cells(self):
        jobs = evaluation._views()
        self.assertEqual(len(jobs), 35)
        self.assertEqual(sum(row[2] == "clean" for row in jobs), 5)
        self.assertEqual(sum(row[2] == "feed_like_v1" for row in jobs), 10)
        self.assertEqual(sum(row[2] == "feed_like_v2" for row in jobs), 20)

    def test_probability_blends_are_normalized_and_distinct(self):
        adopted = np.array([[.60, .25, .15], [.20, .60, .20]], dtype=float)
        calibrated = np.array([[.30, .50, .20], [.50, .30, .20]], dtype=float)
        joint = evaluation.categorical_blend(calibrated, adopted)
        conditional = evaluation.conditional_probability_blend(calibrated, adopted)
        logit = evaluation.conditional_logit_blend(calibrated, adopted)
        for result in (joint, conditional, logit):
            np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-12)
            self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(joint[:, 2] > 0))
        np.testing.assert_array_equal(conditional[:, 2], np.zeros(2))
        self.assertFalse(np.allclose(conditional, logit))

    def test_conditional_blends_reject_zero_ha_mass(self):
        bad = np.array([[0.0, 0.0, 1.0]])
        good = np.array([[0.4, 0.4, 0.2]])
        with self.assertRaises(ValueError):
            evaluation.conditional_probability_blend(bad, good)

    def test_hard_gate_requires_strict_v2_floor_and_zero_regression(self):
        baseline = {view: {metric: .5 for metric in evaluation.METRICS}
                    for view, _ in evaluation.VIEWS}
        candidate = {view: dict(metrics) for view, metrics in baseline.items()}
        candidate["feed_like_v2"]["spell_f1"] = .60
        self.assertFalse(evaluation.hard_success(candidate, baseline)["all_passed"])
        candidate["feed_like_v2"]["spell_f1"] = .6000001
        self.assertTrue(evaluation.hard_success(candidate, baseline)["all_passed"])
        candidate["feed_like_v1"]["d_recall"] -= 1e-12
        self.assertFalse(evaluation.hard_success(candidate, baseline)["all_passed"])

    def test_reference_count_guard_and_macro_tie_policy_are_declared(self):
        baseline = {view: {metric: .5 for metric in evaluation.METRICS}
                    for view, _ in evaluation.VIEWS}
        candidate = {view: dict(metrics) for view, metrics in baseline.items()}
        candidate["feed_like_v2"]["spell_f1"] = .61
        candidate["clean"]["reference_spell_count"] = 101
        result = evaluation.hard_success(candidate, baseline)
        self.assertFalse(result["checks"]["clean/reference_spell_count_invariant"]["passed"])


if __name__ == "__main__":
    unittest.main()
