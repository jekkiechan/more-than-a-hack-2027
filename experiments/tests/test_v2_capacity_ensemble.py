import unittest

import numpy as np

from experiments.v2_capacity_ensemble import uniform_probability
from experiments.v2_data_capacity_eval import hard_gate
from scripts.verify_v2_capacity_ensemble import (
    evaluate_gates as independently_evaluate_gates,
    uniform_probability as independently_average,
    _adopted_baseline,
    _validate_saved_counts,
    verify_dead_mask,
)


class UniformCapacityEnsembleTests(unittest.TestCase):
    def test_equal_weight_mean_is_normalized_and_matches_expected(self):
        members = [np.asarray([[.8, .1, .1]]), np.asarray([[.2, .7, .1]]),
                   np.asarray([[.5, .2, .3]])]
        result = uniform_probability(members)
        np.testing.assert_allclose(result, [[.5, 1/3, 1/6]], atol=1e-15)
        np.testing.assert_allclose(result.sum(axis=1), 1.)

    def test_rejects_misaligned_members(self):
        with self.assertRaises(ValueError):
            uniform_probability([np.ones((2, 3))/3, np.ones((1, 3))/3,
                                 np.ones((2, 3))/3])

    def test_independent_reducer_rejects_invalid_probabilities(self):
        invalid = np.asarray([[.4, .4, .4]])
        with self.assertRaises(ValueError):
            independently_average([invalid, invalid, invalid])
        with self.assertRaises(ValueError):
            independently_average([np.asarray([[np.nan, 0., 1.]]),
                                   np.asarray([[0., 0., 1.]]),
                                   np.asarray([[0., 0., 1.]])])

    def test_protected_dead_mask_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            verify_dead_mask(np.asarray([False, True, False]),
                             np.asarray([False, False, False]))

    def test_saved_counts_follow_runner_cell_schema(self):
        # Runner stores frame/reference counts at cell-arm level, while predicted
        # spell count is nested with the complete scalar metric mapping.
        cell_arm = {"frame_count": 25, "reference_spell_count": 3,
                    "metrics": {"predicted_spell_count": 4,
                                "reference_spell_count": 3}}
        replay = {"predicted_spell_count": 4, "reference_spell_count": 3}
        _validate_saved_counts(cell_arm, replay, 25, 3)
        cell_arm["metrics"]["predicted_spell_count"] = 5
        with self.assertRaises(ValueError):
            _validate_saved_counts(cell_arm, replay, 25, 3)

    def test_adopted_baseline_comes_from_control_replay_schema(self):
        metrics = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                   "predicted_spell_count", "reference_spell_count",
                   "short_spell_recall_iou_0_5_lt3s")
        views = ("clean", "feed_like_v1", "feed_like_v2")
        # The base scalar result aggregate is recipe-keyed; only its matched
        # control replay exposes the adopted candidate aggregate as `cand`.
        base_result_aggregate = {view: {"width96_control/standalone_ha3": {}}
                                 for view in views}
        control_replay = {"aggregate": {
            view: {"cand": {metric: {"equal_provider": .25}
                            for metric in metrics}}
            for view in views}}
        self.assertNotIn("cand", base_result_aggregate["clean"])
        baseline = _adopted_baseline(control_replay)
        self.assertEqual(set(baseline), set(views))
        self.assertTrue(all(set(baseline[view]) == set(metrics) and
                            baseline[view]["spell_f1"] == .25 for view in views))

    def test_independent_strict_gate_has_exact_19_checks(self):
        metrics = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                   "short_spell_recall_iou_0_5_lt3s")
        adopted = {view: {metric: .5 for metric in metrics}
                   for view in ("clean", "feed_like_v1", "feed_like_v2")}
        candidate = {view: dict(values) for view, values in adopted.items()}
        candidate["feed_like_v2"]["spell_f1"] = .5927491913246999
        gate = independently_evaluate_gates(candidate, adopted)
        self.assertEqual(len(gate["checks"]), 19)
        self.assertTrue(gate["all_passed"])
        candidate["clean"]["d_precision"] -= 1e-15
        self.assertFalse(independently_evaluate_gates(candidate, adopted)["all_passed"])

    def test_strict_gate_checks_19_no_regression_floors(self):
        metrics = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
                   "short_spell_recall_iou_0_5_lt3s")
        adopted = {view: {metric: .5 for metric in metrics}
                   for view in ("clean", "feed_like_v1", "feed_like_v2")}
        candidate = {view: dict(values) for view, values in adopted.items()}
        candidate["feed_like_v2"]["spell_f1"] = .5927491913246999
        gate = hard_gate(candidate, adopted)
        self.assertEqual(len(gate["checks"]), 19)
        self.assertTrue(gate["all_passed"])
        candidate["feed_like_v1"]["d_recall"] -= 1e-15
        self.assertFalse(hard_gate(candidate, adopted)["all_passed"])


if __name__ == "__main__":
    unittest.main()
