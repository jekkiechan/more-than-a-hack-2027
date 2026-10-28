import unittest

import numpy as np

from experiments import objective_ensemble as study


def synthetic_score_rows(feed_spell_gain=0.01, skill_clean_d_delta=-0.005):
    rows = []
    base_spell = {"sportec": 0.50, "metrica": 0.60, "skillcorner": 0.40}
    candidate_clean_gain = {"sportec": 0.01, "metrica": 0.02, "skillcorner": 0.04}
    for cell in study._cells():
        provider = cell["source"]
        for recipe in (study.RECIPE_CONTROL, study.RECIPE_CANDIDATE):
            spell = base_spell[provider] + (0.10 if cell["view"] == "feed_like" else 0)
            macro = 0.80 + (0.01 if provider == "metrica" else 0)
            d_f1 = 0.70
            d_precision, d_recall = 0.72, 0.68
            short = 0.40
            if recipe == study.RECIPE_CANDIDATE:
                spell += (feed_spell_gain if cell["view"] == "feed_like"
                          else candidate_clean_gain[provider])
                macro += 0.005
                d_f1 += (skill_clean_d_delta if provider == "skillcorner"
                         and cell["view"] == "clean" else 0.004)
                d_precision += 0.003
                d_recall += 0.004
                short += 0.02
            rows.append({
                **cell,
                "recipe": recipe,
                "spell_f1": spell,
                "macro_f1": macro,
                "weighted_70_30": 0.70 * spell + 0.30 * macro,
                "d_f1": d_f1,
                "d_precision": d_precision,
                "d_recall": d_recall,
                "short_spell_recall": short,
                "spell_by_iou": {
                    f"{iou:.1f}": spell - (iou - 0.3) / 100
                    for iou in study.IOU_THRESHOLDS
                },
            })
    # Each cell has one row per recipe. The number of feed-like rows per match is
    # already two because _cells() expands the two fixed corruptions.
    return rows


class ObjectiveEnsembleTests(unittest.TestCase):
    def test_probability_blend_is_fixed_equal_mean_and_does_not_mutate_inputs(self):
        p2026 = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], dtype=np.float64)
        p2027 = np.array([[0.2, 0.6, 0.2], [0.4, 0.1, 0.5]], dtype=np.float64)
        before0, before1 = p2026.copy(), p2027.copy()
        actual = study.blend_probabilities(p2026, p2027)
        np.testing.assert_array_equal(actual, (before0 + before1) / 2)
        np.testing.assert_array_equal(p2026, before0)
        np.testing.assert_array_equal(p2027, before1)

    def test_probability_blend_accepts_float32_upsampling_roundoff_without_normalizing(self):
        p0 = np.array([[0.4, 0.3, 0.3000001257285476]], dtype=np.float64)
        self.assertGreater(abs(float(p0.sum(axis=1)[0] - 1)), 1e-8)
        p1 = np.array([[0.2, 0.3, 0.5]], dtype=np.float64)
        actual = study.blend_probabilities(p0, p1)
        np.testing.assert_array_equal(actual, (p0 + p1) / 2)

    def test_probability_blend_rejects_malformed_or_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            study.blend_probabilities(np.ones((3, 2)) / 2, np.ones((3, 2)) / 2)
        with self.assertRaises(ValueError):
            study.blend_probabilities(np.array([[0.7, 0.2, 0.2]]),
                                      np.array([[0.4, 0.3, 0.3]]))
        with self.assertRaises(ValueError):
            study.blend_probabilities(np.array([[np.nan, 0.5, 0.5]]),
                                      np.array([[0.4, 0.3, 0.3]]))

    def test_shared_metric_scoring_reports_iou_short_fragment_censor_and_timing(self):
        labels = np.array(list("H" * 10 + "D" * 4 + "A" * 20 + "D" * 3 +
                               "H" * 40 + "X" * 2 + "A" * 10 + "D" * 3 + "H" * 15))
        prediction = labels.copy()
        prediction[4:6] = "D"  # Split the first short H spell into two predictions.
        period = np.zeros(len(labels), dtype=np.int8)
        period[49:] = 1
        timestamps = np.arange(len(labels), dtype=np.float64) / 5
        inputs = {"period": period, "fps": 5.0, "timestamps": timestamps}
        cell = {"source": "sportec", "match": "toy", "view": "clean", "replicate": "clean"}
        row = study.score_prediction(
            labels, prediction, inputs, study.RECIPE_CANDIDATE, cell,
            study.digest_array(np.zeros((len(labels), 3), dtype=np.float64)))
        self.assertEqual(set(row["spell_by_iou"]), {"0.3", "0.4", "0.5", "0.6", "0.7"})
        self.assertEqual(row["n_pred_spells"], row["n_gt_spells"] + 1)
        self.assertGreaterEqual(row["short_reference_spells"], 1)
        self.assertGreaterEqual(row["fragmented_reference_spells"], 1)
        self.assertGreaterEqual(row["reference_spells_touching_graded_run_edge"], 1)
        timing = row["complete_spell_boundary_timing_iou_0_5"]
        self.assertIn("start", timing)
        self.assertIn("end", timing)
        self.assertIsInstance(row["weighted_70_30"], float)
        self.assertEqual(row["graded_frames"], int(np.sum(labels != "X")))

    def test_aggregation_uses_equal_provider_means_and_applies_all_gates(self):
        aggregate = study.aggregate_rows(synthetic_score_rows())
        self.assertEqual(len(aggregate["recipes"][study.RECIPE_CONTROL]["clean"]["per_provider"]), 3)
        self.assertAlmostEqual(
            aggregate["candidate_minus_control"]["clean"]["equal_provider_mean"]["spell_f1"],
            (0.01 + 0.02 + 0.04) / 3,
        )
        self.assertGreater(
            aggregate["candidate_minus_control"]["clean"]["equal_provider_mean"]["spell_f1"],
            0.01,
        )
        self.assertTrue(aggregate["selection_gate"]["passes"])
        self.assertTrue(aggregate["selection_gate"]["post_hoc"])
        self.assertFalse(aggregate["selection_gate"]["automatic_promotion"])
        self.assertEqual(
            set(aggregate["candidate_minus_control"]["clean"]["equal_provider_mean"]),
            set(study.METRIC_FIELDS) | {f"iou_{iou:.1f}" for iou in study.IOU_THRESHOLDS},
        )

    def test_aggregation_rejects_zero_stress_gain_and_provider_guardrail_failure(self):
        no_feed_gain = study.aggregate_rows(synthetic_score_rows(feed_spell_gain=0))
        self.assertFalse(no_feed_gain["selection_gate"]["passes"])
        self.assertFalse(no_feed_gain["selection_gate"]["checks"]["strict_positive_feed_like_spell_gain"])

        d_failure = study.aggregate_rows(synthetic_score_rows(skill_clean_d_delta=-0.07))
        self.assertFalse(d_failure["selection_gate"]["passes"])
        self.assertFalse(
            d_failure["selection_gate"]["checks"][
                "each_provider_clean_d_f1_delta_at_least_minus_0_02"])

    def test_aggregation_rejects_missing_or_duplicate_cells(self):
        rows = synthetic_score_rows()
        with self.assertRaises(study.StudyError):
            study.aggregate_rows(rows[:-1])
        with self.assertRaises(study.StudyError):
            study.aggregate_rows(rows + [rows[0]])

    def test_current_baseline_parity_checks_probability_prediction_and_metrics(self):
        saved = {
            "spell_f1": 0.7,
            "macro_f1": 0.8,
            "weighted_70_30": 0.73,
            "d_f1": 0.6,
            "d_precision": 0.5,
            "d_recall": 0.75,
            "n_gt_spells": 4,
            "n_pred_spells": 5,
            "probability_sha256": "p",
            "prediction_sha256": "y",
            "spell_thresholds": {"0.3": 0.8, "0.4": 0.75, "0.5": 0.7,
                                 "0.6": 0.65, "0.7": 0.6},
        }
        actual = {
            "spell_f1": 0.7,
            "macro_f1": 0.8,
            "weighted_70_30": 0.73,
            "d_f1": 0.6,
            "d_precision": 0.5,
            "d_recall": 0.75,
            "n_gt_spells": 4,
            "n_pred_spells": 5,
            "probability_sha256": "p",
            "prediction_sha256": "y",
            "spell_by_iou": {k: v for k, v in saved["spell_thresholds"].items()},
        }
        study._verify_row_parity(actual, saved, ("sportec", "toy", "clean", "clean"))
        actual["prediction_sha256"] = "different"
        with self.assertRaises(study.StudyError):
            study._verify_row_parity(actual, saved, ("sportec", "toy", "clean", "clean"))


if __name__ == "__main__":
    unittest.main()
