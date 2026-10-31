import unittest
import math

from experiments.provider_touch_calibration import candidate_grid, select_by_provider_mean


class ProviderTouchCalibrationTests(unittest.TestCase):
    def test_grid_is_bounded_and_has_distinct_rules(self):
        grid = candidate_grid()
        self.assertEqual(len(grid), 24)
        keys = {(
            tuple(sorted(row["uncontrolled_end_rules"].items())), row["uncontrolled_opponent"],
            row["censored_run_controlled"], row["out_tail_grace_seconds"],
        ) for row in grid}
        self.assertEqual(len(keys), 24)
        self.assertEqual({row["out_tail_grace_seconds"] for row in grid}, {0.0, 3.0})
        v1 = next(row for row in grid if row["uncontrolled_rule"] == "v1_ch_re_any_cl_single"
                  and row["uncontrolled_opponent"] == "hold"
                  and row["censored_run_controlled"]
                  and row["out_tail_grace_seconds"] == 3.0)
        self.assertEqual(v1["uncontrolled_end_rules"],
                         {"CH": "any", "RE": "any", "CL": "single_frame"})
        self.assertTrue(v1["censored_run_controlled"])
        self.assertEqual(v1["out_tail_grace_seconds"], 3.0)

    def test_selection_weights_providers_equally_and_excludes_holdout(self):
        scores = {
            "metrica": {"a": 0.2, "b": 0.4},
            "sportec": {"a": 0.8, "b": 0.4},
            # A much larger row count must not implicitly weight this group.
        }
        selected = select_by_provider_mean(scores, excluded_provider="skillcorner")
        self.assertEqual(selected["candidate"], "b")
        self.assertEqual(selected["training_providers"], ["metrica", "sportec"])
        self.assertEqual(selected["provider_mean_loss"], 0.4)

    def test_one_eligible_training_provider_is_supported(self):
        selected = select_by_provider_mean({"metrica": {"a": 0.3, "b": 0.2}},
                                           excluded_provider="sportec")
        self.assertEqual(selected["candidate"], "b")
        self.assertEqual(selected["training_providers"], ["metrica"])

    def test_selection_rejects_leakage_and_incomplete_candidate_sets(self):
        with self.assertRaisesRegex(ValueError, "excluded provider"):
            select_by_provider_mean({"sportec": {"a": 1}, "skillcorner": {"a": 2}},
                                    excluded_provider="skillcorner")
        with self.assertRaisesRegex(ValueError, "candidate sets"):
            select_by_provider_mean({"sportec": {"a": 1}, "metrica": {"b": 2}},
                                    excluded_provider="skillcorner")

    def test_selection_rejects_nonfinite_and_boolean_losses(self):
        for invalid in (True, float("nan"), float("inf"), -math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "finite real"):
                select_by_provider_mean({"metrica": {"a": invalid}},
                                        excluded_provider="sportec")


if __name__ == "__main__":
    unittest.main()
