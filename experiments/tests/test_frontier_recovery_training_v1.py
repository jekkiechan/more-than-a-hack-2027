import unittest

from experiments import frontier_recovery_training_v1 as recovery


class FrontierRecoveryTrainingTests(unittest.TestCase):
    def test_fixed_reconstruction_inventory(self):
        cells = recovery.expected_cells()
        self.assertEqual(len(cells), 35)
        self.assertEqual(len(set(cells)), 35)
        for provider, match, _filename in recovery.AUDIT.MATCHES:
            own = [key for key in cells if key.startswith(f"{provider}|{match}|")]
            self.assertEqual(len(own), 7)
            self.assertEqual(sum("|clean|" in key for key in own), 1)
            self.assertEqual(sum("|feed_like_v1|" in key for key in own), 2)
            self.assertEqual(sum("|feed_like_v2|" in key for key in own), 4)

    def test_recipe_is_the_selected_stage_1024_control(self):
        self.assertEqual(recovery.BASELINE_RECIPE, {
            "recipe_id": "ce0.5_cal0.375_ad0.125_arithmetic_ha3",
            "ce_weight": 0.5, "calibrated_weight": 0.375,
            "adopted_weight": 0.125, "pooling": "arithmetic",
            "ha_cost": 3.0, "training_arm": "base", "training_stage": 1024,
        })
        self.assertEqual(recovery.LR_STAGE1, 3e-4)
        self.assertEqual(recovery.LR_STAGE2, 1e-4)
        self.assertEqual(recovery.SEEDS, (2026, 2027))

    def test_fit_outputs_have_isolated_non_overwriting_namespace(self):
        self.assertEqual(recovery.OUT.name, "frontier_recovery_v1")
        self.assertEqual(recovery.TRAIN_DECL.name, "frontier_recovery_v1_training_declaration.json")
        self.assertIn("training_views", recovery.TRAIN_VIEW_OUT.parts)
        self.assertIn("stage512", recovery._stage_path(512, recovery.FOLDS[0], 2026).parts)


if __name__ == "__main__":
    unittest.main()
