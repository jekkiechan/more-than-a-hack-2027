import unittest
from unittest.mock import patch
import json
from pathlib import Path
import tempfile

import numpy as np

from experiments import v2_iterative_followup as followup


class IterativeFollowupTests(unittest.TestCase):
    def test_freeze_rejects_an_incomplete_control_cohort_before_reading_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps({"passed": True, "candidate_scoring": False,
                                       "emission_cache": {}}))
            with patch.object(followup.base, "REPLAY", path), \
                 patch.object(followup.base, "DECLARATION", path), \
                 patch.object(followup.base, "_require_frozen"):
                with self.assertRaisesRegex(RuntimeError, "exact 35-cell"):
                    followup._control_facts()

    def test_round_two_declares_four_fixed_training_pipelines(self):
        self.assertEqual(followup.ARMS, (
            "ce_control_fixed_d_ha3", "boundary_margin_fixed_d_ha3",
            "ce_control_conditional_blend_fixed_d_ha3",
            "boundary_margin_conditional_blend_fixed_d_ha3"))

    def test_equal_provider_result_is_bitwise_independent_of_completion_order(self):
        items = [(('p1', f'm{i}', 0), value) for i, value in enumerate((1e16, 1.0, -1e16))]
        items += [(('p2', 'm3', 0), .25)]
        forward, _ = followup._equal_provider(dict(items))
        reverse, _ = followup._equal_provider(dict(reversed(items)))
        self.assertEqual(np.float64(forward).tobytes(), np.float64(reverse).tobytes())

    def test_seed_aggregate_uses_replicates_then_matches_and_provider_equal(self):
        rows = {
            "p1|m1|feed_like_v2|1|ce_control|2026": {metric: .2 for metric in followup.QUALITY},
            "p1|m1|feed_like_v2|2|ce_control|2026": {metric: .4 for metric in followup.QUALITY},
            "p1|m2|feed_like_v2|1|ce_control|2026": {metric: .8 for metric in followup.QUALITY},
            "p2|m3|feed_like_v2|1|ce_control|2026": {metric: .6 for metric in followup.QUALITY},
            "p1|m1|feed_like_v2|1|boundary_margin|2026": {metric: .9 for metric in followup.QUALITY},
            "p1|m1|feed_like_v2|1|ce_control|2027_blend": {metric: .5 for metric in followup.QUALITY},
        }
        result = followup._aggregate_seed_scores(rows)
        self.assertAlmostEqual(result["ce_control"]["fixed_d"]["2026"]["feed_like_v2"]["spell_f1"][
            "providers"]["p1"], .55)
        self.assertAlmostEqual(result["ce_control"]["fixed_d"]["2026"]["feed_like_v2"]["spell_f1"][
            "equal_provider"], .575)
        self.assertAlmostEqual(result["boundary_margin"]["fixed_d"]["2026"]["feed_like_v2"][
            "spell_f1"]["equal_provider"], .9)
        self.assertIn("2027", result["ce_control"]["conditional_blend"])

    def test_conditional_blend_arm_uses_fixed_adopted_d(self):
        n = 4
        control = {"adopted": np.tile([.6, .2, .2], (n, 1)),
                   "model_period": np.zeros(n), "fps": 25.0,
                   "labels": np.array(["H", "H", "A", "A"]),
                   "period": np.zeros(n), "timestamps": np.arange(n)}
        candidate = np.tile([.2, .6, .2], (n, 1))
        expected = followup.base.conditional_probability_blend(candidate, control["adopted"])
        with patch.object(followup.base, "fixed_d_prediction", return_value=np.array(["H"] * n)) as fixed, \
             patch.object(followup.base.audit, "score", return_value={"spell_f1": 1.0}):
            metrics, _ = followup._score_arm(control, candidate,
                "boundary_margin_conditional_blend_fixed_d_ha3")
        np.testing.assert_allclose(fixed.call_args.args[0], expected)
        np.testing.assert_array_equal(fixed.call_args.args[1], control["adopted"])
        self.assertEqual(fixed.call_args.args[-1], 3.0)
        self.assertEqual(metrics["spell_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
