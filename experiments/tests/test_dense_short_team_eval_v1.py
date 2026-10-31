"""Contracts for the frozen short-team evaluator and its aggregate diagnostics."""
import unittest

import numpy as np
import pandas as pd

from experiments import dense_short_team_eval_v1 as study
from experiments.dense_error_audit_v1 import frame_table


class DenseShortTeamEvaluation(unittest.TestCase):
    def test_paired_native_arrays_must_match_including_masks_and_time(self):
        base = {"labels": np.array(list("HAD")), "period": np.array([1, 1, 1]),
                "timestamps": np.array([0., .04, .08]),
                "ball_observed": np.array([True, False, True]), "fps": np.array(25.)}
        study.assert_native_arrays_equal(base, {k: v.copy() for k, v in base.items()})
        changed = {k: v.copy() for k, v in base.items()}
        changed["timestamps"][1] += .001
        with self.assertRaisesRegex(ValueError, "timestamps"):
            study.assert_native_arrays_equal(base, changed)

    def test_cohort_requires_exact_15_cells_for_each_member_and_arm(self):
        expected = {(f"p{i % 3}", f"m{i}", f"v{i % 3}", f"r{i % 3}") for i in range(15)}
        rows = []
        for arm in study.ARMS:
            for member in ("2026", "2027", "ensemble"):
                for source, match, view, replicate in expected:
                    rows.append(dict(arm=arm, member=member, source=source, match=match,
                                     view=view, replicate=replicate))
        study.validate_complete_cell_cohort(rows, expected)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            study.validate_complete_cell_cohort(rows[:-1], expected)

    def test_complete_le_one_second_recall_uses_official_global_assignment(self):
        labels = np.array(list("DHH DHH D".replace(" ", "")))
        inputs = {"period": np.zeros(7, dtype=np.int8), "fps": 2.,
                  "timestamps": np.arange(7) / 2}
        exact = study.short_spell_metrics(labels, labels.copy(), inputs)
        self.assertEqual(exact["complete_le_1s_reference_spells"], 2)
        self.assertEqual(exact["complete_le_1s_matched_spells"], 2)
        self.assertEqual(exact["complete_le_1s_spell_recall"], 1.)

    def test_public_error_aggregate_drops_match_level_detail(self):
        detail = {"providers": {"p": {"matches": 1, "segments": 1,
            "wrong_team_frames": 1, "false_d_frames": 1, "segment_details": [{
                "scored_frames": 4, "gt_spell_length_bins": {
                    name: {"gt_spells": 1, "gt_alive_frames": 2, "wrong_team_frames": 1,
                           "false_d_frames": 1, "false_d_spells": 1}
                    for name in ("short_le_1s", "medium_1_to_5s", "long_gt_5s")}}]}},
            "interpretation": "aggregate only"}
        result = study.public_error_aggregate(detail)
        self.assertNotIn("segment_details", result["providers"]["p"])
        self.assertAlmostEqual(result["providers"]["p"]["wrong_team_rate"], .25)
        self.assertEqual(result["providers"]["p"]["short_spell_bins"]["short_le_1s"]["false_d_spells"], 1)

    def test_diagnostics_keep_views_separate_and_weight_replicates_then_matches_equally(self):
        rows = []
        for match, n, wrong in (("short", 10, False), ("long", 100, True)):
            for view, rep_wrong in (("clean", wrong), ("feed_like", False)):
                for replicate in (("clean",) if view == "clean" else ("corrupt_1", "corrupt_2")):
                    pred_wrong = rep_wrong if view == "clean" else rep_wrong
                    gt = np.array(["H"] * n, dtype="U1")
                    pred = np.array(["A" if pred_wrong else "H"] * n, dtype="U1")
                    frame = frame_table("provider_a", match, gt, pred, np.ones(n, bool),
                                        np.zeros(n, int), np.arange(n) / 10, 10.)
                    frame["arm"] = "short_team_2x"
                    frame["view"] = view
                    frame["replicate"] = replicate
                    rows.append(frame)
        result = study.aggregate_error_cells(pd.concat(rows, ignore_index=True))
        clean = result["short_team_2x"]["clean"]["per_provider"]["provider_a"]
        stress = result["short_team_2x"]["feed_like"]["per_provider"]["provider_a"]
        # The short perfect match and long all-wrong match receive equal weight.
        self.assertAlmostEqual(clean["wrong_team_rate"], .5)
        self.assertAlmostEqual(stress["wrong_team_rate"], 0.)
        # Feed-like raw counts include both corruption replicates, clean is separate.
        self.assertEqual(clean["evaluation_cells"], 2)
        self.assertEqual(stress["evaluation_cells"], 4)

    def test_diagnostics_sum_time_segments_inside_each_realization(self):
        gt = np.array(["H"] * 10, dtype="U1")
        pred = np.array(["H"] * 5 + ["A"] * 5, dtype="U1")
        period = np.array([0] * 5 + [1] * 5)
        timestamps = np.r_[np.arange(5) / 10, 1 + np.arange(5) / 10]
        frame = frame_table("provider_a", "match_a", gt, pred, np.ones(10, bool),
                            period, timestamps, 10.)
        frame["arm"], frame["view"], frame["replicate"] = "control_D2", "clean", "clean"
        result = study.aggregate_error_cells(frame)
        cell = result["control_D2"]["clean"]["per_provider"]["provider_a"]
        self.assertEqual(cell["wrong_team_frames"], 5)
        self.assertEqual(cell["scored_frames"], 10)
        self.assertAlmostEqual(cell["wrong_team_rate"], .5)


if __name__ == "__main__":
    unittest.main()
