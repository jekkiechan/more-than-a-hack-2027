import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.dense_error_audit_v1 import audit, audit_archive, frame_table, summarize_archives


def rows(gt, pred, missing=None, segment="p1"):
    missing = missing or [False] * len(gt)
    return pd.DataFrame({"provider": ["P"] * len(gt), "match_id": ["M"] * len(gt),
                         "segment_id": [segment] * len(gt), "frame": range(len(gt)),
                         "gt": list(gt), "pred": list(pred), "ball_missing": missing,
                         "fps": [25.0] * len(gt)})


class DenseErrorAuditTests(unittest.TestCase):
    def test_counts_wrong_team_false_d_and_missingness_separately(self):
        result = audit(rows("HHHDDD", "AADHHA", [True, False, False, True, False, False]))
        seg = result["match_segments"][0]
        self.assertEqual(seg["wrong_team_frames"], 2)
        self.assertEqual(seg["false_d_frames"], 1)
        self.assertEqual(seg["missed_dead_frames"], 3)
        self.assertEqual(seg["scored_frames_ball_missing"], 2)
        self.assertEqual(result["providers"]["P"]["matches"], 1)

    def test_breaks_are_independent_and_no_spells_cross_them(self):
        f = rows("HH", "HH")
        f2 = rows("AA", "AA", segment="p2")
        result = audit(pd.concat([f, f2], ignore_index=True))
        self.assertEqual(len(result["match_segments"]), 2)
        self.assertEqual(result["providers"]["P"]["segment_details"][0]["gt_alive_spells"], 1)
        self.assertEqual(result["providers"]["P"]["matches"], 1)
        self.assertEqual(result["providers"]["P"]["segments"], 2)

    def test_string_boolean_missingness_is_parsed_as_false_or_true(self):
        f = rows("HH", "AH", ["false", "true"])
        seg = audit(f)["match_segments"][0]
        self.assertEqual(seg["scored_frames_ball_missing"], 1)
        self.assertEqual(seg["wrong_team_rate_ball_missing"], 0.0)
        self.assertEqual(seg["wrong_team_rate_ball_observed"], 1.0)

    def test_frame_table_splits_period_and_large_timestamp_gaps(self):
        f = frame_table("P", "M", "HHHH", "HHHH", [1, 1, 1, 1],
                        [1, 1, 2, 2], [0.0, 0.04, 0.08, 2.0], 25.0)
        self.assertEqual(f.segment_id.tolist(), [0, 0, 1, 2])

    def test_audits_dense_label_weight_prediction_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.npz"
            np.savez_compressed(path, prediction=np.array(list("HHHD")), labels=np.array(list("HHDD")),
                period=np.array([1, 1, 1, 1]), fps=np.array(25.), timestamps=np.arange(4)/25,
                ball_observed=np.array([1, 0, 0, 1]), source=np.array("P"), match=np.array("M"),
                view=np.array("clean"), replicate=np.array("clean"), seed=np.array("2026"),
                arm=np.array("control_D2"), fold=np.array("leave_out_P"))
            report = audit_archive(path)
        self.assertEqual(report["arm"], "control_D2")
        self.assertEqual(report["match_segments"][0]["missed_dead_frames"], 1)
        self.assertEqual(len(report["archive_sha256"]), 64)

    def test_summary_uses_equal_provider_and_match_rates_separate_from_raw_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            def save(provider, match, seed, gt, pred):
                path = Path(tmp) / f"{provider}_{match}_{seed}.npz"
                n = len(gt)
                np.savez_compressed(path, prediction=np.array(list(pred)), labels=np.array(list(gt)),
                    period=np.ones(n), fps=np.array(25.), timestamps=np.arange(n)/25,
                    ball_observed=np.ones(n, dtype=bool), source=np.array(provider),
                    match=np.array(match), view=np.array("clean"), replicate=np.array("clean"),
                    seed=np.array(str(seed)), arm=np.array("D2"), fold=np.array("fold"))
                paths.append(path)
            save("A", "a1", 2026, "H", "A")
            save("A", "a1", 2027, "H", "H")
            save("B", "b1", 2026, "H" * 100, "A" + "H" * 99)
            save("A", "a1", "ensemble", "H", "A")
            summary = summarize_archives(paths)
            cell = next(c for c in summary["cells"] if c["realization_kind"] == "individual_seed_members")
        self.assertAlmostEqual(cell["provider_balanced_match_rates"]["wrong_team_rate_scored"], .255)
        self.assertEqual(cell["raw_counts_across_archives"]["scored_frames"], 102)
        self.assertEqual(cell["raw_counts_across_archives"]["wrong_team_frames"], 2)
        self.assertEqual(len(summary["cells"]), 2)
        self.assertEqual(next(c for c in summary["cells"] if c["realization_kind"] == "ensemble")
                         ["raw_counts_across_archives"]["wrong_team_frames"], 1)

    def test_rejects_noncontiguous_frames_inside_segment(self):
        f = rows("HHH", "HHH")
        f.loc[2, "frame"] = 4
        with self.assertRaisesRegex(ValueError, "contiguous"):
            audit(f)

    def test_rejects_infinite_or_mixed_segment_fps(self):
        f = rows("HH", "HH")
        f.loc[0, "fps"] = float("inf")
        with self.assertRaisesRegex(ValueError, "fps"):
            audit(f)
        f = rows("HH", "HH")
        f.loc[1, "fps"] = 24.0
        with self.assertRaisesRegex(ValueError, "constant"):
            audit(f)

    def test_rejects_duplicate_frame(self):
        f = rows("HH", "HH")
        f.loc[1, "frame"] = 0
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit(f)

    def test_detects_same_team_split_merge_and_boundary_shift(self):
        split = audit(rows("HHHH", "HHDH"))["match_segments"][0]
        self.assertEqual(split["split_ground_truth_spells"], 1)
        self.assertEqual(split["split_ground_truth_spells"], 1)
        merged = audit(rows("HHDDHH", "HHHHHH"))["match_segments"][0]
        self.assertEqual(merged["merge_predicted_spells"], 1)
        shifted = audit(rows("HHDD", "HHHD"))["match_segments"][0]
        self.assertEqual(shifted["transition_shift_median_frames"], 1)

    def test_spell_bins_use_match_native_fps_and_x_masks_break_predicted_spells(self):
        f = frame_table("P", "M", "HHHHHH", "HHHHHH", [1] * 6,
                        [1] * 6, np.arange(6) / 10, 10.0)
        f.loc[2:3, "gt"] = "X"
        seg = audit(f)["match_segments"][0]
        self.assertEqual(seg["pred_alive_spells"], 2)
        self.assertEqual(seg["gt_spell_length_bins"]["short_le_1s"]["gt_spells"], 2)

    def test_boundary_match_rejects_unrelated_transition_outside_one_second(self):
        f = rows("HHHHDDDD", "DDDDHHHH")
        seg = audit(f)["match_segments"][0]
        self.assertEqual(seg["matched_transition_count"], 0)
        self.assertEqual(seg["unmatched_gt_transition_count"], 1)


if __name__ == "__main__":
    unittest.main()
