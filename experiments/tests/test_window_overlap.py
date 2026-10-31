"""Synthetic contracts for the fixed shifted-core inference intervention."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments import live_gap_experiment
from experiments import window_overlap_experiment as wo


class ShiftedTilingContracts(unittest.TestCase):
    def test_shifted_cores_cover_partial_right_edges_without_crossing_blocks(self):
        record = {
            "x": np.zeros((200, 3), dtype=np.float32),
            "period": np.r_[np.zeros(130, dtype=int), np.ones(70, dtype=int)],
        }
        windows = wo.tiled_windows(record, offset=25)
        retained = [(w["retained_start"], w["retained_stop"]) for w in windows]
        self.assertEqual(retained, [(25, 75), (75, 125), (125, 130), (155, 200)])
        self.assertTrue(all(len(w["full"]) == wo.CORE + 2 * wo.CONTEXT for w in windows))
        self.assertEqual(windows[0]["core_offset"], 0)
        self.assertEqual(windows[-1]["size"], 45)
        for window in windows:
            lo, hi = window["block_start"], window["block_stop"]
            self.assertTrue(np.all((window["full"] >= lo) & (window["full"] < hi)))
            self.assertEqual(window["retained_start"] - window["window_start"],
                             window["core_offset"])

    def test_zero_offset_matches_the_training_window_partition(self):
        period = np.r_[np.zeros(120, dtype=int), np.ones(55, dtype=int)]
        record = {"x": np.zeros((len(period), 2)), "period": period}
        actual = wo.tiled_windows(record, offset=0)
        expected = list(__import__("experiments.learned_temporal", fromlist=["chunks"])
                        .chunks(record))
        self.assertEqual(len(actual), len(expected))
        for got, (start, size, full) in zip(actual, expected):
            self.assertEqual(got["retained_start"], start)
            self.assertEqual(got["size"], size)
            np.testing.assert_array_equal(got["full"], full)

    def test_native_upsample_marks_only_shifted_supported_rows(self):
        record = {
            "indices": np.array([0, 2, 4, 10, 12, 14]),
        }
        p = np.tile(np.array([[.2, .3, .5]]), (6, 1))
        mask = np.array([False, True, True, False, False, True])
        native_period = np.r_[np.zeros(10, dtype=int), np.ones(10, dtype=int)]
        shifted, valid = wo.upsample_shifted(record, p, mask, native_period)
        self.assertEqual(np.flatnonzero(valid).tolist(), list(range(2, 10)) + list(range(14, 20)))
        self.assertTrue(np.isnan(shifted[~valid]).all())
        np.testing.assert_allclose(shifted[valid], np.tile([.2, .3, .5], (14, 1)))

    def test_equal_tiling_average_uses_original_at_uncovered_edges(self):
        original = np.array([[.6, .2, .2], [.3, .4, .3]])
        shifted = np.array([[.2, .5, .3], [.1, .2, .7]])
        actual = wo.average_tilings(original, shifted, np.array([True, False]))
        np.testing.assert_allclose(actual[0], [.4, .35, .25])
        np.testing.assert_array_equal(actual[1], original[1])

    def test_inference_feature_builder_supplies_only_unknown_placeholders(self):
        observed = {}

        def fake_feature_view(inputs, labels, source, match):
            observed["labels"] = labels.copy()
            observed["source"] = source
            observed["match"] = match
            return {"joint": {"x": np.zeros((3, 1)), "names": ["feature"]}}

        with patch.object(live_gap_experiment, "feature_view", side_effect=fake_feature_view):
            wo._feature_record({"period": np.zeros(3)}, "metrica", "metrica_3")
        self.assertEqual(observed["labels"].tolist(), ["X", "X", "X"])
        self.assertEqual((observed["source"], observed["match"]), ("metrica", "metrica_3"))

    def test_cached_inference_record_discards_target_arrays(self):
        base_record = {
            "x": np.zeros((2, 1), dtype=np.float32),
            "names": ["feature"],
            "period": np.array([0, 0]),
            "indices": np.array([0, 1]),
            "native_n": 2,
            "fps": 25.0,
            "y": np.array([0, 1]),
        }
        first = wo._inference_record_from_cache({
            "record": base_record, "labels": np.array(["H", "A"])
        })
        changed = dict(base_record, y=np.array([2, 2]))
        second = wo._inference_record_from_cache({
            "record": changed, "labels": np.array(["D", "D"])
        })
        self.assertNotIn("y", first)
        self.assertNotIn("labels", first)
        self.assertEqual(first.keys(), second.keys())
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_preflight_archive_contract_includes_all_seed_arrays(self):
        expected = {
            "original_ensemble", "candidate_ensemble", "shifted_filled_ensemble",
            "native_coverage_mask",
        }
        for seed in (2026, 2027):
            expected.update({f"original_{seed}", f"candidate_{seed}",
                             f"shifted_filled_{seed}"})
        self.assertEqual(wo._preflight_archive_keys(), expected)

    def test_original_probability_parity_is_required(self):
        p = np.array([[.2, .3, .5], [.7, .2, .1]])
        wo._assert_exact_parity(p, p.copy(), {"match": "synthetic"})
        with self.assertRaises(wo.StudyError):
            wo._assert_exact_parity(p, p[::-1].copy(), {"match": "synthetic"})


if __name__ == "__main__":
    unittest.main()
