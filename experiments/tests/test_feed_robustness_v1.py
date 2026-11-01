"""Contracts for raw defects and matched training-view replacement."""
import copy
import unittest

import numpy as np

from experiments import feed_robustness_v1 as study, supervised
from experiments.tests.test_preparation import synthetic


class FeedRobustnessContracts(unittest.TestCase):
    def test_raw_defects_ignore_labels_preserve_native_masks_and_time(self):
        data = synthetic(203)
        data["period"][103:] = 2
        data["player_observed"] = np.isfinite(data["players"]).all(axis=2)
        data["ball_observed"] = np.isfinite(data["ball"]).all(axis=1)
        data["player_observed"][10:20, :2] = False
        data["ball_observed"][20:30] = False
        original = copy.deepcopy(data)
        a = study.raw_view(data, study.sn.default_ball_calibration(), 551103)
        data["labels"] = np.full(203, "D")
        b = study.raw_view(data, study.sn.default_ball_calibration(), 551103)
        self.assertNotIn("labels", a)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for key in ("period", "timestamps", "fps", "teams"):
            np.testing.assert_array_equal(a[key], original[key])
        self.assertTrue(np.isnan(a["players"][10:20, :2]).all())
        self.assertTrue(np.isnan(a["ball"][20:30]).all())
        np.testing.assert_array_equal(data["players"], original["players"])
        np.testing.assert_array_equal(data["ball"], original["ball"])

    def test_replacement_preserves_targets_windows_clean_and_swaps(self):
        names = ["H_distance1", "A_distance1"]
        x = np.arange(24, dtype=float).reshape(12, 2)
        base = {"x": x, "alternate": x + 1, "physical_x": x + 1,
                "names": names, "y": np.resize([0, 1, 2, -1], 12),
                "period": np.r_[np.ones(6), np.full(6, 2)],
                "indices": np.arange(12), "replace_windows": [True, False]}
        records = [copy.deepcopy(base) for _ in range(4)]
        snapshots = copy.deepcopy(records)
        features = x + 100
        result = study.replace_view(records, features)
        for i, r in enumerate(result):
            for key in ("y", "period", "indices", "replace_windows"):
                np.testing.assert_array_equal(r[key], snapshots[i][key])
            np.testing.assert_array_equal(records[i]["x"], snapshots[i]["x"])
        for i in (0, 1):
            np.testing.assert_array_equal(result[i]["x"], snapshots[i]["x"])
            np.testing.assert_array_equal(result[i]["alternate"], snapshots[i]["alternate"])
        np.testing.assert_array_equal(result[2]["x"], features)
        np.testing.assert_array_equal(result[3]["x"], supervised.swap_team_features(features, names))
        for i in (2, 3):
            np.testing.assert_array_equal(result[i]["alternate"], result[i]["x"])

    def test_misalignment_rejected_and_decoder_team_symmetry(self):
        with self.assertRaises(ValueError):
            study.replace_view([], np.zeros((2, 2)))
        p = np.random.default_rng(42).dirichlet([1, 1, 1], size=301)
        period = np.r_[np.ones(151), np.full(150, 2)]
        original = study.decode(p, period, 25.)
        swapped = study.decode(p[:, [1, 0, 2]], period, 25.)
        np.testing.assert_array_equal(swapped, np.where(original == "H", "A", np.where(original == "A", "H", original)))
        # Future-half predictions cannot change previous-half paths.
        changed = p.copy()
        changed[151:] = [0., 0., 1.]
        np.testing.assert_array_equal(study.decode(changed, period, 25.)[:151], original[:151])


if __name__ == "__main__":
    unittest.main()
