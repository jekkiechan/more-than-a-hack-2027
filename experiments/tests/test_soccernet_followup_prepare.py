import unittest
from unittest.mock import patch

import numpy as np

from experiments import soccernet_followup_prepare as follow
from experiments import soccernet_segments as segments


class FollowupRawRecoveryTests(unittest.TestCase):
    def test_raw_reconstruction_matches_frozen_native_feature_builder(self):
        rows = []
        for i in range(12):
            rows.append({
                "period": 1, "frame": 100 + i, "timestamp": (100 + i) / 30,
                "home": [("7", "VISIBLE", float(i), -1.0),
                         ("8", "ESTIMATED", 3.0, 4.0)],
                "away": [("2", "VISIBLE", -float(i), 1.0)],
                "balls": [(None, "INVISIBLE", 0.0, 0.0),
                          (None, "VISIBLE", 2.0 + i, 3.0)],
            })
        jerseys = [["7", "8"], ["2"]]
        raw = follow._raw_inputs(rows, jerseys, 30.0)
        expected, names, player_mask, ball_mask, _ = segments._feature_arrays(rows, jerseys, 30.0)
        from experiments import learned_temporal as lt, refine_features
        team, team_names = refine_features.features(raw)
        player, player_names = lt.player_features(raw)
        actual = np.column_stack((team, player)).astype(np.float32)
        self.assertEqual(list(team_names) + ["D_" + n for n in player_names], names)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(raw["player_observed"], player_mask)
        np.testing.assert_array_equal(raw["ball_observed"], ball_mask)
        self.assertTrue(raw["tracking_valid"].all())
        self.assertEqual(raw["players"].shape, (12, 3, 2))

    def test_seed_is_segment_identity_stable(self):
        a = follow._identity_seed("3812", "3812_p1_1-20", 2027)
        self.assertEqual(a, follow._identity_seed("3812", "3812_p1_1-20", 2027))
        self.assertNotEqual(a, follow._identity_seed("3812", "3812_p1_21-40", 2027))
        self.assertNotEqual(a, follow._identity_seed("3812", "3812_p1_1-20", 2028))

    def test_capture_uses_frozen_assembler_and_preserves_segment_identity(self):
        rows = []
        for i in range(18):
            rows.append({"frameNum": i, "period": 1, "videoTimeMs": i / 30 * 1000,
                         "tracking_valid": True,
                         "homePlayers": '[{"jerseyNum":"7","visibility":"VISIBLE","x":0,"y":0}]',
                         "awayPlayers": '[{"jerseyNum":"2","visibility":"VISIBLE","x":1,"y":1}]',
                         "balls": '[{"visibility":"VISIBLE","x":2,"y":3}]',
                         "home_team": None})
        sample = {"sample_id": "a", "start_frame": 0, "center_frame": 6,
                  "team": "home", "action": "PASS", "fps": 30.0, "period_hint": 1}
        with patch.object(segments, "_feature_arrays", side_effect=AssertionError("feature extraction must be skipped")) as feature_call:
            raw_by_segment = follow._capture_raw_segments("77", [sample], {"a": rows},
                                                          ["dummy", "schema"])
        feature_call.assert_not_called()
        self.assertEqual(set(raw_by_segment), {"77_p1_0-17"})
        raw = raw_by_segment["77_p1_0-17"]
        self.assertEqual(len(raw["ball"]), 18)
        self.assertEqual(raw["period"].tolist(), [1] * 18)
        np.testing.assert_array_equal(raw["timestamps"], np.arange(18) / 30)


if __name__ == "__main__":
    unittest.main()
