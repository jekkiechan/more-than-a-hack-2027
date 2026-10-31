import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments import smart_tagging_audit as audit


class SmartTaggingAuditTests(unittest.TestCase):
    def test_long_ball_gap_marks_only_bounded_missing_run(self):
        n = 400
        observed = np.ones(n, dtype=bool)
        observed[20:280] = False  # 10.4 s from observed left to right.
        observed[310:350] = False
        period = np.ones(n, dtype=np.int8)
        timestamps = np.arange(n) / 25.0
        marked = audit.ball_gap_ge_10s(observed, period, timestamps, 25.0)
        self.assertTrue(marked[20:280].all())
        self.assertFalse(marked[0:20].any())
        self.assertFalse(marked[280:].any())
        self.assertFalse(marked[310:350].any())

    def test_long_gap_does_not_cross_period_or_time_break(self):
        n = 400
        observed = np.ones(n, dtype=bool)
        observed[20:150] = False
        observed[151:280] = False
        period = np.ones(n, dtype=np.int8)
        period[150:] = 2
        timestamps = np.arange(n) / 25.0
        marked = audit.ball_gap_ge_10s(observed, period, timestamps, 25.0)
        self.assertFalse(marked.any())

    def test_tracking_anchor_requires_observed_ball_and_clear_margin(self):
        ball = np.array([[10., 10.], [10., 10.], [np.nan, np.nan]])
        players = np.array([
            [[10.5, 10.], [15., 10.]],
            [[10.5, 10.], [12., 10.]],
            [[10.5, 10.], [15., 10.]],
        ])
        inputs = {"ball": ball, "players": players, "teams": np.array([0, 1])}
        side, support = audit.tracking_anchors(inputs, np.isfinite(ball).all(axis=1))
        self.assertEqual(side.tolist(), [0, -1, -1])
        self.assertEqual(support.tolist(), [True, False, False])

    def test_input_loader_never_touches_labels_member(self):
        n = 199319
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "inputs.npz"
            np.savez_compressed(
                path,
                ball=np.zeros((n, 2), dtype=np.float32),
                players=np.zeros((n, 2, 2), dtype=np.float32),
                teams=np.array([0, 1], dtype=np.int8),
                period=np.zeros(n, dtype=np.int16),
                fps=np.array(25.0),
                timestamps=np.arange(n, dtype=np.float64) / 25,
                pitch=np.array([105.3, 68.0]),
                labels=np.array([{"not": "read"}], dtype=object),
            )
            loaded = audit.load_inputs(path)
        self.assertEqual(set(loaded), set(audit.INPUT_KEYS))
        self.assertEqual(loaded["ball"].shape, (n, 2))

    def test_probability_loader_reads_only_probability_member(self):
        n = 199319
        probabilities = np.tile(np.array([0.4, 0.3, 0.3], dtype=np.float32), (n, 1))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "probabilities.npz"
            np.savez_compressed(
                path, probabilities=probabilities,
                labels=np.array([{"not": "read"}], dtype=object),
            )
            loaded = audit.load_probabilities(path, n)
        self.assertEqual(loaded.shape, (n, 3))
        self.assertTrue(np.allclose(loaded[0], [0.4, 0.3, 0.3]))

    def test_summary_is_aggregate_only_and_agreement_is_not_accuracy(self):
        n = 199319
        period = np.zeros(n, dtype=np.int8)
        for half, first, last in __import__("games").half_video_blocks("BAR-ATH"):
            period[first:last + 1] = half
        ball = np.tile(np.array([[10., 10.]], dtype=np.float32), (n, 1))
        players = np.empty((n, 2, 2), dtype=np.float32)
        players[:, 0] = [25., 25.]
        players[:, 1] = [80., 45.]
        center = 11185 + 80
        ball[center] = [25., 25.]
        players[center, 0] = [25.5, 25.]
        players[center, 1] = [80., 45.]
        inputs = {
            "ball": ball, "players": players, "teams": np.array([0, 1]),
            "period": period, "timestamps": np.arange(n) / 25.0,
            "fps": np.array(25.0), "pitch": np.array([105.3, 68.0]),
        }
        event = {"code": "BUILD UP", "start_frame": center - 60,
                 "end_frame": center + 60,
                 "tags": {"Team": "Home Club", "Half": "1st Half"}}
        probabilities = np.tile(np.array([0.2, 0.7, 0.1]), (n, 1))
        result = audit.summarize([event], {"Home Club": "H", "Away Club": "A"},
                                 inputs, probabilities, missing_ball_only=False)
        all_rows = result["strata"]["graded_all"]
        self.assertEqual(all_rows["tag_anchor_agreement_frames"], 1)
        self.assertEqual(all_rows["tag_anchor_contradiction_frames"], 0)
        self.assertEqual(all_rows["baseline_tracking_anchor_agreement_frames"], 0)
        self.assertIn("AGREEMENT ONLY; not accuracy", all_rows["agreement_interpretation"])
        self.assertIn("model AGREEMENT ONLY; not accuracy", all_rows["model_comparison_interpretation"])
        self.assertNotIn("frames_by_tag", result)
        self.assertNotIn("labels", str(result).lower())


if __name__ == "__main__":
    unittest.main()
