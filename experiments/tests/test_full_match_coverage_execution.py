import unittest

import numpy as np

from experiments.full_match_coverage_execution import stoppage_episodes, summarize_records


def record(frames, *, game="g1", game_period=1, block=1, fps=10.0, outs=(), touches=(), prov=None):
    n = len(frames)
    runs = np.asarray([[x, x, 0, 0, 1, 0] for x in touches], dtype=np.int32).reshape((-1, 6))
    return {
        "full_frame_numbers": np.asarray(frames, dtype=np.int64),
        "native_fps": fps,
        "match": game,
        "game_period": game_period,
        "full_period": np.full(n, block, dtype=np.int32),
        "indices": np.arange(n),
        "out_frames": np.asarray(outs, dtype=np.int64),
        "touch_runs": runs,
        "full_prov": np.asarray(prov if prov is not None else np.zeros(n), dtype=np.int8),
    }


class StoppageEpisodeTests(unittest.TestCase):
    def test_out_and_restart_inside_record_is_complete(self):
        episodes = stoppage_episodes([3, 4], [0, 8], 12)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["status"], "complete_within_record")
        self.assertEqual(episodes[0]["duration_frames_observed"], 5)

    def test_out_without_restart_is_right_censored(self):
        episodes = stoppage_episodes([8], [1, 3], 10)
        self.assertEqual(episodes[-1]["status"], "right_censored_record_end")
        self.assertEqual(episodes[-1]["duration_frames_observed"], 2)

    def test_record_start_out_is_left_censored_even_with_restart(self):
        episodes = stoppage_episodes([0], [7], 10)
        self.assertEqual(episodes[0]["status"], "left_censored_record_start")
        self.assertEqual(episodes[0]["restart"], 7)

    def test_multiple_out_runs_before_restart_do_not_duplicate_one_dead_spell(self):
        episodes = stoppage_episodes([2, 6], [1, 9], 12)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["restart"], 9)
        self.assertEqual(episodes[0]["duration_frames_observed"], 7)

    def test_adjacent_out_frames_are_one_start(self):
        episodes = stoppage_episodes([2, 3, 4], [7], 10)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["out_start"], 2)
        self.assertEqual(episodes[0]["duration_frames_observed"], 5)


class CoverageAggregationTests(unittest.TestCase):
    def test_half_envelope_reports_gaps_and_counts_time_blocks_separately(self):
        rows = [
            record(range(0, 10), block=1, outs=[2], touches=[0, 8]),
            record(range(20, 30), block=1, outs=[], touches=[1]),
            record(range(100, 110), block=2, outs=[], touches=[1]),
        ]
        result = summarize_records(rows)
        self.assertEqual(result["records"], 3)
        self.assertEqual(result["time_blocks_with_observed_records"], 2)
        self.assertEqual(result["union_covered_hours"], round(30 / 10 / 3600, 4))
        self.assertEqual(result["first_to_last_clip_envelope_hours"], round(110 / 10 / 3600, 4))
        self.assertEqual(result["unobserved_gaps_inside_observed_envelope"]["count"], 2)
        self.assertEqual(result["unobserved_gaps_inside_observed_envelope"]["seconds"]["max"], 7.0)
        self.assertEqual(result["out_to_touch_episodes"]["complete_within_same_record_and_bounded"], 1)
        self.assertEqual(result["out_to_touch_episodes"]["right_censored_at_record_end"], 0)

    def test_record_edge_censoring_is_not_joined_to_next_record(self):
        rows = [
            record(range(0, 10), block=1, outs=[8], touches=[1]),
            record(range(20, 30), block=1, outs=[], touches=[2]),
        ]
        result = summarize_records(rows)
        self.assertEqual(result["out_to_touch_episodes"]["right_censored_at_record_end"], 1)
        self.assertEqual(result["out_to_touch_episodes"]["complete_within_same_record_and_bounded"], 0)

    def test_internal_frame_break_must_be_split_before_audit(self):
        with self.assertRaisesRegex(ValueError, "internal frame gap"):
            summarize_records([record([0, 1, 3, 4])])


if __name__ == "__main__":
    unittest.main()
