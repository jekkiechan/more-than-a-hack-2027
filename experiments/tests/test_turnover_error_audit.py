import unittest

import numpy as np

from experiments.turnover_error_audit import audit_cell, duration_bin, one_to_one_matches, segments


class TurnoverErrorAuditContracts(unittest.TestCase):
    def test_segments_break_on_unknown_period_model_time_and_timestamp_gap(self):
        labels = np.array(list("HHXXAAHHHHHH"))
        period = np.array([1] * 4 + [1, 1, 1, 1, 1, 1, 1, 0])
        model = np.array([2] * 7 + [3] * 4 + [3])
        ts = np.arange(12, dtype=float) / 25
        ts[9:] += .2
        got = list(segments(labels, period, model, ts, 25))
        self.assertEqual(got, [(0, 7), (7, 9), (9, 11)])

    def test_unknown_rows_are_preserved_inside_valid_block(self):
        labels = np.array(list("HHXAA"))
        period = np.ones(5, dtype=int)
        model = np.zeros(5, dtype=int)
        ts = np.arange(5) / 25
        self.assertEqual(list(segments(labels, period, model, ts, 25)), [(0, 5)])

    def test_matching_is_one_to_one_and_maximizes_recoveries(self):
        pairs = one_to_one_matches([0.0, 1.8], [0.8, 2.6], tolerance=1.0)
        self.assertEqual(pairs, [(0.0, 0.8), (1.8, 2.6)])
        self.assertEqual(len(one_to_one_matches([0.0], [0.0, .1])), 1)
        self.assertEqual(one_to_one_matches([0.0], [1.0001]), [])

    def test_both_directions_match_only_within_blocks_and_cross_tab_miss_evidence(self):
        labels = np.array(list("HHAAAAHH"))
        # The second block repeats the first block's timestamps; blocks stay distinct.
        period = np.ones(8, dtype=int)
        model = np.array([0] * 4 + [1] * 4)
        ts = np.array([0., 1., 2., 3., 0., 1., 2., 3.])
        probs = np.array([[.9, .1], [.9, .1], [.1, .9], [.1, .9],
                          [.1, .9], [.1, .9], [.9, .1], [.9, .1]])
        decoded = np.array(list("HHD DHHHH".replace(" ", "")))
        meta = {"identity": {"provider": "P", "match": "M", "view": "V", "rep": "0"}}
        row, events = audit_cell("P|M|V|0", meta, (labels, period, model, ts, 1., probs, decoded))
        self.assertEqual({e["direction"] for e in events}, {"H->A", "A->H"})
        self.assertEqual(row["eligible_HA_edges"], 2)
        self.assertEqual(row["recovered_one_to_one_within_1s"], 0)
        self.assertEqual(row["misses_by_D_and_posterior"]["baseline_D_at_boundary|both_correct_over_0.5"], 1)
        self.assertEqual(row["misses_by_D_and_posterior"]["no_D_within_1s|both_correct_over_0.5"], 1)

    def test_duration_bins(self):
        self.assertEqual(duration_bin(1.0), "<=3s")
        self.assertEqual(duration_bin(3.0), "<=3s")
        self.assertEqual(duration_bin(3.04), "3-10s")
        self.assertEqual(duration_bin(10.0), "3-10s")
        self.assertEqual(duration_bin(10.04), ">10s")


if __name__ == "__main__":
    unittest.main()
