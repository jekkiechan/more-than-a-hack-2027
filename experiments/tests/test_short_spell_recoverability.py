import unittest

import numpy as np

from experiments import short_spell_recoverability as audit


class ShortSpellRecoverabilityTests(unittest.TestCase):
    def test_duration_bins_have_fixed_half_open_edges(self):
        self.assertEqual(audit._duration_bin(.499), "under_0_5s")
        self.assertEqual(audit._duration_bin(.5), "0_5_to_under_1s")
        self.assertEqual(audit._duration_bin(1), "1_to_under_2s")
        self.assertEqual(audit._duration_bin(2), "2_to_under_3s")
        self.assertEqual(audit._duration_bin(3), "3s_or_longer")

    def test_scorer_spans_map_back_to_native_indices_across_inserted_separator(self):
        inserts = np.array([3])
        self.assertEqual(audit._original_span(1, 3, inserts), (1, 3))
        self.assertEqual(audit._original_span(4, 6, inserts), (3, 5))

    def test_miss_mechanisms_are_exclusive_and_preserve_global_conflicts(self):
        common = dict(candidate_matched=False, candidate_best_iou=0.1,
                      decoded_correct_frames=0, raw_best_iou=0.0)
        self.assertEqual(audit._classify_miss(**common, raw_correct_frames=2),
                         "viterbi_suppressed_correct_emission")
        self.assertEqual(audit._classify_miss(**{**common, "decoded_correct_frames": 1},
                                              raw_correct_frames=0),
                         "correct_emission_but_timing_or_duration_insufficient")
        self.assertEqual(audit._classify_miss(**common, raw_correct_frames=0),
                         "no_sufficient_local_correct_team_emission")
        self.assertEqual(audit._classify_miss(**{**common, "candidate_best_iou": .5},
                                              raw_correct_frames=0),
                         "global_assignment_conflict")

    def test_global_matcher_leaves_second_reference_unmatched_when_one_prediction_is_shared(self):
        overlaps = [(0.8, 0, 0), (0.6, 1, 0)]
        matched = audit.pm.matched_gt_flags(overlaps, 2, .5)
        np.testing.assert_array_equal(matched, [True, False])

    def test_nearest_emission_distance_is_measured_within_valid_run(self):
        raw = np.array([0, 0, 1, 0, 2, 2])
        self.assertEqual(audit._nearest_emission_distance(1, 3, 4, 0, 6, raw, 1), 1.0)
        self.assertEqual(audit._nearest_emission_distance(0, 1, 3, 0, 6, raw, 1), 0.0)
        self.assertIsNone(audit._nearest_emission_distance(1, 3, 4, 3, 6, raw, 1))


if __name__ == "__main__":
    unittest.main()
