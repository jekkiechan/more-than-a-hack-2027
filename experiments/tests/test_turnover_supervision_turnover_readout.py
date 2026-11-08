import unittest

from experiments.turnover_supervision_turnover_readout import _aggregate


class TurnoverDiagnosticAggregateTests(unittest.TestCase):
    def test_provider_view_counts_sum_without_emitting_events(self):
        rows = [
            {'provider': 'Metrica', 'view': 'clean', 'cell': 'c1',
             'all_direct_HA_edges': 6, 'eligible_HA_edges': 5, 'recovered_one_to_one_within_1s': 3,
             'misses': 2, 'predicted_HA_edges': 4,
             'misses_baseline_D_at_boundary': 0, 'misses_baseline_D_within_1s': 0,
             'miss_reason_counts': {'no_D_within_1s': 2},
             'miss_posterior_evidence_counts': {'reversed_or_weak': 2},
             'misses_by_D_and_posterior': {'no_D|weak': 2},
             'posterior_evidence_counts_all_eligible': {'both_correct_over_0.5': 3},
             'pre_duration_bins_all_edges': {'<=3s': 2},
             'post_duration_bins_all_edges': {'<=3s': 2},
             'pre_post_duration_bins_all_edges': {'<=3s|<=3s': 1}},
            {'provider': 'Metrica', 'view': 'clean', 'cell': 'c2',
             'all_direct_HA_edges': 9, 'eligible_HA_edges': 7, 'recovered_one_to_one_within_1s': 4,
             'misses': 3, 'predicted_HA_edges': 6,
             'misses_baseline_D_at_boundary': 3, 'misses_baseline_D_within_1s': 3,
             'miss_reason_counts': {'baseline_D_at_boundary': 3},
             'miss_posterior_evidence_counts': {'both_correct_over_0.5': 3},
             'misses_by_D_and_posterior': {'D|good': 3},
             'posterior_evidence_counts_all_eligible': {'both_correct_over_0.5': 4},
             'pre_duration_bins_all_edges': {'<=3s': 3},
             'post_duration_bins_all_edges': {'<=3s': 3},
             'pre_post_duration_bins_all_edges': {'<=3s|<=3s': 2}},
        ]
        result = _aggregate(rows, ('provider', 'view'))[0]
        self.assertEqual(result['eligible_HA_edges'], 12)
        self.assertEqual(result['recovered_one_to_one_within_1s'], 7)
        self.assertEqual(result['miss_reason_counts'], {
            'baseline_D_at_boundary': 3, 'no_D_within_1s': 2})
        self.assertNotIn('events', result)


if __name__ == '__main__':
    unittest.main()
