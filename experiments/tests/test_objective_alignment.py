"""Synthetic contracts for the frozen objective-alignment study runner."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments import objective_alignment as oa


class ObjectiveAlignmentContracts(unittest.TestCase):
    def test_known_transition_diagnostics_break_at_unknown_and_period_edges(self):
        labels = np.array(['H', 'H', 'A', 'D', 'D', 'X', 'A'])
        prediction = np.array(['H', 'A', 'A', 'D', 'H', 'X', 'A'])
        period = np.array([0, 0, 0, 0, 1, 1, 1])
        result = oa._transition_diagnostics(labels, prediction, period)
        self.assertEqual(result['known_adjacent_pairs'], 3)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (1, 1, 1))
        self.assertEqual(result['turnover']['reference'], 1)
        self.assertEqual(result['turnover']['predicted'], 1)
        self.assertEqual(result['turnover']['f1'], 0.0)

    def test_score_row_reports_official_transition_timing_and_spell_iou(self):
        labels = np.array(list('DDHHHAAADD'))
        n = len(labels)
        inputs = dict(period=np.ones(n, dtype=np.int8), timestamps=np.arange(n) / 25,
                      fps=25.0)
        probability = np.full((n, 3), 1 / 3, dtype=np.float32)
        row = oa._score_row(inputs, labels, labels.copy(), probability,
                            'sportec', 'synthetic', 'clean', 'clean',
                            'control', 2026, 0.0, 0.0)
        self.assertEqual(set(row['spell_by_iou']), {'0.3', '0.4', '0.5', '0.6', '0.7'})
        self.assertEqual(row['spell_f1'], 1.0)
        self.assertEqual(row['transitions']['all_state_timing']['0.5s']['f1'], 1.0)
        self.assertEqual(row['transitions']['live_dead_timing']['0.48']['f1'], 1.0)
        self.assertEqual(row['transitions']['turnover_timing']['0.5s']['f1'], 1.0)

    def test_aggregate_uses_equal_provider_weights_and_fixed_auxiliary(self):
        split = {
            'provider_a': {'eval': [Path('match_a')], 'train': []},
            'provider_b': {'eval': [Path('match_b')], 'train': []},
        }
        rows = []
        for source, spell in (('provider_a', .2), ('provider_b', .8)):
            for arm in oa.ARMS:
                for seed in oa.SEEDS:
                    for view in oa.VIEWS:
                        replicates = ('clean',) if view == 'clean' else oa.REPLICATES
                        for replicate in replicates:
                            rows.append(dict(
                                source=source, match=split[source]['eval'][0].stem,
                                view=view, replicate=replicate, arm=arm, seed=seed,
                                spell_f1=spell, macro_f1=.6, weighted_70_30=.7*spell+.3*.6,
                                d_f1=.7, d_precision=.8, d_recall=.6,
                                short_spell_recall=.5, false_possessions_per_minute=.2,
                                n_gt_spells=5, n_pred_spells=6,
                                spell_by_iou={iou: {'f1': spell} for iou in ('0.3', '0.4', '0.5', '0.6', '0.7')},
                            ))
        with patch.object(oa.dt, 'SPLITS', split):
            summaries = oa._aggregate(rows)
        row = next(value for value in summaries
                   if (value['arm'], value['seed'], value['view']) == ('control', 2026, 'clean'))
        self.assertAlmostEqual(row['spell_f1'], .5)
        self.assertAlmostEqual(row['macro_f1'], .6)
        self.assertAlmostEqual(row['weighted_70_30'], .53)
        self.assertAlmostEqual(row['spell_by_iou']['0.5'], .5)


if __name__ == '__main__':
    unittest.main()
