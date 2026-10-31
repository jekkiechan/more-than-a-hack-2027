"""Contract fixtures for the competition objective, independent of model fitting."""
import json
import unittest

import numpy as np

from evaluation import possession_metrics as pm
from experiments.metrics import evaluate_tracking
from experiments.spell_diagnostics import spell_diagnostics


class SpellMetricContractTests(unittest.TestCase):
    def score(self, gt, pred):
        return evaluate_tracking(list(gt), list(pred))

    def test_wrong_team_never_matches_identical_interval(self):
        self.assertEqual(self.score('HHHH', 'AAAA')['spell_f1_mean'], 0)

    def test_stoppages_are_not_possessions(self):
        r = self.score('DDDD', 'DDDD')
        self.assertEqual(r['n_gt_spells'], 0)
        self.assertEqual(r['spell_f1_mean'], 0)
        self.assertEqual(r['accuracy'], 1)

    def test_one_frame_false_dead_splits_long_possession(self):
        r = self.score('H'*101, 'H'*50+'D'+'H'*50)
        self.assertAlmostEqual(r['accuracy'], 100/101)
        self.assertEqual(r['n_pred_spells'], 2)
        # Both fragments have IoU 50/101: one match at .3/.4 and none above.
        self.assertAlmostEqual(r['spell_f1_mean'], (2/3)*2/5)

    def test_short_and_long_possessions_have_equal_count_weight(self):
        r = self.score('H'+'D'+'A'*100, 'D'+'D'+'A'*100)
        self.assertEqual(r['n_gt_spells'], 2)
        self.assertAlmostEqual(r['spell_f1_mean'], 2/3)

    def test_threshold_equality_is_inclusive(self):
        r = self.score('H'*10, 'H'*3+'D'*7)
        self.assertEqual(r['spell_f1'][.3]['n_matched'], 1)
        self.assertEqual(r['spell_f1'][.4]['n_matched'], 0)
        self.assertAlmostEqual(r['spell_f1_mean'], .2)

    def test_adjacent_intervals_have_zero_overlap(self):
        self.assertEqual(pm.spell_overlaps([(0, 5, pm.H)], [(5, 10, pm.H)]), [])

    def test_abstention_filled_for_spells_but_not_frames(self):
        r = self.score('HHHHH', 'XHXXH')
        self.assertEqual(r['spell_f1_mean'], 1)
        self.assertAlmostEqual(r['accuracy'], .4)
        self.assertEqual(r['n_abstain_on_graded'], 3)

    def test_entire_abstained_run_has_no_predictions(self):
        r = self.score('HHHHH', 'XXXXX')
        self.assertEqual(r['spell_f1_mean'], 0)
        self.assertEqual(r['n_pred_spells'], 0)

    def test_ground_truth_unknown_splits_and_ignores_prediction(self):
        r = self.score('HHXXHH', 'HHAAHH')
        self.assertEqual(r['n_gt_spells'], 2)
        self.assertEqual(r['n_pred_spells'], 2)
        self.assertEqual(r['spell_f1_mean'], 1)
        self.assertEqual(r['accuracy'], 1)

    def test_greedy_matching_is_not_maximum_cardinality(self):
        # Eligible edges at .3: G1/P1=.3, G1/P2=.4, G2/P2=4/11.
        # Official greedy takes G1/P2, preventing the possible two-edge match.
        r = self.score('H'*10+'D'+'H'*4, 'H'*3+'D'+'H'*11)
        self.assertEqual(r['spell_f1'][.3]['n_matched'], 1)
        self.assertAlmostEqual(r['spell_f1_mean'], .2)

    def test_iou_ties_are_deterministic(self):
        overlaps = pm.spell_overlaps([(0, 10, pm.H)], [(0, 4, pm.H), (6, 10, pm.H)])
        self.assertEqual(overlaps, [(.4, 0, 0), (.4, 0, 1)])


class SpellDiagnosticTests(unittest.TestCase):
    def test_parity_with_official_and_json_serializable(self):
        gt, pred = list('DDHHHDAAXXHHDDA'), list('DDHXHDHAXXHHADA')
        periods = np.array([1]*8+[2]*7)
        timestamps = np.arange(15)/10
        timestamps[12:] += 5
        diagnostic = spell_diagnostics(gt, pred, periods, fps=10, timestamps=timestamps)
        official = evaluate_tracking(gt, pred, periods, fps=10, timestamps=timestamps)
        self.assertEqual(diagnostic['spell_f1_mean'], official['spell_f1_mean'])
        self.assertEqual(diagnostic['n_gt_spells'], official['n_gt_spells'])
        self.assertEqual(diagnostic['n_pred_spells'], official['n_pred_spells'])
        for threshold in pm.SPELL_IOUS:
            for key, value in official['spell_f1'][threshold].items():
                self.assertEqual(diagnostic['per_iou'][str(threshold)][key], value)
        json.dumps(diagnostic, allow_nan=False)

    def test_subgroup_recall_uses_global_assignment(self):
        d = spell_diagnostics(list('H'*10+'D'+'H'*4), list('H'*3+'D'+'H'*11), fps=2)
        groups = d['per_iou']['0.3']['reference_recall_groups']
        self.assertEqual(groups['under_3_seconds']['recall'], 0)
        self.assertEqual(groups['3_to_under_10_seconds']['recall'], 1)
        self.assertEqual(d['per_iou']['0.3']['unmatched_with_eligible_overlap'], 1)

    def test_cut_spells_excluded_from_boundary_timing(self):
        d = spell_diagnostics(list('HHDDDAAADHH'), list('HHDDDAAADHH'), fps=10)
        timing = d['matched_complete_boundary_errors_seconds_at_iou_0_5']
        self.assertEqual(d['reference_spells_touching_graded_run_edge'], 2)
        self.assertEqual(timing['excluded_run_edge_matches'], 2)
        self.assertEqual(timing['start']['n'], 1)
        self.assertEqual(timing['start']['mean'], 0)

    def test_no_spells_is_explicit_and_serializable(self):
        d = spell_diagnostics(list('DDXXDD'), list('DDXXDD'))
        self.assertEqual(d['per_iou']['0.5']['reference_recall_groups']['home']['recall'], None)
        self.assertEqual(d['matched_complete_boundary_errors_seconds_at_iou_0_5']['start']['n'], 0)
        json.dumps(d, allow_nan=False)

    def test_error_geometry_detects_fragment_and_merge(self):
        d = spell_diagnostics(list('HHHHH'), list('HHDHH'))
        self.assertEqual(d['overlap_geometry']['reference_overlapped_by_multiple_same_team_predictions'], 1)
        d = spell_diagnostics(list('HHDHH'), list('HHHHH'))
        self.assertEqual(d['overlap_geometry']['predictions_overlapping_multiple_same_team_references'], 1)


if __name__ == '__main__':
    unittest.main()
