import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import pcbas_event_anchor_v1 as study


def rows(frames=(32, 33, 36)):
    result = []
    for frame in frames:
        for player, team in ((100, 0), (200, 1)):
            result.append([frame, player, team, 7, 1, .25 + team*.5, .5,
                           0, 0, 10, 20, 30, 40])
    return np.array(result, dtype=float)


class PCBASAdapterContracts(unittest.TestCase):
    def test_scale_native_clock_half_and_gap_preserved(self):
        from experiments import supervised
        inputs, frames, audit = study.adapt_rows(rows(), 2)
        np.testing.assert_array_equal(frames, [32, 33, 36])
        np.testing.assert_allclose(inputs['timestamps'], np.array([32, 33, 36])/25)
        np.testing.assert_array_equal(inputs['period'], [2, 2, 2])
        np.testing.assert_allclose(inputs['players'][0], [[26.25, 34], [78.75, 34]])
        np.testing.assert_array_equal(supervised.model_periods(inputs), [0, 0, 1])
        self.assertFalse(np.isfinite(inputs['ball']).any())
        self.assertFalse(inputs['ball_observed'].any())
        self.assertEqual(audit['source_rows'], 6)

    def test_offscreen_imputation_is_masked_without_clipping(self):
        data = rows()
        data[0, 9:13] = np.nan
        data[1, 5:7] = [1.1, -.2]
        inputs, _, audit = study.adapt_rows(data, 1)
        self.assertFalse(inputs['player_observed'][0, 0])
        self.assertTrue(np.isnan(inputs['players'][0, 0]).all())
        np.testing.assert_allclose(inputs['players'][0, 1], [115.5, -13.6])
        self.assertEqual(audit['source_unobserved_or_imputed_rows'], 1)

    def test_event_role_shirt_and_source_velocity_do_not_enter_features(self):
        data = rows()
        expected, _, _ = study.adapt_rows(data, 1)
        data[:, [3, 4, 7, 8]] = 987654
        actual, _, _ = study.adapt_rows(data, 1)
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])
        with self.assertRaisesRegex(ValueError, '13 feature'):
            study.adapt_rows(np.column_stack((data, np.ones(len(data)))), 1)

    def test_rejects_duplicate_fractional_unknown_and_changing_identity(self):
        for mutation, message in ((lambda x: np.concatenate([x, x[:1]]), 'Duplicate'),
                                  (lambda x: self.change(x, 0, 0, 32.5), 'Noninteger'),
                                  (lambda x: self.change(x, 0, 2, 2), 'unknown direction'),
                                  (lambda x: self.change(x, 0, 2, 1), 'changes direction')):
            with self.assertRaisesRegex(ValueError, message):
                study.adapt_rows(mutation(rows()), 1)

    @staticmethod
    def change(data, row, column, value):
        data[row, column] = value
        return data

    def test_exact_package_dispatch_strips_targets_and_preserves_masks(self):
        from experiments import original_full_phase_baseline
        inputs, _, _ = study.adapt_rows(rows(), 1)
        inputs.update(labels=object(), event_team=object(), action_class=object())
        with mock.patch.object(original_full_phase_baseline, 'predict_labels', return_value=np.array(list('HAD'))) as call:
            result = study.infer(inputs, {'fixed': 1}, {'package': 'sealed'})
        np.testing.assert_array_equal(result, list('HAD'))
        passed = call.call_args.args[0]
        self.assertNotIn('labels', passed)
        self.assertNotIn('event_team', passed)
        self.assertNotIn('action_class', passed)
        self.assertFalse(passed['ball_observed'].any())
        self.assertEqual(call.call_args.args[1:], ({'fixed': 1}, {'package': 'sealed'}))


class PCBASAnchorContracts(unittest.TestCase):
    def test_replay_background_collision_duplicate_and_no_time_tolerance(self):
        events = [[10, 0, 7, 1, 0, 0], [10, 0, 7, 1, 0, 0],
                  [11, 1, 8, 2, 1, 1], [12, 0, 7, 0, 1, 0],
                  [13, 0, 7, 1, 1, 0], [13, 1, 8, 6, 1, 0],
                  [14, 0, 7, 2, 1, 0]]
        selected, excluded = study.select_anchors(events, [10, 11, 12, 13, 15])
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]['bbox_flag'], 0)  # bbox missing is not replay.
        self.assertEqual(excluded, dict(exact_duplicate=1, replay=1,
            nonselected_class=2, opposing_team_same_frame=1, no_exact_tracking_frame=1))

    def test_dead_and_unknown_count_wrong_and_team_swap_is_symmetric(self):
        events = [[1, 0, 7, 1, 1, 0], [2, 1, 8, 1, 0, 0],
                  [3, 0, 7, 1, 1, 0], [4, 1, 8, 1, 1, 0]]
        selected, _ = study.select_anchors(events, [1, 2, 3, 4])
        measured = study.anchor_counts(selected, list('HADX'))['drive']
        self.assertEqual((measured['n'], measured['correct']), (4, 2))
        self.assertEqual(measured['predicted_state_counts']['D'], 1)
        swapped = [dict(row, team=1-row['team']) for row in selected]
        self.assertEqual(study.anchor_counts(swapped, list('AHDX'))['drive']['correct'], 2)

    def test_halves_sum_inside_games_then_games_have_equal_weight(self):
        def cell(n, correct):
            return {name:dict(n=n, correct=correct, predicted_state_counts={'H':n}) for name in study.CLASSES.values()}
        out = study.summarize({'game_18_H1':cell(1, 1), 'game_18_H2':cell(9, 0),
                               'game_24_H1':cell(1, 1)})['drive']
        self.assertAlmostEqual(out['equal_game_team_agreement'], .55)
        self.assertAlmostEqual(out['pooled_team_agreement'], 2/11)

    def test_write_once_preserves_closed_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'gate.json'
            study.write_once(path, {'closed':True})
            with self.assertRaises(FileExistsError):
                study.write_once(path, {'closed':False})
            self.assertTrue(json.loads(path.read_text())['closed'])


if __name__ == '__main__':
    unittest.main()
