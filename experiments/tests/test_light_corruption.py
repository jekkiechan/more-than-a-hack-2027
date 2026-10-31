"""Budget, group isolation and team-symmetry contracts for lighter corruption."""
import importlib.util
import unittest
from unittest.mock import patch
import numpy as np
from experiments import learned_temporal as lt, light_corruption as light, tracking_corruption as tc
from experiments.tests.test_tracking_corruption import synthetic


class LightCorruptionContracts(unittest.TestCase):
    def test_fixed_fraction_and_seed(self):
        a = light.chosen(400, 42)
        self.assertEqual(a.sum(), 100)
        np.testing.assert_array_equal(a, light.chosen(400, 42))
        self.assertFalse(np.array_equal(a, light.chosen(400, 43)))
        self.assertEqual(light.chosen(0, 42).shape, (0,))

    def test_row_budget_and_team_swap_pairs_preserved(self):
        clean = np.arange(400, dtype=float).reshape(200, 2)
        legacy, gap = clean+1000, np.full_like(clean, -1000)
        result = light.mix_team(clean, legacy, gap, 'synthetic')
        self.assertEqual(result.shape, (400, 2))
        for block in (result[:200], result[200:]):
            replaced = block[:, 0] == -1000
            self.assertEqual(replaced.sum(), 50)
            np.testing.assert_array_equal(replaced[:100], replaced[100:])
        with self.assertRaisesRegex(ValueError, 'equal'):
            light.mix_team(clean, legacy[:-1], gap, 'synthetic')

    def test_ball_gap_group_preserves_players_and_observed_coordinates(self):
        inputs = synthetic()
        original = {k: v.copy() for k, v in inputs.items()}
        a = light.ball_gap_view(dict(inputs, labels=np.full(4000, 'H')), tc.profile(inputs), 44)
        b = light.ball_gap_view(dict(inputs, labels=np.full(4000, 'D')), tc.profile(inputs), 44)
        np.testing.assert_array_equal(a['players'], inputs['players'])
        np.testing.assert_array_equal(a['ball'], b['ball'])
        seen = np.isfinite(a['ball']).all(axis=1)
        np.testing.assert_array_equal(a['ball'][seen], inputs['ball'][seen])
        self.assertLess(seen.sum(), np.isfinite(inputs['ball']).all(axis=1).sum())
        self.assertNotIn('labels', a)
        for key in original: np.testing.assert_array_equal(inputs[key], original[key])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch not installed')
    def test_neural_optimizer_budget_and_targets_unchanged(self):
        records = []
        for variant in (0, 1):
            r = dict(source='synthetic', match='one', variant=variant,
                     x=np.sin(np.arange(400, dtype=np.float32)*.01)[:, None],
                     physical_x=np.zeros((400, 1), np.float32),
                     y=np.repeat([0, 1], 200).astype(np.int8), period=np.ones(400))
            windows = list(lt.chunks(r))
            select = light.chosen(len(windows), 50+variant)
            r['replace_windows'] = select
            r['replace_rows'] = np.repeat(select, lt.CORE)
            records.append(r)
        old_targets = [r['y'].copy() for r in records]
        with patch.object(lt, 'EPOCHS', 1):
            model = light.fit_neural(records)
        self.assertEqual(model['windows_per_epoch'], 16)
        self.assertEqual(model['optimizer_updates'], 1)
        for r, y in zip(records, old_targets): np.testing.assert_array_equal(r['y'], y)


if __name__ == '__main__': unittest.main()
