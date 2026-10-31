"""Corruption leakage, continuity and matched-cohort contracts; synthetic only."""
import unittest
import numpy as np
from experiments import corruption_experiment as ce, dead_transfer, tracking_corruption as tc


def synthetic(n=4000):
    t = np.arange(n)/10
    p = np.stack([np.column_stack((25+i+t*.01, 20+np.sin(t*.1+i)))
                  for i in range(12)], axis=1)
    ball = p[:, 0].copy()
    ball[50:80] = np.nan
    return dict(players=p, ball=ball, teams=np.repeat([0, 1], 6),
                period=np.ones(n, dtype=int), timestamps=t, fps=np.array(10.),
                pitch=np.array([105., 68.]))


class CorruptionContracts(unittest.TestCase):
    def test_targets_never_influence_profile_or_corruption(self):
        x = synthetic()
        a = dict(x, labels=np.full(4000, 'H'), owner=np.zeros(4000))
        b = dict(x, labels=np.full(4000, 'D'), owner=np.ones(4000))
        self.assertEqual(tc.profile(a), tc.profile(b))
        for left, right in zip((tc.corrupt(a, tc.profile(a), 13),),
                               (tc.corrupt(b, tc.profile(b), 13),)):
            self.assertNotIn('labels', left)
            self.assertNotIn('owner', left)
            for key in left:
                np.testing.assert_array_equal(left[key], right[key])

    def test_no_mutation_or_fabricated_observations_and_consistent_masks(self):
        x = synthetic()
        saved = {k: v.copy() for k, v in x.items()}
        y = tc.corrupt(x, tc.profile(x), 22)
        for k in x:
            np.testing.assert_array_equal(x[k], saved[k])
        for key, mask in (('ball', 'ball_observed'), ('players', 'player_observed')):
            np.testing.assert_array_equal(y[mask], np.isfinite(y[key]).all(axis=-1))
        self.assertTrue(np.isnan(y['ball'][50:80]).all())
        self.assertTrue(np.all(y['player_observed'].sum(axis=1) <= 12))
        self.assertGreater(np.mean(~y['ball_observed']), np.mean(~np.isfinite(x['ball']).all(axis=1)))
        self.assertGreater(np.sum(y['player_observed'].sum(axis=1) <= 8), 0)
        for k in ('timestamps', 'period', 'fps', 'teams', 'pitch'):
            np.testing.assert_array_equal(x[k], y[k])

    def test_discontinuity_blocks_and_inactive_periods(self):
        x = synthetic(500)
        x['period'][200:300] = 0
        x['period'][300:] = 2
        calibration = tc.profile(x)
        x2 = {k: v.copy() for k, v in x.items()}
        x2['players'][300:] += 20
        x2['ball'][300:] += 20
        y, z = tc.corrupt(x, calibration, 42), tc.corrupt(x2, calibration, 42)
        np.testing.assert_array_equal(y['players'][:300], z['players'][:300])
        np.testing.assert_array_equal(y['players'][200:300], x['players'][200:300])
        np.testing.assert_array_equal(y['ball'][200:300], x['ball'][200:300])
        other = tc.corrupt(x, calibration, 43)
        self.assertFalse(np.array_equal(y['ball'], other['ball'], equal_nan=True))

    def test_masks_override_finite_coordinates(self):
        x = synthetic(100)
        x['ball_observed'] = np.zeros(100, bool)
        x['player_observed'] = np.zeros((100, 12), bool)
        y = tc.corrupt(x, tc.profile(x), 4)
        self.assertTrue(np.isnan(y['ball']).all())
        self.assertTrue(np.isnan(y['players']).all())

    def test_replicates_are_averaged_within_match_and_missing_cohort_refused(self):
        rows = []
        for source, split in dead_transfer.SPLITS.items():
            for path in split['eval']:
                for view, value in (('corrupt_1', .2), ('corrupt_2', .4)):
                    rows.append(dict(source=source, match=path.stem, fold='pooled',
                                     model='baseline', view=view, **{k: value for k in ce.METRICS}))
        means = ce.match_means(rows, 'baseline', 'corrupted', False)
        self.assertEqual(len(means), 5)
        self.assertAlmostEqual(means[0]['spell_f1'], .3)
        with self.assertRaisesRegex(ValueError, 'Incomplete cohort'):
            ce.match_means(rows[:-1], 'baseline', 'corrupted', False)


if __name__ == '__main__':
    unittest.main()
