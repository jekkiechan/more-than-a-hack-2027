import unittest
import numpy as np
from experiments import supervised, tracking_research as tr, broadcast_fusion as bf


def tracking(n=160, fps=5.):
    t = np.arange(n)/fps
    ball = np.column_stack((20+t, 20+np.sin(t)))
    players = np.stack([ball+[dx, dy] for dx, dy in ((0, 1), (4, 3), (-5, 2), (1, -1), (-3, -4), (5, -4))], axis=1)
    return dict(ball=ball, players=players, teams=np.array([0, 0, 0, 1, 1, 1]),
                period=np.ones(n, dtype=int), timestamps=t, fps=np.array(fps), pitch=np.array([105., 68.]))


class TrackingResearchTests(unittest.TestCase):
    def test_endpoints_and_unbounded_gaps(self):
        data = tracking()
        data['ball'][10:60] = np.nan
        x, names = tr.gap_features(data)
        col = {name:i for i, name in enumerate(names)}
        self.assertEqual(x[20, col['gap_duration_s']], 10)
        np.testing.assert_allclose(x[20, 6:8], data['ball'][9])
        np.testing.assert_allclose(x[20, 8:10], data['ball'][60])
        self.assertEqual(x[60, col['gap_present']], 0)
        self.assertTrue(np.isnan(data['ball'][20]).all())

    def test_half_and_timestamp_edges_do_not_supply_endpoints(self):
        data = tracking()
        data['ball'][70:90] = np.nan
        for time_gap in (False, True):
            view = {k:v.copy() for k,v in data.items()}
            if time_gap:
                view['timestamps'][80:] += 10
            else:
                view['period'][80:] = 2
            x, _ = tr.gap_features(view)
            self.assertTrue(np.isnan(x[75, 8:10]).all())
            self.assertTrue(np.isnan(x[85, 6:8]).all())
            self.assertEqual(x[75, 1], 2)

    def test_target_independence_and_team_symmetry(self):
        data = tracking()
        data['ball'][10:60] = np.nan
        x, names = tr.features(dict(data, labels=np.full(160, 'H'), owner=np.zeros(160)))
        y, _ = tr.features(dict(data, labels=np.full(160, 'D'), owner=np.ones(160)))
        np.testing.assert_array_equal(x, y)
        swapped, other = tr.features(dict(data, teams=1-data['teams']))
        self.assertEqual(names, other)
        np.testing.assert_allclose(swapped, supervised.swap_team_features(x, names), equal_nan=True)

    def test_duplicates_preserve_opposing_players_and_observation_masks(self):
        data = tracking()
        data['players'][:, 1] = data['players'][:, 0]+.1
        data['players'][:, 3] = data['players'][:, 0]+.1
        data['player_observed'] = np.ones((160, 6), dtype=bool)
        data['player_observed'][:10, 2] = False
        out = tr.deduplicate(data)
        self.assertTrue(np.isnan(out['players'][:, 1]).all())
        self.assertTrue(np.isfinite(out['players'][:, 3]).all())
        self.assertTrue(np.isnan(out['players'][:10, 2]).all())
        self.assertTrue(np.isfinite(data['players']).all())

    def test_conditioned_views_are_reproducible_target_independent_and_block_local(self):
        data = tracking(600)
        data['ball'][25:40] = np.nan
        data['ball'][110:150] = np.nan
        data['ball'][260:320] = np.nan
        data['period'][350:] = 2
        cal = tr.fit_missingness(dict(data, labels=np.full(600, 'H')))
        one = tr.challenge_view(data, cal, 2026)
        two = tr.challenge_view(dict(data, labels=np.full(600, 'D'), owner=np.ones(600)), cal, 2026)
        for key in one:
            np.testing.assert_array_equal(one[key], two[key])
        self.assertNotIn('labels', one)
        self.assertNotIn('owner', one)
        self.assertTrue(np.isnan(one['ball'][25:40]).all())
        self.assertEqual(one['players'].shape[0], 600)
        np.testing.assert_array_equal(one['timestamps'], data['timestamps'])
        self.assertLessEqual((~one['ball_observed']).mean(), cal['profile']['ball_missing_fraction']+1/600)
        clean = tracking(600)
        clean['period'][350:] = 2
        injected = tr.challenge_view(clean, cal, 2027, players=False)
        self.assertEqual(int((~injected['ball_observed']).sum()), 115)
        self.assertTrue(np.isfinite(clean['ball']).all())

    def test_missingness_requires_both_outcomes(self):
        with self.assertRaises(ValueError):
            tr.fit_missingness(tracking())


class BroadcastFusionTests(unittest.TestCase):
    def test_assignment_needs_both_teams(self):
        data = tracking()
        p = data['players'][0].copy()
        p[3:] = np.nan
        self.assertFalse(bf.accepted(bf.geometry(p, data['teams'], p, data['teams'])))

    def test_known_offset_and_gaps(self):
        tactical = tracking(180, fps=1.)
        t = tactical['timestamps']
        shift = np.column_stack((12*np.sin(t/13), 5*np.cos(t/19)))
        tactical['players'] += shift[:, None]
        broadcast = {k:v.copy() for k,v in tactical.items()}
        for key in ('ball', 'players', 'period', 'timestamps'):
            broadcast[key] = broadcast[key][5:155].copy()
        broadcast['timestamps'] = np.arange(150, dtype=float)
        alignment, report = bf.align(tactical, broadcast)
        ok = alignment >= 0
        np.testing.assert_array_equal(alignment[ok], np.arange(150)[ok]+5)
        self.assertGreater(report['accepted_fraction'], .9)
        tactical['ball'][40:60] = np.nan
        fused, audit = bf.fuse(tactical, broadcast, alignment)
        self.assertEqual(audit['recovered_frames'], 20)
        np.testing.assert_allclose(fused['ball'][40:60], broadcast['ball'][35:55])
        np.testing.assert_array_equal(fused['ball'][:40], tactical['ball'][:40])

    def test_duplicate_mapping_mask_and_bounds(self):
        data = tracking(10)
        data['ball'][2:5] = np.nan
        broadcast = tracking(10)
        broadcast['ball_observed'] = np.ones(10, dtype=bool)
        broadcast['ball_observed'][4] = False
        alignment = np.arange(10)
        alignment[3] = 2
        _, audit = bf.fuse(data, broadcast, alignment)
        self.assertEqual(audit['recovered_frames'], 0)
        with self.assertRaises(ValueError):
            bf.fuse(data, broadcast, np.full(10, 10))


if __name__ == '__main__':
    unittest.main()
