"""Physical, mask, symmetry and timeline contracts for full H/A/D features."""
import unittest

import numpy as np

from experiments import possession_geometry as pg, supervised


def fixture(n=100, fps=10):
    t=np.arange(n)/fps
    ball=np.column_stack((np.full(n,20.),np.full(n,30.)))
    players=np.stack([ball+[.5,0],ball+[2,0],ball+[5,0],ball+[7,0]],axis=1)
    return dict(ball=ball,players=players,teams=np.array([0,0,1,1]),
                period=np.ones(n),timestamps=t,fps=fps,pitch=np.array([105.,68.]),
                ball_observed=np.ones(n,bool),player_observed=np.ones((n,4),bool))


class PossessionGeometryTests(unittest.TestCase):
    def test_stationary_observed_control_is_not_unknown_motion(self):
        data=fixture()
        x,names=pg.features(data)
        idx={n:i for i,n in enumerate(names)}
        expected=np.exp(-.5*(.5/1.8)**2)
        np.testing.assert_allclose(x[:,idx['H_geometry_best_control']],expected,rtol=1e-6)
        np.testing.assert_allclose(x[:,idx['H_geometry_control_candidate_fraction']],1)
        self.assertTrue(np.isnan(x[:,idx['H_geometry_approach_time_s']]).all())
        np.testing.assert_allclose(x[:,idx['H_geometry_approach_candidate_fraction']],0)

    def test_best_control_can_be_second_nearest_player(self):
        data=fixture(21)
        t=data['timestamps']
        data['players'][:,0,0]=20+.2+10*(t-1)
        data['players'][:,1,0]=21
        x,names=pg.features(data)
        self.assertAlmostEqual(x[10,names.index('H_geometry_best_control')],np.exp(-.5*(1/1.8)**2),places=6)
        expected=np.exp(-.5*(.2/1.8)**2)*np.exp(-.5*(10/5)**2)
        self.assertAlmostEqual(x[10,names.index('H_geometry_second_control')],expected,places=6)

    def test_projected_interception_has_physical_time_and_miss(self):
        data=fixture(21)
        t=data['timestamps']
        data['players'][:,0,0]=22-t
        data['players'][:,1,0]=24-2*t
        x,names=pg.features(data)
        self.assertAlmostEqual(x[10,names.index('H_geometry_approach_time_s')],1,places=6)
        self.assertAlmostEqual(x[10,names.index('H_geometry_approach_miss_distance_m')],0,places=6)
        self.assertAlmostEqual(x[10,names.index('H_geometry_approach_improvement_m')],1,places=6)
        permutation=[1,0,3,2]
        changed=dict(data,players=data['players'][:,permutation],teams=data['teams'][permutation],
                     player_observed=data['player_observed'][:,permutation])
        swapped,other_names=pg.features(changed)
        self.assertEqual(names,other_names)
        np.testing.assert_allclose(swapped,x,rtol=1e-6,atol=1e-6,equal_nan=True)

    def test_raw_team_swap_matches_schema_swap_with_anchors(self):
        data=fixture()
        data['ball_observed'][30:60]=False
        x,names=pg.features(data,include_anchors=True)
        actual,other_names=pg.features(dict(data,teams=1-data['teams']),include_anchors=True)
        self.assertEqual(names,other_names)
        np.testing.assert_allclose(actual,supervised.swap_team_features(x,names),rtol=1e-6,atol=1e-6,equal_nan=True)

    def test_target_and_event_metadata_are_never_features(self):
        data=fixture()
        original,names=pg.features(data,include_anchors=True)
        changed=dict(data,labels=np.full(100,'D'),owner=np.ones(100),possession=np.zeros(100),events=['restart']*100)
        actual,other_names=pg.features(changed,include_anchors=True)
        self.assertEqual(names,other_names)
        np.testing.assert_array_equal(original,actual)

    def test_finite_but_unobserved_positions_cannot_affect_features(self):
        data=fixture()
        data['ball_observed'][30:60]=False
        data['player_observed'][:,0]=False
        original,names=pg.features(data,include_anchors=True)
        changed=dict(data,ball=data['ball'].copy(),players=data['players'].copy())
        changed['ball'][30:60]=1e6
        changed['players'][:,0]=-1e6
        actual,_=pg.features(changed,include_anchors=True)
        np.testing.assert_array_equal(original,actual)
        self.assertTrue(np.isnan(actual[30:60,names.index('H_geometry_best_control')]).all())

    def test_missing_team_is_unknown_not_zero_control(self):
        data=fixture()
        data['player_observed'][:,:2]=False
        x,names=pg.features(data,include_anchors=True)
        self.assertTrue(np.isnan(x[:,names.index('H_geometry_best_control')]).all())
        self.assertTrue(np.isnan(x[:,names.index('home_geometry_control_margin')]).all())
        self.assertTrue(np.isfinite(x[:,names.index('A_geometry_best_control')]).all())
        np.testing.assert_array_equal(x[:,names.index('reliable_contact_valid_before')],0)

    def test_anchors_bridge_observation_gap_without_coordinates(self):
        data=fixture()
        data['ball_observed'][30:60]=False
        x,names=pg.features(data,include_anchors=True)
        row=dict(zip(names,x[45]))
        self.assertTrue(np.isnan(row['H_geometry_best_control']))
        self.assertEqual(row['reliable_contact_bracket_valid'],1)
        self.assertEqual(row['reliable_contact_same_team'],1)
        self.assertGreater(row['reliable_contact_age_before_s'],0)
        self.assertGreater(row['reliable_contact_age_after_s'],0)
        self.assertGreater(row['home_reliable_margin_before'],.2)

    def test_anchors_expire_and_never_cross_time_breaks(self):
        data=fixture(300)
        data['ball_observed'][30:270]=False
        x,names=pg.features(data,include_anchors=True)
        self.assertEqual(x[150,names.index('reliable_contact_bracket_valid')],0)
        self.assertTrue(np.isnan(x[150,names.index('reliable_contact_age_before_s')]))
        self.assertTrue(np.isnan(x[150,names.index('reliable_contact_age_after_s')]))
        data=fixture()
        data['timestamps'][50:]+=20
        x,names=pg.features(data,include_anchors=True)
        self.assertEqual(x[49,names.index('reliable_contact_valid_after')],0)
        self.assertEqual(x[50,names.index('reliable_contact_valid_before')],0)
        changed=dict(data,ball=data['ball'].copy(),players=data['players'].copy())
        changed['players'][50:]+=2
        actual,_=pg.features(changed,include_anchors=True)
        np.testing.assert_array_equal(actual[:50],x[:50])

    def test_isolated_evidence_does_not_create_reliable_anchor(self):
        # Single observed frame has no supported velocity, unlike stationary
        # observed tracking. It must not become a strong contact anchor.
        data=fixture()
        data['ball_observed'][:]=False
        data['ball_observed'][50]=True
        x,names=pg.features(data,include_anchors=True)
        np.testing.assert_array_equal(x[:,names.index('reliable_contact_valid_before')],0)
        np.testing.assert_array_equal(x[:,names.index('reliable_contact_valid_after')],0)

    def test_signed_collective_acceleration_distinguishes_restart(self):
        data=fixture(31)
        t=data['timestamps']
        data['players'][:,:,0]+=.5*t[:,None]**2
        x,names=pg.features(data)
        np.testing.assert_allclose(x[5:25,names.index('geometry_collective_acceleration_median')],1,atol=1e-5)
        np.testing.assert_allclose(x[5:25,names.index('geometry_collective_accelerating_fraction')],1)
        np.testing.assert_allclose(x[5:25,names.index('geometry_collective_braking_fraction')],0)

    def test_chunk_halo_preserves_full_block_derivatives(self):
        data=fixture(8200,25)
        data['players'][:,:,0]+=.02*np.sin(data['timestamps'])[:,None]
        data['player_observed'][8188:8191,0]=False
        actual,names=pg.features(data)
        expected=pg._chunk_geometry(data,0,8200,data['timestamps'],data['teams'])
        self.assertEqual(len(names),len(pg.GEOMETRY_NAMES))
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-6,equal_nan=True)

    def test_manifest_and_group_separation(self):
        x,names=pg.features(fixture(),include_anchors=True)
        self.assertEqual(len(pg.GEOMETRY_NAMES),23)
        self.assertEqual(len(pg.ANCHOR_NAMES),14)
        self.assertEqual(names,list(pg.GEOMETRY_NAMES+pg.ANCHOR_NAMES))
        self.assertEqual(len(set(names)),37)
        self.assertEqual(x.dtype,np.float32)


if __name__=='__main__':unittest.main()
