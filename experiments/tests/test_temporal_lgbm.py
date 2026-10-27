"""Contracts for temporal direction, masks, block edges and D-only fusion."""
import unittest
import numpy as np
from experiments import temporal_lgbm as tl


class TemporalLgbmTests(unittest.TestCase):
    def test_direction_and_window_support(self):
        x = np.arange(90, dtype=np.float32)[:, None]
        features, names = tl.features(x, ['speed'], np.ones(90))
        col = {n:i for i,n in enumerate(names)}
        self.assertEqual(features[40,col['speed__before_1s']], 35)
        self.assertEqual(features[40,col['speed__after_1s']], 45)
        self.assertEqual(features[40,col['speed__past_mean_1s']], 37)
        self.assertEqual(features[40,col['speed__future_mean_1s']], 43)
        self.assertEqual(features[40,col['speed__after_minus_before_1s']], 10)
        reverse, _ = tl.features(x[::-1], ['speed'], np.ones(90))
        self.assertEqual(reverse[40,col['speed__after_minus_before_1s']], -10)
        self.assertEqual(features[0,col['speed__past_coverage_1s']], 0)
        self.assertTrue(np.isnan(features[0,col['speed__past_mean_1s']]))

    def test_no_context_crosses_input_blocks(self):
        x = np.r_[np.zeros(30), np.full(30, 100)].astype(np.float32)[:,None]
        f,n = tl.features(x,['motion'], np.repeat([1,2],30))
        c={v:i for i,v in enumerate(n)}
        self.assertTrue(np.isnan(f[30,c['motion__before_1s']]))
        self.assertTrue(np.isnan(f[29,c['motion__after_1s']]))
        self.assertEqual(f[29,c['motion__future_coverage_5s']],0)
        self.assertEqual(f[30,c['motion__past_coverage_5s']],0)
        self.assertEqual(f[31,c['motion__past_mean_5s']],100)

    def test_missing_values_are_not_zero_motion(self):
        x = np.zeros((40,1),np.float32);x[15:20]=np.nan
        f,n = tl.features(x,['speed'],np.ones(40));c={v:i for i,v in enumerate(n)}
        self.assertEqual(f[20,c['speed__past_coverage_1s']],0)
        self.assertTrue(np.isnan(f[20,c['speed__past_mean_1s']]))
        self.assertEqual(f[10,c['speed__past_mean_1s']],0)
        self.assertEqual(f[10,c['speed__past_coverage_1s']],1)
        self.assertEqual(f[16,c['speed__observed']],0)

    def test_record_metadata_cannot_change_features(self):
        r=dict(x=np.zeros((30,1)), names=['speed'], period=np.ones(30), y=np.zeros(30))
        a,n=tl.design(r)
        b,m=tl.design(dict(r,y=np.ones(30),ball=np.ones((30,2)),labels=np.full(30,'D')))
        np.testing.assert_array_equal(a,b);self.assertEqual(n,m)

    def test_source_match_and_view_weighting(self):
        records=[]
        for s,m,length,views in [('a','a1',20,1),('a','a2',40,2),('b','b1',60,1)]:
            for v in range(views):
                records.append(dict(source=s,match=m,variant=v,x=np.zeros((length,1)),y=np.zeros(length),period=np.ones(length),names=['speed']))
        _,_,w,_=tl.matrix(records,False)
        # Source a: 10 rows + 20 + 20; source b: 30 rows.
        self.assertAlmostEqual(w[:50].sum(),w[50:].sum())
        self.assertAlmostEqual(w[:10].sum(),w[10:50].sum())

    def test_fusion_preserves_conditional_team_probability(self):
        gru=np.array([[.6,.2,.2],[0.,0.,1.]])
        result=tl.fuse(gru,np.array([.8,.4]),.5)
        np.testing.assert_allclose(result[0],[.375,.125,.5])
        np.testing.assert_allclose(result[1],[.15,.15,.7])
        np.testing.assert_allclose(result.sum(1),1)
        np.testing.assert_allclose(tl.fuse(gru,np.zeros(2),0),gru)
        with self.assertRaises(ValueError):tl.fuse(gru,np.array([1.1,0]),.5)


if __name__=='__main__': unittest.main()
