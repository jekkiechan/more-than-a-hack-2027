"""Full-class sampling, nonlinear symmetry and coherent-view contracts."""
import unittest
import numpy as np
from experiments import joint_lgbm as j, possession_geometry as g, supervised


def record(n=100):
    # The real204-column schema is not needed to verify mathematical parity;
    # retain all curated existing names and pad neutral columns to204.
    names=list(j.CORE_NAMES)
    names += [f'neutral_{i}' for i in range(204-len(names))]
    names+=list(g.GEOMETRY_NAMES)+list(g.ANCHOR_NAMES)
    rng=np.random.default_rng(23)
    x=rng.uniform(.1,2,(n,len(names))).astype(np.float32)
    return dict(x=x,names=names,base_count=204,period=np.repeat([1,2],n//2),y=np.resize([0,1,2,-1],n))


class JointLgbmContracts(unittest.TestCase):
    def test_sampling_preserves_every_class_mass_and_short_runs(self):
        y=np.array([0,1,1,2,2,2,-1,-1]+[0]*35+[1]*2+[2]*14)
        period=np.ones(len(y));period[24:]=2
        rows,weight=j.sampling(y,period)
        self.assertTrue(np.all(weight>0))
        self.assertFalse(np.any(y[rows]<0))
        for k in (0,1,2):self.assertEqual(weight[y[rows]==k].sum(),(y==k).sum())
        self.assertIn(0,rows)
        self.assertIn(1,rows)
        self.assertIn(2,rows)
        self.assertIn(24,rows)
        self.assertIn(23,rows)
        self.assertLess(len(rows),(y>=0).sum())
        empty,w=j.sampling(np.array([],int),np.array([],int))
        self.assertEqual(len(empty),0);self.assertEqual(len(w),0)

    def test_engineering_has_correct_even_odd_and_paired_parity(self):
        r=record();r['x'][12:17,0]=np.nan
        for family in ('legacy','geometry','full'):
            direct,manifest=j.design(r,family)
            swapped,other=j.design(j.swapped_record(r),family)
            self.assertEqual(manifest,other)
            np.testing.assert_allclose(swapped,j.swap_engineered(direct,manifest),equal_nan=True,atol=2e-6)
            np.testing.assert_allclose(direct,j.swap_engineered(swapped,manifest),equal_nan=True,atol=2e-6)

    def test_no_context_leaks_across_blocks_or_reads_targets(self):
        r=record();a,m=j.design(r,'full')
        changed=dict(r,x=r['x'].copy(),y=np.full(len(r['x']),2),labels=np.full(len(r['x']),'D'))
        changed['x'][50:]+=100
        b,_=j.design(changed,'full')
        np.testing.assert_array_equal(a[:50],b[:50])
        c,_=j.design(dict(r,y=np.full(len(r['x']),-1),owner=np.ones(len(r['x']))),'full')
        np.testing.assert_array_equal(a,c)

    def test_training_views_are_engineered_before_core_replacement(self):
        r=record();r['period']=np.ones(100);r['x'][:]=1
        r['alternate']=np.full_like(r['x'],100)
        r['replace_rows']=np.arange(100)>=50
        x,m=j.training_design(r,'full')
        col=next(i for i,v in enumerate(m) if v['name']=='context__physical_ball_speed__after_minus_before_1s')
        np.testing.assert_allclose(x[45:55,col],0)

    def test_full_vector_blend_changes_all_classes_and_has_exact_endpoints(self):
        tree=np.array([[.7,.2,.1],[.1,.3,.6]])
        gru=np.array([[.1,.6,.3],[.5,.4,.1]])
        np.testing.assert_allclose(j.blend(tree,gru),[[.4,.4,.2],[.3,.35,.35]])
        np.testing.assert_array_equal(j.blend(tree,gru,0),gru)
        np.testing.assert_array_equal(j.blend(tree,gru,1),tree)
        np.testing.assert_allclose(j.blend(tree,gru).sum(1),1)

    def test_symmetrized_three_class_prediction_is_target_independent(self):
        class Estimator:
            classes_=np.array([0,1,2])
            def predict_proba(self,x):
                logits=np.nan_to_num(x[:,:3]);p=np.exp(logits-logits.max(1)[:,None]);return p/p.sum(1)[:,None]
        r=record();_,manifest=j.design(r,'full')
        fitted=dict(family='full',feature_manifest=manifest,estimator=Estimator())
        p=j.predict(r,fitted)
        np.testing.assert_allclose(j.predict(j.swapped_record(r),fitted),p[:,[1,0,2]])
        np.testing.assert_array_equal(j.predict(dict(r,y=np.full(100,2)),fitted),p)
        self.assertEqual(p.shape,(100,3))


if __name__=='__main__':unittest.main()
