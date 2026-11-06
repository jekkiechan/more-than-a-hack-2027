import unittest
import numpy as np
from experiments import v2_iterative_round3 as study

class CombinationContracts(unittest.TestCase):
    def test_equal_experts_are_identity_and_inputs_remain_unchanged(self):
        p=np.array([[.2,.6,.2],[.7,.1,.2],[.2,.2,.6]])
        before=p.copy()
        for value in study.combinations(p,p,p).values():
            np.testing.assert_allclose(value[:,:2],p[:,:2]/p[:,:2].sum(1,keepdims=True),rtol=0,atol=1e-15)
            np.testing.assert_array_equal(value[:,2],0)
            np.testing.assert_allclose(value.sum(1),1,rtol=0,atol=1e-15)
        np.testing.assert_array_equal(before,p)

    def test_team_swap_equivariance_and_distinct_mechanisms(self):
        a=np.array([[.8,.1,.1],[.1,.3,.6]])
        b=np.array([[.3,.2,.5],[.3,.6,.1]])
        c=np.array([[.1,.8,.1],[.6,.2,.2]])
        regular=study.combinations(a,b,c)
        swapped=study.combinations(a[:,[1,0,2]],b[:,[1,0,2]],c[:,[1,0,2]])
        for arm in study.ARMS:
            np.testing.assert_allclose(regular[arm][:,[1,0,2]],swapped[arm],rtol=0,atol=1e-15)
        self.assertFalse(np.allclose(regular['trained_calibrated_mean'],regular['three_expert_mean']))
        self.assertFalse(np.allclose(regular['trained_calibrated_mean'],regular['trained_calibrated_log']))

    def test_bad_probability_and_alignment_rejected(self):
        good=np.ones((4,3))/3
        with self.assertRaises(ValueError): study.combinations(good,good[:2],good)
        with self.assertRaises(ValueError): study.combinations(np.zeros((4,3)),good,good)

    def test_combination_preserves_adopted_dead_mask(self):
        n=100
        adopted=np.tile([.8,.1,.1],(n,1));adopted[30:70]=[.01,.01,.98]
        opposite=adopted[:,[1,0,2]]
        period=np.zeros(n,dtype=int)
        baseline=study.base.audit.decode(adopted,period,5.)
        for p in study.combinations(opposite,opposite,adopted).values():
            pred=study.base.fixed_d_prediction(p,adopted,period,5.,3.)
            np.testing.assert_array_equal(pred=='D',baseline=='D')

if __name__=='__main__': unittest.main()
