import unittest
import numpy as np
from experiments import feed_spell_refine as study, feed_spell_push as first


class FeedRefinementContracts(unittest.TestCase):
    def test_all_arms_swap_breaks_and_unavailable_classes(self):
        x=np.random.default_rng(40).random((2,61,3));x/=x.sum(axis=2,keepdims=True)
        original=x.copy();period=np.r_[np.zeros(26),np.ones(35)]
        self.assertEqual(len(study.configurations()),92)
        for c in study.configurations():
            y=study.decode(x,period,10,c)
            np.testing.assert_array_equal(study.decode(x[:,:,[1,0,2]],period,10,c),
                np.where(y=='H','A',np.where(y=='A','H','D')))
            np.testing.assert_array_equal(y,np.r_[study.decode(x[:,:26],period[:26],10,c),
                                                  study.decode(x[:,26:],period[26:],10,c)])
            for k in range(3):
                p=x.copy();p[:,:,k]=0
                self.assertNotIn('HAD'[k],study.decode(p,period,10,c))
        np.testing.assert_array_equal(x,original)

    def test_single_member_calibration_order_is_equivalent(self):
        x=np.random.default_rng(41).random((1,301,3));x/=x.sum(axis=2,keepdims=True)
        for c in study.configurations():
            if c['family']=='member_calibration':
                np.testing.assert_array_equal(study.decode(x,np.zeros(301),25,c),
                    first.decode(x,np.zeros(301),25,dict(c,family='calibration')))

    def test_member_order_is_irrelevant(self):
        x=np.random.default_rng(42).random((2,117,3));x/=x.sum(axis=2,keepdims=True)
        for c in study.configurations():
            np.testing.assert_array_equal(study.decode(x,np.zeros(117),25,c),
                                          study.decode(x[::-1],np.zeros(117),25,c))
