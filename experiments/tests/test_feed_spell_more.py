import unittest
import numpy as np
from experiments import feed_spell_more as study, feed_spell_push as initial


class AdditionalFeedContracts(unittest.TestCase):
    def test_all_rounds_preserve_swap_breaks_and_classes(self):
        x=np.random.default_rng(76).random((2,83,3));x/=x.sum(axis=2,keepdims=True)
        legacy=np.random.default_rng(77).random((2,83,3));legacy/=legacy.sum(axis=2,keepdims=True)
        original=x.copy();period=np.r_[np.zeros(37),np.ones(46)]
        for number,count in ((3,91),(4,78),(5,67)):
            configs=study.configurations(number,study.BASE,study.BASE)
            self.assertEqual(len(configs),count)
            for c in configs:
                y=study.decode(x,period,10,c,legacy)
                np.testing.assert_array_equal(study.decode(x[:,:,[1,0,2]],period,10,c,legacy[:,:,[1,0,2]]),
                    np.where(y=='H','A',np.where(y=='A','H','D')),err_msg=str(c))
                np.testing.assert_array_equal(y,np.r_[study.decode(x[:,:37],period[:37],10,c,legacy[:,:37]),
                    study.decode(x[:,37:],period[37:],10,c,legacy[:,37:])])
                for k in range(3):
                    p=x.copy();q=legacy.copy();p[:,:,k]=0;q[:,:,k]=0
                    self.assertNotIn('HAD'[k],study.decode(p,period,10,c,q))
        np.testing.assert_array_equal(x,original)

    def test_base_matches_previous_winner(self):
        x=np.random.default_rng(78).random((2,103,3));x/=x.sum(axis=2,keepdims=True)
        period=np.zeros(103)
        np.testing.assert_array_equal(study.decode(x,period,25,study.BASE),
            initial.decode(x,period,25,dict(family='calibration',dt=4,ht=1,ha=3,dead=3)))

    def test_consensus_identical_parents_and_seed_swap(self):
        x=np.random.default_rng(79).random((2,203,3));x/=x.sum(axis=2,keepdims=True)
        period=np.zeros(203)
        base=study.emissions(x,period,25,study.BASE)
        for kind in ('arithmetic','geometric'):
            c=dict(family='consensus',left=study.BASE,right=study.BASE,weight=.25,pooling=kind,ha=3,dead=3)
            np.testing.assert_allclose(study.emissions(x,period,25,c),base,atol=1e-15)
        for c in study.configurations(3):
            np.testing.assert_array_equal(study.decode(x,period,25,c),study.decode(x[::-1],period,25,c))
