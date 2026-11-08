import unittest
import numpy as np
from experiments import spell_frontier_gar_sixphase as fix

class SixPhaseCompatibility(unittest.TestCase):
    def test_same_float64_pooling_for_all_native_rates(self):
        rng=np.random.default_rng(129)
        for phases in (2,5,6):
            array=rng.uniform(.1,1,(2,phases,23,3)).astype(np.float32)
            array/=array.sum(axis=-1,keepdims=True)
            expected=array.astype(np.float64).mean(axis=0).mean(axis=0)
            np.testing.assert_array_equal(fix.sixphase_pool(array.tolist()),expected)
    def test_invalid_inventory_and_probabilities_rejected(self):
        for shape in ((2,4,23,3),(1,6,23,3)):
            with self.assertRaises(ValueError):fix.sixphase_pool(np.full(shape,1/3))
        with self.assertRaises(ValueError):fix.sixphase_pool(np.zeros((2,6,23,3)))
    def test_scoped_patch_restores_original(self):
        before=fix.study.package._pool_seed_phase_probabilities
        with fix.installed():self.assertIs(fix.study.package._pool_seed_phase_probabilities,fix.sixphase_pool)
        self.assertIs(fix.study.package._pool_seed_phase_probabilities,before)

if __name__=='__main__':unittest.main()
