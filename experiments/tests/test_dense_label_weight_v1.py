"""Targeted contracts for the bounded dense-label reliability experiment."""
import unittest

import numpy as np

from experiments import dense_label_weight_v1 as study
from experiments import sn_dense_train as sn
from experiments.tests.test_sn_dense_train import old_record, sn_record, teacher


class DenseLabelWeight(unittest.TestCase):
    def test_mask_clips_to_record_period_and_ignores_unknown_changes(self):
        y = np.zeros(50, dtype=np.int8)
        y[10:25] = 1
        period = np.r_[np.zeros(25, dtype=np.int32), np.ones(25, dtype=np.int32)]
        provenance = np.zeros(50, dtype=np.int8)
        provenance[10] = study.dl.P_TOUCH
        record = dict(full_y=y, full_period=period, full_prov=provenance,
                      native_fps=10.0, fps=10.0)
        mask = study.ambiguous_native_mask(record)
        expected = np.zeros(50, dtype=bool)
        expected[5:16] = True
        np.testing.assert_array_equal(mask, expected)
        # The label flip at the period boundary is not a transition target.
        self.assertFalse(mask[24] or mask[25])

    def test_sample_weights_follow_native_indices_and_preserve_interiors(self):
        y=np.zeros(40,dtype=np.int8);y[20:]=1
        provenance=np.zeros(40,dtype=np.int8);provenance[20]=study.dl.P_PROPAGATED
        record=dict(full_y=y,full_period=np.zeros(40,dtype=np.int32),full_prov=provenance,
                    native_fps=10.0,fps=10.0,indices=np.asarray([0,15,20,25,39]))
        mask=study.ambiguous_native_mask(record)
        got=study.sample_weights(record,mask)
        np.testing.assert_array_equal(got,np.asarray([1,.25,.25,.25,1],dtype=np.float32))

    def test_neutral_weighted_loop_matches_frozen_trainer_after_two_updates(self):
        import torch
        torch.set_num_threads(1)
        t=teacher()
        old=[old_record("o1",1),old_record("o2",2)]
        new=[sn_record("g1",3),sn_record("g2",4)]
        vectors=[np.ones(len(r["y"]),dtype=np.float32) for r in new]
        control=sn.train(t,old,new,width=96,lr=3e-4,seed=2026,steps=2,
                         fraction=.5,batch_size=32,ha_target="dense",train_d=True,
                         require_full_native=True,verbose=False)["model"]
        adapted=study.train_weighted(t,old,new,vectors,seed=2026,steps=2)["model"]
        self.assertEqual(set(control["state"]),set(adapted["state"]))
        for key in control["state"]:
            np.testing.assert_allclose(control["state"][key],adapted["state"][key],atol=2e-7,rtol=0,
                                       err_msg=key)
        self.assertEqual(control["step"],adapted["step"])


if __name__ == "__main__":
    unittest.main()
