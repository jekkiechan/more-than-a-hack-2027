"""Contracts for actual two-seed H/A/D averaging, using one shared feature pass."""
import unittest
from unittest.mock import patch
import numpy as np
from experiments import current_baseline as baseline, factorized_had, live_gap_experiment


class SeedEnsembleBaselineTests(unittest.TestCase):
    def setUp(self):
        self.inputs=dict(ball=np.zeros((4,2)),period=np.ones(4),fps=np.asarray(5.),timestamps=np.arange(4)/5)
        self.record=dict(indices=np.arange(4),native_n=4,period=np.zeros(4))
        self.members={'joint_gru_2026':dict(seed=2026),'joint_gru_2027':dict(seed=2027)}
        self.manifest=dict(architecture=dict(kind='joint_gru_ensemble',members={'joint_gru_2026':.5,'joint_gru_2027':.5}))
        self.a=np.array([[.8,.1,.1]]*4)
        self.b=np.array([[.2,.6,.2]]*4)

    def test_native_probabilities_are_averaged_after_one_feature_pass(self):
        with patch.object(live_gap_experiment,'feature_view',return_value={'joint':self.record}) as features, \
             patch.object(factorized_had,'predict',side_effect=[self.a,self.b]) as predict:
            result=baseline.predict_probabilities(self.inputs,self.members,self.manifest)
        np.testing.assert_allclose(result,np.tile([.5,.35,.15],(4,1)))
        self.assertEqual(features.call_count,1)
        self.assertEqual(predict.call_count,2)
        self.assertTrue(np.all(features.call_args.args[1]=='X'))

    def test_invalid_members_seeds_weights_and_context_are_rejected(self):
        cases=[({'joint_gru_2026':.4,'joint_gru_2027':.6},self.members),
               ({'joint_gru_2026':1.},self.members),
               (self.manifest['architecture']['members'],{'joint_gru_2026':dict(seed=2026),'joint_gru_2027':dict(seed=2026)}),
               (self.manifest['architecture']['members'],{'joint_gru_2026':dict(seed=2026),'joint_gru_2027':dict(seed=2027,context_seconds=15)})]
        for weights,members in cases:
            with self.subTest(weights=weights),self.assertRaises(ValueError):
                baseline.predict_probabilities(self.inputs,members,dict(architecture=dict(kind='joint_gru_ensemble',members=weights)))

    def test_invalid_member_probability_is_rejected(self):
        with patch.object(live_gap_experiment,'feature_view',return_value={'joint':self.record}), \
             patch.object(factorized_had,'predict',side_effect=[self.a,np.full_like(self.b,np.nan)]), \
             self.assertRaises(ValueError):
            baseline.predict_probabilities(self.inputs,self.members,self.manifest)


if __name__=='__main__':unittest.main()
