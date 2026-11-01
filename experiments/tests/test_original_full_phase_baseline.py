import unittest
from unittest import mock

import numpy as np

from experiments import original_full_phase_baseline as inference, original_full_baseline, current_baseline


class PhaseDeploymentContracts(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(ball=np.zeros((12, 2)), period=np.ones(12), timestamps=np.arange(12)/25, fps=25.)
        self.manifest = dict(architecture=dict(kind=inference.KIND, phase_tta=True,
            phase_averaging=inference.PHASE_AVERAGING, postprocessing=original_full_baseline.POSTPROCESSING))
        self.phases = [np.tile(p, (12, 1)) for p in
            [[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.4,.3,.3],[.2,.5,.3]]]

    def test_equal_phase_mean_and_label_free_dispatch(self):
        inputs = dict(self.inputs, labels=np.full(12,'H'), ball_status=np.ones(12))
        with mock.patch.object(inference.feed_like_v2_public,'phase_native',return_value=self.phases) as call:
            result = inference.native_probability(inputs, {}, self.manifest)
        np.testing.assert_array_equal(result, np.mean(self.phases, axis=0))
        self.assertNotIn('labels', call.call_args.args[0])
        self.assertNotIn('ball_status', call.call_args.args[0])
        with mock.patch.object(inference,'predict_probabilities',return_value=result) as predict:
            np.testing.assert_array_equal(current_baseline.predict_probabilities(self.inputs,{},self.manifest),result)
        predict.assert_called_once()

    def test_rejects_partial_phases_and_non_native_feed(self):
        with mock.patch.object(inference.feed_like_v2_public,'phase_native',return_value=self.phases[:4]):
            with self.assertRaisesRegex(ValueError,'five sampling'):
                inference.native_probability(self.inputs,{},self.manifest)
        with self.assertRaisesRegex(ValueError,'25 Hz'):
            inference.native_probability(dict(self.inputs,fps=30.),{},self.manifest)

    def test_refuses_decoder_or_averaging_drift(self):
        for update in [dict(phase_tta=False),dict(phase_averaging={}),dict(postprocessing={})]:
            with self.assertRaisesRegex(ValueError,'sealed original_full'):
                inference.native_probability(self.inputs,{},dict(architecture=dict(self.manifest['architecture'],**update)))

    def test_phase0_package_keeps_its_own_dispatch(self):
        with mock.patch.object(original_full_baseline,'predict_probabilities',return_value='phase0'):
            self.assertEqual(current_baseline.predict_probabilities({}, {}, dict(architecture=dict(kind=original_full_baseline.KIND))), 'phase0')
