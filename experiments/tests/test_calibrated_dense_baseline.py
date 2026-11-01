import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import calibrated_dense_baseline as inference, current_baseline as baseline
from experiments import promote_calibrated_dense_baseline as promotion
from experiments import feed_spell_push


class CalibratedDenseContracts(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(160).random((2, 123, 3))
        self.x /= self.x.sum(axis=2, keepdims=True)
        self.inputs = dict(ball=np.zeros((123, 2)), period=np.r_[np.ones(53), np.full(70, 2)],
                           timestamps=np.arange(123)/25, fps=25.)
        self.manifest = dict(architecture=dict(kind=inference.KIND, postprocessing=inference.POSTPROCESSING))

    def test_deployment_uses_exact_frozen_decoder_and_preserves_members(self):
        original = self.x.copy()
        members = {f'dense_{seed}': self.x[i] for i, seed in enumerate((2026, 2027))}
        with mock.patch.object(inference.sn_dense_baseline, 'member_probabilities', return_value=members):
            guess = inference.predict_labels(self.inputs, {}, self.manifest)
            probability = inference.predict_probabilities(self.inputs, {}, self.manifest)
        expected = feed_spell_push.decode(self.x, baseline.supervised.model_periods(self.inputs), 25, inference.CONFIG)
        np.testing.assert_array_equal(guess, expected)
        np.testing.assert_allclose(probability.sum(axis=1), 1, atol=1e-15)
        np.testing.assert_array_equal(self.x, original)

    def test_rejects_unselected_or_missing_temperature(self):
        for settings in ({}, dict(inference.POSTPROCESSING, temperature=3), dict(inference.POSTPROCESSING, ha_switch_cost=6)):
            with self.assertRaisesRegex(ValueError, 'f062'):
                inference.predict_labels(self.inputs, {}, dict(architecture=dict(kind=inference.KIND, postprocessing=settings)))

    def test_probability_dispatch_keeps_predecessor_route(self):
        with mock.patch.object(inference, 'predict_probabilities', return_value='calibrated') as new, \
             mock.patch.object(inference.sn_dense_baseline, 'predict_probabilities', return_value='original') as old:
            self.assertEqual(baseline.predict_probabilities({}, {}, self.manifest), 'calibrated')
            self.assertEqual(baseline.predict_probabilities({}, {}, {'architecture': {'kind': 'sn_dense_ensemble'}}), 'original')
        new.assert_called_once(); old.assert_called_once()

    def test_current_baseline_cli_uses_frozen_path_once_and_masks_only_inactive_periods(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); manifest = root/'manifest.json'; output = root/'candidate.csv'
            manifest.write_text(json.dumps(dict(self.manifest, model=promotion.MODEL)))
            inputs = dict(ball=np.zeros((4,2)), period=np.array([0,1,1,2]), fps=25.)
            inputs_path = root/'synthetic_inputs.npz'
            np.savez(inputs_path, **inputs)
            def write(labels, target):
                np.testing.assert_array_equal(labels, np.array(list('XHAD')))
                target.write_text('fixture')
            with mock.patch('sys.argv', ['baseline','predict','--manifest',str(manifest),'--inputs',str(inputs_path),'--output',str(output)]), \
                 mock.patch.object(baseline,'load_inputs',return_value=inputs), \
                 mock.patch.object(baseline,'load_models',return_value={}), \
                 mock.patch.object(inference,'predict_labels',return_value=np.array(list('DHAD'))) as decode, \
                 mock.patch.object(baseline,'predict_probabilities',side_effect=AssertionError('double inference')), \
                 mock.patch.object(baseline.challenge,'write_submission',side_effect=write):
                baseline.main()
            decode.assert_called_once()
            receipt=json.loads(output.with_suffix('.json').read_text())
            self.assertFalse(receipt['target_labels_read']); self.assertFalse(receipt['organizer_scored'])

    def test_policy_archives_exception_without_rewriting_predecessor(self):
        with tempfile.TemporaryDirectory() as temp:
            staged=Path(temp)/'manifest.json';staged.write_text('fixture')
            previous=dict(stress_guardrail={'reference_recipe':'old'},clean_improvement_screen={'current_adoption_exception':'old failed screen'})
            original=json.loads(json.dumps(previous))
            with mock.patch.object(promotion,'STAGED',staged):
                policy=promotion.updated_policy(previous,dict(promoted_utc='2026-11-01T00:00:00Z',status='adopted'))
            self.assertEqual(previous,original)
            self.assertTrue(policy['clean_improvement_screen']['current_screen_passes'])
            self.assertIsNone(policy['clean_improvement_screen']['current_adoption_exception'])
            self.assertEqual(policy['working_candidate']['model'],promotion.MODEL)
            self.assertEqual(policy['primary_evaluation_view'],'feed_like')

    def test_existing_preserved_metadata_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'history.json';promotion.preserve(path,b'old')
            with self.assertRaisesRegex(ValueError,'Refusing to replace'):
                promotion.preserve(path,b'new')
            self.assertEqual(path.read_bytes(),b'old')
