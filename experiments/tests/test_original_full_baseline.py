import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import current_baseline, original_full_baseline as inference
from experiments import feed_spell_push, calibrated_dense_baseline
from experiments import promote_original_full_baseline as promotion


class OriginalFullContracts(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(630)
        self.members = {f'dense_{s}': rng.dirichlet([2, 2, 1], 45) * scale
                        for s, scale in ((2026, .9999999), (2027, 1.0000001))}
        self.inputs = dict(ball=np.zeros((45, 2)), period=np.r_[np.ones(20), np.full(25, 2)],
                           timestamps=np.arange(45)/25, fps=25.)
        self.manifest = dict(architecture=dict(kind=inference.KIND,
                                               postprocessing=inference.POSTPROCESSING))

    def test_pool_before_normalization_matches_sealed_public_path(self):
        expected = np.mean(list(self.members.values()), axis=0)
        source = np.stack(list(self.members.values())).copy()
        with mock.patch.object(inference.sn_dense_baseline, 'member_probabilities', return_value=self.members):
            native = inference.native_probability(self.inputs, {}, self.manifest)
            labels = inference.predict_labels(self.inputs, {}, self.manifest)
        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(labels, feed_spell_push.decode(expected[None],
            current_baseline.supervised.model_periods(self.inputs), 25., calibrated_dense_baseline.CONFIG))
        np.testing.assert_array_equal(source, np.stack(list(self.members.values())))

    def test_targets_are_stripped_before_feature_inference(self):
        inputs = dict(self.inputs, labels=np.full(45, 'H'), ball_status=np.ones(45))
        with mock.patch.object(inference.sn_dense_baseline, 'member_probabilities', return_value=self.members) as predict:
            inference.native_probability(inputs, {}, self.manifest)
        self.assertNotIn('labels', predict.call_args.args[0])
        self.assertNotIn('ball_status', predict.call_args.args[0])

    def test_rejects_changed_decoder_or_normalization_order(self):
        for processing in ({}, dict(inference.POSTPROCESSING, temperature=3),
                           dict(calibrated_dense_baseline.POSTPROCESSING)):
            with self.assertRaisesRegex(ValueError, 'sealed original_full'):
                inference.native_probability(self.inputs, {}, dict(architecture=dict(
                    kind=inference.KIND, postprocessing=processing)))

    def test_cli_dispatch_once_preserves_inactive_frame_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); manifest = root/'manifest.json'; output = root/'out.csv'
            manifest.write_text(json.dumps(dict(self.manifest, model=promotion.MODEL)))
            inputs = dict(ball=np.zeros((4, 2)), period=np.array([0, 1, 1, 2]), fps=25.)
            path = root/'inputs.npz'; np.savez(path, **inputs)
            def write(labels, target):
                np.testing.assert_array_equal(labels, np.array(list('XHAD')))
                target.write_text('fixture')
            with mock.patch('sys.argv', ['baseline', 'predict', '--manifest', str(manifest),
                            '--inputs', str(path), '--output', str(output)]), \
                 mock.patch.object(current_baseline, 'load_inputs', return_value=inputs), \
                 mock.patch.object(current_baseline, 'load_models', return_value={}), \
                 mock.patch.object(inference, 'predict_labels', return_value=np.array(list('DHAD'))) as predict, \
                 mock.patch.object(current_baseline, 'predict_probabilities', side_effect=AssertionError('double inference')), \
                 mock.patch.object(current_baseline.challenge, 'write_submission', side_effect=write):
                current_baseline.main()
            predict.assert_called_once()
            receipt = json.loads(output.with_suffix('.json').read_text())
            self.assertFalse(receipt['target_labels_read']); self.assertFalse(receipt['organizer_scored'])

    def test_adoption_exception_preserves_failed_screen_and_predecessor(self):
        with tempfile.TemporaryDirectory() as folder:
            staged = Path(folder)/'manifest.json'; staged.write_text('fixed')
            previous = dict(stress_guardrail={'reference_recipe': 'old'},
                            clean_improvement_screen={'current_screen_passes': True})
            original = json.loads(json.dumps(previous))
            with mock.patch.object(promotion, 'STAGED', staged):
                policy = promotion.updated_policy(previous, dict(promoted_utc='2026-11-01', status='adopted'))
            self.assertEqual(original, previous)
            self.assertFalse(policy['clean_improvement_screen']['current_screen_passes'])
            self.assertIn('Explicit user adoption', policy['clean_improvement_screen']['current_adoption_exception'])
            self.assertEqual(policy['working_candidate']['model'], promotion.MODEL)

    def test_checkpoint_tamper_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root/'dense_2026.joblib').write_bytes(b'tampered')
            manifest = root/'manifest.json'; manifest.write_text(json.dumps(dict(
                model_directory=str(root), models_sha256={'dense_2026.joblib': 'sealed-hash'})))
            with mock.patch.object(current_baseline.joblib, 'load', side_effect=AssertionError('unverified load')):
                with self.assertRaisesRegex(ValueError, 'model hash mismatch'):
                    current_baseline.load_models(root, manifest)

    def test_preserve_refuses_different_existing_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'gate.json'; promotion.preserve(path, b'closed')
            with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
                promotion.preserve(path, b'reopen')
            self.assertEqual(path.read_bytes(), b'closed')
