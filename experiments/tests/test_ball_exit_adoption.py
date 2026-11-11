from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import ball_exit_adoption as adoption
from experiments import ball_exit_delivery as delivery
from experiments.ball_exit_followup import sha


class ProvisionalAdoptionContracts(unittest.TestCase):
    def test_inventory_distinguishes_absent_and_corrupt_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'valid.joblib').write_bytes(b'exact')
            manifest = {'model_directory': '.', 'models_sha256': {
                'valid.joblib': sha(root / 'valid.joblib'), 'missing.joblib': 'a' * 64}}
            with patch.object(adoption, 'ROOT', root):
                result = adoption.checkpoint_inventory(manifest)
                self.assertEqual(list(result['verified']), ['valid.joblib'])
                self.assertEqual(list(result['missing']), ['missing.joblib'])
                (root / 'valid.joblib').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Checkpoint hash mismatch'):
                    adoption.checkpoint_inventory(manifest)

    def test_adoption_never_replaces_a_different_active_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, active = root / 'parent', root / 'active'
            parent.write_text('old'); active.write_text('new')
            with patch.multiple(delivery, PARENT=parent, ACTIVE=active):
                with self.assertRaisesRegex(ValueError, 'Active predecessor changed'):
                    adoption.adopt()
            self.assertEqual(active.read_text(), 'new')

    def test_pending_weights_still_block_diagnostic_staging(self):
        with patch.object(delivery, 'read', side_effect=[{'manifest_sha256': 'hash'},
                {'source_sha256': {}}]), \
                patch.object(adoption, 'sha', return_value='hash'), \
                patch.object(adoption, 'VERSIONED') as versioned, \
                patch.object(delivery, 'ACTIVE') as active, \
                patch.object(delivery.runtime, 'frontier_manifest'), \
                patch.object(delivery.current_baseline, 'load_models',
                    side_effect=FileNotFoundError('ce_2026.joblib')), \
                patch.object(adoption, 'write_once') as writer:
            active.read_bytes.return_value = versioned.read_bytes.return_value = b'same'
            with self.assertRaises(FileNotFoundError):
                adoption.stage()
            writer.assert_not_called()


if __name__ == '__main__':
    unittest.main()
