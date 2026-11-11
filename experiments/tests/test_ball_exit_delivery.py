from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import ball_exit_delivery as delivery
from experiments import current_baseline


class DeliveryContracts(unittest.TestCase):
    def test_missing_checkpoints_block_stage_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp) / "parent.json"
            parent.write_text('{}')
            staged = Path(tmp) / "staged.json"
            with patch.multiple(delivery, PARENT=parent, ACTIVE=parent, STAGED=staged), \
                    patch.object(current_baseline, "load_models", side_effect=FileNotFoundError("ce_2026.joblib")):
                with self.assertRaises(FileNotFoundError):
                    delivery.stage()
            self.assertFalse(staged.exists())

    def test_public_gate_is_claimed_exclusively(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gate.json"
            delivery.write_once(path, {"first": True})
            saved = path.read_bytes()
            with self.assertRaises(FileExistsError):
                delivery.write_once(path, {"second": True})
            self.assertEqual(path.read_bytes(), saved)

    def test_missing_reports_block_activation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.multiple(delivery, GAR=Path(tmp)/"gar.json", PUBLIC=Path(tmp)/"public.json"), \
                    patch.object(delivery, "check_manifest", return_value={}), \
                    patch.object(delivery, "check_seal", return_value={}):
                with self.assertRaises(FileNotFoundError):
                    delivery.activate()

    def test_probability_api_cannot_bypass_corrected_mask(self):
        with self.assertRaises(ValueError):
            current_baseline.predict_probabilities({}, {}, {"architecture": {"kind": delivery.runtime.KIND}})

    def test_hash_verification_fails_before_public_gate(self):
        with patch.object(delivery, "check_seal", side_effect=ValueError("changed CSV")), \
                patch.object(delivery, "write_once") as writer:
            with self.assertRaises(ValueError):
                delivery.public()
            writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
