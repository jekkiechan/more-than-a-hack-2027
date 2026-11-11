import json
from pathlib import Path
import tempfile
import unittest

from experiments import frontier_pooled_reconstruction as study
from experiments import v2_iterative_train as iterative


class FrontierPooledReconstructionContracts(unittest.TestCase):
    def test_json_roundtrip_stage_lookup_uses_string_keys(self):
        value = {
            "original_checkpoint_sha256": {
                "2026": {"512": "stage512", "1024": "stage1024"},
                "2027": {"512": "stage512b", "1024": "stage1024b"},
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "declaration.json"
            path.write_text(json.dumps(value))
            loaded = json.loads(path.read_text())
        self.assertEqual(study.original_checkpoint_sha(loaded, 2026, 1024), "stage1024")
        self.assertEqual(study.original_checkpoint_sha(loaded, 2027, 512), "stage512b")

    def test_fixed_output_paths_and_budget(self):
        self.assertEqual(study.STUDY, "frontier_pooled_reconstruction_v1")
        self.assertEqual(study.output_path(2026, 512).name, "stage512_2026.joblib")
        self.assertEqual(study.output_path(2027, 1024).name, "stage1024_2027.joblib")
        with self.assertRaises(ValueError):
            study.output_path(2025, 512)
        with self.assertRaises(ValueError):
            study.output_path(2026, 2048)

    def test_trainer_scope_restores_shared_globals(self):
        old_lr, old_study = iterative.LR, iterative.STUDY
        with study._trainer_scope(1e-4):
            self.assertEqual(iterative.LR, 1e-4)
            self.assertEqual(iterative.STUDY, study.STUDY)
        self.assertEqual(iterative.LR, old_lr)
        self.assertEqual(iterative.STUDY, old_study)


if __name__ == "__main__":
    unittest.main()
