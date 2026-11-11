"""Tests for append-only supplemental binding around the frozen trainer."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import frontier_recovery_execution_v1 as execution
from experiments import frontier_recovery_training_v1 as training


class SupplementalFreezeTests(unittest.TestCase):
    def test_supplement_binds_frozen_runner_and_refuses_changed_dependency(self):
        original_runner_sha = execution.sha(Path(training.__file__))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiments = root / "experiments"
            experiments.mkdir()
            copied_runner = experiments / "frontier_recovery_training_v1.py"
            copied_runner.write_bytes(Path(training.__file__).read_bytes())
            train_decl = root / "training_declaration.json"
            train_decl.write_text(json.dumps({
                "source_sha256": {
                    "experiments/frontier_recovery_training_v1.py": execution.sha(copied_runner)
                }
            }))
            extra_paths = []
            for index in range(5):
                path = experiments / f"dependency_{index}.py"
                path.write_text(f"dependency {index}\n")
                extra_paths.append(path)
            out = root / "new-output"
            supplement_path = out / "source_supplement.json"

            patches = [
                patch.object(execution, "ROOT", root),
                patch.object(execution, "SUPPLEMENT", supplement_path),
                patch.object(execution, "EXECUTION", out / "execution_receipt.json"),
                patch.object(training, "TRAIN_DECL", train_decl),
                patch.object(training, "FIT_SEAL", out / "training_seal.json"),
                patch.object(training, "_check_training_binding", lambda: {}),
                patch.object(execution, "_extra_source_paths", lambda: extra_paths),
            ]
            for item in patches:
                item.start()
            self.addCleanup(lambda: [item.stop() for item in reversed(patches)])

            frozen = execution.freeze_sources()
            self.assertEqual(frozen["training_runner_sha256"], execution.sha(copied_runner))
            self.assertEqual(len(frozen["additional_numerical_sources_sha256"]), 5)
            self.assertEqual(execution.check_sources(), frozen)

            extra_paths[2].write_text("changed after freeze\n")
            with self.assertRaisesRegex(ValueError, "changed after freeze"):
                execution.check_sources()

        self.assertEqual(execution.sha(Path(training.__file__)), original_runner_sha)

    def test_wrapper_requires_two_workers(self):
        with self.assertRaisesRegex(ValueError, "two-worker"):
            execution.run(workers=1)


if __name__ == "__main__":
    unittest.main()
