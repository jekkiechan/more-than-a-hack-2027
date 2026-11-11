"""Contract tests for the fresh-process recovery execution amendment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import frontier_recovery_execution_v1 as parallel
from experiments import frontier_recovery_serial_v1 as serial
from experiments import frontier_recovery_training_v1 as training


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SerialRecoveryContracts(unittest.TestCase):
    def test_fixed_stage_order_contains_twelve_distinct_jobs(self):
        jobs = serial._jobs()
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len(set(jobs)), 12)
        self.assertEqual(jobs[:6], [(512, fold, seed)
                         for fold in training.FOLDS for seed in training.SEEDS])
        self.assertEqual(jobs[6:], [(1024, fold, seed)
                         for fold in training.FOLDS for seed in training.SEEDS])

    def test_existing_output_requires_complete_matching_pair(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "out"
            declaration = root / "training_declaration.json"
            declaration.write_text("{}")
            with patch.object(training, "ROOT", root), \
                 patch.object(training, "OUT", output), \
                 patch.object(training, "TRAIN_DECL", declaration):
                job = serial._jobs()[0]
                model_path = training._stage_path(*job)
                model_path.parent.mkdir(parents=True)
                model_path.write_bytes(b"model")
                row = {"study_id": training.STUDY, "stage": job[0], "fold": job[1],
                       "seed": job[2], "source_status": "reconstructed",
                       "declaration_sha256": file_sha(declaration),
                       "sha256": file_sha(model_path)}
                Path(str(model_path) + ".json").write_text(json.dumps(row))
                self.assertEqual(serial._verified_existing(job), row)
                Path(str(model_path) + ".json").unlink()
                with self.assertRaises(FileExistsError):
                    serial._verified_existing(job)

    def test_source_amendment_rejects_dependency_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            experiments = root / "experiments"
            experiments.mkdir()
            runner = experiments / "frontier_recovery_training_v1.py"
            runner.write_text("frozen runner\n")
            declaration = root / "training_declaration.json"
            declared_sources = {"experiments/frontier_recovery_training_v1.py": file_sha(runner)}
            declaration.write_text(json.dumps({"source_sha256": declared_sources}))
            extra = []
            for index in range(5):
                path = experiments / f"extra_{index}.py"
                path.write_text(f"extra {index}\n")
                extra.append(path)
            supplement = root / "source_supplement.json"
            supplement_value = {
                "training_declaration_sha256": file_sha(declaration),
                "execution_wrapper_sha256": file_sha(Path(parallel.__file__)),
                "training_runner_sha256": file_sha(runner),
                "additional_numerical_sources_sha256": {
                    str(path.relative_to(root)): file_sha(path) for path in extra
                },
            }
            supplement.write_text(json.dumps(supplement_value))
            prior_log = root / "failed.log"
            prior_log.write_text("OOM\n")
            amendment = root / "serial_amendment.json"
            amendment.write_text(json.dumps({
                "serial_execution_wrapper_sha256": file_sha(Path(serial.__file__)),
                "training_declaration_sha256": file_sha(declaration),
                "supplement_sha256": file_sha(supplement),
                "parallel_execution_wrapper_sha256": file_sha(Path(parallel.__file__)),
                "parallel_failure": {"log_sha256": file_sha(prior_log)},
            }))
            with patch.object(serial, "ROOT", root), \
                 patch.object(serial, "AMENDMENT", amendment), \
                 patch.object(serial, "PRIOR_FAILED_LOG", prior_log), \
                 patch.object(training, "TRAIN_DECL", declaration), \
                 patch.object(training, "_training_source_hashes", lambda: declared_sources), \
                 patch.object(parallel, "SUPPLEMENT", supplement), \
                 patch.object(parallel, "_extra_source_paths", lambda: extra):
                serial._check_child_sources()
                extra[1].write_text("changed\n")
                with self.assertRaisesRegex(ValueError, "dependencies changed"):
                    serial._check_child_sources()

    def test_run_uses_fresh_process_order_and_resumes_only_verified_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "data/experiments/frontier_recovery_v1"
            declaration = root / "training_declaration.json"
            declaration.write_text("frozen declaration")
            amendment = root / "serial_amendment.json"
            amendment.write_text("frozen amendment")
            supplement = root / "source_supplement.json"
            supplement.write_text("frozen source supplement")
            log = root / "prior-parallel.log"
            log.write_text("failed parallel run\n")
            serial_log = root / "serial.log"
            execution_receipt = output / "execution_receipt.json"
            fit_seal = output / "training_seal.json"
            calls = []
            fail_once = {"enabled": True}

            def fake_subprocess(command, *, cwd, capture_output, text):
                self.assertEqual(command[:3], [serial.sys.executable, "-m",
                                              "experiments.frontier_recovery_serial_v1"])
                self.assertEqual(command[3], "fit-one")
                job = (int(command[4]), command[5], int(command[6]))
                calls.append(job)
                if fail_once["enabled"] and len(calls) == 3:
                    return type("Process", (), {"returncode": 1, "stdout": "", "stderr": "failed"})()
                stage, fold, seed = job
                model = training._stage_path(stage, fold, seed)
                model.parent.mkdir(parents=True, exist_ok=True)
                model.write_bytes(f"model {job}".encode())
                row = {"study_id": training.STUDY, "stage": stage, "fold": fold,
                       "seed": seed, "source_status": "reconstructed",
                       "declaration_sha256": file_sha(declaration), "sha256": file_sha(model)}
                Path(str(model) + ".json").write_text(json.dumps(row))
                return type("Process", (), {"returncode": 0, "stdout": "ok\n", "stderr": ""})()

            bindings = [
                patch.object(serial, "ROOT", root),
                patch.object(serial, "OUT", output),
                patch.object(serial, "AMENDMENT", amendment),
                patch.object(serial, "SERIAL_LOG", serial_log),
                patch.object(serial, "PRIOR_FAILED_LOG", log),
                patch.object(training, "ROOT", root),
                patch.object(training, "OUT", output),
                patch.object(training, "TRAIN_DECL", declaration),
                patch.object(training, "FIT_SEAL", fit_seal),
                patch.object(parallel, "SUPPLEMENT", supplement),
                patch.object(parallel, "EXECUTION", execution_receipt),
                patch.object(training, "_check_training_binding", lambda: {}),
                patch.object(parallel, "check_sources", lambda: {}),
                patch.object(serial, "_validate_frozen", lambda: {}),
                patch.object(serial, "_check_child_sources", lambda: None),
                patch.object(serial.subprocess, "run", fake_subprocess),
            ]
            for binding in bindings:
                binding.start()
            self.addCleanup(lambda: [binding.stop() for binding in reversed(bindings)])

            with self.assertRaisesRegex(RuntimeError, "Fresh-process fit failed"):
                serial._run()
            first_completed = [call for call in calls if serial._verified_existing(call)]
            self.assertEqual(first_completed, serial._jobs()[:2])
            previous_hashes = {job: file_sha(training._stage_path(*job))
                               for job in first_completed}
            fail_once["enabled"] = False
            calls.clear()
            result = serial._run()
            self.assertEqual(calls, serial._jobs()[2:])
            self.assertEqual(result["fit_count"], 12)
            self.assertEqual(result["source_supplement_sha256"], file_sha(supplement))
            self.assertTrue(result["all_provider_folds_complete"])
            self.assertEqual(len(json.loads(fit_seal.read_text())["fits"]), 12)
            for job, digest in previous_hashes.items():
                self.assertEqual(file_sha(training._stage_path(*job)), digest)


if __name__ == "__main__":
    unittest.main()
