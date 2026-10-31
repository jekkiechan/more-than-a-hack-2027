"""Integrity contracts for the one-time public diagnostic gate.

Every artifact is synthetic. The repeated-STARTED test proves the code stops
before invoking Git or opening the organizer reference.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments import skillcorner_expansion as study
from experiments import skillcorner_expansion_public_check as public_check


def digest(path: Path) -> str:
    return study.sha(path)


class SkillCornerPublicCheckContracts(unittest.TestCase):
    def test_seal_rejects_changed_prediction_model_and_external_decision(self):
        with tempfile.TemporaryDirectory(prefix="skillcorner-public-seal-") as temporary:
            root = Path(temporary)
            data = root / "data"
            out = data / "experiments/skillcorner_expansion_v1"
            csv_path = out / "candidate.csv"
            inputs_path = data / "challenge/inputs.npz"
            decision_path = out / "decision.json"
            baseline_path = root / "experiments/results/baseline.json"
            model_path = data / "models/candidate.joblib"
            for path, content in ((csv_path, b"frame,label\n0,H\n"),
                                  (inputs_path, b"synthetic inputs"),
                                  (decision_path, b'{"adopt_dataset":true}\n'),
                                  (baseline_path, b"synthetic baseline"),
                                  (model_path, b"synthetic checkpoint")):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            source_hashes = {"source.py": "synthetic-source-hash"}
            seal = {
                "prediction_sha256": digest(csv_path),
                "input_sha256": digest(inputs_path),
                "external_decision_sha256": digest(decision_path),
                "saved_baseline_report_sha256": digest(baseline_path),
                "source_sha256": source_hashes,
                "checkpoints": {"models/candidate.joblib": digest(model_path)},
            }

            with patch.object(study, "DATA", data), \
                 patch.object(study, "OUT", out), \
                 patch.object(study, "check"), \
                 patch.object(study, "seal"), \
                 patch.object(study, "source_hashes", return_value=source_hashes), \
                 patch.object(public_check, "OUT", out), \
                 patch.object(public_check, "DATA", data), \
                 patch.object(public_check, "CSV", csv_path), \
                 patch.object(public_check, "INPUTS", inputs_path), \
                 patch.object(public_check, "BASELINE", baseline_path):
                public_check.verify_seal(seal)
                for path, original, reason in (
                    (csv_path, b"frame,label\n0,A\n", "prediction seal changed"),
                    (model_path, b"different checkpoint", "checkpoint changed"),
                    (decision_path, b'{"adopt_dataset":false}\n', "prediction seal changed"),
                ):
                    saved = path.read_bytes()
                    path.write_bytes(original)
                    with self.subTest(changed=path.name), self.assertRaisesRegex(ValueError, reason):
                        public_check.verify_seal(seal)
                    path.write_bytes(saved)

    def test_existing_started_marker_stops_before_any_reference_fetch(self):
        with tempfile.TemporaryDirectory(prefix="skillcorner-public-started-") as temporary:
            root = Path(temporary)
            out = root / "out"
            seal_path = out / "competition_prediction_seal.json"
            started_path = out / "competition_check_started.json"
            result_path = out / "competition_check_result.json"
            seal_path.parent.mkdir(parents=True)
            seal_path.write_text("{}\n", encoding="utf-8")
            started_path.write_text('{"started":true}\n', encoding="utf-8")
            reference_path = out / "public_reference.csv"
            spec = {"reference_git_commit": "synthetic-commit"}

            with patch.object(public_check, "OUT", out), \
                 patch.object(public_check, "SEAL", seal_path), \
                 patch.object(public_check, "STARTED", started_path), \
                 patch.object(public_check, "RESULT", result_path), \
                 patch.object(public_check.study, "read", return_value=spec), \
                 patch.object(public_check, "verify_seal"), \
                 patch.object(public_check.subprocess, "run") as run, \
                 patch.object(public_check.subprocess, "check_output") as check_output:
                with self.assertRaises(FileExistsError):
                    public_check.score()
                run.assert_not_called()
                check_output.assert_not_called()
                self.assertFalse(reference_path.exists())


if __name__ == "__main__":
    unittest.main()
