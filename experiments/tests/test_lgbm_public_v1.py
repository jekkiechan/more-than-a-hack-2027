"""Gate integrity tests for the pooled public diagnostic.

All artifacts and the reference path are temporary. The scorer is mocked, so
these tests never read or score the organizer reference labels.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments import lgbm_public_v1 as study


class PublicGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        out = root / "data" / "experiments" / study.STUDY
        results = root / "experiments" / "results"
        reference = root / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
        paths = {
            "ROOT": root,
            "OUT": out,
            "RESULTS": results,
            "DECL": results / "declaration.json",
            "PREP": out / "preparation.json",
            "FIT": results / "fit.json",
            "MODEL": out / "refiner.txt",
            "CSV": out / "candidate.csv",
            "SEAL": results / "seal.json",
            "GATE": results / "gate.json",
            "RESULT": results / "public.json",
            "BASELINE": results / "baseline.json",
            "MANIFEST": root / "experiments" / "manifest.json",
            "INPUT": root / "data" / "inputs.npz",
            "reference": reference,
        }
        attributes = {
            "ROOT": "root", "OUT": "out", "DECL": "decl", "PREP": "prep",
            "FIT": "fit", "MODEL": "model", "CSV": "csv", "SEAL": "seal_path",
            "GATE": "gate_path", "RESULT": "result_path", "BASELINE": "baseline_path",
            "MANIFEST": "manifest", "INPUT": "input",
        }
        for key, attribute in attributes.items():
            setattr(self, attribute, paths[key])
        self.reference = reference

        artifacts = {
            self.decl: b"declaration",
            self.prep: b"preparation",
            self.fit: b"fit",
            self.model: b"model",
            self.csv: b"H\nA\n",
            self.manifest: b"manifest",
            self.input: b"input",
            self.out / "baseline_replay.csv": b"exact baseline replay",
            self.reference: b"synthetic reference placeholder",
        }
        for path, content in artifacts.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.baseline = {"candidate": {"spell_f1_mean": 0.6, "macro_f1": 0.8, "d_f1": 0.7}}
        self.baseline_path.write_text(json.dumps(self.baseline))
        self.seal = {
            "prediction_sha256": study.sha(self.csv),
            "fit_sha256": study.sha(self.fit),
            "model_sha256": study.sha(self.model),
            "input_sha256": study.sha(self.input),
            "baseline_manifest_sha256": study.sha(self.manifest),
            "declaration_sha256": study.sha(self.decl),
            "preparation_sha256": study.sha(self.prep),
            "source_sha256": {"test": "fixed"},
            "baseline_replay_sha256": study.sha(self.out / "baseline_replay.csv"),
        }
        self.seal_path.write_text(json.dumps(self.seal))
        self.declaration = {
            "authorization": "test authorization",
            "expected_control_sha256": study.sha(self.out / "baseline_replay.csv"),
            "saved_baseline_sha256": study.sha(self.baseline_path),
            # The runner binds this hash in the public declaration.
            "expected_reference_sha256": study.sha(self.reference),
        }
        self.declaration_patcher = mock.patch.object(study, "check_declaration", return_value=self.declaration)
        self.source_patcher = mock.patch.object(study, "sources", return_value=self.seal["source_sha256"])
        self.sha_patcher = mock.patch.object(study, "sha", wraps=study.sha)
        self.declaration_patcher.start()
        self.source_patcher.start()
        self.sha_patcher.start()
        self.addCleanup(self.declaration_patcher.stop)
        self.addCleanup(self.source_patcher.stop)
        self.addCleanup(self.sha_patcher.stop)

    def _patch_public_scoring(self, read_label_csv):
        from evaluation import possession_metrics as pm
        score = {"n_frames": 199319, "n_graded": 7500, "n_gt_spells": 17}
        return (
            mock.patch.object(pm, "read_label_csv", side_effect=read_label_csv),
            mock.patch.object(pm, "label_codes", side_effect=lambda labels: labels),
            mock.patch.object(pm, "evaluate", return_value=score),
            mock.patch("experiments.combined_objective_public_check._score_summary",
                       return_value={"spell_f1_mean": 0.61, "macro_f1": 0.81, "d_f1": 0.71}),
        )

    def test_gate_is_claimed_before_reference_or_prediction_csv_is_read(self):
        def read_label_csv(path):
            path = Path(path)
            self.assertTrue(self.gate_path.exists(), "gate must be claimed before label CSV access")
            self.assertIn(path, (self.reference, self.csv))
            return ["synthetic"]

        patches = self._patch_public_scoring(read_label_csv)
        with patches[0], patches[1], patches[2], patches[3], self._patched_paths():
            study.public()
        self.assertTrue(self.gate_path.exists())
        result = json.loads(self.result_path.read_text())
        self.assertFalse(result["adopted"])
        self.assertFalse(result["selection_or_tuning"])

    def test_replay_mismatch_is_refused_before_gate_claim(self):
        self.seal["baseline_replay_sha256"] = "wrong"
        self.seal_path.write_text(json.dumps(self.seal))
        with self._patched_paths():
            with self.assertRaisesRegex(ValueError, "Control changed"):
                study.public()
        self.assertFalse(self.gate_path.exists())

    def test_saved_baseline_change_is_refused_before_gate_claim(self):
        self.baseline_path.write_text(json.dumps({"candidate": {"spell_f1_mean": 0.9}}))
        with self._patched_paths():
            with self.assertRaisesRegex(ValueError, "Control changed"):
                study.public()
        self.assertFalse(self.gate_path.exists())

    def test_changed_sealed_prediction_fails_before_gate_claim(self):
        self.csv.write_text("tampered\n")
        with self._patched_paths():
            with self.assertRaisesRegex(ValueError, "Sealed evidence changed"):
                study.public()
        self.assertFalse(self.gate_path.exists())

    def test_changed_reference_is_refused_after_gate_and_before_label_read(self):
        self.declaration["expected_reference_sha256"] = "wrong"
        from evaluation import possession_metrics as pm
        with self._patched_paths(), mock.patch.object(
            pm, "read_label_csv", side_effect=AssertionError("reference reader must not run")
        ):
            with self.assertRaisesRegex(ValueError, "Public reference changed"):
                study.public()
        self.assertTrue(self.gate_path.exists())

    def _patched_paths(self):
        stack = mock.patch.multiple(
            study, ROOT=self.root, GATE=self.gate_path, RESULT=self.result_path,
            SEAL=self.seal_path, CSV=self.csv, FIT=self.fit, MODEL=self.model,
            INPUT=self.input, MANIFEST=self.manifest, DECL=self.decl, PREP=self.prep,
            BASELINE=self.baseline_path,
        )
        return stack


if __name__ == "__main__":
    unittest.main()
