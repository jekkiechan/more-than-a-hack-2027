"""Synthetic contracts for the bounded public-score gate; no organizer assets."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import combined_objective_public_check as check


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("frame", "label"))
        for index, label in enumerate(labels):
            writer.writerow((index, label))


class CombinedObjectivePublicCheckContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="combined-public-check-")
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.out = self.data / "experiments/combined_objective_public_check_v1"
        self.combined_csv = self.data / "experiments/current_baseline/joint_gru_combined_ensemble.csv"
        self.preceding_csv = self.data / "experiments/current_baseline/preceding_seed_ensemble_replay.csv"
        self.receipt = self.combined_csv.with_suffix(".json")
        self.prediction_declaration = self.combined_csv.with_suffix(".declaration.json")
        self.inputs = self.data / "challenge/inputs.npz"
        self.manifest = self.root / "experiments/baseline_manifest.json"
        self.policy = self.root / "experiments/model_decision_policy.json"
        self.promotion = self.root / "experiments/results/baseline_combined_objective_promotion.json"
        self.comparison = self.root / "experiments/results/objective_ensemble.json"
        self.comparison_verification = self.root / "experiments/results/objective_ensemble_verification.json"
        self.comparison_runs = self.data / "experiments/objective_ensemble_v1/runs.json"
        self.history = self.root / "experiments/results/current_baseline_public_check.json"
        self.reference = self.root / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
        self.public_report = self.root / "experiments/results/combined_objective_public_check.json"
        self.private_report = self.out / "public_check_result.json"
        self.started = self.out / "public_check_started.json"
        self.predeclared = self.out / "public_check_predeclared.json"

        self.model_dir = self.data / "models/joint_gru_combined_ensemble_v1"
        self.model_dir.mkdir(parents=True)
        self.model_hashes = {}
        for name, value in (("joint_gru_2026.joblib", b"seed 2026"),
                            ("joint_gru_2027.joblib", b"seed 2027")):
            model_path = self.model_dir / name
            model_path.write_bytes(value)
            self.model_hashes[name] = digest(model_path)

        for relative in check.SOURCE_RELATIVE:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"synthetic source {relative}\n", encoding="utf-8")

        self.inputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez(self.inputs, period=np.array([0, 1, 1, 0], dtype=np.int16))
        write_csv(self.combined_csv, ["X", "H", "D", "X"])
        write_csv(self.preceding_csv, ["X", "A", "D", "X"])

        manifest = {
            "model": "joint_gru_combined_ensemble",
            "package_version": "joint_gru_combined_ensemble_v1",
            "model_directory": "data/models/joint_gru_combined_ensemble_v1",
            "models_sha256": self.model_hashes,
            "organizer_public_diagnostic": {
                "report": "experiments/results/combined_objective_public_check.json",
                "known_at_promotion": False,
                "used_for_training_tuning_selection_or_adoption": False,
                "status": "Separately requested diagnostic; completion and scores live in the report",
            },
        }
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode()
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.manifest.write_bytes(manifest_bytes)
        (self.model_dir / "baseline_manifest.json").write_bytes(manifest_bytes)
        self.policy.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.policy, {"working_candidate": {
            "model": manifest["model"],
            "baseline_manifest_sha256": digest(self.manifest),
        }})

        self.old_manifest_sha = "c" * 64
        write_json(self.comparison, {"study_id": "objective_ensemble_v1"})
        write_json(self.comparison_runs, {"rows": []})
        write_json(self.comparison_verification, {
            "study_id": "objective_ensemble_v1",
            "inputs": {"report_sha256": digest(self.comparison),
                       "runs_sha256": digest(self.comparison_runs)},
        })
        write_json(self.promotion, {
            "baseline_manifest_sha256": digest(self.manifest),
            "previous_manifest_sha256": self.old_manifest_sha,
            "models_sha256": self.model_hashes,
            "model_decision_policy_sha256": digest(self.policy),
            "evidence": {"comparison": {
                "report_sha256": digest(self.comparison),
                "independent_verification_sha256": digest(self.comparison_verification),
            }},
            "organizer_scored": False,
            "organizer_labels_read": False,
        })
        # The synthetic reference is present, but freeze must not open it.
        write_csv(self.reference, ["X", "H", "D", "X"])
        self.reference_sha = digest(self.reference)
        write_json(self.history, {"organizer_gt_sha256": self.reference_sha})

        source_map = {relative: digest(self.root / relative) for relative in check.SOURCE_RELATIVE}
        write_json(self.prediction_declaration, {
            "input_sha256": digest(self.inputs),
            "current_manifest_sha256": self.old_manifest_sha,
            "model_sha256": self.model_hashes,
            "sources_sha256": source_map,
            "organizer_reference_labels_read": False,
            "organizer_scored": False,
        })
        write_json(self.receipt, {
            "declaration_sha256": digest(self.prediction_declaration),
            "input_sha256": digest(self.inputs),
            "output_sha256": digest(self.combined_csv),
            "rows": 4,
            "model_sha256": self.model_hashes,
            "member_probability_sha256": {
                "joint_gru_2026": "e" * 64,
                "joint_gru_2027": "f" * 64,
            },
            "blended_probability_sha256": "1" * 64,
            "current_baseline_replay_sha256": digest(self.preceding_csv),
            "current_baseline_replay_path": "data/experiments/current_baseline/preceding_seed_ensemble_replay.csv",
            "format_validation": {"valid": True, "rows": 4},
            "organizer_reference_labels_read": False,
            "organizer_scored": False,
            "organizer_score": None,
        })

        self.patchers = [
            patch.object(check, "ROOT", self.root),
            patch.object(check, "DATA", self.data),
            patch.object(check, "OUT", self.out),
            patch.object(check, "PREDECLARED", self.predeclared),
            patch.object(check, "STARTED", self.started),
            patch.object(check, "PRIVATE_REPORT", self.private_report),
            patch.object(check, "PUBLIC_REPORT", self.public_report),
            patch.object(check, "COMBINED_CSV", self.combined_csv),
            patch.object(check, "PRECEDING_CSV", self.preceding_csv),
            patch.object(check, "PREDICTION_RECEIPT", self.receipt),
            patch.object(check, "PREDICTION_DECLARATION", self.prediction_declaration),
            patch.object(check, "INPUTS", self.inputs),
            patch.object(check, "BASELINE_MANIFEST", self.manifest),
            patch.object(check, "DECISION_POLICY", self.policy),
            patch.object(check, "PROMOTION_RECEIPT", self.promotion),
            patch.object(check, "COMPARISON_REPORT", self.comparison),
            patch.object(check, "COMPARISON_VERIFICATION", self.comparison_verification),
            patch.object(check, "COMPARISON_RUNS", self.comparison_runs),
            patch.object(check, "HISTORICAL_PUBLIC_REPORT", self.history),
            patch.object(check, "REFERENCE_CSV", self.reference),
            patch.object(check, "EXPECTED_INPUT_SHA256", digest(self.inputs)),
            patch.object(check, "EXPECTED_PRECEDING_CSV_SHA256", digest(self.preceding_csv)),
            patch.object(check, "EXPECTED_FRAMES", 4),
            patch.object(check, "EXPECTED_GRADED", 2),
            patch.object(check, "EXPECTED_GT_SPELLS", 1),
            patch.object(check, "BOOTSTRAP_DRAWS", 1000),
        ]
        for item in self.patchers:
            item.start()

    def tearDown(self):
        for item in reversed(self.patchers):
            item.stop()
        self.temporary.cleanup()

    def test_freeze_binds_fixed_pair_mask_and_provenance_without_reference_access(self):
        original_sha = check.sha256

        def reject_reference_hash(path):
            self.assertNotEqual(Path(path), self.reference, "freeze opened the public reference")
            return original_sha(path)

        with patch.object(check, "sha256", side_effect=reject_reference_hash):
            spec = check.freeze()
        self.assertFalse(self.started.exists())
        self.assertEqual([item["name"] for item in spec["candidate_set"]], [
            "combined_objective_ensemble", "preceding_seed_ensemble_replay"])
        self.assertEqual(spec["prediction_validation"]["input_mask"]["out_of_play_frames"], 2)
        self.assertEqual(spec["reference"]["organizer_gt_sha256"], self.reference_sha)
        self.assertFalse(spec["organizer_reference_labels_read_before_gate"])
        self.assertEqual(set(spec["active_package"]["models"]), set(self.model_hashes))

    def test_pre_gate_csv_tampering_fails_without_claiming_gate(self):
        check.freeze()
        with self.combined_csv.open("a", encoding="utf-8") as stream:
            stream.write("4,H\n")
        from evaluation import possession_metrics as pm
        original_read = pm.read_label_csv

        def reject_reference(path):
            if Path(path) == self.reference:
                raise AssertionError("tampered pre-gate run opened reference")
            return original_read(path)

        with patch.object(pm, "read_label_csv", side_effect=reject_reference), \
             patch.object(check, "_score_fixed_csvs", side_effect=AssertionError("must not score")):
            with self.assertRaises(check.PublicCheckError):
                check.check()
        self.assertFalse(self.started.exists())

    def test_exclusive_gate_precedes_reference_and_scores_exact_pair_once(self):
        check.freeze()
        from evaluation import possession_metrics as pm
        original_read = pm.read_label_csv
        original_evaluate = pm.evaluate
        calls = []

        def read_after_gate(path):
            if Path(path) == self.reference:
                self.assertTrue(self.started.is_file(), "reference opened before exclusive gate")
            return original_read(path)

        def eval_spy(truth, prediction, **kwargs):
            calls.append(kwargs.get("bootstrap_n"))
            return original_evaluate(truth, prediction, **kwargs)

        old_manifest = self.manifest.read_bytes()
        old_policy = self.policy.read_bytes()
        with patch.object(pm, "read_label_csv", side_effect=read_after_gate), \
             patch.object(pm, "evaluate", side_effect=eval_spy):
            result = check.check()
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls, [1000, 1000])
            self.assertIn("spell_f1_mean", result["scores"]["combined_objective_ensemble"])
            self.assertIn("d_precision", result["scores"]["preceding_seed_ensemble_replay"])
            with self.assertRaisesRegex(check.PublicCheckError, "gate is closed"):
                check.check()
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.manifest.read_bytes(), old_manifest)
        self.assertEqual(self.policy.read_bytes(), old_policy)
        self.assertTrue(result["interpretation"]["no_model_selection_or_tuning_after_public_score"])
        self.assertFalse(result["interpretation"]["paired_significance_test"])

    def test_failure_after_gate_is_irreversible(self):
        check.freeze()
        with patch.object(check, "_score_fixed_csvs", side_effect=RuntimeError("synthetic scorer error")):
            with self.assertRaisesRegex(RuntimeError, "synthetic scorer error"):
                check.check()
        self.assertTrue(self.started.is_file())
        with patch.object(check, "_score_fixed_csvs", side_effect=AssertionError("must not retry")):
            with self.assertRaisesRegex(check.PublicCheckError, "gate is closed"):
                check.check()


if __name__ == "__main__":
    unittest.main()
