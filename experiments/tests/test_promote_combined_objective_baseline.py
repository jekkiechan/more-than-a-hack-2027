"""Pure validation contracts for the conditional combined-objective promoter."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib

from experiments import promote_combined_objective_baseline as promoter


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CombinedObjectivePromotionContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="combined-promotion-test-")
        self.root = Path(self.temp.name)
        self.baseline_path = self.root / "experiments/baseline_manifest.json"
        self.policy_path = self.root / "experiments/model_decision_policy.json"
        self.promotion_path = self.root / "experiments/results/baseline_seed_ensemble_promotion.json"
        self.report_path = self.root / "data/experiments/objective_ensemble_v1/report.json"
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.declaration_path = self.report_path.parent / "declaration.json"
        self.runs_path = self.report_path.parent / "runs.json"
        self.declaration_path.write_text("{}\n", encoding="utf-8")
        self.runs_path.write_text("[]\n", encoding="utf-8")

        old_models = {"joint_gru_2026.joblib": "a" * 64,
                      "joint_gru_2027.joblib": "b" * 64}
        write_json(self.baseline_path, dict(
            model="joint_gru_seed_ensemble", package_version="joint_gru_seed_ensemble_v1",
            models_sha256=old_models,
            evidence=dict(report_sha256="c" * 64, verification_sha256="d" * 64)))
        write_json(self.policy_path, dict(working_candidate=dict(
            model="joint_gru_seed_ensemble",
            baseline_manifest_sha256=digest(self.baseline_path))))
        write_json(self.promotion_path, {"old": True})

        providers = {
            "sportec": {"J03WOY"},
            "metrica": {"metrica_3"},
            "skillcorner": {"skillcorner_had_1996435", "skillcorner_had_1996436",
                            "skillcorner_had_2006229"},
        }
        cohort = [{"source": source, "match": match}
                  for source, matches in providers.items() for match in sorted(matches)]
        candidate = {
            "clean": {"equal_provider_mean": {"spell_f1": .64},
                      "per_provider": {name: dict(spell_f1=.61, macro_f1=.85, d_f1=.84)
                                       for name in providers}},
            "feed_like": {"equal_provider_mean": {"spell_f1": .50}},
        }
        control = {
            "clean": {"equal_provider_mean": {"spell_f1": .63},
                      "per_provider": {name: dict(spell_f1=.62, macro_f1=.85, d_f1=.84)
                                       for name in providers}},
            "feed_like": {"equal_provider_mean": {"spell_f1": .49}},
        }
        checks = {
            "strict_positive_clean_spell_gain": True,
            "strict_positive_feed_like_spell_gain": True,
            "each_provider_clean_spell_delta_at_least_minus_0_02": True,
            "each_provider_clean_macro_delta_at_least_minus_0_02": True,
            "each_provider_clean_d_f1_delta_at_least_minus_0_02": True,
        }
        self.report = dict(
            study_id="objective_ensemble_v1",
            declaration_sha256=digest(self.declaration_path),
            runs_sha256=digest(self.runs_path),
            baseline=dict(
                manifest_sha256=digest(self.baseline_path),
                decision_policy_sha256=digest(self.policy_path),
                model="joint_gru_seed_ensemble",
                package_version="joint_gru_seed_ensemble_v1",
                model_sha256=old_models,
                saved_report_sha256="c" * 64,
                verification_sha256="d" * 64,
                promotion_receipt_sha256=digest(self.promotion_path)),
            cohort=cohort,
            evaluation_status=dict(organizer_labels_or_scores_used=False,
                                  independent_confirmation=False, post_hoc=True),
            comparison=dict(
                candidate="combined_objective_equal_probability_ensemble",
                control="current_baseline_equal_probability_ensemble",
                decoder="Viterbi switch cost 3",
                probability_rule="Equal arithmetic mean of member probabilities"),
            result=dict(
                recipes={"combined_objective_equal_probability_ensemble": candidate,
                         "current_baseline_equal_probability_ensemble": control},
                candidate_minus_control={
                    "clean": {"equal_provider_mean": {"spell_f1": .01}},
                    "feed_like": {"equal_provider_mean": {"spell_f1": .01}},
                },
                selection_gate=dict(automatic_promotion=False, post_hoc=True,
                                    passes=True, checks=checks)),
        )
        write_json(self.report_path, self.report)
        self.verification_path = self.root / "experiments/results/objective_ensemble_verification.json"
        write_json(self.verification_path, dict(
            study_id="objective_ensemble_v1",
            inputs=dict(report_sha256=digest(self.report_path),
                        runs_sha256=digest(self.runs_path),
                        declaration_sha256=digest(self.declaration_path),
                        prior_baseline_report_sha256="c" * 64),
            checks=dict(
                expected_cells=15,
                baseline_saved_score_rows_match_prior_report_at_abs_tolerance_1e_12=15,
                baseline_probability_and_decode_digests_reproduced=15,
                candidate_two_seed_native_probability_means_exact=15,
                candidate_ensemble_decode_prediction_digests_reproduced=15,
                candidate_output_and_member_file_array_digests_match=15,
                independently_recomputed_candidate_minus_control_deltas_match=True,
                independently_recomputed_provider_and_equal_provider_aggregates_max_abs_delta=0.0,
                post_hoc_selection_gate_passes=True,
                selection_gate_checks=checks),
            equal_provider_spell_f1_delta={"clean": .01, "feed_like": .01}))
        self.patchers = [
            patch.object(promoter, "ROOT", self.root),
            patch.object(promoter, "BASELINE", self.baseline_path),
            patch.object(promoter, "POLICY", self.policy_path),
            patch.object(promoter, "PRIOR_PROMOTION", self.promotion_path),
            patch.object(promoter, "COMPARISON_VERIFICATION", self.verification_path),
            patch.object(promoter, "EXPECTED_COMPARISON_VERIFICATION_SHA256",
                         digest(self.verification_path)),
        ]
        for item in self.patchers:
            item.start()

    def tearDown(self):
        for item in reversed(self.patchers):
            item.stop()
        self.temp.cleanup()

    def _resync_verifier_report_hash(self):
        verification = json.loads(self.verification_path.read_text(encoding="utf-8"))
        verification["inputs"]["report_sha256"] = digest(self.report_path)
        write_json(self.verification_path, verification)
        promoter.EXPECTED_COMPARISON_VERIFICATION_SHA256 = digest(self.verification_path)

    def _prediction_sources(self):
        sources = {}
        for relative, contents in (
                ("scripts/predict_combined_objective.py", "label-free prediction source\n"),
                ("experiments/combined_objective_deployment.py", "deployment inference source\n")):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            sources[relative] = digest(path)
        return sources

    def test_replay_supports_linked_data_but_requires_candidate_directory_and_seal(self):
        with tempfile.TemporaryDirectory() as external:
            actual = Path(external)
            linked = self.root / "linked_data"
            linked.symlink_to(actual, target_is_directory=True)
            replay = actual / "replay.csv"
            replay.write_bytes(b"sealed preceding baseline")
            candidate = linked / "candidate.csv"
            with patch.object(promoter, "EXPECTED_PRIOR_CSV_SHA256", digest(replay)):
                self.assertEqual(promoter._verify_replay_file(
                    "linked_data/replay.csv", candidate), replay)
                with self.assertRaisesRegex(ValueError, "byte-parity seal"):
                    promoter._verify_replay_file("linked_data/replay.csv", self.root / "elsewhere.csv")
                replay.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "byte-parity seal"):
                    promoter._verify_replay_file("linked_data/replay.csv", candidate)

    def test_comparison_requires_exact_active_baseline_and_passing_screen(self):
        result = promoter.verify_comparison(self.report_path)
        self.assertEqual(result["report_sha256"], digest(self.report_path))
        self.assertTrue(result["selection_gate"]["passes"])
        self.assertFalse(result["independent_confirmation"])

    def test_comparison_rejects_changed_independent_receipt(self):
        self.verification_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "verification receipt changed"):
            promoter.verify_comparison(self.report_path)

    def test_comparison_rejects_changed_baseline_binding(self):
        self.report["baseline"]["manifest_sha256"] = "e" * 64
        write_json(self.report_path, self.report)
        self._resync_verifier_report_hash()
        with self.assertRaisesRegex(ValueError, "different baseline"):
            promoter.verify_comparison(self.report_path)

    def test_comparison_rejects_nonpositive_independently_verified_delta(self):
        self.report["result"]["candidate_minus_control"]["clean"][
            "equal_provider_mean"]["spell_f1"] = -.01
        write_json(self.report_path, self.report)
        verification = json.loads(self.verification_path.read_text(encoding="utf-8"))
        verification["inputs"]["report_sha256"] = digest(self.report_path)
        verification["equal_provider_spell_f1_delta"]["clean"] = -.01
        write_json(self.verification_path, verification)
        promoter.EXPECTED_COMPARISON_VERIFICATION_SHA256 = digest(self.verification_path)
        with self.assertRaisesRegex(ValueError, "clean spell delta"):
            promoter.verify_comparison(self.report_path)

    def test_label_free_receipt_binds_csv_input_models_and_no_public_scoring(self):
        input_path = self.root / "data/challenge/inputs.npz"
        input_path.parent.mkdir(parents=True)
        input_path.write_bytes(b"sealed input")
        csv_path = self.root / "candidate.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            for index in range(promoter.EXPECTED_ROWS):
                writer.writerow((index, "X"))
        model_hashes = {"joint_gru_2026.joblib": "1" * 64,
                        "joint_gru_2027.joblib": "2" * 64}
        receipt = dict(
            input_sha256=digest(input_path), output_sha256=digest(csv_path),
            rows=promoter.EXPECTED_ROWS,
            model_sha256=model_hashes,
            member_probability_sha256={"joint_gru_2026": "3" * 64,
                                       "joint_gru_2027": "4" * 64},
            blended_probability_sha256="5" * 64,
            current_baseline_replay_sha256=promoter.EXPECTED_PRIOR_CSV_SHA256,
            format_validation={"valid": True, "rows": promoter.EXPECTED_ROWS},
            organizer_reference_labels_read=False, organizer_scored=False,
            organizer_score=None)
        sources = self._prediction_sources()
        declaration_path = csv_path.with_suffix(".declaration.json")
        write_json(declaration_path, dict(
            input_sha256=digest(input_path),
            current_manifest_sha256=digest(self.baseline_path),
            model_sha256=model_hashes,
            sources_sha256=sources,
            organizer_reference_labels_read=False, organizer_scored=False))
        receipt["declaration_sha256"] = digest(declaration_path)
        receipt_path = self.root / "candidate.json"
        write_json(receipt_path, receipt)
        with patch.object(promoter, "INPUTS", input_path), \
             patch.object(promoter, "EXPECTED_INPUT_SHA256", digest(input_path)):
            checked = promoter.verify_label_free_validation(receipt_path, csv_path, model_hashes)
        self.assertEqual(checked["csv_sha256"], digest(csv_path))
        self.assertEqual(checked["rows"], promoter.EXPECTED_ROWS)
        self.assertFalse(checked["organizer_scored"])
        self.assertEqual(checked["prediction_sources_sha256"], sources)
        # The promoter checks every declared inference dependency, not just
        # the top-level prediction script.
        declaration = json.loads(declaration_path.read_text(encoding="utf-8"))
        declaration["sources_sha256"]["experiments/combined_objective_deployment.py"] = "0" * 64
        write_json(declaration_path, declaration)
        receipt["declaration_sha256"] = digest(declaration_path)
        write_json(receipt_path, receipt)
        with patch.object(promoter, "INPUTS", input_path), \
             patch.object(promoter, "EXPECTED_INPUT_SHA256", digest(input_path)):
            with self.assertRaisesRegex(ValueError, "inference source changed"):
                promoter.verify_label_free_validation(receipt_path, csv_path, model_hashes)

    def test_prediction_source_map_rejects_paths_outside_repository(self):
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            promoter._verify_prediction_sources({
                "scripts/predict_combined_objective.py": "a" * 64,
                "../outside.py": "b" * 64,
            })

    def test_label_free_receipt_rejects_public_scoring_and_wrong_member_hash(self):
        input_path = self.root / "inputs.npz"
        input_path.write_bytes(b"sealed input")
        csv_path = self.root / "candidate.csv"
        csv_path.write_text("frame,label\n0,X\n", encoding="utf-8")
        model_hashes = {"joint_gru_2026.joblib": "1" * 64,
                        "joint_gru_2027.joblib": "2" * 64}
        receipt_path = self.root / "candidate.json"
        receipt = dict(input_sha256=digest(input_path), output_sha256=digest(csv_path),
                       rows=promoter.EXPECTED_ROWS, model_sha256={"wrong": "f" * 64},
                       member_probability_sha256={"joint_gru_2026": "3" * 64,
                                                  "joint_gru_2027": "4" * 64},
                       blended_probability_sha256="5" * 64,
                       current_baseline_replay_sha256=promoter.EXPECTED_PRIOR_CSV_SHA256,
                       format_validation={"valid": True, "rows": promoter.EXPECTED_ROWS},
                       organizer_reference_labels_read=False, organizer_scored=True,
                       organizer_score=None)
        sources = self._prediction_sources()
        declaration_path = csv_path.with_suffix(".declaration.json")
        write_json(declaration_path, dict(
            input_sha256=digest(input_path),
            current_manifest_sha256=digest(self.baseline_path),
            model_sha256=model_hashes,
            sources_sha256=sources,
            organizer_reference_labels_read=False, organizer_scored=False))
        receipt["declaration_sha256"] = digest(declaration_path)
        write_json(receipt_path, receipt)
        with patch.object(promoter, "INPUTS", input_path), \
             patch.object(promoter, "EXPECTED_INPUT_SHA256", digest(input_path)):
            with self.assertRaisesRegex(ValueError, "different deployment checkpoints"):
                promoter.verify_label_free_validation(receipt_path, csv_path, model_hashes)
            receipt["model_sha256"] = model_hashes
            write_json(receipt_path, receipt)
            with self.assertRaisesRegex(ValueError, "organizer scoring"):
                promoter.verify_label_free_validation(receipt_path, csv_path, model_hashes)

    def test_deployment_loader_requires_both_sealed_combined_members(self):
        from experiments import combined_objective_deployment as deployment

        out = self.root / "deployment"
        paths, rows = {}, []
        matches = [f"match_{index}" for index in range(13)]
        for seed in promoter.MEMBERS:
            seed_num = int(seed.rsplit("_", 1)[1])
            path = out / f"combined_objective_{seed_num}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            model = dict(seed=seed_num, objective="combined",
                         loss_coefficients={"spell": 1.0, "transition": .1},
                         player_only=True)
            joblib.dump(model, path)
            write_json(path.with_suffix(".json"), dict(
                sha256=digest(path), seed=seed_num, objective="combined",
                training_matches=matches))
            paths[seed_num] = path
            rows.append(dict(file=path.name, sha256=digest(path)))
        seal_path = out / "checkpoints_frozen.json"
        write_json(seal_path, dict(checkpoints=rows))
        declaration = dict(
            split_plan={"train": [{"match": match} for match in matches]},
            objective="combined", loss_coefficients={"spell": 1.0, "transition": .1})
        write_json(out / "predeclared.json", declaration)
        with patch.object(deployment, "check", return_value=declaration), \
             patch.object(deployment, "checkpoint", side_effect=lambda seed: paths[seed]), \
             patch.object(deployment, "verify_model", return_value={}), \
             patch.object(deployment, "load_model", side_effect=lambda seed: joblib.load(paths[seed])), \
             patch.object(deployment, "SEAL", seal_path), \
             patch.object(deployment, "OUT", out):
            fitted, loaded_paths, hashes = promoter._deployment_assets()
        self.assertEqual(set(loaded_paths), set(promoter.MEMBERS))
        self.assertEqual(len([key for key in hashes if key.endswith(".joblib")]), 2)
        self.assertEqual(fitted["declaration"]["objective"], "combined")
        self.assertEqual(fitted["declaration_path"], str((out / "predeclared.json").relative_to(promoter.ROOT)))


if __name__ == "__main__":
    unittest.main()
