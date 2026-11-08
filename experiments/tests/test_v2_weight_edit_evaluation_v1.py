import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import v2_weight_edit_evaluation_v1 as study
from experiments import v2_weight_edit_deployment as deployment
from experiments import feed_spell_push


ROOT = Path(__file__).resolve().parents[2]


class WeightEditEvaluationProtocolTests(unittest.TestCase):
    def _temporary_directory(self):
        return tempfile.TemporaryDirectory(dir=ROOT / "data/experiments")

    def test_saved_comparator_sources_have_expected_active_scores_and_test_games(self):
        gar = json.loads((ROOT / "experiments/results/gar_deployment_eval_v2.json").read_text())
        public = json.loads((ROOT / "experiments/results/feed_like_v2_public_result.json").read_text())
        per_game = gar["splits"]["test"]["scores"]["v2"]["original_full"]["phase_avg"]["per_game"]

        self.assertEqual(len(per_game), 10)
        self.assertAlmostEqual(gar["splits"]["test"]["scores"]["v2"]["original_full"]["phase_avg"]["mean"]["spell_f1"], .701288850, places=8)
        self.assertAlmostEqual(public["candidate"]["spell_f1_mean"], .611764705882353, places=12)
        self.assertAlmostEqual(public["candidate"]["macro_f1"], .8494725958535332, places=12)

    def test_four_fixed_realizations_are_averaged_within_game(self):
        rows = {
            "v2_1": {"spell_f1": .50, "graded_frames": 10, "reference_spell_count": 3},
            "v2_2": {"spell_f1": .60, "graded_frames": 10, "reference_spell_count": 3},
            "v2_3": {"spell_f1": .70, "graded_frames": 10, "reference_spell_count": 3},
            "v2_4": {"spell_f1": .80, "graded_frames": 10, "reference_spell_count": 3},
        }
        result = study.average_realizations(rows, ("spell_f1", "graded_frames", "reference_spell_count"))
        self.assertAlmostEqual(result["spell_f1"], .65)
        self.assertEqual(result["graded_frames"], 10)
        self.assertEqual(result["reference_spell_count"], 3)

    def test_realization_aggregation_rejects_changed_scoring_population(self):
        rows = {f"v2_{i}": {"graded_frames": 100 + (i == 4), "reference_spell_count": 3}
                for i in range(1, 5)}
        with self.assertRaisesRegex(ValueError, "graded_frames changed"):
            study.average_realizations(rows, ("graded_frames", "reference_spell_count"))

    def test_paired_bootstrap_uses_ten_whole_game_deltas(self):
        candidate = np.linspace(.60, .69, 10)
        baseline = np.linspace(.59, .68, 10)
        result = study.paired_game_summary(candidate, baseline, np.random.default_rng(9), draws=1000)
        self.assertEqual(result["games"], 10)
        self.assertEqual(result["games_positive"], 10)
        self.assertAlmostEqual(result["delta"], .01)
        self.assertLessEqual(result["paired_game_bootstrap_95"][0], result["delta"] + 1e-12)
        self.assertGreaterEqual(result["paired_game_bootstrap_95"][1] + 1e-12, result["delta"])

    def test_corruption_namespace_reuses_frozen_gar_study(self):
        game = json.loads((ROOT / "experiments/results/gar_deployment_eval_v2.json").read_text())["splits"]["test"]
        game_id = next(iter(json.loads((ROOT / "experiments/results/gar_deployment_eval_v2_declaration.json").read_text())["match_ids"]["test"]))
        self.assertEqual(study.corruption_seed_key(game_id), f"gar_deployment_eval_v2/{game_id}")
        self.assertEqual(game["games"], 10)

    def test_pooled_control_gate_and_projection_edit_scopes_are_exact(self):
        hidden = 2
        state = {
            "output.weight": np.asarray([[1., 2., 0.], [3., 4., 0.]], dtype=np.float32),
            "output.bias": np.zeros(2, dtype=np.float32),
            "sequence.bias_ih_l0": np.arange(3 * hidden, dtype=np.float32),
            "sequence.bias_ih_l0_reverse": np.arange(3 * hidden, dtype=np.float32) + 10,
        }
        model = {"state": state, "d_state": {"weight": np.asarray([4.])}, "hidden": hidden,
                 "feature_names": ["x"], "median": np.asarray([0.]), "scale": np.asarray([1.]),
                 "training": {"seed": 2026}}

        control, changed = deployment.apply_selected_edit(model, {"kind": "control"}, 0, 0)
        self.assertEqual(changed, [])
        for key in state:
            np.testing.assert_array_equal(control["state"][key], model["state"][key])

        gate, changed = deployment.apply_selected_edit(model, {"kind": "gate_bias"}, 0, -.25)
        self.assertEqual(set(changed), {"sequence.bias_ih_l0", "sequence.bias_ih_l0_reverse"})
        np.testing.assert_array_equal(gate["state"]["sequence.bias_ih_l0"][:hidden], state["sequence.bias_ih_l0"][:hidden])
        np.testing.assert_allclose(gate["state"]["sequence.bias_ih_l0"][hidden:2 * hidden]
                                   - state["sequence.bias_ih_l0"][hidden:2 * hidden], -.25)
        np.testing.assert_array_equal(gate["d_state"]["weight"], model["d_state"]["weight"])

        direction = {"direction": [1., 0., 0.]}
        projection, changed = deployment.apply_selected_edit(model, {"kind": "ha_projection"}, .5, 0, direction)
        self.assertEqual(changed, ["output.weight"])
        np.testing.assert_array_equal(projection["state"]["output.bias"], state["output.bias"])
        np.testing.assert_array_equal(projection["state"]["sequence.bias_ih_l0"], state["sequence.bias_ih_l0"])
        np.testing.assert_array_equal(projection["d_state"]["weight"], model["d_state"]["weight"])

    def test_selected_cost_1_5_decoder_config_runs_on_probability_trace(self):
        config = {"family": "calibration", "dt": 4, "ht": 1, "ha": 1.5, "dead": 3}
        probabilities = np.tile(np.asarray([[.49, .46, .05]], dtype=np.float64), (1, 16, 1))
        labels = feed_spell_push.decode(probabilities, np.ones(16, dtype=np.int32), 25.0, config)
        self.assertEqual(labels.shape, (16,))
        self.assertTrue(set(labels.tolist()) <= {"H", "A", "D", "X"})

    def test_actual_dev_choice_schema_normalizes_each_edit_kind(self):
        base = {"study_id": "v2_weight_edit_development_v1",
                "development_report_path": "experiments/results/dev.json",
                "development_report_sha256": "0" * 64,
                "winner_id": "arm", "decoder_config": {"family": "calibration", "ha": 1.5}}
        cases = [
            ({"kind": "none", "strength": 0., "alpha": None, "delta": None}, "control", 0., 0.),
            ({"kind": "projection", "strength": .4, "alpha": .4, "delta": None,
              "direction_path": "unused", "direction_sha256": "unused"}, "ha_projection", .4, 0.),
            ({"kind": "update_bias", "strength": -.2, "alpha": None, "delta": -.2}, "gate_bias", 0., -.2),
        ]
        for edit, kind, alpha, delta in cases:
            choice = dict(base, edit=edit)
            if kind == "ha_projection":
                with self._temporary_directory() as directory:
                    path = Path(directory) / "direction.json"
                    path.write_text("{}")
                    choice["edit"] = dict(edit, direction_path=str(path),
                                          direction_sha256=deployment.sha(path))
                    normalized, strength, got_alpha, got_delta = deployment._normalize_choice(choice)
            else:
                normalized, strength, got_alpha, got_delta = deployment._normalize_choice(choice)
            self.assertEqual(normalized["kind"], kind)
            self.assertAlmostEqual(got_alpha, alpha)
            self.assertAlmostEqual(got_delta, delta)
            self.assertAlmostEqual(strength, edit["strength"])

    def test_public_summary_accepts_official_scorer_result(self):
        from evaluation import possession_metrics as pm
        gt = pm.label_codes(np.asarray(list("HHHDDDAAADDDXXX")))
        pred = pm.label_codes(np.asarray(list("HHHDDDAAADDDXXX")))
        raw = pm.evaluate(gt, pred, bootstrap_n=10)
        summary = study._public_summary(raw)
        self.assertEqual(summary["frames"], 15)
        self.assertEqual(summary["graded_frames"], 12)
        self.assertEqual(summary["ground_truth_spells"], 2)
        self.assertEqual(summary["bootstrap_draws"], 10)
        self.assertEqual(len(summary["spell_f1_bootstrap_5_95"]), 2)

    def _public_fixture(self, directory):
        from evaluation import possession_metrics as pm
        patched_names = ("PUBLIC_OUT", "PUBLIC_CSV", "PUBLIC_RECEIPT", "PUBLIC_DECLARATION",
                         "PUBLIC_STARTED", "PUBLIC_RESULT_PRIVATE", "PUBLIC_RESULT", "PUBLIC_SCORE_RAW",
                         "REFERENCE", "INPUTS", "PACKAGE_RECEIPT", "BASELINE_PUBLIC",
                         "BASELINE_PUBLIC_RECEIPT", "DECLARATION", "EXPECTED_FRAMES",
                         "EXPECTED_GRADED", "EXPECTED_GT_SPELLS", "PUBLIC_BOOTSTRAP_DRAWS")
        old_values = {name: getattr(study, name) for name in patched_names}
        self.addCleanup(lambda: [setattr(study, name, value) for name, value in old_values.items()])
        out = Path(directory)
        public_out = out / "public"
        public_out.mkdir()
        candidate = public_out / "candidate.csv"
        reference = out / "reference.csv"
        inputs = out / "inputs.npz"
        package = out / "deployment.json"
        baseline_public = out / "baseline.json"
        baseline_receipt = out / "baseline_receipt.json"
        labels = np.asarray(list("HHHDDDAAADDDXXX"))
        import pandas as pd
        pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(candidate, index=False)
        pd.DataFrame({"frame": np.arange(len(labels)), "label": labels}).to_csv(reference, index=False)
        np.savez(inputs, period=np.r_[np.ones(12, dtype=np.int32), np.zeros(3, dtype=np.int32)])
        package.write_text("{}")
        baseline_public.write_text(json.dumps({"candidate_label": "original_full + 5 Hz phase TTA (5 phases, equal mean, f062)",
                                               "candidate": {"spell_f1_mean": .611764705882353,
                                                             "macro_f1": .8494725958535332}}))
        baseline_receipt.write_text(json.dumps({"reference_sha256": study.sha256(reference)}))
        study.PUBLIC_OUT = public_out
        study.PUBLIC_CSV = candidate
        study.PUBLIC_RECEIPT = public_out / "candidate.json"
        study.PUBLIC_DECLARATION = out / "declaration.json"
        study.PUBLIC_STARTED = public_out / "public_check_started.json"
        study.PUBLIC_RESULT_PRIVATE = public_out / "public_result.json"
        study.PUBLIC_RESULT = out / "public_result.json"
        study.PUBLIC_SCORE_RAW = public_out / "score_raw.json"
        study.REFERENCE = reference
        study.INPUTS = inputs
        study.PACKAGE_RECEIPT = package
        study.BASELINE_PUBLIC = baseline_public
        study.BASELINE_PUBLIC_RECEIPT = baseline_receipt
        study.DECLARATION = out / "prediction_declaration.json"
        study.DECLARATION.write_text("{}")
        study.EXPECTED_FRAMES = len(labels)
        study.EXPECTED_GRADED = 12
        study.EXPECTED_GT_SPELLS = 2
        study.PUBLIC_BOOTSTRAP_DRAWS = 10
        study.PUBLIC_RECEIPT.write_text(json.dumps({
            "candidate_csv_sha256": study.sha256(candidate),
            "package_receipt_sha256": study.sha256(package),
            "input_sha256": study.sha256(inputs),
            "organizer_reference_labels_read": False,
            "organizer_scored": False, "organizer_score": None,
        }))
        return pm

    def test_invalid_candidate_is_rejected_without_reference_hash_or_read(self):
        with self._temporary_directory() as directory:
            pm = self._public_fixture(directory)
            study.PUBLIC_CSV.write_text("frame,label\n0,Q\n")
            receipt = json.loads(study.PUBLIC_RECEIPT.read_text())
            receipt["candidate_csv_sha256"] = study.sha256(study.PUBLIC_CSV)
            study.PUBLIC_RECEIPT.write_text(json.dumps(receipt))
            with mock.patch.object(study, "sha256", wraps=study.sha256) as hashed:
                with self.assertRaisesRegex(study.EvaluationError, "invalid"):
                    study.freeze_public()
                self.assertFalse(any(call.args[0] == study.REFERENCE for call in hashed.call_args_list))
            self.assertFalse(study.PUBLIC_STARTED.exists())

    def test_score_claims_gate_before_reference_hash_and_preserves_raw_on_format_failure(self):
        with self._temporary_directory() as directory:
            pm = self._public_fixture(directory)
            study._source_hashes = lambda: {}
            study.freeze_public()
            original_sha = study.sha256
            def assert_claimed_sha(path):
                if Path(path) == study.REFERENCE:
                    self.assertTrue(study.PUBLIC_STARTED.is_file())
                return original_sha(path)
            with mock.patch.object(study, "sha256", side_effect=assert_claimed_sha), \
                 mock.patch.object(study, "_public_summary", side_effect=RuntimeError("format failed")):
                with self.assertRaisesRegex(RuntimeError, "format failed"):
                    study.score_public()
            self.assertTrue(study.PUBLIC_STARTED.is_file())
            self.assertTrue(study.PUBLIC_SCORE_RAW.is_file())
            raw = json.loads(study.PUBLIC_SCORE_RAW.read_text())
            self.assertEqual(raw["raw_score"]["n_frames"], 15)
            with self.assertRaisesRegex(study.EvaluationError, "already claimed"):
                study.score_public()


if __name__ == "__main__":
    unittest.main()
