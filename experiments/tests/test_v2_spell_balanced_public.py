import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import possession_metrics as pm
from experiments import v2_spell_balanced_public as public


class PublicGateTests(unittest.TestCase):
    def test_selection_requires_frozen_supported_recipe(self):
        selection = {
            "study_id": public.STUDY,
            "final_selection": True,
            "recipe": {"id": "anchor", "ce_weight": .5,
                       "calibrated_weight": 0.0, "adopted_weight": .5,
                       "pooling": "arithmetic", "ha_cost": 3.0},
            "training_stage": 512,
        }
        self.assertEqual(public._selection_contract(selection), selection["recipe"])
        selection["recipe"]["adopted_weight"] = .25
        with self.assertRaises(ValueError):
            public._selection_contract(selection)
        selection["recipe"]["adopted_weight"] = float("nan")
        with self.assertRaises(ValueError):
            public._selection_contract(selection)

    def test_seed_then_phase_pooling_uses_both_native_axes(self):
        import numpy as np
        values = np.zeros((2, 5, 3, 3), dtype=float)
        values[..., 0] = .6
        values[..., 1] = .3
        values[..., 2] = .1
        values[1, 4, :, :] = [.3, .6, .1]
        pooled = public._pool_seed_phase_probabilities(values)
        expected = values.mean(axis=0).mean(axis=0)
        np.testing.assert_allclose(pooled, expected, rtol=0, atol=1e-15)
        self.assertEqual(pooled.shape, (3, 3))

    def test_saved_public_baseline_requires_all_known_summary_values(self):
        with tempfile.TemporaryDirectory() as temp:
            baseline = Path(temp) / "prior.json"
            baseline.write_text(json.dumps({"candidate": {
                "spell_f1_mean": public.PUBLIC_BASELINE_SPELL_F1,
                "macro_f1": public.PUBLIC_BASELINE_MACRO_F1,
                "d_f1": public.PUBLIC_BASELINE_D_F1}}))
            with patch.object(public, "BASELINE", baseline):
                self.assertEqual(public._verify_public_baseline()["spell_f1_mean"],
                                 public.PUBLIC_BASELINE_SPELL_F1)
                baseline.write_text(json.dumps({"candidate": {
                    "spell_f1_mean": .7, "macro_f1": .8, "d_f1": .8}}))
                with self.assertRaises(ValueError):
                    public._verify_public_baseline()

    def test_gate_claim_is_exclusive_and_binds_seal(self):
        with tempfile.TemporaryDirectory() as temp:
            seal_path, gate_path = Path(temp) / "seal.json", Path(temp) / "gate.json"
            seal_path.write_text('{"bound":true}\n')
            with patch.object(public, "SEAL", seal_path), patch.object(public, "GATE", gate_path):
                claim = public._claim_gate({"candidate_csv_sha256": "csv-sha",
                                            "selection_sha256": "selection-sha"})
                saved = json.loads(gate_path.read_text())
                self.assertEqual(saved, claim)
                self.assertEqual(claim["prediction_seal_sha256"], public.sha(seal_path))
                with self.assertRaises(FileExistsError):
                    public._claim_gate({"candidate_csv_sha256": "other",
                                        "selection_sha256": "other"})

    def test_closed_gate_prevents_any_reference_access(self):
        with tempfile.TemporaryDirectory() as temp:
            gate = Path(temp) / "already_claimed.json"
            gate.write_text("{}")
            result = Path(temp) / "result.json"
            with patch.object(public, "GATE", gate), patch.object(public, "RESULT", result), \
                    patch.object(public, "REFERENCE", Path(temp) / "must_not_open.csv"):
                with self.assertRaises(FileExistsError):
                    public.score_once()
            self.assertFalse((Path(temp) / "must_not_open.csv").exists())

    def test_synthetic_public_score_claims_once_and_binds_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seal_path, gate_path, result_path = (root / "seal.json", root / "gate.json",
                                                 root / "result.json")
            reference, csv_path, baseline_path = (root / "synthetic_public_reference.csv",
                                                   root / "prediction.csv", root / "baseline.json")
            reference.write_text("frame,label\n0,H\n")
            csv_path.write_text("temporary candidate fixture\n")
            baseline_path.write_text(json.dumps({"candidate": {
                "spell_f1_mean": public.PUBLIC_BASELINE_SPELL_F1,
                "macro_f1": public.PUBLIC_BASELINE_MACRO_F1,
                "d_f1": public.PUBLIC_BASELINE_D_F1}}))
            seal = {"study_id": public.STUDY,
                    "candidate_csv_path": csv_path.name,
                    "candidate_csv_sha256": public.sha(csv_path),
                    "selection_sha256": "sel-sha",
                    "selected_recipe": {"id": "fixture"}, "training_stage": 512,
                    "saved_public_baseline_sha256": public.sha(baseline_path)}
            seal_path.write_text(json.dumps(seal))
            measured = {"n_frames": public.EXPECTED_FRAMES,
                        "n_graded": public.EXPECTED_GRADED,
                        "n_gt_spells": public.EXPECTED_SPELLS}
            summary = {"spell_f1_mean": .62, "macro_f1": .85, "d_f1": .86}
            seen = []

            class FakeMetrics:
                @staticmethod
                def label_codes(value):
                    return value

                @staticmethod
                def read_label_csv(path):
                    seen.append(Path(path))
                    return Path(path).read_text()

                @staticmethod
                def evaluate(truth, prediction, bootstrap_n):
                    self.assertEqual(bootstrap_n, 1000)
                    return measured

            with patch.object(public, "SEAL", seal_path), patch.object(public, "GATE", gate_path), \
                    patch.object(public, "RESULT", result_path), patch.object(public, "CSV", csv_path), \
                    patch.object(public, "REFERENCE", reference), patch.object(public, "BASELINE", baseline_path), \
                    patch.object(public, "ROOT", root), patch.object(public, "_verify_seal"), \
                    patch.object(pm, "label_codes", FakeMetrics.label_codes), \
                    patch.object(pm, "read_label_csv", FakeMetrics.read_label_csv), \
                    patch.object(pm, "evaluate", FakeMetrics.evaluate), \
                    patch("experiments.combined_objective_public_check._score_summary", return_value=summary):
                result = public.score_once()
                self.assertEqual(result["delta_vs_saved_baseline"]["spell_f1_mean"],
                                 .62 - public.PUBLIC_BASELINE_SPELL_F1)
                self.assertEqual(result["candidate_recipe"], {"id": "fixture"})
                self.assertEqual(result["prediction_seal_sha256"], public.sha(seal_path))
                self.assertEqual(result["public_gate_claim_sha256"], public.sha(gate_path))
                with self.assertRaises(FileExistsError):
                    public.score_once()
            self.assertEqual(seen, [reference, csv_path])
            self.assertTrue(result_path.is_file())


if __name__ == "__main__":
    unittest.main()
