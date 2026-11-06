"""Synthetic contracts for the independent v2 iterative artifact verifier."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import verify_v2_iterative as verifier


class VerifierShapeContracts(unittest.TestCase):
    def test_round_two_seed_keys_include_training_arm(self):
        cell = "Metrica|metrica_3|feed_like_v2|1"
        self.assertEqual(verifier._round2_seed_key(cell, "boundary_margin", "2027_blend"),
                         f"{cell}|boundary_margin|2027_blend")
        expected = verifier._expected_round2_seed_keys({cell})
        self.assertEqual(len(expected), 8)
        self.assertIn(f"{cell}|boundary_margin|2027_blend", expected)

    def test_seed_metrics_are_nested_by_cell_then_train_arm_and_seed(self):
        cell = "Metrica|metrica_3|clean|0"
        metrics = {"reference_spell_count": 4, "spell_f1": 0.5}
        saved = {}

        verifier._seed_metrics_by_cell(cell, "ce_control", "2026", metrics, saved)
        verifier._seed_metrics_by_cell(cell, "ce_control", "2026_blend", metrics, saved)

        self.assertEqual(set(saved), {cell})
        self.assertEqual(saved[cell]["ce_control|2026"], metrics)
        self.assertEqual(saved[cell]["ce_control|2026_blend"], metrics)

    def test_seed_aggregate_matches_followup_arm_variant_seed_view_shape(self):
        saved = {}
        first_by_provider = {}
        for provider, match, _ in verifier.round1_eval.runtime.frozen.audit.MATCHES:
            first_by_provider.setdefault(provider, match)
        for provider, match in first_by_provider.items():
            for view, reps in verifier.VIEWS:
                for rep in reps:
                    cell = f"{provider}|{match}|{view}|{rep}"
                    for train_arm in ("ce_control", "boundary_margin"):
                        for seed in (2026, 2027):
                            for seed_label in (str(seed), f"{seed}_blend"):
                                metrics = {name: .5 for name in verifier.QUALITY}
                                verifier._seed_metrics_by_cell(cell, train_arm, seed_label,
                                                               metrics, saved)

        result = verifier._round2_seed_aggregate(saved)
        self.assertAlmostEqual(result["ce_control"]["fixed_d"]["2026"]["feed_like_v2"]["spell_f1"]["equal_provider"], .5)
        self.assertEqual(result["boundary_margin"]["conditional_blend"]["2027"]["clean"]["spell_f1"]["cells"], 3)

    def test_reference_count_contract(self):
        verifier._check_reference_count(7, {"reference_spell_count": 7}, "synthetic")
        with self.assertRaises(AssertionError):
            verifier._check_reference_count(7, {"reference_spell_count": 8}, "synthetic")

    def test_metric_comparison_checks_nested_spell_iou_values(self):
        expected = {"spell_f1": .5, "spell_by_iou": {"0.3": .7, "0.5": .6}}
        verifier._compare_metrics(expected, expected, "synthetic")
        changed = {"spell_f1": .5, "spell_by_iou": {"0.3": .7, "0.5": .61}}
        with self.assertRaises(AssertionError):
            verifier._compare_metrics(expected, changed, "synthetic")


class VerifierHashContracts(unittest.TestCase):
    def test_sealed_sidecar_must_match_the_embedded_fit_receipt(self):
        seal_row = {"path": "model.joblib", "sha256": "abc", "seed": 2026}
        self.assertTrue(verifier._fit_receipt_matches(seal_row, dict(seal_row)))
        self.assertFalse(verifier._fit_receipt_matches(seal_row, {**seal_row, "seed": 2027}))

    def test_hash_map_checks_real_file_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "artifact.json"
            artifact.write_text(json.dumps({"fit": 1}))
            with patch.object(verifier, "ROOT", root):
                verifier._verify_hash_map({"artifact.json": verifier.sha(artifact)}, "fixture")
                with self.assertRaises(AssertionError):
                    verifier._verify_hash_map({"artifact.json": "0" * 64}, "fixture")


class RoundThreeContracts(unittest.TestCase):
    def test_four_combinations_match_independent_conditional_math(self):
        trained = np.array([[.72, .18, .10], [.10, .70, .20]])
        calibrated = np.array([[.30, .60, .10], [.56, .24, .20]])
        adopted = np.array([[.50, .30, .20], [.32, .48, .20]])
        actual = verifier._round3_combinations(trained, calibrated, adopted)

        def cond(prob):
            ha = prob[:, :2]
            return ha / ha.sum(axis=1, keepdims=True)

        experts = [cond(p) for p in (trained, calibrated, adopted)]
        expected = {}
        for arm in verifier.round3_eval.ARMS:
            members = experts if arm.startswith("three_") else experts[:2]
            if arm.endswith("_log"):
                ha = np.exp(np.mean([np.log(x) for x in members], axis=0))
            else:
                ha = np.mean(members, axis=0)
            ha /= ha.sum(axis=1, keepdims=True)
            expected[arm] = np.column_stack([ha, np.zeros(len(ha))])
        self.assertEqual(set(actual), set(expected))
        for arm in expected:
            np.testing.assert_allclose(actual[arm], expected[arm], rtol=0, atol=1e-14)
            self.assertTrue(np.all(actual[arm][:, 2] == 0))

    def test_prediction_artifact_binds_bytes_and_exact_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "cell.npz"
            arrays = {"pred__ensemble__trained_calibrated_mean": np.array([0, 1, 2]),
                      "pred__2026__trained_calibrated_mean": np.array([2, 1, 0])}
            np.savez_compressed(artifact, **arrays)
            fact = {"path": artifact.name, "sha256": verifier.sha(artifact)}
            with patch.object(verifier, "ROOT", root):
                loaded = verifier._read_round3_prediction_artifact(fact, set(arrays), "fixture")
                np.testing.assert_array_equal(loaded[next(iter(arrays))], arrays[next(iter(arrays))])
                with self.assertRaises(AssertionError):
                    verifier._read_round3_prediction_artifact(
                        {**fact, "sha256": "0" * 64}, set(arrays), "tampered")
                with self.assertRaises(AssertionError):
                    verifier._read_round3_prediction_artifact(fact, {"extra"}, "wrong inventory")

    def test_round_three_guards_enforce_retention_spell_floor_and_counts(self):
        baseline = {view: {metric: .4 for metric in verifier.METRICS}
                    for view, _ in verifier.VIEWS}
        aggregate = {}
        for arm in verifier.round3_eval.ARMS:
            aggregate[arm] = {}
            for view, _ in verifier.VIEWS:
                aggregate[arm][view] = {
                    metric: {"equal_provider": baseline[view][metric]}
                    for metric in verifier.METRICS}
                aggregate[arm][view]["spell_f1"]["equal_provider"] = .61
                aggregate[arm][view]["reference_spell_count"]["equal_provider"] = 12
        baseline["feed_like_v2"]["spell_f1"] = .60
        for view, _ in verifier.VIEWS:
            baseline[view]["reference_spell_count"] = 12
        gates = verifier._round3_gates(aggregate, baseline)
        self.assertTrue(all(row["all_passed"] for row in gates.values()))
        aggregate[verifier.round3_eval.ARMS[0]]["clean"]["macro_f1"]["equal_provider"] = .39
        gates = verifier._round3_gates(aggregate, baseline)
        self.assertFalse(gates[verifier.round3_eval.ARMS[0]]["all_passed"])


if __name__ == "__main__":
    unittest.main()
