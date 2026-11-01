"""Synthetic contract tests for the frozen spell-MBR evaluation runner."""
from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import spell_mbr_run as runner


class SpellMbrRunnerTests(unittest.TestCase):
    def test_candidate_inference_api_has_no_target_parameter(self):
        parameters = inspect.signature(runner.candidate_prediction).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("truth", parameters)

    def test_candidate_forwards_only_inputs_and_label_free_valid_mask(self):
        probabilities = np.array([[.8, .1, .1], [.1, .8, .1]], dtype=float)
        model_period = np.array([0, 0])
        valid_mask = np.array([True, False])
        observed = {}

        def fake_decode(p, period, fps, **kwargs):
            observed.update(p=p, period=period, fps=fps, kwargs=kwargs)
            return np.array(["H", "X"])

        with patch.object(runner.spell_mbr, "decode", side_effect=fake_decode):
            result = runner.candidate_prediction(probabilities, model_period, 25.0, valid_mask)
        self.assertEqual(result.tolist(), ["H", "X"])
        self.assertIs(observed["p"], probabilities)
        self.assertIs(observed["period"], model_period)
        self.assertEqual(observed["fps"], 25.0)
        self.assertIs(observed["kwargs"]["valid_mask"], valid_mask)
        self.assertEqual(observed["kwargs"]["switch_cost"], 3.0)
        self.assertEqual(observed["kwargs"]["candidate_samples"], 16)
        self.assertEqual(observed["kwargs"]["reference_samples"], 32)
        self.assertEqual(observed["kwargs"]["seed"], 20261101)

    def test_equal_hierarchy_weights_providers_equally_and_averages_feed_repeats(self):
        rows = []
        for provider, matches in runner.EXPECTED_MATCHES.items():
            # Provider values are deliberately unequal across providers and
            # SkillCorner has three matches, so frame/match pooling would differ.
            base = {"Metrica": 1.0, "Sportec": 0.0, "SkillCorner": 0.25}[provider]
            for match in matches:
                for replicate, offset in (("corrupt_1", 0.0), ("corrupt_2", .2)):
                    rows.append({
                        "provider": provider, "match": match,
                        "view": "feed_like_" + replicate, "replicate": replicate,
                        "member": "ensemble", "spell_f1": base + offset,
                        "macro_f1": base + offset, "weighted_70_30": base + offset,
                        "d_f1": base + offset, "d_precision": base + offset,
                        "d_recall": base + offset,
                        "short_spell_recall_iou_0_5_lt3s": base + offset,
                    })
        result = runner._aggregate(rows, "ensemble", "feed_like")
        self.assertAlmostEqual(result["by_provider"]["Metrica"]["spell_f1"], 1.1)
        self.assertAlmostEqual(result["by_provider"]["Sportec"]["spell_f1"], .1)
        self.assertAlmostEqual(result["by_provider"]["SkillCorner"]["spell_f1"], .35)
        self.assertAlmostEqual(result["equal_provider_mean"]["spell_f1"], (1.1 + .1 + .35) / 3)

    def test_score_uses_official_evaluator_and_reports_short_spell_recall(self):
        labels = np.array(["H"] * 4 + ["D"] * 4 + ["A"] * 4 + ["X"] * 2)
        prediction = np.array(["H"] * 4 + ["D"] * 4 + ["A"] * 4 + ["X"] * 2)
        period = np.ones(len(labels), dtype=np.int8)
        timestamps = np.arange(len(labels), dtype=float) / 2.0
        score = runner._score_prediction(labels, prediction, period, timestamps, 2.0)
        self.assertEqual(score["reference_spell_count"], 2)
        self.assertEqual(score["predicted_spell_count"], 2)
        self.assertEqual(score["short_reference_spell_count"], 2)
        self.assertAlmostEqual(score["spell_f1"], 1.0)
        self.assertAlmostEqual(score["short_spell_recall_iou_0_5_lt3s"], 1.0)

    def test_exclusive_output_claim_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "private"
            runner._claim_output(output, "a" * 64)
            self.assertTrue((output / "run.lock.json").is_file())
            with self.assertRaises(FileExistsError):
                runner._claim_output(output, "a" * 64)


if __name__ == "__main__":
    unittest.main()
