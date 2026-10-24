"""Synthetic checks for score selection and one-shot evaluation guards.

No external match or competition annotation file is opened by these tests.
"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from experiments import refinement, refine_combined, refine_features, refine_scale, refine_weighted
from experiments.tests.test_contracts import tracking


class SyntheticEstimator:
    """Deterministic feature-dependent probabilities without a trained artifact."""

    def __init__(self, bias=0.):
        self.bias = bias

    def predict_proba(self, matrix):
        clean = np.nan_to_num(matrix, nan=0.)
        logits = np.column_stack((clean[:, 2] * .05 + self.bias,
                                  clean[:, 5] * .03 - self.bias,
                                  np.full(len(matrix), -.4)))
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)


class RefinementGateTests(unittest.TestCase):
    def test_score_normalizer_selects_clean_selected_score(self):
        clean = {"macro_f1": .91, "spell_f1_mean": .72}
        masked = {"macro_f1": .98, "spell_f1_mean": .95}
        alternatives = [
            clean,
            {"selected": {"clean": {"scores": {"J03WOY": clean}}, "masked": masked}},
            {"selected": {"clean": clean, "masked": masked}},
            {"conditions": {"clean": clean, "masked": masked}},
            {"selected_validation": clean, "candidates": [masked]},
        ]
        for payload in alternatives:
            with self.subTest(payload=payload):
                self.assertEqual(refinement.score_in(payload), clean)

    def test_score_normalizer_rejects_missing_clean_score(self):
        with self.assertRaises(ValueError):
            refinement.score_in({"masked": {"macro_f1": 1, "spell_f1_mean": 1}})

    def test_started_guard_records_manifest_and_refuses_repetition(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text("{}\n")
            with patch.object(refinement, "OUT", root), patch.object(refinement, "MANIFEST", manifest):
                refinement.started("synthetic_check")
                record = json.loads((root / "synthetic_check_started.json").read_text())
                self.assertEqual(record["manifest_sha256"], refinement.robust.sha256(manifest))
                with self.assertRaises(RuntimeError):
                    refinement.started("synthetic_check")

    def test_weighting_changes_training_importance_without_joining_gaps(self):
        # A two-second live spell on either side of a time gap must not become
        # one four-second spell. D/X remain neutral training weights.
        data = {"labels": np.array(list("HHHHDDXX")), "period": np.ones(8),
                "timestamps": np.array([0., 1., 5., 6., 7., 8., 9., 10.]), "fps": 1.}
        weight, duration = refine_weighted.spell_weights(data, .5)
        np.testing.assert_allclose(duration, [2, 2])
        np.testing.assert_allclose(weight[:4], np.sqrt(2.5))
        np.testing.assert_array_equal(weight[4:], np.ones(4))

    def test_scaled_decoder_ignores_targets_and_isolates_time_gaps(self):
        rng = np.random.default_rng(21)
        probability = rng.dirichlet([1, 1, 1], 80)
        data = {"period": np.ones(80), "timestamps": np.arange(80) / 25, "fps": 25.}
        data["timestamps"][40:] += 100
        for config in refine_scale.DECODERS:
            with self.subTest(config=config):
                prediction = refine_scale.decode(probability, data, config)
                changed = dict(data, labels=np.full(80, "D"), BallPossession=np.ones(80))
                np.testing.assert_array_equal(prediction, refine_scale.decode(probability, changed, config))
                separate = []
                for left, right in ((0, 40), (40, 80)):
                    part = {"period": data["period"][left:right],
                            "timestamps": data["timestamps"][left:right], "fps": 25.}
                    separate.append(refine_scale.decode(probability[left:right], part, config))
                np.testing.assert_array_equal(prediction, np.concatenate(separate))

    def test_fresh_feature_cache_matches_public_prediction_apis(self):
        data = tracking()
        _, names = refine_features.features(data)
        base_model = {"feature_names": names, "classes": [0, 1, 2],
                      "estimator": SyntheticEstimator(), "config": {"switch_cost": 3}}
        scaled_model = {"feature_names": names, "classes": [0, 1, 2],
                        "estimator": SyntheticEstimator(.7),
                        "config": {"decoder": "separate", "team_cost": 3., "dead_cost": 8.,
                                   "boundary_power": 0., "switch_cost": 3}}
        cases = [("refine_features", "base.joblib", base_model, None),
                 ("refine_combined", "base.joblib", base_model, refine_combined.FIXED_CONFIG),
                 ("refine_scale", "scaled.joblib", scaled_model, None)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            joblib.dump(base_model, root / "base.joblib")
            joblib.dump(scaled_model, root / "scaled.joblib")
            with patch.object(refinement, "ROOT", root):
                cache = {}
                for module, model_path, model, config in cases:
                    record = {"module": module, "model_path": model_path,
                              "config": config or model["config"]}
                    normal = refinement.predict(data, record)
                    cached = refinement.predict(data, record, cache)
                    np.testing.assert_array_equal(cached, normal)
                with self.assertRaises(ValueError):
                    refinement.predict(dict(data), record, cache)


if __name__ == "__main__":
    unittest.main()
