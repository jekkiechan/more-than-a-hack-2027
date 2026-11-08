from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments import turnover_supervision_eval as study


class TurnoverSupervisionEvalContracts(unittest.TestCase):
    def test_native_inventory_requires_exact_35_cells(self):
        inventory = {key: {} for key in study.EXPECTED_CELLS}
        study._validate_cell_inventory(inventory)
        inventory.pop(next(iter(inventory)))
        with self.assertRaisesRegex(ValueError, "exact frozen 35-cell cohort"):
            study._validate_cell_inventory(inventory)

    def test_training_inventory_is_three_arms_by_three_folds_by_two_seeds(self):
        keys = study.expected_fit_keys()
        self.assertEqual(len(keys), 18)
        self.assertEqual({arm for arm, _, _ in keys}, set(study.ARMS))
        self.assertEqual({seed for _, _, seed in keys}, set(study.SEEDS))

    def test_seed_and_phase_pooling_order_is_explicit(self):
        p2026 = [np.array([[.8, .1, .1]]), np.array([[.2, .7, .1]])]
        p2027 = [np.array([[.2, .7, .1]]), np.array([[.6, .3, .1]])]
        pooled = study._candidate_probability({2026: p2026, 2027: p2027})
        expected = np.mean([(p2026[i] + p2027[i]) / 2 for i in range(2)], axis=0)
        np.testing.assert_allclose(pooled, expected, atol=1e-15, rtol=0)
        seed_only = study._candidate_probability({2026: p2026, 2027: p2027}, seed=2026)
        np.testing.assert_allclose(seed_only, np.mean(p2026, axis=0), atol=1e-15, rtol=0)
        self.assertFalse(np.array_equal(seed_only, pooled))
        five_phase = {seed: [np.array([[.8, .1, .1]]) if i % 2 == 0 else
                             np.array([[.2, .7, .1]]) for i in range(5)]
                      for seed in study.SEEDS}
        expected_five = np.mean([five_phase[2026][i] for i in range(5)], axis=0)
        np.testing.assert_allclose(study._candidate_probability(five_phase), expected_five,
                                   atol=1e-15, rtol=0)

    def test_family_weights_match_the_approved_conditional_mixture(self):
        ce = np.array([[.6, .3, .1]])
        calibrated = np.array([[.2, .6, .2]])
        predecessor = np.array([[.5, .25, .25]])
        actual = study.candidate_mixture(ce, calibrated, predecessor)
        expected = .5 * (ce[:, :2] / ce[:, :2].sum(axis=1, keepdims=True))
        expected += .375 * (calibrated[:, :2] / calibrated[:, :2].sum(axis=1, keepdims=True))
        expected += .125 * (predecessor[:, :2] / predecessor[:, :2].sum(axis=1, keepdims=True))
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)

    def test_inference_adapter_strips_target_fields(self):
        phase = {"x": np.zeros((2, 1)), "names": ["feature"],
            "period": np.array([0, 0]), "indices": np.array([0, 1]),
            "native_n": 2, "fps": 25., "y": np.array([0, 1]),
            "labels": np.array(["H", "A"]), "full_y": np.array([0, 1]),
            "target": np.array([0, 1])}
        inputs = {"period": np.array([0, 0]), "timestamps": np.array([0., .04]), "fps": 25.}

        def predict(record, model, teacher):
            self.assertFalse({"y", "labels", "full_y", "target"} & set(record))
            return np.array([[.7, .2, .1], [.2, .7, .1]])

        sampled = np.array([[.7, .2, .1], [.2, .7, .1]])
        with patch.object(study.sn, "predict", side_effect=predict), \
             patch.object(study.old, "upsample", return_value=sampled) as upsample:
            actual = study._predict_phase(phase, {}, {}, inputs)
        upsample.assert_called_once()
        np.testing.assert_array_equal(actual, sampled)


if __name__ == "__main__":
    unittest.main()
