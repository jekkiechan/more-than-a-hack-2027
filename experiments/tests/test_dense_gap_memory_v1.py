import unittest

import numpy as np
from unittest.mock import patch

from experiments.dense_gap_memory_v1 import apply_memory
from experiments.dense_gap_memory_v1_run import replay_control_cell
from experiments import supervised


class DenseGapMemoryContractTests(unittest.TestCase):
    def test_identity_without_reliable_anchor_and_on_observed_ball(self):
        p = np.array([[.55, .45, 0, 0], [.4, .3, .2, .1]])
        got = apply_memory(p, [True, False], [0, 1], [1, 1])
        np.testing.assert_array_equal(got, p)

    def test_memory_is_team_symmetric_and_confined_to_gap(self):
        p = np.array([[.9, .1, 0, 0], [.4, .3, .2, .1], [.9, .1, 0, 0]])
        obs = np.array([True, False, True])
        q = apply_memory(p, obs, [0, 1, 2], [1, 1, 1])
        swapped = p[:, [1, 0, 2, 3]]
        q_swapped = apply_memory(swapped, obs, [0, 1, 2], [1, 1, 1])
        np.testing.assert_allclose(q_swapped, q[:, [1, 0, 2, 3]])
        np.testing.assert_array_equal(q[obs], p[obs])
        self.assertGreater(q[1, 0], p[1, 0])
        self.assertAlmostEqual(q[1].sum(), 1.0)
        np.testing.assert_array_equal(q[1, 2:], p[1, 2:])

    def test_native_dense_had_probability_shape(self):
        p = np.array([[.9, .1, 0], [.4, .3, .3]])
        got = apply_memory(p, [True, False], [0, 1], [1, 1])
        self.assertEqual(got.shape[1], 3)
        self.assertAlmostEqual(got[1, 2], p[1, 2])
        self.assertAlmostEqual(got[1].sum(), 1.0)

    def test_period_break_and_unreliable_observation_clear_memory(self):
        p = np.array([[.9, .1, 0, 0], [.4, .3, .2, .1],
                      [.4, .3, .2, .1], [.4, .3, .2, .1]])
        got = apply_memory(p, [True, False, False, False],
                           [0, 1, 2, 3], [1, 1, 2, 2])
        np.testing.assert_array_equal(got[2:], p[2:])
        cleared = apply_memory(p, [True, False, True, False],
                               [0, 1, 2, 3], [1, 1, 1, 1])
        np.testing.assert_array_equal(cleared[3], p[3])

    def test_native_period_ids_are_preserved_for_scoring_and_block_ids_for_memory(self):
        inputs = {"period": np.array([1, 1, 1, 2]), "fps": 25.,
                  "timestamps": np.array([0., .04, .08, .12])}
        model_period = supervised.model_periods(inputs)
        np.testing.assert_array_equal(inputs["period"], [1, 1, 1, 2])
        np.testing.assert_array_equal(model_period, [0, 0, 0, 1])
        p = np.array([[.9, .1, 0], [.4, .3, .3], [.4, .3, .3], [.4, .3, .3]])
        got = apply_memory(p, [True, False, False, False],
                           inputs["timestamps"], model_period)
        self.assertGreater(got[1, 0], p[1, 0])
        np.testing.assert_array_equal(got[3], p[3])

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            apply_memory(np.ones((2, 2)) / 2, [True, False], [0, 1], [1, 1])
        with self.assertRaises(ValueError):
            apply_memory(np.ones((2, 4)) / 4, [True], [0, 1], [1, 1])
        with self.assertRaises(ValueError):
            apply_memory(np.ones((2, 3)) / 3, [True, False], [0, 1], [1, 1],
                         max_log_odds=float("nan"))
        with self.assertRaises(ValueError):
            apply_memory(np.ones((2, 3)) / 3, [1, 0], [0, 1], [1, 1])

    def test_runner_replays_two_seeds_and_probability_ensemble(self):
        p26 = np.array([[.8, .1, .1], [.1, .8, .1]])
        p27 = np.array([[.7, .2, .1], [.2, .7, .1]])
        ensemble = .5 * (p26 + p27)

        def fake_score(p, *_args):
            return {}, np.argmax(p, axis=1).astype(str)

        inputs = {"period": np.array([0, 0]), "fps": np.array(25.),
                  "timestamps": np.array([0., .04])}
        with patch("experiments.dense_gap_memory_v1_run.capacity._score_probability",
                   side_effect=fake_score), \
             patch("experiments.dense_gap_memory_v1_run._metric_parity"):
            result = replay_control_cell(item={"source": "provider", "match": "game"},
                view="clean", replicate="clean", inputs=inputs,
                labels=np.array(["H", "A"]), probabilities={2026: p26, 2027: p27},
                saved_control_rows={2026: {}, 2027: {}, "ensemble": {}},
                saved_control_predictions={2026: np.array(["0", "1"]),
                    2027: np.array(["0", "1"]), "ensemble": np.array(["0", "1"])})
        np.testing.assert_array_equal(result["probabilities"]["ensemble"], ensemble)
        self.assertEqual(set(result["rows"]), {2026, 2027, "ensemble"})

    def test_runner_rejects_control_probability_hash_mismatch(self):
        p = np.array([[.8, .1, .1]])
        inputs = {"period": np.array([1]), "fps": np.array(25.),
                  "timestamps": np.array([0.])}
        with self.assertRaisesRegex(ValueError, "probability hash"):
            replay_control_cell(item={"source": "provider", "match": "game"},
                view="clean", replicate="clean", inputs=inputs,
                labels=np.array(["H"]), probabilities={2026: p, 2027: p},
                saved_control_rows={2026: {"probabilities_sha256": "wrong"},
                    2027: {}, "ensemble": {}},
                saved_control_predictions={2026: np.array(["0"]),
                    2027: np.array(["0"]), "ensemble": np.array(["0"])})


if __name__ == "__main__":
    unittest.main()
