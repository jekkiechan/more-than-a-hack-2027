from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import v2_calibration_seed_readout_v2 as readout


class CalibrationSeedReadoutTests(unittest.TestCase):
    def test_duplicate_best_arm_collapses(self):
        self.assertEqual(readout._select({"selected_candidate": "a",
                                          "best_numerical_arm_before_guards": "a"}), ["a"])

    def test_seed_scores_use_model_family_and_phase_mean(self):
        per_seed = {
            ("adop", 0): {2026: np.array([[.8, .1, .1]])},
            ("adop", 1): {2026: np.array([[.2, .7, .1]])},
            ("cand", 0): {2026: np.array([[.1, .8, .1]])},
            ("cand", 1): {2026: np.array([[.7, .2, .1]])},
        }
        view = {"period": np.array([1]), "timestamps": np.array([0.]), "fps": 10.}
        seen = {}

        def decode(members, period, fps, config):
            seen["probability"] = members.copy()
            seen["config"] = config
            return np.array(["H"])

        with (patch.object(readout.decoder, "decode", side_effect=decode),
              patch.object(readout.audit, "score", return_value={"spell_f1": .5})):
            score = readout._seed_cell_scores(
                view, np.array(["H"]), np.array([0]), [object(), object()], per_seed,
                {"family": "adop", "config": audit.CONFIG}, 2026)

        np.testing.assert_allclose(seen["probability"][0, 0], [.5, .4, .1])
        self.assertEqual(seen["config"], audit.CONFIG)
        self.assertEqual(score["spell_f1"], .5)


if __name__ == "__main__":
    unittest.main()
