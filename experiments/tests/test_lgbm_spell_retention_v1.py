import unittest

import numpy as np

from experiments import lgbm_public_runtime, lgbm_spell_retention_v1, supervised


def tracking_fixture(n=40):
    fps = 10.0
    t = np.arange(n, dtype=np.float64) / fps
    ball = np.column_stack((30 + t, 20 + np.sin(t)))
    players = np.zeros((n, 4, 2), dtype=np.float64)
    players[:, :, 0] = np.arange(4)[None, :] + t[:, None]
    players[:, :, 1] = np.arange(4)[None, :] * 2 + 5
    return {
        "ball": ball,
        "ball_observed": np.ones(n, dtype=bool),
        "players": players,
        "pitch": np.array([105.0, 68.0]),
        "fps": fps,
        "period": np.r_[np.zeros(n // 2), np.ones(n - n // 2)],
        "timestamps": t,
    }


def components(n):
    rows = np.tile([0.42, 0.38, 0.20], (n, 1))
    calibrated = np.tile([0.35, 0.45, 0.20], (n, 1))
    ce = np.tile([0.45, 0.35, 0.20], (n, 1))
    return {"adopted": rows, "calibrated": calibrated, "ce": ce}


class ConstantBooster:
    def __init__(self, value):
        self.value = value

    def predict(self, matrix):
        return np.full(len(matrix), self.value, dtype=np.float64)

    def feature_name(self):
        return []


class SpellRetentionTests(unittest.TestCase):
    def test_decode_matches_frozen_public_runtime(self):
        inputs = tracking_fixture()
        comp = components(len(inputs["ball"]))
        booster = ConstantBooster(0.31)
        expected = lgbm_public_runtime.predict_labels(inputs, comp, booster)
        cell = {
            "adopted": comp["adopted"],
            "calibrated": comp["calibrated"],
            "ce": comp["ce"],
            "model_period": supervised.model_periods(inputs),
            "fps": inputs["fps"],
            "ball": inputs["ball"],
            "pitch": inputs["pitch"],
        }
        actual = lgbm_spell_retention_v1.decode(cell, np.full(len(inputs["ball"]), 0.31))
        np.testing.assert_array_equal(actual, expected)

    def test_aggregation_averages_realizations_then_matches_then_providers(self):
        # Provider A: match A1 has two realizations averaging .5, A2 has one at 1;
        # provider mean is .75. Provider B has three zero realizations, mean 0.
        # The final equal-provider value is .375, regardless of raw row counts.
        rows = {}
        for view in ("clean", "feed_like_v1", "feed_like_v2"):
            offset = {"clean": 0.0, "feed_like_v1": 0.1, "feed_like_v2": 0.2}[view]
            values = [
                ("Alpha", "A1", "1", 0.0), ("Alpha", "A1", "2", 1.0),
                ("Alpha", "A2", "1", 1.0),
                ("Beta", "B1", "1", 0.0), ("Beta", "B1", "2", 0.0),
                ("Beta", "B1", "3", 0.0),
            ]
            for provider, match, rep, score in values:
                key = f"{provider}__{match}__{view}__{rep}"
                rows[key] = {metric: score + offset for metric in lgbm_spell_retention_v1.METRICS}

        forward = lgbm_spell_retention_v1.aggregate(rows)
        reversed_result = lgbm_spell_retention_v1.aggregate(dict(reversed(list(rows.items()))))
        for view in ("clean", "feed_like_v1", "feed_like_v2"):
            expected = 0.375 + {"clean": 0.0, "feed_like_v1": 0.1, "feed_like_v2": 0.2}[view]
            self.assertAlmostEqual(forward[view]["spell_f1"], expected)
            self.assertEqual(set(forward[view]["spell_f1/providers"]), {"Alpha", "Beta"})
            self.assertAlmostEqual(forward[view]["spell_f1/providers"]["Alpha"], 0.75 + expected - 0.375)
            self.assertAlmostEqual(forward[view]["spell_f1/providers"]["Beta"], expected - 0.375)
            self.assertAlmostEqual(forward[view]["spell_f1"], reversed_result[view]["spell_f1"])
            for provider in ("Alpha", "Beta"):
                self.assertAlmostEqual(forward[view]["spell_f1/providers"][provider],
                                       reversed_result[view]["spell_f1/providers"][provider])


if __name__ == "__main__":
    unittest.main()
