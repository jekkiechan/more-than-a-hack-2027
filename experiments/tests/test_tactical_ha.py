"""Synthetic contracts for the tactical H/A expert: no private arrays, fitting data or organizer labels."""
import unittest

import numpy as np

from experiments import tactical_ha_features as tactical
from experiments import tactical_ha_runtime as runtime


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    teams = np.r_[np.zeros(11, dtype=np.int8), np.ones(11, dtype=np.int8)]
    base = np.column_stack((np.r_[rng.uniform(5, 60, 11), rng.uniform(45, 100, 11)], rng.uniform(2, 66, 22)))
    players = base[None] + np.cumsum(rng.normal(0, .05, (n, 22, 2)), axis=0)
    ball = np.column_stack((52.5 + np.cumsum(rng.normal(0, .3, n)), 34 + np.cumsum(rng.normal(0, .2, n))))
    ball[200:260] = np.nan
    players[300:340, 3] = np.nan
    period = np.r_[np.ones(n // 2, dtype=np.int8), np.full(n - n // 2, 2, dtype=np.int8)]
    players[n // 2:, :, 0] = 105 - players[n // 2:, :, 0]  # teams change ends at half time
    timestamps = np.r_[np.arange(n // 2), np.arange(n - n // 2)] / 25.0
    return dict(ball=ball, players=players, teams=teams, period=period, fps=25.0, timestamps=timestamps,
                pitch=np.array([105., 68.]))


class Stub:
    """Deterministic stand-in estimator: P(H) rises with the first feature column (oriented ball x)."""
    def predict_proba(self, x):
        z = np.nan_to_num(np.asarray(x, dtype=float)[:, 0]) / 20.0
        p = 1 / (1 + np.exp(-z))
        return np.column_stack((1 - p, p))


class TacticalContracts(unittest.TestCase):
    def test_schema_and_orientation(self):
        data = synthetic()
        x, names = tactical.features(data)
        self.assertEqual(x.shape, (600, len(names)))
        self.assertEqual(len(set(names)), len(names))
        sign = tactical.orientation(data["players"], data["teams"], data["period"])
        self.assertTrue((sign[:300] == 1).all() and (sign[300:] == -1).all())

    def test_team_swap_exchanges_columns(self):
        data = synthetic()
        x, names = tactical.features(data)
        s, swapped_names = tactical.features(tactical.swap_inputs(data))
        self.assertEqual(names, swapped_names)
        index = {n: i for i, n in enumerate(names)}
        np.testing.assert_allclose(s[:, index["ball_x"]], -x[:, index["ball_x"]], equal_nan=True, atol=1e-5)
        for name in names:
            if name.startswith("H_"):
                np.testing.assert_allclose(s[:, index[name]], x[:, index["A_" + name[2:]]], equal_nan=True, atol=1e-4)

    def test_probability_is_team_antisymmetric(self):
        data = synthetic()
        x, names = tactical.features(data)
        model = {"estimator": Stub(), "names": names}
        p = runtime.tactical_probability(data, model)
        q = runtime.tactical_probability(tactical.swap_inputs(data), model)
        np.testing.assert_allclose(p + q, 1.0, atol=1e-9)

    def test_short_spells_are_protected(self):
        labels = np.array(list("D" * 50 + "H" * 200 + "A" * 50 + "H" * 125 + "D" * 25))
        weight = runtime.evidence_weight(labels, np.zeros(len(labels), dtype=int), 25.0)
        self.assertTrue((weight[50:250] == 1).all())      # 8 s spell keeps evidence
        self.assertTrue((weight[250:300] == 0).all())     # 2 s spell is protected
        self.assertTrue((weight[300:425] == 1).all())     # exactly 5 s is not shorter than 5 s
        self.assertTrue((weight[:50] == 1).all())

    def test_zero_weight_reproduces_mixture(self):
        probability = np.column_stack((np.linspace(.1, .9, 9), 1 - np.linspace(.1, .9, 9)))
        out = runtime.combine(probability, np.full(9, .99), np.zeros(9))
        np.testing.assert_allclose(out, probability, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
