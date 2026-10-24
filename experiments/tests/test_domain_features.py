"""The anti-artifact D head must not consume ball or target information."""

import unittest
import numpy as np
from experiments import refine_domain


class Guard(dict):
    def __getitem__(self, key):
        if key in ("ball", "ball_observed", "labels", "BallStatus", "BallPossession"):
            raise AssertionError(f"Forbidden D-head input {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in ("ball", "ball_observed", "labels", "BallStatus", "BallPossession"):
            raise AssertionError(f"Forbidden D-head input {key}")
        return super().get(key, default)


def sample():
    t = np.arange(100)/25
    players = np.zeros((len(t), 8, 2))
    for i in range(8):
        players[:, i, 0] = 20+i*7+np.sin(t+i)
        players[:, i, 1] = 10+i*5+np.cos(t+i)
    return {"players": players, "teams": np.repeat([0, 1], 4),
            "period": np.ones(len(t)), "timestamps": t, "fps": 25., "pitch": np.array([105., 68.])}


class DomainFeatureTests(unittest.TestCase):
    def test_no_ball_or_target_dependency(self):
        data = sample()
        guarded = Guard(data)
        guarded.update({name: object() for name in ("ball", "labels", "BallStatus")})
        np.testing.assert_allclose(refine_domain.features(data)[0], refine_domain.features(guarded)[0], equal_nan=True)

    def test_player_order_and_home_away_invariance(self):
        data = sample()
        order = np.arange(8)[::-1]
        other = dict(data, players=data["players"][:, order], teams=1-data["teams"][order])
        np.testing.assert_allclose(refine_domain.features(data)[0], refine_domain.features(other)[0],
                                   rtol=1e-6, atol=1e-7, equal_nan=True)

    def test_period_windows_are_isolated(self):
        data = sample()
        data["period"][50:] = 2
        parts = [{key: value[start:stop] if key in ("players", "period", "timestamps") else value
                  for key, value in data.items()} for start, stop in ((0, 50), (50, 100))]
        np.testing.assert_allclose(refine_domain.features(data)[0],
                                   np.concatenate([refine_domain.features(p)[0] for p in parts]), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
