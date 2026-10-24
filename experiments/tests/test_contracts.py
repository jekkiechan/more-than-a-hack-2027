"""Synthetic checks for target isolation, team symmetry, and temporal boundaries.

These checks never read an external match or its targets.
"""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from experiments import state_machine, supervised, temporal
from experiments import pathcrf_eval


class TargetGuard(dict):
    """Fail immediately if geometry code attempts to access any target field."""

    forbidden = {"labels", "possession", "BallPossession", "BallStatus", "ball_status"}

    def __getitem__(self, key):
        if key in self.forbidden:
            raise AssertionError(f"Model input computation accessed target {key}")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.forbidden:
            raise AssertionError(f"Model input computation accessed target {key}")
        return super().get(key, default)


def tracking(n=80):
    t = np.arange(n) / 10.0
    ball = np.column_stack((30 + 2 * np.sin(t), 20 + np.cos(t)))
    players = np.stack((ball + [0.8, 0.2], ball + [7, 3],
                        ball + [4, 0.5], ball + [10, 2]), axis=1)
    ball[14:16] = np.nan  # Short interpolation is permitted within a block.
    players[25:28, 1] = np.nan
    return {"ball": ball, "players": players, "teams": np.array([0, 0, 1, 1]),
            "period": np.ones(n, dtype=int), "timestamps": t,
            "fps": np.array(10.0), "pitch": np.array([105.0, 68.0]),
            "ball_observed": np.isfinite(ball).all(axis=1)}


def window(data, left, right):
    frame_fields = {"ball", "players", "period", "timestamps", "ball_observed"}
    return {key: value[left:right].copy() if key in frame_fields else value
            for key, value in data.items()}


class ModelContractTests(unittest.TestCase):
    def test_pathcrf_loader_never_deserializes_targets(self):
        data = tracking()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            # Accessing this object target with allow_pickle=False would fail.
            np.savez(path, **data, labels=np.array([object()], dtype=object))
            loaded = pathcrf_eval.load_geometry(path)
        self.assertEqual(set(loaded), set(pathcrf_eval.GEOMETRY_KEYS))
        self.assertNotIn("ball", loaded)

    def test_pathcrf_phases_do_not_access_ball_or_targets(self):
        data = tracking()
        guarded = TargetGuard({key: value for key, value in data.items() if key != "ball"})
        guarded.update({key: object() for key in TargetGuard.forbidden})
        expected = list(pathcrf_eval.feature_phases(data))
        actual = list(pathcrf_eval.feature_phases(guarded))
        self.assertEqual(len(expected), len(actual))
        for left, right in zip(expected, actual):
            self.assertEqual(left[:2], right[:2])
            np.testing.assert_allclose(left[2], right[2])
            np.testing.assert_array_equal(left[3], right[3])

    def test_features_and_state_predictions_do_not_access_targets(self):
        data = tracking()
        guarded = TargetGuard(data)
        guarded.update({key: object() for key in TargetGuard.forbidden})
        np.testing.assert_array_equal(state_machine.predict(data),
                                      state_machine.predict(guarded))
        np.testing.assert_allclose(supervised.features(data)[0],
                                   supervised.features(guarded)[0], equal_nan=True)
        np.testing.assert_allclose(temporal.features(data), temporal.features(guarded))

    def test_feature_extraction_does_not_modify_source_positions(self):
        data = tracking()
        ball, players = data["ball"].copy(), data["players"].copy()
        state_machine.prepare(data, interpolate_s=0.5)
        supervised.features(data)
        temporal.features(data)
        np.testing.assert_array_equal(data["ball"], ball)
        np.testing.assert_array_equal(data["players"], players)

    def test_home_away_augmentation_matches_recomputed_geometry(self):
        data = tracking()
        swapped = dict(data, teams=1 - data["teams"])
        original, names = supervised.features(data)
        recomputed, swapped_names = supervised.features(swapped)
        self.assertEqual(names, swapped_names)
        np.testing.assert_allclose(supervised.swap_team_features(original, names),
                                   recomputed, equal_nan=True)
        np.testing.assert_allclose(temporal.features(data)[:, temporal.SWAP],
                                   temporal.features(swapped))

    def test_halves_and_time_gaps_isolate_all_temporal_features(self):
        for boundary in ("half", "gap", "clock_reset"):
            with self.subTest(boundary=boundary):
                data = tracking()
                cut = 40
                data["ball"][cut - 1:cut + 1] = np.nan
                data["ball_observed"] = np.isfinite(data["ball"]).all(axis=1)
                if boundary == "half":
                    data["period"][cut:] = 2
                elif boundary == "gap":
                    data["timestamps"][cut:] += 100
                else:
                    data["timestamps"][cut:] -= 4
                pieces = [window(data, 0, cut), window(data, cut, len(data["ball"]))]
                expected_supervised = np.concatenate([supervised.features(p)[0] for p in pieces])
                expected_temporal = np.concatenate([temporal.features(p) for p in pieces])
                np.testing.assert_allclose(supervised.features(data)[0], expected_supervised,
                                           equal_nan=True)
                np.testing.assert_allclose(temporal.features(data), expected_temporal)
                np.testing.assert_array_equal(
                    state_machine.predict(data),
                    np.concatenate([state_machine.predict(p) for p in pieces]))


if __name__ == "__main__":
    unittest.main()
