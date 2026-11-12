import unittest

import numpy as np

from experiments import lgbm_public_runtime as runtime


def fixture(n=24):
    fps = 10.0
    t = np.arange(n, dtype=np.float64) / fps
    ball = np.column_stack((20 + t, 15 + np.sin(t)))
    players = np.zeros((n, 4, 2), dtype=np.float64)
    players[:, :, 0] = np.arange(4)[None, :] + t[:, None]
    players[:, :, 1] = np.arange(4)[None, :] * 2
    inputs = {
        "ball": ball,
        "players": players,
        "pitch": np.array([105.0, 68.0]),
        "fps": fps,
        "period": np.r_[np.zeros(n // 2), np.ones(n - n // 2)],
        "timestamps": t,
        "labels": np.array(["H", "A", "D", "X"] * (n // 4)),
    }
    p = np.tile([0.4, 0.4, 0.2], (n, 1))
    components = {"adopted": p.copy(), "calibrated": p.copy(), "ce": p.copy()}
    return inputs, components


class PublicRuntimeTests(unittest.TestCase):
    def test_features_are_frame_aligned_and_schema_order_is_stable(self):
        inputs, components = fixture()
        names, matrix = runtime.features(inputs, components)
        self.assertEqual(matrix.shape, (len(inputs["ball"]), len(names)))
        self.assertEqual(len(names), len(set(names)))
        names2, matrix2 = runtime.features(inputs, components)
        self.assertEqual(names, names2)
        np.testing.assert_equal(matrix, matrix2)

    def test_temporal_features_do_not_cross_model_block_boundary(self):
        inputs, components = fixture()
        names, original = runtime.features(inputs, components)
        changed_inputs = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                          for k, v in inputs.items()}
        changed_components = {k: np.array(v, copy=True) for k, v in components.items()}
        boundary = len(inputs["ball"]) // 2
        changed_inputs["ball"][boundary:, 0] += 900
        changed_inputs["players"][boundary:] += 500
        changed_components["adopted"][boundary:] = [0.025, 0.025, 0.95]
        changed_components["calibrated"][boundary:] = [0.05, 0.05, 0.9]
        names2, changed = runtime.features(changed_inputs, changed_components)
        self.assertEqual(names, names2)
        np.testing.assert_equal(original[:boundary], changed[:boundary])

    def test_features_ignore_labels_and_are_team_swap_invariant(self):
        inputs, components = fixture()
        names, original = runtime.features(inputs, components)
        changed = dict(inputs, labels=np.full(len(inputs["ball"]), "D"))
        names2, relabeled = runtime.features(changed, components)
        self.assertEqual(names, names2)
        np.testing.assert_equal(original, relabeled)

        swapped = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                   for k, v in inputs.items()}
        swapped["players"] = swapped["players"][:, ::-1]
        swapped_components = {k: np.array(v, copy=True) for k, v in components.items()}
        for value in swapped_components.values():
            value[:, [0, 1]] = value[:, [1, 0]]
        names3, symmetric = runtime.features(swapped, swapped_components)
        self.assertEqual(names, names3)
        np.testing.assert_allclose(original, symmetric, equal_nan=True)

    def test_prediction_ignores_labels_and_respects_team_swap(self):
        class Booster:
            def predict(self, matrix):
                return np.full(len(matrix), 0.2)

            def feature_name(self):
                return []

        inputs, components = fixture()
        pred = runtime.predict_labels(inputs, components, Booster())
        relabeled = dict(inputs, labels=np.full(len(inputs["ball"]), "X"))
        pred_relabeled = runtime.predict_labels(relabeled, components, Booster())
        np.testing.assert_equal(pred, pred_relabeled)

        swapped_inputs = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v
                          for k, v in inputs.items()}
        swapped_inputs["players"] = swapped_inputs["players"][:, ::-1]
        swapped_components = {k: np.array(v, copy=True) for k, v in components.items()}
        for value in swapped_components.values():
            value[:, [0, 1]] = value[:, [1, 0]]
        pred_swapped = runtime.predict_labels(swapped_inputs, swapped_components, Booster())
        inverse = np.where(pred == "H", "A", np.where(pred == "A", "H", pred))
        np.testing.assert_equal(pred_swapped, inverse)


if __name__ == "__main__":
    unittest.main()
