"""Inference isolation and repair contracts; no match data or labels are loaded."""

import unittest
from pathlib import Path
import tempfile

import numpy as np

from experiments import pipeline_eval, possession_pipeline as pipeline, refine_domain, supervised
from experiments.challenge_pathcrf import load_inputs
from experiments.tests.test_contracts import TargetGuard, tracking, window


class PipelineTests(unittest.TestCase):
    def test_complete_prediction_accepts_tracking_only_and_npz_skips_targets(self):
        from sklearn.ensemble import HistGradientBoostingClassifier

        data = tracking()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracking.npz"
            np.savez(path, **data, labels=np.array([object()], dtype=object))
            inputs = load_inputs(path)
        self.assertNotIn("labels", inputs)
        features, clean = pipeline.feature_views(inputs)
        y = np.arange(len(inputs["ball"])) % 3
        members = {}
        for name, (x, names) in features.items():
            estimator = HistGradientBoostingClassifier(max_iter=2, max_leaf_nodes=3, random_state=17).fit(x, y)
            members[name] = {"estimator": estimator, "feature_names": names, "classes": [0, 1, 2]}
        dx, names = refine_domain.features(clean)
        model = {"members": members, "dead_feature_names": names,
                 "dead_estimator": HistGradientBoostingClassifier(max_iter=2, random_state=17).fit(dx, y == 2),
                 "candidate": pipeline_eval.CANDIDATES[-1]}
        guarded = TargetGuard(inputs)
        guarded["labels"] = object()
        predictions = pipeline.predict(guarded, model)
        self.assertEqual(predictions.shape, (len(y),))
        self.assertTrue(np.isin(predictions, ["H", "A", "D"]).all())

    def test_gap_repair_requires_plausible_speed_and_internal_endpoints(self):
        values = np.array([[np.nan, np.nan], [0., 0.], [np.nan, np.nan], [1., 0.],
                           [np.nan, np.nan], [90., 0.], [np.nan, np.nan]])
        repaired, _, filled = pipeline._repair(values, np.ones(7), 10., 13., .2)
        np.testing.assert_allclose(repaired[2], [.5, 0.])
        self.assertTrue(np.isnan(repaired[[0, 4, 6]]).all())
        np.testing.assert_array_equal(np.flatnonzero(filled), [2])

    def test_spike_removal_preserves_persistent_displacement(self):
        values = np.array([[0., 0.], [.1, 0.], [40., 50.], [.3, 0.], [.4, 0.]])
        repaired, removed, _ = pipeline._repair(values, np.ones(5), 25., 65., .5)
        np.testing.assert_allclose(repaired[2], [.2, 0.])
        np.testing.assert_array_equal(np.flatnonzero(removed), [2])
        shifted = values.copy()
        shifted[2:] = [[40., 50.], [40.1, 50.], [40.2, 50.]]
        actual, removed, _ = pipeline._repair(shifted, np.ones(5), 25., 65., .5)
        np.testing.assert_array_equal(actual, shifted)
        self.assertFalse(removed.any())

    def test_observation_flags_and_input_immutability(self):
        data = tracking()
        data["ball_observed"][30:50] = False
        data["player_observed"] = np.isfinite(data["players"]).all(axis=2)
        data["player_observed"][30:50, 0] = False
        original = {k: v.copy() for k, v in data.items()}
        out, _ = pipeline.preprocess(data)
        self.assertTrue(np.isnan(out["ball"][30:50]).all())
        self.assertTrue(np.isnan(out["players"][30:50, 0]).all())
        for key in data:
            np.testing.assert_array_equal(data[key], original[key])

    def test_target_isolation_and_team_symmetry(self):
        data = tracking()
        guarded = TargetGuard(data)
        guarded.update({key: object() for key in TargetGuard.forbidden})
        original, _ = pipeline.feature_views(data)
        actual, _ = pipeline.feature_views(guarded)
        swapped, _ = pipeline.feature_views(dict(data, teams=1 - data["teams"]))
        for key, (x, names) in original.items():
            np.testing.assert_allclose(actual[key][0], x, equal_nan=True)
            self.assertEqual(names, swapped[key][1])
            np.testing.assert_allclose(supervised.swap_team_features(x, names), swapped[key][0],
                                       atol=1e-6, rtol=1e-5, equal_nan=True)

    def test_all_context_and_repair_respect_time_discontinuities(self):
        for boundary in ("period", "gap", "reset"):
            with self.subTest(boundary=boundary):
                data = tracking()
                if boundary == "period":
                    data["period"][40:] = 2
                elif boundary == "gap":
                    data["timestamps"][40:] += 10
                else:
                    data["timestamps"][40:] -= 4
                actual, _ = pipeline.feature_views(data)
                pieces = [pipeline.feature_views(window(data, a, b))[0] for a, b in ((0, 40), (40, 80))]
                for name in actual:
                    np.testing.assert_allclose(actual[name][0], np.concatenate([p[name][0] for p in pieces]),
                                               equal_nan=True)

    def test_control_memory_expires_and_resets(self):
        evidence = np.r_[1., np.zeros(99)]
        memory, age = pipeline._memory(evidence, np.r_[np.zeros(80), np.ones(20)], 10.)
        self.assertGreater(memory[20], 0)
        self.assertEqual(memory[51], 0)
        self.assertEqual(memory[80], 0)
        self.assertEqual(age[80], 5.)

    def test_degradation_is_reproducible_and_does_not_read_targets(self):
        data = tracking()
        guarded = TargetGuard(data)
        guarded["labels"] = object()
        first = pipeline.degraded_inputs(guarded, 17)
        second = pipeline.degraded_inputs(data, 17)
        self.assertNotIn("labels", first)
        for key in ("ball", "players", "ball_observed", "player_observed"):
            np.testing.assert_array_equal(first[key], second[key])

    def test_blend_preserves_conditional_team_votes(self):
        base = np.array([[.09, .01, .9], [0., 0., 1.]])
        context = np.array([[.1, .9, 0.], [.3, .7, 0.]])
        dead = np.array([.2, .4])
        p = pipeline.blend_probabilities(base, context, dead, .5, 1.)
        np.testing.assert_allclose(p, [[.4, .4, .2], [.24, .36, .4]])
        np.testing.assert_allclose(p.sum(axis=1), 1)
        swapped = pipeline.blend_probabilities(base[:, [1, 0, 2]], context[:, [1, 0, 2]], dead)
        np.testing.assert_allclose(swapped, pipeline.blend_probabilities(base, context, dead)[:, [1, 0, 2]])
        with self.assertRaises(ValueError):
            pipeline.blend_probabilities(base, context, np.array([np.nan, .4]))

    def test_selection_rejects_incomplete_or_duplicate_results_and_balances_sources(self):
        rows = [{"source": source, "match": path.stem, "condition": condition,
                 "candidate": candidate["name"], "spell_f1": float(source == "skillcorner"),
                 "macro_f1": .5, "d_f1": .5}
                for source in pipeline_eval.SOURCES for path in pipeline_eval.dead_transfer.SPLITS[source]["eval"]
                for condition in ("clean", "degraded") for candidate in pipeline_eval.CANDIDATES]
        self.assertAlmostEqual(pipeline_eval.summarize(rows)[0]["spell_f1"], 1 / 3)
        for invalid in (rows[:-1], rows + rows[:1]):
            with self.assertRaises(ValueError):
                pipeline_eval.summarize(invalid)


if __name__ == "__main__":
    unittest.main()
