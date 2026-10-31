import unittest

import numpy as np

from experiments import native_short_spell as native
from experiments import possession_geometry
from experiments import supervised


class NativeShortSpellContractTests(unittest.TestCase):
    def test_native_geometry_is_label_independent_and_excludes_anchors(self):
        n = 80
        time = np.arange(n, dtype=np.float64) / 25.0
        players = np.zeros((n, 4, 2), dtype=np.float32)
        players[:, 0, 0] = time * 1.0
        players[:, 1, 0] = time * -0.7
        players[:, 2, 0] = 8.0 - time * 0.8
        players[:, 3, 0] = 10.0 + time * 0.5
        ball = np.column_stack((2.0 + time, np.sin(time))).astype(np.float32)
        inputs = {
            "ball": ball,
            "players": players,
            "teams": np.asarray([0, 0, 1, 1]),
            "period": np.zeros(n, dtype=np.int8),
            "timestamps": time,
            "fps": 25.0,
            "ball_observed": np.ones(n, dtype=bool),
            "player_observed": np.ones((n, 4), dtype=bool),
        }
        before, names = native.geometry_features(inputs)
        after, names_after = native.geometry_features(
            dict(inputs, labels=np.asarray(["H", "A", "D", "X"] * 20)))
        self.assertEqual(names, list(possession_geometry.GEOMETRY_NAMES))
        self.assertEqual(names, names_after)
        self.assertFalse(any("reliable_contact" in name for name in names))
        np.testing.assert_array_equal(before, after)

    def test_native_and_sampled_heads_train_and_preserve_frozen_prior(self):
        # Repeated three-frame away spells sit between 5 Hz sample points. The
        # native head sees the tracking cue and training labels; its matched
        # sampled head never sees either during its fit or inference.
        rng = np.random.default_rng(74)
        per_block = 500
        n = 2 * per_block
        period = np.r_[np.zeros(per_block, dtype=np.int8),
                       np.ones(per_block, dtype=np.int8)]
        target_home = np.ones(n, dtype=np.float32)
        for offset in (0, per_block):
            for start in range(offset + 2, offset + per_block, 25):
                target_home[start:start + 3] = 0.0

        names = list(possession_geometry.GEOMETRY_NAMES)
        geometry = rng.normal(0.0, 0.05, size=(n, len(names))).astype(np.float32)
        margin = names.index("home_geometry_control_margin")
        geometry[:, margin] = np.where(target_home > 0.5, 1.0, -1.0)
        geometry[::37, 0] = np.nan

        prior_logit = np.zeros(n, dtype=np.float32)
        dead = np.full(n, 0.25, dtype=np.float64)
        base = np.column_stack((0.5 * (1 - dead), 0.5 * (1 - dead), dead))
        base = base.astype(np.float64)

        indices = np.concatenate((np.arange(0, per_block, 5),
                                  np.arange(per_block, n, 5))).astype(np.int64)
        native_model = native.fit_residual(
            geometry, prior_logit, target_home, backbone_seed=2026,
            updates=48, batch_size=128)
        sampled_model = native.fit_residual(
            geometry[indices], prior_logit[indices], target_home[indices],
            backbone_seed=2026, updates=48, batch_size=128)

        native_residual = native.predict_symmetric_residual(geometry, native_model, names)
        sampled_rows = native.predict_symmetric_residual(
            geometry[indices], sampled_model, names)
        sampled_residual = native.upsample_blockwise(sampled_rows, indices, period)
        swapped_geometry = np.asarray(
            supervised.swap_team_features(geometry, names), dtype=np.float32)
        swapped_residual = native.predict_symmetric_residual(
            swapped_geometry, native_model, names)
        native_probability = native.apply_residual(base, native_residual)
        sampled_probability = native.apply_residual(base, sampled_residual)

        short = target_home == 0
        self.assertGreater(int(short.sum()), 0)
        self.assertLess(float(native_residual[short].mean()),
                        float(native_residual[~short].mean()) - 0.2)
        self.assertLess(abs(float(sampled_residual[short].mean() -
                                  sampled_residual[~short].mean())), 0.1)
        np.testing.assert_array_equal(native_probability[:, 2], base[:, 2])
        np.testing.assert_array_equal(sampled_probability[:, 2], base[:, 2])
        np.testing.assert_allclose(native_probability.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(sampled_probability.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(np.isfinite(native_probability).all())
        self.assertTrue(np.isfinite(sampled_probability).all())
        self.assertEqual(native_model["optimizer_updates"], sampled_model["optimizer_updates"])
        self.assertEqual(native_model["batch_size"], sampled_model["batch_size"])
        self.assertTrue(native_model["missing_values_use_train_fold_imputation_and_finite_mask"])
        np.testing.assert_allclose(swapped_residual, -native_residual, rtol=0, atol=1e-7)

    def test_provider_match_view_swap_mass_is_equal_at_both_rates(self):
        match_counts = {"sportec": 2, "metrica": 3, "skillcorner": 4}
        arm_support = {
            "native_rate": {"sportec": [101, 147], "metrica": [88, 132, 95],
                            "skillcorner": [110, 122, 105, 98]},
            "matched_5hz": {"sportec": [21, 30], "metrica": [18, 27, 20],
                            "skillcorner": [23, 25, 22, 20]},
        }
        for arm, providers in arm_support.items():
            for source, match_rows in providers.items():
                source_mass = 0.0
                for rows in match_rows:
                    for _view in range(4):
                        for _swap in range(2):
                            source_mass += float(native.equal_support_weights(
                                rows, match_counts[source], 4, 2).sum())
                self.assertAlmostEqual(source_mass, 1.0, places=6, msg=f"{arm}/{source}")
        for arm in arm_support:
            masses = [sum(float(native.equal_support_weights(
                rows, match_counts[source], 4, 2).sum())
                for rows in arm_support[arm][source] for _ in range(8))
                for source in arm_support[arm]]
            np.testing.assert_allclose(masses, np.ones(3), rtol=0, atol=1e-6)

    def test_zero_residual_is_bit_exact_and_nonzero_residual_keeps_dead_mass(self):
        probability = np.asarray([
            [0.216123456789, 0.483876543211, 0.3],
            [0.81, 0.09, 0.1],
            [0.0, 0.75, 0.25],
        ], dtype=np.float64)
        zero = native.apply_residual(probability, np.zeros(len(probability)))
        np.testing.assert_array_equal(zero, probability)

        result = native.apply_residual(probability, np.asarray([0.0, 0.4, -0.2]))
        np.testing.assert_array_equal(result[:, 2], probability[:, 2])
        np.testing.assert_array_equal(result[0], probability[0])
        np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-15)

    def test_sampled_residual_interpolation_resets_at_time_breaks(self):
        period = np.r_[np.zeros(10, dtype=np.int8), np.ones(10, dtype=np.int8)]
        indices = np.asarray([0, 5, 10, 15], dtype=np.int64)
        sampled = np.asarray([0.0, 0.0, 2.0, 2.0])
        result = native.upsample_blockwise(sampled, indices, period)
        self.assertEqual(float(result[9]), 0.0)
        self.assertEqual(float(result[10]), 2.0)


if __name__ == "__main__":
    unittest.main()
