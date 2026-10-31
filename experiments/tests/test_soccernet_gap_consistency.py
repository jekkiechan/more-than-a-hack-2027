import unittest

import numpy as np

from experiments import soccernet_gap_consistency as gap


def make_inputs(n=30):
    return {
        "ball": np.column_stack((np.arange(n, dtype=float), np.arange(n, dtype=float))),
        "ball_observed": np.ones(n, dtype=bool),
        "tracking_valid": np.ones(n, dtype=bool),
        "players": np.arange(n * 4, dtype=float).reshape(n, 2, 2),
        "period": np.zeros(n, dtype=int),
        "timestamps": np.arange(n, dtype=float) / 25,
        "fps": np.asarray(25.0),
    }


class SoccerNetGapConsistencyTests(unittest.TestCase):
    def test_corruption_is_seeded_and_preserves_other_inputs(self):
        raw = make_inputs()
        a, mask_a = gap.corrupt_ball_gaps(raw, rng=np.random.default_rng(8),
                                           start_probability=1.0, max_gap_frames=5)
        b, mask_b = gap.corrupt_ball_gaps(raw, rng=np.random.default_rng(8),
                                           start_probability=1.0, max_gap_frames=5)
        np.testing.assert_array_equal(mask_a, mask_b)
        np.testing.assert_array_equal(a["ball"], b["ball"])
        np.testing.assert_array_equal(a["players"], raw["players"])
        np.testing.assert_array_equal(raw["ball_observed"], np.ones(30, dtype=bool))
        np.testing.assert_array_equal(a["ball_observed"], ~mask_a)
        self.assertTrue(np.isnan(a["ball"][mask_a]).all())

    def test_never_corrupts_invalid_rows_or_crosses_segment_breaks(self):
        raw = make_inputs(12)
        raw["tracking_valid"][5] = False
        raw["period"][8:] = 1
        raw["timestamps"][10:] += 2.0
        _, mask = gap.corrupt_ball_gaps(raw, rng=np.random.default_rng(3),
                                         start_probability=1.0, max_gap_frames=12)
        self.assertFalse(mask[5])
        # Every valid run can be corrupted independently, including adjacent
        # rows on opposite sides of a break. The invalid row itself stays intact.
        self.assertTrue(mask[4])
        self.assertTrue(mask[6])

    def test_unknown_ball_is_not_reclassified_as_corruption(self):
        raw = make_inputs(8)
        raw["ball_observed"][2] = False
        raw["ball"][2] = np.nan
        view, mask = gap.corrupt_ball_gaps(raw, rng=np.random.default_rng(4),
                                            start_probability=1.0, max_gap_frames=1)
        self.assertFalse(mask[2])
        self.assertFalse(view["ball_observed"][2])
        reliable = gap.reliable_teacher_mask(raw)
        self.assertFalse(reliable[2])
        self.assertTrue(reliable[1])

    def test_ball_corruption_is_team_swap_equivariant(self):
        raw = make_inputs()
        swapped = dict(raw, players=raw["players"][:, ::-1].copy())
        a, ma = gap.corrupt_ball_gaps(raw, rng=np.random.default_rng(99), start_probability=.5)
        b, mb = gap.corrupt_ball_gaps(swapped, rng=np.random.default_rng(99), start_probability=.5)
        np.testing.assert_array_equal(ma, mb)
        np.testing.assert_array_equal(a["ball"], b["ball"])
        np.testing.assert_array_equal(b["players"], a["players"][:, ::-1])

    def test_teacher_kl_is_detached_and_empty_mask_is_zero(self):
        import torch

        clean = torch.tensor([[[4.0, -1.0], [-1.0, 3.0]]], requires_grad=True)
        corrupted = torch.zeros((1, 2, 2), requires_grad=True)
        loss = gap.consistency_loss(clean, corrupted, np.array([[True, False]]))
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNone(clean.grad)
        self.assertNotEqual(float(corrupted.grad.abs().sum()), 0.0)
        empty = gap.consistency_loss(clean, corrupted, np.zeros((1, 2), dtype=bool))
        self.assertEqual(float(empty.detach()), 0.0)
        uncertain = torch.zeros((1, 1, 2), requires_grad=True)
        ignored = gap.consistency_loss(uncertain, torch.ones_like(uncertain),
                                      np.ones((1, 1), dtype=bool))
        self.assertEqual(float(ignored.detach()), 0.0)

    def test_paired_feature_builder_recomputes_aligned_schema_from_raw(self):
        from experiments import learned_temporal as lt, refine_features

        n = 144
        t = np.arange(n, dtype=float) / 30
        players = np.zeros((n, 4, 2), dtype=float)
        for slot in range(4):
            players[:, slot, 0] = 10 + 15 * slot + t * (slot + 1)
            players[:, slot, 1] = 12 + 8 * slot + np.sin(t + slot)
        ball = np.column_stack((30 + t * 2, 28 + np.sin(t * 2)))
        observed = np.ones(n, dtype=bool)
        observed[20] = False
        ball[20] = np.nan
        valid = np.ones(n, dtype=bool)
        period = np.ones(n, dtype=np.int32)
        raw = {
            "ball": ball, "ball_observed": observed, "players": players,
            "teams": np.array([0, 0, 1, 1], dtype=np.int8), "period": period,
            "fps": np.asarray(30.), "timestamps": t,
            "pitch": np.asarray([105., 68.]), "tracking_valid": valid,
        }
        clean_team, clean_team_names = refine_features.features(raw)
        clean_player, clean_player_names = lt.player_features(raw)
        names = list(clean_team_names) + ["D_" + name for name in clean_player_names]
        saved_ball = raw["ball"].copy()
        record = gap.prepare_consistency_record(raw, seed=17, names=names)

        self.assertEqual(record["clean_x"].shape, record["gap_x"].shape)
        self.assertEqual(record["clean_x"].shape, (24, len(names)))
        self.assertEqual(record["names"], names)
        np.testing.assert_array_equal(record["indices"], np.arange(0, n, 6))
        np.testing.assert_array_equal(record["clean_x"],
                                      np.column_stack((clean_team, clean_player))[::6])
        np.testing.assert_array_equal(record["native_ball_observed"], observed)
        np.testing.assert_array_equal(record["native_tracking_valid"], valid)
        self.assertFalse(record["native_reliable_mask"][20])
        sampled_native = record["native_corruption_mask"][record["indices"]]
        np.testing.assert_array_equal(record["consistency_mask"],
                                      sampled_native & record["native_reliable_mask"][record["indices"]])
        np.testing.assert_array_equal(raw["ball"], saved_ball)
        self.assertEqual(record["provenance"]["recipe"], gap.RECIPE_VERSION)
        self.assertEqual(len(record["provenance"]["raw_input_sha256"]), 64)

        base_record = {
            "source": "soccernet", "x": record["clean_x"], "y": np.full(24, -1),
            "names": names, "indices": record["indices"], "native_n": np.asarray(n),
            "native_fps": np.asarray(30.), "full_timestamps": t.copy(),
            "full_period": period.copy(),
        }
        attached = gap.attach_consistency_view(base_record, raw, seed=17)
        self.assertIsNot(attached, base_record)
        self.assertIs(attached["y"], base_record["y"])
        np.testing.assert_array_equal(attached["gap_x"], record["gap_x"])
        np.testing.assert_array_equal(attached["consistency_mask"], record["consistency_mask"])
        self.assertIn("gap_provenance", attached)

    def test_paired_feature_builder_requires_explicit_validity(self):
        raw = make_inputs(24)
        del raw["tracking_valid"]
        with self.assertRaisesRegex(ValueError, "tracking_valid is required"):
            gap.prepare_consistency_record(raw, seed=1, names=[])

    def test_paired_feature_builder_allows_empty_reliable_mask_and_rejects_breaks(self):
        raw = make_inputs(144)
        raw["fps"] = np.asarray(30.)
        raw["timestamps"] = np.arange(144) / 30
        raw["ball"][:] = np.nan
        raw["ball_observed"][:] = False
        raw["teams"] = np.array([0, 1], dtype=np.int8)
        raw["pitch"] = np.array([105., 68.])
        from experiments import learned_temporal as lt, refine_features
        team, team_names = refine_features.features(raw)
        player, player_names = lt.player_features(raw)
        names = list(team_names) + ["D_" + name for name in player_names]
        record = gap.prepare_consistency_record(raw, seed=5, names=names)
        self.assertFalse(record["consistency_mask"].any())
        self.assertFalse(record["native_reliable_mask"].any())

        discontinuous = make_inputs(144)
        discontinuous["fps"] = np.asarray(30.)
        discontinuous["timestamps"] = np.arange(144) / 30
        discontinuous["timestamps"][80:] += 1
        with self.assertRaisesRegex(ValueError, "discontinuity; split it first"):
            gap.prepare_consistency_record(discontinuous, seed=5, names=names)
        invalid = make_inputs(144)
        invalid["fps"] = np.asarray(30.)
        invalid["timestamps"] = np.arange(144) / 30
        invalid["tracking_valid"][80] = False
        with self.assertRaisesRegex(ValueError, "every row valid"):
            gap.prepare_consistency_record(invalid, seed=5, names=names)


if __name__ == "__main__":
    unittest.main()
