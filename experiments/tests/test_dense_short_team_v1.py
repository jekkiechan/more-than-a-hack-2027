"""Synthetic contracts for dense short-spell team weighting."""
import unittest
import importlib.util

import numpy as np

from experiments import dense_short_team_v1 as study


def record(full_y, full_period=None, stride=1, fps=25.0, native_fps=None):
    full_y = np.asarray(full_y, dtype=np.int8)
    if full_period is None:
        full_period = np.zeros(len(full_y), dtype=np.int32)
    full_period = np.asarray(full_period)
    indices = np.arange(0, len(full_y), stride, dtype=np.int64)
    return {
        "full_y": full_y,
        "full_period": full_period,
        "indices": indices,
        "y": full_y[indices].copy(),
        "period": full_period[indices].copy(),
        "fps": np.asarray(fps),
        **({"native_fps": np.asarray(native_fps)} if native_fps is not None else {}),
    }


class ShortTeamContracts(unittest.TestCase):
    def test_complete_spells_at_or_below_second_included(self):
        # 25 frames is exactly one second at 25 Hz; 26 frames exceeds it.
        y = np.array([2] + [0] * 25 + [1] * 26 + [2], dtype=np.int8)
        selected = study.complete_short_spell_mask(y, np.zeros(len(y)), 25.0)
        self.assertEqual(int(selected.sum()), 25)
        self.assertTrue(selected[1:26].all())
        self.assertFalse(selected[26:52].any())

    def test_excludes_unknown_censored_and_period_edge_spells(self):
        y = np.array([-1, 0, 0, 1, 1, 0, 2, 1, 1, 2], dtype=np.int8)
        period = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        selected = study.complete_short_spell_mask(y, period, 25.0)
        # H immediately after X and the A run ending at the period boundary
        # are censored. The final short A run is bounded by known D labels.
        np.testing.assert_array_equal(np.flatnonzero(selected), np.array([7, 8]))

    def test_maps_native_fps_weights_to_sampled_rows_and_normalizes_live_mass(self):
        y = np.array([2] + [0] * 20 + [1] * 20 + [0] * 30 + [1] * 20 + [2], dtype=np.int8)
        rec = record(y, stride=5, fps=5.0, native_fps=25.0)
        weights = study.short_spell_frame_weights(rec)
        self.assertEqual(weights.shape, rec["y"].shape)
        live = (rec["y"] == 0) | (rec["y"] == 1)
        self.assertTrue(np.all(np.isin(weights[live], (1.0, 2.0))))
        self.assertEqual(float(weights[live].max()), 2.0)
        # X/unknown is not part of known H/A training mass and never receives
        # the boost, even when adjacent to a short spell.
        unknown = rec["y"] == -1
        self.assertTrue(np.all(weights[unknown] == 1.0))
        swapped = dict(rec, y=np.where(rec["y"] == 0, 1,
                                      np.where(rec["y"] == 1, 0, rec["y"])))
        swapped["full_y"] = np.where(y == 0, 1, np.where(y == 1, 0, y))
        np.testing.assert_array_equal(weights, study.short_spell_frame_weights(swapped))

    def test_old_provider_fps_fallback_and_bad_mapping(self):
        rec = record([2, 0, 0, 1, 1, 2], stride=1, fps=25.0)
        weights = study.short_spell_frame_weights(rec)
        self.assertEqual(len(weights), len(rec["y"]))
        rec["indices"] = np.array([0, 1])
        with self.assertRaises(ValueError):
            study.short_spell_frame_weights(rec)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_all_one_weights_replay_baseline_per_window_ce(self):
        import torch
        from torch.nn import functional as F

        torch.manual_seed(11)
        logits = torch.randn(3, 8, 2)
        targets = torch.tensor([[0, 1, -1, 0, 1, 1, 0, 1],
                                [1, 1, 0, -1, 0, 1, 0, 0],
                                [0, 1, 0, 1, 0, 1, 0, -1]])
        weights = torch.ones_like(targets, dtype=torch.float32)
        got = study.weighted_window_ha_loss(logits, targets, weights)
        safe = targets.clamp(min=0)
        row = F.cross_entropy(logits.transpose(1, 2), safe, reduction="none")
        expected = (row * (targets >= 0)).sum(dim=1) / (targets >= 0).sum(dim=1)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_one_step_weight_one_replays_d2_update(self):
        from experiments import sn_dense_train as sn
        from experiments.tests.test_sn_dense_train import old_record, sn_record, teacher

        initial = teacher(2026)
        old_records = [old_record("o1", 1), old_record("o2", 2)]
        new_records = [sn_record("g1", 3), sn_record("g2", 4), sn_record("g3", 5)]
        control = sn.train(initial, old_records, new_records, width=96, lr=3e-4,
            seed=2026, steps=2, fraction=0.5, batch_size=32, ha_target="dense",
            train_d=True, d_lr=1e-4, input_dropout=0.1,
            require_full_native=True, verbose=False)["model"]
        candidate = study.train_candidate(initial, old_records, new_records,
            seed=2026, steps=2, short_multiplier=1.0)["model"]
        for key in control["state"]:
            np.testing.assert_array_equal(control["state"][key], candidate["state"][key])
        for key in control["d_state"]:
            np.testing.assert_array_equal(control["d_state"][key], candidate["d_state"][key])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is unavailable")
    def test_two_x_changes_ha_but_not_dead_head_on_short_spells(self):
        from experiments.tests.test_sn_dense_train import old_record, sn_record, teacher

        # Every live run is one 5 Hz sample (=0.2s at 25 native FPS), bounded
        # by known H/A/D labels. This guarantees the treatment has support.
        def short_labels(rec):
            rec = dict(rec)
            y = np.resize(np.array([0, 1, 2, 1], dtype=np.int8), len(rec["y"]))
            rec["y"] = y
            rec["full_y"] = np.repeat(y, 5)
            rec["period"] = np.zeros(len(y), dtype=np.int32)
            rec["full_period"] = np.zeros(len(rec["full_y"]), dtype=np.int32)
            rec["indices"] = np.arange(len(y), dtype=np.int64) * 5
            return rec

        initial = teacher(2026)
        old_records = [short_labels(old_record("o1", 1)), short_labels(old_record("o2", 2))]
        new_records = [short_labels(sn_record("g1", 3)), short_labels(sn_record("g2", 4)),
                       short_labels(sn_record("g3", 5))]
        one_x = study.train_candidate(initial, old_records, new_records,
            seed=2026, steps=2, short_multiplier=1.0)["model"]
        two_x = study.train_candidate(initial, old_records, new_records,
            seed=2026, steps=2, short_multiplier=2.0)["model"]
        self.assertTrue(any(not np.array_equal(one_x["state"][k], two_x["state"][k])
                            for k in one_x["state"]))
        for key in one_x["d_state"]:
            np.testing.assert_array_equal(one_x["d_state"][key], two_x["d_state"][key])

    def test_training_declaration_and_paths_are_fixed(self):
        declared = study.declaration()
        self.assertEqual(declared["common"]["candidate_fits"], 6)
        self.assertEqual(declared["candidate_weight"]["multiplier"], 2.0)
        self.assertEqual(study.fit_id("leave_out_metrica", 2026),
                         "short_team_2x__metrica__s2026")
        self.assertEqual(study.candidate_model_path("leave_out_sportec", 2027).name,
                         "short_team_2x__sportec__s2027.joblib")
        self.assertTrue(study.control_model_path("leave_out_skillcorner", 2026)
                        .name.endswith("control_D2__skillcorner__s2026.joblib"))

    def test_eligibility_summary_counts_only_sampled_short_spell_rows(self):
        y = np.array([-1] + [0] * 10 + [1] * 10 + [0] * 40 + [2], dtype=np.int8)
        rec = record(y, stride=1, fps=25.0)
        summary = study.eligibility_summary([rec])
        self.assertEqual(summary["native_eligible_short_ha_frames"], 10)
        self.assertEqual(summary["sampled_eligible_short_ha_frames"], 10)
        self.assertGreater(summary["windows_with_eligible_short_ha"], 0)


if __name__ == "__main__":
    unittest.main()
