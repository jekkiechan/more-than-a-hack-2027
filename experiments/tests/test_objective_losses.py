"""Contracts for train-only spell weighting and probability transition losses."""
import unittest
from unittest.mock import patch

import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt
from experiments import objective_losses as ol


def record(n=100):
    rng = np.random.default_rng(31)
    y = np.resize(np.array([0, 0, 1, 1, 2, 2, -1, 1], dtype=np.int8), n)
    period = np.zeros(n, dtype=np.int32)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    names = ["H_gap", "A_gap", "D_speed", "D_accel"]
    windows = list(lt.chunks(dict(y=y, period=period)))
    return dict(source="Sportec", match="synthetic", x=x, alternate=x + .1,
                y=y, period=period, names=names,
                replace_rows=np.zeros(n, dtype=bool),
                replace_windows=np.zeros(len(windows), dtype=bool),
                full_y=np.repeat(y, 5), full_period=np.repeat(period, 5),
                indices=np.arange(n, dtype=np.int64) * 5, fps=25.0)


class ObjectiveLossContracts(unittest.TestCase):
    def test_capped_sqrt_spell_factors_are_full_resolution_and_censor_aware(self):
        self.assertEqual(ol.capped_spell_weight(.01), 2.0)
        self.assertEqual(ol.capped_spell_weight(100.0), .5)
        self.assertEqual(ol.capped_spell_weight(5.0), 1.0)

        # Complete one-second H and two-second A spells; H runs touching X
        # are censored on either side and must remain neutral.
        full_y = np.array([2]*25 + [0]*25 + [1]*50 + [2]*25
                          + [0]*25 + [-1]*25 + [0]*25 + [2]*25, dtype=np.int8)
        period = np.zeros(len(full_y), dtype=np.int32)
        indices = np.arange(0, len(full_y), 5, dtype=np.int64)
        sampled = full_y[indices]
        factors = ol.spell_duration_factors(full_y, period, indices, sampled, 25)

        eligible = (indices >= 25) & (indices < 100)
        self.assertAlmostEqual(float(factors[eligible].mean()), 1.0, places=6)
        h_factor = float(factors[(indices >= 25) & (indices < 50)].mean())
        a_factor = float(factors[(indices >= 50) & (indices < 100)].mean())
        self.assertAlmostEqual(h_factor / a_factor, 2 / np.sqrt(2.5), places=5)
        self.assertTrue(np.all(factors[sampled == 2] == 1))
        self.assertTrue(np.all(factors[sampled < 0] == 1))
        self.assertTrue(np.all(factors[(indices >= 125) & (indices < 150)] == 1))
        self.assertTrue(np.all(factors[(indices >= 175) & (indices < 200)] == 1))

        swapped_full = np.where((full_y >= 0) & (full_y < 2), 1-full_y, full_y)
        swapped_sampled = np.where((sampled >= 0) & (sampled < 2), 1-sampled, sampled)
        np.testing.assert_array_equal(
            ol.spell_duration_factors(swapped_full, period, indices, swapped_sampled, 25),
            factors)

    def test_spell_factors_censor_at_period_and_time_block_edges(self):
        full_y = np.array([2]*25 + [0]*25 + [1]*25 + [2]*25, dtype=np.int8)
        full_period = np.array([0]*50 + [1]*50, dtype=np.int32)
        indices = np.arange(0, len(full_y), 5, dtype=np.int64)
        sampled = full_y[indices]
        factors = ol.spell_duration_factors(full_y, full_period, indices, sampled, 25)
        self.assertTrue(np.all(factors[(indices >= 25) & (indices < 50)] == 1))
        self.assertTrue(np.all(factors[(indices >= 50) & (indices < 75)] == 1))

    def test_transition_masks_unknowns_and_full_native_interval(self):
        y = np.array([0, 0, 1, 2, 2, 1, -1, 1, 0], dtype=np.int8)
        period = np.array([0]*7 + [1]*2, dtype=np.int32)
        pair = ol.transition_targets(y, period, core_size=len(y))
        np.testing.assert_array_equal(np.flatnonzero(pair["dead_mask"]), [0, 1, 2, 3, 4, 7])
        np.testing.assert_array_equal(np.flatnonzero(pair["live_mask"]), [0, 1, 7])
        np.testing.assert_array_equal(pair["dead_target"][[0, 1, 2, 3, 4, 7]], [0, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(pair["live_target"][[0, 1, 7]], [0, 1, 1])

        # The sampled endpoints are known, but an X/unknown native frame in
        # between invalidates the pair. The same applies across a time break.
        sampled = np.array([0, 1], dtype=np.int8)
        indices = np.array([0, 5], dtype=np.int64)
        same_block = np.zeros(2, dtype=np.int32)
        hidden_unknown = np.array([0, 0, -1, 1, 1, 1], dtype=np.int8)
        native_period = np.zeros(6, dtype=np.int32)
        masked = ol.transition_targets(
            sampled, same_block, core_size=2, full_y=hidden_unknown,
            full_period=native_period, indices=indices)
        self.assertFalse(masked["dead_mask"][0])
        self.assertFalse(masked["live_mask"][0])

        across_break_period = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        sampled_period = across_break_period[indices]
        broken = ol.transition_targets(
            sampled, sampled_period, core_size=2,
            full_y=np.array([0, 0, 0, 0, 1, 1], dtype=np.int8),
            full_period=across_break_period, indices=indices)
        self.assertFalse(broken["dead_mask"][0])
        self.assertFalse(broken["live_mask"][0])

    def test_core_seam_edge_is_owned_once_and_final_edge_is_safe(self):
        y = np.array([0]*50 + [1]*5, dtype=np.int8)
        period = np.zeros(len(y), dtype=np.int32)
        full_y = np.repeat(y, 5)
        full_period = np.repeat(period, 5)
        indices = np.arange(len(y), dtype=np.int64) * 5
        first = ol.transition_targets(y, period, start=0, size=50, core_size=50,
                                      full_y=full_y, full_period=full_period,
                                      indices=indices)
        self.assertTrue(first["live_mask"][49])
        self.assertEqual(first["live_target"][49], 1)
        last = ol.transition_targets(y, period, start=50, size=5, core_size=50,
                                     full_y=full_y, full_period=full_period,
                                     indices=indices)
        self.assertFalse(last["dead_mask"][4])
        self.assertFalse(last["live_mask"][4])

    def test_group_balance_and_absent_groups_keep_fixed_half_mass(self):
        import torch
        single = ol.group_balanced_binary_loss(
            torch.tensor([2.0]), torch.tensor([0.0]),
            torch.tensor([True]), torch.tensor([1.0]))
        self.assertEqual(float(single), 1.0)  # absent changed group contributes zero
        both = ol.group_balanced_binary_loss(
            torch.tensor([2.0, 4.0]), torch.tensor([0.0, 1.0]),
            torch.tensor([True, True]), torch.tensor([1.0, 1.0]))
        self.assertEqual(float(both), 3.0)
        self.assertIsNone(ol.group_balanced_binary_loss(
            torch.tensor([1.0]), torch.tensor([0.0]),
            torch.tensor([False]), torch.tensor([0.0])))

        records = [dict(source="s"), dict(source="s")]
        pairs = [
            dict(dead_mask=np.ones(4, bool), dead_target=np.array([0, 0, 0, 1])),
            dict(dead_mask=np.ones(4, bool), dead_target=np.array([0, 1, 1, 1])),
        ]
        weights = ol._record_pair_weights(records, pairs, "dead")
        for weight, pair in zip(weights, pairs):
            for group in (0, 1):
                self.assertAlmostEqual(float(weight[pair["dead_target"] == group].sum()), .5)

    def test_dead_transition_gradient_is_player_only_and_spell_weight_leaves_d_rows(self):
        import torch
        torch.manual_seed(8)
        model = fh.network(4, [2, 3], player_only=True)
        x = torch.randn(1, 3, 4, requires_grad=True)
        log_prob = model(x)
        p_dead = log_prob[..., 2].exp()
        p_change = p_dead[:, :-1] * (1-p_dead[:, 1:]) + (1-p_dead[:, 1:]) * p_dead[:, :-1]
        dead_transition = torch.nn.functional.binary_cross_entropy(
            p_change, torch.ones_like(p_change))
        gradient = torch.autograd.grad(dead_transition, x)[0]
        self.assertTrue(torch.equal(gradient[..., :2], torch.zeros_like(gradient[..., :2])))
        self.assertGreater(float(gradient[..., 2:].abs().sum()), 0.0)

        targets = torch.full((1, 3), 2, dtype=torch.long)
        base = ol.frame_nll_losses(log_prob, targets)
        weighted = ol.frame_nll_losses(
            log_prob, targets, spell_factors=torch.tensor([[.5, 2.0, 1.5]]),
            spell_coefficient=1.0)
        self.assertTrue(torch.equal(base, weighted))
        base_gradient = torch.autograd.grad(base.sum(), log_prob, retain_graph=True)[0]
        weighted_gradient = torch.autograd.grad(weighted.sum(), log_prob)[0]
        self.assertTrue(torch.equal(base_gradient, weighted_gradient))

    def test_zero_coefficients_delegate_to_exact_factorized_control(self):
        r = record()
        with patch.object(lt, "EPOCHS", 1):
            baseline = fh.fit([r], player_only=True, seed=2026)
            zero = ol.fit([r], player_only=True, seed=2026, objective="combined",
                          spell_coefficient=0.0, transition_coefficient=0.0)
        self.assertEqual(baseline.keys(), zero.keys())
        for key in baseline["state"]:
            np.testing.assert_array_equal(baseline["state"][key], zero["state"][key])
        self.assertEqual(baseline["history"], zero["history"])
        np.testing.assert_array_equal(fh.predict(r, baseline), ol.predict(r, zero))

    def test_spell_transition_and_combined_arms_fit_without_changing_inference(self):
        r = record()
        with patch.object(lt, "EPOCHS", 1):
            models = [ol.fit([r], player_only=True, seed=2026, objective=arm)
                      for arm in ("spell_sqrt", "transition", "combined")]
        for model in models:
            p = ol.predict(r, model)
            self.assertEqual(p.shape, (len(r["y"]), 3))
            np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
