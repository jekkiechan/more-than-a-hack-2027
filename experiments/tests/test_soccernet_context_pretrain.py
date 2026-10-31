"""Synthetic contracts for the opt-in SoccerNet context auxiliary loss."""
import unittest

import numpy as np

from experiments import soccernet_context_pretrain as ctx


class SoccerNetContextPretrainTests(unittest.TestCase):
    def test_masking_hides_values_and_natural_masks_only_on_valid_rows(self):
        import torch
        # Two features, with the second naturally absent at t=3.
        values = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
        observed = torch.ones_like(values)
        observed[0, 3, 1] = 0
        x = torch.cat((values, observed), dim=-1)
        valid = torch.tensor([[False, True, True, True, True, False]])
        masked, target, loss_mask = ctx.mask_context(x, valid, seed=19,
                                                      mask_fraction=1.0, span=1)
        selected = masked[0, :, :2].eq(0).all(dim=-1)
        self.assertTrue(torch.equal(selected, valid[0]))
        self.assertTrue(torch.equal(masked[0, valid[0], 2:], torch.zeros(4, 2)))
        self.assertFalse(loss_mask[0, 3, 1])
        self.assertTrue(loss_mask[0, 3, 0])
        self.assertTrue(torch.equal(target, values))
        self.assertFalse(loss_mask[0, 0].any())
        self.assertFalse(loss_mask[0, 5].any())

    def test_exact_budget_and_spans_stop_at_invalid_time_rows(self):
        import torch
        values = torch.arange(20, dtype=torch.float32).reshape(1, 10, 2)
        x = torch.cat((values, torch.ones_like(values)), dim=-1)
        # The valid rows form two short segments. No mask span may bridge row 4.
        valid = torch.tensor([[True, True, True, True, False,
                               True, True, True, True, True]])
        masked, _, loss_mask = ctx.mask_context(x, valid, seed=11,
                                                  mask_fraction=1.0, span=3)
        selected = masked[0, :, :2].eq(0).all(dim=-1)
        self.assertEqual(int(selected.sum()), int(valid.sum()))
        self.assertTrue(torch.equal(selected, valid[0]))
        self.assertFalse(loss_mask[0, 4].any())

    def test_loss_updates_recurrent_and_disposable_head(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(4)
        net = __import__("experiments.soccernet_capacity_model", fromlist=["network"]).network(6, 24)
        head = ctx.reconstruction_head(24, 3)
        raw = torch.randn(2, 12, 3)
        finite = torch.ones_like(raw)
        normalized = torch.cat((raw, finite), dim=-1)
        valid = torch.ones(2, 12, dtype=torch.bool)
        masked, target, mask = ctx.mask_context(normalized, valid, seed=5)
        before = [p.detach().clone() for p in net.sequence.parameters()]
        loss = ctx.masked_reconstruction_loss(net, head, masked, target, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in net.sequence.parameters()))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in head.parameters()))

    def test_auxiliary_api_is_target_independent_and_keeps_checkpoint_shape(self):
        import torch
        torch.set_num_threads(1)
        from experiments import soccernet_capacity_model as cap
        torch.manual_seed(19)
        net = cap.network(4, 24)
        state_keys = set(net.state_dict())
        head = ctx.reconstruction_head(24, 2)
        x = torch.randn(1, 8, 4)
        x[..., 2:] = 1
        valid = torch.ones(1, 8, dtype=torch.bool)
        # Labels are not arguments to the training-step API; callers can change
        # sparse labels without changing the deterministic masking contract.
        labels_a = np.zeros(8, dtype=np.int8)
        labels_b = np.full(8, -1, dtype=np.int8)
        self.assertEqual(labels_a.shape, labels_b.shape)
        masked_a, target_a, loss_mask_a = ctx.mask_context(x, valid, seed=8)
        masked_b, target_b, loss_mask_b = ctx.mask_context(x, valid, seed=8)
        self.assertTrue(torch.equal(masked_a, masked_b))
        self.assertTrue(torch.equal(target_a, target_b))
        self.assertTrue(torch.equal(loss_mask_a, loss_mask_b))
        ctx.masked_reconstruction_loss(net, head, masked_a, target_a, loss_mask_a)
        self.assertEqual(set(net.state_dict()), state_keys)
        self.assertEqual(net.output.out_features, 2)


if __name__ == "__main__":
    unittest.main()
