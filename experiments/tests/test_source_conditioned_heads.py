import unittest

import numpy as np
import torch

from experiments import source_conditioned_heads as sch


class SourceConditionedHeadsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.net = sch.network(8, 3, [2, 3, 6, 7], d_hidden=2)
        self.x = torch.randn(2, 5, 8)
        self.y = torch.tensor([[0, 1, 2, -1, 0], [2, 1, 0, 2, -1]])

    def test_source_loss_routes_to_selected_heads_and_shared_encoder(self):
        out = self.net(self.x, sch.TARGET)
        sch.routed_loss(out, self.y, source=sch.TARGET).backward()
        self.assertTrue(any(p.grad is not None for p in self.net.ha_sequence.parameters()))
        self.assertTrue(any(p.grad is not None for p in self.net.dead_sequence.parameters()))
        self.assertTrue(all(p.grad is not None for p in self.net.ha[sch.TARGET].parameters()))
        self.assertTrue(all(p.grad is not None for p in self.net.dead[sch.TARGET].parameters()))
        self.assertTrue(all(p.grad is None for p in self.net.ha[sch.SOCCERNET].parameters()))
        self.assertTrue(all(p.grad is None for p in self.net.dead[sch.SOCCERNET].parameters()))

    def test_source_heads_start_from_cloned_weights(self):
        for part in (self.net.ha, self.net.dead):
            for target, gar in zip(part[sch.TARGET].parameters(),
                                   part[sch.SOCCERNET].parameters()):
                torch.testing.assert_close(target, gar)

    def test_dead_path_uses_only_player_columns(self):
        x = self.x.clone().requires_grad_(True)
        original = self.net(x, sch.TARGET)["d"]
        changed = x.detach().clone()
        changed[..., [0, 1, 4, 5]] += 1000
        altered = self.net(changed, sch.TARGET)["d"]
        torch.testing.assert_close(original.detach(), altered)
        grad = torch.autograd.grad(original.sum(), x)[0]
        self.assertTrue(torch.count_nonzero(grad[..., [0, 1, 4, 5]]) == 0)
        self.assertGreater(int(torch.count_nonzero(grad[..., [2, 3, 6, 7]])), 0)

    def test_target_inference_does_not_depend_on_soccernet_head(self):
        before = sch.target_probabilities(self.net, self.x).detach().numpy()
        with torch.no_grad():
            for parameter in self.net.ha[sch.SOCCERNET].parameters():
                parameter.add_(100)
            for parameter in self.net.dead[sch.SOCCERNET].parameters():
                parameter.sub_(100)
        after = sch.target_probabilities(self.net, self.x).detach().numpy()
        np.testing.assert_array_equal(before, after)
        np.testing.assert_allclose(after.sum(axis=-1), 1, atol=1e-6)

    def test_exported_state_restores_identical_predictions(self):
        clone = sch.network(8, 3, [2, 3, 6, 7], d_hidden=2)
        clone.load_state_dict({k: torch.as_tensor(v) for k, v in sch.export_state(self.net).items()})
        np.testing.assert_array_equal(sch.target_probabilities(self.net, self.x).detach().numpy(),
                                      sch.target_probabilities(clone, self.x).detach().numpy())
        np.testing.assert_array_equal(self.net(self.x, sch.SOCCERNET)["ha"].detach().numpy(),
                                      clone(self.x, sch.SOCCERNET)["ha"].detach().numpy())

    def test_unknown_targets_have_zero_loss_and_invalid_route_fails(self):
        outputs = self.net(self.x, sch.TARGET)
        zero = sch.routed_loss(outputs, torch.full_like(self.y, -1), source=sch.TARGET)
        self.assertEqual(float(zero.detach()), 0.)
        with self.assertRaises(ValueError):
            self.net(self.x, "organizer-provider-name")

    def test_player_columns_are_validated(self):
        self.assertEqual(sch.player_columns(["H_x", "A_x", "D_0_x", "D_1_x"]), [2, 3, 6, 7])
        with self.assertRaises(ValueError):
            sch.player_columns(["H_x", "A_x"])

    def test_separate_h_and_d_widths_are_configurable(self):
        net = sch.network(8, 96, [2, 3, 6, 7], d_hidden=24)
        self.assertEqual(net.ha_sequence.hidden_size, 96)
        self.assertEqual(net.dead_sequence.hidden_size, 24)
        self.assertEqual(net.dead_sequence.input_size, 4)
        h_arm = sch.network(8, 96, [2, 3, 6, 7], d_hidden=24, separate_d=False)
        self.assertEqual(list(h_arm.dead.keys()), [sch.TARGET])

if __name__ == "__main__":
    unittest.main()
