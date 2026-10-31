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

    def test_mixed_batch_routes_each_row_without_cross_head_gradients(self):
        self.net.eval()
        source_mask = torch.tensor([False, True])
        output = self.net(self.x, source_mask)
        target = self.net(self.x[:1], sch.TARGET)
        gar = self.net(self.x[1:], sch.SOCCERNET)
        torch.testing.assert_close(output["ha"][0], target["ha"][0])
        torch.testing.assert_close(output["ha"][1], gar["ha"][0])
        grads = torch.autograd.grad(output["ha"][0].sum(),
            tuple(self.net.ha[sch.SOCCERNET].parameters()), allow_unused=True)
        self.assertTrue(all(value is None or torch.count_nonzero(value) == 0 for value in grads))

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
        self.net.eval()
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
        self.net.eval(); clone.eval()
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
        self.assertEqual(net.dead_sequence.input_size, 8)
        h_arm = sch.network(8, 96, [2, 3, 6, 7], d_hidden=24, separate_d=False)
        self.assertEqual(list(h_arm.dead.keys()), [sch.TARGET])
        c_arm = sch.network(8, 96, [2, 3, 6, 7], d_hidden=24,
                             separate_ha=False, separate_d=False)
        self.assertEqual(list(c_arm.ha.keys()), [sch.TARGET])

    def test_teacher_initialization_matches_existing_ha_and_d_paths(self):
        from experiments import factorized_had as fh
        from experiments import factorized_had as fh
        from experiments import soccernet_capacity_model as cap

        torch.manual_seed(72)
        teacher_net = fh.network(8, [2, 3, 6, 7], True)
        teacher = {"input_features": 8, "player_indices": [2, 3, 6, 7],
                   "state": sch.export_state(teacher_net)}
        net = sch.network(8, 24, [2, 3, 6, 7], d_hidden=24, input_dropout=0.)
        sch.initialize_from_teacher(net, teacher)
        base_ha = cap.network(8, 24, input_dropout=0.)
        cap._copy_teacher_function(teacher, base_ha)
        probe = torch.randn(2, 5, 8)
        torch.testing.assert_close(net.ha_sequence(probe)[0], base_ha.sequence(probe)[0])
        torch.testing.assert_close(net.ha[sch.TARGET](net.ha_sequence(probe)[0]),
                                   base_ha.output(base_ha.sequence(probe)[0]))
        from experiments.sn_dense_train import _dead_logit
        torch.testing.assert_close(net(self.x, sch.TARGET)["d"], _dead_logit(teacher_net, self.x))
        torch.testing.assert_close(net.ha[sch.TARGET].weight, net.ha[sch.SOCCERNET].weight)
        torch.testing.assert_close(net.dead[sch.TARGET].weight, net.dead[sch.SOCCERNET].weight)

    def test_initialization_preserves_dense_d2_torch_random_stream(self):
        from experiments import factorized_had as fh
        from experiments import soccernet_capacity_model as cap

        torch.manual_seed(314)
        _ = cap.network(8, 24, 0.1)
        _ = fh.network(8, [2, 3, 6, 7], True)
        baseline_state = torch.random.get_rng_state()
        torch.manual_seed(314)
        _ = sch.network(8, 24, [2, 3, 6, 7], d_hidden=24)
        candidate_state = torch.random.get_rng_state()
        self.assertTrue(torch.equal(baseline_state, candidate_state))

        teacher_net = fh.network(8, [2, 3, 6, 7], True)
        teacher = {"input_features": 8, "player_indices": [2, 3, 6, 7],
                   "state": sch.export_state(teacher_net)}
        torch.manual_seed(811)
        baseline_ha = cap.network(8, 96, 0.1)
        cap._copy_teacher_function(teacher, baseline_ha)
        torch.manual_seed(811)
        candidate = sch.network(8, 96, [2, 3, 6, 7], d_hidden=24)
        sch.initialize_from_teacher(candidate, teacher)
        for name, expected in baseline_ha.sequence.state_dict().items():
            torch.testing.assert_close(candidate.ha_sequence.state_dict()[name], expected,
                                       rtol=0, atol=0)
        torch.testing.assert_close(candidate.ha[sch.TARGET].weight,
                                   baseline_ha.output.weight, rtol=0, atol=0)

    def test_target_inference_matches_frozen_capacity_inference(self):
        from experiments import factorized_had as fh
        from experiments import source_conditioned_train as sct
        from experiments import soccernet_capacity_model as cap
        from experiments import supervised

        names = ["H_value", "A_value", "D_0_value", "D_1_value"]
        indices = sch.player_columns(names)
        teacher_net = fh.network(8, indices, True)
        teacher = {"kind": "factorized_had", "input_features": 8,
                   "player_indices": indices, "player_only": True,
                   "state": sch.export_state(teacher_net), "feature_names": names,
                   "median": np.zeros(4), "scale": np.ones(4)}
        candidate = sch.network(8, 96, indices, d_hidden=24, separate_ha=False,
                                separate_d=False)
        sch.initialize_from_teacher(candidate, teacher)
        source_model = {"kind": "source_conditioned_dense", "arm": "C",
            "state": sch.export_state(candidate), "hidden": 96, "d_hidden": 24,
            "input_dropout": 0.1, "separate_ha": False, "separate_d": False,
            "median": teacher["median"], "scale": teacher["scale"],
            "feature_names": names, "input_features": 8, "player_indices": indices}
        base_ha = cap.network(8, 96, 0.1)
        cap._copy_teacher_function(teacher, base_ha)
        capacity = {"kind": "soccernet_capacity", "state": sch.export_state(base_ha),
            "hidden": 96, "input_features": 8, "training": {"input_dropout": 0.1},
            "median": teacher["median"], "scale": teacher["scale"], "feature_names": names}
        rng = np.random.default_rng(7)
        record = {"x": rng.normal(size=(151, 4)).astype(np.float32), "names": names,
                  "period": np.r_[np.ones(75), np.full(76, 2)].astype(np.int32)}
        actual = sct.predict(record, source_model)
        expected = cap.predict(record, capacity, teacher)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
        swapped = dict(record, x=supervised.swap_team_features(record["x"], names))
        np.testing.assert_allclose(sct.predict(swapped, source_model),
                                   actual[:, [1, 0, 2]], rtol=0, atol=2e-6)
        broken = dict(record, x=record["x"].copy())
        broken["x"][75:] += 100
        np.testing.assert_array_equal(sct.predict(broken, source_model)[:75], actual[:75])

    def test_shared_control_train_path_matches_frozen_d2_updates(self):
        from experiments import sn_dense_train as sn
        from experiments import source_conditioned_train as sct
        from experiments.tests.test_sn_dense_train import teacher, old_record, sn_record

        initial = teacher(seed=901)
        old_records = [old_record("old_a", 1), old_record("old_b", 2)]
        gar_records = [sn_record("gar_a", 3), sn_record("gar_b", 4)]
        baseline = sn.train(initial, old_records, gar_records, width=24, lr=1e-4,
            seed=12, steps=3, batch_size=8, ha_target="dense", train_d=True,
            d_lr=1e-4, require_full_native=True, verbose=False)
        candidate = sct.train(initial, old_records, gar_records, arm="C", width=24,
            d_width=24, lr=1e-4, seed=12, steps=3, batch_size=8, fraction=0.5,
            d_lr=1e-4, input_dropout=0.1, require_full_native=True, verbose=False)
        self.assertEqual(baseline["exposures"], candidate["exposures"])
        for k, v in zip(baseline["loss_intervals"], candidate["loss_intervals"]):
            self.assertEqual(k["step"], v["step"])
            for name, expected in k.items():
                self.assertAlmostEqual(expected, v[name], places=7)
        model = candidate["model"]
        for key, value in baseline["model"]["state"].items():
            mapped = key.replace("sequence.", "ha_sequence.").replace("output.", "ha.target.")
            np.testing.assert_array_equal(model["state"][mapped], value)
        for key, value in baseline["model"]["d_state"].items():
            if key.startswith("sequence."):
                np.testing.assert_allclose(model["state"][key.replace("sequence.", "dead_sequence.")],
                                           value, rtol=0, atol=2e-8)
        np.testing.assert_array_equal(model["state"]["dead.target.weight"],
                                      baseline["model"]["d_state"]["output.weight"][2:3])
        np.testing.assert_array_equal(model["state"]["dead.target.bias"],
                                      baseline["model"]["d_state"]["output.bias"][2:3])
        for record in (old_records[0], gar_records[0]):
            np.testing.assert_allclose(sct.predict(record, model),
                                       sn.predict(record, baseline["model"], initial),
                                       rtol=0, atol=2e-6)

if __name__ == "__main__":
    unittest.main()
