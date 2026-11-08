"""Focused contracts for the scoped v2 capacity training adapter."""
import unittest

import numpy as np

from experiments import factorized_had as fh
from experiments import sn_dense_train as sn
from experiments import soccernet_capacity_model as cap
from experiments import v2_data_capacity as study


class V2DataCapacityAdapterTests(unittest.TestCase):
    def test_width_adapter_matches_frozen_width96_two_step_replay(self):
        import torch
        torch.set_num_threads(1)
        names = ["H_a", "A_a", "D_a", "D_b"]
        torch.manual_seed(11)
        teacher_net = fh.network(8, [2, 3, 6, 7], True)
        teacher = dict(kind="factorized_had", player_only=True,
            player_indices=[2, 3, 6, 7],
            state={k: v.detach().cpu().numpy().copy()
                   for k, v in teacher_net.state_dict().items()},
            median=np.zeros(4, np.float32), scale=np.ones(4, np.float32),
            input_features=8, feature_names=names)

        def record(source, match, seed):
            n = 100
            rng = np.random.default_rng(seed)
            y = np.array(([0, 0, 1, 1, 2, -1] * 17)[:n], np.int64)
            period = np.zeros(n, np.int32)
            x = rng.normal(size=(n, 4)).astype(np.float32)
            return dict(x=x, alternate=x.copy(), y=y, period=period,
                names=names, source=source, match=match, full_y=y.copy(),
                full_period=period.copy(), indices=np.arange(n), fps=25., native_fps=25.)

        old_records = [record("sportec", "old1", 31), record("metrica", "old2", 32)]
        gar_records = [record("gar", "gar1", 33), record("gar", "gar2", 34)]
        args = (teacher, old_records, gar_records)
        kwargs = dict(width=96, seed=2026, steps=2, fraction=.5, batch_size=4,
                      ha_target="dense", train_d=True, require_full_native=True,
                      log_every=0, verbose=False)
        reference = sn.train(*args, **kwargs)
        with study._training_adapter(96) as adapted:
            replay = adapted(*args, **kwargs)
        self.assertEqual(reference["config"], replay["config"])
        self.assertEqual(reference["loss_intervals"], replay["loss_intervals"])
        for part in ("state", "d_state"):
            for name, value in reference["model"][part].items():
                np.testing.assert_array_equal(value, replay["model"][part][name])

    def test_width_adapter_widens_teacher_function_and_rejects_other_widths(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(7)
        teacher_net = fh.network(8, [2, 3, 6, 7], True).eval()
        teacher = {"state": {k: v.detach().cpu().numpy().copy()
                              for k, v in teacher_net.state_dict().items()}}
        widened = study._network(8, 192, 0.)
        cap._copy_teacher_function(teacher, widened)
        widened.eval()
        inputs = torch.randn(2, 16, 8)
        with torch.no_grad():
            expected = teacher_net.output(teacher_net.sequence(inputs)[0])[..., :2]
            actual = widened(inputs)
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-6)
        with self.assertRaisesRegex(ValueError, "width must be"):
            study._adapt_train()(None, [], [], width=128)
        with self.assertRaisesRegex(ValueError, "width 96 or 192"):
            study._network(4, 128, .1)

    def test_adapter_restores_shared_globals(self):
        original_train, original_network = sn.train, sn.cap.network
        with study._training_adapter(192):
            self.assertIsNot(sn.train, original_train)
            self.assertIsNot(sn.cap.network, original_network)
        self.assertIs(sn.train, original_train)
        self.assertIs(sn.cap.network, original_network)

    def test_fit_paths_are_arm_and_fold_scoped(self):
        path = study.fit_path("width192_capacity", study.FOLDS[0], study.SEEDS[0])
        self.assertIn("width192_capacity", str(path))
        with self.assertRaises(ValueError):
            study.fit_path("unfrozen", study.FOLDS[0], study.SEEDS[0])


if __name__ == "__main__":
    unittest.main()
