"""Fast contracts for label-free target adaptation (synthetic data, CPU torch)."""
import unittest

import numpy as np

from experiments import factorized_had as fh, target_adaptation as ta

NAMES = ["D_a", "D_b", "H_x", "A_x", "home_z", "flag"]


def make_model(seed=7):
    import torch
    torch.manual_seed(seed)
    n = len(NAMES)
    cols = fh.player_columns(NAMES)
    net = fh.network(2 * n, cols, True)
    return dict(kind="factorized_had", player_only=True, seed=seed, player_indices=cols,
                state={k: v.detach().numpy() for k, v in net.state_dict().items()},
                median=np.zeros(n, np.float32), scale=np.ones(n, np.float32),
                input_features=2 * n, feature_names=NAMES)


def make_record(n=160, seed=0, with_labels=False):
    rng = np.random.default_rng(seed)
    x = rng.normal(2.0, 3.0, (n, len(NAMES))).astype(np.float32)
    x[:, -1] = rng.integers(0, 2, n)
    x[rng.random(x.shape) < .05] = np.nan
    x[:, -1] = np.where(np.isnan(x[:, -1]), 0, x[:, -1])
    r = dict(source="s", match="m", x=x, names=NAMES, period=np.ones(n, np.int8),
             indices=np.arange(n), native_n=n, fps=5.0)
    if with_labels:
        r["y"] = rng.integers(0, 3, n).astype(np.int8)
        r["full_y"] = r["y"]
    return r


class TargetAdaptationTests(unittest.TestCase):
    def test_blind_removes_every_label_field(self):
        b = ta.blind(make_record(with_labels=True))
        self.assertNotIn("y", b)
        self.assertNotIn("full_y", b)

    def test_adaptation_is_independent_of_labels(self):
        model, record = make_model(), make_record()
        labelled = dict(make_record(), y=np.random.default_rng(1).integers(0, 3, 160))
        a = ta.target_statistics(ta.blind(labelled), model)
        b = ta.target_statistics(record, model)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        np.testing.assert_array_equal(ta.sampled_probability(labelled, model),
                                      ta.sampled_probability(record, model))

    def test_target_norm_with_training_stats_reproduces_control_exactly(self):
        model, record = make_model(), make_record()
        same = ta.with_normalization(model, model["median"].copy(), model["scale"].copy())
        np.testing.assert_array_equal(ta.sampled_probability(record, model),
                                      ta.sampled_probability(record, same))

    def test_target_statistics_rules(self):
        model, record = make_model(), make_record()
        model["scale"] = np.full(len(NAMES), 50., np.float32)
        median, scale, use = ta.target_statistics(record, model)
        self.assertFalse(use[-1])                         # binary flag left alone
        self.assertEqual(median[-1], model["median"][-1])
        self.assertEqual(scale[-1], model["scale"][-1])
        self.assertTrue(np.all(scale >= .1 * model["scale"] - 1e-6))   # floor
        self.assertTrue(use[:-1].all())
        # H/A and home_ pooling over swap keeps H and A statistics equal
        self.assertAlmostEqual(float(median[2]), float(median[3]), places=5)
        self.assertAlmostEqual(float(median[4]), 0., places=5)

    def test_pseudo_targets_threshold_and_swap(self):
        p = np.array([[.95, .03, .02], [.5, .4, .1], [.02, .03, .95], [.05, .90, .05]])
        label, keep = ta.pseudo_targets(p)
        self.assertEqual(label.tolist(), [0, 0, 2, 1])
        self.assertEqual(keep.tolist(), [True, False, True, True])

    def test_target_windows_keep_only_confident_frames(self):
        model, record = make_model(), make_record()
        probs = np.tile([.2, .3, .5], (160, 1))
        self.assertEqual(ta.target_windows(record, probs, model), [])
        probs[:10] = [.96, .02, .02]
        items = ta.target_windows(record, probs, model)
        self.assertEqual(len(items), 2)                    # original + swapped window
        self.assertEqual(sum(float(w.sum()) for _, _, w in items), 20.)
        self.assertEqual([int(t[0]) for _, t, _ in items], [0, 1])

    def test_self_train_deterministic_changes_weights_and_keeps_stats(self):
        model, record = make_model(), make_record()
        probs = np.tile([.97, .02, .01], (160, 1))
        items = ta.target_windows(record, probs, model)
        a = ta.self_train(model, items, None, seed=3, tag="t")
        b = ta.self_train(model, items, None, seed=3, tag="t")
        for k in a["state"]:
            np.testing.assert_array_equal(a["state"][k], b["state"][k])
        self.assertTrue(any(not np.array_equal(a["state"][k], model["state"][k])
                            for k in a["state"] if "weight" in k))
        np.testing.assert_array_equal(a["median"], model["median"])
        self.assertEqual(len(a["adaptation_history"]), 2)

    def test_stable_seed_deterministic(self):
        self.assertEqual(ta.stable_seed(1, "a", 0), ta.stable_seed(1, "a", 0))
        self.assertNotEqual(ta.stable_seed(1, "a", 0), ta.stable_seed(1, "a", 1))


if __name__ == "__main__":
    unittest.main()
