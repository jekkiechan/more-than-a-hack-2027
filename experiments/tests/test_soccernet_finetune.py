"""Synthetic contracts for sparse conditional H/A warm-start training."""
import copy
import unittest

import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt
from experiments import soccernet_finetune as sf, supervised


NAMES = ["H_gap", "A_gap", "D_speed", "D_accel"]


def checkpoint(seed=2026):
    import torch

    torch.manual_seed(seed)
    columns = fh.player_columns(NAMES)
    model = fh.network(len(NAMES) * 2, columns, player_only=True)
    return dict(
        kind="factorized_had", player_only=True, seed=seed,
        player_indices=columns,
        state={key: value.detach().numpy().copy()
               for key, value in model.state_dict().items()},
        median=np.zeros(len(NAMES), dtype=np.float32),
        scale=np.ones(len(NAMES), dtype=np.float32),
        input_features=len(NAMES) * 2, feature_names=NAMES.copy(),
    )


def old_record(match="old-1", period_break=False):
    rng = np.random.default_rng(11)
    x = rng.normal(size=(100, len(NAMES))).astype(np.float32)
    y = np.resize(np.array([0, 0, 1, 2, 2, -1, 1, 1], dtype=np.int8), len(x))
    period = np.zeros(len(x), dtype=np.int32)
    if period_break:
        period[50:] = 1
    windows = list(lt.chunks(dict(y=y, period=period)))
    return dict(source="sportec", match=match, x=x, alternate=x + .05,
                y=y, period=period, names=NAMES.copy(),
                replace_windows=np.array([True] + [False] * (len(windows) - 1)),
                replace_rows=np.zeros(len(x), dtype=bool))


def new_record(match="game-1", anchor=12, period_break=False):
    rng = np.random.default_rng(anchor + 3)
    x = rng.normal(size=(24, len(NAMES))).astype(np.float32)
    y = np.full(24, -1, dtype=np.int8)
    y[anchor] = 0
    period = np.zeros(24, dtype=np.int32)
    if period_break:
        period[12:] = 1
        y[:] = -1
        y[5] = 0
        y[18] = 1
    return dict(source="soccernet", match=match, x=x, y=y,
                period=period, names=NAMES.copy())


class SoccerNetFineTuneContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_training_is_deterministic_and_keeps_checkpoint_normalizer(self):
        initial = checkpoint()
        before = copy.deepcopy(initial)
        old = [old_record("old-1"), old_record("old-2")]
        new = [new_record("game-1"), new_record("game-2", anchor=8)]
        first = sf.train(initial, old, new, .25, 71, updates=2)
        second = sf.train(initial, old, new, .25, 71, updates=2)
        self.assertEqual(first["kind"], "factorized_had")
        self.assertEqual(first["optimizer_updates"], 2)
        np.testing.assert_array_equal(first["median"], initial["median"])
        np.testing.assert_array_equal(first["scale"], initial["scale"])
        self.assertEqual(first["feature_names"], initial["feature_names"])
        self.assertEqual(first["finetune"]["soccernet_windows_per_batch"], 16)
        self.assertEqual(first["finetune"]["old_windows_per_batch"], 48)
        self.assertEqual(first["finetune"]["soccernet_matches"], 2)
        for key in first["state"]:
            np.testing.assert_array_equal(first["state"][key], second["state"][key])
            np.testing.assert_array_equal(initial["state"][key], before["state"][key])
        self.assertEqual(first["finetune"], second["finetune"])

    def test_zero_fraction_does_not_read_new_records(self):
        initial = checkpoint()
        result = sf.train(initial, [old_record()], [None], 0.0, 4, updates=2)
        self.assertEqual(result["finetune"]["soccernet_matches"], 0)
        self.assertEqual(result["finetune"]["soccernet_window_exposures_by_game"], {})
        self.assertEqual(result["finetune"]["soccernet_windows_per_batch"], 0)

    def test_sparse_clip_and_period_windows_never_cross_boundaries(self):
        old = old_record(period_break=True)
        groups = sf._record_windows([old], sparse=False)
        entries = groups[("sportec", "old-1")]
        self.assertEqual(len(entries), 2)
        for _, start, size, full, _ in entries:
            block = int(old["period"][start])
            self.assertTrue(np.all(old["period"][full] == block))
            self.assertEqual(size, 50)

        # Identical timelines in different clips remain independent records.
        first = new_record(match="game-1", anchor=12)
        second = new_record(match="game-1", anchor=4)
        groups = sf._record_windows([first, second], sparse=True)
        self.assertEqual(len(groups[("soccernet", "game-1")]), 2)
        self.assertEqual({entry[0] for entry in groups[("soccernet", "game-1")]}, {0, 1})

    def test_prediction_uses_original_dead_probability_and_ignores_labels(self):
        initial = checkpoint()
        tuned = sf.train(initial, [old_record()], [new_record()], .5, 9, updates=2)
        record = new_record()
        inference = {key: record[key] for key in ("x", "names", "period")}
        inference["indices"] = np.arange(len(inference["x"]), dtype=np.int64)
        inference["native_n"] = len(inference["x"])
        baseline = fh.predict(inference, initial)
        predicted = sf.predict(inference, tuned, initial, baseline)
        np.testing.assert_array_equal(predicted[:, 2], baseline[:, 2])
        self.assertTrue(np.allclose(predicted.sum(axis=1), 1.0, atol=2e-7, rtol=0))

        # The tuned model's shared output row for D is discarded and must not
        # alter H/A through any normalization or swap-averaging path.
        altered_dead = copy.deepcopy(tuned)
        altered_dead["state"]["output.weight"][2] += 100.0
        altered_dead["state"]["output.bias"][2] -= 75.0
        altered_prediction = sf.predict(inference, altered_dead, initial, baseline)
        np.testing.assert_array_equal(altered_prediction, predicted)

        # Passing labels through the input mapping cannot change inference.
        with_labels = dict(inference, y=np.zeros(len(inference["x"]), dtype=np.int8))
        np.testing.assert_array_equal(
            sf.predict(with_labels, tuned, initial, baseline), predicted)

        swapped = dict(inference, x=supervised.swap_team_features(
            inference["x"], inference["names"]))
        swapped_baseline = fh.predict(swapped, initial)
        swapped_prediction = sf.predict(swapped, tuned, initial, swapped_baseline)
        np.testing.assert_allclose(swapped_prediction[:, 0], predicted[:, 1], atol=1e-7)
        np.testing.assert_allclose(swapped_prediction[:, 1], predicted[:, 0], atol=1e-7)
        np.testing.assert_array_equal(swapped_prediction[:, 2], predicted[:, 2])

    def test_fraction_and_sparse_target_contracts(self):
        with self.assertRaises(ValueError):
            sf.train(checkpoint(), [old_record()], [new_record()], .1, 1, updates=1)
        invalid = new_record()
        invalid["y"][0] = 2
        with self.assertRaises(ValueError):
            sf.train(checkpoint(), [old_record()], [invalid], .25, 1, updates=1)
        multiple = new_record()
        multiple["y"][0] = 1
        with self.assertRaises(ValueError):
            sf.train(checkpoint(), [old_record()], [multiple], .25, 1, updates=1)


if __name__ == "__main__":
    unittest.main()
