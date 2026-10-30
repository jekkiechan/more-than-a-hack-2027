"""Synthetic contracts for width-scaled conditional H/A warm-starts."""
import copy
import unittest

import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt, soccernet_capacity_model as cap
from experiments import supervised


WIDTH = 204
NAMES = ([f"H_feature_{i}" for i in range(WIDTH // 2)] +
         [f"A_feature_{i}" for i in range(WIDTH // 2)])


def teacher(seed=7):
    import torch

    torch.manual_seed(seed)
    model = fh.network(2 * WIDTH, list(range(20)), player_only=True)
    return dict(kind="factorized_had", player_only=True, seed=seed,
        player_indices=list(range(20)),
        state={key: value.detach().cpu().numpy().copy()
               for key, value in model.state_dict().items()},
        median=np.zeros(WIDTH, dtype=np.float32),
        scale=np.ones(WIDTH, dtype=np.float32),
        input_features=2 * WIDTH, feature_names=NAMES.copy())


def record(*, source="sportec", match="match-1", seed=4, n=100,
           anchors=(10, 55, 90), period_breaks=()):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .3, (n, WIDTH)).astype(np.float32)
    y = np.full(n, -1, dtype=np.int8)
    for i, index in enumerate(anchors):
        if 0 <= index < n:
            y[index] = i % 2
    period = np.zeros(n, dtype=np.int32)
    for index in period_breaks:
        period[index:] += 1
    return dict(source=source, match=match, x=x, y=y, period=period,
        names=NAMES.copy(), timestamps=np.arange(n, dtype=np.float64) * .2,
        fps=5.0,
        alternate=x + np.float32(.01),
        replace_windows=np.zeros(len(list(lt.chunks(dict(y=y, period=period)))), dtype=bool))


def feature_view(value):
    return {key: value[key] for key in ("x", "names", "period", "timestamps", "fps")}


def full_native_record(*, source="sportec", match="native", seed=4):
    # Complete, censored, unknown, dead, and period-edge examples. One native
    # sample per row makes the objective masks directly inspectable.
    value = record(source=source, match=match, seed=seed, n=24,
                   anchors=tuple(range(24)))
    labels = np.asarray([0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0,
                         0, 0, 0, 1, -1, 1, 1, 0, 0, 2, 0, 0], dtype=np.int8)
    value["y"] = labels.copy()
    value["period"] = np.zeros(len(labels), dtype=np.int32)
    value["period"][22:] = 1
    value["full_y"] = labels.copy()
    value["full_period"] = value["period"].copy()
    value["indices"] = np.arange(len(labels), dtype=np.int64)
    value["fps"] = 5.0
    value["replace_windows"] = np.zeros(len(list(lt.chunks(value))), dtype=bool)
    return value


def raw_conditional(model, rec, width):
    import torch

    torch.set_num_threads(1)
    net = cap.network(2 * WIDTH, width)
    net.load_state_dict({key: torch.as_tensor(value) for key, value in model["state"].items()})
    net.eval()
    normalized = lt.normalize(rec["x"], model["median"], model["scale"])
    windows = list(lt.chunks(rec))
    result = np.empty((len(rec["x"]), 2), dtype=np.float32)
    with torch.inference_mode():
        for offset in range(0, len(windows), 32):
            batch = windows[offset:offset + 32]
            x = torch.as_tensor(np.stack([normalized[full] for _, _, full in batch]))
            q = torch.softmax(net(x)[:, cap.CONTEXT:cap.CONTEXT + cap.CORE], dim=-1).numpy()
            for index, (start, size, _) in enumerate(batch):
                result[start:start + size] = q[index, :size]
    return result


class SoccerNetCapacityContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)

    def test_teacher_conditional_function_is_preserved_at_every_width(self):
        initial = teacher()
        rec = record(seed=31, anchors=())
        expected = cap.teacher_conditional(feature_view(rec), initial)
        for width in (24, 48, 96):
            with self.subTest(width=width):
                import torch
                torch.manual_seed(109 + width)
                net = cap.network(2 * WIDTH, width)
                cap._copy_teacher_function(initial, net)
                model = dict(kind="soccernet_capacity", hidden=width,
                    state={key: value.detach().cpu().numpy().copy()
                           for key, value in net.state_dict().items()},
                    median=initial["median"], scale=initial["scale"],
                    feature_names=initial["feature_names"], input_features=initial["input_features"])
                got = raw_conditional(model, rec, width)
                np.testing.assert_allclose(got, expected, atol=1e-6, rtol=0)
                self.assertEqual(net.output.out_features, 2)
                if width == 24:
                    self.assertEqual(net.sequence.hidden_size, 24)

    def test_teacher_dead_logit_does_not_enter_conditional_initialization(self):
        initial = teacher()
        rec = feature_view(record(seed=41, anchors=()))
        expected = cap.teacher_conditional(rec, initial)
        perturbed = copy.deepcopy(initial)
        perturbed["state"]["output.weight"][2] += 500
        perturbed["state"]["output.bias"][2] -= 300
        np.testing.assert_array_equal(cap.teacher_conditional(rec, perturbed), expected)
        import torch
        torch.manual_seed(1)
        net = cap.network(2 * WIDTH, 96)
        cap._copy_teacher_function(perturbed, net)
        model = dict(state={key: value.detach().numpy().copy()
                            for key, value in net.state_dict().items()},
                     median=initial["median"], scale=initial["scale"])
        np.testing.assert_allclose(raw_conditional(model, rec, 96), expected,
                                   atol=1e-6, rtol=0)

    def test_deterministic_training_and_exact_milestone_resume(self):
        initial = teacher()
        old = [record(match="old-a", seed=1), record(match="old-b", seed=2)]
        new = [record(source="soccernet", match="game-a", seed=3,
                      anchors=(15, 87)),
               record(source="soccernet", match="game-b", seed=8,
                      anchors=(48,))]
        before = copy.deepcopy(initial)
        full_a = cap.train(initial, old, new, 48, 1e-4, 123, steps=4,
                           milestones=(2, 4), batch_size=8)
        callback_snapshots = {}
        full_b = cap.train(initial, old, new, 48, 1e-4, 123, steps=4,
            milestones=(2, 4), batch_size=8,
            on_checkpoint=lambda step, checkpoint: callback_snapshots.__setitem__(step, checkpoint))
        first = cap.train(initial, old, new, 48, 1e-4, 123, steps=2,
                          milestones=(2,), batch_size=8)[2]
        resumed = cap.train(initial, old, new, 48, 1e-4, 123, steps=4,
                            milestones=(4,), batch_size=8, resume_from=first)[4]
        self.assertEqual(set(full_a), {2, 4})
        self.assertEqual(full_a[2]["kind"], "soccernet_capacity")
        self.assertEqual(full_a[4]["step"], 4)
        self.assertEqual(full_a[4]["training"]["width"], 48)
        self.assertEqual(full_a[4]["training"]["new_windows_per_batch"], 4)
        self.assertEqual(sum(full_a[4]["metadata"]["old_window_exposures_by_source_match"].values()), 16)
        self.assertEqual(sum(full_a[4]["metadata"]["new_window_exposures_by_game"].values()), 16)
        self.assertEqual(full_a[4]["metadata"]["loss_intervals"][-1]["step"], 4)
        self.assertEqual(full_a[4]["metadata"], full_b[4]["metadata"])
        self.assertEqual(set(callback_snapshots), {2, 4})
        for step in (2, 4):
            for key in full_b[step]["state"]:
                np.testing.assert_array_equal(callback_snapshots[step]["state"][key],
                                              full_b[step]["state"][key])
        self.assertTrue(np.array_equal(full_a[4]["median"], initial["median"]))
        self.assertTrue(np.array_equal(full_a[4]["scale"], initial["scale"]))
        for key in full_a[4]["state"]:
            np.testing.assert_array_equal(full_a[4]["state"][key], full_b[4]["state"][key])
            np.testing.assert_array_equal(full_a[4]["state"][key], resumed["state"][key])
            np.testing.assert_array_equal(initial["state"][key], before["state"][key])
        self.assertEqual(full_a[4]["metadata"], resumed["metadata"])

    def test_gradients_reach_new_units_after_teacher_preserving_step(self):
        fitted = cap.train(teacher(), [record(match="old")],
            [record(source="soccernet", match="game", anchors=(31,))],
            48, 1e-4, 43, steps=2, milestones=(2,), batch_size=8)[2]
        norms = fitted["metadata"]["capacity_gradient_norms_by_step"]
        self.assertEqual(norms[1], 0.0)
        self.assertGreater(norms[2], 0.0)

    def test_prediction_is_label_independent_team_symmetric_and_keeps_original_d(self):
        initial = teacher()
        old = [record(match="old")]
        new = [record(source="soccernet", match="game", anchors=(21, 79))]
        tuned = cap.train(initial, old, new, 48, 1e-4, 55, steps=2,
                          milestones=(2,), batch_size=8)[2]
        rec = feature_view(record(source="soccernet", match="eval", anchors=()))
        baseline = fh.predict(rec, initial)
        prediction = cap.predict(rec, tuned, initial, original_probability=baseline)
        np.testing.assert_array_equal(prediction[:, 2], baseline[:, 2])
        self.assertTrue(np.allclose(prediction.sum(1), 1, atol=1e-6, rtol=0))
        self.assertEqual(tuned["state"]["output.weight"].shape[0], 2)
        self.assertFalse(any("dead" in key or "output.weight" in key and
                             tuned["state"][key].shape[0] == 3 for key in tuned["state"]))

        with_targets = dict(rec, y=np.zeros(len(rec["x"]), dtype=np.int8))
        np.testing.assert_array_equal(cap.predict(with_targets, tuned, initial, baseline), prediction)
        swapped = dict(rec, x=supervised.swap_team_features(rec["x"], rec["names"]))
        swapped_baseline = fh.predict(swapped, initial)
        swapped_result = cap.predict(swapped, tuned, initial, swapped_baseline)
        np.testing.assert_allclose(swapped_result[:, 0], prediction[:, 1], atol=1e-7, rtol=0)
        np.testing.assert_allclose(swapped_result[:, 1], prediction[:, 0], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(swapped_result[:, 2], prediction[:, 2])

    def test_sparse_long_segments_keep_anchor_windows_and_period_breaks(self):
        long_record = record(source="soccernet", match="game-long", n=140,
                             anchors=(5, 49, 75, 129), period_breaks=(60,))
        groups, ids = cap._window_index([long_record], NAMES, old=False)
        windows = groups["game-long"]
        self.assertEqual(len(windows), 3)
        self.assertEqual(ids[0], ("soccernet", "game-long"))
        for _, start, size, full, _ in windows:
            self.assertTrue(np.all(long_record["period"][full] == long_record["period"][start]))
            self.assertTrue(np.isin(long_record["y"][start:start + size], (0, 1)).any())
        # No chunks from independent records are joined, even for the same game.
        second = record(source="soccernet", match="game-long", seed=22, n=75,
                        anchors=(70,))
        combined, _ = cap._window_index([long_record, second], NAMES, old=False)
        self.assertEqual(len(combined["game-long"]), 4)
        self.assertEqual({entry[0] for entry in combined["game-long"]}, {0, 1})

    def test_timestamp_discontinuity_must_be_a_period_break(self):
        broken = record(source="soccernet", match="game-time", n=100,
                        anchors=(10, 70))
        broken["timestamps"][60:] += 3.0
        with self.assertRaisesRegex(ValueError, "period must preserve"):
            cap._window_index([broken], NAMES, old=False)
        broken["period"][60:] = 1
        groups, _ = cap._window_index([broken], NAMES, old=False)
        self.assertEqual(len(groups["game-time"]), 2)

    def test_unknown_and_dead_targets_are_both_masked(self):
        initial = teacher()
        old_a = record(match="old", seed=2)
        new_a = record(source="soccernet", match="game", seed=3, anchors=(22, 84))
        old_b, new_b = copy.deepcopy(old_a), copy.deepcopy(new_a)
        old_b["y"][old_b["y"] == -1] = 2
        new_b["y"][new_b["y"] == -1] = 2
        result_a = cap.train(initial, [old_a], [new_a], 24, 1e-4, 9,
                             steps=2, milestones=(2,), batch_size=8)[2]
        result_b = cap.train(initial, [old_b], [new_b], 24, 1e-4, 9,
                             steps=2, milestones=(2,), batch_size=8)[2]
        for key in result_a["state"]:
            np.testing.assert_array_equal(result_a["state"][key], result_b["state"][key])

    def test_full_native_spell_and_live_transition_masks_are_reused(self):
        native = full_native_record()
        factors, pairs, weights, complete = cap._old_objective_targets(
            [native], require_full_native=True)
        self.assertTrue(complete)
        self.assertEqual(factors[0].shape, native["y"].shape)
        # The initial H run and runs touching X are censored; D and unknown
        # rows remain neutral. Full-native transition targets keep only
        # consecutive known H/A pairs inside an uninterrupted interval.
        np.testing.assert_array_equal(factors[0][[0, 1, 6, 15, 16, 17, 18, 21, 22, 23]], 1.0)
        self.assertFalse(pairs[0]["live_mask"][5])  # H→D
        self.assertFalse(pairs[0]["live_mask"][6])  # D→H
        self.assertFalse(pairs[0]["live_mask"][15])  # H→unknown
        self.assertFalse(pairs[0]["live_mask"][16])  # unknown→H
        self.assertFalse(pairs[0]["live_mask"][21])  # D→H at period edge
        self.assertFalse(pairs[0]["live_mask"][23])  # record edge
        np.testing.assert_array_equal(weights[0], pairs[0]["live_mask"].astype(np.float32))
        self.assertTrue(np.any(factors[0][2:15] != 1.0))

    def test_real_fit_can_require_full_native_old_objective_fields(self):
        with self.assertRaisesRegex(ValueError, "full_y/full_period/indices/fps"):
            cap.train(teacher(), [record(match="old")], [], 24, 1e-4, 8,
                steps=1, milestones=(1,), fraction=0, batch_size=8,
                require_full_native=True)
        fitted = cap.train(teacher(), [full_native_record()], [], 24, 1e-4, 8,
            steps=1, milestones=(1,), fraction=0, batch_size=8,
            require_full_native=True)[1]
        self.assertEqual(fitted["training"]["old_live_transition_coefficient"], .05)
        self.assertEqual(fitted["training"]["old_spell_coefficient"], 1.0)
        self.assertIn("complete-spell", fitted["training"]["old_objective"])

    def test_gar_dense_spell_metadata_never_changes_sparse_anchor_fit(self):
        old = [record(match="old")]
        gar_a = record(source="soccernet", match="game", seed=3, anchors=(22, 84))
        gar_b = copy.deepcopy(gar_a)
        gar_a.update(full_y=np.zeros(100, dtype=np.int8),
                     full_period=np.zeros(100, dtype=np.int32),
                     indices=np.arange(100), fps=5.0)
        gar_b.update(full_y=np.tile(np.asarray([0, 1], dtype=np.int8), 50),
                     full_period=np.zeros(100, dtype=np.int32),
                     indices=np.arange(100), fps=5.0)
        result_a = cap.train(teacher(), old, [gar_a], 24, 1e-4, 19,
            steps=2, milestones=(2,), batch_size=8)[2]
        result_b = cap.train(teacher(), old, [gar_b], 24, 1e-4, 19,
            steps=2, milestones=(2,), batch_size=8)[2]
        for key in result_a["state"]:
            np.testing.assert_array_equal(result_a["state"][key], result_b["state"][key])

    def test_zero_gar_fraction_does_not_inspect_gar_records(self):
        result = cap.train(teacher(), [record(match="old")], [None], 24,
            1e-4, 8, steps=1, milestones=(1,), fraction=0, batch_size=8)[1]
        self.assertEqual(result["training"]["new_record_count"], 0)
        self.assertEqual(result["training"]["new_windows"], 0)
        self.assertEqual(result["metadata"]["new_window_exposures_by_game"], {})

    def test_scalarized_cache_identities_and_old_replacement_choices(self):
        old = record(match=np.array("old-np-id"), seed=1)
        old["source"] = np.str_("sportec")
        old["replace_windows"][:] = True
        groups, identities = cap._window_index([old], NAMES, old=True)
        self.assertIn(("sportec", "old-np-id"), groups)
        self.assertEqual(identities[0], ("sportec", "old-np-id"))
        self.assertTrue(all(entry[4] for entry in groups[("sportec", "old-np-id")]))

    def test_invalid_configurations_fail_closed(self):
        initial = teacher()
        old = [record(match="old")]
        new = [record(source="soccernet", match="game")]
        with self.assertRaises(ValueError):
            cap.train(initial, old, new, 32, 1e-4, 1, steps=1, milestones=(1,))
        with self.assertRaises(ValueError):
            cap.train(initial, old, new, 48, 1e-4, 1, steps=2, milestones=(1,), batch_size=8)
        no_anchor = record(source="soccernet", match="empty", anchors=())
        with self.assertRaises(ValueError):
            cap.train(initial, old, [no_anchor], 48, 1e-4, 1, steps=1,
                      milestones=(1,), batch_size=8)


if __name__ == "__main__":
    unittest.main()
