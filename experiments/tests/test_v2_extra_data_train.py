"""Focused contracts for the bounded extra-data continuation."""
from contextlib import nullcontext
from types import SimpleNamespace
import random
import unittest
from unittest.mock import patch

import numpy as np

from experiments import feed_short_sampler, v2_iterative_train as iterative
from experiments import v2_extra_data_train as extra
from experiments.tests.test_sn_dense_train import old_record, teacher as factorized_teacher
from experiments.tests.test_short_label_quality import record as quality_record


class ExtraDataTrainTests(unittest.TestCase):
    def test_approved_arm_constants_and_scope_restoration(self):
        self.assertEqual(extra.ARMS,
            ("fresh_control", "gar_weight625", "gar_quality50"))
        self.assertEqual(extra.UPDATES, 256)
        self.assertEqual(extra.GAR_LOSS_WEIGHT,
            {"fresh_control": .5, "gar_weight625": .625, "gar_quality50": .5})
        old = (iterative.LR, iterative.FRACTION_GAR, iterative.STUDY,
               iterative.random)
        with extra.training_scope(extra.FRESH_OFFSET, .625):
            self.assertEqual(iterative.LR, 1e-4)
            self.assertEqual(iterative.FRACTION_GAR, .625)
            self.assertEqual(iterative.STUDY, extra.STUDY)
        self.assertEqual((iterative.LR, iterative.FRACTION_GAR, iterative.STUDY), old[:3])
        self.assertIs(iterative.random, old[3])

    def test_gar_quality_rate_is_independent_of_original_rate(self):
        from experiments import short_original_diversity
        sampler = extra.ExtraDataSampler([], [], [], 2026, quality_rate=.5)
        self.assertIsInstance(sampler, short_original_diversity.OriginalDiversitySampler)
        self.assertEqual(sampler.rate, .25)
        self.assertEqual(sampler.quality_rate, .5)
        receipt = sampler.receipt()
        self.assertEqual(receipt["original_replacement_rate"], .25)
        self.assertEqual(receipt["gar_quality_replacement_rate"], .5)

    def test_quality50_keeps_original_draws_and_same_game_exposure_with_two_records(self):
        from experiments import soccernet_capacity_model as cap
        from experiments.tests.test_sn_dense_train import NAMES
        old = [old_record("old", 12)]
        gar = [quality_record("same-game", 0), quality_record("same-game", 1000)]
        reference = extra.ExtraDataSampler(old, gar, NAMES, 71, quality_rate=.25)
        candidate = extra.ExtraDataSampler(old, gar, NAMES, 71, quality_rate=.5)
        for sampler in (reference, candidate):
            sampler.original_rng = random.Random(2026 + extra.FRESH_OFFSET + 1106644)
        old_rng_a, old_rng_b = random.Random(99), random.Random(99)
        gar_rng_a, gar_rng_b = random.Random(101), random.Random(101)
        for _ in range(256):
            old_a = reference.replace("old", cap._draw_old(old_rng_a,
                reference.groups["old"], old))
            old_b = candidate.replace("old", cap._draw_old(old_rng_b,
                candidate.groups["old"], old))
            self.assertEqual(old_a[:3], old_b[:3])
            np.testing.assert_array_equal(old_a[3], old_b[3])
            self.assertEqual(old_a[4], old_b[4])
            new_a = reference.replace("new", cap._draw_new(gar_rng_a,
                reference.groups["new"]))
            new_b = candidate.replace("new", cap._draw_new(gar_rng_b,
                candidate.groups["new"]))
            self.assertEqual(gar[new_a[0]]["match"], gar[new_b[0]]["match"])
            self.assertEqual(gar_rng_a.getstate(), gar_rng_b.getstate())
        self.assertEqual(reference.exposures, candidate.exposures)
        self.assertGreater(candidate.counts["new"]["replacement_draws"],
                           reference.counts["new"]["replacement_draws"])
        self.assertEqual(reference.rate, candidate.rate)
        self.assertEqual(reference.quality_rate, .25)
        self.assertEqual(candidate.quality_rate, .5)

    def test_fit_binds_weight_and_keeps_teacher_normalization_and_d_state(self):
        import torch
        torch.set_num_threads(1)
        teacher = factorized_teacher(seed=17)
        ha = iterative.cap.network(teacher["input_features"], 96)
        checkpoint = {"kind": "soccernet_capacity", "hidden": 96,
            "feature_names": list(teacher["feature_names"]),
            "median": teacher["median"].copy(), "scale": teacher["scale"].copy(),
            "state": {name: value.detach().cpu().numpy().copy()
                      for name, value in ha.state_dict().items()},
            "d_state": {name: value.copy() for name, value in teacher["state"].items()},
            "source": {"checkpoint_sha256": iterative.cap._checkpoint_sha(teacher)}}
        old = [old_record("old-a", 3), old_record("old-b", 4)]
        gar = [quality_record("gar-a", 5), quality_record("gar-b", 6)]
        base = SimpleNamespace(installed=feed_short_sampler.installed)
        recipe = SimpleNamespace(
            short=SimpleNamespace(parent=SimpleNamespace(base=base)),
            _round_robin_v2_windows=lambda _records: nullcontext({"v2/r0": 32}))
        original_random = iterative.random
        with patch("experiments.v2_spell_frontier_train.prior_run.bind",
                   return_value=recipe):
            result = extra.fit_from_checkpoints(teacher, checkpoint, old, gar,
                seed=2026, arm="gar_weight625", verbose=False, steps=2)
        model = result["model"]
        self.assertEqual(result["config"]["fraction_gar"], .625)
        self.assertEqual(result["config"]["gar_loss_weight"], .625)
        self.assertEqual(model["training"]["stage"], 1026)
        self.assertEqual(model["training"]["draw_seed_offset"], extra.FRESH_OFFSET)
        self.assertIs(iterative.random, original_random)
        self.assertEqual(model["feature_names"], checkpoint["feature_names"])
        for key in ("median", "scale"):
            np.testing.assert_array_equal(model[key], checkpoint[key])
        self.assertEqual(set(model["d_state"]), set(checkpoint["d_state"]))
        for key, value in checkpoint["d_state"].items():
            np.testing.assert_array_equal(model["d_state"][key], value)

    def test_fresh_control_replays_the_shared_frontier_loop_exactly(self):
        import torch
        from experiments import short_original_diversity, v2_spell_frontier_train as frontier
        torch.set_num_threads(1)
        teacher = factorized_teacher(seed=23)
        ha = iterative.cap.network(teacher["input_features"], 96)
        checkpoint = {"kind": "soccernet_capacity", "hidden": 96,
            "feature_names": list(teacher["feature_names"]),
            "median": teacher["median"].copy(), "scale": teacher["scale"].copy(),
            "state": {name: value.detach().cpu().numpy().copy()
                      for name, value in ha.state_dict().items()},
            "d_state": {name: value.copy() for name, value in teacher["state"].items()},
            "source": {"checkpoint_sha256": iterative.cap._checkpoint_sha(teacher)}}
        old = [old_record("old-a", 7), old_record("old-b", 8)]
        gar = [quality_record("gar-a", 9), quality_record("gar-b", 10)]
        base = SimpleNamespace(installed=feed_short_sampler.installed)
        recipe = SimpleNamespace(short=SimpleNamespace(parent=SimpleNamespace(base=base),
                OriginalDiversitySampler=short_original_diversity.OriginalDiversitySampler),
            _round_robin_v2_windows=lambda _records: nullcontext({"v2/r0": 32}))
        with patch("experiments.v2_spell_frontier_train.prior_run.bind",
                   return_value=recipe):
            actual = extra.fit_from_checkpoints(teacher, checkpoint, old, gar,
                seed=2026, arm="fresh_control", verbose=False, steps=2)
        sampler = frontier.make_sampler(recipe, old, gar,
            list(teacher["feature_names"]), 2026, extra.FRESH_OFFSET)
        initial = iterative.warmstart_state(teacher, checkpoint)
        with frontier.capture_windows(recipe, old, gar, sampler) as expected_windows:
            with recipe._round_robin_v2_windows(old):
                with frontier.rng_scope(extra.FRESH_OFFSET):
                    expected = iterative._fit_loop(initial, old, gar, seed=2026,
                        arm="ce_control", verbose=False, steps=2)
        self.assertEqual(actual["sampled_windows"], expected_windows)
        for name, array in expected["model"]["state"].items():
            np.testing.assert_array_equal(actual["model"]["state"][name], array)
        self.assertEqual(actual["exposures"], expected["exposures"])


if __name__ == "__main__":
    unittest.main()
