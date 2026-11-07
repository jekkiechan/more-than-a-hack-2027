"""Synthetic RNG/sampler contracts for the bounded frontier continuation."""
from types import SimpleNamespace
from contextlib import nullcontext
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import feed_short_sampler, v2_iterative_train as iterative
from experiments import v2_spell_frontier_train as frontier
from experiments.tests.test_short_label_quality import record as quality_record
from experiments.tests.test_sn_dense_train import old_record, teacher as factorized_teacher


class FrontierDrawScheduleTests(unittest.TestCase):
    def test_pooled_warmstart_uses_balanced_stage1024_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "balanced-stage1024.joblib"
            checkpoint = {"training": {"study": "v2_spell_balanced_v1", "stage": 1024}}
            import joblib
            joblib.dump(checkpoint, path)
            Path(str(path) + ".json").write_text(json.dumps({"sha256": frontier.sha(path)}))
            with patch.object(frontier.parent_train, "_pooled_output_path", return_value=path):
                loaded, used_path = frontier.pooled_warmstart(2026)
            self.assertEqual(used_path, path)
            self.assertEqual(loaded["training"]["study"], "v2_spell_balanced_v1")
            self.assertEqual(loaded["training"]["stage"], 1024)

    def test_declaration_freeze_check_roundtrip_binds_stage_and_rng_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"study_id": frontier.STUDY}))
            pool_seal = root / "balanced_pool_seal.json"
            pool_seal.write_text(json.dumps({"all_authorized_stages_complete": True}))
            decl, seal = root / "declaration.json", root / "seal.json"
            source = {"runner": "frozen"}
            inputs = {"models": "frozen"}
            with patch.multiple(frontier, ROOT=root, PLAN=plan, DECL=decl, SEAL=seal,
                                POOL_DECL=root / "pool_decl.json", POOL_SEAL=root / "pool_seal.json"), \
                 patch.object(frontier, "_sources", return_value=source), \
                 patch.object(frontier, "_input_facts", return_value=inputs), \
                 patch.object(frontier.parent_train, "check", return_value={}), \
                 patch.object(frontier.parent_train, "check_pooled", return_value={}), \
                 patch.object(frontier.parent_train, "POOL_SEAL", pool_seal):
                declaration = frontier.freeze()
                checked = frontier.check(full=True)
            self.assertEqual(checked, declaration)
            self.assertEqual(declaration["arms"], ["repeat", "fresh"])
            self.assertEqual(declaration["offsets"], {"repeat": 0, "fresh": 104729})
            self.assertEqual(declaration["updates_per_arm"], 512)
            self.assertEqual(declaration["cumulative_continuation_updates"], 1536)
            self.assertFalse(declaration["organizer_reference_access"])

    def test_repeat_matches_unoffset_schedule_and_fresh_changes_real_windows(self):
        import torch
        from experiments import short_original_diversity

        torch.set_num_threads(1)
        teacher = factorized_teacher(seed=17)
        ha = iterative.cap.network(teacher["input_features"], 96)
        prior = {"kind": "soccernet_capacity", "hidden": 96,
            "feature_names": list(teacher["feature_names"]),
            "median": teacher["median"].copy(), "scale": teacher["scale"].copy(),
            "state": {name: value.detach().cpu().numpy().copy()
                      for name, value in ha.state_dict().items()},
            "d_state": {name: value.copy() for name, value in teacher["state"].items()}}
        initial = iterative.warmstart_state(teacher, prior)
        old = [old_record("old-a", 3), old_record("old-b", 4)]
        gar = [quality_record("gar-a", 5), quality_record("gar-b", 6)]
        recipe = SimpleNamespace(short=SimpleNamespace(parent=SimpleNamespace(base=feed_short_sampler)))
        original_random = iterative.random

        def run(offset):
            sampler = frontier.make_sampler(SimpleNamespace(short=SimpleNamespace(
                OriginalDiversitySampler=short_original_diversity.OriginalDiversitySampler)),
                old, gar, list(teacher["feature_names"]), 2026, offset)
            with frontier.capture_windows(recipe, old, gar, sampler) as window_ids:
                with frontier.rng_scope(offset):
                    result = iterative._fit_loop(initial, old, gar, seed=2026,
                        arm="ce_control", steps=2, verbose=False)
            return result, sampler.receipt(), list(window_ids)

        baseline, baseline_sampler, baseline_windows = run(0)
        repeat, repeat_sampler, repeat_windows = run(0)
        fresh, fresh_sampler, fresh_windows = run(frontier.FRESH_OFFSET)

        self.assertIs(iterative.random, original_random)
        self.assertEqual(baseline["exposures"], repeat["exposures"])
        self.assertEqual(baseline_sampler, repeat_sampler)
        self.assertEqual(baseline_windows, repeat_windows)
        for name, array in baseline["model"]["state"].items():
            np.testing.assert_array_equal(array, repeat["model"]["state"][name])
        self.assertNotEqual(repeat_windows, fresh_windows)
        self.assertNotEqual(repeat["exposures"], fresh["exposures"])
        self.assertNotEqual(repeat_sampler, fresh_sampler)
        self.assertEqual(len(repeat_windows), 64)
        self.assertEqual(len(fresh_windows), 64)
        for result in (repeat, fresh):
            model = result["model"]
            self.assertEqual(set(model["d_state"]), set(prior["d_state"]))
            for name in prior["d_state"]:
                np.testing.assert_array_equal(model["d_state"][name], prior["d_state"][name])
            for name in ("median", "scale"):
                np.testing.assert_array_equal(model[name], prior[name])
            self.assertEqual(model["feature_names"], prior["feature_names"])

    def test_rng_scope_restores_module_global_even_after_exception(self):
        original = iterative.random
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with frontier.rng_scope(frontier.FRESH_OFFSET):
                self.assertIsNot(iterative.random, original)
                raise RuntimeError("fixture")
        self.assertIs(iterative.random, original)

    def test_pooled_fit_passes_balanced_stage1024_as_trainer_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pool_decl = root / "pool_decl.json"
            pool_decl.write_text("{}\n")
            teacher_path, adopted_path, warm_path = (root / "teacher", root / "adopted", root / "balanced1024")
            teacher_path.write_bytes(b"teacher")
            adopted_path.write_bytes(b"adopted-base")
            warm_path.write_bytes(b"balanced-stage1024")
            teacher = {"feature_names": ["a"]}
            adopted = {"feature_names": ["a"], "median": np.array([0.]),
                       "scale": np.array([1.]), "d_state": {"d": np.array([9.])}}
            warm = {"training": {"study": "v2_spell_balanced_v1", "stage": 1024},
                    "source": {"checkpoint_sha256": "teacher-hash"},
                    "feature_names": ["a"], "median": np.array([0.]),
                    "scale": np.array([1.]), "state": {"ha": np.array([2.])},
                    "d_state": {"d": np.array([9.])}}
            recipe = SimpleNamespace(TRAIN_RECORDS=root / "gar",
                rf=SimpleNamespace(dl=SimpleNamespace(load_dense_records=lambda *_a, **_k:
                    [{"match": f"gar-{i}"} for i in range(45)])),
                short=SimpleNamespace(parent=SimpleNamespace(base=feed_short_sampler)),
                _round_robin_v2_windows=lambda _records: _Views())
            seen = {}

            def fake_fit(actual_teacher, initial, old, gar, *, seed, arm, verbose):
                seen["initial"] = initial
                self.assertIs(actual_teacher, teacher)
                self.assertIs(initial, warm)
                model = {"source": {"checkpoint_sha256": "teacher-hash"},
                    "training": {}, "state": {"ha": np.array([2.])},
                    "d_state": {"d": np.array([9.])}, "feature_names": ["a"],
                    "median": np.array([0.]), "scale": np.array([1.]), "step": 512}
                return {"model": model, "config": {}, "exposures": {}, "loss_intervals": []}

            destination = root / "output.joblib"
            declaration = {"training_arm": "repeat", "training_stage": 1536}
            with patch.multiple(frontier, ROOT=root, POOL_DECL=pool_decl,
                                OUT=root / "out"):
                with patch.object(frontier, "check_pooled", return_value=declaration), \
                     patch.object(frontier.prior_run, "bind", return_value=recipe), \
                     patch.object(frontier.parent_train, "pooled_training",
                                  return_value=[{"match": f"old-{i}"} for i in range(13)]), \
                     patch.object(frontier.parent_train, "_pool_teacher_and_adopted",
                                  return_value=(teacher, adopted, teacher_path, adopted_path)), \
                     patch.object(frontier, "pooled_warmstart", return_value=(warm, warm_path)), \
                     patch.object(frontier, "make_sampler", return_value=SimpleNamespace(receipt=lambda: {})), \
                     patch.object(frontier.iterative, "fit_from_checkpoints", side_effect=fake_fit), \
                     patch.object(frontier.iterative.cap, "_checkpoint_sha", return_value="teacher-hash"):
                    receipt = frontier.fit_pooled_one(2026, verbose=False)
            self.assertIs(seen["initial"], warm)
            self.assertEqual(receipt["warm_start_path"], str(warm_path.relative_to(root)))
            self.assertEqual(receipt["stage"], 1536)


class _Views:
    def __enter__(self):
        return {"view-0": 1}

    def __exit__(self, *_args):
        return False


if __name__ == "__main__":
    unittest.main()
