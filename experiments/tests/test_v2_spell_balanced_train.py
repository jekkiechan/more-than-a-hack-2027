"""Contracts for the v2 spell-balanced continuation wrapper."""
from __future__ import annotations

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np

from experiments import v2_iterative_train as iterative
from experiments import v2_spell_balanced_train as study


class SpellBalancedTrainingContracts(unittest.TestCase):
    def test_stage_paths_expose_cumulative_update_counts(self):
        self.assertEqual(study.output_path("leave_out_metrica", 2026).name,
                         "leave_out_metrica_2026.joblib")
        self.assertEqual(study._pooled_output_path(2027, 512).name,
                         "stage512_2027.joblib")
        self.assertEqual(study._pooled_output_path(2027, 1024).name,
                         "stage1024_2027.joblib")
        with self.assertRaises(ValueError):
            study._pooled_output_path(2027, 1536)

    def test_learning_rate_and_study_override_restore_on_error(self):
        original = iterative.LR, iterative.STUDY
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with study._trainer_scope(study.LR_STAGE_1024):
                self.assertEqual(iterative.LR, 1e-4)
                self.assertEqual(iterative.STUDY, study.STUDY)
                raise RuntimeError("synthetic")
        self.assertEqual((iterative.LR, iterative.STUDY), original)

    def test_exported_provenance_marks_study_and_cumulative_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            warmstart = Path(temp) / "warm.joblib"
            warmstart.write_bytes(b"warm-start test fixture")
            state = {"weight": np.array([1.0, 2.0])}
            d_state = {"fixed": np.array([3.0])}
            config = {"study": "prior", "steps": 512, "lr": 1e-4}
            model = {"training": dict(config), "state": state,
                     "d_state": d_state, "step": 512}
            result = {"model": model, "config": dict(config)}

            with patch.object(iterative.cap, "_checkpoint_sha", return_value="teacher-hash"):
                model["source"] = {"checkpoint_sha256": "teacher-hash"}
                exported = study._annotate_result(
                    result, teacher={}, stage=1024, cumulative_updates=1024,
                    warmstart_path=warmstart, arm="ce_control_lr1e-4_continuation",
                    schedule="seed reset; repeats stage-1 sample schedule")

            self.assertIs(exported, model)
            for provenance in (exported["training"], result["config"]):
                self.assertEqual(provenance["study"], study.STUDY)
                self.assertEqual(provenance["stage"], 1024)
                self.assertEqual(provenance["stage_updates"], 512)
                self.assertEqual(provenance["cumulative_updates"], 1024)
                self.assertEqual(provenance["warm_start_file_sha256"], study.sha(warmstart))
            self.assertEqual(exported["step"], 1024)
            np.testing.assert_array_equal(exported["d_state"]["fixed"], [3.0])

    def test_pooled_stage_1024_requires_and_consumes_sealed_stage_512(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            pool_decl = base / "pool-declaration.json"
            pool_decl.write_text("{}\n")
            study_out = base / "train"
            teacher_path, adopted_path = base / "teacher.joblib", base / "adopted.joblib"
            teacher_path.write_bytes(b"teacher")
            adopted_path.write_bytes(b"adopted")
            teacher = {"feature_names": ["a"], "median": np.array([0.]),
                       "scale": np.array([1.]), "state": {"teacher": np.array([1.])}}
            adopted = {"kind": "soccernet_capacity", "hidden": 96,
                       "feature_names": ["a"], "median": np.array([0.]),
                       "scale": np.array([1.]), "state": {"ha": np.array([2.])},
                       "d_state": {"fixed": np.array([3.])},
                       "source": {"checkpoint_sha256": "teacher-hash"}}
            recipe = SimpleNamespace(
                TRAIN_RECORDS=base / "gar",
                rf=SimpleNamespace(dl=SimpleNamespace(
                    load_dense_records=lambda *_args, **_kwargs:
                    [{"match": f"gar-{i}"} for i in range(45)])),
                short=SimpleNamespace(OriginalDiversitySampler=lambda *_args:
                    SimpleNamespace(receipt=lambda: {"sampler": "fixture"}),
                    parent=SimpleNamespace(base=SimpleNamespace(installed=lambda _s: nullcontext()))),
                _round_robin_v2_windows=lambda _records:
                    _ViewContext())
            output_path = lambda seed, stage: study_out / "models" / "pooled" / f"stage{stage}_{seed}.joblib"
            declaration = {"training_stage": 1024}

            def fake_fit(_teacher, _prior, _old, _gar, *, seed, arm, verbose):
                model = {"kind": "soccernet_capacity", "hidden": 96,
                         "feature_names": ["a"], "median": np.array([0.]),
                         "scale": np.array([1.]), "state": {"ha": np.array([2.])},
                         "d_state": {"fixed": np.array([3.])},
                         "source": {"checkpoint_sha256": "teacher-hash"},
                         "training": {}, "step": 512}
                return {"model": model, "config": {"lr": iterative.LR},
                        "exposures": {}, "loss_intervals": []}

            def run_stage(stage):
                with patch.multiple(study, ROOT=base, OUT=study_out, POOL_DECL=pool_decl):
                    with patch.object(study, "check_pooled", return_value=declaration), \
                         patch.object(study, "_bind_prior_recipe", return_value=recipe), \
                         patch.object(study, "_pool_teacher_and_adopted",
                                      return_value=(teacher, adopted, teacher_path, adopted_path)), \
                         patch.object(study, "pooled_training", return_value=[{"match": f"old-{i}"} for i in range(13)]), \
                         patch.object(study, "_pooled_output_path", side_effect=output_path), \
                         patch.object(study.iterative, "fit_from_checkpoints", side_effect=fake_fit), \
                         patch.object(study.iterative.cap, "_checkpoint_sha", return_value="teacher-hash"):
                        return study.fit_pooled_stage(2026, stage, verbose=False)

            # Stage 512 writes the receipt that stage 1024 must validate.
            first = run_stage(512)
            self.assertEqual(first["stage"], 512)
            second = run_stage(1024)
            self.assertEqual(second["stage"], 1024)
            self.assertEqual(second["warmstart_sha256"], first["sha256"])
            model = joblib.load(output_path(2026, 1024))
            self.assertEqual(model["training"]["study"], study.STUDY)
            self.assertEqual(model["training"]["stage"], 1024)
            np.testing.assert_array_equal(model["d_state"]["fixed"], [3.])

    def test_fold_fit_uses_lr_override_and_exports_stage_1024_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            prior_path, teacher_path = base / "prior.joblib", base / "teacher.joblib"
            prior_path.write_bytes(b"prior checkpoint")
            teacher_path.write_bytes(b"paired teacher")
            train_decl = base / "declaration.json"
            train_decl.write_text("{}\n")
            prior = {"kind": "soccernet_capacity", "hidden": 96,
                     "source": {"checkpoint_sha256": "teacher-hash"},
                     "d_state": {"fixed": np.array([3.])},
                     "median": np.array([0.]), "scale": np.array([1.]),
                     "feature_names": ["a"]}
            teacher = {"feature_names": ["a"]}
            recipe = SimpleNamespace(
                TRAIN_RECORDS=base / "gar",
                rf=SimpleNamespace(
                    sc=SimpleNamespace(initial_path=lambda *_: teacher_path),
                    dl=SimpleNamespace(load_dense_records=lambda *_args, **_kwargs:
                        [{"match": "gar-1"}])),
                short=SimpleNamespace(
                    OriginalDiversitySampler=lambda *_args:
                        SimpleNamespace(receipt=lambda: {"sampler": "fixture"}),
                    parent=SimpleNamespace(base=SimpleNamespace(installed=lambda _s: nullcontext()))),
                training=lambda _fold: [{"match": "old-1"}],
                _round_robin_v2_windows=lambda _records: _ViewContext())

            def fake_fit(_teacher, _prior, _old, _gar, *, seed, arm, verbose):
                self.assertEqual(iterative.LR, study.LR_STAGE_1024)
                return {"model": {"source": {"checkpoint_sha256": "teacher-hash"},
                    "training": {}, "d_state": {"fixed": np.array([3.])},
                    "median": np.array([0.]), "scale": np.array([1.]),
                    "feature_names": ["a"], "step": 512},
                    "config": {"lr": iterative.LR}, "exposures": {}, "loss_intervals": []}

            destination = base / "output.joblib"
            real_sha = study.sha
            def fixture_sha(path):
                path = Path(path)
                return real_sha(path if path.is_absolute() else base / path)
            with patch.multiple(study, ROOT=base, TRAIN_DECL=train_decl):
                with patch.object(study, "check", return_value={}), \
                     patch.object(study, "_bind_prior_recipe", return_value=recipe), \
                     patch.object(study, "sha", side_effect=fixture_sha), \
                     patch.object(study, "prior_model_path", return_value=prior_path), \
                     patch.object(study, "output_path", return_value=destination), \
                     patch.object(study.joblib, "load", side_effect=lambda path: teacher if Path(path) == teacher_path else prior), \
                     patch.object(study.iterative, "fit_from_checkpoints", side_effect=fake_fit), \
                     patch.object(study.iterative.cap, "_checkpoint_sha", return_value="teacher-hash"):
                    receipt = study.fit_fold("leave_out_metrica", 2026, verbose=False)
            self.assertEqual(receipt["stage"], 1024)
            self.assertTrue(receipt["unchanged_d_state"])
            self.assertEqual(receipt["config"]["stage"], 1024)
            self.assertEqual(receipt["config"]["study"], study.STUDY)
            model = joblib.load(destination)
            self.assertEqual(model["training"]["cumulative_updates"], 1024)
            np.testing.assert_array_equal(model["d_state"]["fixed"], [3.])


class _ViewContext:
    def __enter__(self):
        return {"view-0": 1}

    def __exit__(self, *_args):
        return False

    def test_declaration_labels_repeated_seed_draw_schedule_honestly(self):
        self.assertEqual(study.LR_STAGE_1024, 1e-4)
        self.assertEqual(study.STAGE_UPDATES, 512)
        self.assertEqual(study.FOLDS, iterative.FOLDS)
        self.assertEqual(study.SEEDS, iterative.SEEDS)


if __name__ == "__main__":
    unittest.main()
