"""Synthetic contracts for pooled combined-objective deployment storage."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import (combined_objective_deployment as deploy,
                         gru_robustness_experiment as gr,
                         learned_temporal as lt,
                         objective_losses as ol)


def _record(match="toy", source="Sportec", offset=0.0, n=105):
    rng = np.random.default_rng(99 + int(offset * 10))
    y = np.resize(np.array([2, 2, 0, 0, 0, 1, 1, 1, 1, 2, 2, 0, 0, -1, 1],
                           dtype=np.int8), n)
    period = np.zeros(n, dtype=np.int32)
    x = (rng.normal(size=(n, 4)) + offset).astype(np.float32)
    chunks = list(lt.chunks(dict(y=y, period=period)))
    replace_windows = np.arange(len(chunks)) % 2 == 0
    replace_rows = np.zeros(n, dtype=bool)
    for selected, (start, size, _) in zip(replace_windows, chunks):
        replace_rows[start:start + size] = selected
    full_y = np.repeat(y, 5)
    full_period = np.repeat(period, 5)
    return dict(
        source=source, match=match, x=x, alternate=(x + np.float32(.07)),
        physical_x=x.copy(), y=y.copy(), period=period.copy(),
        names=["H_gap", "A_gap", "D_speed", "D_accel"],
        indices=np.arange(n, dtype=np.int64) * 5,
        native_n=np.asarray(n * 5), fps=25.0,
        replace_windows=replace_windows, replace_rows=replace_rows,
        full_y=full_y, full_period=full_period)


def _same_value(left, right):
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    return left == right


class CombinedObjectiveDeploymentTests(unittest.TestCase):
    def test_paired_records_match_legacy_slot_dropout_wrapper(self):
        cache = {
            "records": {
                "clean": _record(offset=0.0),
                "gap": _record(offset=1.0),
                "dropout_legacy": _record(offset=2.0),
                "dropout_gap": _record(offset=3.0),
            },
            "full_y": np.repeat(_record()["y"], 5),
            "full_period": np.zeros(525, dtype=np.int32),
            "fps": 25.0,
        }
        actual = deploy.paired_records(cache, "toy")
        reference_cache = {key: {"joint": value}
                           for key, value in cache["records"].items()}
        expected = gr.paired_records(reference_cache, "toy", "slot_dropout")
        self.assertEqual(len(actual), 4)
        for left, right in zip(actual, expected):
            for key in ("x", "alternate", "y", "period", "indices", "names",
                        "replace_rows", "replace_windows"):
                self.assertTrue(_same_value(left[key], right[key]), key)
            np.testing.assert_array_equal(left["full_y"], cache["full_y"])
            np.testing.assert_array_equal(left["full_period"], cache["full_period"])
            self.assertEqual(left["fps"], 25.0)

    def test_file_backed_fit_is_bitwise_equal_and_cleans_normalization_files(self):
        records = [_record(offset=0.0), _record(offset=1.0)]
        temp_parent = Path(tempfile.mkdtemp(prefix="combined_adapter_test_"))
        try:
            with patch.object(lt, "EPOCHS", 1):
                ordinary = ol.fit(records, player_only=True, seed=2026,
                                  objective="combined")
                with deploy.file_backed_normalization(temp_parent) as audit:
                    mapped = ol.fit(records, player_only=True, seed=2026,
                                    objective="combined")
            self.assertEqual(ordinary.keys(), mapped.keys())
            for key in ordinary:
                if key == "state":
                    self.assertEqual(ordinary[key].keys(), mapped[key].keys())
                    for name in ordinary[key]:
                        np.testing.assert_array_equal(ordinary[key][name], mapped[key][name])
                elif isinstance(ordinary[key], np.ndarray):
                    np.testing.assert_array_equal(ordinary[key], mapped[key])
                else:
                    self.assertEqual(ordinary[key], mapped[key], key)
            for record in records:
                np.testing.assert_array_equal(ol.predict(record, ordinary),
                                              ol.predict(record, mapped))
            self.assertEqual(len(audit["normalization_calls"]), 2 * len(records))
            self.assertTrue(all(row["readonly"] for row in audit["normalization_calls"]))
            self.assertTrue(audit["normalizer_restored"])
            self.assertTrue(audit["temporary_files_removed"])
            self.assertEqual(list(temp_parent.iterdir()), [])
        finally:
            temp_parent.rmdir()

    def test_file_backed_context_restores_normalizer_and_removes_temp_after_error(self):
        original = ol.lt.normalize
        temp_parent = Path(tempfile.mkdtemp(prefix="combined_adapter_error_"))
        try:
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with deploy.file_backed_normalization(temp_parent) as audit:
                    ol.lt.normalize(np.zeros((1, 1)), np.zeros(1), np.ones(1))
                    raise RuntimeError("injected")
            self.assertIs(ol.lt.normalize, original)
            self.assertTrue(audit["normalizer_restored"])
            self.assertTrue(audit["temporary_files_removed"])
            self.assertEqual(list(temp_parent.iterdir()), [])
        finally:
            temp_parent.rmdir()

    def test_source_promotion_exception_accepts_only_exact_archives(self):
        with tempfile.TemporaryDirectory(prefix="combined_sources_") as folder:
            root = Path(folder)
            paths = {
                "experiments/baseline_manifest.json":
                    "experiments/baselines/joint_gru_seed_ensemble_v1.json",
                "experiments/model_decision_policy.json":
                    "experiments/baselines/model_decision_policy_joint_gru_seed_ensemble_v1.json",
                "experiments/objective_losses.py": "experiments/objective_losses.py",
            }
            for rel in paths.values():
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(rel)
            frozen = {key: deploy.sha256(root / archive)
                      for key, archive in paths.items()}
            active = dict(frozen)
            (root / "experiments/baseline_manifest.json").write_text("promoted")
            (root / "experiments/model_decision_policy.json").write_text("promoted policy")
            with patch.object(deploy, "ROOT", root):
                actual = {key: deploy.sha256(root / key) for key in paths}
                self.assertTrue(deploy._matches_frozen_sources(actual, frozen))
                (root / "experiments/objective_losses.py").write_text("changed source")
                actual["experiments/objective_losses.py"] = deploy.sha256(
                    root / "experiments/objective_losses.py")
                self.assertFalse(deploy._matches_frozen_sources(actual, frozen))


if __name__ == "__main__":
    unittest.main()
