"""Synthetic bitwise checks for file-backed normalized training arrays."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import learned_temporal as lt, objective_losses as ol
from scripts import fit_objective_filebacked as fb


def synthetic_record(n=130):
    rng = np.random.default_rng(145)
    first_block = np.concatenate((
        np.full(5, 2), np.full(10, 0), np.full(40, 1), np.full(8, 2),
        np.full(15, 0), np.full(2, -1), np.full(10, 1)))
    second_block = np.concatenate((
        np.full(5, 2), np.full(20, 0), np.full(10, 1), np.full(5, 2)))
    y = np.concatenate((first_block, second_block)).astype(np.int8)
    if len(y) != n:
        raise ValueError("Synthetic label runs must fill the declared record")
    period = np.zeros(n, dtype=np.int32)
    period[90:] = 1  # Keep synthetic windows on both sides of a time-block break.
    x = rng.normal(size=(n, 4)).astype(np.float32)
    x[::7, 1] = np.nan
    alternate = (x + np.float32(.125)).astype(np.float32)
    chunks = list(lt.chunks(dict(y=y, period=period)))
    replace_windows = np.zeros(len(chunks), dtype=bool)
    replace_windows[1::2] = True
    replace_rows = np.zeros(n, dtype=bool)
    for use, (start, size, _full) in zip(replace_windows, chunks):
        replace_rows[start:start + size] = use
    return dict(
        source="synthetic_provider", match="synthetic_match", x=x,
        alternate=alternate, y=y, period=period,
        names=["H_distance", "A_distance", "D_speed", "D_accel"],
        replace_rows=replace_rows, replace_windows=replace_windows,
        full_y=np.repeat(y, 5), full_period=np.repeat(period, 5),
        indices=np.arange(n, dtype=np.int64) * 5, fps=25.0)


def assert_exact_tree(testcase, left, right, where="model"):
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        a, b = np.asarray(left), np.asarray(right)
        testcase.assertEqual(a.shape, b.shape, where)
        testcase.assertEqual(a.dtype, b.dtype, where)
        np.testing.assert_array_equal(a, b, err_msg=where)
    elif isinstance(left, dict) and isinstance(right, dict):
        testcase.assertEqual(set(left), set(right), where)
        for key in left:
            assert_exact_tree(testcase, left[key], right[key], f"{where}.{key}")
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        testcase.assertEqual(len(left), len(right), where)
        for index, (a, b) in enumerate(zip(left, right)):
            assert_exact_tree(testcase, a, b, f"{where}[{index}]")
    else:
        testcase.assertEqual(left, right, where)


class FileBackedObjectiveContracts(unittest.TestCase):
    def test_all_noncontrol_objectives_are_bitwise_identical(self):
        record = synthetic_record()
        factors = ol.spell_duration_factors(
            record["full_y"], record["full_period"], record["indices"],
            record["y"], record["fps"])
        self.assertGreater(len(np.unique(factors)), 1,
                           "Synthetic spell arm must exercise non-neutral duration weights")
        base_normalizer = lt.normalize
        with tempfile.TemporaryDirectory(prefix="objective-filebacked-test-") as temp_root:
            for arm in ("spell_sqrt", "transition", "combined"):
                seed = 2026
                with patch.object(lt, "EPOCHS", 1):
                    expected = ol.fit([record], player_only=True, seed=seed, objective=arm)
                    with fb.filebacked_normalize(Path(temp_root)) as storage:
                        actual = ol.fit([record], player_only=True, seed=seed, objective=arm)

                self.assertIs(lt.normalize, base_normalizer)
                self.assertTrue(storage["normalizer_restored"])
                self.assertTrue(storage["temporary_files_removed"])
                self.assertTrue(storage["all_mappings_read_only"])
                self.assertEqual(storage["normalize_calls"], 2)
                self.assertEqual(len(storage["mapped_arrays"]), 2)
                self.assertTrue(all(not item["writeable"]
                                    for item in storage["mapped_arrays"]))
                self.assertFalse(Path(storage["temp_directory"]).exists())

                # This traverses every returned field, including every tensor
                # state array, training history, and normalization statistic.
                assert_exact_tree(self, expected, actual, f"{arm}.model")
                expected_prediction = ol.predict(record, expected)
                actual_prediction = ol.predict(record, actual)
                np.testing.assert_array_equal(actual_prediction, expected_prediction,
                                              err_msg=f"{arm}.prediction")

    def test_wrapper_restores_normalizer_after_training_exception(self):
        base_normalizer = lt.normalize
        with tempfile.TemporaryDirectory(prefix="objective-filebacked-error-") as temp_root:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                with fb.filebacked_normalize(Path(temp_root)) as storage:
                    mapped = lt.normalize(np.ones((3, 2), np.float32),
                                          np.zeros(2), np.ones(2))
                    self.assertFalse(mapped.flags.writeable)
                    raise RuntimeError("synthetic failure")
            self.assertIs(lt.normalize, base_normalizer)
            self.assertTrue(storage["normalizer_restored"])
            self.assertTrue(storage["temporary_files_removed"])
            self.assertEqual(storage["normalize_calls"], 1)


if __name__ == "__main__":
    unittest.main()
