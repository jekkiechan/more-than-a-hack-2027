import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import v2_training_eval as evaluation
from experiments import supervised


class ConditionalHATransferTests(unittest.TestCase):
    def test_archived_dead_probability_is_preserved_exactly(self):
        full = np.array([[.70, .20, .10], [.15, .35, .50], [.10, .10, .80]])
        d = np.array([.31, .72, 1.0])
        crossed = evaluation.cross_conditional_ha(full, d)
        np.testing.assert_array_equal(crossed[:, 2], d)
        np.testing.assert_allclose(crossed.sum(axis=1), 1.0, atol=1e-15)
        np.testing.assert_allclose(crossed[0, :2] / crossed[0, :2].sum(),
                                   full[0, :2] / full[0, :2].sum())

    def test_team_swap_symmetry(self):
        full = np.array([[.62, .28, .10], [.11, .09, .80]])
        d = np.array([.4, .7])
        original = evaluation.cross_conditional_ha(full, d)
        swapped_full = full[:, [1, 0, 2]]
        swapped = evaluation.cross_conditional_ha(swapped_full, d)
        np.testing.assert_allclose(swapped, original[:, [1, 0, 2]], atol=1e-15, rtol=0)
        np.testing.assert_array_equal(swapped[:, 2], original[:, 2])

    def test_bad_probability_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            evaluation.cross_conditional_ha(np.array([[0., 0., 1.]]), np.array([.3]))
        with self.assertRaises(ValueError):
            evaluation.cross_conditional_ha(np.array([[.5, .5, 0.]]), np.array([1.1]))


class PeriodPathTests(unittest.TestCase):
    def test_native_period_and_timestamp_breaks_are_retained(self):
        view = {"period": np.array([1, 1, 1, 2, 2, 2, 2]),
                "timestamps": np.array([0., .04, .08, 0., .04, .5, .54]),
                "fps": 25.}
        period = supervised.model_periods(view)
        np.testing.assert_array_equal(period, np.array([0, 0, 0, 1, 1, 2, 2]))

    def test_phase_decode_uses_supplied_period_path_and_masks_outside_play(self):
        probability = np.tile(np.array([[.99, .005, .005], [.005, .99, .005],
                                        [.005, .005, .99]]), (2, 1))
        period = np.array([0, 0, 1, 2, 2, 3])
        prediction = evaluation.upstream.frozen.audit.decode(probability, period, 25.)
        self.assertEqual(prediction.shape, (6,))
        self.assertEqual(prediction[2], "D")
        self.assertEqual(prediction[5], "D")
        self.assertTrue(np.isin(prediction, np.array(["H", "A", "D", "X"])).all())


class RuntimeContractTests(unittest.TestCase):
    def test_source_hash_manifest_resolves_real_files(self):
        hashes = evaluation._source_hashes()
        self.assertTrue(hashes)
        for path, digest in hashes.items():
            resolved = evaluation.ROOT / path if not path.startswith("/") else path
            self.assertTrue(resolved.is_file(), path)
            self.assertEqual(len(digest), 64, path)

    def test_upstream_bindings_match_runtime_wrapper_api(self):
        frozen = evaluation.upstream.frozen
        self.assertTrue(callable(frozen._roots))
        self.assertTrue(hasattr(frozen, "OUT"))
        self.assertTrue(hasattr(frozen, "_verify_frozen_bindings"))
        with tempfile.TemporaryDirectory(prefix="v2-eval-runtime-contract-",
                                         dir=evaluation.ROOT) as temporary:
            scratch = Path(temporary)
            declaration_path = scratch / "runtime_declaration.json"
            declaration = {"study_id": "v2_spell_target_runtime"}
            declaration_path.write_text(json.dumps(declaration))
            prepared_root = scratch / "prepared"
            prepared_root.mkdir()
            with (patch.object(frozen, "_roots"),
                  patch.object(frozen, "DECLARATION", declaration_path),
                  patch.object(frozen, "OUT", prepared_root),
                  patch.object(frozen, "_verify_frozen_bindings") as verify):
                binding = evaluation._upstream_bindings()
                verify.assert_called_once_with(declaration)
                self.assertTrue((evaluation.ROOT / binding["declaration_path"]).is_file())
                self.assertTrue((evaluation.ROOT / binding["prepared_cache_root"]).is_dir())
        self.assertEqual(binding["study_id"], "v2_spell_target_runtime")
        self.assertEqual(len(binding["declaration_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
