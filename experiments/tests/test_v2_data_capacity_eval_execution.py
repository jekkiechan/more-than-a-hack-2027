import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import v2_data_capacity_eval_execution as execution


class PreparedTupleCacheTests(unittest.TestCase):
    def test_cached_tuple_reuses_exact_value_and_rejects_input_mutation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "match_clean_clean.joblib"
            raw.write_bytes(b"raw-input")
            calibration = root / "calibration.json"
            calibration.write_text("{}")
            amendment = root / "amendment.json"
            amendment.write_text("{}\n")
            cache_root = root / "cache"
            original = {"native": [1, 2, 3], "phases": [{"x": [4, 5]}]}
            with (patch.object(execution.core.runtime.frozen, "_roots", return_value=None),
                  patch.object(execution.core.audit, "EVAL_INPUTS", root),
                  patch.object(execution.core.audit, "FILE", {"match": "match"}),
                  patch.object(execution.core.runtime.frozen, "CALIBRATION", calibration),
                  patch.object(execution, "OUT", cache_root),
                  patch.object(execution, "AMENDMENT", amendment),
                  patch.object(execution, "_ORIGINAL_RAW_CACHE", return_value=original) as build):
                first = execution.cached_raw_cache("P", "match", "clean", 0)
                second = execution.cached_raw_cache("P", "match", "clean", 0)
                self.assertEqual(first, original)
                self.assertEqual(second, original)
                build.assert_called_once()
                raw.write_bytes(b"mutated-input")
                with self.assertRaises(RuntimeError):
                    execution.cached_raw_cache("P", "match", "clean", 0)

    def test_freeze_core_uses_only_verified_paths_and_restores_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            declaration = Path(temp) / "core-declaration.json"
            raw_path = Path(temp) / "raw.joblib"
            original = execution.core._raw_cache
            observed = []
            def fake_freeze():
                observed.append(execution.core._raw_cache("P", "m", "clean", 0))
                return {"frozen": True}
            with (patch.object(execution.core, "DECLARATION", declaration),
                  patch.object(execution.core, "_raw_cache", original),
                  patch.object(execution.core, "freeze", side_effect=fake_freeze),
                  patch.object(execution, "_identity", return_value={"raw_path": str(raw_path)})):
                self.assertEqual(execution.freeze_core(), {"frozen": True})
                self.assertEqual(observed, [(raw_path,)])
                self.assertIs(execution.core._raw_cache, original)


if __name__ == "__main__":
    unittest.main()
