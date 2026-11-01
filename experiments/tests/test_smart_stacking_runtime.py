import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from experiments import smart_stacking_runtime_v1 as runtime


class MetadataAdapterTests(unittest.TestCase):
    def test_ensemble_path_loads_identical_numeric_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "record.joblib"
            joblib.dump(dict(x=np.array([[1., 2.]])), path)
            digest = runtime.study.sha(path)
            rel = "data/external/record.joblib"
            cell = dict(record=dict(evaluation_input_hashes_json=np.asarray(json.dumps({rel:digest}))))
            with patch.object(runtime.study.shared, "_resolve_artifact", return_value=path), patch.object(runtime.study, "read", return_value=dict(evaluation_input_hashes={rel:digest})):
                np.testing.assert_array_equal(runtime.cached_input(cell)["x"], [[1., 2.]])

    def test_ambiguous_ensemble_inputs_fail_closed(self):
        c = dict(record=dict(evaluation_input_hashes_json=np.asarray('{"a":"b","c":"d"}')))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            runtime.cached_input(c)

    def test_wrong_hash_fails_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"x"
            path.write_bytes(b"not a checkpoint")
            c = dict(record=dict(evaluation_input_hashes_json=np.asarray('{"data/x":"wrong"}')))
            with patch.object(runtime.study.shared, "_resolve_artifact", return_value=path), patch.object(runtime.study, "read", return_value=dict(evaluation_input_hashes={"data/x":"wrong"})), patch.object(runtime.joblib, "load") as load:
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    runtime.cached_input(c)
                load.assert_not_called()
