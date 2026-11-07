import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import v2_data_capacity_eval_recovery as recovery


class ControlsRecoveryTests(unittest.TestCase):
    def test_roots_are_bound_before_recovery_artifact_lookups(self):
        def bind_roots():
            recovery.core.audit.DATA = Path("/workspace/mtah-dt")
            recovery.core.audit.SQD = Path(
                "/workspace/mtah-sqd/experiments/short_quality_diversity_delivery_v1/artifacts")
        with patch.object(recovery.core.runtime.frozen, "_roots", side_effect=bind_roots):
            recovery._roots()

    def test_saved_control_npz_is_verified_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "control.npz"
            row = {"labels": np.asarray(["H", "D"]),
                   "timestamps": np.asarray([0., 1.]),
                   "period": np.asarray([1, 1]),
                   "model_period": np.asarray([1, 1]),
                   "fps": 25.0,
                   "probabilities": {"cand": np.asarray([[.6, .3, .1], [.1, .2, .7]])}}
            identity = {"study_id": "id", "provider": "P"}
            arrays = {key: np.asarray(value) for key, value in row.items()
                      if key in {"labels", "timestamps", "period", "model_period", "fps"}}
            arrays.update({f"prob__{name}": value for name, value in row["probabilities"].items()})
            arrays.update({f"identity__{key}": np.asarray(str(value))
                           for key, value in identity.items()})
            np.savez_compressed(path, **arrays)
            before = recovery.sha(path)
            self.assertEqual(recovery._assert_unchanged_cache(path, row, identity), before)
            self.assertEqual(recovery.sha(path), before)
            row["probabilities"]["cand"][0, 0] = .5
            with self.assertRaises(RuntimeError):
                recovery._assert_unchanged_cache(path, row, identity)

    def test_canonical_aggregation_is_insertion_order_independent(self):
        pairs = [(('P2', 'm2', 1), .1), (('P1', 'm1', 1), .2),
                 (('P2', 'm1', 1), .3), (('P1', 'm2', 1), .4)]
        first = recovery._canonical_equal_provider(dict(pairs))
        second = recovery._canonical_equal_provider(dict(reversed(pairs)))
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
