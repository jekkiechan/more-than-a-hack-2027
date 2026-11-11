from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import frontier_postprocess_round2_readout as readout


class FrontierPostprocessRound2ReadoutTests(unittest.TestCase):
    def test_independent_composition_preserves_fixed_masks_and_block_run_order(self):
        labels = np.asarray(list("HHHHHAAAAADDDHHHHXXXXXAAAAA"), dtype="U1")
        period = np.r_[np.ones(15, dtype=int), np.full(len(labels) - 15, 2)]
        times = np.r_[np.arange(15) / 25, np.arange(len(labels) - 15) / 25]
        ha = np.tile([.5, .5], (len(labels), 1))
        ha[2:8] = [.02, .98]
        cell = {"model_period": period, "timestamps": times, "fps": 25.}
        with patch("experiments.frontier_postprocess_round2.compose",
                   side_effect=AssertionError("runner composition must not be called")):
            actual = readout._compose_independent(labels, cell, ha)
        self.assertTrue(np.array_equal(actual == "D", labels == "D"))
        self.assertTrue(np.array_equal(actual == "X", labels == "X"))
        self.assertEqual(readout._block_sequences(actual, period),
                         readout._block_sequences(labels, period))

    def test_hash_inventory_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            artifact = root / "artifact.json"
            artifact.write_text("sealed")
            receipt = {"artifact.json": readout.sha256(artifact)}
            readout._check_inventory(receipt, root)
            artifact.write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                readout._check_inventory(receipt, root)

    def test_boundary_run_order_detects_changes_across_model_period_blocks(self):
        baseline = np.asarray(list("HHAAAHH"))
        changed = np.asarray(list("HAHAAHH"))
        period = np.asarray([1, 1, 1, 2, 2, 2, 2])
        self.assertNotEqual(readout._block_sequences(baseline, period),
                            readout._block_sequences(changed, period))


if __name__ == "__main__":
    unittest.main()
