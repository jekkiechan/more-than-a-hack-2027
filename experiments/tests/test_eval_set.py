"""Discover the assets-free evaluation protocol contracts in the normal suite."""

import unittest

from experiments.eval_set.test_protocol import ProtocolTests  # noqa: F401
from experiments.eval_set.metrics import d_boundary_metrics


class TemporalMetricsEntrypointTests(unittest.TestCase):
    def test_runner_compatible_signature(self):
        result = d_boundary_metrics(["H", "H", "D", "D"], ["A", "A", "D", "D"],
                                    period=[1, 1, 1, 1], fps=10,
                                    timestamps=[0, .1, .2, .3])
        self.assertEqual(result["0.5"]["f1"], 1)
