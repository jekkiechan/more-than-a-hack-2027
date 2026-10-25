"""Compatibility entrypoints for the temporal experiment's evaluation helpers."""

from experiments.eval_set.protocol import summarize_d_boundaries


def d_boundary_metrics(labels, predicted, period, fps, timestamps=None):
    """Official live/dead transition matching, with latency reported in seconds."""
    return summarize_d_boundaries(labels, predicted, period, fps, timestamps)
