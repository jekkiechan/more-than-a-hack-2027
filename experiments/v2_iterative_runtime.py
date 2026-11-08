"""Exact fast path for the two row-wise NaN quantiles used by refinement.

The optional patch is process-local and additive. It leaves the frozen feature
builder untouched and delegates every unsupported call to NumPy's original
``nanquantile`` implementation.
"""
from __future__ import annotations

from contextlib import contextmanager
import warnings

import numpy as np

_ORIGINAL_NANQUANTILE = np.nanquantile


def _eligible(a, q, axis, out, overwrite_input, method, keepdims, weights):
    if type(a) is not np.ndarray or a.dtype != np.dtype(np.float64) or a.ndim != 2:
        return False
    if a.shape[0] == 0 or a.shape[1] == 0 or axis != 1:
        return False
    if out is not None or overwrite_input or method != "linear" or keepdims or weights is not None:
        return False
    if np.ndim(q) != 0:
        return False
    try:
        quantile = float(q)
    except (TypeError, ValueError, OverflowError):
        return False
    if quantile not in (0.5, 0.9):
        return False
    return not bool(np.isinf(a).any())


def nanquantile(a, q, axis=None, out=None, overwrite_input=False,
                method="linear", keepdims=False, *, weights=None,
                _original=None):
    """Use sort/count/linear interpolation for the exact supported call shape."""
    original = _ORIGINAL_NANQUANTILE if _original is None else _original
    if not _eligible(a, q, axis, out, overwrite_input, method, keepdims, weights):
        kwargs = dict(axis=axis, out=out, overwrite_input=overwrite_input,
                      method=method, keepdims=keepdims)
        if weights is not None:
            kwargs["weights"] = weights
        return original(a, q, **kwargs)

    quantile = float(q)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ordered = np.sort(a, axis=1)
        counts = np.count_nonzero(~np.isnan(a), axis=1)
        valid = counts > 0
        result = np.full(a.shape[0], np.nan, dtype=np.float64)
        if valid.any():
            virtual = quantile * (counts[valid] - 1)
            lower_index = np.floor(virtual).astype(np.intp)
            upper_index = np.minimum(lower_index + 1, counts[valid] - 1)
            rows = np.flatnonzero(valid)
            lower = ordered[rows, lower_index]
            upper = ordered[rows, upper_index]
            fraction = virtual - lower_index
            difference = upper - lower
            # Keep NumPy's _lerp operation order and its upper-branch form for
            # gamma >= .5; algebraically equivalent rewrites can change bits.
            with np.errstate(over="ignore", invalid="ignore"):
                interpolated = lower + difference * fraction
                upper_branch = upper - difference * (1.0 - fraction)
                interpolated = np.where(fraction >= 0.5, upper_branch, interpolated)
            result[rows] = interpolated
        return result


@contextmanager
def installed():
    """Patch ``np.nanquantile`` only inside the caller's process/context."""
    original = np.nanquantile

    def patched(a, q, axis=None, out=None, overwrite_input=False,
                method="linear", keepdims=False, *, weights=None):
        return nanquantile(a, q, axis=axis, out=out,
                           overwrite_input=overwrite_input, method=method,
                           keepdims=keepdims, weights=weights, _original=original)

    np.nanquantile = patched
    try:
        yield patched
    finally:
        np.nanquantile = original
