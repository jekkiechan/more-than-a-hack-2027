"""Experimental NumPy-only exact-parity Potts decoder runtime candidate.

This module is not integrated into the adopted baseline. It provides an opt-in
decoder and a reproducible synthetic parity/timing probe against
``experiments.supervised.decode``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments import supervised


def decode(probabilities, period, fps, switch_cost=0):
    """Decode with the supervised Potts objective using a smaller recurrence.

    The recurrence retains the reference implementation's first-index tie
    behavior for both predecessor selection and final-state selection.
    """
    valid_classes = np.flatnonzero(probabilities.sum(axis=0) > 0)
    probabilities = probabilities[:, valid_classes]
    if switch_cost <= 0:
        return supervised.LABELS[valid_classes[np.argmax(probabilities, axis=1)]]

    logp = np.log(np.maximum(probabilities, 1e-7))
    penalty = switch_cost * fps / 10
    result = np.empty(len(period), dtype=np.int8)
    for start, stop in supervised.blocks(period):
        nclasses = len(valid_classes)
        back = np.zeros((stop - start, nclasses), dtype=np.int8)
        value = logp[start].copy()
        for row in range(start + 1, stop):
            penalized = value - penalty
            next_value = np.empty(nclasses, dtype=value.dtype)
            previous = np.empty(nclasses, dtype=np.int8)
            for destination in range(nclasses):
                alternatives = penalized.copy()
                alternatives[destination] = -np.inf
                source = int(np.argmax(alternatives))
                # The reference argmaxes after subtracting the penalty, so
                # rounding can create ties between distinct predecessor values.
                switched = alternatives[source]
                stayed = value[destination]
                # np.argmax chooses the lower source index on a tie.
                if switched > stayed or (switched == stayed and source < destination):
                    previous[destination] = source
                    next_value[destination] = switched
                else:
                    previous[destination] = destination
                    next_value[destination] = stayed
            back[row - start] = previous
            value = next_value + logp[row]

        state = int(np.argmax(value))
        for row in range(stop - 1, start - 1, -1):
            result[row] = valid_classes[state]
            state = int(back[row - start, state])
    return supervised.LABELS[result]


def parity_case_count(seed=991):
    """Return exact candidate/reference parity count for deterministic cases."""
    rng = np.random.default_rng(seed)
    count = 0
    for n in (1, 2, 3, 10, 97, 1000):
        for classes in (1, 2, 3):
            for trial in range(40):
                probabilities = rng.random((n, 3))
                probabilities[:, classes:] = 0
                if trial % 4 == 0 and n > 1:
                    probabilities[:] = 1
                    probabilities[:, classes:] = 0
                elif trial % 4 == 1 and n > 1:
                    probabilities[:] = .25
                    probabilities[:, classes:] = 0
                if n > 2 and trial % 5 == 0:
                    probabilities[1] = 0
                period = np.zeros(n, dtype=np.int64)
                if n > 1:
                    period[n // 2:] = 1
                for cost in (0, 3, 30, 300):
                    reference = supervised.decode(probabilities, period, 25., cost)
                    candidate = decode(probabilities, period, 25., cost)
                    if not np.array_equal(reference, candidate):
                        raise AssertionError(
                            f"parity failed: n={n}, classes={classes}, trial={trial}, cost={cost}")
                    count += 1
    return count


def benchmark(seed=991, frames=199319, repeats=3):
    """Run fixed-seed edge parity and full-length synthetic timings."""
    if frames < 1 or repeats < 1:
        raise ValueError("frames and repeats must be positive")
    parity_cases = parity_case_count(seed)
    rng = np.random.default_rng(seed)
    probabilities = rng.random((frames, 3))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    probabilities[::997] = 1 / 3
    period = np.zeros(frames, dtype=np.int32)
    if frames > 1:
        period[frames // 2:] = 1

    reference_times = []
    candidate_times = []
    reference_output = candidate_output = None
    for _ in range(repeats):
        start = time.perf_counter()
        reference_output = supervised.decode(probabilities, period, 25., 3)
        reference_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        candidate_output = decode(probabilities, period, 25., 3)
        candidate_times.append(time.perf_counter() - start)
    exact = bool(np.array_equal(reference_output, candidate_output))
    if not exact:
        raise AssertionError("full-length benchmark output differs from reference")
    return {
        "seed": seed,
        "parity_cases": parity_cases,
        "frames": frames,
        "fps": 25.0,
        "switch_cost": 3,
        "repeats": repeats,
        "full_length_exact_parity": exact,
        "reference_seconds": reference_times,
        "candidate_seconds": candidate_times,
        "reference_median_seconds": float(np.median(reference_times)),
        "candidate_median_seconds": float(np.median(candidate_times)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--frames", type=int, default=199319)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json-output", type=Path,
                        help="write JSON report with exclusive creation")
    args = parser.parse_args()
    report = benchmark(args.seed, args.frames, args.repeats)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.json_output:
        with args.json_output.open("x") as handle:
            handle.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
