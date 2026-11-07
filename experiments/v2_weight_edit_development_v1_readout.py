"""Execution-only v1 readout adapter for the frozen development runner.

The frozen v1 action includes the archived original_full control in the same
arm list as editable new-model arms. This adapter filters that control (which
the scorer adds automatically) and deduplicates a winner equal to the
unedited-blend control before dispatching cells. No model, decoder, cohort,
metric, or selection logic changes.
"""
from __future__ import annotations

import argparse
import json
from unittest import mock

from experiments import v2_weight_edit_development as development
from experiments import v2_weight_edit_development_execution as execution


def _editable_v1_cell_call(original, *args, **kwargs):
    if len(args) < 5:
        raise TypeError("expected the frozen _score_cell positional signature")
    args = list(args)
    args[4] = list(dict.fromkeys(arm for arm in args[4]
                                  if arm != "deployed_original_full"))
    if not args[4]:
        raise ValueError("v1 readout requires at least one editable blend arm")
    return original(*args, **kwargs)


def run_v1(workers: int = 3):
    execution.verify_binding()
    original = development._score_cell
    def safe_score_cell(*args, **kwargs):
        return _editable_v1_cell_call(original, *args, **kwargs)
    with mock.patch.object(development, "_score_cell", side_effect=safe_score_cell):
        return development.score_v1(workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = run_v1(workers=args.workers)
    print(json.dumps(result, indent=1, sort_keys=True, default=float), flush=True)


if __name__ == "__main__":
    main()
