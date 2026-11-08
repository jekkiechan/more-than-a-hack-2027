"""Command-line dispatch for the frozen v2 weight-edit development functions.

The sealed scientific runner remains unchanged. This wrapper only fixes its
CLI argument forwarding and verifies the runner/declaration hashes before any
action is dispatched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments import v2_weight_edit_development as development


FROZEN_SOURCE_SHA256 = "6fd19be1bddb409dd99faeba119a4c4a5ec3d28e7af47af33b2cafb799830fb1"
FROZEN_DECLARATION_SHA256 = "d94dcf3db9dd0af5b73bdca87d6e5139a3baea8afd4522e0ae77a4d36d13bad5"


def verify_binding() -> None:
    source_hash = development.sha(Path(development.__file__))
    declaration_hash = development.sha(development.DECLARATION)
    if source_hash != FROZEN_SOURCE_SHA256:
        raise RuntimeError(f"Frozen development source hash mismatch: {source_hash}")
    if declaration_hash != FROZEN_DECLARATION_SHA256:
        raise RuntimeError(f"Frozen development declaration hash mismatch: {declaration_hash}")


def dispatch(action: str, workers: int = 3):
    verify_binding()
    if workers < 1 or workers > 5:
        raise ValueError("workers must be in [1, 5]")
    if action == "freeze":
        return development.freeze()
    if action == "control-preflight":
        return development.control_preflight()
    if action == "control":
        return development.control_replay(workers=workers)
    if action == "score-primary":
        return development.run_primary(workers=workers)
    if action == "freeze-winner":
        return development.freeze_winner()
    if action == "score-v1":
        return development.score_v1(workers=workers)
    raise ValueError(f"Unknown development action: {action}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "control-preflight", "control",
                                             "score-primary", "freeze-winner", "score-v1"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = dispatch(args.action, workers=args.workers)
    print(json.dumps(result, indent=1, sort_keys=True, default=float), flush=True)


if __name__ == "__main__":
    main()
