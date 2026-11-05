"""JSON-safe controls replay amendment for the v2 data/capacity evaluator.

The frozen core remains untouched. This wrapper uses the verified v2 recovery
path, writes a new receipt filename, and strips native prediction arrays from
JSON rows while the private NPZ emission caches retain the probabilities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments import v2_data_capacity_eval as core
from experiments import v2_data_capacity_eval_recovery as recovery


ROOT = core.ROOT
STUDY = "v2_data_capacity_eval_execution_v3"
AMENDMENT = ROOT / f"experiments/results/{STUDY}_amendment.json"
REPLAY = ROOT / "experiments/results/v2_data_capacity_eval_control_replay_v3.json"
PARTIAL = ROOT / "data/experiments/v2_data_capacity_v1/preserved_control_replay_partial_v2.json"
FAILURE_LOG = ROOT / "data/experiments/v2_data_capacity_v1/control_replay_recovery.log"
_ORIGINAL_SCORE_CELL = core._score_cell


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _strip_predictions(*args, **kwargs):
    row = _ORIGINAL_SCORE_CELL(*args, **kwargs)
    row.pop("predictions", None)
    return row


def freeze():
    if AMENDMENT.exists() or REPLAY.exists():
        raise RuntimeError("v3 amendment/replay already exists")
    if not recovery.AMENDMENT.is_file() or not PARTIAL.is_file() or not FAILURE_LOG.is_file():
        raise FileNotFoundError("Prior recovery amendment, preserved partial receipt, or failure log missing")
    value = {"study_id": STUDY,
             "core_declaration_sha256": sha(core.DECLARATION),
             "prior_recovery_amendment_sha256": sha(recovery.AMENDMENT),
             "core_source_sha256": sha(Path(core.__file__)),
             "recovery_source_sha256": sha(Path(recovery.__file__)),
             "wrapper_source_sha256": sha(Path(__file__)),
             "preserved_partial_receipt": {"path": str(PARTIAL.relative_to(ROOT)),
                                           "sha256": sha(PARTIAL), "bytes": PARTIAL.stat().st_size},
             "failure_log_sha256": sha(FAILURE_LOG),
             "new_replay_path": str(REPLAY.relative_to(ROOT)),
             "change": "write a versioned controls receipt and omit native prediction arrays from JSON rows; keep sealed NPZ probabilities unchanged",
             "candidate_scoring": False, "organizer_reference_access": False}
    AMENDMENT.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    return value


def _bind():
    if not AMENDMENT.is_file():
        raise RuntimeError("Freeze v3 execution amendment first")
    value = json.loads(AMENDMENT.read_text())
    expected = {"study_id": STUDY,
                "core_declaration_sha256": sha(core.DECLARATION),
                "prior_recovery_amendment_sha256": sha(recovery.AMENDMENT),
                "core_source_sha256": sha(Path(core.__file__)),
                "recovery_source_sha256": sha(Path(recovery.__file__)),
                "wrapper_source_sha256": sha(Path(__file__)),
                "failure_log_sha256": sha(FAILURE_LOG),
                "new_replay_path": str(REPLAY.relative_to(ROOT))}
    if any(value.get(key) != expected_value for key, expected_value in expected.items()):
        raise RuntimeError("v3 execution amendment binding changed")
    fact = value["preserved_partial_receipt"]
    if sha(ROOT / fact["path"]) != fact["sha256"]:
        raise RuntimeError("Preserved partial receipt changed")
    return value


def replay_controls():
    _bind()
    recovery._roots()
    original_replay, original_score = core.REPLAY, core._score_cell
    core.REPLAY = REPLAY
    core._score_cell = _strip_predictions
    try:
        return recovery.replay_controls()
    finally:
        core.REPLAY, core._score_cell = original_replay, original_score


def seal():
    _bind()
    recovery._roots()
    original_replay = core.REPLAY
    core.REPLAY = REPLAY
    try:
        return recovery.seal()
    finally:
        core.REPLAY = original_replay


def run():
    _bind()
    recovery._roots()
    original_replay, original_score = core.REPLAY, core._score_cell
    core.REPLAY = REPLAY
    core._score_cell = _strip_predictions
    try:
        return recovery.run()
    finally:
        core.REPLAY, core._score_cell = original_replay, original_score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "replay-controls", "seal", "run"))
    action = parser.parse_args().action
    {"freeze": freeze, "replay-controls": replay_controls, "seal": seal, "run": run}[action]()


if __name__ == "__main__":
    main()
