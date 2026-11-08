"""Supplemental source freeze and execution wrapper for frontier recovery.

The frozen training module is deliberately left byte-identical to its
declaration. This wrapper binds the five transitive numerical dependencies
which were omitted from its original source list before delegating fitting.
"""
from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path

from experiments import frontier_recovery_training_v1 as training

ROOT = Path(__file__).resolve().parents[1]
OUT = training.OUT
SUPPLEMENT = OUT / "source_supplement.json"
EXECUTION = OUT / "execution_receipt.json"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extra_source_paths() -> list[Path]:
    return sorted({
        ROOT / "experiments/datasets.py",
        ROOT / "experiments/dense_short_team_v1.py",
        ROOT / "experiments/feed_short_sampler.py",
        ROOT / "experiments/refine_features.py",
        ROOT / "experiments/short_label_quality.py",
    })


def _frozen_training_hash() -> str:
    declaration = json.loads(training.TRAIN_DECL.read_text())
    key = "experiments/frontier_recovery_training_v1.py"
    expected = declaration.get("source_sha256", {}).get(key)
    current = sha(ROOT / key)
    if expected is None or current != expected:
        raise ValueError("Frozen training runner no longer matches its training declaration")
    return current


def _bindings() -> dict:
    training._check_training_binding()
    if not training.TRAIN_DECL.is_file():
        raise FileNotFoundError("Freeze training inputs before freezing supplemental sources")
    train_decl_sha = sha(training.TRAIN_DECL)
    training_hash = _frozen_training_hash()
    missing = [str(path) for path in _extra_source_paths() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Supplemental numerical source missing: " + ", ".join(missing))
    return {
        "schema": "frontier_recovery_source_supplement_v1",
        "study_id": training.STUDY,
        "training_declaration_sha256": train_decl_sha,
        "training_runner_sha256": training_hash,
        "additional_numerical_sources_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in _extra_source_paths()
        },
        "execution_wrapper_sha256": sha(Path(__file__)),
        "scoring_performed": False,
        "organizer_reference_access": False,
    }


def freeze_sources() -> dict:
    """Write once the supplemental source inventory before any fits."""
    if SUPPLEMENT.exists() or training.FIT_SEAL.exists() or EXECUTION.exists():
        raise FileExistsError("Supplemental freeze or training execution already exists")
    value = _bindings()
    SUPPLEMENT.parent.mkdir(parents=True, exist_ok=True)
    with SUPPLEMENT.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return value


def check_sources() -> dict:
    """Refuse execution if any frozen training or supplement binding changed."""
    if not SUPPLEMENT.is_file():
        raise FileNotFoundError("Freeze supplemental sources before fitting")
    declared = json.loads(SUPPLEMENT.read_text())
    actual = _bindings()
    if declared != actual:
        raise ValueError("Supplemental training source/input binding changed after freeze")
    return declared


def run(workers: int = 2) -> dict:
    """Validate all frozen inputs, fit only the declared six provider folds, receipt."""
    if workers != 2:
        raise ValueError("Frontier recovery uses the frozen two-worker execution budget")
    if EXECUTION.exists():
        raise FileExistsError("Execution receipt already exists; preserve frozen outputs")
    supplement = check_sources()
    fit_seal = training.fit(workers=workers)
    receipt = {
        "schema": "frontier_recovery_execution_v1",
        "study_id": training.STUDY,
        "source_supplement_sha256": sha(SUPPLEMENT),
        "training_declaration_sha256": sha(training.TRAIN_DECL),
        "training_fit_seal_sha256": sha(training.FIT_SEAL),
        "fit_count": len(fit_seal.get("fits", [])),
        "all_provider_folds_complete": fit_seal.get("all_provider_folds_complete") is True,
        "source_status": "reconstructed",
        "organizer_reference_access": False,
        "scoring_performed": False,
        "promotion": False,
    }
    if receipt["fit_count"] != 12 or not receipt["all_provider_folds_complete"]:
        raise RuntimeError("Recovery fit seal is incomplete")
    EXECUTION.parent.mkdir(parents=True, exist_ok=True)
    with EXECUTION.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze-sources", "run"))
    parser.add_argument("--workers", type=int, default=2, choices=(2,))
    args = parser.parse_args()
    result = freeze_sources() if args.action == "freeze-sources" else run(args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
