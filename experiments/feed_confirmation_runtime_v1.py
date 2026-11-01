"""Bounded metadata-only correction for frozen fresh-feature evaluation.

The preparation helper stores feature names as a NumPy array; the frozen
factorized predictor compares Python lists. Convert names at the feature-only
boundary, preserving all numerical values, sources, models, masks and targets.
"""
import argparse
from pathlib import Path

import numpy as np
from experiments import feed_robustness_v1 as parent
from experiments import feed_robustness_confirmation_v1 as confirmation
from experiments import feed_seed_diversity_v1 as diversity
from experiments import soccernet_extension as old

ORIGINAL_INFERENCE = old.inference_record


def normalized_inference(record):
    result = ORIGINAL_INFERENCE(record)
    result["names"] = np.asarray(result["names"]).tolist()
    return result


def run(mode):
    parent.check()
    confirmation.check()
    paths = [Path(__file__), parent.ROOT / "experiments/feed_robustness_confirmation_v1.py",
             parent.ROOT / "experiments/feed_seed_diversity_v1.py"]
    amendment = parent.RESULTS / f"feed_confirmation_runtime_v1_{mode}_amendment.json"
    original_write = parent.write
    receipt = {"mode": mode, "correction": "Convert feature-only names ndarray to an equal Python list before frozen inference",
        "source_sha256": {str(p.relative_to(parent.ROOT)): parent.sha(p) for p in paths},
        "numeric_features_changed": False, "labels_or_models_changed": False,
        "new_arms_or_parameters": False, "organizer_reference_access": False}
    if mode == "parent":
        lock = confirmation.OUT / "score_lock.json"
        prior = parent.read(lock)
        if any((confirmation.OUT / "predictions").rglob("*.npz")):
            raise ValueError("This amendment resumes only the recorded pre-scoring failure")
        failure = confirmation.OUT / "initial_runtime_failure.log"
        if not failure.exists() or "truth value of an array" not in failure.read_text():
            raise ValueError("Preserved original failure log required")
        receipt.update(failed_lock_sha256=parent.sha(lock), failure_log_sha256=parent.sha(failure),
                       prior_prediction_files=0, lock_preserved=True)
        def preserving_write(path, value):
            if Path(path) == lock:
                if prior != value or parent.read(lock) != value:
                    raise ValueError("Original scoring lock differs")
                return
            original_write(path, value)
        parent.write = preserving_write
    original_write(amendment, receipt)
    old.inference_record = normalized_inference
    try:
        confirmation.run() if mode == "parent" else diversity.fresh()
    finally:
        old.inference_record = ORIGINAL_INFERENCE
        parent.write = original_write


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("parent", "diversity"))
    run(parser.parse_args().mode)
