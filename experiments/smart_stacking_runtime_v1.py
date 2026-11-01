"""Metadata-only adapter for the preserved pre-inference cache lookup failure."""
import argparse
from functools import lru_cache
import json

import joblib

from experiments import smart_stacking_v1 as study

AMENDMENT = study.RESULTS / "smart_stacking_v1_runtime_amendment.json"


def cached_input(cell):
    paths = json.loads(study.shared._scalar(cell["record"], "evaluation_input_hashes_json"))
    if len(paths) != 1:
        raise ValueError("Expected exactly one identical ensemble evaluation input")
    rel, digest = next(iter(paths.items()))
    path = study.shared._resolve_artifact(rel)
    declaration = study.read(study.DECL)
    if declaration["evaluation_input_hashes"].get(rel) != digest or study.sha(path) != digest:
        raise ValueError("Ensemble evaluation input hash mismatch")
    return joblib.load(path)


def amend():
    study.check()
    assert not list((study.OUT / "level1").rglob("*.npz"))
    assert not (study.OUT / "score_lock.json").exists()
    failures = {}
    for provider in study.PROVIDERS:
        path = study.OUT / f"infer_{provider}.log"
        assert "Source prediction lacks metadata field evaluation_input_path" in path.read_text()
        failures[str(path.relative_to(study.ROOT))] = study.sha(path)
    source = study.ROOT / "experiments/smart_stacking_runtime_v1.py"
    test = study.ROOT / "experiments/tests/test_smart_stacking_runtime.py"
    study.write(AMENDMENT, dict(correction="Read the single hash-verified path in ensemble evaluation_input_hashes_json rather than the member-only evaluation_input_path. Memoize unchanged source cells to avoid repeated file I/O.",
        before_any_level1_or_candidate_score=True, failed_logs=failures,
        original_runner_preserved=True, numeric_change=False, new_parameters=False,
        declaration_sha256=study.sha(study.DECL),
        source_sha256={str(p.relative_to(study.ROOT)):study.sha(p) for p in (source,test)}))


def run(action, provider=None):
    receipt = study.read(AMENDMENT)
    for rel, digest in receipt["source_sha256"].items():
        assert study.sha(study.ROOT / rel) == digest
    study.original_cells = lru_cache(maxsize=1)(study.original_cells)
    study.cached_input = cached_input
    study.infer(provider) if action == "infer" else getattr(study, action)()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("amend", "infer", "fit", "evaluate", "fresh"))
    parser.add_argument("--provider", choices=study.PROVIDERS)
    args = parser.parse_args()
    amend() if args.action == "amend" else run(args.action, args.provider)
