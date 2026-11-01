"""Preserve the pre-score failure; normalize scalar NPZ metadata only."""
import argparse
from pathlib import Path

from experiments import smart_stacking_runtime_v1 as initial

study = initial.study
AMENDMENT = study.RESULTS / "smart_stacking_v1_scalar_amendment.json"
ORIGINAL_LEVEL1 = study.level1
ORIGINAL_WRITE = study.write


def level1(provider, member):
    cells = ORIGINAL_LEVEL1(provider, member)
    for cell in cells:
        cell["fps"] = float(cell["fps"])
    return cells


def preserving_write(path, value):
    if Path(path) == study.OUT / "score_lock.json":
        if study.read(path) != value:
            raise ValueError("Cannot replace the original scoring lock")
        return
    ORIGINAL_WRITE(path, value)


def amend():
    study.check()
    failure = study.OUT / "evaluate.log"
    assert "numpy.ndarray doesn't define __round__" in failure.read_text()
    assert not list((study.OUT / "predictions").rglob("*.npz"))
    assert not (study.RESULTS / f"{study.STUDY}_aggregate.json").exists()
    paths = [Path(__file__), study.ROOT / "experiments/tests/test_smart_stacking_scalar_runtime.py"]
    study.write(AMENDMENT, dict(correction="Convert NPZ zero-dimensional fps metadata to the identical Python float before official scoring. Preserve the existing identical scoring lock; no completed score/output exists.",
        failure_sha256=study.sha(failure), scoring_lock_sha256=study.sha(study.OUT / "score_lock.json"),
        model_seal_sha256=study.sha(study.RESULTS / f"{study.STUDY}_model_seal.json"),
        source_sha256={str(p.relative_to(study.ROOT)):study.sha(p) for p in paths},
        new_arms=False, numerical_value_changed=False, prior_score_outputs=0))


def run():
    for rel, digest in study.read(AMENDMENT)["source_sha256"].items():
        assert study.sha(study.ROOT / rel) == digest
    study.level1 = level1
    study.write = preserving_write
    initial.run("evaluate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("amend", "evaluate"))
    amend() if parser.parse_args().action == "amend" else run()
