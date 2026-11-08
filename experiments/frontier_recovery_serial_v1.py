"""Serial, fresh-process execution amendment for frontier recovery.

The numerical fits remain the frozen ``frontier_recovery_training_v1`` jobs.
Each fit runs in a separate Python process so provider-fold caches are released
before the next job starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments import frontier_recovery_execution_v1 as execution
from experiments import frontier_recovery_training_v1 as training

ROOT = training.ROOT
OUT = training.OUT
AMENDMENT = OUT / "serial_execution_amendment.json"
SERIAL_LOG = Path("/tmp/frontier-recovery-serial-fit.log")
PRIOR_FAILED_LOG = Path("/tmp/frontier-recovery-fit.log")


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _jobs() -> list[tuple[int, str, int]]:
    return ([(512, fold, seed) for fold in training.FOLDS for seed in training.SEEDS]
            + [(1024, fold, seed) for fold in training.FOLDS for seed in training.SEEDS])


def _inventory() -> list[str]:
    models = OUT / "models"
    return sorted(str(path.relative_to(ROOT)) for path in models.rglob("*")
                  if path.is_file()) if models.exists() else []


def _check_child_sources() -> None:
    """Cheap between-fit guard; the full data/input hash check runs in parent."""
    declaration = json.loads(training.TRAIN_DECL.read_text())
    train_runner = "experiments/frontier_recovery_training_v1.py"
    if sha(ROOT / train_runner) != declaration.get("source_sha256", {}).get(train_runner):
        raise ValueError("Frozen training runner changed after source freeze")
    if training._training_source_hashes() != declaration.get("source_sha256"):
        raise ValueError("Frozen training sources changed during serial recovery")

    supplement = json.loads(execution.SUPPLEMENT.read_text())
    if supplement.get("training_declaration_sha256") != sha(training.TRAIN_DECL):
        raise ValueError("Training declaration changed after supplemental freeze")
    if supplement.get("execution_wrapper_sha256") != sha(Path(execution.__file__)):
        raise ValueError("Frozen execution wrapper changed after supplemental freeze")
    if supplement.get("training_runner_sha256") != sha(ROOT / train_runner):
        raise ValueError("Supplemental binding points at another training runner")
    extra_paths = execution._extra_source_paths()
    extra = {str(path.relative_to(ROOT)): sha(path) for path in extra_paths}
    if extra != supplement.get("additional_numerical_sources_sha256"):
        raise ValueError("Supplemental numerical dependencies changed after freeze")
    if not AMENDMENT.is_file():
        raise FileNotFoundError("Serial execution amendment is not frozen")
    amendment = json.loads(AMENDMENT.read_text())
    if (amendment.get("serial_execution_wrapper_sha256") != sha(Path(__file__)) or
            amendment.get("training_declaration_sha256") != sha(training.TRAIN_DECL) or
            amendment.get("supplement_sha256") != sha(execution.SUPPLEMENT) or
            amendment.get("parallel_execution_wrapper_sha256") != sha(Path(execution.__file__))):
        raise ValueError("Serial execution wrapper changed after freeze")
    if amendment.get("parallel_failure", {}).get("log_sha256") != sha(PRIOR_FAILED_LOG):
        raise ValueError("Preserved failed parallel log changed after freeze")


def _freeze() -> dict:
    if AMENDMENT.exists() or training.FIT_SEAL.exists() or execution.EXECUTION.exists():
        raise FileExistsError("A serial amendment or execution receipt already exists")
    if not PRIOR_FAILED_LOG.is_file():
        raise FileNotFoundError(f"Expected preserved failed parallel log: {PRIOR_FAILED_LOG}")
    if not execution.SUPPLEMENT.is_file():
        raise FileNotFoundError("Freeze the supplemental numerical source inventory first")
    training._check_training_binding()
    supplement = execution.check_sources()
    inventory = _inventory()
    if inventory:
        raise FileExistsError("Refusing serial recovery because fit outputs already exist: "
                              + ", ".join(inventory))
    jobs = _jobs()
    value = {
        "schema": "frontier_recovery_serial_execution_amendment_v1",
        "study_id": training.STUDY,
        "status": "frozen_serial_recovery_before_fit",
        "training_declaration_sha256": sha(training.TRAIN_DECL),
        "training_runner_sha256": sha(ROOT / "experiments/frontier_recovery_training_v1.py"),
        "supplement_sha256": sha(execution.SUPPLEMENT),
        "parallel_execution_wrapper_sha256": sha(Path(execution.__file__)),
        "serial_execution_wrapper_sha256": sha(Path(__file__)),
        "parallel_failure": {
            "log_path": str(PRIOR_FAILED_LOG), "log_sha256": sha(PRIOR_FAILED_LOG),
            "worker_count": 2, "cgroup_memory_limit_bytes": 17179869184,
            "oom_kill_events": 1, "fit_receipts_at_freeze": 0,
            "fit_outputs_at_freeze": inventory,
        },
        "execution_change": "same twelve frozen training jobs and numerical configuration, run sequentially in fresh Python processes; no change to source, inputs, seed, sampler, or fit implementation",
        "jobs": [{"stage": stage, "fold": fold, "seed": seed} for stage, fold, seed in jobs],
        "job_count": len(jobs), "workers": 1, "fresh_process_per_fit": True,
        "organizer_reference_access": False, "scoring_performed": False,
        "candidate_scoring": False, "promotion": False,
    }
    if len(jobs) != 12 or not supplement.get("additional_numerical_sources_sha256"):
        raise RuntimeError("Unexpected serial job inventory or missing source supplement")
    AMENDMENT.parent.mkdir(parents=True, exist_ok=True)
    with AMENDMENT.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return value


def _validate_frozen() -> dict:
    if not AMENDMENT.is_file():
        raise FileNotFoundError("Freeze the serial execution amendment before fitting")
    amendment = json.loads(AMENDMENT.read_text())
    training._check_training_binding()
    supplement = execution.check_sources()
    if amendment.get("training_declaration_sha256") != sha(training.TRAIN_DECL):
        raise ValueError("Training declaration changed after serial freeze")
    if amendment.get("training_runner_sha256") != sha(ROOT / "experiments/frontier_recovery_training_v1.py"):
        raise ValueError("Frozen training runner changed after serial freeze")
    if amendment.get("supplement_sha256") != sha(execution.SUPPLEMENT):
        raise ValueError("Supplement changed after serial freeze")
    if amendment.get("parallel_execution_wrapper_sha256") != sha(Path(execution.__file__)):
        raise ValueError("Parallel execution wrapper changed after serial freeze")
    if amendment.get("serial_execution_wrapper_sha256") != sha(Path(__file__)):
        raise ValueError("Serial execution wrapper changed after serial freeze")
    if amendment.get("parallel_failure", {}).get("log_sha256") != sha(PRIOR_FAILED_LOG):
        raise ValueError("Preserved failed parallel log changed after serial freeze")
    expected_jobs = [{"stage": s, "fold": f, "seed": seed} for s, f, seed in _jobs()]
    if amendment.get("jobs") != expected_jobs or amendment.get("job_count") != 12:
        raise ValueError("Serial fit job inventory changed after freeze")
    if not supplement.get("additional_numerical_sources_sha256"):
        raise ValueError("Supplemental numerical closure is empty")
    return amendment


def _fit_one(stage: int, fold: str, seed: int) -> dict:
    if (stage, fold, seed) not in _jobs():
        raise ValueError("Undeclared frontier recovery fit")
    _check_child_sources()
    row = training._fit_stage((stage, fold, seed))
    model_path = training._stage_path(stage, fold, seed)
    receipt_path = Path(str(training._stage_path(stage, fold, seed)) + ".json")
    if not receipt_path.is_file() or sha(model_path) != row.get("sha256"):
        raise RuntimeError("Fit receipt was not written")
    sidecar = json.loads(receipt_path.read_text())
    if sidecar != row or sidecar.get("declaration_sha256") != sha(training.TRAIN_DECL):
        raise RuntimeError("Fit sidecar does not match the returned recovery receipt")
    return row


def _verified_existing(job: tuple[int, str, int]) -> dict | None:
    stage, fold, seed = job
    model_path = training._stage_path(stage, fold, seed)
    receipt_path = Path(str(model_path) + ".json")
    if model_path.exists() != receipt_path.exists():
        raise FileExistsError(f"Refusing partial fit output pair: {model_path}")
    if not model_path.exists():
        return None
    row = json.loads(receipt_path.read_text())
    if (row.get("stage") != stage or row.get("fold") != fold or
            int(row.get("seed", -1)) != seed or
            row.get("declaration_sha256") != sha(training.TRAIN_DECL) or
            row.get("sha256") != sha(model_path)):
        raise ValueError(f"Refusing inconsistent completed fit: {model_path}")
    if row.get("study_id") != training.STUDY or row.get("source_status") != "reconstructed":
        raise ValueError(f"Completed fit has wrong provenance: {model_path}")
    return row


def _run() -> dict:
    amendment = _validate_frozen()
    if training.FIT_SEAL.exists() or execution.EXECUTION.exists():
        raise FileExistsError("A fit seal or execution receipt already exists")
    jobs = _jobs()
    receipts = []
    SERIAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    SERIAL_LOG.touch(exist_ok=True)
    for index, (stage, fold, seed) in enumerate(jobs, 1):
        existing = _verified_existing((stage, fold, seed))
        if existing is not None:
            receipts.append(existing)
            print(f"SERIAL FIT {index}/{len(jobs)} already-complete; verified stage={stage} fold={fold} seed={seed}", flush=True)
            continue
        _check_child_sources()
        command = [sys.executable, "-m", "experiments.frontier_recovery_serial_v1",
                   "fit-one", str(stage), fold, str(seed)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        with SERIAL_LOG.open("a") as log:
            log.write(f"JOB {index}/{len(jobs)} stage={stage} fold={fold} seed={seed}\n")
            log.write(result.stdout)
            log.write(result.stderr)
            log.flush()
            os.fsync(log.fileno())
        if result.returncode:
            raise RuntimeError(f"Fresh-process fit failed for stage{stage}/{fold}/{seed}; "
                               f"inspect {SERIAL_LOG}")
        receipt_path = Path(str(training._stage_path(stage, fold, seed)) + ".json")
        row = json.loads(receipt_path.read_text())
        if row.get("stage") != stage or row.get("fold") != fold or int(row.get("seed", -1)) != seed:
            raise ValueError(f"Fresh-process fit returned wrong receipt identity: {receipt_path}")
        receipts.append(row)
        print(f"SERIAL FIT {index}/{len(jobs)} stage={stage} fold={fold} seed={seed}", flush=True)

    if len(receipts) != 12:
        raise RuntimeError("Expected exactly twelve fresh-process fit receipts")
    seal = {
        "study_id": training.STUDY,
        "declaration_sha256": sha(training.TRAIN_DECL),
        "fits": receipts,
        "all_provider_folds_complete": True,
        "training_stages": [512, 1024],
        "new_fit_count": 12,
        "source_status": "reconstructed",
        "organizer_reference_access": False,
        "scoring_performed": False,
        "candidate_scoring": False,
    }
    if {(r["stage"], r["fold"], int(r["seed"])) for r in receipts} != set(jobs):
        raise ValueError("Fresh-process receipt inventory is incomplete or duplicated")
    training._write_once(training.FIT_SEAL, seal)
    receipt = {
        "schema": "frontier_recovery_execution_v1_serial_amendment",
        "study_id": training.STUDY,
        "execution_mode": "sequential_fresh_process_per_fit",
        "serial_amendment_sha256": sha(AMENDMENT),
        "source_supplement_sha256": sha(execution.SUPPLEMENT),
        "training_declaration_sha256": sha(training.TRAIN_DECL),
        "training_fit_seal_sha256": sha(training.FIT_SEAL),
        "serial_log_path": str(SERIAL_LOG), "serial_log_sha256": sha(SERIAL_LOG),
        "fit_count": len(receipts), "all_provider_folds_complete": True,
        "source_status": "reconstructed", "organizer_reference_access": False,
        "scoring_performed": False, "candidate_scoring": False, "promotion": False,
    }
    execution.EXECUTION.parent.mkdir(parents=True, exist_ok=True)
    with execution.EXECUTION.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    sub.add_parser("run")
    fit_one = sub.add_parser("fit-one")
    fit_one.add_argument("stage", type=int, choices=(512, 1024))
    fit_one.add_argument("fold", choices=training.FOLDS)
    fit_one.add_argument("seed", type=int, choices=training.SEEDS)
    args = parser.parse_args()
    if args.action == "freeze":
        result = _freeze()
    elif args.action == "run":
        result = _run()
    else:
        result = _fit_one(args.stage, args.fold, args.seed)
    print(json.dumps(result, indent=2, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
