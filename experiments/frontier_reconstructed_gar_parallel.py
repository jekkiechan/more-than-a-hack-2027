"""Execute the fixed reconstructed GAR report in four frozen game partitions.

Call ``freeze`` only after the delivery adapter has prepared the complete GAR
declaration. Workers run the existing per-game verifier in fresh processes;
the original full report runs only after all disjoint partitions are sealed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments import ball_exit_gar as legacy_gar
from experiments import current_baseline
from experiments import frontier_reconstructed_delivery as delivery

ROOT = delivery.ROOT
DECL = ROOT / f"experiments/results/{delivery.STUDY}_gar_execution.json"
FULL_DECL = delivery.GAR_DECL
OUT = delivery.OUT / "gar"
PARTITIONS = 4
TEST_FILE = ROOT / "experiments/tests/test_frontier_reconstructed_gar_parallel.py"
VIEWS = ("clean", "v2_1", "v2_2", "v2_3", "v2_4")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def partition_games(games, count=PARTITIONS):
    games = list(games)
    if len(games) != 10 or len(set(games)) != 10 or count != 4:
        raise ValueError("Execution requires the exact ten unique TEST games and four partitions")
    parts = [games[i::count] for i in range(count)]
    if any(not part for part in parts) or sorted(g for part in parts for g in part) != sorted(games):
        raise ValueError("TEST partition inventory is incomplete")
    if sum(map(len, parts)) != len(set(g for part in parts for g in part)):
        raise ValueError("TEST partitions overlap")
    return parts


def audit_model_exposure(models, test_games):
    test_games = set(test_games)
    exposed = {}
    for role, model in models.items():
        trained = {str(game).removeprefix("soccernet_")
                   for game in model.get("training", {}).get("soccernet_games", [])}
        overlap = sorted(test_games & trained)
        if overlap:
            exposed[role] = overlap
    if exposed:
        raise ValueError(f"TEST exposure in checkpoint(s): {exposed}")
    return sorted(models)


def assert_no_test_exposure(games):
    models = current_baseline.load_models(manifest_path=delivery.STAGED)
    return audit_model_exposure(models, games)


def _source_inventory():
    return {"experiments/frontier_reconstructed_gar_parallel.py": sha(__file__),
            "experiments/tests/test_frontier_reconstructed_gar_parallel.py": sha(TEST_FILE)}


def freeze():
    if DECL.exists():
        raise FileExistsError("GAR execution declaration is immutable")
    if not FULL_DECL.is_file():
        raise FileNotFoundError("Prepare the full GAR declaration before freezing execution partitions")
    full = read(FULL_DECL)
    games = full.get("games", [])
    parts = partition_games(games)
    if tuple(full.get("views", ())) != VIEWS:
        raise ValueError("The frozen GAR view inventory differs from clean plus four fixed v2 views")
    if full.get("manifest_sha256") != sha(delivery.STAGED):
        raise ValueError("Full GAR declaration does not bind the staged model")
    if any((OUT / f"{game}.json").exists() or (OUT / f"{game}.npz").exists() for game in games):
        raise FileExistsError("Freeze partitions before any game score or prediction artifact exists")
    delivery.check_seal()
    exposure_checked_roles = assert_no_test_exposure(games)
    write_once(DECL, {"study": delivery.STUDY, "full_gar_declaration": str(FULL_DECL.relative_to(ROOT)),
        "full_gar_declaration_sha256": sha(FULL_DECL), "staged_manifest_sha256": sha(delivery.STAGED),
        "input_sha256": delivery.read(delivery.SEAL)["input_sha256"],
        "models_sha256": delivery.read(delivery.STAGED)["models_sha256"],
        "full_gar_sources": full["sources"], "full_gar_record_files": full["record_files"],
        "full_gar_raw_files": full["raw_files"], "games": games, "views": list(VIEWS),
        "test_exposure_audit": {"all_test_games_checked_before_partitioning": True,
                                "model_roles": exposure_checked_roles, "overlap": False},
        "partition_count": PARTITIONS,
        "partitions": {str(i): group for i, group in enumerate(parts)},
        "execution_sources": _source_inventory(), "scoring_started_before_freeze": False,
        "worker_processes": "fresh Python process per partition; Torch/BLAS constrained to one thread",
        "aggregate": "Only the original full report aggregates all ten games and five views"})


def check_execution():
    declaration = read(DECL)
    full = read(FULL_DECL)
    if declaration.get("full_gar_declaration_sha256") != sha(FULL_DECL):
        raise ValueError("Full GAR declaration changed after execution freeze")
    if (declaration.get("staged_manifest_sha256") != sha(delivery.STAGED) or
            declaration.get("input_sha256") != delivery.read(delivery.SEAL)["input_sha256"] or
            declaration.get("models_sha256") != delivery.read(delivery.STAGED)["models_sha256"]):
        raise ValueError("Execution package, input or model inventory changed")
    if declaration.get("execution_sources") != _source_inventory():
        raise ValueError("Parallel execution adapter or contract test changed after freeze")
    for rel, expected in {**full["sources"], **full["record_files"], **declaration["execution_sources"]}.items():
        if sha(ROOT / rel) != expected:
            raise ValueError("GAR source/record changed: " + rel)
    tracking = legacy_gar.historical.dense.tracking_root(legacy_gar.historical.dense.DEFAULT_RAW)
    for row in declaration["full_gar_raw_files"]:
        if sha(tracking / row["path"]) != row["sha256"]:
            raise ValueError("GAR raw input changed: " + row["path"])
    games = declaration["games"]
    if len(games) != 10 or len(set(games)) != 10 or declaration["views"] != list(VIEWS):
        raise ValueError("Frozen execution inventory is invalid")
    groups = [declaration["partitions"].get(str(i)) for i in range(PARTITIONS)]
    if any(group is None for group in groups):
        raise ValueError("Frozen partition inventory is incomplete")
    flattened = [game for group in groups for game in group]
    if len(flattened) != len(set(flattened)) or sorted(flattened) != sorted(games):
        raise ValueError("Frozen partitions overlap or omit TEST games")
    return declaration


@contextmanager
def _game_locks(games):
    lock_dir = OUT / "execution_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    held = []
    try:
        for game in games:
            lock = lock_dir / f"{game}.lock"
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(f"pid={os.getpid()}\n")
            held.append(lock)
        yield
    finally:
        for lock in held:
            lock.unlink(missing_ok=True)


def worker(index):
    declaration = check_execution()
    key = str(index)
    if key not in declaration["partitions"]:
        raise ValueError("Unknown frozen partition")
    games = declaration["partitions"][key]
    if not games:
        raise ValueError("Empty worker partition")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    if audit_model_exposure(current_baseline.load_models(manifest_path=delivery.STAGED), declaration["games"]) != declaration["test_exposure_audit"]["model_roles"]:
        raise ValueError("Full TEST exposure audit model inventory changed")
    with delivery.legacy_gar_context():
        original_check, original_report = legacy_gar.check, legacy_gar.report

        def checked_partition():
            manifest, full = original_check()
            if full["games"] != declaration["games"]:
                raise ValueError("Worker sees a different full TEST game order")
            subset = dict(full, games=list(games))
            return manifest, subset

        legacy_gar.check = checked_partition
        legacy_gar.report = lambda: None
        try:
            with _game_locks(games):
                legacy_gar.run()
        finally:
            legacy_gar.check, legacy_gar.report = original_check, original_report
    artifacts = {}
    for game in games:
        game_json, game_npz = OUT / f"{game}.json", OUT / f"{game}.npz"
        row = read(game_json)
        if row.get("declaration_sha256") != declaration["full_gar_declaration_sha256"]:
            raise ValueError("Worker artifact does not bind the full GAR declaration")
        artifacts[game] = {"json_sha256": sha(game_json), "prediction_sha256": sha(game_npz)}
    receipt = {"execution_declaration_sha256": sha(DECL), "full_gar_declaration_sha256": declaration["full_gar_declaration_sha256"],
        "partition": index, "games": list(games), "artifacts": artifacts,
        "scoring_worker": True, "aggregation_performed": False}
    write_once(OUT / "execution_partitions" / f"partition{index}.json", receipt)


def _worker_env():
    env = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    return env


def run_workers():
    declaration = check_execution()
    if assert_no_test_exposure(declaration["games"]) != declaration["test_exposure_audit"]["model_roles"]:
        raise ValueError("Full TEST exposure audit model inventory changed before worker launch")
    receipts = OUT / "execution_partitions"
    if receipts.exists() and any(receipts.iterdir()):
        raise FileExistsError("Preserve any prior or partial partition run")
    command = [sys.executable, "-m", "experiments.frontier_reconstructed_gar_parallel", "worker"]
    with ThreadPoolExecutor(max_workers=PARTITIONS) as pool:
        futures = [pool.submit(subprocess.run, command + [str(i)], cwd=ROOT, env=_worker_env(), check=True)
                   for i in range(PARTITIONS)]
        for future in futures:
            future.result()
    return declaration


def report():
    declaration = check_execution()
    for index in range(PARTITIONS):
        receipt_path = OUT / "execution_partitions" / f"partition{index}.json"
        receipt = read(receipt_path)
        expected = declaration["partitions"][str(index)]
        if (receipt.get("execution_declaration_sha256") != sha(DECL) or
                receipt.get("full_gar_declaration_sha256") != sha(FULL_DECL) or
                receipt.get("games") != expected or set(receipt.get("artifacts", {})) != set(expected)):
            raise ValueError("Partition receipt is missing or bound to a different execution")
        for game, hashes in receipt["artifacts"].items():
            if (hashes["json_sha256"] != sha(OUT / f"{game}.json") or
                    hashes["prediction_sha256"] != sha(OUT / f"{game}.npz")):
                raise ValueError("Partition artifact changed: " + game)
    # The unmodified original report performs independent replay of all 50 cells.
    delivery.report_gar()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run", "report", "worker"))
    parser.add_argument("partition", nargs="?", type=int)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "run":
        run_workers()
    elif args.action == "report":
        report()
    else:
        if args.partition is None:
            parser.error("worker requires its frozen partition index")
        worker(args.partition)


if __name__ == "__main__":
    main()
