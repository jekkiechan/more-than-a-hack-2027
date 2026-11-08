"""Bounded process scheduling for the unchanged frozen GAR comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path

from experiments import spell_frontier_gar as study

AMENDMENT = study.ROOT / "experiments/results/spell_frontier_gar_v1_runtime.json"
BINDING = study.ROOT / "experiments/results/spell_frontier_gar_v1_execution_binding.json"
SOURCE_GUARD = study.ROOT / "experiments/results/spell_frontier_gar_v1_sourceguard_amendment.json"


def freeze():
    if AMENDMENT.exists() or study.DECLARATION.exists():
        raise FileExistsError("Preserve frozen runtime/declaration")
    source_guard = study.read(SOURCE_GUARD)
    if study.digest(study.PLAN) != source_guard["old_plan_sha256"]:
        raise ValueError("Original GAR plan must remain intact")
    if study.digest(study.ABORTED_PREFLIGHT) != source_guard["old_aborted_predeclaration_receipt_sha256"]:
        raise ValueError("Original aborted-attempt record must remain intact")
    if study.digest(Path(study.__file__)) != source_guard["new_runner_sha256"]:
        raise ValueError("Reviewed graded-frame guard source changed")
    value = dict(study=study.STUDY, created_utc=datetime.now(timezone.utc).isoformat(),
        reason="Use at most four independent single-thread inference processes after training finishes; retain the exact frozen 95-cell comparison and output paths.",
        max_workers=4, threads_per_worker=1, allowed_workers=[1, 2, 4],
        first_complete_game="test/3850; require passing paired-control checks before scheduling the rest",
        source_sha256={str(p.relative_to(study.ROOT)):study.digest(p) for p in
            (Path(__file__), Path(study.__file__), study.PLAN, study.ABORTED_PREFLIGHT, SOURCE_GUARD)},
        recipe_change=False, weights_change=False, data_change=False,
        view_or_seed_change=False, metric_or_tolerance_change=False,
        fitting=False, organizer_reference_access=False, public_scoring=False)
    study.write_once(AMENDMENT, value)
    return value


def check():
    amendment = study.read(AMENDMENT)
    for path, digest in amendment["source_sha256"].items():
        if study.digest(study.ROOT / path) != digest:
            raise ValueError("Frozen execution input changed: " + path)
    return amendment


def declare():
    check()
    declaration = study.declare()
    study.write_once(BINDING, dict(study=study.STUDY,
        declaration_sha256=study.digest(study.DECLARATION),
        runtime_sha256=study.digest(AMENDMENT),
        source_guard_amendment_sha256=study.digest(SOURCE_GUARD),
        organizer_reference_access=False, candidate_scoring_started=False))
    return declaration


def worker_init():
    import torch
    torch.set_num_threads(1)


def run(workers, first_only=False):
    check()
    binding = study.read(BINDING)
    if (binding["declaration_sha256"] != study.digest(study.DECLARATION) or
            binding["runtime_sha256"] != study.digest(AMENDMENT)):
        raise ValueError("Execution binding changed")
    if workers not in (1, 2, 4) or (first_only and workers != 1):
        raise ValueError("Use the declared process counts; first game uses one")
    declaration = study.check_declaration()
    jobs = [(split, game) for split in study.SPLITS for game in declaration["game_ids"][split]]
    if first_only:
        jobs = [("test", "3850")]
    elif not (study.GAMES / "test__3850.json").is_file():
        raise ValueError("Complete the declared first-game replay before the full batch")
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init) as pool:
        for future in as_completed([pool.submit(study._run_game, job) for job in jobs]):
            print("GAR COMPLETE", future.result(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "declare", "first", "games", "report"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "declare":
        declare()
    elif args.action == "first":
        run(1, first_only=True)
    elif args.action == "games":
        run(args.workers)
    else:
        check()
        study.report()
