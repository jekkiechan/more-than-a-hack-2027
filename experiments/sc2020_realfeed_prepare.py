"""Prepare development cells and the fixed four-view v2 TRAIN caches (`sc2020_realfeed_v1`)."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments import sc2020_realfeed_common as C


def _cell(job):
    C.configure()
    C.prepared(*job)
    return C.cell_key(*job)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("cells", "train"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    C.configure()
    if args.action == "train":
        C.recipe.prepare(workers=args.workers)
        return
    with ProcessPoolExecutor(args.workers) as pool:
        for future in as_completed([pool.submit(_cell, job) for job in C.cells()]):
            print("CELL", future.result(), flush=True)


if __name__ == "__main__":
    main()
