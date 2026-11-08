"""Run a named `v2_postproc_v1` grid over cached development emissions.

Usage: python -m experiments.v2_postproc_v1_grid <grid-name> [--workers N]
Writes per-cell metrics to data/experiments/v2_postproc_v1/<grid>.pkl and an
aggregate JSON to experiments/results/v2_postproc_v1_<grid>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
from pathlib import Path

from experiments import v2_postproc_v1_arms as A
from experiments import v2_postproc_v1_score as S

ROOT = Path(__file__).resolve().parents[1]


def _w(ce, cal, ce_family="ce1024"):
    out = {ce_family: ce, "adop": cal, "cand": round(1 - ce - cal, 6)}
    if isinstance(ce_family, tuple):
        out = {f: ce / len(ce_family) for f in ce_family} | {"adop": cal, "cand": round(1 - ce - cal, 6)}
    return out


def grids():
    g = {}
    controls = {"cand_f062": A.P(A.family_f062, family="cand"),
                "adop_f062": A.P(A.family_f062, family="adop"),
                "frontier_replica_ha3": A.P(A.blend, weights=_w(.5, .375), ha_cost=3.0)}
    g["controls"] = controls
    weights = [(ce, cal) for ce in (.375, .5, .625, .75) for cal in (.125, .25, .375, .5)
               if ce + cal <= 1 + 1e-9]
    g["ce_cost"] = dict(controls)
    for (ce, cal), cost in itertools.product(weights, (1.5, 2.0, 2.5, 3.0)):
        g["ce_cost"][f"ce{ce}_cal{cal}_ha{cost}"] = A.P(A.blend, weights=_w(ce, cal), ha_cost=cost)
    g["snapshot"] = dict(controls)
    for (ce, cal), cost in itertools.product(((.5, .375), (.625, .25), (.5, .25), (.75, .25)), (2.0, 2.5, 3.0)):
        g["snapshot"][f"snap_ce{ce}_cal{cal}_ha{cost}"] = A.P(
            A.blend, weights=_w(ce, cal, ("ce512", "ce1024")), ha_cost=cost)
        g["snapshot"][f"ce512_ce{ce}_cal{cal}_ha{cost}"] = A.P(
            A.blend, weights=_w(ce, cal, "ce512"), ha_cost=cost)
    g["seeds4"] = dict(controls)
    g["seeds4"]["extra_seeds_only_ha3"] = A.P(A.blend, weights=_w(.5, .375, "ce1024@x"), ha_cost=3.0)
    for (ce, cal), cost in itertools.product(((.5, .375), (.5, .5), (.625, .375), (.5, .25), (.625, .25), (.75, .25)),
                                             (2.0, 2.5, 3.0)):
        g["seeds4"][f"x4_ce{ce}_cal{cal}_ha{cost}"] = A.P(A.blend, weights=_w(ce, cal, "ce1024+x"), ha_cost=cost)
    g["high_cost"] = dict(controls)
    for (ce, cal), cost in itertools.product(((.5, .375), (.5, .5), (.625, .375), (.5, .25)), (3.5, 4.0, 5.0)):
        g["high_cost"][f"ce{ce}_cal{cal}_ha{cost}"] = A.P(A.blend, weights=_w(ce, cal), ha_cost=cost)
        g["high_cost"][f"x4_ce{ce}_cal{cal}_ha{cost}"] = A.P(A.blend, weights=_w(ce, cal, "ce1024+x"), ha_cost=cost)
    g["adopwarm"] = dict(controls)
    for cost in (2.5, 3.0):
        for fam, label in ((("ce1024", "ceadop1024"), "both"), (("ce1024+x", "ceadop1024"), "x4both")):
            for ce, cal in ((.5, .375), (.625, .25), (.75, .25), (.5, .5), (.625, .375)):
                g["adopwarm"][f"{label}_ce{ce}_cal{cal}_ha{cost}"] = A.P(
                    A.blend, weights=_w(ce, cal, fam), ha_cost=cost)
        g["adopwarm"][f"adopwarm_only_ce.5_cal.375_ha{cost}"] = A.P(
            A.blend, weights=_w(.5, .375, "ceadop1024"), ha_cost=cost)
    return g


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    pipes = grids()[args.grid]
    rows = S.run(pipes, workers=args.workers)
    out = ROOT / "data/experiments/v2_postproc_v1" / f"{args.grid}.pkl"
    out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(rows, out.open("wb"))
    summary = {name: S.aggregate(rows, name) for name in pipes}
    res = ROOT / "experiments/results" / f"v2_postproc_v1_{args.grid}.json"
    res.write_text(json.dumps({"study_id": "v2_postproc_v1", "grid": args.grid,
                               "cells": len(rows), "aggregate": summary,
                               "organizer_reference_access": False}, indent=1, sort_keys=True) + "\n")
    print(S.table(rows, list(pipes)))


if __name__ == "__main__":
    main()
