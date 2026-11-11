"""Score the twelve declared `sc2020_realfeed_v1` candidates against the approved frontier."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments import sc2020_realfeed_common as C
from experiments import sc2020_realfeed_score as S

DECLARATION = C.ROOT / "experiments/results/sc2020_realfeed_v1_declaration.json"
RESULT = C.ROOT / "experiments/results/sc2020_realfeed_v1.json"
CONTROL = {"clean": 0.6701365499557103, "feed_like_v1": 0.532128306931943, "feed_like_v2": 0.5936187198896373}


def candidates():
    declaration = json.loads(DECLARATION.read_text())
    arms = {"control": {}}
    for name in declaration["candidates"]["arms_x_snapshots"]:
        arms[f"{name}/replace"] = {"ce": {name: 1.0}}
        arms[f"{name}/half"] = {"ce": {"control": 0.5, name: 0.5}}
    return arms


def main():
    C.configure()
    arms = candidates()
    rows = S.run(arms)
    control = S.aggregate(rows, "control")
    for view, expected in CONTROL.items():
        if abs(control[view]["spell_f1"] - expected) > 1e-12:
            raise RuntimeError(f"Approved frontier control did not replay: {view}")
    out = {"schema": "sc2020_realfeed_v1_result", "study_id": C.STUDY,
           "declaration_sha256": hashlib.sha256(DECLARATION.read_bytes()).hexdigest(),
           "control_replay_exact": True, "organizer_reference_access": False, "candidates": {}, "cells": rows}
    for name in arms:
        agg = S.aggregate(rows, name)
        failed = [] if name == "control" else S.guards(agg, control)
        out["candidates"][name] = {"aggregate": agg, "failed_retention_checks": failed,
                                   "retained": name != "control" and not failed,
                                   "meets_0_60": agg["feed_like_v2"]["spell_f1"] > 0.60}
    effects = {}
    for name in arms:
        if name.startswith("sc"):
            arm, pipeline = name.split("/")
            matched = f"ctrl_{arm.split('_')[1]}/{pipeline}"
            effects[name] = {view: out["candidates"][name]["aggregate"][view]["spell_f1"] -
                             out["candidates"][matched]["aggregate"][view]["spell_f1"] for view in S.VIEW_NAMES}
            effects[name]["feed_like_v2/providers"] = {
                p: v - out["candidates"][matched]["aggregate"]["feed_like_v2"]["spell_f1/providers"][p]
                for p, v in out["candidates"][name]["aggregate"]["feed_like_v2"]["spell_f1/providers"].items()}
    out["data_effect_vs_matched_ctrl"] = effects
    retained = sorted((n for n, c in out["candidates"].items() if c["retained"]),
                      key=lambda n: -out["candidates"][n]["aggregate"]["feed_like_v2"]["spell_f1"])
    out["retained"] = retained
    out["success"] = any(out["candidates"][n]["meets_0_60"] for n in retained)
    RESULT.write_text(json.dumps(out, indent=1) + "\n")
    print(S.table(rows, list(arms)))
    print("\ndata effect (arm - matched ctrl), spell clean/v1/v2 and v2 by provider")
    for name, e in effects.items():
        pv = e["feed_like_v2/providers"]
        print(f"{name:20s} {e['clean']:+.6f} {e['feed_like_v1']:+.6f} {e['feed_like_v2']:+.6f} | "
              f"Met {pv['Metrica']:+.4f} Spo {pv['Sportec']:+.4f} SC {pv['SkillCorner']:+.4f}")
    print("retained:", retained, "success:", out["success"])


if __name__ == "__main__":
    main()
