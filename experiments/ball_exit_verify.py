"""Independently replay saved ball-exit study outputs and aggregates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import ball_exit_followup as shared
from experiments import ball_exit_dmask_v1 as fable
from experiments import frontier_postprocess_readout as official
from experiments import sc2020_realfeed_common as common, sc2020_realfeed_score as scoring


def verify(result_path, output_path):
    result = json.loads(result_path.read_text())
    declaration_path = result_path.with_name(result_path.stem + "_declaration.json")
    if shared.sha(declaration_path) != result["declaration_sha256"]:
        raise ValueError("Declaration binding changed")
    declaration = json.loads(declaration_path.read_text())
    facts = declaration.get("cells", declaration.get("cohort"))
    if set(facts) != set(result["prediction_artifacts"]) or len(facts) != 35:
        raise ValueError("Incomplete saved cohort")
    rows, count, maximum = {}, 0, 0.
    complement = "complement" in result["study"]
    control = "snap__frontier" if complement else "snap"
    for key, artifact in result["prediction_artifacts"].items():
        native = facts[key].get("frontier") or {"path": facts[key]["native"], "sha256": facts[key]["native_sha256"]}
        if shared.sha(native["path"]) != native["sha256"]:
            raise ValueError("Native input changed: " + key)
        path = shared.ROOT / artifact["path"]
        if shared.sha(path) != artifact["sha256"]:
            raise ValueError("Prediction changed: " + key)
        cell = common.frontier_cell(key)
        rows[key] = {}
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != set(result["aggregate"]):
                raise ValueError("Incomplete saved arm inventory")
            for arm in saved.files:
                metrics = official._official_score(cell, saved[arm])
                rows[key][arm] = {m: float(metrics[m]) for m in scoring.METRICS}
                for metric, measured in rows[key][arm].items():
                    error = abs(measured - result["cells"][key][arm][metric])
                    maximum = max(maximum, error)
                    if error > 1e-10:
                        raise ValueError(f"Scalar replay mismatch: {key}/{arm}/{metric}")
                if complement:
                    reference = saved["snap__frontier" if arm.startswith("snap__") else "original__frontier"]
                    np.testing.assert_array_equal(saved[arm] == "D", reference == "D")
                count += 1
    aggregates = {arm: scoring.aggregate(rows, arm) for arm in result["aggregate"]}
    for arm, by_view in aggregates.items():
        for view, metrics in by_view.items():
            for metric, value in metrics.items():
                expected = result["aggregate"][arm][view][metric]
                if isinstance(value, dict):
                    for provider, number in value.items():
                        if abs(number - expected[provider]) > 1e-10:
                            raise ValueError("Provider aggregation mismatch")
                elif abs(value - expected) > 1e-10:
                    raise ValueError("Aggregation mismatch")
    failed = {arm: fable.guards(aggregates[arm], aggregates[control]) for arm in result["failed_guards"]}
    if failed != result["failed_guards"]:
        raise ValueError("Guard replay mismatch")
    eligible = [arm for arm, checks in failed.items() if not checks]
    selected = max(eligible, key=lambda a: (aggregates[a]["feed_like_v2"]["spell_f1"], aggregates[a]["feed_like_v2"]["macro_f1"])) if eligible else control
    if selected != result["selected"]:
        raise ValueError("Selection mismatch")
    receipt = {"result_sha256": shared.sha(result_path), "verifier_sha256": shared.sha(__file__),
        "prediction_sets": count, "scalar_comparisons": count * len(scoring.METRICS),
        "maximum_scalar_error": maximum, "all_passed": True, "aggregates_guards_selection_verified": True,
        "D_masks_verified": complement, "selected": selected, "organizer_reference_access": False}
    shared.write_once(output_path, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    verify(args.result, args.output)
