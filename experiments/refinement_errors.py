"""Diagnose saved final external-test predictions without changing any model."""

import json
from pathlib import Path

import numpy as np

from experiments.error_analysis import analyze, digest, write_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/external_audit/sportec/processed/J03WQQ.npz"
OUT = ROOT / "data/experiments/refinement"


if __name__ == "__main__":
    test = json.loads((OUT / "test.json").read_text())
    with np.load(DATA, allow_pickle=False) as source:
        data = {key: source[key] for key in ("labels", "period", "timestamps", "fps")}
    comparisons = {}
    for name in ("baseline", test["selected_before_test"]):
        path = OUT / f"J03WQQ_clean_{name}.npy"
        result = analyze(data, np.load(path, allow_pickle=False))
        previous = test["conditions"]["clean"][name]
        for metric in ("macro_f1", "spell_f1_mean", "n_gt_spells", "n_pred_spells"):
            assert np.isclose(result["metrics"][metric], previous[metric], rtol=0, atol=1e-14)
        result["provenance"] = {
            "match": "J03WQQ", "model": name, "condition": "clean",
            "data_sha256": digest(DATA), "prediction_sha256": digest(path),
            "manifest_sha256": digest(OUT / "frozen_manifest.json"),
        }
        if name == test["selected_before_test"]:
            result["conditional_bootstrap"] = {
                "macro_f1": [previous["bootstrap_low"], previous["bootstrap_high"]],
                "spell_f1_mean": [previous["spell_bootstrap_low"], previous["spell_bootstrap_high"]],
            }
            write_report(result, ROOT / "experiments/REFINE_ERRORS.md")
        comparisons[name] = result
        print(name, [(row["duration"], row["recall_at_iou_0_5"])
                     for row in result["by_gt_duration"]])
    (ROOT / "experiments/results/refinement_errors.json").write_text(
        json.dumps(comparisons, indent=2) + "\n")
