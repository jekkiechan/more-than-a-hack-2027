"""One fixed comparison of pretrained PathCRF and our frozen model on J03WR9.

Prepare predictions before reading targets. Reference-live masks are used only
when scoring the separately named H/A diagnostic, never to define model windows.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import state_machine, supervised
from experiments.benchmark import verified_manifest
from experiments.export_reports import aggregates
from experiments.metrics import evaluate_tracking
from experiments.robust import INPUT_KEYS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/pathcrf_comparison"
UPSTREAM = ROOT / "data/experiments/pathcrf"
SOURCE = ROOT / "data/external_audit/sportec/processed/J03WR9.npz"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "frozen_config.json").exists():
        raise RuntimeError("Comparison already frozen")
    primary = verified_manifest()
    pathcrf = read(UPSTREAM / "frozen_config.json")
    if sha(ROOT / "experiments/pathcrf_eval.py") != pathcrf["adapter_sha256"]:
        raise RuntimeError("PathCRF adapter changed after freeze")
    selected = primary["selected_model"]
    model_path = ROOT / primary["models"][selected]["model_path"]
    model = joblib.load(model_path)
    with np.load(SOURCE, allow_pickle=False) as source:
        data = {key: source[key] for key in INPUT_KEYS if key in source.files}
    custom = supervised.predict(data, model)
    rule = primary["models"]["state_machine"]["config"]
    dead = state_machine.dead_evidence(state_machine.prepare(data, rule["interpolate_s"]), rule)
    output = OUT / "custom_predictions_and_dead_cue.npz"
    np.savez_compressed(output, custom=custom, dead=dead)
    manifest = {
        "match": "J03WR9", "selected_custom_model": selected,
        "custom_training_matches": primary["split"]["train"],
        "pathcrf_training_matches": ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY"],
        "pathcrf": pathcrf, "dead_cue_config": rule,
        "pathcrf_output": "prediction (original CRF); macro_diagnostic is not evaluated or selected",
        "hybrid": "PathCRF H/A overridden by the frozen geometric D cue",
        "conditional_diagnostic": "H/A F1 and accuracy on true-live rows; reference used only for scoring",
        "labels_deserialized": False,
        "sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
            Path(__file__), ROOT / "experiments/pathcrf_eval.py", model_path,
            UPSTREAM / "frozen_config.json", output, SOURCE)},
    }
    (OUT / "frozen_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Comparison frozen; custom inference complete without targets")


def score():
    manifest = read(OUT / "frozen_config.json")
    report_path = OUT / "results.json"
    if report_path.exists():
        raise RuntimeError("Comparison already scored; no test-driven adaptation")
    for name, expected in manifest["sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"Changed after comparison freeze: {name}")
    predicted_path = UPSTREAM / "J03WR9_predictions.npz"
    with np.load(predicted_path, allow_pickle=False) as source:
        pathcrf = source["prediction"].copy()
    with np.load(OUT / "custom_predictions_and_dead_cue.npz", allow_pickle=False) as source:
        custom, dead = source["custom"].copy(), source["dead"].copy()
    data = supervised.load(SOURCE)
    truth = data["labels"]
    if pathcrf.shape != truth.shape or not np.isin(pathcrf, ["H", "A"]).all():
        raise ValueError("Expected complete, full-match H/A PathCRF predictions")
    hybrid = pathcrf.copy()
    hybrid[dead] = "D"
    candidates = {"pathcrf_team_only": pathcrf,
                  "pathcrf_plus_geometric_dead": hybrid,
                  "frozen_custom_robust": custom}
    scores = {}
    live_truth = np.where(np.isin(truth, ["H", "A"]), truth, "X")
    for name, prediction in candidates.items():
        metric = evaluate_tracking(truth, prediction, data["period"], float(data["fps"]), data["timestamps"])
        diagnostic = evaluate_tracking(live_truth, prediction, data["period"], float(data["fps"]), data["timestamps"])
        scores[name] = {
            "whole_match": aggregates(metric),
            "reference_live_diagnostic": {
                "ha_macro_f1": diagnostic["macro_f1_present_classes"],
                "accuracy": diagnostic["accuracy"], "n_frames": diagnostic["n_graded"],
                "per_class": diagnostic["per_class"],
            },
            "prediction_coverage": float(np.mean(prediction != "X")),
        }
        print(name, "whole macro", metric["macro_f1"], "spell", metric["spell_f1_mean"],
              "live H/A macro", diagnostic["macro_f1_present_classes"], flush=True)
    results = {
        "match": "J03WR9", "n_frames": len(truth), "scores": scores,
        "selection_changed": False, "inference": read(UPSTREAM / "J03WR9_inference.json"),
        "comparison_manifest_sha256": sha(OUT / "frozen_config.json"),
        "pathcrf_prediction_sha256": sha(predicted_path),
        "limitation": "Adapted whole-match team attribution; not the paper's true-live event benchmark. D cue uses frozen-ball source artifact.",
    }
    report_path.write_text(json.dumps(results, indent=2) + "\n")
    public = ROOT / "experiments/results"
    (public / "pathcrf_comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    (public / "pathcrf_comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "score"))
    args = parser.parse_args()
    {"prepare": prepare, "score": score}[args.action]()
