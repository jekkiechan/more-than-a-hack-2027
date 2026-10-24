"""Freeze validation-selected models, then run one held-out benchmark.

All predictors receive dictionaries with the target labels removed. The manifest
records code and model hashes before test access. Predictions stay under data/;
only aggregate reports are suitable for version control.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import state_machine, supervised, temporal
from experiments.metrics import evaluate_tracking
from experiments.stress import mask_ball

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external_audit/sportec"
OUT = ROOT / "data/experiments/benchmark"
MANIFEST = OUT / "frozen_manifest.json"
CONFIGS = {
    "state_machine": "data/experiments/state_machine/sportec/frozen_config.json",
    "temporal": "data/experiments/temporal_sportec/frozen_config.json",
    "supervised": "data/experiments/supervised/frozen_config.json",
    "robust": "data/experiments/robust/frozen_config.json",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def predictor(name, record):
    config = record["config"]
    if name == "state_machine":
        return lambda data: state_machine.predict(data, config)
    if name == "temporal":
        config = dict(config, model_path=str(ROOT / record["model_path"]))
        return lambda data: temporal.predict(data, config)
    model = joblib.load(ROOT / record["model_path"])
    return lambda data: supervised.predict(data, model)


def score(data, predictions, bootstrap_n=0):
    result = evaluate_tracking(data["labels"], predictions,
                               period=data["period"], fps=float(data["fps"]),
                               timestamps=data["timestamps"], bootstrap_n=bootstrap_n)
    result["prediction_coverage"] = float(np.mean(predictions != "X"))
    result["ball_missing_fraction"] = float(np.mean(~np.isfinite(data["ball"]).all(axis=1)))
    return result


def inference_inputs(data):
    return {key: value for key, value in data.items() if key != "labels"}


def freeze():
    if MANIFEST.exists():
        raise RuntimeError("Manifest already frozen; do not replace it after test access")
    if (OUT / "test.json").exists():
        raise RuntimeError("Test results already exist")
    split = read(BASE / "split.json")
    validation = {name: supervised.load(BASE / "processed" / f"{name}.npz")
                  for name in split["validation"]}
    records = {}
    for name, config_path in CONFIGS.items():
        raw = read(ROOT / config_path)
        config = raw.get("config", raw)
        record = {"config_path": config_path, "config": config}
        if name != "state_machine":
            path = config.get("model_path") if name == "temporal" else str(Path(config_path).parent / "model.joblib")
            record["model_path"] = relative(ROOT / path)
        infer = predictor(name, record)
        record["validation"] = {match: score(data, infer(inference_inputs(data)))
                                for match, data in validation.items()}
        record["mean_validation_macro_f1"] = float(np.mean([
            value["macro_f1"] for value in record["validation"].values()]))
        record["mean_validation_spell_f1"] = float(np.mean([
            value["spell_f1_mean"] for value in record["validation"].values()]))
        records[name] = record
        print(name, record["mean_validation_macro_f1"], record["mean_validation_spell_f1"], flush=True)
    def selection(name):
        macro = records[name]["mean_validation_macro_f1"]
        spell = records[name]["mean_validation_spell_f1"]
        return (1, spell, macro) if macro >= .9 else (0, macro, spell)
    selected = max(records, key=selection)
    files = [ROOT / "experiments" / name for name in (
        "benchmark.py", "state_machine.py", "supervised.py", "temporal.py",
        "robust.py", "metrics.py", "stress.py", "challenge.py", "audit_dataset.py")]
    files += [ROOT / "evaluation/possession_metrics.py", BASE / "split.json"]
    files += [ROOT / record["config_path"] for record in records.values()]
    files += [ROOT / record["model_path"] for record in records.values() if "model_path" in record]
    manifest = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split, "selected_model": selected, "models": records,
        "selection": "Highest validation spell F1 among models with frame macro F1 >= 0.90; otherwise highest macro F1",
        "stress": {"missing_rate": .41, "mean_gap_s": 1.0, "seed": 2026},
        "bootstrap": "1000 official segment/spell draws for selected clean test; conditional on one match",
        "sha256": {relative(path): digest(path) for path in files},
        "test_accessed": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("FROZEN", selected, MANIFEST, flush=True)


def verified_manifest():
    manifest = read(MANIFEST)
    for name, expected in manifest["sha256"].items():
        if digest(ROOT / name) != expected:
            raise RuntimeError(f"Changed since freeze: {name}")
    return manifest


def test():
    manifest = verified_manifest()
    report_path = OUT / "test.json"
    started = OUT / "test_started.json"
    if report_path.exists() or started.exists():
        raise RuntimeError("Held-out evaluation already started; inspect saved artifacts")
    started.write_text(json.dumps({"started_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "manifest_sha256": digest(MANIFEST)}, indent=2) + "\n")
    predictors = {name: predictor(name, record) for name, record in manifest["models"].items()}
    rule_config = manifest["models"]["state_machine"]["config"]
    predictors["nearest_team"] = state_machine.nearest_baseline
    predictors["nearest_team_dead_cue"] = lambda data: state_machine.nearest_with_dead_baseline(data, rule_config)
    results = {"selected_before_test": manifest["selected_model"],
               "manifest_sha256": digest(MANIFEST), "matches": {}}
    for match in manifest["split"]["test"]:
        data = supervised.load(BASE / "processed" / f"{match}.npz")
        variants = {"clean": data, "masked": mask_ball(data, **manifest["stress"])}
        results["matches"][match] = {}
        for variant, dataset in variants.items():
            rows = {}
            for name, infer in predictors.items():
                prediction = infer(inference_inputs(dataset))
                draws = 1000 if variant == "clean" and name == manifest["selected_model"] else 0
                rows[name] = score(dataset, prediction, bootstrap_n=draws)
                np.save(OUT / f"{match}_{variant}_{name}.npy", prediction)
                print(match, variant, name, rows[name]["macro_f1"], rows[name]["spell_f1_mean"], flush=True)
                results["matches"][match][variant] = rows
                # Write each completed result so a failure cannot erase test access.
                report_path.write_text(json.dumps(results, indent=2) + "\n")
    return results


def challenge():
    from experiments.challenge import load_challenge, write_submission
    manifest = verified_manifest()
    name = manifest["selected_model"]
    data = load_challenge()
    predictions = predictor(name, manifest["models"][name])(inference_inputs(data))
    predictions[data["period"] == 0] = "X"
    output = write_submission(predictions, OUT / f"challenge_{name}.csv")
    print("Validated challenge submission (no ground truth read):", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "test", "challenge"))
    args = parser.parse_args()
    {"freeze": freeze, "test": test, "challenge": challenge}[args.action]()
