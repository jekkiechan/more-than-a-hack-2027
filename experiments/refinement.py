"""Freeze refinements, score a fresh external match, then check competition IO.

The public competition labels are a final pipeline diagnostic, never a tuning
set. The configuration and transfer candidates are fixed before that file opens.
"""

import argparse
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluation import possession_metrics as pm
from experiments import benchmark, robust, supervised
from experiments.challenge import write_submission
from experiments.export_reports import aggregates
from experiments.metrics import evaluate_tracking
from experiments.stress import mask_ball

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/experiments/refinement"
PUBLIC = ROOT / "experiments/results"
MATCHES = ROOT / "data/external_audit/sportec/processed"
MANIFEST = OUT / "frozen_manifest.json"


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def score_in(value):
    if "macro_f1" in value and "spell_f1_mean" in value:
        return value
    for key in ("selected", "conditions", "clean", "scores", "J03WOY", "selected_validation"):
        if key in value:
            return score_in(value[key])
    raise ValueError("No clean validation score in artifact")


def check_hashes(mapping):
    for path, expected in mapping.items():
        if robust.sha256(ROOT / path) != expected:
            raise RuntimeError(f"Artifact differs from its prior freeze: {path}")


def freeze():
    if MANIFEST.exists():
        raise RuntimeError("Refinement manifest already frozen")
    primary = benchmark.verified_manifest()
    protocol = read(ROOT / "data/experiments/refinement_protocol.json")
    records = {}
    validation_data = supervised.load(MATCHES / "J03WOY.npz")
    prediction_cache = {}
    specifications = [
        ("baseline", "robust", "supervised", "robust/model.joblib"),
        ("decoder", "refine_decode", "refine_decode", "robust/model.joblib"),
        ("physical", "refine_features", "refine_features", "refine_features/model.joblib"),
        ("combined", "refine_combined", "refine_combined", "refine_features/model.joblib"),
        ("domain", "refine_domain", "refine_domain", "refine_domain/model.joblib"),
        ("weighted", "refine_weighted", "refine_features", "refine_weighted/model.joblib"),
        ("scaled", "refine_scale", "refine_scale", "refine_scale/model.joblib"),
        ("ensemble", "refine_ensemble", "refine_ensemble", "refine_ensemble/model.joblib"),
    ]
    paths = {Path(__file__), ROOT / "data/experiments/refinement_protocol.json"}
    for name, folder, module, model in specifications:
        directory = ROOT / "data/experiments" / folder
        config_path, model_path = directory / "frozen_config.json", ROOT / "data/experiments" / model
        config = read(config_path)
        if "model_sha256" in config and robust.sha256(model_path) != config["model_sha256"]:
            raise RuntimeError(f"Model differs from prior freeze: {name}")
        if "robust_model_sha256" in config and robust.sha256(model_path) != config["robust_model_sha256"]:
            raise RuntimeError("Decoder's base model changed")
        sources = config.get("source_sha256", {})
        if isinstance(sources, str):
            sources = {f"experiments/{folder}.py": sources}
        check_hashes(sources)
        check_hashes(config.get("sha256", {}))
        if name == "baseline":
            validation = primary["models"]["robust"]["validation"]["J03WOY"]
        elif name == "decoder":
            validation = config["selected_validation"]
        else:
            validation = score_in(read(directory / "validation.json"))
        if (directory / "validation.json").exists():
            paths.add(directory / "validation.json")
        records[name] = {
            "module": module, "model_path": str(model_path.relative_to(ROOT)),
            "config": config.get("config", config),
            "validation": aggregates(validation),
            "config_path": str(config_path.relative_to(ROOT)),
        }
        paths.update((config_path, model_path, ROOT / "experiments" / f"{module}.py",
                      ROOT / "experiments" / f"{folder}.py"))
        reproduced = benchmark.score(validation_data, predict(validation_data, records[name], prediction_cache))
        for metric in ("macro_f1", "spell_f1_mean"):
            if not np.isclose(reproduced[metric], validation[metric], atol=1e-12, rtol=0):
                raise RuntimeError(f"Validation reproduction mismatch: {name} {metric}")
        records[name]["validation"] = aggregates(reproduced)
        print("VERIFIED", name, reproduced["macro_f1"], reproduced["spell_f1_mean"], flush=True)
    def key(name):
        score = records[name]["validation"]
        macro, spell = score["macro_f1"], score["spell_f1_mean"]
        return (1, spell, macro) if macro >= .9 else (0, macro, spell)
    selected = max(records, key=key)
    for name in ("supervised", "robust", "stress", "metrics", "challenge", "refine_features",
                 "refine_decode", "benchmark", "export_reports"):
        paths.add(ROOT / "experiments" / f"{name}.py")
    paths.add(ROOT / "evaluation/possession_metrics.py")
    for match in protocol["train"] + protocol["validation"] + protocol["test"]:
        paths.add(MATCHES / f"{match}.npz")
    paths.add(ROOT / "data/challenge/inputs.npz")
    transfer_dir = ROOT / "data/experiments/challenge_pathcrf"
    transfer = read(transfer_dir / "frozen_config.json")
    check_hashes({"experiments/challenge_pathcrf.py": transfer["adapter_sha256"],
                  "experiments/pathcrf_eval.py": transfer["original_adapter_sha256"],
                  "data/challenge/inputs.npz": transfer["input_sha256"]})
    for filename in ("frozen_config.json", "inference.json", "predictions.npz"):
        paths.add(transfer_dir / filename)
    pure_path = ROOT / "data/experiments/refine_domain/player_only_model.joblib"
    paths.add(pure_path)
    manifest = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "models": records, "selected_model": selected,
        "selection": protocol["selection_policy"],
        "test_metadata_note": "The converter sidecar's aggregate class counts were incidentally visible during a metadata read after all fits were predeclared. No sequence labels or scores were inspected before freeze; counts did not change training, candidates, or selection.",
        "stress": {"missing_rate": .41, "mean_gap_s": 1., "seed": 2026},
        "pure_player_dead_diagnostic": str(pure_path.relative_to(ROOT)),
        "competition_candidates": ["baseline_robust", "selected_refinement", "player_only_dead_diagnostic",
                                   "pathcrf_team_robust_dead", "pathcrf_team_refined_dead"],
        "competition_use": "One final public-five-minute pipeline check only; never training, tuning, or model selection",
        "sha256": {str(path.relative_to(ROOT)): robust.sha256(path) for path in sorted(paths)},
    }
    save(MANIFEST, manifest)
    save(PUBLIC / "refinement_manifest.json", manifest)
    print("FROZEN", selected, records[selected]["validation"]["macro_f1"],
          records[selected]["validation"]["spell_f1_mean"], flush=True)


def verified():
    manifest = read(MANIFEST)
    for path, expected in manifest["sha256"].items():
        if robust.sha256(ROOT / path) != expected:
            raise RuntimeError(f"Changed after freeze: {path}")
    return manifest


def predict(data, record, cache=None):
    module = importlib.import_module("experiments." + record["module"])
    model = joblib.load(ROOT / record["model_path"])
    inputs = robust.tracking_inputs(data)
    # Reuse freshly computed geometry for models with the identical feature
    # schema. No labels enter this cache; its lifetime is one input condition.
    if cache is not None and record["module"] in ("refine_features", "refine_combined", "refine_scale", "refine_ensemble"):
        feature_module = importlib.import_module("experiments.refine_features")
        if "input_id" in cache and cache["input_id"] != id(data):
            raise ValueError("Prediction cache cannot cross input conditions")
        cache["input_id"] = id(data)
        if "physical" not in cache:
            cache["physical"] = feature_module.features(inputs)
        if record["module"] == "refine_ensemble":
            matrix, names = cache["physical"]
            return module.predict_from_features(matrix, names, inputs, model)
        key = ("probabilities", record["model_path"])
        if key not in cache:
            matrix, names = cache["physical"]
            cache[key] = feature_module.probabilities_from_features(matrix, names, model)
        probabilities = cache[key]
        if record["module"] == "refine_combined":
            return module.predict_from_probabilities(probabilities, inputs, record["config"])
        if record["module"] == "refine_scale":
            return module.decode(probabilities, inputs, model["config"])
        return supervised.decode(probabilities, supervised.model_periods(inputs), float(inputs["fps"]),
                                 model["config"]["switch_cost"])
    if record["module"] in ("refine_decode", "refine_combined"):
        return module.predict(inputs, model, record["config"])
    return module.predict(inputs, model)


def started(name):
    path = OUT / f"{name}_started.json"
    if path.exists():
        raise RuntimeError(f"{name} already started; inspect existing artifacts, do not repeat test-driven selection")
    save(path, {"utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": robust.sha256(MANIFEST)})


def test():
    manifest = verified()
    started("test")
    match = manifest["protocol"]["test"][0]
    data = supervised.load(MATCHES / f"{match}.npz")
    report = {"match": match, "selected_before_test": manifest["selected_model"], "conditions": {}}
    for condition, view in (("clean", data), ("masked", mask_ball(data, **manifest["stress"]))):
        rows = {}
        prediction_cache = {}
        for name, record in manifest["models"].items():
            prediction = predict(view, record, prediction_cache)
            np.save(OUT / f"{match}_{condition}_{name}.npy", prediction)
            draws = 1000 if name == manifest["selected_model"] and condition == "clean" else 0
            rows[name] = aggregates(benchmark.score(view, prediction, bootstrap_n=draws))
            report["conditions"][condition] = rows
            save(OUT / "test.json", report)
            print(condition, name, rows[name]["macro_f1"], rows[name]["spell_f1_mean"], flush=True)
    save(PUBLIC / "refinement_test.json", report)


def challenge():
    manifest = verified()
    if (OUT / "competition/frozen_predictions.json").exists() or (OUT / "competition_check_started.json").exists():
        raise RuntimeError("Competition predictions already frozen; no regeneration after public inspection")
    data = supervised.load(ROOT / "data/challenge/inputs.npz")
    selected = predict(data, manifest["models"][manifest["selected_model"]])
    baseline = predict(data, manifest["models"]["baseline"])
    domain = importlib.import_module("experiments.refine_domain")
    pure = domain.predict(robust.tracking_inputs(data), joblib.load(ROOT / manifest["pure_player_dead_diagnostic"]))
    pretrained = ROOT / "data/experiments/challenge_pathcrf"
    with np.load(pretrained / "predictions.npz", allow_pickle=False) as source:
        native = source["pathcrf_ha"].copy()
        hybrid = source["pathcrf_team_robust_dead"].copy()
    hybrid_refined = native.copy()
    hybrid_refined[selected == "D"] = "D"
    outputs = {"baseline_robust": baseline, "selected_refinement": selected,
               "player_only_dead_diagnostic": pure, "pathcrf_team_robust_dead": hybrid,
               "pathcrf_team_refined_dead": hybrid_refined}
    signatures = {}
    for name, prediction in outputs.items():
        prediction[data["period"] == 0] = "X"
        path = write_submission(prediction, OUT / "competition" / f"{name}.csv")
        signatures[str(path.relative_to(ROOT))] = robust.sha256(path)
    save(OUT / "competition/frozen_predictions.json", {
        "utc": datetime.now(timezone.utc).isoformat(), "sha256": signatures,
        "selected_model": manifest["selected_model"], "ground_truth_read": False,
        "pathcrf_manifest_sha256": robust.sha256(pretrained / "frozen_config.json"),
        "note": "All candidates predeclared; public scores cannot select a model or alter these outputs",
    })
    print("Validated and froze competition predictions; no ground truth read")


def competition_check():
    manifest = verified()
    frozen = read(OUT / "competition/frozen_predictions.json")
    predictions = {}
    for path, expected in frozen["sha256"].items():
        if robust.sha256(ROOT / path) != expected:
            raise RuntimeError(f"Competition prediction changed: {path}")
        frame = pd.read_csv(ROOT / path)
        error = pm.validate_submission(frame, 199319)
        if error:
            raise ValueError(error)
        predictions[Path(path).stem] = frame.label.to_numpy()
    started("competition_check")
    gt_path = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    truth = pm.label_codes(pm.read_label_csv(str(gt_path)))
    predictions["organizer_smooth50"] = pm.read_label_csv(str(ROOT / "evaluation/results/baseline_smooth50.csv"))
    report = {
        "use": manifest["competition_use"], "full_match_ground_truth_available": False,
        "selected_before_public_check": manifest["selected_model"],
        "public_ground_truth_sha256": robust.sha256(gt_path),
        "graded_frames": int(np.sum(truth != pm.X)), "scores": {},
    }
    for name, prediction in predictions.items():
        scores = pm.evaluate(truth, pm.label_codes(prediction), bootstrap_n=1000)
        report["scores"][name] = aggregates(scores)
        save(OUT / "competition/public_pipeline.json", report)
        print(name, scores["macro_f1"], scores["spell_f1_mean"], scores["accuracy"], flush=True)
    save(PUBLIC / "refinement_competition.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "test", "challenge", "competition-check"))
    args = parser.parse_args()
    {"freeze": freeze, "test": test, "challenge": challenge,
     "competition-check": competition_check}[args.action]()
