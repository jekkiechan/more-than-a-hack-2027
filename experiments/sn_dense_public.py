"""Pooled D2 fits and one user-requested public diagnostic for sn_dense_v1.

``fit --seed S`` trains the selected D2 recipe (dense targets, D-path training,
SoccerNet fraction .5, 45 TRAIN games, width 96, lr 3e-4, 2,048 updates) on all
thirteen original TRAIN matches from the pooled combined-objective teacher.
``predict`` averages both seeds' probabilities, decodes once at cost 3 and seals
the 199,319-row CSV. ``score`` scores the sealed CSV once against the public
five-minute reference behind an exclusive gate. The recipe was selected from
external evidence before this module existed; no tuning follows the score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

import joblib
import numpy as np

from experiments import sn_dense_run as run, sn_dense_train as sn, soccernet_capacity as sc

ROOT, DATA = run.ROOT, run.DATA
OUT = run.OUT / "public"
SEEDS = (2026, 2027)
CSV = OUT / "competition_candidate.csv"
SEAL = OUT / "competition_prediction_seal.json"
STARTED = OUT / "competition_check_started.json"
RESULT = OUT / "competition_check_result.json"
INPUTS = DATA / "challenge/inputs.npz"
INPUT_SHA = "7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3"
ADOPTED = ROOT / "experiments/results/soccernet_capacity_public_check.json"
PUBLISHED = ROOT / "experiments/results/sn_dense_v1_public_check.json"
REQUEST = ("User asked on 2026-10-31 for the public-window score of the two-seed "
           "D2 ensemble after reviewing the completed external study.")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(value, handle, indent=2, default=float)
        handle.write("\n")


def model_path(seed: int) -> Path:
    return OUT / f"D2_f050_g45_pooled_s{seed}.joblib"


def fit(seed: int) -> None:
    path = model_path(seed)
    if path.exists():
        raise ValueError(f"Pooled checkpoint already exists: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    teacher = joblib.load(sc.initial_path("pooled", seed))
    old_records = run.old.load_old("pooled")
    train_records = run.load_train(False)
    out = sn.train(teacher, old_records, train_records, width=96, lr=3e-4, seed=seed,
                   steps=2048, fraction=0.5, batch_size=32, ha_target="dense",
                   train_d=True, require_full_native=True, log_every=256)
    joblib.dump(out["model"], path)
    print("POOLED_FIT_DONE", seed, sha(path), flush=True)


def predict() -> None:
    from experiments import current_baseline, live_gap_experiment
    if SEAL.exists() or STARTED.exists() or RESULT.exists():
        raise ValueError("Public prediction is already sealed")
    if sha(INPUTS) != INPUT_SHA:
        raise ValueError("Unexpected organizer input hash")
    inputs = current_baseline.load_inputs(INPUTS)
    placeholder = np.full(len(inputs["ball"]), "X", dtype="U1")
    record = sc.old.inference_record(live_gap_experiment.feature_view(
        inputs, placeholder, "competition", "BAR-ATH")["joint"])
    members, seals = [], {}
    for seed in SEEDS:
        tuned = joblib.load(model_path(seed))
        teacher_path = sc.initial_path("pooled", seed)
        teacher = joblib.load(teacher_path)
        probability = sn.predict(record, tuned, teacher)
        if not np.array_equal(probability, sn.predict(record, tuned, teacher)):
            raise ValueError("Inference is not exactly reproducible")
        members.append(sc.old.upsample(probability, record, inputs))
        seals[str(model_path(seed).relative_to(DATA))] = sha(model_path(seed))
        seals[str(teacher_path.relative_to(DATA))] = sha(teacher_path)
    probability = (members[0] + members[1]) * 0.5
    prediction = sc.supervised.decode(probability, sc.supervised.model_periods(inputs),
                                      float(inputs["fps"]), sc.lt.COST)
    prediction = np.asarray(prediction).astype("U1")
    prediction[np.asarray(inputs["period"]) <= 0] = "X"
    if len(prediction) != 199319:
        raise ValueError("Unexpected organizer row count")
    with CSV.open("x") as handle:
        handle.write("frame,label\n")
        for frame, label in enumerate(prediction):
            handle.write(f"{frame},{label}\n")
    from evaluation import possession_metrics as pm
    error = pm.validate_submission(pm.read_label_csv(CSV), 199319)
    if error:
        raise ValueError(error)
    np.savez_compressed(OUT / "competition_probability.npz", probability=probability,
                        prediction=prediction)
    adopted = json.loads(ADOPTED.read_text())
    exclusive(SEAL, dict(
        study_id="sn_dense_v1_public_check", created_utc=datetime.now(timezone.utc).isoformat(),
        recipe="D2_f050_g45: dense targets + D-path training, fraction .5, 45 games, w96, lr 3e-4, 2048 updates",
        declaration_sha256=sha(ROOT / "experiments/results/sn_dense_v1_declaration.json"),
        progress_sha256=sha(ROOT / "experiments/results/sn_dense_v1_progress.md"),
        prediction_sha256=sha(CSV), input_sha256=sha(INPUTS), models_sha256=seals,
        source_sha256={p: sha(ROOT / p) for p in ("experiments/sn_dense_public.py",
                       "experiments/sn_dense_train.py", "experiments/sn_dense_run.py",
                       "experiments/soccernet_dense_labels.py")},
        expected_reference_sha256=adopted.get("reference_sha256") or json.loads(
            (ROOT / "experiments/results/combined_objective_public_check.json").read_text())["organizer_gt_sha256"],
        reference_git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                     text=True).strip(),
        rows=199319, format_valid=True, inference_replay_exact=True,
        organizer_reference_labels_read=False, public_used_for_selection=False,
        explicit_user_request=REQUEST))
    print("SEALED", sha(CSV), flush=True)


def score() -> dict:
    if RESULT.exists():
        return json.loads(RESULT.read_text())
    spec = json.loads(SEAL.read_text())
    if sha(CSV) != spec["prediction_sha256"]:
        raise ValueError("Sealed prediction changed")
    for rel, value in spec["models_sha256"].items():
        if sha(DATA / rel) != value:
            raise ValueError("Sealed model changed: " + rel)
    exclusive(STARTED, dict(started_utc=datetime.now(timezone.utc).isoformat(),
                            prediction_seal_sha256=sha(SEAL)))
    raw = subprocess.run(["git", "show", f"{spec['reference_git_commit']}:evaluation/gt/BAR-ATH_possession_gt_public.csv"],
                         cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout
    if hashlib.sha256(raw).hexdigest() != spec["expected_reference_sha256"]:
        raise ValueError("Reference hash differs")
    reference = OUT / "public_reference.csv"
    reference.write_bytes(raw)
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    value = pm.evaluate(pm.label_codes(pm.read_label_csv(reference)),
                        pm.label_codes(pm.read_label_csv(CSV)), bootstrap_n=1000)
    summary = _score_summary(value)
    if (summary["graded_frames"], summary["ground_truth_spells"]) != (7500, 17):
        raise ValueError("Unexpected public population")
    adopted = json.loads(ADOPTED.read_text())["candidate"]
    result = dict(study_id="sn_dense_v1_public_check", candidate=summary, adopted_baseline=adopted,
                  deltas={k: summary[k] - adopted[k] for k in ("spell_f1_mean", "macro_f1", "d_f1")},
                  prediction_seal_sha256=sha(SEAL), organizer_reference_labels_read=True,
                  public_used_for_selection=False, explicit_user_request=REQUEST,
                  interpretation=("Exposed five-minute pipeline diagnostic with 17 reference "
                                  "possessions; no tuning, selection or adoption follows it."))
    exclusive(RESULT, result)
    exclusive(PUBLISHED, result)
    print("PUBLIC_RESULT", json.dumps(result["deltas"]), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit"); f.add_argument("--seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("predict"); sub.add_parser("score")
    args = parser.parse_args()
    if args.cmd == "fit":
        fit(args.seed)
    elif args.cmd == "predict":
        predict()
    else:
        score()


if __name__ == "__main__":
    main()
