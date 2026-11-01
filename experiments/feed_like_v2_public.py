"""One user-requested fixed public diagnostic: adopted `original_full` + 5 Hz phase TTA.

The candidate is fixed before any reference access: the promoted deployment package
(verified hashes), equal two-seed native probability mean per sampling phase, equal mean
over all 5 phases, unchanged f062 decoder. Phase 0 must replay the sealed adopted CSV
byte for byte. The public gate is a one-time exclusive marker; once claimed it stays
closed, and the result is reused. No tuning, reselection or promotion follows from it.

Steps: `predict` (label-free CSV + seal), then `score` (claims gate, official scorer).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from experiments import current_baseline as baseline, feed_spell_push, calibrated_dense_baseline as calibration
from experiments import sn_dense_baseline, tracking_corruption, supervised, challenge
from experiments import feed_benchmark_audit_v1 as audit, live_gap_experiment, soccernet_extension as old
from experiments import sn_dense_train as sn

STUDY = "feed_like_v2_public"
ROOT = audit.ROOT
OUT = ROOT / "data/experiments" / STUDY
RESULT = ROOT / "experiments/results" / f"{STUDY}_result.json"
SEAL = OUT / "prediction_seal.json"
GATE = OUT / "public_gate_claimed.json"
INPUT = ROOT / "data/challenge/inputs.npz"
REFERENCE = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
SEALED_ADOPTED_CSV_SHA256 = "7fe3da71b0cbd66bee046597611aac016fd714ddfaec6ba924dc400d7466bd1f"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, default=float)
        f.write("\n")


def phase_native(inputs, models, manifest):
    """Per-phase two-seed native probability means; phase 0 is the deployment record."""
    sn_dense_baseline.validate_models(models, manifest["architecture"])
    clean = tracking_corruption.inputs_only(inputs)
    placeholder = np.full(len(clean["ball"]), "X", dtype="U1")
    base = old.inference_record(live_gap_experiment.feature_view(clean, placeholder, "competition", "BAR-ATH")["joint"])
    records = audit.phase_records(clean, base)
    if not (np.array_equal(records[0]["indices"], base["indices"])
            and np.allclose(records[0]["x"], base["x"], equal_nan=True)):
        raise ValueError("Phase-0 record differs from the deployment feature record")
    out = []
    for record in records:
        record = dict(record, names=list(record["names"]))
        members = [old.upsample(sn.predict(record, models[m], models[t]), record, clean)
                   for m, t in sn_dense_baseline.TEACHERS.items()]
        out.append(np.mean(members, axis=0))
    return out


def predict():
    if SEAL.exists():
        return
    manifest = json.loads(baseline.MANIFEST.read_text())
    if manifest["package_version"] != "joint_gru_sn_dense_original_diversity_ensemble_v1":
        raise ValueError("Expected the promoted original_full package")
    inputs = baseline.load_inputs(INPUT)  # tracking keys only; no label key is loaded
    models = baseline.load_models(None, baseline.MANIFEST)  # verifies model hashes
    phases = phase_native(inputs, models, manifest)
    period = supervised.model_periods(inputs)
    fps = float(inputs["fps"])
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, p in (("phase0", phases[0]), ("phase_tta", np.mean(phases, axis=0))):
        labels = feed_spell_push.decode(p[None], period, fps, calibration.CONFIG)
        labels[np.asarray(inputs["period"]) <= 0] = "X"
        path = OUT / f"original_full_{name}.csv"
        challenge.write_submission(labels, path)
        subprocess.run([sys.executable, str(ROOT / "evaluation/validate_submission.py"), str(path)], check=True,
                       capture_output=True)
        paths[name] = path
    if sha(paths["phase0"]) != SEALED_ADOPTED_CSV_SHA256:
        raise ValueError("Phase 0 does not replay the sealed adopted CSV")
    _write_once(SEAL, dict(candidate="original_full + 5 Hz phase TTA (5 phases, equal mean, f062)",
                           candidate_csv_sha256=sha(paths["phase_tta"]), phase0_csv_sha256=sha(paths["phase0"]),
                           phase0_replays_sealed_adopted=True, input_sha256=sha(INPUT),
                           manifest_sha256=sha(baseline.MANIFEST), models_sha256=manifest["models_sha256"],
                           reference_accessed=False))
    print("SEALED", sha(paths["phase_tta"]))


def score():
    if RESULT.exists():
        print(RESULT.read_text())
        return
    seal = json.loads(SEAL.read_text())
    path = OUT / "original_full_phase_tta.csv"
    if sha(path) != seal["candidate_csv_sha256"] or sha(INPUT) != seal["input_sha256"]:
        raise ValueError("Sealed candidate changed")
    _write_once(GATE, dict(seal_sha256=sha(SEAL), note="one-time public gate; never rescored"))
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    truth = pm.label_codes(pm.read_label_csv(REFERENCE))
    candidate = pm.read_label_csv(path)
    assert pm.validate_submission(candidate, 199319) is None
    measured = pm.evaluate(truth, pm.label_codes(candidate), bootstrap_n=1000)
    assert (measured["n_frames"], measured["n_graded"], measured["n_gt_spells"]) == (199319, 7500, 17)
    summary = _score_summary(measured)
    prior = json.loads((ROOT / "experiments/results/short_quality_public_v1_result.json").read_text())
    out = dict(study=STUDY, candidate_label=seal["candidate"], candidate=summary,
               original_full_reused=prior["candidate"], calibrated_dense_reused=prior["adopted_baseline_reused"],
               delta_vs_original_full={k: summary[k]-prior["candidate"][k] for k in ("spell_f1_mean", "macro_f1", "d_f1")},
               delta_vs_calibrated_dense={k: summary[k]-prior["adopted_baseline_reused"][k]
                                          for k in ("spell_f1_mean", "macro_f1", "d_f1")},
               candidate_csv_sha256=seal["candidate_csv_sha256"], prediction_seal_sha256=sha(SEAL),
               new_unique_csvs_scored=1, public_used_for_training_tuning_selection=False, promotion=False)
    _write_once(RESULT, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("predict", "score"))
    {"predict": predict, "score": score}[parser.parse_args().step]()
