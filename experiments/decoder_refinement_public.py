"""Two frozen predictions, one user-requested public diagnostic, no tuning."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from experiments import decoder_refinement as decoder, decoder_refinement_run as study
from experiments import current_baseline as baseline, decoder_search, supervised
from experiments import decoder_search_run as previous

ROOT = study.ROOT
OUT = study.OUT / "public"
SEAL = OUT / "seal.json"
STARTED = OUT / "started.json"
RESULT = study.RESULTS / "decoder_refinement_v3_public_check.json"
INPUT = ROOT / "data/challenge/inputs.npz"
MODELS = ROOT.parent / "mtah-data/models/joint_gru_sn_dense_ensemble_v1"
PRIOR_CSV = previous.OUT / "competition_candidate.csv"


def prepare():
    study.verify()
    if RESULT.exists() or OUT.exists():
        raise ValueError("Public preparation/gate already exists")
    aggregate = json.loads(study.AGGREGATE.read_text())
    selected = aggregate["selected_config"]
    manifest = json.loads(baseline.MANIFEST.read_text())
    inputs = baseline.load_inputs(INPUT)
    models = baseline.load_models(MODELS)
    probability = baseline.predict_probabilities(inputs, models, manifest)
    period = supervised.model_periods(inputs)
    fps = float(inputs["fps"])
    old = decoder_search.decode(probability, period, fps, dict(family="matrix", ha=3, dead=12))
    refined = decoder.decode(probability, period, fps, selected)
    ordinary = supervised.decode(probability, period, fps, 3)
    for prediction in (old, refined, ordinary):
        prediction[np.asarray(inputs["period"]) <= 0] = "X"
    OUT.mkdir(parents=True, exist_ok=False)
    # Verify exact deployed model inference before any organizer reference access.
    baseline_path = OUT / "baseline_replay.csv"
    baseline.challenge.write_submission(ordinary, baseline_path)
    if study.shared.sha256_file(baseline_path) != manifest["evidence"]["prediction_csv_sha256"]:
        raise ValueError("Adopted baseline CSV does not exactly reproduce sealed deployment")
    paths = {}
    for name, prediction in (("original_cost3_dead12", old), ("refined", refined)):
        path = OUT / (name+".csv")
        baseline.challenge.write_submission(prediction, path)
        paths[name] = {"path": str(path.relative_to(ROOT)), "sha256": study.shared.sha256_file(path)}
    if paths["original_cost3_dead12"]["sha256"] != study.shared.sha256_file(PRIOR_CSV):
        raise ValueError("Original split-cost CSV does not reproduce prior candidate")
    np.savez_compressed(OUT / "probabilities.npz", probabilities=probability)
    input_sources = sorted((ROOT / "experiments").glob("*.py")) + [ROOT / "evaluation/possession_metrics.py"]
    expected_gt = json.loads((study.RESULTS / "combined_objective_public_check.json").read_text())["organizer_gt_sha256"]
    previous.write_once(SEAL, {
        "study": "decoder_refinement_v3_public", "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "explicit_user_request": json.loads(study.DECLARATION.read_text())["explicit_user_request"],
        "selected_config": selected, "predictions": paths,
        "declaration_sha256": study.shared.sha256_file(study.DECLARATION),
        "aggregate_sha256": study.shared.sha256_file(study.AGGREGATE),
        "baseline_manifest_sha256": study.shared.sha256_file(baseline.MANIFEST),
        "models_sha256": {str((MODELS/name).relative_to(ROOT.parent)): digest for name,digest in manifest["models_sha256"].items()},
        "input_sha256": study.shared.sha256_file(INPUT),
        "source_sha256": {str(p.relative_to(ROOT)): study.shared.sha256_file(p) for p in input_sources},
        "baseline_replay_exact": True, "original_candidate_replay_exact": True,
        "expected_reference_sha256": expected_gt, "rows": 199319,
        "reference_labels_read": False, "public_used_for_selection": False,
        "candidate_set": "Original split-cost winner and externally selected refinement; identical CSV hashes scored once; adopted baseline reused from saved report",
        "no_selection_tuning_or_promotion_after_public": True,
    })
    print("Sealed public candidates", paths, flush=True)


def validate_seal(spec):
    for rel,digest in spec["source_sha256"].items():
        if study.shared.sha256_file(ROOT / rel) != digest:
            raise ValueError(f"Sealed source changed: {rel}")
    for entry in spec["predictions"].values():
        if study.shared.sha256_file(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("Sealed CSV changed")
    for key,path in (("declaration_sha256",study.DECLARATION), ("aggregate_sha256",study.AGGREGATE),
                     ("input_sha256",INPUT), ("baseline_manifest_sha256",baseline.MANIFEST)):
        if study.shared.sha256_file(path) != spec[key]:
            raise ValueError(f"Sealed artifact changed: {path}")
    for rel,digest in spec["models_sha256"].items():
        if study.shared.sha256_file(ROOT.parent / rel) != digest:
            raise ValueError("Sealed model changed")


def score():
    if RESULT.exists():
        return json.loads(RESULT.read_text())
    if STARTED.exists():
        raise ValueError("Public gate permanently closed; do not rescore")
    spec = json.loads(SEAL.read_text())
    validate_seal(spec)
    # Gate is exclusive and permanent even if reference validation/scoring fails.
    previous.write_once(STARTED, {"started_utc": datetime.now(timezone.utc).isoformat(),
        "seal_sha256": study.shared.sha256_file(SEAL)})
    raw = (ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv").read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec["expected_reference_sha256"]:
        raise ValueError("Reference SHA mismatch; gate remains closed")
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    truth = pm.label_codes(pm.read_label_csv(io.BytesIO(raw)))
    by_digest, scores = {}, {}
    for name, entry in spec["predictions"].items():
        if entry["sha256"] not in by_digest:
            frame = pm.read_label_csv(ROOT / entry["path"])
            error = pm.validate_submission(frame, 199319)
            if error:
                raise ValueError(error)
            value = pm.evaluate(truth, pm.label_codes(frame), bootstrap_n=1000)
            if (value["n_graded"],value["n_gt_spells"]) != (7500,17):
                raise ValueError("Unexpected public grading population")
            by_digest[entry["sha256"]] = _score_summary(value)
        scores[name] = by_digest[entry["sha256"]]
    adopted = json.loads((study.RESULTS / "sn_dense_v1_public_check.json").read_text())["candidate"]
    result = {"study": "decoder_refinement_v3_public", "candidate_scores": scores,
        "adopted_baseline_reused": adopted,
        "delta_vs_adopted": {name:{k:s[k]-adopted[k] for k in ("spell_f1_mean","macro_f1","d_f1")} for name,s in scores.items()},
        "prediction_seal_sha256": study.shared.sha256_file(SEAL),
        "unique_csvs_scored": len(by_digest), "reference_sha256": spec["expected_reference_sha256"],
        "public_used_for_training_tuning_selection_or_promotion": False,
        "interpretation": "One exposed public five-minute diagnostic, 17 possessions; full-match score unknown; no tuning or selection follows",
        "explicit_user_request": spec["explicit_user_request"]}
    previous.write_once(RESULT,result)
    print(json.dumps(result["delta_vs_adopted"],indent=2),flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","score"))
    args=parser.parse_args()
    prepare() if args.stage=="prepare" else score()
