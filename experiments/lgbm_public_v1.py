"""One predeclared pooled, block-safe refiner fit and requested public diagnostic.

Public labels are opened only by ``public`` after an exclusive gate is claimed.
The adopted package and every historical scoring gate remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from experiments import ball_exit_runtime, challenge, current_baseline, supervised
from experiments import feed_benchmark_audit_v1 as audit
from experiments import feed_like_v2, tracking_corruption
from experiments import spell_frontier_baseline as frontier
from experiments import v2_target_push_refiner as original

ROOT = Path(__file__).resolve().parents[1]
STUDY = "lgbm_public_v1"
OUT = ROOT / "data/experiments" / STUDY
RESULTS = ROOT / "experiments/results"
DECL = RESULTS / f"{STUDY}_declaration.json"
PREP = OUT / "preparation.json"
FIT = RESULTS / f"{STUDY}_fit.json"
MODEL = OUT / "refiner.txt"
CSV = OUT / "candidate.csv"
SEAL = RESULTS / f"{STUDY}_prediction_seal.json"
GATE = RESULTS / f"{STUDY}_public_gate.json"
RESULT = RESULTS / f"{STUDY}_public.json"
BASELINE = RESULTS / "ball_exit_reconstructed_v1_public.json"
MANIFEST = ROOT / "experiments/baseline_manifest.json"
INPUT = ROOT / "data/challenge/inputs.npz"
REFERENCE_SHA = "e596a9af9626c623bb94be6fd350e12ee9a75c76a696cfadca3bc01735e92487"
RESTORE_RECEIPT = RESULTS / f"{STUDY}_restore.json"
VIEWS = (("clean", (0,)), ("feed_like_v1", (1, 2)), ("feed_like_v2", (1, 2, 3, 4)))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def sources():
    # Include transitive experiment code, official scorer, and output formatting.
    paths = list((ROOT / "experiments").glob("*.py"))
    paths += list((ROOT / "evaluation").glob("*.py"))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def declare():
    import lightgbm
    write_once(DECL, dict(study=STUDY, utc=datetime.now(timezone.utc).isoformat(),
        authorization="User requested public evaluation first on 2026-11-12; no automatic adoption",
        question="Does the fixed pooled block-safe R2_full_w0.5 adaptation execute correctly on the challenge feed?",
        candidate="R2_full_w0.5 pooled block-safe adaptation; not the original LOPO checkpoint",
        changes_from_original=["One pooled fit on all five exposed development matches",
            "All temporal features reset at model-period/half/time breaks; frozen original untouched"],
        training="All 35 clean/v1x2/v2x4 development cells; native stride5; X excluded; D versus H/A",
        params=original.PARAMS, lightgbm_version=lightgbm.__version__,
        matches=[list(x) for x in audit.MATCHES], blend_weight=.5,
        budget="Exactly one seed2026 pooled LightGBM fit; one sealed public score; no search",
        controls="Exact full baseline CSV parity before reference access; reuse saved baseline scores",
        expected_control_sha256="56425990299fcdda15bf799cfc7384d53f444885e8141ebdd600965f64c5274f",
        expected_input_sha256="7c8763e926c933efa0d4324fa4dad3cf9f4416192db56cccf507250ff77698a3",
        expected_reference_sha256=REFERENCE_SHA, restore_receipt_sha256=sha(RESTORE_RECEIPT),
        public_population=dict(frames=199319, graded=7500, spells=17), bootstrap_n=1000,
        source_sha256=sources(), baseline_manifest_sha256=sha(MANIFEST),
        saved_baseline_sha256=sha(BASELINE),
        limitations=["Exposed development and label-conditioned v2 training corruption",
            "Training uses cached provider-fold upstream probabilities; deployment uses pooled upstream models",
            "No nested full-pipeline holdout or independent transfer confirmation claimed",
            "Original failed v1 macro guard remains failed; adaptation has no matched development score"],
        stopping_rule="Do not tune, select, refit or promote using public results. Gate never resets."))


def check_declaration():
    d = read(DECL)
    if (d["source_sha256"] != sources() or d["baseline_manifest_sha256"] != sha(MANIFEST)
            or d["restore_receipt_sha256"] != sha(RESTORE_RECEIPT)):
        raise ValueError("Declared source or baseline changed")
    return d


def components(cell):
    from experiments import frontier_postprocess_v1 as fp
    from experiments import sc2020_realfeed_score as score
    return {role: fp.pool_family(score._family(cell, family), .5)
            for role, family in (("ce", "ce"), ("adopted", "predecessor"), ("calibrated", "calibrated"))}


def prepare(raw_dir, cells_dir, calibration):
    from experiments import lgbm_public_runtime as runtime
    check_declaration()
    if PREP.exists() or FIT.exists() or GATE.exists():
        raise FileExistsError("Preparation/fitting already sealed")
    cal = read(calibration)["calibration"]
    restored = read(RESTORE_RECEIPT)
    if sha(calibration) != restored["calibration_sha256"]:
        raise ValueError("Pinned calibration changed")
    records, names = [], None
    for provider, match, filename in audit.MATCHES:
        for view, reps in VIEWS:
            for rep in reps:
                key = f"{provider}__{match}__{view}__{rep}"
                suffix = f"feed_like_corrupt_{rep}" if view == "feed_like_v1" else "clean_clean"
                raw = raw_dir / f"{filename}_{suffix}.joblib"
                if sha(raw) != restored["raw_inputs"][raw.name]:
                    raise ValueError("Pinned raw development input changed")
                payload = joblib.load(raw)
                labels = np.asarray(payload["labels"]).astype("U1")
                inputs = tracking_corruption.inputs_only(payload["inputs"])
                if view == "feed_like_v2":
                    seed = int(hashlib.sha256(f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8], 16)
                    inputs = feed_like_v2.corrupt(inputs, labels, cal, seed)
                cp = cells_dir / f"{key}.npz"
                if sha(cp) != restored["cells"][cp.name]:
                    raise ValueError("Pinned probability cell changed")
                with np.load(cp, allow_pickle=False) as z:
                    cell = {k: z[k] for k in z.files if not k.startswith("identity__")}
                for field, actual in (("labels", labels), ("period", inputs["period"]),
                                      ("model_period", supervised.model_periods(inputs)), ("timestamps", inputs["timestamps"])):
                    if not np.array_equal(cell[field], actual):
                        raise ValueError(f"Restored cell mismatch {key}/{field}")
                nm, X = runtime.features(inputs, components(cell))
                if names is not None and nm != names:
                    raise ValueError("Feature ordering changed")
                names = nm
                idx = np.arange(0, len(labels), 5)
                idx = idx[labels[idx] != "X"]
                target = OUT / "training" / f"{key}.npz"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(target)
                np.savez_compressed(target, X=X[idx], y=(labels[idx] == "D").astype(np.int8))
                records.append(dict(key=key, rows=len(idx), path=str(target.relative_to(ROOT)),
                    sha256=sha(target), raw_path=str(raw), raw_sha256=sha(raw),
                    cell_path=str(cp), cell_sha256=sha(cp)))
                print("Prepared", key, len(idx), flush=True)
    assert len(records) == 35
    write_once(PREP, dict(declaration_sha256=sha(DECL), feature_names=names,
        calibration_path=str(calibration), calibration_sha256=sha(calibration), cells=records))


def fit():
    import lightgbm as lgb
    check_declaration()
    p = read(PREP)
    if p["declaration_sha256"] != sha(DECL) or FIT.exists() or MODEL.exists() or GATE.exists():
        raise ValueError("Fit already exists or preparation changed")
    xs, ys = [], []
    roster = {f"{provider}__{match}__{view}__{rep}" for provider, match, _ in audit.MATCHES
              for view, reps in VIEWS for rep in reps}
    if len(p["cells"]) != 35 or {r["key"] for r in p["cells"]} != roster:
        raise ValueError("Preparation is not the canonical 35-cell cohort")
    for r in p["cells"]:
        f = ROOT / r["path"]
        if f != OUT / "training" / f"{r['key']}.npz":
            raise ValueError("Noncanonical training cache path")
        if sha(f) != r["sha256"]:
            raise ValueError("Training cache changed")
        with np.load(f) as z:
            xs.append(z["X"]); ys.append(z["y"])
    X, y = np.concatenate(xs), np.concatenate(ys)
    model = lgb.LGBMClassifier(**original.PARAMS).fit(X, y, feature_name=p["feature_names"])
    model.booster_.save_model(str(MODEL))
    write_once(FIT, dict(study=STUDY, declaration_sha256=sha(DECL), preparation_sha256=sha(PREP),
        model_sha256=sha(MODEL), feature_names=p["feature_names"], training_rows=len(y),
        training_D_fraction=float(y.mean()), fits=1, organizer_labels_used=False))
    print("Fit sealed", len(y), flush=True)


def challenge_components():
    """Reference-free preparation can run while development assets restore."""
    out = OUT / "challenge_components.npz"
    if out.exists():
        raise FileExistsError(out)
    models = current_baseline.load_models()
    manifest = read(MANIFEST)
    inputs = current_baseline.load_inputs(INPUT)
    c = frontier.native_components(inputs, models, ball_exit_runtime.frontier_manifest(manifest))
    labels = ball_exit_runtime.decode_components(inputs, c)
    labels[np.asarray(inputs["period"]) <= 0] = "X"
    OUT.mkdir(parents=True, exist_ok=True)
    control = OUT / "baseline_replay.csv"
    challenge.write_submission(labels, control)
    expected = "56425990299fcdda15bf799cfc7384d53f444885e8141ebdd600965f64c5274f"
    if sha(control) != expected:
        raise ValueError("Baseline full CSV parity failed before scoring")
    np.savez_compressed(out, **c)
    write_once(OUT / "challenge_components.json", dict(input_sha256=sha(INPUT),
        manifest_sha256=sha(MANIFEST), components_sha256=sha(out), control_sha256=sha(control),
        sources=sources(), models=manifest["models_sha256"], reference_access=False))
    print("Baseline CSV matches exactly; native components saved", flush=True)


def predict():
    import lightgbm as lgb
    from experiments import lgbm_public_runtime as runtime
    d = check_declaration(); fit_record = read(FIT)
    if fit_record["model_sha256"] != sha(MODEL) or fit_record["declaration_sha256"] != sha(DECL):
        raise ValueError("Model/declaration changed")
    if CSV.exists() or SEAL.exists() or GATE.exists():
        raise FileExistsError("Prediction or gate already exists")
    cp = OUT / "challenge_components.npz"; rec = read(OUT / "challenge_components.json")
    if (rec["components_sha256"] != sha(cp) or rec["input_sha256"] != sha(INPUT)
            or rec["manifest_sha256"] != sha(MANIFEST) or rec["sources"] != sources()
            or sha(INPUT) != d["expected_input_sha256"]):
        raise ValueError("Challenge components/input/source changed")
    inputs = current_baseline.load_inputs(INPUT)
    with np.load(cp) as z:
        c = {k: z[k] for k in z.files}
    names, _ = runtime.features(inputs, c)
    if names != fit_record["feature_names"]:
        raise ValueError("Deployment feature order differs from training")
    pred = runtime.predict_labels(inputs, c, lgb.Booster(model_file=str(MODEL)))
    pred[np.asarray(inputs["period"]) <= 0] = "X"
    challenge.write_submission(pred, CSV)
    write_once(SEAL, dict(study=STUDY, declaration_sha256=sha(DECL), fit_sha256=sha(FIT),
        preparation_sha256=sha(PREP), model_sha256=sha(MODEL), prediction_sha256=sha(CSV),
        input_sha256=sha(INPUT), baseline_manifest_sha256=sha(MANIFEST),
        baseline_replay_sha256=sha(OUT / "baseline_replay.csv"),
        source_sha256=sources(), reference_access=False, rows=len(pred)))
    print("Candidate generated and sealed without reference access", flush=True)


def public():
    d = check_declaration(); s = read(SEAL)
    checks = ((CSV, "prediction_sha256"), (FIT, "fit_sha256"), (MODEL, "model_sha256"),
              (INPUT, "input_sha256"), (MANIFEST, "baseline_manifest_sha256"),
              (DECL, "declaration_sha256"), (PREP, "preparation_sha256"))
    if any(sha(p) != s[k] for p, k in checks) or s["source_sha256"] != sources():
        raise ValueError("Sealed evidence changed")
    if (s["baseline_replay_sha256"] != d["expected_control_sha256"]
            or sha(BASELINE) != d["saved_baseline_sha256"] or RESULT.exists()):
        raise ValueError("Control changed or result exists")
    write_once(GATE, dict(study=STUDY, seal_sha256=sha(SEAL),
        authorization=d["authorization"], claimed_utc=datetime.now(timezone.utc).isoformat()))
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    reference = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
    if sha(reference) != d["expected_reference_sha256"]:
        raise ValueError("Public reference changed; gate remains closed")
    score = pm.evaluate(pm.label_codes(pm.read_label_csv(reference)),
        pm.label_codes(pm.read_label_csv(CSV)), bootstrap_n=1000)
    if (score["n_frames"], score["n_graded"], score["n_gt_spells"]) != (199319, 7500, 17):
        raise ValueError("Unexpected public population; gate remains closed")
    candidate, baseline = _score_summary(score), read(BASELINE)["candidate"]
    write_once(RESULT, dict(study=STUDY, candidate=candidate, saved_baseline=baseline,
        delta={k: candidate[k]-baseline[k] for k in ("spell_f1_mean", "macro_f1", "d_f1")},
        seal_sha256=sha(SEAL), gate_sha256=sha(GATE), selection_or_tuning=False,
        reference_sha256=d["expected_reference_sha256"],
        adopted=False, interpretation="Requested diagnostic on 17 exposed possessions; no significance or full-match claim"))
    print(json.dumps(read(RESULT), indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("declare", "prepare", "fit", "components", "predict", "public"))
    p.add_argument("--raw-dir", type=Path)
    p.add_argument("--cells-dir", type=Path)
    p.add_argument("--calibration", type=Path)
    a = p.parse_args()
    if a.action == "prepare":
        if not all((a.raw_dir, a.cells_dir, a.calibration)):
            p.error("prepare needs raw-dir, cells-dir, calibration")
        prepare(a.raw_dir, a.cells_dir, a.calibration)
    else:
        {"declare": declare, "fit": fit, "components": challenge_components,
         "predict": predict, "public": public}[a.action]()


if __name__ == "__main__":
    main()
