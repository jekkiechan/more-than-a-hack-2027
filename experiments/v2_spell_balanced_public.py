"""Reference-free prediction seal and one-time public diagnostic for v2 spell balancing.

The CSV must be produced from the frozen selection and pooled TRAIN checkpoints
before this module is asked to score. ``seal`` validates and binds that CSV,
selection, input, models, and source files without touching organizer labels.
``score`` atomically claims a new study-specific gate before opening the public
reference. The public result is diagnostic only and cannot change selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_spell_balanced_v1"
SELECTION = ROOT / "experiments/results/v2_spell_balanced_v1_selection.json"
PLAN = ROOT / "experiments/results/v2_spell_balanced_v1_plan.json"
INPUT = ROOT / "data/challenge/inputs.npz"
REFERENCE = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
OUT = ROOT / "data/experiments" / STUDY
CSV = OUT / "public_candidate.csv"
SEAL = OUT / "public_prediction_seal.json"
GATE = OUT / "public_gate_claimed.json"
RESULT = ROOT / "experiments/results/v2_spell_balanced_v1_public_result.json"
BASELINE = ROOT / "experiments/results/feed_like_v2_public_result.json"
EXPECTED_FRAMES = 199319
EXPECTED_GRADED = 7500
EXPECTED_SPELLS = 17
PUBLIC_BASELINE_SPELL_F1 = 0.611764705882353
PUBLIC_BASELINE_MACRO_F1 = 0.8494725958535332
PUBLIC_BASELINE_D_F1 = 0.862869869551384


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _selection_contract(selection: dict) -> dict:
    if selection.get("study_id") != STUDY or selection.get("final_selection") is not True:
        raise ValueError("Selection belongs to another study")
    recipe = selection.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Selection has no frozen recipe")
    for name in ("ce_weight", "calibrated_weight", "adopted_weight"):
        if (name not in recipe or not isinstance(recipe[name], (int, float)) or
                not math.isfinite(float(recipe[name])) or float(recipe[name]) < 0.0):
            raise ValueError(f"Selection recipe is missing {name}")
    if abs(sum(float(recipe[name]) for name in
               ("ce_weight", "calibrated_weight", "adopted_weight")) - 1.0) > 1e-12:
        raise ValueError("Frozen expert weights must sum to one")
    if recipe.get("pooling") not in ("arithmetic", "log"):
        raise ValueError("Unsupported frozen pooling rule")
    if recipe.get("ha_cost") not in (1.5, 3, 3.0):
        raise ValueError("Unsupported frozen H/A decoder cost")
    if selection.get("training_stage") not in (512, 1024):
        raise ValueError("Selection has no valid frozen training stage")
    return recipe


def _expected_public_model_roles() -> set[str]:
    return {f"{role}/{seed}" for role in
            ("ce", "adopted", "calibrated", "teacher_adopted", "teacher_calibrated")
            for seed in (2026, 2027)}


def _csv_valid(csv_path: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    from evaluation import possession_metrics as pm
    frame = pm.read_label_csv(csv_path)
    error = pm.validate_submission(frame, EXPECTED_FRAMES)
    if error:
        raise ValueError(f"Public candidate CSV is invalid: {error}")


def _phase_public_probabilities(selection: dict) -> tuple[dict, dict[str, Path]]:
    """Run frozen pooled experts on organizer tracking input only."""
    import joblib
    import numpy as np
    from experiments import current_baseline as baseline
    from experiments import feed_benchmark_audit_v1 as audit
    from experiments import feed_spell_push as decoder
    from experiments import live_gap_experiment, sn_dense_baseline
    from experiments import sn_dense_train as sn
    from experiments import soccernet_extension as old
    from experiments import supervised, tracking_corruption, v2_iterative_runtime
    from experiments import v2_spell_balanced_train as training

    stage = int(selection["training_stage"])
    training.check_pooled(full=True)
    pool_seal = _json(training.POOL_SEAL)
    if pool_seal.get("study_id") != STUDY or not pool_seal.get("all_authorized_stages_complete"):
        raise ValueError("Pooled CE continuation is not fully sealed")

    original_manifest = _json(ROOT / "experiments/baseline_manifest.json")
    calibrated_manifest_path = ROOT / "experiments/baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json"
    calibrated_manifest = _json(calibrated_manifest_path)
    original_directory = ROOT / original_manifest["model_directory"]
    calibrated_directory = ROOT / calibrated_manifest["model_directory"]
    pooled_records = {(int(row["seed"]), int(row["stage"])): row for row in pool_seal["fits"]}

    model_files: dict[str, Path] = {}
    bundles = {name: {} for name in ("ce", "adopted", "calibrated")}
    for seed in (2026, 2027):
        ce_path = training._pooled_output_path(seed, stage)
        adopted_path = original_directory / f"dense_{seed}.joblib"
        calibrated_path = calibrated_directory / f"dense_{seed}.joblib"
        teacher_path = original_directory / f"teacher_{seed}.joblib"
        calibrated_teacher_path = calibrated_directory / f"teacher_{seed}.joblib"
        for name, path, manifest in (
                ("adopted", adopted_path, original_manifest),
                ("calibrated", calibrated_path, calibrated_manifest)):
            expected = manifest["models_sha256"][path.name]
            if sha(path) != expected:
                raise ValueError(f"Pooled {name} model hash mismatch: {path}")
            model_files[f"{name}/{seed}"] = path
        if sha(teacher_path) != original_manifest["models_sha256"][teacher_path.name]:
            raise ValueError(f"Adopted pooled teacher hash mismatch: {teacher_path}")
        if sha(calibrated_teacher_path) != calibrated_manifest["models_sha256"][calibrated_teacher_path.name]:
            raise ValueError(f"Calibrated pooled teacher hash mismatch: {calibrated_teacher_path}")
        model_files[f"teacher_adopted/{seed}"] = teacher_path
        model_files[f"teacher_calibrated/{seed}"] = calibrated_teacher_path
        fit = pooled_records.get((seed, stage))
        if fit is None or fit.get("sha256") != sha(ce_path):
            raise ValueError(f"Selected pooled CE stage is absent from its seal: {seed}/{stage}")
        model_files[f"ce/{seed}"] = ce_path
        bundles["ce"][seed] = (joblib.load(ce_path), joblib.load(teacher_path))
        bundles["adopted"][seed] = (joblib.load(adopted_path), joblib.load(teacher_path))
        bundles["calibrated"][seed] = (joblib.load(calibrated_path), joblib.load(calibrated_teacher_path))

    inputs = baseline.load_inputs(INPUT)
    clean = tracking_corruption.inputs_only(inputs)
    placeholder = np.full(len(clean["ball"]), "X", dtype="U1")
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass
    with v2_iterative_runtime.installed():
        feature = live_gap_experiment.feature_view(clean, placeholder, "competition", "BAR-ATH")["joint"]
        base = old.inference_record(feature)
        records = audit.phase_records(clean, base)
    if len(records) != 5:
        raise ValueError(f"Expected five native 5 Hz phase records, got {len(records)}")

    pooled = {expert: [] for expert in bundles}
    for phase_record in records:
        record = dict(phase_record, names=list(phase_record["names"]))
        for expert, members in bundles.items():
            seed_probs = []
            for seed in (2026, 2027):
                model, teacher = members[seed]
                sampled = sn.predict(record, model, teacher)
                seed_probs.append(old.upsample(sampled, record, clean))
            pooled[expert].append(seed_probs)
    return ({name: _pool_seed_phase_probabilities(np.transpose(
                np.asarray(phases, dtype=np.float64), (1, 0, 2, 3)))
             for name, phases in pooled.items()}, model_files)


def predict_and_seal() -> dict:
    """Infer and seal one full-length candidate, without accessing reference labels."""
    import numpy as np
    from experiments import current_baseline as baseline
    from experiments import feed_benchmark_audit_v1 as audit
    from experiments import v2_spell_balanced_eval as evaluator
    from experiments import (challenge, feed_spell_push, live_gap_experiment,
                             sn_dense_baseline, sn_dense_train, soccernet_extension,
                             supervised, tracking_corruption,
                             v2_spell_balanced_train as training)

    if SEAL.exists() or GATE.exists() or RESULT.exists() or CSV.exists():
        raise FileExistsError("Public candidate/gate artifacts already exist for this study")
    selection = _json(SELECTION)
    recipe = _selection_contract(selection)
    if not selection.get("public_input_sha256") or sha(INPUT) != selection["public_input_sha256"]:
        raise ValueError("Public tracking input differs from the frozen selection")
    components, model_files = _phase_public_probabilities(selection)
    inputs = baseline.load_inputs(INPUT)
    period = supervised.model_periods(inputs)
    adopted_decoded = audit.decode(components["adopted"], period,
                                   float(inputs["fps"]), audit.CONFIG)
    h_a = evaluator.recipe_probability(components, float(recipe["ce_weight"]),
                                       float(recipe["calibrated_weight"]), recipe["pooling"])
    prediction = evaluator.decode_fixed_d(h_a, adopted_decoded, period, float(inputs["fps"]),
                                          float(recipe["ha_cost"]))
    prediction = np.asarray(prediction).astype("U1", copy=False)
    prediction[np.asarray(inputs["period"]) <= 0] = "X"
    if not np.array_equal(prediction[np.asarray(inputs["period"]) <= 0],
                          np.full(np.sum(np.asarray(inputs["period"]) <= 0), "X", dtype="U1")):
        raise ValueError("Invalid/untracked frames must be masked as X")
    tracked = np.asarray(inputs["period"]) > 0
    if not np.isin(prediction[tracked], ("H", "A", "D")).all():
        raise ValueError("Tracked frames must have H/A/D labels")
    if not np.array_equal(prediction[tracked] == "D", adopted_decoded[tracked] == "D"):
        raise ValueError("Candidate decoding changed the fixed adopted D mask")

    # Label-free whole-pipeline replay against the saved prior public prediction hash.
    adopted_prediction = np.asarray(audit.decode(components["adopted"], period,
        float(inputs["fps"]), audit.CONFIG)).astype("U1")
    adopted_prediction[np.asarray(inputs["period"]) <= 0] = "X"
    adopted_bytes = ("frame,label\n" + "".join(
        f"{i},{label}\n" for i, label in enumerate(adopted_prediction))).encode("utf8")
    adopted_hash = hashlib.sha256(adopted_bytes).hexdigest()
    saved_public_hash = _json(BASELINE)["candidate_csv_sha256"]
    if adopted_hash != saved_public_hash:
        raise ValueError("Adopted pooled render does not match the saved public prediction hash")
    manifest = _json(ROOT / "experiments/baseline_manifest.json")
    default_output = ROOT / manifest["default_output"]
    adopted_replay = {"default_output_path": str(default_output.relative_to(ROOT)),
                      "available": default_output.is_file(), "replay_checked": True,
                      "saved_public_prediction_sha256": saved_public_hash,
                      "adopted_prediction_sha256": adopted_hash}
    if default_output.is_file():
        expected_hash = sha(default_output)
        adopted_replay.update(expected_sha256=expected_hash, exact_match=expected_hash == adopted_hash)
        if expected_hash != adopted_hash:
            raise ValueError("Adopted prediction does not exactly replay the current baseline CSV")
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("x") as stream:
        stream.write("frame,label\n")
        for frame, label in enumerate(prediction):
            stream.write(f"{frame},{label}\n")
    sources = tuple(ROOT / path for path in sorted(_expected_public_source_paths()))
    return seal_prediction(csv_path=CSV, model_files=model_files, source_files=sources,
                           pooled_seal_path=training.POOL_SEAL,
                           adopted_baseline_replay=adopted_replay)


def seal_prediction(*, csv_path: Path = CSV, selection_path: Path = SELECTION,
                    input_path: Path = INPUT, model_files: dict[str, Path],
                    source_files: tuple[Path, ...], pooled_seal_path: Path | None = None,
                    adopted_baseline_replay: dict | None = None) -> dict:
    """Bind a validated full-length CSV without reading organizer reference labels."""
    if SEAL.exists() or GATE.exists() or RESULT.exists():
        raise FileExistsError("This study's public prediction or scoring gate is already sealed")
    selection = _json(selection_path)
    recipe = _selection_contract(selection)
    if (not isinstance(adopted_baseline_replay, dict) or
            adopted_baseline_replay.get("replay_checked") is not True or
            adopted_baseline_replay.get("saved_public_prediction_sha256") !=
                _json(BASELINE).get("candidate_csv_sha256") or
            adopted_baseline_replay.get("adopted_prediction_sha256") !=
                _json(BASELINE).get("candidate_csv_sha256") or
            (adopted_baseline_replay.get("available") and
             adopted_baseline_replay.get("exact_match") is not True)):
        raise ValueError("A successful adopted public CSV hash replay is required before sealing")
    bindings = selection.get("development_result_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Final selection must bind both development rounds and controls")
    binding_hashes = {}
    for raw_path, expected_hash in bindings.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected_hash:
            raise ValueError(f"Frozen development evidence changed: {path}")
        binding_hashes[str(path)] = expected_hash
    if set(model_files) != _expected_public_model_roles():
        raise ValueError("Seal requires every selected CE, adopted, calibrated and paired teacher model")
    if pooled_seal_path is None:
        from experiments import v2_spell_balanced_train as training
        pooled_seal_path = training.POOL_SEAL
    if not pooled_seal_path.is_file():
        raise FileNotFoundError("A sealed pooled TRAIN model inventory is required")
    pooled_seal = _json(pooled_seal_path)
    if pooled_seal.get("study_id") != STUDY or pooled_seal.get("all_authorized_stages_complete") is not True:
        raise ValueError("Pooled model seal is invalid or incomplete")
    from experiments import v2_spell_balanced_train as training
    original_manifest = _json(ROOT / "experiments/baseline_manifest.json")
    calibrated_manifest = _json(ROOT / "experiments/baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json")
    expected_paths = {}
    for seed in (2026, 2027):
        expected_paths[f"ce/{seed}"] = training._pooled_output_path(seed, selection["training_stage"])
        expected_paths[f"adopted/{seed}"] = ROOT / original_manifest["model_directory"] / f"dense_{seed}.joblib"
        expected_paths[f"calibrated/{seed}"] = ROOT / calibrated_manifest["model_directory"] / f"dense_{seed}.joblib"
        expected_paths[f"teacher_adopted/{seed}"] = ROOT / original_manifest["model_directory"] / f"teacher_{seed}.joblib"
        expected_paths[f"teacher_calibrated/{seed}"] = ROOT / calibrated_manifest["model_directory"] / f"teacher_{seed}.joblib"
    if {key: Path(path).resolve() for key, path in model_files.items()} != {
            key: Path(path).resolve() for key, path in expected_paths.items()}:
        raise ValueError("Public model paths do not match the selected pooled stages and adopted manifests")
    fit_rows = {(int(row["seed"]), int(row["stage"])): row for row in pooled_seal["fits"]}
    for seed in (2026, 2027):
        row = fit_rows.get((seed, int(selection["training_stage"])))
        model_path = expected_paths[f"ce/{seed}"]
        if row is None or row.get("sha256") != sha(model_path):
            raise ValueError(f"Selected pooled fit is absent from the sealed inventory: {seed}")
    _verify_public_baseline()
    _csv_valid(csv_path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    model_hashes = {str(path): sha(path) for path in sorted(model_files.values())}
    if len(model_hashes) != len(model_files):
        raise ValueError("Model paths are duplicated")
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_files}
    expected_sources = _expected_public_source_paths()
    if set(source_hashes) != expected_sources:
        raise ValueError("Public source hash inventory is incomplete or contains extras")
    policy = _json(PLAN)
    if not policy["public_validation"]["authorized"]:
        raise ValueError("The frozen parent plan does not authorize public validation")
    value = {
        "study_id": STUDY,
        "plan_sha256": sha(PLAN),
        "pooled_seal_path": str(pooled_seal_path.relative_to(ROOT)),
        "pooled_seal_sha256": sha(pooled_seal_path),
        "selection_path": str(selection_path.relative_to(ROOT)),
        "selection_sha256": sha(selection_path),
        "selected_recipe": recipe,
        "training_stage": selection["training_stage"],
        "development_result_bindings": selection.get("development_result_bindings"),
        "development_binding_sha256": binding_hashes,
        "candidate_csv_path": str(csv_path.relative_to(ROOT)),
        "candidate_csv_sha256": sha(csv_path),
        "saved_public_baseline_sha256": sha(BASELINE),
        "input_path": str(input_path.relative_to(ROOT)),
        "input_sha256": sha(input_path),
        "model_sha256": model_hashes,
        "model_role_paths": {role: str(Path(path).resolve()) for role, path in model_files.items()},
        "source_sha256": source_hashes,
        "rows": EXPECTED_FRAMES,
        "adopted_baseline_replay": adopted_baseline_replay or {
            "available": False, "replay_checked": False,
            "reason": "No local CSV at the manifest default-output path"},
        "format_valid": True,
        "reference_read_or_hashed": False,
        "public_used_for_selection": False,
        "promotion": False,
    }
    _exclusive_json(SEAL, value)
    return value


def _expected_public_source_paths() -> set[str]:
    from experiments import (challenge, current_baseline, feed_benchmark_audit_v1,
                             feed_spell_push, live_gap_experiment, sn_dense_baseline,
                             sn_dense_train, soccernet_extension, supervised,
                             tracking_corruption, v2_iterative_runtime, v2_spell_balanced_eval,
                             v2_spell_balanced_train)
    paths = {str(Path(p).resolve().relative_to(ROOT)) for p in (
        __file__, v2_spell_balanced_eval.__file__, v2_spell_balanced_train.__file__,
        current_baseline.__file__, feed_benchmark_audit_v1.__file__,
        feed_spell_push.__file__, live_gap_experiment.__file__, sn_dense_baseline.__file__,
        sn_dense_train.__file__, soccernet_extension.__file__, supervised.__file__,
        tracking_corruption.__file__, v2_iterative_runtime.__file__, challenge.__file__)}
    paths.update(v2_spell_balanced_eval._source_hashes())
    paths.update({"evaluation/possession_metrics.py",
                  "experiments/combined_objective_public_check.py"})
    return paths


def _pool_seed_phase_probabilities(seed_phases: list[list]):
    """Mean paired seeds within each native phase, then mean the phases."""
    import numpy as np
    values = np.asarray(seed_phases, dtype=np.float64)
    if (values.ndim != 4 or values.shape[0] != 2 or values.shape[1] not in (2, 5) or
            values.shape[-1] != 3 or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("Expected aligned seed × phase × frame × H/A/D probabilities")
    if not np.allclose(values.sum(axis=-1), 1.0, rtol=0, atol=1e-6):
        raise ValueError("Every seed/phase probability row must sum to one")
    return values.mean(axis=0).mean(axis=0)


def _verify_public_baseline() -> dict:
    baseline = _json(BASELINE)["candidate"]
    expected = {"spell_f1_mean": PUBLIC_BASELINE_SPELL_F1,
                "macro_f1": PUBLIC_BASELINE_MACRO_F1,
                "d_f1": PUBLIC_BASELINE_D_F1}
    for key, value in expected.items():
        if abs(float(baseline[key]) - value) > 1e-15:
            raise ValueError(f"Saved adopted public baseline {key} differs from the declared value")
    return baseline


def _verify_seal(seal: dict) -> None:
    if seal.get("study_id") != STUDY or seal.get("reference_read_or_hashed") is not False:
        raise ValueError("Invalid public prediction seal identity")
    if seal.get("selection_sha256") != sha(SELECTION):
        raise ValueError("Frozen selection changed after CSV sealing")
    if sha(INPUT) != _json(SELECTION).get("public_input_sha256"):
        raise ValueError("Organizer tracking input differs from the final selection seal")
    if seal.get("plan_sha256") != sha(PLAN) or _json(PLAN)["public_validation"]["authorized"] is not True:
        raise ValueError("Parent public-validation authorization changed or is missing")
    if seal.get("saved_public_baseline_sha256") != sha(BASELINE):
        raise ValueError("Saved public baseline evidence changed after sealing")
    _verify_public_baseline()
    replay = seal.get("adopted_baseline_replay", {})
    prior_csv_hash = _json(BASELINE).get("candidate_csv_sha256")
    if (replay.get("replay_checked") is not True or
            replay.get("saved_public_prediction_sha256") != prior_csv_hash or
            replay.get("adopted_prediction_sha256") != prior_csv_hash or
            (replay.get("available") and replay.get("exact_match") is not True)):
        raise ValueError("Adopted whole-pipeline public prediction replay is missing or mismatched")
    pooled_seal_path = ROOT / seal["pooled_seal_path"]
    if sha(pooled_seal_path) != seal["pooled_seal_sha256"]:
        raise ValueError("Pooled TRAIN seal changed after prediction sealing")
    selection = _json(SELECTION)
    if _selection_contract(selection) != seal.get("selected_recipe") or selection.get("training_stage") != seal.get("training_stage"):
        raise ValueError("Selected recipe/stage differs from the prediction seal")
    if selection.get("development_result_bindings") != seal.get("development_result_bindings"):
        raise ValueError("Frozen development evidence bindings changed")
    if set(seal.get("development_binding_sha256", {})) != {
            str(Path(path).resolve()) if not Path(path).is_absolute() else str(Path(path))
            for path in selection.get("development_result_bindings", {})}:
        raise ValueError("Development result binding inventory changed")
    for raw_path, expected_hash in seal.get("development_binding_sha256", {}).items():
        if not Path(raw_path).is_file() or sha(Path(raw_path)) != expected_hash:
            raise ValueError(f"Frozen development result changed: {raw_path}")
    csv_path = ROOT / seal["candidate_csv_path"]
    input_path = ROOT / seal["input_path"]
    if sha(csv_path) != seal["candidate_csv_sha256"] or sha(input_path) != seal["input_sha256"]:
        raise ValueError("Candidate CSV or organizer input changed after sealing")
    _csv_valid(csv_path)
    for raw_path, expected in seal["model_sha256"].items():
        if sha(Path(raw_path)) != expected:
            raise ValueError(f"Selected pooled model changed: {raw_path}")
    if set(seal.get("model_role_paths", {})) != _expected_public_model_roles():
        raise ValueError("Public seal has incomplete selected model roles")
    if {sha(Path(path)) for path in seal["model_role_paths"].values()} != set(seal["model_sha256"].values()):
        raise ValueError("Public model role/path hash inventory differs from the model hash map")
    if set(seal.get("source_sha256", {})) != _expected_public_source_paths():
        raise ValueError("Public seal source inventory is incomplete or contains extras")
    current_sources = {name: sha(ROOT / name) for name in seal["source_sha256"]}
    if current_sources != seal["source_sha256"]:
        raise ValueError("Public prediction source changed after sealing")


def _claim_gate(seal: dict) -> dict:
    claim = {
        "study_id": STUDY,
        "claimed_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_seal_sha256": sha(SEAL),
        "candidate_csv_sha256": seal["candidate_csv_sha256"],
        "selection_sha256": seal["selection_sha256"],
        "one_time_public_score": True,
        "no_post_public_tuning_or_reselection": True,
    }
    _exclusive_json(GATE, claim)
    return claim


def score_once() -> dict:
    """Claim the exclusive public gate, then make one official reference read/score."""
    if RESULT.exists() or GATE.exists():
        raise FileExistsError("The public score gate is closed and cannot be reused")
    seal = _json(SEAL)
    _verify_seal(seal)  # reference-free validation precedes atomic gate claim
    claim = _claim_gate(seal)

    # Public reference access is deliberately below the irreversible one-time claim.
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    truth = pm.label_codes(pm.read_label_csv(REFERENCE))
    candidate_path = ROOT / seal["candidate_csv_path"]
    prediction = pm.label_codes(pm.read_label_csv(candidate_path))
    measured = pm.evaluate(truth, prediction, bootstrap_n=1000)
    if (measured["n_frames"], measured["n_graded"], measured["n_gt_spells"]) != (
            EXPECTED_FRAMES, EXPECTED_GRADED, EXPECTED_SPELLS):
        raise ValueError("Public reference population differs from the previously recorded protocol")
    summary = _score_summary(measured)
    prior = _json(BASELINE)
    baseline = prior["candidate"]
    for key, expected in (("spell_f1_mean", PUBLIC_BASELINE_SPELL_F1),
                          ("macro_f1", PUBLIC_BASELINE_MACRO_F1),
                          ("d_f1", PUBLIC_BASELINE_D_F1)):
        if abs(float(baseline[key]) - expected) > 1e-15:
            raise ValueError(f"Saved adopted public baseline {key} differs from the declared value")
    result = {
        "study_id": STUDY,
        "candidate_recipe": seal["selected_recipe"],
        "training_stage": seal["training_stage"],
        "candidate": summary,
        "reused_public_baseline": baseline,
        "delta_vs_saved_baseline": {
            key: summary[key] - baseline[key]
            for key in ("spell_f1_mean", "macro_f1", "d_f1")},
        "graded_frames": EXPECTED_GRADED,
        "reference_spells": EXPECTED_SPELLS,
        "prediction_seal_sha256": sha(SEAL),
        "public_gate_claim_sha256": sha(GATE),
        "saved_public_baseline_sha256": seal["saved_public_baseline_sha256"],
        "organizer_reference_labels_read": True,
        "public_used_for_selection": False,
        "independent_confirmation": False,
        "promotion": False,
    }
    _exclusive_json(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "seal", "score"))
    parser.add_argument("--csv", type=Path, default=CSV)
    parser.add_argument("--model", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--source", action="append", default=[], type=Path)
    args = parser.parse_args()
    if args.action == "predict":
        print(json.dumps(predict_and_seal(), indent=2))
    elif args.action == "seal":
        model_files = {}
        for item in args.model:
            name, sep, raw_path = item.partition("=")
            if not sep or not name or not raw_path or name in model_files:
                parser.error("--model must be unique NAME=PATH entries")
            model_files[name] = Path(raw_path)
        seal_prediction(csv_path=args.csv, model_files=model_files,
                        source_files=tuple(args.source) + (Path(__file__).resolve(),))
    else:
        print(json.dumps(score_once(), indent=2))


if __name__ == "__main__":
    main()
