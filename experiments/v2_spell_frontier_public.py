"""Reference-free public prediction and one-time frontier-study score gate.

The previous balanced study remains sealed and unscored. This module writes
only v2_spell_frontier_v1 artifacts and binds the final frontier selection,
pooled TRAIN models, public input, and adopted-baseline replay.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

from experiments import v2_spell_balanced_public as balanced_public

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_spell_frontier_v1"
SELECTION = ROOT / f"experiments/results/{STUDY}_selection.json"
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
INPUT = ROOT / "data/challenge/inputs.npz"
REFERENCE = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
BASELINE = ROOT / "experiments/results/feed_like_v2_public_result.json"
BALANCED_CSV = ROOT / "data/experiments/v2_spell_balanced_v1/public_candidate.csv"
BALANCED_SEAL = ROOT / "data/experiments/v2_spell_balanced_v1/public_prediction_seal.json"
BALANCED_GATE = ROOT / "data/experiments/v2_spell_balanced_v1/public_gate_claimed.json"
BALANCED_RESULT = ROOT / "experiments/results/v2_spell_balanced_v1_public_result.json"
OUT = ROOT / "data/experiments" / STUDY
CSV = OUT / "public_candidate.csv"
SEAL = OUT / "public_prediction_seal.json"
GATE = OUT / "public_gate_claimed.json"
RESULT = ROOT / f"experiments/results/{STUDY}_public_result.json"
EXPECTED_FRAMES = balanced_public.EXPECTED_FRAMES
EXPECTED_GRADED = balanced_public.EXPECTED_GRADED
EXPECTED_SPELLS = balanced_public.EXPECTED_SPELLS
PUBLIC_BASELINE_SPELL_F1 = balanced_public.PUBLIC_BASELINE_SPELL_F1
PUBLIC_BASELINE_MACRO_F1 = balanced_public.PUBLIC_BASELINE_MACRO_F1
PUBLIC_BASELINE_D_F1 = balanced_public.PUBLIC_BASELINE_D_F1
OLD_CSV_SHA256 = "b9ab4216485e986417aaad7dd1c2dc103c86ee0682e209ad26bd6429a8ae8509"
OLD_SEAL_SHA256 = "b49400f38f9f5e4969013bdc44e8207ff7a8fd1ef7898c3fab6ec27a83f2f1bd"


def sha(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
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
        raise ValueError("Selection is not a frozen final frontier selection")
    if not selection.get("public_input_sha256"):
        raise ValueError("Final selection does not bind the public tracking input")
    recipe = selection.get("recipe")
    if not isinstance(recipe, dict):
        raise ValueError("Final selection has no mixture recipe")
    for name in ("ce_weight", "calibrated_weight", "adopted_weight"):
        value = recipe.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"Invalid frozen recipe weight: {name}")
    if abs(sum(float(recipe[name]) for name in
               ("ce_weight", "calibrated_weight", "adopted_weight")) - 1.0) > 1e-12:
        raise ValueError("Frozen mixture weights must sum to one")
    if recipe.get("pooling") not in ("arithmetic", "log"):
        raise ValueError("Unsupported frozen pooling rule")
    if recipe.get("ha_cost") not in (3, 3.0):
        raise ValueError("Frontier selection must preserve fixed H/A cost 3")
    arm = selection.get("training_arm")
    stage = selection.get("training_stage")
    if (arm, stage) not in (("base", 1024), ("repeat", 1536), ("fresh", 1536)):
        raise ValueError("Unsupported frozen training arm/stage")
    expected_id = recipe.get("recipe_id", recipe.get("id"))
    if not isinstance(expected_id, str) or not expected_id:
        raise ValueError("Frozen recipe needs a stable recipe_id")
    return recipe


def _expected_model_roles() -> set[str]:
    return {f"{role}/{seed}" for role in
            ("ce", "adopted", "calibrated", "teacher_adopted", "teacher_calibrated")
            for seed in (2026, 2027)}


def _prior_public_facts() -> dict:
    """Check the old candidate/seal bindings without touching reference labels."""
    plan = _json(PLAN)
    previous = plan["previous_study"]
    if previous.get("old_public_gate_claimed") is not False or previous.get("old_public_score_exists") is not False:
        raise ValueError("Previous public gate is already claimed; preserve its one-score limit")
    if not BALANCED_CSV.is_file() or sha(BALANCED_CSV) != previous.get("old_reference_free_csv_sha256", OLD_CSV_SHA256):
        raise ValueError("Previously sealed balanced CSV changed")
    if not BALANCED_SEAL.is_file() or sha(BALANCED_SEAL) != previous.get("old_prediction_seal_sha256", OLD_SEAL_SHA256):
        raise ValueError("Previously sealed balanced prediction seal changed")
    if sha(BALANCED_CSV) != OLD_CSV_SHA256 or sha(BALANCED_SEAL) != OLD_SEAL_SHA256:
        raise ValueError("Previous balanced artifacts differ from their declared immutable hashes")
    if BALANCED_GATE.exists() or BALANCED_RESULT.exists():
        raise FileExistsError("The balanced public study has already been scored")
    return {"csv_sha256": sha(BALANCED_CSV), "seal_sha256": sha(BALANCED_SEAL)}


def _recipe_matches_balanced(recipe: dict) -> bool:
    old = _json(BALANCED_SEAL)["selected_recipe"]
    keys = ("ce_weight", "calibrated_weight", "adopted_weight", "pooling", "ha_cost")
    return all(float(recipe[k]) == float(old[k]) if k.endswith("weight") or k == "ha_cost"
                   else recipe[k] == old[k] for k in keys)


def _copy_balanced_prediction() -> None:
    """Reuse old bytes only when the final frontier pipeline is identical."""
    if CSV.exists():
        raise FileExistsError(CSV)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with BALANCED_CSV.open("rb") as source, CSV.open("xb") as target:
        shutil.copyfileobj(source, target)


def _model_inventory(selection: dict) -> tuple[dict[str, Path], Path, dict]:
    import joblib
    from experiments import v2_spell_balanced_train as base_training
    from experiments import v2_spell_frontier_train as frontier_training

    arm, stage = selection["training_arm"], int(selection["training_stage"])
    original_manifest = _json(ROOT / "experiments/baseline_manifest.json")
    calibrated_manifest = _json(ROOT / "experiments/baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json")
    original_dir = ROOT / original_manifest["model_directory"]
    calibrated_dir = ROOT / calibrated_manifest["model_directory"]
    if arm == "base":
        base_training.check_pooled(full=True)
        pooled_seal_path = base_training.POOL_SEAL
        pooled_seal = _json(pooled_seal_path)
        if pooled_seal.get("study_id") != base_training.STUDY or pooled_seal.get("all_authorized_stages_complete") is not True:
            raise ValueError("Balanced stage-1024 pool is not sealed")
        ce_paths = {seed: base_training._pooled_output_path(seed, stage) for seed in (2026, 2027)}
        fits = {(int(row["seed"]), int(row["stage"])): row for row in pooled_seal["fits"]}
        for seed, path in ce_paths.items():
            row = fits.get((seed, stage))
            if row is None or row.get("sha256") != sha(path):
                raise ValueError(f"Balanced pooled fit seal does not bind {seed}/{stage}")
    else:
        frontier_training.check_pooled(full=True)
        pooled_seal_path = frontier_training.POOL_SEAL
        if not pooled_seal_path.is_file():
            raise FileNotFoundError(pooled_seal_path)
        pooled_seal = _json(pooled_seal_path)
        pool_declaration = _json(frontier_training.POOL_DECL)
        ce_paths = {seed: frontier_training.selected_pooled_model_path(selection, seed)
                    for seed in (2026, 2027)}
        _validate_frontier_pool_seal(selection, pooled_seal, pool_declaration,
                                     sha(frontier_training.POOL_DECL), ce_paths)
    paths = {}
    for seed in (2026, 2027):
        adopted = original_dir / f"dense_{seed}.joblib"
        calibrated = calibrated_dir / f"dense_{seed}.joblib"
        teacher_adopted = original_dir / f"teacher_{seed}.joblib"
        teacher_calibrated = calibrated_dir / f"teacher_{seed}.joblib"
        for path, manifest in ((adopted, original_manifest), (teacher_adopted, original_manifest),
                               (calibrated, calibrated_manifest), (teacher_calibrated, calibrated_manifest)):
            if sha(path) != manifest["models_sha256"][path.name]:
                raise ValueError(f"Baseline model hash mismatch: {path}")
        paths.update({f"ce/{seed}": ce_paths[seed], f"adopted/{seed}": adopted,
                      f"calibrated/{seed}": calibrated, f"teacher_adopted/{seed}": teacher_adopted,
                      f"teacher_calibrated/{seed}": teacher_calibrated})
    return paths, pooled_seal_path, pooled_seal


def _validate_frontier_pool_seal(selection: dict, pooled_seal: dict,
                                 pool_declaration: dict, declaration_sha256: str,
                                 ce_paths: dict[int, Path]) -> None:
    if (pooled_seal.get("study_id") != STUDY or
            pooled_seal.get("all_selected_pooled_models_ready") is not True or
            pooled_seal.get("training_arm") != selection.get("training_arm") or
            int(pooled_seal.get("training_stage", -1)) != int(selection.get("training_stage", -2)) or
            pooled_seal.get("pooled_declaration_sha256") != declaration_sha256 or
            pool_declaration.get("selection_sha256") != sha(SELECTION)):
        raise ValueError("Frontier pooled model seal does not bind the final selected arm/stage")
    if (pool_declaration.get("study_id") != STUDY or
            pool_declaration.get("training_arm") != selection.get("training_arm") or
            int(pool_declaration.get("training_stage", -1)) != int(selection.get("training_stage", -2))):
        raise ValueError("Frontier pooled declaration differs from final selection")
    models = pooled_seal.get("models")
    if not isinstance(models, dict) or set(models) != {"2026", "2027"}:
        raise ValueError("Pooled seal must list exactly the two deployed seed models")
    for seed, path in ce_paths.items():
        row = models[str(seed)]
        if (row.get("path") != str(path.relative_to(ROOT)) or
                row.get("sha256") != sha(path)):
            raise ValueError(f"Pooled seal does not bind selected CE model: {seed}")
    fits = pooled_seal.get("fits")
    if selection["training_arm"] == "base":
        if fits != []:
            raise ValueError("Base pooled seal must not claim new fits")
    else:
        if not isinstance(fits, list) or len(fits) != 2:
            raise ValueError("Repeat/fresh pooled seal must bind exactly two selected-seed fits")
        by_seed = {int(row.get("seed", -1)): row for row in fits}
        if set(by_seed) != {2026, 2027}:
            raise ValueError("Pooled fit receipt seed inventory is incomplete")
        for seed, path in ce_paths.items():
            row = by_seed[seed]
            model_fact = models[str(seed)]
            if (row.get("study_id") != STUDY or row.get("arm") != selection["training_arm"] or
                    int(row.get("stage", -1)) != 1536 or row.get("path") != model_fact["path"] or
                    row.get("sha256") != model_fact["sha256"] or
                    row.get("declaration_sha256") != declaration_sha256 or
                    row.get("d_state_unchanged") is not True):
                raise ValueError(f"Pooled fit receipt mismatch: {seed}")


def _phase_probabilities(selection: dict) -> tuple[dict, dict[str, Path], Path, dict]:
    """Build aligned pooled CE/adopted/calibrated H/A/D probabilities."""
    import joblib
    import numpy as np
    from experiments import current_baseline as baseline
    from experiments import feed_benchmark_audit_v1 as audit
    from experiments import live_gap_experiment, sn_dense_baseline
    from experiments import sn_dense_train as sn
    from experiments import soccernet_extension as old
    from experiments import tracking_corruption, v2_iterative_runtime

    paths, pooled_seal_path, pooled_seal = _model_inventory(selection)
    models = {name: {} for name in ("ce", "adopted", "calibrated")}
    for seed in (2026, 2027):
        models["ce"][seed] = (joblib.load(paths[f"ce/{seed}"]), joblib.load(paths[f"teacher_adopted/{seed}"]))
        models["adopted"][seed] = (joblib.load(paths[f"adopted/{seed}"]), joblib.load(paths[f"teacher_adopted/{seed}"]))
        models["calibrated"][seed] = (joblib.load(paths[f"calibrated/{seed}"]), joblib.load(paths[f"teacher_calibrated/{seed}"]))
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
    if len(records) not in (2, 5):
        raise ValueError(f"Expected two or five native phases, got {len(records)}")
    by_expert = {name: [] for name in models}
    for phase in records:
        record = dict(phase, names=list(phase["names"]))
        for name, seed_models in models.items():
            paired = []
            for seed in (2026, 2027):
                model, teacher = seed_models[seed]
                sampled = sn.predict(record, model, teacher)
                paired.append(old.upsample(sampled, record, clean))
            by_expert[name].append(paired)
    arrays = {name: np.transpose(np.asarray(phase_rows, dtype=np.float64), (1, 0, 2, 3))
              for name, phase_rows in by_expert.items()}
    components = {name: balanced_public._pool_seed_phase_probabilities(array.tolist())
                  for name, array in arrays.items()}
    return components, paths, pooled_seal_path, pooled_seal


def _render_csv(selection: dict, recipe: dict, components: dict) -> dict:
    import numpy as np
    from experiments import current_baseline as baseline
    from experiments import feed_benchmark_audit_v1 as audit
    from experiments import v2_spell_balanced_eval as evaluator
    import experiments.supervised as supervised

    inputs = baseline.load_inputs(INPUT)
    period = supervised.model_periods(inputs)
    adopted_decoded = audit.decode(components["adopted"], period, float(inputs["fps"]), audit.CONFIG)
    probability = evaluator.recipe_probability(components, float(recipe["ce_weight"]),
        float(recipe["calibrated_weight"]), recipe["pooling"])
    prediction = np.asarray(evaluator.decode_fixed_d(probability, adopted_decoded, period,
        float(inputs["fps"]), float(recipe["ha_cost"]))).astype("U1", copy=True)
    invalid = np.asarray(inputs["period"]) <= 0
    prediction[invalid] = "X"
    tracked = ~invalid
    if not np.isin(prediction[tracked], ("H", "A", "D")).all():
        raise ValueError("Tracked public frames must be H/A/D")
    if not np.array_equal(prediction[tracked] == "D", adopted_decoded[tracked] == "D"):
        raise ValueError("Frontier recipe changed the fixed adopted D mask")

    adopted_prediction = np.asarray(adopted_decoded).astype("U1", copy=True)
    adopted_prediction[invalid] = "X"
    adopted_bytes = ("frame,label\n" + "".join(f"{i},{label}\n" for i, label in enumerate(adopted_prediction))).encode()
    adopted_hash = hashlib.sha256(adopted_bytes).hexdigest()
    baseline_hash = _json(BASELINE)["candidate_csv_sha256"]
    if adopted_hash != baseline_hash:
        raise ValueError("Adopted full public render does not replay saved baseline prediction hash")
    if CSV.exists():
        raise FileExistsError(CSV)
    CSV.parent.mkdir(parents=True, exist_ok=True)
    with CSV.open("x", encoding="utf-8") as stream:
        stream.write("frame,label\n")
        for frame, label in enumerate(prediction):
            stream.write(f"{frame},{label}\n")
    return {"replay_checked": True, "saved_public_prediction_sha256": baseline_hash,
            "adopted_prediction_sha256": adopted_hash, "available": False,
            "default_output_path": "data/experiments/current_baseline/joint_gru_sn_dense_original_diversity_phase_ensemble.csv"}


def _copy_or_render(selection: dict, recipe: dict) -> tuple[dict[str, Path], Path, dict, dict]:
    prior = _prior_public_facts()
    if selection["training_arm"] == "base" and int(selection["training_stage"]) == 1024 and _recipe_matches_balanced(recipe):
        balanced_public._verify_seal(_json(BALANCED_SEAL))
        _copy_balanced_prediction()
        paths, pooled_seal_path, pooled_seal = _model_inventory(selection)
        old_model_hashes = _json(BALANCED_SEAL)["model_sha256"]
        if {str(p): sha(p) for p in paths.values()} != old_model_hashes:
            raise ValueError("Base pooled model inventory differs from the sealed balanced pipeline")
        adopted_replay = _json(BALANCED_SEAL)["adopted_baseline_replay"]
        return paths, pooled_seal_path, pooled_seal, {**adopted_replay, "reused_balanced_prediction": True,
                                                        "previous_csv_sha256": prior["csv_sha256"]}
    components, paths, pooled_seal_path, pooled_seal = _phase_probabilities(selection)
    adopted_replay = _render_csv(selection, recipe, components)
    return paths, pooled_seal_path, pooled_seal, adopted_replay


def _expected_source_paths() -> set[str]:
    from experiments import v2_spell_frontier_eval as evaluator
    from experiments import v2_spell_frontier_train as training
    expected = balanced_public._expected_public_source_paths()
    expected.update({str(Path(p).resolve().relative_to(ROOT)) for p in
                     (Path(__file__), Path(balanced_public.__file__), Path(evaluator.__file__),
                      Path(training.__file__), ROOT / "scripts/verify_v2_spell_frontier.py", BASELINE)})
    return expected


def _validate_evidence(selection: dict) -> dict:
    bindings = selection.get("development_result_bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Final selection must bind its frontier development evidence")
    hashed = {}
    for raw, expected in bindings.items():
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Frontier development binding changed: {path}")
        hashed[str(path.resolve())] = expected
    return hashed


def predict_and_seal() -> dict:
    if SEAL.exists() or GATE.exists() or RESULT.exists() or CSV.exists():
        raise FileExistsError("Frontier public artifacts already exist")
    selection = _json(SELECTION)
    recipe = _selection_contract(selection)
    balanced_public._verify_public_baseline()
    if sha(INPUT) != selection["public_input_sha256"]:
        raise ValueError("Public tracking input differs from frozen frontier selection")
    plan = _json(PLAN)
    if plan.get("public_validation", {}).get("authorized") is not True:
        raise ValueError("Frontier plan does not authorize the final diagnostic")
    _prior_public_facts()
    binding_hashes = _validate_evidence(selection)
    paths, pooled_seal_path, pooled_seal, adopted_replay = _copy_or_render(selection, recipe)
    balanced_public._csv_valid(CSV)
    if set(paths) != _expected_model_roles():
        raise ValueError("Incomplete frontier model-role inventory")
    source_paths = _expected_source_paths()
    source_files = {name: ROOT / name for name in source_paths}
    if any(not p.is_file() for p in source_files.values()):
        raise FileNotFoundError("Frontier source inventory has missing files")
    model_hashes = {str(path.resolve()): sha(path) for path in paths.values()}
    value = {
        "study_id": STUDY, "plan_sha256": sha(PLAN),
        "selection_path": str(SELECTION.relative_to(ROOT)), "selection_sha256": sha(SELECTION),
        "selected_recipe": recipe, "training_arm": selection["training_arm"],
        "training_stage": selection["training_stage"],
        "development_result_bindings": selection["development_result_bindings"],
        "development_binding_sha256": binding_hashes,
        "candidate_csv_path": str(CSV.relative_to(ROOT)), "candidate_csv_sha256": sha(CSV),
        "input_path": str(INPUT.relative_to(ROOT)), "input_sha256": sha(INPUT),
        "pooled_seal_path": str(pooled_seal_path.relative_to(ROOT)),
        "pooled_seal_sha256": sha(pooled_seal_path),
        "model_role_paths": {role: str(path.resolve()) for role, path in paths.items()},
        "model_sha256": model_hashes,
        "source_sha256": {name: sha(path) for name, path in sorted(source_files.items())},
        "adopted_baseline_replay": adopted_replay,
        "saved_public_baseline_sha256": sha(BASELINE), "rows": EXPECTED_FRAMES,
        "format_valid": True, "reference_read_or_hashed": False,
        "public_used_for_selection": False, "promotion": False,
        "reused_balanced_prediction": bool(adopted_replay.get("reused_balanced_prediction", False)),
    }
    _exclusive_json(SEAL, value)
    return value


def _verify_seal(seal: dict) -> None:
    if seal.get("study_id") != STUDY or seal.get("reference_read_or_hashed") is not False:
        raise ValueError("Frontier prediction seal has invalid identity")
    selection = _json(SELECTION)
    recipe = _selection_contract(selection)
    if seal.get("selection_sha256") != sha(SELECTION) or seal.get("plan_sha256") != sha(PLAN):
        raise ValueError("Frontier selection or plan changed after sealing")
    if seal.get("selected_recipe") != recipe or seal.get("training_arm") != selection.get("training_arm") or seal.get("training_stage") != selection.get("training_stage"):
        raise ValueError("Frontier selected pipeline differs from prediction seal")
    if sha(INPUT) != selection["public_input_sha256"] or seal.get("input_sha256") != sha(INPUT):
        raise ValueError("Frontier public tracking input changed after sealing")
    if seal.get("saved_public_baseline_sha256") != sha(BASELINE):
        raise ValueError("Saved adopted public summary changed after sealing")
    balanced_public._verify_public_baseline()
    prior = _prior_public_facts()
    if seal.get("reused_balanced_prediction") and seal.get("candidate_csv_sha256") != prior["csv_sha256"]:
        raise ValueError("Reused candidate no longer matches the sealed balanced CSV")
    replay = seal.get("adopted_baseline_replay", {})
    expected_baseline_prediction = _json(BASELINE).get("candidate_csv_sha256")
    if (replay.get("replay_checked") is not True or
            replay.get("saved_public_prediction_sha256") != expected_baseline_prediction or
            replay.get("adopted_prediction_sha256") != expected_baseline_prediction):
        raise ValueError("Adopted prediction baseline replay is absent or mismatched")
    if seal.get("development_result_bindings") != selection.get("development_result_bindings"):
        raise ValueError("Frontier development bindings changed")
    for raw, expected in seal.get("development_binding_sha256", {}).items():
        if not Path(raw).is_file() or sha(Path(raw)) != expected:
            raise ValueError(f"Frontier development evidence changed: {raw}")
    if set(seal.get("model_role_paths", {})) != _expected_model_roles():
        raise ValueError("Frontier model role inventory is incomplete")
    for raw, expected in seal.get("model_sha256", {}).items():
        if sha(Path(raw)) != expected:
            raise ValueError(f"Frontier model changed after sealing: {raw}")
    path_map = {str(Path(path).resolve()): sha(Path(path)) for path in seal["model_role_paths"].values()}
    if path_map != seal.get("model_sha256"):
        raise ValueError("Frontier role/path model hashes are inconsistent")
    for name, expected in seal.get("source_sha256", {}).items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frontier source changed after sealing: {name}")
    if set(seal.get("source_sha256", {})) != _expected_source_paths():
        raise ValueError("Frontier public source inventory is incomplete")
    csv_path = ROOT / seal["candidate_csv_path"]
    if sha(csv_path) != seal.get("candidate_csv_sha256"):
        raise ValueError("Frontier candidate CSV changed after sealing")
    balanced_public._csv_valid(csv_path)
    if sha(ROOT / seal["pooled_seal_path"]) != seal.get("pooled_seal_sha256"):
        raise ValueError("Frontier pooled seal changed after prediction sealing")


def _claim_gate(seal: dict) -> dict:
    value = {"study_id": STUDY, "claimed_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_seal_sha256": sha(SEAL), "candidate_csv_sha256": seal["candidate_csv_sha256"],
        "selection_sha256": seal["selection_sha256"], "one_time_public_score": True,
        "no_post_public_tuning_or_reselection": True}
    _exclusive_json(GATE, value)
    return value


def score_once() -> dict:
    """One-time final diagnostic. This function must run only after all development is complete."""
    if BALANCED_GATE.exists() or BALANCED_RESULT.exists():
        raise FileExistsError("The prior balanced gate/result must remain unused")
    if GATE.exists() or RESULT.exists():
        raise FileExistsError("The frontier public score gate is closed")
    seal = _json(SEAL)
    _verify_seal(seal)
    _claim_gate(seal)
    # The sole public reference access in this study is below the exclusive claim.
    from evaluation import possession_metrics as pm
    from experiments.combined_objective_public_check import _score_summary
    truth = pm.label_codes(pm.read_label_csv(REFERENCE))
    prediction = pm.label_codes(pm.read_label_csv(ROOT / seal["candidate_csv_path"]))
    measured = pm.evaluate(truth, prediction, bootstrap_n=1000)
    if (measured["n_frames"], measured["n_graded"], measured["n_gt_spells"]) != (EXPECTED_FRAMES, EXPECTED_GRADED, EXPECTED_SPELLS):
        raise ValueError("Public reference population differs from the frozen protocol")
    summary = _score_summary(measured)
    baseline = _json(BASELINE)["candidate"]
    for key, expected in (("spell_f1_mean", PUBLIC_BASELINE_SPELL_F1),
                          ("macro_f1", PUBLIC_BASELINE_MACRO_F1), ("d_f1", PUBLIC_BASELINE_D_F1)):
        if abs(float(baseline[key]) - expected) > 1e-15:
            raise ValueError(f"Saved public baseline {key} differs from its frozen value")
    result = {"study_id": STUDY, "candidate_recipe": seal["selected_recipe"],
        "training_arm": seal["training_arm"], "training_stage": seal["training_stage"],
        "candidate": summary, "reused_public_baseline": baseline,
        "delta_vs_saved_baseline": {k: summary[k] - baseline[k]
            for k in ("spell_f1_mean", "macro_f1", "d_f1")},
        "graded_frames": EXPECTED_GRADED, "reference_spells": EXPECTED_SPELLS,
        "prediction_seal_sha256": sha(SEAL), "public_gate_claim_sha256": sha(GATE),
        "saved_public_baseline_sha256": seal["saved_public_baseline_sha256"],
        "organizer_reference_labels_read": True, "public_used_for_selection": False,
        "independent_confirmation": False, "promotion": False}
    _exclusive_json(RESULT, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("predict", "score"))
    args = parser.parse_args()
    if args.action == "predict":
        print(json.dumps(predict_and_seal(), indent=2))
    else:
        print(json.dumps(score_once(), indent=2))


if __name__ == "__main__":
    main()
