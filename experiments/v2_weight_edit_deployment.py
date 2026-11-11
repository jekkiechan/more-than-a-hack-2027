"""Two-fit pooled deployment reconstruction and inference for v2 weight edit.

The final deployment package contains two pooled four-view fits edited with a
frozen selected edit (control, projection direction, or update-gate bias),
paired with the archived calibrated-D members and teachers. The selected edit
strength/decoder are supplied by one frozen development-choice file. This
module has no scoring action.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MTAH_DATA_DIR", str(ROOT.parent / "mtah-data"))
os.environ.setdefault("SOCCERNET_GAR_TRAIN_RECORDS",
                      str(ROOT / "data/experiments/v2_weight_edit_reconstruction_v1/gar_records_archived_pr8"))
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import joblib
import numpy as np

from experiments import current_baseline as baseline
from experiments import feed_benchmark_audit_v1 as audit
from experiments import live_gap_experiment
from experiments import soccernet_extension as old
from experiments import sn_dense_train as sn
from experiments import v2_weight_edit_reconstruction as reconstruction
from experiments import weight_space_edit as weight_edit
from experiments import feed_spell_push
from experiments import supervised, tracking_corruption

STUDY = "v2_weight_edit_pooled_deployment_v1"
OUT = ROOT / "data/experiments" / STUDY
MODELS = OUT / "models"
DECLARATION = ROOT / "experiments/results" / f"{STUDY}_declaration.json"
FIT_RECEIPT = ROOT / "experiments/results" / f"{STUDY}_fit_receipt.json"
DEPLOYMENT_RECEIPT = OUT / "deployment.json"
CHOICE_DEFAULT = ROOT / "data/experiments/v2_weight_edit_selected_choice.json"
CALIBRATED_MANIFEST = ROOT / "experiments/baselines/joint_gru_sn_dense_calibrated_ensemble_v1.json"
SEEDS = (2026, 2027)


class DeploymentError(RuntimeError):
    pass


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _tree_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return (set(left) == set(right)
                and all(_tree_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            return False
    return left == right


def _source_map() -> dict[str, str]:
    from experiments import (calibrated_dense_baseline, decoder_refinement, decoder_search,
                             factorized_had, learned_temporal, live_gap_experiment,
                             refine_domain, refine_features, soccernet_capacity_model,
                             soccernet_data, soccernet_segments, sn_dense_baseline)
    sources = (Path(__file__), Path(reconstruction.__file__), Path(weight_edit.__file__),
               Path(baseline.__file__), Path(audit.__file__), Path(old.__file__), Path(sn.__file__),
               Path(feed_spell_push.__file__), Path(supervised.__file__),
               Path(tracking_corruption.__file__), Path(calibrated_dense_baseline.__file__),
               Path(decoder_refinement.__file__), Path(decoder_search.__file__),
               Path(factorized_had.__file__), Path(learned_temporal.__file__),
               Path(live_gap_experiment.__file__), Path(refine_domain.__file__),
               Path(refine_features.__file__), Path(soccernet_capacity_model.__file__),
               Path(soccernet_data.__file__), Path(soccernet_segments.__file__),
               Path(sn_dense_baseline.__file__), ROOT / "evaluation/possession_metrics.py")
    return {str(path.relative_to(ROOT)): sha(path) for path in sources}


def _calibrated_files() -> dict[str, str]:
    manifest = read(CALIBRATED_MANIFEST)
    directory = ROOT / manifest["model_directory"]
    result = {str((directory / name).relative_to(ROOT)): digest
              for name, digest in manifest["models_sha256"].items()}
    for rel, expected in result.items():
        if sha(ROOT / rel) != expected:
            raise DeploymentError(f"Calibrated pooled artifact missing or changed: {rel}")
    return result


def _teacher_paths() -> dict[int, Path]:
    recipe = reconstruction.recipe
    return {seed: recipe.rf.sc.initial_path("pooled", seed) for seed in SEEDS}


def _normalize_choice(choice: dict) -> tuple[dict, float, float, float]:
    """Translate the actual development freeze schema into deployment settings."""
    if not isinstance(choice, dict):
        raise DeploymentError("Selected choice must be a JSON object")
    if not (choice.get("study") or choice.get("study_id")):
        raise DeploymentError("Selected choice must identify its development study")
    if not choice.get("development_report_path") or not choice.get("development_report_sha256"):
        raise DeploymentError("Selected choice must bind its development report")
    if not choice.get("winner_id"):
        raise DeploymentError("Selected choice must identify the frozen winner")
    edit = choice.get("edit")
    if isinstance(edit, (list, tuple)) and len(edit) == 2:
        edit = {"kind": edit[0], "strength": edit[1]}
    if not isinstance(edit, dict):
        raise DeploymentError("Selected edit must be a mapping or a [kind, strength] pair")
    aliases = {"none": "control", "identity": "control", "control": "control",
               "projection": "ha_projection", "ha_projection": "ha_projection",
               "update_bias": "gate_bias", "gate_bias": "gate_bias"}
    if edit.get("kind") not in aliases:
        raise DeploymentError("Selected edit kind must be none, projection, or update_bias")
    raw_kind = edit["kind"]
    edit = dict(edit, kind=aliases[raw_kind])
    strength_raw = edit.get("strength")
    if strength_raw is None:
        strength_raw = edit.get("alpha") if edit["kind"] == "ha_projection" else edit.get("delta")
    if strength_raw is None:
        strength_raw = choice.get("strength")
    strength = float(0.0 if strength_raw is None and edit["kind"] == "control" else strength_raw)
    alpha_raw = edit.get("alpha") if edit.get("alpha") is not None else choice.get("alpha")
    delta_raw = edit.get("delta") if edit.get("delta") is not None else choice.get("delta")
    alpha = float((strength if edit["kind"] == "ha_projection" else 0.0)
                  if alpha_raw is None else alpha_raw)
    gate_delta = float((strength if edit["kind"] == "gate_bias" else 0.0)
                       if delta_raw is None else delta_raw)
    if not np.isfinite(strength) or not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise DeploymentError("Selected edit strength/alpha must be finite and alpha in [0,1]")
    if not np.isfinite(gate_delta):
        raise DeploymentError("Selected update-gate delta must be finite")
    if edit["kind"] == "control" and (alpha != 0 or gate_delta != 0):
        raise DeploymentError("Control edit must use alpha=0 and gate delta=0")
    if edit["kind"] == "ha_projection" and (gate_delta != 0 or alpha == 0):
        raise DeploymentError("Projection edit must have alpha >0 and no update-gate delta")
    if edit["kind"] == "gate_bias" and (alpha != 0 or gate_delta == 0):
        raise DeploymentError("Update-gate edit must have nonzero delta and no projection alpha")
    if edit["kind"] == "ha_projection":
        direction_path = edit.get("direction_path")
        direction_hash = edit.get("direction_sha256")
        if not direction_path or not direction_hash:
            raise DeploymentError("Projection choice must bind its train-only direction artifact")
        path = Path(direction_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != direction_hash:
            raise DeploymentError("Selected projection direction path/hash does not verify")
    return edit, strength, alpha, gate_delta


def freeze(choice_path: Path = CHOICE_DEFAULT) -> dict:
    """Freeze the root-selected dev winner before any pooled refit."""
    if DECLARATION.exists() or FIT_RECEIPT.exists() or DEPLOYMENT_RECEIPT.exists():
        raise DeploymentError("Pooled deployment ID already exists; preserve its artifacts")
    if not choice_path.is_file():
        raise DeploymentError(f"Waiting for a frozen dev winner choice: {choice_path}")
    reconstruction.check()
    choice = read(choice_path)
    required = ("development_report_path", "development_report_sha256", "winner_id",
                "edit", "decoder_config")
    missing = [key for key in required if key not in choice]
    if missing:
        raise DeploymentError(f"Selected choice lacks required fields: {missing}")
    report_path = ROOT / choice["development_report_path"]
    if not report_path.is_file() or sha(report_path) != choice["development_report_sha256"]:
        raise DeploymentError("Frozen development report path/hash does not verify")
    edit, strength, alpha, gate_delta = _normalize_choice(choice)
    config = choice["decoder_config"]
    if not isinstance(config, dict) or config.get("family") not in {"calibration", "reference", "pool", "filter", "dead_bias", "disagreement", "entropy", "anchors"}:
        raise DeploymentError("Selected decoder config is not a feed_spell_push configuration")
    if config.get("ha") != 1.5:
        raise DeploymentError("Deployment decoder must retain the selected H/A transition cost 1.5")
    smoke_prob = np.tile(np.asarray([[.47, .48, .05]], dtype=np.float64), (12, 1))
    smoke = feed_spell_push.decode(smoke_prob[None, ...], np.ones(12, dtype=np.int32), 25.0, config)
    if np.asarray(smoke).shape != (12,) or not set(np.asarray(smoke).tolist()) <= {"H", "A", "D", "X"}:
        raise DeploymentError("Selected feed_spell_push decoder config failed its synthetic contract")
    # The reconstruction declaration binds all original/GAR TRAIN inputs and
    # source code before the pooled fitting step begins.
    recon_decl = read(reconstruction.DECLARATION)
    teachers = _teacher_paths()
    teacher_hashes = {str(seed): sha(path) for seed, path in teachers.items()}
    calibrated_files = _calibrated_files()
    declaration = {
        "study": STUDY, "created_utc": datetime.now(timezone.utc).isoformat(),
        "selected_dev_choice_path": str(choice_path.relative_to(ROOT)),
        "selected_dev_choice_sha256": sha(choice_path), "selected_dev_choice": choice,
        "development_report_sha256": choice["development_report_sha256"],
        "winner_id": choice["winner_id"], "edit": edit, "alpha": alpha,
        "gate_delta": gate_delta,
        "direction": {
            "method": "weight_space_edit.diagnose_model on clean + fixed first-v2 original-provider TRAIN records, both native and team-swapped orientations; equal-match late/control means",
            "eligible_matches": "The thirteen original-provider TRAIN matches only; omit matches lacking either qualifying activation group and record counts/omissions. No GAR TRAIN, GAR VALID/TEST, or organizer labels enter direction derivation.",
        },
        "pooled_recipe": {
            "reconstruction_declaration_sha256": sha(reconstruction.DECLARATION),
            "reconstruction_fit_receipt_sha256": sha(reconstruction.FIT_RECEIPT),
            "fit_helper": "v2_weight_edit_reconstruction.fit_pooled_member",
            "seeds": list(SEEDS), "fit_count": 2,
            "training_scope": "all thirteen original TRAIN matches plus all 45 GAR TRAIN games; two pooled fits only",
            "provider_fold_average": False,
            "teacher_sha256": teacher_hashes,
        },
        "calibrated_blend": {
            "manifest": str(CALIBRATED_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha(CALIBRATED_MANIFEST), "model_files_sha256": calibrated_files,
            "arithmetic": "For each matched seed and native phase, average edited-new and calibrated-family H/A/D probabilities 50/50; then equal-average the two matched seeds; then equal-average native phases. Apply the one decoder config frozen from dev.",
        },
        "decoder_config": config,
        "source_sha256": _source_map(),
        "organizer_reference_access": False, "public_scoring": False, "evaluation": False,
        "promotion": False,
    }
    write_once(DECLARATION, declaration)
    return declaration


def check() -> dict:
    if not DECLARATION.is_file():
        raise DeploymentError("Freeze the selected development winner before pooled fitting")
    declaration = read(DECLARATION)
    if declaration.get("source_sha256") != _source_map():
        raise DeploymentError("Pooled deployment source changed after declaration")
    for rel, expected in declaration["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise DeploymentError(f"Pooled deployment source changed: {rel}")
    if sha(ROOT / declaration["selected_dev_choice_path"]) != declaration["selected_dev_choice_sha256"]:
        raise DeploymentError("Development winner choice changed")
    if sha(reconstruction.DECLARATION) != declaration["pooled_recipe"]["reconstruction_declaration_sha256"]:
        raise DeploymentError("Reconstruction inputs changed")
    if sha(reconstruction.FIT_RECEIPT) != declaration["pooled_recipe"]["reconstruction_fit_receipt_sha256"]:
        raise DeploymentError("Six-fit reconstruction receipt changed")
    reconstruction.check()
    for seed, path in _teacher_paths().items():
        if sha(path) != declaration["pooled_recipe"]["teacher_sha256"][str(seed)]:
            raise DeploymentError(f"Pooled teacher changed for seed {seed}")
    if sha(CALIBRATED_MANIFEST) != declaration["calibrated_blend"]["manifest_sha256"]:
        raise DeploymentError("Calibrated pooled manifest changed")
    for rel, expected in declaration["calibrated_blend"]["model_files_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise DeploymentError(f"Calibrated pooled artifact changed: {rel}")
    return declaration


def _direction_records(records: list[dict]) -> list[dict]:
    """Select clean and first-v2 direct records; diagnose_model adds both orientations."""
    if len(records) % 4:
        raise DeploymentError("Expected exactly four paired records per TRAIN match")
    selected = []
    for i in range(0, len(records), 4):
        clean, _clean_swap, first_v2, _v2_swap = records[i:i + 4]
        for value, view in ((clean, "clean"), (first_v2, "v2_first")):
            record = dict(value)
            record["view_name"] = view
            selected.append(record)
    matches = {str(record["match"]) for record in selected}
    if len(matches) != 13 or len(selected) != 26:
        raise DeploymentError("Direction source must contain clean and first-v2 records for all thirteen TRAIN matches")
    return selected


def _derive_member_direction(model: dict, teacher: dict, training_records: list[dict]) -> dict:
    return weight_edit.diagnose_model(model, teacher, _direction_records(training_records))


def apply_selected_edit(model: dict, edit: dict, alpha: float, gate_delta: float,
                        direction: dict | None = None) -> tuple[dict, list[str]]:
    """Apply one frozen mechanism and assert its exact tensor scope."""
    kind = edit["kind"]
    edited = deepcopy(model)
    if kind == "control":
        pass
    elif kind == "ha_projection":
        if direction is None:
            raise DeploymentError("Projection mechanism requires its pooled TRAIN direction")
        edited["state"] = weight_edit.edit_conditional_ha_state_projection(
            model["state"], np.asarray(direction["direction"], dtype=np.float32), alpha=float(alpha))
    elif kind == "gate_bias":
        edited["state"] = weight_edit.shift_gru_update_bias(
            model["state"], hidden_size=int(model["hidden"]), delta=float(gate_delta))
    else:
        raise DeploymentError(f"Unsupported selected edit kind: {kind}")
    if set(edited) != set(model) or any(not _tree_equal(edited[key], model[key])
            for key in model if key != "state"):
        raise DeploymentError("Weight edit changed model metadata, normalization or separate D path")
    allowed = set() if kind == "control" else (
        {"output.weight"} if kind == "ha_projection" else
        {"sequence.bias_ih_l0", "sequence.bias_ih_l0_reverse"})
    changed = [key for key in model["state"]
               if not np.array_equal(edited["state"][key], model["state"][key])]
    if set(changed) - allowed or (kind != "control" and not changed):
        raise DeploymentError(f"Unexpected changed tensors for {kind}: {sorted(changed)}")
    if kind == "gate_bias":
        hidden = int(model["hidden"])
        for key in allowed:
            before, after = model["state"][key], edited["state"][key]
            np.testing.assert_array_equal(after[:hidden], before[:hidden])
            np.testing.assert_allclose(after[hidden:2 * hidden] - before[hidden:2 * hidden],
                                       float(gate_delta), atol=1e-7, rtol=0)
            np.testing.assert_array_equal(after[2 * hidden:], before[2 * hidden:])
    if kind == "control" and changed:
        raise DeploymentError("Control must preserve every model tensor exactly")
    return edited, sorted(changed)


def _seal_deployment_package(fit_receipt: dict, declaration: dict) -> dict:
    if fit_receipt.get("declaration_sha256") != sha(DECLARATION):
        raise DeploymentError("Pooled fit receipt does not bind the selected deployment declaration")
    for rel, expected in fit_receipt["model_files_sha256"].items():
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != expected:
            raise DeploymentError(f"Pooled fit artifact changed before package seal: {path}")
    files_by_seed = fit_receipt["deployment_files"]
    deployment_receipt = {
        "study": STUDY, "declaration_path": str(DECLARATION.relative_to(ROOT)),
        "declaration_sha256": sha(DECLARATION), "fit_receipt_path": str(FIT_RECEIPT.relative_to(ROOT)),
        "fit_receipt_sha256": sha(FIT_RECEIPT), "adapter_module": "experiments.v2_weight_edit_deployment",
        "adapter_source_sha256": sha(Path(__file__)), "model_files_sha256": fit_receipt["model_files_sha256"],
        "training_game_ids": fit_receipt["training_game_ids"], "winner_id": declaration["winner_id"],
        "edit": declaration["edit"], "alpha": declaration["alpha"], "gate_delta": declaration["gate_delta"],
        "blend_weight_new_family": .5, "decoder_config": declaration["decoder_config"],
        "decoder_description": "Selected frozen feed_spell_push config with H/A cost 1.5; same probability pooling and decoder for GAR and public diagnostics.",
        "new_member_paths": files_by_seed["new_models"],
        "new_teacher_paths": files_by_seed["new_teachers"],
        "calibrated_manifest_path": files_by_seed["calibrated_manifest"],
        "source_sha256": _source_map(), "organizer_reference_access": False,
        "public_scoring": False, "evaluation": False, "promotion": False,
    }
    if not DEPLOYMENT_RECEIPT.exists():
        write_once(DEPLOYMENT_RECEIPT, deployment_receipt)
    elif read(DEPLOYMENT_RECEIPT) != deployment_receipt:
        raise DeploymentError("Existing deployment package receipt differs; preserve it")
    return deployment_receipt


def _public_member_fact(member: dict) -> dict:
    """Keep reconstructive direction vectors only in ignored per-member receipts."""
    private_path = MODELS / f"pooled_{member['seed']}.json"
    if not private_path.is_file():
        raise DeploymentError(f"Missing private pooled member receipt: {private_path}")
    safe_fields = (
        "seed", "unedited_model_path", "unedited_model_sha256", "edited_model_path",
        "edited_model_sha256", "pooled_teacher_path", "pooled_teacher_sha256",
        "changed_state_tensors", "direction_sha256", "direction_contributing_matches",
        "direction_contributing_match_count", "direction_omitted_matches",
        "direction_omitted_match_count", "training_match_count", "view_count",
    )
    result = {key: member[key] for key in safe_fields if key in member}
    result["private_member_receipt_path"] = str(private_path.relative_to(ROOT))
    result["private_member_receipt_sha256"] = sha(private_path)
    return result


def fit() -> dict:
    declaration = check()
    if DEPLOYMENT_RECEIPT.exists():
        existing = read(FIT_RECEIPT)
        _seal_deployment_package(existing, declaration)
        return existing
    if FIT_RECEIPT.exists():
        existing = read(FIT_RECEIPT)
        _seal_deployment_package(existing, declaration)
        return existing
    reconstruction._bind_recipe_namespace()
    training_records = reconstruction.pooled_training_records()
    model_rows, model_files = [], {}
    teacher_paths = _teacher_paths()
    for seed in SEEDS:
        unedited_path = MODELS / f"pooled_{seed}_unedited.joblib"
        edited_path = MODELS / f"pooled_{seed}.joblib"
        member_receipt_path = MODELS / f"pooled_{seed}.json"
        if member_receipt_path.exists():
            member = read(member_receipt_path)
            if (member.get("declaration_sha256") != sha(DECLARATION)
                    or member.get("edited_model_sha256") != sha(edited_path)
                    or member.get("unedited_model_sha256") != sha(unedited_path)
                    or member.get("pooled_teacher_sha256") != sha(teacher_paths[seed])):
                raise DeploymentError(f"Existing pooled member receipt is stale for seed {seed}")
            model_files[str(edited_path.relative_to(ROOT))] = member["edited_model_sha256"]
            model_files[str(teacher_paths[seed])] = member["pooled_teacher_sha256"]
            model_rows.append(member)
            continue
        if edited_path.exists():
            raise DeploymentError(f"Edited pooled model exists without its receipt: {edited_path}")
        if unedited_path.exists():
            unedited_receipt = Path(str(unedited_path) + ".json")
            if not unedited_receipt.is_file():
                raise DeploymentError(f"Unedited fit exists without a sealed helper receipt: {unedited_path}")
            fit_receipt = read(unedited_receipt)
            if fit_receipt.get("declaration_sha256") != sha(reconstruction.DECLARATION) or fit_receipt.get("sha256") != sha(unedited_path):
                raise DeploymentError(f"Unedited pooled fit receipt is stale: {unedited_path}")
        else:
            fit_receipt = reconstruction.fit_pooled_member(seed, unedited_path)
            write_once(Path(str(unedited_path) + ".json"), {
                **fit_receipt, "declaration_sha256": sha(reconstruction.DECLARATION),
                "deployment_declaration_sha256": sha(DECLARATION),
            })
        model = joblib.load(unedited_path)
        teacher = joblib.load(teacher_paths[seed])
        direction = (_derive_member_direction(model, teacher, training_records)
                     if declaration["edit"]["kind"] == "ha_projection" else None)
        direction_contributors = (sorted(
            {row["match"] for row in direction["activation_diagnostic"] if row["late_1_sample"] > 0}
            & {row["match"] for row in direction["activation_diagnostic"] if row["correct_at_boundary"] > 0}
        ) if direction else [])
        direction_omitted = (sorted({str(row["match"]) for row in _direction_records(training_records)}
                             - set(direction_contributors)) if direction else [])
        edited_model, changed = apply_selected_edit(model, declaration["edit"],
            declaration["alpha"], declaration["gate_delta"], direction)
        edited_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(edited_model, edited_path)
        model_files[str(edited_path.relative_to(ROOT))] = sha(edited_path)
        model_files[str(teacher_paths[seed])] = sha(teacher_paths[seed])
        member = {
            "seed": seed, "declaration_sha256": sha(DECLARATION),
            "unedited_model_path": str(unedited_path.relative_to(ROOT)),
            "unedited_model_sha256": sha(unedited_path), "edited_model_path": str(edited_path.relative_to(ROOT)),
            "edited_model_sha256": sha(edited_path), "pooled_teacher_path": str(teacher_paths[seed]),
            "pooled_teacher_sha256": sha(teacher_paths[seed]), "fit_receipt": fit_receipt,
            "changed_state_tensors": changed, "direction_sha256": direction["direction_sha256"] if direction else None,
            "direction": direction["direction"] if direction else None,
            "activation_diagnostic": direction["activation_diagnostic"] if direction else [],
            "direction_contributing_matches": direction_contributors,
            "direction_contributing_match_count": len(direction_contributors),
            "direction_omitted_matches": direction_omitted,
            "direction_omitted_match_count": len(direction_omitted),
            "training_match_count": direction["training_match_count"] if direction else None,
            "view_count": direction["view_count"] if direction else None,
        }
        write_once(member_receipt_path, member)
        model_rows.append(member)

    calibrated_files = declaration["calibrated_blend"]["model_files_sha256"]
    model_files.update(calibrated_files)
    from experiments import current_baseline
    cal_models = current_baseline.load_models(manifest_path=CALIBRATED_MANIFEST)
    from experiments import sn_dense_baseline
    cal_manifest = read(CALIBRATED_MANIFEST)
    sn_dense_baseline.validate_models(cal_models, cal_manifest["architecture"])
    training_ids = {str(r["match"]) for r in training_records}
    gar_manifest_path = reconstruction.recipe.TRAIN_RECORDS / "train/manifest.json"
    gar_train_ids = ["soccernet_" + str(game) for game in read(gar_manifest_path)["games"]]
    training_ids.update(gar_train_ids)
    if len(training_ids) != 58 or len(set(gar_train_ids)) != 45:
        raise DeploymentError(f"Expected 13 original + 45 GAR TRAIN games; got {len(training_ids)}")
    files_by_seed = {
        "new_models": {str(row["seed"]): row["edited_model_path"] for row in model_rows},
        "new_teachers": {str(seed): str(path) for seed, path in teacher_paths.items()},
        "calibrated_manifest": str(CALIBRATED_MANIFEST.relative_to(ROOT)),
    }
    fit_receipt = {
        "study": STUDY, "declaration_sha256": sha(DECLARATION), "created_utc": datetime.now(timezone.utc).isoformat(),
        "fit_count": 2, "members": [_public_member_fact(row) for row in model_rows],
        "model_files_sha256": model_files,
        "training_game_ids": sorted(training_ids), "training_game_count": len(training_ids),
        "deployment_files": files_by_seed, "edit": declaration["edit"],
        "edit_alpha": declaration["alpha"], "edit_gate_delta": declaration["gate_delta"],
        "winner_id": declaration["winner_id"],
        "organizer_reference_access": False, "public_scoring": False, "evaluation": False,
    }
    write_once(FIT_RECEIPT, fit_receipt)
    _seal_deployment_package(fit_receipt, declaration)
    return fit_receipt


def load_deployment(receipt_path: Path) -> dict:
    receipt = read(receipt_path)
    for raw, expected in receipt["model_files_sha256"].items():
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != expected:
            raise DeploymentError(f"Pooled inference file changed: {path}")
    if receipt.get("adapter_source_sha256") != sha(Path(__file__)):
        raise DeploymentError("Deployment adapter changed after fit")
    for rel, expected in receipt["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise DeploymentError(f"Deployment inference source changed: {rel}")
    new_models, teachers = {}, {}
    for seed in SEEDS:
        new_models[seed] = joblib.load(ROOT / receipt["new_member_paths"][str(seed)])
        teacher_path = Path(receipt["new_teacher_paths"][str(seed)])
        teachers[seed] = joblib.load(teacher_path)
    cal_manifest_path = ROOT / receipt["calibrated_manifest_path"]
    cal_manifest = read(cal_manifest_path)
    cal_models = baseline.load_models(manifest_path=cal_manifest_path)
    from experiments import sn_dense_baseline
    sn_dense_baseline.validate_models(cal_models, cal_manifest["architecture"])
    for seed in SEEDS:
        member = new_models[seed]
        teacher = teachers[seed]
        training = member.get("training", {})
        if (member.get("kind") != "soccernet_capacity" or "d_state" not in member
                or training.get("seed") != seed or training.get("steps") != 2048
                or training.get("lr") != 3e-4 or training.get("fraction") != .5
                or training.get("ha_target") != "dense" or training.get("train_d") is not True
                or member.get("hidden") != 96
                or teacher.get("seed") != seed
                or member.get("source", {}).get("checkpoint_sha256")
                    != reconstruction.recipe.rf.sn.cap._checkpoint_sha(teacher)
                or list(member.get("feature_names", [])) != list(teacher.get("feature_names", []))
                or not np.array_equal(member.get("median"), teacher.get("median"))
                or not np.array_equal(member.get("scale"), teacher.get("scale"))):
            raise DeploymentError(f"Pooled member {seed} does not match its teacher/schema")
    return {"receipt": receipt, "new_models": new_models, "teachers": teachers,
            "calibrated_models": cal_models, "calibrated_manifest": cal_manifest}


def predict_probabilities(record: dict, inputs: dict, package: dict) -> np.ndarray:
    """Blend the edited and calibrated family 50/50 within matched seeds."""
    record = old.inference_record(record)
    record["names"] = list(record["names"])
    native = audit.tc.inputs_only(inputs)
    members = []
    for seed in SEEDS:
        name = f"dense_{seed}"
        teacher_name = f"teacher_{seed}"
        new_probability = old.upsample(
            sn.predict(record, package["new_models"][seed], package["teachers"][seed]), record, native)
        calibrated_probability = old.upsample(
            sn.predict(record, package["calibrated_models"][name], package["calibrated_models"][teacher_name]), record, native)
        if new_probability.shape != calibrated_probability.shape:
            raise DeploymentError("Matched pooled members produced different probability shapes")
        member = .5 * new_probability + .5 * calibrated_probability
        if not np.isfinite(member).all() or not np.allclose(member.sum(axis=1), 1, atol=1e-6, rtol=0):
            raise DeploymentError("Matched pooled blend is not a probability simplex")
        members.append(member)
    result = .5 * members[0] + .5 * members[1]
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1, atol=1e-6, rtol=0):
        raise DeploymentError("Two-fit pooled deployment is not a probability simplex")
    return result


def decode_probabilities(probabilities: np.ndarray, period: np.ndarray, fps: float, receipt: dict) -> np.ndarray:
    config = receipt["decoder_config"]
    result = feed_spell_push.decode(np.asarray(probabilities)[None, ...], period, float(fps), config)
    return np.asarray(result)


def predict_challenge(package: dict, inputs_path: Path):
    """Label-free full-match features, native phases, pooling and decoding."""
    from experiments import challenge
    inputs = baseline.load_inputs(inputs_path)
    clean = tracking_corruption.inputs_only(inputs)
    placeholder = np.full(len(clean["ball"]), "X", dtype="U1")
    base = old.inference_record(live_gap_experiment.feature_view(
        clean, placeholder, "competition", "BAR-ATH")["joint"])
    phases = audit.phase_records(clean, base)
    phase_probabilities = [predict_probabilities(record, clean, package) for record in phases]
    if len(phase_probabilities) != int(round(float(clean["fps"]) / 5)):
        raise DeploymentError("Competition deployment phase count differs from its native 5 Hz grid")
    probabilities = np.mean(np.stack(phase_probabilities, axis=0), axis=0)
    labels = decode_probabilities(probabilities, supervised.model_periods(clean), float(clean["fps"]), package["receipt"])
    labels[np.asarray(clean["period"]) <= 0] = "X"
    import pandas as pd
    return pd.DataFrame({"frame": np.arange(len(labels)), "label": labels})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p_freeze = sub.add_parser("freeze")
    p_freeze.add_argument("--choice", type=Path, default=CHOICE_DEFAULT)
    sub.add_parser("fit")
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.choice)
    else:
        result = fit()
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
