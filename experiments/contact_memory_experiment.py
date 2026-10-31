"""Frozen-backbone contact-memory residual for the combined-objective GRU.

Only the small conditional H/A residual is fitted. The saved baseline GRU,
normalization, H/A hidden states, and player-only D path stay frozen.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import time

import joblib
import numpy as np

from experiments import (
    combined_objective_deployment as deployment,
    contact_memory as cm,
    corruption_experiment as ce,
    dead_transfer as dt,
    factorized_had as fh,
    gru_robustness_experiment as gr,
    learned_temporal as lt,
    live_gap_experiment as old,
    light_corruption as lc,
    multisource_eval as ms,
    objective_ensemble as oe,
    objective_losses as ol,
    preparation as prep,
    refine_domain,
    robust,
    seed_gap_context as sg,
    supervised,
)
from experiments.eval_set.protocol import sha256


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")).resolve()
OUT = DATA / "experiments/contact_memory_v1"
BASE_CACHE = DATA / "experiments/objective_alignment_v1"
EVAL_CACHE = DATA / "experiments/seed_gap_context_v1"
ENSEMBLE_CACHE = DATA / "experiments/objective_ensemble_v1"
TRAIN_SEEDS = (2026, 2027)
HELDOUT = ("sportec", "metrica", "skillcorner")
EPOCHS = 8
BATCH_SIZE = 64
LR = .001
WEIGHT_DECAY = .0001
CLIP_NORM = 1.0
MAX_LOGIT_DELTA = .5
ADAPTER_WIDTH = 16
TRANSITION_COEFFICIENT = .1
DECLARATION = OUT / "predeclared.json"
PUBLIC_DECLARATION = ROOT / "experiments/results/contact_memory_declaration.json"
FIT_REPORT = OUT / "runs.json"
AGGREGATE_REPORT = ROOT / "experiments/results/contact_memory_readout.json"


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def fingerprint(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path):
    return json.loads(Path(path).read_text())


def _write_exclusive(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False).encode() + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def source_paths() -> tuple[Path, ...]:
    return tuple(ROOT / relative for relative in (
        "experiments/contact_memory.py",
        "experiments/contact_memory_experiment.py",
        "experiments/possession_geometry.py",
        "experiments/supervised.py",
        "experiments/factorized_had.py",
        "experiments/objective_losses.py",
        "experiments/combined_objective_deployment.py",
        "experiments/gru_robustness_experiment.py",
        "experiments/learned_temporal.py",
        "experiments/live_gap_experiment.py",
        "experiments/light_corruption.py",
        "experiments/corruption_experiment.py",
        "experiments/refine_domain.py",
        "experiments/robust.py",
        "experiments/tracking_corruption.py",
        "experiments/multisource_eval.py",
        "experiments/seed_gap_context.py",
        "experiments/objective_ensemble.py",
        "experiments/preparation.py",
        "experiments/refine_features.py",
        "experiments/dead_transfer.py",
        "experiments/metrics.py",
        "experiments/spell_diagnostics.py",
        "experiments/eval_set/manifest.json",
        "experiments/eval_set/protocol.py",
        "experiments/baseline_manifest.json",
        "experiments/model_decision_policy.json",
        "evaluation/possession_metrics.py",
    ))


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in source_paths()}


def _resolve_data_relative(relative: str) -> Path:
    value = Path(relative)
    if value.parts and value.parts[0] == "data":
        return DATA.joinpath(*value.parts[1:])
    return ROOT / value


def _baseline_paths() -> dict:
    spec = _json(BASE_CACHE / "predeclared.json")
    frozen = _json(BASE_CACHE / "checkpoints_frozen.json")
    new_by_name = {entry["file"]: entry for entry in frozen["new_checkpoints"]}
    checkpoints = {}
    for source in HELDOUT:
        checkpoints[source] = {}
        for seed in TRAIN_SEEDS:
            name = f"without_{source}_combined_{seed}.joblib"
            entry = new_by_name.get(name)
            model_path = BASE_CACHE / name
            receipt_path = model_path.with_suffix(".json")
            if (entry is None or not model_path.is_file() or
                    sha256(model_path) != entry["sha256"] or
                    not receipt_path.is_file() or
                    sha256(receipt_path) != entry["receipt_sha256"]):
                raise ValueError(f"Missing or changed frozen baseline checkpoint: {name}")
            checkpoints[source][str(seed)] = {
                "path": str(model_path.relative_to(DATA)),
                "sha256": sha256(model_path),
                "receipt_sha256": sha256(receipt_path),
            }
    training_caches = {}
    for item in prep.split_plan(fold="pooled")["train"]:
        record = spec["prepared_training_caches"][item["match"]]
        cache = _resolve_data_relative(record["cache"])
        receipt = _resolve_data_relative(record["receipt"])
        if (not cache.is_file() or not receipt.is_file() or
                sha256(cache) != record["sha256"] or
                sha256(receipt) != record["receipt_sha256"] or
                record["input_sha256"] != item["sha256"]):
            raise ValueError(f"Missing or changed prepared baseline cache: {item['match']}")
        training_caches[item["match"]] = {
            "cache": str(cache.relative_to(DATA)),
            "cache_sha256": sha256(cache),
            "receipt": str(receipt.relative_to(DATA)),
            "receipt_sha256": sha256(receipt),
        }
    return {"preparation_sha256": sha256(BASE_CACHE / "predeclared.json"),
            "checkpoint_manifest_sha256": sha256(BASE_CACHE / "checkpoints_frozen.json"),
            "checkpoints": checkpoints, "training_caches": training_caches}


def _provider_fold_plans() -> dict[str, dict]:
    baseline_spec = _json(BASE_CACHE / "predeclared.json")
    result = {}
    for source in HELDOUT:
        plan = prep.split_plan(fold=f"leave_out_{source}")
        train_matches = [item["match"] for item in plan["train"]]
        selection_matches = [item["match"] for item in plan["selection"]]
        if (set(train_matches) != set(baseline_spec["training_matches"][source]) or
                set(selection_matches) != set(baseline_spec["evaluation_matches"][source])):
            raise ValueError(f"Provider fold differs from frozen combined-GRU plan: {source}")
        result[source] = {
            "fingerprint": prep.fingerprint(plan),
            "train_matches": train_matches,
            "selection_matches": selection_matches,
        }
    return result


def _evaluation_facts() -> dict:
    ensemble_rows = _json(ENSEMBLE_CACHE / "runs.json")["rows"]
    combined_rows = {
        (row["source"], row["match"], row["view"], row["replicate"]): row
        for row in ensemble_rows
        if row["recipe"] == oe.RECIPE_CANDIDATE
    }
    alignment_rows = _json(BASE_CACHE / "report.json")["rows"]
    member_rows = {
        (row["source"], row["match"], row["view"], row["replicate"], int(row["seed"])): row
        for row in alignment_rows if row["arm"] == "combined"
    }
    def compact(row: dict) -> dict:
        return {key: row[key] for key in (
            "source", "match", "view", "replicate", "spell_f1", "macro_f1",
            "weighted_70_30", "d_f1", "d_precision", "d_recall",
            "short_spell_recall", "spell_by_iou", "n_gt_spells",
            "n_pred_spells", "graded_frames",
        )}

    cells = []
    for cell in oe._cells():
        key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
        if key not in combined_rows:
            raise ValueError(f"Missing saved combined-baseline row: {key}")
        raw_cache = EVAL_CACHE / f"{cell['match']}_{cell['view']}_{cell['replicate']}.joblib"
        ensemble_probability = ENSEMBLE_CACHE / (
            f"candidate_{cell['match']}_{cell['view']}_{cell['replicate']}.npz")
        if not raw_cache.is_file() or not ensemble_probability.is_file():
            raise ValueError(f"Missing frozen baseline evaluation input: {key}")
        cell_fact = {
            **cell,
            "evaluation_cache": str(raw_cache.relative_to(DATA)),
            "evaluation_cache_sha256": sha256(raw_cache),
            "ensemble_probability": str(ensemble_probability.relative_to(DATA)),
            "ensemble_probability_sha256": sha256(ensemble_probability),
            "ensemble_baseline_row": compact(combined_rows[key]),
            "member_baselines": {},
        }
        for seed in TRAIN_SEEDS:
            probability = BASE_CACHE / (
                f"{cell['source']}_{cell['match']}_{cell['view']}_"
                f"{cell['replicate']}_combined_{seed}.npz")
            member_key = (*key, seed)
            if member_key not in member_rows or not probability.is_file():
                raise ValueError(f"Missing frozen member baseline: {member_key}")
            cell_fact["member_baselines"][str(seed)] = {
                "probability": str(probability.relative_to(DATA)),
                "probability_sha256": sha256(probability),
                "row": compact(member_rows[member_key]),
            }
        cells.append(cell_fact)
    return {
        "cells": cells,
        "ensemble_report_sha256": sha256(ENSEMBLE_CACHE / "runs.json"),
        "ensemble_declaration_sha256": sha256(ENSEMBLE_CACHE / "declaration.json"),
        "member_report_sha256": sha256(BASE_CACHE / "report.json"),
        "evaluation_cache_manifest_sha256": sha256(BASE_CACHE / "evaluation_cache_manifest.json"),
    }


def _build_declaration() -> dict:
    baseline_manifest_path = ROOT / "experiments/baseline_manifest.json"
    policy_path = ROOT / "experiments/model_decision_policy.json"
    baseline_manifest = _json(baseline_manifest_path)
    policy = _json(policy_path)
    if baseline_manifest.get("package_version") != "joint_gru_combined_ensemble_v1":
        raise ValueError("Active baseline is no longer the combined-objective ensemble")
    if policy.get("working_candidate", {}).get("baseline_manifest_sha256") != sha256(
            baseline_manifest_path):
        raise ValueError("Active policy does not bind the current baseline manifest")
    plan = prep.split_plan(fold="pooled")
    input_hashes = {}
    for item in plan["train"] + plan["selection"]:
        path = prep.data_path(item, DATA)
        if sha256(path) != item["sha256"]:
            raise ValueError(f"External input changed: {item['match']}")
        input_hashes[str(path.relative_to(DATA))] = item["sha256"]
    baseline_artifacts = _baseline_paths()
    eval_facts = _evaluation_facts()
    provider_fold_plans = _provider_fold_plans()
    objective_calibration = _json(BASE_CACHE / "predeclared.json")["calibration"]
    previous_calibration = _json(EVAL_CACHE / "predeclared.json")["calibration"]
    if objective_calibration != previous_calibration:
        raise ValueError("Baseline and saved evaluation calibration differ")
    calibration = objective_calibration
    return {
        "schema_version": 1,
        "study_id": "contact_memory_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before adapter fitting and candidate scoring",
        "question": "Does a zero-initialized, bounded H/A contact-memory residual improve the current combined-GRU ensemble's full-provider spell F1?",
        "historical_status": {
            "development_matches_previously_exposed": True,
            "independent_confirmation": False,
            "organizer_labels_or_scores_used": False,
            "selection_provider_excluded_from_every_fitted_component": True,
        },
        "hypothesis": "Sparse stable contact events before and after uncertain intervals provide a time- and evidence-aware H/A bridge, recovering some short spells and turnovers without forcing an unchanged-team hold.",
        "mechanism": {
            "control": "Saved combined-objective two-seed GRU probability ensemble and exact full-provider member outputs.",
            "treatment": "Freeze each matching provider-held-out combined-GRU checkpoint, hidden path, normalization and D path; fit only a small conditional H/A residual over frozen hidden/logit rows and two query-local reliable-contact event tokens.",
            "memory_spec": cm.MEMORY_SPEC,
            "residual": "Two-slot learned attention over prior/future event tokens with log-age and direction; tanh-bounded logit correction within +/-0.5, exactly zero when neither event is valid; final layer initializes to zero.",
            "D_path": "No contact tokens or residual enter the player-only D logit. Candidate predictions copy the exact archived baseline P(D) and rescale only conditional H/A mass.",
            "training_view_parity": "Regenerate clean/gap/dropout_legacy/dropout_gap with the exact archived seeds/functions and require bitwise augmented x/y/index parity against objective_alignment cached records before memory fitting.",
            "uncertainty": "Training hidden states are produced by each frozen backbone on its own training matches; adapter training rows are not cross-fitted. Evaluation provider remains excluded from backbone and adapter labels.",
        },
        "arms": ["combined_baseline_control", "contact_memory_residual"],
        "seeds": list(TRAIN_SEEDS),
        "heldout_providers": list(HELDOUT),
        "fold_plan": prep.split_plan(fold="pooled"),
        "training_plan_sha256": prep.fingerprint(plan),
        "provider_fold_plans": provider_fold_plans,
        "input_sha256": input_hashes,
        "baseline": {
            "manifest": "experiments/baseline_manifest.json",
            "manifest_sha256": sha256(baseline_manifest_path),
            "decision_policy": "experiments/model_decision_policy.json",
            "decision_policy_sha256": sha256(policy_path),
            "preparation": baseline_artifacts,
            "evaluation": eval_facts,
        },
        "calibration": calibration,
        "views": {
            "primary": "clean",
            "stress": "feed_like",
            "feed_like_replicates": ["corrupt_1", "corrupt_2"],
            "stress_generator": "The exact saved feed-like tracking views from the prior full-provider evaluation cache.",
        },
        "training": {
            "adapters": 6,
            "fits": "one adapter per heldout provider and seed; no GRU or control refits",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip": CLIP_NORM,
            "window_selection": "Exact cached paired_records, direct/team-swapped rows and replacement-window masks.",
            "normalization": "Reuse frozen matched backbone training-only median/std; fit no new normalization.",
            "objective": "Conditional H/A CE with the existing complete-native-spell duration factors plus the live-transition half of the existing .1 auxiliary. Frozen D and dead-transition losses have no adapter gradient.",
            "parameter_budget": "Six small residual adapters; no new recurrent, D, backbone or normalization parameters.",
            "representation_cache": "Frozen base hidden and H/A logits per declared window are computed once per seed/fold and retained only in process memory.",
        },
        "metric": {
            "primary": "official greedy one-to-one possession spell F1 averaged across IoU .3/.4/.5/.6/.7",
            "tie_break": "macro F1 only for exact spell-F1 ties",
            "auxiliary": "0.70 * spell F1 + 0.30 * macro F1; not ranked",
            "aggregation": "Mean matches within provider, then equal mean across providers; feed_like replicates averaged within match.",
            "guardrails": {
                "clean_spell": "positive equal-provider delta",
                "feed_like_spell": "candidate no worse than active baseline - .01",
                "each_provider_clean_spell": "delta >= -.02",
                "each_provider_clean_macro": "delta >= -.02",
                "each_provider_clean_d_f1": "delta >= -.02; decoded D labels may shift under the shared unchanged Viterbi decoder",
            },
            "secondary": ["macro F1", "D precision/recall/F1", "all five IoU spell F1 values", "short-spell recall", "fragmentation", "spell counts", "long-gap and contact-memory support"],
        },
        "budget": {
            "new_adapter_fits": 6,
            "baseline_fits": 0,
            "new_ensemble_score_cells": 15,
            "new_seed_diagnostic_cells": 30,
            "total_candidate_rows": 45,
            "additional_arms_or_sweeps": 0,
        },
        "artifacts": {
            "private_output": str(OUT.relative_to(DATA)),
            "public_readout": "experiments/CONTACT_MEMORY.md",
            "public_aggregate": "experiments/results/contact_memory_readout.json",
            "public_declaration": "experiments/results/contact_memory_declaration.json",
        },
        "source_sha256": _source_hashes(),
        "organizer_labels_read": False,
        "organizer_scored": False,
        "automatic_promotion": False,
    }


def freeze() -> dict:
    if PUBLIC_DECLARATION.exists():
        check()
        return {"declaration": str(PUBLIC_DECLARATION.relative_to(ROOT)),
                "sha256": sha256(PUBLIC_DECLARATION), "already_frozen": True}
    spec = _build_declaration()
    payload = json.dumps(spec, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False).encode() + b"\n"
    descriptor = os.open(PUBLIC_DECLARATION, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return {"declaration": str(PUBLIC_DECLARATION.relative_to(ROOT)),
            "sha256": sha256(PUBLIC_DECLARATION), "already_frozen": False}


def check() -> dict:
    if not PUBLIC_DECLARATION.is_file():
        raise ValueError("Contact-memory declaration is not frozen")
    spec = _json(PUBLIC_DECLARATION)
    if spec.get("source_sha256") != _source_hashes():
        raise ValueError("Contact-memory source changed after freeze; use a new study ID")
    if sha256(ROOT / spec["baseline"]["manifest"]) != spec["baseline"]["manifest_sha256"]:
        raise ValueError("Current baseline manifest changed after freeze")
    if sha256(ROOT / spec["baseline"]["decision_policy"]) != spec["baseline"]["decision_policy_sha256"]:
        raise ValueError("Current model policy changed after freeze")
    for relative, expected in spec["input_sha256"].items():
        if sha256(DATA / relative) != expected:
            raise ValueError(f"External source changed after freeze: {relative}")
    baseline_now = _baseline_paths()
    if baseline_now != spec["baseline"]["preparation"]:
        raise ValueError("Frozen baseline cache/checkpoint receipts changed")
    if _evaluation_facts() != spec["baseline"]["evaluation"]:
        raise ValueError("Frozen baseline evaluation files or rows changed")
    spec["_declaration_sha256"] = sha256(PUBLIC_DECLARATION)
    return spec


def prepare() -> dict:
    """Materialize only training memory arrays and prove exact cached-view parity."""
    spec = check()
    plan = prep.split_plan(fold="pooled")
    receipts = []
    for item in plan["train"]:
        cache_path = OUT / f"memory_train_{item['match']}.npz"
        receipt_path = cache_path.with_suffix(".json")
        receipts.append(_create_memory_cache(
            item, spec["calibration"], cache_path, receipt_path,
            spec["_declaration_sha256"]))
        print("PREPARED CONTACT MEMORY", item["source"], item["match"], flush=True)
    if len(receipts) != 13 or any(
            not row.get("memory_builder_label_independent") for row in receipts):
        raise ValueError("Expected 13 parity-verified, label-independent memory caches")
    receipt = {
        "study_id": spec["study_id"],
        "declaration_sha256": spec["_declaration_sha256"],
        "training_match_count": len(receipts),
        "cache_receipts": receipts,
        "augmented_feature_parity": "all four cached views, target vectors and sampled indices checked before fitting",
        "selection_labels_loaded": False,
    }
    path = OUT / "preparation_receipt.json"
    if path.exists():
        saved = _json(path)
        if saved != receipt:
            raise ValueError("Frozen contact-memory preparation receipt changed")
    else:
        _write_exclusive(path, receipt)
    return {"receipt": str(path.relative_to(DATA)), "matches": len(receipts),
            "sha256": sha256(path)}


def _adapter_path(heldout: str, seed: int) -> Path:
    return OUT / f"adapter_without_{heldout}_{seed}.joblib"


def fit(heldout: str, seed: int) -> dict:
    """Fit one of the six predeclared H/A adapters; no GRU weights change."""
    spec = check()
    heldout = str(heldout).lower()
    seed = int(seed)
    if heldout not in HELDOUT or seed not in TRAIN_SEEDS:
        raise ValueError("Expected a declared provider fold and seed")
    if not (OUT / "preparation_receipt.json").is_file():
        raise ValueError("Run label-free contact preparation before fitting")
    if (OUT / "checkpoints_frozen.json").exists():
        raise ValueError("All adapter fitting was sealed; use a new study ID")
    model_path = _adapter_path(heldout, seed)
    receipt_path = model_path.with_suffix(".json")
    if model_path.exists() or receipt_path.exists():
        if not model_path.is_file() or not receipt_path.is_file():
            raise ValueError("Incomplete adapter fit artifact")
        saved = _json(receipt_path)
        if saved.get("model_sha256") != sha256(model_path):
            raise ValueError("Saved adapter changed")
        return saved

    control = spec["baseline"]["preparation"]["checkpoints"][heldout][str(seed)]
    current_fold = prep.split_plan(fold=f"leave_out_{heldout}")
    if prep.fingerprint(current_fold) != spec["provider_fold_plans"][heldout]["fingerprint"]:
        raise ValueError(f"Provider training fold changed after freeze: {heldout}")
    base_path = DATA / control["path"]
    if sha256(base_path) != control["sha256"]:
        raise ValueError("Frozen base checkpoint changed")
    fitted_base, base_model = _load_base(base_path)
    records = _fold_training_records(heldout, spec)
    if list(fitted_base["feature_names"]) != list(records[0]["names"]):
        raise ValueError("Adapter records differ from frozen base feature schema")
    started = time.monotonic()
    fitted = fit_adapter(records, fitted_base, base_model, seed=seed)
    fitted.update(heldout=heldout, seed=seed,
                  base_checkpoint_path=control["path"],
                  base_checkpoint_sha256=control["sha256"],
                  declaration_sha256=spec["_declaration_sha256"])
    with tempfile.NamedTemporaryFile(prefix=f".{model_path.name}.", suffix=".joblib",
                                     dir=OUT, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(fitted, temporary)
        os.replace(temporary, model_path)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "heldout_provider": heldout,
        "seed": seed,
        "model": str(model_path.relative_to(DATA)),
        "model_sha256": sha256(model_path),
        "declaration_sha256": spec["_declaration_sha256"],
        "base_checkpoint_sha256": control["sha256"],
        "parameters": int(fitted["parameters"]),
        "windows_per_epoch": int(fitted["windows_per_epoch"]),
        "adapter_optimizer_updates": int(fitted["optimizer_updates"]),
        "epochs": EPOCHS,
        "normalization_reused_exactly": True,
        "base_parameters_updated": False,
        "dead_probability_path_updated": False,
        "seconds": time.monotonic() - started,
        "history": fitted["history"],
        "training_matches": [record["match"] for record in records[::4]],
        "selection_labels_read": False,
        "organizer_labels_read": False,
    }
    _write_exclusive(receipt_path, receipt)
    print("FROZEN CONTACT ADAPTER", heldout, seed,
          "updates", receipt["adapter_optimizer_updates"],
          "seconds", round(receipt["seconds"], 1), flush=True)
    return receipt


def seal() -> dict:
    """Close the six-adapter fit budget before any selection score is opened."""
    spec = check()
    seal_path = OUT / "checkpoints_frozen.json"
    required = []
    for heldout in HELDOUT:
        for seed in TRAIN_SEEDS:
            model = _adapter_path(heldout, seed)
            receipt = model.with_suffix(".json")
            if not model.is_file() or not receipt.is_file():
                raise ValueError(f"Missing predeclared contact adapter: {heldout}/{seed}")
            metadata = _json(receipt)
            if (metadata.get("model_sha256") != sha256(model) or
                    metadata.get("declaration_sha256") != spec["_declaration_sha256"]):
                raise ValueError(f"Contact adapter receipt mismatch: {heldout}/{seed}")
            required.append({
                "heldout_provider": heldout,
                "seed": seed,
                "path": str(model.relative_to(DATA)),
                "sha256": sha256(model),
                "receipt_sha256": sha256(receipt),
            })
    result = {
        "study_id": spec["study_id"],
        "declaration_sha256": spec["_declaration_sha256"],
        "new_fits": len(required),
        "baseline_fits": 0,
        "models": required,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "organizer_labels_read": False,
        "selection_scores_opened": False,
    }
    if seal_path.exists():
        saved = _json(seal_path)
        for key in ("study_id", "declaration_sha256", "new_fits", "baseline_fits", "models"):
            if saved.get(key) != result[key]:
                raise ValueError("Adapter seal is immutable and does not match outputs")
        return saved
    _write_exclusive(seal_path, result)
    return result


def _verify_seal(spec: dict) -> dict:
    seal_path = OUT / "checkpoints_frozen.json"
    if not seal_path.is_file():
        raise ValueError("All six adapters must be sealed before evaluation")
    seal_spec = _json(seal_path)
    if seal_spec.get("declaration_sha256") != spec["_declaration_sha256"]:
        raise ValueError("Adapter seal belongs to a different declaration")
    for item in seal_spec["models"]:
        path = DATA / item["path"]
        receipt = path.with_suffix(".json")
        if (sha256(path) != item["sha256"] or
                sha256(receipt) != item["receipt_sha256"]):
            raise ValueError("A sealed contact adapter or receipt changed")
    return seal_spec


def retain_dead_probability(probability: np.ndarray,
                            reference_dead: np.ndarray) -> np.ndarray:
    """Keep archived P(D) exactly and preserve adapted conditional H/A odds."""
    values = np.asarray(probability, dtype=np.float64)
    dead = np.asarray(reference_dead, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != 3 or dead.shape != (len(values),)
            or not np.isfinite(values).all() or not np.isfinite(dead).all()
            or (dead < 0).any() or (dead > 1).any()):
        raise ValueError("Expected finite three-class probabilities and reference P(D)")
    conditional_mass = values[:, 0] + values[:, 1]
    if (conditional_mass <= 0).any():
        raise ValueError("Cannot preserve H/A odds when conditional mass is zero")
    home_fraction = values[:, 0] / conditional_mass
    live_mass = 1.0 - dead
    result = np.empty_like(values, dtype=np.float64)
    result[:, 0] = live_mass * home_fraction
    result[:, 1] = live_mass - result[:, 0]
    result[:, 2] = dead
    return result


def upsample_probabilities(sampled: np.ndarray, record: dict,
                           native_period: np.ndarray) -> np.ndarray:
    """Interpolate each sampled probability channel on the native input grid."""
    values = np.asarray(sampled)
    if values.shape != (len(record["indices"]), 3):
        raise ValueError("Expected one sampled three-class row per contact record index")
    native_period = np.asarray(native_period)
    if native_period.shape != (int(record["native_n"]),):
        raise ValueError("Native period grid does not match the contact record")
    return np.column_stack([
        lt.upsample(values[:, channel], record, native_period)
        for channel in range(3)
    ])


def _score_cell(cell: dict, spec: dict, base_models: dict,
                adapters: dict) -> dict:
    key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
    cache_path = DATA / cell["evaluation_cache"]
    if sha256(cache_path) != cell["evaluation_cache_sha256"]:
        raise ValueError(f"Evaluation tracking cache changed: {key}")
    cache = joblib.load(cache_path)
    inputs = cache["inputs"]
    labels = np.asarray(cache["labels"]).astype(str)
    record = dict(cache["record"])
    # Labels are not part of inference; replacing the cache targets makes this
    # boundary explicit. Selection labels enter only after the adapter seal.
    record["y"] = np.full(len(record["x"]), -1, dtype=np.int8)
    memory = cm.memory_features(inputs, record["indices"])
    cm.validate_memory(memory, len(record["x"]))
    member_probabilities = []
    base_probabilities = []
    member_rows = []
    source = cell["source"]
    for seed in TRAIN_SEEDS:
        fitted_base, base_model = base_models[(source, seed)]
        fitted_adapter, adapter = adapters[(source, seed)]
        base_probability = upsample_probabilities(
            fh.predict(record, fitted_base), record,
            supervised.model_periods(inputs))
        baseline_path = DATA / cell["member_baselines"][str(seed)]["probability"]
        if sha256(baseline_path) != cell["member_baselines"][str(seed)]["probability_sha256"]:
            raise ValueError(f"Frozen saved member probability changed: {key}/{seed}")
        with np.load(baseline_path, allow_pickle=False) as archive:
            saved_probability = archive["probability"]
        if not np.allclose(base_probability, saved_probability, rtol=0, atol=1e-6):
            raise ValueError(f"Frozen control probability parity failed: {key}/{seed}")
        candidate_sampled = predict(
            record, memory, fitted_base, base_model, adapter)
        candidate_probability = upsample_probabilities(
            candidate_sampled, record, supervised.model_periods(inputs))
        if not np.array_equal(candidate_probability[:, 2], base_probability[:, 2]):
            raise ValueError(f"Frozen D probabilities changed: {key}/{seed}")
        candidate_probability = retain_dead_probability(
            candidate_probability, saved_probability[:, 2])
        if not np.array_equal(candidate_probability[:, 2], saved_probability[:, 2]):
            raise ValueError(f"Candidate D probabilities differ from frozen saved member: {key}/{seed}")
        candidate_prediction = supervised.decode(
            candidate_probability, supervised.model_periods(inputs),
            float(inputs["fps"]), lt.COST)
        member_file = OUT / (
            f"candidate_{cell['match']}_{cell['view']}_{cell['replicate']}_seed_{seed}.npz")
        with tempfile.NamedTemporaryFile(prefix=f".{member_file.name}.", suffix=".npz",
                                         dir=OUT, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, probability=candidate_probability,
                                prediction=candidate_prediction.astype("U1"))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if member_file.exists():
                raise ValueError(f"Candidate output is immutable: {member_file.name}")
            os.replace(temporary, member_file)
        finally:
            temporary.unlink(missing_ok=True)
        row = oe.score_prediction(
            labels, candidate_prediction, inputs,
            f"contact_memory_seed_{seed}", cell,
            digest_array(candidate_probability))
        baseline_row = cell["member_baselines"][str(seed)]["row"]
        if row["n_gt_spells"] != baseline_row["n_gt_spells"] or (
                row["graded_frames"] != baseline_row["graded_frames"]):
            raise ValueError(f"Candidate changed the graded spell population: {key}/{seed}")
        member_probabilities.append(candidate_probability)
        base_probabilities.append(saved_probability)
        member_rows.append({
            "candidate": row,
            "control": baseline_row,
            "candidate_probability_file": str(member_file.relative_to(DATA)),
            "candidate_probability_file_sha256": sha256(member_file),
            "control_probability_sha256": digest_array(saved_probability),
            "d_probability_exact": True,
        })

    candidate_ensemble = oe.blend_probabilities(*member_probabilities)
    reconstructed_control = oe.blend_probabilities(*base_probabilities)
    control_path = DATA / cell["ensemble_probability"]
    if sha256(control_path) != cell["ensemble_probability_sha256"]:
        raise ValueError(f"Frozen current-ensemble probabilities changed: {key}")
    with np.load(control_path, allow_pickle=False) as archive:
        control_ensemble = archive["probability"]
    if not np.array_equal(candidate_ensemble[:, 2], control_ensemble[:, 2]):
        raise ValueError(f"Candidate ensemble D probabilities differ from frozen baseline: {key}")
    if not np.allclose(reconstructed_control, control_ensemble, rtol=0, atol=1e-6):
        raise ValueError(f"Candidate ensemble changed frozen D probabilities: {key}")
    prediction = supervised.decode(
        candidate_ensemble, supervised.model_periods(inputs),
        float(inputs["fps"]), lt.COST)
    ensemble_file = OUT / (
        f"candidate_{cell['match']}_{cell['view']}_{cell['replicate']}_ensemble.npz")
    with tempfile.NamedTemporaryFile(prefix=f".{ensemble_file.name}.", suffix=".npz",
                                     dir=OUT, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, probability=candidate_ensemble,
                            prediction=prediction.astype("U1"))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if ensemble_file.exists():
            raise ValueError(f"Candidate output is immutable: {ensemble_file.name}")
        os.replace(temporary, ensemble_file)
    finally:
        temporary.unlink(missing_ok=True)
    candidate_row = oe.score_prediction(
        labels, prediction, inputs, "contact_memory_ensemble", cell,
        digest_array(candidate_ensemble))
    baseline_row = cell["ensemble_baseline_row"]
    if (candidate_row["n_gt_spells"] != baseline_row["n_gt_spells"] or
            candidate_row["graded_frames"] != baseline_row["graded_frames"]):
        raise ValueError(f"Candidate changed the graded population: {key}")
    return {
        "cell": cell,
        "control_ensemble": baseline_row,
        "candidate_ensemble": candidate_row,
        "seed_members": member_rows,
        "candidate_ensemble_probability_file": str(ensemble_file.relative_to(DATA)),
        "candidate_ensemble_probability_file_sha256": sha256(ensemble_file),
        "d_probability_exact": True,
    }


def evaluate() -> dict:
    """Score only the sealed candidate on the fixed full-provider cohort."""
    spec = check()
    _verify_seal(spec)
    if FIT_REPORT.exists():
        raise ValueError("Saved candidate scores are immutable")
    base_models = {}
    adapters = {}
    for source in HELDOUT:
        for seed in TRAIN_SEEDS:
            control = spec["baseline"]["preparation"]["checkpoints"][source][str(seed)]
            base_path = DATA / control["path"]
            fitted_base, base_model = _load_base(base_path)
            adapter_path = _adapter_path(source, seed)
            fitted_adapter, adapter = _load_adapter(adapter_path)
            if fitted_adapter.get("base_checkpoint_sha256") != control["sha256"]:
                raise ValueError(f"Adapter/backbone binding mismatch: {source}/{seed}")
            base_models[(source, seed)] = (fitted_base, base_model)
            adapters[(source, seed)] = (fitted_adapter, adapter)
    cells = []
    for cell in oe._cells():
        cells.append(_score_cell(cell, spec, base_models, adapters))
        print("SCORED CONTACT CELL", cell["source"], cell["match"],
              cell["view"], cell["replicate"], flush=True)
    result = {
        "study_id": spec["study_id"],
        "declaration_sha256": spec["_declaration_sha256"],
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "rows": cells,
        "new_ensemble_cells": len(cells),
        "new_seed_diagnostic_rows": sum(len(row["seed_members"]) for row in cells),
        "d_probability_exact_for_every_member_and_ensemble": all(
            row["d_probability_exact"] for row in cells),
        "organizer_labels_read": False,
        "organizer_scored": False,
        "automatic_promotion": False,
    }
    _write_exclusive(FIT_REPORT, result)
    return {"runs": str(FIT_REPORT.relative_to(DATA)), "sha256": sha256(FIT_REPORT),
            "cells": len(cells)}


def _row_metric(row: dict, field: str) -> float:
    if field.startswith("iou_"):
        key = field.removeprefix("iou_")
        value = row["spell_by_iou"][key]
        return float(value["f1"] if isinstance(value, dict) else value)
    return float(row[field])


def _mean(rows: list[dict], field: str) -> float:
    return float(np.mean([_row_metric(row, field) for row in rows]))


def summarize(runs: dict) -> dict:
    """Aggregate paired rows by match, provider and declared view."""
    rows = runs["rows"]
    recipes = {
        "combined_baseline_ensemble": [
            item["control_ensemble"] for item in rows],
        "contact_memory_ensemble": [
            item["candidate_ensemble"] for item in rows],
    }
    for seed_index, seed in enumerate(TRAIN_SEEDS):
        recipes[f"combined_baseline_seed_{seed}"] = [
            item["seed_members"][seed_index]["control"] for item in rows]
        recipes[f"contact_memory_seed_{seed}"] = [
            item["seed_members"][seed_index]["candidate"] for item in rows]
    fields = (
        "spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
        "d_recall", "short_spell_recall",
    )
    fields += tuple(f"iou_{threshold:.1f}" for threshold in (.3, .4, .5, .6, .7))
    summaries = {}
    for recipe, recipe_rows in recipes.items():
        summaries[recipe] = {}
        for view in ("clean", "feed_like"):
            per_provider = {}
            for provider in HELDOUT:
                selected = [row for row in recipe_rows
                            if row["source"] == provider and row["view"] == view]
                expected = len(dt.SPLITS[provider]["eval"]) * (1 if view == "clean" else 2)
                if len(selected) != expected:
                    raise ValueError(f"Incomplete readout cells: {recipe}/{provider}/{view}")
                by_match = {}
                for row in selected:
                    by_match.setdefault(row["match"], []).append(row)
                match_means = [{
                    field: _mean(match_rows, field)
                    for field in fields
                } for match_rows in by_match.values()]
                per_provider[provider] = {
                    field: float(np.mean([item[field] for item in match_means]))
                    for field in fields
                }
                per_provider[provider]["n_matches"] = len(match_means)
                per_provider[provider]["n_rows"] = len(selected)
            summaries[recipe][view] = {
                "per_provider": per_provider,
                "equal_provider_mean": {
                    field: float(np.mean([per_provider[p][field] for p in HELDOUT]))
                    for field in fields
                },
                "aggregation": "mean within provider, then equal mean across providers; feed_like replicates are averaged within match",
            }
    comparisons = {}
    pairs = (
        ("contact_memory_ensemble", "combined_baseline_ensemble"),
        ("contact_memory_seed_2026", "combined_baseline_seed_2026"),
        ("contact_memory_seed_2027", "combined_baseline_seed_2027"),
    )
    for candidate, control in pairs:
        comparisons[candidate] = {}
        for view in ("clean", "feed_like"):
            candidate_summary = summaries[candidate][view]
            control_summary = summaries[control][view]
            comparisons[candidate][view] = {
                "equal_provider_delta": {
                    field: candidate_summary["equal_provider_mean"][field]
                    - control_summary["equal_provider_mean"][field]
                    for field in fields
                },
                "per_provider_delta": {
                    provider: {
                        field: candidate_summary["per_provider"][provider][field]
                        - control_summary["per_provider"][provider][field]
                        for field in fields
                    } for provider in HELDOUT
                },
            }
    ensemble_clean = comparisons["contact_memory_ensemble"]["clean"]
    ensemble_stress = comparisons["contact_memory_ensemble"]["feed_like"]
    guards = {
        "positive_equal_provider_clean_spell_delta":
            ensemble_clean["equal_provider_delta"]["spell_f1"] > 0,
        "feed_like_spell_delta_at_least_minus_0_01":
            ensemble_stress["equal_provider_delta"]["spell_f1"] >= -0.01,
        "each_provider_clean_spell_delta_at_least_minus_0_02": all(
            ensemble_clean["per_provider_delta"][provider]["spell_f1"] >= -0.02
            for provider in HELDOUT),
        "each_provider_clean_macro_delta_at_least_minus_0_02": all(
            ensemble_clean["per_provider_delta"][provider]["macro_f1"] >= -0.02
            for provider in HELDOUT),
        "each_provider_clean_d_f1_delta_at_least_minus_0_02": all(
            ensemble_clean["per_provider_delta"][provider]["d_f1"] >= -0.02
            for provider in HELDOUT),
        "d_probabilities_exact": bool(runs[
            "d_probability_exact_for_every_member_and_ensemble"]),
    }
    return {
        "recipes": summaries,
        "candidate_minus_control": comparisons,
        "guardrails": {"criteria": {
            "clean_spell_delta": "positive equal-provider mean",
            "feed_like_spell_delta": "at least -0.01 equal-provider mean",
            "provider_clean_spell_macro_d": "each delta at least -0.02",
                "D_probability": "bit-exact conditional-model member and ensemble probability equality",
        }, "checks": guards, "passes": all(guards.values())},
        "primary_order": ["spell_f1", "macro_f1"],
        "weighted_70_30_used_for_ranking": False,
        "no_automatic_promotion": True,
    }


def report() -> dict:
    """Write the public aggregate readout from sealed saved rows; never rescore."""
    spec = check()
    _verify_seal(spec)
    if not FIT_REPORT.is_file():
        raise ValueError("Candidate evaluation has not completed")
    if AGGREGATE_REPORT.exists():
        raise ValueError("Contact-memory readout is immutable")
    runs = _json(FIT_REPORT)
    if runs.get("declaration_sha256") != spec["_declaration_sha256"]:
        raise ValueError("Saved candidate scores do not match the frozen declaration")
    result = {
        "study_id": spec["study_id"],
        "declaration_sha256": spec["_declaration_sha256"],
        "runs_sha256": sha256(FIT_REPORT),
        "reported_utc": datetime.now(timezone.utc).isoformat(),
        "question": spec["question"],
        "scope": spec["historical_status"],
        "baseline_manifest_sha256": spec["baseline"]["manifest_sha256"],
        "decision_policy_sha256": spec["baseline"]["decision_policy_sha256"],
        "cohort": [{key: cell[key] for key in ("source", "match", "view", "replicate")}
                    for cell in spec["baseline"]["evaluation"]["cells"]],
        "mechanism": spec["mechanism"],
        "metric": spec["metric"],
        "budget": spec["budget"],
        "result": summarize(runs),
        "d_probability_exact_for_every_member_and_ensemble":
            runs["d_probability_exact_for_every_member_and_ensemble"],
        "limitations": [
            "All evaluation matches were previously exposed and are development evidence, not independent confirmation.",
            "Metrica and SkillCorner labels include event-derived silver targets; feed_like is synthetic stress.",
            "The adapter is trained on frozen-backbone hidden states from matches used by that backbone, so adapter training rows are not cross-fitted.",
            "An unchanged P(D) can still yield a changed decoded D label because Viterbi jointly decodes H/A/D paths.",
            "Only the saved provider-held-out five-match cohort and two seeds were measured; full-match accuracy is unknown.",
        ],
        "private_artifacts": {
            "runs": str(FIT_REPORT.relative_to(DATA)),
            "runs_sha256": sha256(FIT_REPORT),
            "sealed_checkpoints": str((OUT / "checkpoints_frozen.json").relative_to(DATA)),
            "organizer_labels_read": False,
            "organizer_scored": False,
        },
    }
    _write_exclusive(AGGREGATE_REPORT, result)
    return {"report": str(AGGREGATE_REPORT.relative_to(ROOT)),
            "sha256": sha256(AGGREGATE_REPORT)}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze")
    sub.add_parser("prepare")
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--heldout", choices=HELDOUT, required=True)
    fit_parser.add_argument("--seed", type=int, choices=TRAIN_SEEDS, required=True)
    sub.add_parser("seal")
    sub.add_parser("evaluate")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    actions = {
        "freeze": freeze,
        "prepare": prepare,
        "seal": seal,
        "evaluate": evaluate,
        "report": report,
    }
    result = fit(args.heldout, args.seed) if args.action == "fit" else actions[args.action]()
    print(json.dumps(result, sort_keys=True, indent=2))


def build_adapter():
    """Small zero-initialized H/A-only residual that requires contact evidence."""
    import torch
    from torch import nn

    class Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.query = nn.Linear(50, ADAPTER_WIDTH)
            self.key = nn.Linear(len(cm.EVENT_FIELDS) + 2, ADAPTER_WIDTH)
            self.value = nn.Linear(len(cm.EVENT_FIELDS) + 2, ADAPTER_WIDTH)
            self.attention = nn.Linear(ADAPTER_WIDTH, 1, bias=False)
            self.residual = nn.Sequential(
                nn.Linear(48 + 2 + ADAPTER_WIDTH, ADAPTER_WIDTH),
                nn.Tanh(),
                nn.Linear(ADAPTER_WIDTH, 2),
            )
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

        def forward(self, hidden, base_logits, tokens, ages, valid):
            age = torch.log1p(ages.clamp(min=0, max=cm.MAX_AGE_SECONDS))
            age = age / math.log1p(cm.MAX_AGE_SECONDS)
            direction = torch.tensor(
                [-1.0, 1.0], dtype=tokens.dtype, device=tokens.device
            ).view(1, 1, 2, 1).expand(tokens.shape[0], tokens.shape[1], -1, -1)
            event = torch.cat((tokens, age.unsqueeze(-1), direction), dim=-1)
            query = self.query(torch.cat((hidden, base_logits), dim=-1)).unsqueeze(-2)
            keys = self.key(event)
            values = self.value(event)
            scores = self.attention(torch.tanh(query + keys)).squeeze(-1)
            mask = valid.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask, -1e9)
            weights = torch.softmax(scores, dim=-1) * mask.to(scores.dtype)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            memory = (weights.unsqueeze(-1) * values).sum(dim=-2)
            joined = torch.cat((hidden, base_logits, memory), dim=-1)
            delta = MAX_LOGIT_DELTA * torch.tanh(self.residual(joined))
            delta = delta * mask.any(dim=-1, keepdim=True).to(delta.dtype)
            return delta

    return Adapter()


def adapted_log_probabilities(base_model, adapter, x, tokens, ages, valid):
    """Run a frozen factorized GRU plus a conditional H/A residual."""
    import torch
    from torch.nn import functional as F

    hidden = base_model.sequence(x)[0]
    live_logits = base_model.output(hidden)[..., :2]
    dead_input = x * base_model.dead_input_mask
    dead_logits = base_model.output(base_model.sequence(dead_input)[0])[..., 2]
    delta = adapter(hidden, live_logits, tokens, ages, valid)
    conditional = F.log_softmax(live_logits + delta, dim=-1)
    return torch.cat((F.logsigmoid(-dead_logits)[..., None] + conditional,
                      F.logsigmoid(dead_logits)[..., None]), dim=-1)


def _load_base(checkpoint: Path):
    import torch
    fitted = joblib.load(checkpoint)
    if fitted.get("kind") != "factorized_had" or fitted.get("objective") != "combined":
        raise ValueError(f"Unexpected combined-objective checkpoint: {checkpoint}")
    network = fh.network(fitted["input_features"], fitted["player_indices"],
                         fitted["player_only"])
    network.load_state_dict({key: torch.as_tensor(value)
                             for key, value in fitted["state"].items()})
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    fitted["_checkpoint_sha256"] = sha256(checkpoint)
    return fitted, network


def _memory_record(record: dict, base: dict, alternate: dict) -> dict:
    """Attach matched view memories, preserving the cached direct/swap order."""
    cm.validate_memory(base, len(record["x"]))
    cm.validate_memory(alternate, len(record["alternate"]))
    result = dict(record)
    result.update(
        memory_tokens=np.asarray(base["tokens"], dtype=np.float32),
        memory_ages=np.asarray(base["ages"], dtype=np.float32),
        memory_valid=np.asarray(base["valid"], dtype=bool),
        alternate_memory_tokens=np.asarray(alternate["tokens"], dtype=np.float32),
        alternate_memory_ages=np.asarray(alternate["ages"], dtype=np.float32),
        alternate_memory_valid=np.asarray(alternate["valid"], dtype=bool),
    )
    return result


def _swapped_record_memories(record: dict) -> dict:
    result = dict(record)
    for prefix in ("memory", "alternate_memory"):
        memory = cm.swap_memory({
            "tokens": result[f"{prefix}_tokens"],
            "ages": result[f"{prefix}_ages"],
            "valid": result[f"{prefix}_valid"],
        })
        result[f"{prefix}_tokens"] = memory["tokens"]
        result[f"{prefix}_ages"] = memory["ages"]
        result[f"{prefix}_valid"] = memory["valid"]
    return result


def _training_index(source: str, match: str) -> int:
    return [path.stem for path in dt.SPLITS[source]["train"]].index(match)


def _training_views(inputs: dict, source: str, match: str,
                    calibration: dict) -> dict[str, dict]:
    index = _training_index(source, match)
    seed = lt.SEED + index
    legacy = robust.masked_inputs(inputs, seed)
    gap = lc.ball_gap_view(inputs, calibration, ce.seed(match, "train"))
    return {
        "clean": inputs,
        "gap": gap,
        "dropout_legacy": gr.slot_view(legacy, seed),
        "dropout_gap": gr.slot_view(gap, seed),
    }


def _create_memory_cache(item: dict, calibration: dict,
                         cache_path: Path, receipt_path: Path,
                         declaration_sha256: str) -> dict[str, str]:
    """Prepare four unnormalized, tracking-only memory views for one train match."""
    input_path = prep.data_path(item, DATA)
    if sha256(input_path) != item["sha256"]:
        raise ValueError(f"Training input changed: {item['match']}")
    if cache_path.exists() or receipt_path.exists():
        if not cache_path.is_file() or not receipt_path.is_file():
            raise ValueError(f"Incomplete contact cache: {item['match']}")
        receipt = _json(receipt_path)
        if (receipt.get("cache_sha256") != sha256(cache_path) or
                receipt.get("declaration_sha256") != declaration_sha256 or
                receipt.get("input_sha256") != item["sha256"]):
            raise ValueError(f"Contact memory cache changed: {item['match']}")
        return receipt
    inputs, labels = ms.view(input_path)
    views = _training_views(inputs, item["source"], item["match"], calibration)
    prep_spec = _json(BASE_CACHE / "predeclared.json")
    training_cache = prep_spec["prepared_training_caches"][item["match"]]
    training_cache_path = ROOT / training_cache["cache"]
    if sha256(training_cache_path) != training_cache["sha256"]:
        raise ValueError(f"Baseline training cache changed: {item['match']}")
    cached = joblib.load(training_cache_path)
    arrays = {}
    for view_name, viewed in views.items():
        indices = lt.sampled_indices(viewed)
        rebuilt = old.feature_view(viewed, labels,
                                   item["source"], item["match"])["joint"]
        reference = cached["records"][view_name]
        rebuilt_x = np.asarray(rebuilt["x"], dtype=np.float32)
        if (list(rebuilt["names"]) != list(reference["names"]) or
                not np.array_equal(rebuilt["indices"], reference["indices"]) or
                not np.array_equal(rebuilt["y"], reference["y"]) or
                not np.array_equal(rebuilt_x, reference["x"], equal_nan=True)):
            raise ValueError(f"Regenerated augmented feature parity failed: {item['match']}/{view_name}")
        if not np.array_equal(indices, reference["indices"]):
            raise ValueError(f"Augmented contact sample clock differs: {item['match']}/{view_name}")
        memory = cm.memory_features(viewed, indices)
        for key, value in memory.items():
            arrays[f"{view_name}_{key}"] = value
        arrays[f"{view_name}_indices"] = indices.astype(np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{cache_path.name}.", suffix=".npz",
                                     dir=cache_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    receipt = {
        "match": item["match"], "source": item["source"],
        "input_sha256": item["sha256"],
        "cache_sha256": sha256(cache_path),
        "declaration_sha256": declaration_sha256,
        "rows_per_view": {name: int(len(arrays[f"{name}_indices"]))
                          for name in views},
        "training_labels_used_for_feature_parity": True,
        "memory_builder_label_independent": True,
    }
    _write_exclusive(receipt_path, receipt)
    return receipt


def _load_memory_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    with np.load(path, allow_pickle=False) as archive:
        for view in ("clean", "gap", "dropout_legacy", "dropout_gap"):
            result[view] = {
                "tokens": archive[f"{view}_tokens"],
                "ages": archive[f"{view}_ages"],
                "valid": archive[f"{view}_valid"],
                "indices": archive[f"{view}_indices"],
            }
    return result


def _fold_training_records(heldout: str, declaration: dict) -> list[dict]:
    plan = prep.split_plan(fold=f"leave_out_{heldout}")
    prep_spec = _json(BASE_CACHE / "predeclared.json")
    records = []
    for item in plan["train"]:
        cache_record = prep_spec["prepared_training_caches"][item["match"]]
        cache_path = ROOT / cache_record["cache"]
        receipt_path = ROOT / cache_record["receipt"]
        if (sha256(cache_path) != cache_record["sha256"] or
                sha256(receipt_path) != cache_record["receipt_sha256"]):
            raise ValueError(f"Prepared baseline cache changed: {item['match']}")
        cache = joblib.load(cache_path)
        if (cache.get("input_sha256") != item["sha256"] or
                cache.get("match") != item["match"] or
                cache.get("source") != item["source"]):
            raise ValueError(f"Prepared baseline cache identity mismatch: {item['match']}")
        memory_path = OUT / f"memory_train_{item['match']}.npz"
        memory_receipt_path = memory_path.with_suffix(".json")
        memory_receipt = _json(memory_receipt_path)
        if (memory_receipt.get("cache_sha256") != sha256(memory_path) or
                memory_receipt.get("declaration_sha256") !=
                declaration["_declaration_sha256"]):
            raise ValueError(f"Prepared contact cache changed: {item['match']}")
        memories = _load_memory_cache(memory_path)
        base_records = deployment.paired_records(cache, item["match"])
        mapping = (("clean", "gap"), ("clean", "gap"),
                   ("dropout_legacy", "dropout_gap"),
                   ("dropout_legacy", "dropout_gap"))
        for position, record in enumerate(base_records):
            original_view, alternate_view = mapping[position]
            base_mem = memories[original_view]
            alt_mem = memories[alternate_view]
            if not np.array_equal(record["indices"], base_mem["indices"]) or not np.array_equal(
                    record["indices"], alt_mem["indices"]):
                raise ValueError(f"Contact sample clock mismatch: {item['match']}/{position}")
            augmented = _memory_record(record, base_mem, alt_mem)
            if position % 2:
                augmented = _swapped_record_memories(augmented)
            augmented.update(full_y=np.asarray(cache["full_y"]),
                             full_period=np.asarray(cache["full_period"]),
                             fps=float(cache["fps"]))
            records.append(augmented)
    expected = sum(4 for item in plan["train"])
    if len(records) != expected:
        raise ValueError(f"Expected {expected} matched records, got {len(records)}")
    return records


def _fit_samples(records: list[dict], fitted_base: dict, base_model,
                 seed: int) -> dict[str, np.ndarray]:
    """Cache frozen backbone outputs for each already-declared training window."""
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    original = [lt.normalize(record["x"], fitted_base["median"],
                             fitted_base["scale"]) for record in records]
    alternate = [lt.normalize(record["alternate"], fitted_base["median"],
                              fitted_base["scale"]) for record in records]
    samples = [
        (index, start, size, full, bool(use))
        for index, record in enumerate(records)
        for (start, size, full), use in zip(lt.chunks(record),
                                            record["replace_windows"])
    ]
    count = len(samples)
    row_count = lt.CORE + 1
    hidden_rows = np.zeros((count, row_count, 48), dtype=np.float32)
    logits_rows = np.zeros((count, row_count, 2), dtype=np.float32)
    tokens_rows = np.zeros((count, row_count, 2, len(cm.EVENT_FIELDS)), dtype=np.float32)
    ages_rows = np.zeros((count, row_count, 2), dtype=np.float32)
    valid_rows = np.zeros((count, row_count, 2), dtype=bool)
    base_model.eval()
    with torch.inference_mode():
        for offset in range(0, count, BATCH_SIZE):
            batch = samples[offset:offset+BATCH_SIZE]
            x_values = []
            for index, _, _, full, use in batch:
                x_values.append((alternate if use else original)[index][full])
            x_batch = torch.as_tensor(np.stack(x_values))
            h_batch = base_model.sequence(x_batch)[0]
            logits_batch = base_model.output(h_batch)[..., :2]
            for position, (index, start, size, full, use) in enumerate(batch):
                selected_rows = slice(lt.CONTEXT, lt.CONTEXT + size + 1)
                destination_rows = slice(0, size + 1)
                hidden_rows[offset+position, destination_rows] = (
                    h_batch[position, selected_rows].cpu().numpy())
                logits_rows[offset+position, destination_rows] = (
                    logits_batch[position, selected_rows].cpu().numpy())
                record = records[index]
                prefix = "alternate_memory" if use else "memory"
                tokens = record[f"{prefix}_tokens"][full][selected_rows]
                ages = record[f"{prefix}_ages"][full][selected_rows]
                valid = record[f"{prefix}_valid"][full][selected_rows]
                tokens_rows[offset+position, destination_rows] = tokens
                ages_rows[offset+position, destination_rows] = ages
                valid_rows[offset+position, destination_rows] = valid

    masses = np.asarray([
        1.0 / (sum(other["source"] == record["source"] for other in records)
               * max(1, int(np.sum(record["y"] >= 0))))
        for record in records
    ], dtype=np.float64)
    masses /= masses.mean()
    spell_factors = [ol._record_full_fields(record) for record in records]
    pair_targets = [
        ol.transition_targets(
            record["y"], record["period"], core_size=len(record["y"]),
            full_y=record["full_y"], full_period=record["full_period"],
            indices=record["indices"])
        for record in records
    ]
    live_pair_weights = ol._record_pair_weights(records, pair_targets, "live")
    targets = np.zeros((count, lt.CORE), dtype=np.int64)
    frame_weights = np.zeros((count, lt.CORE), dtype=np.float32)
    spell_weights = np.ones((count, lt.CORE), dtype=np.float32)
    live_targets = np.zeros((count, lt.CORE), dtype=np.float32)
    live_masks = np.zeros((count, lt.CORE), dtype=bool)
    live_weights = np.zeros((count, lt.CORE), dtype=np.float32)
    for row, (index, start, size, _, _) in enumerate(samples):
        record = records[index]
        y = record["y"][start:start+size]
        targets[row, :size] = np.maximum(y, 0)
        frame_weights[row, :size] = (y >= 0) * masses[index]
        spell_weights[row, :size] = spell_factors[index][start:start+size]
        live_pair = pair_targets[index]
        for key, destination in (("live_target", live_targets),
                                 ("live_mask", live_masks)):
            destination[row, :size] = live_pair[key][start:start+size]
        live_weights[row, :size] = live_pair_weights[index][start:start+size]
    return {
        "hidden": hidden_rows,
        "base_logits": logits_rows,
        "tokens": tokens_rows,
        "ages": ages_rows,
        "valid": valid_rows,
        "targets": targets,
        "frame_weights": frame_weights,
        "spell_weights": spell_weights,
        "live_targets": live_targets,
        "live_masks": live_masks,
        "live_weights": live_weights,
        "windows": np.asarray(count, dtype=np.int64),
    }


def fit_adapter(records: list[dict], fitted_base: dict, base_model,
                *, seed: int) -> dict:
    """Train one tiny H/A residual while keeping every baseline parameter frozen."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.nn import functional as F

    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    batches = _fit_samples(records, fitted_base, base_model, seed)
    arrays = {key: value for key, value in batches.items() if key != "windows"}

    class EncodedWindows(Dataset):
        def __len__(self):
            return int(batches["windows"])

        def __getitem__(self, index):
            return tuple(torch.from_numpy(arrays[name][index]) for name in (
                "hidden", "base_logits", "tokens", "ages", "valid", "targets",
                "frame_weights", "spell_weights", "live_targets", "live_masks",
                "live_weights"))

    loader = DataLoader(
        EncodedWindows(), batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(int(seed)))
    adapter = build_adapter()
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    history = []
    updates = 0
    for epoch in range(EPOCHS):
        adapter.train()
        numerator = denominator = transition_sum = 0.0
        transition_batches = 0
        for (hidden, base_logits, tokens, ages, valid, targets, frame_weights,
             spell_weights, live_targets, live_masks, live_weights) in loader:
            if not frame_weights.sum():
                continue
            logits = base_logits + adapter(hidden, base_logits, tokens, ages, valid)
            log_q = F.log_softmax(logits, dim=-1)
            live_rows = targets < 2
            core_log_q = log_q[:, :lt.CORE]
            q_loss = F.nll_loss(
                core_log_q.transpose(1, 2), targets.clamp(max=1), reduction="none")
            factor = 1.0 + (spell_weights - 1.0)
            nll = (q_loss * live_rows * frame_weights * factor).sum() / frame_weights.sum()
            transition_terms = []
            if live_masks.any() and live_weights.sum() > 0:
                q = torch.softmax(logits, dim=-1)
                change = q[:, :lt.CORE, 0] * q[:, 1:lt.CORE+1, 1]
                change += q[:, :lt.CORE, 1] * q[:, 1:lt.CORE+1, 0]
                edge_loss = F.binary_cross_entropy(
                    change.clamp(1e-6, 1-1e-6), live_targets, reduction="none")
                transition = ol.group_balanced_binary_loss(
                    edge_loss, live_targets, live_masks, live_weights)
                if transition is not None:
                    transition_terms.append(0.5 * transition)
                    transition_sum += float(transition.detach())
                    transition_batches += 1
            has_weighted_live = bool((frame_weights * live_rows).sum() > 0)
            if not has_weighted_live and not transition_terms:
                continue
            total = nll
            if transition_terms:
                # The fixed dead/live transition objective allocates half its
                # coefficient to each. The dead term has no adapter gradient.
                total = total + TRANSITION_COEFFICIENT * torch.stack(
                    transition_terms).sum()
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), CLIP_NORM)
            optimizer.step()
            updates += 1
            numerator += float((q_loss.detach() * live_rows *
                                frame_weights * factor).sum())
            denominator += float(frame_weights.sum())
        history.append({
            "conditional_nll": numerator / denominator if denominator else 0.0,
            "live_transition": transition_sum / transition_batches
            if transition_batches else None,
            "adapter_updates": updates,
        })
    return {
        "kind": "contact_memory_residual",
        "seed": int(seed),
        "state": {key: value.detach().cpu().numpy()
                  for key, value in adapter.state_dict().items()},
        "history": history,
        "optimizer_updates": updates,
        "windows_per_epoch": int(batches["windows"]),
        "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "base_checkpoint_sha256": fitted_base["_checkpoint_sha256"],
        "base_feature_names": list(fitted_base["feature_names"]),
        "base_input_features": int(fitted_base["input_features"]),
        "normalization_sha256": digest_array(np.concatenate(
            (fitted_base["median"], fitted_base["scale"]))),
        "objective": "conditional H/A combined objective; dead gate frozen",
        "loss": {
            "conditional_spell_weight": "same complete-native-spell sqrt(5s/duration) factor from objective_losses",
            "transition_coefficient": TRANSITION_COEFFICIENT,
            "live_transition_share": TRANSITION_COEFFICIENT / 2,
            "dead_transition": "constant under frozen D path; no adapter gradient",
        },
    }


def _load_adapter(path: Path):
    import torch
    fitted = joblib.load(path)
    if fitted.get("kind") != "contact_memory_residual":
        raise ValueError(f"Unexpected adapter checkpoint: {path}")
    adapter = build_adapter()
    adapter.load_state_dict({key: torch.as_tensor(value)
                             for key, value in fitted["state"].items()})
    adapter.eval()
    return fitted, adapter


def predict(record: dict, memory: dict[str, np.ndarray], fitted_base: dict,
            base_model, adapter) -> np.ndarray:
    """Symmetrize H/A residual predictions while retaining frozen P(D)."""
    import torch

    cm.validate_memory(memory, len(record["x"]))
    if list(record["names"]) != list(fitted_base["feature_names"]):
        raise ValueError("Contact adapter base feature schema differs")
    torch.set_num_threads(1)
    windows = list(lt.chunks(record))
    outputs = []
    base_model.eval()
    adapter.eval()
    for swapped in (False, True):
        if swapped:
            raw = supervised.swap_team_features(record["x"], record["names"])
            swapped_memory = cm.swap_memory(memory)
        else:
            raw = record["x"]
            swapped_memory = memory
        x = lt.normalize(raw, fitted_base["median"], fitted_base["scale"])
        out = np.empty((len(x), 3), dtype=np.float32)
        with torch.inference_mode():
            for offset in range(0, len(windows), BATCH_SIZE):
                batch = windows[offset:offset+BATCH_SIZE]
                x_batch = torch.as_tensor(np.stack([x[full] for _, _, full in batch]))
                token_batch = torch.as_tensor(np.stack([
                    swapped_memory["tokens"][full] for _, _, full in batch]))
                age_batch = torch.as_tensor(np.stack([
                    swapped_memory["ages"][full] for _, _, full in batch]))
                valid_batch = torch.as_tensor(np.stack([
                    swapped_memory["valid"][full] for _, _, full in batch]))
                probabilities = adapted_log_probabilities(
                    base_model, adapter, x_batch, token_batch, age_batch,
                    valid_batch).exp().cpu().numpy()
                for index, (start, size, _) in enumerate(batch):
                    out[start:start+size] = probabilities[
                        index, lt.CONTEXT:lt.CONTEXT+size]
        outputs.append(out[:, [1, 0, 2]] if swapped else out)
    return (outputs[0] + outputs[1]) * .5


if __name__ == "__main__":
    main()


