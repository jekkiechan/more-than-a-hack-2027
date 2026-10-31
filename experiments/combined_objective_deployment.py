"""Fit two pooled combined-loss deployment members from frozen train caches.

This runner fits only seeds 2026 and 2027 for the already completed combined
objective. It reads the thirteen TRAIN records only. It does not score matches,
load selection labels, or perform organizer inference.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import threading
import time
import warnings

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")).resolve()
OUT = DATA / "experiments/combined_objective_deployment_v1"
CACHE_ROOT = DATA / "experiments/objective_alignment_v1"
MODEL_ROOT = DATA / "models/joint_gru_seed_ensemble_v1"
SEEDS = (2026, 2027)
BATCH_SIZE = 64
OBJECTIVE = "combined"
SEAL = OUT / "checkpoints_frozen.json"

from experiments import (corruption_experiment as ce, factorized_had as fh,
                         learned_temporal as lt, light_corruption as lc,
                         objective_losses, preparation as prep, supervised)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


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


def _source_paths() -> tuple[Path, ...]:
    return tuple(ROOT / relative for relative in (
        "experiments/combined_objective_deployment.py",
        "experiments/objective_losses.py",
        "experiments/factorized_had.py",
        "experiments/learned_temporal.py",
        "experiments/light_corruption.py",
        "experiments/corruption_experiment.py",
        "experiments/supervised.py",
        "experiments/preparation.py",
        "experiments/eval_set/manifest.json",
        "experiments/baseline_manifest.json",
        "experiments/model_decision_policy.json",
        "experiments/seed_ensemble_deployment.py",
    ))


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()}


def _matches_frozen_sources(actual: dict[str, str], frozen: dict[str, str]) -> bool:
    """Allow the exact archived pre-promotion baseline/policy after promotion."""
    if actual.keys() != frozen.keys():
        return False
    mutable_archives = {
        "experiments/baseline_manifest.json":
            "experiments/baselines/joint_gru_seed_ensemble_v1.json",
        "experiments/model_decision_policy.json":
            "experiments/baselines/model_decision_policy_joint_gru_seed_ensemble_v1.json",
    }
    for relative, expected in frozen.items():
        if actual[relative] == expected:
            continue
        archive = mutable_archives.get(relative)
        if archive is None or not (ROOT / archive).is_file():
            return False
        if sha256(ROOT / archive) != expected:
            return False
    return True


def _pooled_plan() -> dict:
    plan = prep.split_plan(fold="pooled")
    if len(plan["train"]) != 13 or len(plan["selection"]) != 5:
        raise ValueError("Expected the fixed 13-match pooled TRAIN plan and five selections")
    if len({item["match"] for item in plan["train"]}) != 13:
        raise ValueError("Pooled TRAIN plan contains duplicate matches")
    return plan


def _cache_file_info(plan: dict, preparation: dict) -> list[dict]:
    info = []
    declared_caches = preparation["prepared_training_caches"]
    if set(declared_caches) != {item["match"] for item in plan["train"]}:
        raise ValueError("Prepared cache list does not equal the pooled 13-match TRAIN plan")
    for item in plan["train"]:
        cache_record = declared_caches[item["match"]]
        cache_path = ROOT / cache_record["cache"]
        receipt_path = ROOT / cache_record["receipt"]
        input_path = prep.data_path(item, DATA)
        if sha256(input_path) != item["sha256"]:
            raise ValueError(f"Pooled training source changed: {item['match']}")
        if (cache_path.stat().st_size != cache_record["bytes"] or
                sha256(cache_path) != cache_record["sha256"] or
                sha256(receipt_path) != cache_record["receipt_sha256"]):
            raise ValueError(f"Prepared training cache changed: {item['match']}")
        receipt = _json(receipt_path)
        if (receipt.get("sha256") != cache_record["sha256"] or
                receipt.get("input_sha256") != item["sha256"] or
                receipt.get("preparation_sha256") != preparation["preparation_declaration_sha256"] or
                receipt.get("source") != item["source"] or
                receipt.get("match") != item["match"]):
            raise ValueError(f"Prepared training receipt mismatch: {item['match']}")
        info.append({
            "source": item["source"], "match": item["match"],
            "input": str(input_path.relative_to(ROOT)),
            "input_sha256": sha256(input_path),
            "cache": cache_record["cache"],
            "cache_bytes": cache_record["bytes"],
            "cache_sha256": cache_record["sha256"],
            "receipt": cache_record["receipt"],
            "receipt_sha256": cache_record["receipt_sha256"],
        })
    return info


def _copy_legacy_mixed_record(original: dict, alternate: dict, match: str,
                              variant: int, swap: bool) -> dict:
    """Copy the legacy gr.paired_records window wrapper without importing gr."""
    result = dict(original)
    windows = list(lt.chunks(result))
    chosen = lc.chosen(len(windows), ce.seed(match, f"live_window_{variant}"))
    replace_rows = np.zeros(len(result["y"]), dtype=bool)
    for use, (start, size, _) in zip(chosen, windows):
        replace_rows[start:start+size] = use
    result.update(alternate=alternate["x"], physical_x=alternate["x"],
                  replace_windows=chosen, replace_rows=replace_rows)
    if swap:
        result["x"] = supervised.swap_team_features(result["x"], result["names"])
        result["alternate"] = supervised.swap_team_features(
            result["alternate"], result["names"])
        result["y"] = np.where(np.isin(result["y"], (0, 1)), 1-result["y"], result["y"])
    return result


def paired_records(cache: dict, match: str) -> list[dict]:
    """Four fixed legacy slot-dropout records with full-native training targets."""
    required_views = {"clean", "gap", "dropout_legacy", "dropout_gap"}
    if set(cache.get("records", {})) != required_views:
        raise ValueError(f"Expected the four cached joint views: {sorted(required_views)}")
    records = []
    for variant in (0, 1):
        original = "clean" if variant == 0 else "dropout_legacy"
        alternate = "gap" if variant == 0 else "dropout_gap"
        for swap in (False, True):
            record = _copy_legacy_mixed_record(
                cache["records"][original], cache["records"][alternate],
                match, variant, swap)
            record["full_y"] = np.asarray(cache["full_y"])
            record["full_period"] = np.asarray(cache["full_period"])
            record["fps"] = float(cache["fps"])
            records.append(record)
    return records


def _load_records(plan: dict) -> list[dict]:
    records = []
    prep_spec = _json(CACHE_ROOT / "predeclared.json")
    for item in plan["train"]:
        entry = prep_spec["prepared_training_caches"][item["match"]]
        cache = joblib.load(ROOT / entry["cache"])
        if (cache.get("match") != item["match"] or cache.get("source") != item["source"] or
                cache.get("input_sha256") != item["sha256"] or
                cache.get("preparation_sha256") != prep_spec["preparation_declaration_sha256"]):
            raise ValueError(f"Prepared cache metadata mismatch: {item['match']}")
        native_y = np.asarray(cache.get("full_y"))
        native_period = np.asarray(cache.get("full_period"))
        if (native_y.ndim != 1 or native_period.shape != native_y.shape or
                not np.issubdtype(native_y.dtype, np.integer)):
            raise ValueError(f"Missing full-native labels or periods: {item['match']}")
        for name in ("clean", "gap", "dropout_legacy", "dropout_gap"):
            base = cache["records"][name]
            if (base.get("source") != item["source"] or base.get("match") != item["match"] or
                    len(base["y"]) != len(base["x"]) or
                    not np.array_equal(base["period"], native_period[base["indices"]]) or
                    not np.array_equal(base["y"], native_y[base["indices"]])):
                raise ValueError(f"Cached joint record labels/schema do not align: {item['match']}/{name}")
        records.extend(paired_records(cache, item["match"]))
        del cache
    if len(records) != 52:
        raise ValueError(f"Expected 52 legacy paired records from thirteen matches, got {len(records)}")
    return records


def _expected_updates(records: list[dict], seed: int) -> tuple[int, int]:
    import torch
    from torch.utils.data import DataLoader, Dataset

    valid = [bool(np.any(record["y"][start:start+size] >= 0))
             for record in records for start, size, _ in lt.chunks(record)]

    class ValidWindows(Dataset):
        def __len__(self):
            return len(valid)

        def __getitem__(self, index):
            return valid[index]

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(ValidWindows(), batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=0, generator=generator)
    updates = 0
    for _ in range(lt.EPOCHS):
        for batch in loader:
            updates += int(bool(batch.any()))
    return len(valid), updates


def _no_fit_contract(records: list[dict], controls: dict[int, dict],
                     control_hashes: dict[int, str]) -> dict:
    import torch

    if not records or len(records) != 52:
        raise ValueError("The pooled no-fit contract requires the exact 52 paired records")
    names = list(records[0]["names"])
    if len(names) != 204 or any(list(record["names"]) != names for record in records):
        raise ValueError("Expected one shared 204-feature joint schema")
    mixed = [dict(record, x=np.where(record["replace_rows"][:, None],
                                     record["alternate"], record["x"]))
             for record in records]
    median, scale = lt.normalization(mixed)
    del mixed
    columns = fh.player_columns(names)
    with torch.random.fork_rng(devices=[]):
        network = fh.network(len(median) * 2, columns, player_only=True)
        parameters = sum(parameter.numel() for parameter in network.parameters())
        del network
    windows_and_updates = {seed: _expected_updates(records, seed) for seed in SEEDS}
    result = {
        "fit_performed": False,
        "record_count": len(records),
        "training_matches": sorted({record["match"] for record in records}),
        "records_per_match": {match: sum(r["match"] == match for r in records)
                              for match in sorted({r["match"] for r in records})},
        "feature_names": names,
        "feature_names_sha256": fingerprint(names),
        "n_features": len(names),
        "network_input_features": len(median) * 2,
        "parameters": int(parameters),
        "normalization": {
            "median_sha256": array_sha256(median),
            "scale_sha256": array_sha256(scale),
            "median_dtype": median.dtype.str,
            "scale_dtype": scale.dtype.str,
        },
        "seeds": {},
    }
    for seed in SEEDS:
        windows, updates = windows_and_updates[seed]
        control = controls[seed]
        checks = {
            "median": np.array_equal(median, control["median"]),
            "scale": np.array_equal(scale, control["scale"]),
            "feature_names": names == control["feature_names"],
            "player_only": control["player_only"] is True,
            "parameters": int(parameters) == int(control["parameters"]),
            "windows_per_epoch": windows == int(control["windows_per_epoch"]),
            "optimizer_updates": updates == int(control["optimizer_updates"]),
        }
        if not all(checks.values()):
            raise ValueError(f"Pooled pre-fit parity failed for seed {seed}: {checks}")
        result["seeds"][str(seed)] = {
            "checks": checks,
            "windows_per_epoch": int(windows),
            "optimizer_updates": int(updates),
            "parameters": int(parameters),
            "control_checkpoint_sha256": control_hashes[seed],
        }
    return result


def _control_manifest_path(expected_sha256: str | None = None) -> Path:
    active = ROOT / "experiments/baseline_manifest.json"
    if expected_sha256 is None or sha256(active) == expected_sha256:
        return active
    archived = ROOT / "experiments/baselines/joint_gru_seed_ensemble_v1.json"
    if archived.is_file() and sha256(archived) == expected_sha256:
        return archived
    raise ValueError("Frozen control manifest is neither active nor exactly archived")


def _load_verified_controls(expected_manifest_sha256: str | None = None) -> tuple[dict[int, dict], dict[int, str]]:
    manifest_path = _control_manifest_path(expected_manifest_sha256)
    manifest = _json(manifest_path)
    if manifest.get("package_version") != "joint_gru_seed_ensemble_v1":
        raise ValueError("The active package is no longer joint_gru_seed_ensemble_v1")
    controls, hashes = {}, {}
    for seed in SEEDS:
        name = f"joint_gru_{seed}.joblib"
        path = MODEL_ROOT / name
        expected = manifest["models_sha256"].get(name)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Restored control checkpoint is missing or changed: {name}")
        model = joblib.load(path)
        if model.get("seed") != seed or model.get("kind") != "factorized_had":
            raise ValueError(f"Unexpected control model metadata: {name}")
        controls[seed] = model
        hashes[seed] = expected
    return controls, hashes


def _build_declaration() -> dict:
    plan = _pooled_plan()
    prep_spec_path = CACHE_ROOT / "predeclared.json"
    prep_spec = _json(prep_spec_path)
    training = _cache_file_info(plan, prep_spec)
    controls, control_hashes = _load_verified_controls()
    records = _load_records(plan)
    contract = _no_fit_contract(records, controls, control_hashes)

    baseline_manifest = ROOT / "experiments/baseline_manifest.json"
    package_manifest = MODEL_ROOT / "baseline_manifest.json"
    if sha256(package_manifest) != sha256(baseline_manifest):
        raise ValueError("Restored model package manifest differs from the active manifest")
    preparation_files = []
    for filename in ("predeclared.json", "preparation_declaration.json",
                     "prepared_training_manifest.json", "training_parity.json"):
        path = CACHE_ROOT / filename
        preparation_files.append({
            "path": str(path.relative_to(ROOT)), "sha256": sha256(path),
        })
    old_deployment = ROOT / "experiments/results/seed_ensemble_deployment_declaration.json"
    if not old_deployment.is_file():
        raise ValueError("Current pooled deployment declaration is missing")
    return {
        "schema_version": 1,
        "study_id": "combined_objective_deployment_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pooled deployment members for the completed combined objective.",
        "training_scope": "Exactly the thirteen TRAIN matches from the canonical pooled whole-match plan.",
        "seeds": list(SEEDS),
        "objective": OBJECTIVE,
        "loss_coefficients": {
            "spell": float(objective_losses.OBJECTIVES[OBJECTIVE][0]),
            "transition": float(objective_losses.OBJECTIVES[OBJECTIVE][1]),
        },
        "recipe": {
            "records": "Exact cached objective-alignment records; legacy gr.paired_records slot_dropout pairing, direct and H/A-swapped.",
            "full_native_targets": "Attach each cached training match's full_y/full_period/fps for train-only spell weights and transition masks.",
            "model": "Factorized player-only-D BiGRU, unchanged 204-feature schema, hidden 24.",
            "training": {"epochs": int(lt.EPOCHS), "batch_size": BATCH_SIZE,
                         "learning_rate": .001, "weight_decay": .0001,
                         "gradient_clip": 1.0, "core": int(lt.CORE),
                         "context": int(lt.CONTEXT), "rate_hz": 5,
                         "normalization": "Training-only median/std on the legacy mixed records."},
            "decoder": {"switch_cost": int(lt.COST)},
        },
        "pool_plan": plan,
        "split_plan": plan,
        "pool_plan_sha256": prep.fingerprint(plan),
        "split_plan_sha256": prep.fingerprint(plan),
        "training_matches": [item["match"] for item in plan["train"]],
        "excluded_selection_matches": [item["match"] for item in plan["selection"]],
        "training_cache_files": training,
        "preparation_receipts": preparation_files,
        "control_package": {
            "baseline_manifest": str(baseline_manifest.relative_to(ROOT)),
            "baseline_manifest_sha256": sha256(baseline_manifest),
            "decision_policy": "experiments/model_decision_policy.json",
            "decision_policy_sha256": sha256(ROOT / "experiments/model_decision_policy.json"),
            "model_directory": str(MODEL_ROOT.relative_to(DATA)),
            "model_package_manifest_sha256": sha256(package_manifest),
            "checkpoints": {
                str(seed): {
                    "path": str((MODEL_ROOT / f"joint_gru_{seed}.joblib").relative_to(ROOT)),
                    "sha256": control_hashes[seed],
                }
                for seed in SEEDS
            },
            "prior_pooled_fit_declaration_sha256": sha256(old_deployment),
        },
        "prefit_parity": contract,
        "source_sha256": _source_hashes(),
        "organizer_labels_read": False,
        "selection_data_loaded": False,
        "fit_performed": False,
    }


def freeze() -> dict:
    path = OUT / "predeclared.json"
    if path.exists():
        check()
        return _json(path)
    spec = _build_declaration()
    OUT.mkdir(parents=True, exist_ok=True)
    _write_exclusive(path, spec)
    return spec


def check() -> dict:
    path = OUT / "predeclared.json"
    if not path.is_file():
        raise ValueError("No frozen deployment declaration; freeze before fitting")
    spec = _json(path)
    if not _matches_frozen_sources(_source_hashes(), spec.get("source_sha256", {})):
        raise ValueError("Frozen deployment source changed")
    if prep.fingerprint(spec["pool_plan"]) != spec.get("pool_plan_sha256"):
        raise ValueError("Frozen pooled whole-match plan changed")
    if prep.fingerprint(_pooled_plan()) != spec.get("pool_plan_sha256"):
        raise ValueError("Current canonical pooled plan differs from the frozen plan")
    # Verify every raw TRAIN input, prepared-cache payload and receipt against
    # the declaration. Selection records are named in the plan but never loaded.
    for entry in spec["training_cache_files"]:
        for key in ("input", "cache", "receipt"):
            path_value = ROOT / entry[key]
            if sha256(path_value) != entry[f"{key}_sha256"]:
                raise ValueError(f"Frozen training input changed: {entry[key]}")
    for item in spec["preparation_receipts"]:
        path_value = ROOT / item["path"]
        if sha256(path_value) != item["sha256"]:
            raise ValueError(f"Frozen preparation receipt changed: {item['path']}")
    manifest = _control_manifest_path(spec["control_package"]["baseline_manifest_sha256"])
    policy_hash = spec["control_package"]["decision_policy_sha256"]
    active_policy = ROOT / "experiments/model_decision_policy.json"
    if sha256(active_policy) != policy_hash:
        archived_policy = ROOT / "experiments/baselines/model_decision_policy_joint_gru_seed_ensemble_v1.json"
        if not archived_policy.is_file() or sha256(archived_policy) != policy_hash:
            raise ValueError("Frozen control decision policy changed or was not exactly archived")
    package_manifest = MODEL_ROOT / "baseline_manifest.json"
    if sha256(package_manifest) != spec["control_package"]["model_package_manifest_sha256"]:
        raise ValueError("Restored control package manifest changed")
    _, control_hashes = _load_verified_controls(spec["control_package"]["baseline_manifest_sha256"])
    for seed in SEEDS:
        if control_hashes[seed] != spec["control_package"]["checkpoints"][str(seed)]["sha256"]:
            raise ValueError(f"Frozen control checkpoint changed for seed {seed}")
    return spec


_NORMALIZE_LOCK = threading.RLock()


@contextmanager
def file_backed_normalization(temp_parent: Path | None = None):
    """Context-scoped, byte-preserving memmaps for objective_losses.fit only."""
    _NORMALIZE_LOCK.acquire()
    original = objective_losses.lt.normalize
    temporary = None
    audit = {"normalization_calls": [], "normalizer_restored": False,
             "temporary_files_removed": False}
    try:
        if temp_parent is not None:
            Path(temp_parent).mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix="combined_normalize_", dir=str(temp_parent) if temp_parent else None)
        folder = Path(temporary.name)

        def mapped_normalize(x, median, scale):
            result = original(x, median, scale)
            index = len(audit["normalization_calls"])
            path = folder / f"normalized_{index:04d}.npy"
            with path.open("wb") as stream:
                np.save(stream, result, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            mapped = np.load(path, mmap_mode="r", allow_pickle=False)
            if (mapped.dtype != result.dtype or mapped.shape != result.shape or
                    not np.array_equal(mapped, result) or mapped.flags.writeable):
                raise RuntimeError("File-backed normalization changed normalized values")
            audit["normalization_calls"].append({
                "shape": list(mapped.shape), "dtype": mapped.dtype.str,
                "sha256": array_sha256(mapped), "bytes": int(mapped.nbytes),
                "readonly": True,
            })
            return mapped

        objective_losses.lt.normalize = mapped_normalize
        yield audit
    finally:
        objective_losses.lt.normalize = original
        audit["normalizer_restored"] = objective_losses.lt.normalize is original
        if temporary is not None:
            temporary.cleanup()
            audit["temporary_files_removed"] = not Path(temporary.name).exists()
        _NORMALIZE_LOCK.release()


def _assert_fitted_contract(model: dict, spec: dict, seed: int) -> None:
    expected = spec["prefit_parity"]["seeds"][str(seed)]
    control = _load_verified_controls(
        spec["control_package"]["baseline_manifest_sha256"])[0][seed]
    if (model.get("objective") != OBJECTIVE or model.get("seed") != seed or
            model.get("player_only") is not True or
            not np.array_equal(model["median"], control["median"]) or
            not np.array_equal(model["scale"], control["scale"]) or
            model["feature_names"] != spec["prefit_parity"]["feature_names"] or
            model["parameters"] != expected["parameters"] or
            model["windows_per_epoch"] != expected["windows_per_epoch"] or
            model["optimizer_updates"] != expected["optimizer_updates"]):
        raise ValueError(f"Fitted seed {seed} changed a frozen architecture or budget contract")


def fit_seed(seed: int) -> dict:
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"Expected seed 2026 or 2027, got {seed}")
    spec = check()
    destination = OUT / f"combined_objective_{seed}.joblib"
    receipt_path = destination.with_suffix(".json")
    if destination.exists() or receipt_path.exists():
        return verify_model(seed)
    records = _load_records(spec["pool_plan"])
    controls, control_hashes = _load_verified_controls(
        spec["control_package"]["baseline_manifest_sha256"])
    current_contract = _no_fit_contract(records, controls, control_hashes)
    if current_contract != spec["prefit_parity"]:
        raise ValueError("Pooled no-fit parity differs from the frozen declaration")
    started = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with file_backed_normalization(OUT) as adapter_audit:
            fitted = objective_losses.fit(records, player_only=True, seed=seed,
                                          objective=OBJECTIVE)
    _assert_fitted_contract(fitted, spec, seed)
    if (not adapter_audit["normalizer_restored"] or
            not adapter_audit["temporary_files_removed"] or
            len(adapter_audit["normalization_calls"]) != 2 * len(records)):
        raise ValueError("File-backed normalization runtime contract failed")
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=OUT, delete=False) as stream:
        temporary_model = Path(stream.name)
    try:
        joblib.dump(fitted, temporary_model)
        os.replace(temporary_model, destination)
    finally:
        temporary_model.unlink(missing_ok=True)
    receipt = {
        "model": destination.name,
        "sha256": sha256(destination),
        "predeclared_sha256": sha256(OUT / "predeclared.json"),
        "seed": seed,
        "objective": OBJECTIVE,
        "loss_coefficients": {
            "spell": float(objective_losses.OBJECTIVES[OBJECTIVE][0]),
            "transition": float(objective_losses.OBJECTIVES[OBJECTIVE][1]),
        },
        "deployment": "pooled_external_train_only",
        "training_matches": list(spec["training_matches"]),
        "training_record_count": len(records),
        "excluded_selection_matches": list(spec["excluded_selection_matches"]),
        "parameters": int(fitted["parameters"]),
        "windows_per_epoch": int(fitted["windows_per_epoch"]),
        "optimizer_updates": int(fitted["optimizer_updates"]),
        "normalization_matched_control": True,
        "normalization_sha256": spec["prefit_parity"]["normalization"],
        "file_backed_normalization": adapter_audit,
        "seconds": time.monotonic() - started,
        "organizer_labels_read": False,
        "selection_data_loaded": False,
        "external_scoring_performed": False,
    }
    _write_exclusive(receipt_path, receipt)
    return receipt


def _verify_receipt(seed: int, path: Path) -> dict:
    spec = check()
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file():
        raise ValueError(f"Missing pooled deployment model or receipt: {path}")
    receipt = _json(receipt_path)
    if (receipt.get("sha256") != sha256(path) or
            receipt.get("predeclared_sha256") != sha256(OUT / "predeclared.json") or
            receipt.get("seed") != seed or receipt.get("objective") != OBJECTIVE or
            receipt.get("deployment") != "pooled_external_train_only" or
            receipt.get("training_matches") != spec["training_matches"] or
            receipt.get("excluded_selection_matches") != spec["excluded_selection_matches"] or
            receipt.get("organizer_labels_read") is not False or
            receipt.get("selection_data_loaded") is not False or
            receipt.get("external_scoring_performed") is not False):
        raise ValueError(f"Pooled deployment fit receipt mismatch: {receipt_path}")
    model = joblib.load(path)
    _assert_fitted_contract(model, spec, seed)
    return receipt


def verify_model(seed: int, path: Path | None = None) -> dict:
    """Validate a promoted-runner input path against its exact frozen fit receipt."""
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"Expected seed 2026 or 2027, got {seed}")
    expected = OUT / f"combined_objective_{seed}.joblib"
    actual = Path(path).resolve() if path is not None else expected.resolve()
    if actual != expected.resolve():
        raise ValueError("Model path is not the sealed pooled combined-objective output")
    receipt = _verify_receipt(seed, expected)
    manifest_path = SEAL
    if not manifest_path.is_file():
        raise ValueError("Pooled deployment checkpoints have not been sealed")
    manifest = _json(manifest_path)
    row = next((item for item in manifest.get("checkpoints", [])
                if item.get("seed") == seed), None)
    if (manifest.get("predeclared_sha256") != sha256(OUT / "predeclared.json") or
            row is None or row.get("file") != expected.name or
            row.get("sha256") != receipt["sha256"] or
            row.get("receipt_sha256") != sha256(expected.with_suffix(".json"))):
        raise ValueError(f"Sealed deployment manifest does not bind seed {seed}")
    return receipt


def load_model(seed: int):
    verify_model(seed)
    return joblib.load(OUT / f"combined_objective_{int(seed)}.joblib")


def seal() -> dict:
    check()
    rows = []
    for seed in SEEDS:
        path = OUT / f"combined_objective_{seed}.joblib"
        receipt = _verify_receipt(seed, path)
        rows.append({"seed": seed, "file": path.name, "sha256": receipt["sha256"],
                     "receipt": path.with_suffix(".json").name,
                     "receipt_sha256": sha256(path.with_suffix(".json"))})
    manifest = {
        "study_id": "combined_objective_deployment_v1",
        "predeclared_sha256": sha256(OUT / "predeclared.json"),
        "objective": OBJECTIVE,
        "seeds": list(SEEDS),
        "checkpoints": rows,
        "all_training_only": True,
        "external_scoring_performed": False,
    }
    path = SEAL
    if path.exists():
        if _json(path) != manifest:
            # created_utc is intentionally absent; the seal is deterministic.
            raise ValueError("Sealed pooled checkpoints changed")
    else:
        _write_exclusive(path, manifest)
    for seed in SEEDS:
        verify_model(seed)
    return manifest


def checkpoint(seed: int) -> Path:
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"Expected seed 2026 or 2027, got {seed}")
    return OUT / f"combined_objective_{seed}.joblib"


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "check", "fit", "seal"))
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze()
        print(json.dumps({"declaration_sha256": sha256(OUT / "predeclared.json"),
                          "matches": len(result["training_matches"]),
                          "fit_performed": False}, indent=2))
    elif args.action == "check":
        result = check()
        print(json.dumps({"declaration_sha256": sha256(OUT / "predeclared.json"),
                          "fit_performed": False}, indent=2))
    elif args.action == "fit":
        if args.seed is None:
            parser.error("--seed is required for fit")
        print(json.dumps(fit_seed(args.seed), sort_keys=True, indent=2))
    else:
        print(json.dumps(seal(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
