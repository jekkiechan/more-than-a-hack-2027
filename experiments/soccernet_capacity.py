"""Declared capacity/data-scaling continuation for sparse SoccerNet H/A.

This runner is intentionally separate from ``soccernet_extension.py``. It
binds only the explicit source and data dependencies listed here and never
calls the stage-1 global source/seal checks. No public-scoring action is
implemented in this file; the requested public diagnostic remains owned by
``soccernet_capacity_public.py`` after final external model selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import joblib
import numpy as np

from experiments import factorized_had as fh, learned_temporal as lt
from experiments import preparation as prep
from experiments import soccernet_capacity_model as model
from experiments import soccernet_extension as old
from experiments import supervised


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")).resolve()
OUT = DATA / "experiments/soccernet_capacity_v1"
SEGMENTS = OUT / "segments"

STUDY_ID = "soccernet_capacity_v1"
FOLDS = ("leave_out_sportec", "leave_out_metrica", "leave_out_skillcorner")
SEEDS = (2026, 2027)
COARSE_SEED = 2026
WIDTHS = (24, 48, 96)
LEARNING_RATES = (1e-4, 3e-4)
STEPS = (512, 2048)
MILESTONES = (512, 2048)
FRACTION = 0.5
BATCH_SIZE = 32
INPUT_DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
MEMBERS = ("2026", "2027", "ensemble")
VIEWS = old.VIEWS
METRICS = old.METRICS

# Exact source ownership. The public module is included in the declaration
# because it defines the later public path, but this runner never invokes it.
SOURCE_PATHS = (
    "experiments/soccernet_capacity.py",
    "experiments/soccernet_capacity_model.py",
    "experiments/objective_losses.py",
    "experiments/soccernet_capacity_public.py",
    "experiments/soccernet_segments.py",
    "experiments/soccernet_data.py",
    "experiments/soccernet_extension.py",
    "experiments/soccernet_finetune.py",
    "experiments/combined_objective_deployment.py",
    "experiments/preparation.py",
    "experiments/refine_features.py",
    "experiments/refine_domain.py",
    "experiments/robust.py",
    "experiments/supervised.py",
    "experiments/learned_temporal.py",
    "experiments/factorized_had.py",
    "experiments/objective_ensemble.py",
    "experiments/metrics.py",
    "experiments/spell_diagnostics.py",
    "experiments/current_baseline.py",
    "experiments/live_gap_experiment.py",
    "experiments/combined_objective_public_check.py",
    "evaluation/possession_metrics.py",
    "experiments/eval_set/manifest.json",
    "experiments/baseline_manifest.json",
    "experiments/model_decision_policy.json",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Cannot write non-finite JSON value")
    return value


def write_once(path: Path, value: dict) -> None:
    """Create an immutable JSON receipt; identical reruns are idempotent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _jsonable(value)
    if path.exists():
        if read_json(path) != clean:
            raise ValueError(f"Refusing to overwrite sealed artifact: {path}")
        return
    payload = json.dumps(clean, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_joblib_once(path: Path, value: Any, receipt_path: Path,
                       receipt: dict) -> None:
    path = Path(path)
    receipt_path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not receipt_path.is_file():
            raise ValueError(f"Model exists without its immutable receipt: {path}")
        sealed = read_json(receipt_path)
        expected = _jsonable(receipt)
        actual = {key: value for key, value in sealed.items() if key != "model_sha256"}
        if sha256(path) != sealed.get("model_sha256") or actual != expected:
            raise ValueError(f"Existing model does not match its receipt: {path}")
        return
    if receipt_path.exists():
        raise ValueError(f"Receipt exists without its model: {receipt_path}")
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        joblib.dump(value, tmp)
        model_sha = sha256(tmp)
        os.replace(tmp, path)
        write_once(receipt_path, dict(receipt, model_sha256=model_sha))
    finally:
        tmp.unlink(missing_ok=True)


def rate_tag(lr: float) -> str:
    if np.isclose(lr, 1e-4, rtol=0, atol=1e-12):
        return "1e-4"
    if np.isclose(lr, 3e-4, rtol=0, atol=1e-12):
        return "3e-4"
    raise ValueError("Undeclared learning rate")


def recipe_id(width: int, lr: float, steps: int) -> str:
    if width not in WIDTHS or steps not in STEPS:
        raise ValueError("Undeclared width or update budget")
    return f"w{int(width)}_lr{rate_tag(lr)}_step{int(steps)}"


def coarse_configs() -> tuple[dict, ...]:
    return tuple(dict(width=w, lr=lr, steps=step,
                      recipe=recipe_id(w, lr, step))
                 for w, lr, step in itertools.product(WIDTHS, LEARNING_RATES, STEPS))


def initial_path(fold: str, seed: int) -> Path:
    if fold not in (*FOLDS, "pooled") or seed not in SEEDS:
        raise ValueError("Unknown initial checkpoint fold or seed")
    return old.control_path(fold, seed)


def coarse_path(fold: str, width: int, lr: float, step: int,
                seed: int = COARSE_SEED) -> Path:
    if fold not in FOLDS or seed != COARSE_SEED:
        raise ValueError("Coarse fits use a full-provider fold and seed 2026")
    return OUT / "coarse" / recipe_id(width, lr, step) / fold / f"seed{seed}.joblib"


def confirm_path(fold: str, width: int, lr: float, step: int,
                 seed: int = 2027) -> Path:
    if fold not in FOLDS or seed != 2027:
        raise ValueError("Confirmation fits use a full-provider fold and seed 2027")
    return OUT / "confirm" / recipe_id(width, lr, step) / fold / f"seed{seed}.joblib"


def control_fit_path(fold: str, width: int, lr: float, step: int,
                     seed: int) -> Path:
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown matched-control fold or seed")
    return OUT / "old_only" / recipe_id(width, lr, step) / fold / f"seed{seed}.joblib"


def pooled_path(width: int, lr: float, step: int, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError("Pooled seed must be 2026 or 2027")
    return OUT / "pooled" / recipe_id(width, lr, step) / f"seed{seed}.joblib"


def _rel_key(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return "data/" + path.relative_to(DATA).as_posix()
    except ValueError:
        try:
            return "repo/" + path.relative_to(ROOT).as_posix()
        except ValueError as exc:
            raise ValueError(f"Dependency escapes repository/data roots: {path}") from exc


def _resolve_key(key: str) -> Path:
    if key.startswith("data/"):
        return DATA / key.removeprefix("data/")
    if key.startswith("repo/"):
        return ROOT / key.removeprefix("repo/")
    raise ValueError(f"Unknown dependency key: {key}")


def _manifest_file_entries(manifest: dict) -> list[dict]:
    """Normalize the segment manifest's explicit relative file table."""
    entries = manifest.get("files", manifest.get("game_files"))
    if isinstance(entries, dict):
        entries = [dict(path=k, **(v if isinstance(v, dict) else {"sha256": v}))
                   for k, v in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise ValueError("Segment manifest must contain a nonempty files list/map")
    normalized = []
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed segment file receipt")
        rel = entry.get("path", entry.get("file"))
        expected = entry.get("sha256", entry.get("hash"))
        if not isinstance(rel, str) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Segment file receipt requires path and SHA-256")
        p = Path(rel)
        if p.is_absolute() or ".." in p.parts or not rel.startswith("games/"):
            raise ValueError("Segment payload must be under games/ without traversal")
        if rel in seen:
            raise ValueError("Duplicate segment payload path")
        seen.add(rel)
        full = SEGMENTS / p
        if not full.is_file() or sha256(full) != expected:
            raise ValueError(f"Segment payload is absent or hash-mismatched: {rel}")
        normalized.append(dict(path=rel, sha256=expected, bytes=full.stat().st_size))
    if not all(Path(item["path"]).suffix == ".joblib" for item in normalized):
        raise ValueError("Only game-segment joblib payloads are allowed")
    return sorted(normalized, key=lambda item: item["path"])


def _validate_segment_audit(manifest: dict) -> dict:
    """Require preparation's duplicate/label/time checks before training."""
    counts = manifest.get("counts", {})
    if manifest.get("schema") != "soccernet_observed_segments_v1":
        raise ValueError("Unsupported SoccerNet segment manifest schema")
    if manifest.get("split_used") != "train" or manifest.get("organizer_reference_access") is not False:
        raise ValueError("Only the official SoccerNet TRAIN split is admissible")
    recipe = manifest.get("segment_recipe", {})
    feature_recipe = manifest.get("feature_recipe", {})
    anchors = manifest.get("eligible_anchors", {})
    dedup = str(recipe.get("duplicate_policy", ""))
    if not all(term in dedup for term in (
            "Exact row signature equality", "visibility", "geometry conflicts exclude")):
        raise ValueError("Segment recipe does not seal exact geometry/time/visibility duplicate handling")
    breaks = set(recipe.get("breaks", []))
    if not {"uncovered frameNum gap", "period change", "nonmonotonic timestamps",
            "timestamp delta > 1.5/fps", "invalid/conflicting row"}.issubset(breaks):
        raise ValueError("Segment recipe omits required invalid/uncovered/time breaks")
    anchor_recipe = str(recipe.get("anchors", ""))
    if "within 3 native frames" not in anchor_recipe or "conflicting teams" not in anchor_recipe:
        raise ValueError("Anchors must use nearest sampled frame within three native frames")
    if (anchors.get("actions") != ["CROSS", "PASS", "SHOT"] or
            anchors.get("unlabeled_rows") != -1 or anchors.get("D_targets") != 0):
        raise ValueError("GAR anchors must be sparse PASS/CROSS/SHOT H/A only, without D labels")
    if feature_recipe.get("sample_stride") != 6 or "fps/6" not in str(feature_recipe.get("sample_rate_hz", "")):
        raise ValueError("Segment sample rate/stride differs from declared native-grid recipe")
    games = manifest.get("games", [])
    if len(games) != 45 or len(set(games)) != 45:
        raise ValueError(f"Expected exactly 45 official TRAIN games; got {len(games)}")
    metadata_clips = manifest.get("train_metadata_rows")
    eligible = counts.get("anchor_candidates")
    for name, value, expected in (("TRAIN metadata clips", metadata_clips, 62159),
                                  ("eligible event-team anchor candidates", eligible, 42979),
                                  ("clips_read", counts.get("clips_read"), 62159)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) != expected:
            raise ValueError(f"Unexpected {name}: {value}; expected {expected}")
    mapped = counts.get("anchor_mapped_known")
    if isinstance(mapped, bool) or not isinstance(mapped, (int, np.integer)) or not 0 < int(mapped) <= int(eligible):
        raise ValueError(f"Invalid usable mapped-anchor count: {mapped}")
    rejected = {
        "center_unobserved": counts.get("anchor_center_unobserved"),
        "center_invalid_period": counts.get("anchor_center_invalid_period"),
        "center_home_team_metadata_conflict": counts.get("anchor_center_metadata_flag_conflict"),
        "center_team_conflict": counts.get("anchor_unknown_team_conflict"),
        "center_removed_after_merge": counts.get("anchor_center_removed_after_merge"),
        "out_of_grid_tolerance": counts.get("anchor_out_of_grid_tolerance"),
        "grid_conflict": counts.get("anchor_grid_conflict"),
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0
           for value in rejected.values()):
        raise ValueError("Segment manifest lacks classified anchor rejection counts")
    if int(mapped) + sum(map(int, rejected.values())) != int(eligible):
        raise ValueError("Eligible event-team anchors are not fully accounted as mapped or rejected")
    retention = int(mapped) / int(eligible)
    if retention < 0.90:
        raise ValueError(f"Mapped-anchor retention below prefit 90% floor: {retention:.3%}")
    exact_duplicates = counts.get("duplicate_rows_exact")
    conflicting_duplicates = counts.get("duplicate_rows_conflicting")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           for value in (exact_duplicates, conflicting_duplicates)):
        raise ValueError("Segment manifest lacks exact/conflicting duplicate-row audit counts")
    return dict(train_games=len(games), metadata_clips=int(metadata_clips),
        eligible_anchors=int(eligible), mapped_anchors=int(mapped),
        mapped_anchor_retention=retention,
        rejected_anchors={k: int(v) for k, v in rejected.items()},
        duplicate_rows_exact=int(exact_duplicates),
        duplicate_rows_conflicting=int(conflicting_duplicates),
        sample_stride=6, nominal_sample_rate_hz=5,
        validation=dict(duplicate_policy=dedup, breaks=sorted(breaks),
            anchor_policy=anchor_recipe, d_targets=anchors.get("D_targets")))


def load_segments() -> tuple[list[dict], dict, list[dict]]:
    """Load verified game segments from the prepared TRAIN-only cache."""
    manifest_path = SEGMENTS / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing prepared segments manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    entries = _manifest_file_entries(manifest)
    audit = _validate_segment_audit(manifest)
    from experiments import soccernet_segments
    records = soccernet_segments.load_records(output=SEGMENTS)
    if not isinstance(records, list) or not records:
        raise ValueError("soccernet_segments.load_records must return a nonempty list")
    names = list(joblib.load(initial_path("pooled", 2026))["feature_names"])
    groups, ids = model._window_index(records, names, old=False)
    if not groups:
        raise ValueError("Segment cache contains no H/A-supervised windows")
    if len({match for _, match in ids}) != audit["train_games"]:
        raise ValueError("Segment records do not cover exactly the official TRAIN games")
    total_labels = sum(int(np.isin(np.asarray(record["y"]), (0, 1)).sum()) for record in records)
    mapped_anchors = sum(sum(anchor.get("status") == "mapped"
                             for anchor in record.get("anchor_frame_mapping", []))
                         for record in records)
    if mapped_anchors != audit["mapped_anchors"] or total_labels > mapped_anchors:
        raise ValueError("Segment mapped event anchors differ from manifest or sampled labels exceed anchors")
    for record in records:
        if np.any(np.asarray(record["y"]) == 2):
            raise ValueError("GAR segments may contain only H/A anchors and unknown labels")
        if list(record["names"]) != names:
            raise ValueError("Segment feature schema differs from the original model")
        if str(np.asarray(record["source"]).reshape(()).item()).lower() not in (
                "soccernet", "soccernet_gar", "gar"):
            raise ValueError("Unexpected SoccerNet segment source identity")
    return records, manifest, entries


def _source_hashes() -> dict[str, str]:
    result = {}
    for rel in SOURCE_PATHS:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Declared source is missing: {rel}")
        result[rel] = sha256(path)
    return result


def _dependency_paths() -> set[Path]:
    paths: set[Path] = {
        ROOT / "experiments/baseline_manifest.json",
        ROOT / "experiments/model_decision_policy.json",
        ROOT / "experiments/eval_set/manifest.json",
        DATA / "experiments/objective_ensemble_v1/report.json",
        DATA / "experiments/soccernet_extension_v1/report.json",
        DATA / "experiments/soccernet_extension_v1/predeclared.json",
        DATA / "experiments/soccernet_extension_v1/checkpoints_frozen.json",
        DATA / "experiments/soccernet_extension_v1/restore_manifest.json",
        SEGMENTS / "manifest.json",
    }
    # Fold teachers have per-checkpoint receipts. Pooled deployment teachers
    # are authenticated by the active package manifest and promotion receipt,
    # not by fabricated per-model .json files.
    for fold in FOLDS:
        for seed in SEEDS:
            p = initial_path(fold, seed)
            paths.update((p, p.with_suffix(".json")))
    package_dir = DATA / "models/joint_gru_combined_ensemble_v1"
    for seed in SEEDS:
        paths.add(initial_path("pooled", seed))
    paths.update((package_dir / "baseline_manifest.json",
                  ROOT / "experiments/results/baseline_combined_objective_promotion.json"))
    # Stage-1 verified cache inputs: original training cache and receipt pairs.
    spec = old.old_spec()
    plans = {fold: prep.split_plan(fold=fold) for fold in (*FOLDS, "pooled")}
    for plan in plans.values():
        for item in plan["train"]:
            entry = spec["prepared_training_caches"][item["match"]]
            for key in ("cache", "receipt"):
                value = Path(entry[key])
                paths.add(ROOT / value if value.is_absolute() is False else value)
    # The exact selected external cells and their historic probability/decode files.
    for item in old.selection_items():
        for view, replicate in VIEWS:
            paths.add(old.eval_path(item, view, replicate))
            for seed in SEEDS:
                paths.add(old.historical_path(item, view, replicate, seed))
    # All current game payloads are individually bound as well as the manifest.
    manifest = read_json(SEGMENTS / "manifest.json")
    for entry in _manifest_file_entries(manifest):
        paths.add(SEGMENTS / entry["path"])
    return paths


def _freeze_dependencies() -> dict[str, str]:
    result = {}
    for path in sorted(_dependency_paths()):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen study dependency: {path}")
        result[_rel_key(path)] = sha256(path)
    return result


def _verify_initial_checkpoint_provenance(plans: dict[str, dict]) -> dict:
    """Bind fold checkpoints to their fit receipts and pooled teachers to the
    actual active-package manifest and baseline-promotion receipt.
    """
    active_path = ROOT / "experiments/baseline_manifest.json"
    package_manifest_path = DATA / "models/joint_gru_combined_ensemble_v1/baseline_manifest.json"
    promotion_path = ROOT / "experiments/results/baseline_combined_objective_promotion.json"
    active, package, promotion = (read_json(path) for path in
                                  (active_path, package_manifest_path, promotion_path))
    if (active.get("package_version") != "joint_gru_combined_ensemble_v1" or
            package.get("package_version") != active.get("package_version") or
            package.get("models_sha256") != active.get("models_sha256") or
            promotion.get("package") != active.get("package_version") or
            promotion.get("models_sha256") != active.get("models_sha256") or
            promotion.get("baseline_manifest_sha256") != sha256(package_manifest_path)):
        raise ValueError("Pooled teacher package/promotion provenance differs from active baseline")
    model_hashes = active.get("models_sha256", {})
    pooled = {}
    for seed in SEEDS:
        name = f"joint_gru_{seed}.joblib"
        path = initial_path("pooled", seed)
        expected = model_hashes.get(name)
        if not expected or sha256(path) != expected:
            raise ValueError(f"Pooled teacher differs from active package manifest: {name}")
        pooled[str(seed)] = dict(path=_rel_key(path), sha256=expected,
            package_manifest_sha256=sha256(package_manifest_path),
            promotion_receipt_sha256=sha256(promotion_path))

    folds = {}
    for fold in FOLDS:
        plan = plans[fold]
        expected_matches = {item["match"] for item in plan["train"]}
        expected_sources = {item["source"] for item in plan["train"]}
        for seed in SEEDS:
            path = initial_path(fold, seed)
            receipt_path = path.with_suffix(".json")
            if not receipt_path.is_file():
                raise FileNotFoundError(f"Missing fold teacher receipt: {receipt_path}")
            receipt = read_json(receipt_path)
            expected_heldout = fold.removeprefix("leave_out_")
            if (receipt.get("sha256") != sha256(path) or
                    receipt.get("heldout") != expected_heldout or
                    receipt.get("seed") != seed or
                    receipt.get("objective") != "combined" or
                    set(receipt.get("training_matches", [])) != expected_matches or
                    set(receipt.get("sources", [])) != expected_sources):
                raise ValueError(f"Fold teacher receipt does not match declared fold/seed: {receipt_path}")
            folds[f"{fold}:{seed}"] = dict(model_path=_rel_key(path),
                model_sha256=sha256(path), receipt_path=_rel_key(receipt_path),
                receipt_sha256=sha256(receipt_path), heldout=expected_heldout,
                training_matches=sorted(expected_matches), sources=sorted(expected_sources))
    return dict(pooled=pooled, fold_checkpoints=folds,
        package_manifest_path=_rel_key(package_manifest_path),
        package_manifest_sha256=sha256(package_manifest_path),
        promotion_receipt_path=_rel_key(promotion_path),
        promotion_receipt_sha256=sha256(promotion_path))


def _verify_raw_train_source(dataset_root: Path | None = None) -> dict:
    """Cryptographically verify the current TRAIN source at declaration time.

    The multi-gigabyte shard hashes are recorded once here. Per-fit ``check``
    relies on the resulting acquisition receipt and prepared-segment seals, so
    it does not reread the raw shards.
    """
    root = Path(dataset_root) if dataset_root is not None else DATA / "independent/soccernet_gar"
    receipt_path = root / "acquisition_manifest.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Missing SoccerNet acquisition receipt: {receipt_path}")
    receipt_sha = sha256(receipt_path)
    receipt = read_json(receipt_path)
    if (receipt.get("dataset") != "OpenSportsLab/SoccerNet-GAR" or
            receipt.get("branch") != "tracking-full"):
        raise ValueError("Raw source receipt is not pinned to SoccerNet-GAR tracking-full")
    entries = [entry for entry in receipt.get("files", [])
               if str(entry.get("path", "")).startswith("tracking-full/train/")]
    if not entries:
        raise ValueError("Acquisition receipt contains no official TRAIN files")
    file_hashes: dict[str, str] = {}
    seen: set[str] = set()
    for entry in entries:
        rel = entry.get("path")
        if not isinstance(rel, str):
            raise ValueError("Malformed acquisition file path")
        path_part = Path(rel)
        if path_part.is_absolute() or ".." in path_part.parts:
            raise ValueError("Acquisition file path escapes the dataset root")
        if rel in seen:
            raise ValueError(f"Duplicate acquisition receipt path: {rel}")
        seen.add(rel)
        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing content SHA-256 for TRAIN file: {rel}")
        if rel.endswith(".tar") and entry.get("verified_lfs_sha256") is not True:
            raise ValueError(f"TRAIN TAR lacks a verified LFS hash receipt: {rel}")
        if rel.endswith(".parquet") and rel not in (
                "tracking-full/train/metadata.parquet",
                "tracking-full/train/shard_manifest.parquet"):
            raise ValueError(f"Unexpected TRAIN parquet source: {rel}")
        raw_path = root / rel
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing receipt-listed TRAIN source: {rel}")
        expected_size = entry.get("bytes")
        if (isinstance(expected_size, bool) or not isinstance(expected_size, (int, np.integer)) or
                raw_path.stat().st_size != int(expected_size)):
            raise ValueError(f"TRAIN source size differs from acquisition receipt: {rel}")
        actual = sha256(raw_path)
        if actual != expected:
            raise ValueError(f"Current TRAIN content hash differs from acquisition receipt: {rel}")
        file_hashes[rel] = actual
    required = {"tracking-full/train/metadata.parquet",
                "tracking-full/train/shard_manifest.parquet"}
    required.update(f"tracking-full/train/shards/shard-{index:06d}.tar"
                    for index in range(5))
    if set(file_hashes) != required:
        missing, extra = sorted(required - set(file_hashes)), sorted(set(file_hashes) - required)
        raise ValueError(f"TRAIN acquisition file set differs; missing={missing}, extra={extra}")
    tar_files = sorted(rel for rel in file_hashes if rel.endswith(".tar"))
    if sha256(receipt_path) != receipt_sha:
        raise ValueError("Acquisition receipt changed during raw source verification")
    return dict(acquisition_manifest_sha256=receipt_sha,
        verification_utc=datetime.now(timezone.utc).isoformat(),
        acquired_at_utc=receipt.get("acquired_at_utc"),
        dataset=receipt["dataset"], branch=receipt["branch"], revision=receipt.get("revision"),
        file_sha256=dict(sorted(file_hashes.items())), tar_count=len(tar_files))


def freeze() -> dict:
    if (OUT / "predeclared.json").exists():
        return check()
    records, manifest, segment_files = load_segments()
    # Recheck raw bytes after the segment build completed and before the
    # immutable study declaration is written.
    raw_source = _verify_raw_train_source()
    prepared_receipt = manifest.get("acquisition_receipt", {})
    if prepared_receipt.get("sha256") != raw_source["acquisition_manifest_sha256"]:
        raise ValueError("Prepared segments and current raw acquisition receipt differ")
    if prepared_receipt.get("revision") != raw_source["revision"]:
        raise ValueError("Prepared segments and current raw source revision differ")
    # Ensure every fold's original-data component excludes its held-out provider.
    plans = {fold: prep.split_plan(fold=fold) for fold in (*FOLDS, "pooled")}
    old_training = {}
    for fold in (*FOLDS, "pooled"):
        train = old.load_old(fold)
        sources = {str(record["source"]) for record in train}
        excluded = fold.removeprefix("leave_out_") if fold != "pooled" else None
        if excluded is not None and excluded in sources:
            raise ValueError(f"Original training leaks held-out provider: {fold}")
        old_training[fold] = dict(matches=sorted({str(r["match"]) for r in train}),
                                  sources=sorted(sources),
                                  records=len(train))
    initial_hashes = {_rel_key(initial_path(fold, seed)): sha256(initial_path(fold, seed))
                      for fold in (*FOLDS, "pooled") for seed in SEEDS}
    initial_provenance = _verify_initial_checkpoint_provenance(plans)
    value = dict(study_id=STUDY_ID, created_utc=datetime.now(timezone.utc).isoformat(),
        question="Does capacity-scaled, function-preserving conditional H/A warm-start transfer sparse SoccerNet event-team evidence to the exposed five-match full-provider cohort?",
        baseline=dict(package="joint_gru_combined_ensemble_v1",
            manifest_sha256=sha256(ROOT / "experiments/baseline_manifest.json"),
            policy_sha256=sha256(ROOT / "experiments/model_decision_policy.json")),
        source_sha256=_source_hashes(), dependency_sha256=_freeze_dependencies(),
        raw_source_sha256=raw_source,
        split_manifest_sha256=sha256(ROOT / "experiments/eval_set/manifest.json"),
        initial_checkpoint_sha256=initial_hashes,
        initial_training_data_provenance=old_training,
        initial_checkpoint_provenance=initial_provenance,
        segments=dict(manifest_sha256=sha256(SEGMENTS / "manifest.json"),
            files=segment_files, audit=_validate_segment_audit(manifest),
            record_count=len(records), game_count=len({str(np.asarray(r["match"]).reshape(()).item())
                                                       for r in records})),
        folds=FOLDS, seeds=SEEDS, coarse_seed=COARSE_SEED,
        widths=WIDTHS, learning_rates=LEARNING_RATES, steps=STEPS,
        milestones=MILESTONES, input_dropout=INPUT_DROPOUT,
        fraction=FRACTION, batch_size=BATCH_SIZE,
        weight_decay=WEIGHT_DECAY, gradient_clip=GRADIENT_CLIP,
        initial_function="Copy teacher GRU gate rows into first 24 units; zero old-row→new-column recurrent weights; retain random new rows; initialize conditional output from teacher H/A rows with new-unit columns zero.",
        training="Conditional H/A only; original checkpoint supplies bitwise P(D); frozen fold normalizer; old source→match→window sampling and retained replace_windows augmentation; GAR game→window sampling, random 50% team swap; old per-window H/A CE uses complete-spell factors from full-native labels (censored/D/unknown neutral), plus old-only conditional H/A transition surrogate coefficient 0.05 using full-native known-interval masks and half stable/change group means; GAR contributes sparse-anchor H/A CE only and never duration or transition targets; old/GAR per-window group means mixed at fraction 0.5. Real fits require full-native old objective fields.",
        selection_stage_a="For each width select eligible seed-2026 recipe by unrounded clean spell F1, macro tie-break, fewer updates, lower LR. Eligibility: each provider clean D-F1 and macro no worse than baseline-2026 by .02, and feed_like spell no worse by .01; if no recipe eligible mark highest clean recipe exploratory.",
        selection_final="Among guard-eligible width ensembles rank unrounded clean spell, macro, then lower width. Report +.01 clean improvement screen and no automatic adoption.",
        evaluation=[dict(item=item, view=view, replicate=replicate)
                    for item in old.selection_items() for view, replicate in VIEWS],
        public="No public scoring in this runner. Separate public check waits for final external model selection and is owned by soccernet_capacity_public.py.",
        organizer_labels_read=False, automatic_promotion=False)
    write_once(OUT / "predeclared.json", value)
    print("FROZEN", sha256(OUT / "predeclared.json"), flush=True)
    return value


def check() -> dict:
    path = OUT / "predeclared.json"
    if not path.is_file():
        raise FileNotFoundError("Run freeze before any fitting or evaluation")
    spec = read_json(path)
    if spec.get("study_id") != STUDY_ID or spec.get("source_sha256") != _source_hashes():
        raise ValueError("Frozen capacity-study source changed")
    if spec.get("split_manifest_sha256") != sha256(ROOT / "experiments/eval_set/manifest.json"):
        raise ValueError("Full-provider split manifest changed")
    raw_source = spec.get("raw_source_sha256", {})
    raw_receipt_path = DATA / "independent/soccernet_gar/acquisition_manifest.json"
    if (not raw_receipt_path.is_file() or
            sha256(raw_receipt_path) != raw_source.get("acquisition_manifest_sha256")):
        raise ValueError("SoccerNet raw acquisition receipt changed after freeze")
    for key, expected in spec.get("dependency_sha256", {}).items():
        actual_path = _resolve_key(key)
        if not actual_path.is_file() or sha256(actual_path) != expected:
            raise ValueError(f"Frozen dependency changed or missing: {key}")
    for key, expected in spec.get("initial_checkpoint_sha256", {}).items():
        p = _resolve_key(key)
        if not p.is_file() or sha256(p) != expected:
            raise ValueError(f"Initial checkpoint changed: {key}")
    return spec


def _fold_records(fold: str, *, include_gar: bool = True):
    if fold not in (*FOLDS, "pooled"):
        raise ValueError("Unknown training fold")
    old_records = old.load_old(fold)
    if not include_gar:
        return old_records, []
    records, _, _ = load_segments()
    names = list(joblib.load(initial_path(fold, 2026))["feature_names"])
    model._window_index(records, names, old=False)
    return old_records, records


def _training_receipt(model_value: dict, *, stage: str, fold: str,
                      recipe: str, seed: int, step: int,
                      teacher_path: Path) -> dict:
    return dict(study_id=STUDY_ID, stage=stage, fold=fold, recipe=recipe,
        seed=seed, step=step, teacher_path=_rel_key(teacher_path),
        teacher_sha256=sha256(teacher_path),
        declaration_sha256=sha256(OUT / "predeclared.json"),
        checkpoint=dict(kind=model_value["kind"], hidden=model_value["hidden"],
            step=model_value["step"], source=model_value["source"],
            schema=model_value["schema"], training=model_value["training"],
            metadata=model_value["metadata"]))


def _save_model(path: Path, value: dict, *, receipt: dict) -> None:
    _write_joblib_once(path, value, path.with_suffix(".json"), receipt)


def _load_sealed_model(path: Path) -> dict:
    receipt_path = path.with_suffix(".json")
    if not path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"Missing model/checkpoint receipt: {path}")
    receipt = read_json(receipt_path)
    if sha256(path) != receipt.get("model_sha256"):
        raise ValueError(f"Model SHA differs from receipt: {path}")
    result = joblib.load(path)
    if result.get("kind") != "soccernet_capacity" or result.get("step") != receipt.get("step"):
        raise ValueError(f"Model payload differs from receipt: {path}")
    checkpoint = dict(kind=result["kind"], hidden=result["hidden"], step=result["step"],
        source=result["source"], schema=result["schema"], training=result["training"],
        metadata=result["metadata"])
    if receipt.get("checkpoint") != _jsonable(checkpoint):
        raise ValueError(f"Model contents differ from immutable checkpoint receipt: {path}")
    if receipt.get("declaration_sha256") != sha256(OUT / "predeclared.json"):
        raise ValueError(f"Checkpoint receipt declaration differs: {path}")
    return result


def _validate_existing_fit(path: Path, *, stage: str, fold: str, recipe: str,
                           seed: int, step: int, teacher_path: Path) -> dict:
    """Validate an already-written fit before treating it as complete."""
    fitted = _load_sealed_model(path)
    receipt = read_json(path.with_suffix(".json"))
    expected = dict(study_id=STUDY_ID, stage=stage, fold=fold, recipe=recipe,
        seed=int(seed), step=int(step), teacher_path=_rel_key(teacher_path),
        teacher_sha256=sha256(teacher_path),
        declaration_sha256=sha256(OUT / "predeclared.json"))
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Existing checkpoint receipt has the wrong fit identity: {path}")
    if fitted.get("source", {}).get("checkpoint_sha256") != expected["teacher_sha256"]:
        raise ValueError(f"Existing checkpoint was initialized from a different teacher: {path}")
    return fitted


def fit_coarse(fold: str, width: int, lr: float) -> None:
    spec = check()
    if fold not in FOLDS or width not in WIDTHS:
        raise ValueError("Unknown coarse fit fold/width")
    rate_tag(lr)
    teacher_path = initial_path(fold, COARSE_SEED)
    recipe512, recipe2048 = recipe_id(width, lr, 512), recipe_id(width, lr, 2048)
    step512 = coarse_path(fold, width, lr, 512)
    step2048 = coarse_path(fold, width, lr, 2048)
    if step2048.exists() or step2048.with_suffix(".json").exists():
        _validate_existing_fit(step2048, stage="coarse", fold=fold, recipe=recipe2048,
                               seed=COARSE_SEED, step=2048, teacher_path=teacher_path)
        if step512.exists() or step512.with_suffix(".json").exists():
            _validate_existing_fit(step512, stage="coarse", fold=fold, recipe=recipe512,
                                   seed=COARSE_SEED, step=512, teacher_path=teacher_path)
        print("FIT_COARSE_ALREADY_COMPLETE", fold, width, rate_tag(lr), flush=True)
        return
    resume = None
    if step512.exists() or step512.with_suffix(".json").exists():
        resume = _validate_existing_fit(step512, stage="coarse", fold=fold, recipe=recipe512,
            seed=COARSE_SEED, step=512, teacher_path=teacher_path)
    old_records, new_records = _fold_records(fold, include_gar=True)
    teacher = joblib.load(teacher_path)
    start = time.monotonic()
    milestones = (2048,) if resume is not None else MILESTONES

    def persist_milestone(step: int, fitted: dict) -> None:
        if (not np.array_equal(fitted["median"], teacher["median"]) or
                not np.array_equal(fitted["scale"], teacher["scale"])):
            raise ValueError("Capacity model changed its frozen teacher normalizer")
        path = coarse_path(fold, width, lr, step)
        receipt = _training_receipt(fitted, stage="coarse", fold=fold,
            recipe=recipe_id(width, lr, step), seed=COARSE_SEED, step=step,
            teacher_path=teacher_path)
        _save_model(path, fitted, receipt=receipt)

    snapshots = model.train(teacher, old_records, new_records, width, lr,
        COARSE_SEED, steps=2048, milestones=milestones, fraction=FRACTION,
        batch_size=BATCH_SIZE, input_dropout=INPUT_DROPOUT,
        require_full_native=True, resume_from=resume,
        on_checkpoint=persist_milestone)
    for step in milestones:
        persist_milestone(step, snapshots[step])
    action = "FIT_COARSE_RESUMED" if resume is not None else "FIT_COARSE"
    print(action, fold, width, rate_tag(lr), round(time.monotonic() - start, 1), flush=True)


def _all_coarse_models() -> dict[str, str]:
    entries = {}
    for fold in FOLDS:
        for config in coarse_configs():
            for step in STEPS:
                path = coarse_path(fold, config["width"], config["lr"], step)
                _load_sealed_model(path)
                entries[_rel_key(path)] = sha256(path)
                entries[_rel_key(path.with_suffix(".json"))] = sha256(path.with_suffix(".json"))
    return entries


def seal_coarse() -> dict:
    check()
    models = _all_coarse_models()
    value = dict(declaration_sha256=sha256(OUT / "predeclared.json"),
                 models=models, expected_models=36)
    write_once(OUT / "coarse_checkpoints_frozen.json", value)
    return value


def _check_seal(name: str) -> dict:
    value = read_json(OUT / name)
    if value.get("declaration_sha256") != sha256(OUT / "predeclared.json"):
        raise ValueError(f"Checkpoint seal declaration mismatch: {name}")
    for key, expected in value["models"].items():
        path = _resolve_key(key)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Sealed model changed: {key}")
    return value


def _cell_id(item: dict, view: str, replicate: str) -> str:
    return f"{item['match']}_{view}_{replicate}"


def _selection_items_for_match(match: str | None = None) -> list[dict]:
    items = old.selection_items()
    if match is None:
        return items
    selected = [item for item in items if item["match"] == match]
    if not selected:
        raise ValueError(f"Unknown selection match: {match}")
    if len(selected) != 1:
        raise ValueError(f"Selection match is ambiguous: {match}")
    return selected


def _cache_cell(item: dict, view: str, replicate: str):
    path = old.eval_path(item, view, replicate)
    cached = joblib.load(path)
    inputs, labels = cached["inputs"], cached["labels"]
    record = old.inference_record(cached["record"])
    return path, inputs, labels, record


def _baseline_probability(item: dict, view: str, replicate: str, seed: int,
                          inputs: dict, record: dict, teacher: dict):
    raw = fh.predict(record, teacher)
    native = old.upsample(raw, record, inputs)
    historic = old.historical_path(item, view, replicate, seed)
    with np.load(historic, allow_pickle=False) as saved:
        old_probability = saved["probability"]
        old_prediction = saved["prediction"]
    if not np.allclose(native, old_probability, atol=1e-6, rtol=0):
        raise ValueError(f"Baseline probability parity exceeds 1e-6: {historic.name}")
    decoded = supervised.decode(native, supervised.model_periods(inputs),
                                float(inputs["fps"]), lt.COST)
    if not np.array_equal(decoded, old_prediction):
        raise ValueError(f"Baseline decoded prediction differs from archive: {historic.name}")
    return raw, native, decoded


def _historical_report_rows() -> dict[tuple, dict]:
    path = DATA / "experiments/soccernet_extension_v1/report.json"
    report = read_json(path)
    rows = report.get("rows", [])
    return {(r["source"], r["match"], r["view"], r["replicate"], r["arm"], r["member"]): r
            for r in rows}


def _metric_parity(row: dict, prior: dict) -> None:
    for name in (*METRICS, "weighted_70_30", "n_gt_spells", "n_pred_spells",
                 "short_spell_recall", "fragmented_reference_spells", "graded_frames"):
        if name not in prior or row.get(name) is None or prior.get(name) is None:
            continue
        if abs(float(row[name]) - float(prior[name])) > 1e-12:
            raise ValueError(f"Saved baseline score parity failed for {name}")
    for threshold, value in row["spell_by_iou"].items():
        if abs(float(value) - float(prior["spell_by_iou"][threshold])) > 1e-12:
            raise ValueError(f"Saved baseline IoU parity failed at {threshold}")


def _score_probability(probability, inputs, labels, item, view, replicate,
                       arm: str, member: str):
    row, prediction = old.metric_row(probability, inputs, labels, item,
                                     view, replicate, arm, member)
    return row, prediction


def _save_cell(stage: str, item: dict, view: str, replicate: str,
               rows: list[dict], arrays: dict[str, tuple[np.ndarray, np.ndarray]],
               seal_name: str) -> None:
    stem = f"{stage}_{_cell_id(item, view, replicate)}"
    path = OUT / "cells" / f"{stem}.json"
    if path.exists():
        saved = read_json(path)
        for rel, expected in saved.get("files", {}).items():
            if sha256(DATA / rel) != expected:
                raise ValueError(f"Existing cell array changed: {rel}")
        if saved.get("rows") != _jsonable(rows):
            raise ValueError(f"Existing cell results differ; refusing overwrite: {path}")
        return
    files = {}
    for name, (probability, prediction) in arrays.items():
        output = OUT / "arrays" / f"{stem}_{name}.npz"
        if output.exists():
            raise ValueError(f"Array exists without its cell receipt: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            np.savez_compressed(stream, probability=probability, prediction=prediction)
            stream.flush(); os.fsync(stream.fileno())
        files[_rel_key(output).removeprefix("data/")] = sha256(output)
    write_once(path, dict(declaration_sha256=sha256(OUT / "predeclared.json"),
        checkpoint_seal_sha256=sha256(OUT / seal_name), item=item, view=view,
        replicate=replicate, rows=rows, files=files))


def evaluate_coarse(match: str | None = None) -> None:
    items = _selection_items_for_match(match)
    spec = check()
    _check_seal("coarse_checkpoints_frozen.json")
    config_values = coarse_configs()
    prior_rows = _historical_report_rows()
    for item in items:
        fold = "leave_out_" + item["source"]
        teacher = joblib.load(initial_path(fold, COARSE_SEED))
        for view, replicate in VIEWS:
            cache_path, inputs, labels, record = _cache_cell(item, view, replicate)
            raw, baseline, _ = _baseline_probability(item, view, replicate, COARSE_SEED,
                                                       inputs, record, teacher)
            rows, arrays = [], {}
            base_row, base_prediction = _score_probability(baseline, inputs, labels,
                item, view, replicate, "baseline", str(COARSE_SEED))
            prior = prior_rows.get((item["source"], item["match"], view, replicate,
                                    "baseline", str(COARSE_SEED)))
            if prior is None:
                raise ValueError("Stage-1 report lacks seed-2026 baseline score row")
            _metric_parity(base_row, prior)
            rows.append(base_row)
            arrays["baseline_seed2026"] = (baseline, base_prediction)
            for config in config_values:
                fitted = _load_sealed_model(coarse_path(fold, config["width"],
                                                        config["lr"], config["steps"]))
                probability = model.predict(record, fitted, teacher,
                                            original_probability=raw)
                if not np.array_equal(probability[:, 2], raw[:, 2]):
                    raise ValueError("Coarse model altered sampled P(D)")
                native = old.upsample(probability, record, inputs)
                if not np.array_equal(native[:, 2], baseline[:, 2]):
                    raise ValueError("Coarse model altered native P(D)")
                row, prediction = _score_probability(native, inputs, labels, item,
                    view, replicate, config["recipe"], str(COARSE_SEED))
                rows.append(row)
                arrays[config["recipe"]] = (native, prediction)
            if len(rows) != 13:
                raise ValueError("Coarse cell must contain baseline plus twelve recipes")
            _save_cell("coarse", item, view, replicate, rows, arrays,
                       "coarse_checkpoints_frozen.json")
            print("EVALUATE_COARSE", item["match"], view, replicate, flush=True)


def _aggregate_provider(rows: list[dict], arm: str, view: str,
                        member: str) -> dict:
    selected = [row for row in rows if (row["arm"], row["view"], row["member"]) ==
                (arm, view, member)]
    providers = {}
    fields = (*METRICS, "weighted_70_30", "n_gt_spells", "n_pred_spells",
              "d_precision", "d_recall", "short_spell_recall",
              "fragmented_reference_spells", "graded_frames")
    for source in ("sportec", "metrica", "skillcorner"):
        group = [row for row in selected if row["source"] == source]
        if not group:
            raise ValueError(f"Missing {source} rows for {arm}/{view}/{member}")
        summary = {}
        for field in fields:
            values = [row.get(field) for row in group if row.get(field) is not None]
            if values:
                summary[field] = float(np.mean(values))
        summary["spell_by_iou"] = {threshold: float(np.mean(
            [row["spell_by_iou"][threshold] for row in group]))
            for threshold in ("0.3", "0.4", "0.5", "0.6", "0.7")}
        providers[source] = summary
    means = {field: float(np.mean([value[field] for value in providers.values()]))
             for field in fields if all(field in value for value in providers.values())}
    means["spell_by_iou"] = {threshold: float(np.mean(
        [value["spell_by_iou"][threshold] for value in providers.values()]))
        for threshold in ("0.3", "0.4", "0.5", "0.6", "0.7")}
    return dict(equal_provider_mean=means, per_provider=providers,
                cell_count=len(selected))


def _load_cell_rows(stage: str, expected_cells: int, seal_name: str) -> list[dict]:
    seal = _check_seal(seal_name)
    rows = []
    for item in old.selection_items():
        for view, replicate in VIEWS:
            path = OUT / "cells" / f"{stage}_{_cell_id(item, view, replicate)}.json"
            value = read_json(path)
            if (value.get("declaration_sha256") != sha256(OUT / "predeclared.json") or
                    value.get("checkpoint_seal_sha256") != sha256(OUT / seal_name)):
                raise ValueError(f"Cell seal mismatch: {path}")
            for rel, expected in value.get("files", {}).items():
                if sha256(DATA / rel) != expected:
                    raise ValueError(f"Scored array hash differs: {rel}")
            rows.extend(value["rows"])
    if len(rows) != expected_cells:
        raise ValueError(f"Expected {expected_cells} rows for {stage}; got {len(rows)}")
    return rows


def assert_exact_keys(rows: list[dict], expected: set[tuple], fields: tuple[str, ...]) -> None:
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete, duplicated, or unexpected study cells")


def _expected_coarse_rows() -> set[tuple]:
    return {(item["source"], item["match"], view, replicate, arm, "2026")
            for item in old.selection_items() for view, replicate in VIEWS
            for arm in ("baseline", *(config["recipe"] for config in coarse_configs()))}


def _rows_by_cell(stage: str, key_count: int, seal_name: str) -> list[dict]:
    rows = _load_cell_rows(stage, key_count, seal_name)
    assert_exact_keys(rows, _expected_coarse_rows(),
        ("source", "match", "view", "replicate", "arm", "member"))
    return rows


def coarse_summary(rows: list[dict], config: dict | None, view: str) -> dict:
    arm = "baseline" if config is None else config["recipe"]
    return _aggregate_provider(rows, arm, view, "2026")


def guard_checks(candidate: dict, baseline: dict, *, d_macro_floor: float = 0.02,
                 feed_spell_floor: float = 0.01) -> dict:
    checks = {f"{provider}_{metric}":
        candidate["clean"]["per_provider"][provider][metric] >=
        baseline["clean"]["per_provider"][provider][metric] - d_macro_floor
        for provider in ("sportec", "metrica", "skillcorner")
        for metric in ("d_f1", "macro_f1")}
    checks["feed_like_spell"] = candidate["feed_like"]["equal_provider_mean"]["spell_f1"] >= \
        baseline["feed_like"]["equal_provider_mean"]["spell_f1"] - feed_spell_floor
    return checks


def select_coarse_for_width(summaries: dict[str, dict], width: int) -> dict:
    baseline = summaries["baseline"]
    configs = [config for config in coarse_configs() if config["width"] == width]
    guards = {}
    eligible = []
    for config in configs:
        summary = summaries[config["recipe"]]
        checks = guard_checks(summary, baseline)
        guards[config["recipe"]] = checks
        if all(checks.values()):
            eligible.append(config)
    pool = eligible or configs
    selected = max(pool, key=lambda config: (
        summaries[config["recipe"]]["clean"]["equal_provider_mean"]["spell_f1"],
        summaries[config["recipe"]]["clean"]["equal_provider_mean"]["macro_f1"],
        -config["steps"], -config["lr"]))
    return dict(width=width, recipe=selected["recipe"], lr=selected["lr"],
        steps=selected["steps"], eligible=selected in eligible,
        selection_mode="guard_eligible" if eligible else "exploratory_no_guard_eligible_recipe",
        guards=guards)


def report_coarse() -> dict:
    check()
    _check_seal("coarse_checkpoints_frozen.json")
    rows = _rows_by_cell("coarse", 195, "coarse_checkpoints_frozen.json")
    summaries = {"baseline": {view: _aggregate_provider(rows, "baseline", view, "2026")
                              for view in ("clean", "feed_like")}}
    for config in coarse_configs():
        summaries[config["recipe"]] = {view: _aggregate_provider(
            rows, config["recipe"], view, "2026") for view in ("clean", "feed_like")}
    selections = {str(width): select_coarse_for_width(summaries, width)
                  for width in WIDTHS}
    decision = dict(study_id=STUDY_ID,
        declaration_sha256=sha256(OUT / "predeclared.json"),
        coarse_seal_sha256=sha256(OUT / "coarse_checkpoints_frozen.json"),
        selected_by_width=selections, scoring_rows=195,
        baseline_parity="30 individual seed-2026 arrays within 1e-6 and exact decoded predictions; saved report metrics within 1e-12")
    value = dict(study_id=STUDY_ID, rows=rows, summaries=summaries,
        decision=decision, organizer_labels_read=False)
    write_once(OUT / "coarse_report.json", value)
    write_once(OUT / "coarse_decision.json", decision)
    return value


def _selected_coarse() -> dict[int, dict]:
    decision = read_json(OUT / "coarse_decision.json")
    if decision.get("declaration_sha256") != sha256(OUT / "predeclared.json"):
        raise ValueError("Coarse decision declaration mismatch")
    return {int(width): value for width, value in decision["selected_by_width"].items()}


def fit_confirm(fold: str, width: int) -> None:
    check(); _check_seal("coarse_checkpoints_frozen.json")
    if fold not in FOLDS or width not in WIDTHS:
        raise ValueError("Unknown confirmation fold/width")
    selection = _selected_coarse()[width]
    lr, steps = float(selection["lr"]), int(selection["steps"])
    teacher_path = initial_path(fold, 2027)
    path = confirm_path(fold, width, lr, steps)
    if path.exists() or path.with_suffix(".json").exists():
        _validate_existing_fit(path, stage="confirm", fold=fold,
            recipe=selection["recipe"], seed=2027, step=steps, teacher_path=teacher_path)
        print("FIT_CONFIRM_ALREADY_COMPLETE", fold, width, flush=True)
        return
    old_records, new_records = _fold_records(fold, include_gar=True)
    teacher = joblib.load(teacher_path)
    snapshot = model.train(teacher, old_records, new_records, width, lr, 2027,
        steps=steps, milestones=(steps,), fraction=FRACTION,
        batch_size=BATCH_SIZE, input_dropout=INPUT_DROPOUT,
        require_full_native=True)[steps]
    if not np.array_equal(snapshot["median"], teacher["median"]) or not np.array_equal(snapshot["scale"], teacher["scale"]):
        raise ValueError("Confirmation fit changed fold normalizer")
    receipt = _training_receipt(snapshot, stage="confirm", fold=fold,
        recipe=selection["recipe"], seed=2027, step=steps, teacher_path=teacher_path)
    _save_model(path, snapshot, receipt=receipt)
    print("FIT_CONFIRM", fold, selection["recipe"], flush=True)


def seal_confirm() -> dict:
    check(); coarse_seal = _check_seal("coarse_checkpoints_frozen.json")
    selected = _selected_coarse()
    entries = {}
    for width, recipe in selected.items():
        for fold in FOLDS:
            seed26 = coarse_path(fold, width, recipe["lr"], recipe["steps"])
            seed27 = confirm_path(fold, width, recipe["lr"], recipe["steps"])
            for path in (seed26, seed27):
                _load_sealed_model(path)
                entries[_rel_key(path)] = sha256(path)
                entries[_rel_key(path.with_suffix(".json"))] = sha256(path.with_suffix(".json"))
    value = dict(declaration_sha256=sha256(OUT / "predeclared.json"),
        coarse_seal_sha256=sha256(OUT / "coarse_checkpoints_frozen.json"),
        coarse_models=coarse_seal["models"], selected_models=entries,
        expected_confirmation_models=9)
    write_once(OUT / "checkpoints_confirm_frozen.json", value)
    return value


def _expected_final_rows() -> set[tuple]:
    return {(item["source"], item["match"], view, replicate, arm, member)
            for item in old.selection_items() for view, replicate in VIEWS
            for arm in ("baseline", "w24", "w48", "w96") for member in MEMBERS}


def evaluate_final(match: str | None = None) -> None:
    items = _selection_items_for_match(match)
    check(); _check_seal("checkpoints_confirm_frozen.json")
    selected = _selected_coarse()
    prior_rows = _historical_report_rows()
    for item in items:
        fold = "leave_out_" + item["source"]
        teachers = {seed: joblib.load(initial_path(fold, seed)) for seed in SEEDS}
        configs = {width: selected[width] for width in WIDTHS}
        models = {width: {seed: _load_sealed_model(
            coarse_path(fold, width, configs[width]["lr"], configs[width]["steps"]) if seed == 2026 else
            confirm_path(fold, width, configs[width]["lr"], configs[width]["steps"]))
            for seed in SEEDS} for width in WIDTHS}
        for view, replicate in VIEWS:
            _, inputs, labels, record = _cache_cell(item, view, replicate)
            raw = {seed: fh.predict(record, teachers[seed]) for seed in SEEDS}
            baseline_members = {}
            baseline_preds = {}
            for seed in SEEDS:
                native, prediction = None, None
                _, native, prediction = _baseline_probability(item, view, replicate,
                    seed, inputs, record, teachers[seed])
                prior = prior_rows.get((item["source"], item["match"], view,
                                        replicate, "baseline", str(seed)))
                if prior is None:
                    raise ValueError("Historical report lacks individual baseline row")
                row, decoded = _score_probability(native, inputs, labels, item,
                    view, replicate, "baseline", str(seed))
                _metric_parity(row, prior)
                baseline_members[seed] = native
                baseline_preds[seed] = decoded
            all_native = {"baseline": dict(baseline_members)}
            for width in WIDTHS:
                all_native[f"w{width}"] = {}
                for seed in SEEDS:
                    probability = model.predict(record, models[width][seed],
                        teachers[seed], original_probability=raw[seed])
                    if not np.array_equal(probability[:, 2], raw[seed][:, 2]):
                        raise ValueError("Final capacity member altered P(D)")
                    native = old.upsample(probability, record, inputs)
                    if not np.array_equal(native[:, 2], baseline_members[seed][:, 2]):
                        raise ValueError("Final capacity native P(D) differs from baseline")
                    all_native[f"w{width}"][seed] = native
            rows, arrays = [], {}
            for arm, members in all_native.items():
                members = dict(members, ensemble=(members[2026] + members[2027]) * 0.5)
                for seed_name, probability in members.items():
                    row, prediction = _score_probability(probability, inputs, labels,
                        item, view, replicate, arm, str(seed_name))
                    if arm == "baseline":
                        prior = prior_rows.get((item["source"], item["match"], view,
                                                replicate, "baseline", "ensemble"))
                        if prior is None:
                            raise ValueError("Historical report lacks baseline ensemble row")
                        _metric_parity(row, prior)
                    rows.append(row)
                    arrays[f"{arm}_{seed_name}"] = (probability, prediction)
            if len(rows) != 12:
                raise ValueError("Final cell must contain four recipes × three members")
            _save_cell("final", item, view, replicate, rows, arrays,
                       "checkpoints_confirm_frozen.json")
            print("EVALUATE_FINAL", item["match"], view, replicate, flush=True)


def select_final_width(summaries: dict[str, dict]) -> dict:
    baseline = summaries["baseline"]
    guards = {f"w{width}": guard_checks(summaries[f"w{width}"], baseline)
              for width in WIDTHS}
    eligible = [width for width in WIDTHS if all(guards[f"w{width}"].values())]
    pool = eligible or list(WIDTHS)
    selected = max(pool, key=lambda width: (
        summaries[f"w{width}"]["clean"]["equal_provider_mean"]["spell_f1"],
        summaries[f"w{width}"]["clean"]["equal_provider_mean"]["macro_f1"],
        -width))
    clean_delta = (summaries[f"w{selected}"]["clean"]["equal_provider_mean"]["spell_f1"] -
                   baseline["clean"]["equal_provider_mean"]["spell_f1"])
    return dict(selected_width=selected, selected_recipe=summaries[f"w{selected}"].get("recipe"),
        eligible=selected in eligible,
        selection_mode="guard_eligible" if eligible else "exploratory_no_guard_eligible_width",
        guards=guards, clean_spell_delta_vs_baseline=clean_delta,
        improvement_screen_passes=selected in eligible and clean_delta >= 0.01,
        adopt_model=False)


def report_final() -> dict:
    check(); _check_seal("checkpoints_confirm_frozen.json")
    rows = _load_cell_rows("final", 180, "checkpoints_confirm_frozen.json")
    assert_exact_keys(rows, _expected_final_rows(),
        ("source", "match", "view", "replicate", "arm", "member"))
    summaries = {}
    for arm in ("baseline", "w24", "w48", "w96"):
        summaries[arm] = {view: _aggregate_provider(rows, arm, view, "ensemble")
                          for view in ("clean", "feed_like")}
        summaries[arm]["recipe"] = (None if arm == "baseline" else
            _selected_coarse()[int(arm.removeprefix("w"))]["recipe"])
    decision = select_final_width(summaries)
    decision.update(study_id=STUDY_ID,
        declaration_sha256=sha256(OUT / "predeclared.json"),
        checkpoint_seal_sha256=sha256(OUT / "checkpoints_confirm_frozen.json"),
        baseline_parity="Both seed probability arrays within 1e-6 of historical arrays; decoded predictions exact; saved seed/member scores match stage 1.")
    value = dict(study_id=STUDY_ID, rows=rows, summaries=summaries,
        seed_summaries={arm:{seed:{view:_aggregate_provider(rows, arm, view, seed)
            for view in ("clean", "feed_like")} for seed in MEMBERS[:2]}
            for arm in ("baseline", "w24", "w48", "w96")},
        decision=decision, organizer_labels_read=False,
        limitations=["Previously exposed five-match development cohort; feed_like is synthetic.",
            "Event-team labels are sparse silver, not dense SoccerNet possession gold.",
            "The original checkpoint supplies P(D); changed conditional H/A can still alter decoded D metrics.",
            "Warm-start conditional H/A capacity study; not a from-scratch combined-loss comparison."])
    write_once(OUT / "final_report.json", value)
    write_once(OUT / "final_decision.json", decision)
    return value


def fit_control(fold: str, seed: int) -> None:
    check(); _check_seal("checkpoints_confirm_frozen.json")
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown old-only control fit")
    decision = read_json(OUT / "final_decision.json")
    width = int(decision["selected_width"])
    recipe = _selected_coarse()[width]
    lr, steps = float(recipe["lr"]), int(recipe["steps"])
    teacher_path = initial_path(fold, seed)
    path = control_fit_path(fold, width, lr, steps, seed)
    if path.exists() or path.with_suffix(".json").exists():
        _validate_existing_fit(path, stage="old_only", fold=fold,
            recipe=recipe["recipe"], seed=seed, step=steps, teacher_path=teacher_path)
        print("FIT_CONTROL_ALREADY_COMPLETE", fold, seed, flush=True)
        return
    old_records, _ = _fold_records(fold, include_gar=False)
    teacher = joblib.load(teacher_path)
    fitted = model.train(teacher, old_records, [], width, lr, seed,
        steps=steps, milestones=(steps,), fraction=0.0, batch_size=BATCH_SIZE,
        input_dropout=INPUT_DROPOUT, require_full_native=True)[steps]
    receipt = _training_receipt(fitted, stage="old_only", fold=fold,
        recipe=recipe["recipe"], seed=seed, step=steps, teacher_path=teacher_path)
    _save_model(path, fitted, receipt=receipt)
    print("FIT_CONTROL", fold, seed, recipe["recipe"], flush=True)


def seal_control() -> dict:
    check(); _check_seal("checkpoints_confirm_frozen.json")
    decision = read_json(OUT / "final_decision.json")
    width = int(decision["selected_width"])
    recipe = _selected_coarse()[width]
    entries = {}
    for fold in FOLDS:
        for seed in SEEDS:
            path = control_fit_path(fold, width, recipe["lr"], recipe["steps"], seed)
            _load_sealed_model(path)
            entries[_rel_key(path)] = sha256(path)
            entries[_rel_key(path.with_suffix(".json"))] = sha256(path.with_suffix(".json"))
    value = dict(declaration_sha256=sha256(OUT / "predeclared.json"),
        final_decision_sha256=sha256(OUT / "final_decision.json"),
        models=entries, expected_models=6)
    write_once(OUT / "control_checkpoints_frozen.json", value)
    return value


def evaluate_control(match: str | None = None) -> None:
    items = _selection_items_for_match(match)
    check(); _check_seal("control_checkpoints_frozen.json")
    decision = read_json(OUT / "final_decision.json")
    width = int(decision["selected_width"])
    recipe = _selected_coarse()[width]
    for item in items:
        fold = "leave_out_" + item["source"]
        teachers = {seed: joblib.load(initial_path(fold, seed)) for seed in SEEDS}
        controls = {seed: _load_sealed_model(control_fit_path(fold, width,
            recipe["lr"], recipe["steps"], seed)) for seed in SEEDS}
        for view, replicate in VIEWS:
            _, inputs, labels, record = _cache_cell(item, view, replicate)
            members = {}
            rows, arrays = [], {}
            for seed in SEEDS:
                raw = fh.predict(record, teachers[seed])
                probability = model.predict(record, controls[seed], teachers[seed],
                                            original_probability=raw)
                if not np.array_equal(probability[:, 2], raw[:, 2]):
                    raise ValueError("Matched old-only control changed P(D)")
                native = old.upsample(probability, record, inputs)
                members[seed] = native
            members["ensemble"] = (members[2026] + members[2027]) * 0.5
            for member, probability in members.items():
                row, prediction = _score_probability(probability, inputs, labels,
                    item, view, replicate, "old_only", str(member))
                rows.append(row); arrays[f"old_only_{member}"] = (probability, prediction)
            _save_cell("control", item, view, replicate, rows, arrays,
                       "control_checkpoints_frozen.json")
            print("EVALUATE_CONTROL", item["match"], view, replicate, flush=True)


def report_control() -> dict:
    check(); _check_seal("control_checkpoints_frozen.json")
    rows = _load_cell_rows("control", 45, "control_checkpoints_frozen.json")
    expected = {(item["source"], item["match"], view, replicate, "old_only", member)
        for item in old.selection_items() for view, replicate in VIEWS for member in MEMBERS}
    assert_exact_keys(rows, expected,
        ("source", "match", "view", "replicate", "arm", "member"))
    control = {view: _aggregate_provider(rows, "old_only", view, "ensemble")
               for view in ("clean", "feed_like")}
    final = read_json(OUT / "final_report.json")
    width = int(read_json(OUT / "final_decision.json")["selected_width"])
    recipe_id_width = f"w{width}"
    chosen = final["summaries"][recipe_id_width]
    deltas = {view: chosen[view]["equal_provider_mean"]["spell_f1"] -
              control[view]["equal_provider_mean"]["spell_f1"]
              for view in ("clean", "feed_like")}
    value = dict(study_id=STUDY_ID, rows=rows, summaries={"old_only": control},
        selected_width=width, selected_recipe=_selected_coarse()[width]["recipe"],
        selected_gar_vs_old_only_spell_delta=deltas,
        note="Matched control is attribution evidence only; it does not alter the previously frozen final width/recipe selection.",
        organizer_labels_read=False)
    write_once(OUT / "control_report.json", value)
    return value


def fit_pooled(seed: int) -> None:
    check(); _check_seal("checkpoints_confirm_frozen.json")
    if seed not in SEEDS:
        raise ValueError("Pooled deployment fit seed must be 2026 or 2027")
    decision = read_json(OUT / "final_decision.json")
    width = int(decision["selected_width"])
    recipe = _selected_coarse()[width]
    steps = int(recipe["steps"])
    teacher_path = initial_path("pooled", seed)
    path = pooled_path(width, float(recipe["lr"]), steps, seed)
    if path.exists() or path.with_suffix(".json").exists():
        _validate_existing_fit(path, stage="pooled", fold="pooled",
            recipe=recipe["recipe"], seed=seed, step=steps, teacher_path=teacher_path)
        print("FIT_POOLED_ALREADY_COMPLETE", seed, flush=True)
        return
    old_records, new_records = _fold_records("pooled", include_gar=True)
    teacher = joblib.load(teacher_path)
    fitted = model.train(teacher, old_records, new_records, width,
        float(recipe["lr"]), seed, steps=steps, milestones=(steps,),
        fraction=FRACTION, batch_size=BATCH_SIZE,
        input_dropout=INPUT_DROPOUT, require_full_native=True)[steps]
    receipt = _training_receipt(fitted, stage="pooled", fold="pooled",
        recipe=recipe["recipe"], seed=seed, step=steps, teacher_path=teacher_path)
    _save_model(path, fitted, receipt=receipt)


def seal_pooled() -> dict:
    check()
    decision = read_json(OUT / "final_decision.json")
    width = int(decision["selected_width"]); recipe = _selected_coarse()[width]
    entries = {}
    for seed in SEEDS:
        path = pooled_path(width, recipe["lr"], recipe["steps"], seed)
        _load_sealed_model(path)
        entries[_rel_key(path)] = sha256(path)
        entries[_rel_key(path.with_suffix(".json"))] = sha256(path.with_suffix(".json"))
    value = dict(declaration_sha256=sha256(OUT / "predeclared.json"),
        final_decision_sha256=sha256(OUT / "final_decision.json"),
        models=entries, expected_models=2)
    write_once(OUT / "pooled_checkpoints_frozen.json", value)
    return value


def _write_public_ready_receipt():
    """Seal pooled models for the separately owned public checker; no gate claim."""
    seal = _check_seal("pooled_checkpoints_frozen.json")
    decision = read_json(OUT / "final_decision.json")
    value = dict(study_id=STUDY_ID, declaration_sha256=sha256(OUT / "predeclared.json"),
        final_decision_sha256=sha256(OUT / "final_decision.json"),
        pooled_seal_sha256=sha256(OUT / "pooled_checkpoints_frozen.json"),
        selected_width=decision["selected_width"], models=seal["models"],
        public_gate_claimed=False, organizer_reference_access=False)
    write_once(OUT / "public_ready.json", value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(
        "freeze", "check", "fit-coarse", "seal-coarse", "evaluate-coarse",
        "report-coarse", "fit-confirm", "seal-confirm", "evaluate-final",
        "report-final", "fit-control", "seal-control", "evaluate-control",
        "report-control", "fit-pooled", "seal-pooled", "public-ready"))
    parser.add_argument("--fold", choices=(*FOLDS, "pooled"))
    parser.add_argument("--width", type=int, choices=WIDTHS)
    parser.add_argument("--lr", type=float, choices=LEARNING_RATES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--match", help="Evaluate only this known development match")
    args = parser.parse_args()
    if args.action == "fit-coarse":
        if args.fold is None or args.width is None or args.lr is None:
            parser.error("fit-coarse requires --fold --width --lr")
        fit_coarse(args.fold, args.width, args.lr)
    elif args.action == "fit-confirm":
        if args.fold is None or args.width is None:
            parser.error("fit-confirm requires --fold --width")
        fit_confirm(args.fold, args.width)
    elif args.action == "fit-control":
        if args.fold is None or args.seed is None:
            parser.error("fit-control requires --fold --seed")
        fit_control(args.fold, args.seed)
    elif args.action == "fit-pooled":
        if args.seed is None:
            parser.error("fit-pooled requires --seed")
        fit_pooled(args.seed)
    elif args.action == "evaluate-coarse":
        result = evaluate_coarse(match=args.match)
        if result is not None:
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    elif args.action == "evaluate-final":
        result = evaluate_final(match=args.match)
        if result is not None:
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    elif args.action == "evaluate-control":
        result = evaluate_control(match=args.match)
        if result is not None:
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    else:
        if args.match is not None:
            parser.error("--match is only valid with evaluate-coarse, evaluate-final, or evaluate-control")
        mapping = {
            "seal-coarse": seal_coarse, "evaluate-coarse": evaluate_coarse,
            "report-coarse": report_coarse, "seal-confirm": seal_confirm,
            "evaluate-final": evaluate_final, "report-final": report_final,
            "seal-control": seal_control, "evaluate-control": evaluate_control,
            "report-control": report_control, "seal-pooled": seal_pooled,
            "public-ready": _write_public_ready_receipt,
            "freeze": freeze, "check": check,
        }
        result = mapping[args.action]()
        if result is not None:
            print(json.dumps(_jsonable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
