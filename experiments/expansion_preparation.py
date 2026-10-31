"""Immutable external-match caches for the SkillCorner expansion study.

This module prepares raw views only. It never fits normalization or a model and
never reads organizer inputs. Native SkillCorner grading is applied by
``multisource_eval.view`` before any augmentation; the model record receives
tracking fields only.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile

import joblib
import numpy as np

from experiments import (corruption_experiment as ce, gru_robustness_experiment as gr,
                         learned_temporal as lt, light_corruption as lc,
                         multisource_eval as ms, preparation as prep, robust,
                         supervised, stress_components as sc,
                         tracking_corruption as tc)

ROOT = prep.ROOT
TRAIN_RECORD_KEYS = ("x", "y", "names", "indices", "period", "native_n",
                     "fps", "source", "match")
FEEDLIKE_COMPONENTS = ("ball_gaps", "coordinates", "slot_gaps", "id_swaps")
NEW_SKILLCORNER_TRAIN = (
    "skillcorner_had_2006363", "skillcorner_had_2007448",
    "skillcorner_had_2007721", "skillcorner_had_2010085",
    "skillcorner_had_2011166", "skillcorner_had_2013725",
)
SOURCES = (
    "experiments/expansion_preparation.py",
    "experiments/preparation.py",
    "experiments/multisource_eval.py",
    "experiments/dead_transfer.py",
    "experiments/robust.py",
    "experiments/stress.py",
    "experiments/light_corruption.py",
    "experiments/corruption_experiment.py",
    "experiments/tracking_corruption.py",
    "experiments/stress_components.py",
    "experiments/gru_robustness_experiment.py",
    "experiments/refine_domain.py",
    "experiments/refine_features.py",
    "experiments/learned_temporal.py",
    "experiments/supervised.py",
)


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _fingerprint(value) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {relative: prep.sha256(ROOT / relative) for relative in SOURCES}


def _calibration_copy(calibration: dict) -> dict:
    if not isinstance(calibration, dict):
        raise ValueError("calibration must be a dict")
    required = {"ball_missing_fraction", "gap_edges_seconds", "gap_bin_probability"}
    if not required.issubset(calibration):
        raise ValueError("calibration is missing required fields")
    result = json.loads(_canonical(calibration))
    edges = np.asarray(result["gap_edges_seconds"], dtype=float)
    probabilities = np.asarray(result["gap_bin_probability"], dtype=float)
    missing = float(result["ball_missing_fraction"])
    if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
            or np.any(np.diff(edges) <= 0)):
        raise ValueError("gap edges must be finite and strictly increasing")
    if (probabilities.shape != (len(edges) - 1,)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1.0)):
        raise ValueError("gap probabilities must align with bins and sum to one")
    if not np.isfinite(missing) or not 0 <= missing <= 1:
        raise ValueError("ball missing fraction must be finite and in [0, 1]")
    return result


def _input_path(item: dict, data_dir: Path) -> tuple[Path, str]:
    if not isinstance(item, dict):
        raise ValueError("item must be a manifest row")
    for key in ("source", "match", "group_id", "path", "sha256"):
        if key not in item:
            raise ValueError(f"manifest row is missing {key}")
    path = prep.data_path(item, data_dir)
    if (path.suffix != ".npz" or path.stem != item["match"]
            or item["group_id"] != item["match"]):
        raise ValueError("expected a data-relative NPZ named for its match")
    actual = prep.sha256(path)
    if actual != item["sha256"]:
        raise ValueError(f"Input checksum mismatch: {item['match']}")
    return path, actual


def _encode_labels(labels) -> np.ndarray:
    values = np.asarray(labels).astype(str)
    if values.ndim != 1 or not np.isin(values, ("H", "A", "D", "X")).all():
        raise ValueError("expected aligned H/A/D/X labels")
    output = np.full(len(values), -1, dtype=np.int8)
    output[values == "H"] = 0
    output[values == "A"] = 1
    output[values == "D"] = 2
    return output


def _clean_record(record: dict, source: str, match: str) -> dict:
    result = {key: record[key] for key in TRAIN_RECORD_KEYS}
    result["x"] = np.asarray(result["x"])
    result["y"] = np.asarray(result["y"], dtype=np.int8)
    result["names"] = list(result["names"])
    result["indices"] = np.asarray(result["indices"], dtype=np.int64)
    result["period"] = np.asarray(result["period"])
    result["source"] = str(source)
    result["match"] = str(match)
    if (result["x"].ndim != 2 or result["y"].shape != (len(result["x"]),)
            or result["indices"].shape != result["y"].shape
            or result["period"].shape != result["y"].shape
            or result["x"].shape[1] != len(result["names"])):
        raise ValueError(f"invalid prepared feature record: {match}")
    return result


def _identity(kind: str, item: dict, input_sha256: str, recipe: dict,
              source_hashes: dict[str, str]) -> dict:
    return dict(schema=1, kind=kind, source=item["source"], match=item["match"],
                group_id=item["group_id"], role=item.get("role"),
                input_sha256=input_sha256,
                recipe=recipe, source_hashes=source_hashes)


def _versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}


def _verify_cache(path: Path, receipt_path: Path, identity: dict,
                  preparation_sha256: str) -> dict:
    if not path.is_file() or not receipt_path.is_file():
        raise ValueError(f"incomplete immutable cache pair: {path.name}")
    receipt = json.loads(receipt_path.read_text())
    actual_hash = prep.sha256(path)
    actual_bytes = path.stat().st_size
    if (receipt.get("identity") != identity
            or receipt.get("preparation_sha256") != preparation_sha256
            or receipt.get("source_hashes") != identity["source_hashes"]
            or receipt.get("cache_sha256") != actual_hash
            or receipt.get("sha256") != actual_hash
            or receipt.get("cache_bytes") != actual_bytes
            or receipt.get("bytes") != actual_bytes):
        raise ValueError(f"immutable cache receipt mismatch: {path.name}")
    return receipt


def _write_cache(path: Path, receipt_path: Path, payload: dict,
                 identity: dict, preparation_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or receipt_path.exists():
        _verify_cache(path, receipt_path, identity, preparation_sha256)
        return

    with tempfile.TemporaryDirectory(prefix=".expansion-cache-", dir=path.parent) as temp:
        staged = Path(temp) / "payload.joblib"
        joblib.dump(payload, staged, compress=0)
        cache_hash = prep.sha256(staged)
        cache_bytes = staged.stat().st_size
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            _verify_cache(path, receipt_path, identity, preparation_sha256)
            return
        with os.fdopen(descriptor, "wb") as destination, staged.open("rb") as source:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())

    receipt = dict(schema=1, identity=identity,
                   input_sha256=identity["input_sha256"],
                   preparation_sha256=preparation_sha256,
                   source_hashes=identity["source_hashes"],
                   cache_sha256=cache_hash, sha256=cache_hash,
                   cache_bytes=cache_bytes, bytes=cache_bytes,
                   cache_path=path.name)
    try:
        with receipt_path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        _verify_cache(path, receipt_path, identity, preparation_sha256)


def verify_cache(path: Path, receipt_path: Path, expected_identity: dict,
                 expected_preparation_sha256: str) -> dict:
    """Validate a cache/receipt pair and return the receipt without unpickling."""
    return _verify_cache(Path(path), Path(receipt_path), expected_identity,
                         expected_preparation_sha256)


def prepare_training(item: dict, source_index: int, out: Path,
                     calibration: dict, data_dir: Path) -> tuple[Path, Path]:
    """Prepare one of the six new SkillCorner training matches immutably.

    ``source_index`` is its position in the expanded SkillCorner TRAIN list and
    must be 8..13, preserving the existing match-indexed augmentation seeds.
    """
    if (isinstance(source_index, bool) or not isinstance(source_index, (int, np.integer))
            or not 8 <= int(source_index) <= 13):
        raise ValueError("new SkillCorner source_index must be in 8..13")
    if item.get("source") != "skillcorner":
        raise ValueError("training expansion helper accepts only SkillCorner rows")
    if item.get("match") != NEW_SKILLCORNER_TRAIN[int(source_index) - 8]:
        raise ValueError("source_index does not match the frozen six-match expansion order")
    if item.get("role") not in (None, "train"):
        raise ValueError("expected a SkillCorner TRAIN manifest row")
    calibration = _calibration_copy(calibration)
    input_path, input_sha256 = _input_path(item, data_dir)
    out = Path(out)

    clean_config = prep.recipe("clean")
    clean_directory = prep.prepare_match(
        item, data_dir, out / "base_records", clean_config)
    clean_base = prep.load_record(clean_directory)
    clean_receipt_path = clean_directory / "receipt.json"
    clean_receipt_sha256 = prep.sha256(clean_receipt_path)

    # Grading and native masks are fixed before any training corruption.
    inputs, labels = ms.view(input_path)
    labels = np.asarray(labels).astype(str)
    full_y = _encode_labels(labels)
    full_period = supervised.model_periods(inputs)
    fps = float(inputs["fps"])
    expected_indices = lt.sampled_indices(inputs)
    if (not np.array_equal(clean_base.get("native_labels"), labels)
            or not np.array_equal(clean_base.get("native_period"), inputs["period"])
            or not np.array_equal(clean_base.get("indices"), expected_indices)
            or not np.array_equal(clean_base.get("y"), full_y[expected_indices])
            or not np.array_equal(clean_base.get("period"), full_period[expected_indices])):
        raise ValueError(f"shared clean preparation does not align with native grading: {item['match']}")
    clean_record = _clean_record(clean_base, item["source"], item["match"])

    slot_seed = int(lt.SEED + int(source_index))
    gap_seed = ce.seed(item["match"], "train")
    native = tc.inputs_only(inputs)
    legacy = robust.masked_inputs(native, slot_seed)
    gap = lc.ball_gap_view(native, calibration, gap_seed)
    dropout_legacy = gr.slot_view(legacy, slot_seed)
    dropout_gap = gr.slot_view(gap, slot_seed)

    records = {
        "clean": clean_record,
        "gap": _clean_record(prep.build_record(gap, labels, item["source"], item["match"]),
                             item["source"], item["match"]),
        "dropout_legacy": _clean_record(
            prep.build_record(dropout_legacy, labels, item["source"], item["match"]),
            item["source"], item["match"]),
        "dropout_gap": _clean_record(
            prep.build_record(dropout_gap, labels, item["source"], item["match"]),
            item["source"], item["match"]),
    }
    for name, record in records.items():
        if (not np.array_equal(record["indices"], expected_indices)
                or not np.array_equal(record["period"], full_period[expected_indices])
                or not np.array_equal(record["y"], full_y[expected_indices])):
            raise ValueError(f"view changed native targets or time blocks: {item['match']}/{name}")

    recipe = dict(
        schema=1, kind="skillcorner_expansion_combined_training_v1",
        source_index=int(source_index),
        legacy_ball_mask="robust.masked_inputs with lt.SEED + source_index",
        gap_view="light_corruption.ball_gap_view with ce.seed(match, 'train')",
        slot_mask="gru_robustness_experiment.slot_view with lt.SEED + source_index",
        clean_base_recipe=clean_config,
        versions=_versions(),
        calibration=calibration,
        clean_base_cache=clean_directory.name,
        clean_base_receipt_sha256=clean_receipt_sha256,
    )
    source_hashes = _source_hashes()
    identity = _identity("training", item, input_sha256, recipe, source_hashes)
    preparation_sha256 = _fingerprint(identity)
    payload = dict(records=records, full_y=full_y,
                   full_period=np.asarray(full_period), fps=fps,
                   source=item["source"], match=item["match"],
                   input_sha256=input_sha256,
                   preparation_sha256=preparation_sha256)
    cache_path = out / f"train_{item['match']}.joblib"
    receipt_path = cache_path.with_suffix(".json")
    _write_cache(cache_path, receipt_path, payload, identity, preparation_sha256)
    _verify_cache(cache_path, receipt_path, identity, preparation_sha256)
    return cache_path, receipt_path


def prepare_eval(item: dict, view: str, replicate: str, out: Path,
                 calibration: dict, data_dir: Path) -> tuple[Path, Path]:
    """Prepare one clean/feed-like external evaluation view, without fitting."""
    if (view, replicate) not in (("clean", "clean"),
                                 ("feed_like", "corrupt_1"),
                                 ("feed_like", "corrupt_2")):
        raise ValueError("expected clean/clean or feed_like/corrupt_{1,2}")
    if view == "feed_like":
        calibration = _calibration_copy(calibration)
    elif calibration is not None:
        # Keep clean cache identity independent of an unused calibration.
        calibration = None
    input_path, input_sha256 = _input_path(item, data_dir)
    inputs, labels = ms.view(input_path)
    labels = np.asarray(labels).astype(str)
    native_labels = labels.copy()
    native_period = supervised.model_periods(inputs)
    if view == "clean":
        viewed = tc.inputs_only(inputs)
    else:
        viewed = sc.corrupt(inputs, calibration,
                            ce.seed(item["match"], replicate),
                            keep=FEEDLIKE_COMPONENTS)
    viewed = tc.inputs_only(viewed)
    record = _clean_record(
        prep.build_record(viewed, labels, item["source"], item["match"]),
        item["source"], item["match"])
    full_y = _encode_labels(native_labels)
    indices = lt.sampled_indices(inputs)
    if (not np.array_equal(record["indices"], indices)
            or not np.array_equal(record["period"], native_period[indices])
            or not np.array_equal(record["y"], full_y[indices])):
        raise ValueError(f"evaluation corruption changed targets/time: {item['match']}/{replicate}")

    recipe = dict(schema=1, kind="skillcorner_expansion_eval_v1", view=view,
                  replicate=replicate, calibration=calibration,
                  corruption_components=list(FEEDLIKE_COMPONENTS) if view == "feed_like" else [],
                  versions=_versions(),
                  seed=ce.seed(item["match"], replicate) if view == "feed_like" else None)
    source_hashes = _source_hashes()
    identity = _identity("evaluation", item, input_sha256, recipe, source_hashes)
    preparation_sha256 = _fingerprint(identity)
    payload = dict(inputs=tc.inputs_only(viewed), labels=native_labels,
                   record=record, source=item["source"], match=item["match"],
                   view=view, replicate=replicate, input_sha256=input_sha256,
                   preparation_sha256=preparation_sha256)
    out = Path(out)
    cache_path = out / f"eval_{item['match']}_{view}_{replicate}.joblib"
    receipt_path = cache_path.with_suffix(".json")
    _write_cache(cache_path, receipt_path, payload, identity, preparation_sha256)
    _verify_cache(cache_path, receipt_path, identity, preparation_sha256)
    return cache_path, receipt_path

