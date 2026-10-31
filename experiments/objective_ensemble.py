"""Compare saved combined-objective probabilities with the active baseline.

This follow-up performs no fitting. It averages the two already saved combined
loss probabilities, applies the current cost-3 decoder, and scores those outputs
beside the saved current-baseline probabilities on the same full-provider
development cells.

The comparison was conceived after the objective-alignment results had been
inspected. Its thresholds are a new, explicitly post-hoc practical screen; this
is not blind confirmation or an independent holdout.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import dead_transfer, learned_temporal as lt  # noqa: E402
from experiments import metrics, spell_diagnostics, supervised  # noqa: E402


STUDY_ID = "objective_ensemble_v1"
OUT_RELATIVE = Path("experiments") / STUDY_ID
OBJECTIVE_ARCHIVE_MANIFEST_SHA256 = (
    "cfd124e4091d14bb6deff126a5c87df0c3ae266a300c46bd2203ace6fdae02e0"
)
SEED_GAP_ARCHIVE_MANIFEST_SHA256 = (
    "11ba724c305c9dc42a65f0a62742e89256d292b8df31f45fa274a774916f4a59"
)
OBJECTIVE_ARCHIVE_MANIFEST = "objective_alignment_cache_manifest.json"
SEED_GAP_ARCHIVE_MANIFEST = "seed_gap_context_cache_manifest.json"
OBJECTIVE_ROOT = Path("experiments/objective_alignment_v1")
SEED_GAP_ROOT = Path("experiments/seed_gap_context_v1")
BASELINE_REPORT_RELATIVE = Path("experiments/results/seed_gap_context.json")
BASELINE_VERIFICATION_RELATIVE = Path(
    "experiments/results/seed_gap_context_verification.json"
)
BASELINE_PROMOTION_RELATIVE = Path(
    "experiments/results/baseline_seed_ensemble_promotion.json"
)
CURRENT_BASELINE_MANIFEST_RELATIVE = Path("experiments/baseline_manifest.json")
CURRENT_POLICY_RELATIVE = Path("experiments/model_decision_policy.json")
EVAL_MANIFEST_RELATIVE = Path("experiments/eval_set/manifest.json")
PREVIOUS_OBJECTIVE_FILES = (
    "predeclared.json",
    "report.json",
    "runtime_addendum.json",
    "checkpoints_frozen.json",
    "source_metadata.json",
    "evaluation_cache_manifest.json",
)
SOURCE_FILES = (
    "experiments/objective_ensemble.py",
    "experiments/metrics.py",
    "experiments/supervised.py",
    "experiments/learned_temporal.py",
    "experiments/spell_diagnostics.py",
    "experiments/dead_transfer.py",
    "experiments/current_baseline.py",
    "experiments/eval_set/manifest.json",
    "experiments/baseline_manifest.json",
    "experiments/model_decision_policy.json",
    "experiments/results/seed_gap_context.json",
    "experiments/results/seed_gap_context_verification.json",
    "experiments/results/baseline_seed_ensemble_promotion.json",
    "evaluation/possession_metrics.py",
)
VIEWS = (
    ("clean", "clean"),
    ("feed_like", "corrupt_1"),
    ("feed_like", "corrupt_2"),
)
SEEDS = (2026, 2027)
RECIPE_CONTROL = "current_baseline_equal_probability_ensemble"
RECIPE_CANDIDATE = "combined_objective_equal_probability_ensemble"
METRIC_FIELDS = (
    "spell_f1",
    "macro_f1",
    "weighted_70_30",
    "d_f1",
    "d_precision",
    "d_recall",
    "short_spell_recall",
)
IOU_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
PROBABILITY_SUM_ATOL = 1e-5
GATE = {
    "clean_spell_delta": "strictly greater than 0.0 at equal-provider mean",
    "feed_like_spell_delta": "strictly greater than 0.0 at equal-provider mean",
    "provider_clean_spell_delta": "each provider at least -0.02",
    "provider_clean_macro_f1_delta": "each provider at least -0.02",
    "provider_clean_d_f1_delta": "each provider at least -0.02",
}


class StudyError(RuntimeError):
    """Raised when frozen inputs, score rows, or declaration do not validate."""


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_array(value: np.ndarray) -> str:
    return digest_bytes(np.ascontiguousarray(value).tobytes())


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _data_root(path: Path | None = None) -> Path:
    chosen = path or (Path(os.environ["MTAH_DATA_DIR"])
                      if os.environ.get("MTAH_DATA_DIR") else ROOT / "data")
    return chosen.expanduser().resolve()


def _archive_root(path: Path | None = None) -> Path:
    chosen = path or (Path(os.environ["MTAH_ARCHIVE_ROOT"])
                      if os.environ.get("MTAH_ARCHIVE_ROOT") else Path("/workspace/mtah-data"))
    return chosen.expanduser().resolve()


def _relative_path(path: str | Path) -> str:
    return PurePosixPath(str(path).replace("\\", "/")).as_posix()


def _archive_manifest(archive_root: Path, filename: str, expected_sha: str) -> dict[str, dict]:
    path = archive_root / filename
    if not path.is_file():
        raise StudyError(f"Missing pinned private archive manifest: {path}")
    actual_sha = digest_file(path)
    if actual_sha != expected_sha:
        raise StudyError(f"Archive manifest SHA mismatch for {path}: {actual_sha}")
    payload = _json(path)
    files = payload.get("files")
    if not isinstance(files, list):
        raise StudyError(f"Malformed archive manifest {path}: files must be a list")
    result = {}
    for entry in files:
        rel = _relative_path(entry.get("path", ""))
        if not rel or rel in result:
            raise StudyError(f"Malformed or duplicate archive manifest path: {rel}")
        result[rel] = entry
    return result


def _archived_fact(data_root: Path, file_index: dict[str, dict], rel: str) -> dict:
    if rel not in file_index:
        raise StudyError(f"{rel} is absent from the pinned private file manifest")
    entry = file_index[rel]
    path = data_root.joinpath(*PurePosixPath(rel).parts)
    if not path.is_file():
        raise StudyError(f"Missing archived file: {path}")
    actual_bytes = path.stat().st_size
    actual_sha = digest_file(path)
    if actual_bytes != entry.get("bytes") or actual_sha != entry.get("sha256"):
        raise StudyError(f"Archived file does not match pinned manifest: {path}")
    return {"path": rel, "bytes": actual_bytes, "sha256": actual_sha}


def _cohort() -> list[dict[str, str]]:
    rows = []
    for source, split in dead_transfer.SPLITS.items():
        for path in split["eval"]:
            rows.append({"source": source, "match": path.stem})
    expected = {
        "sportec": {"J03WOY"},
        "metrica": {"metrica_3"},
        "skillcorner": {
            "skillcorner_had_1996435",
            "skillcorner_had_1996436",
            "skillcorner_had_2006229",
        },
    }
    observed = {provider: set() for provider in expected}
    for row in rows:
        observed[row["source"]].add(row["match"])
    if observed != expected:
        raise StudyError(f"Full provider evaluation cohort changed: {observed}")
    return rows


def _cells() -> list[dict[str, str]]:
    cells = []
    for match in _cohort():
        for view, replicate in VIEWS:
            cells.append({**match, "view": view, "replicate": replicate})
    if len(cells) != 15:
        raise StudyError(f"Expected 15 matched evaluation cells, got {len(cells)}")
    return cells


def _cell_key(row: dict) -> tuple[str, str, str, str]:
    return row["source"], row["match"], row["view"], row["replicate"]


def _expected_cell_paths(cell: dict[str, str]) -> tuple[str, str]:
    stem = f"{cell['match']}_{cell['view']}_{cell['replicate']}"
    return (
        f"{SEED_GAP_ROOT.as_posix()}/{stem}.joblib",
        f"{SEED_GAP_ROOT.as_posix()}/{stem}.npz",
    )


def _required_archive_paths() -> tuple[list[str], list[str]]:
    objective = [f"{OBJECTIVE_ROOT.as_posix()}/{name}" for name in PREVIOUS_OBJECTIVE_FILES]
    for source in dead_transfer.SPLITS:
        for seed in SEEDS:
            objective.extend([
                f"{OBJECTIVE_ROOT.as_posix()}/without_{source}_combined_{seed}.joblib",
                f"{OBJECTIVE_ROOT.as_posix()}/without_{source}_combined_{seed}.json",
            ])
            # One candidate was among the three fits started before runtime
            # receipts were introduced. Validate this exemption against the
            # pinned runtime addendum before freezing.
            if (source, seed) != ("skillcorner", 2027):
                objective.append(
                    f"{OBJECTIVE_ROOT.as_posix()}/without_{source}_combined_{seed}.runtime.json")
    objective.extend(
        f"{OBJECTIVE_ROOT.as_posix()}/{source}_{cell['match']}_{cell['view']}_"
        f"{cell['replicate']}_combined_{seed}.npz"
        for cell in _cells()
        for seed in SEEDS
        for source in (cell["source"],)
    )
    seed = [path for cell in _cells() for path in _expected_cell_paths(cell)]
    return sorted(set(objective)), sorted(set(seed))


def _path_fact(data_root: Path, rel: str) -> dict[str, Any]:
    del data_root
    path = ROOT.joinpath(*PurePosixPath(rel).parts)
    if not path.is_file():
        raise StudyError(f"Missing comparison input: {path}")
    return {"path": rel, "bytes": path.stat().st_size, "sha256": digest_file(path)}


def _verify_current_baseline(data_root: Path) -> tuple[dict, dict, dict[str, str]]:
    baseline_path = ROOT / CURRENT_BASELINE_MANIFEST_RELATIVE
    policy_path = ROOT / CURRENT_POLICY_RELATIVE
    if not baseline_path.is_file() or not policy_path.is_file():
        raise StudyError("Current baseline manifest or decision policy is missing")
    manifest = _json(baseline_path)
    policy = _json(policy_path)
    manifest_sha = digest_file(baseline_path)
    policy_sha = digest_file(policy_path)
    active = policy.get("working_candidate", {})
    if active.get("model") != "joint_gru_seed_ensemble":
        raise StudyError("Active model is no longer joint_gru_seed_ensemble")
    if active.get("baseline_manifest_sha256") != manifest_sha:
        raise StudyError("Decision policy does not bind the current baseline manifest")
    model_hashes = manifest.get("models_sha256", {})
    expected_members = {"joint_gru_2026.joblib", "joint_gru_2027.joblib"}
    if set(model_hashes) != expected_members:
        raise StudyError(f"Unexpected active ensemble members: {sorted(model_hashes)}")
    architecture = manifest.get("architecture", {})
    members = architecture.get("members", {})
    if members != {"joint_gru_2026": 0.5, "joint_gru_2027": 0.5}:
        raise StudyError("Active baseline is not the frozen equal-probability two-seed ensemble")
    if architecture.get("decoder_switch_cost") != 3:
        raise StudyError("Current baseline decoder cost changed from 3")
    evidence = manifest.get("evidence", {})
    report_sha = digest_file(ROOT / BASELINE_REPORT_RELATIVE)
    verification_sha = digest_file(ROOT / BASELINE_VERIFICATION_RELATIVE)
    if evidence.get("report_sha256") != report_sha:
        raise StudyError("Active baseline manifest does not bind the saved seed-gap report")
    if evidence.get("verification_sha256") != verification_sha:
        raise StudyError("Active baseline manifest does not bind the saved verification receipt")
    return manifest, policy, {"baseline_manifest": manifest_sha, "decision_policy": policy_sha}


def _prior_baseline_rows(data_root: Path) -> dict[tuple[str, str, str, str], dict]:
    del data_root
    report_path = ROOT / BASELINE_REPORT_RELATIVE
    report = _json(report_path)
    rows = {}
    for row in report.get("rows", []):
        if row.get("recipe") != "equal_probability_ensemble":
            continue
        key = _cell_key(row)
        if key in rows:
            raise StudyError(f"Duplicate archived baseline row for {key}")
        rows[key] = row
    expected = {_cell_key(cell) for cell in _cells()}
    if set(rows) != expected:
        raise StudyError("Saved baseline report does not contain the exact 15 comparison cells")
    return rows


def _build_declaration(data_root: Path, archive_root: Path) -> dict[str, Any]:
    objective_manifest_sha = digest_file(archive_root / OBJECTIVE_ARCHIVE_MANIFEST)
    seed_gap_manifest_sha = digest_file(archive_root / SEED_GAP_ARCHIVE_MANIFEST)
    if objective_manifest_sha != OBJECTIVE_ARCHIVE_MANIFEST_SHA256:
        raise StudyError("Pinned objective-alignment archive manifest has changed")
    if seed_gap_manifest_sha != SEED_GAP_ARCHIVE_MANIFEST_SHA256:
        raise StudyError("Pinned seed-gap archive manifest has changed")
    objective_index = _archive_manifest(
        archive_root, OBJECTIVE_ARCHIVE_MANIFEST, OBJECTIVE_ARCHIVE_MANIFEST_SHA256)
    seed_gap_index = _archive_manifest(
        archive_root, SEED_GAP_ARCHIVE_MANIFEST, SEED_GAP_ARCHIVE_MANIFEST_SHA256)
    objective_paths, seed_paths = _required_archive_paths()
    archive_facts = []
    for rel in objective_paths:
        archive_facts.append(_archived_fact(data_root, objective_index, rel))
    for rel in seed_paths:
        archive_facts.append(_archived_fact(data_root, seed_gap_index, rel))

    runtime_addendum = _json(data_root / OBJECTIVE_ROOT / "runtime_addendum.json")
    runtime_addendum_sha = digest_file(data_root / OBJECTIVE_ROOT / "runtime_addendum.json")
    initial_fits = set(runtime_addendum.get("initial_fits_started_without_adapter", []))
    expected_initial_fits = {
        "skillcorner/spell_sqrt/2026",
        "sportec/spell_sqrt/2026",
        "skillcorner/combined/2027",
    }
    if initial_fits != expected_initial_fits:
        raise StudyError("Pinned adapter addendum does not declare the exact initial-fit set")

    facts_by_path = {fact["path"]: fact for fact in archive_facts}
    checkpoint_manifest = _json(data_root / OBJECTIVE_ROOT / "checkpoints_frozen.json")
    prior_declaration = _json(data_root / OBJECTIVE_ROOT / "predeclared.json")
    prior_declaration_sha = digest_file(data_root / OBJECTIVE_ROOT / "predeclared.json")
    expected_checkpoint_names = {
        f"without_{source}_combined_{seed}.joblib"
        for source in dead_transfer.SPLITS for seed in SEEDS
    }
    checkpoint_rows = {
        entry["file"]: entry for entry in checkpoint_manifest.get("new_checkpoints", [])
        if entry.get("file") in expected_checkpoint_names
    }
    if set(checkpoint_rows) != expected_checkpoint_names:
        raise StudyError("Frozen checkpoint manifest does not list all six combined models")
    checkpoint_facts = []
    for name, entry in sorted(checkpoint_rows.items()):
        heldout = name.removeprefix("without_").split("_combined_")[0]
        seed = int(name.removesuffix(".joblib").rsplit("_", 1)[1])
        model_rel = f"{OBJECTIVE_ROOT.as_posix()}/{name}"
        receipt_rel = f"{OBJECTIVE_ROOT.as_posix()}/{name.removesuffix('.joblib')}.json"
        model_fact = facts_by_path[model_rel]
        receipt_fact = facts_by_path[receipt_rel]
        if model_fact["sha256"] != entry.get("sha256"):
            raise StudyError(f"Checkpoint manifest model hash mismatch: {name}")
        if receipt_fact["sha256"] != entry.get("receipt_sha256"):
            raise StudyError(f"Checkpoint manifest receipt hash mismatch: {name}")
        receipt = _json(data_root / OBJECTIVE_ROOT / Path(receipt_rel).name)
        if receipt.get("sha256") != model_fact["sha256"]:
            raise StudyError(f"Candidate model receipt does not bind its checkpoint: {name}")
        if (receipt.get("heldout") != heldout or receipt.get("seed") != seed or
                receipt.get("arm") != "combined" or receipt.get("objective") != "combined"):
            raise StudyError(f"Candidate fit receipt has the wrong fold or arm: {name}")
        if receipt.get("predeclared_sha256") != prior_declaration_sha:
            raise StudyError(f"Candidate fit receipt does not bind the frozen declaration: {name}")
        if receipt.get("sources") != prior_declaration["training_sources"][heldout]:
            raise StudyError(f"Candidate fit receipt has the wrong training sources: {name}")
        if receipt.get("training_matches") != prior_declaration["training_matches"][heldout]:
            raise StudyError(f"Candidate fit receipt has the wrong training matches: {name}")
        runtime_rel = f"{OBJECTIVE_ROOT.as_posix()}/{name.removesuffix('.joblib')}.runtime.json"
        runtime_fact = facts_by_path.get(runtime_rel)
        if runtime_fact is None and name != "without_skillcorner_combined_2027.joblib":
            raise StudyError(f"Missing runtime receipt without a declared exemption: {name}")
        if runtime_fact is not None:
            runtime_receipt = _json(data_root / OBJECTIVE_ROOT / Path(runtime_rel).name)
            runtime_checks = runtime_receipt.get("checks", {})
            if (runtime_receipt.get("heldout") != heldout or
                    runtime_receipt.get("seed") != seed or
                    runtime_receipt.get("arm") != "combined" or
                    runtime_receipt.get("objective") != "combined" or
                    runtime_receipt.get("adapter_sha256") != runtime_addendum.get("adapter_sha256") or
                    runtime_receipt.get("runtime_addendum_sha256") != runtime_addendum_sha or
                    runtime_receipt.get("objective_predeclared_sha256") != prior_declaration_sha or
                    runtime_receipt.get("model_sha256") != model_fact["sha256"] or
                    runtime_receipt.get("model_receipt_sha256") != receipt_fact["sha256"] or
                    not runtime_checks or not all(runtime_checks.values())):
                raise StudyError(f"Candidate runtime receipt does not validate its fit: {name}")
        checkpoint_facts.append({
            "model": model_fact,
            "fit_receipt": receipt_fact,
            "runtime_receipt": runtime_fact,
        })

    baseline_manifest, policy, baseline_hashes = _verify_current_baseline(data_root)
    prior_rows = _prior_baseline_rows(data_root)
    eval_manifest = _json(ROOT / EVAL_MANIFEST_RELATIVE)
    eval_sources = {}
    for provider, entries in eval_manifest["sources"].items():
        selection = entries.get("selection", [])
        for record in selection:
            if record["match"] in {cell["match"] for cell in _cohort()}:
                eval_sources[record["match"]] = {
                    "source": provider,
                    "path": record["path"],
                    "sha256": record["sha256"],
                    "prior_exposure": record.get("prior_exposure"),
                }
    if set(eval_sources) != {row["match"] for row in _cohort()}:
        raise StudyError("Evaluation manifest does not identify every saved evaluation match")

    tracked_facts = [_path_fact(data_root, rel) for rel in SOURCE_FILES]
    manifest_proofs = {
        "objective_alignment_archive": {
            "path": OBJECTIVE_ARCHIVE_MANIFEST,
            "bytes": (archive_root / OBJECTIVE_ARCHIVE_MANIFEST).stat().st_size,
            "sha256": objective_manifest_sha,
        },
        "seed_gap_context_archive": {
            "path": SEED_GAP_ARCHIVE_MANIFEST,
            "bytes": (archive_root / SEED_GAP_ARCHIVE_MANIFEST).stat().st_size,
            "sha256": seed_gap_manifest_sha,
        },
    }
    policy_weights = policy.get("weighted_summary", {})
    if policy_weights.get("spell_weight") != 0.7 or policy_weights.get("macro_weight") != 0.3:
        raise StudyError("Current policy no longer declares the 70/30 auxiliary summary")
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before this comparison scores any cells",
        "question": (
            "Does the fixed 50/50 probability ensemble of the two saved combined-objective "
            "full-provider-holdout models outperform the active two-seed baseline on the "
            "same clean and feed-like development cells?"
        ),
        "historical_status": {
            "post_hoc": True,
            "disclosure": (
                "The prior objective-alignment study and its outcomes were already inspected. "
                "This is a newly declared follow-up with fixed weights and post-hoc practical "
                "criteria; it is neither blind confirmation nor an independent holdout."
            ),
            "prior_declaration_sha256": next(
                x["sha256"] for x in archive_facts
                if x["path"] == f"{OBJECTIVE_ROOT.as_posix()}/predeclared.json"),
            "prior_report_sha256": next(
                x["sha256"] for x in archive_facts
                if x["path"] == f"{OBJECTIVE_ROOT.as_posix()}/report.json"),
            "prior_runtime_addendum_sha256": next(
                x["sha256"] for x in archive_facts
                if x["path"] == f"{OBJECTIVE_ROOT.as_posix()}/runtime_addendum.json"),
            "organizer_labels_or_scores_used": False,
            "independent_confirmation": False,
            "runtime_receipt_exemptions": sorted(initial_fits),
            "runtime_receipt_exemptions": sorted(initial_fits),
        },
        "recipes": {
            "control": RECIPE_CONTROL,
            "candidate": RECIPE_CANDIDATE,
            "probability_rule": (
                "Equal arithmetic mean of the saved native probability arrays for seeds "
                "2026 and 2027, before decoding; no weight search or score averaging."
            ),
            "decoder": "Unchanged supervised Viterbi decoder, switch cost 3, frame-native fps.",
            "training": "No training or checkpoint fitting occurs in this comparison.",
        },
        "primary_metric": "Equal-provider mean official possession-spell F1.",
        "auxiliary_summary": {
            "formula": "0.70 * spell_f1 + 0.30 * macro_f1",
            "used_for_ranking": False,
            "weights_from_policy": policy_weights,
        },
        "aggregation": (
            "Within each provider, average its matched evaluation matches and feed-like "
            "replicates; then average the three provider means equally."
        ),
        "selection_criteria": {
            **GATE,
            "context": "Post-hoc practical screen declared before this comparison is scored.",
            "automatic_promotion": False,
        },
        "cohort": _cohort(),
        "cells": _cells(),
        "views": [
            {"view": view, "replicates": [rep for v, rep in VIEWS if v == view]}
            for view in ("clean", "feed_like")
        ],
        "baseline": {
            "model": baseline_manifest.get("model"),
            "package_version": baseline_manifest.get("package_version"),
            "manifest_sha256": baseline_hashes["baseline_manifest"],
            "decision_policy_sha256": baseline_hashes["decision_policy"],
            "model_sha256": baseline_manifest.get("models_sha256"),
            "saved_report_sha256": digest_file(ROOT / BASELINE_REPORT_RELATIVE),
            "verification_sha256": digest_file(ROOT / BASELINE_VERIFICATION_RELATIVE),
            "promotion_receipt_sha256": digest_file(ROOT / BASELINE_PROMOTION_RELATIVE),
            "manifest_evidence": baseline_manifest.get("evidence"),
            "weights_available_in_this_workspace": False,
        },
        "evaluation_sources": eval_sources,
        "candidate_checkpoints": checkpoint_facts,
        "private_archive_manifests": manifest_proofs,
        "verified_archive_files": sorted(archive_facts, key=lambda item: item["path"]),
        "tracked_source_files": sorted(tracked_facts, key=lambda item: item["path"]),
        "source_sha256": {fact["path"]: fact["sha256"] for fact in tracked_facts},
        "prior_baseline_row_count": len(prior_rows),
        "baseline_probabilities": [
            {
                "source": source,
                "match": match,
                "view": view,
                "replicate": replicate,
                "probability_sha256": row["probability_sha256"],
                "prediction_sha256": row["prediction_sha256"],
            }
            for (source, match, view, replicate), row in sorted(prior_rows.items())
        ],
    }


def freeze(data_root: Path | None = None, archive_root: Path | None = None) -> dict[str, Any]:
    data = _data_root(data_root)
    archive = _archive_root(archive_root)
    out = data / OUT_RELATIVE
    declaration_path = out / "declaration.json"
    if declaration_path.exists():
        raise StudyError(f"Study declaration is immutable and already exists: {declaration_path}")
    declaration = _build_declaration(data, archive)
    out.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(declaration, sort_keys=True, indent=2, ensure_ascii=False,
                         allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(declaration_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    declaration_sha = digest_bytes(payload)
    return {"declaration": str(declaration_path), "declaration_sha256": declaration_sha,
            "cells": len(declaration["cells"]), "scored": False}


def _verify_declaration(data_root: Path, archive_root: Path) -> tuple[dict, str]:
    path = data_root / OUT_RELATIVE / "declaration.json"
    if not path.is_file():
        raise StudyError(f"No frozen declaration found at {path}; run freeze first")
    declaration = _json(path)
    source_hashes = declaration.get("source_sha256", {})
    for rel, expected in source_hashes.items():
        source_path = ROOT.joinpath(*PurePosixPath(rel).parts)
        if not source_path.is_file() or digest_file(source_path) != expected:
            raise StudyError(f"Frozen source changed since declaration: {rel}")
    # Recompute all declared input checksums, not only the top-level manifests.
    current = _build_declaration(data_root, archive_root)
    immutable_keys = (
        "question", "recipes", "primary_metric", "auxiliary_summary", "aggregation",
        "selection_criteria", "cohort", "cells", "views", "baseline", "evaluation_sources",
        "private_archive_manifests", "verified_archive_files", "tracked_source_files",
        "source_sha256", "baseline_probabilities", "prior_baseline_row_count", "historical_status",
        "candidate_checkpoints",
    )
    for key in immutable_keys:
        if canonical_json(current.get(key)) != canonical_json(declaration.get(key)):
            raise StudyError(f"Frozen declaration input changed: {key}")
    return declaration, digest_file(path)


def check(data_root: Path | None = None, archive_root: Path | None = None) -> dict[str, Any]:
    declaration, sha = _verify_declaration(_data_root(data_root), _archive_root(archive_root))
    return {"declaration_sha256": sha, "cells": len(declaration["cells"]),
            "inputs_verified": True, "scored": False}


def blend_probabilities(probability_2026: np.ndarray,
                        probability_2027: np.ndarray) -> np.ndarray:
    """Return the fixed 50/50 native-probability ensemble after strict validation."""
    p0 = np.asarray(probability_2026)
    p1 = np.asarray(probability_2027)
    if p0.ndim != 2 or p0.shape[1] != 3 or p1.shape != p0.shape:
        raise ValueError("Both inputs must have matching shape (n_frames, 3)")
    if not np.issubdtype(p0.dtype, np.number) or not np.issubdtype(p1.dtype, np.number):
        raise ValueError("Probability arrays must be numeric")
    if not np.isfinite(p0).all() or not np.isfinite(p1).all():
        raise ValueError("Probability arrays must be finite")
    if (p0 < 0).any() or (p1 < 0).any():
        raise ValueError("Probability arrays cannot contain negative values")
    if not np.allclose(p0.sum(axis=1), 1.0, rtol=0, atol=PROBABILITY_SUM_ATOL):
        raise ValueError("2026 probability rows must sum to one")
    if not np.allclose(p1.sum(axis=1), 1.0, rtol=0, atol=PROBABILITY_SUM_ATOL):
        raise ValueError("2027 probability rows must sum to one")
    return (p0.astype(np.float64, copy=False) + p1.astype(np.float64, copy=False)) / 2.0


def score_prediction(labels: np.ndarray, prediction: np.ndarray, inputs: dict,
                     recipe: str, cell: dict[str, str], probability_sha256: str,
                     probability_file_sha256: str | None = None) -> dict[str, Any]:
    """Score one already-decoded synthetic or saved prediction with shared metrics."""
    truth = np.asarray(labels).astype(str)
    guess = np.asarray(prediction).astype(str)
    period = np.asarray(inputs["period"])
    fps = float(np.asarray(inputs["fps"]))
    timestamps = np.asarray(inputs["timestamps"])
    if truth.ndim != 1 or guess.shape != truth.shape:
        raise ValueError("Labels and prediction must be equal one-dimensional arrays")
    if period.shape != truth.shape or timestamps.shape != truth.shape:
        raise ValueError("period and timestamps must align with labels")
    result = metrics.evaluate_tracking(
        truth, guess, period=period, fps=fps, timestamps=timestamps)
    diag = spell_diagnostics.spell_diagnostics(
        truth, guess, period=period, fps=fps, timestamps=timestamps)
    d = result["per_class"]["D"]
    short = diag["per_iou"]["0.5"]["reference_recall_groups"]["under_3_seconds"]
    row = {
        **cell,
        "recipe": recipe,
        "spell_f1": float(result["spell_f1_mean"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_70_30": float(.70 * result["spell_f1_mean"] + .30 * result["macro_f1"]),
        "d_f1": float(d["f1"]),
        "d_precision": float(d["precision"]),
        "d_recall": float(d["recall"]),
        "spell_by_iou": {
            str(threshold): float(result["spell_f1"][threshold]["f1"])
            for threshold in IOU_THRESHOLDS
        },
        "n_gt_spells": int(result["n_gt_spells"]),
        "n_pred_spells": int(result["n_pred_spells"]),
        "short_spell_recall": short["recall"],
        "short_reference_spells": int(short["n_gt"]),
        "short_matched_spells": int(short["n_matched"]),
        "fragmented_reference_spells": int(
            diag["overlap_geometry"]["reference_overlapped_by_multiple_same_team_predictions"]),
        "reference_spells_touching_graded_run_edge": int(
            diag["reference_spells_touching_graded_run_edge"]),
        "complete_spell_boundary_timing_iou_0_5": diag[
            "matched_complete_boundary_errors_seconds_at_iou_0_5"],
        "graded_frames": int(result["n_graded"]),
        "probability_sha256": probability_sha256,
        "prediction_sha256": digest_array(guess),
        "baseline_probability_file_sha256": probability_file_sha256,
    }
    return row


def _verify_probability(probability: np.ndarray, n_frames: int, description: str) -> None:
    p = np.asarray(probability)
    if p.shape != (n_frames, 3) or not np.issubdtype(p.dtype, np.number):
        raise StudyError(f"{description} must have shape ({n_frames}, 3)")
    if not np.isfinite(p).all() or (p < 0).any():
        raise StudyError(f"{description} contains invalid probabilities")
    if not np.allclose(p.sum(axis=1), 1, rtol=0, atol=PROBABILITY_SUM_ATOL):
        raise StudyError(f"{description} contains rows that do not sum to one")


def _baseline_archive_row(data_root: Path, cell: dict[str, str]) -> tuple[np.ndarray, Path]:
    rel = f"{SEED_GAP_ROOT.as_posix()}/{cell['match']}_{cell['view']}_{cell['replicate']}.npz"
    path = data_root.joinpath(*PurePosixPath(rel).parts)
    with np.load(path, allow_pickle=False) as archive:
        if "equal_probability_ensemble" not in archive.files:
            raise StudyError(f"Missing baseline ensemble probability in {path}")
        probability = archive["equal_probability_ensemble"]
    return probability, path


def _candidate_probabilities(data_root: Path, cell: dict[str, str]) -> tuple[np.ndarray, list[dict]]:
    members = []
    member_rows = []
    for seed in SEEDS:
        rel = (
            f"{OBJECTIVE_ROOT.as_posix()}/{cell['source']}_{cell['match']}_"
            f"{cell['view']}_{cell['replicate']}_combined_{seed}.npz"
        )
        path = data_root.joinpath(*PurePosixPath(rel).parts)
        with np.load(path, allow_pickle=False) as archive:
            if not {"probability", "prediction"}.issubset(archive.files):
                raise StudyError(f"Candidate archive is missing arrays: {path}")
            p = archive["probability"]
            saved_prediction = archive["prediction"]
        members.append(p)
        member_rows.append({"path": rel, "path_sha256": digest_file(path),
                            "probability_sha256": digest_array(p),
                            "saved_prediction_sha256": digest_array(saved_prediction),
                            "saved_prediction": saved_prediction})
    return blend_probabilities(*members), member_rows


def _verify_row_parity(actual: dict, saved: dict, key: tuple) -> None:
    scalar_fields = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision", "d_recall")
    for name in scalar_fields:
        if not math.isclose(float(actual[name]), float(saved[name]), rel_tol=0, abs_tol=1e-12):
            raise StudyError(f"Baseline saved-score parity failed for {key}: {name}")
    if actual["n_gt_spells"] != saved["n_gt_spells"] or actual["n_pred_spells"] != saved["n_pred_spells"]:
        raise StudyError(f"Baseline spell-count parity failed for {key}")
    if actual["probability_sha256"] != saved["probability_sha256"]:
        raise StudyError(f"Baseline probability digest parity failed for {key}")
    if actual["prediction_sha256"] != saved["prediction_sha256"]:
        raise StudyError(f"Baseline prediction digest parity failed for {key}")
    for threshold, value in actual["spell_by_iou"].items():
        if not math.isclose(float(value), float(saved["spell_thresholds"][threshold]),
                            rel_tol=0, abs_tol=1e-12):
            raise StudyError(f"Baseline IoU parity failed for {key} at {threshold}")


def run(data_root: Path | None = None, archive_root: Path | None = None) -> dict[str, Any]:
    data = _data_root(data_root)
    archive = _archive_root(archive_root)
    declaration, declaration_sha = _verify_declaration(data, archive)
    out = data / OUT_RELATIVE
    result_path = out / "runs.json"
    if result_path.exists():
        raise StudyError(f"Saved run already exists and is immutable: {result_path}")
    baseline_rows = _prior_baseline_rows(data)
    rows = []
    for cell in declaration["cells"]:
        cache_rel = _expected_cell_paths(cell)[0]
        cache_path = data.joinpath(*PurePosixPath(cache_rel).parts)
        cache = joblib.load(cache_path)
        if not {"labels", "inputs"}.issubset(cache):
            raise StudyError(f"Evaluation cache lacks labels/inputs: {cache_path}")
        labels = np.asarray(cache["labels"]).astype(str)
        inputs = cache["inputs"]
        fps = float(np.asarray(inputs["fps"]))
        if labels.ndim != 1 or labels.size == 0 or np.asarray(inputs["period"]).shape != labels.shape:
            raise StudyError(f"Invalid cached evaluation row shapes: {cache_path}")
        if np.asarray(inputs["timestamps"]).shape != labels.shape:
            raise StudyError(f"Cached timestamps do not align: {cache_path}")
        time_blocks = supervised.model_periods(inputs)

        baseline_prob, baseline_path = _baseline_archive_row(data, cell)
        _verify_probability(baseline_prob, len(labels), "baseline probabilities")
        baseline_pred = supervised.decode(baseline_prob, time_blocks, fps, lt.COST)
        baseline_row = score_prediction(
            labels, baseline_pred, inputs, RECIPE_CONTROL, cell,
            digest_array(baseline_prob), digest_file(baseline_path))
        key = _cell_key(cell)
        _verify_row_parity(baseline_row, baseline_rows[key], key)
        baseline_row["probability_file"] = _relative_path(baseline_path.relative_to(data))
        rows.append(baseline_row)

        candidate_prob, members = _candidate_probabilities(data, cell)
        _verify_probability(candidate_prob, len(labels), "candidate ensemble probabilities")
        # Validate that each source archive's saved member prediction corresponds
        # exactly to its own P array under the same frozen decoder.
        for member in members:
            with np.load(data.joinpath(*PurePosixPath(member["path"]).parts),
                         allow_pickle=False) as member_archive:
                member_prob = member_archive["probability"]
            expected_member_prediction = supervised.decode(member_prob, time_blocks, fps, lt.COST)
            if not np.array_equal(member["saved_prediction"], expected_member_prediction):
                raise StudyError(f"Saved member prediction disagrees with its probabilities: {member['path']}")
            member.pop("saved_prediction")
        candidate_pred = supervised.decode(candidate_prob, time_blocks, fps, lt.COST)
        candidate_row = score_prediction(
            labels, candidate_pred, inputs, RECIPE_CANDIDATE, cell,
            digest_array(candidate_prob))
        candidate_name = (
            f"candidate_{cell['match']}_{cell['view']}_{cell['replicate']}.npz"
        )
        candidate_file = out / candidate_name
        np.savez_compressed(candidate_file, probability=candidate_prob,
                            prediction=candidate_pred.astype("U1"))
        candidate_row["probability_file"] = candidate_name
        candidate_row["probability_file_sha256"] = digest_file(candidate_file)
        candidate_row["members"] = members
        rows.append(candidate_row)

    expected_rows = len(declaration["cells"]) * 2
    if len(rows) != expected_rows:
        raise StudyError(f"Expected {expected_rows} recipe rows, got {len(rows)}")
    payload = {
        "study_id": STUDY_ID,
        "declaration_sha256": declaration_sha,
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "baseline_parity": {
            "matched_cells": len(declaration["cells"]),
            "all_probability_and_score_rows_reproduced": True,
        },
        "organizer_labels_read": False,
        "organizer_scored": False,
    }
    raw = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False,
                     allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(result_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"runs": str(result_path), "rows": len(rows),
            "baseline_parity": True, "runs_sha256": digest_bytes(raw)}


def _mean_metric(rows: list[dict], metric_name: str) -> float | None:
    values = []
    for row in rows:
        if metric_name == "short_spell_recall":
            value = row.get(metric_name)
        elif metric_name.startswith("iou_"):
            value = row["spell_by_iou"][metric_name.removeprefix("iou_")]
        else:
            value = row.get(metric_name)
        if value is not None:
            values.append(float(value))
    return float(np.mean(values)) if values else None


def aggregate_rows(rows: list[dict]) -> dict[str, Any]:
    """Equal-provider scorecard and the fixed post-hoc practical screen."""
    expected_keys = {
        (recipe, *_cell_key(cell))
        for recipe in (RECIPE_CONTROL, RECIPE_CANDIDATE)
        for cell in _cells()
    }
    observed = set()
    for row in rows:
        key = (row["recipe"], *_cell_key(row))
        if key in observed:
            raise StudyError(f"Duplicate output row: {key}")
        observed.add(key)
    if observed != expected_keys:
        raise StudyError(f"Scorecard cell mismatch: missing={len(expected_keys-observed)} "
                         f"extra={len(observed-expected_keys)}")

    all_metrics = list(METRIC_FIELDS) + [f"iou_{t:.1f}" for t in IOU_THRESHOLDS]
    providers = tuple(dead_transfer.SPLITS)
    summaries = {}
    for recipe in (RECIPE_CONTROL, RECIPE_CANDIDATE):
        summaries[recipe] = {}
        for view in ("clean", "feed_like"):
            provider_values = {}
            for provider in providers:
                provider_rows = [row for row in rows
                                 if row["recipe"] == recipe and row["source"] == provider
                                 and row["view"] == view]
                # Every match contributes once on clean, twice on feed-like.
                expected_count = 1 if view == "clean" else 2
                expected_provider_count = (
                    len(dead_transfer.SPLITS[provider]["eval"]) * expected_count
                )
                if len(provider_rows) != expected_provider_count:
                    raise StudyError(f"Incomplete provider cells for {recipe}/{provider}/{view}")
                provider_values[provider] = {
                    metric_name: _mean_metric(provider_rows, metric_name)
                    for metric_name in all_metrics
                }
                provider_values[provider]["n_cells"] = len(provider_rows)
            summaries[recipe][view] = {
                "per_provider": provider_values,
                "equal_provider_mean": {
                    metric_name: float(np.mean([
                        provider_values[p][metric_name] for p in providers
                    ])) for metric_name in all_metrics
                },
                "aggregation": "mean within provider, then equal mean across providers",
            }

    deltas = {}
    for view in ("clean", "feed_like"):
        control = summaries[RECIPE_CONTROL][view]
        candidate = summaries[RECIPE_CANDIDATE][view]
        deltas[view] = {
            "equal_provider_mean": {
                name: candidate["equal_provider_mean"][name]
                - control["equal_provider_mean"][name]
                for name in all_metrics
            },
            "per_provider": {
                provider: {
                    name: candidate["per_provider"][provider][name]
                    - control["per_provider"][provider][name]
                    for name in all_metrics
                }
                for provider in providers
            },
        }
    clean = deltas["clean"]["equal_provider_mean"]
    feed = deltas["feed_like"]["equal_provider_mean"]
    clean_provider = deltas["clean"]["per_provider"]
    checks = {
        "strict_positive_clean_spell_gain": clean["spell_f1"] > 0,
        "strict_positive_feed_like_spell_gain": feed["spell_f1"] > 0,
        "each_provider_clean_spell_delta_at_least_minus_0_02": all(
            clean_provider[p]["spell_f1"] >= -0.02 for p in providers),
        "each_provider_clean_macro_delta_at_least_minus_0_02": all(
            clean_provider[p]["macro_f1"] >= -0.02 for p in providers),
        "each_provider_clean_d_f1_delta_at_least_minus_0_02": all(
            clean_provider[p]["d_f1"] >= -0.02 for p in providers),
    }
    return {
        "recipes": summaries,
        "candidate_minus_control": deltas,
        "selection_gate": {
            "criteria": dict(GATE),
            "checks": checks,
            "passes": all(checks.values()),
            "post_hoc": True,
            "automatic_promotion": False,
        },
    }


def report(data_root: Path | None = None, archive_root: Path | None = None) -> dict[str, Any]:
    data = _data_root(data_root)
    archive = _archive_root(archive_root)
    declaration, declaration_sha = _verify_declaration(data, archive)
    out = data / OUT_RELATIVE
    runs_path = out / "runs.json"
    report_path = out / "report.json"
    if not runs_path.is_file():
        raise StudyError(f"No saved run results found at {runs_path}")
    if report_path.exists():
        raise StudyError(f"Saved report already exists and is immutable: {report_path}")
    runs = _json(runs_path)
    if runs.get("declaration_sha256") != declaration_sha:
        raise StudyError("Runs file belongs to a different frozen declaration")
    summary = aggregate_rows(runs.get("rows", []))
    result = {
        "study_id": STUDY_ID,
        "declaration_sha256": declaration_sha,
        "runs_sha256": digest_file(runs_path),
        "reported_utc": datetime.now(timezone.utc).isoformat(),
        "question": declaration["question"],
        "cohort": declaration["cohort"],
        "evaluation_status": declaration["historical_status"],
        "baseline": declaration["baseline"],
        "comparison": declaration["recipes"],
        "primary_metric": declaration["primary_metric"],
        "auxiliary_summary": declaration["auxiliary_summary"],
        "aggregation": declaration["aggregation"],
        "selection_criteria": declaration["selection_criteria"],
        "result": summary,
        "limitations": [
            "Previously exposed development matches; no independent confirmation was measured.",
            "Feed-like is a synthetic corruption recipe and does not establish real broadcast performance.",
            "The study compares saved full-provider-holdout model probabilities and does not refit them.",
            "A passing development screen would not itself authorize baseline promotion.",
        ],
    }
    raw = json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False,
                     allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"report": str(report_path), "report_sha256": digest_bytes(raw),
            "gate_passes": summary["selection_gate"]["passes"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check", "run", "report"))
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--archive-root", type=Path, default=None)
    args = parser.parse_args(argv)
    functions = {"freeze": freeze, "check": check, "run": run, "report": report}
    result = functions[args.command](args.data_dir, args.archive_root)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
