"""Zero-fit shifted-core inference experiment for the adopted H/A/D GRU.

The frozen candidate averages the existing 10-second core tiling with a
second tiling offset by five seconds. Each pass keeps the existing five-second
context and the existing full-provider checkpoints. Frames before the first
shifted core in a continuous block use the original-tiling probability; this
keeps every temporal break local and avoids fabricating a partial left context.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import time
from typing import Any

import joblib
import numpy as np

from experiments import (
    current_baseline,
    factorized_had,
    learned_temporal as lt,
    live_gap_experiment,
    objective_ensemble as oe,
    spell_diagnostics,
    supervised,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_ID = "window_overlap_v4"
OUT_RELATIVE = Path("experiments/window_overlap_v4")
DECLARATION = ROOT / "experiments/results/window_overlap_declaration_v4.json"
AGGREGATE = ROOT / "experiments/results/window_overlap_v4.json"
ARTIFACT_RECEIPT = ROOT / "experiments/results/window_overlap_artifacts_v4.json"
PRIOR_DECLARATION = ROOT / "experiments/results/objective_ensemble_declaration.json"
PRIOR_REPORT = ROOT / "experiments/results/objective_ensemble.json"
PRIOR_RUNS_RELATIVE = Path("experiments/objective_ensemble_v1/runs.json")
PRIOR_VERIFICATION = ROOT / "experiments/results/objective_ensemble_verification.json"
BASELINE_MANIFEST = ROOT / "experiments/baseline_manifest.json"
DECISION_POLICY = ROOT / "experiments/model_decision_policy.json"
INPUT_ROOT = Path("experiments/seed_gap_context_v1")
MODEL_ROOT = Path("experiments/objective_alignment_v1")
MEMBER_ROOT = MODEL_ROOT
PRIOR_OUTPUT_ROOT = Path("experiments/objective_ensemble_v1")
SEEDS = (2026, 2027)
RATE_HZ = 5
CORE_SECONDS = 10
CONTEXT_SECONDS = 5
CORE = CORE_SECONDS * RATE_HZ
CONTEXT = CONTEXT_SECONDS * RATE_HZ
OFFSET_SECONDS = 5
OFFSET = OFFSET_SECONDS * RATE_HZ
SWITCH_COST = 3
PROBABILITY_SUM_ATOL = 1e-5
SOURCE_FILES = (
    "experiments/window_overlap_experiment.py",
    "experiments/factorized_had.py",
    "experiments/live_gap_experiment.py",
    "experiments/learned_temporal.py",
    "experiments/supervised.py",
    "experiments/current_baseline.py",
    "experiments/objective_ensemble.py",
    "experiments/metrics.py",
    "experiments/eval_set/metrics.py",
    "evaluation/possession_metrics.py",
    "experiments/spell_diagnostics.py",
    "experiments/eval_set/manifest.json",
    "experiments/baseline_manifest.json",
    "experiments/model_decision_policy.json",
    "experiments/results/objective_ensemble_declaration.json",
    "experiments/results/objective_ensemble.json",
    "experiments/results/objective_ensemble_verification.json",
    "experiments/results/window_overlap_v1_prescore_deviation.json",
    "experiments/results/window_overlap_v2_prescore_deviation.json",
    "experiments/results/window_overlap_v2_replay_drift.json",
    "experiments/results/window_overlap_v3_prescore_deviation.json",
)


class StudyError(RuntimeError):
    """Raised when a frozen source, artifact, cohort, or score is inconsistent."""


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
    return json.loads(path.read_text(encoding="utf-8"))


def _data_root(path: Path | None = None) -> Path:
    chosen = path or (Path(os.environ["MTAH_DATA_DIR"])
                      if os.environ.get("MTAH_DATA_DIR") else ROOT / "data")
    return chosen.expanduser().resolve()


def _fact(path: Path, relative: str | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise StudyError(f"Missing frozen input: {path}")
    return {"path": relative or path.as_posix(), "bytes": path.stat().st_size,
            "sha256": digest_file(path)}


def _rel_fact(data_root: Path, relative: str) -> dict[str, Any]:
    return _fact(data_root.joinpath(*PurePosixPath(relative).parts), relative)


def _cells() -> list[dict[str, str]]:
    prior = _json(PRIOR_DECLARATION)
    cells = [dict(source=row["source"], match=row["match"], view=row["view"],
                  replicate=row["replicate"]) for row in prior["cells"]]
    expected = {
        ("sportec", "J03WOY"),
        ("metrica", "metrica_3"),
        ("skillcorner", "skillcorner_had_1996435"),
        ("skillcorner", "skillcorner_had_1996436"),
        ("skillcorner", "skillcorner_had_2006229"),
    }
    if len(cells) != 15 or {(c["source"], c["match"]) for c in cells} != expected:
        raise StudyError("Prior comparison does not contain the frozen five-match cohort")
    return cells


def _active_package() -> tuple[dict, dict, dict[str, str]]:
    manifest = _json(BASELINE_MANIFEST)
    policy = _json(DECISION_POLICY)
    manifest_sha = digest_file(BASELINE_MANIFEST)
    policy_sha = digest_file(DECISION_POLICY)
    working = policy.get("working_candidate", {})
    architecture = manifest.get("architecture", {})
    if working.get("model") != "joint_gru_combined_ensemble":
        raise StudyError("The adopted working baseline changed before this study")
    if working.get("baseline_manifest_sha256") != manifest_sha:
        raise StudyError("Decision policy no longer binds the active manifest")
    if manifest.get("package_version") != "joint_gru_combined_ensemble_v1":
        raise StudyError("Unexpected active combined-objective package version")
    if architecture.get("kind") != "joint_gru_ensemble" or \
            architecture.get("members") != {"joint_gru_2026": .5, "joint_gru_2027": .5}:
        raise StudyError("Active package is not the fixed equal two-seed GRU ensemble")
    if architecture.get("rate_hz") != RATE_HZ or \
            architecture.get("core_seconds") != CORE_SECONDS or \
            architecture.get("context_each_side_seconds") != CONTEXT_SECONDS or \
            architecture.get("decoder_switch_cost") != SWITCH_COST:
        raise StudyError("Active model tiling or decoder no longer matches this study")
    return manifest, policy, {"baseline_manifest": manifest_sha,
                              "decision_policy": policy_sha}


def _required_artifacts(data_root: Path) -> list[dict[str, Any]]:
    prior = _json(PRIOR_DECLARATION)
    prior_index = {entry["path"]: entry for entry in prior["verified_archive_files"]}
    artifacts: dict[str, dict[str, Any]] = {}

    def pinned(relative: str) -> None:
        if relative not in prior_index:
            raise StudyError(f"Prior verified archive does not pin {relative}")
        actual = _rel_fact(data_root, relative)
        expected = prior_index[relative]
        if actual["sha256"] != expected["sha256"] or actual["bytes"] != expected["bytes"]:
            raise StudyError(f"Restored artifact differs from the prior receipt: {relative}")
        artifacts[relative] = actual

    for cell in _cells():
        stem = f"{cell['match']}_{cell['view']}_{cell['replicate']}"
        pinned(f"{INPUT_ROOT.as_posix()}/{stem}.joblib")
        for seed in SEEDS:
            pinned(f"{MEMBER_ROOT.as_posix()}/{cell['source']}_{stem}_combined_{seed}.npz")

    for source in ("sportec", "metrica", "skillcorner"):
        for seed in SEEDS:
            stem = f"without_{source}_combined_{seed}"
            pinned(f"{MODEL_ROOT.as_posix()}/{stem}.joblib")
            pinned(f"{MODEL_ROOT.as_posix()}/{stem}.json")
            receipt = f"{MODEL_ROOT.as_posix()}/{stem}.runtime.json"
            # The original combined-objective declaration documents one
            # pre-receipt checkpoint; preserve that stated exemption.
            if any(x["path"] == receipt for x in prior["verified_archive_files"]):
                pinned(receipt)

    runs_path = data_root / PRIOR_RUNS_RELATIVE
    runs = _json(runs_path)
    if runs.get("study_id") != "objective_ensemble_v1" or len(runs.get("rows", [])) != 30:
        raise StudyError("Saved combined-objective run is missing or malformed")
    for row in runs["rows"]:
        if row.get("recipe") != "combined_objective_equal_probability_ensemble":
            continue
        relative = f"{PRIOR_OUTPUT_ROOT.as_posix()}/{row['probability_file']}"
        actual = _rel_fact(data_root, relative)
        if actual["sha256"] != row.get("probability_file_sha256"):
            raise StudyError(f"Saved original-tiling ensemble changed: {relative}")
        artifacts[relative] = actual
    if len([p for p in artifacts if p.startswith(f"{INPUT_ROOT.as_posix()}/")
            and p.endswith(".joblib")]) != 15:
        raise StudyError("Expected exactly 15 evaluation source caches")
    if len([p for p in artifacts if p.startswith(f"{MEMBER_ROOT.as_posix()}/")
            and p.endswith(".npz")]) != 30:
        raise StudyError("Expected 30 saved per-seed member predictions")
    if len([p for p in artifacts if p.startswith(f"{PRIOR_OUTPUT_ROOT.as_posix()}/")
            and p.endswith(".npz")]) != 15:
        raise StudyError("Expected 15 original-tiling control probability files")
    return [artifacts[key] for key in sorted(artifacts)]


def _build_declaration(data_root: Path) -> dict[str, Any]:
    manifest, policy, baseline_hashes = _active_package()
    cells = _cells()
    artifacts = _required_artifacts(data_root)
    source_facts = [_fact(ROOT / relative, relative) for relative in SOURCE_FILES]
    prior_runs = data_root / PRIOR_RUNS_RELATIVE
    prior_report = _json(PRIOR_REPORT)
    if prior_report.get("study_id") != "objective_ensemble_v1":
        raise StudyError("Prior adopted-baseline comparison report is missing")
    return {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "status": "frozen before shifted-tiling scoring",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does equal averaging of original and five-second-shifted inference cores improve full-provider spell F1 for the adopted combined-objective GRU ensemble?",
        "hypothesis": "Inference core alignment variance at ten-second seams obscures short possession boundaries; averaging probabilities from a five-second shifted tiling will improve official spell F1 without changing the trained model or decoder.",
        "falsifier": "No increase in equal-provider spell F1 or a failed current clean/feed_like guardrail will falsify the practical value of this tiling change. Seam probability disagreement is descriptive and is not an independent selection criterion.",
        "intervention": {
            "training_fits": 0,
            "candidate_arm_count": 1,
            "control": "The exact saved native H/A/D probabilities from the completed original 10-second-core/5-second-context tiling, with its existing cost-3 decode and score receipt.",
            "candidate": "Reuse each seed's exact saved original-tiling native probability array, then run one additional tiling shifted forward by 5 seconds on the pinned float32 evaluation feature record within each continuous sampled block. Average those native H/A/D probabilities 50/50 where the shifted tiling has retained-core support. At a block's leading 5-second region, retain the original prediction because shifted cores have not begun. Average each seed's two tilings first, average seeds 2026/2027 equally next, then use the existing cost-3 decoder.",
            "changed_mechanism": "Inference tiling phase only; saved original native probabilities provide the exact matched control. Shifted inference uses the frozen float32 feature record and the same combined-objective held-out checkpoints, features, full-provider exclusions, probability classes and decoder.",
            "sampling_rate_hz": RATE_HZ,
            "core_seconds": CORE_SECONDS,
            "context_each_side_seconds": CONTEXT_SECONDS,
            "shift_seconds": OFFSET_SECONDS,
            "shift_sample_frames": OFFSET,
            "partial_edges": "Tiling windows are clipped to their own continuous block. The shifted grid begins 25 samples after each block start, ends with a partial retained core if needed, and leaves its first 25 samples to the original prediction. No window crosses a period or timestamp break.",
            "decoder_switch_cost": SWITCH_COST,
            "seed_weights": {"2026": .5, "2027": .5},
            "tiling_weights_where_available": {"original": .5, "shifted": .5},
        },
        "cohort": cells,
        "cohort_interpretation": "Previously exposed full-provider development: Sportec J03WOY, Metrica Game 3, SkillCorner 1996435/1996436/2006229; one clean view plus two fixed feed_like replicates per match. No new holdout is created.",
        "provider_exclusions": "Every full-provider checkpoint, including normalization, excludes all matches from the evaluated provider; the two seed checkpoints are 2026 and 2027.",
        "prior_exposure": "All five development matches were previously exposed in earlier studies. feed_like is synthetic tracking stress. This is development evidence, not independent confirmation.",
        "inputs": {
            "labels": "Native labels are read only after shifted inference, to score the saved candidate and parity control. No label, target or label-derived feature is passed to inference.",
            "tracking": "Existing full-provider clean/feed_like evaluation cache records, their sealed float32 joint feature matrices, 5 Hz sampled blocks, native visibility and time boundaries.",
            "native_masks_and_breaks": "Unchanged saved cache inputs, official evaluator masks, period boundaries, and timestamp discontinuities.",
        },
        "metrics": {
            "primary": "Official possession spell F1, equal provider mean; rank by unrounded spell F1 and macro F1 only for exact ties.",
            "auxiliary": "0.70 spell F1 + 0.30 macro F1, shown but not ranked.",
            "diagnostics": ["D precision/recall/F1", "spell F1 at IoU .3/.4/.5/.6/.7", "short spell recall under 3 s at IoU .5", "reference and predicted spell counts", "fragmented reference spells", "graded-edge-censored spells", "complete-boundary timing errors", "per-seed metrics", "boundary seam probability disagreement"],
        },
        "aggregation": "Average matches and feed_like replicates within provider, then average three providers equally.",
        "guardrails": {
            "source": "Current experiments/model_decision_policy.json and docs/research/STATUS.md; values are evaluated against the adopted combined-objective ensemble.",
            "clean_spell_f1": "At least +0.01 equal-provider spell F1 for the default clean screen.",
            "feed_like_spell_f1": "No worse than -0.01 equal-provider spell F1 versus the adopted ensemble.",
            "per_provider_clean_d_f1": "No provider decline greater than 0.02.",
            "per_provider_clean_macro_f1": "No provider decline greater than 0.02.",
            "automatic_promotion": False,
        },
        "baseline": {
            "package_version": manifest["package_version"],
            "manifest_sha256": baseline_hashes["baseline_manifest"],
            "decision_policy_sha256": baseline_hashes["decision_policy"],
            "prior_objective_report_sha256": digest_file(PRIOR_REPORT),
            "prior_objective_runs_sha256": digest_file(prior_runs),
            "prior_declaration_sha256": digest_file(PRIOR_DECLARATION),
            "prior_verification_sha256": digest_file(PRIOR_VERIFICATION),
        },
        "original_replay_diagnostic": {
            "receipt": "experiments/results/window_overlap_v2_replay_drift.json",
            "interpretation": "On the first clean Sportec cell, cached and recomputed all-X float32 feature matrices match exactly, while current Torch re-inference differs from saved member probabilities by less than 7e-7. The frozen saved probabilities are the exact control; the replay drift is descriptive only.",
            "used_for_candidate_or_selection": False,
        },
        "source_files": source_facts,
        "input_artifacts": artifacts,
        "organizer_labels_read": False,
        "organizer_scored": False,
        "scoring_gate": "None; no organizer score or public gate is part of this study.",
    }


def freeze(data_root: Path | None = None) -> dict[str, Any]:
    data = _data_root(data_root)
    if DECLARATION.exists():
        declaration, declaration_sha = _verify_declaration(data)
        return {"declaration": str(DECLARATION.relative_to(ROOT)),
                "declaration_sha256": declaration_sha, "already_frozen": True,
                "cells": len(declaration["cohort"])}
    declaration = _build_declaration(data)
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(declaration, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n", encoding="utf-8")
    return {"declaration": str(DECLARATION.relative_to(ROOT)),
            "declaration_sha256": digest_file(DECLARATION), "already_frozen": False}


def _verify_declaration(data_root: Path) -> tuple[dict, str]:
    if not DECLARATION.is_file():
        raise StudyError("Study declaration is not frozen; run freeze first")
    declaration = _json(DECLARATION)
    frozen_hash = digest_file(DECLARATION)
    if declaration.get("study_id") != STUDY_ID:
        raise StudyError("Unexpected study declaration ID")
    if declaration.get("status") != "frozen before shifted-tiling scoring":
        raise StudyError("Study declaration is not in its frozen pre-score state")
    _, _, active_hashes = _active_package()
    if declaration.get("baseline", {}).get("manifest_sha256") != active_hashes["baseline_manifest"] or \
            declaration.get("baseline", {}).get("decision_policy_sha256") != active_hashes["decision_policy"]:
        raise StudyError("Active baseline or policy changed after freeze")
    actual_sources = [_fact(ROOT / relative, relative) for relative in SOURCE_FILES]
    if canonical_json(actual_sources) != canonical_json(declaration.get("source_files")):
        raise StudyError("Frozen inference source/config changed; use a new study ID")
    actual_artifacts = _required_artifacts(data_root)
    if canonical_json(actual_artifacts) != canonical_json(declaration.get("input_artifacts")):
        raise StudyError("Frozen inference or checkpoint artifacts changed")
    if len(declaration.get("cohort", [])) != 15:
        raise StudyError("Frozen cohort is incomplete")
    return declaration, frozen_hash


def tiled_windows(record: dict, offset: int) -> list[dict[str, Any]]:
    """Return retained-core windows, clipped to each sampled continuity block."""
    if offset < 0 or offset >= CORE:
        raise ValueError("offset must be between zero and the core length")
    period = np.asarray(record["period"])
    if period.ndim != 1 or len(period) != len(record["x"]):
        raise ValueError("record period must align with sampled features")
    result = []
    for block_start, block_stop in supervised.blocks(period):
        start = block_start + offset
        while start < block_stop:
            retained_start = max(block_start, start)
            retained_stop = min(block_stop, start + CORE)
            if retained_stop > retained_start:
                full = np.clip(
                    np.arange(start - CONTEXT, start + CORE + CONTEXT),
                    block_start, block_stop - 1)
                result.append({
                    "block_start": block_start,
                    "block_stop": block_stop,
                    "window_start": start,
                    "retained_start": retained_start,
                    "retained_stop": retained_stop,
                    "size": retained_stop - retained_start,
                    "core_offset": retained_start - start,
                    "full": full,
                })
            start += CORE
    return result


def shifted_tiling_sampled(record: dict, model: dict,
                           offset: int = OFFSET) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Predict retained cores on one shifted grid, symmetrizing H/A as usual."""
    import torch

    if record["names"] != model["feature_names"]:
        raise ValueError("Feature schema differs")
    torch.set_num_threads(1)
    network = factorized_had.network(model["input_features"],
                                     model["player_indices"], model["player_only"])
    network.load_state_dict({key: torch.as_tensor(value)
                             for key, value in model["state"].items()})
    network.eval()
    windows = tiled_windows(record, offset)
    sampled = np.asarray(record["x"])
    mask = np.zeros(len(sampled), dtype=bool)
    results = []
    pass_masks = []
    started = time.monotonic()
    for swap in (False, True):
        raw = supervised.swap_team_features(sampled, record["names"]) if swap else sampled
        normalized = lt.normalize(raw, model["median"], model["scale"])
        output = np.full((len(sampled), 3), np.nan, dtype=np.float32)
        pass_mask = np.zeros(len(sampled), dtype=bool)
        with torch.inference_mode():
            for batch_start in range(0, len(windows), 64):
                batch = windows[batch_start:batch_start + 64]
                tensors = torch.as_tensor(np.stack([normalized[w["full"]] for w in batch]))
                probabilities = network(tensors).exp().numpy()
                for index, window in enumerate(batch):
                    left = CONTEXT + window["core_offset"]
                    right = left + window["size"]
                    out_left = window["retained_start"]
                    out_right = window["retained_stop"]
                    if pass_mask[out_left:out_right].any():
                        raise StudyError("Shifted tiling assigned a sampled frame more than once")
                    output[out_left:out_right] = probabilities[index, left:right]
                    pass_mask[out_left:out_right] = True
        results.append(output[:, [1, 0, 2]] if swap else output)
        pass_masks.append(pass_mask)
    if not results:
        raise StudyError("No shifted inference passes were produced")
    if not np.array_equal(pass_masks[0], pass_masks[1]):
        raise StudyError("Team-symmetrized tilings do not cover the same sampled rows")
    mask = pass_masks[0]
    covered = mask
    # Match factorized_had.predict's native float32 team-symmetry arithmetic
    # before the probabilities are upsampled to the native frame grid.
    symmetrized = (results[0][covered] + results[1][covered]) * .5
    output = np.full((len(sampled), 3), np.nan, dtype=np.float64)
    output[covered] = symmetrized
    if not np.isfinite(output[covered]).all() or \
            not np.allclose(output[covered].sum(axis=1), 1, rtol=0,
                            atol=PROBABILITY_SUM_ATOL):
        raise StudyError("Shifted inference produced invalid probabilities")
    coverage = {
        "offset_samples": offset,
        "sampled_frames": int(len(mask)),
        "covered_sampled_frames": int(mask.sum()),
        "fallback_sampled_frames": int((~mask).sum()),
        "coverage_fraction": float(mask.mean()) if len(mask) else 0.0,
        "window_count": len(windows),
        "elapsed_seconds": time.monotonic() - started,
    }
    return output, mask, coverage


def upsample_shifted(record: dict, sampled_probability: np.ndarray,
                     sampled_mask: np.ndarray, native_period: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Upsample shifted probabilities only where that tiling has core support."""
    native_period = np.asarray(native_period)
    indices = np.asarray(record["indices"], dtype=int)
    sample_mask = np.asarray(sampled_mask, dtype=bool)
    p = np.asarray(sampled_probability)
    if sample_mask.shape != (len(indices),) or p.shape != (len(indices), 3):
        raise ValueError("Shifted samples, mask and record indices must align")
    result = np.full((len(native_period), 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(native_period), dtype=bool)
    for start, stop in supervised.blocks(native_period):
        selected = (indices >= start) & (indices < stop) & sample_mask
        x = indices[selected]
        if not len(x):
            continue
        rows = np.arange(max(start, int(x[0])), stop)
        for column in range(3):
            result[rows, column] = np.interp(rows, x, p[selected, column])
        valid[rows] = True
    return result, valid


def average_tilings(original_native: np.ndarray, shifted_native: np.ndarray,
                    shifted_valid: np.ndarray) -> np.ndarray:
    """Give each available tiling equal weight and retain original at uncovered edges."""
    original = np.asarray(original_native, dtype=np.float64)
    shifted = np.asarray(shifted_native, dtype=np.float64)
    valid = np.asarray(shifted_valid, dtype=bool)
    if original.ndim != 2 or original.shape[1] != 3 or shifted.shape != original.shape:
        raise ValueError("Both tilings must have matching shape (n_frames, 3)")
    if valid.shape != (len(original),):
        raise ValueError("Coverage mask must align with native frames")
    if not np.isfinite(original).all() or not np.isfinite(shifted[valid]).all():
        raise ValueError("Available tiling probabilities must be finite")
    if not np.allclose(original.sum(axis=1), 1, rtol=0, atol=PROBABILITY_SUM_ATOL):
        raise ValueError("Original probability rows must sum to one")
    if valid.any() and not np.allclose(shifted[valid].sum(axis=1), 1, rtol=0,
                                      atol=PROBABILITY_SUM_ATOL):
        raise ValueError("Shifted probability rows must sum to one")
    result = original.copy()
    result[valid] = (original[valid] + shifted[valid]) / 2.0
    return result


def _feature_record(inputs: dict, source: str, match: str) -> dict:
    """Prepare inference features with an all-X placeholder, never targets."""
    placeholder = np.full(len(inputs["period"]), "X", dtype="U1")
    return live_gap_experiment.feature_view(inputs, placeholder, source, match)["joint"]


def _inference_record_from_cache(cache: dict) -> dict:
    """Allowlist cached inference arrays and discard all target-bearing fields."""
    source = cache.get("record")
    if not isinstance(source, dict):
        raise StudyError("Evaluation cache lacks its frozen joint feature record")
    required = {"x", "names", "period", "indices", "native_n", "fps"}
    if not required.issubset(source):
        raise StudyError(f"Frozen feature record lacks fields: {sorted(required-set(source))}")
    # In particular, do not copy the cached `y` array or any top-level labels.
    return {key: source[key] for key in sorted(required)}


def _verify_record_inputs(record: dict, inputs: dict) -> None:
    """Check cached row/time alignment without reading any target values."""
    x = np.asarray(record["x"])
    indices = np.asarray(record["indices"])
    native_period = supervised.model_periods(inputs)
    expected_indices = lt.sampled_indices(inputs)
    if x.ndim != 2 or x.shape[1] != len(record["names"]) or x.dtype != np.float32:
        raise StudyError("Frozen joint feature matrix must be float32 with aligned names")
    if not np.array_equal(indices, expected_indices):
        raise StudyError("Frozen feature rows do not match the pinned 5 Hz input sampling")
    if not np.array_equal(np.asarray(record["period"]), native_period[indices]):
        raise StudyError("Frozen sampled period blocks do not match native inputs")
    if int(record["native_n"]) != len(native_period) or \
            float(record["fps"]) != float(np.asarray(inputs["fps"])):
        raise StudyError("Frozen feature record length/fps do not match native inputs")


def _verify_target_independent_features(record: dict, inputs: dict,
                                        source: str, match: str) -> dict[str, Any]:
    """Check the frozen cached matrix against all-X feature construction once."""
    fresh = _feature_record(inputs, source, match)
    if record["names"] != fresh["names"] or \
            not np.array_equal(record["indices"], fresh["indices"]) or \
            not np.array_equal(record["period"], fresh["period"]):
        raise StudyError("All-X feature reconstruction changed the cached schema or blocks")
    cached_x = np.asarray(record["x"])
    fresh_x = np.asarray(fresh["x"]).astype(cached_x.dtype, copy=False)
    if not np.array_equal(cached_x, fresh_x, equal_nan=True):
        finite = np.isfinite(cached_x) & np.isfinite(fresh_x)
        maximum = float(np.max(np.abs(cached_x[finite].astype(np.float64)
                                      - fresh_x[finite].astype(np.float64)))) \
            if finite.any() else None
        raise StudyError(f"All-X features differ from pinned cache: max_abs={maximum}")
    return {"schema_matches": True, "indices_match": True, "period_blocks_match": True,
            "float32_feature_values_match_including_nan": True,
            "feature_columns": int(cached_x.shape[1]),
            "sampled_rows": int(cached_x.shape[0])}


def _preflight_archive_keys() -> set[str]:
    keys = {"original_ensemble", "candidate_ensemble", "shifted_filled_ensemble",
            "native_coverage_mask"}
    for seed in SEEDS:
        keys.update({f"original_{seed}", f"candidate_{seed}",
                     f"shifted_filled_{seed}"})
    return keys


def _load_preflight_inference(data_root: Path, cache: dict,
                              receipt: dict) -> dict[str, Any]:
    """Load the already-sealed first-cell probabilities for the scoring pass."""
    relative = receipt.get("candidate_archive", {}).get("path")
    if not relative:
        raise StudyError("Preflight candidate archive path is missing")
    path = data_root.joinpath(*PurePosixPath(relative).parts)
    if not path.is_file() or digest_file(path) != receipt["candidate_archive"].get("sha256"):
        raise StudyError("Preflight candidate archive changed after sealing")
    with np.load(path, allow_pickle=False) as archive:
        required = _preflight_archive_keys()
        if not required.issubset(archive.files):
            raise StudyError("Preflight candidate archive is incomplete")
        arrays = {name: archive[name] for name in required}
    record = _inference_record_from_cache(cache)
    _verify_record_inputs(record, cache["inputs"])
    member_results = {}
    for seed in SEEDS:
        key = str(seed)
        coverage = receipt["seed_coverage"][key]
        member_results[seed] = {
            "original_probability": arrays[f"original_{seed}"],
            "candidate_probability": arrays[f"candidate_{seed}"],
            "shifted_filled_probability": arrays[f"shifted_filled_{seed}"],
            "coverage": coverage,
            "native_covered_frames": int(arrays["native_coverage_mask"].sum()),
            "native_fallback_frames": int((~arrays["native_coverage_mask"]).sum()),
            "native_coverage_mask": arrays["native_coverage_mask"],
        }
    return {
        "record": record,
        "member_results": member_results,
        "original_ensemble": arrays["original_ensemble"],
        "candidate_ensemble": arrays["candidate_ensemble"],
        "shifted_filled_ensemble": arrays["shifted_filled_ensemble"],
        "native_coverage_mask": arrays["native_coverage_mask"],
    }


def _verify_model(model: dict, source: str, seed: int,
                  data_root: Path | None = None) -> None:
    if model.get("kind") != "factorized_had" or model.get("player_only") is not True:
        raise StudyError("Restored checkpoint is not the player-only factorized H/A/D GRU")
    if int(model.get("seed", -1)) != seed or model.get("objective") != "combined":
        raise StudyError(f"Checkpoint seed/objective mismatch for {source}, seed {seed}")
    data = _data_root(data_root)
    receipt_path = data / MODEL_ROOT / f"without_{source}_combined_{seed}.json"
    receipt = _json(receipt_path)
    if receipt.get("heldout") != source or receipt.get("seed") != seed or \
            receipt.get("objective") != "combined" or receipt.get("arm") != "combined":
        raise StudyError(f"Checkpoint receipt does not confirm full provider holdout: {receipt_path}")


def _load_member_probabilities(data_root: Path, cell: dict[str, str]
                               ) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    members = {}
    receipts = []
    for seed in SEEDS:
        relative = (f"{MEMBER_ROOT.as_posix()}/{cell['source']}_{cell['match']}_"
                    f"{cell['view']}_{cell['replicate']}_combined_{seed}.npz")
        path = data_root.joinpath(*PurePosixPath(relative).parts)
        with np.load(path, allow_pickle=False) as archive:
            if not {"probability", "prediction"}.issubset(archive.files):
                raise StudyError(f"Saved original member lacks probability/prediction: {relative}")
            probability = archive["probability"]
            saved_prediction = archive["prediction"]
        oe._verify_probability(probability, len(saved_prediction), relative)
        members[seed] = probability
        receipts.append({"seed": seed, "path": relative,
                         "path_sha256": digest_file(path),
                         "probability_sha256": digest_array(probability),
                         "saved_prediction_sha256": digest_array(saved_prediction),
                         "prediction": saved_prediction})
    return members, receipts


def _control_archive(data_root: Path, cell: dict[str, str]) -> tuple[np.ndarray, dict, Path]:
    runs_path = data_root / PRIOR_RUNS_RELATIVE
    runs = _json(runs_path)
    key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
    rows = [row for row in runs["rows"]
            if (row.get("source"), row.get("match"), row.get("view"),
                row.get("replicate")) == key
            and row.get("recipe") == "combined_objective_equal_probability_ensemble"]
    if len(rows) != 1:
        raise StudyError(f"Saved original-tiling row missing/duplicated for {key}")
    row = rows[0]
    path = data_root / PRIOR_OUTPUT_ROOT / row["probability_file"]
    with np.load(path, allow_pickle=False) as archive:
        if not {"probability", "prediction"}.issubset(archive.files):
            raise StudyError(f"Original-tiling ensemble archive is incomplete: {path}")
        probability = archive["probability"]
        prediction = archive["prediction"]
    if digest_array(probability) != row["probability_sha256"] or \
            digest_array(prediction) != row["prediction_sha256"]:
        raise StudyError(f"Original-tiling ensemble changed after its report: {path}")
    return probability, row, path


def _score(labels: np.ndarray, probability: np.ndarray, inputs: dict,
           cell: dict[str, str], recipe: str) -> tuple[dict, np.ndarray]:
    _verify_native_probability(probability, len(labels), recipe)
    predicted = supervised.decode(probability, supervised.model_periods(inputs),
                                  float(inputs["fps"]), SWITCH_COST)
    row = oe.score_prediction(labels, predicted, inputs, recipe, cell,
                              digest_array(probability))
    row["recipe"] = recipe
    return row, predicted


def _verify_native_probability(probability: np.ndarray, n_frames: int,
                               description: str) -> None:
    p = np.asarray(probability)
    if p.shape != (n_frames, 3) or not np.issubdtype(p.dtype, np.number):
        raise StudyError(f"{description} must have shape ({n_frames}, 3)")
    if not np.isfinite(p).all() or (p < -1e-8).any():
        raise StudyError(f"{description} contains invalid values")
    if not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=PROBABILITY_SUM_ATOL):
        raise StudyError(f"{description} rows do not sum to one")


def _assert_exact_parity(actual: np.ndarray, saved: np.ndarray, cell: dict) -> None:
    if actual.shape != saved.shape or not np.array_equal(actual, saved):
        maximum = float(np.max(np.abs(actual.astype(np.float64) - saved.astype(np.float64)))) \
            if actual.shape == saved.shape else None
        raise StudyError(f"Original-tiling probability parity failed for {cell}: max_abs={maximum}")


def _cell_inference(data_root: Path, cell: dict[str, str],
                    models: dict[int, dict], cache: dict) -> dict[str, Any]:
    """Infer without passing the cached label vector to feature or model code."""
    inputs = cache["inputs"]
    record = _inference_record_from_cache(cache)
    _verify_record_inputs(record, inputs)
    native_period = supervised.model_periods(inputs)
    saved_members, member_receipts = _load_member_probabilities(data_root, cell)
    receipt_by_seed = {row["seed"]: row for row in member_receipts}
    member_results: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        model = models[seed]
        _verify_model(model, cell["source"], seed, data_root)
        if record["names"] != model["feature_names"]:
            raise StudyError(f"Feature schema drift for {cell['match']} seed {seed}")
        original_native = np.asarray(saved_members[seed])
        saved_prediction = receipt_by_seed[seed]["prediction"]
        expected_prediction = supervised.decode(original_native, native_period,
                                                float(inputs["fps"]), SWITCH_COST)
        if not np.array_equal(expected_prediction, saved_prediction):
            raise StudyError(f"Saved member decoder parity failed for {cell}, seed {seed}")
        shifted_sampled, shifted_mask, coverage = shifted_tiling_sampled(record, model)
        shifted_native, native_mask = upsample_shifted(
            record, shifted_sampled, shifted_mask, native_period)
        # The archived original probability is the exact control side. Rows
        # without shifted support retain that saved original prediction.
        shifted_for_blend = np.where(native_mask[:, None], shifted_native, original_native)
        candidate_native = average_tilings(original_native, shifted_for_blend, native_mask)
        _verify_native_probability(original_native, len(inputs["period"]),
                                   f"original seed {seed}")
        _verify_native_probability(candidate_native, len(inputs["period"]),
                                   f"shifted seed {seed}")
        member_results[seed] = {
            "original_probability": original_native,
            "candidate_probability": candidate_native,
            "shifted_filled_probability": shifted_for_blend,
            "coverage": coverage,
            "native_covered_frames": int(native_mask.sum()),
            "native_fallback_frames": int((~native_mask).sum()),
            "native_coverage_mask": native_mask,
            "saved_member_probability_sha256": receipt_by_seed[seed]["probability_sha256"],
        }
    if not np.array_equal(member_results[2026]["native_coverage_mask"],
                          member_results[2027]["native_coverage_mask"]):
        raise StudyError("Both seed tilings must cover identical native frames")
    native_coverage = member_results[2026]["native_coverage_mask"]
    original_ensemble = oe.blend_probabilities(
        member_results[2026]["original_probability"],
        member_results[2027]["original_probability"])
    candidate_ensemble = oe.blend_probabilities(
        member_results[2026]["candidate_probability"],
        member_results[2027]["candidate_probability"])
    shifted_filled_ensemble = oe.blend_probabilities(
        member_results[2026]["shifted_filled_probability"],
        member_results[2027]["shifted_filled_probability"])
    return {"record": record, "member_results": member_results,
            "original_ensemble": original_ensemble,
            "candidate_ensemble": candidate_ensemble,
            "shifted_filled_ensemble": shifted_filled_ensemble,
            "native_coverage_mask": native_coverage}


def boundary_probability_disagreement(inputs: dict, original: np.ndarray,
                                      shifted_filled: np.ndarray,
                                      coverage: np.ndarray) -> dict[str, Any]:
    """Measure original-versus-shifted TV near original core seams and elsewhere."""
    native_period = supervised.model_periods(inputs)
    indices = lt.sampled_indices(inputs)
    sampled_period = native_period[indices]
    fps = float(inputs["fps"])
    seam_frames = []
    for start, stop in supervised.blocks(sampled_period):
        for sampled_seam in range(start + CORE, stop, CORE):
            seam_frames.append(int(indices[sampled_seam]))
    seam_zone = np.zeros(len(native_period), dtype=bool)
    radius = max(1, round(fps))
    native_rows = np.arange(len(native_period))
    for seam in seam_frames:
        same_block = native_period == native_period[seam]
        seam_zone |= same_block & (np.abs(native_rows - seam) <= radius)
    overlap = np.asarray(coverage, dtype=bool)
    tv = .5 * np.abs(np.asarray(original) - np.asarray(shifted_filled)).sum(axis=1)
    selected = overlap & np.isfinite(tv)
    at_seams = selected & seam_zone
    away = selected & ~seam_zone
    return {
        "original_core_seam_count": len(seam_frames),
        "seam_window_radius_seconds": 1.0,
        "seam_window_frames_with_both_tilings": int(at_seams.sum()),
        "other_frames_with_both_tilings": int(away.sum()),
        "mean_total_variation_near_original_seams":
            float(tv[at_seams].mean()) if at_seams.any() else None,
        "mean_total_variation_elsewhere":
            float(tv[away].mean()) if away.any() else None,
    }


def preflight(data_root: Path | None = None) -> dict[str, Any]:
    """Run first-cell inference and exact control parity without scoring labels."""
    data = _data_root(data_root)
    declaration, declaration_sha = _verify_declaration(data)
    output_dir = data / OUT_RELATIVE
    preflight_path = output_dir / "preflight.json"
    if preflight_path.exists():
        raise StudyError(f"Preflight receipt already exists and is immutable: {preflight_path}")
    cell = declaration["cohort"][0]
    cache_path = data / INPUT_ROOT / f"{cell['match']}_{cell['view']}_{cell['replicate']}.joblib"
    cache = joblib.load(cache_path)
    inputs = cache["inputs"]
    cached_record = _inference_record_from_cache(cache)
    _verify_record_inputs(cached_record, inputs)
    feature_audit = _verify_target_independent_features(
        cached_record, inputs, cell["source"], cell["match"])
    models = {}
    model_facts = []
    for seed in SEEDS:
        relative = f"{MODEL_ROOT.as_posix()}/without_{cell['source']}_combined_{seed}.joblib"
        model_path = data.joinpath(*PurePosixPath(relative).parts)
        model = joblib.load(model_path)
        _verify_model(model, cell["source"], seed, data)
        models[seed] = model
        model_facts.append(_rel_fact(data, relative))
    inference = _cell_inference(data, cell, models, cache)
    saved_original, saved_row, saved_path = _control_archive(data, cell)
    _assert_exact_parity(inference["original_ensemble"], saved_original,
                         (cell["source"], cell["match"], cell["view"], cell["replicate"]))
    candidate_archive = output_dir / "preflight_candidate.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    with candidate_archive.open("xb") as stream:
        np.savez_compressed(
            stream,
            original_ensemble=inference["original_ensemble"],
            candidate_ensemble=inference["candidate_ensemble"],
            shifted_filled_ensemble=inference["shifted_filled_ensemble"],
            native_coverage_mask=inference["native_coverage_mask"],
            **{f"{name}_{seed}": inference["member_results"][seed][field]
               for seed in SEEDS
               for name, field in (
                   ("original", "original_probability"),
                   ("candidate", "candidate_probability"),
                   ("shifted_filled", "shifted_filled_probability"))},
        )
        stream.flush()
        os.fsync(stream.fileno())
    receipt = {
        "study_id": STUDY_ID,
        "status": "pre-score inference and parity only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": declaration_sha,
        "cell": cell,
        "input_cache": _rel_fact(
            data, f"{INPUT_ROOT.as_posix()}/{cell['match']}_{cell['view']}_{cell['replicate']}.joblib"),
        "all_x_feature_audit": feature_audit,
        "checkpoints": model_facts,
        "saved_control_probability_sha256": digest_array(saved_original),
        "reused_control_probability_sha256": digest_array(inference["original_ensemble"]),
        "candidate_probability_sha256": digest_array(inference["candidate_ensemble"]),
        "candidate_probability_valid": True,
        "candidate_covered_native_frames": int(inference["native_coverage_mask"].sum()),
        "candidate_fallback_native_frames": int((~inference["native_coverage_mask"]).sum()),
        "candidate_archive": {"path": str(candidate_archive.relative_to(data)),
                              "bytes": candidate_archive.stat().st_size,
                              "sha256": digest_file(candidate_archive)},
        "seed_coverage": {str(seed): inference["member_results"][seed]["coverage"]
                          for seed in SEEDS},
        "saved_control_path": str(saved_path.relative_to(data)),
        "saved_control_path_sha256": digest_file(saved_path),
        "labels_used_by_inference": False,
        "metrics_computed": False,
        "organizer_labels_read": False,
        "organizer_scored": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    return {"preflight": str(preflight_path),
            "candidate_probability_sha256": receipt["candidate_probability_sha256"],
            "metrics_computed": False}


def _validate_saved_score(row: dict, saved: dict, key: tuple[str, ...]) -> None:
    for metric in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                   "d_precision", "d_recall"):
        if not math.isclose(float(row[metric]), float(saved[metric]), rel_tol=0,
                            abs_tol=1e-12):
            raise StudyError(f"Original-tiling score parity failed for {key}: {metric}")
    if row["n_gt_spells"] != saved["n_gt_spells"] or \
            row["n_pred_spells"] != saved["n_pred_spells"]:
        raise StudyError(f"Original-tiling spell-count parity failed for {key}")
    if row["probability_sha256"] != saved["probability_sha256"] or \
            row["prediction_sha256"] != saved["prediction_sha256"]:
        raise StudyError(f"Original-tiling digest parity failed for {key}")


def run(data_root: Path | None = None) -> dict[str, Any]:
    data = _data_root(data_root)
    declaration, declaration_sha = _verify_declaration(data)
    output_dir = data / OUT_RELATIVE
    runs_path = output_dir / "runs.json"
    if runs_path.exists():
        raise StudyError(f"Saved results are immutable and already exist: {runs_path}")
    preflight_path = output_dir / "preflight.json"
    if not preflight_path.is_file():
        raise StudyError("Run requires the first-cell pre-score inference/parity receipt")
    preflight_receipt = _json(preflight_path)
    if preflight_receipt.get("declaration_sha256") != declaration_sha or \
            preflight_receipt.get("metrics_computed") is not False:
        raise StudyError("Preflight receipt does not match this frozen declaration")
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = declaration["cohort"]
    model_cache: dict[str, dict[int, dict]] = {}
    for source in ("sportec", "metrica", "skillcorner"):
        model_cache[source] = {}
        for seed in SEEDS:
            model_path = data / MODEL_ROOT / f"without_{source}_combined_{seed}.joblib"
            model = joblib.load(model_path)
            _verify_model(model, source, seed, data)
            model_cache[source][seed] = model

    rows: list[dict[str, Any]] = []
    run_receipts: list[dict[str, Any]] = []
    for cell in cells:
        cache_path = data / INPUT_ROOT / f"{cell['match']}_{cell['view']}_{cell['replicate']}.joblib"
        cache = joblib.load(cache_path)
        if not {"inputs", "labels"}.issubset(cache):
            raise StudyError(f"Evaluation cache lacks inputs or scoring labels: {cache_path}")
        inputs = cache["inputs"]
        # No label value is passed into this function; it receives only the
        # cached tracking arrays and the provider-held-out models.
        is_preflight_cell = tuple(cell.items()) == tuple(declaration["cohort"][0].items())
        inference = (_load_preflight_inference(data, cache, preflight_receipt)
                     if is_preflight_cell else
                     _cell_inference(data, cell, model_cache[cell["source"]], cache))
        if is_preflight_cell:
            if digest_array(inference["candidate_ensemble"]) != \
                    preflight_receipt.get("candidate_probability_sha256"):
                raise StudyError("First-cell candidate differs from sealed pre-score inference")
        labels = np.asarray(cache["labels"]).astype(str)
        if labels.ndim != 1 or labels.shape != np.asarray(inputs["period"]).shape:
            raise StudyError(f"Invalid label/input alignment for {cell}")
        if np.asarray(inputs["timestamps"]).shape != labels.shape:
            raise StudyError(f"Invalid timestamp/input alignment for {cell}")
        original, original_saved, control_path = _control_archive(data, cell)
        _assert_exact_parity(inference["original_ensemble"], original, tuple(cell.values()))
        for seed in SEEDS:
            member = inference["member_results"][seed]
            member_archive = (f"{MEMBER_ROOT.as_posix()}/{cell['source']}_{cell['match']}_"
                              f"{cell['view']}_{cell['replicate']}_combined_{seed}.npz")
            with np.load(data.joinpath(*PurePosixPath(member_archive).parts),
                         allow_pickle=False) as archived:
                saved_member = archived["probability"]
                saved_member_pred = archived["prediction"]
            _assert_exact_parity(member["original_probability"], saved_member,
                                 tuple(cell.values()) + (seed,))
            expected_member_prediction = supervised.decode(
                saved_member, supervised.model_periods(inputs), float(inputs["fps"]),
                SWITCH_COST)
            if not np.array_equal(expected_member_prediction, saved_member_pred):
                raise StudyError(f"Saved original member decoder parity failed for {cell}, {seed}")

        control_row, control_prediction = _score(
            labels, inference["original_ensemble"], inputs, cell,
            "original_tiling_control")
        key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
        _validate_saved_score(control_row, original_saved, key)
        candidate_row, candidate_prediction = _score(
            labels, inference["candidate_ensemble"], inputs, cell,
            "overlapping_core_candidate")
        candidate_row["boundary_probability_disagreement"] = \
            boundary_probability_disagreement(
                inputs, inference["original_ensemble"],
                inference["shifted_filled_ensemble"],
                inference["native_coverage_mask"])
        seed_rows = {}
        for seed in SEEDS:
            member = inference["member_results"][seed]
            original_seed_row, _ = _score(
                labels, member["original_probability"], inputs, cell,
                f"original_seed_{seed}")
            candidate_seed_row, _ = _score(
                labels, member["candidate_probability"], inputs, cell,
                f"overlap_seed_{seed}")
            seed_rows[str(seed)] = {
                "original_tiling": original_seed_row,
                "shifted_tiling": candidate_seed_row,
                "coverage": member["coverage"],
                "native_covered_frames": member["native_covered_frames"],
                "native_fallback_frames": member["native_fallback_frames"],
            }

        candidate_file = output_dir / f"candidate_{cell['match']}_{cell['view']}_{cell['replicate']}.npz"
        with candidate_file.open("xb") as stream:
            np.savez_compressed(stream,
                                probability=inference["candidate_ensemble"],
                                prediction=candidate_prediction.astype("U1"),
                                original_probability=inference["original_ensemble"],
                                original_prediction=control_prediction.astype("U1"),
                                shifted_filled_probability=inference["shifted_filled_ensemble"],
                                native_coverage=inference["native_coverage_mask"])
            stream.flush()
            os.fsync(stream.fileno())
        candidate_row["probability_file"] = candidate_file.name
        candidate_row["probability_file_sha256"] = digest_file(candidate_file)
        candidate_row["members"] = seed_rows
        control_row["probability_file"] = str(control_path.relative_to(data))
        control_row["probability_file_sha256"] = digest_file(control_path)
        run_receipts.extend((control_row, candidate_row))
        rows.append({"cell": cell, "control": control_row,
                     "candidate": candidate_row, "seeds": seed_rows})
        print("SCORED", cell["source"], cell["match"], cell["view"],
              cell["replicate"], round(candidate_row["spell_f1"], 6), flush=True)

    if len(rows) != 15 or len(run_receipts) != 30:
        raise StudyError("Declared full-provider cohort was not completed")
    payload = {
        "study_id": STUDY_ID,
        "declaration_sha256": declaration_sha,
        "scored_utc": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "baseline_parity": {
            "matched_cells": 15,
            "saved_member_probability_and_decode_parity": True,
            "saved_original_ensemble_probability_and_score_parity": True,
            "all_original_tiling_cells_reproduced": True,
        },
        "organizer_labels_read": False,
        "organizer_scored": False,
        "public_gate_touched": False,
    }
    raw = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False,
                     allow_nan=False).encode("utf-8") + b"\n"
    fd = os.open(runs_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"runs": str(runs_path), "rows": len(rows),
            "baseline_parity": True, "runs_sha256": digest_bytes(raw)}


def _mean(rows: list[dict], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return float(np.mean(values)) if values else None


def _summarize(rows: list[dict], recipe_key: str) -> dict[str, Any]:
    recipes = [row[recipe_key] for row in rows]
    providers = {}
    for source in ("sportec", "metrica", "skillcorner"):
        source_rows = [row for row in recipes if row["source"] == source]
        if not source_rows:
            raise StudyError(f"Missing provider from aggregate: {source}")
        providers[source] = {}
        for view in ("clean", "feed_like"):
            view_rows = [row for row in source_rows if row["view"] == view]
            providers[source][view] = {
                field: _mean(view_rows, field)
                for field in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                              "d_precision", "d_recall", "short_spell_recall")
            }
    equal_provider = {}
    for view in ("clean", "feed_like"):
        equal_provider[view] = {
            field: float(np.mean([providers[source][view][field]
                                  for source in providers]))
            for field in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                          "d_precision", "d_recall", "short_spell_recall")
        }
    return {"providers": providers, "equal_provider_mean": equal_provider}


def _aggregate_payload(runs: dict, declaration: dict) -> dict[str, Any]:
    rows = runs["rows"]
    control = _summarize(rows, "control")
    candidate = _summarize(rows, "candidate")
    deltas = {}
    for view in ("clean", "feed_like"):
        deltas[view] = {
            field: candidate["equal_provider_mean"][view][field]
            - control["equal_provider_mean"][view][field]
            for field in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                          "d_precision", "d_recall", "short_spell_recall")
        }
    guardrails = declaration["guardrails"]
    provider_guards = {}
    for source in ("sportec", "metrica", "skillcorner"):
        provider_guards[source] = {
            "clean_d_f1_delta": candidate["providers"][source]["clean"]["d_f1"]
            - control["providers"][source]["clean"]["d_f1"],
            "clean_macro_f1_delta": candidate["providers"][source]["clean"]["macro_f1"]
            - control["providers"][source]["clean"]["macro_f1"],
        }
    gate = {
        "clean_spell_f1_delta_at_least_0_01": deltas["clean"]["spell_f1"] >= .01,
        "feed_like_spell_f1_delta_at_least_minus_0_01": deltas["feed_like"]["spell_f1"] >= -.01,
        "each_provider_clean_d_f1_delta_at_least_minus_0_02": all(
            row["clean_d_f1_delta"] >= -.02 for row in provider_guards.values()),
        "each_provider_clean_macro_f1_delta_at_least_minus_0_02": all(
            row["clean_macro_f1_delta"] >= -.02 for row in provider_guards.values()),
    }
    disagreements = [{"source": row["cell"]["source"],
                      "match": row["cell"]["match"],
                      "view": row["cell"]["view"],
                      "replicate": row["cell"]["replicate"],
                      **row["candidate"]["boundary_probability_disagreement"]}
                     for row in rows]
    return {
        "study_id": STUDY_ID,
        "declaration_sha256": runs["declaration_sha256"],
        "runs_sha256": digest_bytes(canonical_json(runs)),
        "reported_utc": datetime.now(timezone.utc).isoformat(),
        "question": declaration["question"],
        "evidence_status": "Previously exposed full-provider development; feed_like is synthetic stress.",
        "decision": "Retain current baseline pending review; this study never promotes automatically.",
        "primary_metric": "Official possession spell F1, equal provider mean; macro F1 breaks exact ties.",
        "protocol": {
            "cells_completed": len(rows),
            "declared_cells": 15,
            "full_provider_exclusions": True,
            "seeds": list(SEEDS),
            "training_fits": 0,
            "organizer_labels_read": False,
            "organizer_scored": False,
            "public_gate_touched": False,
        },
        "control": control,
        "candidate": candidate,
        "candidate_minus_control": deltas,
        "provider_clean_guardrails": provider_guards,
        "guardrails": {"criteria": guardrails, "checks": gate,
                       "passes": all(gate.values()), "automatic_promotion": False},
        "boundary_probability_disagreement": {
            "interpretation": "Descriptive total-variation difference between original and shifted filled probabilities near original ten-second seams and elsewhere; not a separate selection criterion.",
            "cells": disagreements,
        },
        "rows": rows,
        "limitations": [
            "The five evaluation matches were previously exposed; the comparison is historical development evidence.",
            "feed_like is synthetic, not independent broadcast confirmation.",
            "Leading block edges without shifted-core support retain the original tiling prediction.",
            "No organizer public diagnostic or full-match evaluation was run.",
        ],
    }


def report(data_root: Path | None = None) -> dict[str, Any]:
    data = _data_root(data_root)
    declaration, declaration_sha = _verify_declaration(data)
    runs_path = data / OUT_RELATIVE / "runs.json"
    if not runs_path.is_file():
        raise StudyError("Study has no complete saved run; run the frozen study first")
    runs = _json(runs_path)
    if runs.get("study_id") != STUDY_ID or runs.get("declaration_sha256") != declaration_sha:
        raise StudyError("Saved run does not match the frozen declaration")
    if len(runs.get("rows", [])) != 15 or not runs.get("baseline_parity", {}).get(
            "all_original_tiling_cells_reproduced"):
        raise StudyError("Saved run is incomplete or original-tiling parity failed")
    payload = _aggregate_payload(runs, declaration)
    if AGGREGATE.exists():
        raise StudyError(f"Aggregate report already exists and is immutable: {AGGREGATE}")
    AGGREGATE.write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    return {"aggregate": str(AGGREGATE.relative_to(ROOT)),
            "aggregate_sha256": digest_file(AGGREGATE),
            "candidate_spell": payload["candidate"]["equal_provider_mean"]}


def write_artifact_receipt(data_root: Path | None = None) -> dict[str, Any]:
    data = _data_root(data_root)
    declaration, declaration_sha = _verify_declaration(data)
    runs_path = data / OUT_RELATIVE / "runs.json"
    if not runs_path.is_file() or not AGGREGATE.is_file():
        raise StudyError("Run and aggregate report are required before writing artifact receipt")
    receipt = {
        "study_id": STUDY_ID,
        "declaration_sha256": declaration_sha,
        "aggregate_report_sha256": digest_file(AGGREGATE),
        "private_data_root": str(data),
        "private_run": {"path": str(runs_path.relative_to(data)),
                        "bytes": runs_path.stat().st_size,
                        "sha256": digest_file(runs_path)},
        "preflight": {
            "path": str((data / OUT_RELATIVE / "preflight.json").relative_to(data)),
            "sha256": digest_file(data / OUT_RELATIVE / "preflight.json"),
        },
        "preflight_candidate": {
            "path": str((data / OUT_RELATIVE / "preflight_candidate.npz").relative_to(data)),
            "sha256": digest_file(data / OUT_RELATIVE / "preflight_candidate.npz"),
        },
        "source_and_input_artifacts_verified": True,
        "full_provider_checkpoints": 6,
        "evaluation_caches": 15,
        "saved_original_member_predictions": 30,
        "public_diagnostic_or_gate": False,
        "organizer_labels_read": False,
        "organizer_scored": False,
        "independent_remote_restore": None,
    }
    if ARTIFACT_RECEIPT.exists():
        raise StudyError(f"Artifact receipt already exists and is immutable: {ARTIFACT_RECEIPT}")
    ARTIFACT_RECEIPT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    return {"artifact_receipt": str(ARTIFACT_RECEIPT.relative_to(ROOT)),
            "sha256": digest_file(ARTIFACT_RECEIPT)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "check", "preflight", "run", "report", "receipt"))
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args(argv)
    if args.action == "freeze":
        result = freeze(args.data_dir)
    elif args.action == "check":
        declaration, declaration_sha = _verify_declaration(_data_root(args.data_dir))
        result = {"study_id": declaration["study_id"],
                  "declaration_sha256": declaration_sha,
                  "artifacts": len(declaration["input_artifacts"]),
                  "cells": len(declaration["cohort"])}
    elif args.action == "preflight":
        result = preflight(args.data_dir)
    elif args.action == "run":
        result = run(args.data_dir)
    elif args.action == "report":
        result = report(args.data_dir)
    else:
        result = write_artifact_receipt(args.data_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
