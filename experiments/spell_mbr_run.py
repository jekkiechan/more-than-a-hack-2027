"""Frozen, paired evaluation runner for the spell-F1 posterior MBR decoder.

This runner has two deliberate stages: ``freeze`` verifies and binds the
restored control evidence and code; ``run`` claims a fresh ignored output
directory, replays the unchanged cost-3 comparator, then evaluates the one
declared decoder on the same 45 cells. Target labels are passed only to the
metric function after candidate inference has returned.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from evaluation import possession_metrics as pm
from experiments import dense_boundary_v1, metrics, spell_mbr, supervised


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"
DEFAULT_OUTPUT = ROOT / "data/experiments/spell_mbr_v1"
DECLARATION = RESULTS / "spell_mbr_v1_declaration.json"
PROVENANCE = RESULTS / "dense_label_weight_v1_prediction_provenance_control_D2.json"
AUDIT = RESULTS / "dense_label_weight_v1_control_prediction_audit.json"
CONTROL_REPLAY = RESULTS / "dense_label_weight_v1_control_replay.json"
MEMBER_SCORES = RESULTS / "dense_label_weight_v1_member_scores_control_D2.json"
ENSEMBLE_SCORES = RESULTS / "dense_label_weight_v1_ensemble_scores_control_D2.json"
SOURCE_FILES = (
    "experiments/spell_mbr.py",
    "experiments/spell_mbr_run.py",
    "experiments/dense_boundary_v1.py",
    "experiments/metrics.py",
    "experiments/supervised.py",
    "evaluation/possession_metrics.py",
)
PROVIDERS = ("Metrica", "Sportec", "SkillCorner")
MEMBERS = ("2026", "2027", "ensemble")
EXPECTED_CONTROL = {"clean": 0.6472145590743718,
                    "feed_like": 0.5058107894247519}
CONTROL_TOLERANCE = 0.000051
RECONSTRUCTION_STATUS = (
    "reconstructed from exact raw TRAIN source, source modules, frozen fold "
    "teachers, and declared D2 recipe; per-game joblib binaries differ from "
    "historical hashes; original trained D2 fold weights were unavailable"
)
FOLD_MAP = {"metrica": "leave_out_metrica",
            "sportec": "leave_out_sportec",
            "skillcorner": "leave_out_skillcorner"}
EXPECTED_MATCHES = {
    "Metrica": ("metrica_3",),
    "Sportec": ("J03WOY",),
    "SkillCorner": ("1996435", "1996436", "2006229"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_artifact(rel: str) -> Path:
    """Resolve provenance paths in the repository cache or sibling mtah-data."""
    relative = Path(rel)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe provenance path: {rel}")
    candidates = [ROOT / relative]
    if relative.parts and relative.parts[0] == "data":
        candidates.append(ROOT.parent / "mtah-data" / Path(*relative.parts[1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing provenance artifact {rel}; searched {candidates}")


def _check_sha_mapping(mapping: dict, field: str) -> None:
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"Provenance {field} must be a nonempty path-to-SHA256 mapping")
    for rel, digest in mapping.items():
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(ch not in "0123456789abcdef" for ch in digest.lower())):
            raise ValueError(f"Invalid SHA256 in {field}: {rel}")
        if sha256_file(_resolve_artifact(rel)) != digest:
            raise ValueError(f"Provenance hash verification failed for {field}: {rel}")


def _load_sources() -> dict:
    """Verify all sealed upstream evidence and return its exact source hashes."""
    for path in (PROVENANCE, AUDIT, CONTROL_REPLAY, MEMBER_SCORES, ENSEMBLE_SCORES):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen source receipt is missing: {path}")
    provenance, audit = _json(PROVENANCE), _json(AUDIT)
    replay = _json(CONTROL_REPLAY)
    members, ensemble = _json(MEMBER_SCORES), _json(ENSEMBLE_SCORES)
    provenance_sha = sha256_file(PROVENANCE)
    if (provenance.get("source_study_id") != "dense_label_weight_v1" or
            provenance.get("source_arm") != "control_D2" or
            provenance.get("inference_seed") != "seedensemble" or
            provenance.get("reconstruction_status") != RECONSTRUCTION_STATUS or
            provenance.get("fold_mapping") != FOLD_MAP):
        raise ValueError("Source provenance does not identify reconstructed control_D2")
    if (audit.get("passed") is not True or audit.get("source_arm") != "control_D2" or
            audit.get("source_study_id") != "dense_label_weight_v1" or
            audit.get("reconstruction_status") != RECONSTRUCTION_STATUS or
            audit.get("member_count") != 30 or audit.get("ensemble_count") != 15 or
            len(audit.get("files", [])) != 45):
        raise ValueError("The sealed source prediction audit is incomplete or failed")
    if (replay.get("passed") is not True or replay.get("scored_cells") != 15 or
            replay.get("public_labels_accessed") is not False or
            replay.get("prediction_provenance_sha256") != provenance_sha or
            replay.get("prediction_provenance_receipt_sha256") != sha256_file(AUDIT)):
        raise ValueError("The upstream control replay does not bind sealed provenance")
    if (members.get("arm") != "control_D2" or len(members.get("rows", [])) != 30 or
            ensemble.get("arm") != "control_D2" or len(ensemble.get("rows", [])) != 15):
        raise ValueError("Source score receipts do not cover 30 members and 15 ensembles")
    if (sha256_file(MEMBER_SCORES) != audit.get("member_scores_sha256") or
            sha256_file(ENSEMBLE_SCORES) != audit.get("ensemble_scores_sha256")):
        raise ValueError("Source score receipts differ from the sealed audit")
    for key in ("model_checkpoint_hashes", "evaluation_input_hashes"):
        _check_sha_mapping(provenance.get(key), key)
    members_files = provenance.get("member_prediction_files")
    ensemble_files = provenance.get("ensemble_prediction_files")
    if len(members_files or {}) != 30 or len(ensemble_files or {}) != 15:
        raise ValueError("Source provenance must bind all 45 prediction files")
    _check_sha_mapping(members_files, "member_prediction_files")
    _check_sha_mapping(ensemble_files, "ensemble_prediction_files")
    audit_files = {row["path"]: row["sha256"] for row in audit["files"]}
    expected_files = dict(members_files) | dict(ensemble_files)
    if audit_files != expected_files:
        raise ValueError("Prediction audit and provenance file lists differ")
    if (members.get("prediction_files") != members_files or
            ensemble.get("prediction_files") != ensemble_files):
        raise ValueError("Prediction hashes differ from score receipts")
    expected_model_folds = {"metrica", "sportec", "skillcorner"}
    for provider in expected_model_folds:
        mapping = provenance["model_checkpoint_hashes"]
        if sum(f"__{provider}__" in path for path in mapping) != 2:
            raise ValueError(f"Expected two control checkpoints for {provider}")
    return {
        "provenance_sha256": provenance_sha,
        "audit_sha256": sha256_file(AUDIT),
        "control_replay_sha256": sha256_file(CONTROL_REPLAY),
        "member_scores_sha256": sha256_file(MEMBER_SCORES),
        "ensemble_scores_sha256": sha256_file(ENSEMBLE_SCORES),
        "provenance": provenance,
        "audit": audit,
        "member_scores": members,
        "ensemble_scores": ensemble,
    }


def _expected_cell_paths(source_hashes: dict) -> dict[tuple[str, str, str, str], str]:
    """Map each expected provider/match/view/replicate/member to its source NPZ."""
    result = {}
    maps = source_hashes["provenance"]
    for mapping, member in ((maps["member_prediction_files"], "members"),
                            (maps["ensemble_prediction_files"], "ensemble")):
        for rel in mapping:
            name = Path(rel).name
            fields = name.removesuffix(".npz").split("__")
            if len(fields) != 4:
                raise ValueError(f"Unexpected prediction basename: {name}")
            seed, match_raw, view, replicate = fields
            record = {"source": "", "match": match_raw}
            # Canonical metadata validation also checks the provider/match pair.
            provider, match, canonical_view, realization = dense_boundary_v1._canonical_record_metadata({
                "provider": _provider_for_match(match_raw), "match": match_raw,
                "view": view, "replicate": replicate,
                "realization": seed.removeprefix("seed"),
            })
            seed_tag = "ensemble" if seed in {"seedensemble", "seed2026", "seed2027"} and (
                seed == "seedensemble") else seed.removeprefix("seed")
            if seed_tag not in MEMBERS:
                raise ValueError(f"Unexpected source realization {seed}")
            if ((member == "ensemble") != (seed_tag == "ensemble")):
                raise ValueError("Prediction source map has a seed/member mismatch")
            key = (provider, match, canonical_view, seed_tag)
            if key in result:
                raise ValueError(f"Duplicate paired source cell: {key}")
            result[key] = rel
    expected = {
        (provider, match, view, member)
        for provider, matches in EXPECTED_MATCHES.items()
        for match in matches
        for view in ("clean", "feed_like_corrupt_1", "feed_like_corrupt_2")
        for member in MEMBERS
    }
    if set(result) != expected:
        raise ValueError(f"Source cohort mismatch: missing={sorted(expected-set(result))}, "
                         f"extra={sorted(set(result)-expected)}")
    return result


def _provider_for_match(match: str) -> str:
    match = match.removeprefix("skillcorner_had_").removeprefix("skillcorner_")
    for provider, matches in EXPECTED_MATCHES.items():
        if match in matches:
            return provider
    raise ValueError(f"Unknown expected match: {match}")


def _declaration_payload(source_hashes: dict) -> dict:
    source_files = {rel: sha256_file(ROOT / rel) for rel in SOURCE_FILES}
    return {
        "schema": 1,
        "study_id": "spell_mbr_v1",
        "status": "frozen_before_candidate_scoring",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "experiments/SPELL_MBR_V1.md",
        "question": "Does a fixed posterior-sample minimum-risk decision rule improve official possession spell F1 over the unchanged cost-3 decoder?",
        "configuration": {
            "source_study": "dense_label_weight_v1",
            "source_arm": "control_D2",
            "cohort": "5 matches x 3 views x seeds 2026/2027/equal probability ensemble = 45 paired cells",
            "decoder": "cost-3 Potts posterior; MAP plus 16 FFBS candidate paths; 32 independently sampled FFBS references",
            "switch_cost": 3.0,
            "candidate_samples": spell_mbr.DEFAULT_CANDIDATE_SAMPLES,
            "reference_samples": spell_mbr.DEFAULT_REFERENCE_SAMPLES,
            "seed": spell_mbr.DEFAULT_SEED,
            "validity": "No target-derived mask; inference receives only probabilities, recomputed model_period, and fps. Unknown labels remain for official grading to exclude.",
            "primary": "equal-provider official spell_f1_mean; ensemble realization",
            "tie_break": "macro_f1 for exact unrounded primary ties",
            "auxiliary": "0.70 * spell_f1 + 0.30 * macro_f1; descriptive only",
            "guards": {
                "clean_spell_gain_at_least": 0.01,
                "feed_like_spell_delta_at_least": -0.01,
                "each_provider_clean_d_f1_delta_at_least": -0.02,
                "each_provider_clean_macro_f1_delta_at_least": -0.02,
                "promotion_authorized": False,
            },
            "control_replay": {
                "clean_spell_f1": EXPECTED_CONTROL["clean"],
                "feed_like_spell_f1": EXPECTED_CONTROL["feed_like"],
                "absolute_tolerance": CONTROL_TOLERANCE,
            },
        },
        "source_sha256": source_files,
        "upstream": {key: source_hashes[key] for key in (
            "provenance_sha256", "audit_sha256", "control_replay_sha256",
            "member_scores_sha256", "ensemble_scores_sha256")},
        "prediction_files": source_hashes["provenance"]["member_prediction_files"] | source_hashes["provenance"]["ensemble_prediction_files"],
        "evaluation_input_hashes": source_hashes["provenance"]["evaluation_input_hashes"],
        "model_checkpoint_hashes": source_hashes["provenance"]["model_checkpoint_hashes"],
    }


def freeze(declaration_path: Path = DECLARATION) -> dict:
    """Verify all 45 upstream predictions and write the code/input freeze once."""
    sources = _load_sources()
    _expected_cell_paths(sources)
    payload = _declaration_payload(sources)
    declaration_path = Path(declaration_path)
    declaration_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if declaration_path.exists():
        existing = _json(declaration_path)
        # A repeat freeze is a verification only; the original timestamp is stable.
        payload["frozen_utc"] = existing.get("frozen_utc")
        if existing != payload:
            raise ValueError(f"Refusing to replace a different frozen declaration: {declaration_path}")
        return existing
    fd = os.open(declaration_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def verify_declaration(declaration_path: Path = DECLARATION) -> dict:
    declaration_path = Path(declaration_path)
    if not declaration_path.is_file():
        raise RuntimeError("Freeze the spell MBR declaration before running")
    declaration = _json(declaration_path)
    if declaration.get("status") != "frozen_before_candidate_scoring":
        raise ValueError("Spell MBR declaration is not frozen")
    current_sources = _load_sources()
    expected = _declaration_payload(current_sources)
    # frozen_utc records the original declaration time and is intentionally fixed.
    expected["frozen_utc"] = declaration.get("frozen_utc")
    if declaration != expected:
        raise ValueError("Frozen spell MBR declaration differs from current source, config, or input hashes")
    for rel, expected_sha in declaration["source_sha256"].items():
        if sha256_file(ROOT / rel) != expected_sha:
            raise ValueError(f"Frozen source changed after declaration: {rel}")
    return declaration


def _scalar(npz, name) -> str:
    keys = npz.keys() if isinstance(npz, dict) else npz.files
    if name not in keys:
        raise ValueError(f"Source prediction lacks metadata field {name}")
    return str(np.asarray(npz[name]).reshape(()).item())


def _load_cells(declaration: dict) -> list[dict]:
    provenance = _json(PROVENANCE)
    audit_rows = {row["path"]: row for row in _json(AUDIT)["files"]}
    member_rows = {(row["source"], row["match"], row["view"], row["replicate"],
                    str(row["member"])): row for row in _json(MEMBER_SCORES)["rows"]}
    ensemble_rows = {(row["source"], row["match"], row["view"], row["replicate"],
                      "ensemble"): row for row in _json(ENSEMBLE_SCORES)["rows"]}
    paths = _expected_cell_paths({"provenance": provenance})
    cells = []
    for (provider, match, view, member), rel in sorted(paths.items()):
        expected_sha = declaration["prediction_files"].get(rel)
        path = _resolve_artifact(rel)
        if not expected_sha or sha256_file(path) != expected_sha:
            raise ValueError(f"Frozen source prediction hash mismatch: {rel}")
        audit = audit_rows.get(rel)
        if not audit or audit.get("sha256") != expected_sha:
            raise ValueError(f"Prediction audit row differs for {rel}")
        fold = FOLD_MAP[provider.lower()]
        expected_replicate = "clean" if view == "clean" else view.removeprefix("feed_like_")
        # The canonical view names feed_like_corrupt_1/2 match the producer's
        # feed_like + corrupt_1/2 tags stored in each private NPZ.
        with np.load(path, allow_pickle=False) as src:
            record = {key: src[key].copy() for key in src.files}
        source_view = "clean" if view == "clean" else "feed_like"
        source_name = _scalar(record, "source")
        raw_match = _scalar(record, "match")
        raw_view = _scalar(record, "view")
        raw_replicate = _scalar(record, "replicate")
        tags = dense_boundary_v1._canonical_record_metadata({
            "source": source_name, "match": raw_match, "view": raw_view,
            "replicate": raw_replicate,
            "realization": "ensemble" if member == "ensemble" else member,
        })
        if tags != (provider, match, view, member):
            raise ValueError(f"Source prediction canonical identity mismatch: {rel}: {tags}")
        if (_scalar(record, "source_study_id") != "dense_label_weight_v1" or
                _scalar(record, "source_arm") != "control_D2" or
                _scalar(record, "arm") != "control_D2" or
                _scalar(record, "reconstruction_status") != RECONSTRUCTION_STATUS or
                source_name != provider.lower() or
                raw_view != source_view or
                raw_replicate != expected_replicate or
                _scalar(record, "fold") != fold):
            raise ValueError(f"Source prediction metadata mismatch: {rel}")
        expected_seed = "seedensemble" if member == "ensemble" else f"seed{member}"
        if _scalar(record, "seed") != ("ensemble" if member == "ensemble" else member):
            # NPZ stores seedensemble as "ensemble" for the pooled array.
            raise ValueError(f"Source prediction seed tag mismatch: {rel}")
        inference_seed = _scalar(record, "inference_seed")
        if inference_seed != expected_seed:
            raise ValueError(f"Source inference seed mismatch: {rel}")
        model_period = supervised.model_periods({
            "period": record["period"], "timestamps": record["timestamps"],
            "fps": float(np.asarray(record["fps"]).item()),
        })
        if not np.array_equal(model_period, record["model_period"]):
            raise ValueError(f"Recomputed model_period differs from saved source metadata: {rel}")
        p = np.asarray(record["probabilities"])
        labels = np.asarray(record["labels"]).astype("U1")
        period = np.asarray(record["period"])
        timestamps = np.asarray(record["timestamps"], dtype=np.float64)
        fps = float(np.asarray(record["fps"]).item())
        n = len(labels)
        if (p.shape != (n, 3) or period.shape != (n,) or model_period.shape != (n,) or
                timestamps.shape != (n,) or not np.isfinite(p).all() or (p < 0).any() or
                not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-5) or
                not np.isin(labels, ("H", "A", "D", "X")).all()):
            raise ValueError(f"Malformed source prediction record: {rel}")
        if len(record["prediction"]) != n or not np.isin(record["prediction"], ("H", "A", "D", "X")).all():
            raise ValueError(f"Malformed saved comparator prediction: {rel}")
        if member == "ensemble":
            eval_hashes = json.loads(_scalar(record, "evaluation_input_hashes_json"))
            expected_eval = audit["evaluation_input_hashes"]
            if eval_hashes != expected_eval:
                raise ValueError(f"Ensemble evaluation cache source/hash differs from audit: {rel}")
            model_hashes = json.loads(_scalar(record, "model_checkpoint_hashes_json"))
            expected_models = audit["model_checkpoint_hashes"]
            if model_hashes != expected_models:
                raise ValueError(f"Ensemble checkpoint metadata differs from provider fold: {rel}")
        else:
            eval_path = _scalar(record, "evaluation_input_path")
            eval_sha = _scalar(record, "evaluation_input_sha256")
            if audit.get("evaluation_input_path") != eval_path or audit.get("evaluation_input_sha256") != eval_sha:
                raise ValueError(f"Evaluation cache source/hash differs from audit: {rel}")
            if declaration["evaluation_input_hashes"].get(eval_path) != eval_sha:
                raise ValueError(f"Evaluation input is not bound by declaration: {rel}")
            expected_eval = {eval_path: eval_sha}
            model_hashes = json.loads(_scalar(record, "model_checkpoint_hashes_json"))
            expected_models = {audit["model_checkpoint_path"]: audit["model_checkpoint_sha256"]}
            if model_hashes != expected_models:
                raise ValueError(f"Member checkpoint metadata differs from provider fold: {rel}")
        if (any(declaration["evaluation_input_hashes"].get(path) != digest
                for path, digest in expected_eval.items()) or
                any(declaration["model_checkpoint_hashes"].get(path) != digest
                    for path, digest in expected_models.items())):
            raise ValueError(f"Prediction metadata inputs are not bound by declaration: {rel}")
        if json.loads(_scalar(record, "fold_mapping_json")) != FOLD_MAP:
            raise ValueError(f"Fold mapping metadata mismatch: {rel}")
        scores = ensemble_rows if member == "ensemble" else member_rows
        score = scores.get((provider.lower(), raw_match, source_view, expected_replicate, member))
        if not score:
            raise ValueError(f"Source score row missing: {rel}")
        cells.append({
            "provider": provider, "match": match, "view": view,
            "replicate": expected_replicate, "member": member,
            "path": path, "relative_path": rel, "source_sha256": expected_sha,
            "record": record, "probabilities": p, "labels": labels,
            "period": period, "model_period": model_period,
            "timestamps": timestamps, "fps": fps, "score": score,
        })
    if len(cells) != 45:
        raise ValueError(f"Expected 45 complete paired cells, found {len(cells)}")
    return cells


def _score_prediction(labels: np.ndarray, prediction: np.ndarray, period: np.ndarray,
                      timestamps: np.ndarray, fps: float) -> dict:
    """Return official metrics and bounded scalar spell/D diagnostics."""
    result = metrics.evaluate_tracking(labels, prediction, period=period,
                                       fps=fps, timestamps=timestamps)
    # Use the same separators and reference-valid runs as the official wrapper.
    _, truth_codes, guess_codes, _ = metrics.prepare_tracking_codes(
        labels, prediction, period=period, fps=fps, timestamps=timestamps)
    runs = pm.valid_runs(truth_codes)
    gt_spells = pm.possession_spells(truth_codes, runs)
    pred_spells = pm.possession_spells(guess_codes, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    short = np.asarray([(b - a) / fps < 3.0 for a, b, _ in gt_spells], dtype=bool)
    matched_05 = pm.matched_gt_flags(overlaps, len(gt_spells), 0.5)
    d = result["per_class"]["D"]
    per_iou = result.get("spell_f1", {})
    spell_by_iou = {str(threshold): float(per_iou[threshold]["f1"])
                    for threshold in pm.SPELL_IOUS}
    return {
        "spell_f1": float(result["spell_f1_mean"]),
        "macro_f1": float(result["macro_f1"]),
        "weighted_70_30": float(.70 * result["spell_f1_mean"] + .30 * result["macro_f1"]),
        "d_f1": float(d["f1"]), "d_precision": float(d["precision"]),
        "d_recall": float(d["recall"]), "spell_by_iou": spell_by_iou,
        "reference_spell_count": len(gt_spells),
        "predicted_spell_count": len(pred_spells),
        "short_reference_spell_count": int(short.sum()),
        "short_matched_spell_count": int(matched_05[short].sum()),
        "short_spell_recall_iou_0_5_lt3s": (float(matched_05[short].mean())
                                             if short.any() else None),
        "graded_frames": int(result["n_graded"]),
    }


def candidate_prediction(probabilities: np.ndarray, model_period: np.ndarray,
                         fps: float, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Inference boundary: target labels are not accepted by this function."""
    return spell_mbr.decode(
        probabilities, model_period, fps, switch_cost=3.0,
        valid_mask=valid_mask,
        candidate_samples=spell_mbr.DEFAULT_CANDIDATE_SAMPLES,
        reference_samples=spell_mbr.DEFAULT_REFERENCE_SAMPLES,
        seed=spell_mbr.DEFAULT_SEED,
    )


def _metric_parity(actual: dict, expected: dict, atol: float = 1e-12) -> None:
    fields = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
              "d_recall", "n_gt_spells", "n_pred_spells", "graded_frames")
    for field in fields:
        actual_field = "reference_spell_count" if field == "n_gt_spells" else (
            "predicted_spell_count" if field == "n_pred_spells" else field)
        if field not in expected or actual_field not in actual:
            continue
        left, right = actual[actual_field], expected[field]
        if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
            if left != right:
                raise ValueError(f"Control replay {field} mismatch: {left} != {right}")
        elif not np.isclose(float(left), float(right), atol=atol, rtol=0):
            raise ValueError(f"Control replay {field} mismatch: {left} != {right}")


def _aggregate(rows: list[dict], member: str, view: str) -> dict:
    """Mean view replicates within match, matches within provider, providers equally."""
    metrics_to_mean = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                       "d_precision", "d_recall", "short_spell_recall_iou_0_5_lt3s",
                       "reference_spell_count", "predicted_spell_count",
                       "short_reference_spell_count", "short_matched_spell_count")
    provider_values = {}
    for provider in PROVIDERS:
        match_values = []
        for match in EXPECTED_MATCHES[provider]:
            cell_rows = [row for row in rows if row["provider"] == provider and
                         row["match"] == match and row["member"] == member and
                         (row["view"] == "clean" if view == "clean" else row["view"].startswith("feed_like_"))]
            expected_count = 1 if view == "clean" else 2
            if len(cell_rows) != expected_count:
                raise ValueError(f"Incomplete aggregate cells for {provider}/{match}/{view}/{member}")
            match_values.append({metric: (float(np.mean([r[metric] for r in cell_rows
                                                         if r.get(metric) is not None]))
                                          if any(r.get(metric) is not None for r in cell_rows)
                                          else None) for metric in metrics_to_mean})
        provider_values[provider] = {
            metric: (float(np.mean([m[metric] for m in match_values if m[metric] is not None]))
                     if any(m[metric] is not None for m in match_values) else None)
            for metric in metrics_to_mean
        }
    summary = {
        metric: float(np.mean([provider_values[p][metric] for p in PROVIDERS
                               if provider_values[p][metric] is not None]))
        if any(provider_values[p][metric] is not None for p in PROVIDERS) else None
        for metric in metrics_to_mean
    }
    return {"equal_provider_mean": summary, "by_provider": provider_values}


def _claim_output(output: Path, declaration_sha: str) -> None:
    """Exclusively claim a new private output directory before scoring starts."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    lock = {"study_id": "spell_mbr_v1", "status": "claimed_before_scoring",
            "declaration_sha256": declaration_sha,
            "claimed_utc": datetime.now(timezone.utc).isoformat()}
    lock_path = output / "run.lock.json"
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(lock, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(output: Path = DEFAULT_OUTPUT, declaration_path: Path = DECLARATION) -> dict:
    """Claim output, replay all ensemble controls, then score the 45 MBR cells."""
    declaration = verify_declaration(declaration_path)
    declaration_sha = sha256_file(declaration_path)
    _claim_output(output, declaration_sha)
    output = Path(output)
    # Load and re-validate all frozen records only after lock claim. No metrics
    # are computed until the complete 45-cell provenance/cohort checks finish.
    cells = _load_cells(declaration)
    replay_rows = []
    baseline_rows = []
    for cell in cells:
        record = cell["record"]
        baseline = supervised.decode(cell["probabilities"], cell["model_period"],
                                     cell["fps"], switch_cost=3.0).astype("U1")
        saved = np.asarray(record["prediction"]).astype("U1")
        if not np.array_equal(baseline, saved):
            raise ValueError(f"Cost-3 saved ensemble prediction replay failed: {cell['relative_path']}")
        # Native period <= 0 marks unavailable input frames; it is label-free.
        # The saved cost-3 path remains checked byte-for-byte before masking.
        baseline[ np.asarray(cell["period"]) <= 0 ] = "X"
        score = _score_prediction(cell["labels"], baseline, cell["period"],
                                  cell["timestamps"], cell["fps"])
        _metric_parity(score, cell["score"])
        row = {
            "provider": cell["provider"], "match": cell["match"],
            "view": cell["view"], "replicate": cell["replicate"],
            "member": cell["member"], **score,
            "prediction_sha256": sha256_array(baseline),
            "probabilities_sha256": sha256_array(cell["probabilities"]),
            "source_prediction_sha256": cell["source_sha256"],
        }
        baseline_rows.append(row)
        if cell["member"] == "ensemble":
            replay_rows.append(row)
        if cell["member"] != "ensemble":
            print(f"BASELINE {len(baseline_rows)}/45 {cell['provider']}/{cell['match']}/"
                  f"{cell['view']}/{cell['member']}", flush=True)
    if len(baseline_rows) != 45:
        raise ValueError(f"Expected 45 paired cost-3 baseline cells, got {len(baseline_rows)}")
    (output / "private_baseline_cell_scores.json").write_text(
        json.dumps(baseline_rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if len(replay_rows) != 15:
        raise ValueError(f"Expected 15 ensemble control replays, got {len(replay_rows)}")
    control_clean = _aggregate(replay_rows, "ensemble", "clean")["equal_provider_mean"]["spell_f1"]
    control_feed = _aggregate(replay_rows, "ensemble", "feed_like")["equal_provider_mean"]["spell_f1"]
    replay = {
        "status": "passed" if (abs(control_clean - EXPECTED_CONTROL["clean"]) <= CONTROL_TOLERANCE
                                 and abs(control_feed - EXPECTED_CONTROL["feed_like"]) <= CONTROL_TOLERANCE)
        else "failed",
        "scored_ensemble_cells": len(replay_rows),
        "clean_spell_f1": control_clean,
        "expected_clean_spell_f1": EXPECTED_CONTROL["clean"],
        "feed_like_spell_f1": control_feed,
        "expected_feed_like_spell_f1": EXPECTED_CONTROL["feed_like"],
        "absolute_tolerance": CONTROL_TOLERANCE,
        "rows": replay_rows,
    }
    (output / "control_replay.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if replay["status"] != "passed":
        raise ValueError("Reconstructed cost-3 controls failed the predeclared replay gate")

    rows = []
    private_dir = output / "candidate_predictions"
    private_dir.mkdir(mode=0o700)
    for cell in cells:
        p = cell["probabilities"]
        # Candidate inference boundary takes only p/model_period/fps. `labels`
        # is not in scope for this call and is used below only by the evaluator.
        start = time.perf_counter()
        valid_mask = np.asarray(cell["period"]) > 0
        candidate = candidate_prediction(p, cell["model_period"], cell["fps"], valid_mask)
        elapsed = time.perf_counter() - start
        candidate = np.asarray(candidate).astype("U1")
        if candidate.shape != cell["labels"].shape or not np.isin(candidate, ("H", "A", "D", "X")).all():
            raise ValueError(f"Candidate decoder returned malformed output for {cell['relative_path']}")
        score = _score_prediction(cell["labels"], candidate, cell["period"],
                                  cell["timestamps"], cell["fps"])
        pred_hash = sha256_array(candidate)
        base_key = f"{cell['provider']}_{cell['match']}_{cell['view']}_{cell['member']}"
        prediction_path = private_dir / f"{base_key}.npz"
        if prediction_path.exists():
            raise ValueError(f"Refusing to overwrite private candidate output: {prediction_path}")
        np.savez_compressed(prediction_path, prediction=candidate)
        rows.append({
            "provider": cell["provider"], "match": cell["match"],
            "view": cell["view"], "replicate": cell["replicate"],
            "member": cell["member"], **score,
            "runtime_seconds": float(elapsed),
            "prediction_sha256": pred_hash,
            "candidate_file": str(prediction_path.relative_to(output)),
            "source_prediction_sha256": cell["source_sha256"],
            "probabilities_sha256": sha256_array(p),
        })
        print(f"CELL {len(rows)}/45 {cell['provider']}/{cell['match']}/"
              f"{cell['view']}/{cell['member']} runtime={elapsed:.3f}s", flush=True)
    if len(rows) != 45:
        raise ValueError(f"Expected 45 candidate rows, got {len(rows)}")
    # Detailed cell metrics and prediction hashes remain in ignored private data.
    (output / "private_cell_scores.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, allow_nan=False) + "\n")
    summary = {}
    for member in MEMBERS:
        summary[member] = {
            "clean": _aggregate(rows, member, "clean"),
            "feed_like": _aggregate(rows, member, "feed_like"),
        }
    baseline_summary = {
        member: {
            "clean": _aggregate(baseline_rows, member, "clean"),
            "feed_like": _aggregate(baseline_rows, member, "feed_like"),
        } for member in MEMBERS
    }
    deltas = {}
    for view in ("clean", "feed_like"):
        deltas[view] = {}
        for member in MEMBERS:
            deltas[view][member] = {
                metric_name: (
                    summary[member][view]["equal_provider_mean"][metric_name] -
                    baseline_summary[member][view]["equal_provider_mean"][metric_name]
                )
                for metric_name in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                                    "d_precision", "d_recall")
            }
    providers_clean = {}
    for provider in PROVIDERS:
        providers_clean[provider] = {
            metric_name: (summary["ensemble"]["clean"]["by_provider"][provider][metric_name] -
                          baseline_summary["ensemble"]["clean"]["by_provider"][provider][metric_name])
            for metric_name in ("macro_f1", "d_f1")
        }
    guards = {
        "clean_spell_gain_at_least_0_01": deltas["clean"]["ensemble"]["spell_f1"] >= 0.01,
        "feed_like_spell_delta_at_least_minus_0_01": deltas["feed_like"]["ensemble"]["spell_f1"] >= -0.01,
        "provider_clean_d_f1_within_0_02": all(v["d_f1"] >= -0.02 for v in providers_clean.values()),
        "provider_clean_macro_f1_within_0_02": all(v["macro_f1"] >= -0.02 for v in providers_clean.values()),
        "promotion_authorized": False,
    }
    aggregate = {
        "study_id": "spell_mbr_v1",
        "status": "completed",
        "declaration_sha256": declaration_sha,
        "runner_sha256": sha256_file(Path(__file__)),
        "spell_mbr_sha256": sha256_file(ROOT / "experiments/spell_mbr.py"),
        "metrics_sha256": sha256_file(ROOT / "experiments/metrics.py"),
        "supervised_sha256": sha256_file(ROOT / "experiments/supervised.py"),
        "source_provenance_sha256": declaration["upstream"]["provenance_sha256"],
        "control_replay": {key: replay[key] for key in (
            "status", "scored_ensemble_cells", "clean_spell_f1", "expected_clean_spell_f1",
            "feed_like_spell_f1", "expected_feed_like_spell_f1", "absolute_tolerance")},
        "baseline": baseline_summary,
        "candidate": summary,
        "candidate_minus_baseline": deltas,
        "provider_clean_guard_deltas": providers_clean,
        "declared_guards": guards,
        "cell_runtime_seconds": {
            row["provider"] + "/" + row["match"] + "/" + row["view"] + "/" + row["member"]:
            row["runtime_seconds"] for row in rows
        },
        "private_artifacts": {
            "candidate_prediction_directory": str(private_dir.relative_to(ROOT)),
            "private_baseline_cell_scores": str((output / "private_baseline_cell_scores.json").relative_to(ROOT)),
            "private_cell_scores": str((output / "private_cell_scores.json").relative_to(ROOT)),
            "candidate_prediction_count": len(rows),
        },
        "interpretation": "Development-only matched evidence; no confirmation or promotion is authorized.",
    }
    report_path = output / "aggregate_report.json"
    if report_path.exists():
        raise ValueError(f"Refusing to overwrite aggregate report: {report_path}")
    report_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze", help="verify evidence and write the frozen declaration")
    freeze_parser.add_argument("--declaration", type=Path, default=DECLARATION)
    run_parser = sub.add_parser("run", help="replay controls and evaluate the frozen MBR decoder")
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--declaration", type=Path, default=DECLARATION)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.declaration)
        print(f"Frozen declaration: {args.declaration}")
    else:
        result = run(args.output, args.declaration)
        print(json.dumps({"status": result["status"], "report": str(args.output / "aggregate_report.json")},
                         sort_keys=True))


if __name__ == "__main__":
    main()
