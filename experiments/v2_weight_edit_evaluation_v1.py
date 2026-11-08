"""One-shot GAR TEST and public diagnostics for the selected v2 weight edit.

This runner reuses the frozen GAR v2 input assembler, corruption namespace,
scorer and deployment decoder. It never edits or reopens the earlier GAR gate.
The selected pooled deployment is supplied by a sealed adapter receipt under
the ignored data directory; see ``ADAPTER_CONTRACT`` below.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np

from experiments import gar_deployment_eval_v2 as gar
from experiments import gar_deployment_eval_v2_execution as gar_exec
from experiments import feed_benchmark_audit_v1 as audit
from experiments import soccernet_dense_labels as dense

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_weight_edit_evaluation_v1"
OUT = ROOT / "data/experiments" / STUDY
RECORDS = OUT / "records"
PACKAGE_RECEIPT = ROOT / "data/experiments/v2_weight_edit_pooled_deployment_v1/deployment.json"
DECLARATION = ROOT / "experiments/results" / f"{STUDY}_declaration.json"
RESULT = ROOT / "experiments/results" / f"{STUDY}.json"
PUBLIC_OUT = ROOT / "data/experiments" / f"{STUDY}_public"
PUBLIC_CSV = PUBLIC_OUT / "candidate.csv"
PUBLIC_RECEIPT = PUBLIC_OUT / "candidate.json"
PUBLIC_DECLARATION = ROOT / "experiments/results" / f"{STUDY}_public_declaration.json"
PUBLIC_STARTED = PUBLIC_OUT / "public_check_started.json"
PUBLIC_RESULT_PRIVATE = PUBLIC_OUT / "public_result.json"
PUBLIC_RESULT = ROOT / "experiments/results" / f"{STUDY}_public_result.json"
PUBLIC_SCORE_RAW = PUBLIC_OUT / "score_raw.json"
REFERENCE = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
INPUTS = ROOT / "data/challenge/inputs.npz"
BASELINE_PUBLIC = ROOT / "experiments/results/feed_like_v2_public_result.json"
BASELINE_PUBLIC_RECEIPT = ROOT / "experiments/results/short_quality_public_v1_result.json"
BASELINE_GAR = ROOT / "experiments/results/gar_deployment_eval_v2.json"
EXPECTED_FRAMES = 199_319
EXPECTED_GRADED = 7_500
EXPECTED_GT_SPELLS = 17
PUBLIC_BOOTSTRAP_DRAWS = 1_000
GAR_BOOTSTRAP_DRAWS = 2_000
GAR_BOOTSTRAP_SEED = 20261104
SPLIT = "test"
VIEWS = ("clean", "v2_1", "v2_2", "v2_3", "v2_4")
METRICS = tuple(gar.previous.KEYS)
PREPARATION_RECEIPT = OUT / "test_preparation.json"
_GAR_RAW_VERIFIED = False

ADAPTER_CONTRACT = {
    "receipt": "deployment.json has adapter_module, adapter_source_sha256, model_files_sha256, training_game_ids, package_description.",
    "load": "adapter_module.load_deployment(receipt_path) returns an opaque package object.",
    "predict": "adapter_module.predict_probabilities(record, inputs, package) returns finite N×3 H/A/D probabilities aligned to inputs.",
    "decoder": "Probabilities are post-package pooling/normalization. Adapter decode_probabilities(probabilities, period, fps, receipt) applies the winner's frozen decoder configuration; GAR runner performs native-rate phase averaging.",
}


class EvaluationError(RuntimeError):
    """Raised when a frozen package, source, or one-use receipt fails validation."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_write_once(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_jsonable(value), sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _package_files(receipt: dict) -> dict[str, str]:
    files = receipt.get("model_files_sha256")
    if not isinstance(files, dict) or not files:
        raise EvaluationError("Pooled deployment receipt must bind every model artifact")
    resolved = {}
    for raw, expected in files.items():
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != expected:
            raise EvaluationError(f"Missing or changed pooled deployment artifact: {raw}")
        resolved[str(path)] = expected
    return resolved


def _load_receipt() -> dict:
    if not PACKAGE_RECEIPT.is_file():
        raise EvaluationError(f"Missing selected pooled deployment receipt: {PACKAGE_RECEIPT}")
    receipt = json_read(PACKAGE_RECEIPT)
    module_name = receipt.get("adapter_module")
    source_sha = receipt.get("adapter_source_sha256")
    if not isinstance(module_name, str) or not module_name.startswith("experiments."):
        raise EvaluationError("Deployment receipt must identify a repository experiments.* adapter")
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve()
    if sha256(source) != source_sha:
        raise EvaluationError("Pooled inference adapter source differs from its receipt")
    _package_files(receipt)
    if not receipt.get("training_game_ids"):
        raise EvaluationError("Deployment receipt must disclose its training-match scope")
    return receipt


def _source_hashes() -> dict[str, str]:
    paths = {
        Path(__file__), Path(gar.__file__), Path(gar_exec.__file__),
        Path(gar.previous.__file__), Path(dense.__file__), Path(audit.__file__),
        ROOT / "experiments/feed_like_v2.py", ROOT / "evaluation/possession_metrics.py",
        BASELINE_GAR, BASELINE_PUBLIC, BASELINE_PUBLIC_RECEIPT,
    }
    result = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
    receipt = _load_receipt()
    module = importlib.import_module(receipt["adapter_module"])
    result[str(Path(module.__file__).resolve().relative_to(ROOT))] = sha256(Path(module.__file__))
    return result


def _verify_frozen_gar_lineage() -> tuple[dict, list[dict], str]:
    """Validate the old helper lineage and TEST input files without its gate."""
    gar_decl = json_read(gar.DECLARATION)
    amendment = json_read(gar_exec.AMENDMENT)
    if amendment.get("original_declaration_sha256") != sha256(gar.DECLARATION):
        raise EvaluationError("Frozen GAR execution amendment no longer binds its declaration")
    for rel, expected in gar_decl["source_sha256"].items():
        if sha256(ROOT / rel) != expected:
            raise EvaluationError(f"Frozen GAR input helper source changed: {rel}")
    for rel, expected in amendment["source_sha256"].items():
        if sha256(ROOT / rel) != expected:
            raise EvaluationError(f"Frozen GAR execution helper source changed: {rel}")
    calibration_sha = sha256(gar.previous.CALIBRATION)
    if calibration_sha != gar_decl["calibration_sha256"]:
        raise EvaluationError("Frozen GAR v2 calibration differs from its original declaration")
    tracking = dense.tracking_root(dense.DEFAULT_RAW)
    test_files = [entry for entry in gar_decl["raw_files"] if entry["path"].startswith("test/")]
    if len(test_files) != 4:
        raise EvaluationError("Frozen GAR receipt does not list the four TEST source files")
    for entry in test_files:
        path = tracking / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise EvaluationError(f"GAR TEST source missing or changed: {entry['path']}")
    return gar_decl, test_files, calibration_sha


def declare() -> dict:
    """Freeze the chosen pooled package and the bounded TEST-only protocol."""
    if DECLARATION.exists() or RESULT.exists():
        raise EvaluationError("Evaluation declaration/result already exists; preserve the one-shot study")
    # Read the frozen cohort/source lineage without invoking its checkpoint
    # loader or reopening its completed gate. The original sources stay frozen.
    gar_decl, test_files, calibration_sha = _verify_frozen_gar_lineage()
    if len(gar_decl["match_ids"][SPLIT]) != 10:
        raise EvaluationError("Frozen GAR TEST cohort no longer contains ten games")
    records_manifest_path = RECORDS / "test/manifest.json"
    if not records_manifest_path.is_file():
        raise EvaluationError("Prepare this study's GAR TEST dense records before freezing the evaluation")
    records_manifest = json_read(records_manifest_path)
    if (records_manifest.get("split") != "test"
            or set(records_manifest.get("games", [])) != set(gar_decl["match_ids"][SPLIT])
            or len(records_manifest.get("game_files", {})) != 10):
        raise EvaluationError("Prepared GAR TEST records differ from the frozen ten-game cohort")
    if not PREPARATION_RECEIPT.is_file():
        raise EvaluationError("Validate/adopt prepared TEST records before freezing the evaluation")
    preparation = json_read(PREPARATION_RECEIPT)
    if (preparation.get("manifest_sha256") != sha256(records_manifest_path)
            or preparation.get("gar_test_source_files") != test_files
            or preparation.get("gar_v2_calibration_sha256") != calibration_sha):
        raise EvaluationError("TEST preparation receipt does not bind the current records/source lineage")
    receipt = _load_receipt()
    test_ids = set(gar_decl["match_ids"][SPLIT])
    train_ids = {str(x).removeprefix("soccernet_") for x in receipt["training_game_ids"]}
    if test_ids & train_ids:
        raise EvaluationError("Pooled checkpoint training scope overlaps GAR TEST")
    saved_gar = json_read(BASELINE_GAR)
    saved_public = json_read(BASELINE_PUBLIC)
    active = saved_gar["splits"]["test"]["scores"]
    if (abs(active["v2"]["original_full"]["phase_avg"]["mean"]["spell_f1"] - .701288850) > 1e-8
            or abs(saved_public["candidate"]["spell_f1_mean"] - .611764705882353) > 1e-12
            or abs(saved_public["candidate"]["macro_f1"] - .8494725958535332) > 1e-12):
        raise EvaluationError("Saved active baseline metrics differ from the expected phase-averaged package")
    declaration = {
        "study": STUDY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does the frozen direct weight-edit pooled deployment improve exact-package GAR TEST spell F1 under clean inputs and the same four feed_like_v2 corruption realizations?",
        "selected_package": {
            "receipt_path": str(PACKAGE_RECEIPT.relative_to(ROOT)),
            "receipt_sha256": sha256(PACKAGE_RECEIPT),
            "receipt": receipt,
            "artifact_sha256": _package_files(receipt),
        },
        "cohort": {
            "split": SPLIT,
            "match_ids": gar_decl["match_ids"][SPLIT],
            "games": 10,
            "exposure": "Repeatedly exposed GAR TEST: prior package scores, phase diagnostics and label-free bridge work exist. This is a bounded diagnostic for one frozen new candidate, not fresh confirmation.",
            "valid_excluded": "Nine GAR VALID games remain selected-on development and are not evaluated here.",
        },
        "gar_test_source_files": test_files,
        "gar_v2_calibration_sha256": calibration_sha,
        "frozen_gar_declaration_sha256": sha256(gar.DECLARATION),
        "frozen_gar_execution_amendment_sha256": sha256(gar_exec.AMENDMENT),
        "test_records_manifest_path": str(records_manifest_path.relative_to(ROOT)),
        "test_records_manifest_sha256": sha256(records_manifest_path),
        "test_preparation_receipt_path": str(PREPARATION_RECEIPT.relative_to(ROOT)),
        "test_preparation_receipt_sha256": sha256(PREPARATION_RECEIPT),
        "test_record_game_files_sha256": records_manifest["game_files"],
        "views": list(VIEWS),
        "corruption_seed_namespace": "gar_deployment_eval_v2/<game_id>, exactly reusing frozen realization seeds 1–4",
        "aggregation": "Average the four v2 realizations within each game, then average ten games equally.",
        "primary": "Official spell F1; macro F1 breaks exact ties.",
        "diagnostics": ["macro_f1", "d_f1", "d_precision", "d_recall", "short_spell_recall_iou_0_5_lt3s", "predicted_spell_count", "reference_spell_count", "graded_frames"],
        "baseline": {
            "gar_report": str(BASELINE_GAR.relative_to(ROOT)),
            "gar_report_sha256": sha256(BASELINE_GAR),
            "package": "active original_full with five-phase averaging; saved per-game clean/v2 means reused, no baseline rescoring",
            "public_report": str(BASELINE_PUBLIC.relative_to(ROOT)),
            "public_report_sha256": sha256(BASELINE_PUBLIC),
            "public_spell_f1": saved_public["candidate"]["spell_f1_mean"],
            "public_macro_f1": saved_public["candidate"]["macro_f1"],
        },
        "decoder": receipt.get("decoder_description", "Adapter-bound selected decoder and package-specific normalization; native GAR phases from the frozen phase_set helper."),
        "bootstrap": {"unit": "whole game", "draws": GAR_BOOTSTRAP_DRAWS, "seed": GAR_BOOTSTRAP_SEED, "paired_to_saved_baseline": True},
        "limits": ["GAR labels are event-derived silver labels over discontinuous segments, not full-match gold", "Corruption placement uses the frozen v2 simulator and GAR labels; inference receives tracking only", "This is repeated use of GAR TEST for a new predeclared candidate and makes it development exposure for subsequent selection", "No VALID evaluation, organizer scoring, tuning, or automatic promotion"],
        "source_sha256": _source_hashes(),
        "fitting": False,
        "organizer_reference_access": False,
        "public_scoring": False,
        "promotion": False,
    }
    json_write_once(DECLARATION, declaration)
    return declaration


def check_declaration() -> dict:
    global _GAR_RAW_VERIFIED
    declaration = json_read(DECLARATION)
    for rel, expected in declaration["source_sha256"].items():
        if sha256(ROOT / rel) != expected:
            raise EvaluationError(f"Evaluation source changed after freeze: {rel}")
    if (sha256(gar.DECLARATION) != declaration["frozen_gar_declaration_sha256"]
            or sha256(gar_exec.AMENDMENT) != declaration["frozen_gar_execution_amendment_sha256"]):
        raise EvaluationError("Frozen GAR declaration or execution amendment changed")
    if sha256(PACKAGE_RECEIPT) != declaration["selected_package"]["receipt_sha256"]:
        raise EvaluationError("Selected deployment receipt changed after freeze")
    manifest_path = ROOT / declaration["test_records_manifest_path"]
    if sha256(manifest_path) != declaration["test_records_manifest_sha256"]:
        raise EvaluationError("Prepared TEST record manifest changed")
    preparation_path = ROOT / declaration["test_preparation_receipt_path"]
    if sha256(preparation_path) != declaration["test_preparation_receipt_sha256"]:
        raise EvaluationError("TEST preparation/adoption receipt changed")
    if not _GAR_RAW_VERIFIED:
        tracking = dense.tracking_root(dense.DEFAULT_RAW)
        for entry in declaration["gar_test_source_files"]:
            path = tracking / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
                raise EvaluationError(f"GAR TEST source changed after declaration: {entry['path']}")
        if sha256(gar.previous.CALIBRATION) != declaration["gar_v2_calibration_sha256"]:
            raise EvaluationError("GAR v2 calibration changed after declaration")
        _GAR_RAW_VERIFIED = True
    receipt = _load_receipt()
    if receipt != declaration["selected_package"]["receipt"]:
        raise EvaluationError("Selected deployment receipt differs from the declaration")
    return declaration


def _load_deployment():
    declaration = check_declaration()
    module = importlib.import_module(declaration["selected_package"]["receipt"]["adapter_module"])
    if (not callable(getattr(module, "load_deployment", None))
            or not callable(getattr(module, "predict_probabilities", None))
            or not callable(getattr(module, "decode_probabilities", None))):
        raise EvaluationError("Adapter must implement load_deployment, predict_probabilities and decode_probabilities")
    return module, module.load_deployment(PACKAGE_RECEIPT)


def _candidate_probability(record, inputs, adapter, deployment):
    values = np.asarray(adapter.predict_probabilities(record, inputs, deployment))
    if values.shape != (len(inputs["period"]), 3):
        raise EvaluationError(f"Adapter returned probability shape {values.shape}; expected {(len(inputs['period']), 3)}")
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0)):
        raise EvaluationError("Pooled deployment returned invalid H/A/D probabilities")
    return values


def corruption_seed_key(game_id: str) -> str:
    """Preserve the exact namespace used to make the saved GAR v2 views."""
    return gar.STUDY + "/" + str(game_id)


def paired_game_summary(candidate_values, baseline_values, rng, draws=GAR_BOOTSTRAP_DRAWS) -> dict:
    """Summarize equal-game candidate scores and a paired whole-game interval."""
    candidate_values = np.asarray(candidate_values, dtype=float)
    baseline_values = np.asarray(baseline_values, dtype=float)
    if candidate_values.ndim != 1 or candidate_values.shape != baseline_values.shape or not len(candidate_values):
        raise ValueError("Candidate and baseline must be nonempty, aligned one-dimensional game scores")
    delta = candidate_values - baseline_values
    sample = rng.integers(0, len(delta), size=(draws, len(delta)))
    boot = delta[sample].mean(axis=1)
    return {
        "candidate_equal_game_mean": float(candidate_values.mean()),
        "saved_baseline_equal_game_mean": float(baseline_values.mean()),
        "delta": float(delta.mean()),
        "games_positive": int((delta > 0).sum()),
        "games": int(len(delta)),
        "paired_game_bootstrap_95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
    }


def average_realizations(realization_rows: dict[str, dict[str, float]], metrics=METRICS) -> dict:
    """Average v2 realizations within a game and verify invariant counts."""
    names = [f"v2_{index}" for index in range(1, 5)]
    if set(names) - set(realization_rows):
        raise ValueError("A game must have all four fixed v2 realizations")
    result = {}
    invariant = {"graded_frames", "reference_spell_count"}
    for metric in metrics:
        values = [float(realization_rows[name][metric]) for name in names]
        if metric in invariant and len(set(values)) != 1:
            raise ValueError(f"{metric} changed across fixed v2 realizations")
        result[metric] = float(np.mean(values))
    return result


def evaluate_game(game_id: str) -> Path:
    declaration = check_declaration()
    if game_id not in declaration["cohort"]["match_ids"]:
        raise EvaluationError(f"Undeclared GAR TEST game: {game_id}")
    target = OUT / "games" / f"test__{game_id}.json"
    if target.exists():
        saved = json_read(target)
        if saved.get("declaration_sha256") != sha256(DECLARATION):
            raise EvaluationError(f"Existing game output has stale provenance: {game_id}")
        return target
    adapter, deployment = _load_deployment()
    records = gar.previous.load_game_records("test", game_id, RECORDS)
    captured = gar.previous.capture_inputs("test", game_id)
    clean_inputs, labels, saved_x, names = gar.previous.build_game(records, captured)
    calibration = json_read(gar.previous.CALIBRATION)["calibration"]
    fps = float(records[0]["fps"])
    scores, predictions, phase_counts = {}, {}, {}
    started = time.monotonic()
    for view in VIEWS:
        if view == "clean":
            inputs = clean_inputs
        else:
            # Keep the frozen prior-study seed namespace byte-for-byte; the new
            # evaluation study ID must not generate new corruption draws.
            inputs = gar.previous.feed_view(clean_inputs, labels, calibration,
                corruption_seed_key(game_id), int(view[-1]))[0]
        phases = gar.previous.phase_set(inputs, names, fps)
        phase_counts[view] = len(phases)
        expected_phases = int(round(fps / 5.0))
        if phase_counts[view] != expected_phases:
            raise EvaluationError(
                f"GAR {view} phase count {phase_counts[view]} != expected native grid {expected_phases}")
        if view == "clean" and not np.array_equal(phases[0]["x"], saved_x, equal_nan=True):
            raise EvaluationError("Clean phase-0 features differ from frozen GAR dense records")
        native = audit.tc.inputs_only(inputs)
        period = audit.supervised.model_periods(native)
        fps_for_decode = float(inputs["fps"])
        phase_probabilities = [_candidate_probability(record, native, adapter, deployment) for record in phases]
        if len({p.shape for p in phase_probabilities}) != 1:
            raise EvaluationError("Native phase probability arrays are misaligned")
        probabilities = np.mean(np.stack(phase_probabilities, axis=0), axis=0)
        predicted = np.asarray(adapter.decode_probabilities(probabilities, period, fps_for_decode,
            declaration["selected_package"]["receipt"]))
        if predicted.shape != labels.shape:
            raise EvaluationError("Selected adapter decoder returned a prediction with the wrong frame count")
        cell = {"labels": labels, "period": inputs["period"], "timestamps": inputs["timestamps"], "fps": fps_for_decode}
        measured = audit.score(cell, predicted)
        scores[view] = {key: measured[key] for key in METRICS if key in measured}
        predictions[view] = predicted.astype("S1")
    counts = {(row["graded_frames"], row["reference_spell_count"]) for row in scores.values()}
    if len(counts) != 1:
        raise EvaluationError("Grading population or reference count changed across candidate views")
    target.parent.mkdir(parents=True, exist_ok=True)
    native_path = target.with_suffix(".npz")
    np.savez_compressed(native_path, labels=labels.astype("S1"), period=clean_inputs["period"],
        timestamps=clean_inputs["timestamps"], fps=clean_inputs["fps"], **predictions)
    json_write_once(target, {
        "study": STUDY, "split": "test", "game_id": game_id,
        "declaration_sha256": sha256(DECLARATION), "package_receipt_sha256": sha256(PACKAGE_RECEIPT),
        "records": len(records), "native_frames": len(labels), "graded_frames": int((labels != "X").sum()),
        "phase_counts": phase_counts, "seed_namespace_reused": corruption_seed_key(game_id),
        "scores": scores, "prediction_sha256": sha256(native_path),
        "grading_and_reference_counts_equal": True, "seconds": time.monotonic() - started,
    })
    return target


def summarize() -> dict:
    declaration = check_declaration()
    saved_baseline = json_read(BASELINE_GAR)["splits"]["test"]["scores"]
    cells = {}
    for game in declaration["cohort"]["match_ids"]:
        path = OUT / "games" / f"test__{game}.json"
        value = json_read(path)
        npz_path = path.with_suffix(".npz")
        if (value.get("declaration_sha256") != sha256(DECLARATION)
                or value.get("package_receipt_sha256") != declaration["selected_package"]["receipt_sha256"]
                or value.get("prediction_sha256") != sha256(npz_path)
                or value.get("game_id") != game):
            raise EvaluationError(f"Game result provenance or prediction changed: {game}")
        cells[game] = value
    metrics = [key for key in METRICS if key in next(iter(cells.values()))["scores"]["clean"]]
    per_game, summaries, realization_means = {}, {}, {}
    rng = np.random.default_rng(GAR_BOOTSTRAP_SEED)
    game_ids = declaration["cohort"]["match_ids"]
    per_game["clean"] = {g: cells[g]["scores"]["clean"] for g in game_ids}
    per_game["v2"] = {g: average_realizations(cells[g]["scores"], metrics) for g in game_ids}
    per_game["paired"] = {"clean": {}, "v2": {}}
    realization_means = {f"v2_{index}": {
        metric: float(np.mean([cells[g]["scores"][f"v2_{index}"][metric] for g in game_ids]))
        for metric in metrics
    } for index in range(1, 5)}
    for summary_view, saved_view in (("clean", "clean"), ("v2", "v2")):
        baseline_game = saved_baseline[saved_view]["original_full"]["phase_avg"]["per_game"]
        summaries[summary_view] = {}
        for metric in metrics:
            candidate_values = np.asarray([per_game[summary_view][g][metric] for g in game_ids], dtype=float)
            base_values = np.asarray([baseline_game[g][metric] for g in game_ids], dtype=float)
            deltas = candidate_values - base_values
            per_game["paired"][summary_view][metric] = {
                g: {"candidate": float(c), "saved_baseline": float(b), "delta": float(d)}
                for g, c, b, d in zip(game_ids, candidate_values, base_values, deltas)
            }
            summaries[summary_view][metric] = paired_game_summary(candidate_values, base_values, rng)
    result = {
        "study": STUDY, "reported_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": sha256(DECLARATION), "primary_view": "v2 game mean across four fixed realizations",
        "primary_metric": "spell_f1", "ranking_tie_breaker": "macro_f1",
        "per_game": per_game, "per_realization_equal_game_means": realization_means, "summary": summaries,
        "saved_baseline_source": str(BASELINE_GAR.relative_to(ROOT)),
        "public_baseline_reused": {"source": str(BASELINE_PUBLIC.relative_to(ROOT)),
                                   "spell_f1": .611764705882353, "macro_f1": .8494725958535332},
        "limits": declaration["limits"], "organizer_reference_access": False,
        "public_scoring": False, "fitting": False, "promotion": False,
    }
    json_write_once(RESULT, result)
    return result


def prepare() -> None:
    gar_decl, test_files, calibration_sha = _verify_frozen_gar_lineage()
    if PREPARATION_RECEIPT.exists():
        saved = json_read(PREPARATION_RECEIPT)
        manifest = RECORDS / "test/manifest.json"
        if (not manifest.is_file() or saved["manifest_sha256"] != sha256(manifest)
                or saved["gar_test_source_files"] != test_files
                or saved["gar_v2_calibration_sha256"] != calibration_sha):
            raise EvaluationError("Prepared TEST records or source lineage changed; preserve and investigate")
        print(json.dumps(saved, indent=2, sort_keys=True))
        return
    manifest = RECORDS / "test/manifest.json"
    if manifest.exists():
        prepared_manifest = json_read(manifest)
        if (prepared_manifest.get("split") != "test"
                or set(prepared_manifest.get("games", [])) != set(gar_decl["match_ids"][SPLIT])
                or len(prepared_manifest.get("game_files", {})) != 10):
            raise EvaluationError("Existing TEST manifest does not match the frozen cohort; preserve it")
        game_files = prepared_manifest["game_files"]
        for name, expected in game_files.items():
            path = RECORDS / "test/games" / name
            if not path.is_file() or sha256(path) != expected:
                raise EvaluationError(f"Prepared TEST game record missing or changed: {name}")
        prepared = {
            "games": prepared_manifest["games"],
            "game_records": prepared_manifest.get("game_records", {}),
            "game_files": game_files,
        }
        adopted_existing = True
    else:
        prepared = dense.prepare_split("test", root=dense.DEFAULT_RAW, output=RECORDS, workers=2)
        if len(prepared["games"]) != 10:
            raise EvaluationError("Prepared TEST records do not contain ten games")
        adopted_existing = False
    receipt = {
        "study": STUDY, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "split": "test", "games": prepared["games"], "records": sum(prepared["game_records"].values()),
        "manifest_sha256": sha256(manifest),
        "game_record_sha256": prepared["game_files"],
        "gar_declaration_sha256": sha256(gar.DECLARATION),
        "gar_execution_amendment_sha256": sha256(gar_exec.AMENDMENT),
        "gar_test_source_files": test_files, "gar_v2_calibration_sha256": calibration_sha,
        "labels_accessed_for_preparation": True, "model_scoring": False,
        "organizer_reference_access": False, "public_scoring": False,
        "adopted_existing_preparation": adopted_existing,
    }
    json_write_once(PREPARATION_RECEIPT, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def games(workers: int = 1) -> None:
    declaration = check_declaration()
    ids = declaration["cohort"]["match_ids"]
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(evaluate_game, game_id) for game_id in ids]
        for future in as_completed(futures):
            print("DONE", future.result(), flush=True)


def render_public() -> dict:
    """Generate one label-free sealed full-match CSV via the package adapter."""
    declaration = check_declaration()
    if (PUBLIC_CSV.exists() or PUBLIC_RECEIPT.exists() or PUBLIC_DECLARATION.exists()
            or PUBLIC_STARTED.exists() or PUBLIC_RESULT_PRIVATE.exists() or PUBLIC_SCORE_RAW.exists()
            or PUBLIC_RESULT.exists()):
        raise EvaluationError("Public gate is already declared or claimed; never render a replacement CSV")
    if not INPUTS.is_file():
        raise EvaluationError(f"Missing sealed tracking input: {INPUTS}")
    adapter, deployment = _load_deployment()
    predict = getattr(adapter, "predict_challenge", None)
    if not callable(predict):
        raise EvaluationError("Deployment adapter must implement label-free predict_challenge(package, inputs_npz)")
    from evaluation import possession_metrics as pm
    frame = predict(deployment, INPUTS)
    error = pm.validate_submission(frame, EXPECTED_FRAMES)
    if error:
        raise EvaluationError(f"Generated public CSV is invalid: {error}")
    labels = np.asarray([str(v).strip().upper() for v in frame["label"]])
    with np.load(INPUTS, allow_pickle=False) as data:
        if "period" not in data.files or np.asarray(data["period"]).shape != (EXPECTED_FRAMES,):
            raise EvaluationError("Challenge tracking input has an invalid period mask")
        period = np.asarray(data["period"])
        input_sha = sha256(INPUTS)
    if not np.array_equal(labels == "X", period <= 0):
        raise EvaluationError("Generated candidate X mask differs from sealed challenge tracking")
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PUBLIC_CSV, index=False, lineterminator="\n")
    # Re-read the written artifact so its receipt certifies the exact CSV bytes
    # that will be validated and scored later.
    saved = pm.read_label_csv(PUBLIC_CSV)
    error = pm.validate_submission(saved, EXPECTED_FRAMES)
    if error:
        raise EvaluationError(f"Written public CSV is invalid: {error}")
    receipt = {
        "study": STUDY, "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_csv": str(PUBLIC_CSV.relative_to(ROOT)), "candidate_csv_sha256": sha256(PUBLIC_CSV),
        "rows": EXPECTED_FRAMES, "format_validation": {"valid": True, "rows": EXPECTED_FRAMES},
        "input_sha256": input_sha,
        "package_receipt_sha256": declaration["selected_package"]["receipt_sha256"],
        "adapter_source_sha256": declaration["selected_package"]["receipt"]["adapter_source_sha256"],
        "prediction_adapter": declaration["selected_package"]["receipt"]["adapter_module"],
        "organizer_reference_labels_read": False, "organizer_scored": False,
        "organizer_score": None,
    }
    json_write_once(PUBLIC_RECEIPT, receipt)
    return receipt


def _public_declaration_data() -> dict:
    """Verify the one sealed CSV and return a declaration without opening GT."""
    if not PUBLIC_CSV.is_file() or not PUBLIC_RECEIPT.is_file():
        raise EvaluationError("Generate the candidate CSV and its label-free receipt first")
    receipt = json_read(PUBLIC_RECEIPT)
    if (receipt.get("candidate_csv_sha256") != sha256(PUBLIC_CSV)
            or receipt.get("package_receipt_sha256") != sha256(PACKAGE_RECEIPT)
            or receipt.get("input_sha256") != sha256(INPUTS)
            or receipt.get("organizer_reference_labels_read") is not False
            or receipt.get("organizer_scored") is not False
            or receipt.get("organizer_score") is not None):
        raise EvaluationError("Candidate CSV receipt is stale or claims organizer access")
    from evaluation import possession_metrics as pm
    frame = pm.read_label_csv(PUBLIC_CSV)
    error = pm.validate_submission(frame, EXPECTED_FRAMES)
    if error:
        raise EvaluationError(f"Candidate CSV is invalid: {error}")
    labels = np.asarray([str(v).strip().upper() for v in frame["label"]])
    with np.load(INPUTS, allow_pickle=False) as inputs:
        period = np.asarray(inputs["period"])
    if not np.array_equal(labels == "X", period <= 0):
        raise EvaluationError("Candidate X mask differs from the sealed tracking input")
    prior_public = json_read(BASELINE_PUBLIC)
    prior_receipt = json_read(BASELINE_PUBLIC_RECEIPT)
    if (prior_public.get("candidate_label") != "original_full + 5 Hz phase TTA (5 phases, equal mean, f062)"
            or prior_public["candidate"]["spell_f1_mean"] != .611764705882353
            or abs(prior_public["candidate"]["macro_f1"] - .8494725958535332) > 1e-12):
        raise EvaluationError("Saved active public baseline score changed")
    return {
        "study": STUDY + "_public", "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "One candidate diagnostic on the fixed 199,319-row organizer public window; no full-match claim.",
        "candidate_csv": str(PUBLIC_CSV.relative_to(ROOT)),
        "candidate_csv_sha256": sha256(PUBLIC_CSV),
        "candidate_receipt_sha256": sha256(PUBLIC_RECEIPT),
        "prediction_declaration_sha256": sha256(DECLARATION),
        "package_receipt_sha256": sha256(PACKAGE_RECEIPT),
        "input_sha256": sha256(INPUTS),
        "organizer_reference_sha256": prior_receipt["reference_sha256"],
        "saved_active_baseline": {
            "source": str(BASELINE_PUBLIC.relative_to(ROOT)),
            "source_sha256": sha256(BASELINE_PUBLIC),
            "spell_f1_mean": prior_public["candidate"]["spell_f1_mean"],
            "macro_f1": prior_public["candidate"]["macro_f1"],
            "new_unique_csvs_scored": 0,
        },
        "expected": {"frames": EXPECTED_FRAMES, "graded_frames": EXPECTED_GRADED,
                     "ground_truth_spells": EXPECTED_GT_SPELLS},
        "primary_metric": "official spell F1; macro F1 breaks exact ties",
        "candidate_count": 1,
        "source_sha256": _source_hashes(),
        "organizer_reference_labels_read_before_gate": False,
        "organizer_scored_before_gate": False,
        "existing_gates_reset_or_modified": False,
        "automatic_selection_or_promotion": False,
    }


def freeze_public() -> dict:
    """Format-validate and exclusively seal the single CSV before GT access."""
    if (PUBLIC_DECLARATION.exists() or PUBLIC_STARTED.exists() or PUBLIC_RESULT.exists()
            or PUBLIC_RESULT_PRIVATE.exists() or PUBLIC_SCORE_RAW.exists()):
        raise EvaluationError("Public diagnostic gate is closed; do not reseal or rescore")
    data = _public_declaration_data()
    json_write_once(PUBLIC_DECLARATION, data)
    return data


def _verify_public_frozen() -> dict:
    declaration = json_read(PUBLIC_DECLARATION)
    for rel, expected in declaration["source_sha256"].items():
        if sha256(ROOT / rel) != expected:
            raise EvaluationError(f"Public diagnostic source changed after freeze: {rel}")
    current = _public_declaration_data()
    for key in ("candidate_csv_sha256", "candidate_receipt_sha256", "prediction_declaration_sha256",
                "package_receipt_sha256", "input_sha256", "organizer_reference_sha256"):
        if declaration.get(key) != current.get(key):
            raise EvaluationError(f"Public candidate changed after freeze: {key}")
    return declaration


def _public_summary(score: dict) -> dict:
    spell = float(score["spell_f1_mean"])
    macro = float(score["macro_f1"])
    return {
        "primary_metric": "spell_f1_mean", "spell_f1_mean": spell, "macro_f1": macro,
        "auxiliary_70_30": 0.7 * spell + 0.3 * macro,
        "d_precision": float(score["per_class"]["D"]["precision"]),
        "d_recall": float(score["per_class"]["D"]["recall"]),
        "d_f1": float(score["per_class"]["D"]["f1"]),
        "spell_by_iou": {f"{t:.1f}": {k: score["spell_f1"][t][k]
            for k in ("precision", "recall", "f1", "n_matched", "n_spurious", "n_missed")}
            for t in (0.3, 0.4, 0.5, 0.6, 0.7)},
        "frames": int(score["n_frames"]), "graded_frames": int(score["n_graded"]),
        "ground_truth_spells": int(score["n_gt_spells"]), "predicted_spells": int(score["n_pred_spells"]),
        "spell_f1_bootstrap_5_95": [float(score["spell_bootstrap_low"]), float(score["spell_bootstrap_high"])],
        "macro_f1_bootstrap_5_95": [float(score["bootstrap_low"]), float(score["bootstrap_high"])],
        "bootstrap_draws": int(score["bootstrap_n"]),
    }


def score_public() -> dict:
    """Claim the exclusive new gate, then access and score the reference once."""
    if PUBLIC_STARTED.exists() or PUBLIC_SCORE_RAW.exists() or PUBLIC_RESULT_PRIVATE.exists() or PUBLIC_RESULT.exists():
        raise EvaluationError("Public gate is already claimed; saved result is final")
    declaration = _verify_public_frozen()
    json_write_once(PUBLIC_STARTED, {
        "study": declaration["study"], "started_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": sha256(PUBLIC_DECLARATION),
        "candidate_csv_sha256": declaration["candidate_csv_sha256"],
        "organizer_reference_sha256_from_prior_receipt": declaration["organizer_reference_sha256"],
        "reference_opened_before_marker": False,
    })
    # The exclusive marker permanently closes this new diagnostic before any
    # organizer reference read. A failed read/score also closes it forever.
    declaration = _verify_public_frozen()
    if sha256(REFERENCE) != declaration["organizer_reference_sha256"]:
        raise EvaluationError("Organizer reference differs from the previously recorded public receipt")
    from evaluation import possession_metrics as pm
    reference = pm.read_label_csv(REFERENCE)
    error = pm.validate_submission(reference, EXPECTED_FRAMES)
    if error:
        raise EvaluationError(f"Organizer reference format differs: {error}")
    predicted = pm.read_label_csv(PUBLIC_CSV)
    error = pm.validate_submission(predicted, EXPECTED_FRAMES)
    if error:
        raise EvaluationError(f"Frozen candidate CSV became invalid: {error}")
    score = pm.evaluate(pm.label_codes(reference), pm.label_codes(predicted), bootstrap_n=PUBLIC_BOOTSTRAP_DRAWS)
    raw_receipt = {
        "study": declaration["study"], "scored_utc": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": sha256(PUBLIC_DECLARATION),
        "reference_sha256": declaration["organizer_reference_sha256"],
        "candidate_csv_sha256": declaration["candidate_csv_sha256"],
        "bootstrap_draws": PUBLIC_BOOTSTRAP_DRAWS,
        "official_scorer": "evaluation.possession_metrics.evaluate",
        "raw_score": score,
    }
    # Preserve the official scorer return value before coverage checks or any
    # presentation formatting. The gate is already claimed and cannot rescore.
    json_write_once(PUBLIC_SCORE_RAW, raw_receipt)
    if (score["n_frames"] != EXPECTED_FRAMES or score["n_graded"] != EXPECTED_GRADED
            or score["n_gt_spells"] != EXPECTED_GT_SPELLS or score["bootstrap_n"] != PUBLIC_BOOTSTRAP_DRAWS):
        raise EvaluationError("Public reference coverage differs from frozen five-minute window")
    result = {
        "study": declaration["study"], "reported_utc": datetime.now(timezone.utc).isoformat(),
        "frozen": declaration, "candidate": _public_summary(score),
        "saved_active_baseline_reused": declaration["saved_active_baseline"],
        "candidate_minus_saved_baseline": {
            "spell_f1_mean": _public_summary(score)["spell_f1_mean"] - declaration["saved_active_baseline"]["spell_f1_mean"],
            "macro_f1": _public_summary(score)["macro_f1"] - declaration["saved_active_baseline"]["macro_f1"],
        },
        "organizer_reference_labels_read": True, "organizer_scored": True,
        "raw_score_receipt": str(PUBLIC_SCORE_RAW.relative_to(ROOT)),
        "raw_score_receipt_sha256": sha256(PUBLIC_SCORE_RAW),
        "organizer_reference_sha256": declaration["organizer_reference_sha256"],
        "interpretation": {"public_pipeline_diagnostic_only": True, "independent_confirmation": False,
            "full_match_score": None, "public_used_for_training_tuning_selection": False,
            "existing_gates_reset_or_modified": False, "promotion": False,
            "interval_note": "Bootstrap interval is conditional on this single public match and is not a paired difference interval."},
    }
    json_write_once(PUBLIC_RESULT_PRIVATE, result)
    json_write_once(PUBLIC_RESULT, result)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("declare", "prepare", "games", "report",
                    "render-public", "freeze-public", "score-public"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.action == "declare":
        output = declare()
    elif args.action == "prepare":
        prepare(); output = {"prepared": "test"}
    elif args.action == "games":
        games(args.workers); output = {"games": "complete"}
    elif args.action == "report":
        output = summarize()
    elif args.action == "render-public":
        output = render_public()
    elif args.action == "freeze-public":
        output = freeze_public()
    else:
        output = score_public()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
