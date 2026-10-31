"""Rebuild the selected SoccerNet reference and run its matched continuations.

This executor keeps the historical capacity rebuild declaration separate from
the later mechanism declaration. It only trains from verified TRAIN and old
cache inputs; scoring belongs to ``soccernet_followup_evaluate``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import tempfile
import time
from typing import Any

import joblib
import numpy as np

from experiments import preparation as prep
from experiments import soccernet_anchor_balance as balance
from experiments import soccernet_capacity as cap
from experiments import soccernet_capacity_model as model
from experiments import soccernet_extension as old
from experiments import soccernet_followup_evaluate as evaluation
from experiments import soccernet_segments as segments
from experiments import soccernet_training_ideas as ideas

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")).resolve()
EXECUTION_ROOT = Path(os.environ.get("MTAH_EXECUTION_DIR", ROOT / "data")).resolve()
GAR_ROOT = Path(os.environ.get("MTAH_GAR_ROOT", ROOT / "data/independent/soccernet_gar")).resolve()
OUT = EXECUTION_ROOT / "experiments/soccernet_followup_execution_v1"
SEGMENTS = OUT / "segments"
GAPS = OUT / "gaps"
REBUILD_DECLARATION = OUT / "rebuild_declaration.json"
DECLARATION = OUT / "predeclared.json"
CAPACITY_DECLARATION = ROOT / "experiments/results/soccernet_capacity_declaration.json"
CAPACITY_REPORT = ROOT / "experiments/results/soccernet_capacity_final_report.json"
RESTORE_RECEIPT = OUT / "old_restore.json"

FOLDS = tuple(cap.FOLDS)
SEEDS = tuple(cap.SEEDS)
ARMS = ("reference", *ideas.ARMS)
REFERENCE_CONFIG = dict(width=96, lr=3e-4, steps=2048, milestones=[512, 2048],
                        fraction=0.5, batch_size=32, input_dropout=0.1,
                        weight_decay=1e-4, gradient_clip=1.0,
                        require_full_native=True)

REBUILD_SOURCES = (
    "experiments/soccernet_followup_execute.py",
    "experiments/soccernet_capacity.py",
    "experiments/soccernet_capacity_model.py",
    "experiments/soccernet_capacity_public.py",
    "experiments/soccernet_data.py",
    "experiments/soccernet_extension.py",
    "experiments/soccernet_finetune.py",
    "experiments/soccernet_segments.py",
    "experiments/preparation.py",
    "experiments/combined_objective_deployment.py",
    "experiments/combined_objective_public_check.py",
    "experiments/objective_losses.py",
    "experiments/metrics.py",
    "experiments/objective_ensemble.py",
    "experiments/current_baseline.py",
    "experiments/live_gap_experiment.py",
    "experiments/robust.py",
    "experiments/spell_diagnostics.py",
    "experiments/factorized_had.py",
    "experiments/learned_temporal.py",
    "experiments/refine_features.py",
    "experiments/refine_domain.py",
    "experiments/supervised.py",
    "experiments/eval_set/manifest.json",
    "experiments/baseline_manifest.json",
    "experiments/model_decision_policy.json",
    "experiments/results/soccernet_capacity_declaration.json",
    "experiments/results/soccernet_capacity_final_report.json",
    "evaluation/possession_metrics.py",
)
def _followup_source_paths() -> tuple[str, ...]:
    paths = {p.relative_to(ROOT).as_posix() for p in (ROOT / "experiments").glob("*.py")}
    paths.update(("evaluation/possession_metrics.py", "experiments/eval_set/protocol.py",
                  "experiments/eval_set/manifest.json", "experiments/baseline_manifest.json",
                  "experiments/model_decision_policy.json",
                  "experiments/results/soccernet_capacity_declaration.json",
                  "experiments/results/soccernet_capacity_final_report.json"))
    return tuple(sorted(paths))


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = cap._jsonable(value)
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            # Permit semantic equality if a prior writer used insertion order.
            if read_json(path) != value:
                raise ValueError(f"Refusing to overwrite sealed declaration: {path}")
        return
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _atomic_joblib(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(name)
    try:
        joblib.dump(value, tmp)
        digest = sha256(tmp)
        os.replace(tmp, path)
        return digest
    finally:
        tmp.unlink(missing_ok=True)


def _save_model(path: Path, value: Any, receipt: dict) -> None:
    receipt_path = path.with_suffix(".receipt.json")
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError(f"Model/receipt pair is incomplete: {path}")
        actual = read_json(receipt_path)
        if (actual.get("model_sha256") != sha256(path)
                or {k: v for k, v in actual.items() if k != "model_sha256"} != receipt):
            raise ValueError(f"Refusing to reuse mismatched or modified model: {path}")
        return
    digest = _atomic_joblib(path, value)
    _write_once(receipt_path, dict(receipt, model_sha256=digest))


def _model_path(arm: str, fold: str, seed: int) -> Path:
    if arm not in ARMS or fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown arm, full-provider fold, or seed")
    return OUT / "models" / arm / fold / f"seed{seed}.joblib"


def _resume_path(fold: str, seed: int) -> Path:
    return OUT / "models" / "reference" / fold / f"seed{seed}.resume512.joblib"


def _source_hashes(paths: tuple[str, ...]) -> dict[str, str]:
    result = {}
    for rel in paths:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Declared source is missing: {rel}")
        result[rel] = sha256(path)
    return result


ALLOWED_HISTORICAL_METADATA_SOURCE_CHANGES = frozenset((
    "experiments/baseline_manifest.json",
    "experiments/current_baseline.py",
    "experiments/model_decision_policy.json",
))


def _historical_source_audit(historical: dict[str, str], current: dict[str, str]) -> dict:
    """Require exact historical training code, allowing known current metadata drift."""
    if set(historical) != set(current):
        raise ValueError("Historical capacity source inventory differs from current source map")
    mismatches = sorted(path for path in historical if historical[path] != current[path])
    unexpected = sorted(set(mismatches) - ALLOWED_HISTORICAL_METADATA_SOURCE_CHANGES)
    if unexpected:
        raise ValueError("Historical training-core source changed: " + ", ".join(unexpected))
    return dict(
        historical_source_sha256=dict(sorted(historical.items())),
        current_source_sha256=dict(sorted(current.items())),
        exact_training_core_sha256={path: current[path] for path in sorted(current)
                                    if path not in ALLOWED_HISTORICAL_METADATA_SOURCE_CHANGES},
        allowed_metadata_mismatches=mismatches,
        allowed_metadata_paths=sorted(ALLOWED_HISTORICAL_METADATA_SOURCE_CHANGES),
        outcome="exact training-core source parity; only named current adoption metadata may differ")


def _verify_hash_map(entries: dict[str, str], description: str) -> None:
    for raw, expected in entries.items():
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{description} hash mismatch: {path}")


def _restore_inputs() -> dict[str, str]:
    if not RESTORE_RECEIPT.is_file():
        raise FileNotFoundError(f"Missing local old-input restore receipt: {RESTORE_RECEIPT}")
    receipt = read_json(RESTORE_RECEIPT)
    if receipt.get("status") != "verified" or receipt.get("training_cache_count") != 13:
        raise ValueError("Old-input restore receipt is not complete and verified")
    files = receipt.get("files", [])
    digest_map = {}
    for entry in files:
        path = DATA / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError(f"Restored old input changed: {path}")
        digest_map[str(path)] = entry["sha256"]
    return digest_map


def _teacher_files() -> dict[str, str]:
    historical = read_json(CAPACITY_DECLARATION)
    pinned = historical.get("initial_checkpoint_sha256", {})
    provenance = historical.get("initial_checkpoint_provenance", {}).get("fold_checkpoints", {})
    files = {}
    for fold in FOLDS:
        for seed in SEEDS:
            path = old.control_path(fold, seed)
            rel = "data/" + path.relative_to(DATA).as_posix()
            expected = pinned.get(rel)
            if not expected or sha256(path) != expected:
                raise ValueError(f"Original teacher differs from capacity declaration: {path}")
            receipt = path.with_suffix(".json")
            fit_receipt = read_json(receipt)
            declared = provenance.get(f"{fold}:{seed}")
            plan = prep.split_plan(fold=fold)
            if (not declared or fit_receipt.get("sha256") != expected
                    or fit_receipt.get("heldout") != fold.removeprefix("leave_out_")
                    or fit_receipt.get("seed") != seed
                    or fit_receipt.get("objective") != "combined"
                    or set(fit_receipt.get("training_matches", [])) !=
                       {item["match"] for item in plan["train"]}
                    or sha256(receipt) != declared.get("receipt_sha256")
                    or expected != declared.get("model_sha256")):
                raise ValueError(f"Original teacher receipt differs from capacity declaration: {receipt}")
            files[str(path)] = expected
            files[str(receipt)] = sha256(receipt)
    return files


def _segment_provenance(*, load_records: bool = True) -> tuple[dict, list[dict] | None]:
    manifest_path = SEGMENTS / "manifest.json"
    records = segments.load_records(SEGMENTS) if load_records else None
    manifest = read_json(manifest_path) if load_records else segments._load_manifest(SEGMENTS)
    audit = cap._validate_segment_audit(manifest)
    raw_entries = manifest.get("files", manifest.get("game_files"))
    if isinstance(raw_entries, dict):
        raw_entries = [dict(path=key, **(value if isinstance(value, dict)
                                         else {"sha256": value}))
                       for key, value in raw_entries.items()]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Segment manifest must list its verified game payloads")
    entries = []
    for entry in raw_entries:
        relative = entry.get("path", entry.get("file"))
        expected = entry.get("sha256", entry.get("hash"))
        path_part = Path(relative) if isinstance(relative, str) else Path("/")
        if (not isinstance(relative, str) or path_part.is_absolute() or ".." in path_part.parts
                or not relative.startswith("games/") or not isinstance(expected, str)):
            raise ValueError("Malformed segment file provenance entry")
        full = SEGMENTS / path_part
        # segments._load_manifest above verifies each game_files digest. Avoid
        # reading every large payload a second time merely to record its size.
        if not full.is_file():
            raise ValueError(f"Segment payload hash mismatch: {full}")
        entries.append(dict(path=relative, sha256=expected, bytes=full.stat().st_size))
    entries.sort(key=lambda item: item["path"])
    games = {str(x) for x in manifest.get("games", [])}
    if len(games) != 45:
        raise ValueError("Prepared segment cache must contain all 45 SoccerNet TRAIN games")
    record_count = int(manifest.get("counts", {}).get("records_emitted", -1))
    if records is not None:
        if len({str(r["match"]) for r in records}) != 45:
            raise ValueError("Prepared records must contain all 45 SoccerNet TRAIN games")
        if len(records) != record_count:
            raise ValueError("Prepared records do not match manifest record count")
        record_count = len(records)
    return dict(manifest_sha256=sha256(manifest_path), files=entries, audit=audit,
                game_count=len(games), record_count=record_count,
                historical_dependency_versions=manifest.get("dependency_versions", {}),
                current_runtime_versions=_runtime_versions()), records


def _runtime_versions() -> dict[str, str]:
    packages = ("numpy", "torch", "joblib", "scipy", "scikit-learn", "pyarrow")
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in packages}}


def _rebuild_declaration_value() -> dict:
    # Load all fold plans now so the six original cache-derived training inputs
    # and provider exclusions are frozen before any reference fit begins.
    restored = _restore_inputs()
    teachers = _teacher_files()
    old_records = {}
    for fold in FOLDS:
        records = old.load_old(fold)
        plan = prep.split_plan(fold=fold)
        expected = {(item["source"], item["match"]) for item in plan["train"]}
        actual = {(str(r["source"]), str(r["match"])) for r in records}
        if actual != expected:
            raise ValueError(f"Historical full-provider training cohort differs: {fold}")
        old_records[fold] = dict(matches=sorted({str(r["match"]) for r in records}),
                                 sources=sorted({str(r["source"]) for r in records}),
                                 record_count=len(records))
    segment_info, _ = _segment_provenance()
    report = read_json(CAPACITY_REPORT)
    selected = report.get("decision", {})
    expected_recipe = "w96_lr3e-4_step2048"
    if (selected.get("selected_recipe") != expected_recipe or
            selected.get("selected_width") != 96):
        raise ValueError("Published capacity decision does not select the declared width-96 recipe")
    source_hashes = _source_hashes(REBUILD_SOURCES)
    historical_sources = read_json(CAPACITY_DECLARATION)["source_sha256"]
    historical_audit = _historical_source_audit(
        historical_sources,
        {path: source_hashes[path] for path in historical_sources})
    return dict(
        schema_version=1, study_id="soccernet_followup_execution_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Rebuild the selected historical width-96 reference from original fold teachers and verified sparse SoccerNet TRAIN records; no package restore is claimed.",
        selected_historical_recipe=expected_recipe, reference_training=REFERENCE_CONFIG,
        capacity_declaration_sha256=sha256(CAPACITY_DECLARATION),
        capacity_report_sha256=sha256(CAPACITY_REPORT),
        old_restore_receipt_sha256=sha256(RESTORE_RECEIPT),
        old_restore_file_sha256=restored, teacher_files=teachers,
        segments=segment_info, old_training=old_records,
        folds=list(FOLDS), seeds=list(SEEDS),
        source_sha256=source_hashes,
        historical_source_audit=historical_audit,
        runtime_versions=_runtime_versions(),
        organizer_labels_read=False, automatic_promotion=False)


def _check_rebuild(*, verify_artifacts: bool = True) -> dict:
    if not REBUILD_DECLARATION.is_file():
        raise FileNotFoundError("Run freeze-rebuild before reference fits")
    value = read_json(REBUILD_DECLARATION)
    current_sources = _source_hashes(REBUILD_SOURCES)
    if value.get("source_sha256") != current_sources:
        raise ValueError("Reference source changed after rebuild declaration")
    if value.get("capacity_declaration_sha256") != sha256(CAPACITY_DECLARATION):
        raise ValueError("Historical capacity declaration changed")
    if value.get("old_restore_receipt_sha256") != sha256(RESTORE_RECEIPT):
        raise ValueError("Old input restore receipt changed")
    if value.get("runtime_versions") != _runtime_versions():
        raise ValueError("Runtime package versions changed after rebuild declaration")
    historical_sources = read_json(CAPACITY_DECLARATION)["source_sha256"]
    historical_audit = _historical_source_audit(
        historical_sources,
        {path: current_sources[path] for path in historical_sources})
    if value.get("historical_source_audit") != historical_audit:
        raise ValueError("Historical capacity source parity audit changed")
    _verify_hash_map(value["teacher_files"], "historical teacher")
    if verify_artifacts:
        _verify_hash_map(value["old_restore_file_sha256"], "restored input")
        current, _ = _segment_provenance(load_records=False)
        if value["segments"] != current:
            raise ValueError("Prepared segment provenance changed after rebuild declaration")
    elif sha256(SEGMENTS / "manifest.json") != value["segments"]["manifest_sha256"]:
        # The subsequent TRAIN cache loader hashes every game payload exactly once.
        raise ValueError("Prepared segment manifest changed after rebuild declaration")
    return value


def freeze_rebuild() -> dict:
    if REBUILD_DECLARATION.is_file():
        return _check_rebuild()
    value = _rebuild_declaration_value()
    _write_once(REBUILD_DECLARATION, value)
    return value


def fit_reference(fold: str, seed: int) -> None:
    declaration = _check_rebuild(verify_artifacts=False)
    path = _model_path("reference", fold, seed)
    output_receipt = path.with_suffix(".receipt.json")
    teacher_path = old.control_path(fold, seed)
    declaration_sha = sha256(REBUILD_DECLARATION)
    expected = dict(kind="reference", fold=fold, seed=seed,
                    rebuild_declaration_sha256=declaration_sha,
                    teacher_path=str(teacher_path), teacher_sha256=sha256(teacher_path),
                    recipe=declaration["selected_historical_recipe"],
                    config=REFERENCE_CONFIG, organizer_labels_read=False)
    if path.exists() or output_receipt.exists():
        if not path.is_file() or not output_receipt.is_file():
            raise ValueError(f"Reference checkpoint/receipt pair is incomplete: {path}")
        receipt = read_json(output_receipt)
        if (receipt.get("model_sha256") != sha256(path) or
                any(receipt.get(k) != v for k, v in expected.items())):
            raise ValueError(f"Existing reference checkpoint is not a verified match: {path}")
        return
    old_records = old.load_old(fold)
    gar_records = segments.load_records(SEGMENTS)
    if len(gar_records) != declaration["segments"]["record_count"]:
        raise ValueError("Reference GAR records differ from frozen segment count")
    if len({str(r["match"]) for r in gar_records}) != 45:
        raise ValueError("Reference GAR records do not cover all frozen TRAIN fixtures")
    teacher = joblib.load(teacher_path)
    resume = None
    resume_path = _resume_path(fold, seed)
    resume_receipt = resume_path.with_suffix(".receipt.json")
    if resume_path.exists() or resume_receipt.exists():
        if not resume_path.is_file() or not resume_receipt.is_file():
            raise ValueError(f"Reference resume checkpoint/receipt pair is incomplete: {resume_path}")
        prior = read_json(resume_receipt)
        expected_resume = dict(kind="reference_resume", fold=fold, seed=seed, step=512,
            rebuild_declaration_sha256=declaration_sha,
            teacher_sha256=sha256(teacher_path), recipe=expected["recipe"], config=REFERENCE_CONFIG)
        if (prior.get("model_sha256") != sha256(resume_path) or
                {k: v for k, v in prior.items() if k != "model_sha256"} != expected_resume):
            raise ValueError(f"Existing resume checkpoint is not a verified match: {resume_path}")
        resume = joblib.load(resume_path)

    def checkpoint(step: int, snapshot: dict) -> None:
        if step != 512 or resume is not None:
            return
        resume_expected = dict(kind="reference_resume", fold=fold, seed=seed, step=512,
            rebuild_declaration_sha256=declaration_sha,
            teacher_sha256=sha256(teacher_path), recipe=expected["recipe"], config=REFERENCE_CONFIG)
        _save_model(resume_path, snapshot, resume_expected)

    started = time.monotonic()
    milestones = (2048,) if resume is not None else (512, 2048)
    snapshots = model.train(teacher, old_records, gar_records, 96, 3e-4, seed,
        steps=2048, milestones=milestones, fraction=0.5, batch_size=32,
        input_dropout=0.1, require_full_native=True, resume_from=resume,
        on_checkpoint=checkpoint)
    fitted = snapshots[2048]
    if (not np.array_equal(fitted["median"], teacher["median"]) or
            not np.array_equal(fitted["scale"], teacher["scale"])):
        raise ValueError("Reference rebuild changed the original fold normalizer")
    receipt = dict(expected, seconds=time.monotonic() - started,
                   training=fitted.get("training", {}))
    _save_model(path, fitted, receipt)


def _reference_hashes() -> tuple[dict[str, str], dict[str, str]]:
    models, receipts = {}, {}
    for fold in FOLDS:
        for seed in SEEDS:
            path = _model_path("reference", fold, seed)
            receipt_path = path.with_suffix(".receipt.json")
            if not path.is_file() or not receipt_path.is_file():
                raise FileNotFoundError(f"All six reference fits are required before continuation freeze: {path}")
            receipt = read_json(receipt_path)
            if receipt.get("model_sha256") != sha256(path):
                raise ValueError(f"Reference checkpoint hash differs from receipt: {path}")
            if receipt.get("rebuild_declaration_sha256") != sha256(REBUILD_DECLARATION):
                raise ValueError(f"Reference receipt is bound to another rebuild declaration: {path}")
            models[str(path)] = sha256(path)
            receipts[str(receipt_path)] = sha256(receipt_path)
    return models, receipts


def _evaluation_inputs() -> dict[str, str]:
    values = {}
    cells = evaluation.expected_cells()
    if len(cells) != 15:
        raise ValueError(f"Expected 15 frozen selection cells, found {len(cells)}")
    for cell in cells:
        item, view, replicate = cell["item"], cell["view"], cell["replicate"]
        cache = old.eval_path(item, view, replicate)
        values[str(cache)] = sha256(cache)
        for seed in SEEDS:
            archive = old.historical_path(item, view, replicate, seed)
            values[str(archive)] = sha256(archive)
    if len(values) != 45:
        raise ValueError(f"Expected 45 unique selection inputs, found {len(values)}")
    return values


def _gap_records(base_records: list[dict] | None) -> tuple[list[dict] | None, dict]:
    manifest_path = GAPS / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing prepared raw-gap manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    segment_manifest = read_json(SEGMENTS / "manifest.json")
    expected_games = sorted(str(x) for x in segment_manifest.get("games", []))
    if (manifest.get("schema") != "soccernet_followup_raw_gap_views_v1" or
            manifest.get("complete") is not True or
            sorted(str(x) for x in manifest.get("games", [])) != expected_games):
        raise ValueError("Gap-view manifest is incomplete or bound to another TRAIN cohort")
    if manifest.get("base_manifest_sha256") != sha256(SEGMENTS / "manifest.json"):
        raise ValueError("Gap-view cache is bound to a different base segment manifest")
    if manifest.get("acquisition_receipt", {}).get("sha256") is None:
        raise ValueError("Gap-view manifest lacks SoccerNet acquisition provenance")
    acquisition_path = GAR_ROOT / manifest["acquisition_receipt"]["path"]
    if (not acquisition_path.is_file() or
            sha256(acquisition_path) != manifest["acquisition_receipt"]["sha256"]):
        raise ValueError("Gap-view acquisition receipt differs from the verified TRAIN source")
    expected_sources = {
        "soccernet_followup_prepare.py": sha256(ROOT / "experiments/soccernet_followup_prepare.py"),
        Path(segments.__file__).name: sha256(Path(segments.__file__)),
        "soccernet_gap_consistency.py": sha256(ROOT / "experiments/soccernet_gap_consistency.py"),
        "refine_features.py": sha256(ROOT / "experiments/refine_features.py"),
        "learned_temporal.py": sha256(ROOT / "experiments/learned_temporal.py"),
    }
    if manifest.get("feature_source_sha256") != expected_sources:
        raise ValueError("Gap-view feature-source hashes differ from current code")
    gap_by_game = {}
    files = {str(manifest_path): sha256(manifest_path)}
    for game, entry in manifest["game_files"].items():
        path = GAPS / "games" / f"{game}.joblib"
        if not path.is_file() or sha256(path) != entry.get("sha256"):
            raise ValueError(f"Prepared game gap payload hash mismatch: {path}")
        files[str(path)] = entry["sha256"]
        if base_records is not None:
            payload = joblib.load(path)
            if payload.get("game_id") != str(game):
                raise ValueError(f"Prepared game gap payload identity mismatch: {path}")
            gap_by_game[str(game)] = payload["records"]
    if base_records is None:
        return None, dict(manifest_sha256=sha256(manifest_path), files=files,
                          recipe=manifest.get("feature_source_sha256"),
                          games=len(manifest["game_files"]),
                          segments=sum(int(value["segments"]) for value in manifest["game_files"].values()))
    enriched = []
    seen = set()
    for record in base_records:
        game = str(record["match"]).removeprefix("soccernet_")
        segment_id = str(record["segment_id"])
        pair = gap_by_game.get(game, {}).get(segment_id)
        if pair is None:
            raise ValueError(f"Gap view missing retained base segment {segment_id}")
        combined = dict(record, **pair)
        if np.shape(combined["gap_x"]) != np.shape(combined["x"]):
            raise ValueError(f"Gap feature shape differs for {segment_id}")
        seen.add((game, segment_id))
        enriched.append(combined)
    declared = {(game, segment) for game, records in gap_by_game.items() for segment in records}
    if seen != declared:
        raise ValueError("Gap view and retained base segments do not have exact identity parity")
    return enriched, dict(manifest_sha256=sha256(manifest_path), files=files,
                          recipe=manifest.get("feature_source_sha256"),
                          games=len(gap_by_game), segments=len(seen))


def _followup_declaration_value() -> dict:
    rebuild = _check_rebuild()
    reference_models, reference_receipts = _reference_hashes()
    segment_info, base_records = _segment_provenance()
    gar_records, gap_info = _gap_records(base_records)
    # Validate each specific mechanism's preconditions before freezing it.
    anchor_weights, anchor_balance = balance.build_anchor_weights(
        gar_records, tuple(ideas.TRAIN_FIXTURES))
    positive = np.concatenate([weight[weight > 0] for weight in anchor_weights])
    anchor_balance = dict(anchor_balance,
        post_cap_positive_row_count=int(positive.size),
        post_cap_positive_weight_mean=float(positive.mean()),
        post_cap_weight_at_cap_count=int(np.count_nonzero(positive >= anchor_balance["max_weight"])),
        post_cap_weight_at_cap_fraction=float(np.mean(positive >= anchor_balance["max_weight"])))
    for fold in FOLDS:
        old_records = old.load_old(fold)
        plan = prep.split_plan(fold=fold)
        teacher = joblib.load(old.control_path(fold, SEEDS[0]))
        initial_path = _model_path("reference", fold, SEEDS[0])
        initial = joblib.load(initial_path)
        ideas._check_training(initial, teacher, old_records, gar_records, plan,
                              list(ideas.TRAIN_FIXTURES))
    source = _source_hashes(_followup_source_paths())
    return dict(
        schema_version=1, study_id="soccernet_followup_execution_v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        question="Do within-fixture anchor balancing, raw ball-gap teacher consistency, or temporal context reconstruction improve matched SoccerNet continuation?",
        rebuild_declaration_sha256=sha256(REBUILD_DECLARATION),
        reference_models=reference_models, reference_receipts=reference_receipts,
        source_sha256=source,
        teacher_files=rebuild["teacher_files"],
        evaluation_inputs=_evaluation_inputs(), evaluation_cells=evaluation.expected_cells(),
        old_restore_receipt_sha256=sha256(RESTORE_RECEIPT),
        old_restore_file_sha256=rebuild["old_restore_file_sha256"],
        segments=segment_info, gaps=gap_info,
        training=dict(arms=list(ideas.ARMS), folds=list(FOLDS), seeds=list(SEEDS),
                      config=ideas.CONFIG, reference=REFERENCE_CONFIG,
                      anchor_balance=anchor_balance,
                      total_fits=24, updates_matched_not_wall_time=True,
                      heldout_components="full provider for all original H/A, D and normalization; SoccerNet TRAIN shared"),
        evaluation=dict(views=["clean", "feed_like"], primary="spell_f1",
                        tie_break="macro_f1", actual_probability_ensemble=True,
                        public_scoring=False, organizer_labels_read=False),
        guardrails=dict(clean_spell_delta_min=0.01, feed_like_spell_delta_min=-0.01,
                        provider_clean_macro_min=-0.02, provider_clean_d_f1_min=-0.02,
                        primary="spell_f1", tie_break="macro_f1"),
        runtime_versions=_runtime_versions(),
        automatic_promotion=False)


def _check_followup(*, verify_artifacts: bool = True) -> dict:
    if not DECLARATION.is_file():
        raise FileNotFoundError("Run freeze-followup before continuation fits")
    value = read_json(DECLARATION)
    if value.get("source_sha256") != _source_hashes(_followup_source_paths()):
        raise ValueError("Follow-up sources changed after predeclaration")
    if value.get("rebuild_declaration_sha256") != sha256(REBUILD_DECLARATION):
        raise ValueError("Follow-up declaration references a different rebuild declaration")
    if value.get("runtime_versions") != _runtime_versions():
        raise ValueError("Runtime package versions changed after follow-up freeze")
    current_models, current_receipts = _reference_hashes()
    if value.get("reference_models") != current_models or value.get("reference_receipts") != current_receipts:
        raise ValueError("Reference checkpoint hashes changed after follow-up freeze")
    _verify_hash_map(value["teacher_files"], "historical teacher")
    if verify_artifacts:
        _verify_hash_map(value["evaluation_inputs"], "evaluation input")
        _verify_hash_map(value["old_restore_file_sha256"], "restored old input")
        segment_info, base_records = _segment_provenance(load_records=False)
        if value.get("segments") != segment_info:
            raise ValueError("Segment provenance changed after follow-up freeze")
        _, gap_info = _gap_records(None)
        if value.get("gaps") != gap_info:
            raise ValueError("Gap-view provenance changed after follow-up freeze")
    elif (sha256(SEGMENTS / "manifest.json") != value["segments"]["manifest_sha256"] or
          sha256(GAPS / "manifest.json") != value["gaps"]["manifest_sha256"]):
        # Each fit below revalidates and hashes the TRAIN payloads it consumes.
        raise ValueError("Prepared TRAIN manifests changed after follow-up freeze")
    return value


def freeze_followup() -> dict:
    if DECLARATION.is_file():
        return _check_followup()
    value = _followup_declaration_value()
    _write_once(DECLARATION, value)
    return value


def fit_followup(fold: str, seed: int, arm: str) -> None:
    declaration = _check_followup(verify_artifacts=False)
    if arm not in ideas.ARMS or fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Unknown continuation arm, full-provider fold, or seed")
    path = _model_path(arm, fold, seed)
    teacher_path = old.control_path(fold, seed)
    reference_path = _model_path("reference", fold, seed)
    expected = dict(kind="continuation", arm=arm, fold=fold, seed=seed,
        declaration_sha256=sha256(DECLARATION),
        rebuild_declaration_sha256=sha256(REBUILD_DECLARATION),
        reference_model_path=str(reference_path), reference_model_sha256=sha256(reference_path),
        teacher_path=str(teacher_path), teacher_sha256=sha256(teacher_path),
        config=ideas.CONFIG, organizer_labels_read=False)
    receipt_path = path.with_suffix(".receipt.json")
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError(f"Continuation checkpoint/receipt pair is incomplete: {path}")
        receipt = read_json(receipt_path)
        if (receipt.get("model_sha256") != sha256(path) or
                any(receipt.get(k) != v for k, v in expected.items())):
            raise ValueError(f"Existing continuation checkpoint is not a verified match: {path}")
        return
    old_records = old.load_old(fold)
    _, base_records = _segment_provenance()
    gar_records, gap_info = _gap_records(base_records)
    if gap_info != declaration["gaps"]:
        raise ValueError("Gap-view provenance changed after follow-up freeze")
    teacher = joblib.load(teacher_path)
    initial = joblib.load(reference_path)
    started = time.monotonic()
    result = ideas.train(initial, teacher, old_records, gar_records,
        plan=prep.split_plan(fold=fold), fixtures=list(ideas.TRAIN_FIXTURES),
        arm=arm, seed=seed, updates=512, batch_size=32)
    receipt = dict(expected, seconds=time.monotonic() - started,
                   training=result.get("training", {}).get("followup", {}))
    _save_model(path, result, receipt)


def seal() -> dict:
    declaration = _check_followup()
    models, receipts = {}, {}
    supervised_draw_hashes = {}
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = _model_path(arm, fold, seed)
                receipt_path = path.with_suffix(".receipt.json")
                if not path.is_file() or not receipt_path.is_file():
                    raise FileNotFoundError(f"Cannot seal incomplete 30-fit grid: {path}")
                receipt = read_json(receipt_path)
                if receipt.get("model_sha256") != sha256(path):
                    raise ValueError(f"Checkpoint hash differs from receipt: {path}")
                expected_decl = (sha256(REBUILD_DECLARATION) if arm == "reference"
                                 else sha256(DECLARATION))
                key = "rebuild_declaration_sha256" if arm == "reference" else "declaration_sha256"
                if receipt.get(key) != expected_decl:
                    raise ValueError(f"Checkpoint receipt declaration binding differs: {path}")
                if arm != "reference" and receipt.get("rebuild_declaration_sha256") != sha256(REBUILD_DECLARATION):
                    raise ValueError(f"Continuation receipt lacks rebuild binding: {path}")
                if arm != "reference" and not receipt.get("training", {}).get("supervised_draws_sha256"):
                    raise ValueError(f"Continuation receipt lacks supervised draw seal: {path}")
                rel = path.relative_to(OUT).as_posix()
                rel_receipt = receipt_path.relative_to(OUT).as_posix()
                models[rel], receipts[rel_receipt] = sha256(path), sha256(receipt_path)
    for fold in FOLDS:
        for seed in SEEDS:
            hashes = {}
            for arm in ideas.ARMS:
                receipt = read_json(_model_path(arm, fold, seed).with_suffix(".receipt.json"))
                hashes[arm] = receipt["training"]["supervised_draws_sha256"]
            if len(set(hashes.values())) != 1:
                raise ValueError(f"Matched continuation arms changed supervised draws: {fold}/seed{seed}")
            supervised_draw_hashes[f"{fold}:seed{seed}"] = next(iter(hashes.values()))
    if len(models) != 30 or len(receipts) != 30:
        raise ValueError("Final seal must contain 30 models and 30 receipts")
    value = dict(schema_version=1, study_id="soccernet_followup_execution_v1",
        sealed_utc=datetime.now(timezone.utc).isoformat(),
        reference_declaration_sha256=sha256(REBUILD_DECLARATION),
        declaration_sha256=sha256(DECLARATION), source_sha256=declaration["source_sha256"],
        reference_models=declaration["reference_models"],
        reference_receipts=declaration["reference_receipts"],
        supervised_draws_sha256=supervised_draw_hashes,
        models=models, receipts=receipts, organizer_labels_read=False,
        automatic_promotion=False)
    target = OUT / "models_sealed.json"
    if target.is_file():
        prior = read_json(target)
        if {k: v for k, v in prior.items() if k != "sealed_utc"} != {
                k: v for k, v in value.items() if k != "sealed_utc"}:
            raise ValueError("Existing final seal differs from verified checkpoint grid")
        return prior
    _write_once(target, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze-rebuild")
    fit_ref = commands.add_parser("fit-reference")
    fit_ref.add_argument("--fold", choices=FOLDS, required=True)
    fit_ref.add_argument("--seed", choices=SEEDS, type=int, required=True)
    commands.add_parser("freeze-followup")
    fit = commands.add_parser("fit-followup")
    fit.add_argument("--fold", choices=FOLDS, required=True)
    fit.add_argument("--seed", choices=SEEDS, type=int, required=True)
    fit.add_argument("--arm", choices=ideas.ARMS, required=True)
    commands.add_parser("seal")
    args = parser.parse_args()
    if args.command == "freeze-rebuild":
        result = freeze_rebuild()
    elif args.command == "fit-reference":
        result = fit_reference(args.fold, args.seed)
    elif args.command == "freeze-followup":
        result = freeze_followup()
    elif args.command == "fit-followup":
        result = fit_followup(args.fold, args.seed, args.arm)
    else:
        result = seal()
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
