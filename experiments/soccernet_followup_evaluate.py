"""Sealed external evaluation for the SoccerNet continuation follow-up.

This runner only scores the predeclared five-match selection cohort.  It reads
the execution's sealed checkpoints, cached feature/label cells, and restored
objective teachers; it never invokes a frozen study gate or organizer scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from experiments import factorized_had as fh
from experiments import soccernet_capacity as cap
from experiments import soccernet_capacity_model as cap_model
from experiments import soccernet_extension as ext
from experiments import supervised


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("reference", "control", "balanced_anchors", "gap_consistency",
        "context_reconstruction")
SEEDS = (2026, 2027)
MEMBERS = ("2026", "2027", "ensemble")
VIEWS = ext.VIEWS
IOU_KEYS = ("0.3", "0.4", "0.5", "0.6", "0.7")
SUMMARY_FIELDS = (
    "spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
    "d_recall", "short_spell_recall", "n_gt_spells", "n_pred_spells",
    "fragmented_reference_spells", "graded_frames",
)
PUBLISHED_TOLERANCE = 1e-9


class EvaluationError(RuntimeError):
    """Raised when any declaration, seal, cohort, or output is incomplete."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expect_hashes(entries: dict[str, str], *, base: Path, description: str) -> None:
    if not isinstance(entries, dict) or not entries:
        raise EvaluationError(f"Missing {description} hash map")
    for raw, expected in entries.items():
        path = Path(raw)
        if not path.is_absolute():
            path = base / path
        if not path.is_file() or sha256(path) != expected:
            raise EvaluationError(f"{description} hash mismatch: {path}")


def _verify_checkpoint_receipt(receipt: dict, model_rel: str, model_digest: str,
                               declaration_sha: str, rebuild_sha: str) -> None:
    if receipt.get("model_sha256") != model_digest:
        raise EvaluationError(f"Checkpoint receipt does not bind {model_rel}")
    if model_rel.startswith("models/reference/"):
        if receipt.get("rebuild_declaration_sha256") != rebuild_sha:
            raise EvaluationError(f"Checkpoint receipt rebuild binding differs: {model_rel}")
    elif (receipt.get("declaration_sha256") != declaration_sha or
          receipt.get("rebuild_declaration_sha256") != rebuild_sha):
        raise EvaluationError(f"Checkpoint receipt declaration binding differs: {model_rel}")


def _verify_output_seal(report: dict, evaluation_seal: dict, execution: Path,
                        report_path: Path, rows_path: Path) -> None:
    declaration_sha = sha256(execution / "predeclared.json")
    model_seal_sha = sha256(execution / "models_sealed.json")
    rebuild_sha = sha256(execution / "rebuild_declaration.json")
    if (report.get("declaration_sha256") != declaration_sha or
            report.get("model_seal_sha256") != model_seal_sha):
        raise EvaluationError("Aggregate is bound to a different declaration or model seal")
    if (evaluation_seal.get("declaration_sha256") != declaration_sha or
            evaluation_seal.get("model_seal_sha256") != model_seal_sha or
            evaluation_seal.get("reference_declaration_sha256") != rebuild_sha or
            evaluation_seal.get("aggregate_sha256") != sha256(report_path)):
        raise EvaluationError("Evaluation output seal does not bind current report and declarations")
    rows_sha = sha256(rows_path)
    if (report.get("private_rows_sha256") != rows_sha or
            evaluation_seal.get("scored_rows_sha256") != rows_sha):
        raise EvaluationError("Evaluation output seal does not bind private score rows")
    if evaluation_seal.get("arrays") != report.get("private_array_sha256"):
        raise EvaluationError("Evaluation output seal does not bind the probability array manifest")


def expected_cells() -> list[dict[str, Any]]:
    return [dict(item=item, view=view, replicate=replicate)
            for item in ext.selection_items() for view, replicate in VIEWS]


def _verify_declaration(execution: Path, data: Path) -> tuple[dict, dict]:
    declaration_path = execution / "predeclared.json"
    seal_path = execution / "models_sealed.json"
    if not declaration_path.is_file() or not seal_path.is_file():
        raise EvaluationError("Predeclared protocol or final model seal is missing")
    declaration = read_json(declaration_path)
    seal = read_json(seal_path)
    rebuild_path = execution / "rebuild_declaration.json"
    if not rebuild_path.is_file():
        raise EvaluationError("Rebuilt-reference declaration is missing")
    rebuild_declaration = read_json(rebuild_path)
    if (rebuild_declaration.get("organizer_labels_read") is not False or
            rebuild_declaration.get("automatic_promotion") is not False or
            declaration.get("evaluation", {}).get("organizer_labels_read") is not False or
            declaration.get("evaluation", {}).get("public_scoring") is not False or
            declaration.get("automatic_promotion") is not False or
            seal.get("organizer_labels_read") is not False or
            seal.get("automatic_promotion") is not False):
        raise EvaluationError("Declarations/seal do not preserve external-only, no-promotion scope")
    declared_cells = declaration.get("evaluation_cells")
    if declared_cells != expected_cells():
        raise EvaluationError("Predeclared evaluation cells differ from the 15 selection cells")
    expected_inputs = {str(ext.eval_path(item, view, replicate).resolve())
        for item in ext.selection_items() for view, replicate in VIEWS}
    expected_inputs.update(str(ext.historical_path(item, view, replicate, seed).resolve())
        for item in ext.selection_items() for view, replicate in VIEWS for seed in SEEDS)
    evaluation_inputs = declaration.get("evaluation_inputs", {})
    if set(evaluation_inputs) != expected_inputs:
        raise EvaluationError("Evaluation input seal does not cover exactly 15 caches and 30 historical arrays")
    _expect_hashes(evaluation_inputs, base=Path("/"),
                   description="evaluation input")
    expected_teachers = {str(ext.control_path(f"leave_out_{provider}", seed).resolve())
                         for provider in ("sportec", "metrica", "skillcorner")
                         for seed in SEEDS}
    expected_teachers.update(str(ext.control_path(f"leave_out_{provider}", seed)
                                  .with_suffix(".json").resolve())
                             for provider in ("sportec", "metrica", "skillcorner")
                             for seed in SEEDS)
    teacher_files = declaration.get("teacher_files", {})
    if set(teacher_files) != expected_teachers:
        raise EvaluationError("Teacher seal must contain six restored checkpoints and six receipts")
    _expect_hashes(teacher_files, base=Path("/"),
                   description="restored teacher")
    source_hashes = declaration.get("source_sha256", {})
    rebuild_sources = rebuild_declaration.get("source_sha256", {})
    if not rebuild_sources:
        raise EvaluationError("Rebuilt-reference declaration lacks source hashes")
    _expect_hashes(rebuild_sources, base=ROOT,
                   description="rebuilt-reference source")
    report_rel = "experiments/results/soccernet_capacity_final_report.json"
    if report_rel not in source_hashes:
        raise EvaluationError("Published width-96 report is not bound by source hashes")
    _expect_hashes(source_hashes, base=ROOT,
                   description="source")
    declaration_sha = sha256(declaration_path)
    rebuild_sha = sha256(rebuild_path)
    if seal.get("declaration_sha256") != declaration_sha:
        raise EvaluationError("Final model seal references a different declaration")
    if seal.get("reference_declaration_sha256") != rebuild_sha:
        raise EvaluationError("Final model seal references a different rebuild declaration")
    if seal.get("source_sha256") != declaration.get("source_sha256"):
        raise EvaluationError("Final model seal source hashes differ from declaration")
    reference_models = declaration.get("reference_models", {})
    reference_receipts = declaration.get("reference_receipts", {})
    expected_reference_models = {
        str((execution / f"models/reference/{fold}/seed{seed}.joblib").resolve())
        for fold in ("leave_out_sportec", "leave_out_metrica", "leave_out_skillcorner")
        for seed in SEEDS
    }
    expected_reference_receipts = {
        str((execution / f"models/reference/{fold}/seed{seed}.receipt.json").resolve())
        for fold in ("leave_out_sportec", "leave_out_metrica", "leave_out_skillcorner")
        for seed in SEEDS
    }
    if (set(reference_models) != expected_reference_models or
            set(reference_receipts) != expected_reference_receipts):
        raise EvaluationError("Predeclared reference hash maps are incomplete")
    if (seal.get("reference_models") != reference_models or
            seal.get("reference_receipts") != reference_receipts):
        raise EvaluationError("Final model seal reference hashes differ from predeclaration")
    _expect_hashes(reference_models, base=Path("/"),
                   description="rebuilt reference checkpoint")
    _expect_hashes(reference_receipts, base=Path("/"),
                   description="rebuilt reference receipt")
    if seal.get("models") is None or seal.get("receipts") is None:
        raise EvaluationError("Final model seal must list models and receipts")
    _expect_hashes(seal["models"], base=execution, description="checkpoint")
    _expect_hashes(seal["receipts"], base=execution, description="checkpoint receipt")
    expected = {f"models/{arm}/leave_out_{provider}/seed{seed}.joblib"
                for arm in ARMS for provider in ("sportec", "metrica", "skillcorner")
                for seed in SEEDS}
    if set(seal["models"]) != expected:
        raise EvaluationError("Final model seal does not contain all 30 expected checkpoints")
    for model_path in expected_reference_models:
        relative = Path(model_path).relative_to(execution).as_posix()
        if seal["models"].get(relative) != reference_models[model_path]:
            raise EvaluationError(f"Reference model map differs between declaration and seal: {relative}")
    expected_receipts = {f"models/{arm}/leave_out_{provider}/seed{seed}.receipt.json"
                         for arm in ARMS for provider in ("sportec", "metrica", "skillcorner")
                         for seed in SEEDS}
    if set(seal["receipts"]) != expected_receipts:
        raise EvaluationError("Final model seal does not contain all 30 checkpoint receipts")
    for receipt_path in expected_reference_receipts:
        relative = Path(receipt_path).relative_to(execution).as_posix()
        if seal["receipts"].get(relative) != reference_receipts[receipt_path]:
            raise EvaluationError(f"Reference receipt map differs between declaration and seal: {relative}")
    for model_rel in expected:
        receipt_rel = model_rel.removesuffix(".joblib") + ".receipt.json"
        receipt = read_json(execution / receipt_rel)
        _verify_checkpoint_receipt(receipt, model_rel, seal["models"][model_rel],
                                   declaration_sha, rebuild_sha)
    expected_draw_keys = {f"{fold}:seed{seed}"
                          for fold in ("leave_out_sportec", "leave_out_metrica", "leave_out_skillcorner")
                          for seed in SEEDS}
    draw_seal = seal.get("supervised_draws_sha256", {})
    if set(draw_seal) != expected_draw_keys:
        raise EvaluationError("Final model seal lacks matched supervised-draw hashes")
    for fold in ("leave_out_sportec", "leave_out_metrica", "leave_out_skillcorner"):
        for seed in SEEDS:
            hashes = set()
            for arm in ARMS[1:]:
                receipt_rel = f"models/{arm}/{fold}/seed{seed}.receipt.json"
                receipt = read_json(execution / receipt_rel)
                value = receipt.get("training", {}).get("supervised_draws_sha256")
                if not isinstance(value, str) or len(value) != 64:
                    raise EvaluationError(f"Continuation receipt lacks supervised draws: {receipt_rel}")
                hashes.add(value)
            draw_key = f"{fold}:seed{seed}"
            if len(hashes) != 1 or next(iter(hashes)) != draw_seal[draw_key]:
                raise EvaluationError(f"Continuation arms do not share the sealed draws: {draw_key}")
    return declaration, seal


def _verify_probability(p: np.ndarray, n_frames: int, name: str) -> np.ndarray:
    p = np.asarray(p)
    if p.shape != (n_frames, 3) or not np.issubdtype(p.dtype, np.number):
        raise EvaluationError(f"{name} must be a numeric ({n_frames}, 3) array")
    if (not np.isfinite(p).all() or (p < 0).any()
            or not np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-5)):
        raise EvaluationError(f"{name} contains invalid probabilities")
    return p


def _aggregate(rows: list[dict], arm: str, view: str, member: str) -> dict:
    chosen = [r for r in rows if (r["arm"], r["view"], r["member"]) ==
              (arm, view, member)]
    providers = {}
    for provider in ("sportec", "metrica", "skillcorner"):
        values = [r for r in chosen if r["source"] == provider]
        if not values:
            raise EvaluationError(f"Missing provider rows for {arm}/{view}/{member}/{provider}")
        summary = {}
        for key in SUMMARY_FIELDS:
            nums = [r[key] for r in values if r.get(key) is not None]
            if nums:
                summary[key] = float(np.mean(nums))
        summary["spell_by_iou"] = {
            key: float(np.mean([r["spell_by_iou"][key] for r in values]))
            for key in IOU_KEYS
        }
        providers[provider] = summary
    mean = {key: float(np.mean([p[key] for p in providers.values()]))
            for key in SUMMARY_FIELDS if all(key in p for p in providers.values())}
    mean["spell_by_iou"] = {
        key: float(np.mean([p["spell_by_iou"][key] for p in providers.values()]))
        for key in IOU_KEYS
    }
    return {"equal_provider_mean": mean, "per_provider": providers,
            "cell_count": len(chosen)}


def _published_comparison(rebuilt: dict, published: dict) -> dict:
    """Numerically compare a reproduction without labeling it an exact restore."""
    comparisons = {}
    all_within = True
    for member in ("2026", "2027", "ensemble"):
        prior = (published.get("seed_summaries", {}).get("w96", {}).get(member)
                 if member != "ensemble" else published.get("summaries", {}).get("w96"))
        if prior is None:
            raise EvaluationError(f"Published capacity report lacks w96/{member} scores")
        comparisons[member] = {}
        for view in ("clean", "feed_like"):
            current = rebuilt[member][view]
            old = prior[view]
            deltas = {}
            checks = {}
            for key in SUMMARY_FIELDS:
                a = current["equal_provider_mean"].get(key)
                b = old["equal_provider_mean"].get(key)
                if (a is None) != (b is None):
                    raise EvaluationError(f"Published summary field mismatch: {member}/{view}/{key}")
                if a is None or b is None:
                    continue
                delta = float(a) - float(b)
                deltas[key] = delta
                checks[key] = abs(delta) <= PUBLISHED_TOLERANCE
            for threshold in IOU_KEYS:
                delta = (current["equal_provider_mean"]["spell_by_iou"][threshold]
                         - old["equal_provider_mean"]["spell_by_iou"][threshold])
                deltas[f"spell_by_iou_{threshold}"] = float(delta)
                checks[f"spell_by_iou_{threshold}"] = abs(delta) <= PUBLISHED_TOLERANCE
            cell_count_delta = int(current["cell_count"]) - int(old["cell_count"])
            deltas["cell_count"] = cell_count_delta
            checks["cell_count"] = cell_count_delta == 0
            for provider in ("sportec", "metrica", "skillcorner"):
                current_provider = current["per_provider"][provider]
                prior_provider = old["per_provider"][provider]
                for key in SUMMARY_FIELDS:
                    a = current_provider.get(key)
                    b = prior_provider.get(key)
                    if (a is None) != (b is None):
                        raise EvaluationError(
                            f"Published provider field mismatch: {member}/{view}/{provider}/{key}")
                    if a is None or b is None:
                        continue
                    delta = float(a) - float(b)
                    name = f"{provider}_{key}"
                    deltas[name] = delta
                    checks[name] = abs(delta) <= PUBLISHED_TOLERANCE
                for threshold in IOU_KEYS:
                    delta = (current_provider["spell_by_iou"][threshold]
                             - prior_provider["spell_by_iou"][threshold])
                    name = f"{provider}_spell_by_iou_{threshold}"
                    deltas[name] = float(delta)
                    checks[name] = abs(delta) <= PUBLISHED_TOLERANCE
            comparisons[member][view] = {
                "published_report_tolerance": PUBLISHED_TOLERANCE,
                "deltas_rebuilt_minus_published": deltas,
                "within_tolerance": checks,
                "all_within_tolerance": all(checks.values()),
            }
            all_within &= all(checks.values())
    return {"published_report": "experiments/results/soccernet_capacity_final_report.json",
            "label": "reproduced reference comparison; not an exact package restore",
            "all_within_tolerance": all_within,
            "members": comparisons}


def _guardrails(summaries: dict) -> dict:
    guards = {}
    reference = summaries["reference"]
    for arm in ARMS[1:]:
        candidate = summaries[arm]
        guards[arm] = {
            "clean_spell_delta_vs_reference": (
                candidate["clean"]["equal_provider_mean"]["spell_f1"] -
                reference["clean"]["equal_provider_mean"]["spell_f1"]),
            "clean_spell_at_least_reference_plus_0_01": (
                candidate["clean"]["equal_provider_mean"]["spell_f1"] >=
                reference["clean"]["equal_provider_mean"]["spell_f1"] + 0.01),
            "feed_like_spell_delta_vs_reference": (
                candidate["feed_like"]["equal_provider_mean"]["spell_f1"] -
                reference["feed_like"]["equal_provider_mean"]["spell_f1"]),
            "feed_like_spell_at_least_reference_minus_0_01": (
                candidate["feed_like"]["equal_provider_mean"]["spell_f1"] >=
                reference["feed_like"]["equal_provider_mean"]["spell_f1"] - 0.01),
            "per_provider_clean_macro_and_d_at_least_reference_minus_0_02": {
                provider: {
                    metric: candidate["clean"]["per_provider"][provider][metric] >=
                    reference["clean"]["per_provider"][provider][metric] - 0.02
                    for metric in ("macro_f1", "d_f1")
                }
                for provider in ("sportec", "metrica", "skillcorner")
            },
        }
    return guards


def _rank_arms(seed_summaries: dict) -> dict:
    rankings = {}
    for member in MEMBERS:
        rankings[member] = {}
        for view in ("clean", "feed_like"):
            ordered = sorted(ARMS, key=lambda arm: (
                seed_summaries[arm][member][view]["equal_provider_mean"]["spell_f1"],
                seed_summaries[arm][member][view]["equal_provider_mean"]["macro_f1"]),
                reverse=True)
            rankings[member][view] = [
                {"rank": rank, "arm": arm,
                 "spell_f1": seed_summaries[arm][member][view]["equal_provider_mean"]["spell_f1"],
                 "macro_f1": seed_summaries[arm][member][view]["equal_provider_mean"]["macro_f1"]}
                for rank, arm in enumerate(ordered, 1)]
    return rankings


def _write_json_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _verify_historical_teacher(raw: np.ndarray, record: dict, inputs: dict, item: dict,
                               view: str, replicate: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    archive_path = ext.historical_path(item, view, replicate, seed)
    with np.load(archive_path, allow_pickle=False) as saved:
        if not {"probability", "prediction"}.issubset(saved.files):
            raise EvaluationError(f"Historical teacher archive is incomplete: {archive_path}")
        prior_probability = saved["probability"]
        prior_prediction = saved["prediction"]
    native = ext.upsample(raw, record, inputs)
    if not np.allclose(native, prior_probability, atol=1e-6, rtol=0):
        raise EvaluationError(f"Restored teacher probability parity failed: {archive_path.name}")
    decoded = supervised.decode(native, supervised.model_periods(inputs),
                                float(inputs["fps"]), cap.lt.COST)
    if not np.array_equal(decoded, prior_prediction):
        raise EvaluationError(f"Restored teacher decoded parity failed: {archive_path.name}")
    return native, decoded


def _verify_all_teacher_parity(declaration: dict) -> list[dict]:
    """Finish every historical teacher parity check before candidate scoring."""
    results = []
    for cell in declaration["evaluation_cells"]:
        item, view, replicate = cell["item"], cell["view"], cell["replicate"]
        _, inputs, _, record = cap._cache_cell(item, view, replicate)
        fold = f"leave_out_{item['source']}"
        for seed in SEEDS:
            teacher = joblib.load(ext.control_path(fold, seed))
            raw = fh.predict(record, teacher)
            native, decoded = _verify_historical_teacher(
                raw, record, inputs, item, view, replicate, seed)
            archive_path = ext.historical_path(item, view, replicate, seed)
            with np.load(archive_path, allow_pickle=False) as saved:
                maximum_probability_delta = float(np.max(np.abs(native - saved["probability"])))
                exact_decoded = bool(np.array_equal(decoded, saved["prediction"]))
            results.append({"source": item["source"], "match": item["match"],
                "view": view, "replicate": replicate, "seed": seed,
                "archive": str(archive_path), "probability_atol": 1e-6,
                "maximum_absolute_probability_delta": maximum_probability_delta,
                "decoded_prediction_exact": exact_decoded})
    if len(results) != 30 or not all(r["decoded_prediction_exact"] for r in results):
        raise EvaluationError("Original teacher parity is incomplete")
    return results


def run(execution_dir: Path, data_dir: Path) -> dict:
    execution = Path(execution_dir).resolve()
    data = Path(data_dir).resolve()
    # Existing experiment helpers use module-level roots; bind them to the
    # explicitly supplied private cache for this invocation.
    cap.DATA = data
    ext.DATA = data
    declaration, seal = _verify_declaration(execution, data)
    if not data.is_dir():
        raise EvaluationError(f"Private data directory is missing: {data}")
    teacher_parity = _verify_all_teacher_parity(declaration)
    output = execution / "evaluation"
    if output.exists():
        raise EvaluationError(f"Evaluation output already exists; refusing overwrite: {output}")
    # Load and validate every checkpoint before the first inference/scoring call.
    models = {}
    for arm in ARMS:
        models[arm] = {}
        for provider in ("sportec", "metrica", "skillcorner"):
            fold = f"leave_out_{provider}"
            models[arm][fold] = {}
            for seed in SEEDS:
                path = execution / f"models/{arm}/{fold}/seed{seed}.joblib"
                models[arm][fold][seed] = joblib.load(path)
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    (output / "arrays").mkdir()
    cell_keys = set()
    for cell in declaration["evaluation_cells"]:
        item, view, replicate = cell["item"], cell["view"], cell["replicate"]
        cell_id = f"{item['match']}_{view}_{replicate}"
        key = (item["source"], item["match"], view, replicate)
        if key in cell_keys:
            raise EvaluationError(f"Duplicate declared cell: {key}")
        cell_keys.add(key)
        cache_path = ext.eval_path(item, view, replicate)
        cached = joblib.load(cache_path)
        inputs, labels = cached["inputs"], np.asarray(cached["labels"])
        record = ext.inference_record(cached["record"])
        n = len(labels)
        if n != len(inputs["period"]) or n != len(inputs["timestamps"]):
            raise EvaluationError(f"Cached native arrays are misaligned: {cache_path}")
        fold = f"leave_out_{item['source']}"
        originals = {}
        teacher_native = {}
        for seed in SEEDS:
            teacher_path = ext.control_path(fold, seed)
            teacher = joblib.load(teacher_path)
            raw = fh.predict(record, teacher)
            teacher_native[seed], _ = _verify_historical_teacher(
                raw, record, inputs, item, view, replicate, seed)
            originals[seed] = raw
        probabilities: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in ARMS}
        predictions: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in ARMS}
        for arm in ARMS:
            for seed in SEEDS:
                sampled = cap_model.predict(record, models[arm][fold][seed],
                    joblib.load(ext.control_path(fold, seed)),
                    original_probability=originals[seed])
                native = ext.upsample(sampled, record, inputs)
                if not np.array_equal(native[:, 2], teacher_native[seed][:, 2]):
                    raise EvaluationError(f"Native P(D) changed for {arm}/{seed}/{cell_id}")
                native = _verify_probability(native, n, f"{arm}/{seed}/{cell_id}")
                decoded = supervised.decode(native, supervised.model_periods(inputs),
                                             float(inputs["fps"]), cap.lt.COST)
                probabilities[arm][str(seed)] = native
                predictions[arm][str(seed)] = decoded
            # Match the historical capacity runner exactly, including NumPy's
            # native dtype behavior before the equal-probability blend.
            probabilities[arm]["ensemble"] = (
                probabilities[arm]["2026"] + probabilities[arm]["2027"]) * 0.5
            predictions[arm]["ensemble"] = supervised.decode(
                probabilities[arm]["ensemble"], supervised.model_periods(inputs),
                float(inputs["fps"]), cap.lt.COST)
        cell_rows = []
        for arm in ARMS:
            for member in MEMBERS:
                probability = probabilities[arm][member]
                prediction = predictions[arm][member]
                row = cap._score_probability(probability, inputs, labels, item,
                                             view, replicate, arm, member)[0]
                row["source"] = item["source"]
                row["arm"] = arm
                row["member"] = member
                cell_rows.append(row)
                rows.append(row)
                array_path = output / "arrays" / f"{cell_id}_{arm}_{member}.npz"
                with array_path.open("xb") as stream:
                    np.savez_compressed(stream, probability=probability,
                                        prediction=prediction)
        if len({r["n_gt_spells"] for r in cell_rows}) != 1 or \
                len({r["graded_frames"] for r in cell_rows}) != 1:
            raise EvaluationError(f"Models changed grading population for {cell_id}")
    expected_keys = {(i["source"], i["match"], v, r, arm, member)
                     for i in ext.selection_items() for v, r in VIEWS
                     for arm in ARMS for member in MEMBERS}
    actual_keys = {(r["source"], r["match"], r["view"], r["replicate"],
                    r["arm"], r["member"]) for r in rows}
    if actual_keys != expected_keys or len(rows) != len(expected_keys):
        raise EvaluationError("Scored rows do not exactly cover the 15 cells × 5 arms × 3 members")
    summaries = {arm: {view: _aggregate(rows, arm, view, "ensemble")
                       for view in ("clean", "feed_like")} for arm in ARMS}
    seed_summaries = {arm: {member: {view: _aggregate(rows, arm, view, member)
                                     for view in ("clean", "feed_like")}
                            for member in MEMBERS} for arm in ARMS}
    published = read_json(ROOT / "experiments/results/soccernet_capacity_final_report.json")
    reference_rebuild = {
        member: {view: seed_summaries["reference"][member][view]
                 for view in ("clean", "feed_like")}
        for member in MEMBERS
    }
    comparison = _published_comparison(reference_rebuild, published)
    guards = _guardrails(summaries)
    private_rows_path = output / "scored_rows.json"
    private_rows_path.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    array_hashes = {str(path.relative_to(execution)): sha256(path)
                    for path in sorted((output / "arrays").glob("*.npz"))}
    report = {
        "study_id": "soccernet_followup_execution_v1",
        "declaration_sha256": sha256(execution / "predeclared.json"),
        "model_seal_sha256": sha256(execution / "models_sealed.json"),
        "evaluation_cells": 15,
        "summaries": summaries,
        "seed_summaries": seed_summaries,
        "descriptive_ranking_spell_then_macro": _rank_arms(seed_summaries),
        "reproduced_reference_vs_published_capacity": comparison,
        "guardrails_vs_rebuilt_reference": guards,
        "original_teacher_historical_parity": teacher_parity,
        "private_rows_sha256": sha256(private_rows_path),
        "private_array_sha256": array_hashes,
        "organizer_labels_read": False,
        "limitations": [
            "Previously exposed five-match development cohort; feed_like is synthetic.",
            "The rebuilt width-96 reference is compared numerically with the published report; it is not called an exact restore.",
            "Probability and decoded-prediction parity for each original teacher was checked against objective_alignment archives before candidate scoring.",
        ],
    }
    report_path = output / "aggregate.json"
    _write_json_once(report_path, report)
    evaluation_seal = {
        "declaration_sha256": report["declaration_sha256"],
        "model_seal_sha256": report["model_seal_sha256"],
        "reference_declaration_sha256": sha256(execution / "rebuild_declaration.json"),
        "aggregate_sha256": sha256(report_path),
        "scored_rows_sha256": report["private_rows_sha256"],
        "arrays": report["private_array_sha256"],
    }
    _write_json_once(output / "evaluation_sealed.json", evaluation_seal)
    return report


def verify_saved(execution_dir: Path, data_dir: Path) -> dict:
    """Replay saved native arrays through the scorer and verify every row."""
    execution = Path(execution_dir).resolve()
    data = Path(data_dir).resolve()
    cap.DATA = data
    ext.DATA = data
    declaration, _ = _verify_declaration(execution, data)
    output = execution / "evaluation"
    report_path = output / "aggregate.json"
    rows_path = output / "scored_rows.json"
    seal_path = output / "evaluation_sealed.json"
    if not report_path.is_file() or not rows_path.is_file() or not seal_path.is_file():
        raise EvaluationError("Saved aggregate, private rows, or evaluation seal is missing")
    report = read_json(report_path)
    saved_rows = read_json(rows_path)
    evaluation_seal = read_json(seal_path)
    _verify_output_seal(report, evaluation_seal, execution, report_path, rows_path)
    if (evaluation_seal.get("declaration_sha256") != report["declaration_sha256"] or
            evaluation_seal.get("model_seal_sha256") != report["model_seal_sha256"] or
            evaluation_seal.get("reference_declaration_sha256") !=
            sha256(execution / "rebuild_declaration.json") or
            evaluation_seal.get("aggregate_sha256") != sha256(report_path)):
        raise EvaluationError("Evaluation output seal does not bind the current report and declarations")
    teacher_parity = report.get("original_teacher_historical_parity", [])
    expected_teacher_keys = {
        (cell["item"]["source"], cell["item"]["match"], cell["view"],
         cell["replicate"], seed)
        for cell in declaration["evaluation_cells"] for seed in SEEDS
    }
    actual_teacher_keys = {
        (r.get("source"), r.get("match"), r.get("view"), r.get("replicate"), r.get("seed"))
        for r in teacher_parity
    }
    if (len(teacher_parity) != 30 or actual_teacher_keys != expected_teacher_keys or
            any(not r.get("decoded_prediction_exact") or
                r.get("maximum_absolute_probability_delta", float("inf")) > 1e-6
                for r in teacher_parity)):
        raise EvaluationError("Aggregate lacks complete original teacher parity evidence")
    recomputed_teacher_parity = _verify_all_teacher_parity(declaration)
    if recomputed_teacher_parity != teacher_parity:
        raise EvaluationError("Saved original-teacher parity receipt does not replay")
    array_hashes = report.get("private_array_sha256", {})
    expected_arrays = {f"evaluation/arrays/{item['match']}_{view}_{replicate}_{arm}_{member}.npz"
                       for item in ext.selection_items() for view, replicate in VIEWS
                       for arm in ARMS for member in MEMBERS}
    if set(array_hashes) != expected_arrays:
        raise EvaluationError("Saved array manifest does not cover the exact expected cohort")
    actual_arrays = {str(p.relative_to(execution)) for p in (output / "arrays").glob("*.npz")}
    if actual_arrays != expected_arrays:
        raise EvaluationError("Evaluation output contains missing or unsealed probability arrays")
    _expect_hashes(array_hashes, base=execution, description="saved probability array")
    expected_rows = {(item["source"], item["match"], view, replicate, arm, member)
                     for item in ext.selection_items() for view, replicate in VIEWS
                     for arm in ARMS for member in MEMBERS}
    indexed = {(r["source"], r["match"], r["view"], r["replicate"],
                r["arm"], r["member"]): r for r in saved_rows}
    if len(indexed) != len(saved_rows) or set(indexed) != expected_rows:
        raise EvaluationError("Saved private rows are incomplete or duplicated")
    replayed = []
    replay_probabilities = {}
    for cell in declaration["evaluation_cells"]:
        item, view, replicate = cell["item"], cell["view"], cell["replicate"]
        _, inputs, labels, cached_record = cap._cache_cell(item, view, replicate)
        record = ext.inference_record(cached_record)
        teacher_native = {}
        fold = f"leave_out_{item['source']}"
        for seed in SEEDS:
            teacher = joblib.load(ext.control_path(fold, seed))
            teacher_native[seed] = ext.upsample(fh.predict(record, teacher), record, inputs)
        cell_id = f"{item['match']}_{view}_{replicate}"
        for arm in ARMS:
            for member in MEMBERS:
                array_rel = f"evaluation/arrays/{cell_id}_{arm}_{member}.npz"
                with np.load(execution / array_rel, allow_pickle=False) as archive:
                    probability = _verify_probability(archive["probability"], len(labels), array_rel)
                    prediction = archive["prediction"]
                replay_probabilities[(item["match"], view, replicate, arm, member)] = probability
                if member in {str(seed) for seed in SEEDS}:
                    seed = int(member)
                    if not np.array_equal(probability[:, 2], teacher_native[seed][:, 2]):
                        raise EvaluationError(f"Saved native P(D) differs from teacher: {array_rel}")
                decoded = supervised.decode(probability, supervised.model_periods(inputs),
                                            float(inputs["fps"]), cap.lt.COST)
                if not np.array_equal(decoded, prediction):
                    raise EvaluationError(f"Saved prediction does not decode from probability: {array_rel}")
                actual = cap._score_probability(probability, inputs, labels, item,
                    view, replicate, arm, member)[0]
                actual["source"] = item["source"]
                actual["arm"] = arm
                actual["member"] = member
                key = (item["source"], item["match"], view, replicate, arm, member)
                if actual != indexed[key]:
                    raise EvaluationError(f"Scored row replay differs for {key}")
                replayed.append(actual)
        for arm in ARMS:
            expected_ensemble = (replay_probabilities[(item["match"], view, replicate, arm, "2026")] +
                                 replay_probabilities[(item["match"], view, replicate, arm, "2027")]) * 0.5
            stored_ensemble = replay_probabilities[(item["match"], view, replicate, arm, "ensemble")]
            if not np.array_equal(expected_ensemble, stored_ensemble):
                raise EvaluationError(f"Saved ensemble differs from native seed mean: {arm}/{item['match']}")
    if report.get("evaluation_cells") != 15:
        raise EvaluationError("Aggregate reports the wrong number of evaluation cells")
    rebuilt_seed_summaries = {arm: {member: {view: _aggregate(replayed, arm, view, member)
                                            for view in ("clean", "feed_like")}
                                     for member in MEMBERS} for arm in ARMS}
    for arm in ARMS:
        for member in MEMBERS:
            for view in ("clean", "feed_like"):
                actual = rebuilt_seed_summaries[arm][member][view]
                if actual != report.get("seed_summaries", {}).get(arm, {}).get(member, {}).get(view):
                    raise EvaluationError(f"Aggregate replay differs for {arm}/{member}/{view}")
    rebuilt_summaries = {arm: {view: _aggregate(replayed, arm, view, "ensemble")
                               for view in ("clean", "feed_like")} for arm in ARMS}
    if report.get("summaries") != rebuilt_summaries:
        raise EvaluationError("Aggregate summary does not replay from saved rows")
    if report.get("descriptive_ranking_spell_then_macro") != _rank_arms(rebuilt_seed_summaries):
        raise EvaluationError("Descriptive arm ranking does not replay from saved scores")
    if report.get("guardrails_vs_rebuilt_reference") != _guardrails(rebuilt_summaries):
        raise EvaluationError("Reported guardrails do not replay from saved aggregate")
    published = read_json(ROOT / "experiments/results/soccernet_capacity_final_report.json")
    rebuilt_reference = {member: rebuilt_seed_summaries["reference"][member]
                         for member in MEMBERS}
    if report.get("reproduced_reference_vs_published_capacity") != \
            _published_comparison(rebuilt_reference, published):
        raise EvaluationError("Published-reference comparison does not replay from saved aggregate")
    return {"verified_rows": len(replayed), "verified_arrays": len(array_hashes),
            "report": str(report_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("evaluate", "verify"))
    parser.add_argument("--execution-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(os.environ.get("MTAH_DATA_DIR", "/workspace/mtah-data")))
    args = parser.parse_args()
    if args.action == "evaluate":
        report = run(args.execution_dir, args.data_dir)
        result = {"report": str(args.execution_dir / "evaluation/aggregate.json"),
                  "reference_parity": report["reproduced_reference_vs_published_capacity"]["all_within_tolerance"]}
    else:
        result = verify_saved(args.execution_dir, args.data_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
