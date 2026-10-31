"""One bounded confidence-dependent boundary decoder for dense ensemble outputs.

The candidate retains cost-3 Potts smoothing except where adjacent-frame
probabilities support a particular state transition. Its edge-specific penalty
is 3 * (1 - sqrt(p_left[i] * p_right[j])). No parameters are fitted.
"""
from __future__ import annotations

import numpy as np
import argparse
import json
from pathlib import Path
import hashlib

from experiments import supervised
from experiments.metrics import evaluate_tracking, prepare_tracking_codes
from evaluation import possession_metrics as pm

STUDY_ID = "dense_boundary_v1"
SWITCH_COST = 3.0


def decode(probabilities: np.ndarray, period: np.ndarray, fps: float,
           switch_cost: float = SWITCH_COST) -> np.ndarray:
    """Decode H/A/D, applying transition-evidence costs within each period."""
    p = np.asarray(probabilities, dtype=np.float64)
    period = np.asarray(period)
    if p.ndim != 2 or p.shape != (len(period), 3):
        raise ValueError("expected one H/A/D probability row per frame")
    if not np.isfinite(p).all() or (p < 0).any() or (p.sum(axis=1) <= 0).any():
        raise ValueError("probabilities must be finite, nonnegative, and nonempty")
    if not np.isfinite(fps) or fps <= 0 or switch_cost < 0:
        raise ValueError("fps must be positive and switch cost nonnegative")
    p = p / p.sum(axis=1, keepdims=True)
    logp = np.log(np.maximum(p, 1e-7))
    result = np.empty(len(period), dtype=np.int8)
    scale = float(fps) / 10.0
    for start, stop in supervised.blocks(period):
        n = stop - start
        back = np.zeros((n, 3), dtype=np.int8)
        value = logp[start].copy()
        for row in range(start + 1, stop):
            # Evidence is directional: probability of the old state before
            # the edge and the new state after it. Diagonal transitions remain free.
            evidence = np.sqrt(p[row - 1, :, None] * p[row, None, :])
            penalty = switch_cost * scale * (1.0 - evidence)
            np.fill_diagonal(penalty, 0.0)
            transition = value[:, None] - penalty
            previous = np.argmax(transition, axis=0)
            back[row - start] = previous
            value = transition[previous, np.arange(3)] + logp[row]
        state = int(np.argmax(value))
        for row in range(stop - 1, start - 1, -1):
            result[row] = state
            state = int(back[row - start, state])
    return supervised.LABELS[result]


def baseline_decode(probabilities: np.ndarray, period: np.ndarray, fps: float) -> np.ndarray:
    """Exact comparator path through the frozen cost-3 decoder."""
    return supervised.decode(probabilities, period, fps, SWITCH_COST)


def score_record(record: dict) -> dict:
    """Score paired decoders from a frozen record of probabilities and labels."""
    p = np.asarray(record["probabilities"], dtype=np.float64)
    truth = np.asarray(record["labels"]).astype(str)
    period = np.asarray(record["period"])
    fps = float(np.asarray(record["fps"]).item())
    if p.shape != (len(truth), 3) or period.shape != truth.shape:
        raise ValueError("record shape mismatch")
    timestamps = np.asarray(record["timestamps"]) if "timestamps" in record else None
    if timestamps is not None:
        model_period = supervised.model_periods({"period": period, "timestamps": timestamps, "fps": fps})
    else:
        model_period = supervised.model_periods({"period": period, "fps": fps})
    if "model_period" in record and not np.array_equal(np.asarray(record["model_period"]), model_period):
        raise ValueError("saved model_period does not match recomputed breaks from original period/timestamps")
    baseline = baseline_decode(p, model_period, fps).astype("U1")
    candidate = decode(p, model_period, fps).astype("U1")
    invalid = period <= 0
    if "valid_mask" in record:
        valid_mask = np.asarray(record["valid_mask"], dtype=bool)
        if valid_mask.shape != truth.shape:
            raise ValueError("valid_mask must have one value per frame")
        invalid |= ~valid_mask
    baseline[invalid] = "X"
    candidate[invalid] = "X"
    common = dict(period=period, fps=fps, timestamps=timestamps)
    results = {}
    for name, prediction in (("cost3", baseline), ("confidence_boundary", candidate)):
        metrics = evaluate_tracking(truth, prediction, **common)
        metrics["d_precision"] = metrics["per_class"]["D"]["precision"]
        metrics["d_recall"] = metrics["per_class"]["D"]["recall"]
        metrics["d_f1"] = metrics["per_class"]["D"]["f1"]
        metrics["spell_f1"] = metrics["spell_f1_mean"]
        _, gt_codes, pred_codes, inserts = prepare_tracking_codes(
            truth, prediction, period=period, fps=fps, timestamps=timestamps)
        valid_runs = pm.valid_runs(gt_codes)
        gt_spells = pm.possession_spells(gt_codes, valid_runs)
        pred_spells = pm.possession_spells(pred_codes, valid_runs)
        matched = pm.matched_gt_flags(pm.spell_overlaps(gt_spells, pred_spells), len(gt_spells), .5)
        short = np.asarray([(b-a) / fps < 3 for a, b, _ in gt_spells])
        metrics["short_spell_recall_iou_0_5_lt3s"] = (
            float(np.mean(matched[short])) if short.any() else None)
        metrics["short_reference_spells_lt3s"] = int(short.sum())
        metrics["predicted_spell_count"] = len(pred_spells)
        results[name] = metrics
    return {"provider": str(record.get("provider", "unknown")),
            "match": str(record.get("match", "unknown")),
            "view": str(record.get("view", "unknown")),
            "realization": str(record.get("realization", "ensemble")),
            "n_frames": len(truth), "scores": results,
            "candidate_minus_cost3": {
                key: results["confidence_boundary"][key] - results["cost3"][key]
                for key in ("spell_f1", "macro_f1", "short_spell_recall_iou_0_5_lt3s",
                            "d_precision", "d_recall", "d_f1")
                if isinstance(results["confidence_boundary"].get(key), (int, float))
                and isinstance(results["cost3"].get(key), (int, float))}}


def _canonical_record_metadata(record: dict) -> tuple[str, str, str, str]:
    provider_aliases = {"metrica": "Metrica", "sportec": "Sportec", "skillcorner": "SkillCorner"}
    provider_raw = str(record.get("provider", record.get("source", ""))).strip().lower()
    provider = provider_aliases.get(provider_raw)
    if not provider:
        raise ValueError(f"unknown provider tag {provider_raw!r}")
    match_raw = str(record.get("match", "")).strip().lower().replace("-", "_")
    match_aliases = {"metrica_3": "metrica_3", "j03woy": "J03WOY",
                     "1996435": "1996435", "1996436": "1996436", "2006229": "2006229"}
    match = match_aliases.get(match_raw)
    if not match:
        raise ValueError(f"unknown held-out match tag {match_raw!r}")
    if ((provider == "Metrica" and match != "metrica_3")
            or (provider == "Sportec" and match != "J03WOY")
            or (provider == "SkillCorner" and match not in {"1996435", "1996436", "2006229"})):
        raise ValueError(f"provider/match mismatch: {provider}/{match}")
    view_raw = str(record.get("view", "")).strip().lower()
    view_aliases = {"clean": "clean", "feed_like": "feed_like_corrupt_1",
                    "feed_like_corrupt_1": "feed_like_corrupt_1",
                    "feed_like_corrupt_2": "feed_like_corrupt_2"}
    view = view_aliases.get(view_raw)
    if not view:
        raise ValueError(f"unknown view tag {view_raw!r}")
    realization = str(record.get("realization", record.get("seed", "ensemble"))).strip()
    if not realization:
        raise ValueError("realization tag must be nonempty")
    return provider, match, view, realization


def _equal_hierarchy(rows: list[dict]) -> dict:
    """Mean realizations, then matches within provider, then providers equally."""
    metric_names = ("spell_f1", "macro_f1", "short_spell_recall_iou_0_5_lt3s",
                    "d_precision", "d_recall", "d_f1")
    result = {}
    for view in ("clean", "feed_like_corrupt_1", "feed_like_corrupt_2"):
        for arm in ("cost3", "confidence_boundary"):
            provider_values = {}
            for provider in ("Metrica", "Sportec", "SkillCorner"):
                match_values = []
                matches = sorted({r["match"] for r in rows if r["provider"] == provider and r["view"] == view})
                for match in matches:
                    cells = [r for r in rows if (r["provider"], r["match"], r["view"]) == (provider, match, view)]
                    match_values.append({metric: float(np.mean([c["scores"][arm][metric]
                        for c in cells if isinstance(c["scores"][arm].get(metric), (int, float))]))
                        for metric in metric_names})
                provider_values[provider] = {metric: float(np.mean([m[metric] for m in match_values]))
                    for metric in metric_names} if match_values else None
            result.setdefault(view + "_by_provider", {})[arm] = provider_values
            result.setdefault(view + "_arms", {})[arm] = {
                metric: float(np.mean([provider_values[p][metric] for p in provider_values
                    if provider_values[p] is not None])) for metric in metric_names}
    for view in ("clean", "feed_like_corrupt_1", "feed_like_corrupt_2"):
        result[view] = result[view + "_arms"]["confidence_boundary"]
        result[view + "_candidate_minus_cost3"] = {
            metric: result[view + "_arms"]["confidence_boundary"][metric]
                    - result[view + "_arms"]["cost3"][metric]
            for metric in metric_names}
    feed = {metric: float(np.mean([result[f"feed_like_corrupt_{i}"][metric] for i in (1, 2)]))
            for metric in metric_names}
    result["feed_like_mean"] = feed
    result["feed_like_cost3_mean"] = {metric: float(np.mean([
        result[f"feed_like_corrupt_{i}_arms"]["cost3"][metric] for i in (1, 2)])) for metric in metric_names}
    result["feed_like_candidate_minus_cost3"] = {metric: feed[metric] - result["feed_like_cost3_mean"][metric]
                                                   for metric in metric_names}
    clean_delta = result["clean_candidate_minus_cost3"]
    result["declared_guardrails"] = {
        "positive_clean_spell": clean_delta["spell_f1"] > 0,
        "feed_like_noninferior_minus_0_01": result["feed_like_candidate_minus_cost3"]["spell_f1"] >= -0.01,
        "short_spell_recall_nondecreasing_clean": clean_delta["short_spell_recall_iou_0_5_lt3s"] >= 0,
        "current_policy_plus_0_01_clean_screen": clean_delta["spell_f1"] >= 0.01,
        "provider_clean_d_f1_within_0_02": all(
            result["clean_by_provider"]["confidence_boundary"][p]["d_f1"]
            - result["clean_by_provider"]["cost3"][p]["d_f1"] >= -0.02
            for p in ("Metrica", "Sportec", "SkillCorner")),
        "promotion_authorized": False}
    return result


def run(input_dir: Path, output: Path, provenance_path: Path | None = None) -> None:
    """Score declared NPZ records; no input file may contain organizer labels."""
    files = sorted(input_dir.rglob("*.npz"))
    if not files:
        raise ValueError(f"no frozen prediction records found in {input_dir}")
    records = []
    for path in files:
        with np.load(path, allow_pickle=False) as src:
            record = {k: src[k] for k in src.files}
        if str(record.get("arm", "")) != "control_D2":
            continue
        if str(record.get("seed", "")) != "seedensemble":
            continue
        tags = _canonical_record_metadata(record)
        expected_fold = {"Metrica": "leave_out_metrica", "Sportec": "leave_out_sportec",
                         "SkillCorner": "leave_out_skillcorner"}[tags[0]]
        if str(record.get("fold", "")) != expected_fold:
            raise ValueError(f"prediction fold mismatch in {path.name}: expected {expected_fold}")
        records.append((path, record, tags))
    expected = {(p, m, v) for p, m in (("Metrica", "metrica_3"), ("Sportec", "J03WOY"),
                ("SkillCorner", "1996435"), ("SkillCorner", "1996436"), ("SkillCorner", "2006229"))
                for v in ("clean", "feed_like_corrupt_1", "feed_like_corrupt_2")}
    actual = {(provider, match, view) for _, _, (provider, match, view, _) in records}
    full_expected = expected
    triples = [(provider, match, view) for _, _, (provider, match, view, _) in records]
    realization_keys = [(provider, match, view, realization)
                        for _, _, (provider, match, view, realization) in records]
    if actual != full_expected or len(realization_keys) != len(set(realization_keys)):
        raise ValueError(f"incomplete, duplicate realization, or unexpected paired cohort: missing={sorted(full_expected-actual)}, extra={sorted(actual-full_expected)}")
    if provenance_path is None or not provenance_path.is_file():
        raise ValueError("a source provenance JSON is required before scoring")
    provenance = json.loads(provenance_path.read_text())
    required_provenance = ("source_study_id", "reconstruction_status", "model_checkpoint_hashes",
                           "evaluation_input_hashes", "source_arm", "fold_mapping")
    missing_provenance = [key for key in required_provenance if not provenance.get(key)]
    if missing_provenance:
        raise ValueError(f"provenance missing required reconstruction bindings: {missing_provenance}")
    if provenance["source_arm"] != "control_D2" or provenance.get("inference_seed") != "seedensemble":
        raise ValueError("provenance must bind the declared control_D2/seedensemble source")
    for field in ("model_checkpoint_hashes", "evaluation_input_hashes"):
        values = provenance[field]
        if not isinstance(values, dict) or not values:
            raise ValueError(f"provenance {field} must be a nonempty path-to-SHA256 mapping")
        if any(not isinstance(digest, str) or len(digest) != 64 or
               any(ch not in "0123456789abcdef" for ch in digest.lower()) for digest in values.values()):
            raise ValueError(f"provenance {field} contains an invalid SHA256")
    rows = []
    for path, record, (provider, match, view, realization) in records:
        row = score_record(record)
        row.update(provider=provider, match=match, view=view, realization=realization,
                   input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        rows.append(row)
    results_dir = Path(__file__).resolve().parents[1] / "experiments/results"
    report = {"study_id": STUDY_ID, "declaration_sha256": hashlib.sha256(
        (results_dir / "dense_boundary_v1_declaration.json").read_bytes()).hexdigest(),
        "prerun_addendum_sha256": hashlib.sha256(
        (results_dir / "dense_boundary_v1_prerun_addendum.json").read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "official_evaluator_sha256": hashlib.sha256(
        (Path(__file__).resolve().parents[1] / "evaluation/possession_metrics.py").read_bytes()).hexdigest(),
        "scoring": "official evaluation/possession_metrics.py via experiments.metrics.evaluate_tracking",
        "prediction_source": provenance,
        "provenance_sha256": hashlib.sha256(provenance_path.read_bytes()).hexdigest(),
        "aggregation": "mean realizations within match, mean matches within provider, mean providers equally; feed_like is mean of two declared corruptions",
        "summary": _equal_hierarchy(rows), "rows": rows}
    if output.exists():
        raise ValueError(f"refusing to overwrite prior report {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True,
                        help="JSON receipt identifying exact/reconstructed prediction source and hashes")
    args = parser.parse_args()
    run(args.input_dir, args.output, args.provenance)


if __name__ == "__main__":
    main()
