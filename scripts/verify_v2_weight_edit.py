#!/usr/bin/env python3
"""Independently replay saved v2 weight-edit cell metrics and cohort aggregates.

This verifier reads only already-scored NPZ/JSON artifacts. It never loads model
inputs or checkpoints, and never opens the organizer's public reference. It replays
GAR scalar scores from its saved labels/predictions and checks development decoder
outputs from saved probabilities, without model inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import joblib

from evaluation import possession_metrics as pm
from experiments import calibrated_dense_baseline
from experiments import feed_spell_push
from experiments import metrics as tracking_metrics

ROOT = Path(__file__).resolve().parents[1]
STUDY = "v2_weight_edit_development_v1"
RESULTS = ROOT / "experiments/results"
OUT = ROOT / "data/experiments" / STUDY
PROVIDERS = ("Sportec", "Metrica", "SkillCorner")
MATCHES = (("Sportec", "J03WOY"), ("Metrica", "metrica_3"),
           ("SkillCorner", "1996435"), ("SkillCorner", "1996436"),
           ("SkillCorner", "2006229"))
VIEWS = {"clean": (0,), "feed_like_v2": (1, 2, 3, 4)}
METRICS = ("spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
           "predicted_spell_count", "reference_spell_count",
           "short_spell_recall_iou_0_5_lt3s")
GAR_METRICS = (*METRICS, "graded_frames")
GAR_VIEWS = ("clean", "v2_1", "v2_2", "v2_3", "v2_4")
EPS = 1e-10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def score_saved(labels, pred, period, timestamps, fps: float) -> dict:
    """Official scalar definitions, reconstructed from frozen labels/predictions."""
    labels = np.asarray(labels).astype(str)
    pred = np.asarray(pred).astype(str)
    if labels.ndim != 1 or pred.shape != labels.shape or not len(labels):
        raise ValueError("saved labels/predictions must be equal nonempty vectors")
    if not np.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("invalid saved fps")
    _, gt, guess, _ = tracking_metrics.prepare_tracking_codes(
        labels, pred, period=period, fps=float(fps), timestamps=timestamps)
    runs = pm.valid_runs(gt)
    result = pm.evaluate(gt, guess, taus=(max(1, round(.5 * fps)), max(1, round(fps))),
                         bootstrap_n=0)
    gt_spells = pm.possession_spells(gt, runs)
    pred_spells = pm.possession_spells(guess, runs)
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    short = np.asarray([(b-a) / float(fps) < 3.0 for a, b, _ in gt_spells], dtype=bool)
    matched = pm.matched_gt_flags(overlaps, len(gt_spells), .5)
    d = result["per_class"]["D"]
    return {
        "spell_f1": float(result["spell_f1_mean"]),
        "macro_f1": float(result["macro_f1"]),
        "d_f1": float(d["f1"]), "d_precision": float(d["precision"]),
        "d_recall": float(d["recall"]),
        "predicted_spell_count": int(len(pred_spells)),
        "reference_spell_count": int(len(gt_spells)),
        "short_spell_recall_iou_0_5_lt3s": float(matched[short].mean()) if short.any() else None,
    }


def model_period_ids(period, timestamps, fps: float) -> np.ndarray:
    """Mirror `supervised.model_periods` using just the sealed cell time columns."""
    period = np.asarray(period)
    timestamps = np.asarray(timestamps, dtype=float)
    breaks = np.diff(period) != 0
    delta = np.diff(timestamps)
    breaks |= (delta <= 0) | (delta > 1.5 / float(fps))
    return np.r_[0, np.cumsum(breaks)]


def equal_provider(values: dict[tuple[str, str, str], float]) -> tuple[float, dict[str, float]]:
    by_provider: dict[str, dict[str, list[float]]] = {}
    for (provider, match, _), value in values.items():
        by_provider.setdefault(provider, {}).setdefault(match, []).append(float(value))
    per_provider = {
        provider: float(np.mean([np.mean(reps) for reps in matches.values()]))
        for provider, matches in by_provider.items()
    }
    if set(per_provider) != set(PROVIDERS):
        raise AssertionError(f"provider coverage mismatch: {sorted(per_provider)}")
    return float(np.mean(list(per_provider.values()))), per_provider


def expected_keys(v1: bool = False) -> set[tuple[str, str, str, int]]:
    reps_by_view = {"feed_like_v1": (1, 2)} if v1 else VIEWS
    keys = set()
    for provider, match in MATCHES:
        for view, reps in reps_by_view.items():
            keys.update((provider, match, view, rep) for rep in reps)
    return keys


def key_stem(key: tuple[str, str, str, int]) -> str:
    return "__".join(map(str, key))


def load_cell(stage: str, key: tuple[str, str, str, int], declaration_hash: str):
    stem = key_stem(key)
    npz_path = OUT / "cell_artifacts" / stage / f"{stem}.npz"
    receipt_path = OUT / "cell_artifacts" / stage / f"{stem}.json"
    if not npz_path.is_file() or not receipt_path.is_file():
        raise FileNotFoundError(f"missing immutable cell artifact: {npz_path}")
    receipt = read_json(receipt_path)
    if (receipt.get("study_id") != STUDY or tuple(receipt.get("key", ())) != key
            or receipt.get("stage") != stage
            or receipt.get("arrays_sha256") != sha(npz_path)
            or receipt.get("declaration_sha256") != declaration_hash):
        raise AssertionError(f"cell receipt verification failed: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as stored:
        item = {name: stored[name] for name in stored.files}
    required = {"labels", "period", "timestamps", "fps", "indices", "reference_denominator",
                "metrics_json", "input_sha256", "declaration_sha256",
                "deployed_original_predictions", "deployed_original_probabilities"}
    if not required <= item.keys():
        raise AssertionError(f"missing replay arrays in {npz_path}: {sorted(required-item.keys())}")
    labels = item["labels"].astype(str)
    if (not np.array_equal(item["indices"], np.arange(len(labels)))
            or int(item["reference_denominator"].item()) != len(labels)):
        raise AssertionError(f"frame population/index mismatch: {npz_path}")
    if (int(item.get("frame_count", np.asarray(-1)).item()) != len(labels)
            or int(item.get("graded_frames", np.asarray(-1)).item()) != int(np.sum(labels != "X"))):
        raise AssertionError(f"saved frame counts disagree with labels: {npz_path}")
    if item["declaration_sha256"].item() != declaration_hash:
        raise AssertionError(f"NPZ declaration hash mismatch: {npz_path}")
    if receipt.get("input_sha256") != item["input_sha256"].item():
        raise AssertionError(f"input fingerprint mismatch: {npz_path}")
    return item, json.loads(item["metrics_json"].item()), receipt


def compare_metric_sets(actual: dict, expected: dict, context: str) -> int:
    checks = 0
    for metric in METRICS:
        if metric not in actual or metric not in expected:
            raise AssertionError(f"{context}: missing metric {metric}")
        a, e = actual[metric], expected[metric]
        if a is None or e is None:
            if a is not None or e is not None:
                raise AssertionError(f"{context}/{metric}: {a!r} != {e!r}")
        elif abs(float(a) - float(e)) > EPS:
            raise AssertionError(f"{context}/{metric}: saved {a} != replay {e}")
        checks += 1
    return checks


def replay_stage(stage: str, arm_names_by_stage: dict, declaration_hash: str,
                 saved_rows: dict | None = None, v1: bool = False):
    keys = expected_keys(v1)
    directory = OUT / "cell_artifacts" / stage
    actual_npz = {path.stem for path in directory.glob("*.npz")}
    expected_npz = {key_stem(key) for key in keys}
    if actual_npz != expected_npz:
        raise AssertionError(
            f"{stage}: expected exactly {len(expected_npz)} cells; "
            f"missing={sorted(expected_npz-actual_npz)}, extra={sorted(actual_npz-expected_npz)}")
    scores, receipt_hashes = {}, {}
    for key in sorted(keys):
        item, saved_metrics, receipt = load_cell(stage, key, declaration_hash)
        label = "|".join(map(str, key))
        arms = arm_names_by_stage
        if saved_rows is not None:
            if label not in saved_rows:
                raise AssertionError(f"result JSON missing cell {label}")
            for arm in arms:
                if arm not in saved_rows[label] or arm not in saved_metrics:
                    raise AssertionError(f"result-json/{label} misses arm {arm}")
                compare_metric_sets(saved_rows[label][arm], saved_metrics[arm],
                                    f"result-json/{label}/{arm}")
        row = {}
        decode_period = model_period_ids(item["period"], item["timestamps"],
                                         float(item["fps"].item()))
        for arm in arms:
            pred_key = ("deployed_original_predictions" if arm == "deployed_original_full"
                        else f"predictions__{arm}")
            if pred_key not in item:
                raise AssertionError(f"{stage}/{label} lacks {pred_key}")
            calculated = score_saved(item["labels"], item[pred_key], item["period"],
                                     item["timestamps"], float(item["fps"].item()))
            if arm not in saved_metrics:
                raise AssertionError(f"{stage}/{label} metrics JSON lacks {arm}")
            compare_metric_sets(saved_metrics[arm], calculated, f"npz-replay/{stage}/{label}/{arm}")
            if arm != "deployed_original_full":
                probability_key = f"probabilities__{arm}"
                if probability_key not in item:
                    raise AssertionError(f"{stage}/{label} lacks {probability_key}")
                probabilities = item[probability_key]
                if (probabilities.shape != (len(item["labels"]), 3)
                        or not np.isfinite(probabilities).all()
                        or (probabilities < 0).any()
                        or not np.allclose(probabilities.sum(axis=1), 1., atol=1e-5, rtol=0)):
                    raise AssertionError(f"invalid normalized probabilities: {stage}/{label}/{arm}")
                decoded = feed_spell_push.decode(
                    probabilities[None], decode_period, float(item["fps"].item()),
                    {"family": "calibration", "dt": 4, "dead": 3, "ha": 1.5, "ht": 1.0})
                if not np.array_equal(decoded.astype(str), item[pred_key].astype(str)):
                    raise AssertionError(f"saved decoded prediction does not match probability: {stage}/{label}/{arm}")
                seed_keys = [f"seed_probabilities__{arm}__{seed}" for seed in (2026, 2027)]
                if not all(seed_key in item for seed_key in seed_keys):
                    raise AssertionError(f"{stage}/{label}/{arm} lacks seed probabilities")
                if "_seed_metrics" not in saved_metrics or arm not in saved_metrics["_seed_metrics"]:
                    raise AssertionError(f"{stage}/{label}/{arm} lacks supplemental seed metrics")
                for seed, seed_key in zip((2026, 2027), seed_keys):
                    seed_probs = item[seed_key]
                    if (seed_probs.shape != (len(item["labels"]), 3)
                            or not np.isfinite(seed_probs).all()
                            or (seed_probs < 0).any()
                            or not np.allclose(seed_probs.sum(axis=1), 1., atol=1e-5, rtol=0)):
                        raise AssertionError(f"invalid seed probabilities: {stage}/{label}/{seed}")
                    seed_pred = feed_spell_push.decode(
                        seed_probs[None], decode_period, float(item["fps"].item()),
                        {"family": "calibration", "dt": 4, "dead": 3, "ha": 1.5, "ht": 1.0})
                    seed_score = score_saved(item["labels"], seed_pred, item["period"],
                                             item["timestamps"], float(item["fps"].item()))
                    compare_metric_sets(saved_metrics["_seed_metrics"][arm][str(seed)], seed_score,
                                        f"seed-replay/{stage}/{label}/{arm}/{seed}")
                    if saved_rows is not None:
                        compare_metric_sets(saved_rows[label]["_seed_metrics"][arm][str(seed)],
                                            saved_metrics["_seed_metrics"][arm][str(seed)],
                                            f"result-json-seed/{label}/{arm}/{seed}")
            row[arm] = calculated
        original_probability = item["deployed_original_probabilities"]
        if (original_probability.shape != (len(item["labels"]), 3)
                or not np.isfinite(original_probability).all()
                or (original_probability < 0).any()
                or not np.allclose(original_probability.sum(axis=1), 1., atol=1e-5, rtol=0)):
            raise AssertionError(f"invalid deployed original probabilities: {stage}/{label}")
        original_decoded = feed_spell_push.decode(
            original_probability[None], decode_period, float(item["fps"].item()),
            calibrated_dense_baseline.CONFIG)
        if not np.array_equal(original_decoded.astype(str),
                              item["deployed_original_predictions"].astype(str)):
            raise AssertionError(f"deployed prediction does not match saved probabilities: {stage}/{label}")
        scores[key] = row
        receipt_hashes[label] = sha(OUT / "cell_artifacts" / stage / f"{key_stem(key)}.json")
    if set(scores) != keys:
        raise AssertionError(f"{stage} cell cohort mismatch")
    return scores, receipt_hashes


def aggregate(scores: dict, arm: str, view: str) -> dict:
    output = {}
    for metric in METRICS:
        values = {(provider, match, str(rep)): row[arm][metric]
                  for (provider, match, view_name, rep), row in scores.items()
                  if view_name == view}
        mean, providers = equal_provider(values)
        output[metric] = {"equal_provider_mean": mean, "providers": providers}
    return output


def compare_aggregates(saved: dict, replayed: dict, context: str):
    for arm, views in replayed.items():
        if arm not in saved:
            raise AssertionError(f"{context}: missing saved arm {arm}")
        for view, metric_set in views.items():
            if view not in saved[arm]:
                raise AssertionError(f"{context}: missing {arm}/{view}")
            for metric, value in metric_set.items():
                stored = saved[arm][view][metric]
                if abs(float(stored["equal_provider_mean"]) - value["equal_provider_mean"]) > EPS:
                    raise AssertionError(f"{context}/{arm}/{view}/{metric}: mean mismatch")
                for provider, actual in value["providers"].items():
                    if abs(float(stored["providers"][provider]) - actual) > EPS:
                        raise AssertionError(f"{context}/{arm}/{view}/{metric}/{provider}: mismatch")


def verify() -> dict:
    declaration_path = RESULTS / f"{STUDY}_declaration.json"
    control_path = RESULTS / f"{STUDY}_control_replay.json"
    primary_path = RESULTS / f"{STUDY}.json"
    winner_path = RESULTS / f"{STUDY}_winner.json"
    if not all(path.is_file() for path in (declaration_path, control_path, primary_path)):
        raise FileNotFoundError("development declaration, control replay, and primary result are required")
    declaration_hash = sha(declaration_path)
    declaration, control, primary = map(read_json, (declaration_path, control_path, primary_path))
    if primary.get("declaration_sha256") != declaration_hash:
        raise AssertionError("primary result binds a different declaration")
    if declaration.get("study_id") != STUDY or declaration.get("promotion") is not False:
        raise AssertionError("unexpected study identity or promotion flag")
    declared_cohort = declaration.get("cohort", {}).get("matches", [])
    if {(r["provider"], r["match"]) for r in declared_cohort} != set(MATCHES):
        raise AssertionError("declared match population differs from fixed 5-match cohort")

    control_arms = ("blend_unedited_ha15", "deployed_original_full")
    control_scores, control_receipts = replay_stage(
        "control", control_arms, declaration_hash, control.get("rows"))
    control_aggregate = {
        arm: {view: aggregate(control_scores, arm, view) for view in VIEWS}
        for arm in control_arms
    }
    compare_aggregates({"deployed_original_full": control["archived_original_full_replay"]},
                       {"deployed_original_full": control_aggregate["deployed_original_full"]},
                       "control-archived")
    compare_aggregates({"blend_unedited_ha15": control["reconstructed_unedited_blend"]},
                       {"blend_unedited_ha15": control_aggregate["blend_unedited_ha15"]},
                       "control-blend")
    if control.get("archived_original_full_replay_passed") is not True:
        raise AssertionError("control replay report did not pass archived baseline check")

    primary_arms = tuple(arm for arm in declaration["provenance"]["active_arms"])
    if primary.get("cell_count") != 25 or primary.get("organizer_reference_access") is not False:
        raise AssertionError("primary cohort size or reference access flag mismatch")
    primary_stage_scores, primary_receipts = replay_stage(
        "candidate", (*primary_arms, "deployed_original_full"), declaration_hash,
        primary.get("cells"))
    for key in expected_keys():
        control_npz = OUT / "cell_artifacts" / "control" / f"{key_stem(key)}.npz"
        candidate_npz = OUT / "cell_artifacts" / "candidate" / f"{key_stem(key)}.npz"
        with np.load(control_npz, allow_pickle=False) as left, \
                np.load(candidate_npz, allow_pickle=False) as right:
            for name in ("labels", "period", "timestamps", "indices", "reference_denominator"):
                if not np.array_equal(left[name], right[name]):
                    raise AssertionError(f"reference population changed for {key}/{name}")
            if left["input_sha256"].item() != right["input_sha256"].item():
                raise AssertionError(f"input fingerprint changed between stages for {key}")
            if float(left["fps"].item()) != float(right["fps"].item()):
                raise AssertionError(f"sampling rate changed between stages for {key}")
    replayed_aggregates = {
        arm: {view: aggregate(primary_stage_scores, arm, view) for view in VIEWS}
        for arm in (*primary_arms, "deployed_original_full")
    }
    compare_aggregates(primary["arms"], replayed_aggregates, "primary")
    if primary.get("guard_results") != expected_guards(primary["arms"]):
        raise AssertionError("saved guards differ from independently applied fixed rules")
    winner = select_winner(primary["arms"], primary["guard_results"])
    if primary.get("eligible_winner") != winner:
        raise AssertionError("saved winner differs from independently selected arm")
    if winner_path.is_file():
        receipt = read_json(winner_path)
        if (receipt.get("primary_result_sha256") != sha(primary_path)
                or receipt.get("winner") != (winner or max(
                    primary_arms, key=lambda arm: (primary["arms"][arm]["feed_like_v2"]
                                                       ["spell_f1"]["equal_provider_mean"],
                                                   primary["arms"][arm]["feed_like_v2"]
                                                       ["macro_f1"]["equal_provider_mean"])) )
                or receipt.get("eligible") is not (winner is not None)):
            raise AssertionError("winner freeze receipt mismatch")
    verified_v1 = False
    v1_path = RESULTS / f"{STUDY}_v1.json"
    if v1_path.is_file():
        v1_result = read_json(v1_path)
        if not winner_path.is_file():
            raise AssertionError("v1 readout exists before the winner freeze receipt")
        frozen_winner = read_json(winner_path)["winner"]
        if (v1_result.get("winner") != frozen_winner
                or v1_result.get("selection_used_v1") is not False):
            raise AssertionError("v1 result is not bound to the frozen pre-v1 winner")
        v1_arms = tuple(v1_result["arms"])
        v1_scores, _ = replay_stage("v1", v1_arms, declaration_hash, v1_result.get("cells"), v1=True)
        for arm in v1_arms:
            expected_aggregate = aggregate(v1_scores, arm, "feed_like_v1")
            for metric, value in expected_aggregate.items():
                stored = v1_result["arms"][arm][metric]
                if abs(float(stored["equal_provider_mean"]) - value["equal_provider_mean"]) > EPS:
                    raise AssertionError(f"v1/{arm}/{metric}: mean mismatch")
                for provider, actual in value["providers"].items():
                    if abs(float(stored["providers"][provider]) - actual) > EPS:
                        raise AssertionError(f"v1/{arm}/{metric}/{provider}: mismatch")
        verified_v1 = True
    fit_facts = verify_fit_receipt()
    gar_facts = verify_gar_readout()
    public_facts = verify_public_saved_result()
    pooled_tensor_facts = verify_pooled_tensor_edits()
    return {
        "study_id": STUDY, "independent_scalar_replay": True,
        "development_cell_count": len(primary_stage_scores),
        "control_cell_count": len(control_scores), "candidate_arms": list(primary_arms),
        "eligible_winner": winner, "winner_receipt_verified": winner_path.is_file(),
        "v1_readout_verified": verified_v1, "reconstruction_fit_facts": fit_facts,
        "paired_pooled_tensor_audit": pooled_tensor_facts or {"status": "pending_deployment_fit"},
        "gar_readout_verification": gar_facts or {"status": "pending_saved_output"},
        "public_saved_result_verification": public_facts or {"status": "pending_saved_output"},
        "control_receipts": control_receipts, "candidate_receipts": primary_receipts,
        "declaration_sha256": declaration_hash, "primary_result_sha256": sha(primary_path),
        "control_result_sha256": sha(control_path), "gar_or_public_labels_read": False,
        "model_inference_rerun": False,
    }


def verify_fit_receipt() -> dict:
    """Check every immutable reconstructed checkpoint against its fit receipts."""
    fit_path = RESULTS / "v2_weight_edit_reconstruction_v1_fit_receipt.json"
    fit_declaration_path = RESULTS / "v2_weight_edit_reconstruction_v1_declaration.json"
    if not fit_path.is_file() or not fit_declaration_path.is_file():
        raise FileNotFoundError("six-fit reconstruction receipt/declaration are required")
    fit_report = read_json(fit_path)
    fit_declaration = read_json(fit_declaration_path)
    prior_recipe_path = RESULTS / "v2_short_original_diversity_declaration.json"
    prior_training_audit_path = RESULTS / "short_original_diversity_v1_training_audit.json"
    prior_recipe = read_json(prior_recipe_path)
    prior_training_audit = read_json(prior_training_audit_path)
    if (fit_report.get("all_six_fits_complete") is not True
            or fit_report.get("byte_exact_restore_claim") is not False
            or fit_report.get("declaration_sha256") != sha(fit_declaration_path)
            or len(fit_report.get("fits", [])) != 6):
        raise AssertionError("incomplete or invalid six-fit receipt")
    expected = {f"leave_out_{provider.lower()}_{seed}.joblib"
                for provider in ("sportec", "metrica", "skillcorner")
                for seed in (2026, 2027)}
    seen = set()
    checks = []
    prior_sampler = {(row["fold"], int(row["seed"])): row["sampling"]
                     for row in prior_training_audit.get("rows", [])}
    for row in fit_report["fits"]:
        model_path = ROOT / "data/experiments/v2_weight_edit_reconstruction_v1/models" / \
            f"{row['fold']}_{row['seed']}.joblib"
        if model_path.name in seen or model_path.name not in expected or not model_path.is_file():
            raise AssertionError(f"unexpected/missing reconstructed model path: {model_path}")
        seen.add(model_path.name)
        model_receipt_path = Path(str(model_path) + ".json")
        receipt = read_json(model_receipt_path)
        if (sha(model_path) != row.get("sha256")
                or receipt.get("sha256") != row.get("sha256")
                or receipt.get("declaration_sha256") != sha(fit_declaration_path)
                or receipt.get("fold") != row.get("fold")
                or int(receipt.get("seed", -1)) != int(row.get("seed", -2))):
            raise AssertionError(f"reconstructed fit checksum/receipt mismatch: {model_path}")
        sampler_key = (row["fold"], int(row["seed"]))
        if row.get("sampler") != prior_sampler.get(sampler_key):
            raise AssertionError(f"reconstructed sampler receipt differs from prior recipe: {sampler_key}")
        checks.append({"path": str(model_path.relative_to(ROOT)), "sha256": row["sha256"],
                       "fold": row["fold"], "seed": row["seed"]})
    if seen != expected:
        raise AssertionError(f"six-fit model set mismatch: {sorted(seen)}")
    # The reconstructed four-view TRAIN artifacts were regenerated from the
    # declared recipe inputs; verify their 52 joblib byte hashes against the
    # earlier frozen PR82 declaration before accepting the fit chain.
    prior_input_hashes = set(prior_recipe["input_sha256"].values())
    view_jobs = {path: digest for path, digest in fit_declaration["inputs"]["files"].items()
                 if "/v2_weight_edit_reconstruction_v1/views/" in path
                 and path.endswith(".joblib")}
    if len(view_jobs) != 52 or not all(digest in prior_input_hashes for digest in view_jobs.values()):
        raise AssertionError("reconstructed v2 TRAIN view hashes diverge from the frozen PR82 input inventory")
    for raw_path, expected_hash in view_jobs.items():
        view_path = Path(raw_path)
        if not view_path.is_absolute():
            view_path = ROOT / view_path
        if not view_path.is_file() or sha(view_path) != expected_hash:
            raise AssertionError(f"reconstructed v2 TRAIN view changed: {view_path}")
    return {"all_six_fit_receipts_valid": True,
            "fit_receipt_sha256": sha(fit_path),
            "fit_declaration_sha256": sha(fit_declaration_path), "models": checks,
            "reconstructed_v2_train_view_joblibs_match_prior_declaration": len(view_jobs),
            "sampler_receipts_match_prior_training_audit": len(checks),
            "prior_training_declaration_sha256": sha(prior_recipe_path),
            "byte_exact_restore_claim": False}


def verify_gar_readout() -> dict | None:
    """Replay saved GAR TEST predictions; no GAR inference or public access."""
    gar_result_path = RESULTS / "v2_weight_edit_evaluation_v1.json"
    gar_declaration_path = RESULTS / "v2_weight_edit_evaluation_v1_declaration.json"
    gar_root = ROOT / "data/experiments/v2_weight_edit_evaluation_v1"
    if not gar_result_path.is_file():
        return None
    if not gar_declaration_path.is_file():
        raise AssertionError("GAR result has no bound declaration")
    declaration, report = read_json(gar_declaration_path), read_json(gar_result_path)
    declaration_sha = sha(gar_declaration_path)
    if report.get("declaration_sha256") != declaration_sha:
        raise AssertionError("GAR report declaration checksum mismatch")
    gar_baseline_path = RESULTS / "gar_deployment_eval_v2.json"
    public_baseline_path = RESULTS / "feed_like_v2_public_result.json"
    if (declaration.get("baseline", {}).get("gar_report_sha256") != sha(gar_baseline_path)
            or declaration.get("baseline", {}).get("public_report_sha256") != sha(public_baseline_path)):
        raise AssertionError("GAR declaration does not bind the reused baseline reports")
    if report.get("organizer_reference_access") is not False or report.get("public_scoring") is not False:
        raise AssertionError("GAR report unexpectedly claims public/reference scoring")
    game_ids = declaration.get("cohort", {}).get("match_ids", [])
    if len(game_ids) != 10 or len(set(map(str, game_ids))) != 10:
        raise AssertionError("GAR TEST cohort is not the frozen 10-game cohort")
    cells = {}
    for game in game_ids:
        game = str(game)
        json_path = gar_root / "games" / f"test__{game}.json"
        npz_path = json_path.with_suffix(".npz")
        cell_report = read_json(json_path)
        if (cell_report.get("game_id") != game
                or cell_report.get("declaration_sha256") != declaration_sha
                or cell_report.get("package_receipt_sha256")
                    != declaration["selected_package"]["receipt_sha256"]
                or cell_report.get("prediction_sha256") != sha(npz_path)
                or cell_report.get("split") != "test"
                or cell_report.get("seed_namespace_reused") != f"gar_deployment_eval_v2/{game}"
                or cell_report.get("grading_and_reference_counts_equal") is not True):
            raise AssertionError(f"GAR game receipt mismatch: {game}")
        with np.load(npz_path, allow_pickle=False) as saved:
            labels = saved["labels"].astype(str)
            period, timestamps, fps = saved["period"], saved["timestamps"], float(saved["fps"].item())
            expected_phases = int(round(fps / 5.0))
            if set(cell_report.get("phase_counts", {}).values()) != {expected_phases}:
                raise AssertionError(f"GAR native phase count mismatch for {game}: {cell_report.get('phase_counts')}")
            if int(np.sum(labels != "X")) != int(cell_report["graded_frames"]):
                raise AssertionError(f"GAR graded population mismatch: {game}")
            scores = {}
            for view in GAR_VIEWS:
                if view not in saved:
                    raise AssertionError(f"GAR prediction NPZ missing {game}/{view}")
                pred = saved[view].astype(str)
                scores[view] = score_saved(labels, pred, period, timestamps, fps)
                scores[view]["graded_frames"] = int(np.sum(labels != "X"))
                for metric in GAR_METRICS:
                    stored = cell_report["scores"][view].get(metric)
                    actual = scores[view][metric]
                    if stored is None or abs(float(stored) - float(actual)) > EPS:
                        raise AssertionError(f"GAR scalar mismatch {game}/{view}/{metric}: {stored} vs {actual}")
            if len(labels) != int(cell_report["native_frames"]):
                raise AssertionError(f"GAR frame count mismatch: {game}")
            counts = {(row["graded_frames"], row["reference_spell_count"]) for row in scores.values()}
            if len(counts) != 1:
                raise AssertionError(f"GAR scoring population changed across views: {game}")
        cells[game] = scores

    metrics = tuple(GAR_METRICS)
    per_game = {"clean": {g: cells[g]["clean"] for g in game_ids}, "v2": {}}
    for game in game_ids:
        per_game["v2"][game] = {}
        for metric in metrics:
            values = [cells[game][f"v2_{i}"][metric] for i in range(1, 5)]
            if metric in ("graded_frames", "reference_spell_count") and len(set(values)) != 1:
                raise AssertionError(f"GAR invariant metric varies over v2 views: {game}/{metric}")
            per_game["v2"][game][metric] = float(np.mean(values))
    for view in ("clean", "v2"):
        for game in game_ids:
            for metric in metrics:
                expected_value = per_game[view][game][metric]
                stored = report["per_game"][view][game][metric]
                if abs(float(stored) - float(expected_value)) > EPS:
                    raise AssertionError(f"GAR per-game aggregate mismatch {view}/{game}/{metric}")
    realization_means = {f"v2_{i}": {
        metric: float(np.mean([cells[g][f"v2_{i}"][metric] for g in game_ids]))
        for metric in metrics} for i in range(1, 5)}
    for realization, vals in realization_means.items():
        for metric, expected_value in vals.items():
            if abs(float(report["per_realization_equal_game_means"][realization][metric])
                   - expected_value) > EPS:
                raise AssertionError(f"GAR equal-game realization mean mismatch {realization}/{metric}")

    baseline = read_json(gar_baseline_path)["splits"]["test"]["scores"]
    rng = np.random.default_rng(20261104)
    for summary_view in ("clean", "v2"):
        saved_view = summary_view
        for metric in metrics:
            candidate_values = np.asarray([per_game[summary_view][str(g)][metric] for g in game_ids], dtype=float)
            base_values = np.asarray([
                baseline[saved_view]["original_full"]["phase_avg"]["per_game"][str(g)][metric]
                for g in game_ids], dtype=float)
            deltas = candidate_values - base_values
            recorded_pairs = report["per_game"]["paired"][summary_view][metric]
            for game, candidate, prior, delta in zip(game_ids, candidate_values, base_values, deltas):
                record = recorded_pairs[str(game)]
                if any(abs(float(record[k]) - float(v)) > EPS for k, v in
                       (("candidate", candidate), ("saved_baseline", prior), ("delta", delta))):
                    raise AssertionError(f"GAR saved baseline pair mismatch {summary_view}/{game}/{metric}")
            picks = rng.integers(0, len(deltas), size=(2000, len(deltas)))
            boot = deltas[picks].mean(axis=1)
            expected_summary = {
                "candidate_equal_game_mean": float(candidate_values.mean()),
                "saved_baseline_equal_game_mean": float(base_values.mean()),
                "delta": float(deltas.mean()), "games_positive": int((deltas > 0).sum()),
                "games": int(len(deltas)),
                "paired_game_bootstrap_95": [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
            }
            for key, val in expected_summary.items():
                stored = report["summary"][summary_view][metric][key]
                if isinstance(val, list):
                    if not np.allclose(stored, val, atol=EPS, rtol=0):
                        raise AssertionError(f"GAR bootstrap summary mismatch {summary_view}/{metric}")
                elif isinstance(val, int):
                    if int(stored) != val:
                        raise AssertionError(f"GAR game count mismatch {summary_view}/{metric}")
                elif abs(float(stored) - val) > EPS:
                    raise AssertionError(f"GAR paired aggregate mismatch {summary_view}/{metric}/{key}")
    return {"saved_gar_test_npz_metrics_replayed": True, "gar_game_count": len(cells),
            "gar_declaration_sha256": declaration_sha, "gar_result_sha256": sha(gar_result_path),
            "public_reference_accessed": False}


def verify_public_saved_result() -> dict | None:
    """Check only saved public score/provenance JSON; never open reference or candidate CSV."""
    result_path = RESULTS / "v2_weight_edit_evaluation_v1_public_result.json"
    raw_path = ROOT / "data/experiments/v2_weight_edit_evaluation_v1_public/score_raw.json"
    declaration_path = RESULTS / "v2_weight_edit_evaluation_v1_public_declaration.json"
    if not result_path.is_file():
        return None
    if not raw_path.is_file() or not declaration_path.is_file():
        raise AssertionError("public score report is missing its saved raw receipt or freeze declaration")
    result, raw, declaration = read_json(result_path), read_json(raw_path), read_json(declaration_path)
    if (result.get("raw_score_receipt_sha256") != sha(raw_path)
            or raw.get("declaration_sha256") != sha(declaration_path)
            or raw.get("reference_sha256") != declaration.get("organizer_reference_sha256")
            or raw.get("candidate_csv_sha256") != declaration.get("candidate_csv_sha256")
            or result.get("organizer_reference_sha256") != declaration.get("organizer_reference_sha256")
            or result.get("organizer_reference_labels_read") is not True
            or result.get("organizer_scored") is not True):
        raise AssertionError("saved public result provenance chain is inconsistent")
    score = raw["raw_score"]
    summary = result["candidate"]
    scalar_pairs = (("spell_f1_mean", score["spell_f1_mean"]), ("macro_f1", score["macro_f1"]),
                    ("frames", score["n_frames"]), ("graded_frames", score["n_graded"]),
                    ("ground_truth_spells", score["n_gt_spells"]),
                    ("predicted_spells", score["n_pred_spells"]),
                    ("bootstrap_draws", score["bootstrap_n"]))
    for key, value in scalar_pairs:
        if abs(float(summary[key]) - float(value)) > EPS:
            raise AssertionError(f"saved public summary/raw scalar mismatch: {key}")
    if (result.get("interpretation", {}).get("public_used_for_training_tuning_selection") is not False
            or result.get("interpretation", {}).get("promotion") is not False):
        raise AssertionError("saved public report claims selection/promotion")
    return {"saved_public_result_provenance_checked": True,
            "public_result_sha256": sha(result_path), "raw_score_receipt_sha256": sha(raw_path),
            "organizer_reference_file_opened_by_verifier": False,
            "public_score_recomputed": False}


def _tree_equal(left, right) -> bool:
    if isinstance(left, dict) and isinstance(right, dict):
        return (set(left) == set(right)
                and all(_tree_equal(left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
        except (TypeError, ValueError):
            return False
    return left == right


def verify_pooled_tensor_edits() -> dict | None:
    """Audit the two deployed edited checkpoints against their paired pre-edit fits."""
    study = "v2_weight_edit_pooled_deployment_v1"
    fit_path = RESULTS / f"{study}_fit_receipt.json"
    declaration_path = RESULTS / f"{study}_declaration.json"
    deployment_path = ROOT / f"data/experiments/{study}/deployment.json"
    if not fit_path.is_file():
        return None
    if not declaration_path.is_file() or not deployment_path.is_file():
        raise AssertionError("pooled deployment fit receipt lacks its declaration/package")
    fit, declaration, deployment = map(read_json, (fit_path, declaration_path, deployment_path))
    if (fit.get("declaration_sha256") != sha(declaration_path)
            or deployment.get("fit_receipt_sha256") != sha(fit_path)
            or deployment.get("declaration_sha256") != sha(declaration_path)
            or fit.get("winner_id") != declaration.get("winner_id")
            or fit.get("winner_id") != deployment.get("winner_id")):
        raise AssertionError("pooled deployment receipt chain does not bind selected choice")
    model_hashes = fit.get("model_files_sha256", {})
    if not model_hashes or deployment.get("model_files_sha256") != model_hashes:
        raise AssertionError("deployment package model checksum manifest mismatch")
    for raw_path, expected_hash in model_hashes.items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha(path) != expected_hash:
            raise AssertionError(f"pooled package model hash mismatch: {path}")
    members = fit.get("members", [])
    if len(members) != 2:
        raise AssertionError("expected exactly two pooled deployment member receipts")
    kind = fit.get("edit", {}).get("kind")
    delta = float(fit.get("edit_gate_delta", deployment.get("gate_delta", 0.0)))
    alpha = float(fit.get("edit_alpha", deployment.get("alpha", 0.0)))
    model_facts = []
    seen_seeds = set()
    for public_member in members:
        private_path = ROOT / public_member["private_member_receipt_path"]
        if (not private_path.is_file()
                or sha(private_path) != public_member.get("private_member_receipt_sha256")):
            raise AssertionError(f"private member receipt hash mismatch: {private_path}")
        member = read_json(private_path)
        seed = int(public_member["seed"])
        if (seed in seen_seeds or member.get("seed") != seed
                or member.get("declaration_sha256") != sha(declaration_path)):
            raise AssertionError(f"duplicate/stale pooled member receipt for seed {seed}")
        seen_seeds.add(seed)
        for key in ("unedited_model_path", "unedited_model_sha256", "edited_model_path",
                    "edited_model_sha256", "pooled_teacher_path", "pooled_teacher_sha256"):
            if member.get(key) != public_member.get(key):
                raise AssertionError(f"public/private member receipt mismatch: seed={seed}, {key}")
        before_path, after_path = ROOT / member["unedited_model_path"], ROOT / member["edited_model_path"]
        if (sha(before_path) != member["unedited_model_sha256"]
                or sha(after_path) != member["edited_model_sha256"]):
            raise AssertionError(f"paired model checksum mismatch for seed {seed}")
        before, after = joblib.load(before_path), joblib.load(after_path)
        if set(before) != set(after) or set(before.get("state", {})) != set(after.get("state", {})):
            raise AssertionError(f"model structure changed for seed {seed}")
        for key in before:
            if key != "state" and not _tree_equal(before[key], after[key]):
                raise AssertionError(f"non-state model metadata changed for seed {seed}: {key}")
        before_state, after_state = before["state"], after["state"]
        changed_keys = {key for key in before_state
                        if not np.array_equal(np.asarray(before_state[key]), np.asarray(after_state[key]))}
        if kind == "gate_bias":
            hidden = int(before["hidden"])
            expected_keys = {"sequence.bias_ih_l0", "sequence.bias_ih_l0_reverse"}
            if changed_keys != expected_keys:
                raise AssertionError(f"gate edit changed wrong tensors for seed {seed}: {sorted(changed_keys)}")
            changed_values = 0
            for key in expected_keys:
                old, new = np.asarray(before_state[key]), np.asarray(after_state[key])
                if old.shape != (3 * hidden,) or new.shape != old.shape:
                    raise AssertionError(f"wrong GRU bias shape for seed {seed}/{key}")
                np.testing.assert_array_equal(new[:hidden], old[:hidden])
                np.testing.assert_allclose(new[hidden:2*hidden] - old[hidden:2*hidden],
                                           delta, atol=1e-7, rtol=0)
                np.testing.assert_array_equal(new[2*hidden:], old[2*hidden:])
                changed_values += int(np.count_nonzero(new != old))
            if changed_values != 2 * hidden or changed_values != 192:
                raise AssertionError(f"expected 192 changed update-gate values for seed {seed}, got {changed_values}")
            expected_changed = sorted(expected_keys)
        elif kind == "ha_projection":
            if changed_keys != {"output.weight"}:
                raise AssertionError(f"projection edit changed wrong tensors for seed {seed}: {sorted(changed_keys)}")
            vector = np.asarray(member["direction"], dtype=np.float64)
            old = np.asarray(before_state["output.weight"], dtype=np.float64)
            actual = np.asarray(after_state["output.weight"], dtype=np.float64)
            if old.shape != (2, len(vector)):
                raise AssertionError(f"projection direction/output shape mismatch for seed {seed}")
            expected = old - alpha * np.outer(old @ vector, vector)
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
            if not np.any(actual != old):
                raise AssertionError(f"projection edit was numerically a no-op for seed {seed}")
            expected_changed = ["output.weight"]
        elif kind == "control":
            if changed_keys:
                raise AssertionError(f"control edit changed tensors for seed {seed}")
            expected_changed = []
        else:
            raise AssertionError(f"unrecognized deployment edit kind: {kind}")
        if sorted(member.get("changed_state_tensors", [])) != expected_changed:
            raise AssertionError(f"private tensor-change summary mismatch for seed {seed}")
        model_facts.append({"seed": seed, "edited_model_sha256": member["edited_model_sha256"],
                            "unedited_model_sha256": member["unedited_model_sha256"],
                            "changed_state_tensors": expected_changed,
                            "changed_value_count": (192 if kind == "gate_bias" else
                                                    int(np.count_nonzero(np.asarray(after_state["output.weight"])
                                                        != np.asarray(before_state["output.weight"])))
                                                    if kind == "ha_projection" else 0)})
    if seen_seeds != {2026, 2027}:
        raise AssertionError(f"pooled model seed set mismatch: {sorted(seen_seeds)}")
    return {"paired_pooled_tensor_audit_passed": True, "edit_kind": kind,
            "edit_alpha": alpha, "edit_gate_delta": delta, "members": model_facts,
            "deployment_fit_receipt_sha256": sha(fit_path),
            "deployment_package_receipt_sha256": sha(deployment_path),
            "direction_vectors_emitted": False}


def expected_guards(aggregates: dict) -> dict:
    baseline_guard = read_json(RESULTS / "phase_average_guard_v1.json")
    clean_control = baseline_guard["summaries"]["ensemble"]["phase0"]["clean"]
    results = {}
    for arm, views in aggregates.items():
        if arm == "deployed_original_full":
            continue
        checks = {
            "clean_spell_floor": views["clean"]["spell_f1"]["equal_provider_mean"] >= .662555533,
            "v2_spell_floor": views["feed_like_v2"]["spell_f1"]["equal_provider_mean"] >= .573571208,
        }
        provider_checks = {provider: {
            "macro_f1": views["clean"]["macro_f1"]["providers"][provider]
            >= clean_control["macro_f1"]["providers"][provider] - .02,
            "d_f1": views["clean"]["d_f1"]["providers"][provider]
            >= clean_control["d_f1"]["providers"][provider] - .02,
        } for provider in PROVIDERS}
        checks["clean_provider_floors"] = all(v for vals in provider_checks.values() for v in vals.values())
        results[arm] = {"eligible": all(checks.values()), "checks": checks,
                        "providers": provider_checks}
    return results


def select_winner(aggregates: dict, guards: dict):
    eligible = [arm for arm, result in guards.items() if result["eligible"]]
    if not eligible:
        return None
    return max(eligible, key=lambda arm: (
        aggregates[arm]["feed_like_v2"]["spell_f1"]["equal_provider_mean"],
        aggregates[arm]["feed_like_v2"]["macro_f1"]["equal_provider_mean"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true",
                        help="write a new immutable verification report beside the result")
    parser.add_argument("--report-suffix", default="_independent_verification",
                        help="immutable report filename suffix (use a new suffix for later GAR/public readouts)")
    args = parser.parse_args()
    report = verify()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.write_report:
        path = RESULTS / f"{STUDY}{args.report_suffix}.json"
        with path.open("x") as stream:
            stream.write(rendered + "\n")


if __name__ == "__main__":
    main()
