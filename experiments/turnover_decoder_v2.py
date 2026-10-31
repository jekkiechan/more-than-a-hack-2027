"""Regularization-unit follow-up for the frozen turnover-edge experiment.

This is a post-v1 numerical diagnosis, not a hyperparameter sweep.  It changes
only the global scale of source/match-balanced logistic sample weights from
sum=1 to mean=1; the relative weights, weighted transforms, natural event
prior, C, feature/target design, cost map, frozen models and evaluation cohort
are unchanged.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import objective_ensemble as oe  # noqa: E402
from experiments import turnover_decoder as v1  # noqa: E402
from experiments import turnover_decoder_provenance as v1_provenance  # noqa: E402

STUDY_ID = "turnover_decoder_v2_unit_weights"
OUT_RELATIVE = Path("experiments") / STUDY_ID
DECLARATION = ROOT / "experiments/results/turnover_decoder_v2_declaration.json"
FIT_RECEIPT = ROOT / "experiments/results/turnover_decoder_v2_fit_receipt.json"
SCORE_RECEIPT = ROOT / "experiments/results/turnover_decoder_v2_score_receipt.json"
ARTIFACT_RECEIPT = ROOT / "experiments/results/turnover_decoder_v2_artifacts.json"
V1_DECLARATION = ROOT / "experiments/results/turnover_decoder_declaration.json"
V1_FIT = ROOT / "experiments/results/turnover_decoder_fit_receipt.json"
V1_SCORE = ROOT / "experiments/results/turnover_decoder_score_receipt.json"
V1_PROVENANCE = ROOT / "experiments/results/turnover_decoder_provenance_receipt_v2.json"
RAW_INPUTS = ROOT / "experiments/results/short_spell_raw_inputs_verification.json"
V1_ARTIFACTS = ROOT / "experiments/results/turnover_decoder_artifacts.json"
V1_OUT = Path("experiments") / "turnover_decoder_v1"
PROVIDERS = v1.PROVIDERS
SEEDS = v1.SEEDS
V2_SOURCE_FILES = (
    "experiments/turnover_decoder_v2.py",
    "experiments/tests/test_turnover_decoder_v2.py",
    "experiments/turnover_decoder.py",
    "experiments/turnover_decoder_provenance.py",
    "experiments/results/turnover_decoder_declaration.json",
    "experiments/results/turnover_decoder_fit_receipt.json",
    "experiments/results/turnover_decoder_score_receipt.json",
    "experiments/results/turnover_decoder_provenance_receipt_v2.json",
    "experiments/results/short_spell_raw_inputs_verification.json",
    "experiments/results/turnover_decoder_artifacts.json",
)


class StudyError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_root(data: Path | None) -> Path:
    root = v1.data_root(data)
    return root


def _fact_path(data: Path, fact: dict[str, Any]) -> Path:
    rel = Path(str(fact["path"]))
    if rel.parts and rel.parts[0] == "data":
        rel = Path(*rel.parts[1:])
    return data / rel


def _verify_fact(data: Path, index: dict[str, dict], fact: dict[str, Any]) -> None:
    actual = v1_provenance._verify_indexed_file(data, index, str(fact["path"]))
    if actual != {key: fact[key] for key in ("path", "bytes", "sha256")}:
        raise StudyError(f"Pinned v1 input fact changed: {fact['path']}")


def verify_v1_chain(data: Path) -> dict[str, Any]:
    """Recheck the prior archive chain, source bytes, and all pinned inputs."""
    old, old_sha = v1._load_frozen_declaration(data)
    for path in (V1_FIT, V1_SCORE, V1_PROVENANCE, RAW_INPUTS, V1_ARTIFACTS):
        if not path.is_file():
            raise StudyError(f"Missing v1 predecessor receipt: {path}")
    fit = json.loads(V1_FIT.read_text())
    score = json.loads(V1_SCORE.read_text())
    provenance = json.loads(V1_PROVENANCE.read_text())
    raw = json.loads(RAW_INPUTS.read_text())
    artifacts = json.loads(V1_ARTIFACTS.read_text())
    if fit.get("declaration_sha256") != old_sha or fit.get("fit_count") != 3:
        raise StudyError("v1 fit receipt is not bound to the frozen v1 declaration")
    if (score.get("declaration_sha256") != old_sha or score.get("scored_cells") != 15 or
            score.get("scores_read_after_all_heads_sealed") is not True):
        raise StudyError("v1 score receipt is incomplete or not bound to its declaration")
    if (provenance.get("declaration_sha256") != old_sha or
            provenance.get("stage") != "pre_fit_no_turnover_head_fit_or_evaluation_scoring" or
            provenance.get("head_fits") != 0 or provenance.get("scores_read") != 0):
        raise StudyError("v1 independent provenance receipt does not match its pre-fit state")
    if raw.get("all18raw_inputs_size_and_sha256_verified") is not True or raw.get("organizer_inputs_read") is not False:
        raise StudyError("Independent raw-input receipt is incomplete")
    if (artifacts.get("artifacts", {}).get("turnover_decoder_score_receipt.json", {}).get("sha256") !=
            sha256_file(V1_SCORE)):
        raise StudyError("v1 artifact receipt does not bind its score receipt")
    run_path = data / Path(score["run_path"])
    seal_path = data / V1_OUT / "model_seal.json"
    if (not run_path.is_file() or sha256_file(run_path) != score.get("run_sha256") or
            not seal_path.is_file() or sha256_file(seal_path) != score.get("model_seal_sha256")):
        raise StudyError("v1 sealed model/run bytes do not match their score receipt")
    v1._verify_model_seal(data, old_sha)

    archive_root = oe._archive_root()
    objective_index, objective_sha = v1_provenance._local_index(
        archive_root, oe.OBJECTIVE_ARCHIVE_MANIFEST, oe.OBJECTIVE_ARCHIVE_MANIFEST_SHA256)
    seed_index, seed_sha = v1_provenance._local_index(
        archive_root, oe.SEED_GAP_ARCHIVE_MANIFEST, oe.SEED_GAP_ARCHIVE_MANIFEST_SHA256)
    ensemble_fact = provenance["archive_manifests"]["objective_ensemble"]
    ensemble_index, ensemble_sha = v1_provenance._local_index(
        archive_root, ensemble_fact["path"], ensemble_fact["sha256"])
    expected_manifests = provenance["archive_manifests"]
    for name, current_sha in (("objective_alignment", objective_sha),
                              ("seed_gap_evaluation", seed_sha),
                              ("objective_ensemble", ensemble_sha)):
        if expected_manifests[name]["sha256"] != current_sha:
            raise StudyError(f"Pinned v1 archive manifest changed: {name}")

    for row in provenance["training_caches"]:
        _verify_fact(data, objective_index, row["cache"])
        _verify_fact(data, objective_index, row["receipt"])
    for row in provenance["training_raw_inputs"]:
        _verify_fact(data, objective_index, row["raw_input"])
    for row in provenance["frozen_fold_checkpoints"]:
        _verify_fact(data, objective_index, row["model"])
        _verify_fact(data, objective_index, row["fit_receipt"])
    _verify_fact(data, seed_index, provenance["seed_gap_predeclaration"])
    for row in provenance["selection_raw_inputs"]:
        _verify_fact(data, objective_index, row["raw_input"])
    for row in provenance["evaluation_cells"]:
        _verify_fact(data, seed_index, row["inputs_labels_record"])
        _verify_fact(data, seed_index, row["seed_gap_probability_archive"])
        _verify_fact(data, ensemble_index, row["objective_ensemble_candidate"])
        for member in row["member_probabilities"].values():
            _verify_fact(data, objective_index, member)
            _verify_fact(data, ensemble_index, member)
    return {"declaration_sha256": old_sha, "fit_receipt_sha256": sha256_file(V1_FIT),
            "score_receipt_sha256": sha256_file(V1_SCORE),
            "provenance_receipt_sha256": sha256_file(V1_PROVENANCE),
            "raw_inputs_receipt_sha256": sha256_file(RAW_INPUTS),
            "artifact_receipt_sha256": sha256_file(V1_ARTIFACTS),
            "archive_manifest_sha256": {"objective_alignment": objective_sha,
                                        "seed_gap_evaluation": seed_sha,
                                        "objective_ensemble": ensemble_sha},
            "training_caches": len(provenance["training_caches"]),
            "training_raw_inputs": len(provenance["training_raw_inputs"]),
            "checkpoints": len(provenance["frozen_fold_checkpoints"]),
            "evaluation_cells": len(provenance["evaluation_cells"]),
            "raw_selection_inputs": len(provenance["selection_raw_inputs"])}


def freeze(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    if DECLARATION.exists() or FIT_RECEIPT.exists() or SCORE_RECEIPT.exists():
        raise StudyError("v2 declaration and receipts are immutable once created")
    previous, previous_sha = v1._load_frozen_declaration(root)
    chain = verify_v1_chain(root)
    if chain["declaration_sha256"] != previous_sha:
        raise StudyError("v1 predecessor binding changed during v2 freeze")
    source_paths = sorted(set(V2_SOURCE_FILES) | set(v1.SOURCE_FILES))
    source_hashes = {name: sha256_file(ROOT / name) for name in source_paths}
    previous_score = json.loads(V1_SCORE.read_text())
    baseline_outcome = previous_score["aggregate"]["candidate_minus_baseline"]
    declaration = {
        "study_id": STUDY_ID,
        "status": "post_v1_regularization_units_followup_frozen_before_v2_fits",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "possession/short-spell-recovery",
        "rationale": ("v1's fitted heads changed neither clean spells nor costs materially. "
                      "Its source/match-balanced sample weights were normalized to sum 1, "
                      "while sklearn LogisticRegression minimizes a summed weighted loss plus L2. "
                      "The resulting data term had only one effective observation relative to L2. "
                      "This follow-up changes only global weight scale to mean 1."),
        "hypothesis": ("With the same relative source/match weights, features, prevalence, C and "
                      "cost map, mean-one sample weights give the fixed C=.1 data term N times "
                      "the v1 strength and may produce useful calibrated edge evidence."),
        "prior_v1_outcome": {
            "clean_spell_delta": baseline_outcome["clean"]["spell_f1"],
            "feed_like_spell_delta": baseline_outcome["feed_like"]["spell_f1"],
            "clean_screen_passed": previous_score["aggregate"]["predeclared_guardrails"]["checks"]["clean_spell_plus_0_01"],
            "v1_declaration_sha256": previous_sha,
            "v1_score_receipt_sha256": chain["score_receipt_sha256"],
        },
        "only_change": {
            "from_v1": "normalized source/match sample weights sum to 1",
            "v2": "multiply every source/match sample weight by N directed training rows, making mean weight 1 and sum weight N",
            "C": 0.1,
            "objective_relationship": ("sklearn's L2 logistic objective is 0.5*||coef||^2 + "
                                        "C*sum_i(weight_i*logloss_i). Multiplying all weights "
                                        "by N leaves relative masses and weighted transforms/prevalence unchanged "
                                        "and is mathematically equivalent "
                                        "to retaining normalized weights with effective C=N*C. "
                                        "Weights are fractional source/match masses, so this is "
                                        "objective equivalence to a replicated loss sum, not literal integer row duplication."),
            "unchanged": ["TRAIN fold and full-provider exclusion", "frozen two-seed ensemble predictions",
                          "features and transform", "exact directed targets/native eligibility",
                          "natural prevalence and source/match mass ratios", "solver and max_iter",
                          "cost mapping and bounds", "emissions", "D/live costs", "evaluation cells and masks"],
        },
        "fixed_configuration": {
            "solver": "lbfgs", "C": 0.1, "max_iter": 300, "random_state": 0,
            "weight_scale": "N = number of directed eligible edge rows in the provider's TRAIN fold",
            "global_weight_mean": 1.0,
            "cost_mapping": "clip(3 - (logit(q_edge) - logit(weighted_natural_training_prevalence)), 1.5, 4.5)",
            "head_fits": 3, "evaluation_cells": 15, "extra_arms": 0, "sweeps": 0,
        },
        "predecessor_bindings": {**chain,
                                 "v1_declaration": str(V1_DECLARATION.relative_to(ROOT)),
                                 "v1_fit_receipt": str(V1_FIT.relative_to(ROOT)),
                                 "v1_score_receipt": str(V1_SCORE.relative_to(ROOT)),
                                 "v1_provenance_receipt": str(V1_PROVENANCE.relative_to(ROOT)),
                                 "raw_input_receipt": str(RAW_INPUTS.relative_to(ROOT)),
                                 "v1_artifact_receipt": str(V1_ARTIFACTS.relative_to(ROOT))},
        "cohort": {"provider_folds": deepcopy(previous["cohort"]["provider_folds"]),
                   "cells": deepcopy(previous["cohort"]["cells"]),
                   "historical_exposure": previous["cohort"]["historical_exposure"]},
        "evaluation": {"same_as_v1": True, "frozen_v1_cells": 15,
                       "primary": "official greedy H/A spell F1, mean IoU 0.3/0.4/0.5/0.6/0.7",
                       "diagnostics": ["clean/feed-like short recall", "false extra spells/merges",
                                       "per-provider clean macro and D-F1", "per-provider feed-like spell F1",
                                       "training AUC/logloss (descriptive, in-sample)",
                                       "coefficient norms and weight scale", "native event cost range/bound hits"],
                       "guardrails": deepcopy(previous["evaluation"]["guardrails"]),
                       "organizer_scoring": False, "automatic_promotion": False},
        "source_sha256": source_hashes,
        "data_root": str(root), "output_directory": str((root / OUT_RELATIVE).relative_to(root)),
    }
    DECLARATION.parent.mkdir(parents=True, exist_ok=True)
    DECLARATION.write_text(json.dumps(declaration, indent=2, allow_nan=False) + "\n")
    return {"declaration": str(DECLARATION.relative_to(ROOT)),
            "declaration_sha256": sha256_file(DECLARATION),
            "source_count": len(source_hashes), "v1_inputs_reverified": chain}


def _load_frozen(data: Path) -> tuple[dict[str, Any], str]:
    if not DECLARATION.is_file():
        raise StudyError("Run v2 freeze before fitting")
    declaration = json.loads(DECLARATION.read_text())
    for name, expected in declaration["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise StudyError(f"Frozen v2 source changed: {name}")
    if Path(declaration["data_root"]).resolve() != data.resolve():
        raise StudyError("Frozen v2 data root differs from invocation")
    previous_sha = json.loads(V1_DECLARATION.read_text())["source_sha256"]
    chain = verify_v1_chain(data)
    if chain["declaration_sha256"] != declaration["predecessor_bindings"]["declaration_sha256"]:
        raise StudyError("V1 declaration hash differs from v2 frozen predecessor")
    for name, digest in declaration["source_sha256"].items():
        if name in previous_sha and previous_sha[name] != digest:
            raise StudyError(f"V1 source changed between studies: {name}")
    for key in ("fit_receipt_sha256", "score_receipt_sha256", "provenance_receipt_sha256",
                "raw_inputs_receipt_sha256", "artifact_receipt_sha256"):
        if chain[key] != declaration["predecessor_bindings"][key]:
            raise StudyError(f"V1 predecessor receipt changed after v2 freeze: {key}")
    return declaration, sha256_file(DECLARATION)


def _source_mass(blocks: list[dict[str, Any]], weights: np.ndarray) -> dict[str, float]:
    result = {}
    offset = 0
    for block in blocks:
        stop = offset + len(block["target"])
        result[block["source"]] = result.get(block["source"], 0.0) + float(weights[offset:stop].sum())
        offset = stop
    return result


def _train_metrics(target: np.ndarray, probability: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {"weighted_auc": float(roc_auc_score(target, probability, sample_weight=weights)),
            "weighted_log_loss": float(log_loss(target, probability, sample_weight=weights, labels=[0, 1]))}


def fit_scaled_head(blocks: list[dict[str, Any]], provider: str,
                    previous_head: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit mean-one weights; all preprocessing/statistics remain v1-identical."""
    blocks = [row for row in blocks if len(row["target"]) > 0]
    features = np.concatenate([row["features"] for row in blocks], axis=0)
    target = np.concatenate([row["target"] for row in blocks], axis=0).astype(np.int8)
    weights_v1 = v1._balanced_source_match_weights(blocks)
    if len(weights_v1) != len(target) or np.unique(target).size != 2:
        raise StudyError(f"Invalid TRAIN event rows for {provider}")
    n = len(weights_v1)
    weights_v2 = weights_v1 * n
    if not np.isclose(weights_v1.sum(), 1.) or not np.isclose(weights_v2.mean(), 1.):
        raise StudyError("Weight normalization did not produce the frozen v1/v2 sums and means")
    transformed, transform = v1._fit_transform(features, weights_v1)
    prior = float(np.average(target, weights=weights_v1))
    if not 0 < prior < 1:
        raise StudyError("Natural event prevalence must lie strictly between 0 and 1")

    estimator = LogisticRegression(C=v1.LOGISTIC_C, solver="lbfgs", max_iter=v1.MAX_ITER,
                                   random_state=0, fit_intercept=True)
    estimator.fit(transformed, target, sample_weight=weights_v2)
    q_v2 = estimator.predict_proba(transformed)[:, 1]
    previous_transform = v1._transform_features(previous_head, features)
    q_v1 = previous_head["estimator"].predict_proba(previous_transform)[:, 1]
    if not np.allclose(transform["median"], previous_head["transform"]["median"], rtol=0, atol=1e-14) or not np.allclose(
            transform["mean"], previous_head["transform"]["mean"], rtol=1e-12, atol=1e-12) or not np.allclose(
            transform["scale"], previous_head["transform"]["scale"], rtol=1e-12, atol=1e-12):
        raise StudyError("Global weight scale changed the v1 weighted transform")
    if not np.isclose(prior, previous_head["natural_event_prevalence"], rtol=0, atol=1e-14):
        raise StudyError("Global weight scale changed natural event prevalence")

    source_mass_v1 = _source_mass(blocks, weights_v1)
    source_mass_v2 = _source_mass(blocks, weights_v2)
    if any(not np.isclose(source_mass_v2[s] / n, source_mass_v1[s], rtol=0, atol=1e-12)
           for s in source_mass_v1):
        raise StudyError("Global weight scale changed relative source masses")
    metrics = {
        "v1": {"coefficient_l2_norm": float(np.linalg.norm(previous_head["estimator"].coef_)),
               "weight_sum": float(weights_v1.sum()), "weight_mean": float(weights_v1.mean()),
               "event_auc_logloss": _train_metrics(target, q_v1, weights_v1)},
        "v2": {"coefficient_l2_norm": float(np.linalg.norm(estimator.coef_)),
               "weight_sum": float(weights_v2.sum()), "weight_mean": float(weights_v2.mean()),
               "event_auc_logloss": _train_metrics(target, q_v2, weights_v1)},
        "directed_rows": int(n), "directed_positive_rows": int(target.sum()),
        "natural_prevalence": prior,
        "relative_source_mass_v1": source_mass_v1,
        "relative_source_mass_v2": {s: x / n for s, x in source_mass_v2.items()},
        "global_weight_factor": n,
        "effective_C_equivalent_on_v1_normalized_weights": float(v1.LOGISTIC_C * n),
        "transform_unchanged": True, "prevalence_unchanged": True,
    }
    head = {
        "study_id": STUDY_ID, "provider_excluded": provider,
        "feature_names": list(v1.FEATURE_NAMES), "track_feature_names": list(v1.TRACK_FEATURES),
        "transform": transform, "estimator": estimator,
        "natural_event_prevalence": prior,
        "training": {"sources": sorted({row["source"] for row in blocks}),
                     "matches": [row["match"] for row in blocks],
                     "directed_edges": int(n), "directed_events": int(target.sum()),
                     "source_match_balanced_weights": True, "weight_sum": float(weights_v2.sum()),
                     "weight_mean": float(weights_v2.mean()), "weight_scale_factor": int(n),
                     "weighted_natural_prevalence": prior, "class_balancing": False,
                     "event_undersampling": False,
                     "target": "exact directed H/A change across adjacent eligible native-checked 5 Hz rows",
                     "training_is_in_sample_for_backbone": True,
                     "training_backbone_provider_exclusion": provider,
                     "n_iter": int(estimator.n_iter_[0]),
                     "converged": bool(estimator.n_iter_[0] < v1.MAX_ITER)},
        "mapping": {"base_cost_10hz": v1.BASE_COST, "lower_cost_10hz": v1.COST_MIN,
                    "upper_cost_10hz": v1.COST_MAX,
                    "rule": "clip(3 - (logit(q_edge) - logit(weighted_training_event_prevalence)), 1.5, 4.5)"},
    }
    return head, metrics


def fit_all(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    declaration, declaration_sha = _load_frozen(root)
    output = root / OUT_RELATIVE
    output.mkdir(parents=True, exist_ok=True)
    if FIT_RECEIPT.exists() or any((output / f"head_{p}.joblib").exists() for p in PROVIDERS):
        raise StudyError("v2 heads and fit receipt are immutable")
    previous_heads = {p: joblib.load(root / V1_OUT / f"head_{p}.joblib") for p in PROVIDERS}
    receipts, summaries = [], {}
    for provider in PROVIDERS:
        expected = declaration["cohort"]["provider_folds"][provider]
        blocks, facts = v1.build_training_blocks(root, provider)
        if (facts["fold"]["train"] != expected["fold"]["train"] or
                facts["training_cache_sha256"] != expected["train_cache_sha256"] or
                facts["checkpoint_facts"] != expected["checkpoint_facts"]):
            raise StudyError(f"TRAIN cache/fold/checkpoint facts changed for {provider}")
        previous_fit = json.loads(V1_FIT.read_text())
        previous_row = next(row for row in previous_fit["heads"] if row["provider_excluded"] == provider)
        if facts["train_probability_facts"] != previous_row["train_probability_facts"]:
            raise StudyError(f"Frozen train backbone predictions changed for {provider}")
        head, scale_metrics = fit_scaled_head(blocks, provider, previous_heads[provider])
        head["scale_diagnostic"] = scale_metrics
        path = output / f"head_{provider}.joblib"
        joblib.dump(head, path, compress=3)
        row = {"provider_excluded": provider, "path": str(path.relative_to(root)),
               "sha256": sha256_file(path), "fit": head["training"],
               "scale_diagnostic": scale_metrics,
               "training_cache_sha256": facts["training_cache_sha256"],
               "checkpoint_facts": facts["checkpoint_facts"],
               "train_probability_facts": facts["train_probability_facts"],
               "training_blocks": [{k: v for k, v in item.items() if k not in ("features", "target")}
                                   for item in blocks]}
        receipts.append(row)
        summaries[provider] = {"model_sha256": row["sha256"], "fit": head["training"],
                               "scale_diagnostic": scale_metrics}
    receipt = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
               "source_sha256": declaration["source_sha256"],
               "predecessor_score_receipt_sha256": declaration["predecessor_bindings"]["score_receipt_sha256"],
               "fitted_utc": datetime.now(timezone.utc).isoformat(),
               "fit_count": 3, "heads": receipts,
               "scores_read_before_fit_seal": False}
    FIT_RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return {"fit_receipt": str(FIT_RECEIPT.relative_to(ROOT)),
            "fit_receipt_sha256": sha256_file(FIT_RECEIPT), "heads": summaries}


def seal(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    declaration, declaration_sha = _load_frozen(root)
    if not FIT_RECEIPT.is_file():
        raise StudyError("Fit all three v2 heads before sealing")
    fit = json.loads(FIT_RECEIPT.read_text())
    if (fit.get("declaration_sha256") != declaration_sha or fit.get("fit_count") != 3 or
            fit.get("source_sha256") != declaration["source_sha256"]):
        raise StudyError("v2 fit receipt differs from frozen declaration")
    output = root / OUT_RELATIVE
    heads = {}
    for provider, item in zip(PROVIDERS, fit["heads"]):
        path = output / f"head_{provider}.joblib"
        if item["provider_excluded"] != provider or sha256_file(path) != item["sha256"]:
            raise StudyError(f"v2 head changed before sealing: {provider}")
        heads[provider] = sha256_file(path)
    seal = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
            "fit_receipt_sha256": sha256_file(FIT_RECEIPT), "head_sha256": heads,
            "sealed_utc": datetime.now(timezone.utc).isoformat(),
            "before_evaluation_score_read": True}
    seal_path = output / "model_seal.json"
    if seal_path.exists():
        raise StudyError("v2 model seal is immutable")
    seal_path.write_text(json.dumps(seal, indent=2) + "\n")
    return {"model_seal": str(seal_path.relative_to(root)), "model_seal_sha256": sha256_file(seal_path)}


def _verify_v2_seal(root: Path, declaration: dict[str, Any], declaration_sha: str) -> dict[str, dict]:
    output = root / OUT_RELATIVE
    seal_path = output / "model_seal.json"
    if not seal_path.is_file() or not FIT_RECEIPT.is_file():
        raise StudyError("v2 scoring requires a sealed three-head model")
    seal = json.loads(seal_path.read_text())
    fit = json.loads(FIT_RECEIPT.read_text())
    if (seal.get("declaration_sha256") != declaration_sha or
            seal.get("fit_receipt_sha256") != sha256_file(FIT_RECEIPT) or
            fit.get("declaration_sha256") != declaration_sha):
        raise StudyError("v2 seal/fit/declaration mismatch")
    heads = {}
    for provider in PROVIDERS:
        path = output / f"head_{provider}.joblib"
        if sha256_file(path) != seal["head_sha256"].get(provider):
            raise StudyError(f"v2 head differs from seal: {provider}")
        heads[provider] = joblib.load(path)
    return heads


def _score_row(labels: np.ndarray, prediction: np.ndarray, inputs: dict,
               cell: dict[str, str], probability: np.ndarray,
               probability_file_sha: str | None = None) -> dict[str, Any]:
    return v1.score_prediction(labels, prediction, inputs, cell, probability, probability_file_sha)


def score_all(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    declaration, declaration_sha = _load_frozen(root)
    heads = _verify_v2_seal(root, declaration, declaration_sha)
    if SCORE_RECEIPT.exists():
        raise StudyError("v2 score receipt is immutable")
    outputs = root / OUT_RELATIVE
    previous_rows = v1._baseline_score_rows(root)
    frozen_cells = {(r["cell"]["source"], r["cell"]["match"], r["cell"]["view"], r["cell"]["replicate"]): r
                    for r in declaration["cohort"]["cells"]}
    rows, component_rows = [], []
    for cell in v1.cells():
        key = (cell["source"], cell["match"], cell["view"], cell["replicate"])
        current = v1._verify_eval_cell_snapshot(root, cell)
        if current != frozen_cells[key]:
            raise StudyError(f"v1 frozen evaluation snapshot changed before v2 scoring: {key}")
        paths = v1._record_paths_for_cell(cell, root)
        cache = joblib.load(paths["inputs_labels_record"])
        labels = np.asarray(cache["labels"]).astype(str)
        inputs, record = cache["inputs"], cache["record"]
        probability, members = v1._load_member_probabilities(root, cell)
        with np.load(paths["candidate_ensemble"], allow_pickle=False) as file:
            if not np.array_equal(file["probability"], probability):
                raise StudyError(f"Evaluation ensemble mismatch: {key}")
            archived_pred = file["prediction"].astype("U1")
        blocks = v1.supervised.model_periods(inputs)
        fps = float(inputs["fps"])
        baseline_pred = v1.supervised.decode(probability, blocks, fps, v1.BASE_COST)
        if not np.array_equal(baseline_pred.astype("U1"), archived_pred):
            raise StudyError(f"Cost-3 baseline emission/decode mismatch: {key}")
        baseline = _score_row(labels, baseline_pred, inputs, cell, probability,
                              v1.sha256_file(paths["candidate_ensemble"]))
        v1._verify_baseline_row(baseline, previous_rows[key], key)

        edge_costs, diag = v1.event_costs_for_native(heads[cell["source"]], record, probability)
        predicted = v1.decode_dynamic(probability, blocks, fps, edge_costs)
        candidate = _score_row(labels, predicted, inputs, cell, probability,
                               v1.sha256_file(paths["candidate_ensemble"]))
        event_costs = np.concatenate([diag["cost_home_away"], diag["cost_away_home"]]).astype(float)
        event_costs = event_costs[np.isfinite(event_costs)]
        candidate.update({"recipe": "turnover_v2_mean_one_weights",
                          "edge_cost_diagnostics": {
                              "q_home_away_mean": float(np.mean(diag["q_home_away"])) if diag["pairs"] else None,
                              "q_away_home_mean": float(np.mean(diag["q_away_home"])) if diag["pairs"] else None,
                              "cost_mean_home_away": diag["cost_mean_home_away"],
                              "cost_mean_away_home": diag["cost_mean_away_home"],
                              "event_cost_min": float(np.min(event_costs)) if event_costs.size else v1.BASE_COST,
                              "event_cost_max": float(np.max(event_costs)) if event_costs.size else v1.BASE_COST,
                              "cost_clipped_fraction": diag["cost_clipped_fraction"],
                              "eligible_sampled_edge_intervals": len(diag["pairs"]),
                              "native_costs_sha256": v1.sha256_array(edge_costs)}})
        baseline_file = outputs / f"baseline_{cell['match']}_{cell['view']}_{cell['replicate']}.npz"
        candidate_file = outputs / f"candidate_{cell['match']}_{cell['view']}_{cell['replicate']}.npz"
        np.savez_compressed(baseline_file, probability=probability,
                            prediction=baseline_pred.astype("U1"))
        np.savez_compressed(candidate_file, probability=probability,
                            prediction=predicted.astype("U1"), edge_costs=edge_costs)
        baseline["probability_file"] = baseline_file.name
        baseline["probability_file_sha256"] = v1.sha256_file(baseline_file)
        candidate["probability_file"] = candidate_file.name
        candidate["probability_file_sha256"] = v1.sha256_file(candidate_file)
        rows.extend((baseline, candidate))
        for seed in SEEDS:
            seed_baseline = v1.supervised.decode(members[seed], blocks, fps, v1.BASE_COST)
            seed_candidate = v1.decode_dynamic(members[seed], blocks, fps, edge_costs)
            component_rows.append({**_score_row(labels, seed_baseline, inputs, cell, members[seed]),
                                   "component_seed": seed, "recipe": f"cost3_seed_{seed}"})
            component_rows.append({**_score_row(labels, seed_candidate, inputs, cell, members[seed]),
                                   "component_seed": seed, "recipe": f"shared_head_seed_{seed}"})

    baseline = [r for r in rows if r.get("recipe") != "turnover_v2_mean_one_weights"]
    candidate = [r for r in rows if r.get("recipe") == "turnover_v2_mean_one_weights"]
    if len(baseline) != 15 or len(candidate) != 15 or len(component_rows) != 60:
        raise StudyError("v2 evaluation count differs from frozen cohort")
    baseline_aggregate, candidate_aggregate = v1.aggregate_rows(baseline), v1.aggregate_rows(candidate)
    component_aggregate = {
        str(seed): {recipe: v1.aggregate_rows([r for r in component_rows
                                               if r["component_seed"] == seed and r["recipe"] == recipe])
                    for recipe in (f"cost3_seed_{seed}", f"shared_head_seed_{seed}")}
        for seed in SEEDS}
    delta = {}
    for view in ("clean", "feed_like"):
        delta[view] = {k: candidate_aggregate[view]["equal_provider_mean"][k] -
                         baseline_aggregate[view]["equal_provider_mean"][k]
                       for k in candidate_aggregate[view]["equal_provider_mean"]}
        delta[view]["per_provider"] = {
            p: {k: candidate_aggregate[view]["per_provider"][p][k] -
                  baseline_aggregate[view]["per_provider"][p][k]
                for k in candidate_aggregate[view]["per_provider"][p]}
            for p in PROVIDERS}
    checks = {"clean_spell_plus_0_01": delta["clean"]["spell_f1"] >= .01,
              "feed_like_spell_no_worse_than_minus_0_01": delta["feed_like"]["spell_f1"] >= -.01,
              "each_provider_clean_macro_within_0_02": all(delta["clean"]["per_provider"][p]["macro_f1"] >= -.02 for p in PROVIDERS),
              "each_provider_clean_d_f1_within_0_02": all(delta["clean"]["per_provider"][p]["d_f1"] >= -.02 for p in PROVIDERS)}
    aggregate = {"baseline": baseline_aggregate, "candidate": candidate_aggregate,
                 "component_seed_diagnostics": component_aggregate,
                 "candidate_minus_baseline": delta,
                 "predeclared_guardrails": {"checks": checks, "passes": all(checks.values()),
                                            "automatic_promotion": False}}
    run = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
           "model_seal_sha256": sha256_file(outputs / "model_seal.json"),
           "scored_utc": datetime.now(timezone.utc).isoformat(),
           "rows": rows, "component_seed_rows": component_rows, "aggregate": aggregate}
    run_path = outputs / "runs.json"
    if run_path.exists():
        raise StudyError("v2 runs output is immutable")
    run_path.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    receipt = {"study_id": STUDY_ID, "declaration_sha256": declaration_sha,
               "model_seal_sha256": sha256_file(outputs / "model_seal.json"),
               "run_path": str(run_path.relative_to(root)), "run_sha256": sha256_file(run_path),
               "scored_cells": len(candidate), "baseline_rows": len(baseline),
               "candidate_rows": len(candidate), "component_seed_rows": len(component_rows),
               "source_sha256": declaration["source_sha256"], "no_organizer_scoring": True,
               "automatic_promotion": False, "scores_read_after_all_heads_sealed": True,
               "aggregate": aggregate}
    SCORE_RECEIPT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def artifact_receipt(data: Path | None = None) -> dict[str, Any]:
    root = data_root(data)
    artifacts = {}
    for path in (DECLARATION, FIT_RECEIPT, SCORE_RECEIPT):
        if path.is_file():
            artifacts[path.name] = {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)}
    output = root / OUT_RELATIVE
    if output.is_dir():
        artifacts["private_outputs"] = {
            p.name: {"path": str(p.relative_to(root)), "bytes": p.stat().st_size,
                     "sha256": sha256_file(p)} for p in sorted(output.iterdir()) if p.is_file()}
    for path in (Path("experiments/turnover_decoder_v2.py"),
                 Path("experiments/tests/test_turnover_decoder_v2.py"),
                 Path("experiments/results/turnover_decoder_declaration.json"),
                 Path("experiments/results/turnover_decoder_fit_receipt.json"),
                 Path("experiments/results/turnover_decoder_score_receipt.json"),
                 Path("experiments/results/turnover_decoder_provenance_receipt_v2.json"),
                 Path("experiments/results/short_spell_raw_inputs_verification.json"),
                 Path("TURNOVER_DECODER.md")):
        if (ROOT / path).is_file():
            artifacts[str(path)] = {"path": str(path), "bytes": (ROOT / path).stat().st_size,
                                    "sha256": sha256_file(ROOT / path)}
    payload = {"study_id": STUDY_ID, "created_utc": datetime.now(timezone.utc).isoformat(),
               "artifacts": artifacts, "private_artifacts_are_local_only": True}
    ARTIFACT_RECEIPT.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "seal", "score", "receipt"))
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    actions = {"freeze": freeze, "fit": fit_all, "seal": seal,
               "score": score_all, "receipt": artifact_receipt}
    print(json.dumps(actions[args.command](args.data_dir), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
