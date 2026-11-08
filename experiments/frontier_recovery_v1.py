"""Rebuild per-cell native probabilities for the selected spell frontier."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib, json, os
from pathlib import Path
import joblib
import numpy as np
from experiments import frontier_recovery_training_v1 as training
from experiments import frontier_recovery_execution_v1 as execution
from experiments import frontier_recovery_serial_v1 as serial
from experiments import frontier_input_prepare_v1 as input_preparation
from experiments import v2_data_capacity_eval as capacity
from experiments import v2_spell_balanced_eval as balanced_eval
from experiments import v2_iterative_train as iterative
from experiments import v2_spell_balanced_train as balanced_train
from experiments import v2_spell_target_runtime as model_runtime
from experiments import sn_dense_train as sn
from experiments import soccernet_extension as old
from experiments import feed_spell_push
from experiments import tracking_corruption

ROOT=training.ROOT; STUDY=training.STUDY; OUT=training.OUT
TRAIN_DECL=training.TRAIN_DECL; FIT_SEAL=training.FIT_SEAL
CELL_MANIFEST=OUT / "frontier_native_probability_manifest.json"
FIT_AUDIT=OUT / "training_reconstruction_audit.json"
SOURCE_SUPPLEMENT=OUT / "source_supplement.json"
EXECUTION_RECEIPT=OUT / "execution_receipt.json"
SERIAL_AMENDMENT=OUT / "serial_execution_amendment.json"
INDEPENDENT_FIT_AUDIT=OUT / "independent_training_audit.json"
SEEDS=training.SEEDS; VIEWS=training.VIEWS; AUDIT=training.AUDIT
BASELINE_RECIPE=training.BASELINE_RECIPE; recipe=training.recipe
sha=training.sha; _write_once=training._write_once
_configure=training._configure; _stage_path=training._stage_path
prepare_training=training.prepare_training; freeze_training=training.freeze_training
fit=training.fit
model_runtime.frozen.tc=tracking_corruption

def _worker_init():
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

def _cell_key(provider: str, match: str, view: str, rep: int) -> str:
    return f"{provider}|{match}|{view}|{rep}"


def _graded_native_mask(inputs: dict) -> np.ndarray:
    """Native label periods are authoritative; model_period may start at zero."""
    period = np.asarray(inputs["period"])
    if period.ndim != 1:
        raise ValueError("Expected aligned one-dimensional native period")
    return period > 0


def _eval_source_hashes() -> dict:
    paths = [Path(__file__), Path(capacity.__file__), Path(model_runtime.__file__),
             Path(model_runtime.frozen.__file__), Path(balanced_eval.__file__),
             Path(balanced_eval.controls.__file__), Path(balanced_eval.controls.audit.__file__),
             Path(balanced_eval.controls.audit.decoder.__file__),
             Path(balanced_eval.controls.audit.metrics.__file__),
             Path(feed_spell_push.__file__), Path(sn.__file__), Path(old.__file__),
             Path(iterative.cap.__file__), Path(iterative.__file__),
             Path(AUDIT.__file__), Path(recipe.prep.__file__), Path(recipe.ms.__file__),
             Path(recipe.v2.__file__), Path(recipe.rf.__file__),
             ROOT / "experiments/supervised.py", ROOT / "experiments/tracking_corruption.py",
             ROOT / "experiments/feed_like_v2.py", ROOT / "experiments/refine_features.py",
             ROOT / "experiments/learned_temporal.py", ROOT / "evaluation/possession_metrics.py",
             ROOT / "experiments/results/v2_spell_frontier_v1_round1_declaration.json",
             ROOT / "experiments/results/v2_spell_frontier_v1_round1.json",
             ROOT / "experiments/results/v2_spell_frontier_v1_replay.json",
             ROOT / "experiments/results/v2_spell_frontier_v1_selection.json",
             ROOT / "data/experiments/complementary_spell_v1/native_probability/preparation_manifest.json",
             ROOT / "experiments/frontier_recovery_training_v1.py", TRAIN_DECL, FIT_SEAL,
             ROOT / "experiments/frontier_recovery_execution_v1.py",
             ROOT / "experiments/frontier_recovery_serial_v1.py",
             ROOT / "experiments/frontier_input_prepare_v1.py",
             input_preparation.DECLARATION, input_preparation.MANIFEST,
             FIT_AUDIT, SOURCE_SUPPLEMENT, EXECUTION_RECEIPT, SERIAL_AMENDMENT,
             INDEPENDENT_FIT_AUDIT]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def _predict_cell(job: tuple[str, str, str, int]) -> tuple[str, dict, dict]:
    provider, match, view, rep = job
    _configure()
    prepared = input_preparation.load(_cell_key(provider, match, view, rep))
    raw_path = Path(prepared["path"])
    inputs, labels, phases, model_period = (prepared["inputs"], prepared["labels"],
                                             prepared["phases"], prepared["model_period"])
    if len(phases) not in (2, 5):
        raise ValueError(f"Unexpected native phase count for {provider}/{match}: {len(phases)}")
    key = _cell_key(provider, match, view, rep)
    cell = _load_predecessor_cell(provider, match, view, rep)
    input_manifest_sha = sha(input_preparation.MANIFEST)
    input_declaration_sha = sha(input_preparation.DECLARATION)
    input_fact = json.loads(input_preparation.MANIFEST.read_text())["cells"][key]
    # Fold aliases are fixed; use the canonical audit mapping for this provider.
    fold = f"leave_out_{AUDIT.FOLD[provider]}"
    ce_models = {seed: joblib.load(_stage_path(1024, fold, seed)) for seed in SEEDS}
    teachers = {seed: joblib.load(recipe.rf.sc.initial_path(fold, seed)) for seed in SEEDS}
    per_family = {name: {seed: [] for seed in SEEDS}
                  for name in ("ce", "calibrated", "predecessor")}
    for phase_idx, phase_record in enumerate(phases):
        for seed in SEEDS:
            record = old.inference_record(phase_record)
            record["names"] = list(record["names"])
            raw_ce = sn.predict(record, ce_models[seed], teachers[seed])
            ce = old.upsample(raw_ce, record, inputs)
            per_family["ce"][seed].append(ce)
            calibrated = model_runtime.frozen.predict_seed(inputs, phase_record, "adop", provider, seed)
            per_family["calibrated"][seed].append(calibrated)
            # Reuse the hash-verified original_full phase probabilities and D
            # mask prepared from these exact restored model sources earlier.
            previous = cell["arrays"][f"prob__candidate__seed{seed}__phase{phase_idx}"]
            per_family["predecessor"][seed].append(previous)
    arrays = {"labels": labels, "period": np.asarray(inputs["period"]),
              "timestamps": np.asarray(inputs["timestamps"]), "model_period": model_period,
              "fps": np.asarray(float(inputs["fps"]))}
    for field in ("labels", "period", "timestamps", "model_period", "fps"):
        if not np.array_equal(np.asarray(arrays[field]), np.asarray(cell["arrays"][field])):
            raise ValueError(f"Predecessor/native metadata mismatch for {key}: {field}")
    pooled = {}
    for family, by_seed in per_family.items():
        phase_means = []
        for seed in SEEDS:
            for phase_idx, probability in enumerate(by_seed[seed]):
                if probability.shape != (len(labels), 3):
                    raise ValueError(f"Misaligned native probabilities: {family}/{seed}/{phase_idx}")
                arrays[f"prob__{family}__seed{seed}__phase{phase_idx}"] = probability
        for phase_idx in range(len(phases)):
            phase_means.append(np.mean([by_seed[seed][phase_idx] for seed in SEEDS], axis=0))
        pooled[family] = np.mean(phase_means, axis=0)
    adopted_decoded = balanced_eval.controls.audit.decode(
        pooled["predecessor"], model_period, float(inputs["fps"]),
        balanced_eval.controls.audit.CONFIG)
    historical_decode = cell["arrays"]["decoded__historical_f062"].astype("U1", copy=False)
    graded = _graded_native_mask(inputs)
    if not np.array_equal(adopted_decoded[graded], historical_decode[graded]):
        raise ValueError(f"Restored predecessor probabilities differ from historical f062 on graded rows: {key}")
    ungraded_differences = int(np.count_nonzero(adopted_decoded[~graded] != historical_decode[~graded]))
    frontier_prob = balanced_eval.recipe_probability(
        {"ce": pooled["ce"], "calibrated": pooled["calibrated"], "adopted": pooled["predecessor"]},
        BASELINE_RECIPE["ce_weight"], BASELINE_RECIPE["calibrated_weight"], "arithmetic")
    frontier_pred = balanced_eval.decode_fixed_d(frontier_prob, adopted_decoded,
        model_period, float(inputs["fps"]), BASELINE_RECIPE["ha_cost"])
    arrays["prob__approved_baseline"] = frontier_prob
    arrays["decoded__predecessor"] = adopted_decoded
    arrays["decoded__approved_baseline"] = frontier_pred
    identity = {
        "cell": key, "provider": provider, "match": match, "view": view, "rep": rep,
        "source_status": "reconstructed", "source_study_id": "v2_spell_frontier_v1",
        "frontier_study_id": "v2_spell_frontier_v1",
        "frontier_declaration_sha256": sha(ROOT / "experiments/results/v2_spell_frontier_v1_round1_declaration.json"),
        "frontier_result_sha256": sha(ROOT / "experiments/results/v2_spell_frontier_v1_round1.json"),
        "frontier_replay_sha256": sha(ROOT / "experiments/results/v2_spell_frontier_v1_replay.json"),
        "selection_sha256": sha(ROOT / "experiments/results/v2_spell_frontier_v1_selection.json"),
        "input_sha256": sha(raw_path), "phase_count": len(phases),
        "view": view, "rep": rep,
        "corruption_seed": (int(hashlib.sha256(f"feed_like_v2_benchmark/{rep}/{match}".encode()).hexdigest()[:8], 16)
                            if view == "feed_like_v2" else None),
        "predecessor_cache_sha256": cell["fact"]["sha256"],
        "prepared_input_cache_sha256": input_fact["cache_sha256"],
        "input_preparation_manifest_sha256": input_manifest_sha,
        "input_preparation_declaration_sha256": input_declaration_sha,
        "pilot_ungraded_f062_x_difference_count": ungraded_differences,
        "predecessor_manifest_sha256": cell["manifest_sha256"],
        "model_manifest_sha256": str(cell["arrays"]["identity__model_manifest_sha256"].item()),
        "training_declaration_sha256": sha(TRAIN_DECL),
        "training_seal_sha256": sha(FIT_SEAL),
        "training_audit_sha256": sha(FIT_AUDIT),
        "independent_training_audit_sha256": sha(INDEPENDENT_FIT_AUDIT),
        "training_source_supplement_sha256": sha(SOURCE_SUPPLEMENT),
        "training_execution_receipt_sha256": sha(EXECUTION_RECEIPT),
        "serial_execution_amendment_sha256": sha(SERIAL_AMENDMENT),
        "reconstruction_provenance": json.dumps({"stages": [
            {"updates": 512, "learning_rate": 3e-4},
            {"updates": 512, "learning_rate": 1e-4}],
            "fold_seed_pairs": [f"{fold}/{seed}" for seed in SEEDS]}, sort_keys=True),
        "calibration_sha256": sha(recipe.CALIBRATION),
    }
    for seed in SEEDS:
        identity[f"ce_stage512_seed{seed}_sha256"] = sha(_stage_path(512, fold, seed))
        identity[f"ce_stage1024_seed{seed}_sha256"] = sha(_stage_path(1024, fold, seed))
        identity[f"ce_stage512_seed{seed}_receipt_sha256"] = sha(
            Path(str(_stage_path(512, fold, seed)) + ".json"))
        identity[f"ce_stage1024_seed{seed}_receipt_sha256"] = sha(
            Path(str(_stage_path(1024, fold, seed)) + ".json"))
        identity[f"calibrated_model_seed{seed}_sha256"] = sha(
            model_runtime.frozen._model_path("adop", provider, seed))
        identity[f"predecessor_model_seed{seed}_sha256"] = sha(
            model_runtime.frozen._model_path("cand", provider, seed))
        identity[f"teacher_seed{seed}_sha256"] = sha(recipe.rf.sc.initial_path(fold, seed))
    for name, value in identity.items():
        arrays[f"identity__{name}"] = np.asarray(str(value))
    metadata = {"key": key, "identity": identity, "phase_count": len(phases),
                "rows": len(labels), "input_path": str(raw_path)}
    return key, arrays, metadata


def _load_predecessor_cell(provider: str, match: str, view: str, rep: int) -> dict:
    manifest_path = ROOT / "data/experiments/complementary_spell_v1/native_probability/preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    fact_key = f"{provider}|{match}|{view}|{rep}"
    fact = manifest["cells"].get(fact_key)
    if not fact:
        raise KeyError(f"Missing predecessor cell in verified preparation manifest: {fact_key}")
    path = ROOT / fact["path"]
    if not path.is_file() or sha(path) != fact["sha256"]:
        raise ValueError(f"Historical predecessor probability cache changed: {fact_key}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    saved_identity = {name.removeprefix("identity__"): str(arrays[name].item())
                      for name in arrays if name.startswith("identity__")}
    required_identity = ("cell", "provider", "match", "view", "rep", "phase_count",
                         "input_sha256", "model_manifest_sha256", "source_study_id")
    if any(name not in saved_identity or
           saved_identity[name] != str(fact.get("identity", {}).get(name))
           for name in required_identity):
        raise ValueError(f"Predecessor cache payload identity differs from manifest: {fact_key}")
    return {"arrays": arrays, "fact": fact, "manifest_sha256": sha(manifest_path)}


def _verify_execution_chain() -> dict:
    """Validate the frozen recovery declaration, supplement, serial amendment, and receipt."""
    training._check_training_binding()
    execution.check_sources()
    amendment = serial._validate_frozen()
    receipt = json.loads(EXECUTION_RECEIPT.read_text())
    expected = {
        "schema": "frontier_recovery_execution_v1_serial_amendment",
        "study_id": training.STUDY,
        "serial_amendment_sha256": sha(SERIAL_AMENDMENT),
        "source_supplement_sha256": sha(SOURCE_SUPPLEMENT),
        "training_declaration_sha256": sha(TRAIN_DECL),
        "training_fit_seal_sha256": sha(FIT_SEAL),
        "fit_count": 12,
        "execution_mode": "sequential_fresh_process_per_fit",
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise ValueError("Training execution receipt does not bind the verified serial reconstruction chain")
    fit_seal = json.loads(FIT_SEAL.read_text())
    if fit_seal.get("all_provider_folds_complete") is not True or len(fit_seal.get("fits", [])) != 12:
        raise ValueError("Fit seal is incomplete for the frozen serial job inventory")
    independent = json.loads(INDEPENDENT_FIT_AUDIT.read_text())
    required_true = ("model_and_sidecar_hashes_verified_for_all_12",
                     "sampling_view_draws_exposures_exact_for_all_12",
                     "teacher_identity_and_warmstart_pairing_verified_for_all_12",
                     "D_state_normalization_and_feature_schema_preserved_for_all_12")
    if (independent.get("schema") != "frontier_recovery_independent_fit_audit_v1" or
            independent.get("status") != "all_12_reconstructed_fit_artifacts_independently_verified_before_native_replay" or
            independent.get("fit_count") != 12 or independent.get("expected_fit_count") != 12 or
            independent.get("training_declaration_sha256") != sha(TRAIN_DECL) or
            independent.get("training_seal_sha256") != sha(FIT_SEAL) or
            independent.get("execution_receipt_sha256") != sha(EXECUTION_RECEIPT) or
            independent.get("serial_amendment_sha256") != sha(SERIAL_AMENDMENT) or
            independent.get("source_supplement_sha256") != sha(SOURCE_SUPPLEMENT) or
            any(independent.get(name) is not True for name in required_true)):
        raise ValueError("Independent all-12 fit audit does not bind the verified recovery chain")
    return {"amendment": amendment, "receipt": receipt, "independent": independent}


def prepare_cells(workers: int = 2) -> dict:
    _configure()
    if workers not in (1, 2):
        raise ValueError("Use one or two workers")
    if not FIT_SEAL.is_file():
        raise FileNotFoundError("Complete both training stages before preparing frontier cells")
    fit_seal = json.loads(FIT_SEAL.read_text())
    if fit_seal.get("all_provider_folds_complete") is not True or len(fit_seal["fits"]) != 12:
        raise ValueError("Recovery fit seal is incomplete")
    execution_chain = _verify_execution_chain()
    training_audit = verify_training_reconstruction()
    source_snapshot = _eval_source_hashes()
    jobs = [(provider, match, view, rep)
            for provider, match, _ in AUDIT.MATCHES
            for view, reps in VIEWS for rep in reps]
    cells = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
        futures = {pool.submit(_predict_cell, job): job for job in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            key, arrays, metadata = future.result()
            path = OUT / "cells" / f"{key.replace('|', '__')}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite recovered cell cache: {path}")
            temp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
            np.savez_compressed(temp, **arrays)
            os.replace(temp, path)
            cell_fact = {"path": str(path.relative_to(ROOT)), "sha256": sha(path),
                         "phase_count": metadata["phase_count"], "identity": metadata["identity"]}
            cells[key] = cell_fact
            print(f"RECOVERED CELL {done}/{len(jobs)} {key}", flush=True)
    frontier_paths = {
        "declaration": ROOT / "experiments/results/v2_spell_frontier_v1_round1_declaration.json",
        "result": ROOT / "experiments/results/v2_spell_frontier_v1_round1.json",
        "replay": ROOT / "experiments/results/v2_spell_frontier_v1_replay.json",
        "selection": ROOT / "experiments/results/v2_spell_frontier_v1_selection.json",
    }
    frontier_artifacts = {name: {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
                          for name, path in frontier_paths.items()}
    round1 = json.loads(frontier_paths["result"].read_text())
    per_cell = {key: round1["cell_metrics"][key][BASELINE_RECIPE["recipe_id"]]
                for key in sorted(cells)}
    agg = round1["aggregate"][BASELINE_RECIPE["recipe_id"]]
    baseline_metrics = {view: {metric: float(values[metric]["equal_provider"])
                               for metric in ("spell_f1", "macro_f1")}
                        for view, values in agg.items()}
    sources = _eval_source_hashes()
    if sources != source_snapshot:
        raise ValueError("Native inference source closure changed during cell preparation")
    data = {"schema": "frontier_postprocess_v1_cache_manifest",
            "source_status": "reconstructed", "source_study_id": "v2_spell_frontier_v1",
            "evidence_scope": "reconstructed current selected frontier control; no new candidate scores",
            "frontier_study_id": "v2_spell_frontier_v1", "frontier_artifacts": frontier_artifacts,
            "frontier_result_sha256": frontier_artifacts["result"]["sha256"],
            "frontier_cell_metrics": per_cell,
            "frontier_cell_metric_source": frontier_artifacts["result"],
            "source_hashes": sources,
            "baseline_recipe": BASELINE_RECIPE,
            "family_weights": {"ce": 0.5, "calibrated": 0.375, "predecessor": 0.125},
            "pooling_order": "paired seeds within native phase; equal phase mean; arithmetic conditional H/A family mix",
            "seeds": list(SEEDS), "expected_cells": sorted(cells),
            "cohort": {"matches": [[p, m] for p, m, _ in AUDIT.MATCHES],
                       "view_replicates": {"clean": [0], "feed_like_v1": [1, 2],
                                           "feed_like_v2": [1, 2, 3, 4]}},
            "baseline_metrics": baseline_metrics, "cells": cells,
            "ungraded_f062_x_difference_counts": {
                key: int(fact["identity"].get("pilot_ungraded_f062_x_difference_count", 0))
                for key, fact in cells.items()
                if fact["identity"].get("pilot_ungraded_f062_x_difference_count", 0)},
            "training_declaration": {"path": str(TRAIN_DECL.relative_to(ROOT)), "sha256": sha(TRAIN_DECL)},
            "training_seal": {"path": str(FIT_SEAL.relative_to(ROOT)), "sha256": sha(FIT_SEAL)},
            "training_audit": {"path": str(FIT_AUDIT.relative_to(ROOT)), "sha256": sha(FIT_AUDIT)},
            "training_source_supplement": {"path": str(SOURCE_SUPPLEMENT.relative_to(ROOT)), "sha256": sha(SOURCE_SUPPLEMENT)},
            "training_execution_receipt": {"path": str(EXECUTION_RECEIPT.relative_to(ROOT)), "sha256": sha(EXECUTION_RECEIPT)},
            "serial_execution_amendment": {"path": str(SERIAL_AMENDMENT.relative_to(ROOT)), "sha256": sha(SERIAL_AMENDMENT)},
            "independent_training_audit": {"path": str(INDEPENDENT_FIT_AUDIT.relative_to(ROOT)),
                                           "sha256": sha(INDEPENDENT_FIT_AUDIT)},
            "training_source_supplement_details": execution_chain["amendment"],
            "training_execution_mode": execution_chain["receipt"]["execution_mode"],
            "independent_training_audit_summary": {
                "status": execution_chain["independent"]["status"],
                "fit_count": execution_chain["independent"]["fit_count"],
                "original_checkpoint_file_sha_matches": execution_chain["independent"]["original_checkpoint_file_sha_matches"],
                "original_checkpoint_file_sha_mismatches": execution_chain["independent"]["original_checkpoint_file_sha_mismatches"],
                "max_loss_interval_abs_difference": execution_chain["independent"]["max_loss_interval_abs_difference"]},
            "prepared_input_manifest": {"path": str(input_preparation.MANIFEST.relative_to(ROOT)),
                                        "sha256": sha(input_preparation.MANIFEST)},
            "prepared_input_declaration": {"path": str(input_preparation.DECLARATION.relative_to(ROOT)),
                                           "sha256": sha(input_preparation.DECLARATION)},
            "calibration_sha256": sha(recipe.CALIBRATION),
            "reconstruction_provenance": {
                "source_status": "reconstructed",
                "ce_training": [
                    {"stage": 512, "updates": 512, "learning_rate": 3e-4,
                     "initialization": "archived original_full fold checkpoint and paired teacher"},
                    {"stage": 1024, "updates": 512, "learning_rate": 1e-4,
                     "initialization": "same-fold/seed reconstructed stage512 checkpoint"}],
                "fold_seed_pairs": [f"{fold}/{seed}" for fold in training.FOLDS for seed in SEEDS],
                "calibrated_predecessor": "restored and private-manifest verified models; predecessor probabilities reused from verified original_full phase cache",
                "historical_exact_artifact_identity_claim": False,
                "fit_count": 12},
            "source_status_notes": "Reconstructed CE stage1024 fold experts. The historical calibrated experts and original_full per-phase probabilities were restored/hash-verified; the prior native cache is reused for original_full. Exact selected-frontier replay is required before candidate scoring.",
            "historical_pilot_decoder_note": "The prior complementary pilot forced ungraded SkillCorner period<=0 rows to X; the original frontier decoder did not. This cache uses the original frontier decoder unchanged. Its decoded predecessor is checked against the prior pilot only on graded rows, and ungraded differences are recorded per cell.",
            "organizer_reference_access": False, "scoring_performed": False,
            "candidate_scoring": False, "promotion": False}
    _write_once(CELL_MANIFEST, data)
    return data


def verify_training_reconstruction() -> dict:
    """Check recovered data order/exposure against both sealed historical stages."""
    fit_seal = json.loads(FIT_SEAL.read_text())
    historical_512, historical_1024 = training._original_fit_rows()
    comparisons = []
    max_loss_delta = 0.0
    max_loss_context = None
    for row in fit_seal["fits"]:
        stage = int(row["stage"])
        old_row = (historical_512 if stage == 512 else historical_1024)[(row["fold"], int(row["seed"]))]
        for field in ("sampling", "view_draws", "exposures"):
            if row[field] != old_row[field]:
                raise ValueError(f"Reconstructed stage{stage} {field} differs from the historical seal: {row['fold']}/{row['seed']}")
        old_losses = old_row.get("loss_intervals", [])
        new_losses = row.get("loss_intervals", [])
        if len(old_losses) != len(new_losses):
            raise ValueError(f"Loss interval inventory mismatch for stage{stage}/{row['fold']}/{row['seed']}")
        for old_loss, new_loss in zip(old_losses, new_losses):
            if int(old_loss["step"]) != int(new_loss["step"]):
                raise ValueError(f"Loss interval step mismatch for stage{stage}/{row['fold']}/{row['seed']}")
            delta = abs(float(old_loss["mean_loss"]) - float(new_loss["mean_loss"]))
            if delta > max_loss_delta:
                max_loss_delta, max_loss_context = delta, f"stage{stage}/{row['fold']}/{row['seed']}/{old_loss['step']}"
        comparisons.append({"stage": stage, "fold": row["fold"], "seed": int(row["seed"]),
                            "sampling_exact": True, "view_draws_exact": True,
                            "exposures_exact": True, "loss_intervals_compared": len(old_losses)})
    receipt = {"study_id": STUDY, "status": "verified_before_frontier_control_replay",
               "training_declaration_sha256": sha(TRAIN_DECL),
               "training_seal_sha256": sha(FIT_SEAL), "fit_count": len(comparisons),
               "exact_sampling_view_draws_exposures": True,
               "max_loss_interval_abs_difference": max_loss_delta,
               "max_loss_difference_fit_interval": max_loss_context,
               "comparisons": comparisons, "candidate_scoring": False,
               "organizer_reference_access": False}
    if FIT_AUDIT.exists():
        existing = json.loads(FIT_AUDIT.read_text())
        if existing != receipt:
            raise ValueError("Existing training reconstruction audit differs from recomputed fit-seal parity")
        return existing
    training._write_once(FIT_AUDIT, receipt)
    return receipt


def _worker_init() -> None:
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare-training", "freeze-training", "fit", "prepare-cells"))
    parser.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    _configure()
    if args.action == "prepare-training":
        print(json.dumps(prepare_training(args.workers), indent=2))
    elif args.action == "freeze-training":
        print(json.dumps(freeze_training(), indent=2))
    elif args.action == "fit":
        print(json.dumps(fit(args.workers), indent=2))
    else:
        print(json.dumps(prepare_cells(args.workers), indent=2))


if __name__ == "__main__":
    main()
