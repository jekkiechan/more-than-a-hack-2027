"""Auditable pre-fit amendment and bounded scheduler for native_short_spell_v1."""
from __future__ import annotations

import concurrent.futures
import multiprocessing
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import time
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import factorized_had as fh
from experiments import native_short_spell_experiment as runner
from experiments.eval_set.protocol import sha256
from experiments import learned_temporal as lt
from experiments import native_short_spell as nss


torch.set_num_threads(1)
OUT = runner.OUT
AMENDMENT_DRAFT = OUT / "pre_fit_runtime_amendment.json"
AMENDMENT = OUT / "pre_fit_runtime_amendment_v2.json"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _patch_training_mass_guard():
    original = inspect.getsource(runner._assemble_training_data)
    needle = '''    weight_sum_by_source = {source: 0.0 for source in PROVIDERS}\n    for key, detail in row_support.items():\n        weight_sum_by_source[detail["source"]] += (\n            1.0 / (matches_per_source[detail["source"]] * len(VIEWS)))\n    if any(not np.isclose(value, 1.0, rtol=0, atol=1e-8)\n           for value in weight_sum_by_source.values()):\n'''
    replacement = '''    weight_sum_by_source = {source: 0.0 for source in PROVIDERS}\n    for key, detail in row_support.items():\n        weight_sum_by_source[detail["source"]] += (\n            1.0 / (matches_per_source[detail["source"]] * len(VIEWS)))\n    active_sources = {source for source, count in matches_per_source.items() if count > 0}\n    if any(not np.isclose(value, 1.0 if source in active_sources else 0.0,\n                          rtol=0, atol=1e-8)\n           for source, value in weight_sum_by_source.items()):\n'''
    if original.count(needle) != 1:
        raise RuntimeError("Expected exactly one frozen source-mass guard")
    effective = original.replace(needle, replacement)
    needle2 = '''    if not np.isclose(float(data["sample_weight"].sum()), len(PROVIDERS), atol=1e-5):\n'''
    replacement2 = '''    if not np.isclose(float(data["sample_weight"].sum()), len(active_sources), atol=1e-5):\n'''
    if effective.count(needle2) != 1:
        raise RuntimeError("Expected exactly one frozen total-mass guard")
    effective = effective.replace(needle2, replacement2)
    namespace = dict(runner.__dict__)
    exec(compile(effective, inspect.getsourcefile(runner._assemble_training_data), "exec"),
         namespace)
    runner._assemble_training_data = namespace["_assemble_training_data"]
    return original, effective


def _validate_mass_contract():
    declaration, declaration_sha = runner._verify_declaration()
    results = []
    for provider in runner.PROVIDERS:
        fold = runner.prep.split_plan(fold=f"leave_out_{provider}")
        active = sorted({item["source"] for item in fold["train"]})
        heldout = [source for source in runner.PROVIDERS if source not in active]
        if heldout != [provider] or len(active) != len(runner.PROVIDERS) - 1:
            raise RuntimeError(f"Unexpected provider holdout in {provider}: {active}")
        for arm in runner.ARMS:
            training = runner._assemble_training_data(provider, runner.SEEDS[0], arm,
                                                      declaration)
            mass = training["support"]["sample_weight_sum_by_provider"]
            expected = {source: (1.0 if source in active else 0.0)
                        for source in runner.PROVIDERS}
            if any(not np.isclose(mass[source], expected[source], atol=1e-7, rtol=0)
                   for source in runner.PROVIDERS):
                raise RuntimeError(f"Provider mass contract failed: {provider}/{arm}: {mass}")
            if not np.isclose(float(training["sample_weight"].sum()), 2.0,
                              atol=1e-5, rtol=0):
                raise RuntimeError(f"Total sample mass is not 2: {provider}/{arm}")
            if len(training["target"]) != len(training["sample_weight"]):
                raise RuntimeError("Target/weight rows differ")
            results.append({
                "heldout_provider": provider, "arm": arm,
                "active_sources": active, "heldout_sources_zero_mass": heldout,
                "sample_weight_sum_by_provider": mass,
                "total_sample_weight": float(training["sample_weight"].sum()),
                "training_rows": int(len(training["target"])),
                "rows_by_match_view": training["support"]["rows_by_match_view"],
            })
            del training
    return declaration_sha, results


def _prepare_one_prior(provider, seed, declaration, declaration_sha):
    torch.set_num_threads(1)
    destination = runner._prior_cache_path(provider, seed)
    receipt_path = destination.with_suffix(".json")
    if destination.exists() or receipt_path.exists():
        if not destination.is_file() or not receipt_path.is_file():
            raise ValueError(f"Partial training prior cache: {provider}/{seed}")
        receipt = runner._json(receipt_path)
        if (receipt.get("declaration_sha256") != declaration_sha or
                receipt.get("cache_sha256") != sha256(destination)):
            raise ValueError(f"Changed training prior cache: {provider}/{seed}")
        return receipt
    fold = runner.prep.split_plan(fold=f"leave_out_{provider}")
    model = runner._saved_checkpoint(provider, seed, declaration)
    arrays = {}
    support = {}
    started = time.monotonic()
    for item in fold["train"]:
        feature_path = runner._feature_cache_path(item["match"])
        if not feature_path.is_file():
            raise ValueError(f"Missing TRAIN geometry cache: {item['match']}")
        training_meta = declaration["baseline"]["training"]["matches"][item["match"]]
        cache = joblib.load(runner._data_path(training_meta["cache"]))
        if (cache.get("match") != item["match"] or
                cache.get("source") != item["source"] or
                cache.get("input_sha256") != item["sha256"]):
            raise ValueError(f"TRAIN cache identity mismatch: {item['match']}")
        with np.load(feature_path, allow_pickle=False) as feature_archive:
            full_period = feature_archive["full_period"]
            for view_name in runner.VIEWS:
                record = cache["records"][view_name]
                prediction_record = {key: record[key] for key in
                                     ("x", "names", "indices", "period", "native_n", "fps")}
                sampled = fh.predict(prediction_record, model)
                native_prob = np.column_stack([
                    lt.upsample(sampled[:, channel], prediction_record, full_period)
                    for channel in range(3)
                ])
                if (native_prob.shape != (len(full_period), 3) or
                        not np.isfinite(native_prob).all()):
                    raise ValueError(f"Invalid TRAIN prior: {item['match']}/{view_name}")
                key = f"{item['match']}__{view_name}"
                arrays[key] = nss.conditional_home_logit(native_prob).astype(np.float32)
                support[key] = {
                    "sampled_rows": int(len(sampled)),
                    "native_rows": int(len(native_prob)),
                    "prior_sha256": runner.digest_array(arrays[key]),
                }
        del cache
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".npz",
                                     dir=OUT, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    checkpoint_fact = declaration["baseline"]["fold_checkpoints"][provider][str(seed)]
    receipt = {
        "heldout_provider": provider, "backbone_seed": seed,
        "checkpoint_sha256": checkpoint_fact["sha256"],
        "declaration_sha256": declaration_sha,
        "cache_sha256": sha256(destination),
        "training_matches_only": [item["match"] for item in fold["train"]],
        "views": support, "evaluation_archives_read": False,
        "seconds": time.monotonic() - started,
        "runtime_wrapper_sha256": sha256(Path(__file__)),
    }
    runner._write_exclusive(receipt_path, receipt)
    print("NATIVE PRIOR", provider, seed, "seconds", round(receipt["seconds"], 1),
          flush=True)
    del model, arrays
    return receipt


def _prepare_priors_parallel(declaration, declaration_sha):
    tasks = [(provider, seed) for provider in runner.PROVIDERS for seed in runner.SEEDS]
    results = []
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,
                                                mp_context=context) as pool:
        future_map = {
            pool.submit(_prepare_one_prior, provider, seed, declaration, declaration_sha):
            (provider, seed) for provider, seed in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())
    if len(results) != len(runner.PROVIDERS) * len(runner.SEEDS):
        raise RuntimeError("Provider-held-out TRAIN priors are incomplete")
    return results


def _write_amendment(original, effective, declaration_sha, mass_contract,
                     prior_receipts):
    receipt = {
        "study_id": "native_short_spell_v1",
        "declaration_sha256": declaration_sha,
        "amendment_type": "pre-fit runtime validation correction and prior scheduling",
        "source_files_changed": False,
        "source_hashes_still_match_frozen_declaration": True,
        "original_function_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "effective_function_sha256": hashlib.sha256(effective.encode()).hexdigest(),
        "only_effective_guard_changes": [
            "Require each present training provider to have total mass 1 and the held-out provider to have mass 0.",
            "Require total sample mass to equal the number of present providers (2).",
        ],
        "actual_features_targets_weights_loss_and_optimizer_unchanged": True,
        "mass_contract_checks": mass_contract,
        "training_prior_receipts": [
            {"heldout_provider": item["heldout_provider"],
             "backbone_seed": item["backbone_seed"],
             "cache_sha256": item["cache_sha256"],
             "receipt_sha256": sha256(runner._prior_cache_path(
                 item["heldout_provider"], item["backbone_seed"]).with_suffix(".json"))}
            for item in sorted(prior_receipts,
                               key=lambda value: (value["heldout_provider"],
                                                  value["backbone_seed"]))
        ],
        "prior_scheduler": {
            "tasks": [f"{provider}/{seed}" for provider in runner.PROVIDERS
                       for seed in runner.SEEDS],
            "worker_count": 3, "torch_threads_per_worker": 1,
            "disjoint_output_files": True,
            "only_training_folds_read": True,
        },
        "runtime_wrapper_sha256": sha256(Path(__file__)),
        "supersedes_amendment_sha256": sha256(AMENDMENT_DRAFT),
        "fits_started": False, "candidate_scores_opened": False,
        "organizer_labels_or_scores_used": False,
    }
    runner._write_exclusive(AMENDMENT, receipt)
    return receipt


def _verify_amendment_contract(receipt, original, effective, declaration_sha):
    expected_original = hashlib.sha256(original.encode()).hexdigest()
    expected_effective = hashlib.sha256(effective.encode()).hexdigest()
    if (receipt.get("original_function_sha256") != expected_original or
            receipt.get("effective_function_sha256") != expected_effective):
        raise RuntimeError("Runtime validation patch differs from reviewed amendment")
    if receipt.get("declaration_sha256") != declaration_sha:
        raise RuntimeError("Runtime amendment belongs to a different declaration")
    expected_changes = [
        "Require each present training provider to have total mass 1 and the held-out provider to have mass 0.",
        "Require total sample mass to equal the number of present providers (2).",
    ]
    if receipt.get("only_effective_guard_changes") != expected_changes:
        raise RuntimeError("Runtime amendment guard description changed")
    expected_scheduler = [f"{provider}/{seed}" for provider in runner.PROVIDERS
                          for seed in runner.SEEDS]
    scheduler = receipt.get("prior_scheduler", {})
    if (scheduler.get("tasks") != expected_scheduler or
            scheduler.get("worker_count") != 3 or
            scheduler.get("torch_threads_per_worker") != 1 or
            scheduler.get("disjoint_output_files") is not True or
            scheduler.get("only_training_folds_read") is not True):
        raise RuntimeError("Prior scheduler differs from reviewed amendment")
    _, observed_mass_contract = _validate_mass_contract()
    if _canonical(observed_mass_contract) != _canonical(receipt.get("mass_contract_checks")):
        raise RuntimeError("Current provider mass checks differ from reviewed amendment")
    prior_entries = receipt.get("training_prior_receipts", [])
    expected_keys = {(provider, seed) for provider in runner.PROVIDERS
                     for seed in runner.SEEDS}
    if {(item.get("heldout_provider"), item.get("backbone_seed"))
            for item in prior_entries} != expected_keys:
        raise RuntimeError("Amendment does not bind all six prior caches")
    for item in prior_entries:
        provider = item["heldout_provider"]
        seed = item["backbone_seed"]
        cache_path = runner._prior_cache_path(provider, seed)
        prior_receipt_path = cache_path.with_suffix(".json")
        prior_receipt = runner._json(prior_receipt_path)
        if (sha256(cache_path) != item.get("cache_sha256") or
                sha256(prior_receipt_path) != item.get("receipt_sha256") or
                prior_receipt.get("cache_sha256") != item.get("cache_sha256") or
                prior_receipt.get("declaration_sha256") != declaration_sha or
                prior_receipt.get("heldout_provider") != provider or
                prior_receipt.get("backbone_seed") != seed or
                prior_receipt.get("evaluation_archives_read") is not False):
            raise RuntimeError(f"Training prior differs from amendment: {provider}/{seed}")


def validate():
    original, effective = _patch_training_mass_guard()
    declaration, declaration_sha = runner._verify_declaration()
    prior_receipts = _prepare_priors_parallel(declaration, declaration_sha)
    declaration_sha, mass_contract = _validate_mass_contract()
    receipt = _write_amendment(original, effective, declaration_sha, mass_contract,
                               prior_receipts)
    print(json.dumps({
        "amendment_path": str(AMENDMENT),
        "declaration_sha256": declaration_sha,
        "original_function_sha256": receipt["original_function_sha256"],
        "effective_function_sha256": receipt["effective_function_sha256"],
        "runtime_wrapper_sha256": receipt["runtime_wrapper_sha256"],
        "fold_arm_contracts_checked": len(mass_contract),
        "total_mass_each": 2.0,
        "fits_started": False,
    }, sort_keys=True, indent=2))


def fit():
    if not AMENDMENT.is_file():
        raise RuntimeError("Run validate and obtain amendment review before fitting")
    receipt = runner._json(AMENDMENT)
    wrapper_sha = sha256(Path(__file__))
    if (receipt.get("runtime_wrapper_sha256") != wrapper_sha or
            receipt.get("source_files_changed") is not False or
            receipt.get("fits_started") is not False or
            receipt.get("candidate_scores_opened") is not False):
        raise RuntimeError("Runtime amendment receipt does not match this wrapper")
    if any(OUT.glob("head_*.joblib")) or any(OUT.glob("head_*.json")):
        raise RuntimeError("Refusing to start with existing or partial heads")
    original, effective = _patch_training_mass_guard()
    declaration, declaration_sha = runner._verify_declaration()
    _verify_amendment_contract(receipt, original, effective, declaration_sha)
    runner._prepare_training_priors = _prepare_priors_parallel
    result = runner.fit_all()
    print(json.dumps({"frozen_heads": len(result), "fits_started": True,
                      "candidate_scores_opened": False,
                      "declaration_sha256": declaration_sha,
                      "runtime_amendment_sha256": sha256(AMENDMENT)}, indent=2))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["validate"]:
        validate()
    elif sys.argv[1:] == ["fit"]:
        fit()
    else:
        raise SystemExit("usage: native_short_spell_runtime.py {validate|fit}")
