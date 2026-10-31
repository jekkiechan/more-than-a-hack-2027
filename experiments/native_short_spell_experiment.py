"""Bounded native-rate H/A emission refinement over the frozen GRU ensemble.

The runner has an immutable declaration, fixed twelve-head budget, fit seal,
and one scoring pass over the fifteen saved full-provider cells. It never
accesses the organizer's labels or prediction gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

import joblib
import numpy as np

from experiments import (
    contact_memory_experiment as cmx,
    factorized_had as fh,
    learned_temporal as lt,
    multisource_eval as ms,
    native_short_spell as nss,
    objective_ensemble as oe,
    preparation as prep,
    possession_geometry,
    supervised,
)
from experiments.eval_set.protocol import sha256


ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")).resolve()
OUT = DATA / "experiments/native_short_spell_v1"
BASE_CACHE = DATA / "experiments/objective_alignment_v1"
ENSEMBLE_CACHE = DATA / "experiments/objective_ensemble_v1"
EVAL_CACHE = DATA / "experiments/seed_gap_context_v1"
ARMS = ("native_rate", "matched_5hz")
SEEDS = (2026, 2027)
PROVIDERS = ("sportec", "metrica", "skillcorner")
VIEWS = ("clean", "gap", "dropout_legacy", "dropout_gap")
FIT_UPDATES = nss.FIT_UPDATES
BATCH_SIZE = nss.BATCH_SIZE
DECLARATION = ROOT / "experiments/results/native_short_spell_declaration.json"
READOUT = ROOT / "experiments/results/native_short_spell_readout.json"
FIT_SEAL = OUT / "checkpoints_frozen.json"


def digest_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _json(path: Path):
    return json.loads(Path(path).read_text())


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _write_exclusive(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2,
                                ensure_ascii=False, allow_nan=False).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())


def _data_path(relative: str) -> Path:
    value = Path(relative)
    if value.parts and value.parts[0] == "data":
        return DATA.joinpath(*value.parts[1:])
    return DATA / value


def _source_paths() -> tuple[Path, ...]:
    return tuple(ROOT / relative for relative in (
        "experiments/native_short_spell.py",
        "experiments/native_short_spell_experiment.py",
        "experiments/tests/test_native_short_spell.py",
        "experiments/contact_memory_experiment.py",
        "experiments/possession_geometry.py",
        "experiments/supervised.py",
        "experiments/factorized_had.py",
        "experiments/learned_temporal.py",
        "experiments/preparation.py",
        "experiments/multisource_eval.py",
        "experiments/objective_ensemble.py",
        "experiments/eval_set/manifest.json",
        "experiments/eval_set/protocol.py",
        "experiments/live_gap_experiment.py",
        "experiments/gru_robustness_experiment.py",
        "experiments/light_corruption.py",
        "experiments/corruption_experiment.py",
        "experiments/robust.py",
        "experiments/tracking_corruption.py",
        "experiments/refine_features.py",
        "experiments/refine_domain.py",
        "experiments/baseline_manifest.json",
        "experiments/model_decision_policy.json",
        "evaluation/possession_metrics.py",
    ))


def _source_hashes() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): sha256(path) for path in _source_paths()}


def _checkpoint_facts() -> dict:
    frozen = _json(BASE_CACHE / "checkpoints_frozen.json")
    entries = {item["file"]: item for item in frozen["new_checkpoints"]}
    result = {}
    for provider in PROVIDERS:
        result[provider] = {}
        for seed in SEEDS:
            name = f"without_{provider}_combined_{seed}.joblib"
            if name not in entries:
                raise ValueError(f"Missing frozen provider-held-out checkpoint: {name}")
            model = BASE_CACHE / name
            receipt = model.with_suffix(".json")
            entry = entries[name]
            if (not model.is_file() or not receipt.is_file() or
                    sha256(model) != entry["sha256"] or
                    sha256(receipt) != entry["receipt_sha256"]):
                raise ValueError(f"Changed frozen checkpoint: {name}")
            result[provider][str(seed)] = {
                "path": str(model.relative_to(DATA)),
                "sha256": entry["sha256"],
                "receipt": str(receipt.relative_to(DATA)),
                "receipt_sha256": entry["receipt_sha256"],
            }
    return result


def _training_facts(plan: dict) -> dict:
    preparation = _json(BASE_CACHE / "predeclared.json")
    declared = preparation["prepared_training_caches"]
    rows = {}
    for item in plan["train"]:
        entry = declared[item["match"]]
        cache = _data_path(entry["cache"])
        receipt = _data_path(entry["receipt"])
        raw = prep.data_path(item, DATA)
        if (sha256(raw) != item["sha256"] or sha256(cache) != entry["sha256"] or
                sha256(receipt) != entry["receipt_sha256"]):
            raise ValueError(f"Changed TRAIN input/cache: {item['match']}")
        rows[item["match"]] = {
            "source": item["source"], "input": str(raw.relative_to(DATA)),
            "input_sha256": item["sha256"], "cache": entry["cache"],
            "cache_sha256": entry["sha256"], "cache_receipt": entry["receipt"],
            "cache_receipt_sha256": entry["receipt_sha256"],
        }
    return {
        "preparation_declaration": str((BASE_CACHE / "predeclared.json").relative_to(DATA)),
        "preparation_declaration_sha256": sha256(BASE_CACHE / "predeclared.json"),
        "calibration": preparation["calibration"],
        "matches": rows,
    }


def _predecessor_verification() -> dict:
    declaration_path = ROOT / "experiments/results/contact_memory_declaration.json"
    verification_path = ROOT / "experiments/results/contact_memory_independent_verification.json"
    checked = cmx.check()
    verification = _json(verification_path)
    checks = verification.get("checks", {})
    required = {
        "frozen_declaration_and_inputs_verified": True,
        "declared_source_hashes_match": True,
        "native_member_pd_exact_to_archived_baseline": 30,
        "native_ensemble_pd_exact_to_archived_baseline": 15,
        "candidate_seed_decodes_and_official_rows_exact": 30,
        "candidate_ensemble_decodes_and_official_rows_exact": 15,
    }
    if (verification.get("status") != "passed" or
            verification.get("study_id") != "contact_memory_v1" or
            verification.get("provenance", {}).get("declaration_sha256") != sha256(declaration_path) or
            any(checks.get(key) != value for key, value in required.items()) or
            checked.get("_declaration_sha256") != verification["provenance"]["declaration_sha256"]):
        raise ValueError("Prior contact-memory control verification is missing or stale")
    return {
        "declaration": str(declaration_path.relative_to(ROOT)),
        "declaration_sha256": sha256(declaration_path),
        "verification": str(verification_path.relative_to(ROOT)),
        "verification_sha256": sha256(verification_path),
        "verification_status": verification["status"],
        "checks": {key: checks[key] for key in required},
    }


def _build_declaration(*, require_preflight: bool = False) -> dict:
    manifest_path = ROOT / "experiments/baseline_manifest.json"
    policy_path = ROOT / "experiments/model_decision_policy.json"
    manifest = _json(manifest_path)
    policy = _json(policy_path)
    if manifest.get("package_version") != "joint_gru_combined_ensemble_v1":
        raise ValueError("Active baseline is no longer the combined-objective ensemble")
    if policy.get("working_candidate", {}).get("baseline_manifest_sha256") != sha256(manifest_path):
        raise ValueError("Active decision policy does not bind the baseline manifest")
    if policy.get("primary_order") != ["spell_f1", "macro_f1"]:
        raise ValueError("The active spell-first ranking policy changed")
    plan = prep.split_plan(fold="pooled")
    if len(plan["train"]) != 13 or len(plan["selection"]) != 5:
        raise ValueError("Expected thirteen TRAIN and five selection matches")
    folds = {
        provider: prep.split_plan(fold=f"leave_out_{provider}")
        for provider in PROVIDERS
    }
    for provider, fold in folds.items():
        if any(item["source"] == provider for item in fold["train"]):
            raise ValueError(f"Held-out provider leaked into training fold: {provider}")
        if not any(item["source"] == provider for item in fold["selection"]):
            raise ValueError(f"Held-out provider is absent from evaluation fold: {provider}")

    native_rates = {provider: set() for provider in PROVIDERS}
    for item in plan["train"] + plan["selection"]:
        with np.load(prep.data_path(item, DATA), allow_pickle=False) as archive:
            native_rates[item["source"]].add(float(archive["fps"]))
    if native_rates != {"sportec": {25.0}, "metrica": {25.0}, "skillcorner": {10.0}}:
        raise ValueError(f"Unexpected provider native frame rates: {native_rates}")

    training = _training_facts(plan)
    evaluation = cmx._evaluation_facts()
    if len(evaluation["cells"]) != 15:
        raise ValueError("Expected the frozen fifteen-cell clean/feed_like cohort")
    for cell in evaluation["cells"]:
        for relative, expected in (
                (cell["evaluation_cache"], cell["evaluation_cache_sha256"]),
                (cell["ensemble_probability"], cell["ensemble_probability_sha256"])):
            if sha256(_data_path(relative)) != expected:
                raise ValueError(f"Changed evaluation artifact: {relative}")
        for member in cell["member_baselines"].values():
            if sha256(_data_path(member["probability"])) != member["probability_sha256"]:
                raise ValueError("Changed saved baseline member output")

    declaration = {
        "schema_version": 1,
        "study_id": "native_short_spell_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen before residual fitting and candidate scoring",
        "question": "Do instantaneous native-rate control geometry emissions improve the frozen combined-GRU H/A outputs beyond the same linear head trained and sampled at 5 Hz?",
        "native_rate_hz_by_provider": {provider: next(iter(rates))
                                       for provider, rates in native_rates.items()},
        "scope": {
            "historical_development_matches_previously_exposed": True,
            "independent_confirmation": False,
            "organizer_labels_or_scores_used": False,
            "provider_held_out_from_every_learned_component": True,
        },
        "prior_art": {
            "pr11_live_attribution": "Pooled live-only/short-spell-weighted HGB at existing approximately 4 Hz; not frozen-GRU residuals or full-provider holdout.",
            "pr13_short_spell": "Complete-spell duration weighting for the older ballgap25 HGB at its sampled rate.",
            "pr13_boundary": "5 Hz directional D/live stoppage/restart head that modifies decoder transition costs; H/A switches remain unchanged.",
            "distinct_mechanism": "Native instantaneous H/A control geometry refines frozen GRU conditional H/A emissions, with a same-capacity, same-update 5 Hz control and unchanged decoder.",
        },
        "arms": {
            "native_rate": "Train the fixed linear residual at every native geometry/label row; emit one residual per native frame (25 Hz Sportec/Metrica; 10 Hz SkillCorner).",
            "matched_5hz": "Train the same fixed linear residual on exact GRU 5 Hz rows; interpolate only this residual block-locally onto the native prior.",
            "common": "Same 23 geometry fields plus 23 finite masks; frozen member-specific native H/A log-odds is a fixed offset. Copy member P(D) exactly and rescale live H/A mass; 50/50 true probability ensemble before the unchanged cost-3 decoder.",
        },
        "feature_spec": {
            "source": "possession_geometry.features with include_anchors=False",
            "fields": list(possession_geometry.GEOMETRY_NAMES),
            "finite_values": "Training-fold weighted mean and scale; impute missing with the same training-fold mean and append a binary finite-value mask for each field.",
            "missingness": "NaNs use training-fold weighted-mean imputation and one binary finite mask per geometry field; the same 46 inputs are used by both arms. A missingness-dependent residual can be learned, subject to full provider-held-out fitting and declared guards.",
            "lookback": "Only the feature builder's native local derivative halo; no event/token memory or long history/future windows.",
            "labels_at_inference": False,
        },
        "fit": {
            "model": "One binary linear logistic residual; fixed frozen prior offset, no hidden layer.",
            "optimizer": "CPU PyTorch Adam, one thread, zero initialization, fixed 512 updates × batch 2048, learning rate 0.01, L2 0.02.",
            "support_mass": "Equal total weight by provider, then match, view, and direct/team-swapped copy; uniform over known H/A rows within each group.",
            "training_views": list(VIEWS),
            "training_cache_contract": "Regenerate each raw view with contact_memory_experiment._training_views; require bit-exact cached joint feature, H/A/D/X target vector, sample index and native period parity before native geometry is built for that same view.",
            "prior_source": "Corresponding provider-held-out combined-objective GRU member inferred only on that fold's TRAIN caches; never on evaluation archives.",
            "heads": len(ARMS) * len(PROVIDERS) * len(SEEDS),
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "heldout_providers": list(PROVIDERS),
        },
        "runtime": {
            "python": os.sys.version.split()[0],
            "packages": {name: importlib.metadata.version(name)
                         for name in ("numpy", "torch", "joblib")},
            "fit_workers": 1,
            "torch_threads": 1,
            "feature_preparation_workers": 1,
        },
        "selection": {
            "primary": "Official equal-provider spell F1 over IoU .3-.7; macro F1 breaks exact ties.",
            "default_clean_screen": "candidate clean spell F1 at least baseline +0.01",
            "feed_like_guard": "candidate feed_like spell F1 at least baseline -0.01",
            "provider_clean_guards": "Each provider clean macro F1 and D-F1 delta at least -0.02.",
            "secondary": "Under-three-second recall at IoU .5, all IoU scores, D precision/recall, counts, fragmentation, graded-edge and complete-boundary diagnostics.",
            "automatic_promotion": False,
        },
        "baseline": {
            "manifest_sha256": sha256(manifest_path),
            "decision_policy_sha256": sha256(policy_path),
            "manifest": "experiments/baseline_manifest.json",
            "decision_policy": "experiments/model_decision_policy.json",
            "training": training,
            "fold_checkpoints": _checkpoint_facts(),
            "predecessor_replay_verification": _predecessor_verification(),
            "evaluation": evaluation,
        },
        "split_plans": {
            "pooled": prep.fingerprint(plan),
            **{provider: prep.fingerprint(fold) for provider, fold in folds.items()},
        },
        "source_sha256": _source_hashes(),
        "organizer_labels_or_scores_used": False,
        "fits_started": False,
        "candidate_scores_opened": False,
    }
    if require_preflight:
        receipt_path = OUT / "preflight.json"
        if not receipt_path.is_file():
            raise ValueError("Preflight must pass before the declaration is frozen")
        receipt = _json(receipt_path)
        if (receipt.get("source_sha256") != declaration["source_sha256"] or
                receipt.get("zero_residual_member_and_ensemble_parity") is not True or
                receipt.get("fit_performed") is not False or
                receipt.get("candidate_score_opened") is not False):
            raise ValueError("Preflight receipt does not bind current sources/contracts")
        declaration["preflight"] = {
            "path": str(receipt_path.relative_to(DATA)),
            "sha256": sha256(receipt_path),
            "probability_rows_checked": receipt["probability_rows_checked"],
        }
    return declaration


def _verify_declaration() -> tuple[dict, str]:
    if not DECLARATION.is_file():
        raise ValueError("No frozen native short-spell declaration; run freeze first")
    declaration = _json(DECLARATION)
    actual = _source_hashes()
    frozen = declaration.get("source_sha256", {})
    if actual != frozen:
        changed = sorted(key for key in set(actual) | set(frozen)
                         if actual.get(key) != frozen.get(key))
        raise ValueError(f"Frozen source changed; use a new study ID: {changed}")
    if (sha256(ROOT / "experiments/baseline_manifest.json") !=
            declaration["baseline"]["manifest_sha256"] or
            sha256(ROOT / "experiments/model_decision_policy.json") !=
            declaration["baseline"]["decision_policy_sha256"]):
        raise ValueError("Active baseline/policy changed after declaration")
    if _predecessor_verification() != declaration["baseline"]["predecessor_replay_verification"]:
        raise ValueError("Predecessor baseline replay receipt or evaluation cache changed")
    preflight = declaration.get("preflight", {})
    if (not preflight or sha256(_data_path(preflight["path"])) != preflight["sha256"]):
        raise ValueError("Frozen preflight receipt is missing or changed")
    for item in declaration["baseline"]["training"]["matches"].values():
        for key, expected_key in (("input", "input_sha256"),
                                  ("cache", "cache_sha256"),
                                  ("cache_receipt", "cache_receipt_sha256")):
            path = _data_path(item[key])
            if sha256(path) != item[expected_key]:
                raise ValueError(f"Frozen TRAIN artifact changed: {item[key]}")
    return declaration, sha256(DECLARATION)


def preflight() -> dict:
    """Check archived probability contracts without labels, fitting or scoring."""
    declaration = _build_declaration()
    tested = 0
    for cell in declaration["baseline"]["evaluation"]["cells"]:
        for member in cell["member_baselines"].values():
            path = _data_path(member["probability"])
            with np.load(path, allow_pickle=False) as archive:
                probability = archive["probability"]
            zero = nss.apply_residual(probability, np.zeros(len(probability)))
            if not np.array_equal(zero, probability):
                raise ValueError("Zero residual did not reproduce frozen member output")
            tested += 1
        path = _data_path(cell["ensemble_probability"])
        with np.load(path, allow_pickle=False) as archive:
            probability = archive["probability"]
        if not np.array_equal(nss.apply_residual(probability, np.zeros(len(probability))),
                              probability):
            raise ValueError("Zero residual did not reproduce saved ensemble control")
        tested += 1
    result = {
        "study_id": "native_short_spell_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": declaration["source_sha256"],
        "probability_rows_checked": tested,
        "zero_residual_member_and_ensemble_parity": True,
        "organizer_labels_or_scores_used": False,
        "fit_performed": False,
        "candidate_score_opened": False,
    }
    path = OUT / "preflight.json"
    if path.exists():
        saved = _json(path)
        if any(saved.get(key) != value for key, value in result.items()
               if key != "created_utc"):
            raise ValueError("Native short-spell preflight receipt is immutable")
        return saved
    _write_exclusive(path, result)
    return result


def freeze() -> dict:
    """Freeze arms, folds, source/input artifacts and score rules."""
    built = _build_declaration(require_preflight=True)
    path = DECLARATION
    if path.exists():
        saved = _json(path)
        for key, value in built.items():
            if key != "created_utc" and saved.get(key) != value:
                raise ValueError("Existing declaration is immutable; use a new study ID")
        return saved
    _write_exclusive(path, built)
    return built


def _saved_checkpoint(provider: str, seed: int, declaration: dict):
    fact = declaration["baseline"]["fold_checkpoints"][provider][str(seed)]
    model_path = _data_path(fact["path"])
    if sha256(model_path) != fact["sha256"]:
        raise ValueError(f"Provider-held-out checkpoint changed: {provider}/{seed}")
    model = joblib.load(model_path)
    if model.get("kind") != "factorized_had" or model.get("objective") != "combined":
        raise ValueError("Unexpected frozen GRU checkpoint kind")
    return model


def _feature_cache_path(match: str) -> Path:
    return OUT / f"train_features_{match}.npz"


def _prepare_training_features(declaration: dict, declaration_sha: str) -> list[dict]:
    plan = prep.split_plan(fold="pooled")
    calibration = declaration["baseline"]["training"]["calibration"]
    preparation = _json(BASE_CACHE / "predeclared.json")
    receipts = []
    for item in plan["train"]:
        destination = _feature_cache_path(item["match"])
        receipt_path = destination.with_suffix(".json")
        if destination.exists() or receipt_path.exists():
            if not destination.is_file() or not receipt_path.is_file():
                raise ValueError(f"Partial native geometry cache: {item['match']}")
            receipt = _json(receipt_path)
            if (receipt.get("declaration_sha256") != declaration_sha or
                    receipt.get("cache_sha256") != sha256(destination) or
                    receipt.get("input_sha256") != item["sha256"]):
                raise ValueError(f"Changed native geometry cache: {item['match']}")
            receipts.append(receipt)
            continue

        input_path = prep.data_path(item, DATA)
        inputs, labels = ms.view(input_path)
        views = cmx._training_views(inputs, item["source"], item["match"], calibration)
        training_cache_entry = preparation["prepared_training_caches"][item["match"]]
        training_cache_path = _data_path(training_cache_entry["cache"])
        if sha256(training_cache_path) != training_cache_entry["sha256"]:
            raise ValueError(f"Changed cached feature contract: {item['match']}")
        cached = joblib.load(training_cache_path)
        full_y = np.asarray(cached["full_y"], dtype=np.int8)
        full_period = np.asarray(cached["full_period"])
        arrays = {"full_y": full_y, "full_period": full_period,
                  "native_fps": np.asarray(float(inputs["fps"]))}
        view_facts = {}
        for view_name in VIEWS:
            if view_name not in views or view_name not in cached["records"]:
                raise ValueError(f"Missing regenerated or cached training view: {item['match']}/{view_name}")
            viewed = views[view_name]
            reference = cached["records"][view_name]
            rebuilt = cmx.old.feature_view(
                viewed, labels, item["source"], item["match"])["joint"]
            if (list(rebuilt["names"]) != list(reference["names"]) or
                    not np.array_equal(rebuilt["indices"], reference["indices"]) or
                    not np.array_equal(rebuilt["y"], reference["y"]) or
                    not np.array_equal(np.asarray(rebuilt["x"], dtype=np.float32),
                                       np.asarray(reference["x"], dtype=np.float32),
                                       equal_nan=True)):
                raise ValueError(f"Cached 5 Hz feature/target parity failed: {item['match']}/{view_name}")
            indices = lt.sampled_indices(viewed)
            if not np.array_equal(indices, reference["indices"]):
                raise ValueError(f"Sample clock parity failed: {item['match']}/{view_name}")
            geometry, names = nss.geometry_features(viewed)
            if (len(geometry) != len(full_y) or
                    not np.array_equal(full_period, supervised.model_periods(viewed))):
                raise ValueError(f"Native geometry/target alignment failed: {item['match']}/{view_name}")
            arrays[f"{view_name}_geometry"] = geometry
            arrays[f"{view_name}_indices"] = indices.astype(np.int64)
            view_facts[view_name] = {
                "native_rows": int(len(full_y)),
                "sampled_rows": int(len(indices)),
                "native_rate_hz": float(viewed["fps"]),
                "known_ha_native": int(np.isin(full_y, (0, 1)).sum()),
                "cached_5hz_x_sha256": digest_array(np.asarray(reference["x"], dtype=np.float32)),
                "geometry_sha256": digest_array(geometry),
                "geometry_features": len(names),
                "features_recomputed_from_this_view": True,
            }
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
        receipt = {
            "match": item["match"], "source": item["source"],
            "input_sha256": item["sha256"],
            "training_cache_sha256": training_cache_entry["sha256"],
            "declaration_sha256": declaration_sha,
            "cache_sha256": sha256(destination),
            "views": view_facts,
            "labels_used_only_for_training_cache_parity_and_targets": True,
            "labels_used_for_features": False,
        }
        _write_exclusive(receipt_path, receipt)
        receipts.append(receipt)
        print("NATIVE GEOMETRY", item["source"], item["match"], flush=True)
        del inputs, labels, views, cached, arrays
    if len(receipts) != len(plan["train"]):
        raise ValueError("Native geometry preparation did not cover all TRAIN matches")
    return receipts


def _prior_cache_path(provider: str, seed: int) -> Path:
    return OUT / f"train_prior_{provider}_{seed}.npz"


def _prepare_training_priors(declaration: dict, declaration_sha: str) -> list[dict]:
    results = []
    for provider in PROVIDERS:
        fold = prep.split_plan(fold=f"leave_out_{provider}")
        for seed in SEEDS:
            destination = _prior_cache_path(provider, seed)
            receipt_path = destination.with_suffix(".json")
            if destination.exists() or receipt_path.exists():
                if not destination.is_file() or not receipt_path.is_file():
                    raise ValueError(f"Partial training prior cache: {provider}/{seed}")
                receipt = _json(receipt_path)
                if (receipt.get("declaration_sha256") != declaration_sha or
                        receipt.get("cache_sha256") != sha256(destination)):
                    raise ValueError(f"Changed training prior cache: {provider}/{seed}")
                results.append(receipt)
                continue
            model = _saved_checkpoint(provider, seed, declaration)
            arrays = {}
            support = {}
            started = time.monotonic()
            for item in fold["train"]:
                feature_path = _feature_cache_path(item["match"])
                if not feature_path.is_file():
                    raise ValueError(f"Missing TRAIN geometry cache: {item['match']}")
                training_meta = declaration["baseline"]["training"]["matches"][item["match"]]
                cache = joblib.load(_data_path(training_meta["cache"]))
                if (cache.get("match") != item["match"] or
                        cache.get("source") != item["source"] or
                        cache.get("input_sha256") != item["sha256"]):
                    raise ValueError(f"TRAIN cache identity mismatch: {item['match']}")
                with np.load(feature_path, allow_pickle=False) as feature_archive:
                    full_period = feature_archive["full_period"]
                    for view_name in VIEWS:
                        record = cache["records"][view_name]
                        prediction_record = {
                            key: record[key] for key in
                            ("x", "names", "indices", "period", "native_n", "fps")
                        }
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
                            "prior_sha256": digest_array(arrays[key]),
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
                "views": support,
                "evaluation_archives_read": False,
                "seconds": time.monotonic() - started,
            }
            _write_exclusive(receipt_path, receipt)
            results.append(receipt)
            print("NATIVE PRIOR", provider, seed,
                  "seconds", round(receipt["seconds"], 1), flush=True)
            del model, arrays
    if len(results) != len(PROVIDERS) * len(SEEDS):
        raise ValueError("Provider-held-out TRAIN priors are incomplete")
    return results


def _assemble_training_data(provider: str, seed: int, arm: str,
                            declaration: dict) -> dict:
    if arm not in ARMS:
        raise ValueError(f"Unknown native-rate arm: {arm}")
    fold = prep.split_plan(fold=f"leave_out_{provider}")
    matches_per_source = {
        source: sum(item["source"] == source for item in fold["train"])
        for source in PROVIDERS
    }
    prior_path = _prior_cache_path(provider, seed)
    x_parts, prior_parts, target_parts, weight_parts = [], [], [], []
    row_support = {}
    with np.load(prior_path, allow_pickle=False) as prior_archive:
        for item in fold["train"]:
            feature_path = _feature_cache_path(item["match"])
            training_meta = declaration["baseline"]["training"]["matches"][item["match"]]
            cache = joblib.load(_data_path(training_meta["cache"]))
            with np.load(feature_path, allow_pickle=False) as feature_archive:
                full_y = feature_archive["full_y"].astype(np.int8, copy=False)
                full_period = feature_archive["full_period"]
                native_fps = float(feature_archive["native_fps"])
                for view_name in VIEWS:
                    geometry = feature_archive[f"{view_name}_geometry"]
                    indices = feature_archive[f"{view_name}_indices"]
                    prior_native = prior_archive[f"{item['match']}__{view_name}"]
                    if (len(geometry) != len(full_y) or
                            len(prior_native) != len(full_y) or
                            not np.array_equal(indices,
                                              cache["records"][view_name]["indices"])):
                        raise ValueError(f"Training row alignment changed: {item['match']}/{view_name}")
                    if arm == "native_rate":
                        x_value = geometry
                        prior_value = prior_native
                        y_value = full_y
                    else:
                        x_value = geometry[indices]
                        prior_value = prior_native[indices]
                        y_value = full_y[indices]
                    selected = np.isin(y_value, (0, 1))
                    count = int(selected.sum())
                    if count == 0:
                        raise ValueError(f"No known H/A training rows: {item['match']}/{view_name}")
                    group_weights = nss.equal_support_weights(
                        count, matches_per_source[item["source"]], len(VIEWS), 2)
                    target_home = (y_value == 0).astype(np.float32)
                    for swap in (False, True):
                        if swap:
                            x_parts.append(supervised.swap_team_features(
                                np.asarray(x_value, dtype=np.float32),
                                possession_geometry.GEOMETRY_NAMES)[selected])
                            target_parts.append(1.0 - target_home[selected])
                            prior_parts.append(-prior_value[selected])
                        else:
                            x_parts.append(np.asarray(x_value, dtype=np.float32)[selected])
                            target_parts.append(target_home[selected])
                            prior_parts.append(prior_value[selected])
                        weight_parts.append(group_weights)
                    row_support[f"{item['match']}__{view_name}"] = {
                        "source": item["source"],
                        "native_rows": int(len(full_y)),
                        "sampled_rows": int(len(indices)),
                        "native_rate_hz": native_fps,
                        "labeled_ha_rows": count,
                    }
            del cache
    weight_sum_by_source = {source: 0.0 for source in PROVIDERS}
    for key, detail in row_support.items():
        weight_sum_by_source[detail["source"]] += (
            1.0 / (matches_per_source[detail["source"]] * len(VIEWS)))
    if any(not np.isclose(value, 1.0, rtol=0, atol=1e-8)
           for value in weight_sum_by_source.values()):
        raise ValueError(f"Training source mass is not equal: {weight_sum_by_source}")
    data = {
        "x": np.concatenate(x_parts),
        "prior": np.concatenate(prior_parts),
        "target": np.concatenate(target_parts),
        "sample_weight": np.concatenate(weight_parts),
    }
    if not np.isclose(float(data["sample_weight"].sum()), len(PROVIDERS), atol=1e-5):
        raise ValueError("Training sample weights do not have equal provider mass")
    data["support"] = {
        "arm": arm, "heldout_provider": provider, "backbone_seed": seed,
        "rows": int(len(data["target"])), "rows_by_match_view": row_support,
        "sample_weight_sum_by_provider": weight_sum_by_source,
        "optimizer_updates": FIT_UPDATES, "batch_size": BATCH_SIZE,
    }
    return data


def _head_path(arm: str, provider: str, seed: int) -> Path:
    return OUT / f"head_{arm}_{provider}_{seed}.joblib"


def fit_all() -> list[dict]:
    declaration, declaration_sha = _verify_declaration()
    preflight_path = OUT / "preflight.json"
    if not preflight_path.is_file():
        raise ValueError("Run preflight before freezing fits")
    preflight_receipt = _json(preflight_path)
    if preflight_receipt.get("source_sha256") != declaration["source_sha256"]:
        raise ValueError("Preflight receipt belongs to different sources")
    _prepare_training_features(declaration, declaration_sha)
    _prepare_training_priors(declaration, declaration_sha)
    receipts = []
    for arm in ARMS:
        for provider in PROVIDERS:
            for seed in SEEDS:
                model_path = _head_path(arm, provider, seed)
                receipt_path = model_path.with_suffix(".json")
                if model_path.exists() or receipt_path.exists():
                    if not model_path.is_file() or not receipt_path.is_file():
                        raise ValueError(f"Partial frozen head: {arm}/{provider}/{seed}")
                    receipt = _json(receipt_path)
                    if (receipt.get("declaration_sha256") != declaration_sha or
                            receipt.get("model_sha256") != sha256(model_path)):
                        raise ValueError(f"Changed frozen head: {arm}/{provider}/{seed}")
                    receipts.append(receipt)
                    continue
                started = time.monotonic()
                training = _assemble_training_data(provider, seed, arm, declaration)
                fitted = nss.fit_residual(
                    training["x"], training["prior"], training["target"],
                    backbone_seed=seed, sample_weight=training["sample_weight"],
                    updates=FIT_UPDATES, batch_size=BATCH_SIZE)
                fitted.update(arm=arm, heldout_provider=provider,
                              declaration_sha256=declaration_sha,
                              support=training["support"])
                with tempfile.NamedTemporaryFile(prefix=f".{model_path.name}.",
                                                 suffix=".joblib", dir=OUT,
                                                 delete=False) as stream:
                    temporary = Path(stream.name)
                try:
                    joblib.dump(fitted, temporary)
                    os.replace(temporary, model_path)
                finally:
                    temporary.unlink(missing_ok=True)
                receipt = {
                    "arm": arm, "heldout_provider": provider, "backbone_seed": seed,
                    "model": str(model_path.relative_to(DATA)),
                    "model_sha256": sha256(model_path),
                    "declaration_sha256": declaration_sha,
                    "parameters": int(np.asarray(fitted["weights"]).size + 1),
                    "optimizer_updates": FIT_UPDATES,
                    "batch_size": BATCH_SIZE,
                    "training_rows": int(len(training["target"])),
                    "training_support": training["support"],
                    "seconds": time.monotonic() - started,
                    "selection_labels_read": False,
                    "organizer_labels_read": False,
                }
                _write_exclusive(receipt_path, receipt)
                receipts.append(receipt)
                print("FROZEN NATIVE HEAD", arm, provider, seed,
                      "rows", receipt["training_rows"],
                      "seconds", round(receipt["seconds"], 1), flush=True)
                del fitted, training
    if len(receipts) != len(ARMS) * len(PROVIDERS) * len(SEEDS):
        raise ValueError("The declared twelve-head budget is incomplete")
    return receipts


def seal_fits() -> dict:
    declaration, declaration_sha = _verify_declaration()
    heads = []
    for arm in ARMS:
        for provider in PROVIDERS:
            for seed in SEEDS:
                model_path = _head_path(arm, provider, seed)
                receipt_path = model_path.with_suffix(".json")
                if not model_path.is_file() or not receipt_path.is_file():
                    raise ValueError(f"Missing predeclared head: {arm}/{provider}/{seed}")
                receipt = _json(receipt_path)
                if (receipt.get("declaration_sha256") != declaration_sha or
                        receipt.get("model_sha256") != sha256(model_path) or
                        receipt.get("optimizer_updates") != FIT_UPDATES or
                        receipt.get("batch_size") != BATCH_SIZE):
                    raise ValueError(f"Head receipt mismatch: {arm}/{provider}/{seed}")
                heads.append({
                    "arm": arm, "provider": provider, "seed": seed,
                    "path": str(model_path.relative_to(DATA)),
                    "sha256": sha256(model_path),
                    "receipt": str(receipt_path.relative_to(DATA)),
                    "receipt_sha256": sha256(receipt_path),
                })
    seal = {
        "study_id": "native_short_spell_v1",
        "declaration_sha256": declaration_sha,
        "heads_frozen": len(heads), "gru_refits": 0,
        "heads": heads,
        "sealed_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_scores_opened": False,
        "organizer_labels_read": False,
    }
    if FIT_SEAL.exists():
        saved = _json(FIT_SEAL)
        for key in ("study_id", "declaration_sha256", "heads_frozen", "gru_refits", "heads"):
            if saved.get(key) != seal[key]:
                raise ValueError("Frozen-head seal is immutable and does not match outputs")
        return saved
    _write_exclusive(FIT_SEAL, seal)
    return seal


def _verify_fit_seal(declaration_sha: str) -> dict:
    if not FIT_SEAL.is_file():
        raise ValueError("All twelve residual heads must be sealed before scoring")
    seal = _json(FIT_SEAL)
    if seal.get("declaration_sha256") != declaration_sha or seal.get("heads_frozen") != 12:
        raise ValueError("Frozen-head seal belongs to another study or is incomplete")
    for item in seal["heads"]:
        if (sha256(_data_path(item["path"])) != item["sha256"] or
                sha256(_data_path(item["receipt"])) != item["receipt_sha256"]):
            raise ValueError("A sealed residual head or receipt changed")
    return seal


def _head_index(seal: dict) -> dict:
    result = {}
    for item in seal["heads"]:
        model = joblib.load(_data_path(item["path"]))
        key = (item["arm"], item["provider"], int(item["seed"]))
        if key in result:
            raise ValueError(f"Duplicate sealed head: {key}")
        result[key] = model
    if len(result) != 12:
        raise ValueError("Expected exactly twelve sealed residual heads")
    return result


def _eval_row(cell: dict) -> dict:
    cache_path = _data_path(cell["evaluation_cache"])
    if sha256(cache_path) != cell["evaluation_cache_sha256"]:
        raise ValueError(f"Evaluation tracking cache changed: {cell['match']}")
    cache = joblib.load(cache_path)
    inputs = cache["inputs"]
    if any(key in inputs for key in ("labels", "targets", "events")):
        raise ValueError("Evaluation labels leaked into tracking-only feature input")
    record = cache["record"]
    indices = np.asarray(record["indices"])
    if not np.array_equal(indices, lt.sampled_indices(inputs)):
        raise ValueError(f"Evaluation 5 Hz grid changed: {cell['match']}")
    geometry, names = nss.geometry_features(inputs)
    if names != list(possession_geometry.GEOMETRY_NAMES):
        raise ValueError("Evaluation geometry schema changed")
    return cache, inputs, geometry, indices


def _save_npz(path: Path, **arrays) -> None:
    if path.exists():
        raise ValueError(f"Candidate output is immutable: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".npz",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _baseline_member(cell: dict, seed: int) -> np.ndarray:
    fact = cell["member_baselines"][str(seed)]
    path = _data_path(fact["probability"])
    if sha256(path) != fact["probability_sha256"]:
        raise ValueError(f"Saved baseline member changed: {cell['match']}/{seed}")
    with np.load(path, allow_pickle=False) as archive:
        probability = archive["probability"]
    return probability


def _candidate_member(arm: str, cell: dict, seed: int, probability: np.ndarray,
                      prediction: np.ndarray, head: dict) -> dict:
    path = OUT / f"candidate_{arm}_{cell['match']}_{cell['view']}_{cell['replicate']}_seed_{seed}.npz"
    _save_npz(path, probability=probability, prediction=prediction.astype("U1"))
    return {
        "recipe": arm, "source": cell["source"], "match": cell["match"],
        "view": cell["view"], "replicate": cell["replicate"], "seed": seed,
        "probability_file": str(path.relative_to(DATA)),
        "probability_file_sha256": sha256(path),
        "probability_sha256": digest_array(probability),
        "prediction_sha256": digest_array(prediction.astype("U1")),
        "p_d_exact": True,
        "head_sha256": head["model_sha256"],
    }


def _aggregate(rows: list[dict], controls: dict) -> dict:
    providers = PROVIDERS
    summaries = {}
    metrics = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
               "d_precision", "d_recall", "short_spell_recall",
               "iou_0.3", "iou_0.4", "iou_0.5", "iou_0.6", "iou_0.7")
    for arm in ARMS:
        summaries[arm] = {}
        for view in ("clean", "feed_like"):
            per_provider = {}
            for provider in providers:
                selected = [row for row in rows if row["recipe"] == arm and
                            row["source"] == provider and row["view"] == view]
                expected = sum(cell["source"] == provider and cell["view"] == view
                               for cell in oe._cells())
                if len(selected) != expected:
                    raise ValueError(f"Incomplete cells: {arm}/{provider}/{view}")
                per_provider[provider] = {
                    name: float(np.mean([row[name] for row in selected]))
                    for name in metrics if not name.startswith("iou_")
                }
                for threshold in ("0.3", "0.4", "0.5", "0.6", "0.7"):
                    per_provider[provider][f"iou_{threshold}"] = float(np.mean([
                        row["spell_by_iou"][threshold] for row in selected]))
                per_provider[provider]["n_cells"] = len(selected)
            summaries[arm][view] = {
                "per_provider": per_provider,
                "equal_provider_mean": {
                    name: float(np.mean([per_provider[p][name] for p in providers]))
                    for name in metrics
                },
                "aggregation": "mean within provider, then equal mean across providers",
            }
    deltas = {}
    for arm in ARMS:
        deltas[arm] = {}
        for view in ("clean", "feed_like"):
            deltas[arm][view] = {
                "equal_provider_mean": {
                    name: summaries[arm][view]["equal_provider_mean"][name] -
                    controls[view]["equal_provider_mean"][name]
                    for name in metrics
                },
                "per_provider": {
                    provider: {
                        name: summaries[arm][view]["per_provider"][provider][name] -
                        controls[view]["per_provider"][provider][name]
                        for name in metrics
                    } for provider in providers
                },
            }
    gates = {}
    for arm in ARMS:
        clean = deltas[arm]["clean"]["equal_provider_mean"]["spell_f1"]
        feed = deltas[arm]["feed_like"]["equal_provider_mean"]["spell_f1"]
        gates[arm] = {
            "clean_spell_at_least_plus_0_01": clean >= .01,
            "feed_like_spell_at_least_minus_0_01": feed >= -.01,
            "each_provider_clean_macro_at_least_minus_0_02": all(
                deltas[arm]["clean"]["per_provider"][p]["macro_f1"] >= -.02
                for p in providers),
            "each_provider_clean_d_at_least_minus_0_02": all(
                deltas[arm]["clean"]["per_provider"][p]["d_f1"] >= -.02
                for p in providers),
        }
        gates[arm]["passes_all"] = all(gates[arm].values())
    return {
        "recipes": summaries, "candidate_minus_saved_baseline": deltas,
        "predeclared_guardrails": gates,
        "rank_rule": "Unrounded spell F1 first, macro F1 for exact ties; short-spell recall is secondary.",
        "metrics": list(metrics),
    }


def score() -> dict:
    declaration, declaration_sha = _verify_declaration()
    seal = _verify_fit_seal(declaration_sha)
    heads = _head_index(seal)
    baseline = declaration["baseline"]["evaluation"]["cells"]
    all_rows, seed_rows, member_files, ensemble_files = [], [], [], []
    d_checks = []
    for cell in baseline:
        cache, inputs, geometry, indices = _eval_row(cell)
        labels = np.asarray(cache["labels"]).astype(str)
        period = supervised.model_periods(inputs)
        base_members = {seed: _baseline_member(cell, seed) for seed in SEEDS}
        base_ensemble_path = _data_path(cell["ensemble_probability"])
        if sha256(base_ensemble_path) != cell["ensemble_probability_sha256"]:
            raise ValueError("Saved current ensemble changed")
        with np.load(base_ensemble_path, allow_pickle=False) as archive:
            saved_ensemble = archive["probability"]
        reconstructed = oe.blend_probabilities(base_members[2026], base_members[2027])
        if not np.allclose(reconstructed, saved_ensemble, rtol=0, atol=1e-12):
            raise ValueError(f"Saved ensemble control does not match members: {cell['match']}")
        candidate_by_arm = {arm: [] for arm in ARMS}
        for arm in ARMS:
            for seed in SEEDS:
                model = heads[(arm, cell["source"], seed)]
                if arm == "native_rate":
                    residual = nss.predict_symmetric_residual(
                        geometry, model, list(possession_geometry.GEOMETRY_NAMES))
                else:
                    sampled_residual = nss.predict_symmetric_residual(
                        geometry[indices], model,
                        list(possession_geometry.GEOMETRY_NAMES))
                    residual = nss.upsample_blockwise(sampled_residual, indices, period)
                probability = nss.apply_residual(base_members[seed], residual)
                if not np.array_equal(probability[:, 2], base_members[seed][:, 2]):
                    raise ValueError(f"Candidate changed frozen member P(D): {arm}/{cell['match']}/{seed}")
                prediction = supervised.decode(probability, period,
                                               float(inputs["fps"]), lt.COST)
                head_receipt = _json(_head_path(arm, cell["source"], seed).with_suffix(".json"))
                member_row = oe.score_prediction(
                    labels, prediction, inputs, f"{arm}_seed_{seed}", cell,
                    digest_array(probability))
                member_control = cell["member_baselines"][str(seed)]["row"]
                if (member_row["graded_frames"] != member_control["graded_frames"] or
                        member_row["n_gt_spells"] != member_control["n_gt_spells"]):
                    raise ValueError(f"Seed changed graded population: {arm}/{cell['match']}/{seed}")
                seed_rows.append({**member_row, "saved_baseline": member_control})
                member_files.append(_candidate_member(
                    arm, cell, seed, probability, prediction, head_receipt))
                candidate_by_arm[arm].append(probability)
            ensemble = oe.blend_probabilities(*candidate_by_arm[arm])
            if not np.array_equal(ensemble[:, 2], saved_ensemble[:, 2]):
                raise ValueError(f"Candidate changed ensemble P(D): {arm}/{cell['match']}")
            prediction = supervised.decode(ensemble, period, float(inputs["fps"]), lt.COST)
            output_path = OUT / f"candidate_{arm}_{cell['match']}_{cell['view']}_{cell['replicate']}_ensemble.npz"
            _save_npz(output_path, probability=ensemble,
                      prediction=prediction.astype("U1"))
            row = oe.score_prediction(
                labels, prediction, inputs, arm, cell, digest_array(ensemble),
                sha256(output_path))
            control_row = cell["ensemble_baseline_row"]
            if (row["graded_frames"] != control_row["graded_frames"] or
                    row["n_gt_spells"] != control_row["n_gt_spells"]):
                raise ValueError(f"Candidate changed graded population: {arm}/{cell['match']}")
            all_rows.append({**row, "saved_baseline": control_row})
            ensemble_files.append({
                "recipe": arm, "source": cell["source"], "match": cell["match"],
                "view": cell["view"], "replicate": cell["replicate"],
                "path": str(output_path.relative_to(DATA)),
                "sha256": sha256(output_path),
                "probability_sha256": digest_array(ensemble),
                "prediction_sha256": digest_array(prediction.astype("U1")),
                "p_d_exact": True,
            })
            d_checks.append({"recipe": arm, "match": cell["match"],
                             "view": cell["view"], "replicate": cell["replicate"],
                             "p_d_exact": True})
        del cache, labels, inputs, geometry
        print("SCORED NATIVE CELL", cell["source"], cell["match"],
              cell["view"], cell["replicate"], flush=True)

    controls = {}
    for view in ("clean", "feed_like"):
        controls[view] = {}
        for provider in PROVIDERS:
            cells = [cell["ensemble_baseline_row"] for cell in baseline
                     if cell["source"] == provider and cell["view"] == view]
            controls[view]["per_provider"] = controls[view].get("per_provider", {})
            controls[view]["per_provider"][provider] = {
                name: float(np.mean([
                    row[name] if not name.startswith("iou_") else
                    row["spell_by_iou"][name.split("_")[1]]
                    for row in cells
                ]))
                for name in ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                             "d_precision", "d_recall", "short_spell_recall",
                             "iou_0.3", "iou_0.4", "iou_0.5", "iou_0.6", "iou_0.7")
            }
            controls[view]["per_provider"][provider]["n_cells"] = len(cells)
        metric_names = ("spell_f1", "macro_f1", "weighted_70_30", "d_f1",
                        "d_precision", "d_recall", "short_spell_recall",
                        "iou_0.3", "iou_0.4", "iou_0.5", "iou_0.6", "iou_0.7")
        controls[view]["equal_provider_mean"] = {
            name: float(np.mean([controls[view]["per_provider"][p][name]
                                 for p in PROVIDERS])) for name in metric_names
        }
        controls[view]["aggregation"] = "mean within provider, then equal mean across providers"

    # Compute paired candidate summaries against the already-saved score rows.
    aggregate_rows = []
    for row in all_rows:
        aggregate_rows.append({
            **{key: row[key] for key in (
                "recipe", "source", "match", "view", "replicate", "spell_f1",
                "macro_f1", "weighted_70_30", "d_f1", "d_precision", "d_recall",
                "short_spell_recall", "spell_by_iou")},
        })
    aggregate = _aggregate(aggregate_rows, controls)
    result = {
        "study_id": "native_short_spell_v1",
        "declaration_sha256": declaration_sha,
        "fit_seal_sha256": sha256(FIT_SEAL),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(ARMS), "scored_cells_per_arm": 15,
        "seed_member_rows": len(seed_rows),
        "control": controls,
        "aggregate": aggregate,
        "rows": all_rows,
        "seed_rows": seed_rows,
        "candidate_seed_artifacts": member_files,
        "candidate_ensemble_artifacts": ensemble_files,
        "p_d_exact_checks": d_checks,
        "organizer_labels_or_scores_used": False,
        "automatic_promotion": False,
    }
    if READOUT.exists():
        raise ValueError("Native short-spell readout is immutable")
    _write_exclusive(READOUT, result)
    artifacts = {
        "study_id": result["study_id"],
        "declaration_sha256": declaration_sha,
        "fit_seal_sha256": sha256(FIT_SEAL),
        "readout": str(READOUT.relative_to(ROOT)),
        "readout_sha256": sha256(READOUT),
        "candidate_seed_files": member_files,
        "candidate_ensemble_files": ensemble_files,
        "organizer_labels_or_scores_used": False,
        "independent_confirmation": False,
    }
    artifact_path = ROOT / "experiments/results/native_short_spell_artifacts.json"
    _write_exclusive(artifact_path, artifacts)
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "freeze", "check", "fit", "seal", "score", "report"))
    args = parser.parse_args()
    if args.action == "preflight":
        result = preflight()
    elif args.action == "freeze":
        result = freeze()
    elif args.action == "check":
        declaration, sha = _verify_declaration()
        result = {"declaration_sha256": sha, "source_and_train_inputs_verified": True,
                  "scored": bool(READOUT.is_file()),
                  "expected_heads": len(ARMS) * len(PROVIDERS) * len(SEEDS)}
    elif args.action == "fit":
        result = fit_all()
    elif args.action == "seal":
        result = seal_fits()
    elif args.action == "score":
        result = score()
    else:
        if not READOUT.is_file():
            raise ValueError("No saved native short-spell readout")
        result = _json(READOUT)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
