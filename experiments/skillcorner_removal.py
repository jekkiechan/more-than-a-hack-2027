"""Bounded comparison of the active two-seed GRU with all SkillCorner TRAIN data removed.

All artifacts stay under ignored data/. This runner is separate from the frozen
SkillCorner-expansion runner and never accesses organizer assets, labels,
scores, or scoring gates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import joblib
import numpy as np

from experiments import (combined_objective_deployment as deployment,
                         learned_temporal as lt, preparation as prep,
                         skillcorner_expansion as expansion)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("MTAH_DATA_DIR", ROOT / "data")).resolve()
OUT = DATA / "experiments/skillcorner_removal_v1"
MANIFEST = ROOT / "experiments/eval_set/candidates/skillcorner_removal_v1.json"
OLD_MANIFEST = ROOT / "experiments/eval_set/manifest.json"
BASELINE_MANIFEST = ROOT / "experiments/baseline_manifest.json"
DECISION_POLICY = ROOT / "experiments/model_decision_policy.json"
OLD_CACHE = DATA / "experiments/objective_alignment_v1"
TRAIN_PREDECLARED = OLD_CACHE / "predeclared.json"
TRAIN_CHECKPOINTS = OLD_CACHE / "checkpoints_frozen.json"
BASELINE_PACKAGE = DATA / "models/joint_gru_combined_ensemble_v1"
DEPLOYMENT_REFERENCE = ROOT / "experiments/results/combined_objective_deployment_declaration.json"
BASELINE_ROWS = ROOT / "experiments/results/objective_ensemble_rows.json"
SEEDS = (2026, 2027)
FIT_FOLDS = ("leave_out_sportec", "leave_out_metrica", "pooled")
ALL_FOLDS = (*FIT_FOLDS, "leave_out_skillcorner")
VIEWS = (("clean", "clean"), ("feed_like", "corrupt_1"), ("feed_like", "corrupt_2"))
PROVIDERS = ("sportec", "metrica", "skillcorner")
TACTICAL_PROVIDERS = ("sportec", "metrica")
BATCH_SIZE = 64
OBJECTIVE = "combined"
SCORE_KEYS = (
    "spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
    "weighted_70_30", "short_spell_recall", "n_gt_spells", "n_pred_spells",
    "short_reference_spells", "fragmented_reference_spells", "graded_frames",
)
GATES = {
    "delta_direction": "candidate_no_skillcorner_minus_current_with_skillcorner",
    "primary_original_five": {
        "clean_spell_delta_min": 0.01,
        "feed_like_spell_delta_min": -0.01,
        "each_provider_clean_macro_f1_delta_min": -0.02,
        "each_provider_clean_d_f1_delta_min": -0.02,
    },
    "pooled_tactical_sportec_metrica": {
        "clean_spell_delta_strictly_positive": True,
        "feed_like_spell_delta_min": -0.01,
        "each_provider_clean_macro_f1_delta_min": -0.02,
        "each_provider_clean_d_f1_delta_min": -0.02,
    },
    "automatic_promotion": False,
}


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path | str):
    return json.loads(Path(path).read_text())


def write_once(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Refusing to change a completed artifact: {path}")
        return
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _relative_data_path(path: Path) -> str:
    return str(Path(path).resolve().relative_to(DATA))


def source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha(path)
        for folder in (ROOT / "experiments", ROOT / "evaluation")
        for path in sorted(folder.rglob("*.py"))
    }


def _dependency_versions() -> dict[str, str]:
    packages = ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl", "torch")
    result = {name: importlib.metadata.version(name) for name in packages}
    result["python"] = platform.python_version()
    return result


def read_manifest() -> dict:
    value = read(MANIFEST)
    original = read(OLD_MANIFEST)
    for fold in ALL_FOLDS:
        prep.split_plan(MANIFEST, fold)
    if value["sources"] != original["sources"]:
        raise ValueError("Removal manifest changed the frozen source/match records")
    if value["historical_tests_excluded"] != original["historical_tests_excluded"]:
        raise ValueError("Historical test exclusions changed")
    if value["organizer_game"].get("training") is not False or value["organizer_game"].get("selection") is not False:
        raise ValueError("Organizer data must be excluded from this experiment")
    design = value.get("skillcorner_removal_v1", {})
    train = {item["match"] for source in value["sources"].values() for item in source["train"]}
    selection = {item["match"] for source in value["sources"].values() for item in source["selection"]}
    skillcorner_train = {item["match"] for item in value["sources"]["skillcorner"]["train"]}
    if (len(train), len(selection), skillcorner_train) != (13, 5, set(design.get("removed_training_matches", []))):
        raise ValueError("Unexpected 13-train/5-selection cohort or SkillCorner removal set")
    if design.get("current_arm_training_matches") != sorted(train):
        raise ValueError("Manifest control training match list changed")
    if design.get("selection_matches") != sorted(selection):
        raise ValueError("Manifest selection match list changed")
    if design.get("views") != [list(view) for view in VIEWS]:
        raise ValueError("Manifest evaluation views changed")
    pooled_design = design["candidate_folds"]["pooled"]
    if (pooled_design.get("selection_sources") != list(PROVIDERS) or
            pooled_design.get("tactical_summary_sources") != list(TACTICAL_PROVIDERS)):
        raise ValueError("Pooled evaluation must retain all five matches and name the tactical summary separately")
    fold_selection_union = set()
    for fold in ALL_FOLDS:
        plan = prep.split_plan(MANIFEST, fold)
        expected_candidate = design["candidate_folds"][fold]
        candidate_train = [item for item in plan["train"] if item["source"] != "skillcorner"]
        actual_matches = sorted(item["match"] for item in candidate_train)
        if actual_matches != sorted(expected_candidate["candidate_training_matches"]):
            raise ValueError(f"Candidate training membership changed for {fold}")
        expected_selection_sources = next(
            item["selection_sources"] for item in value["folds"] if item["name"] == fold)
        if fold == "pooled":
            if expected_candidate.get("selection_sources") != expected_selection_sources:
                raise ValueError("Pooled candidate selection-source declaration differs from canonical split")
        elif expected_candidate.get("selection_source") != expected_selection_sources[0]:
            raise ValueError(f"Candidate selection source differs from canonical fold: {fold}")
        expected_selection = {
            item["match"] for source in expected_selection_sources
            for item in value["sources"][source]["selection"]
        }
        actual_selection = {item["match"] for item in plan["selection"]}
        if actual_selection != expected_selection:
            raise ValueError(f"Selection cohort changed for {fold}")
        if len(actual_selection) != {"leave_out_sportec": 1, "leave_out_metrica": 1,
                                     "leave_out_skillcorner": 3, "pooled": 5}[fold]:
            raise ValueError(f"Unexpected selection match count for {fold}")
        fold_selection_union.update(expected_selection)
    if fold_selection_union != selection:
        raise ValueError("Fold selection union does not match the original five")
    if train & selection or set(original["historical_tests_excluded"]) & (train | selection):
        raise ValueError("Match leakage across train, selection, or historical tests")
    return value


def selection_items() -> list[dict]:
    value = read_manifest()
    result = [dict(item, source=source)
              for source in PROVIDERS
              for item in value["sources"][source]["selection"]]
    if len(result) != 5 or len({item["match"] for item in result}) != 5:
        raise ValueError("Expected the exact original five selection matches")
    return result


def training_items(fold: str) -> list[dict]:
    if fold not in ALL_FOLDS:
        raise ValueError(f"Unknown fold: {fold}")
    plan = prep.split_plan(MANIFEST, fold)
    values = [item for item in plan["train"] if item["source"] != "skillcorner"]
    expected = read_manifest()["skillcorner_removal_v1"]["candidate_folds"][fold]
    if sorted(item["match"] for item in values) != sorted(expected["candidate_training_matches"]):
        raise ValueError(f"Candidate training items are not the frozen split for {fold}")
    if any(item["source"] == "skillcorner" for item in values):
        raise ValueError("A candidate fit contains SkillCorner training data")
    return values


def control_path(fold: str, seed: int) -> Path:
    if seed not in SEEDS or fold not in ALL_FOLDS:
        raise ValueError("Control checkpoint is outside the declared fold/seed set")
    if fold == "pooled":
        return BASELINE_PACKAGE / f"joint_gru_{seed}.joblib"
    source = fold.removeprefix("leave_out_")
    return OLD_CACHE / f"without_{source}_combined_{seed}.joblib"


def candidate_path(fold: str, seed: int) -> Path:
    if fold == "leave_out_skillcorner":
        return control_path(fold, seed)
    if fold not in FIT_FOLDS or seed not in SEEDS:
        raise ValueError("Only the six declared no-SkillCorner fits have candidate paths")
    return OUT / f"no_skillcorner_{fold}_{seed}.joblib"


def _control_checkpoint_declaration(fold: str, seed: int) -> dict:
    path = control_path(fold, seed)
    if fold == "pooled":
        manifest = read(BASELINE_MANIFEST)
        if manifest.get("package_version") != "joint_gru_combined_ensemble_v1":
            raise ValueError("The current combined-objective baseline package changed")
        expected = manifest["models_sha256"].get(path.name)
        if expected is None:
            raise ValueError(f"Current baseline manifest has no hash for {path.name}")
        package_manifest = BASELINE_PACKAGE / "baseline_manifest.json"
        if sha(package_manifest) != sha(BASELINE_MANIFEST):
            raise ValueError("Restored pooled package manifest differs from the active baseline")
        receipt = path.with_suffix(".json")
        return {
            "path": _relative_data_path(path), "sha256": expected,
            "manifest": str(BASELINE_MANIFEST.relative_to(ROOT)),
            "manifest_sha256": sha(BASELINE_MANIFEST),
            "package_manifest": _relative_data_path(package_manifest),
            "package_manifest_sha256": sha(package_manifest),
            "receipt": _relative_data_path(receipt) if receipt.is_file() else None,
            "receipt_sha256": sha(receipt) if receipt.is_file() else None,
        }
    checkpoint_manifest = read(TRAIN_CHECKPOINTS)
    entries = [item for item in checkpoint_manifest["new_checkpoints"] if item["file"] == path.name]
    if len(entries) != 1:
        raise ValueError(f"Historical checkpoint manifest lacks a unique {path.name}")
    entry = entries[0]
    receipt = path.with_suffix(".json")
    if not receipt.is_file() or sha(receipt) != entry["receipt_sha256"]:
        raise ValueError(f"Historical current-GRU fit receipt missing or changed: {receipt}")
    return {
        "path": _relative_data_path(path), "sha256": entry["sha256"],
        "manifest": _relative_data_path(TRAIN_CHECKPOINTS),
        "manifest_sha256": sha(TRAIN_CHECKPOINTS),
        "receipt": _relative_data_path(receipt),
        "receipt_sha256": entry["receipt_sha256"],
    }


def _training_cache_declarations() -> dict[str, dict]:
    old = read(TRAIN_PREDECLARED)
    declarations = {}
    for source in PROVIDERS:
        for item in read_manifest()["sources"][source]["train"]:
            identity = item["match"]
            entry = old["prepared_training_caches"].get(identity)
            if entry is None or entry["input_sha256"] != item["sha256"]:
                raise ValueError(f"No matching restored training-cache receipt for {identity}")
            cache = DATA / Path(entry["cache"]).relative_to("data")
            receipt = DATA / Path(entry["receipt"]).relative_to("data")
            raw = prep.data_path(item, DATA)
            if sha(raw) != item["sha256"] or sha(cache) != entry["sha256"] or sha(receipt) != entry["receipt_sha256"]:
                raise ValueError(f"Restored training input/cache/receipt changed: {identity}")
            declarations[identity] = {
                "source": source, "input": _relative_data_path(raw),
                "input_sha256": item["sha256"], "cache": _relative_data_path(cache),
                "cache_sha256": entry["sha256"], "receipt": _relative_data_path(receipt),
                "receipt_sha256": entry["receipt_sha256"],
                "preparation_sha256": old["preparation_declaration_sha256"],
            }
    if len(declarations) != 13:
        raise ValueError("Expected exactly the original thirteen restored training caches")
    return declarations


def _evaluation_cache_declarations() -> dict[str, dict]:
    old = read(TRAIN_PREDECLARED)["evaluation_feature_cache"]["files"]
    result = {}
    for item in selection_items():
        for view, replicate in VIEWS:
            key = f"{item['source']}/{item['match']}/{view}/{replicate}"
            entry = old.get(key)
            if entry is None:
                raise ValueError(f"Missing restored evaluation cache declaration: {key}")
            path = DATA / Path(entry["path"]).relative_to("data")
            if sha(path) != entry["sha256"]:
                raise ValueError(f"Restored evaluation cache changed: {key}")
            result[key] = {"path": _relative_data_path(path), "sha256": entry["sha256"],
                           "bytes": path.stat().st_size}
    if len(result) != 15:
        raise ValueError("Expected exactly the original fifteen evaluation feature caches")
    return result


def _control_probability_declarations() -> dict[str, dict]:
    result = {}
    for item in selection_items():
        for view, replicate in VIEWS:
            for seed in SEEDS:
                path = OLD_CACHE / (
                    f"{item['source']}_{item['match']}_{view}_{replicate}_combined_{seed}.npz")
                if not path.is_file():
                    raise ValueError(f"Missing saved full-provider control probabilities: {path}")
                result[_relative_data_path(path)] = {
                    "source": item["source"], "match": item["match"],
                    "view": view, "replicate": replicate, "seed": seed,
                    "sha256": sha(path),
                }
    if len(result) != 30:
        raise ValueError("Expected all thirty saved per-seed primary control probability caches")
    return result


def _load_training_records_for(items: list[dict], spec: dict | None = None) -> list[dict]:
    old = read(TRAIN_PREDECLARED)
    records = []
    for item in items:
        entry = old["prepared_training_caches"].get(item["match"])
        if entry is None:
            raise ValueError(f"Training cache missing from restored manifest: {item['match']}")
        path = DATA / Path(entry["cache"]).relative_to("data")
        relative = _relative_data_path(path)
        expected = (spec or {}).get("training_cache_sha256", {}).get(relative, entry["sha256"])
        if sha(path) != expected or expected != entry["sha256"]:
            raise ValueError(f"Frozen training cache changed: {item['match']}")
        receipt = DATA / Path(entry["receipt"]).relative_to("data")
        receipt_relative = _relative_data_path(receipt)
        receipt_expected = (spec or {}).get("training_receipt_sha256", {}).get(receipt_relative,
                                                                                entry["receipt_sha256"])
        if sha(receipt) != receipt_expected or receipt_expected != entry["receipt_sha256"]:
            raise ValueError(f"Frozen training-cache receipt changed: {item['match']}")
        cache = joblib.load(path)
        if (cache.get("match") != item["match"] or cache.get("source") != item["source"] or
                cache.get("input_sha256") != item["sha256"] or
                cache.get("preparation_sha256") != old["preparation_declaration_sha256"]):
            raise ValueError(f"Training cache identity mismatch: {item['match']}")
        native_y = np.asarray(cache.get("full_y"))
        native_period = np.asarray(cache.get("full_period"))
        if native_y.ndim != 1 or native_period.shape != native_y.shape or not np.issubdtype(native_y.dtype, np.integer):
            raise ValueError(f"Missing full-native training labels/periods: {item['match']}")
        for view in ("clean", "gap", "dropout_legacy", "dropout_gap"):
            record = cache["records"].get(view)
            if record is None:
                raise ValueError(f"Missing legacy training record {item['match']}/{view}")
            if (str(record.get("source")) != item["source"] or
                    str(record.get("match")) != item["match"] or
                    len(record["y"]) != len(record["x"]) or
                    not np.array_equal(record["period"], native_period[record["indices"]]) or
                    not np.array_equal(record["y"], native_y[record["indices"]])):
                raise ValueError(f"Training record source/match/label alignment failed: {item['match']}/{view}")
        records.extend(deployment.paired_records(cache, item["match"]))
        del cache
    identities = {(str(record["source"]), str(record["match"])) for record in records}
    expected = {(item["source"], item["match"]) for item in items}
    if identities != expected or any(source == "skillcorner" for source, _ in identities):
        raise ValueError("Candidate records differ from the frozen source/match training partition")
    return records


def _window_count(records: list[dict]) -> int:
    return sum(sum(1 for _ in lt.chunks(record)) for record in records)


def _verify_control(path: Path, fold: str, seed: int, expected_sha: str) -> dict:
    if not path.is_file() or sha(path) != expected_sha:
        raise ValueError(f"Restored control checkpoint missing or changed: {path}")
    fitted = joblib.load(path)
    if (fitted.get("kind") != "factorized_had" or fitted.get("objective") != OBJECTIVE or
            fitted.get("player_only") is not True or fitted.get("seed") != seed or
            fitted.get("parameters") != 62643):
        raise ValueError(f"Unexpected current-GRU control metadata: {fold}/{seed}")
    return fitted


def _build_declaration() -> dict:
    value = read_manifest()
    baseline = read(BASELINE_MANIFEST)
    package_manifest = BASELINE_PACKAGE / "baseline_manifest.json"
    if sha(package_manifest) != sha(BASELINE_MANIFEST):
        raise ValueError("Restored active baseline package differs from its tracked manifest")
    if read(DECISION_POLICY).get("working_candidate", {}).get("model") != "joint_gru_combined_ensemble":
        raise ValueError("The active decision policy no longer names the current GRU control")
    training_caches = _training_cache_declarations()
    evaluation_caches = _evaluation_cache_declarations()
    control_probabilities = _control_probability_declarations()
    all_inputs = {}
    for source in PROVIDERS:
        for role in ("train", "selection"):
            for item in value["sources"][source][role]:
                raw = prep.data_path(item, DATA)
                if sha(raw) != item["sha256"]:
                    raise ValueError(f"External match input changed: {item['match']}")
                all_inputs[_relative_data_path(raw)] = item["sha256"]

    controls, control_budgets = {}, {}
    for fold in ALL_FOLDS:
        for seed in SEEDS:
            entry = _control_checkpoint_declaration(fold, seed)
            model = _verify_control(DATA / entry["path"], fold, seed, entry["sha256"])
            key = f"{fold}:{seed}"
            controls[key] = entry
            control_budgets[key] = {
                name: int(model[name])
                for name in ("windows_per_epoch", "optimizer_updates", "parameters")
            }

    sampling = {}
    for fold in FIT_FOLDS:
        records = _load_training_records_for(training_items(fold))
        dataset_windows = _window_count(records)
        if dataset_windows <= 0:
            raise ValueError(f"No training windows for candidate {fold}")
        expected = control_budgets[f"{fold}:2026"]
        if any(control_budgets[f"{fold}:{seed}"]["windows_per_epoch"] != expected["windows_per_epoch"]
               for seed in SEEDS):
            raise ValueError(f"Seed controls disagree on windows for {fold}")
        budget = expected["windows_per_epoch"]
        batches = (budget + BATCH_SIZE - 1) // BATCH_SIZE
        if expected["optimizer_updates"] != batches * lt.EPOCHS:
            raise ValueError(f"Unexpected control optimizer budget for {fold}")
        full_cycles, remainder = divmod(budget, dataset_windows)
        sampling[fold] = {
            "candidate_dataset_windows": dataset_windows,
            "dataset_unique_windows_per_epoch": min(dataset_windows, budget),
            "requested_windows_per_epoch": budget,
            "full_cycles": full_cycles,
            "remainder": remainder,
            "expected_windows_per_epoch": budget,
            "batch_size": BATCH_SIZE,
            "epochs": int(lt.EPOCHS),
            "batches_per_epoch": batches,
            "expected_optimizer_updates": int(expected["optimizer_updates"]),
            "method": "torch RandomSampler(replacement=False,num_samples=budget): full random permutations and a fresh shuffled remainder",
        }
        del records

    return {
        "schema_version": 1,
        "study_id": "skillcorner_removal_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": value["skillcorner_removal_v1"]["question"],
        "arms": {
            "current": "Adopted combined-objective hidden24 player-only-D two-seed GRU ensemble trained on the original 13 TRAIN matches.",
            "no_skillcorner": "Same current recipe with all eight original SkillCorner TRAIN matches removed and no replacement source.",
        },
        "seeds": list(SEEDS),
        "fit_folds": list(FIT_FOLDS),
        "new_fits": 6,
        "parity_fold": "leave_out_skillcorner reuses the exact saved current no-SkillCorner full-provider controls; zero refits.",
        "cohorts": {
            "original_train_matches": sorted(
                item["match"] for source in PROVIDERS for item in value["sources"][source]["train"]),
            "removed_skillcorner_matches": sorted(
                item["match"] for item in value["sources"]["skillcorner"]["train"]),
            "primary_original_five_selection_matches": sorted(item["match"] for item in selection_items()),
            "primary_treated_providers": list(TACTICAL_PROVIDERS),
            "all_selection_matches_historically_exposed": True,
            "fresh_confirmation_available": False,
        },
        "split_manifest": str(MANIFEST.relative_to(ROOT)),
        "split_manifest_sha256": sha(MANIFEST),
        "canonical_split_manifest_sha256": sha(OLD_MANIFEST),
        "baseline_manifest_sha256": sha(BASELINE_MANIFEST),
        "baseline_package_manifest_sha256": sha(package_manifest),
        "decision_policy_sha256": sha(DECISION_POLICY),
        "deployment_reference_sha256": sha(DEPLOYMENT_REFERENCE),
        "baseline_rows_sha256": sha(BASELINE_ROWS),
        "objective_alignment_predeclared_sha256": sha(TRAIN_PREDECLARED),
        "objective_alignment_checkpoint_manifest_sha256": sha(TRAIN_CHECKPOINTS),
        "source_sha256": source_hashes(),
        "dependency_versions": _dependency_versions(),
        "input_sha256": all_inputs,
        "training_cache_sha256": {
            entry["cache"]: entry["cache_sha256"] for entry in training_caches.values()},
        "training_receipt_sha256": {
            entry["receipt"]: entry["receipt_sha256"] for entry in training_caches.values()},
        "training_caches": training_caches,
        "evaluation_cache_sha256": {
            entry["path"]: entry["sha256"] for entry in evaluation_caches.values()},
        "evaluation_caches": evaluation_caches,
        "control_probability_sha256": {
            path: entry["sha256"] for path, entry in control_probabilities.items()},
        "control_probabilities": control_probabilities,
        "control_checkpoint_sha256": {
            entry["path"]: entry["sha256"] for entry in controls.values()},
        "control_checkpoints": controls,
        "control_budgets": control_budgets,
        "candidate_sampling": sampling,
        "preparation": "Reuse exactly restored unnormalized objective-alignment records. Pair clean/gap with dropout_legacy/dropout_gap and direct/H-A swap using the frozen deployment helper; do not alter features or augmentations.",
        "normalization": "Recompute median/std from each no-SkillCorner training fold only; never reuse the with-SkillCorner control scaler.",
        "model": "Factorized hidden24 player-only-D GRU; combined duration+transition objective; 204 feature schema; AdamW lr .001, weight decay .0001, gradient clip 1.",
        "training": {
            "epochs": int(lt.EPOCHS), "batch_size": BATCH_SIZE,
            "core_frames": int(lt.CORE), "context_frames_each_side": int(lt.CONTEXT),
            "rate_hz": 5, "loss_coefficients": {"spell": 1.0, "transition": 0.1},
        },
        "source_mass": "The unchanged objective loss gives each training source equal total frame mass and matches within source equal mass; removal naturally renormalizes remaining sources. This estimates the full source-removal recipe effect, not an isolated marginal label effect.",
        "sampling": "Eight epochs, batch 64, and the per-fold control windows/updates. For smaller candidates, each epoch samples full random permutations without replacement and one fresh shuffled prefix for the remainder; all unique windows are used and repeated only to fill the frozen window budget. Equal-size DataLoader calls pass through unchanged.",
        "evaluation": {
            "primary": "Original five full-provider selection matches; clean and two fixed feed_like corruptions; saved baseline controls and removal candidates use matched leave-one-provider-out folds.",
            "pooled_supplement": "Same original five selection matches and selection exclusion for both arms; tactical Sportec+Metrica equal-provider summary is secondary.",
            "skillcorner_primary": "Exact current no-SkillCorner checkpoint parity, not a candidate effect; shown in policy context and excluded from source-attribution summary.",
            "metric": "Official greedy spell F1 mean over IoU .3/.4/.5/.6/.7, macro F1, D metrics, weighted 70/30 auxiliary, short recall, spell counts, fragmentation and graded frame counts.",
            "decoder": {"native_probability_mean": "Equal actual probabilities from seeds 2026 and 2027",
                        "viterbi_switch_cost": int(lt.COST)},
        },
        "gates": GATES,
        "gate_interpretation": "The original-five +.01 clean removal threshold implies a +.015 mean gain across the two treated providers because SkillCorner cells are exact parity. Failing a removal recommendation gate does not prove that SkillCorner helps. No automatic baseline/model promotion.",
        "organizer": {
            "inputs_read": False, "labels_read": False, "gates_claimed": False,
            "scores_run": False, "public_scores_used_for_selection": False,
        },
    }


def freeze() -> dict:
    path = OUT / "predeclared.json"
    if path.exists():
        check()
        return read(path)
    declaration = _build_declaration()
    _validate_declaration(declaration)
    write_once(path, declaration)
    print("FROZEN", sha(path), flush=True)
    return declaration


def _validate_declaration(spec: dict) -> dict:
    if spec.get("study_id") != "skillcorner_removal_v1" or spec.get("gates") != GATES:
        raise ValueError("Frozen removal study identity or gates changed")
    if source_hashes() != spec["source_sha256"]:
        raise ValueError("Frozen experiment source changed")
    if (sha(MANIFEST) != spec["split_manifest_sha256"] or
            sha(OLD_MANIFEST) != spec["canonical_split_manifest_sha256"]):
        raise ValueError("Frozen split manifest changed")
    frozen_refs = (
        (BASELINE_MANIFEST, "baseline_manifest_sha256"),
        (BASELINE_PACKAGE / "baseline_manifest.json", "baseline_package_manifest_sha256"),
        (DECISION_POLICY, "decision_policy_sha256"),
        (DEPLOYMENT_REFERENCE, "deployment_reference_sha256"),
        (BASELINE_ROWS, "baseline_rows_sha256"),
        (TRAIN_PREDECLARED, "objective_alignment_predeclared_sha256"),
        (TRAIN_CHECKPOINTS, "objective_alignment_checkpoint_manifest_sha256"),
    )
    if any(sha(path) != spec[key] for path, key in frozen_refs):
        raise ValueError("Frozen current-baseline or preparation reference changed")
    if _dependency_versions() != spec["dependency_versions"]:
        raise ValueError("Frozen dependency versions changed")
    read_manifest()
    for relative, expected in spec["input_sha256"].items():
        if sha(DATA / relative) != expected:
            raise ValueError(f"Frozen external input changed: {relative}")
    for group in ("training_cache_sha256", "training_receipt_sha256",
                  "evaluation_cache_sha256", "control_checkpoint_sha256",
                  "control_probability_sha256"):
        for relative, expected in spec[group].items():
            if sha(DATA / relative) != expected:
                raise ValueError(f"Frozen artifact changed ({group}): {relative}")
    for entry in spec["control_checkpoints"].values():
        for path_key, hash_key in (("receipt", "receipt_sha256"),
                                   ("package_manifest", "package_manifest_sha256")):
            if entry.get(path_key) and sha(DATA / entry[path_key]) != entry[hash_key]:
                raise ValueError(f"Frozen control sidecar changed: {entry[path_key]}")
    return spec


def check() -> dict:
    path = OUT / "predeclared.json"
    if not path.is_file():
        raise ValueError("No frozen removal declaration; freeze before fitting")
    return _validate_declaration(read(path))


def load_training_records(fold: str, spec: dict | None = None) -> list[dict]:
    spec = check() if spec is None else spec
    return _load_training_records_for(training_items(fold), spec)


def fit(fold: str, seed: int) -> dict:
    from experiments import objective_losses, removal_budget

    spec = check()
    if fold not in FIT_FOLDS or seed not in SEEDS:
        raise ValueError("Fit request is outside the exact six declared new fits")
    if (OUT / "checkpoints_frozen.json").exists():
        raise ValueError("Training is sealed")
    path = candidate_path(fold, seed)
    receipt_path = path.with_suffix(".json")
    if path.exists() or receipt_path.exists():
        if not path.is_file() or not receipt_path.is_file():
            raise ValueError("Found an incomplete candidate checkpoint/receipt")
        receipt = read(receipt_path)
        if (sha(path) != receipt.get("sha256") or
                receipt.get("declaration_sha256") != sha(OUT / "predeclared.json")):
            raise ValueError("Existing candidate checkpoint changed")
        return receipt

    records = load_training_records(fold, spec)
    expected = spec["control_budgets"][f"{fold}:{seed}"]
    expected_sampling = spec["candidate_sampling"][fold]
    if _window_count(records) != expected_sampling["candidate_dataset_windows"]:
        raise ValueError(f"Candidate window count changed for {fold}")
    started = time.monotonic()
    with deployment.file_backed_normalization(OUT / "temporary") as storage:
        with removal_budget.matched_window_budget(expected["windows_per_epoch"]) as budget:
            model = objective_losses.fit(records, player_only=True, seed=seed, objective=OBJECTIVE)
    if (not storage.get("normalizer_restored") or
            not storage.get("temporary_files_removed") or
            len(storage.get("normalization_calls", [])) != 2 * len(records)):
        raise ValueError("Training-only file-backed normalization contract failed")
    if (model.get("seed") != seed or model.get("objective") != OBJECTIVE or
            model.get("player_only") is not True or
            model.get("windows_per_epoch") != expected_sampling["candidate_dataset_windows"] or
            model.get("optimizer_updates") != expected["optimizer_updates"] or
            model.get("parameters") != expected["parameters"]):
        raise ValueError("Candidate failed the frozen model/window/update budget contract")
    actual = budget.as_dict()
    for key, expected_value in (
        ("windows_drawn_per_epoch", expected["windows_per_epoch"]),
        ("expected_optimizer_updates", expected["optimizer_updates"]),
        ("dataset_windows", expected_sampling["candidate_dataset_windows"]),
        ("full_cycles", expected_sampling["full_cycles"]),
        ("remainder", expected_sampling["remainder"]),
    ):
        if actual[key] != expected_value:
            raise ValueError(f"Sampler audit changed {key}: {actual[key]} != {expected_value}")
    # Preserve objective_losses' unique dataset count; record the actual
    # matched-budget draw count separately because inference ignores both.
    model["unique_windows_per_epoch"] = int(model["windows_per_epoch"])
    model["effective_windows_drawn_per_epoch"] = int(actual["windows_drawn_per_epoch"])
    model["sampling_audit"] = actual
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        joblib.dump(model, temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)
    receipt = {
        "fold": fold, "seed": seed, "sha256": sha(path),
        "declaration_sha256": sha(OUT / "predeclared.json"),
        "training_matches": [item["match"] for item in training_items(fold)],
        "training_sources": sorted({item["source"] for item in training_items(fold)}),
        "training_record_count": len(records),
        "candidate_dataset_windows": expected_sampling["candidate_dataset_windows"],
        "parameters": int(model["parameters"]),
        "windows_per_epoch": int(model["windows_per_epoch"]),
        "unique_windows_per_epoch": int(model["unique_windows_per_epoch"]),
        "effective_windows_drawn_per_epoch": int(model["effective_windows_drawn_per_epoch"]),
        "optimizer_updates": int(model["optimizer_updates"]),
        "sampling": actual, "file_backed_normalization": storage,
        "seconds": time.monotonic() - started,
        "organizer_inputs_read": False, "organizer_labels_read": False,
        "evaluation_scoring_performed": False,
    }
    write_once(receipt_path, receipt)
    print("FITTED", fold, seed, receipt["seconds"], flush=True)
    return receipt


def seal() -> dict:
    check()
    rows = []
    for fold in FIT_FOLDS:
        for seed in SEEDS:
            path = candidate_path(fold, seed)
            receipt_path = path.with_suffix(".json")
            if not path.is_file() or not receipt_path.is_file():
                raise ValueError(f"Cannot seal incomplete six-fit set: {fold}/{seed}")
            receipt = read(receipt_path)
            if (sha(path) != receipt.get("sha256") or
                    receipt.get("declaration_sha256") != sha(OUT / "predeclared.json") or
                    receipt.get("fold") != fold or receipt.get("seed") != seed):
                raise ValueError(f"Candidate checkpoint or fit receipt changed: {fold}/{seed}")
            rows.append({
                "fold": fold, "seed": seed, "path": _relative_data_path(path),
                "sha256": sha(path), "receipt": _relative_data_path(receipt_path),
                "receipt_sha256": sha(receipt_path),
            })
    value = {
        "study_id": "skillcorner_removal_v1",
        "declaration_sha256": sha(OUT / "predeclared.json"),
        "checkpoints": rows, "all_six_fits_sealed": True,
    }
    write_once(OUT / "checkpoints_frozen.json", value)
    if read(OUT / "checkpoints_frozen.json") != value:
        raise ValueError("Checkpoint seal differs from the six verified fits")
    return value


def _check_seal() -> dict:
    spec = check()
    path = OUT / "checkpoints_frozen.json"
    if not path.is_file():
        raise ValueError("Seal all six candidate checkpoints before evaluation")
    frozen = read(path)
    if frozen.get("declaration_sha256") != sha(OUT / "predeclared.json"):
        raise ValueError("Checkpoint seal belongs to another declaration")
    expected = {(fold, seed) for fold in FIT_FOLDS for seed in SEEDS}
    actual = {(entry["fold"], entry["seed"]) for entry in frozen.get("checkpoints", [])}
    if actual != expected or len(frozen["checkpoints"]) != 6:
        raise ValueError("Candidate checkpoint seal is not the complete six-fit set")
    for entry in frozen["checkpoints"]:
        for path_key, hash_key in (("path", "sha256"), ("receipt", "receipt_sha256")):
            if sha(DATA / entry[path_key]) != entry[hash_key]:
                raise ValueError(f"Sealed candidate checkpoint artifact changed: {entry[path_key]}")
    for fold in ALL_FOLDS:
        for seed in SEEDS:
            entry = spec["control_checkpoints"][f"{fold}:{seed}"]
            if sha(DATA / entry["path"]) != entry["sha256"]:
                raise ValueError(f"Sealed control checkpoint changed: {entry['path']}")
    return frozen


def inference_record(record: dict, native_n: int | None = None) -> dict:
    """Project a prepared record onto tracking-only fields used by inference."""
    if native_n is None:
        native_n = record.get("native_n")
    if isinstance(native_n, bool) or not isinstance(native_n, (int, np.integer)) or native_n <= 0:
        raise ValueError("Inference requires the native row count from tracking inputs")
    required = ("x", "names", "period", "indices")
    if any(key not in record for key in required):
        raise ValueError("Inference record is missing features, schema, periods, or sample indices")
    result = {key: record[key] for key in required}
    result["native_n"] = int(native_n)
    return result


def evaluation_cache(item: dict, view: str, replicate: str,
                     spec: dict | None = None) -> tuple[Path, dict, np.ndarray, dict]:
    spec = check() if spec is None else spec
    key = f"{item['source']}/{item['match']}/{view}/{replicate}"
    entry = spec["evaluation_caches"].get(key)
    if entry is None:
        raise ValueError(f"Evaluation cell is outside the frozen original-five cohort: {key}")
    path = DATA / entry["path"]
    if sha(path) != entry["sha256"]:
        raise ValueError(f"Evaluation feature cache changed: {key}")
    cached = joblib.load(path)
    record = cached.get("record", cached.get("joint"))
    if record is None:
        raise ValueError(f"Evaluation cache has no temporal record: {key}")
    # Bind each loaded view to its frozen provider and match before inference.
    if (str(record.get("source")) != item["source"] or
            str(record.get("match")) != item["match"]):
        raise ValueError(f"Evaluation record source/match differs from manifest: {key}")
    inputs = cached.get("inputs")
    labels = np.asarray(cached.get("labels"))
    n = len(inputs.get("period", [])) if isinstance(inputs, dict) else -1
    if (not isinstance(inputs, dict) or len(labels) != n or
            len(inputs.get("timestamps", [])) != n or
            len(record.get("y", [])) != len(record.get("x", [])) or
            len(record.get("indices", [])) != len(record.get("y", []))):
        raise ValueError(f"Evaluation labels/native rows do not align: {key}")
    indices = np.asarray(record["indices"])
    if (len(indices) == 0 or not np.issubdtype(indices.dtype, np.integer) or
            indices.min() < 0 or indices.max() >= n):
        raise ValueError(f"Evaluation temporal indices are invalid: {key}")
    if not np.isin(labels, ("H", "A", "D", "X")).all():
        raise ValueError(f"Evaluation labels contain unexpected values: {key}")
    return path, inputs, labels, record


def _verify_historical_control_probability(item: dict, view: str, replicate: str,
                                           seed: int, probability: np.ndarray,
                                           spec: dict) -> None:
    relative = (
        f"experiments/objective_alignment_v1/{item['source']}_{item['match']}_"
        f"{view}_{replicate}_combined_{seed}.npz")
    expected = spec["control_probability_sha256"].get(relative)
    if expected is None or sha(DATA / relative) != expected:
        raise ValueError(f"Saved primary control probability changed or is undeclared: {relative}")
    with np.load(DATA / relative, allow_pickle=False) as saved:
        archived = saved["probability"]
    if archived.shape != probability.shape or not np.allclose(probability, archived, rtol=0, atol=1e-6):
        raise ValueError(f"Current control probability parity failed: {item['match']}/{view}/{replicate}/{seed}")


def _evaluate_cell(protocol: str, item: dict, view: str, replicate: str,
                   spec: dict) -> None:
    artifact = OUT / f"{protocol}_{item['match']}_{view}_{replicate}.json"
    if artifact.exists():
        validate_result(artifact, spec)
        return
    probability_path = artifact.with_suffix(".npz")
    if probability_path.exists():
        raise ValueError(f"Unreceipted probability artifact already exists: {probability_path}")
    cache_path, inputs, labels, record = evaluation_cache(item, view, replicate, spec)
    inference = inference_record(record, native_n=len(inputs["period"]))
    fold = f"leave_out_{item['source']}" if protocol == "primary" else "pooled"
    if fold not in ALL_FOLDS:
        raise ValueError(f"Unknown evaluation fold: {fold}")
    arrays, rows = {}, []
    parity = protocol == "primary" and item["source"] == "skillcorner"
    for arm in ("current", "no_skillcorner"):
        for seed in SEEDS:
            if arm == "current":
                entry = spec["control_checkpoints"][f"{fold}:{seed}"]
                model_path = DATA / entry["path"]
                if sha(model_path) != entry["sha256"]:
                    raise ValueError(f"Current control checkpoint changed: {fold}/{seed}")
            elif parity:
                # The SkillCorner holdout model was already trained without SK.
                entry = spec["control_checkpoints"][f"leave_out_skillcorner:{seed}"]
                model_path = DATA / entry["path"]
                if sha(model_path) != entry["sha256"]:
                    raise ValueError(f"SkillCorner parity checkpoint changed: {seed}")
            else:
                model_path = candidate_path(fold, seed)
            if parity and arm == "no_skillcorner":
                probability = arrays[f"current_{seed}"]
            else:
                probability = expansion.predict_native(inference, joblib.load(model_path), inputs)
            arrays[f"{arm}_{seed}"] = probability
            if protocol == "primary" and arm == "current":
                _verify_historical_control_probability(item, view, replicate, seed, probability, spec)
        if parity:
            arrays["no_skillcorner_2026"] = arrays["current_2026"]
            arrays["no_skillcorner_2027"] = arrays["current_2027"]
        arrays[f"{arm}_ensemble"] = .5 * (arrays[f"{arm}_2026"] + arrays[f"{arm}_2027"])
        for member in ("2026", "2027", "ensemble"):
            metrics, prediction = expansion.score_cell(inputs, labels, arrays[f"{arm}_{member}"])
            arrays[f"{arm}_{member}_prediction"] = prediction
            rows.append({
                "protocol": protocol, "source": item["source"], "match": item["match"],
                "view": view, "replicate": replicate, "arm": arm, "member": member,
                "skillcorner_primary_exact_parity": parity, **metrics,
            })
    if parity and any(not np.array_equal(arrays[f"current_{seed}"], arrays[f"no_skillcorner_{seed}"])
                      for seed in SEEDS):
        raise ValueError("SkillCorner primary parity cells must reuse identical probabilities")
    if (len({row["n_gt_spells"] for row in rows}) != 1 or
            len({row["graded_frames"] for row in rows}) != 1):
        raise ValueError("Paired evaluation labels or graded-frame counts differ")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{probability_path.name}.",
                                     dir=artifact.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, probability_path)
    finally:
        temporary.unlink(missing_ok=True)
    write_once(artifact, {
        "rows": rows, "probability_sha256": sha(probability_path),
        "cache": _relative_data_path(cache_path), "cache_sha256": sha(cache_path),
        "declaration_sha256": sha(OUT / "predeclared.json"),
        "candidate_checkpoint_seal_sha256": sha(OUT / "checkpoints_frozen.json"),
    })
    print("SCORED", protocol, item["source"], item["match"], view, replicate, flush=True)


def validate_result(path: Path | str, spec: dict | None = None) -> list[dict]:
    path = Path(path)
    spec = check() if spec is None else spec
    value = read(path)
    if (value.get("declaration_sha256") != sha(OUT / "predeclared.json") or
            value.get("candidate_checkpoint_seal_sha256") != sha(OUT / "checkpoints_frozen.json")):
        raise ValueError("Evaluation result belongs to another declaration or checkpoint seal")
    cache = DATA / value["cache"]
    if sha(cache) != value["cache_sha256"]:
        raise ValueError("Evaluation cache changed after scoring")
    probability_path = path.with_suffix(".npz")
    if sha(probability_path) != value["probability_sha256"]:
        raise ValueError("Saved probabilities changed")
    with np.load(probability_path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    arms = ("current", "no_skillcorner")
    members = ("2026", "2027", "ensemble")
    if len(value["rows"]) != 6 or {
            (row["arm"], row["member"]) for row in value["rows"]} != {
            (arm, member) for arm in arms for member in members}:
        raise ValueError("Incomplete paired rows in saved evaluation result")
    protocol = "primary" if path.name.startswith("primary_") else "pooled" if path.name.startswith("pooled_") else None
    suffix = next(((view, replicate) for view, replicate in VIEWS
                   if path.stem.endswith(f"_{view}_{replicate}")), None)
    if protocol is None or suffix is None:
        raise ValueError("Evaluation artifact path has an invalid protocol/view identity")
    view, replicate = suffix
    match = path.stem[len(protocol) + 1:-(len(f"_{view}_{replicate}"))]
    matches = [item for item in selection_items() if item["match"] == match]
    if len(matches) != 1:
        raise ValueError("Saved result match is not in the frozen original-five manifest")
    item = matches[0]
    key = f"{item['source']}/{item['match']}/{view}/{replicate}"
    expected_cache = spec["evaluation_caches"][key]
    if (value["cache"] != expected_cache["path"] or
            value["cache_sha256"] != expected_cache["sha256"]):
        raise ValueError("Saved score cache provider/match differs from frozen manifest")
    expected_arrays = {
        f"{arm}_{member}{suffix}" for arm in arms for member in members
        for suffix in ("", "_prediction")
    }
    if set(arrays) != expected_arrays:
        raise ValueError("Saved probability archive has unexpected arrays")
    for arm in arms:
        if not np.array_equal(arrays[f"{arm}_ensemble"],
                              .5 * (arrays[f"{arm}_2026"] + arrays[f"{arm}_2027"])):
            raise ValueError(f"{arm} ensemble is not the actual seed probability mean")
    parity = protocol == "primary" and item["source"] == "skillcorner"
    if parity and any(not np.array_equal(arrays[f"current_{seed}"], arrays[f"no_skillcorner_{seed}"])
                      for seed in SEEDS):
        raise ValueError("A SkillCorner primary cell differs from its exact parity arm")
    _, inputs, labels, _ = evaluation_cache(item, view, replicate, spec)
    expected_rows = {
        (protocol, item["source"], item["match"], view, replicate, arm, member, parity)
        for arm in arms for member in members
    }
    actual_rows = {
        (row["protocol"], row["source"], row["match"], row["view"], row["replicate"],
         row["arm"], row["member"], row["skillcorner_primary_exact_parity"])
        for row in value["rows"]
    }
    if actual_rows != expected_rows:
        raise ValueError("Saved row source/provider/match differs from the frozen manifest")
    for row in value["rows"]:
        array_key = f"{row['arm']}_{row['member']}"
        metrics, prediction = expansion.score_cell(inputs, labels, arrays[array_key])
        if not np.array_equal(prediction, arrays[f"{array_key}_prediction"]):
            raise ValueError("Saved decoded prediction differs from unchanged cost-3 decoder")
        for name, actual in metrics.items():
            saved = row.get(name)
            if isinstance(actual, dict):
                if actual != saved:
                    raise ValueError(f"Saved official replay differs on {name}")
            elif isinstance(actual, float):
                if saved is None or abs(actual - saved) > 1e-12:
                    raise ValueError(f"Saved official replay differs on {name}")
            elif actual != saved:
                raise ValueError(f"Saved official replay differs on {name}")
    return value["rows"]


def evaluate(protocol: str = "primary", match: str | None = None) -> None:
    if protocol not in ("primary", "pooled"):
        raise ValueError("Protocol must be primary or pooled")
    spec = check()
    _check_seal()
    for item in selection_items():
        if match is not None and item["match"] != match:
            continue
        for view, replicate in VIEWS:
            _evaluate_cell(protocol, item, view, replicate, spec)


def _choose_rows(rows: list[dict], protocol: str, arm: str, view: str, member: str,
                 sources: tuple[str, ...] | None = None) -> list[dict]:
    return [
        row for row in rows
        if (row["protocol"], row["arm"], row["view"], row["member"]) ==
           (protocol, arm, view, member)
        and (sources is None or row["source"] in sources)
    ]


def summarize(rows: list[dict], protocol: str, arm: str, view: str,
              member: str = "ensemble",
              sources: tuple[str, ...] | None = None) -> dict:
    chosen = _choose_rows(rows, protocol, arm, view, member, sources)
    if not chosen:
        raise ValueError("Empty summary population")
    per_provider = {}
    for source in sorted({row["source"] for row in chosen}):
        provider_rows = [row for row in chosen if row["source"] == source]
        by_match = {}
        for row in provider_rows:
            by_match.setdefault(row["match"], []).append(row)
        match_summaries = []
        for match, match_rows in sorted(by_match.items()):
            values = {"match": match}
            for key in SCORE_KEYS:
                scored = [float(row[key]) for row in match_rows if row[key] is not None]
                values[key] = float(np.mean(scored)) if scored else None
            values["spell_by_iou"] = {
                iou: float(np.mean([row["spell_by_iou"][iou] for row in match_rows]))
                for iou in ("0.3", "0.4", "0.5", "0.6", "0.7")
            }
            match_summaries.append(values)
        summary = {}
        for key in SCORE_KEYS:
            values = [row[key] for row in match_summaries if row[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
        summary["spell_by_iou"] = {
            iou: float(np.mean([row["spell_by_iou"][iou] for row in match_summaries]))
            for iou in ("0.3", "0.4", "0.5", "0.6", "0.7")
        }
        summary["matches"] = match_summaries
        per_provider[source] = summary
    equal_provider = {}
    for key in SCORE_KEYS:
        values = [provider[key] for provider in per_provider.values() if provider[key] is not None]
        equal_provider[key] = float(np.mean(values)) if values else None
    equal_provider["spell_by_iou"] = {
        iou: float(np.mean([provider["spell_by_iou"][iou] for provider in per_provider.values()]))
        for iou in ("0.3", "0.4", "0.5", "0.6", "0.7")
    }
    return {
        "provider_aggregation": "mean within provider, then equal mean across listed providers",
        "providers": per_provider, "equal_provider_mean": equal_provider,
    }


def _delta(candidate: dict, current: dict) -> dict:
    result = {}
    for key in SCORE_KEYS:
        left, right = candidate.get(key), current.get(key)
        result[key] = float(left - right) if left is not None and right is not None else None
    result["spell_by_iou"] = {
        iou: float(candidate["spell_by_iou"][iou] - current["spell_by_iou"][iou])
        for iou in ("0.3", "0.4", "0.5", "0.6", "0.7")
    }
    return result


def _primary_control_parity(rows: list[dict]) -> None:
    baseline_rows = read(ROOT / "experiments/results/objective_ensemble_rows.json")["rows"]
    for row in rows:
        if (row["protocol"], row["arm"], row["member"]) != ("primary", "current", "ensemble"):
            continue
        matched = [
            baseline for baseline in baseline_rows
            if (baseline["source"], baseline["match"], baseline["view"],
                baseline["replicate"], baseline["recipe"]) ==
               (row["source"], row["match"], row["view"], row["replicate"],
                "combined_objective_equal_probability_ensemble")
        ]
        if len(matched) != 1:
            raise ValueError(f"No unique saved current-control row: {row['match']}/{row['view']}/{row['replicate']}")
        for key in ("spell_f1", "macro_f1", "d_f1", "n_gt_spells",
                    "n_pred_spells", "graded_frames"):
            if abs(row[key] - matched[0][key]) > 1e-9:
                raise ValueError(f"Historical current-control metric parity failed: {key}/{row['match']}")


def _gate_readout(summaries: dict) -> dict:
    primary_clean = summaries["primary:all:clean:ensemble"]["delta"]
    primary_feed = summaries["primary:all:feed_like:ensemble"]["delta"]
    primary_providers = summaries["primary:all:clean:ensemble"]["provider_delta"]
    pooled_clean = summaries["pooled:tactical:clean:ensemble"]["delta"]
    pooled_feed = summaries["pooled:tactical:feed_like:ensemble"]["delta"]
    pooled_providers = summaries["pooled:tactical:clean:ensemble"]["provider_delta"]
    checks = {
        "primary_full_five_clean_spell_at_least_plus_0_01": primary_clean["spell_f1"] >= 0.01,
        "primary_full_five_feed_like_spell_at_least_minus_0_01": primary_feed["spell_f1"] >= -0.01,
    }
    for source in PROVIDERS:
        checks[f"primary_{source}_clean_macro_at_least_minus_0_02"] = (
            primary_providers[source]["macro_f1"] >= -0.02)
        checks[f"primary_{source}_clean_d_f1_at_least_minus_0_02"] = (
            primary_providers[source]["d_f1"] >= -0.02)
    checks["pooled_tactical_clean_spell_strictly_positive"] = pooled_clean["spell_f1"] > 0.0
    checks["pooled_tactical_feed_like_spell_at_least_minus_0_01"] = pooled_feed["spell_f1"] >= -0.01
    for source in TACTICAL_PROVIDERS:
        checks[f"pooled_{source}_clean_macro_at_least_minus_0_02"] = (
            pooled_providers[source]["macro_f1"] >= -0.02)
        checks[f"pooled_{source}_clean_d_f1_at_least_minus_0_02"] = (
            pooled_providers[source]["d_f1"] >= -0.02)
    return {
        "removal_recommendation_passes": all(checks.values()),
        "automatic_promotion": False,
        "interpretation": "A failed removal screen does not prove that retaining SkillCorner helps; these historically exposed matches are not independent confirmation.",
        "checks": checks,
    }


def report() -> dict:
    _check_seal()
    spec = check()
    rows = []
    for protocol in ("primary", "pooled"):
        for item in selection_items():
            for view, replicate in VIEWS:
                result_path = OUT / f"{protocol}_{item['match']}_{view}_{replicate}.json"
                rows.extend(validate_result(result_path, spec))
    _primary_control_parity(rows)

    summaries = {}
    for protocol in ("primary", "pooled"):
        for cohort, sources in (("all", None), ("tactical", TACTICAL_PROVIDERS)):
            for view in ("clean", "feed_like"):
                current = summarize(rows, protocol, "current", view, "ensemble", sources)
                candidate = summarize(rows, protocol, "no_skillcorner", view, "ensemble", sources)
                provider_delta = {
                    source: _delta(candidate["providers"][source], current["providers"][source])
                    for source in current["providers"]
                }
                summaries[f"{protocol}:{cohort}:{view}:ensemble"] = {
                    "current": current, "no_skillcorner": candidate,
                    "delta": _delta(candidate["equal_provider_mean"], current["equal_provider_mean"]),
                    "provider_delta": provider_delta,
                    "skillcorner_primary_parity_excluded_from_attribution":
                        protocol == "primary" and cohort == "tactical",
                }
                for member in ("2026", "2027"):
                    seed_current = summarize(rows, protocol, "current", view, member, sources)
                    seed_candidate = summarize(rows, protocol, "no_skillcorner", view, member, sources)
                    summaries[f"{protocol}:{cohort}:{view}:{member}"] = {
                        "current": seed_current, "no_skillcorner": seed_candidate,
                        "delta": _delta(seed_candidate["equal_provider_mean"],
                                        seed_current["equal_provider_mean"]),
                        "provider_delta": {
                            source: _delta(seed_candidate["providers"][source],
                                           seed_current["providers"][source])
                            for source in seed_current["providers"]
                        },
                        "skillcorner_primary_parity_excluded_from_attribution":
                            protocol == "primary" and cohort == "tactical",
                    }
    gates = _gate_readout(summaries)
    result = {
        "study_id": "skillcorner_removal_v1",
        "declaration_sha256": sha(OUT / "predeclared.json"),
        "checkpoint_seal_sha256": sha(OUT / "checkpoints_frozen.json"),
        "delta_direction": GATES["delta_direction"],
        "primary_original_five": {
            "clean": summaries["primary:all:clean:ensemble"],
            "feed_like": summaries["primary:all:feed_like:ensemble"],
            "tactical_sportec_metrica_clean": summaries["primary:tactical:clean:ensemble"],
            "tactical_sportec_metrica_feed_like": summaries["primary:tactical:feed_like:ensemble"],
            "skillcorner_cells": "Exact parity under full-provider holdout; excluded from source-attribution estimand.",
        },
        "pooled_supplement": {
            "tactical_sportec_metrica_clean": summaries["pooled:tactical:clean:ensemble"],
            "tactical_sportec_metrica_feed_like": summaries["pooled:tactical:feed_like:ensemble"],
            "all_five_clean_diagnostic": summaries["pooled:all:clean:ensemble"],
            "all_five_feed_like_diagnostic": summaries["pooled:all:feed_like:ensemble"],
            "skillcorner_and_all_five_are_diagnostics_only": True,
        },
        "per_seed_and_ensemble": summaries,
        "gates": gates,
        "headline_baseline_parity": "passed",
        "organizer_inputs_read": False, "organizer_labels_read": False,
        "organizer_gates_claimed": False, "organizer_scores_run": False,
        "automatic_model_promotion": False,
        "completed_result_rows": len(rows), "new_fits": 6,
    }
    write_once(OUT / "report.json", result)
    write_once(OUT / "external_rows.json", {"rows": rows})
    print(json.dumps({
        "removal_recommendation_passes": gates["removal_recommendation_passes"],
        "primary_clean_delta": summaries["primary:all:clean:ensemble"]["delta"]["spell_f1"],
        "primary_feed_like_delta": summaries["primary:all:feed_like:ensemble"]["delta"]["spell_f1"],
        "tactical_clean_delta": summaries["primary:tactical:clean:ensemble"]["delta"]["spell_f1"],
        "tactical_feed_like_delta": summaries["primary:tactical:feed_like:ensemble"]["delta"]["spell_f1"],
        "checks": gates["checks"],
    }, indent=2), flush=True)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("freeze", help="verify restored controls/caches and freeze the declaration")
    fit_parser = sub.add_parser("fit", help="fit one declared no-SkillCorner seed/fold")
    fit_parser.add_argument("--fold", choices=FIT_FOLDS, required=True)
    fit_parser.add_argument("--seed", choices=tuple(str(seed) for seed in SEEDS), required=True)
    sub.add_parser("seal", help="seal all six candidate checkpoints before scoring")
    evaluate_parser = sub.add_parser("evaluate", help="score one frozen external protocol")
    evaluate_parser.add_argument("--protocol", choices=("primary", "pooled"), required=True)
    evaluate_parser.add_argument("--match", help="optional original-five match id")
    sub.add_parser("report", help="validate saved cells and write the fixed external readout")
    sub.add_parser("check", help="recheck the frozen declaration and all referenced hashes")
    args = parser.parse_args(argv)
    if args.action == "freeze":
        freeze()
    elif args.action == "fit":
        fit(args.fold, int(args.seed))
    elif args.action == "seal":
        seal()
    elif args.action == "evaluate":
        evaluate(args.protocol, args.match)
    elif args.action == "report":
        report()
    else:
        spec = check()
        print(json.dumps({
            "study_id": spec["study_id"],
            "declaration_sha256": sha(OUT / "predeclared.json"),
            "inputs": len(spec["input_sha256"]),
            "training_caches": len(spec["training_caches"]),
            "evaluation_caches": len(spec["evaluation_caches"]),
            "control_checkpoints": len(spec["control_checkpoints"]),
        }, indent=2))


if __name__ == "__main__":
    main()

