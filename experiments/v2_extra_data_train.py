"""Bounded data-mix continuations from the sealed v2 spell frontier checkpoint.

The arms change only the continuation draw schedule, GAR loss share, or the
rate of quality-gated GAR window replacement. All learning targets, views,
normalization and the copied D component remain shared with the frontier fit.
"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random as std_random
import time
from types import SimpleNamespace

import numpy as np
import joblib

from experiments import short_original_diversity_v1 as short
from experiments import v2_iterative_train as iterative
from experiments import v2_spell_frontier_train as frontier

STUDY = "v2_extra_data_v1"
ARMS = ("fresh_control", "gar_weight625", "gar_quality50")
UPDATES = 256
BASE_STAGE = 1024
LR = 1e-4
FRESH_OFFSET = 104729
GAR_LOSS_WEIGHT = {"fresh_control": 0.5, "gar_weight625": 0.625,
                   "gar_quality50": 0.5}
ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / f"experiments/results/{STUDY}_plan.json"
DECL = ROOT / f"experiments/results/{STUDY}_training_declaration.json"
SEAL = ROOT / f"experiments/results/{STUDY}_training_seal.json"
OUT = ROOT / "data/experiments" / STUDY / "train"
FOLDS, SEEDS = iterative.FOLDS, iterative.SEEDS


class ExtraDataSampler(short.OriginalDiversitySampler):
    """Keep original replacement at .25 while independently raising GAR quality rate."""

    def __init__(self, old_records, new_records, names, seed, *, quality_rate=.25):
        if not 0 <= float(quality_rate) <= 1:
            raise ValueError("GAR quality replacement rate must be in [0, 1]")
        super().__init__(old_records, new_records, names, seed)
        self.quality_rate = float(quality_rate)

    def replace(self, group, window):
        if group == "old":
            return super().replace(group, window)

        # Consume the legacy GAR replacement stream at its original .25 rate,
        # preserving the paired source/game and original-quality RNG schedule.
        ri = window[0]
        game = str(self.records["new"][ri]["match"])
        selected = window
        if self.rate and self.rng.random() < self.rate:
            pool = self.pool["new"].get(ri, ())
            if pool:
                self.rng.choice(pool)
        if self.quality_rate and self.quality_rng.random() < self.quality_rate:
            self.counts["new"]["requested_replacement"] += 1
            entries = self.catalog.get(game, {})
            if entries:
                if not self.bags.get(game):
                    self.bags[game] = sorted(entries)
                    self.quality_rng.shuffle(self.bags[game])
                    self.counts["new"]["possession_cycles_started"] += 1
                identity = self.bags[game].pop()
                selected = self.quality_rng.choice(entries[identity])
                self.targets[identity] += 1
                for span in self.window_spans[(selected[0], selected[1])]:
                    self.seen[span] += 1
                self.selected_windows[(selected[0], selected[1])] += 1
                self.counts["new"]["replacement_draws"] += 1
        ri = selected[0]
        assert str(self.records["new"][ri]["match"]) == game
        _, start, size, _, _ = selected
        record = self.records["new"][ri]
        labels = np.asarray(record["y"])[start:start + size]
        known = np.isin(labels, (0, 1))
        eligible = self.eligible["new"][ri][start:start + size] & known
        counts = self.counts["new"]
        counts["draws"] += 1
        counts["known_ha_rows"] += int(known.sum())
        counts["short_ha_rows"] += int(eligible.sum())
        counts["short_containing_draws"] += int(eligible.any())
        self.exposures["new/" + str(record["match"])] += 1
        return selected

    def receipt(self):
        result = super().receipt()
        result["gar_quality_replacement_rate"] = self.quality_rate
        result["original_replacement_rate"] = self.rate
        return result


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=1, sort_keys=True, default=float)
        stream.write("\n")


def output_path(arm, fold, seed):
    if arm not in ARMS or fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Undeclared fit")
    return OUT / "models" / arm / f"{fold}_{seed}.joblib"


def _sources():
    recipe = frontier.prior_run.bind()
    paths = {Path(__file__), Path(__file__).with_name("v2_spell_frontier_train.py"),
        Path(iterative.__file__), Path(frontier.parent_train.__file__),
        Path(frontier.prior_run.__file__), ROOT / "experiments/tests/test_v2_extra_data_train.py",
        PLAN, ROOT / "experiments/results/v2_spell_frontier_v1_training_declaration.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_training_seal.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_pooled_declaration.json",
        ROOT / "experiments/results/v2_spell_frontier_v1_pooled_seal.json",
        ROOT / "experiments/baseline_manifest.json",
        *recipe._source_files(), *recipe._view_pipeline_files()}
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)}


def _input_facts():
    frontier.prior_run.bind()
    frontier.parent_train.check(full=True)
    frontier.parent_train.check_pooled(full=True)
    return frontier._input_facts()


def freeze():
    if DECL.exists() or SEAL.exists():
        raise FileExistsError("Preserve the frozen extra-data training artifacts")
    plan = json.loads(PLAN.read_text())
    if plan.get("study_id") != STUDY:
        raise ValueError("Unexpected extra-data plan")
    budget = plan.get("budget", {})
    if (budget.get("folds") != list(FOLDS) or budget.get("seeds") != list(SEEDS) or
            budget.get("fits") != len(ARMS) * len(FOLDS) * len(SEEDS) or
            budget.get("updates_per_fit") != UPDATES or
            float(budget.get("learning_rate", -1)) != LR):
        raise ValueError("Plan budget differs from the fixed trainer")
    if tuple(plan.get("arms", {})) != ARMS:
        raise ValueError("Plan arms/order differ from the fixed trainer")
    for arm in ARMS:
        settings = plan["arms"][arm]
        if (float(settings["gar_loss_weight"]) != GAR_LOSS_WEIGHT[arm] or
                float(settings["gar_quality_replacement_rate"]) !=
                (0.5 if arm == "gar_quality50" else 0.25)):
            raise ValueError(f"Plan arm differs from implementation: {arm}")
    declaration = {"schema": "v2_extra_data_training_v1", "study_id": STUDY,
        "plan_sha256": sha(PLAN), "source_sha256": _sources(),
        "input_sha256": _input_facts(), "folds": list(FOLDS), "seeds": list(SEEDS),
        "arms": list(ARMS), "updates_per_fit": UPDATES,
        "cumulative_stage": BASE_STAGE + UPDATES, "learning_rate": LR,
        "batch_size": iterative.BATCH_SIZE,
        "gar_loss_weights": GAR_LOSS_WEIGHT,
        "sampling": {"old_windows_per_batch": 16, "gar_windows_per_batch": 16,
            "original_short_replacement_rate": .25,
            "gar_quality_replacement_rate": {"fresh_control": .25,
                "gar_weight625": .25, "gar_quality50": .5},
            "trainer_and_original_rng_offset": FRESH_OFFSET},
        "warm_start": "Sealed v2_spell_balanced_v1 stage-1024 fold/seed checkpoint used by the selected frontier pipeline",
        "organizer_reference_access": False, "gar_valid_test_access": False,
        "public_scoring": False, "promotion": False}
    write_once(DECL, declaration)
    return declaration


def check(full=True):
    declaration = json.loads(DECL.read_text())
    if (declaration.get("study_id") != STUDY or
            declaration.get("source_sha256") != _sources()):
        raise ValueError("Extra-data training source changed after freeze")
    if full and declaration.get("input_sha256") != _input_facts():
        raise ValueError("Extra-data training inputs changed after freeze")
    return declaration


@contextmanager
def training_scope(offset: int, gar_loss_weight: float):
    """Override shared trainer globals for one call and restore them afterward."""
    if offset not in (0, FRESH_OFFSET):
        raise ValueError("Undeclared trainer RNG offset")
    if not 0 <= float(gar_loss_weight) <= 1:
        raise ValueError("GAR loss weight must be in [0, 1]")
    old_random, old_lr = iterative.random, iterative.LR
    old_fraction, old_study = iterative.FRACTION_GAR, iterative.STUDY
    iterative.random = SimpleNamespace(
        Random=lambda seed: std_random.Random(int(seed) + offset))
    iterative.LR = LR
    iterative.FRACTION_GAR = float(gar_loss_weight)
    iterative.STUDY = STUDY
    try:
        yield
    finally:
        iterative.random, iterative.LR = old_random, old_lr
        iterative.FRACTION_GAR, iterative.STUDY = old_fraction, old_study


def fit_from_checkpoints(teacher, checkpoint, old_records, gar_records, *,
                         seed: int, arm: str, verbose: bool = True,
                         steps: int = UPDATES):
    """Fit one predeclared arm; caller owns artifact sealing and evaluation."""
    if arm not in ARMS or seed not in iterative.SEEDS:
        raise ValueError("Use a declared arm and seed")
    if int(steps) != steps or not 0 < steps <= UPDATES:
        raise ValueError("Continuation updates must be in [1, 256]")
    offset = FRESH_OFFSET
    quality_rate = 0.5 if arm == "gar_quality50" else 0.25
    sampler = ExtraDataSampler(old_records, gar_records,
        list(teacher["feature_names"]), seed, quality_rate=quality_rate)
    sampler.original_rng = std_random.Random(seed + offset + 1106644)
    initial = iterative.warmstart_state(teacher, checkpoint)
    recipe = frontier.prior_run.bind()
    teacher_sha = iterative.cap._checkpoint_sha(teacher)
    if checkpoint.get("source", {}).get("checkpoint_sha256") != teacher_sha:
        raise ValueError("Stage-1024 checkpoint teacher identity mismatch")
    old_d = {key: np.array(value, copy=True)
             for key, value in checkpoint.get("d_state", {}).items()}
    weight = GAR_LOSS_WEIGHT[arm]
    with frontier.capture_windows(recipe, old_records, gar_records, sampler) as sampled:
        with recipe._round_robin_v2_windows(old_records) as view_draws:
            with training_scope(offset, weight):
                result = iterative._fit_loop(initial, old_records, gar_records,
                    seed=seed, arm="ce_control", verbose=verbose, steps=int(steps))
    model = result["model"]
    for key, value in old_d.items():
        np.testing.assert_array_equal(model["d_state"][key], value)
    if set(model["d_state"]) != set(old_d):
        raise ValueError("D state changed during H/A continuation")
    for field in ("median", "scale"):
        np.testing.assert_array_equal(model[field], checkpoint[field])
    if model["feature_names"] != checkpoint["feature_names"]:
        raise ValueError("Feature schema changed during continuation")
    if model["source"]["checkpoint_sha256"] != teacher_sha:
        raise ValueError("Trainer export changed teacher identity")
    cumulative = BASE_STAGE + int(steps)
    provenance = {"study": STUDY, "arm": arm, "stage": cumulative,
        "stage_updates": int(steps), "cumulative_updates": cumulative,
        "warm_start_stage": BASE_STAGE, "draw_seed_offset": offset,
        "gar_loss_weight": weight,
        "gar_quality_replacement_rate": quality_rate,
        "original_short_replacement_rate": sampler.rate,
        "gar_train_games": sorted({str(row["match"]) for row in gar_records})}
    model["training"].update(provenance)
    model["step"] = cumulative
    result["config"].update(provenance)
    return {**result, "sampler": sampler, "sampling": sampler.receipt(),
        "sampled_windows": sampled, "view_draws": dict(view_draws),
        "provenance": provenance}


def prior_model_path(fold, seed):
    if fold not in FOLDS or seed not in SEEDS:
        raise ValueError("Undeclared fold/seed")
    return frontier.prior_path(fold, seed)


def fit_one(fold, seed, arm, *, verbose=True):
    declaration = check(full=True)
    path = output_path(arm, fold, seed)
    receipt_path = Path(str(path) + ".json")
    if path.exists() or receipt_path.exists():
        raise FileExistsError(path)
    recipe = frontier.prior_run.bind()
    warm_path = prior_model_path(fold, seed)
    warm_receipt_path = Path(str(warm_path) + ".json")
    warm = joblib.load(warm_path)
    warm_receipt = json.loads(warm_receipt_path.read_text())
    if (warm.get("training", {}).get("study") != frontier.parent_train.STUDY or
            warm.get("training", {}).get("stage") != BASE_STAGE or
            warm_receipt.get("sha256") != sha(warm_path) or
            warm_receipt.get("stage") != BASE_STAGE or
            warm_receipt.get("fold") != fold or
            int(warm_receipt.get("seed", -1)) != seed):
        raise ValueError("Warm start is not the sealed frontier stage-1024 fit")
    teacher_path = recipe.rf.sc.initial_path(fold, seed)
    teacher = joblib.load(teacher_path)
    if warm.get("source", {}).get("checkpoint_sha256") != iterative.cap._checkpoint_sha(teacher):
        raise ValueError("Warm start teacher hash mismatch")
    old_records = recipe.training(fold)
    gar_records = recipe.rf.dl.load_dense_records("train", output=recipe.TRAIN_RECORDS)
    games = sorted({str(row["match"]) for row in gar_records})
    if len(games) != 45:
        raise ValueError("Expected exactly 45 GAR TRAIN games")
    started = time.monotonic()
    result = fit_from_checkpoints(teacher, warm, old_records, gar_records,
        seed=seed, arm=arm, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["model"], path)
    sampled = [list(row) for row in result["sampled_windows"]]
    receipt = {"study_id": STUDY, "arm": arm, "fold": fold, "seed": seed,
        "stage": BASE_STAGE + UPDATES, "stage_updates": UPDATES,
        "path": str(path.relative_to(ROOT)), "sha256": sha(path),
        "declaration_sha256": sha(DECL), "warm_start_path": str(warm_path.relative_to(ROOT)),
        "warm_start_sha256": sha(warm_path), "teacher_path": str(teacher_path),
        "teacher_sha256": sha(teacher_path), "wall_seconds": time.monotonic() - started,
        "config": result["config"], "sampling": result["sampling"],
        "view_draws": result["view_draws"], "exposures": result["exposures"],
        "sampled_window_ids": sampled,
        "draw_window_count": len(sampled),
        "draw_window_sequence_sha256": hashlib.sha256(json.dumps(
            sampled, separators=(",", ":")).encode()).hexdigest(),
        "old_matches": sorted({str(row["match"]) for row in old_records}),
        "gar_train_games": games, "d_state_unchanged": True,
        "normalization_unchanged": True, "organizer_reference_access": False}
    write_once(receipt_path, receipt)
    return receipt


def fit_all(workers=2):
    if workers not in (1, 2):
        raise ValueError("Use one or two workers")
    check(full=True)
    if SEAL.exists():
        raise FileExistsError(SEAL)
    jobs = [(arm, fold, seed) for arm in ARMS for fold in FOLDS for seed in SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_one, fold, seed, arm): (arm, fold, seed)
                   for arm, fold, seed in jobs}
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: (ARMS.index(row["arm"]), row["fold"], row["seed"]))
    _validate_matched_pairs(rows)
    write_once(SEAL, {"study_id": STUDY, "declaration_sha256": sha(DECL),
        "fits": rows, "all_18_complete": len(rows) == 18,
        "paired_fresh_control_weight_draws": True,
        "organizer_reference_access": False, "promotion": False})
    return rows


def _validate_matched_pairs(rows):
    for fold in FOLDS:
        for seed in SEEDS:
            pair = [row for row in rows if row["fold"] == fold and row["seed"] == seed]
            control = next(row for row in pair if row["arm"] == "fresh_control")
            weighted = next(row for row in pair if row["arm"] == "gar_weight625")
            if control["sampled_window_ids"] != weighted["sampled_window_ids"]:
                raise ValueError(f"Fresh control and weight arm draws differ: {fold}/{seed}")
            if control["exposures"] != weighted["exposures"]:
                raise ValueError(f"Fresh control and weight arm exposures differ: {fold}/{seed}")


def seal():
    check(full=True)
    rows = []
    for arm in ARMS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = output_path(arm, fold, seed)
                receipt = json.loads(Path(str(path) + ".json").read_text())
                if (receipt.get("declaration_sha256") != sha(DECL) or
                        receipt.get("sha256") != sha(path) or receipt.get("arm") != arm or
                        receipt.get("fold") != fold or receipt.get("seed") != seed):
                    raise ValueError(f"Fit receipt/hash mismatch: {arm}/{fold}/{seed}")
                rows.append(receipt)
    if len(rows) != 18:
        raise ValueError("Expected all 18 declared fits")
    _validate_matched_pairs(rows)
    if SEAL.exists():
        raise FileExistsError(SEAL)
    return write_once(SEAL, {"study_id": STUDY, "declaration_sha256": sha(DECL),
        "fits": rows, "all_18_complete": True,
        "organizer_reference_access": False, "promotion": False})


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "fit", "seal"))
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    frontier.prior_run.bind()
    if args.action == "freeze":
        freeze()
    elif args.action == "fit":
        fit_all(args.workers)
    else:
        seal()


if __name__ == "__main__":
    main()
