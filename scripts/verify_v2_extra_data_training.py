"""Read-only integrity audit for the v2_extra_data_v1 training seal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np

from experiments import v2_extra_data_train as train
from experiments import v2_spell_frontier_train as frontier


def _digest_rows(rows):
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _equal(left, right, context):
    if left != right:
        raise ValueError(f"{context} differs")


def verify():
    """Verify frozen TRAIN artifacts only; never loads evaluation/public labels."""
    declaration = train.check(full=True)
    seal = json.loads(train.SEAL.read_text())
    if (seal.get("study_id") != train.STUDY or
            seal.get("declaration_sha256") != train.sha(train.DECL) or
            seal.get("all_18_complete") is not True):
        raise ValueError("Training seal is incomplete or bound to another declaration")
    expected = {(arm, fold, seed) for arm in train.ARMS
                for fold in train.FOLDS for seed in train.SEEDS}
    by_key = {}
    for row in seal.get("fits", []):
        key = (row.get("arm"), row.get("fold"), int(row.get("seed", -1)))
        if key in by_key or key not in expected:
            raise ValueError(f"Duplicate or undeclared fit in seal: {key}")
        by_key[key] = row
    if set(by_key) != expected:
        raise ValueError("Training seal does not contain exactly the 18 declared fits")

    recipe = frontier.prior_run.bind()
    cohorts = {fold: recipe.prep.split_plan(fold=fold)["train"] for fold in train.FOLDS}
    cohort_inventory = {}
    for fold, cohort in cohorts.items():
        excluded = fold.removeprefix("leave_out_").lower()
        if any(str(item["source"]).lower() == excluded for item in cohort):
            raise ValueError(f"Held-out provider occurs in declared TRAIN cohort: {fold}")
        cohort_inventory[fold] = sorted({str(item["match"]) for item in cohort})
    report = {"study_id": train.STUDY, "declaration_sha256": train.sha(train.DECL),
        "seal_sha256": train.sha(train.SEAL), "fit_count": len(by_key),
        "fits": [], "paired_checks": [], "organizer_reference_access": False,
        "evaluation_labels_loaded": False, "public_labels_loaded": False}
    receipt_by_key = {}
    for arm, fold, seed in sorted(expected):
        row = by_key[(arm, fold, seed)]
        path = train.output_path(arm, fold, seed)
        receipt_path = Path(str(path) + ".json")
        if not path.is_file() or not receipt_path.is_file():
            raise FileNotFoundError(f"Missing model or receipt: {arm}/{fold}/{seed}")
        receipt = json.loads(receipt_path.read_text())
        if (row.get("path") != str(path.relative_to(train.ROOT)) or
                row.get("sha256") != train.sha(path) or
                receipt.get("sha256") != train.sha(path) or
                row.get("declaration_sha256") != train.sha(train.DECL) or
                receipt.get("declaration_sha256") != train.sha(train.DECL)):
            raise ValueError(f"Model/receipt/seal hash mismatch: {arm}/{fold}/{seed}")
        for field in ("arm", "fold", "seed", "stage", "warm_start_sha256", "teacher_sha256",
                      "sampled_window_ids", "draw_window_count", "draw_window_sequence_sha256",
                      "exposures", "gar_train_games"):
            if field in row and row[field] != receipt.get(field):
                raise ValueError(f"Sealed fit row and receipt differ on {field}: {arm}/{fold}/{seed}")
        sampled = receipt.get("sampled_window_ids")
        if (not isinstance(sampled, list) or len(sampled) != train.UPDATES *
                train.iterative.BATCH_SIZE or
                receipt.get("draw_window_count") != len(sampled) or
                receipt.get("draw_window_sequence_sha256") != _digest_rows(sampled)):
            raise ValueError(f"Sampled-window audit mismatch: {arm}/{fold}/{seed}")

        warm_path = train.prior_model_path(fold, seed)
        warm_receipt = json.loads(Path(str(warm_path) + ".json").read_text())
        warm = joblib.load(warm_path)
        model = joblib.load(path)
        teacher_path = recipe.rf.sc.initial_path(fold, seed)
        teacher = joblib.load(teacher_path)
        teacher_sha = train.iterative.cap._checkpoint_sha(teacher)
        if (warm.get("training", {}).get("study") != frontier.parent_train.STUDY or
                warm.get("training", {}).get("stage") != train.BASE_STAGE or
                warm_receipt.get("stage") != train.BASE_STAGE or
                warm_receipt.get("sha256") != train.sha(warm_path) or
                receipt.get("warm_start_sha256") != train.sha(warm_path) or
                receipt.get("teacher_sha256") != train.sha(teacher_path) or
                warm.get("source", {}).get("checkpoint_sha256") != teacher_sha or
                model.get("source", {}).get("checkpoint_sha256") != teacher_sha):
            raise ValueError(f"Warm-start, teacher or candidate lineage mismatch: {arm}/{fold}/{seed}")
        if (receipt.get("stage") != train.BASE_STAGE + train.UPDATES or
                model.get("training", {}).get("cumulative_updates") !=
                train.BASE_STAGE + train.UPDATES or
                model.get("training", {}).get("arm") != arm):
            raise ValueError(f"Candidate training provenance mismatch: {arm}/{fold}/{seed}")
        _equal(model.get("feature_names"), warm.get("feature_names"), "feature schema")
        for field in ("median", "scale"):
            np.testing.assert_array_equal(model[field], warm[field])
        if set(model.get("d_state", {})) != set(warm.get("d_state", {})):
            raise ValueError(f"D-state keys changed: {arm}/{fold}/{seed}")
        for name, value in warm["d_state"].items():
            np.testing.assert_array_equal(model["d_state"][name], value)

        # Use the cached lightweight TRAIN split metadata rather than reloading
        # large prepared feature arrays for every fit.
        if receipt.get("old_matches") != cohort_inventory[fold]:
            raise ValueError(f"Original TRAIN match inventory differs: {arm}/{fold}/{seed}")
        games = receipt.get("gar_train_games", [])
        if (len(games) != 45 or len(set(games)) != 45 or
                model.get("training", {}).get("gar_train_games") != games):
            raise ValueError(f"GAR TRAIN game inventory differs: {arm}/{fold}/{seed}")
        receipt_by_key[(arm, fold, seed)] = receipt
        report["fits"].append({"arm": arm, "fold": fold, "seed": seed,
            "sha256": train.sha(path), "stage": receipt["stage"],
            "draw_window_count": len(sampled), "old_match_count": len(cohort_inventory[fold]),
            "gar_game_count": len(games), "d_state_unchanged": True,
            "normalization_unchanged": True, "held_out_provider_excluded": True})

    for fold in train.FOLDS:
        for seed in train.SEEDS:
            control = receipt_by_key[("fresh_control", fold, seed)]
            weighted = receipt_by_key[("gar_weight625", fold, seed)]
            quality = receipt_by_key[("gar_quality50", fold, seed)]
            _equal(control["sampled_window_ids"], weighted["sampled_window_ids"],
                   "fresh-control/weight-arm sampled windows")
            _equal(control["exposures"], weighted["exposures"],
                   "fresh-control/weight-arm exposures")
            for arm_receipt in (weighted, quality):
                _equal(control["exposures"], arm_receipt["exposures"],
                       "same-schedule match/game exposures")
                control_old = [r for r in control["sampled_window_ids"] if r[0] == "old"]
                arm_old = [r for r in arm_receipt["sampled_window_ids"] if r[0] == "old"]
                _equal(control_old, arm_old, "original-window sequence across arms")
            q_counts = quality["sampling"]["counts"]["new"]
            c_counts = control["sampling"]["counts"]["new"]
            if (quality["sampling"].get("gar_quality_replacement_rate") != .5 or
                    control["sampling"].get("gar_quality_replacement_rate") != .25 or
                    quality["sampling"].get("original_replacement_rate") != .25 or
                    q_counts.get("replacement_draws", 0) <= c_counts.get("replacement_draws", 0)):
                raise ValueError(f"GAR quality replacement rate/count mismatch: {fold}/{seed}")
            if quality["sampling"].get("quality_counts") != control["sampling"].get("quality_counts"):
                raise ValueError(f"Quality catalog changed between arms: {fold}/{seed}")
            report["paired_checks"].append({"fold": fold, "seed": seed,
                "fresh_weight_windows_identical": True,
                "old_windows_identical_across_arms": True,
                "game_exposures_identical": True,
                "gar_quality_replacement_counts": {
                    "control_25pct": c_counts.get("replacement_draws", 0),
                    "quality_50pct": q_counts.get("replacement_draws", 0)}})
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
        help="Optional path for the aggregate audit report; defaults to stdout")
    args = parser.parse_args()
    result = verify()
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    else:
        print(serialized)


if __name__ == "__main__":
    main()
