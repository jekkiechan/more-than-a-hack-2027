"""Independently rescore frozen GAR predictions and summarize paired game deltas."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import spell_frontier_gar as study
from experiments import spell_frontier_gar_execution as execution

PLAN = ROOT / "experiments/results/spell_frontier_gar_v1_audit_plan.json"
RESULT = ROOT / "experiments/results/spell_frontier_gar_v1_verification.json"
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_DRAWS = 2000


def freeze():
    execution.check()
    study.write_once(PLAN, dict(study=study.STUDY,
        source_sha256=study.digest(Path(__file__)),
        runtime_sha256=study.digest(execution.AMENDMENT),
        bootstrap=dict(unit="whole game, paired candidate minus control",
            draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, percentiles=[5, 95],
            splits_separate=True, use="descriptive only; no selection or promotion threshold"),
        tolerance=1e-12, organizer_reference_access=False, public_scoring=False))


def verify():
    plan = study.read(PLAN)
    if (plan["source_sha256"] != study.digest(Path(__file__)) or
            plan["runtime_sha256"] != study.digest(execution.AMENDMENT)):
        raise ValueError("Frozen audit changed")
    execution.check()
    declaration = study.check_declaration()
    result = study.read(study.RESULT)
    if result["declaration_sha256"] != study.digest(study.DECLARATION):
        raise ValueError("GAR result does not bind its declaration")
    summaries, scored_cells = {}, 0
    for split in study.SPLITS:
        games = declaration["game_ids"][split]
        records = {}
        for game in games:
            row = study.read(study.GAMES / f"{split}__{game}.json")
            path = ROOT / row["prediction_path"]
            if (row["declaration_sha256"] != study.digest(study.DECLARATION) or
                    study.digest(path) != row["prediction_sha256"] or row["phase_count"] != 6):
                raise ValueError("Incomplete or mismatched GAR game: " + game)
            records[game] = {}
            with np.load(path, allow_pickle=False) as cache:
                cell = {k:cache[k] for k in ("period", "timestamps", "fps")}
                cell["labels"] = cache["labels"].astype("U1")
                if len(cell["labels"]) != row["native_frames"] or int((cell["labels"] != "X").sum()) != row["graded_frames"]:
                    raise ValueError("Grading population changed")
                for view in study.VIEWS:
                    pred = cache[f"pred__{view}"].astype("U1")
                    control = cache[f"adopted_control__{view}"].astype("U1")
                    np.testing.assert_array_equal(pred == "D", control == "D")
                    records[game][view] = {}
                    for name, labels in (("spell_frontier", pred), ("original_full_phase_avg", control)):
                        scored = study.audit.score(cell, labels)
                        saved = row["views"][view][name]
                        for metric, value in saved.items():
                            if abs(scored[metric] - value) > plan["tolerance"]:
                                raise ValueError(f"Saved score differs: {split}/{game}/{view}/{name}/{metric}")
                        records[game][view][name] = saved
                    if records[game][view]["spell_frontier"]["reference_spell_count"] != records[game][view]["original_full_phase_avg"]["reference_spell_count"]:
                        raise ValueError("Reference spell count changed")
                    scored_cells += 1
        summary = {}
        for view, subviews in (("clean", ("clean",)), ("v2_mean_over_four_reps", study.VIEWS[1:])):
            expected = result["splits"][split]["scores"][view]
            summary[view] = {}
            for metric in study.MATRIX_KEYS:
                candidate = np.asarray([np.mean([records[g][v]["spell_frontier"][metric] for v in subviews]) for g in games])
                control = np.asarray([np.mean([records[g][v]["original_full_phase_avg"][metric] for v in subviews]) for g in games])
                for actual, field in ((candidate.mean(), "spell_frontier_equal_game"),
                                      (control.mean(), "original_full_phase_avg_equal_game"),
                                      ((candidate-control).mean(), "paired_delta_equal_game")):
                    if abs(actual-expected[field][metric]) > plan["tolerance"]:
                        raise ValueError(f"Equal-game reduction differs: {split}/{view}/{metric}")
                delta = candidate-control
                rng = np.random.default_rng(BOOTSTRAP_SEED)
                means = delta[rng.integers(0, len(games), size=(BOOTSTRAP_DRAWS, len(games)))].mean(axis=1)
                summary[view][metric] = dict(candidate_mean=float(candidate.mean()),
                    control_mean=float(control.mean()), paired_delta=float(delta.mean()),
                    paired_bootstrap_5_95=np.percentile(means, [5, 95]).tolist(),
                    improved_games=int((delta > 0).sum()), regressed_games=int((delta < 0).sum()),
                    equal_games=int((delta == 0).sum()))
        summaries[split] = summary
    if scored_cells != 95:
        raise ValueError("Expected all 95 paired game/view cells")
    receipt = dict(study=study.STUDY, passed=True, game_view_cells=scored_cells,
        official_scorecards_recomputed=2*scored_cells, result_sha256=study.digest(study.RESULT),
        declaration_sha256=study.digest(study.DECLARATION), audit_plan_sha256=study.digest(PLAN),
        paired_summaries=summaries, reused_test=True, valid_separate=True,
        independent_gold_confirmation=False, organizer_reference_access=False, public_scoring=False)
    study.write_once(RESULT, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify"))
    args = parser.parse_args()
    print((freeze if args.action == "freeze" else verify)())
