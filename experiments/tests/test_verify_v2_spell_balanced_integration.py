"""Small independent-audit invariants that do not open organizer labels."""
from contextlib import ExitStack
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import v2_spell_balanced_eval as r1
from experiments import v2_spell_balanced_round2 as r2
from scripts import verify_v2_spell_balanced as verifier


class VerifierNestedSchemaTests(unittest.TestCase):
    def test_nested_string_and_boolean_provenance_are_compared_exactly(self):
        expected = {"gate": {"comparison": "strictly_greater", "passed": True}}
        verifier._close(expected, {"gate": {"comparison": "strictly_greater", "passed": True}},
                        "fixture")
        with self.assertRaises(AssertionError):
            verifier._close(expected, {"gate": {"comparison": "greater_or_equal", "passed": True}},
                            "fixture")
        with self.assertRaises(AssertionError):
            verifier._close(expected, {"gate": {"comparison": "strictly_greater", "passed": False}},
                            "fixture")

    def test_phase_count_uses_native_5hz_sampling_rate(self):
        self.assertEqual(verifier._phase_count(10.0), 2)
        self.assertEqual(verifier._phase_count(25.0), 5)
        with self.assertRaises(AssertionError):
            verifier._phase_count(24.0)

    def test_both_entrypoints_replay_a_full_synthetic_35_cell_study(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cells = {r1._cell_key(*cell) for cell in r1._expected_cells()}
            self.assertEqual(len(cells), 35)
            recipes = list(r1.RECIPES)
            recipe_ids = [row["recipe_id"] for row in recipes]
            selected = [r1.RECIPES[0], r1.RECIPES[1], next(
                row for row in recipes if row["recipe_id"] == r1.recipe_key(r1.ANCHOR))]
            shortlist = [row["recipe_id"] for row in selected]
            labels = np.zeros(8, dtype=np.int64)
            period = np.ones(8, dtype=np.int64)
            timestamps = np.arange(8, dtype=np.float64) / 5.0
            expert = np.tile(np.array([[0.6, 0.4, 0.0]], dtype=np.float64), (8, 1))
            score_row = {
                "spell_f1": .6, "macro_f1": .7, "d_f1": .8,
                "d_precision": .8, "d_recall": .8,
                "predicted_spell_count": 2.0, "reference_spell_count": 2.0,
                "short_spell_recall_iou_0_5_lt3s": .4,
                "short_matched_spell_count": 1, "short_reference_spell_count": 2,
                "graded_frames": 8, "weighted_70_30": .64,
                "spell_by_iou": {"0.3": .6, "0.4": .6, "0.5": .6,
                                 "0.6": .6, "0.7": .6},
            }

            def components_for(cell):
                provider = cell.split("|", 1)[0]
                fps = 10.0 if provider == "SkillCorner" else 25.0
                seed_components = {seed: {"ce": expert.copy(), "adopted": expert.copy(),
                                          "calibrated": expert.copy()}
                                   for seed in r1.SEEDS}
                return {"labels": labels.copy(), "period": period.copy(),
                    "timestamps": timestamps.copy(), "model_period": period.copy(),
                    "fps": fps, "adopted": expert.copy(), "calibrated": expert.copy(),
                    "ce": expert.copy(), "seed": seed_components}

            components = {cell: components_for(cell) for cell in cells}
            baseline = {view: {metric: {"equal_provider":
                (score_row[metric] if metric.startswith("d_") else 0.0),
                "providers": {provider: score_row[metric] if metric.startswith("d_")
                              else 0.0
                              for provider in ("Metrica", "SkillCorner", "Sportec")},
                "cells": sum(1 for cell in cells if cell.split("|")[2] == view)}
                for metric in r1.METRICS} for view, _ in r1.VIEWS}
            references = {cell: 2 for cell in cells}

            # Frozen-source manifests and result paths are temporary, while every
            # saved prediction cache uses the same on-disk NPZ contract as production.
            r1_decl_path = root / "r1_decl.json"
            r1_replay_path = root / "control_replay.json"
            r1_result_path = root / "r1_result.json"
            r2_decl_path = root / "r2_decl.json"
            r2_result_path = root / "r2_result.json"
            r2_amendment_path = root / "r2_amendment.json"
            r2_execution_path = root / "r2_execution.json"
            r1_decl = {"source_sha256": {"fixture": "stable"}, "upstream": {
                "adopted_calibrated_emissions": {cell: {"fixture": cell} for cell in cells}}}
            r1_decl_path.write_text(json.dumps(r1_decl))
            replay = {"passed": True, "candidate_scoring": False,
                      "reference_spell_counts": references, "adopted": baseline}
            r1_replay_path.write_text(json.dumps(replay))
            r1_decl_sha = verifier.sha(r1_decl_path)
            replay_sha = verifier.sha(r1_replay_path)

            candidate_cache, seed_cache = {}, {}
            cell_metrics, seed_cells = {}, {}
            for cell in sorted(cells):
                prediction_path = root / "r1_predictions" / f"{len(candidate_cache)}.npz"
                identity = {"study_id": r1.STUDY, "declaration_sha256": r1_decl_sha,
                            "cell": cell, "control_sha256": "control-fixture",
                            "ce_sha256": "ce-fixture"}
                values = {f"pred__{rid}": np.zeros(8, dtype=np.uint8) for rid in recipe_ids}
                values.update({f"identity__{name}": np.asarray(value)
                               for name, value in identity.items()})
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(prediction_path, **values)
                candidate_cache[cell] = {"path": str(prediction_path.relative_to(root)),
                    "sha256": verifier.sha(prediction_path), "identity": identity}

                seed_path = root / "r1_seed_predictions" / f"{len(seed_cache)}.npz"
                seed_identity = {"study_id": r1.STUDY, "declaration_sha256": r1_decl_sha,
                                 "cell": cell, "recipes": shortlist}
                values = {f"pred__{rid}|{seed}": np.zeros(8, dtype=np.uint8)
                          for rid in shortlist for seed in r1.SEEDS}
                values.update({f"identity__{name}": np.asarray(repr(value) if name == "recipes" else value)
                               for name, value in seed_identity.items()})
                seed_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(seed_path, **values)
                seed_cache[cell] = {"path": str(seed_path.relative_to(root)),
                    "sha256": verifier.sha(seed_path), "identity": seed_identity}
                cell_metrics[cell] = {rid: dict(score_row) for rid in recipe_ids}
                seed_cells[cell] = {f"{rid}|{seed}": dict(score_row)
                                    for rid in shortlist for seed in r1.SEEDS}

            aggregate = verifier.aggregate_cell_metrics(cell_metrics, recipe_ids)
            for view, _ in r1.VIEWS:
                for metric in ("d_f1", "d_precision", "d_recall"):
                    baseline[view][metric]["equal_provider"] = aggregate[recipe_ids[0]][view][metric]["equal_provider"]
            replay["adopted"] = baseline
            r1_replay_path.write_text(json.dumps(replay))
            replay_sha = verifier.sha(r1_replay_path)
            gates = {}
            for rid in recipe_ids:
                candidate = {view: {m: aggregate[rid][view][m]["equal_provider"]
                                    for m in r1.METRICS} for view, _ in r1.VIEWS}
                base = {view: {m: baseline[view][m]["equal_provider"]
                               for m in r1.METRICS} for view, _ in r1.VIEWS}
                gate = r1.eligibility(candidate, base)
                gate["d_metrics_unchanged"] = True
                gate["reference_spell_counts_invariant"] = True
                gate["eligible"] = gate["eligible"] and True
                gates[rid] = gate
            ranking = sorted(recipe_ids, key=lambda rid: (
                -aggregate[rid]["feed_like_v2"]["spell_f1"]["equal_provider"], rid))
            selected_shortlist = r1.select_seed_shortlist(gates, aggregate)
            # Use selection derived from the independent reducer (anchor is retained).
            shortlist = selected_shortlist
            # Rewrite seed files with the actual selected shortlist.
            seed_cells = {cell: {f"{rid}|{seed}": dict(score_row)
                                 for rid in shortlist for seed in r1.SEEDS}
                          for cell in cells}
            for cell in sorted(cells):
                fact = seed_cache[cell]
                path = root / fact["path"]
                seed_identity = {"study_id": r1.STUDY, "declaration_sha256": r1_decl_sha,
                                 "cell": cell, "recipes": shortlist}
                values = {f"pred__{rid}|{seed}": np.zeros(8, dtype=np.uint8)
                          for rid in shortlist for seed in r1.SEEDS}
                values.update({f"identity__{name}": np.asarray(repr(value) if name == "recipes" else value)
                               for name, value in seed_identity.items()})
                np.savez_compressed(path, **values)
                fact["sha256"] = verifier.sha(path)
                fact["identity"] = seed_identity
            seed_metrics = {rid: {str(seed): verifier.aggregate_cell_metrics(
                {cell: {rid: score_row} for cell in cells}, [rid])[rid]
                for seed in r1.SEEDS} for rid in shortlist}
            r1_result = {"study_id": r1.STUDY, "round": 1,
                "declaration_sha256": r1_decl_sha,
                "control_replay_sha256": replay_sha,
                "recipe_ids": recipe_ids, "recipes": recipes,
                "selected_for_round2_seed_diagnostics": shortlist,
                "candidate_prediction_cache": candidate_cache,
                "seed_prediction_cache": seed_cache,
                "cell_metrics": cell_metrics, "seed_cells": seed_cells,
                "aggregate": aggregate, "eligibility": gates,
                "eligible_ranking": ranking, "seed_diagnostics": seed_metrics}
            r1_result_path.write_text(json.dumps(r1_result))

            # R2 fixture uses three selected recipes and per-cell phase arrays.
            r2_decl = {"source_sha256": {"fixture": "stable"},
                       "recipes": [{**next(row for row in recipes if row["recipe_id"] == rid),
                                    "training_stage": 1024} for rid in shortlist]}
            r2_decl_path.write_text(json.dumps(r2_decl))
            r2_decl_sha = verifier.sha(r2_decl_path)
            r2_cache, r2_cell_rows, r2_seed_rows = {}, {}, {}
            for cell in sorted(cells):
                comp = components[cell]
                phase_count = verifier._phase_count(comp["fps"])
                identity = {"study_id": r1.STUDY, "declaration_sha256": r2_decl_sha,
                            "cell": cell, "phase_count": phase_count}
                path = root / "r2_predictions" / f"{len(r2_cache)}.npz"
                values = {f"phase_ce_seed{seed}_{phase}": expert.copy()
                          for seed in r1.SEEDS for phase in range(phase_count)}
                values.update({"pooled_ce": expert.copy(),
                               **{f"seed_ce_{seed}": expert.copy() for seed in r1.SEEDS}})
                values.update({f"pred__{rid}": np.zeros(8, dtype=np.uint8) for rid in shortlist})
                values.update({f"seedpred__{rid}|{seed}": np.zeros(8, dtype=np.uint8)
                               for rid in shortlist for seed in r1.SEEDS})
                values.update({f"identity__{name}": np.asarray(value)
                               for name, value in identity.items()})
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **values)
                r2_cache[cell] = {"path": str(path.relative_to(root)),
                                  "sha256": verifier.sha(path), "identity": identity}
                r2_cell_rows[cell] = {rid: dict(score_row) for rid in shortlist}
                r2_seed_rows[cell] = {f"{rid}|{seed}": dict(score_row)
                                      for rid in shortlist for seed in r1.SEEDS}
            r2_aggregate = verifier._score_rows(r2_cell_rows, shortlist)
            r2_seed_aggregate = verifier._seed_aggregate(r2_seed_rows, shortlist)
            r2_gates = {}
            for rid in shortlist:
                candidate = {view: {metric: r2_aggregate[rid][view][metric]["equal_provider"]
                                    for metric in r1.METRICS} for view, _ in r1.VIEWS}
                adopted_baseline = {view: {metric: baseline[view][metric]["equal_provider"]
                                           for metric in r1.METRICS} for view, _ in r1.VIEWS}
                gate = r1.eligibility(candidate, adopted_baseline)
                gate["d_metrics_unchanged"] = all(
                    candidate[view][metric] == adopted_baseline[view][metric]
                    for view, _ in r1.VIEWS for metric in ("d_f1", "d_precision", "d_recall"))
                gate["reference_spell_counts_invariant"] = True
                gate["eligible"] = gate["eligible"] and gate["d_metrics_unchanged"]
                r2_gates[rid] = gate
            r2_result = {"study_id": r1.STUDY, "round": 2,
                "declaration_sha256": r2_decl_sha,
                "round1_result_sha256": verifier.sha(r1_result_path),
                "recipes": r2_decl["recipes"], "recipe_ids": shortlist,
                "cohort": {"reference_spell_counts": references},
                "cell_metrics": r2_cell_rows, "seed_cells": r2_seed_rows,
                "prediction_cache": r2_cache, "aggregate": r2_aggregate,
                "seed_diagnostics": r2_seed_aggregate,
                "eligibility": r2_gates}
            r2_result_path.write_text(json.dumps(r2_result))
            r2_amendment_path.write_text("{}\n")
            r2_execution_path.write_text(json.dumps({
                "completed": True, "amendment_sha256": verifier.sha(r2_amendment_path),
                "result_sha256": verifier.sha(r2_result_path),
                "source_and_input_sha256": {"fixture": "stable"},
                "organizer_reference_access": False}))

            execution = __import__("experiments.v2_spell_balanced_execution", fromlist=["check"])
            with ExitStack() as stack:
                for target, value, new in (
                    (verifier, "ROOT", root), (verifier, "R1_RESULT", r1_result_path),
                    (verifier, "R1_DECL", r1_decl_path), (verifier, "CONTROL_REPLAY", r1_replay_path),
                    (verifier, "R2_RESULT", r2_result_path), (verifier, "R2_DECL", r2_decl_path),
                    (verifier, "R2_EXECUTION_AMENDMENT", r2_amendment_path),
                    (verifier, "R2_EXECUTION_RECEIPT", r2_execution_path),
                    (r1, "_source_hashes", lambda: {"fixture": "stable"}),
                    (r2, "_source_hashes", lambda: {"fixture": "stable"}),
                    (r1, "load_frozen_emissions", lambda cell: components[cell]),
                    (r1.controls.audit, "score", lambda *_args: dict(score_row)),
                    (r1.controls.audit, "decode", lambda *_args: np.full(8, "H", dtype="U1")),
                    (r1, "decode_fixed_d", lambda *_args: np.full(8, "H", dtype="U1")),
                    (execution, "check", lambda: {}),
                    (execution, "_facts", lambda: {"fixture": "stable"}),
                ):
                    stack.enter_context(patch.object(target, value, new))
                first = verifier.verify_round1()
                second = verifier.verify_round2()
            self.assertTrue(first["passed"])
            self.assertEqual((first["cell_count"], first["recipe_count"]), (35, 36))
            self.assertTrue(second["passed"])
            self.assertEqual((second["cell_count"], second["recipe_count"]), (35, 3))
            self.assertTrue(second["phase_pooling_recomputed"])


if __name__ == "__main__":
    unittest.main()
