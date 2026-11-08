from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import v2_spell_balanced_eval as balanced
from experiments import v2_spell_frontier_eval as frontier


class SpellFrontierEvalTests(unittest.TestCase):
    def test_grid_has_30_unique_valid_recipes_and_fixed_ha3(self):
        self.assertEqual(len(frontier.RECIPES), 30)
        self.assertEqual(len(set(frontier.ARM_IDS)), 30)
        for row in frontier.RECIPES:
            self.assertGreaterEqual(row["adopted_weight"], 0)
            self.assertAlmostEqual(row["ce_weight"] + row["calibrated_weight"] + row["adopted_weight"], 1.)
            self.assertEqual(row["ha_cost"], 3.)
        self.assertEqual(frontier.LEADER_IDS, (
            "ce0.5_cal0.25_ad0.25_arithmetic_ha3", "ce0.5_cal0.25_ad0.25_log_ha3"))

    def test_cache_identity_is_checked_and_stage_metadata_is_not_mutated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            control_path, pred_path = root / "control.npz", root / "pred.npz"
            ident = {"cell": "Metrica|metrica_3|clean|0", "provider": "Metrica"}
            control = np.tile([[.45, .45, .1]], (3, 1))
            np.savez(control_path, **{f"prob__{name}": control for name in
                ("adopted", "calibrated", "cand_seed2026", "cand_seed2027",
                 "adop_seed2026", "adop_seed2027")},
                labels=np.array(["H"]*3), period=np.ones(3), timestamps=np.arange(3),
                model_period=np.ones(3), fps=np.array(1.),
                **{f"identity__{k}": np.asarray(v) for k, v in ident.items()})
            pred_identity = {"cell": ident["cell"], "study_id": "v2_spell_balanced_v1",
                "declaration_sha256": "r2decl", "phase_count": "5"}
            np.savez(pred_path, pooled_ce=control, seed_ce_2026=control, seed_ce_2027=control,
                **{f"identity__{k}": np.asarray(v) for k, v in pred_identity.items()})
            c_fact = {"path": "control.npz", "identity": ident}
            p_fact = {"path": "pred.npz"}
            with ExitStack() as stack:
                stack.enter_context(patch.object(frontier, "ROOT", root))
                stack.enter_context(patch.object(frontier, "sha", lambda _: "r2decl"))
                stack.enter_context(patch.object(frontier.balanced_r2, "DECLARATION", root / "r2decl.json"))
                result = frontier._load_cell(ident["cell"], c_fact, p_fact)
            self.assertEqual(result["ce"].shape, (3, 3))
            self.assertEqual(set(result["seed_ce"]), set(frontier.SEEDS))
            self.assertEqual(pred_identity["phase_count"], "5")

    def test_mocked_freeze_require_roundtrip_preserves_frontier_grid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = root / "plan.json"
            plan.write_text(json.dumps({"study_id": frontier.STUDY}))
            cells = sorted(frontier.expected_cells())
            control = {key: {"sha256": f"c-{i}"} for i, key in enumerate(cells)}
            predictions = {key: {"sha256": f"p-{i}"} for i, key in enumerate(cells)}
            inputs = (control, predictions, {"replay": {}, "round2": {}, "declaration": {}})
            selection = {"study_id": "v2_spell_balanced_v1", "recipe": {"recipe_id": "current"},
                "training_stage": 1024,
                "aggregate": {"feed_like_v2": {"spell_f1": {"equal_provider": frontier.CURRENT_BEST_V2}}}}
            with ExitStack() as stack:
                stack.enter_context(patch.object(frontier, "PLAN", plan))
                stack.enter_context(patch.object(frontier, "DECLARATION", root / "decl.json"))
                stack.enter_context(patch.object(frontier, "REPLAY", root / "replay.json"))
                stack.enter_context(patch.object(frontier, "RESULT", root / "result.json"))
                stack.enter_context(patch.object(frontier, "SELECTION", root / "selection.json"))
                frontier.SELECTION.write_text(json.dumps(selection))
                stack.enter_context(patch.object(frontier, "sha", lambda _: "stable"))
                stack.enter_context(patch.object(frontier, "_inputs", lambda: inputs))
                stack.enter_context(patch.object(frontier, "_source_hashes", lambda: {"test": "source"}))
                declaration = frontier.freeze()
                checked = frontier.require_declaration()
            self.assertEqual(declaration, checked)
            self.assertEqual(checked["recipe_grid"], frontier.RECIPES)
            self.assertEqual(checked["recipe_count"], 30)
            self.assertEqual(checked["fallback"]["training_stage"], 1024)

    def test_mocked_35_cell_run_aggregates_and_ranks_with_protected_gates(self):
        cells = sorted(frontier.expected_cells())
        self.assertEqual(len(cells), 35)
        with tempfile.TemporaryDirectory(dir=frontier.ROOT) as temp:
            root = Path(temp)
            baseline_rows = {key: {name: .5 for name in balanced.METRICS} for key in cells}
            for row in baseline_rows.values():
                row["reference_spell_count"] = 2
            adopted = balanced.aggregate_cell_metrics(baseline_rows)
            refs = {key: 2 for key in cells}
            replay_path = root / "replay.json"
            replay_path.write_text(json.dumps({"adopted": adopted, "reference_spell_counts": refs,
                "cell_counts": balanced._cell_counts(baseline_rows)}))
            decl = {"inputs": {"control": {k: {} for k in cells},
                               "stage1024_probability_cache": {k: {} for k in cells}},
                    "recipe_grid": frontier.RECIPES}
            result_path = root / "result.json"
            result_path.unlink(missing_ok=True)
            with ExitStack() as stack:
                stack.enter_context(patch.object(frontier, "require_declaration", lambda: decl))
                stack.enter_context(patch.object(frontier, "DECLARATION", root / "decl.json"))
                frontier.DECLARATION.write_text("{}")
                stack.enter_context(patch.object(frontier, "RESULT", result_path))
                stack.enter_context(patch.object(frontier, "REPLAY", root / "leader_replay.json"))
                frontier.REPLAY.write_text(json.dumps({"passed": True}))
                stack.enter_context(patch.object(frontier, "OUT", root / "artifacts"))
                stack.enter_context(patch.object(frontier, "CONTROL_REPLAY", replay_path))
                stack.enter_context(patch.object(frontier, "SELECTION", root / "selection.json"))
                frontier.SELECTION.write_text(json.dumps({"recipe": {"recipe_id": "fallback"},
                                                          "training_stage": 1024}))
                stack.enter_context(patch.object(frontier, "sha", lambda _: "fixed"))
                stack.enter_context(patch.object(frontier, "expected_cells", lambda: set(cells)))
                # Make candidate spells beat the baseline but remain below the frozen .591174 floor.
                candidate = {name: .55 for name in balanced.METRICS}
                candidate["reference_spell_count"] = 2
                def worker(job):
                    key = job[0]
                    return key, {rid: candidate.copy() for rid in frontier.ARM_IDS}, {
                        rid: np.zeros(3, dtype=np.int8) for rid in frontier.ARM_IDS}
                stack.enter_context(patch.object(frontier, "_worker_job", worker))

                class InlinePool:
                    def __init__(self, *args, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                    def submit(self, fn, job):
                        fut = Future()
                        try: fut.set_result(fn(job))
                        except Exception as exc: fut.set_exception(exc)
                        return fut

                stack.enter_context(patch.object(frontier, "ProcessPoolExecutor", InlinePool))
                # No recipes qualify, so only the predeclared current leader is used for seed diagnostics.
                cell = {"ce": np.tile([[.5,.4,.1]], (3,1)),
                    "calibrated": np.tile([[.4,.5,.1]], (3,1)),
                    "adopted": np.tile([[.45,.45,.1]], (3,1)),
                    "seed_ce": {s: np.tile([[.5,.4,.1]], (3,1)) for s in frontier.SEEDS},
                    "seed_calibrated": {s: np.tile([[.4,.5,.1]], (3,1)) for s in frontier.SEEDS},
                    "seed_adopted": {s: np.tile([[.45,.45,.1]], (3,1)) for s in frontier.SEEDS},
                    "labels": np.array(["H"]*3), "period": np.ones(3),
                    "timestamps": np.arange(3), "model_period": np.ones(3), "fps": 1.}
                stack.enter_context(patch.object(frontier, "_load_cell", lambda *args: cell))
                stack.enter_context(patch.object(frontier.balanced.runtime.frozen.audit, "decode",
                                                 lambda *args: np.array(["H"]*3)))
                stack.enter_context(patch.object(frontier.balanced, "decode_fixed_d",
                                                 lambda p, d, *args: d.copy()))
                stack.enter_context(patch.object(frontier.balanced.controls.audit, "score",
                                                 lambda *args: candidate.copy()))
                result = frontier.run_round1()
            self.assertEqual(len(result["cell_metrics"]), 35)
            self.assertEqual(len(result["aggregate"]), 30)
            self.assertEqual(len(result["prediction_cache"]), 35)
            self.assertEqual(len(result["seed_prediction_cache"]), 35)
            self.assertEqual(result["recipe_ids"], list(frontier.ARM_IDS))
            pooled_fact = result["prediction_cache"][cells[0]]
            with np.load(frontier.ROOT / pooled_fact["path"], allow_pickle=False) as saved:
                self.assertEqual(json.loads(saved["identity__recipe_order"].item()), list(frontier.ARM_IDS))
            seed_fact = result["seed_prediction_cache"][cells[0]]
            with np.load(frontier.ROOT / seed_fact["path"], allow_pickle=False) as saved:
                self.assertEqual(json.loads(saved["identity__recipe_order"].item()), result["seed_shortlist"])
            self.assertEqual(result["seed_shortlist"], [frontier.LEADER_IDS[0]])
            self.assertTrue(all(not gate["eligible"] for gate in result["eligibility"].values()))
            self.assertEqual(result["fallback_comparison"]["v2_spell_f1"], frontier.CURRENT_BEST_V2)


if __name__ == "__main__":
    unittest.main()
