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
from experiments import v2_spell_frontier_eval as r1
from experiments import v2_spell_frontier_round2 as r2


class FrontierRound2Tests(unittest.TestCase):
    def test_pair_pooling_supports_two_and_five_phases(self):
        for count in (2, 5):
            values = {s: [np.full((3, 3), (s - 2025) / 10) for _ in range(count)] for s in r2.SEEDS}
            pooled, seeds = r2._pool(values)
            np.testing.assert_allclose(pooled, np.full((3, 3), .15))
            self.assertEqual(set(seeds), set(r2.SEEDS))

    def test_pipeline_ids_wrap_immutable_recipe_and_stage(self):
        recipe = r1.RECIPES[0]
        for arm in r2.ARMS:
            pipeline = r2._pipeline(recipe, arm)
            self.assertEqual(pipeline["pipeline_id"], f"{arm}__{recipe['recipe_id']}")
            self.assertEqual(pipeline["recipe"]["recipe_id"], recipe["recipe_id"])
            self.assertEqual(pipeline["training_stage"], 1536)
            self.assertEqual(pipeline["training_arm"], arm)

    def test_freeze_require_roundtrip_rederives_nested_inventory_and_ranked_shortlist(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            declaration_path, result_path = root / "r1-decl.json", root / "r1-result.json"
            replay_path = root / "r1-replay.json"
            declaration_path.write_text("frozen")
            replay_path.write_text("passed")
            recipes = r1.RECIPES
            # Ranking intentionally differs from grid order; the anchor is appended after the top two.
            expected_shortlist = [recipes[4]["recipe_id"], recipes[1]["recipe_id"], r1.LEADER_IDS[0]]
            fake_r1 = {"declaration_sha256": "fixed", "eligible_ranking": expected_shortlist[:2],
                "seed_shortlist": expected_shortlist, "recipes": recipes}
            result_path.write_text(json.dumps(fake_r1))
            controls = {key: {"path": f"{key}.npz"} for key in r1.expected_cells()}
            fit_facts = {f"{arm}/fold/seed": {"sha256": arm} for arm in r2.ARMS}
            with ExitStack() as stack:
                stack.enter_context(patch.object(r1, "DECLARATION", declaration_path))
                stack.enter_context(patch.object(r1, "RESULT", result_path))
                stack.enter_context(patch.object(r1, "REPLAY", replay_path))
                stack.enter_context(patch.object(r2, "DECLARATION", root / "r2-decl.json"))
                stack.enter_context(patch.object(r2, "RESULT", root / "r2-result.json"))
                stack.enter_context(patch.object(r2, "sha", lambda _: "fixed"))
                stack.enter_context(patch.object(r2, "_fit_facts", lambda: fit_facts))
                stack.enter_context(patch.object(r2, "_controls", lambda: controls))
                stack.enter_context(patch.object(r2, "_source_hashes", lambda: {"test": "src"}))
                frozen = r2.freeze()
                checked = r2.require_declaration()
            self.assertEqual(frozen, checked)
            self.assertEqual([r["recipe_id"] for r in frozen["recipes"]], expected_shortlist)
            self.assertEqual(frozen["pipeline_ids"], [f"{arm}__{rid}" for rid in expected_shortlist for arm in r2.ARMS])
            self.assertTrue(all(p["training_stage"] == 1536 and set(p["recipe"]) >= {
                "recipe_id", "ce_weight", "calibrated_weight", "adopted_weight", "pooling", "ha_cost"}
                for p in frozen["pipelines"]))

    def test_cell_job_pools_seed_phases_and_scores_nested_pipelines(self):
        recipe = r1.RECIPES[0]
        pipelines = [_ for _ in (r2._pipeline(recipe, arm) for arm in r2.ARMS)]
        n = 3
        control = {"adopted": np.tile([[.45, .45, .1]], (n, 1)),
            "calibrated": np.tile([[.4, .5, .1]], (n, 1)),
            **{f"cand_seed{s}": np.tile([[.45, .45, .1]], (n, 1)) for s in r2.SEEDS},
            **{f"adop_seed{s}": np.tile([[.4, .5, .1]], (n, 1)) for s in r2.SEEDS},
            "labels": np.array(["H"]*n), "period": np.ones(n), "timestamps": np.arange(n),
            "model_period": np.ones(n), "fps": 1.}
        fake_predict = lambda prepared, provider, arm, seed, facts: [
            np.tile([[.5, .4, .1]], (n, 1)), np.tile([[.6, .3, .1]], (n, 1))]
        with ExitStack() as stack:
            stack.enter_context(patch.object(r2, "_cell_inputs", lambda key, fact:
                ({"phases": [object(), object()], "view": {}}, control)))
            stack.enter_context(patch.object(r2, "_predict_seed", fake_predict))
            stack.enter_context(patch.object(r1.balanced.runtime.frozen.audit, "decode",
                lambda *args: np.array(["H"]*n)))
            stack.enter_context(patch.object(r1.balanced, "decode_fixed_d", lambda p, d, *args: d.copy()))
            metrics = {name: .5 for name in balanced.METRICS}; metrics["reference_spell_count"] = 2
            stack.enter_context(patch.object(r1.balanced.controls.audit, "score", lambda *args: metrics.copy()))
            output = r2._cell_job(("Metrica|metrica_3|clean|0", {}, [recipe], pipelines, {}))
        key, rows, seed_rows, probs, preds, seed_preds = output
        self.assertEqual(set(rows), {p["pipeline_id"] for p in pipelines})
        self.assertEqual(len(seed_rows), 4)
        self.assertEqual(len([k for k in probs if k.startswith("phase__")]), 8)
        self.assertEqual(len(preds), 2)
        self.assertEqual(len(seed_preds), 4)

    def test_prediction_inference_receives_no_reference_labels(self):
        record = {"labels": np.array(["H"]), "features": np.array([1.])}
        frozen = r1.balanced.runtime.frozen
        loaded = iter(({"model": 1}, {"teacher": 1}))
        with ExitStack() as stack:
            stack.enter_context(patch.object(r2.joblib, "load", lambda path: next(loaded)))
            stack.enter_context(patch.object(frozen, "_teacher_path", lambda provider, seed: Path("teacher")))
            stack.enter_context(patch.object(frozen.old, "inference_record", lambda row:
                {"names": ["features"], "features": row["features"]}))
            def predict(infer, model, teacher):
                self.assertNotIn("labels", infer)
                return np.array([[.5, .4, .1]])
            stack.enter_context(patch.object(frozen.sn, "predict", predict))
            stack.enter_context(patch.object(frozen.old, "upsample", lambda sampled, infer, view: sampled))
            stack.enter_context(patch.object(frozen.tc, "inputs_only", lambda view: {}))
            prepared = {"phases": [record, record], "view": {}}
            outputs = r2._predict_seed(prepared, "Metrica", "repeat", 2026,
                                       {"repeat/leave_out_metrica/2026": {"path": "model"}})
        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].shape, (1, 3))

    def test_mocked_35_cell_pipeline_assembly(self):
        cells = sorted(r1.expected_cells())
        recipe = r1.RECIPES[0]
        pipelines = [r2._pipeline(recipe, arm) for arm in r2.ARMS]
        ids = [p["pipeline_id"] for p in pipelines]
        row = {name: .5 for name in balanced.METRICS}
        row["reference_spell_count"] = 2
        baseline_rows = {key: row.copy() for key in cells}
        baseline = balanced.aggregate_cell_metrics(baseline_rows)
        refs = {key: 2 for key in cells}
        with tempfile.TemporaryDirectory(dir=r2.ROOT) as temp:
            root = Path(temp)
            replay = root / "replay.json"
            replay.write_text(json.dumps({"adopted": baseline, "reference_spell_counts": refs}))
            declaration = {"recipes": [recipe], "pipelines": pipelines, "pipeline_ids": ids,
                           "training": {"fits": {}}, "controls": {key: {} for key in cells}}
            with ExitStack() as stack:
                stack.enter_context(patch.object(r2, "require_declaration", lambda: declaration))
                stack.enter_context(patch.object(r2, "DECLARATION", root / "decl.json"))
                r2.DECLARATION.write_text("{}")
                stack.enter_context(patch.object(r2, "RESULT", root / "result.json"))
                stack.enter_context(patch.object(r2, "OUT", root / "artifacts"))
                stack.enter_context(patch.object(r2, "sha", lambda _: "fixed"))
                stack.enter_context(patch.object(r2, "_worker_init", lambda: None))
                stack.enter_context(patch.object(r2, "_cell_job", lambda job:
                    (job[0], {pid: row.copy() for pid in ids},
                     {f"{pid}|{seed}": row.copy() for pid in ids for seed in r2.SEEDS},
                     {"pooled__x": np.zeros((2, 3))},
                     {f"pred__{pid}": np.zeros(2, dtype=np.int8) for pid in ids},
                     {f"seedpred__{pid}|{seed}": np.zeros(2, dtype=np.int8) for pid in ids for seed in r2.SEEDS})))
                class InlinePool:
                    def __init__(self, *args, **kwargs): pass
                    def __enter__(self): return self
                    def __exit__(self, *args): return False
                    def submit(self, fn, job):
                        future = Future()
                        try: future.set_result(fn(job))
                        except Exception as exc: future.set_exception(exc)
                        return future
                stack.enter_context(patch.object(r2, "ProcessPoolExecutor", InlinePool))
                stack.enter_context(patch.object(r1, "CONTROL_REPLAY", replay))
                result = r2.run_round2()
            self.assertEqual(len(result["cell_metrics"]), 35)
            self.assertEqual(set(result["aggregate"]), set(ids))
            self.assertEqual(set(result["eligibility"]), set(ids))
            self.assertEqual(len(result["prediction_cache"]), 35)
            self.assertTrue(all("recipe" in p and p["training_stage"] == 1536 for p in result["pipelines"]))


if __name__ == "__main__":
    unittest.main()
