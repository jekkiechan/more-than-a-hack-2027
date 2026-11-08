from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import Future
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import v2_spell_balanced_eval as r1
from experiments import v2_spell_balanced_round2 as r2


class BalancedRound2Tests(unittest.TestCase):
    def test_mean_seed_phase_pooling(self):
        phases = {
            2026: [np.array([[.1, .9, 0.]]), np.array([[.7, .3, 0.]])],
            2027: [np.array([[.5, .5, 0.]]), np.array([[.9, .1, 0.]])],
        }
        pooled, per_seed = r2._mean_seed_phase(phases)
        np.testing.assert_allclose(pooled, [[.55, .45, 0.]])
        np.testing.assert_allclose(per_seed[2026], [[.4, .6, 0.]])
        np.testing.assert_allclose(per_seed[2027], [[.7, .3, 0.]])

    def test_phase_prediction_drops_labels_before_model_inference(self):
        calls = []
        frozen = r1.runtime.frozen
        record = {"labels": np.array(["H"]), "raw": np.array([1.])}
        teacher = {"teacher": True}
        with ExitStack() as stack:
            stack.enter_context(patch.object(r2, "_stage_model", lambda provider, seed: {"model": True}))
            stack.enter_context(patch.object(r2.joblib, "load", lambda path: teacher))
            stack.enter_context(patch.object(frozen, "_teacher_path", lambda provider, seed: Path("teacher")))
            stack.enter_context(patch.object(frozen.old, "inference_record", lambda row:
                {"names": ["raw"], "raw": row["raw"]}))
            def predict(infer_record, model, paired_teacher):
                self.assertNotIn("labels", infer_record)
                calls.append((model, paired_teacher))
                return np.array([[.2, .8, 0.]])
            stack.enter_context(patch.object(frozen.sn, "predict", predict))
            stack.enter_context(patch.object(frozen.old, "upsample", lambda sampled, infer, inputs: sampled))
            stack.enter_context(patch.object(frozen.tc, "inputs_only", lambda view: {}))
            predictions = r2._predict_phases({"view": {}, "phases": [record, record]}, "Sportec", 2026)
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(pair[0]["model"] and pair[1]["teacher"] for pair in calls))

    def test_shortlist_is_exact_r1_tuple_and_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path, decl_path = root / "round1.json", root / "round1_decl.json"
            anchor_id = r1.recipe_key(r1.ANCHOR)
            row = {**r1.ANCHOR, "adopted_weight": .5, "recipe_id": anchor_id,
                   "training_stage": 512}
            result = {"declaration_sha256": "bound", "selected_for_round2": [row],
                      "eligibility": {anchor_id: {"eligible": False}},
                      "aggregate": {anchor_id: {"feed_like_v2": {"spell_f1": {"equal_provider": .5},
                                                                         "macro_f1": {"equal_provider": .5}}}}}
            result_path.write_text(json.dumps(result))
            decl_path.write_text("frozen")
            with ExitStack() as stack:
                stack.enter_context(patch.object(r1, "RESULT", result_path))
                stack.enter_context(patch.object(r1, "DECLARATION", decl_path))
                stack.enter_context(patch.object(r2, "sha", lambda _: "bound"))
                stack.enter_context(patch.object(r1, "sha", lambda _: "bound"))
                self.assertEqual([row["recipe_id"] for row in r2._shortlist()], [anchor_id])
                result["selected_for_round2"][0]["ha_cost"] = 1.5
                result_path.write_text(json.dumps(result))
                with self.assertRaisesRegex(RuntimeError, "tuple"):
                    r2._shortlist()

    def test_mocked_35_cell_inference_and_result_assembly(self):
        cells = [r1._cell_key(*cell) for cell in r1._expected_cells()]
        self.assertEqual(len(cells), 35)
        with tempfile.TemporaryDirectory(dir=r2.ROOT) as directory:
            root = Path(directory)
            recipe = {**r1.ANCHOR, "adopted_weight": .5,
                      "recipe_id": r1.recipe_key(r1.ANCHOR), "training_stage": 1024}
            recipe_ids = [recipe["recipe_id"]]
            metric_row = {name: .5 for name in r1.METRICS}
            metric_row["reference_spell_count"] = 2
            baseline_rows = {key: metric_row.copy() for key in cells}
            baseline_aggregate = r1.aggregate_cell_metrics(baseline_rows)
            baseline_counts = {key: 2 for key in cells}
            (root / "replay.json").write_text(json.dumps({"adopted": baseline_aggregate,
                "reference_spell_counts": baseline_counts}))
            declaration = {"recipe_ids": recipe_ids, "recipes": [recipe],
                "training": {"fits": {}}, "controls": {key: {"identity": {}, "path": "unused",
                    "prepared_path": "unused"} for key in cells}}
            with ExitStack() as stack:
                stack.enter_context(patch.object(r2, "require_declaration", lambda: declaration))
                stack.enter_context(patch.object(r2, "DECLARATION", root / "decl.json"))
                r2.DECLARATION.write_text("{}")
                stack.enter_context(patch.object(r2, "RESULT", root / "result.json"))
                stack.enter_context(patch.object(r2, "OUT", root / "artifacts"))
                stack.enter_context(patch.object(r2, "sha", lambda _: "frozen"))
                stack.enter_context(patch.object(r2, "_expected_cells", lambda: set(cells)))
                stack.enter_context(patch.object(r1, "CONTROL_REPLAY", root / "replay.json"))
                # The real _cell_job path sees deterministic, mocked inference and scoring.
                n = 4
                adopted = np.tile([[.45, .45, .10]], (n, 1))
                control = {"adopted": adopted, "calibrated": adopted,
                    "labels": np.array(["H"] * n), "period": np.zeros(n, dtype=int),
                    "timestamps": np.arange(n), "model_period": np.zeros(n, dtype=int), "fps": 1.0}
                for seed in r2.SEEDS:
                    control[f"cand_seed{seed}"] = adopted
                    control[f"adop_seed{seed}"] = adopted
                stack.enter_context(patch.object(r2, "_load_control", lambda key, fact: control.copy()))
                stack.enter_context(patch.object(r2, "_load_prepared", lambda fact: {
                    "view": {}, "phases": [object()], "identity": {"cell": cells[0]}}))
                calls = []
                stack.enter_context(patch.object(r2, "_predict_phases", lambda prepared, provider, seed:
                    calls.append((provider, seed)) or [adopted.copy()]))
                stack.enter_context(patch.object(r2.r1.runtime.frozen.audit, "decode",
                    lambda *args: np.array(["H"] * n)))
                stack.enter_context(patch.object(r1, "decode_fixed_d",
                    lambda probability, decoded, *args: decoded.copy()))
                stack.enter_context(patch.object(r1.controls.audit, "score",
                    lambda cell, pred: metric_row.copy()))

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
                result = r2.run_round2()
            self.assertEqual(len(calls), 70)
            self.assertEqual(len(result["cell_metrics"]), 35)
            self.assertEqual(len(result["seed_cells"]), 35)
            self.assertEqual(set(result["aggregate"]), set(recipe_ids))
            self.assertEqual(set(result["eligibility"]), set(recipe_ids))
            self.assertEqual(len(result["prediction_cache"]), 35)
            self.assertEqual(result["cohort"]["cell_count"], 35)
            for seed_values in result["seed_cells"].values():
                self.assertEqual(set(seed_values), {f"{recipe_ids[0]}|{seed}" for seed in r2.SEEDS})


if __name__ == "__main__":
    unittest.main()
