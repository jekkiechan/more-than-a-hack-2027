"""Synthetic reporting contracts for the SoccerNet execution follow-up."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments import soccernet_followup_evaluate as evaluation
from experiments import soccernet_followup_execute as executor


def score_row(source, *, spell, macro, d=0.7, precision=0.8, recall=0.6):
    return {
        "source": source, "arm": "reference", "view": "clean", "member": "ensemble",
        "spell_f1": spell, "macro_f1": macro,
        "weighted_70_30": 0.7 * spell + 0.3 * macro,
        "d_f1": d, "d_precision": precision, "d_recall": recall,
        "short_spell_recall": 0.2, "n_gt_spells": 10,
        "n_pred_spells": 11, "fragmented_reference_spells": 1,
        "graded_frames": 100,
        "spell_by_iou": {key: spell - i * 0.01
                         for i, key in enumerate(evaluation.IOU_KEYS)},
    }


class FollowupEvaluationTests(unittest.TestCase):
    def test_provider_aggregation_is_equal_weighted(self):
        rows = [score_row("sportec", spell=0.5, macro=0.6),
                score_row("metrica", spell=0.7, macro=0.8),
                score_row("skillcorner", spell=0.9, macro=1.0)]
        result = evaluation._aggregate(rows, "reference", "clean", "ensemble")
        mean = result["equal_provider_mean"]
        self.assertAlmostEqual(mean["spell_f1"], 0.7)
        self.assertAlmostEqual(mean["macro_f1"], 0.8)
        self.assertAlmostEqual(mean["weighted_70_30"], 0.73)
        self.assertAlmostEqual(mean["spell_by_iou"]["0.5"], 0.68)
        self.assertEqual(result["cell_count"], 3)
        historical_style = evaluation.cap._aggregate_provider(
            rows, "reference", "clean", "ensemble")
        self.assertEqual(result, historical_style)

    def test_published_comparison_reports_member_deltas_and_tolerance(self):
        providers = {}
        for provider in ("sportec", "metrica", "skillcorner"):
            providers[provider] = {
                "spell_f1": 0.5, "macro_f1": 0.6, "weighted_70_30": 0.53,
                "d_f1": 0.7, "d_precision": 0.8, "d_recall": 0.6,
                "short_spell_recall": 0.2, "n_gt_spells": 10,
                "n_pred_spells": 11, "fragmented_reference_spells": 1,
                "graded_frames": 100,
                "spell_by_iou": {key: 0.5 - i * 0.01
                                 for i, key in enumerate(evaluation.IOU_KEYS)},
            }
        summary = {"equal_provider_mean": dict(providers["sportec"]),
                   "per_provider": providers, "cell_count": 5}
        rebuilt = {member: {view: summary for view in ("clean", "feed_like")}
                   for member in ("2026", "2027", "ensemble")}
        published = {
            "seed_summaries": {"w96": {
                seed: {view: summary for view in ("clean", "feed_like")}
                for seed in ("2026", "2027")}},
            "summaries": {"w96": {view: summary for view in ("clean", "feed_like")}},
        }
        result = evaluation._published_comparison(rebuilt, published)
        self.assertTrue(result["all_within_tolerance"])
        self.assertEqual(result["members"]["ensemble"]["clean"][
            "deltas_rebuilt_minus_published"]["sportec_spell_f1"], 0.0)
        changed = dict(summary)
        changed["equal_provider_mean"] = dict(summary["equal_provider_mean"], spell_f1=0.51)
        rebuilt["2026"]["clean"] = changed
        failed = evaluation._published_comparison(rebuilt, published)
        self.assertFalse(failed["all_within_tolerance"])

    def test_aggregate_rejects_missing_provider(self):
        with self.assertRaises(evaluation.EvaluationError):
            evaluation._aggregate([score_row("sportec", spell=0.5, macro=0.6)],
                                  "reference", "clean", "ensemble")

    def test_guardrails_compare_all_continuations_to_reference(self):
        def summary(spell, macro, d):
            provider_values = {
                provider: {"macro_f1": macro, "d_f1": d}
                for provider in ("sportec", "metrica", "skillcorner")
            }
            mean = {"spell_f1": spell}
            return {"clean": {"equal_provider_mean": mean,
                              "per_provider": provider_values},
                    "feed_like": {"equal_provider_mean": mean,
                                  "per_provider": provider_values}}
        summaries = {"reference": summary(.5, .6, .7)}
        summaries.update({arm: summary(.51, .59, .69)
                          for arm in evaluation.ARMS[1:]})
        result = evaluation._guardrails(summaries)
        self.assertEqual(len(result), 4)
        self.assertTrue(result["control"]["clean_spell_at_least_reference_plus_0_01"])
        self.assertTrue(result["control"]["feed_like_spell_at_least_reference_minus_0_01"])
        self.assertTrue(result["control"][
            "per_provider_clean_macro_and_d_at_least_reference_minus_0_02"]["sportec"]["d_f1"])

    def test_ranking_uses_macro_only_after_spell(self):
        values = {}
        for rank, arm in enumerate(evaluation.ARMS):
            value = .5 - rank * .01
            values[arm] = {member: {view: {"equal_provider_mean": {
                "spell_f1": value, "macro_f1": 0.4 + rank * .01}}
                for view in ("clean", "feed_like")}
                for member in evaluation.MEMBERS}
        values["control"]["ensemble"]["clean"]["equal_provider_mean"]["spell_f1"] = .5
        values["control"]["ensemble"]["clean"]["equal_provider_mean"]["macro_f1"] = .8
        ranked = evaluation._rank_arms(values)["ensemble"]["clean"]
        self.assertEqual(ranked[0]["arm"], "control")

    def test_reference_and_continuation_receipts_bind_their_own_declarations(self):
        evaluation._verify_checkpoint_receipt(
            {"model_sha256": "model", "rebuild_declaration_sha256": "rebuild"},
            "models/reference/leave_out_sportec/seed2026.joblib", "model",
            "main", "rebuild")
        evaluation._verify_checkpoint_receipt(
            {"model_sha256": "model", "declaration_sha256": "main",
             "rebuild_declaration_sha256": "rebuild"},
            "models/control/leave_out_sportec/seed2026.joblib", "model",
            "main", "rebuild")
        with self.assertRaises(evaluation.EvaluationError):
            evaluation._verify_checkpoint_receipt(
                {"model_sha256": "model", "declaration_sha256": "main"},
                "models/reference/leave_out_sportec/seed2026.joblib", "model",
                "main", "rebuild")
        with self.assertRaises(evaluation.EvaluationError):
            evaluation._verify_checkpoint_receipt(
                {"model_sha256": "model", "declaration_sha256": "main"},
                "models/control/leave_out_sportec/seed2026.joblib", "model",
                "main", "rebuild")

    def test_evaluation_seal_detects_report_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            execution = Path(temp)
            for name, value in (("predeclared.json", {}),
                                ("rebuild_declaration.json", {}),
                                ("models_sealed.json", {})):
                (execution / name).write_text(json.dumps(value))
            report_path = execution / "aggregate.json"
            rows_path = execution / "scored_rows.json"
            report_path.write_text("{}")
            rows_path.write_text("[]")
            report = {
                "declaration_sha256": evaluation.sha256(execution / "predeclared.json"),
                "model_seal_sha256": evaluation.sha256(execution / "models_sealed.json"),
                "private_rows_sha256": evaluation.sha256(rows_path),
                "private_array_sha256": {},
            }
            seal = {
                "declaration_sha256": report["declaration_sha256"],
                "model_seal_sha256": report["model_seal_sha256"],
                "reference_declaration_sha256": evaluation.sha256(
                    execution / "rebuild_declaration.json"),
                "aggregate_sha256": evaluation.sha256(report_path),
                "scored_rows_sha256": report["private_rows_sha256"],
                "arrays": {},
            }
            evaluation._verify_output_seal(report, seal, execution, report_path, rows_path)
            report_path.write_text("{\"edited\": true}")
            with self.assertRaises(evaluation.EvaluationError):
                evaluation._verify_output_seal(report, seal, execution, report_path, rows_path)

    def test_mocked_run_then_verify_round_trips_full_private_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = root / "execution"
            data = root / "data"
            execution.mkdir()
            data.mkdir()
            for name in ("predeclared.json", "models_sealed.json", "rebuild_declaration.json"):
                (execution / name).write_text("{}\n")

            items = [
                {"source": "sportec", "match": "S1"},
                {"source": "sportec", "match": "S2"},
                {"source": "metrica", "match": "M1"},
                {"source": "skillcorner", "match": "K1"},
                {"source": "skillcorner", "match": "K2"},
            ]
            cells = [dict(item=item, view=view, replicate=replicate)
                     for item in items for view, replicate in evaluation.VIEWS]
            declaration = {"evaluation_cells": cells}
            bases = {}
            cache_paths = {}
            teacher_paths = {}
            model_paths = {}
            for item in items:
                n = 4
                inputs = {
                    "period": np.zeros(n, dtype=np.int32),
                    "timestamps": np.arange(n, dtype=np.float64) / 25,
                    "fps": np.asarray(25.0),
                }
                record = {"x": np.zeros((n, 1), dtype=np.float32), "names": ["x"],
                          "period": np.zeros(n, dtype=np.int32),
                          "indices": np.arange(n), "native_n": n, "fps": 25.0}
                labels = np.asarray(["H", "A", "D", "X"])
                for view, replicate in evaluation.VIEWS:
                    cell_key = (item["match"], view, replicate)
                    cache_path = root / f"cache_{item['match']}_{view}_{replicate}.joblib"
                    cache_paths[str(cache_path)] = {
                        "inputs": inputs, "labels": labels, "record": record,
                    }
                    bases[cell_key] = (inputs, labels, record)
            for provider in ("sportec", "metrica", "skillcorner"):
                for seed in evaluation.SEEDS:
                    path = root / "teachers" / provider / f"seed{seed}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    teacher_paths[str(path)] = {"base": np.asarray([
                        [0.45, 0.45, 0.10 if seed == 2026 else 0.20],
                        [0.40, 0.40, 0.20 if seed == 2026 else 0.30],
                        [0.35, 0.35, 0.30 if seed == 2026 else 0.30],
                        [0.30, 0.30, 0.40 if seed == 2026 else 0.20],
                    ], dtype=np.float32)}
            for arm in evaluation.ARMS:
                for provider in ("sportec", "metrica", "skillcorner"):
                    for seed in evaluation.SEEDS:
                        path = execution / f"models/{arm}/leave_out_{provider}/seed{seed}.joblib"
                        model_paths[str(path)] = {"arm": arm, "provider": provider, "seed": seed}

            def selection_items():
                return items

            def eval_path(item, view, replicate):
                return root / f"cache_{item['match']}_{view}_{replicate}.joblib"

            def control_path(fold, seed):
                provider = fold.removeprefix("leave_out_")
                return root / "teachers" / provider / f"seed{seed}.joblib"

            def score(probability, inputs, labels, item, view, replicate, arm, member):
                prediction = np.argmax(probability, axis=1).astype("U1")
                arm_offset = evaluation.ARMS.index(arm) * 0.01
                row = {
                    "source": item["source"], "match": item["match"],
                    "view": view, "replicate": replicate,
                    "arm": arm, "member": member,
                    "spell_f1": 0.50 + arm_offset,
                    "macro_f1": 0.70 - arm_offset,
                    "weighted_70_30": 0.7 * (0.50 + arm_offset) + 0.3 * (0.70 - arm_offset),
                    "d_f1": 0.75, "d_precision": 0.80, "d_recall": 0.70,
                    "short_spell_recall": 0.25,
                    "n_gt_spells": 2, "n_pred_spells": 3,
                    "fragmented_reference_spells": 0, "graded_frames": 3,
                    "spell_by_iou": {key: 0.50 + arm_offset - index * 0.01
                                     for index, key in enumerate(evaluation.IOU_KEYS)},
                    "probability_sha256": hashlib.sha256(
                        np.ascontiguousarray(probability).tobytes()).hexdigest(),
                    "prediction_sha256": hashlib.sha256(
                        np.ascontiguousarray(prediction).tobytes()).hexdigest(),
                }
                return row, prediction

            published_summary = {
                "cell_count": 5,
                "equal_provider_mean": {
                    "spell_f1": .5, "macro_f1": .7, "weighted_70_30": .56,
                    "d_f1": .75, "d_precision": .8, "d_recall": .7,
                    "short_spell_recall": .25, "n_gt_spells": 2,
                    "n_pred_spells": 3, "fragmented_reference_spells": 0,
                    "graded_frames": 3,
                    "spell_by_iou": {key: .5 - index * .01
                                     for index, key in enumerate(evaluation.IOU_KEYS)},
                },
                "per_provider": {},
            }
            for provider in ("sportec", "metrica", "skillcorner"):
                published_summary["per_provider"][provider] = dict(
                    published_summary["equal_provider_mean"])
            published = {
                "summaries": {"w96": {view: published_summary
                                      for view in ("clean", "feed_like")}},
                "seed_summaries": {"w96": {
                    str(seed): {view: published_summary for view in ("clean", "feed_like")}
                    for seed in evaluation.SEEDS}},
            }
            real_read_json = evaluation.read_json

            def read_json(path):
                if Path(path).resolve() == (
                    evaluation.ROOT / "experiments/results/soccernet_capacity_final_report.json"
                ).resolve():
                    return published
                return real_read_json(path)

            teacher_parity = [
                {"source": cell["item"]["source"], "match": cell["item"]["match"],
                 "view": cell["view"], "replicate": cell["replicate"],
                 "seed": seed, "archive": f"archive/{cell['item']['match']}/{seed}",
                 "probability_atol": 1e-6, "maximum_absolute_probability_delta": 0.0,
                 "decoded_prediction_exact": True}
                for cell in cells for seed in evaluation.SEEDS
            ]

            def model_predict(record, model, teacher, original_probability=None):
                dead = original_probability[:, 2]
                home = (1.0 - dead) * (0.45 + 0.01 * evaluation.ARMS.index(model["arm"]))
                away = 1.0 - dead - home
                return np.column_stack((home, away, dead)).astype(np.float32)

            cache_by_key = {key: value for key, value in bases.items()}

            def cache_cell(item, view, replicate):
                inputs, labels, record = cache_by_key[(item["match"], view, replicate)]
                return Path("mock-cache.joblib"), inputs, labels, record

            def joblib_load(path):
                name = str(path)
                if name in cache_paths:
                    return cache_paths[name]
                if name in teacher_paths:
                    return teacher_paths[name]
                if name in model_paths:
                    return model_paths[name]
                raise AssertionError(f"Unexpected joblib load in synthetic fixture: {name}")

            decode = lambda probability, *args: np.argmax(probability, axis=1).astype("U1")
            with (
                mock.patch.object(evaluation, "_verify_declaration",
                                  return_value=(declaration, {})),
                mock.patch.object(evaluation, "_verify_all_teacher_parity",
                                  return_value=teacher_parity),
                mock.patch.object(evaluation, "read_json", side_effect=read_json),
                mock.patch.object(evaluation.ext, "selection_items", side_effect=selection_items),
                mock.patch.object(evaluation.ext, "eval_path", side_effect=eval_path),
                mock.patch.object(evaluation.ext, "control_path", side_effect=control_path),
                mock.patch.object(evaluation.ext, "upsample", side_effect=lambda p, record, inputs: p),
                mock.patch.object(evaluation.ext, "historical_path", side_effect=lambda *args: root / "unused.npz"),
                mock.patch.object(evaluation.ext, "inference_record", side_effect=lambda value: value),
                mock.patch.object(evaluation, "_verify_historical_teacher",
                                  side_effect=lambda probability, *args: (probability, decode(probability))),
                mock.patch.object(evaluation.fh, "predict", side_effect=lambda record, teacher: teacher["base"]),
                mock.patch.object(evaluation.cap_model, "predict", side_effect=model_predict),
                mock.patch.object(evaluation.cap, "_cache_cell", side_effect=cache_cell),
                mock.patch.object(evaluation.cap, "_score_probability", side_effect=score),
                mock.patch.object(evaluation.joblib, "load", side_effect=joblib_load),
                mock.patch.object(evaluation.supervised, "decode", side_effect=decode),
            ):
                result = evaluation.run(execution, data)
                self.assertEqual(result["evaluation_cells"], 15)
                self.assertEqual(len(result["original_teacher_historical_parity"]), 30)
                rows = json.loads((execution / "evaluation/scored_rows.json").read_text())
                self.assertEqual(len(rows), 225)
                self.assertTrue(all({"source", "arm", "member"}.issubset(row) for row in rows))
                self.assertNotIn("teacher_probability_parity_atol", rows[0])
                verified = evaluation.verify_saved(execution, data)
                self.assertEqual(verified["verified_rows"], 225)
                self.assertEqual(verified["verified_arrays"], 225)

    def test_executor_seal_is_accepted_by_evaluator_declaration_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = root / "execution"
            data = root / "data"
            source_root = root / "repo"
            execution.mkdir()
            data.mkdir()
            report_path = source_root / "experiments/results/soccernet_capacity_final_report.json"
            report_path.parent.mkdir(parents=True)
            report_path.write_text("{}\n")
            source_path = source_root / "source.txt"
            source_path.write_text("source\n")
            rebuild_source = source_root / "rebuild_source.txt"
            rebuild_source.write_text("rebuild\n")
            rebuild_source_map = {"rebuild_source.txt": evaluation.sha256(rebuild_source)}
            rebuild_path = execution / "rebuild_declaration.json"
            rebuild_path.write_text(json.dumps({"source_sha256": rebuild_source_map,
                                                "organizer_labels_read": False,
                                                "automatic_promotion": False}) + "\n")
            rebuild_sha = evaluation.sha256(rebuild_path)

            items = [
                {"source": "sportec", "match": "S1"},
                {"source": "sportec", "match": "S2"},
                {"source": "metrica", "match": "M1"},
                {"source": "skillcorner", "match": "K1"},
                {"source": "skillcorner", "match": "K2"},
            ]
            cells = [dict(item=item, view=view, replicate=replicate)
                     for item in items for view, replicate in evaluation.VIEWS]
            teacher_files = {}
            for provider in ("sportec", "metrica", "skillcorner"):
                for seed in evaluation.SEEDS:
                    teacher_path = data / "teachers" / provider / f"seed{seed}.joblib"
                    teacher_path.parent.mkdir(parents=True, exist_ok=True)
                    teacher_path.write_bytes(b"teacher\n")
                    receipt_path = teacher_path.with_suffix(".json")
                    receipt_path.write_text("{}\n")
                    teacher_files[str(teacher_path)] = evaluation.sha256(teacher_path)
                    teacher_files[str(receipt_path)] = evaluation.sha256(receipt_path)
            evaluation_inputs = {}
            for item in items:
                for view, replicate in evaluation.VIEWS:
                    cache = data / f"{item['match']}_{view}_{replicate}.joblib"
                    cache.write_bytes(b"cache\n")
                    evaluation_inputs[str(cache)] = evaluation.sha256(cache)
                    for seed in evaluation.SEEDS:
                        historical = data / f"{item['match']}_{view}_{replicate}_{seed}.npz"
                        historical.write_bytes(b"historical\n")
                        evaluation_inputs[str(historical)] = evaluation.sha256(historical)

            source_hashes = {
                "experiments/results/soccernet_capacity_final_report.json": evaluation.sha256(report_path),
                "source.txt": evaluation.sha256(source_path),
            }
            reference_models = {}
            reference_receipts = {}
            for fold in executor.FOLDS:
                for seed in executor.SEEDS:
                    model_path = execution / "models/reference" / fold / f"seed{seed}.joblib"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    model_path.write_bytes(f"reference:{fold}:{seed}".encode())
                    receipt_path = model_path.with_suffix(".receipt.json")
                    receipt_path.write_text(json.dumps({
                        "model_sha256": evaluation.sha256(model_path),
                        "rebuild_declaration_sha256": rebuild_sha,
                    }) + "\n")
                    reference_models[str(model_path)] = evaluation.sha256(model_path)
                    reference_receipts[str(receipt_path)] = evaluation.sha256(receipt_path)
            declaration = {
                "source_sha256": source_hashes,
                "reference_models": reference_models,
                "reference_receipts": reference_receipts,
                "teacher_files": teacher_files,
                "evaluation_inputs": evaluation_inputs,
                "evaluation_cells": cells,
                "evaluation": {"organizer_labels_read": False, "public_scoring": False},
                "automatic_promotion": False,
            }
            declaration_path = execution / "predeclared.json"
            declaration_path.write_text(json.dumps(declaration) + "\n")

            # The real executor seal reads all 30 fit receipts and writes the
            # same manifest that evaluator._verify_declaration consumes.
            for arm in executor.ARMS:
                for fold in executor.FOLDS:
                    for seed in executor.SEEDS:
                        model_path = execution / "models" / arm / fold / f"seed{seed}.joblib"
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        if not model_path.exists():
                            model_path.write_bytes(f"{arm}:{fold}:{seed}".encode())
                        receipt_path = model_path.with_suffix(".receipt.json")
                        if arm == "reference":
                            receipt = {"model_sha256": evaluation.sha256(model_path),
                                       "rebuild_declaration_sha256": rebuild_sha}
                        else:
                            receipt = {"model_sha256": evaluation.sha256(model_path),
                                       "declaration_sha256": evaluation.sha256(declaration_path),
                                       "rebuild_declaration_sha256": rebuild_sha,
                                       "training": {"supervised_draws_sha256":
                                           hashlib.sha256(f"{fold}:{seed}".encode()).hexdigest()}}
                        receipt_path.write_text(json.dumps(receipt) + "\n")

            with (
                mock.patch.object(executor, "OUT", execution),
                mock.patch.object(executor, "DECLARATION", declaration_path),
                mock.patch.object(executor, "REBUILD_DECLARATION", rebuild_path),
                mock.patch.object(executor, "_check_followup", return_value=declaration),
                mock.patch.object(evaluation, "ROOT", source_root),
                mock.patch.object(evaluation.ext, "DATA", data),
                mock.patch.object(evaluation.ext, "selection_items", return_value=items),
                mock.patch.object(evaluation.ext, "eval_path",
                                  side_effect=lambda item, view, replicate:
                                  data / f"{item['match']}_{view}_{replicate}.joblib"),
                mock.patch.object(evaluation.ext, "historical_path",
                                  side_effect=lambda item, view, replicate, seed:
                                  data / f"{item['match']}_{view}_{replicate}_{seed}.npz"),
                mock.patch.object(evaluation.ext, "control_path",
                                  side_effect=lambda fold, seed:
                                  data / "teachers" / fold.removeprefix("leave_out_") / f"seed{seed}.joblib"),
            ):
                sealed = executor.seal()
                self.assertEqual(len(sealed["models"]), 30)
                self.assertEqual(len(sealed["receipts"]), 30)
                parsed, _ = evaluation._verify_declaration(execution, data)
                self.assertEqual(parsed["source_sha256"], source_hashes)
                self.assertEqual(sealed["reference_declaration_sha256"], rebuild_sha)


if __name__ == "__main__":
    unittest.main()
