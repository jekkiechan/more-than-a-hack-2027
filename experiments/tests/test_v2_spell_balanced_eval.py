import unittest
import json
from concurrent.futures import Future
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path

import numpy as np

from experiments import v2_spell_balanced_eval as evaluation


class BalancedRecipeTests(unittest.TestCase):
    def test_declares_36_valid_recipes_and_anchor(self):
        self.assertEqual(len(evaluation.RECIPES), 36)
        self.assertEqual(len(set(evaluation.ARMS)), 36)
        self.assertIn(evaluation.recipe_key(evaluation.ANCHOR), evaluation.ARMS)
        for recipe in evaluation.RECIPES:
            self.assertAlmostEqual(recipe["ce_weight"] + recipe["calibrated_weight"] +
                                   recipe["adopted_weight"], 1.0)
            self.assertGreaterEqual(recipe["adopted_weight"], 0.0)

    def test_arithmetic_and_log_pooling_are_conditional_ha(self):
        components = {
            "ce": np.array([[.45, .45, .10], [.10, .80, .10]]),
            "calibrated": np.array([[.20, .70, .10], [.60, .30, .10]]),
            "adopted": np.array([[.70, .20, .10], [.70, .20, .10]]),
        }
        arithmetic = evaluation.recipe_probability(components, .5, .0, "arithmetic")
        log = evaluation.recipe_probability(components, .5, .0, "log")
        np.testing.assert_allclose(arithmetic, [[.6388888888888888, .3611111111111111],
                                               [.4444444444444444, .5555555555555556]], atol=1e-12)
        np.testing.assert_allclose(arithmetic.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(log.sum(axis=1), 1.0, atol=1e-12)
        self.assertFalse(np.allclose(arithmetic, log))

    def test_fixed_d_decoder_matches_historical_helper_bitwise_on_synthetic_cells(self):
        from experiments import v2_data_capacity_eval as historical
        period = np.array([0] * 5 + [1] * 5 + [2] * 4)
        adopted_probability = np.array(
            [[.62, .18, .20], [.65, .15, .20], [.20, .20, .60], [.22, .58, .20],
             [.18, .62, .20]] * 2 + [[.20, .20, .60], [.55, .25, .20], [.54, .26, .20], [.52, .28, .20]])
        candidate = np.array(
            [[.51, .39, .10], [.48, .42, .10], [.45, .45, .10], [.44, .46, .10],
             [.43, .47, .10]] * 2 + [[.49, .41, .10], [.47, .43, .10], [.45, .45, .10], [.44, .46, .10]])
        adopted_decoded = evaluation.controls.audit.decode(adopted_probability, period, 25.,
                                                           evaluation.controls.audit.CONFIG)
        for ha_cost in (1.5, 3.0):
            actual = evaluation.decode_fixed_d(candidate, adopted_decoded, period, 25., ha_cost)
            expected = historical._adopted_dead_prediction(candidate, adopted_probability,
                period, 25., historical._decoder(ha_cost))
            np.testing.assert_array_equal(actual, expected)

    def test_fixed_d_decoder_matches_historical_helper_on_saved_real_cell(self):
        cache = evaluation.controls.OUT / "controls/Metrica__metrica_3__clean__0.npz"
        if not cache.is_file():
            self.skipTest("Restored development control cache is unavailable")
        from experiments import v2_data_capacity_eval as historical
        with np.load(cache, allow_pickle=False) as saved:
            adopted = saved["prob__adopted"].copy()
            candidate = saved["prob__calibrated"].copy()
            period = saved["model_period"].copy()
            fps = float(saved["fps"])
        adopted_decoded = evaluation.controls.audit.decode(adopted, period, fps,
                                                           evaluation.controls.audit.CONFIG)
        expected = historical._adopted_dead_prediction(candidate, adopted, period, fps,
                                                        historical._decoder(3.0))
        actual = evaluation.decode_fixed_d(candidate, adopted_decoded, period, fps, 3.0)
        np.testing.assert_array_equal(actual, expected)

    def test_eligibility_and_shortlist_schema_are_deterministic(self):
        baseline = {view: {"spell_f1": .5, "macro_f1": .8,
                           "short_spell_recall_iou_0_5_lt3s": .25}
                    for view, _ in evaluation.VIEWS}
        candidate = {view: {"spell_f1": .5001, "macro_f1": .799,
                            "short_spell_recall_iou_0_5_lt3s": .25}
                     for view, _ in evaluation.VIEWS}
        self.assertTrue(evaluation.eligibility(candidate, baseline)["eligible"])
        candidate["feed_like_v2"]["spell_f1"] = .5
        self.assertFalse(evaluation.eligibility(candidate, baseline)["eligible"])

        anchor = evaluation.recipe_key(evaluation.ANCHOR)
        ids = [anchor, "recipe_b", "recipe_c"]
        gates = {recipe: {"eligible": True} for recipe in ids}
        pooled = {recipe: {view: {"spell_f1": {"equal_provider": value},
                                 "macro_f1": {"equal_provider": .8}}
                           for view, _ in evaluation.VIEWS}
                  for recipe, value in zip(ids, (.61, .64, .63))}
        self.assertEqual(evaluation.select_seed_shortlist(gates, pooled),
                         ["recipe_b", "recipe_c", anchor])

    def test_nested_equal_provider_aggregate_uses_exact_frozen_views(self):
        rows = {}
        for provider, match, _filename in evaluation.runtime.frozen.audit.MATCHES:
            for view, reps in evaluation.VIEWS:
                for rep in reps:
                    rows[evaluation._cell_key(provider, match, view, rep)] = {
                        metric: float(rep) + 1.0 for metric in evaluation.METRICS}
        aggregate = evaluation.aggregate_cell_metrics(rows)
        self.assertEqual(set(aggregate), {"clean", "feed_like_v1", "feed_like_v2"})
        self.assertEqual(aggregate["clean"]["spell_f1"]["cells"], 5)
        self.assertEqual(aggregate["feed_like_v1"]["spell_f1"]["cells"], 10)
        self.assertEqual(aggregate["feed_like_v2"]["spell_f1"]["cells"], 20)
        self.assertEqual(set(aggregate["feed_like_v2"]["spell_f1"]["providers"]),
                         {"Metrica", "SkillCorner", "Sportec"})

    def test_round_two_seed_schema_keeps_recipe_seed_and_view_levels(self):
        anchor = evaluation.recipe_key(evaluation.ANCHOR)
        recipe_b = "recipe_b"
        seed_rows = {}
        for provider, match, _filename in evaluation.runtime.frozen.audit.MATCHES:
            for view, reps in evaluation.VIEWS:
                for rep in reps:
                    cell = evaluation._cell_key(provider, match, view, rep)
                    seed_rows[cell] = {}
                    for arm in (anchor, recipe_b):
                        for seed in evaluation.SEEDS:
                            seed_rows[cell][f"{arm}|{seed}"] = {
                                metric: .5 + (seed - 2026) * .01 for metric in evaluation.METRICS}
        summary = evaluation.aggregate_seed_metrics(seed_rows, [anchor, recipe_b])
        self.assertEqual(set(summary), {anchor, recipe_b})
        self.assertEqual(set(summary[anchor]), {"2026", "2027"})
        self.assertEqual(summary[anchor]["2026"]["feed_like_v2"]["spell_f1"]["cells"], 20)

    def test_controls_replay_writes_all_48_checks_before_candidate_scoring(self):
        cells = evaluation._expected_cells()
        keys = [evaluation._cell_key(*cell) for cell in cells]
        baseline_row = {metric: .5 for metric in evaluation.METRICS}
        baseline_row.update(reference_spell_count=101, predicted_spell_count=80)
        anchor_row = {metric: .6 for metric in evaluation.METRICS}
        anchor_row.update(reference_spell_count=101, predicted_spell_count=90)
        expected_baseline = evaluation.aggregate_cell_metrics({key: baseline_row for key in keys})
        expected_anchor = evaluation.aggregate_cell_metrics({key: anchor_row for key in keys})
        fact = {"path": "unused.npz", "sha256": "0" * 64,
                "identity": {"study_id": evaluation.STUDY}}
        declaration = {"controls_replay": {"baseline_expected": expected_baseline,
                                             "anchor_expected": expected_anchor},
                       "upstream": {"adopted_calibrated_emissions": {key: fact for key in keys}}}
        components = {"adopted": np.tile([.6, .2, .2], (2, 1)),
                      "calibrated": np.tile([.3, .5, .2], (2, 1)),
                      "ce": np.tile([.4, .4, .2], (2, 1)),
                      "components": {"adopted": np.tile([.6, .2, .2], (2, 1)),
                                     "calibrated": np.tile([.3, .5, .2], (2, 1)),
                                     "ce": np.tile([.4, .4, .2], (2, 1))},
                      "labels": np.array(["H", "A"]), "period": np.zeros(2),
                      "timestamps": np.arange(2), "model_period": np.zeros(2), "fps": 25.}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration_path, receipt_path = root / "declaration.json", root / "replay.json"
            declaration_path.write_text("{}\n")
            with patch.object(evaluation, "DECLARATION", declaration_path), \
                 patch.object(evaluation, "CONTROL_REPLAY", receipt_path), \
                 patch.object(evaluation, "RESULT", root / "result.json"), \
                 patch.object(evaluation, "require_declaration", return_value=declaration), \
                 patch.object(evaluation, "_load_control_arrays", return_value=components), \
                 patch.object(evaluation, "_load_components_for_cell", return_value=components), \
                 patch.object(evaluation.controls.audit, "decode", return_value=np.array(["H", "A"])), \
                 patch.object(evaluation.controls.audit, "score", return_value=baseline_row), \
                 patch.object(evaluation, "_metrics_for_recipe", return_value=(anchor_row, np.array(["H", "A"]))):
                receipt = evaluation.replay_controls()
        self.assertTrue(receipt["passed"])
        self.assertFalse(receipt["candidate_scoring"])
        self.assertEqual(len(receipt["checks"]), 48)
        self.assertEqual(receipt["cell_count"], 35)
        self.assertEqual(receipt["cell_counts"]["feed_like_v2/4"], 5)

    def test_mocked_round1_assembles_all_cells_recipes_and_shortlist(self):
        cells = evaluation._expected_cells()
        keys = [evaluation._cell_key(*cell) for cell in cells]
        baseline_values = {view: {metric: .5 for metric in evaluation.METRICS}
                           for view, _ in evaluation.VIEWS}
        for view, _ in evaluation.VIEWS:
            baseline_values[view].update(spell_f1=.50, macro_f1=.80,
                short_spell_recall_iou_0_5_lt3s=.25, d_f1=.7, d_precision=.8,
                d_recall=.6, predicted_spell_count=100, reference_spell_count=101)
        baseline_rows = {}
        for provider, match, _filename in evaluation.runtime.frozen.audit.MATCHES:
            for view, reps in evaluation.VIEWS:
                for rep in reps:
                    row = dict(baseline_values[view])
                    baseline_rows[evaluation._cell_key(provider, match, view, rep)] = row
        baseline = evaluation.aggregate_cell_metrics(baseline_rows)
        control_facts = {key: {"path": "control.npz", "sha256": "c" * 64} for key in keys}
        declaration = {"upstream": {"adopted_calibrated_emissions": control_facts}}
        replay = {"adopted": baseline, "reference_spell_counts": {key: 101 for key in keys},
                  "cell_counts": evaluation._cell_counts({key: {} for key in keys}),
                  "passed": True, "candidate_scoring": False}
        anchor = evaluation.recipe_key(evaluation.ANCHOR)
        values = {arm: .61 for arm in evaluation.ARMS}
        values[evaluation.ARMS[0]], values[evaluation.ARMS[1]], values[anchor] = .70, .69, .60
        ce_facts = {key: {"path": "ce.npz", "sha256": "e" * 64} for key in keys}
        declaration["upstream"]["stage512_ce_emissions"] = ce_facts

        def fake_cell(job):
            key, _fact, expected_reference_count = job
            view = key.split("|")[2]
            rows = {}
            for arm in evaluation.ARMS:
                row = dict(baseline_values[view])
                row["spell_f1"] = values[arm]
                row["reference_spell_count"] = expected_reference_count
                rows[arm] = row
            return key, rows, {"path": "prediction.npz", "sha256": "p" * 64,
                               "identity": {"cell": key}}

        def fake_seed(job):
            key, selected = job
            metrics = {f"{recipe['recipe_id']}|{seed}": dict(baseline_values["clean"])
                       for recipe in selected for seed in evaluation.SEEDS}
            return key, metrics, {"path": "seed.npz", "sha256": "s" * 64,
                                  "identity": {"cell": key}}

        class SyncExecutor:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def submit(self, function, job):
                future = Future()
                future.set_result(function(job))
                return future

        with TemporaryDirectory() as directory:
            root = Path(directory)
            declaration_path, replay_path = root / "declaration.json", root / "control_replay.json"
            declaration_path.write_text("{}\n")
            replay_path.write_text(json.dumps(replay))
            with patch.object(evaluation, "DECLARATION", declaration_path), \
                 patch.object(evaluation, "CONTROL_REPLAY", replay_path), \
                 patch.object(evaluation, "RESULT", root / "result.json"), \
                 patch.object(evaluation, "OUT", root / "data"), \
                 patch.object(evaluation, "require_declaration", return_value=declaration), \
                 patch.object(evaluation, "ProcessPoolExecutor", SyncExecutor), \
                 patch.object(evaluation, "_score_cell_job", side_effect=fake_cell), \
                 patch.object(evaluation, "_seed_diagnostic_job", side_effect=fake_seed):
                result = evaluation.run_round1()
        self.assertEqual(result["cohort"]["cell_count"], 35)
        self.assertEqual(len(result["recipe_ids"]), 36)
        self.assertEqual(len(result["cell_metrics"]), 35)
        self.assertTrue(all(len(rows) == 36 for rows in result["cell_metrics"].values()))
        self.assertEqual(len(result["candidate_prediction_cache"]), 35)
        self.assertEqual(result["selected_for_round2_seed_diagnostics"],
                         [evaluation.ARMS[0], evaluation.ARMS[1], anchor],
                         msg=str({arm: result["eligibility"][arm] for arm in evaluation.ARMS[:2]}))
        self.assertEqual(result["selected_for_round2"][0]["training_stage"], 512)
        self.assertEqual(set(result["seed_diagnostics"]), {evaluation.ARMS[0], evaluation.ARMS[1], anchor})


if __name__ == "__main__":
    unittest.main()
