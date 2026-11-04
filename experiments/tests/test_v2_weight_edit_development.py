"""Contract tests for bounded development scoring and weight edit helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

from experiments import v2_weight_edit_development as dev


class EditedModelInterfaceTests(unittest.TestCase):
    def test_each_arm_changes_only_its_declared_h_a_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fold = "leave_out_sportec"
            model_path = root / "models" / f"{fold}_2026.joblib"
            model_path.parent.mkdir(parents=True)
            base = {
                "hidden": 2,
                "state": {
                    "output.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
                    "output.bias": np.array([0.1, -0.1], dtype=np.float32),
                    "sequence.bias_ih_l0": np.arange(6, dtype=np.float32),
                    "sequence.bias_ih_l0_reverse": np.arange(6, dtype=np.float32) + 10,
                    "sequence.weight_ih_l0": np.arange(24, dtype=np.float32).reshape(6, 4),
                },
                "d_state": {"output.weight": np.array([[7.]], dtype=np.float32)},
                "median": np.zeros(2, dtype=np.float32),
                "scale": np.ones(2, dtype=np.float32),
            }
            joblib.dump(base, model_path)
            direction_path = root / "directions.npz"
            np.savez_compressed(direction_path, **{f"{fold}/2026": np.array([1, 0, 0, 0], np.float32)})
            with mock.patch.multiple(dev, TRAIN_DIRECTIONS=direction_path), \
                    mock.patch.object(dev.reconstruction, "OUT", root):
                unchanged = dev._edited_model(fold, 2026, "blend_unedited_ha15")
                projection = dev._edited_model(fold, 2026, "projection_a050")
                gate = dev._edited_model(fold, 2026, "update_bias_m025")

            np.testing.assert_array_equal(unchanged["state"]["output.weight"], base["state"]["output.weight"])
            self.assertFalse(np.array_equal(projection["state"]["output.weight"], base["state"]["output.weight"]))
            for key in base["state"]:
                if key != "output.weight":
                    np.testing.assert_array_equal(projection["state"][key], base["state"][key])
            for key in base["state"]:
                if key not in ("sequence.bias_ih_l0", "sequence.bias_ih_l0_reverse"):
                    np.testing.assert_array_equal(gate["state"][key], base["state"][key])
            for key in ("sequence.bias_ih_l0", "sequence.bias_ih_l0_reverse"):
                np.testing.assert_array_equal(gate["state"][key][:2], base["state"][key][:2])
                np.testing.assert_allclose(gate["state"][key][2:4], base["state"][key][2:4] - .25)
                np.testing.assert_array_equal(gate["state"][key][4:], base["state"][key][4:])
            np.testing.assert_array_equal(projection["d_state"]["output.weight"], base["d_state"]["output.weight"])
            np.testing.assert_array_equal(gate["d_state"]["output.weight"], base["d_state"]["output.weight"])


class PredictionCacheTests(unittest.TestCase):
    def test_dead_probability_cache_is_separate_by_native_phase_and_shared_across_arms(self):
        view = {"id": "same-view"}
        phase0 = {"x": np.array([[0.]], dtype=np.float32), "names": [], "period": np.array([1]),
                  "indices": np.array([0]), "native_n": 1, "fps": 25., "y": np.array([0])}
        phase1 = {"x": np.array([[1.]], dtype=np.float32), "names": [], "period": np.array([1]),
                  "indices": np.array([0]), "native_n": 1, "fps": 25., "y": np.array([1])}
        model = {"d_state": {"id": 3}}
        cache = {}
        d_calls = []
        probability_calls = []

        class Old:
            @staticmethod
            def inference_record(record):
                # Match the real adapter: labels are absent from the network input.
                return {key: record[key] for key in ("x", "names", "period", "indices", "native_n", "fps")}

            @staticmethod
            def upsample(probability, record, inputs):
                return probability

        class Capacity:
            @staticmethod
            def predict(record, model, teacher, *, original_probability):
                if "y" in record:
                    raise AssertionError("training/evaluation labels reached model inference")
                probability_calls.append(float(record["x"][0, 0]))
                return np.asarray(original_probability)

        fake_upstream = mock.Mock()
        fake_upstream.frozen.audit.FOLD = {"sportec": "sportec"}
        fake_upstream.frozen.audit.old = Old
        fake_upstream.frozen.sn.cap = Capacity
        fake_upstream.frozen.tc.inputs_only.return_value = object()
        with mock.patch.object(dev, "upstream", fake_upstream), \
                mock.patch.object(dev, "_teacher_path", return_value=Path("teacher.joblib")), \
                mock.patch.object(dev.joblib, "load", return_value={"d_state": {"old": 1}}), \
                mock.patch("experiments.factorized_had.predict", side_effect=lambda record, teacher: (
                    d_calls.append(float(record["x"][0, 0])) or np.array([[.5, .4, .1]], np.float32)
                )):
            dev._new_prediction(view, phase0, "sportec", "m", "clean", 0, 0, 2026, model, cache)
            dev._new_prediction(view, phase0, "sportec", "m", "clean", 0, 0, 2026, model, cache)
            dev._new_prediction(view, phase1, "sportec", "m", "clean", 0, 1, 2026, model, cache)

        self.assertEqual(d_calls, [0.0, 1.0])
        self.assertEqual(probability_calls, [0.0, 0.0, 1.0])
        self.assertEqual(len(cache), 2)


class FreezeAndAggregationTests(unittest.TestCase):
    def test_freeze_declaration_round_trips_json_and_is_reusable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            declaration_path = root / "declaration.json"
            transfer_path = root / "transfer.json"
            transfer_path.write_text(json.dumps({
                "highest_eligible_arm": "equal_50_50_new_full_plus_archived_calibrated",
                "selected_config": {"config": {"family": "calibration", "dt": 4,
                                                "dead": 3, "ha": 1.5, "ht": 1.0}},
            }))
            fake_facts = {"fits": {}, "directions": {}, "models": {"x": {"sha256": "model"}}}
            fake_direction = {"projection_edit_eligible_all_six": False,
                              "training_only": True, "evaluation_inputs_used": False,
                              "rows": []}
            active_arms = {name: value for name, value in dev.EDIT_ARMS.items()
                           if not name.startswith("projection_")}
            with mock.patch.multiple(dev, ROOT=root, DECLARATION=declaration_path,
                                     CONTROL_RESULT=root / "control.json", RESULT=root / "result.json",
                                     TRAIN_DIRECTIONS=root / "data/directions.npz",
                                     TRAIN_DIRECTION_REPORT=root / "data/directions.json"), \
                    mock.patch.object(dev, "_verify_training_artifacts", return_value=fake_facts), \
                    mock.patch.object(dev, "_direction_artifacts", return_value=fake_direction), \
                    mock.patch.object(dev, "_active_arms", return_value=active_arms), \
                    mock.patch.object(dev, "_source_hashes", return_value={"source.py": "abc"}), \
                    mock.patch.object(dev, "sha", return_value="fixed-hash"):
                # freeze reads this path relative to its patched ROOT
                target = root / "experiments/results/v2_calibration_transfer.json"
                target.parent.mkdir(parents=True)
                target.write_text(transfer_path.read_text())
                frozen = dev.freeze()
                loaded = json.loads(declaration_path.read_text())
                dev._require_declaration()

            self.assertEqual(loaded["arms"], active_arms)
            self.assertEqual(loaded["study_id"], frozen["study_id"])
            self.assertEqual(loaded["cohort"]["primary_views"], ["clean", "feed_like_v2"])

    def test_aggregate_averages_replicates_then_matches_then_providers(self):
        rows = {}
        values = {
            ("p1", "a", "clean", 0): 0.2,
            ("p1", "a", "clean", 1): 0.4,
            ("p1", "b", "clean", 0): 0.8,
            ("p2", "c", "clean", 0): 0.6,
        }
        for key, spell in values.items():
            rows[key] = {"arm": {metric: spell for metric in dev.METRICS}}
        with mock.patch.object(dev.upstream.frozen.audit, "equal_provider", wraps=dev.upstream.frozen.audit.equal_provider):
            result = dev._aggregate(rows, "arm", "clean")
        # p1=(mean(.2,.4)+.8)/2=.55; p2=.6; provider mean=.575
        self.assertAlmostEqual(result["spell_f1"]["equal_provider_mean"], .575)
        self.assertAlmostEqual(result["spell_f1"]["providers"]["p1"], .55)
        self.assertAlmostEqual(result["spell_f1"]["providers"]["p2"], .6)

    @staticmethod
    def _aggregate_row(v2_spell, macro=.8, d_f1=.9, clean_spell=.665):
        def metric(value, providers=None):
            return {"equal_provider_mean": value,
                    "providers": providers or {"Sportec": value, "Metrica": value,
                                                "SkillCorner": value}}
        return {
            "clean": {"spell_f1": metric(clean_spell), "macro_f1": metric(macro), "d_f1": metric(d_f1)},
            "feed_like_v2": {"spell_f1": metric(v2_spell), "macro_f1": metric(macro), "d_f1": metric(d_f1)},
        }

    def test_all_guardrails_gate_selection_and_ties_use_macro_only_after_spell(self):
        control = self._aggregate_row(.5836, macro=.87, d_f1=.90, clean_spell=.6676)
        good_a = self._aggregate_row(.59, macro=.86, d_f1=.89)
        good_b = self._aggregate_row(.59, macro=.87, d_f1=.89)
        better_spell = self._aggregate_row(.5900001, macro=.85, d_f1=.89)
        failing_clean = self._aggregate_row(.59, macro=.86, d_f1=.89, clean_spell=.662)
        failing_v2 = self._aggregate_row(.59, macro=.86, d_f1=.89)
        failing_v2["feed_like_v2"]["spell_f1"]["equal_provider_mean"] = .573
        failing_provider = self._aggregate_row(.59, macro=.86, d_f1=.89)
        failing_provider["clean"]["macro_f1"]["providers"]["Metrica"] = .84

        aggregates = {"blend_unedited_ha15": good_a, "candidate_b": good_b,
                      "candidate_spell": better_spell, "clean_fail": failing_clean,
                      "v2_fail": failing_v2, "provider_fail": failing_provider,
                      "deployed_original_full": control}
        baseline_guards = {"macro_f1": {"providers": {p: .87 for p in dev.upstream.frozen.audit.PROVIDER.values()}},
                           "d_f1": {"providers": {p: .90 for p in dev.upstream.frozen.audit.PROVIDER.values()}}}
        # Guard thresholds are bound to the saved baseline phase-average report,
        # not the candidate-run's archived-control aggregate.
        with mock.patch.object(dev, "_baseline_clean_provider_guards", return_value=baseline_guards):
            guards = dev._evaluate_guards(aggregates, control)
        self.assertTrue(guards["blend_unedited_ha15"]["eligible"])
        self.assertTrue(guards["candidate_b"]["eligible"])
        self.assertFalse(guards["clean_fail"]["eligible"])
        self.assertFalse(guards["v2_fail"]["eligible"])
        self.assertFalse(guards["provider_fail"]["eligible"])
        self.assertEqual(dev._select_winner(aggregates, guards), "candidate_spell")

        del guards["candidate_spell"]
        self.assertEqual(dev._select_winner(aggregates, guards), "candidate_b")
        guards = {arm: {"eligible": False} for arm in guards}
        self.assertIsNone(dev._select_winner(aggregates, guards))

    def test_freeze_winner_binds_primary_result_before_v1_is_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            primary = root / "primary.json"
            winner_path = root / "winner.json"
            selected_choice_path = root / "selected_choice.json"
            v1_path = root / "v1.json"
            selected_arm = "blend_unedited_ha15"
            primary.write_text(json.dumps({"eligible_winner": selected_arm,
                                           "arms": {},
                                           "guard_results": {selected_arm: {"eligible": True}}}))
            with mock.patch.multiple(dev, ROOT=root, RESULT=primary, WINNER_RECEIPT=winner_path,
                                     SELECTED_CHOICE=selected_choice_path, V1_RESULT=v1_path), \
                    mock.patch.object(dev, "sha", return_value="sealed"), \
                    mock.patch.object(dev, "_verify_training_artifacts", return_value={"models": {}}):
                winner = dev.freeze_winner()
                self.assertFalse(v1_path.exists())
                self.assertTrue(winner["v1_scoring_may_start"])
                self.assertEqual(winner["winner"], selected_arm)
                self.assertTrue(winner["eligible"])
                self.assertEqual(json.loads(selected_choice_path.read_text())["winner_id"], selected_arm)

                # v1 is an outcome readout for the frozen winner, never a new
                # selection input; it cannot overwrite a prior result.
                with mock.patch.object(dev, "_json", side_effect=lambda p: json.loads(p.read_text())):
                    original = dev._score_cell
                    calls = []
                    def fake_score(*args, **kwargs):
                        calls.append(args[4])
                        return {arm: {"spell_f1": .5, "macro_f1": .8} for arm in args[4]}
                    with mock.patch.object(dev, "_score_cell", side_effect=fake_score), \
                            mock.patch.object(dev, "_aggregate", side_effect=lambda rows, arm, view: {"spell_f1": .5}), \
                            mock.patch.object(dev.joblib, "load", return_value={
                                "inputs": {"x": np.zeros((2, 1))},
                                "labels": np.array(["H", "A"]), "record": {"x": np.zeros((2, 1))}}), \
                            mock.patch.object(dev.upstream.frozen.audit, "phase_records", return_value=[{}]), \
                            mock.patch.object(dev.upstream.frozen.audit.supervised, "model_periods",
                                              return_value=np.array([1, 1])):
                        result = dev.score_v1()
                self.assertEqual(result["winner"], selected_arm)
                self.assertTrue(all(call == ["candidate", "blend_unedited_ha15", "deployed_original_full"]
                                    for call in calls))
                self.assertFalse(result["selection_used_v1"])
                self.assertNotIn("eligible_winner", result)


if __name__ == "__main__":
    unittest.main()
