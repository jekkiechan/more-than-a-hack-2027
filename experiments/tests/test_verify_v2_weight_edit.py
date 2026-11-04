"""Smoke the external verifier against a real NPZ/receipt-shaped fixture."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments import calibrated_dense_baseline, feed_spell_push
from scripts import verify_v2_weight_edit as verify


class SavedCellReplayTests(unittest.TestCase):
    def test_replay_stage_replays_metrics_and_saved_probability_decodes(self):
        key = ("Sportec", "J03WOY", "clean", 0)
        labels = np.asarray(["H"] * 20 + ["D"] * 10 + ["A"] * 20 + ["D"] * 10)
        fps = 25.0
        period = np.ones(len(labels), dtype=np.int8)
        timestamps = np.arange(len(labels), dtype=float) / fps
        decode_period = verify.model_period_ids(period, timestamps, fps)
        codes = {"H": 0, "A": 1, "D": 2}
        probabilities = np.full((len(labels), 3), .0005, dtype=np.float64)
        probabilities[np.arange(len(labels)), [codes[x] for x in labels]] = .999
        config = {"family": "calibration", "dt": 4, "dead": 3, "ha": 1.5, "ht": 1.0}
        prediction = feed_spell_push.decode(probabilities[None], decode_period, fps, config)
        old_probability = probabilities.copy()
        old_prediction = feed_spell_push.decode(
            old_probability[None], decode_period, fps, calibrated_dense_baseline.CONFIG)
        pool_score = verify.score_saved(labels, prediction, period, timestamps, fps)
        old_score = verify.score_saved(labels, old_prediction, period, timestamps, fps)
        seed_scores = {str(seed): pool_score for seed in (2026, 2027)}
        metrics = {"arm": pool_score, "deployed_original_full": old_score,
                   "_seed_metrics": {"arm": seed_scores}}
        input_hash = "a" * 64
        declaration_hash = "b" * 64

        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            stage_dir = out / "cell_artifacts" / "candidate"
            stage_dir.mkdir(parents=True)
            stem = verify.key_stem(key)
            npz_path = stage_dir / f"{stem}.npz"
            np.savez_compressed(
                npz_path,
                labels=labels.astype("S1"), period=period, timestamps=timestamps,
                fps=np.asarray(fps), indices=np.arange(len(labels), dtype=np.int64),
                reference_denominator=np.asarray(len(labels), dtype=np.int64),
                graded_frames=np.asarray(len(labels), dtype=np.int64),
                frame_count=np.asarray(len(labels), dtype=np.int64),
                probabilities__arm=probabilities, predictions__arm=prediction.astype("S1"),
                seed_probabilities__arm__2026=probabilities,
                seed_probabilities__arm__2027=probabilities,
                deployed_original_probabilities=old_probability,
                deployed_original_predictions=old_prediction.astype("S1"),
                metrics_json=np.asarray(json.dumps(metrics)),
                input_sha256=np.asarray(input_hash), declaration_sha256=np.asarray(declaration_hash))
            receipt = {"study_id": verify.STUDY, "stage": "candidate", "key": list(key),
                       "arrays_sha256": verify.sha(npz_path),
                       "declaration_sha256": declaration_hash, "input_sha256": input_hash}
            (stage_dir / f"{stem}.json").write_text(json.dumps(receipt))
            saved_rows = {"|".join(map(str, key)): metrics}
            with mock.patch.object(verify, "OUT", out), \
                    mock.patch.object(verify, "expected_keys", return_value={key}):
                replayed, receipts = verify.replay_stage(
                    "candidate", ("arm", "deployed_original_full"), declaration_hash,
                    saved_rows=saved_rows)
        self.assertEqual(len(receipts), 1)
        self.assertAlmostEqual(replayed[key]["arm"]["spell_f1"], pool_score["spell_f1"])
        self.assertAlmostEqual(replayed[key]["deployed_original_full"]["macro_f1"],
                               old_score["macro_f1"])

    def test_gar_saved_result_replays_per_game_and_paired_aggregate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = root / "experiments/results"
            gar_root = root / "data/experiments/v2_weight_edit_evaluation_v1/games"
            results.mkdir(parents=True)
            gar_root.mkdir(parents=True)
            games = [f"g{i}" for i in range(10)]
            declaration = {"study": "v2_weight_edit_evaluation_v1",
                           "cohort": {"match_ids": games},
                           "selected_package": {"receipt_sha256": "c" * 64},
                           "baseline": {}}
            declaration_path = results / "v2_weight_edit_evaluation_v1_declaration.json"
            declaration_path.write_text(json.dumps(declaration))
            declaration_hash = verify.sha(declaration_path)
            labels = np.asarray(["H"] * 20 + ["D"] * 10 + ["A"] * 20 + ["D"] * 10)
            fps = 25.
            raw_period = np.ones(len(labels), dtype=np.int8)
            timestamps = np.arange(len(labels), dtype=float) / fps
            scores_by_game = {}
            report = {"declaration_sha256": declaration_hash,
                      "organizer_reference_access": False, "public_scoring": False,
                      "per_game": {"clean": {}, "v2": {}, "paired": {"clean": {}, "v2": {}}},
                      "per_realization_equal_game_means": {}, "summary": {"clean": {}, "v2": {}}}
            for game_index, game in enumerate(games):
                game_scores = {}
                predictions = {}
                for view_index, view in enumerate(verify.GAR_VIEWS):
                    pred = labels.copy()
                    if (game_index + view_index) % 3 == 0:
                        pred[23] = "D"
                    predictions[view] = pred
                    measured = verify.score_saved(labels, pred, raw_period, timestamps, fps)
                    measured["graded_frames"] = int(np.sum(labels != "X"))
                    game_scores[view] = measured
                scores_by_game[game] = game_scores
                npz_path = gar_root / f"test__{game}.npz"
                np.savez_compressed(npz_path, labels=labels.astype("S1"), period=raw_period,
                                    timestamps=timestamps, fps=np.asarray(fps), **predictions)
                game_path = gar_root / f"test__{game}.json"
                game_path.write_text(json.dumps({"game_id": game, "split": "test",
                    "declaration_sha256": declaration_hash, "prediction_sha256": verify.sha(npz_path),
                    "package_receipt_sha256": "c" * 64,
                    "seed_namespace_reused": f"gar_deployment_eval_v2/{game}",
                    "grading_and_reference_counts_equal": True,
                    "phase_counts": {view: 5 for view in verify.GAR_VIEWS},
                    "graded_frames": int(np.sum(labels != "X")),
                    "native_frames": len(labels), "scores": game_scores}))
            for view in ("clean", "v2"):
                for game in games:
                    report["per_game"][view][game] = (scores_by_game[game]["clean"] if view == "clean"
                        else {metric: float(np.mean([scores_by_game[game][f"v2_{i}"][metric]
                                                     for i in range(1, 5)]))
                            for metric in verify.GAR_METRICS})
            for i in range(1, 5):
                report["per_realization_equal_game_means"][f"v2_{i}"] = {
                    metric: float(np.mean([scores_by_game[g][f"v2_{i}"][metric] for g in games]))
                    for metric in verify.GAR_METRICS}
            baseline_scores = {view: {"original_full": {"phase_avg": {"per_game": {
                game: report["per_game"][view][game] for game in games}}}} for view in ("clean", "v2")}
            gar_baseline = results / "gar_deployment_eval_v2.json"
            gar_baseline.write_text(json.dumps({"splits": {"test": {"scores": baseline_scores}}}))
            public_baseline = results / "feed_like_v2_public_result.json"
            public_baseline.write_text(json.dumps({"candidate": {"spell_f1_mean": .6, "macro_f1": .8}}))
            declaration["baseline"] = {"gar_report_sha256": verify.sha(gar_baseline),
                                        "public_report_sha256": verify.sha(public_baseline)}
            declaration_path.write_text(json.dumps(declaration))
            declaration_hash = verify.sha(declaration_path)
            report["declaration_sha256"] = declaration_hash
            for game in games:
                game_path = gar_root / f"test__{game}.json"
                game_report = json.loads(game_path.read_text())
                game_report["declaration_sha256"] = declaration_hash
                game_path.write_text(json.dumps(game_report))
            rng = np.random.default_rng(20261104)
            for view in ("clean", "v2"):
                report["per_game"]["paired"][view] = {}
                report["summary"][view] = {}
                for metric in verify.GAR_METRICS:
                    vals = np.asarray([report["per_game"][view][g][metric] for g in games], dtype=float)
                    report["per_game"]["paired"][view][metric] = {
                        g: {"candidate": float(v), "saved_baseline": float(v), "delta": 0.}
                        for g, v in zip(games, vals)}
                    picks = rng.integers(0, len(vals), size=(2000, len(vals)))
                    boot = np.zeros(2000)
                    report["summary"][view][metric] = {
                        "candidate_equal_game_mean": float(vals.mean()),
                        "saved_baseline_equal_game_mean": float(vals.mean()), "delta": 0.,
                        "games_positive": 0, "games": 10,
                        "paired_game_bootstrap_95": [float(np.quantile(boot, .025)),
                                                     float(np.quantile(boot, .975))]}
            result_path = results / "v2_weight_edit_evaluation_v1.json"
            result_path.write_text(json.dumps(report))
            with mock.patch.object(verify, "ROOT", root), mock.patch.object(verify, "RESULTS", results):
                checked = verify.verify_gar_readout()
        self.assertTrue(checked["saved_gar_test_npz_metrics_replayed"])
        self.assertEqual(checked["gar_game_count"], 10)


if __name__ == "__main__":
    unittest.main()
