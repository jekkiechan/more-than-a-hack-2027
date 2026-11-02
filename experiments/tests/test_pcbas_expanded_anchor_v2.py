"""Contracts for the expanded, label-separated PCBAS anchor evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import pcbas_expanded_anchor_v2 as study
from experiments import pcbas_event_anchor_v1 as v1


def feature_rows(frames=(10, 11, 14)):
    rows = []
    for frame in frames:
        for player, team in ((100, 0), (200, 1)):
            rows.append([frame, player, team, 7, 1, .25 + team * .5, .5,
                         0, 0, 10, 20, 30, 40])
    return np.asarray(rows, dtype=float)


class CohortContracts(unittest.TestCase):
    def test_hash_rank_is_stable_order_independent_and_deduplicated(self):
        keys = [f"game_{i}_H{half}" for i in range(60, 108) for half in (1, 2)]
        selected_a, reserve_a = study.cohort_from_keys(keys)
        selected_b, reserve_b = study.cohort_from_keys(reversed(keys))
        self.assertEqual(selected_a, selected_b)
        self.assertEqual(reserve_a, reserve_b)
        self.assertEqual(len(selected_a), 12)
        self.assertEqual(len({row["game_id"] for row in selected_a}), 12)
        self.assertEqual(selected_a, sorted(selected_a, key=lambda row: (row["rank_sha256"], row["game_id"])))
        self.assertTrue(all(len(row["rank_sha256"]) == 64 for row in selected_a + reserve_a))

    def test_only_complete_games_are_ranked(self):
        keys = [f"game_{i}_H{half}" for i in range(1, 53) for half in (1, 2)]
        keys.remove("game_4_H2")
        selected, reserve = study.cohort_from_keys(keys)
        self.assertNotIn("game_4", {row["game_id"] for row in selected + reserve})

    def test_val_game_ids_are_excluded_from_train_cohort(self):
        val_keys = [f"game_{i}_H{half}" for i in (18, 24, 47) for half in (1, 2)]
        train_keys = [f"game_{i}_H{half}" for i in range(60, 108) for half in (1, 2)]
        selected, reserve = study.cohort_from_keys(val_keys + train_keys)
        selected_ids = {row["game_id"] for row in selected}
        reserve_ids = {row["game_id"] for row in reserve}
        self.assertTrue(selected_ids.isdisjoint({"game_18", "game_24", "game_47"}))
        self.assertTrue(reserve_ids.isdisjoint({"game_18", "game_24", "game_47"}))


class InferenceSeparationContracts(unittest.TestCase):
    def test_half_reader_requests_feature_columns_only(self):
        class Dataset:
            def __init__(self, rows):
                self.rows = rows
                self.key = None
                self.index = None

            def __getitem__(self, key):
                self.key = key
                return self.rows[key]

        dataset = Dataset(feature_rows())
        inputs, frames, audit, feature_hash = study._load_half_features(
            {"game_60_H1": dataset}, "game_60_H1", study.array_sha(feature_rows()))
        self.assertEqual(dataset.key, (slice(None), slice(None, 13)))
        np.testing.assert_array_equal(frames, [10, 11, 14])
        self.assertFalse(np.isfinite(inputs["ball"]).any())
        self.assertFalse(inputs["ball_observed"].any())
        self.assertEqual(audit["source_rows"], 6)
        self.assertEqual(feature_hash, study.array_sha(feature_rows()))

    def test_anchor_targets_do_not_change_features_or_hash_rank(self):
        data = feature_rows()
        original, _, _ = v1.adapt_rows(data, 1)
        # Simulate an appended event/target field; the v2 reader's [:, :13]
        # contract prevents this field from reaching feature adaptation.
        with_targets_a = np.column_stack((data, np.zeros(len(data))))
        with_targets_b = with_targets_a.copy()
        with_targets_b[:, 13] = 987654
        feature_hash = study.array_sha(data)
        got_a, _, _, _ = study._load_half_features(
            {"game_60_H1": _ArrayDataset(with_targets_a)}, "game_60_H1", feature_hash)
        got_b, _, _, _ = study._load_half_features(
            {"game_60_H1": _ArrayDataset(with_targets_b)}, "game_60_H1", feature_hash)
        for key in original:
            np.testing.assert_array_equal(got_a[key], original[key])
            np.testing.assert_array_equal(got_b[key], original[key])
        self.assertEqual(study.rank_games(["game_60"]), study.rank_games(["game_60"]))


class _ArrayDataset:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, key):
        return self.rows[key]


class MetricContracts(unittest.TestCase):
    def test_d_and_x_are_incorrect_at_anchors(self):
        events = [[1, 0, 7, 1, 1, 0], [2, 1, 8, 1, 1, 0],
                  [3, 0, 7, 1, 1, 0], [4, 1, 8, 1, 1, 0]]
        anchors, _ = v1.select_anchors(events, [1, 2, 3, 4])
        measured = v1.anchor_counts(anchors, list("HADX"))["drive"]
        self.assertEqual((measured["n"], measured["correct"]), (4, 2))
        self.assertEqual(measured["predicted_state_counts"]["D"], 1)
        self.assertEqual(measured["predicted_state_counts"]["X"], 1)

    def test_equal_game_summary_is_not_pooled_anchor_accuracy(self):
        def cell(n, correct):
            return {name: dict(n=n, correct=correct, predicted_state_counts={"H": n},
                               target_direction_team_counts={"0": n})
                    for name in v1.CLASSES.values()}

        summary = v1.summarize({"game_1_H1": cell(1, 1), "game_1_H2": cell(9, 0),
                                "game_2_H1": cell(1, 1)})["drive"]
        self.assertAlmostEqual(summary["equal_game_team_agreement"], .55)
        self.assertAlmostEqual(summary["pooled_team_agreement"], 2 / 11)
        self.assertNotAlmostEqual(summary["equal_game_team_agreement"], summary["pooled_team_agreement"])

    def test_visibility_groups_partition_anchors_and_report_d_penalty(self):
        anchors = [dict(index=i, team=0 if i != 2 else 1) for i in range(3)]
        result = study.visibility_summary(anchors, list("HDA"), [3, 8, 14])
        self.assertEqual(set(result), {"0-5", "6-11", "12+"})
        self.assertEqual([result[k]["n"] for k in ("0-5", "6-11", "12+")], [1, 1, 1])
        self.assertEqual([result[k]["accuracy"] for k in ("0-5", "6-11", "12+")], [1.0, 0.0, 1.0])
        self.assertEqual(result["6-11"]["predicted_d_rate"], 1.0)

    def test_bootstrap_resamples_equal_weight_games_not_anchor_counts(self):
        rows = []
        for i in range(study.N_GAMES):
            acc = i / (study.N_GAMES - 1)
            rows.append(dict(game_id=f"g{i}", drive_accuracy=acc,
                             majority_direction_accuracy=.5, drive_n=1 if i == 0 else 10000))
        got = study.bootstrap_game_intervals(rows)
        rng = np.random.default_rng(study.BOOTSTRAP_SEED)
        values = np.asarray([r["drive_accuracy"] for r in rows])
        indices = rng.integers(0, len(rows), size=(study.BOOTSTRAPS, len(rows)))
        expected = np.quantile(values[indices].mean(axis=1), [.025, .975]).tolist()
        np.testing.assert_allclose(got["equal_game_drive_accuracy_95_interval"], expected)
        self.assertEqual(got["replicates"], 10_000)
        self.assertEqual(got["game_count"], 12)
        self.assertTrue(got["no_frame_or_anchor_bootstrap"])
        self.assertAlmostEqual(got["equal_game_drive_accuracy"], .5)


class IntegrityAndResumeContracts(unittest.TestCase):
    def test_declaration_verification_detects_changed_source_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_file = root / "runner.py"
            input_file = root / "source.h5"
            manifest_file = root / "manifest.json"
            model_file = root / "model.bin"
            for path, content in ((source_file, b"runner"), (input_file, b"data"),
                                  (manifest_file, b"manifest"), (model_file, b"model")):
                path.write_bytes(content)
            manifest = dict(package_version=study.PACKAGE, model_directory=".",
                            models_sha256={"model.bin": study.sha(model_file)})
            decl = dict(input_sha256={"source.h5": study.sha(input_file)},
                        source_sha256={"runner.py": study.sha(source_file)},
                        baseline_manifest_sha256=study.sha(manifest_file),
                        model_sha256=manifest["models_sha256"],
                        model_paths_sha256=manifest["models_sha256"])
            declaration_file = root / "declaration.json"
            declaration_file.write_text(json.dumps(decl))
            with mock.patch.multiple(study, ROOT=root, DECLARATION=declaration_file,
                                     MANIFEST=manifest_file), \
                 mock.patch.object(study, "_load_runtime", return_value=(manifest, {})):
                study.verify_declaration()
                source_file.write_bytes(b"changed runner")
                with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                    study.verify_declaration()
                source_file.write_bytes(b"runner")
                model_file.write_bytes(b"changed model")
                with self.assertRaisesRegex(ValueError, "checkpoint files changed"):
                    study.verify_declaration()

    def test_existing_half_is_resumable_only_when_receipt_matches_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            output = out / "game_60_H1.npz"
            output.write_bytes(b"sealed prediction")
            receipt = dict(declaration_sha256="decl", half="game_60_H1",
                           output_sha256=study.sha(output), model_sha256={"model": "sha"},
                           target_free_feature_rows_sha256="feature",
                           target_events_read=False, input_action_class_read=False)
            (out / "game_60_H1.receipt.json").write_text(json.dumps(receipt))
            control = out / "parity_control.npz"
            control.write_bytes(b"control")
            (out / "parity_receipt.json").write_text(json.dumps({
                "declaration_sha256": "decl", "exact_equal": True,
                "control_sha256": study.sha(control)}))
            declaration = {"cohort": {"halves": ["game_60_H1"]},
                           "selected_half_feature_sha256": {"game_60_H1": "feature"},
                           "model_sha256": {"model": "sha"}}
            with mock.patch.multiple(study, OUT=out), \
                 mock.patch.object(study, "verify_declaration", return_value=declaration), \
                 mock.patch.object(study, "sha", side_effect=lambda p: "decl" if p == study.DECLARATION else v1.sha(p)):
                study.predict("game_60_H1")
                output.write_bytes(b"tampered prediction")
                with self.assertRaisesRegex(ValueError, "receipt does not verify"):
                    study.predict("game_60_H1")

    def test_score_gate_rejects_unsealed_outputs_before_reading_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.multiple(study, OUT=Path(tmp)), \
                 mock.patch.object(study, "verify_declaration", return_value={}), \
                 mock.patch.object(study, "EVENTS", Path(tmp) / "targets-must-not-be-read.json"):
                with self.assertRaisesRegex(ValueError, "not sealed"):
                    study.score()

    def test_seal_rejects_missing_half_and_preserves_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "parity_control.npz").write_bytes(b"control")
            (out / "parity_receipt.json").write_text(json.dumps({
                "declaration_sha256": "decl", "exact_equal": True,
                "control_sha256": "decl"}))
            declaration = {"cohort": {"halves": ["game_60_H1"]}}
            with mock.patch.multiple(study, OUT=out), \
                 mock.patch.object(study, "verify_declaration", return_value=declaration), \
                 mock.patch.object(study, "sha", return_value="decl"):
                with self.assertRaisesRegex(ValueError, "Prediction incomplete"):
                    study.seal()
            self.assertFalse((out / "prediction_seal.json").exists())


if __name__ == "__main__":
    unittest.main()
