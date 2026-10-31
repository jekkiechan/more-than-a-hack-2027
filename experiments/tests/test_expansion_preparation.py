"""Synthetic preparation and receipt contracts; no fitting or real match data."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import warnings

import joblib
import numpy as np

from experiments import (expansion_preparation as ep, learned_temporal as lt,
                         preparation as prep, supervised, tracking_corruption as tc)


def synthetic_match(n: int = 240, labels: str = "HADX") -> dict:
    times = np.arange(n, dtype=float) / 10.0
    times[n // 2:] += 48.0
    teams = np.repeat([0, 1], 11)
    phase = np.arange(n, dtype=float) / 10.0
    players = np.empty((n, 22, 2), dtype=float)
    for slot in range(22):
        players[:, slot, 0] = (20.0 if slot < 11 else 80.0) + slot * .15 + phase * .03
        players[:, slot, 1] = 8.0 + (slot % 11) * 4.0 + np.sin(phase * .1 + slot) * .1
    ball = players[:, 0].copy()
    ball[30:45] = np.nan
    period = np.ones(n, dtype=np.int8)
    period[n // 2:] = 2
    player_observed = np.ones((n, 22), dtype=bool)
    # Finite but unobserved coordinates must be masked and graded as X before aug.
    player_observed[80:90, :11] = False
    raw_labels = np.resize(np.asarray(list(labels), dtype="U1"), n)
    return dict(players=players, ball=ball, teams=teams, period=period,
                timestamps=times, fps=np.asarray(10.0), pitch=np.asarray([105., 68.]),
                player_observed=player_observed,
                ball_observed=np.isfinite(ball).all(axis=1), labels=raw_labels)


class ExpansionPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data_dir = self.root / "data"
        self.external = self.data_dir / "external" / "processed"
        self.external.mkdir(parents=True)
        self.out = self.root / "cache"
        self.calibration = dict(
            ball_missing_fraction=.41,
            gap_edges_seconds=[.08, .2, .5, 1., 2., 5., 10., 30.],
            gap_bin_probability=[1/7] * 7,
        )
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def save_item(self, match="skillcorner_had_2006363", data=None, role="train"):
        data = synthetic_match() if data is None else data
        path = self.external / f"{match}.npz"
        np.savez_compressed(path, **data)
        return dict(source="skillcorner", match=match, group_id=match,
                    path=f"data/external/processed/{match}.npz",
                    sha256=prep.sha256(path), role=role)

    def test_training_cache_keeps_pre_aug_grading_x_and_time_blocks(self):
        item = self.save_item()
        cache_path, receipt_path = ep.prepare_training(
            item, 8, self.out, self.calibration, self.data_dir)
        receipt = json.loads(receipt_path.read_text())
        ep.verify_cache(cache_path, receipt_path, receipt["identity"],
                        receipt["preparation_sha256"])
        payload = joblib.load(cache_path)

        inputs, labels = __import__("experiments.multisource_eval", fromlist=["view"]).view(
            prep.data_path(item, self.data_dir))
        expected_y = ep._encode_labels(labels)
        expected_period = supervised.model_periods(inputs)
        np.testing.assert_array_equal(payload["full_y"], expected_y)
        np.testing.assert_array_equal(payload["full_period"], expected_period)
        self.assertTrue(np.all(payload["full_y"][80:90] == -1))
        self.assertEqual(set(payload["records"]),
                         {"clean", "gap", "dropout_legacy", "dropout_gap"})

        clean = payload["records"]["clean"]
        self.assertEqual(clean["x"].shape[1], len(clean["names"]))
        for name, record in payload["records"].items():
            np.testing.assert_array_equal(record["indices"], clean["indices"])
            np.testing.assert_array_equal(record["period"], expected_period[record["indices"]])
            np.testing.assert_array_equal(record["y"], expected_y[record["indices"]])
            self.assertTrue(np.any(record["y"] == -1))
            self.assertEqual(record["source"], item["source"])
            self.assertEqual(record["match"], item["match"])
            for start, size, full in lt.chunks(record):
                block = record["period"][full]
                self.assertEqual(len(np.unique(block)), 1)
                if size > 1:
                    self.assertTrue(np.all(np.diff(record["indices"][start:start+size]) > 0))
        # The augmented player/ball views may change features, never pre-aug labels.
        self.assertFalse(np.array_equal(clean["x"], payload["records"]["dropout_legacy"]["x"],
                                        equal_nan=True))

    def test_features_are_target_independent_and_feedlike_preserves_labels(self):
        a = synthetic_match(labels="H")
        b = synthetic_match(labels="D")
        item_a = self.save_item("skillcorner_a", a, role="selection")
        item_b = self.save_item("skillcorner_b", b, role="selection")
        inputs_a, labels_a = __import__("experiments.multisource_eval", fromlist=["view"]).view(
            prep.data_path(item_a, self.data_dir))
        inputs_b, labels_b = __import__("experiments.multisource_eval", fromlist=["view"]).view(
            prep.data_path(item_b, self.data_dir))
        rec_a = prep.build_record(inputs_a, labels_a, "skillcorner", item_a["match"])
        rec_b = prep.build_record(inputs_b, labels_b, "skillcorner", item_b["match"])
        np.testing.assert_array_equal(rec_a["x"], rec_b["x"])
        self.assertFalse(np.array_equal(rec_a["y"], rec_b["y"]))

        clean_path, _ = ep.prepare_eval(item_a, "clean", "clean", self.out,
                                        None, self.data_dir)
        feed_path, _ = ep.prepare_eval(item_a, "feed_like", "corrupt_1", self.out,
                                       self.calibration, self.data_dir)
        clean, feed = joblib.load(clean_path), joblib.load(feed_path)
        self.assertNotIn("labels", clean["inputs"])
        self.assertNotIn("labels", feed["inputs"])
        np.testing.assert_array_equal(clean["labels"], feed["labels"])
        np.testing.assert_array_equal(clean["record"]["y"], feed["record"]["y"])
        self.assertFalse(np.array_equal(clean["record"]["x"], feed["record"]["x"],
                                        equal_nan=True))

    def test_source_index_is_fixed_and_cache_reuse_is_immutable(self):
        item = self.save_item()
        for invalid in (7, 14, True, 8.0):
            with self.assertRaises(ValueError):
                ep.prepare_training(item, invalid, self.out, self.calibration, self.data_dir)
        first = ep.prepare_training(item, 8, self.out, self.calibration, self.data_dir)
        cache_before = first[0].read_bytes()
        receipt_before = first[1].read_bytes()
        second = ep.prepare_training(item, 8, self.out, self.calibration, self.data_dir)
        self.assertEqual(first, second)
        self.assertEqual(cache_before, second[0].read_bytes())
        self.assertEqual(receipt_before, second[1].read_bytes())

        with first[0].open("ab") as stream:
            stream.write(b"tamper")
        with self.assertRaisesRegex(ValueError, "receipt mismatch"):
            ep.prepare_training(item, 8, self.out, self.calibration, self.data_dir)

        wrong_item = dict(item, match="skillcorner_had_2007448",
                          group_id="skillcorner_had_2007448",
                          path="data/external/processed/skillcorner_had_2007448.npz")
        with self.assertRaisesRegex(ValueError, "frozen six-match expansion order"):
            ep.prepare_training(wrong_item, 8, self.out, self.calibration, self.data_dir)

    def test_input_tampering_and_invalid_calibration_are_rejected(self):
        item = self.save_item()
        path = prep.data_path(item, self.data_dir)
        with path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Input checksum"):
            ep.prepare_training(item, 8, self.out, self.calibration, self.data_dir)
        with self.assertRaisesRegex(ValueError, "required fields"):
            ep.prepare_eval(item, "feed_like", "corrupt_1", self.out,
                            {"ball_missing_fraction": .4}, self.data_dir)


if __name__ == "__main__":
    unittest.main()
