"""Contracts for the fixed post-hoc short-team probability blend."""
import unittest

import numpy as np
import pandas as pd

from experiments import dense_short_team_blend_v1 as blend
from experiments.dense_error_audit_v1 import frame_table


class DenseShortTeamBlend(unittest.TestCase):
    def test_half_blend_is_symmetric_and_does_not_accept_targets(self):
        control = np.array([[.8, .1, .1], [.2, .7, .1]])
        candidate = np.array([[.4, .4, .2], [.6, .2, .2]])
        expected = (control + candidate) / 2
        np.testing.assert_array_equal(blend.blend_probabilities(control, candidate), expected)
        np.testing.assert_array_equal(blend.blend_probabilities(candidate, control), expected)
        # Labels are not an input to blending, so changing them cannot tune inference.
        labels_one = np.array(["H", "A"])
        labels_two = np.array(["D", "X"])
        self.assertFalse(np.array_equal(labels_one, labels_two))
        np.testing.assert_array_equal(blend.blend_probabilities(control, candidate), expected)

    def test_blend_rejects_mismatched_probability_arrays(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            blend.blend_probabilities(np.ones((2, 3)), np.ones((3, 3)))
        with self.assertRaisesRegex(ValueError, "matching"):
            blend.blend_probabilities(np.ones((2, 2)), np.ones((2, 2)))

    def test_synthetic_paired_archive_runs_blend_decode_and_official_score(self):
        labels = np.array(list("DDDDDHHHHHHHHHHDDDDDAAAAAAAAAADDDDDDDDDD"), dtype="U1")
        n = len(labels)
        classes = {"H": 0, "A": 1, "D": 2}
        probability = np.full((n, 3), .005, dtype=np.float64)
        probability[np.arange(n), np.array([classes[x] for x in labels])] = .99
        base = {"probabilities": probability, "labels": labels,
                "period": np.zeros(n, dtype=np.int8), "fps": np.array(10.),
                "timestamps": np.arange(n) / 10,
                "ball_observed": np.ones(n, dtype=bool),
                "source": np.array("metrica"), "match": np.array("synthetic"),
                "view": np.array("clean"), "replicate": np.array("clean"),
                "fold": np.array("leave_out_metrica"), "arm": np.array("control_D2"),
                "seed": np.array("ensemble")}
        candidate = {key: value.copy() for key, value in base.items()}
        candidate["arm"] = np.array("short_team_2x")
        candidate["seed"] = np.array("seedensemble")
        cell = {"source": "metrica", "match": "synthetic", "view": "clean",
                "replicate": "clean"}
        blended, prediction, row = blend.evaluate_cell(base, candidate, cell)
        np.testing.assert_array_equal(blended, probability)
        np.testing.assert_array_equal(prediction, labels)
        self.assertEqual(row["arm"], blend.ARM)
        self.assertAlmostEqual(row["spell_f1"], 1.)
        self.assertIn("complete_le_1s_spell_recall", row)

    def test_source_pair_checks_all_native_masks_and_timeline(self):
        cell = {"source": "metrica", "match": "m1", "view": "clean", "replicate": "clean"}
        base = {"labels": np.array(["H", "D"]), "period": np.array([1, 1]),
                "timestamps": np.array([0., .04]), "ball_observed": np.array([True, False]),
                "fps": np.array(25.), "source": np.array("metrica"), "match": np.array("m1"),
                "view": np.array("clean"), "replicate": np.array("clean"),
                "fold": np.array("leave_out_metrica"), "arm": np.array("control_D2"),
                "seed": np.array("ensemble")}
        candidate = {key: value.copy() for key, value in base.items()}
        candidate["arm"] = np.array("short_team_2x")
        candidate["seed"] = np.array("seedensemble")
        blend.validate_source_pair(base, candidate, cell)
        for field in ("labels", "period", "timestamps", "ball_observed", "fps"):
            changed = {key: value.copy() for key, value in candidate.items()}
            if field == "fps":
                changed[field] = np.array(24.)
            elif field == "labels":
                changed[field][0] = "A"
            elif field == "period":
                changed[field][1] = 2
            elif field == "timestamps":
                changed[field][1] += .01
            else:
                changed[field][1] = True
            with self.subTest(field=field), self.assertRaises(ValueError):
                blend.validate_source_pair(base, changed, cell)

    def test_blend_diagnostics_reuse_candidate_arm_and_keep_time_segments(self):
        gt = np.array(["H"] * 10, dtype="U1")
        pred = np.array(["H"] * 5 + ["A"] * 5, dtype="U1")
        period = np.array([0] * 5 + [1] * 5)
        timestamps = np.r_[np.arange(5) / 10, 1 + np.arange(5) / 10]
        frame = frame_table("metrica", "m1", gt, pred, np.ones(10, bool), period,
                            timestamps, 10.)
        frame["arm"], frame["view"], frame["replicate"] = blend.ARM, "clean", "clean"
        result = blend.aggregate_blend_diagnostics(frame)
        cell = result["clean"]["per_provider"]["metrica"]
        self.assertAlmostEqual(cell["wrong_team_rate"], .5)
        self.assertEqual(cell["wrong_team_frames"], 5)

    def test_short_spell_recall_is_separate_for_clean_and_stress(self):
        rows = []
        for provider in ("metrica", "skillcorner", "sportec"):
            rows.extend([
                {"source": provider, "view": "clean", "complete_le_1s_reference_spells": 4,
                 "complete_le_1s_matched_spells": 2, "short_reference_spells": 8,
                 "short_matched_spells": 6},
                {"source": provider, "view": "feed_like", "complete_le_1s_reference_spells": 2,
                 "complete_le_1s_matched_spells": 2, "short_reference_spells": 4,
                 "short_matched_spells": 2},
            ])
        result = blend.aggregate_short_spell_readout(rows)
        self.assertEqual(result["clean"]["equal_provider_mean"]["complete_le_1s_recall"], .5)
        self.assertEqual(result["clean"]["equal_provider_mean"]["under_3s_recall"], .75)
        self.assertEqual(result["feed_like"]["equal_provider_mean"]["complete_le_1s_recall"], 1.)
        self.assertEqual(result["feed_like"]["equal_provider_mean"]["under_3s_recall"], .5)


if __name__ == "__main__":
    unittest.main()
