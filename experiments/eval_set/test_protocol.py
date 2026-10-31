"""Synthetic evaluation contracts; no external assets or organizer labels."""
import copy
import unittest

import numpy as np

from experiments.eval_set.protocol import paired_source_bootstrap, summarize_d_boundaries, validate_manifest
from experiments.metrics import evaluate_tracking
from evaluation import possession_metrics as pm


class ProtocolTests(unittest.TestCase):
    def test_official_spell_metric_is_unchanged(self):
        labels = np.array(list("HHHDDDAAHH"))
        pred = np.array(list("HHDDDDAAAA"))
        direct = pm.evaluate(pm.label_codes(labels), pm.label_codes(pred))
        adapter = evaluate_tracking(labels, pred)
        self.assertEqual(adapter["spell_f1_mean"], direct["spell_f1_mean"])
        self.assertEqual(adapter["spell_f1"], direct["spell_f1"])

    def test_team_switch_does_not_make_dead_boundary(self):
        result = summarize_d_boundaries(list("HHAAAADDDDHH"), list("AAHHHHDDDDAA"),
                                        np.ones(12), 2)
        self.assertEqual(result["0.5"]["n_gt"], 2)
        self.assertEqual(result["0.5"]["n_pred"], 2)
        self.assertEqual(result["0.5"]["f1"], 1)
        self.assertEqual(result["0.5"]["mean_abs_latency_seconds"], 0)

    def test_dead_boundary_lag_in_seconds_and_one_to_one(self):
        result = summarize_d_boundaries(list("HHHHDDDDHHHH"), list("HHHHHDDDDHHH"),
                                        np.ones(12), 4)
        self.assertEqual(result["0.5"]["n_matched"], 2)
        self.assertEqual(result["0.5"]["mean_latency_seconds"], .25)
        self.assertNotIn("pairs", result["0.5"])

    def test_half_time_gaps_and_ungraded_regions_do_not_make_boundaries(self):
        truth = list("HHDDXXHHDD")
        periods = [1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        timestamps = [0, .1, .2, .3, .4, .5, .6, .7, 5, 5.1]
        result = summarize_d_boundaries(truth, truth, periods, 10, timestamps)
        self.assertEqual(result["0.5"]["n_gt"], 0)
        self.assertIsNone(result["0.5"]["mean_abs_latency_seconds"])

    def test_source_balanced_paired_bootstrap(self):
        baseline = [{"eval_source": "large", "match": str(i), "spell_f1": .2} for i in range(3)]
        baseline.append({"eval_source": "small", "match": "s", "spell_f1": .2})
        candidate = [dict(row, spell_f1=.4 if row["eval_source"] == "large" else .2) for row in baseline]
        result = paired_source_bootstrap(baseline, candidate, repeats=100)
        self.assertAlmostEqual(result["candidate_minus_baseline"], .1)
        self.assertEqual(result["matches_per_source"], {"large": 3, "small": 1})
        with self.assertRaisesRegex(ValueError, "identical"):
            paired_source_bootstrap(baseline, candidate[:-1])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_source_bootstrap(baseline + baseline[:1], candidate)

    def test_split_guard_rejects_leakage_and_fake_confirmation(self):
        base = {"sources": {"a": {"train": [{"group_id": "a1"}], "selection": [{"group_id": "a2"}]},
                            "b": {"train": [{"group_id": "b1"}], "selection": [{"group_id": "b2"}]}},
                "folds": [{"name": "leave_out_b", "train_sources": ["a"], "selection_sources": ["b"]}],
                "confirmation": {"matches": []}}
        self.assertTrue(validate_manifest(base))
        leaked = copy.deepcopy(base)
        leaked["sources"]["a"]["selection"][0]["group_id"] = "a1"
        with self.assertRaisesRegex(ValueError, "both"):
            validate_manifest(leaked)
        leaked = copy.deepcopy(base)
        leaked["folds"][0]["train_sources"].append("b")
        with self.assertRaisesRegex(ValueError, "target source"):
            validate_manifest(leaked)
        leaked = copy.deepcopy(base)
        leaked["confirmation"]["matches"].append("old_match")
        with self.assertRaisesRegex(ValueError, "Fresh confirmation"):
            validate_manifest(leaked)


if __name__ == "__main__":
    unittest.main()
