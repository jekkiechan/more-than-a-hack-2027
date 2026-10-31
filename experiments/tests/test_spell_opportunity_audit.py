"""Contracts for aggregate, official-scorer spell opportunity diagnostics."""
import unittest

import numpy as np

from experiments.spell_opportunity_audit import (
    COHORT,
    _balanced_summary,
    _frame_errors,
    _unmatched_spell_geometry,
)


class SpellOpportunityAuditContracts(unittest.TestCase):
    @staticmethod
    def inputs_for(labels):
        labels = np.asarray(list(labels))
        return np.ones(len(labels), dtype=int), np.arange(len(labels), dtype=float) / 25, 25.0

    def test_frame_error_classes_are_disjoint_and_missed_d_is_separate(self):
        labels = np.array(["H", "A", "H", "D", "D", "D"])
        prediction = np.array(["A", "D", "X", "H", "X", "D"])
        result = _frame_errors(labels, prediction)
        self.assertEqual(result["live"]["wrong_team"], 1)
        self.assertEqual(result["live"]["false_D"], 1)
        self.assertEqual(result["live"]["abstained"], 1)
        self.assertEqual(result["dead"]["missed_D"], 2)
        self.assertEqual(result["dead"]["missed_to_H"], 1)
        self.assertEqual(result["dead"]["missed_to_X"], 1)

    def test_unmatched_wrong_team_spell_uses_both_team_overlap_geometry(self):
        labels, prediction = list("H" * 20), list("A" * 20)
        period, timestamps, fps = self.inputs_for(labels)
        result = _unmatched_spell_geometry(labels, prediction, period, timestamps, fps)
        self.assertEqual(result["unmatched_reference_spells_at_iou_0_5"], 1)
        self.assertEqual(result["unmatched_with_other_team_overlap"], 1)
        self.assertEqual(result["unmatched_without_same_team_alive_overlap"], 1)
        self.assertEqual(result["unmatched_dominant_frame_error_within_reference_span"]["wrong_team"], 1)

    def test_unmatched_false_dead_spell_is_visible_as_a_distinct_mode(self):
        labels, prediction = list("H" * 20), list("D" * 20)
        period, timestamps, fps = self.inputs_for(labels)
        result = _unmatched_spell_geometry(labels, prediction, period, timestamps, fps)
        self.assertEqual(result["unmatched_reference_spells_at_iou_0_5"], 1)
        self.assertEqual(result["unmatched_dominant_frame_error_within_reference_span"]["false_D"], 1)

    def test_summary_weights_matches_within_provider_then_providers_equally(self):
        per_match = {
            "J03WOY": [0.9],
            "metrica_3": [0.6],
            "skillcorner_had_1996435": [0.3],
            "skillcorner_had_1996436": [0.4],
            "skillcorner_had_2006229": [0.5],
        }
        rows = []
        for source, matches in COHORT.items():
            for match in matches:
                score = per_match[match][0]
                rows.append({
                    "source": source,
                    "match": match,
                    "view": "clean",
                    "score": {
                        "spell_f1": score,
                        "macro_f1": score,
                        "d_f1": score,
                        "d_precision": score,
                        "d_recall": score,
                        "short_spell_recall_under_3s_at_iou_0_5": score,
                    },
                    "gap_contact": {"frames": {}, "spells": {}},
                })
        result = _balanced_summary(rows, "clean")["equal_provider_mean"]
        # Provider means: Sportec .9, Metrica .6, SkillCorner .4.
        self.assertAlmostEqual(result["spell_f1"], (0.9 + 0.6 + 0.4) / 3)

    def test_feed_replicates_are_averaged_inside_their_match(self):
        match_values = {
            "J03WOY": [0.2, 0.6],
            "metrica_3": [0.6, 0.6],
            "skillcorner_had_1996435": [0.3, 0.5],
            "skillcorner_had_1996436": [0.5, 0.7],
            "skillcorner_had_2006229": [0.7, 0.9],
        }
        rows = []
        for source, matches in COHORT.items():
            for match in matches:
                for replicate, score in zip(("corrupt_1", "corrupt_2"), match_values[match]):
                    rows.append({
                        "source": source,
                        "match": match,
                        "view": "feed_like",
                        "replicate": replicate,
                        "score": {
                            "spell_f1": score,
                            "macro_f1": score,
                            "d_f1": score,
                            "d_precision": score,
                            "d_recall": score,
                            "short_spell_recall_under_3s_at_iou_0_5": score,
                        },
                        "gap_contact": {"frames": {}, "spells": {}},
                    })
        result = _balanced_summary(rows, "feed_like")["equal_provider_mean"]
        # Match means: Sportec .4, Metrica .6, SkillCorner .4/.6/.8 -> .6.
        self.assertAlmostEqual(result["spell_f1"], (0.4 + 0.6 + 0.6) / 3)


if __name__ == "__main__":
    unittest.main()
