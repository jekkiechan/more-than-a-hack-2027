import unittest

import numpy as np

from experiments.soccernet_anchor_balance import build_anchor_weights


def segment(game, segment_id, events, length=30):
    y = np.full(length, -1, dtype=np.int8)
    mapping = []
    for index, action, team, sample_id, status in events:
        if status == "mapped":
            y[index] = 0 if team == "home" else 1
        mapping.append({"sampled_index": index, "action": action, "team": team,
                        "sample_id": sample_id, "status": status})
    return {"y": y, "source": "soccernet", "match": f"soccernet_{game}",
            "segment_id": segment_id, "anchor_frame_mapping": mapping}


class SoccerNetAnchorBalanceTests(unittest.TestCase):
    def test_inverse_fixture_event_frequency_balances_cells(self):
        records = [
            segment("1", "a", [(0, "PASS", "home", "p1", "mapped"),
                                (1, "PASS", "away", "p2", "mapped"),
                                (2, "PASS", "home", "p3", "mapped"),
                                (3, "CROSS", "away", "c1", "mapped")]),
            segment("2", "b", [(0, "PASS", "away", "p4", "mapped"),
                                (1, "SHOT", "home", "s1", "mapped")]),
        ]
        weights, provenance = build_anchor_weights(records, {"1", "2"})
        totals = {
            "1/PASS": float(weights[0][:3].sum()),
            "1/CROSS": float(weights[0][3]),
            "2/PASS": float(weights[1][0]),
            "2/SHOT": float(weights[1][1]),
        }
        # Within each fixture, present event types have equal total weight;
        # each fixture's total mass remains its anchor count.
        self.assertAlmostEqual(totals["1/PASS"], totals["1/CROSS"], places=6)
        self.assertAlmostEqual(totals["2/PASS"], totals["2/SHOT"], places=6)
        self.assertAlmostEqual(totals["1/PASS"] + totals["1/CROSS"], 4.0, places=6)
        self.assertAlmostEqual(totals["2/PASS"] + totals["2/SHOT"], 2.0, places=6)
        positive = np.concatenate(weights)
        self.assertAlmostEqual(positive[positive > 0].mean(), 1.0, places=6)
        self.assertEqual(provenance["anchor_count"], 6)
        self.assertEqual(provenance["counts_by_fixture_event"], {
            "1/CROSS": 1, "1/PASS": 3, "2/PASS": 1, "2/SHOT": 1,
        })
        self.assertEqual(len(provenance["anchor_index_sha256"]), 64)
        self.assertTrue(provenance["team_symmetric"])
        self.assertTrue(np.all(weights[0][4:] == 0))

    def test_only_mapped_valid_anchor_rows_receive_weight(self):
        record = segment("1", "a", [
            (0, "PASS", "home", "ok", "mapped"),
            (1, "PASS", "home", "unknown", "center_unobserved"),
            (2, "CROSS", "away", "conflict", "grid_conflict"),
            (3, "SHOT", "away", "target-bad", "mapped"),
        ])
        record["y"][3] = 0  # sidecar says away, target says home
        with self.assertRaisesRegex(ValueError, "conflicts with the segment target"):
            build_anchor_weights([record], {"1"})

    def test_duplicate_sample_and_same_row_action_averaging(self):
        duplicated = segment("1", "a", [(0, "PASS", "home", "same", "mapped")])
        duplicate_segment = segment("1", "b", [(0, "PASS", "home", "same", "mapped")])
        _, duplicate_provenance = build_anchor_weights([duplicated, duplicate_segment], {"1"})
        self.assertEqual(duplicate_provenance["masked_counts"]["missing_or_duplicate_anchor_identity"], 1)

        events = [(0, "PASS", "home", "pass0", "mapped"),
                  (0, "SHOT", "home", "shot0", "mapped")]
        events.extend((index, "PASS", "away", f"pass{index}", "mapped")
                      for index in range(1, 21))
        record = segment("1", "crowded", events)
        weights, provenance = build_anchor_weights([record], {"1"})
        self.assertLessEqual(float(weights[0].max()), 5.0)
        self.assertGreater(float(weights[0][0]), 0)
        self.assertEqual(provenance["anchor_count"], 22)
        self.assertLess(float(weights[0].mean()), 1.0)  # cap applied after mean normalization

    def test_rejects_non_train_fixture_and_dense_or_dead_targets(self):
        with self.assertRaisesRegex(ValueError, "outside whole-fixture TRAIN"):
            build_anchor_weights([segment("2", "a", [(0, "PASS", "home", "p", "mapped")])],
                                 {"1"})
        dense = segment("1", "a", [(0, "PASS", "home", "p", "mapped")])
        dense["y"][2] = 1
        with self.assertRaisesRegex(ValueError, "sparse targets"):
            build_anchor_weights([dense], {"1"})
        dead = segment("1", "a", [(0, "PASS", "home", "p", "mapped")])
        dead["y"][2] = 2
        with self.assertRaisesRegex(ValueError, "remain sparse"):
            build_anchor_weights([dead], {"1"})

        fractional = segment("1", "a", [(0, "PASS", "home", "p", "mapped")])
        fractional["anchor_frame_mapping"][0]["sampled_index"] = 0.5
        with self.assertRaisesRegex(ValueError, "sampled_index must be an integer"):
            build_anchor_weights([fractional], {"1"})

    def test_team_swap_preserves_event_weights(self):
        original = segment("1", "a", [(0, "PASS", "home", "p1", "mapped"),
                                        (1, "CROSS", "away", "p2", "mapped")])
        swapped = segment("1", "a", [(0, "PASS", "away", "p1", "mapped"),
                                      (1, "CROSS", "home", "p2", "mapped")])
        left, _ = build_anchor_weights([original], {"1"})
        right, _ = build_anchor_weights([swapped], {"1"})
        np.testing.assert_array_equal(left[0], right[0])


if __name__ == "__main__":
    unittest.main()
