import unittest
from unittest.mock import patch

from experiments import v2_short_original_diversity as study


class RoundRobinViewTests(unittest.TestCase):
    def test_selects_four_v2_views_without_changing_other_records(self):
        seen = []

        def original(record, start, size, full, alternate, **kwargs):
            seen.append((record["x"], alternate, kwargs["swap"]))
            return record["x"]

        records = [
            {"x": "v1", "_v2_view_match": "m1",
             "_v2_feature_views": ["v2a", "v2b", "v2c", "v2d"]},
            {"x": "clean", "_v2_view_match": "m1"},
        ]
        with patch.object(study.rf.sn.cap, "_prepare_window", original):
            with study._round_robin_v2_windows(records) as counts:
                prepare = study.rf.sn.cap._prepare_window
                values = [prepare(records[0], 0, 1, None, False,
                                  median=None, scale=None, swap=False)
                          for _ in range(8)]
                clean = prepare(records[1], 0, 1, None, False,
                                median=None, scale=None, swap=True)

        self.assertEqual(values, ["v2a", "v2b", "v2c", "v2d"] * 2)
        self.assertEqual(clean, "clean")
        self.assertEqual([x[0] for x in seen], values + ["clean"])
        self.assertEqual(dict(counts), {"m1/r0": 2, "m1/r1": 2,
                                        "m1/r2": 2, "m1/r3": 2})


if __name__ == "__main__":
    unittest.main()
