import copy
import random
import unittest

import numpy as np

from experiments import turnover_event_sampler as turnover
from experiments import turnover_supervision_train as train


def event_record(match="m", *, fps=25, edge=100, duration=400):
    y = np.zeros(duration, dtype=np.int8)
    y[edge + 1:] = 1
    period = np.zeros(duration, dtype=np.int8)  # zero is a valid period ID
    indices = np.arange(0, duration, 5, dtype=np.int64)
    sampled_y = y[indices]
    return {"match": match, "source": "metrica", "full_y": y,
            "full_period": period, "fps": float(fps), "native_fps": float(fps),
            "indices": indices, "y": sampled_y}


class TurnoverEventTests(unittest.TestCase):
    def test_requires_stable_known_ha_context_and_same_period(self):
        record = event_record()
        events = turnover.eligible_events(record)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["identity"], ("m", 0, 100))
        self.assertEqual(events[0]["left"], 0)
        self.assertEqual(events[0]["right"], 1)

        unknown = copy.deepcopy(record)
        unknown["full_y"][90] = -1
        self.assertEqual(turnover.eligible_events(unknown), [])
        dead = copy.deepcopy(record)
        dead["full_y"][110] = 2
        self.assertEqual(turnover.eligible_events(dead), [])
        period_break = copy.deepcopy(record)
        period_break["full_period"][101:] = 1
        self.assertEqual(turnover.eligible_events(period_break), [])

    def test_swap_invariance_and_any_surrounding_duration(self):
        record = event_record(duration=1200, edge=500)
        swapped = copy.deepcopy(record)
        swapped["full_y"] = np.where(np.isin(swapped["full_y"], (0, 1)),
                                    1 - swapped["full_y"], swapped["full_y"])
        swapped["y"] = np.where(np.isin(swapped["y"], (0, 1)),
                                1 - swapped["y"], swapped["y"])
        left, right = turnover.eligible_events(record), turnover.eligible_events(swapped)
        self.assertEqual(left[0]["identity"], right[0]["identity"])
        self.assertEqual((left[0]["left"], left[0]["right"]), (0, 1))
        self.assertEqual((right[0]["left"], right[0]["right"]), (1, 0))

    def test_full_stable_context_must_fit_candidate_core(self):
        record = event_record(edge=230)
        windows = [(0, start, 50, np.zeros(100), False) for start in (0, 10, 20)]
        catalog, physical = turnover.event_catalog([record], {0: windows})
        self.assertEqual(set(physical), {("m", 0, 230)})
        self.assertEqual(catalog[0][("m", 0, 230)], [windows[1], windows[2]])

    def test_hard_half_is_per_match_deterministic_and_lowest_signal(self):
        physical = {}
        for match in ("a", "b"):
            for edge, score in ((10, .1), (20, .2), (30, .3), (40, .4)):
                physical[(match, 0, edge)] = {"hardness": {
                    "pre_correct": score, "post_correct": .9}}
        selected = turnover.rank_hardest_half(physical)
        self.assertEqual(selected, {("a", 0, 10), ("a", 0, 20),
                                    ("b", 0, 10), ("b", 0, 20)})

    def test_declared_training_budget_and_safe_scope(self):
        self.assertEqual(train.ARMS, ("fresh_control", "uniform_turnover", "hard_turnover"))
        self.assertEqual((train.UPDATES, train.LR, train.BASE_STAGE), (256, 1e-4, 1024))
        old = (train.iterative.LR, train.iterative.FRACTION_GAR,
               train.iterative.STUDY, train.iterative.random)
        with train._training_scope():
            self.assertEqual(train.iterative.LR, 1e-4)
            self.assertEqual(train.iterative.FRACTION_GAR, .5)
        self.assertEqual((train.iterative.LR, train.iterative.FRACTION_GAR,
                          train.iterative.STUDY), old[:3])
        self.assertIs(train.iterative.random, old[3])


if __name__ == "__main__":
    unittest.main()
