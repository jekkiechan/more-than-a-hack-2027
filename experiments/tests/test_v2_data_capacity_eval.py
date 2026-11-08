import unittest
from unittest.mock import patch

import numpy as np

from experiments import v2_data_capacity_eval as evaluation
from experiments.v2_data_capacity_eval import equal_provider, hard_gate


class DataCapacityEvaluationAggregationTests(unittest.TestCase):
    def test_equal_provider_averages_replicates_then_matches(self):
        values = {
            ("p1", "m1", 1): 0.2,
            ("p1", "m1", 2): 0.4,
            ("p1", "m2", 1): 0.8,
            ("p2", "m3", 1): 0.6,
        }
        mean, providers = equal_provider(values)
        self.assertAlmostEqual(providers["p1"], 0.55)
        self.assertAlmostEqual(providers["p2"], 0.6)
        self.assertAlmostEqual(mean, 0.575)

    def test_hard_gate_has_zero_tolerance_and_pr82_v2_floor(self):
        adopted = {view: {"spell_f1": .6, "macro_f1": .8, "d_f1": .7,
                          "d_precision": .7, "d_recall": .7,
                          "short_spell_recall_iou_0_5_lt3s": .2}
                   for view in ("clean", "feed_like_v1", "feed_like_v2")}
        candidate = {view: dict(metrics) for view, metrics in adopted.items()}
        candidate["feed_like_v2"]["spell_f1"] = .5927491913
        self.assertTrue(hard_gate(candidate, adopted)["all_passed"] is False)
        candidate["feed_like_v2"]["spell_f1"] = .6
        self.assertTrue(hard_gate(candidate, adopted)["all_passed"])
        candidate["clean"]["d_f1"] -= 1e-12
        self.assertFalse(hard_gate(candidate, adopted)["all_passed"])

    def test_width_192_prediction_scopes_compatible_network(self):
        import torch
        cap = evaluation.training.recipe.rf.sn.cap
        frozen_network = cap.network
        record = {"names": [], "x": np.zeros((2, 4), dtype=np.float32),
                  "period": np.ones(2, dtype=np.int8)}
        native = {"ball": np.zeros((2, 2), dtype=np.float32)}
        def fake_predict(_record, _model, _teacher):
            network = cap.network(4, 192, 0.1)
            output = network(torch.zeros((1, 3, 4), dtype=torch.float32))
            self.assertEqual(tuple(output.shape), (1, 3, 2))
            return np.full((2, 3), 1 / 3, dtype=np.float32)
        with (patch.object(evaluation.training, "load_fit", return_value={}),
              patch.object(evaluation.training.recipe.rf.sc, "initial_path", return_value="teacher"),
              patch.object(evaluation.joblib, "load", return_value={}),
              patch.object(evaluation.old, "inference_record", return_value=record),
              patch.object(evaluation.sn, "predict", side_effect=fake_predict),
              patch.object(evaluation.old, "upsample", side_effect=lambda p, _r, _n: p),
              patch.object(evaluation.runtime.frozen.tc, "inputs_only", return_value=native)):
            output = evaluation._predict_new(native, record, "Sportec", "width192_capacity", 2026)
        self.assertEqual(output.shape, (2, 3))
        self.assertIs(cap.network, frozen_network)


if __name__ == "__main__":
    unittest.main()
