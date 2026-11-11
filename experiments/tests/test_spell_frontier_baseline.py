import unittest
from unittest import mock

import numpy as np

from experiments import spell_frontier_baseline as inference


class FrontierDeploymentContracts(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(ball=np.zeros((50, 2)), period=np.ones(50),
                           timestamps=np.arange(50) / 25, fps=25.)
        self.manifest = dict(architecture=dict(kind=inference.KIND,
            recipe=dict(inference.RECIPE), phase_order="seeds_then_phases"))
        self.models = {f"{role}_{seed}": {"role": role, "seed": seed}
                       for role in inference.ROLES for seed in inference.SEEDS}

    def test_paired_seed_then_phase_pooling_and_target_independence(self):
        records = [dict(names=["feature"], x=np.full((50, 1), p), indices=np.arange(50), native_n=50, fps=25.,
                        period=np.ones(50), labels=np.full(50, "H"), phase=p)
                   for p in range(5)]
        arrays = {}
        def predict(record, model, teacher):
            self.assertNotIn("labels", record)
            expected = "teacher_calibrated" if model["role"] == "calibrated" else "teacher_adopted"
            self.assertEqual(teacher["role"], expected)
            self.assertEqual(teacher["seed"], model["seed"])
            phase = int(record["x"][0, 0])
            x = .1 + .03 * phase + .01 * (model["seed"] - 2026)
            value = np.tile([x, .8 - x, .2], (50, 1)).astype(np.float32)
            arrays.setdefault(model["role"], {}).setdefault(phase, []).append(value)
            return value
        with mock.patch.object(inference.sn, "predict", side_effect=predict), \
             mock.patch.object(inference.old, "upsample", side_effect=lambda p, r, i: p):
            result = inference.components_from_records(dict(self.inputs, labels=np.full(50, "D")),
                                                         records, self.models, self.manifest)
        for role in result:
            expected = np.mean([np.mean(np.asarray(arrays[role][p], dtype=np.float64), axis=0)
                                for p in range(5)], axis=0)
            np.testing.assert_array_equal(result[role], expected)

    def test_fixed_d_survives_disagreeing_experts(self):
        adopted = np.tile([.9, .05, .05], (50, 1))
        adopted[20:35] = [.01, .01, .98]
        components = dict(adopted=adopted, ce=np.tile([.01, .98, .01], (50, 1)),
                          calibrated=np.tile([.01, .98, .01], (50, 1)))
        expected = inference.audit.decode(adopted, inference.supervised.model_periods(self.inputs),
                                           25., inference.audit.CONFIG)
        labels = inference.decode_components(self.inputs, components)
        np.testing.assert_array_equal(labels == "D", expected == "D")
        self.assertTrue((labels != expected).any())

    def test_reject_recipe_or_role_drift(self):
        broken = dict(self.manifest, architecture=dict(self.manifest["architecture"],
                      recipe=dict(inference.RECIPE, ce_weight=.4)))
        with self.assertRaises(ValueError):
            inference.validate(self.models, broken)
        with self.assertRaises(ValueError):
            inference.validate({k:v for k,v in self.models.items() if k != "ce_2027"}, self.manifest)


if __name__ == "__main__":
    unittest.main()
