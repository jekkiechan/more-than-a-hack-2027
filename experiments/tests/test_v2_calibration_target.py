from __future__ import annotations

import unittest
import json
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np

from experiments import feed_benchmark_audit_v1 as audit
from experiments import v2_calibration_target as study


class CalibrationGridTests(unittest.TestCase):
    def test_grid_is_exact_and_only_changes_the_four_declared_parameters(self):
        arms = study.configurations()
        self.assertEqual(len(arms), 109)
        self.assertEqual(len({arm["id"] for arm in arms}), 109)
        self.assertEqual(sum(arm["family"] == "adop" for arm in arms), 54)
        self.assertEqual(sum(arm["family"] == "cand" for arm in arms), 54)
        blend = arms[-1]
        self.assertEqual(blend["id"], "blend50_f062")
        self.assertEqual(blend["config"], audit.CONFIG)
        fixed_fields = set(audit.CONFIG) - {"dt", "dead", "ha", "ht"}
        for arm in arms[:-1]:
            config = arm["config"]
            self.assertEqual({k: config[k] for k in fixed_fields},
                             {k: audit.CONFIG[k] for k in fixed_fields})
            self.assertIn(config["dt"], study.TEMP_D)
            self.assertIn(config["dead"], study.SWITCH_COST)
            self.assertIn(config["ha"], study.HA_COST)
            self.assertIn(config["ht"], study.HA_TEMP)
        f062 = next(a for a in arms if a["id"] == "cand_dt4_dead3_ha3_ht1")
        self.assertEqual(f062["config"], audit.CONFIG)
        self.assertIn("cand_dt4_dead3_ha3_ht0.65", {a["id"] for a in arms})
        self.assertIn("D-transition", study.__doc__)

    def test_cell_scoring_runs_each_fixed_arm_on_the_phase_mean(self):
        cell = {"labels": np.array(["H", "A"]), "period": np.ones(2),
                "timestamps": np.array([0., .1]), "fps": 10.}
        probabilities = {family: np.full((2, 3), 1 / 3) for family in study.FAMILIES}
        model_period = np.ones(2, dtype=int)
        seen = []

        def fake_decode(members, period, fps, config):
            seen.append((members.copy(), config.copy()))
            return np.array(["H", "A"])

        with (patch.object(study, "_cell_inputs",
                           return_value=(cell, probabilities, model_period)),
              patch.object(study.decoder, "decode", side_effect=fake_decode),
              patch.object(study.audit, "score", return_value={"spell_f1": 0.5,
                                                                 "macro_f1": 0.6})):
            result = study._cell_scores("Sportec", "m", "clean", 0)

        self.assertEqual(list(result), list(study.ARM_IDS))
        self.assertEqual(len(seen), 109)
        np.testing.assert_allclose(seen[-1][0][0], probabilities["adop"])
        self.assertEqual(seen[-1][1], audit.CONFIG)

    def test_resumable_cell_cache_requires_declaration_and_payload_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(study, "OUT", Path(temporary)):
                path = study._cell_path("Sportec__m__clean__0")
                path.parent.mkdir(parents=True)
                payload = {"declaration_sha256": "decl", "arms": list(study.ARM_IDS),
                           "scores": {"synthetic": {"spell_f1": 0.4}}}
                path.write_text(json.dumps(payload))
                receipt_path = path.with_suffix(".receipt.json")
                receipt_path.write_text(json.dumps({"declaration_sha256": "decl",
                                                    "sha256": study._sha256(path)}))
                self.assertEqual(study._load_cell("Sportec__m__clean__0", "decl"),
                                 payload["scores"])
                path.write_text("{}")
                with self.assertRaises(RuntimeError):
                    study._load_cell("Sportec__m__clean__0", "decl")


if __name__ == "__main__":
    unittest.main()
