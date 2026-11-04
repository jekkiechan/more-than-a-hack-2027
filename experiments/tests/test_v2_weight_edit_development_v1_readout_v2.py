from __future__ import annotations

import unittest
from unittest import mock
import json
import tempfile
from pathlib import Path

import joblib
import numpy as np

from experiments import v2_weight_edit_development as development
from experiments import v2_weight_edit_development_v1_readout_v2 as readout


class V1ReadoutExecutionTests(unittest.TestCase):
    def test_filters_only_archived_control_from_editable_arm_call(self):
        original = mock.Mock(return_value={"ok": True})
        readout._editable_v1_cell_call(
            original, "provider", "match", "feed_like_v1", 1,
            ["update_bias_m025", "blend_unedited_ha15", "deployed_original_full"],
            prepared=(1, 2, 3, 4, 5))
        original.assert_called_once_with(
            "provider", "match", "feed_like_v1", 1,
            ["update_bias_m025", "blend_unedited_ha15"], prepared=(1, 2, 3, 4, 5))

    def test_deduplicates_winner_when_it_is_the_unedited_control(self):
        original = mock.Mock(return_value={"ok": True})
        readout._editable_v1_cell_call(
            original, "provider", "match", "feed_like_v1", 1,
            ["blend_unedited_ha15", "blend_unedited_ha15", "deployed_original_full"])
        original.assert_called_once_with("provider", "match", "feed_like_v1", 1,
                                         ["blend_unedited_ha15"])

    def test_v1_action_uses_bound_frozen_protocol_with_scoped_adapter(self):
        with mock.patch.object(readout.execution, "verify_binding"), \
             mock.patch.object(development, "score_v1", return_value={"ok": True}) as scored:
            self.assertEqual(readout.run_v1(workers=4), {"ok": True})
            scored.assert_called_once_with(workers=4)

    def test_missing_editable_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            readout._editable_v1_cell_call(
                mock.Mock(), "provider", "match", "feed_like_v1", 1,
                ["deployed_original_full"])

    def test_full_v1_orchestration_survives_fork_workers_and_control_seed_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result_path, winner_path = root / "primary.json", root / "winner.json"
            v1_path, declaration_path = root / "v1.json", root / "declaration.json"
            inputs_path, cell_out = root / "inputs", root / "cells"
            inputs_path.mkdir()
            declaration_path.write_text("{}\n")
            result_path.write_text(json.dumps({"eligible_winner": "blend_unedited_ha15"}))
            winner_path.write_text(json.dumps({"winner": "blend_unedited_ha15", "eligible": True,
                                               "primary_result_sha256": development.sha(result_path)}))
            matches = [("Sportec", f"M{i}", "fixture") for i in range(5)]
            file_map = {}
            for _, match, _ in matches:
                filename = f"{match}.joblib"
                file_map[match] = filename
                joblib.dump({"inputs": {"period": np.array([1, 1]),
                                        "timestamps": np.array([0., 0.2]),
                                        "fps": 5.0},
                             "labels": np.array(["H", "A"]), "record": {"x": np.zeros((2, 1))}},
                            inputs_path / f"{filename}_feed_like_corrupt_1.joblib")
                joblib.dump({"inputs": {"period": np.array([1, 1]),
                                        "timestamps": np.array([0., 0.2]),
                                        "fps": 5.0},
                             "labels": np.array(["H", "A"]), "record": {"x": np.zeros((2, 1))}},
                            inputs_path / f"{filename}_feed_like_corrupt_2.joblib")

            def fake_score(provider, match, view, rep, arms, prepared=None):
                return {arm: {"spell_f1": 0.5, "macro_f1": 0.7} for arm in arms}

            with mock.patch.multiple(
                development,
                RESULT=result_path, WINNER_RECEIPT=winner_path, V1_RESULT=v1_path,
                DECLARATION=declaration_path, CELL_OUT=cell_out,
            ), mock.patch.object(development.upstream.frozen.audit, "MATCHES", matches), \
                 mock.patch.object(development.upstream.frozen.audit, "FILE", file_map), \
                 mock.patch.object(development.upstream.frozen.audit, "EVAL_INPUTS", inputs_path), \
                 mock.patch.object(development.upstream.frozen.audit, "phase_records", return_value=[{}]), \
                 mock.patch.object(development.upstream.frozen.audit.supervised, "model_periods",
                                  return_value=np.array([1, 1])), \
                 mock.patch.object(development, "_score_cell", side_effect=fake_score), \
                 mock.patch.object(development, "_aggregate", return_value={"spell_f1": 0.5}), \
                 mock.patch.object(development, "_aggregate_seed", return_value={"spell_f1": 0.5}), \
                 mock.patch.object(readout.execution, "verify_binding"):
                result = readout.run_v1(workers=2)

            self.assertEqual(len(result["cells"]), 10)
            self.assertEqual(result["winner"], "blend_unedited_ha15")
            self.assertEqual(set(result["arms"]), {"blend_unedited_ha15", "deployed_original_full"})
            self.assertEqual(result["component_seed_diagnostics"]["deployed_original_full"],
                             {"2026": {}, "2027": {}})


if __name__ == "__main__":
    unittest.main()
