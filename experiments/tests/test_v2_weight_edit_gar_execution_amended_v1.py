import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from experiments import v2_weight_edit_gar_execution_amended_v1 as amended
from experiments import v2_weight_edit_evaluation_v1 as base


class AmendedGarExecutorTests(unittest.TestCase):
    def test_native_fps_sets_six_phases_while_phase_set_keeps_sampled_fps(self):
        labels = np.asarray(["H", "A", "D", "X"])
        inputs = {"period": np.asarray([1, 1, 1, 0]),
                  "timestamps": np.arange(4, dtype=float) / 29.97,
                  "fps": 29.97}
        saved_x = np.zeros((4, 2), dtype=float)
        records = [{"fps": 4.995}]
        phases = [
            {"x": saved_x.copy(), "phase": index}
            for index in range(6)
        ]
        sample_rates, decode_rates = [], []

        class Adapter:
            def decode_probabilities(self, probabilities, period, fps, receipt):
                decode_rates.append(fps)
                return labels.copy()

        declaration = {
            "cohort": {"match_ids": ["3850"]},
            "selected_package": {"receipt_sha256": "package-sha",
                                 "receipt": {"adapter": "bound"}},
        }
        with tempfile.TemporaryDirectory(dir=base.ROOT / "data/experiments") as tmp:
            amendment_path = Path(tmp) / "amendment.json"
            amendment_path.write_text("{}")
            deployment_receipt = Path(tmp) / "deployment.json"
            deployment_receipt.write_text(json.dumps({"fixture": "synthetic-test-deployment"}))
            with (mock.patch.object(amended, "check_amendment", return_value={}),
                  mock.patch.object(amended, "GAMES", Path(tmp) / "games"),
                  mock.patch.object(amended, "AMENDMENT", amendment_path),
                  mock.patch.object(base, "check_declaration", return_value=declaration),
                  mock.patch.object(base, "PACKAGE_RECEIPT", deployment_receipt),
                  mock.patch.object(base, "_load_deployment", return_value=(Adapter(), object())),
                  mock.patch.object(base.gar.previous, "load_game_records", return_value=records),
                  mock.patch.object(base.gar.previous, "capture_inputs", return_value={}),
                  mock.patch.object(base.gar.previous, "build_game",
                                    return_value=(inputs, labels, saved_x, ["feature"])),
                  mock.patch.object(base, "json_read", return_value={"calibration": {}}),
                  mock.patch.object(base.gar.previous, "feed_view", return_value=(inputs, 123)),
                  mock.patch.object(base.gar.previous, "phase_set",
                                    side_effect=lambda data, names, fps:
                                    (sample_rates.append(fps) or phases)),
                  mock.patch.object(base.audit.tc, "inputs_only", side_effect=lambda data: data),
                  mock.patch.object(base.audit.supervised, "model_periods",
                                    side_effect=lambda data: data["period"]),
                  mock.patch.object(base, "_candidate_probability",
                                    return_value=np.tile([[.5, .4, .1]], (4, 1))),
                  mock.patch.object(base.audit, "score", return_value={
                      **{key: 1.0 for key in base.METRICS},
                      "graded_frames": 3, "reference_spell_count": 1}),):
                path = amended.evaluate_game("3850")
                output = json.loads(path.read_text())
                prediction_exists = path.with_suffix(".npz").is_file()
        self.assertEqual(len(sample_rates), 5)
        self.assertTrue(all(rate == 4.995 for rate in sample_rates))
        self.assertEqual(decode_rates, [29.97] * 5)
        self.assertEqual(output["phase_counts"], {view: 6 for view in base.VIEWS})
        self.assertEqual(output["native_frames"], 4)
        self.assertTrue(prediction_exists)


if __name__ == "__main__":
    unittest.main()
