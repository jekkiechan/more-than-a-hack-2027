import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments import spell_frontier_gar as study


def _metrics():
    return {
        "spell_f1": .5, "macro_f1": .7, "d_f1": .4,
        "d_precision": .3, "d_recall": .6,
        "predicted_spell_count": 2.0, "reference_spell_count": 2.0,
        "short_spell_recall_iou_0_5_lt3s": .25,
    }


class SpellFrontierGarTests(unittest.TestCase):
    def test_corruption_reuses_exact_prior_gar_seed_contract(self):
        inputs, labels, calibration = {}, np.array(["H"]), {"x": 1}
        with patch.object(study.heldout, "feed_view", return_value=("view", 123)) as feed:
            result, seed = study._corrupt_view(inputs, labels, calibration, "3850", 3)
        self.assertEqual((result, seed), ("view", 123))
        feed.assert_called_once_with(inputs, labels, calibration,
                                     "gar_deployment_eval_v2/3850", 3)

    def test_single_game_run_reconstructs_adopted_control_and_seals_paired_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            declaration_path = root / "declaration.json"
            games_path = root / "games"
            calibration_path = root / "calibration.json"
            calibration_path.write_text(json.dumps({"calibration": {"fixture": True}}))
            metrics = _metrics()
            declaration = {
                "prior_game_artifacts": {
                    "test/3850": {"inventory": {"native_frames": 5, "graded_frames": 5}, "clean": metrics, "v2": metrics}
                }
            }
            declaration_path.write_text(json.dumps(declaration))
            inputs = {
                "ball": np.zeros((5, 2)), "period": np.ones(5, dtype=np.int32),
                "timestamps": np.arange(5, dtype=float), "fps": np.asarray(29.97),
            }
            labels = np.array(["H", "A", "D", "H", "A"])
            saved_x = np.ones((5, 1))
            names = ["fixture"]
            phase = {"x": saved_x.copy(), "names": names, "fps": np.asarray(29.97)}
            pred = np.array(["H", "A", "D", "H", "A"])
            components = {"adopted": np.zeros((5, 3))}

            with (
                patch.object(study, "ROOT", root),
                patch.object(study, "DECLARATION", declaration_path),
                patch.object(study, "GAMES", games_path),
                patch.object(study, "check_declaration", return_value=declaration),
                patch.object(study, "read", side_effect=lambda p: json.loads(Path(p).read_text())),
                patch.object(study.heldout, "CALIBRATION", calibration_path),
                patch.object(study.heldout, "load_game_records", return_value=[{"fps": 29.97}]),
                patch.object(study.heldout, "capture_inputs", return_value={}),
                patch.object(study.heldout, "build_game", return_value=(inputs, labels, saved_x, names)),
                patch.object(study.heldout, "phase_set", return_value=[phase] * 6),
                patch.object(study, "_corrupt_view", side_effect=lambda i, l, c, g, r: (i, 100 + r)),
                patch.object(study, "_models", return_value=({}, {})),
                patch.object(study.package, "validate"),
                patch.object(study.package, "components_from_records", return_value=components),
                patch.object(study.package, "decode_components", return_value=pred),
                patch.object(study.audit, "decode", return_value=pred),
                patch.object(study.supervised, "model_periods", return_value=np.ones(5, dtype=np.int32)),
                patch.object(study.audit, "score", return_value=metrics),
            ):
                result_path = Path(study._run_game(("test", "3850")))

            row = json.loads(result_path.read_text())
            self.assertTrue(row["paired_D_mask_equal"])
            self.assertTrue(row["baseline_reconstructed_from_frozen_adopted_package"])
            self.assertEqual(row["corruption_seeds"], {f"v2_{i}": 100 + i for i in range(1, 5)})
            self.assertTrue(all(row["views"][v]["delta"]["spell_f1"] == 0 for v in study.VIEWS))
            with np.load(result_path.with_suffix(".npz"), allow_pickle=False) as cache:
                self.assertTrue(set(f"pred__{v}" for v in study.VIEWS).issubset(cache.files))
                self.assertTrue(set(f"adopted_control__{v}" for v in study.VIEWS).issubset(cache.files))
                for view in study.VIEWS:
                    np.testing.assert_array_equal(cache[f"pred__{view}"], cache[f"adopted_control__{view}"])

    def test_result_writer_is_write_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "result.json"
            study.write_once(path, {"study": "fixture"})
            with self.assertRaises(FileExistsError):
                study.write_once(path, {"study": "replacement"})


if __name__ == "__main__":
    unittest.main()
