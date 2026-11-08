import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments import complementary_pilot_prepare as pilot


class ComplementaryPilotPreparationContracts(unittest.TestCase):
    def test_two_phase_native_cache_preserves_seed_members_and_baseline(self):
        n = 6
        inputs = {"ball": np.zeros((n, 2)), "players": np.zeros((n, 2, 2)),
                  "period": np.ones(n, dtype=int), "timestamps": np.arange(n) / 10,
                  "fps": 10.}
        labels = np.array(["H", "A", "D", "H", "A", "D"])
        phases = [{"phase": i} for i in range(2)]
        seed_probs = {
            (0, 2026): np.tile([.8, .1, .1], (n, 1)),
            (0, 2027): np.tile([.1, .8, .1], (n, 1)),
            (1, 2026): np.tile([.6, .2, .2], (n, 1)),
            (1, 2027): np.tile([.2, .6, .2], (n, 1)),
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "input.joblib"
            input_path.write_bytes(b"verified input fixture")
            baseline_manifest = root / "models/joint_gru_sn_dense_original_diversity_phase_ensemble_v1/baseline_manifest.json"
            baseline_manifest.parent.mkdir(parents=True)
            baseline_manifest.write_text("historical phase predecessor")
            model_files = [root / f"{kind}_{seed}.joblib" for kind in ("model", "teacher")
                           for seed in pilot.SEEDS]
            for path in model_files:
                path.write_bytes(path.name.encode())
            with (mock.patch.object(pilot, "_input_path", return_value=input_path),
                  mock.patch.object(pilot.audit, "DATA", root),
                  mock.patch.object(pilot.shared, "_prepare_cell",
                                    return_value=(inputs, labels, {}, phases, np.ones(n, dtype=int))),
                  mock.patch.object(pilot.shared, "predict_seed",
                                    side_effect=lambda _i, rec, _f, _p, seed: seed_probs[(rec["phase"], seed)]),
                  mock.patch.object(pilot.shared, "_model_path", side_effect=lambda _f, _p, s: model_files[0 if s == 2026 else 2]),
                  mock.patch.object(pilot.shared, "_teacher_path", side_effect=lambda _p, s: model_files[1 if s == 2026 else 3]),
                  mock.patch.object(pilot.feed_spell_push, "decode", return_value=np.array(["H"] * n))):
                arrays, identity = pilot._make_cell("metrica_3", "clean")

        self.assertEqual(identity["phase_count"], 2)
        self.assertEqual(identity["cell"], "Metrica|metrica_3|clean|0")
        self.assertEqual(identity["source_study_id"], "original_full_phase_ensemble_v1")
        for phase in range(2):
            for seed in pilot.SEEDS:
                self.assertIn(f"prob__candidate__seed{seed}__phase{phase}", arrays)
        expected = np.mean([np.mean([seed_probs[(phase, seed)] for seed in pilot.SEEDS], axis=0)
                            for phase in range(2)], axis=0)
        np.testing.assert_array_equal(arrays["prob__historical_ensemble"], expected)
        np.testing.assert_array_equal(arrays["decoded__historical_f062"], np.array(["H"] * n))
        self.assertEqual(arrays["identity__cell"].item(), identity["cell"])

    def test_complete_view_design_is_five_matches_and_thirty_five_cells(self):
        self.assertEqual(len(pilot.audit.MATCHES), 5)
        self.assertEqual(pilot.VIEWS, (
            "clean", "feed_like_v1_r1", "feed_like_v1_r2",
            "feed_like_v2_r1", "feed_like_v2_r2", "feed_like_v2_r3", "feed_like_v2_r4"))
        self.assertEqual(len(pilot.audit.MATCHES) * len(pilot.VIEWS), 35)
        self.assertEqual(pilot.HISTORICAL_CONTROL["clean"], 0.6675555328266306)
        self.assertEqual(pilot.HISTORICAL_CONTROL["feed_like_v2"], 0.5835712076382747)


if __name__ == "__main__":
    unittest.main()
