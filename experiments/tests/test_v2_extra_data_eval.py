from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import v2_extra_data_eval as extra


class ExtraDataEvalTests(unittest.TestCase):
    def test_paired_seed_phase_pooling(self):
        for nphase in (2, 5):
            values = {2026: [np.full((4, 3), .2) for _ in range(nphase)],
                      2027: [np.full((4, 3), .6) for _ in range(nphase)]}
            pooled, per_seed = extra._pool(values)
            np.testing.assert_allclose(pooled, np.full((4, 3), .4))
            np.testing.assert_allclose(per_seed[2026], np.full((4, 3), .2))
            np.testing.assert_allclose(per_seed[2027], np.full((4, 3), .6))

    def test_fit_seal_requires_exact_three_arm_18_fit_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            teachers = {}
            fits = []
            for arm in ("fresh_control", "gar_weight625", "gar_quality50"):
                for fold in ("leave_out_metrica", "leave_out_skillcorner", "leave_out_sportec"):
                    for seed in extra.SEEDS:
                        model = root / f"{arm}_{fold}_{seed}.joblib"
                        model.write_bytes(f"{arm}/{fold}/{seed}".encode())
                        teacher = root / f"teacher_{fold}_{seed}.joblib"
                        if teacher not in teachers:
                            teacher.write_bytes(f"teacher/{fold}/{seed}".encode())
                            teachers[teacher] = extra.sha(teacher)
                        provider = fold.removeprefix("leave_out_")
                        fits.append({"arm": arm, "fold": fold, "seed": seed,
                            "path": model.name, "sha256": extra.sha(model), "stage": 1280,
                            "stage_updates": 256, "declaration_sha256": "decl",
                            "teacher_sha256": teachers[teacher], "d_state_unchanged": True})
                        Path(str(model) + ".json").write_text(json.dumps(fits[-1]))
            seal_path = root / "seal.json"
            train_decl = root / "training.json"
            train_decl.write_text("{}")
            declaration_hash = extra.sha(train_decl)
            for row in fits:
                row["declaration_sha256"] = declaration_hash
                (root / (row["path"] + ".json")).write_text(json.dumps(row))
            seal_path.write_text(json.dumps({"study_id": extra.STUDY, "all_18_complete": True,
                "declaration_sha256": declaration_hash, "fits": fits}))
            plan = root / "plan.json"
            plan.write_text(json.dumps({"arms": {"fresh_control": {}, "gar_weight625": {}, "gar_quality50": {}}}))
            with ExitStack() as stack:
                stack.enter_context(patch.object(extra, "ROOT", root))
                stack.enter_context(patch.object(extra, "TRAIN_DECL", train_decl))
                stack.enter_context(patch.object(extra, "PLAN", plan))
                stack.enter_context(patch.object(extra.balanced.runtime.frozen, "_teacher_path",
                    lambda provider, seed: root / f"teacher_leave_out_{provider.lower()}_{seed}.joblib"))
                # Runtime fold labels are lower-case provider names.
                arms, inventory, _ = extra._load_training_seal(seal_path)
            self.assertEqual(len(arms), 3)
            self.assertEqual(len(inventory), 18)

    def test_control_replay_uses_35_cells_and_checks_every_metric(self):
        cells = sorted(extra._expected_cells())
        metric = {m: .5 for m in extra.METRICS}
        metric["reference_spell_count"] = 2
        rows = {key: metric.copy() for key in cells}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            replay_path = root / "replay.json"
            preflight_path = root / "preflight.json"
            preflight_path.write_text(json.dumps({"passed": True, "aggregate": {},
                "cell_metrics": rows, "checks": {}}))
            declaration = {"preflight_result_sha256": "bound", "preflight_declaration_sha256": "bound"}
            with ExitStack() as stack:
                stack.enter_context(patch.object(extra, "require_declaration", lambda: declaration))
                stack.enter_context(patch.object(extra, "REPLAY", replay_path))
                stack.enter_context(patch.object(extra, "RESULT", root / "result.json"))
                stack.enter_context(patch.object(extra, "PREFLIGHT_RESULT", preflight_path))
                stack.enter_context(patch.object(extra, "PREFLIGHT_DECL", root / "preflight-decl.json"))
                extra.PREFLIGHT_DECL.write_text("frozen")
                stack.enter_context(patch.object(extra, "sha", lambda _: "bound"))
                result = extra.replay_frontier_control()
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["cell_metrics"]), 35)
            self.assertEqual(result["preflight_result_sha256"], "bound")


if __name__ == "__main__":
    unittest.main()
