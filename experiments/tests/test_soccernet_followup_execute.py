import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np

from experiments import soccernet_followup_execute as execute


class FollowupExecutorTests(unittest.TestCase):
    def test_output_paths_cover_frozen_thirty_fit_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.object(execute, "OUT", Path(temporary)):
                paths = {
                    execute._model_path(arm, fold, seed).relative_to(temporary).as_posix()
                    for arm in execute.ARMS for fold in execute.FOLDS for seed in execute.SEEDS
                }
        self.assertEqual(len(paths), 30)
        self.assertIn("models/reference/leave_out_sportec/seed2026.joblib", paths)
        self.assertIn("models/context_reconstruction/leave_out_skillcorner/seed2027.joblib", paths)

    def test_followup_freeze_rejects_changed_segment_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rebuild = root / "rebuild_declaration.json"
            rebuild.write_text("{}\n")
            declaration = {
                "source_sha256": {"source.py": "frozen"},
                "rebuild_declaration_sha256": execute.sha256(rebuild),
                "reference_models": {}, "reference_receipts": {},
                "teacher_files": {}, "evaluation_inputs": {},
                "old_restore_file_sha256": {}, "segments": {"manifest": "old"},
                "gaps": {"manifest": "old"},
                "runtime_versions": {},
            }
            path = root / "predeclared.json"
            path.write_text(json.dumps(declaration))
            with (
                mock.patch.object(execute, "DECLARATION", path),
                mock.patch.object(execute, "REBUILD_DECLARATION", rebuild),
                mock.patch.object(execute, "_source_hashes", return_value={"source.py": "frozen"}),
                mock.patch.object(execute, "_runtime_versions", return_value={}),
                mock.patch.object(execute, "_reference_hashes", return_value=({}, {})),
                mock.patch.object(execute, "_verify_hash_map"),
                mock.patch.object(execute, "_segment_provenance", return_value=({"manifest": "new"}, [])),
            ):
                with self.assertRaisesRegex(ValueError, "Segment provenance changed"):
                    execute._check_followup()

    def test_followup_freeze_rejects_changed_gap_payload_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rebuild = root / "rebuild_declaration.json"
            rebuild.write_text("{}\n")
            declaration = {
                "source_sha256": {"source.py": "frozen"},
                "rebuild_declaration_sha256": execute.sha256(rebuild),
                "reference_models": {}, "reference_receipts": {},
                "teacher_files": {}, "evaluation_inputs": {},
                "old_restore_file_sha256": {}, "segments": {"manifest": "same"},
                "gaps": {"manifest": "old"},
                "runtime_versions": {},
            }
            path = root / "predeclared.json"
            path.write_text(json.dumps(declaration))
            with (
                mock.patch.object(execute, "DECLARATION", path),
                mock.patch.object(execute, "REBUILD_DECLARATION", rebuild),
                mock.patch.object(execute, "_source_hashes", return_value={"source.py": "frozen"}),
                mock.patch.object(execute, "_runtime_versions", return_value={}),
                mock.patch.object(execute, "_reference_hashes", return_value=({}, {})),
                mock.patch.object(execute, "_verify_hash_map"),
                mock.patch.object(execute, "_segment_provenance", return_value=({"manifest": "same"}, [])),
                mock.patch.object(execute, "_gap_records", return_value=([], {"manifest": "new"})),
            ):
                with self.assertRaisesRegex(ValueError, "Gap-view provenance changed"):
                    execute._check_followup()

    def test_declaration_write_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.json"
            execute._write_once(path, {"recipe": "fixed"})
            execute._write_once(path, {"recipe": "fixed"})
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                execute._write_once(path, {"recipe": "changed"})

    def test_historical_source_audit_only_allows_named_metadata_drift(self):
        historical = {"experiments/train.py": "a", "experiments/current_baseline.py": "b"}
        current = {"experiments/train.py": "a", "experiments/current_baseline.py": "changed"}
        audit = execute._historical_source_audit(historical, current)
        self.assertEqual(audit["allowed_metadata_mismatches"],
                         ["experiments/current_baseline.py"])
        current["experiments/train.py"] = "changed"
        with self.assertRaisesRegex(ValueError, "training-core source changed"):
            execute._historical_source_audit(historical, current)

    def test_reference_fit_persists_reuses_and_resumes_with_numpy_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            execution = root / "execution"
            execution.mkdir()
            rebuild = root / "rebuild_declaration.json"
            rebuild.write_text("{}\n")
            teacher_path = root / "teacher.joblib"
            teacher = {"median": np.asarray([1.0, 2.0]),
                       "scale": np.asarray([3.0, 4.0]), "seed": 2026}
            joblib.dump(teacher, teacher_path)
            records = [{"match": f"soccernet_{i}"} for i in range(45)]
            declaration = {"selected_historical_recipe": "w96_lr3e-4_step2048",
                           "segments": {"record_count": len(records)}}

            def snapshot(step):
                return {"kind": "soccernet_capacity", "hidden": 96, "step": step,
                        "median": teacher["median"].copy(), "scale": teacher["scale"].copy(),
                        "state": {"weight": np.asarray([step], dtype=np.float32)},
                        "training": {"diagnostics": np.asarray([1.0, 2.0])}}

            train = mock.Mock()

            def first_fit(*args, **kwargs):
                kwargs["on_checkpoint"](512, snapshot(512))
                return {2048: snapshot(2048)}

            train.side_effect = first_fit
            with (
                mock.patch.object(execute, "OUT", execution),
                mock.patch.object(execute, "REBUILD_DECLARATION", rebuild),
                mock.patch.object(execute, "_check_rebuild", return_value=declaration),
                mock.patch.object(execute.old, "control_path", return_value=teacher_path),
                mock.patch.object(execute.old, "load_old", return_value=[{"old": True}]),
                mock.patch.object(execute.segments, "load_records", return_value=records),
                mock.patch.object(execute.model, "train", train),
            ):
                execute.fit_reference("leave_out_sportec", 2026)
                target = execute._model_path("reference", "leave_out_sportec", 2026)
                resume_path = execute._resume_path("leave_out_sportec", 2026)
                receipt = execute.read_json(target.with_suffix(".receipt.json"))
                self.assertEqual(receipt["training"]["diagnostics"], [1.0, 2.0])
                self.assertTrue(resume_path.is_file())

                # Simulate a process ending after its durable step-512 callback.
                target.unlink()
                target.with_suffix(".receipt.json").unlink()

                def resumed_fit(*args, **kwargs):
                    self.assertEqual(kwargs["milestones"], (2048,))
                    self.assertEqual(kwargs["resume_from"]["step"], 512)
                    return {2048: snapshot(2048)}

                train.side_effect = resumed_fit
                execute.fit_reference("leave_out_sportec", 2026)
                self.assertEqual(train.call_count, 2)
                self.assertTrue(target.is_file())
                execute.fit_reference("leave_out_sportec", 2026)
                self.assertEqual(train.call_count, 2)

    def test_seal_requires_identical_supervised_draw_hashes_across_followup_arms(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rebuild = root / "rebuild_declaration.json"
            rebuild.write_text("rebuild\n")
            declaration_path = root / "predeclared.json"
            declaration_path.write_text("followup\n")
            declaration = {"source_sha256": {"source.py": "source-hash"},
                           "reference_models": {}, "reference_receipts": {}}
            draws = {}
            with (
                mock.patch.object(execute, "OUT", root),
                mock.patch.object(execute, "REBUILD_DECLARATION", rebuild),
                mock.patch.object(execute, "DECLARATION", declaration_path),
                mock.patch.object(execute, "_check_followup", return_value=declaration),
            ):
                for fold in execute.FOLDS:
                    for seed in execute.SEEDS:
                        expected_draw = f"draw-{fold}-{seed}"
                        draws[(fold, seed)] = expected_draw
                        for arm in execute.ARMS:
                            model_path = execute._model_path(arm, fold, seed)
                            model_path.parent.mkdir(parents=True, exist_ok=True)
                            model_path.write_text(f"{arm}-{fold}-{seed}")
                            receipt = {"model_sha256": execute.sha256(model_path)}
                            if arm == "reference":
                                receipt["rebuild_declaration_sha256"] = execute.sha256(rebuild)
                            else:
                                receipt["declaration_sha256"] = execute.sha256(declaration_path)
                                receipt["rebuild_declaration_sha256"] = execute.sha256(rebuild)
                                receipt["training"] = {"supervised_draws_sha256": expected_draw}
                            model_path.with_suffix(".receipt.json").write_text(json.dumps(receipt))

                changed = execute._model_path("gap_consistency", execute.FOLDS[0], execute.SEEDS[0])
                receipt_path = changed.with_suffix(".receipt.json")
                receipt = json.loads(receipt_path.read_text())
                receipt["training"]["supervised_draws_sha256"] = "different-draws"
                receipt_path.write_text(json.dumps(receipt))
                with self.assertRaisesRegex(ValueError, "changed supervised draws"):
                    execute.seal()
                self.assertFalse((root / "models_sealed.json").exists())

                receipt["training"]["supervised_draws_sha256"] = draws[(execute.FOLDS[0], execute.SEEDS[0])]
                receipt_path.write_text(json.dumps(receipt))
                sealed = execute.seal()
                self.assertEqual(len(sealed["models"]), 30)
                self.assertEqual(len(sealed["receipts"]), 30)
                self.assertEqual(len(sealed["supervised_draws_sha256"]), 6)
                self.assertEqual(execute.seal(), sealed)


if __name__ == "__main__":
    unittest.main()
