from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

from experiments import ball_exit_delivery, ball_exit_gar, frontier_reconstructed_delivery as delivery


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReconstructedDeliveryContracts(unittest.TestCase):
    def test_training_seal_binds_every_reconstructed_stage_and_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan, declaration, seal = root / "plan.json", root / "declaration.json", root / "seal.json"
            verification = root / "verification.json"
            models = root / "models"
            models.mkdir()
            outputs = {f"stage{stage}_{seed}.joblib": models / f"stage{stage}_{seed}.joblib"
                       for stage in (512, 1024) for seed in delivery.SEEDS}
            for name, path in outputs.items():
                path.write_bytes(name.encode())
            plan.write_text('{"study":"frontier_pooled_reconstruction_v1"}')
            declaration.write_text(json.dumps({"plan_sha256": digest(plan)}))
            fits = [{"path": f"models/{name}",
                     "sha256": digest(path), "adopted_d_state_unchanged": True} for name, path in outputs.items()]
            seal.write_text(json.dumps({"declaration_sha256": digest(declaration), "all_four_fits_complete": True,
                "original_byte_identity": False, "fits": fits,
                "models_sha256": {f"models/{name}": digest(path) for name, path in outputs.items()}}))
            verification.write_text(json.dumps({"seal_sha256": digest(seal), "declaration_sha256": digest(declaration),
                "all_passed": True, "fit_count": 4,
                "models_sha256": {f"models/{name}": digest(path)
                                  for name, path in outputs.items()}}))
            with patch.multiple(delivery, ROOT=root, RECON=models, RECON_PLAN=plan, RECON_DECL=declaration,
                                TRAIN_RECEIPT=seal, RECON_VERIFY=verification):
                delivery.check_training_seal()
                outputs["stage1024_2026.joblib"].write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "does not bind"):
                    delivery.check_training_seal()

    def test_wrong_training_plan_seal_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            for stage in (512, 1024):
                for seed in delivery.SEEDS:
                    (models / f"stage{stage}_{seed}.joblib").write_bytes(b"weight")
            plan, declaration, seal = root / "plan.json", root / "declaration.json", root / "seal.json"
            verification = root / "verification.json"
            plan.write_text('{"study":"frontier_pooled_reconstruction_v1"}')
            declaration.write_text(json.dumps({"plan_sha256": "0" * 64}))
            fits = [{"path": f"data/experiments/frontier_pooled_reconstruction_v1/models/stage{stage}_{seed}.joblib",
                     "sha256": digest(models / f"stage{stage}_{seed}.joblib"), "adopted_d_state_unchanged": True}
                    for stage in (512, 1024) for seed in delivery.SEEDS]
            seal.write_text(json.dumps({"declaration_sha256": digest(declaration), "all_four_fits_complete": True,
                "original_byte_identity": False, "fits": fits,
                "models_sha256": {f"models/stage{stage}_{seed}.joblib": digest(models / f"stage{stage}_{seed}.joblib")
                                   for stage in (512, 1024) for seed in delivery.SEEDS}}))
            verification.write_text(json.dumps({"seal_sha256": digest(seal), "declaration_sha256": digest(declaration),
                "all_passed": True, "fit_count": 4,
                "models_sha256": {f"models/stage{stage}_{seed}.joblib":
                                  digest(models / f"stage{stage}_{seed}.joblib")
                                  for stage in (512, 1024) for seed in delivery.SEEDS}}))
            with patch.multiple(delivery, ROOT=root, RECON=models, RECON_PLAN=plan, RECON_DECL=declaration,
                                TRAIN_RECEIPT=seal, RECON_VERIFY=verification):
                with self.assertRaisesRegex(ValueError, "predeclared plan"):
                    delivery.check_training_seal()

    def test_staged_source_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "runner.py"
            src.write_text("frozen")
            parent = root / "parent.json"
            frontier = root / "frontier.json"
            frontier.write_text("frontier")
            parent.write_text(json.dumps({"previous_manifest_sha256": digest(frontier)}))
            staged = root / "staged.json"
            seal = root / "seal.json"
            seal.write_text("seal")
            manifest = {"package_version": delivery.PACKAGE,
                "architecture": {}, "previous_manifest_sha256": digest(parent),
                "source_sha256": {"runner.py": digest(src)},
                "reconstruction": {"training_seal_sha256": digest(seal), "verification_sha256": "v",
                    "source_frontier_manifest_sha256": digest(frontier)}}
            staged.write_text(json.dumps(manifest))
            verify = root / "verify.json"
            verify.write_text("verification")
            manifest["reconstruction"]["verification_sha256"] = digest(verify)
            staged.write_text(json.dumps(manifest))
            with patch.multiple(delivery, ROOT=root, STAGED=staged, PARENT=parent, FRONTIER=frontier, TRAIN_RECEIPT=seal,
                                RECON_VERIFY=verify), \
                    patch.object(delivery.ball_exit_runtime, "frontier_manifest"), \
                    patch.object(delivery, "load_models", return_value={}):
                delivery.check_manifest()
                src.write_text("changed")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    delivery.check_manifest()

    def test_legacy_gar_adapter_restores_every_changed_global(self):
        old_delivery = {name: getattr(ball_exit_delivery, name) for name in
                        ("STUDY", "OUT", "STAGED", "PARENT", "ACTIVE", "POLICY", "CSV", "SEAL", "GATE", "PUBLIC", "GAR", "RECEIPT", "sources", "check_manifest")}
        old_gar = {name: getattr(ball_exit_gar, name) for name in ("OUT", "RECORDS", "DECL")}
        with delivery.legacy_gar_context():
            self.assertEqual(ball_exit_delivery.STUDY, delivery.STUDY)
            self.assertEqual(ball_exit_gar.OUT, delivery.OUT / "gar")
        self.assertEqual(old_delivery, {name: getattr(ball_exit_delivery, name) for name in old_delivery})
        self.assertEqual(old_gar, {name: getattr(ball_exit_gar, name) for name in old_gar})

    def test_context_does_not_edit_frozen_runner_or_prior_gates(self):
        paths = [Path(ball_exit_delivery.__file__), Path(ball_exit_gar.__file__),
                 delivery.ROOT / "experiments/results/ball_exit_promotion_v1_gar_declaration.json",
                 delivery.ROOT / "experiments/results/ball_exit_promotion_v1_public.json"]
        before = {path: path.read_bytes() for path in paths if path.is_file()}
        with delivery.legacy_gar_context():
            self.assertNotEqual(ball_exit_gar.DECL, paths[2])
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_public_gate_reuse_fails_before_reference_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = Path(tmp) / "gate.json"
            gate.write_text('{"previous":true}')
            with patch.multiple(delivery, GATE=gate, PUBLIC=Path(tmp) / "result.json"), \
                    patch.object(delivery, "check_seal", return_value={"saved_public_baseline_sha256": "x"}), \
                    patch.object(delivery, "write_once") as writer:
                with self.assertRaisesRegex(FileExistsError, "exclusive"):
                    delivery.public()
                writer.assert_not_called()

    def test_activation_requires_reports_and_exact_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parent, active = root / "parent.json", root / "active.json"
            parent.write_text("same provisional baseline")
            active.write_bytes(parent.read_bytes())
            gar, public = root / "gar.json", root / "public.json"
            gate, seal_path = root / "gate.json", root / "seal.json"
            staged, archive = root / "staged.json", root / "archive.json"
            staged.write_text("manifest")
            seal_path.write_text("seal")
            archive.write_text(json.dumps({"manifest_sha256": digest(staged), "models_sha256": {},
                                           "remote_restore_verified": True}))
            gate.write_text("gate")
            with patch.multiple(delivery, PARENT=parent, ACTIVE=active, GAR=gar, PUBLIC=public,
                                GATE=gate, SEAL=seal_path, STAGED=staged, ARCHIVE_VERIFY=archive), \
                    patch.object(delivery, "check_manifest", return_value={}), \
                    patch.object(delivery, "check_seal", return_value={"control_parity_passed": False}):
                with self.assertRaises(FileNotFoundError):
                    delivery.activate()
                gar.write_text(json.dumps({"manifest_sha256": digest(staged), "independent_prediction_replay_passed": True,
                                           "organizer_reference_access": False}))
                public.write_text(json.dumps({"seal_sha256": digest(seal_path), "gate_sha256": digest(gate)}))
                with self.assertRaisesRegex(ValueError, "parity-verified"):
                    delivery.activate()
            self.assertEqual(active.read_bytes(), parent.read_bytes())


if __name__ == "__main__":
    unittest.main()
