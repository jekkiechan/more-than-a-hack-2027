"""Synthetic guards for capacity-study selection and immutable receipts."""
import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np

from experiments import soccernet_capacity as study


def summary(spell=0.5, macro=0.5, d=0.5, feed=0.5):
    per_provider = {
        provider: {"spell_f1": spell, "macro_f1": macro, "d_f1": d}
        for provider in ("sportec", "metrica", "skillcorner")
    }
    return {
        "clean": {"per_provider": per_provider,
                  "equal_provider_mean": {"spell_f1": spell, "macro_f1": macro}},
        "feed_like": {"per_provider": per_provider,
                      "equal_provider_mean": {"spell_f1": feed, "macro_f1": macro}},
    }


class SoccerNetCapacityStudyContracts(unittest.TestCase):
    def test_fit_coarse_validates_and_skips_complete_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher = root / "teacher.joblib"
            teacher.write_bytes(b"teacher")
            path512, path2048 = root / "step512.joblib", root / "step2048.joblib"
            path2048.write_bytes(b"complete")
            validator = Mock(return_value={})
            with patch.object(study, "check", return_value={}), \
                 patch.object(study, "initial_path", return_value=teacher), \
                 patch.object(study, "coarse_path", side_effect=lambda f, w, lr, step: path512 if step == 512 else path2048), \
                 patch.object(study, "_validate_existing_fit", validator), \
                 patch.object(study, "_fold_records") as load_records, \
                 patch.object(study.model, "train") as train:
                study.fit_coarse(study.FOLDS[0], 24, 1e-4)
            self.assertEqual(validator.call_count, 1)
            load_records.assert_not_called()
            train.assert_not_called()

    def test_fit_coarse_resumes_saved_512_state_to_2048(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_path = root / "teacher.joblib"
            teacher_path.write_bytes(b"teacher")
            path512, path2048 = root / "step512.joblib", root / "step2048.joblib"
            path512.write_bytes(b"milestone")
            resume_state = {"continuation": "saved-at-512"}
            fitted = {"median": np.zeros(1), "scale": np.ones(1), "step": 2048}
            trainer = Mock(return_value={2048: fitted})
            with patch.object(study, "check", return_value={}), \
                 patch.object(study, "initial_path", return_value=teacher_path), \
                 patch.object(study, "coarse_path", side_effect=lambda f, w, lr, step: path512 if step == 512 else path2048), \
                 patch.object(study, "_validate_existing_fit", return_value=resume_state), \
                 patch.object(study, "_fold_records", return_value=([], [])), \
                 patch.object(study.joblib, "load", return_value={"median": np.zeros(1), "scale": np.ones(1)}), \
                 patch.object(study.model, "train", trainer), \
                 patch.object(study, "_training_receipt", return_value={}), \
                 patch.object(study, "_save_model") as save_model:
                study.fit_coarse(study.FOLDS[0], 24, 1e-4)
            self.assertEqual(trainer.call_args.kwargs["milestones"], (2048,))
            self.assertIs(trainer.call_args.kwargs["resume_from"], resume_state)
            self.assertEqual(save_model.call_count, 1)
            self.assertEqual(save_model.call_args.args[0], path2048)

    def test_freeze_raw_source_hashes_every_train_file_and_requires_lfs_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = {
                "tracking-full/train/metadata.parquet": b"metadata bytes",
                "tracking-full/train/shard_manifest.parquet": b"shard manifest bytes",
                **{f"tracking-full/train/shards/shard-{index:06d}.tar":
                   f"tar bytes {index}".encode() for index in range(5)},
            }
            entries = []
            for rel, content in payloads.items():
                path = root / rel
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                entries.append({"path": rel, "bytes": len(content),
                    "sha256": study.sha256(path),
                    "verified_lfs_sha256": rel.endswith(".tar")})
            acquisition = root / "acquisition_manifest.json"
            acquisition.write_text(json.dumps({
                "dataset": "OpenSportsLab/SoccerNet-GAR", "branch": "tracking-full",
                "revision": "revision-1", "files": entries,
            }))
            receipt = study._verify_raw_train_source(root)
            self.assertEqual(receipt["tar_count"], 5)
            self.assertEqual(receipt["revision"], "revision-1")
            self.assertIn("acquired_at_utc", receipt)
            self.assertEqual(set(receipt["file_sha256"]), set(payloads))
            tar_path = root / "tracking-full/train/shards/shard-000000.tar"
            tar_path.write_bytes(b"bad bytes 0")
            with self.assertRaisesRegex(ValueError, "content hash differs"):
                study._verify_raw_train_source(root)
            tar_path.write_bytes(b"tar bytes 0")
            tar_entry = next(entry for entry in entries if entry["path"].endswith("000004.tar"))
            tar_entry["verified_lfs_sha256"] = False
            acquisition.write_text(json.dumps({
                "dataset": "OpenSportsLab/SoccerNet-GAR", "branch": "tracking-full",
                "revision": "revision-1", "files": entries,
            }))
            with self.assertRaisesRegex(ValueError, "verified LFS hash"):
                study._verify_raw_train_source(root)

    def test_teacher_provenance_uses_active_pooled_manifest_and_fold_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            package = data / "models/joint_gru_combined_ensemble_v1"
            package.mkdir(parents=True)
            model_hashes = {}
            for seed in study.SEEDS:
                path = package / f"joint_gru_{seed}.joblib"
                path.write_bytes(f"pooled teacher {seed}".encode())
                model_hashes[path.name] = study.sha256(path)
            package_manifest = {"package_version": "joint_gru_combined_ensemble_v1",
                                "models_sha256": model_hashes}
            package_manifest_path = package / "baseline_manifest.json"
            package_manifest_path.write_text(json.dumps(package_manifest))
            active_path = root / "experiments/baseline_manifest.json"
            active_path.parent.mkdir(parents=True)
            active_path.write_text(json.dumps(package_manifest))
            promotion_path = root / "experiments/results/baseline_combined_objective_promotion.json"
            promotion_path.parent.mkdir(parents=True)
            promotion_path.write_text(json.dumps({
                "package": "joint_gru_combined_ensemble_v1",
                "models_sha256": model_hashes,
                "baseline_manifest_sha256": study.sha256(package_manifest_path),
            }))
            plans = {}
            fold_paths = {}
            for fold in study.FOLDS:
                heldout = fold.removeprefix("leave_out_")
                records = [{"match": f"{heldout}-train-a", "source": "sportec"},
                           {"match": f"{heldout}-train-b", "source": "metrica"}]
                plans[fold] = {"train": records}
                for seed in study.SEEDS:
                    path = data / "folds" / fold / f"seed{seed}.joblib"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{fold} seed {seed}".encode())
                    path.with_suffix(".json").write_text(json.dumps({
                        "sha256": study.sha256(path), "heldout": heldout, "seed": seed,
                        "objective": "combined",
                        "training_matches": sorted(item["match"] for item in records),
                        "sources": sorted({item["source"] for item in records}),
                    }))
                    fold_paths[(fold, seed)] = path
            def initial_path(fold, seed):
                if fold == "pooled":
                    return package / f"joint_gru_{seed}.joblib"
                return fold_paths[(fold, seed)]
            with patch.object(study, "ROOT", root), patch.object(study, "DATA", data), \
                 patch.object(study, "initial_path", side_effect=initial_path):
                provenance = study._verify_initial_checkpoint_provenance(plans)
            self.assertEqual(set(provenance["pooled"]), {"2026", "2027"})
            self.assertFalse((package / "joint_gru_2026.json").exists())
            self.assertEqual(len(provenance["fold_checkpoints"]), 6)
            bad_receipt = fold_paths[(study.FOLDS[0], study.SEEDS[0])].with_suffix(".json")
            value = json.loads(bad_receipt.read_text())
            value["heldout"] = "wrong-provider"
            bad_receipt.write_text(json.dumps(value))
            with patch.object(study, "ROOT", root), patch.object(study, "DATA", data), \
                 patch.object(study, "initial_path", side_effect=initial_path):
                with self.assertRaisesRegex(ValueError, "Fold teacher receipt"):
                    study._verify_initial_checkpoint_provenance(plans)

    def test_segment_manifest_contract_matches_prepared_train_schema(self):
        manifest = {
            "schema": "soccernet_observed_segments_v1", "split_used": "train",
            "organizer_reference_access": False, "games": [str(i) for i in range(45)],
            "train_metadata_rows": 62159,
            "counts": {"clips_read": 62159, "anchor_candidates": 42979,
                       "anchor_mapped_known": 42970,
                       "anchor_center_unobserved": 2, "anchor_center_invalid_period": 1,
                       "anchor_center_metadata_flag_conflict": 1,
                       "anchor_unknown_team_conflict": 1,
                       "anchor_center_removed_after_merge": 1,
                       "anchor_out_of_grid_tolerance": 2, "anchor_grid_conflict": 1,
                       "duplicate_rows_exact": 100, "duplicate_rows_conflicting": 0},
            "segment_recipe": {
                "duplicate_policy": "Exact row signature equality across visibility; geometry conflicts exclude the key",
                "breaks": ["uncovered frameNum gap", "period change", "nonmonotonic timestamps",
                           "timestamp delta > 1.5/fps", "invalid/conflicting row"],
                "anchors": "nearest sampled source frame within 3 native frames; conflicting teams at one grid sample remain unknown",
            },
            "eligible_anchors": {"actions": ["CROSS", "PASS", "SHOT"],
                                 "unlabeled_rows": -1, "D_targets": 0},
            "feature_recipe": {"sample_stride": 6, "sample_rate_hz": "fps/6, approx 4.995"},
            "game_files": {},
        }
        audit = study._validate_segment_audit(manifest)
        self.assertEqual(audit["train_games"], 45)
        self.assertEqual(audit["eligible_anchors"], 42979)
        self.assertEqual(audit["mapped_anchors"], 42970)
        self.assertAlmostEqual(audit["mapped_anchor_retention"], 42970 / 42979)
        with tempfile.TemporaryDirectory() as tmp:
            segment_root = Path(tmp)
            game = segment_root / "games/one.joblib"
            game.parent.mkdir()
            game.write_bytes(b"synthetic game payload")
            manifest["game_files"] = {"games/one.joblib": study.sha256(game)}
            with patch.object(study, "SEGMENTS", segment_root):
                self.assertEqual(study._manifest_file_entries(manifest)[0]["path"],
                                 "games/one.joblib")
        manifest["split_used"] = "valid"
        with self.assertRaises(ValueError):
            study._validate_segment_audit(manifest)

    def test_anchor_accounting_rejects_unclassified_or_massive_drops(self):
        # Share the canonical fixture from the prior test without accessing raw data.
        manifest = {
            "schema": "soccernet_observed_segments_v1", "split_used": "train",
            "organizer_reference_access": False, "games": [str(i) for i in range(45)],
            "train_metadata_rows": 62159,
            "counts": {"clips_read": 62159, "anchor_candidates": 42979,
                       "anchor_mapped_known": 100,
                       "anchor_center_unobserved": 0, "anchor_center_invalid_period": 0,
                       "anchor_center_metadata_flag_conflict": 0,
                       "anchor_unknown_team_conflict": 0,
                       "anchor_center_removed_after_merge": 0,
                       "anchor_out_of_grid_tolerance": 0, "anchor_grid_conflict": 0,
                       "duplicate_rows_exact": 0, "duplicate_rows_conflicting": 0},
            "segment_recipe": {
                "duplicate_policy": "Exact row signature equality across visibility; geometry conflicts exclude the key",
                "breaks": ["uncovered frameNum gap", "period change", "nonmonotonic timestamps",
                           "timestamp delta > 1.5/fps", "invalid/conflicting row"],
                "anchors": "nearest sampled source frame within 3 native frames; conflicting teams at one grid sample remain unknown",
            },
            "eligible_anchors": {"actions": ["CROSS", "PASS", "SHOT"],
                                 "unlabeled_rows": -1, "D_targets": 0},
            "feature_recipe": {"sample_stride": 6, "sample_rate_hz": "fps/6"},
        }
        with self.assertRaisesRegex(ValueError, "fully accounted"):
            study._validate_segment_audit(manifest)
        manifest["counts"]["anchor_grid_conflict"] = 42879
        with self.assertRaisesRegex(ValueError, "retention below"):
            study._validate_segment_audit(manifest)

    def test_declared_grid_has_all_capacity_rate_and_budget_combinations(self):
        configs = study.coarse_configs()
        self.assertEqual(len(configs), 12)
        self.assertEqual({row["width"] for row in configs}, {24, 48, 96})
        self.assertEqual({row["lr"] for row in configs}, {1e-4, 3e-4})
        self.assertEqual({row["steps"] for row in configs}, {512, 2048})
        self.assertEqual(len({row["recipe"] for row in configs}), 12)

    def test_coarse_selection_obeys_guards_then_tiebreaks_by_budget_and_rate(self):
        values = {"baseline": summary()}
        for config in study.coarse_configs():
            values[config["recipe"]] = summary(spell=.6, macro=.6)
        # Highest score is ineligible due to the provider clean-D guard.
        chosen = next(row for row in study.coarse_configs()
                      if row["width"] == 48 and row["steps"] == 2048 and row["lr"] == 3e-4)
        values[chosen["recipe"]] = summary(spell=.95, macro=.9, d=.1)
        selected = study.select_coarse_for_width(values, 48)
        self.assertTrue(selected["eligible"])
        self.assertEqual(selected["steps"], 512)
        self.assertEqual(selected["lr"], 1e-4)

    def test_no_guard_eligible_config_is_marked_exploratory(self):
        values = {"baseline": summary()}
        for config in study.coarse_configs():
            values[config["recipe"]] = summary(spell=.6, d=.1, feed=.1)
        selected = study.select_coarse_for_width(values, 24)
        self.assertFalse(selected["eligible"])
        self.assertEqual(selected["selection_mode"], "exploratory_no_guard_eligible_recipe")

    def test_final_width_ties_choose_lower_capacity_and_require_screen(self):
        values = {"baseline": summary(spell=.5)}
        for width in study.WIDTHS:
            values[f"w{width}"] = summary(spell=.509, macro=.6)
        decision = study.select_final_width(values)
        self.assertEqual(decision["selected_width"], 24)
        self.assertTrue(decision["eligible"])
        self.assertFalse(decision["improvement_screen_passes"])
        self.assertFalse(decision["adopt_model"])

    def test_key_contract_rejects_missing_duplicate_or_extra_rows(self):
        expected = {("sportec", "m1", "clean", "clean", "baseline", "2026"),
                    ("metrica", "m2", "clean", "clean", "baseline", "2026")}
        fields = ("source", "match", "view", "replicate", "arm", "member")
        rows = [dict(zip(fields, key)) for key in sorted(expected)]
        study.assert_exact_keys(rows, expected, fields)
        with self.assertRaises(ValueError):
            study.assert_exact_keys(rows[:1], expected, fields)
        with self.assertRaises(ValueError):
            study.assert_exact_keys(rows + [rows[0]], expected, fields)

    def test_match_filter_rejects_unknown_and_selects_one_known_item(self):
        items = [{"source": "sportec", "match": "match-a"},
                 {"source": "metrica", "match": "match-b"}]
        with patch.object(study.old, "selection_items", return_value=items):
            self.assertEqual(study._selection_items_for_match("match-b"), [items[1]])
            self.assertEqual(study._selection_items_for_match(), items)
            with self.assertRaisesRegex(ValueError, "Unknown selection match"):
                study._selection_items_for_match("not-a-match")

    def test_write_once_receipt_is_idempotent_and_rejects_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            study.write_once(path, {"frozen": True, "values": [1, 2]})
            study.write_once(path, {"frozen": True, "values": [1, 2]})
            with self.assertRaises(ValueError):
                study.write_once(path, {"frozen": True, "values": [1, 3]})


if __name__ == "__main__":
    unittest.main()
