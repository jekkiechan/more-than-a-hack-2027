"""Assets-free integrity contracts for the SkillCorner-removal runner."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import skillcorner_removal as removal


MANIFEST = ROOT / "experiments/eval_set/manifest.json"
CANDIDATE = ROOT / "experiments/eval_set/candidates/skillcorner_removal_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _saved_result(temp: Path) -> tuple[Path, dict, dict]:
    out = temp / "out"
    data = temp / "data"
    out.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    declaration = out / "predeclared.json"
    seal = out / "checkpoints_frozen.json"
    declaration.write_text('{"study_id":"synthetic"}\n')
    seal.write_text('{"seal":"synthetic"}\n')
    cache = data / "eval/cache.joblib"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(b"synthetic label-free feature cache")

    identity = ("sportec", "J03WOY", "clean", "clean")
    probability = {
        "current_2026": np.asarray([[.70, .20, .10], [.15, .75, .10]], dtype=np.float64),
        "current_2027": np.asarray([[.60, .30, .10], [.20, .70, .10]], dtype=np.float64),
        "no_skillcorner_2026": np.asarray([[.65, .25, .10], [.10, .80, .10]], dtype=np.float64),
        "no_skillcorner_2027": np.asarray([[.55, .35, .10], [.25, .65, .10]], dtype=np.float64),
    }
    for arm in ("current", "no_skillcorner"):
        probability[f"{arm}_ensemble"] = .5 * (
            probability[f"{arm}_2026"] + probability[f"{arm}_2027"])
        for member in ("2026", "2027", "ensemble"):
            probability[f"{arm}_{member}_prediction"] = np.asarray(["H", "A"])
    result_path = out / "primary_J03WOY_clean_clean.json"
    with result_path.with_suffix(".npz").open("wb") as stream:
        np.savez_compressed(stream, **probability)
    metrics = {"spell_f1": .5, "macro_f1": .7}
    rows = [
        {
            "protocol": "primary", "source": identity[0], "match": identity[1],
            "view": identity[2], "replicate": identity[3], "arm": arm,
            "member": member, "skillcorner_primary_exact_parity": False, **metrics,
        }
        for arm in ("current", "no_skillcorner")
        for member in ("2026", "2027", "ensemble")
    ]
    value = {
        "declaration_sha256": _sha(declaration),
        "candidate_checkpoint_seal_sha256": _sha(seal),
        "cache": "eval/cache.joblib", "cache_sha256": _sha(cache),
        "probability_sha256": _sha(result_path.with_suffix(".npz")), "rows": rows,
    }
    result_path.write_text(json.dumps(value, indent=2) + "\n")
    spec = {
        "evaluation_caches": {
            "sportec/J03WOY/clean/clean": {
                "path": "eval/cache.joblib", "sha256": _sha(cache),
            },
        },
    }
    return result_path, spec, probability


class SkillCornerRemovalIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_manifest = json.loads(MANIFEST.read_text())
        cls.candidate_manifest = json.loads(CANDIDATE.read_text())

    def _read_manifest_from(self, value):
        with tempfile.TemporaryDirectory() as temp_name:
            path = Path(temp_name) / "candidate_manifest.json"
            path.write_text(json.dumps(value))
            with mock.patch.object(removal, "MANIFEST", path):
                return removal.read_manifest()

    def test_manifest_preserves_provider_roles_training_partitions_and_all_five_pooled(self):
        value = self._read_manifest_from(copy.deepcopy(self.candidate_manifest))
        self.assertEqual(removal.training_items("leave_out_sportec")[0]["source"], "metrica")
        self.assertEqual(len(removal.training_items("leave_out_metrica")), 3)
        self.assertEqual(len(removal.training_items("pooled")), 5)
        design = value["skillcorner_removal_v1"]["candidate_folds"]
        self.assertEqual(design["leave_out_sportec"]["selection_source"], "sportec")
        self.assertEqual(design["leave_out_metrica"]["selection_source"], "metrica")
        self.assertEqual(design["leave_out_skillcorner"]["selection_source"], "skillcorner")
        self.assertEqual(design["pooled"]["selection_sources"], list(removal.PROVIDERS))
        self.assertEqual(design["pooled"]["tactical_summary_sources"],
                         list(removal.TACTICAL_PROVIDERS))
        self.assertEqual(len(removal.selection_items()), 5)

    def test_manifest_rejects_source_or_provider_row_tampering(self):
        value = copy.deepcopy(self.candidate_manifest)
        value["sources"]["metrica"]["train"][0]["prior_exposure"] = "changed"
        with self.assertRaisesRegex(ValueError, "source/match records"):
            self._read_manifest_from(value)

    def test_manifest_rejects_provider_specific_selection_role_tampering(self):
        value = copy.deepcopy(self.candidate_manifest)
        value["skillcorner_removal_v1"]["candidate_folds"][
            "leave_out_sportec"]["selection_source"] = "metrica"
        with self.assertRaises(ValueError):
            self._read_manifest_from(value)

    def test_inference_projection_has_no_target_or_label_fields(self):
        record = {
            "x": np.zeros((3, 204), dtype=np.float32), "names": [f"x{i}" for i in range(204)],
            "period": np.asarray([0, 0, 1]), "indices": np.asarray([0, 5, 10]),
            "y": np.asarray([0, 1, 2]), "labels": ["H", "A", "D"],
            "native_labels": ["H"] * 11, "full_y": [0] * 11,
        }
        projected = removal.inference_record(record, native_n=11)
        self.assertEqual(set(projected), {"x", "names", "period", "indices", "native_n"})
        self.assertEqual(projected["native_n"], 11)
        self.assertTrue(np.array_equal(projected["x"], record["x"]))
        before = {key: copy.deepcopy(value) for key, value in projected.items()}
        record["y"][:] = 2
        record["labels"][:] = "D"
        record["native_labels"][:] = "D"
        record["full_y"] = [2] * len(record["full_y"])
        after = removal.inference_record(record, native_n=11)
        self.assertEqual(set(after), set(before))
        for key in before:
            if isinstance(before[key], np.ndarray):
                self.assertTrue(np.array_equal(after[key], before[key]))
            else:
                self.assertEqual(after[key], before[key])

    def test_declaration_hashes_paths_but_keeps_identity_maps_separate(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            data = temp / "data"
            data.mkdir()

            def asset(relative, payload):
                path = data / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                return _sha(path)

            input_sha = asset("inputs/train.npz", b"train input")
            train_cache_sha = asset("cache/train.joblib", b"train cache")
            train_receipt_sha = asset("cache/train.json", b"train receipt")
            eval_cache_sha = asset("eval/features.joblib", b"evaluation cache")
            checkpoint_sha = asset("controls/model.joblib", b"control checkpoint")
            probability_sha = asset("controls/probability.npz", b"control probability")
            control_receipt_sha = asset("controls/model.json", b"control receipt")
            package_sha = asset("controls/package.json", b"package manifest")

            fixed_files = {}
            for name in ("manifest.json", "old_manifest.json", "baseline.json",
                         "baseline_package.json", "policy.json", "deployment.json",
                         "baseline_rows.json", "predeclared.json", "checkpoint_manifest.json"):
                path = temp / name
                path.write_text(name)
                fixed_files[name] = path
            package_manifest = temp / "baseline_manifest.json"
            package_manifest.write_text("synthetic package manifest")
            fixed_files["package_manifest"] = package_manifest

            spec = {
                "study_id": "skillcorner_removal_v1",
                "gates": removal.GATES,
                "source_sha256": {"experiments/runner.py": "source-hash"},
                "split_manifest_sha256": "manifest-hash",
                "canonical_split_manifest_sha256": "old-manifest-hash",
                "baseline_manifest_sha256": "baseline-hash",
                "baseline_package_manifest_sha256": "package-hash",
                "decision_policy_sha256": "policy-hash",
                "deployment_reference_sha256": "deployment-hash",
                "baseline_rows_sha256": "rows-hash",
                "objective_alignment_predeclared_sha256": "predeclared-hash",
                "objective_alignment_checkpoint_manifest_sha256": "checkpoint-manifest-hash",
                "dependency_versions": {"python": "synthetic"},
                "input_sha256": {"inputs/train.npz": input_sha},
                "training_cache_sha256": {"cache/train.joblib": train_cache_sha},
                "training_receipt_sha256": {"cache/train.json": train_receipt_sha},
                "evaluation_cache_sha256": {"eval/features.joblib": eval_cache_sha},
                "control_checkpoint_sha256": {"controls/model.joblib": checkpoint_sha},
                "control_probability_sha256": {"controls/probability.npz": probability_sha},
                "evaluation_caches": {
                    "sportec/J03WOY/clean/clean": {
                        "path": "eval/features.joblib", "sha256": eval_cache_sha,
                    },
                },
                "control_checkpoints": {
                    "leave_out_sportec:2026": {
                        "path": "controls/model.joblib", "sha256": checkpoint_sha,
                        "receipt": "controls/model.json", "receipt_sha256": control_receipt_sha,
                        "package_manifest": "controls/package.json",
                        "package_manifest_sha256": package_sha,
                    },
                },
            }
            patched_hashes = {
                fixed_files["manifest.json"]: "manifest-hash",
                fixed_files["old_manifest.json"]: "old-manifest-hash",
                fixed_files["baseline.json"]: "baseline-hash",
                fixed_files["package_manifest"]: "package-hash",
                fixed_files["policy.json"]: "policy-hash",
                fixed_files["deployment.json"]: "deployment-hash",
                fixed_files["baseline_rows.json"]: "rows-hash",
                fixed_files["predeclared.json"]: "predeclared-hash",
                fixed_files["checkpoint_manifest.json"]: "checkpoint-manifest-hash",
            }
            real_sha = removal.sha

            def synthetic_sha(path):
                path = Path(path)
                if path in patched_hashes:
                    return patched_hashes[path]
                return real_sha(path)

            patches = [
                mock.patch.object(removal, "DATA", data),
                mock.patch.object(removal, "MANIFEST", fixed_files["manifest.json"]),
                mock.patch.object(removal, "OLD_MANIFEST", fixed_files["old_manifest.json"]),
                mock.patch.object(removal, "BASELINE_MANIFEST", fixed_files["baseline.json"]),
                mock.patch.object(removal, "BASELINE_PACKAGE",
                                  fixed_files["baseline_package.json"].parent),
                mock.patch.object(removal, "DECISION_POLICY", fixed_files["policy.json"]),
                mock.patch.object(removal, "DEPLOYMENT_REFERENCE", fixed_files["deployment.json"]),
                mock.patch.object(removal, "BASELINE_ROWS", fixed_files["baseline_rows.json"]),
                mock.patch.object(removal, "TRAIN_PREDECLARED", fixed_files["predeclared.json"]),
                mock.patch.object(removal, "TRAIN_CHECKPOINTS",
                                  fixed_files["checkpoint_manifest.json"]),
                mock.patch.object(removal, "source_hashes", return_value=spec["source_sha256"]),
                mock.patch.object(removal, "_dependency_versions",
                                  return_value=spec["dependency_versions"]),
                mock.patch.object(removal, "read_manifest", return_value={}),
                mock.patch.object(removal, "sha", side_effect=synthetic_sha),
            ]
            with patches[0]:
                with patches[1], patches[2], patches[3], patches[4], patches[5], patches[6], \
                        patches[7], patches[8], patches[9], patches[10], patches[11], \
                        patches[12], patches[13]:
                    self.assertIs(removal._validate_declaration(spec), spec)

            # Identity keys are deliberately not filesystem paths and must not
            # be fed to a generic data-relative hash loop.
            self.assertEqual(list(spec["evaluation_caches"]), ["sportec/J03WOY/clean/clean"])
            self.assertEqual(list(spec["control_checkpoints"]), ["leave_out_sportec:2026"])

    def test_evaluation_cache_uses_source_match_view_replicate_identity(self):
        with tempfile.TemporaryDirectory() as temp_name:
            data = Path(temp_name)
            cache_path = data / "features/sportec_J03WOY_clean.joblib"
            cache_path.parent.mkdir()
            cache_path.write_bytes(b"synthetic evaluation feature cache")
            record = {
                "source": "sportec", "match": "J03WOY",
                "x": np.zeros((3, 204), dtype=np.float32),
                "y": np.asarray([0, 1, 2]), "indices": np.asarray([0, 1, 2]),
            }
            cached = {
                "record": record,
                "inputs": {"period": np.asarray([0, 0, 0]),
                           "timestamps": np.asarray([0., .04, .08])},
                "labels": np.asarray(["H", "A", "D"]),
            }
            key = "sportec/J03WOY/clean/clean"
            spec = {"evaluation_caches": {key: {
                "path": "features/sportec_J03WOY_clean.joblib",
                "sha256": _sha(cache_path),
            }}}
            with mock.patch.object(removal, "DATA", data), mock.patch.object(
                    removal.joblib, "load", return_value=cached):
                path, inputs, labels, actual_record = removal.evaluation_cache(
                    {"source": "sportec", "match": "J03WOY"}, "clean", "clean", spec)
            self.assertEqual(path, cache_path)
            self.assertEqual(len(labels), 3)
            self.assertEqual(actual_record["source"], "sportec")
            with mock.patch.object(removal, "DATA", data), mock.patch.object(
                    removal.joblib, "load", return_value=cached):
                with self.assertRaisesRegex(ValueError, "outside the frozen original-five cohort"):
                    removal.evaluation_cache(
                        {"source": "metrica", "match": "J03WOY"},
                        "clean", "clean", spec)

    def test_checkpoint_seal_uses_fold_seed_control_identity_keys(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            out, data = temp / "out", temp / "data"
            out.mkdir()
            data.mkdir()
            declaration = out / "predeclared.json"
            declaration.write_text("synthetic declaration\n")
            candidate_entries = []
            for fold in removal.FIT_FOLDS:
                for seed in removal.SEEDS:
                    relative = f"candidates/{fold}_{seed}.joblib"
                    receipt_relative = f"candidates/{fold}_{seed}.json"
                    for path, payload in ((data / relative, b"candidate model"),
                                          (data / receipt_relative, b"candidate receipt")):
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(payload)
                    candidate_entries.append({"fold": fold, "seed": seed,
                                              "path": relative,
                                              "sha256": _sha(data / relative),
                                              "receipt": receipt_relative,
                                              "receipt_sha256": _sha(data / receipt_relative)})
            controls = {}
            for fold in removal.ALL_FOLDS:
                for seed in removal.SEEDS:
                    relative = f"controls/{fold}_{seed}.joblib"
                    path = data / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"saved control checkpoint")
                    controls[f"{fold}:{seed}"] = {
                        "path": relative, "sha256": _sha(path),
                    }
            seal = out / "checkpoints_frozen.json"
            seal.write_text(json.dumps({
                "declaration_sha256": _sha(declaration),
                "checkpoints": candidate_entries,
            }))
            spec = {"control_checkpoints": controls}
            with mock.patch.object(removal, "OUT", out), mock.patch.object(
                    removal, "DATA", data), mock.patch.object(
                    removal, "check", return_value=spec):
                verified = removal._check_seal()
            self.assertEqual(len(verified["checkpoints"]), 6)
            self.assertEqual(set(controls), {
                f"{fold}:{seed}" for fold in removal.ALL_FOLDS for seed in removal.SEEDS
            })

    def test_saved_result_replays_identities_hashes_and_probability_mean(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            result_path, spec, _ = _saved_result(temp)
            data = temp / "data"
            inputs = {"fps": 25.0, "period": np.asarray([0, 0])}
            labels = np.asarray(["H", "A"])
            item = {"source": "sportec", "match": "J03WOY"}
            fake_prediction = np.asarray(["H", "A"])
            with (
                mock.patch.object(removal, "OUT", temp / "out"),
                mock.patch.object(removal, "DATA", data),
                mock.patch.object(removal, "selection_items", return_value=[item]),
                mock.patch.object(removal, "evaluation_cache",
                                  return_value=(data / "eval/cache.joblib", inputs, labels, {})),
                mock.patch.object(removal.expansion, "score_cell",
                                  return_value=({"spell_f1": .5, "macro_f1": .7}, fake_prediction)),
            ):
                rows = removal.validate_result(result_path, spec)
            self.assertEqual(len(rows), 6)

    def test_saved_result_rejects_mean_hash_and_provider_tampering(self):
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            result_path, spec, arrays = _saved_result(temp)
            data, out = temp / "data", temp / "out"
            inputs = {"fps": 25.0, "period": np.asarray([0, 0])}
            labels = np.asarray(["H", "A"])
            item = {"source": "sportec", "match": "J03WOY"}
            fake_prediction = np.asarray(["H", "A"])
            patches = (
                mock.patch.object(removal, "OUT", out),
                mock.patch.object(removal, "DATA", data),
                mock.patch.object(removal, "selection_items", return_value=[item]),
                mock.patch.object(removal, "evaluation_cache",
                                  return_value=(data / "eval/cache.joblib", inputs, labels, {})),
                mock.patch.object(removal.expansion, "score_cell",
                                  return_value=({"spell_f1": .5, "macro_f1": .7}, fake_prediction)),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                value = json.loads(result_path.read_text())
                probability_path = result_path.with_suffix(".npz")
                arrays["current_ensemble"] = arrays["current_ensemble"] + .01
                with probability_path.open("wb") as stream:
                    np.savez_compressed(stream, **arrays)
                value["probability_sha256"] = _sha(probability_path)
                result_path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "probability mean"):
                    removal.validate_result(result_path, spec)

                # Recreate a valid cell and then exercise each receipt identity.
                result_path, spec, _ = _saved_result(temp)
                value = json.loads(result_path.read_text())
                value["declaration_sha256"] = "0" * 64
                result_path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "another declaration"):
                    removal.validate_result(result_path, spec)

                result_path, spec, _ = _saved_result(temp)
                value = json.loads(result_path.read_text())
                value["cache_sha256"] = "0" * 64
                result_path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "cache changed"):
                    removal.validate_result(result_path, spec)

                result_path, spec, _ = _saved_result(temp)
                value = json.loads(result_path.read_text())
                value["rows"][0]["source"] = "metrica"
                result_path.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "row source/provider/match"):
                    removal.validate_result(result_path, spec)

    def test_five_match_control_parity_and_tactical_equal_provider_summary(self):
        five = [
            ("sportec", "J03WOY", [.50]),
            ("metrica", "metrica_3", [.60]),
            ("skillcorner", "skillcorner_had_1996435", [.10]),
            ("skillcorner", "skillcorner_had_1996436", [.30]),
            ("skillcorner", "skillcorner_had_2006229", [.50]),
        ]
        primary_rows = []
        baseline_rows = []
        pooled_rows = []
        for source, match, (spell,) in five:
            baseline_rows.append({
                "source": source, "match": match, "view": "clean", "replicate": "clean",
                "recipe": "combined_objective_equal_probability_ensemble",
                "spell_f1": spell, "macro_f1": .7, "d_f1": .6,
                "n_gt_spells": 4, "n_pred_spells": 5, "graded_frames": 100,
            })
            primary_rows.append({
                "protocol": "primary", "arm": "current", "member": "ensemble",
                "source": source, "match": match, "view": "clean", "replicate": "clean",
                **{key: baseline_rows[-1][key] for key in (
                    "spell_f1", "macro_f1", "d_f1", "n_gt_spells", "n_pred_spells",
                    "graded_frames")},
            })
            pooled_rows.append({
                "protocol": "pooled", "arm": "no_skillcorner", "member": "ensemble",
                "source": source, "match": match, "view": "clean", "replicate": "clean",
                "spell_f1": spell, "macro_f1": .7, "d_f1": .6,
                "d_precision": .6, "d_recall": .6, "weighted_70_30": .65,
                "short_spell_recall": .5, "n_gt_spells": 4, "n_pred_spells": 5,
                "short_reference_spells": 2, "fragmented_reference_spells": 1,
                "graded_frames": 100,
                "spell_by_iou": {str(value): spell for value in (.3, .4, .5, .6, .7)},
            })

        with mock.patch.object(removal, "read", return_value={"rows": baseline_rows}):
            removal._primary_control_parity(primary_rows)
        summary = removal.summarize(
            pooled_rows, "pooled", "no_skillcorner", "clean", "ensemble",
            sources=removal.TACTICAL_PROVIDERS)
        provider_means = summary["providers"]
        self.assertEqual(set(provider_means), {"sportec", "metrica"})
        self.assertAlmostEqual(provider_means["sportec"]["spell_f1"], .5)
        self.assertAlmostEqual(provider_means["metrica"]["spell_f1"], .6)
        self.assertAlmostEqual(summary["equal_provider_mean"]["spell_f1"], .55)

    def test_gate_readout_applies_frozen_direction_and_boundaries(self):
        summaries = {
            "primary:all:clean:ensemble": {"delta": {"spell_f1": .01}},
            "primary:all:feed_like:ensemble": {"delta": {"spell_f1": -.01}},
            "primary:all:clean:ensemble": {
                "delta": {"spell_f1": .01},
                "provider_delta": {source: {"macro_f1": -.02, "d_f1": -.02}
                                    for source in removal.PROVIDERS},
            },
            "pooled:tactical:clean:ensemble": {
                "delta": {"spell_f1": 1e-6},
                "provider_delta": {source: {"macro_f1": -.02, "d_f1": -.02}
                                    for source in removal.TACTICAL_PROVIDERS},
            },
            "pooled:tactical:feed_like:ensemble": {"delta": {"spell_f1": -.01}},
        }
        result = removal._gate_readout(summaries)
        self.assertTrue(result["removal_recommendation_passes"])
        self.assertFalse(result["automatic_promotion"])
        summaries["pooled:tactical:clean:ensemble"]["delta"]["spell_f1"] = 0.0
        self.assertFalse(removal._gate_readout(summaries)["removal_recommendation_passes"])


if __name__ == "__main__":
    unittest.main()
