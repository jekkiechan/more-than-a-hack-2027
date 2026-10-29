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
