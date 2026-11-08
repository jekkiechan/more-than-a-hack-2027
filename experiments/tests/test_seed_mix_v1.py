from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import seed_mix_v1 as study


class SeedMixContractTests(unittest.TestCase):
    def manifest(self, cache: Path):
        key = "Metrica|metrica_3|clean|0"
        source = {"kind": "historical_control", "study_id": "original_full_phase_pilot",
                  "source_sha256": "a" * 64}
        identity = {"cell": key, "source_study_id": source["study_id"], "input_sha256": "b" * 64,
                    "calibration_sha256": "e" * 64, "model_manifest_sha256": "f" * 64}
        return {"schema": "seed_mix_v1_cache_manifest", "source": source,
                "evidence_scope": "historical original_full phase pilot; clean control cell",
                "pooling_order": study.POOLING_ORDER, "seeds": list(study.SEEDS),
                "source_hashes": {"experiments/seed_mix_v1.py": "d" * 64},
                "expected_cells": [key], "cells": {key: {"path": str(cache),
                    "sha256": study.sha256(cache), "identity": identity, "phase_count": 2}},
                "expected_control_spell_f1": {"clean": .5},
                "expected_control_macro_f1": {"clean": .5},
                "weights": list(study.WEIGHTS),
                "cohort": {"matches": [["Metrica", "metrica_3"]],
                           "view_replicates": {"clean": [0]},
                           "view_groups": {"clean": ["clean"], "feed_like_v1": ["feed_like_v1"],
                                           "feed_like_v2": ["feed_like_v2"]}}}

    def save_cache(self, path: Path, *, include_second_phase=True):
        # Different phases and seeds make pooling-order mistakes observable.
        high = np.array([[.8, .1, .1], [.1, .8, .1]])
        low = np.array([[.1, .8, .1], [.8, .1, .1]])
        content = {"prob__candidate__seed2026__phase0": high,
                   "prob__candidate__seed2027__phase0": low,
                   "prob__adopted": high,
                   "prob__historical_ensemble": high,
                   "decoded__historical_f062": np.array(["H", "D"]),
                   "labels": np.array(["H", "D"]), "period": np.array([0, 0]),
                   "timestamps": np.array([0., .04]), "model_period": np.array([0, 0]),
                   "fps": np.array(25.)}
        if include_second_phase:
            content.update({"prob__candidate__seed2026__phase1": low,
                            "prob__candidate__seed2027__phase1": high})
        identity = {"cell": "Metrica|metrica_3|clean|0",
                    "source_study_id": "original_full_phase_pilot", "input_sha256": "b" * 64,
                    "calibration_sha256": "e" * 64, "model_manifest_sha256": "f" * 64}
        content.update({f"identity__{key}": np.asarray(value) for key, value in identity.items()})
        np.savez_compressed(path, **content)

    def test_fixed_grid_and_equal_seed_mix(self):
        self.assertEqual(study.validate_weights(), (0., .25, .5, .75, 1.))
        left = np.array([[.7, .2, .1]])
        right = np.array([[.2, .7, .1]])
        np.testing.assert_allclose(study.seed_mixture({2026: left, 2027: right}, .5),
                                   [[.45, .45, .1]])
        with self.assertRaises(ValueError):
            study.seed_mixture({2026: left}, .5)
        with self.assertRaises(ValueError):
            study.validate_weights((.5, float("nan")))

    def test_source_hashes_bind_real_paths(self):
        hashes = study._source_hashes()
        study.verify_source_hashes(hashes)
        relative = next(iter(hashes))
        hashes[relative] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            study.verify_source_hashes(hashes)

    def test_phase_pooling_weights_within_phase_before_equal_phase_mean(self):
        a0 = np.array([[.8, .1, .1]])
        b0 = np.array([[.1, .8, .1]])
        a1, b1 = b0.copy(), a0.copy()
        result = study.pool_phases({2026: [a0, a1], 2027: [b0, b1]}, .75)
        expected = np.mean([.75 * a0 + .25 * b0, .75 * a1 + .25 * b1], axis=0)
        np.testing.assert_allclose(result, expected, atol=1e-15)
        self.assertEqual(result.shape, (1, 3))

    def test_manifest_rejects_partial_cohort_and_organizer_or_test_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "cell.npz"
            self.save_cache(path)
            manifest = self.manifest(path)
            with self.assertRaisesRegex(ValueError, "all five"):
                # A single-cell declaration cannot masquerade as the frozen 35-cell cohort.
                study.validate_manifest(manifest)
            with self.assertRaisesRegex(ValueError, "Test or organizer"):
                study._safe_cache_path(Path(temp), "evaluation/gt/labels.npz")

    def test_cache_requires_complete_phase_seed_pair_and_exact_identity_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "cell.npz"
            self.save_cache(path, include_second_phase=False)
            manifest = self.manifest(path)
            key, fact = next(iter(manifest["cells"].items()))
            with self.assertRaisesRegex(ValueError, "Missing paired seed phase"):
                study.load_cell(root, key, fact, manifest["source"]["study_id"])
            self.save_cache(path)
            fact["sha256"] = hashlib.sha256(b"tampered").hexdigest()
            with self.assertRaisesRegex(ValueError, "changed native probability"):
                study.load_cell(root, key, fact, manifest["source"]["study_id"])

    def test_predecessor_prediction_is_a_full_native_had_mask(self):
        probs = np.array([[.6, .3, .1], [.2, .7, .1]])
        bad = np.array([[.6, .4], [.3, .7]])
        with self.assertRaises(ValueError):
            study.seed_mixture({2026: probs, 2027: bad}, .5)

    def test_unknown_x_is_decoded_as_a_break_and_restored(self):
        prob = np.tile([[.7, .2, .1]], (5, 1))
        predecessor = np.array(["H", "X", "A", "D", "H"])
        captured = {}
        def decoder(_prob, mask, _period, _fps, _cost):
            captured["mask"] = mask.copy()
            return np.array(["H", "A", "H", "D", "H"])
        with patch.object(study.scoring, "decode_fixed_d", decoder):
            output = study.decode_candidate(prob, predecessor, np.ones(5), 25.)
        self.assertEqual(captured["mask"][1], "D")
        np.testing.assert_array_equal(output, ["H", "X", "H", "D", "H"])

    def test_synthetic_35_cell_freeze_preflight_and_run_bind_all_inputs(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as temp:
            root = Path(temp)
            pairs = [(provider, match) for provider, match, _ in study.preparation.audit.MATCHES]
            view_replicates = {"clean": [0], "feed_like_v1": [1, 2],
                               "feed_like_v2": [1, 2, 3, 4]}
            view_groups = {"clean": ["clean"], "feed_like_v1": ["feed_like_v1"],
                           "feed_like_v2": ["feed_like_v2"]}
            cells, keys = {}, []
            prob = np.array([[.8, .1, .1], [.1, .1, .8], [.1, .8, .1]])
            predecessor = np.array(["H", "D", "A"])
            labels = predecessor.copy()
            for provider, match in pairs:
                for view, reps in view_replicates.items():
                    for rep in reps:
                        key = f"{provider}|{match}|{view}|{rep}"
                        keys.append(key)
                        path = root / (key.replace("|", "__") + ".npz")
                        arrays = {f"prob__candidate__seed{seed}__phase0": prob for seed in study.SEEDS}
                        arrays["prob__historical_ensemble"] = prob
                        arrays.update({"decoded__historical_f062": predecessor,
                            "labels": labels, "period": np.array([1, 1, 1]),
                            "timestamps": np.array([0., .04, .08]),
                            "model_period": np.array([1, 1, 1]), "fps": np.array(25.)})
                        arrays.update({f"identity__{name}": np.asarray(str(value))
                                       for name, value in {"cell": key,
                                           "source_study_id": "original_full_phase_ensemble_v1",
                                           "input_sha256": "a"*64,
                                           "calibration_sha256": "b"*64,
                                           "model_manifest_sha256": "c"*64}.items()})
                        np.savez_compressed(path, **arrays)
                        identity = {"cell": key, "source_study_id": "original_full_phase_ensemble_v1",
                            "provider": provider, "match": match, "view": view, "rep": rep,
                            "input_sha256": "a"*64, "calibration_sha256": "b"*64,
                            "model_manifest_sha256": "c"*64}
                        cells[key] = {"path": str(path.relative_to(study.ROOT)),
                            "sha256": study.sha256(path), "identity": identity, "phase_count": 1}
            preparation = {"scoring_performed": False, "source_hashes": study._source_hashes(),
                           "cells": cells}
            preparation_path = root / "preparation.json"
            preparation_path.write_text(json.dumps(preparation))
            manifest_path, declaration_path = root / "manifest.json", root / "declaration.json"
            manifest, declaration = study.freeze(preparation_path, manifest_path, declaration_path)
            baseline_row = {metric: .5 for metric in study.METRICS}
            baseline_row["reference_spell_count"] = 2
            def fake_score(cell, prediction):
                row = {metric: .5 for metric in study.METRICS}
                row["spell_f1"] = {"clean": .6675555328266306,
                    "feed_like_v1": .5315755112553304,
                    "feed_like_v2": .5835712076382747}[cell["view"]]
                row["macro_f1"] = {"clean": .8712713689886137,
                    "feed_like_v1": .826852823195384,
                    "feed_like_v2": .8499746497420088}[cell["view"]]
                row["reference_spell_count"] = 2
                return row
            with ExitStack() as stack:
                stack.enter_context(patch.object(study, "decode_candidate", lambda p, y, *a: y.copy()))
                stack.enter_context(patch.object(study.scoring.controls.audit, "score", fake_score))
                stack.enter_context(patch.object(study.boundary, "relocate", lambda p, y, *a: y.copy()))
                receipt = study.preflight(manifest_path, declaration_path)
                receipt_path = root / "preflight.json"
                receipt_path.write_text(json.dumps(receipt))
                tampered = json.loads(json.dumps(receipt))
                tampered["control_preflight"]["spell_f1"]["clean"] += .01
                receipt_path.write_text(json.dumps(tampered))
                with self.assertRaisesRegex(ValueError, "immutable preflight receipt"):
                    study.run(manifest_path, declaration_path, receipt_path)
                receipt_path.write_text(json.dumps(receipt))
                result = study.run(manifest_path, declaration_path, receipt_path,
                                   prediction_dir=root / "predictions")
            self.assertTrue(receipt["control_preflight"]["passed"])
            self.assertEqual(len(result["cell_metrics"]), 5)
            self.assertEqual(len(result["boundary"]["cell_metrics"]), 3)
            self.assertEqual(len(result["prediction_inventory"]), 35)
            self.assertEqual(len(list((root / "predictions").glob("*.npz"))), 35)


if __name__ == "__main__":
    unittest.main()
