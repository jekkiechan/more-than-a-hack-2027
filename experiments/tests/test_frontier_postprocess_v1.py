from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import frontier_postprocess_v1 as study
from experiments import v2_spell_frontier_eval as frontier


class FrontierPostprocessContracts(unittest.TestCase):
    def test_phase_pools_keep_paired_seed_weighting_before_phase_mean(self):
        phases = {2026: [np.array([[.8, .1, .1]]), np.array([[.2, .7, .1]])],
                  2027: [np.array([[.1, .8, .1]]), np.array([[.7, .2, .1]])]}
        actual = study.pool_family(phases, .25)
        expected = np.mean([.25 * phases[2026][i] + .75 * phases[2027][i]
                            for i in range(2)], axis=0)
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        self.assertEqual(actual.shape, (1, 3))

    def test_family_recipe_calls_the_frozen_conditional_ha_mixture(self):
        components = {
            "ce": {2026: [np.array([[.7, .2, .1]])] * 2, 2027: [np.array([[.2, .7, .1]])] * 2},
            "calibrated": {2026: [np.array([[.4, .5, .1]])] * 2, 2027: [np.array([[.8, .1, .1]])] * 2},
            "predecessor": {2026: [np.array([[.5, .4, .1]])] * 2, 2027: [np.array([[.3, .6, .1]])] * 2},
        }
        cell = {"components": components}
        pooled = {name: study.pool_family(value, .5) for name, value in components.items()}
        expected = study.balanced.recipe_probability({"ce": pooled["ce"],
            "calibrated": pooled["calibrated"], "adopted": pooled["predecessor"]},
            .5, .375, "arithmetic")
        np.testing.assert_allclose(study.probability_for_weight(cell, .5), expected, atol=1e-15)
        self.assertEqual(expected.shape, (1, 2))
        np.testing.assert_allclose(expected.sum(axis=1), 1., atol=1e-15)

    def test_decoder_is_the_original_fixed_d_call_without_x_rewriting(self):
        p = np.array([[.8, .2], [.4, .6], [.7, .3]])
        predecessor = np.array(["H", "X", "D"])
        period = np.array([1, 1, 1])
        with patch.object(study.balanced, "decode_fixed_d", return_value=np.array(["H", "A", "D"])) as decode:
            actual = study.decode_candidate(p, predecessor, period, 25.)
        decode.assert_called_once_with(p, predecessor, period, 25., 3.0)
        np.testing.assert_array_equal(actual, ["H", "A", "D"])

    def test_frontier_manifest_requires_exact_recipe_artifacts_and_35_cells(self):
        artifact_paths = {"declaration": frontier.DECLARATION, "result": frontier.RESULT,
                          "replay": frontier.REPLAY, "selection": study.FRONTIER_SELECTION}
        artifacts = {name: {"path": str(path.relative_to(study.ROOT)),
                            "sha256": study.sha256(path)} for name, path in artifact_paths.items()}
        cells = {}
        for key in study.EXPECTED_CELLS:
            identity = {"cell": key, "source_status": "reconstructed",
                "frontier_study_id": frontier.STUDY,
                "frontier_declaration_sha256": artifacts["declaration"]["sha256"],
                "frontier_result_sha256": artifacts["result"]["sha256"],
                "frontier_replay_sha256": artifacts["replay"]["sha256"],
                "selection_sha256": artifacts["selection"]["sha256"]}
            cells[key] = {"path": "data/experiments/cache.npz", "sha256": "a"*64,
                          "phase_count": 5, "identity": identity}
        manifest = {"schema": "frontier_postprocess_v1_cache_manifest",
            "source_status": "reconstructed", "reconstruction_provenance": "synthetic contract fixture",
            "frontier_study_id": frontier.STUDY, "frontier_artifacts": artifacts,
            "baseline_recipe": {"recipe_id": study.BASELINE_RECIPE_ID, "ce_weight": .5,
                "calibrated_weight": .375, "adopted_weight": .125, "pooling": "arithmetic",
                "ha_cost": 3.0, "training_arm": "base", "training_stage": 1024},
            "family_weights": study.FAMILY_WEIGHTS, "seeds": list(study.SEEDS),
            "expected_cells": sorted(study.EXPECTED_CELLS), "cells": cells,
            "source_hashes": {"experiments/frontier_postprocess_v1.py": "b"*64},
            "baseline_metrics": study.BASELINE_METRICS}
        study._validate_native_manifest(manifest)
        manifest["cells"].pop(next(iter(manifest["cells"])))
        with self.assertRaisesRegex(ValueError, "35 frozen frontier cells"):
            study._validate_native_manifest(manifest)

    def test_source_hash_inventory_rehashes_actual_source_files(self):
        hashes = study.source_hashes()
        study.verify_source_hashes(hashes)
        first = next(iter(hashes))
        hashes[first] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed frozen source"):
            study.verify_source_hashes(hashes)

    def test_freeze_binds_all_arms_sources_and_unchanged_native_manifest(self):
        artifact_paths = {"declaration": frontier.DECLARATION, "result": frontier.RESULT,
                          "replay": frontier.REPLAY, "selection": study.FRONTIER_SELECTION}
        artifacts = {name: {"path": str(path.relative_to(study.ROOT)),
                            "sha256": study.sha256(path)} for name, path in artifact_paths.items()}
        recipe = {"recipe_id": study.BASELINE_RECIPE_ID, "ce_weight": .5,
            "calibrated_weight": .375, "adopted_weight": .125, "pooling": "arithmetic",
            "ha_cost": 3.0, "training_arm": "base", "training_stage": 1024}
        with tempfile.TemporaryDirectory(dir=study.ROOT) as temp:
            base = Path(temp)
            cache = base / "cell-cache.npz"
            cache.write_bytes(b"synthetic cache receipt")
            cache_hash = study.sha256(cache)
            cells = {}
            for key in study.EXPECTED_CELLS:
                cells[key] = {"path": str(cache.relative_to(study.ROOT)), "sha256": cache_hash,
                    "phase_count": 5, "identity": {"cell": key, "source_status": "reconstructed",
                    "frontier_study_id": frontier.STUDY,
                    "frontier_declaration_sha256": artifacts["declaration"]["sha256"],
                    "frontier_result_sha256": artifacts["result"]["sha256"],
                    "frontier_replay_sha256": artifacts["replay"]["sha256"],
                    "selection_sha256": artifacts["selection"]["sha256"]}}
            native = {"schema": "frontier_postprocess_v1_cache_manifest",
                "source_status": "reconstructed", "reconstruction_provenance": "synthetic fixture",
                "frontier_study_id": frontier.STUDY, "frontier_artifacts": artifacts,
                "baseline_recipe": recipe, "family_weights": study.FAMILY_WEIGHTS,
                "seeds": list(study.SEEDS), "expected_cells": sorted(study.EXPECTED_CELLS),
                "cells": cells, "source_hashes": study.source_hashes(),
                "baseline_metrics": study.BASELINE_METRICS}
            native_path = base / "native.json"
            native_path.write_text(json.dumps(native))
            evaluation_path, declaration_path = base / "evaluation.json", base / "declaration.json"
            evaluation, declaration = study.freeze(native_path, evaluation_path, declaration_path)
            study.validate_manifest(evaluation)
            study._verify_declaration(evaluation_path, declaration_path, evaluation)
            self.assertEqual(declaration["arm_ids"], list(study.ARM_IDS))
            self.assertEqual(declaration["arm_settings"], study.arm_settings())
            self.assertEqual(declaration["native_manifest"]["sha256"], study.sha256(native_path))
            self.assertFalse(declaration["promotion"])
            altered = json.loads(declaration_path.read_text())
            altered["arm_settings"]["seed2026_0.25"]["seed2026_weight"] = .2
            declaration_path.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError, "exact evaluation manifest"):
                study._verify_declaration(evaluation_path, declaration_path, evaluation)

    def test_cell_metric_comparison_uses_tight_float_and_exact_counts(self):
        saved = {"spell_f1": .5936187198, "graded_frames": 199319,
                 "spell_by_iou": {"0.3": .7}}
        close = {"spell_f1": .59361871981, "graded_frames": 199319,
                 "spell_by_iou": {"0.3": .70000000001}}
        study._assert_metric_row(close, saved, "fixture")
        close["graded_frames"] += 1
        with self.assertRaisesRegex(ValueError, "count differs"):
            study._assert_metric_row(close, saved, "fixture")


if __name__ == "__main__":
    unittest.main()
