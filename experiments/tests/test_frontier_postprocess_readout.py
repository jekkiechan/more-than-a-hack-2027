from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import frontier_postprocess_readout as readout
from experiments import frontier_postprocess_v1 as runner


class FrontierPostprocessReadoutTests(unittest.TestCase):
    def test_exact_d_and_x_masks_reject_added_or_removed_rows(self):
        baseline = np.asarray(["H", "D", "A", "X", "H"])
        readout._assert_dx_mask(baseline.copy(), baseline, "identity")
        added_x = baseline.copy()
        added_x[0] = "X"
        with self.assertRaisesRegex(ValueError, "exact X mask"):
            readout._assert_dx_mask(added_x, baseline, "added X")
        removed_d = baseline.copy()
        removed_d[1] = "H"
        with self.assertRaisesRegex(ValueError, "exact D mask"):
            readout._assert_dx_mask(removed_d, baseline, "removed D")

    def test_synthetic_full_cohort_replay_all_ten_arms_and_hash_failure(self):
        repo = runner.ROOT
        with tempfile.TemporaryDirectory(dir=repo) as temp:
            work = Path(temp)
            root = work
            cache_dir, pred_dir = work / "cache", work / "predictions"
            cache_dir.mkdir()
            pred_dir.mkdir()
            frontier_declaration_path = work / "frontier_declaration.json"
            declaration_path = work / "evaluation_declaration.json"
            selection_path = work / "selection.json"
            replay_path = work / "frontier_replay.json"
            frontier_result_path = work / "frontier_result.json"
            frontier_declaration_path.write_text(json.dumps({"study_id": runner.frontier.STUDY}))
            selection_path.write_text(json.dumps({
                "pipeline_id": "base__" + runner.BASELINE_RECIPE_ID,
                "training_arm": "base", "training_stage": 1024,
                "recipe": {"recipe_id": runner.BASELINE_RECIPE_ID}}))
            replay_path.write_text(json.dumps({"study_id": runner.frontier.STUDY,
                "passed": True, "candidate_scoring": False}))
            frontier_artifacts = {"declaration": {"path": str(frontier_declaration_path.relative_to(root)),
                    "sha256": readout.sha256(frontier_declaration_path)},
                "result": {"path": str(frontier_result_path.relative_to(root)), "sha256": "0" * 64},
                "replay": {"path": str(replay_path.relative_to(root)), "sha256": readout.sha256(replay_path)},
                "selection": {"path": str(selection_path.relative_to(root)), "sha256": readout.sha256(selection_path)}}

            expected_cells = sorted(runner.EXPECTED_CELLS)
            labels = np.asarray(list("H" * 30 + "A" * 30 + "D" * 30 + "H" * 30), dtype="U1")
            period = np.ones(len(labels), dtype=np.int32)
            timestamps = np.arange(len(labels), dtype=np.float64) / 25
            probabilities = np.full((len(labels), 3), .01, dtype=np.float64)
            probabilities[labels == "H"] = (.98, .01, .01)
            probabilities[labels == "A"] = (.01, .98, .01)
            probabilities[labels == "D"] = (.01, .01, .98)
            approved_probability = runner.balanced.recipe_probability(
                {"ce": probabilities, "calibrated": probabilities, "adopted": probabilities},
                .5, .375, "arithmetic")
            facts = {}
            loaded = {}
            base_rows = {}
            for index, key in enumerate(expected_cells):
                ident = {"cell": key, "source_status": "reconstructed",
                    "frontier_study_id": runner.frontier.STUDY,
                    "frontier_declaration_sha256": frontier_artifacts["declaration"]["sha256"],
                    "frontier_result_sha256": readout.sha256(frontier_result_path)
                        if frontier_result_path.exists() else "0" * 64,
                    "frontier_replay_sha256": frontier_artifacts["replay"]["sha256"],
                    "selection_sha256": frontier_artifacts["selection"]["sha256"],
                    "input_sha256": f"{index:064x}", "model_sha256": f"{index+1:064x}",
                    "teacher_sha256": f"{index+2:064x}", "calibration_sha256": f"{index+3:064x}"}
                # The immutable round-one result hash participates in cache
                # identity. Its rows are filled after all synthetic controls
                # have been scored, then caches are rewritten once below.
                arrays = {"labels": labels, "period": period, "timestamps": timestamps,
                    "model_period": period, "fps": np.asarray(25.),
                    "decoded__approved_baseline": labels,
                    "decoded__predecessor": labels,
                    "prob__approved_baseline": approved_probability}
                for family in ("ce", "calibrated", "predecessor"):
                    for seed in readout.SEEDS:
                        arrays[f"prob__{family}__seed{seed}__phase0"] = probabilities.copy()
                        arrays[f"prob__{family}__seed{seed}__phase1"] = probabilities.copy()
                arrays.update({f"identity__{name}": np.asarray(str(value))
                               for name, value in ident.items()})
                path = cache_dir / (key.replace("|", "__") + ".npz")
                np.savez_compressed(path, **arrays)
                facts[key] = {"path": str(path.relative_to(root)),
                    "sha256": readout.sha256(path), "phase_count": 2, "identity": ident}
                # Internal fixture cell for generating the official score rows.
                loaded[key] = {"labels": labels, "period": period, "timestamps": timestamps,
                    "model_period": period, "fps": 25., "baseline_probability": approved_probability,
                    "baseline_prediction": labels, "predecessor_prediction": labels,
                    "decoded_predecessor": labels, "components": {
                        family: {seed: [probabilities.copy(), probabilities.copy()]
                                 for seed in readout.SEEDS}
                        for family in ("ce", "calibrated", "predecessor")}}
                base_rows[key] = readout._official_score(loaded[key], labels)

            r1 = {"study_id": runner.frontier.STUDY,
                  "cell_metrics": {key: {runner.BASELINE_RECIPE_ID: row}
                                   for key, row in base_rows.items()}}
            frontier_result_path.write_text(json.dumps(r1))
            r1_hash = readout.sha256(frontier_result_path)
            frontier_artifacts["result"]["sha256"] = r1_hash
            # Update the cache identities now that the immutable result hash is known.
            for key in expected_cells:
                fact = facts[key]
                path = root / fact["path"]
                with np.load(path, allow_pickle=False) as saved:
                    arrays = {name: saved[name].copy() for name in saved.files
                              if not name.startswith("identity__")}
                    ident = dict(fact["identity"])
                ident["frontier_result_sha256"] = r1_hash
                arrays.update({f"identity__{name}": np.asarray(str(value))
                               for name, value in ident.items()})
                np.savez_compressed(path, **arrays)
                fact["identity"] = ident
                fact["sha256"] = readout.sha256(path)

            required_source_paths = ("experiments/frontier_postprocess_v1.py",
                "experiments/frontier_disagreement_v1.py", "experiments/v2_spell_balanced_eval.py",
                "experiments/v2_spell_frontier_eval.py", "experiments/boundary_relocation_v1.py",
                "experiments/spell_mbr_run.py", "experiments/frontier_postprocess_readout.py",
                "evaluation/possession_metrics.py")
            source_hashes = {}
            for relative in required_source_paths:
                source_copy = root / relative
                source_copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repo / relative, source_copy)
                source_hashes[relative] = readout.sha256(source_copy)
            native_manifest_path = work / "native_manifest.json"
            baseline_recipe = {"recipe_id": runner.BASELINE_RECIPE_ID,
                "ce_weight": .5, "calibrated_weight": .375, "adopted_weight": .125,
                "pooling": "arithmetic", "ha_cost": 3., "training_arm": "base",
                "training_stage": 1024}
            family_weights = {"ce": .5, "calibrated": .375, "predecessor": .125}
            native_manifest_path.write_text(json.dumps({"cells": facts,
                "expected_cells": expected_cells, "source_status": "reconstructed",
                "frontier_artifacts": frontier_artifacts, "baseline_recipe": baseline_recipe,
                "family_weights": family_weights, "seeds": list(readout.SEEDS),
                "baseline_metrics": runner.BASELINE_METRICS, "source_hashes": source_hashes}))
            native_ref = {"path": str(native_manifest_path.relative_to(root)),
                          "sha256": readout.sha256(native_manifest_path)}
            manifest = {"schema": "frontier_postprocess_v1_cache_manifest",
                "source_status": "reconstructed", "reconstruction_provenance": {"synthetic": True},
                "frontier_study_id": runner.frontier.STUDY,
                "frontier_artifacts": frontier_artifacts,
                "baseline_recipe": baseline_recipe,
                "family_weights": family_weights,
                "seeds": list(readout.SEEDS), "expected_cells": expected_cells,
                "cells": facts, "source_hashes": source_hashes,
                "preparation_source_hashes": source_hashes,
                "baseline_metrics": runner.BASELINE_METRICS,
                "view_replicates": {"clean": [0], "feed_like_v1": [1, 2],
                                    "feed_like_v2": [1, 2, 3, 4]},
                "frozen_native_manifest": native_ref,
                "arm_ids": list(runner.ARM_IDS), "arm_settings": runner.arm_settings(),
                "guards": runner.GUARDS, "ranking": runner.RANKING,
                "comparison_tolerances": {"per_cell_metric": runner.PER_CELL_TOLERANCE,
                                           "aggregate_metric": runner.AGGREGATE_TOLERANCE},
                "evidence_scope": "synthetic verifier contract"}
            manifest_path = work / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            manifest_sha = readout.sha256(manifest_path)
            declaration = {"study_id": runner.STUDY,
                "status": "frozen_before_candidate_scoring", "source_status": "reconstructed",
                "reconstruction_provenance": {"synthetic": True},
                "native_manifest": native_ref, "evaluation_manifest_sha256": manifest_sha,
                "frontier_study_id": runner.frontier.STUDY,
                "frontier_artifacts": frontier_artifacts,
                "baseline_recipe": manifest["baseline_recipe"],
                "family_weights": manifest["family_weights"], "arm_ids": manifest["arm_ids"],
                "arm_settings": manifest["arm_settings"], "guards": manifest["guards"],
                "ranking": manifest["ranking"],
                "comparison_tolerances": manifest["comparison_tolerances"],
                "source_hashes": source_hashes,
                "expected_baseline_metrics": runner.BASELINE_METRICS,
                "cohort": {"cell_count": 35, "cells": expected_cells,
                           "seeds": list(readout.SEEDS)},
                "organizer_reference_access": False, "promotion": False}
            declaration_path.write_text(json.dumps(declaration))
            declaration_sha = readout.sha256(declaration_path)

            arm_rows = {arm: {} for arm in readout.PREDICTION_KEYS}
            prediction_inventory = {}
            for key in expected_cells:
                predictions = {"baseline_prediction": labels.copy()}
                for arm, npz_key in readout.PREDICTION_KEYS.items():
                    pred = readout._rebuild_prediction(arm, loaded[key])
                    if npz_key is not None:
                        predictions[npz_key] = pred
                    arm_rows[arm][key] = readout._official_score(loaded[key], pred)
                pred_path = pred_dir / (key.replace("|", "__") + ".npz")
                np.savez_compressed(pred_path, **predictions)
                prediction_inventory[key] = {"path": str(pred_path),
                                             "sha256": readout.sha256(pred_path)}
            aggregates = {arm: readout._aggregate(rows, manifest["view_replicates"])
                          for arm, rows in arm_rows.items()}
            preflight = {"study_id": runner.STUDY, "passed": True,
                "candidate_scoring_performed": False, "manifest_sha256": manifest_sha,
                "declaration_sha256": declaration_sha,
                "source_status": "reconstructed", "source_hashes": source_hashes,
                "baseline_rows": base_rows,
                "aggregate": aggregates["seed2026_0.50"]}
            preflight_path = work / "preflight.json"
            preflight_path.write_text(json.dumps(preflight))
            result = {"study_id": runner.STUDY, "source_status": "reconstructed",
                "manifest_sha256": manifest_sha, "evaluation_manifest_sha256": manifest_sha,
                "declaration_sha256": declaration_sha,
                "preflight_sha256": readout.sha256(preflight_path),
                "source_hashes": source_hashes, "arm_ids": list(runner.ARM_IDS),
                "frontier_artifacts": frontier_artifacts,
                "baseline_replay": {"passed": True, "manifest_sha256": manifest_sha},
                "baseline_rows": base_rows,
                "seed_cell_metrics": {name: arm_rows[name] for name in
                    ("seed2026_0.25", "seed2026_0.50", "seed2026_0.75")},
                "seed_aggregate": {name: aggregates[name] for name in
                    ("seed2026_0.25", "seed2026_0.50", "seed2026_0.75")},
                "boundary": {"cell_metrics": {arm.removeprefix("boundary_"): arm_rows[arm]
                    for arm in arm_rows if arm.startswith("boundary_")},
                    "aggregate": {arm.removeprefix("boundary_"): aggregates[arm]
                    for arm in aggregates if arm.startswith("boundary_")}},
                "ha_cost": {"cell_metrics": {arm: arm_rows[arm] for arm in readout.HA_COST_ARMS},
                    "aggregate": {arm: aggregates[arm] for arm in readout.HA_COST_ARMS}},
                "disagreement": {"cell_metrics": {arm: arm_rows[arm] for arm in readout.GATE_ARMS},
                    "aggregate": {arm: aggregates[arm] for arm in readout.GATE_ARMS}},
                "prediction_inventory": prediction_inventory}
            result_path = work / "result.json"
            result_path.write_text(json.dumps(result))

            report = readout.verify(manifest_path, declaration_path, preflight_path, result_path,
                                        pred_dir, root)
            self.assertTrue(report["passed"])
            self.assertEqual(report["cells_verified"], 35)
            self.assertEqual(report["arms_verified"], 10)
            self.assertEqual(report["candidate_arms_verified"], 9)
            self.assertEqual(report["scored_predictions"], 350)
            self.assertEqual(report["boundary_raw_run_order_checks"], 105)
            self.assertEqual(set(report["guards"]), set(readout.PREDICTION_KEYS) - {"seed2026_0.50"})
            self.assertFalse(report["guards"]["seed2026_0.25"]["passed"])
            self.assertNotIn("seed2026_0.25", report["eligible_ranking"])
            self.assertEqual(set(report["all_candidate_ranking"]),
                             set(readout.PREDICTION_KEYS) - {"seed2026_0.50"})
            self.assertEqual(len(report["hashes"]["prediction_files"]), 35)

            # Tampering is detected by the prediction-file receipt before
            # metric comparison, even for a synthetic all-arm archive.
            first = next(iter(prediction_inventory))
            path = Path(prediction_inventory[first]["path"])
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "changed prediction cache"):
                readout.verify(manifest_path, declaration_path, preflight_path, result_path,
                               pred_dir, root)


if __name__ == "__main__":
    unittest.main()
