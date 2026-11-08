from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments import complementary_spell_readout as readout
from experiments import seed_mix_v1 as study


class ComplementarySpellReadoutTests(unittest.TestCase):
    def test_replays_all_35_cells_eight_arms_and_rejects_tampered_prediction(self):
        with tempfile.TemporaryDirectory(dir=study.ROOT) as temp:
            root = Path(temp)
            pairs = [(provider, match) for provider, match, _ in study.preparation.audit.MATCHES]
            view_replicates = {"clean": [0], "feed_like_v1": [1, 2],
                               "feed_like_v2": [1, 2, 3, 4]}
            view_groups = {name: [name] for name in view_replicates}
            labels = np.asarray(list("H" * 10 + "A" * 10 + "D" * 10 + "X" * 10 + "H" * 10))
            probs = np.tile([.9, .05, .05], (len(labels), 1))
            probs[labels == "A"] = [.05, .9, .05]
            probs[labels == "D"] = [.05, .05, .9]
            cells, keys = {}, []
            for provider, match in pairs:
                for view, reps in view_replicates.items():
                    for rep in reps:
                        key = f"{provider}|{match}|{view}|{rep}"
                        keys.append(key)
                        cache = root / (key.replace("|", "__") + ".npz")
                        arrays = {f"prob__candidate__seed{seed}__phase0": probs
                                  for seed in study.SEEDS}
                        arrays.update({"decoded__historical_f062": labels,
                            "prob__historical_ensemble": probs,
                            "labels": labels, "period": np.ones(len(labels), dtype=int),
                            "timestamps": np.arange(len(labels)) / 25,
                            "model_period": np.ones(len(labels), dtype=int),
                            "fps": np.asarray(25.)})
                        np.savez_compressed(cache, **arrays)
                        identity = {"cell": key, "source_study_id": "synthetic",
                                    "input_sha256": "a" * 64, "calibration_sha256": "b" * 64,
                                    "model_manifest_sha256": "c" * 64}
                        # Native caches carry scalar identity fields alongside
                        # arrays; the loader checks those bindings as well.
                        with np.load(cache, allow_pickle=False) as z:
                            full = {name: z[name].copy() for name in z.files}
                        full.update({f"identity__{name}": np.asarray(str(value))
                                     for name, value in identity.items()})
                        np.savez_compressed(cache, **full)
                        cells[key] = {"path": str(cache.relative_to(study.ROOT)),
                            "sha256": study.sha256(cache), "identity": identity, "phase_count": 1}
            manifest = {"schema": "seed_mix_v1_cache_manifest",
                "source": {"kind": "historical_control", "study_id": "synthetic",
                           "source_sha256": "d" * 64},
                "evidence_scope": "synthetic verifier contract fixture",
                "pooling_order": study.POOLING_ORDER, "seeds": list(study.SEEDS),
                "weights": list(study.WEIGHTS), "source_hashes": study._source_hashes(),
                "cohort": {"matches": [list(x) for x in pairs],
                    "view_replicates": view_replicates, "view_groups": view_groups},
                "expected_cells": keys, "cells": cells,
                "expected_control_spell_f1": {v: 1. for v in view_replicates},
                "expected_control_macro_f1": {v: 1. for v in view_replicates}}
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            declaration_path = root / "declaration.json"
            prep_path = root / "synthetic_preparation_manifest.json"
            prep_path.write_text(json.dumps({"scoring_performed": False, "synthetic": True}))
            declaration_path.write_text(json.dumps({"study_id": study.STUDY,
                "manifest_sha256": study.sha256(manifest_path), "source_hashes": manifest["source_hashes"],
                "preparation_receipt": {"path": str(prep_path.relative_to(study.ROOT)),
                                         "sha256": study.sha256(prep_path)}}))
            with patch.object(study, "decode_candidate", side_effect=lambda p, y, *a: y.copy()), \
                 patch.object(study.boundary, "relocate", side_effect=lambda p, y, *a: y.copy()):
                preflight = study.preflight(manifest_path, declaration_path, study.ROOT)
                preflight_path = root / "preflight.json"
                preflight_path.write_text(json.dumps(preflight))
                predictions = root / "predictions"
                result = study.run(manifest_path, declaration_path, preflight_path,
                                   root=study.ROOT, prediction_dir=predictions)
            result_path = root / "result.json"
            result_path.write_text(json.dumps(result))
            report = readout.verify(manifest_path, declaration_path, preflight_path,
                                    result_path, predictions, study.ROOT)
            self.assertTrue(report["passed"])
            self.assertEqual(report["cells_verified"], 35)
            self.assertEqual(report["arms_verified"], 8)
            self.assertEqual(report["scored_predictions"], 280)
            self.assertEqual(report["boundary_count_and_order_checks"], 105)
            self.assertEqual(report["v2_weight_winner_seed2026"], 0.)
            self.assertEqual(set(report["leave_one_provider_out_seed_weight"]),
                             {"Metrica", "SkillCorner", "Sportec"})
            # Bound prediction hashes are checked before any metric comparison.
            one = next(predictions.glob("*.npz"))
            one.write_bytes(one.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "changed saved prediction"):
                readout.verify(manifest_path, declaration_path, preflight_path,
                               result_path, predictions, study.ROOT)


if __name__ == "__main__":
    unittest.main()
