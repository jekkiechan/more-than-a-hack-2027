"""Contracts for SkillCorner expansion summaries and sealed score replay.

All split checks read manifests only. Result replay uses a tiny synthetic cache;
this file never opens competition or organizer ground truth.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np

from experiments import skillcorner_expansion as study


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def synthetic_result(cache_path: Path, declaration_path: Path, result_path: Path):
    """Write a fully replayable six-row artifact using official synthetic scores."""
    labels = np.array(["H"] * 12 + ["D"] * 4 + ["A"] * 10 + ["X"] * 3 + ["H"] * 8)
    fps = 10.0
    inputs = {
        "period": np.ones(len(labels), dtype=np.int16),
        "timestamps": np.arange(len(labels), dtype=float) / fps,
        "fps": np.asarray(fps),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"inputs": inputs, "labels": labels}, cache_path)
    cache_hash = study.sha(cache_path)
    cache_relative = str(cache_path.relative_to(study.DATA))
    write_json(declaration_path, {"evaluation_sha256": {cache_relative: cache_hash}})

    probability = np.zeros((len(labels), 3), dtype=float)
    columns = {"H": 0, "A": 1, "D": 2, "X": 2}
    for index, label in enumerate(labels):
        probability[index] = 0.025
        probability[index, columns[label]] = 0.95
    arrays = {}
    rows = []
    for arm in ("control", "candidate"):
        for member in ("2026", "2027", "ensemble"):
            key = f"{arm}_{member}"
            arrays[key] = probability.copy()
            metrics, prediction = study.score_cell(inputs, labels, arrays[key])
            arrays[f"{key}_prediction"] = prediction
            rows.append(dict(protocol="primary", match="synthetic", view="clean",
                             replicate="clean", arm=arm, member=member, **metrics))
    # The replay contract checks exact ensemble construction for both arms.
    with result_path.with_suffix(".npz").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    write_json(result_path, {
        "rows": rows,
        "probability_sha256": study.sha(result_path.with_suffix(".npz")),
        "cache": cache_relative,
        "cache_sha256": cache_hash,
        "declaration_sha256": study.sha(declaration_path),
    })


class SkillCornerExpansionContracts(unittest.TestCase):
    def test_split_membership_excludes_held_out_provider_and_reserved_match_without_loading_arrays(self):
        # The plan is entirely manifest-derived. Accidental data access should fail.
        with patch.object(study.joblib, "load", side_effect=AssertionError("loaded a match array")), \
             patch.object(study.np, "load", side_effect=AssertionError("loaded a match array")):
            reserved = {item["match"] for item in study.selection_items(reserved=True)}
            pooled = study.training_items("pooled")
            pooled_matches = {item["match"] for item in pooled}
            all_selection = {item["match"] for item in study.selection_items()}
            self.assertTrue(reserved)
            self.assertFalse(pooled_matches & all_selection)
            self.assertFalse(pooled_matches & reserved)
            self.assertFalse(all_selection & reserved)
            for provider in ("sportec", "metrica", "skillcorner"):
                train = study.training_items(f"leave_out_{provider}")
                self.assertTrue(train)
                self.assertNotIn(provider, {item["source"] for item in train})
                self.assertFalse({item["match"] for item in train} & reserved)

    def test_summary_averages_replicates_then_matches_then_equal_providers(self):
        rows = []
        values = (("provider_a", "a1", 0.1),
                  ("provider_a", "a1", 0.3),
                  ("provider_a", "a2", 0.8),
                  ("provider_b", "b1", 0.4))
        for source, match, value in values:
            row = dict(protocol="primary", source=source, match=match, view="clean",
                       arm="control", member="ensemble", replicate="corrupt_1")
            row.update({key: value for key in study.SCORE_KEYS})
            rows.append(row)

        summary = study.summarize(rows, "primary", "control", "clean")
        self.assertAlmostEqual(summary["providers"]["provider_a"]["spell_f1"], 0.5)
        self.assertAlmostEqual(summary["providers"]["provider_b"]["spell_f1"], 0.4)
        self.assertAlmostEqual(summary["equal_provider"]["spell_f1"], 0.45)
        # A flat row average would weight provider_a's replicated match twice.
        self.assertNotAlmostEqual(summary["equal_provider"]["spell_f1"],
                                  np.mean([row["spell_f1"] for row in rows]))

    def test_changed_historical_manifest_is_rejected_by_frozen_check(self):
        with tempfile.TemporaryDirectory(prefix="skillcorner-manifest-seal-") as temporary:
            root = Path(temporary)
            out = root / "out"
            manifest = root / "expanded.json"
            old_manifest = root / "historical.json"
            manifest.write_text('{"expanded": true}\n', encoding="utf-8")
            old_manifest.write_text('{"historical": "original"}\n', encoding="utf-8")
            predeclared = out / "predeclared.json"
            spec = {
                "source_sha256": {"fixed": "source"},
                "split_manifest_sha256": study.sha(manifest),
                "old_split_manifest_sha256": study.sha(old_manifest),
                "input_sha256": {}, "cache_sha256": {},
                "control_sha256": {}, "evaluation_sha256": {},
            }
            write_json(predeclared, spec)
            with patch.object(study, "OUT", out), \
                 patch.object(study, "MANIFEST", manifest), \
                 patch.object(study, "OLD_MANIFEST", old_manifest), \
                 patch.object(study, "source_hashes", return_value={"fixed": "source"}):
                study.check()
                old_manifest.write_text('{"historical": "changed"}\n', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "cohort changed"):
                    study.check()

    def test_validate_result_rejects_changed_declaration_cache_and_probability_hashes(self):
        with tempfile.TemporaryDirectory(prefix="skillcorner-result-hashes-") as temporary:
            root = Path(temporary)
            data = root / "data"
            out = data / "experiments/skillcorner_expansion_v1"
            result = out / "primary_synthetic_clean_clean.json"
            cache = data / "cache/evaluation.joblib"
            declaration = out / "predeclared.json"
            with patch.object(study, "DATA", data), patch.object(study, "OUT", out):
                synthetic_result(cache, declaration, result)

                value = study.read(result)
                value["declaration_sha256"] = "0" * 64
                write_json(result, value)
                with self.assertRaisesRegex(ValueError, "different declaration"):
                    study.validate_result(result)

                synthetic_result(cache, declaration, result)
                cache.write_bytes(cache.read_bytes() + b"changed")
                with self.assertRaisesRegex(ValueError, "changed cache"):
                    study.validate_result(result)

                synthetic_result(cache, declaration, result)
                with result.with_suffix(".npz").open("ab") as stream:
                    stream.write(b"changed")
                with self.assertRaisesRegex(ValueError, "probabilities changed"):
                    study.validate_result(result)

    def test_validate_result_replays_official_score_and_rejects_edited_score_row(self):
        with tempfile.TemporaryDirectory(prefix="skillcorner-result-replay-") as temporary:
            root = Path(temporary)
            data = root / "data"
            out = data / "experiments/skillcorner_expansion_v1"
            result = out / "primary_synthetic_clean_clean.json"
            cache = data / "cache/evaluation.joblib"
            declaration = out / "predeclared.json"
            with patch.object(study, "DATA", data), patch.object(study, "OUT", out):
                synthetic_result(cache, declaration, result)
                self.assertEqual(len(study.validate_result(result)), 6)
                value = study.read(result)
                value["rows"][0]["spell_f1"] += 0.01
                write_json(result, value)
                with self.assertRaisesRegex(ValueError, "official replay: spell_f1"):
                    study.validate_result(result)


if __name__ == "__main__":
    unittest.main()
