"""Replay archived predictions, official scores and aggregation independently."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from experiments import feed_robustness_v1 as study
from experiments import metrics, spell_mbr_run as shared


def main():
    declaration = study.check()
    aggregate = study.read(study.RESULTS / f"{study.STUDY}_aggregate.json")
    model_checks = []
    for fold in study.FOLDS:
        for seed in study.SEEDS:
            p = study.OUT / "models" / f"{fold}_{seed}.joblib"
            model = joblib.load(p)
            receipt = study.read(p.with_suffix(".json"))
            prior = study.read(ROOT / "data/experiments/dense_label_weight_v1/fits" /
                               f"control_D2__{fold.removeprefix('leave_out_')}__s{seed}.json")
            if receipt["exposures"] != prior["exposures"]:
                raise ValueError("Training sampling exposure differs from matched control")
            if model["step"] != 2048 or len(receipt["training_config"]["soccernet_games"]) != 45:
                raise ValueError("Training budget/cohort changed")
            if study.sha(p) != receipt["sha256"]:
                raise ValueError("Model checksum differs")
            model_checks.append({"fold": fold, "seed": seed, "same_sampling_exposures": True,
                                 "steps": model["step"]})
    cells = shared._load_cells(declaration)
    baseline = {tuple(c[k] for k in ("provider", "match", "view", "member")): c for c in cells}
    verified = {a: [] for a in aggregate["rows"]}
    arrays = {}
    maximum_score_difference = 0.
    for arm, rows in aggregate["rows"].items():
        for row in rows:
            key = tuple(row[k] for k in ("provider", "match", "view", "member"))
            c = baseline[key]
            if arm == "baseline":
                p = c["probabilities"]
                prediction = study.decode(p, c["model_period"], c["fps"])
            else:
                path = study.OUT / "predictions" / arm / ("__".join(key) + ".npz")
                if study.sha(path) != aggregate["prediction_sha256"][str(path.relative_to(ROOT))]:
                    raise ValueError("Native prediction hash changed")
                with np.load(path) as z:
                    p, prediction = z["probabilities"], z["prediction"]
                    for field in ("labels", "period", "timestamps", "model_period", "fps"):
                        np.testing.assert_array_equal(z[field], c[field])
                np.testing.assert_array_equal(prediction, study.decode(p, c["model_period"], c["fps"]))
            arrays[(arm,) + key] = p
            official = metrics.evaluate_tracking(c["labels"], prediction, period=c["period"],
                                                 timestamps=c["timestamps"], fps=c["fps"])
            actual = {"spell_f1": float(official["spell_f1_mean"]),
                      "macro_f1": float(official["macro_f1"]),
                      "d_f1": float(official["per_class"]["D"]["f1"])}
            for metric, value in actual.items():
                error = abs(value - row[metric])
                maximum_score_difference = max(error, maximum_score_difference)
                if error > 1e-12:
                    raise ValueError("Official score replay differs")
            verified[arm].append({**{k: row[k] for k in ("provider", "match", "view", "member")}, **actual})
    for key, c in baseline.items():
        if key[-1] == "ensemble":
            for arm in aggregate["rows"]:
                expected = np.mean([arrays[(arm,) + key[:-1] + (str(s),)] for s in study.SEEDS], axis=0)
                # Blend-then-average vs average-then-blend may differ by one float ulp.
                np.testing.assert_allclose(arrays[(arm,) + key], expected, rtol=0, atol=2e-16)
        np.testing.assert_array_equal(arrays[("fixed_half_blend",) + key],
                                      .5 * (arrays[("feed_augmented",) + key] + c["probabilities"]))
    for arm, rows in verified.items():
        for member in shared.MEMBERS:
            for view in ("clean", "feed_like"):
                for metric in ("spell_f1", "macro_f1", "d_f1"):
                    provider_means = []
                    for provider, matches in shared.EXPECTED_MATCHES.items():
                        match_means = []
                        for match in matches:
                            values = [r[metric] for r in rows if r["member"] == member and
                                      r["provider"] == provider and r["match"] == match and
                                      (r["view"] == "clean" if view == "clean" else r["view"].startswith("feed_like_"))]
                            if len(values) != (1 if view == "clean" else 2):
                                raise ValueError("Missing or duplicate cohort cell")
                            match_means.append(sum(values) / len(values))
                        provider_means.append(sum(match_means) / len(match_means))
                    actual = sum(provider_means) / len(provider_means)
                    reported = aggregate["summaries"][arm][member][view]["equal_provider_mean"][metric]
                    if abs(actual - reported) > 1e-12:
                        raise ValueError("Independent scalar aggregation differs")
    result = {"passed": True, "study": study.STUDY, "verified_models": model_checks,
              "scored_cells": sum(map(len, verified.values())), "native_prediction_files": 90,
              "maximum_official_score_difference": maximum_score_difference,
              "ensemble_and_blend_parity": True, "unchanged_native_grading": True,
              "aggregate_sha256": study.sha(study.RESULTS / f"{study.STUDY}_aggregate.json"),
              "verifier_sha256": study.sha(Path(__file__)), "organizer_reference_access": False}
    study.write(study.RESULTS / f"{study.STUDY}_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
