"""Synthetic contracts for objective-alignment verification helpers."""
from copy import deepcopy
import unittest

import numpy as np

from experiments import (dead_transfer as dt, learned_temporal as lt,
                         metrics, objective_alignment as oa)
from scripts import verify_objective_alignment as verifier


IOU_KEYS = ("0.3", "0.4", "0.5", "0.6", "0.7")
METRIC_KEYS = (
    "spell_f1", "macro_f1", "weighted_70_30", "d_f1", "d_precision",
    "d_recall", "short_spell_recall", "false_possessions_per_minute",
    "n_gt_spells", "n_pred_spells",
)

SPELL_DELTA = {
    "spell_sqrt": {2026: .02, 2027: .005},
    "transition": {2026: .015, 2027: -.02},
    "combined": {2026: .03, 2027: .012},
}
D_DELTA = {
    "spell_sqrt": (-.01, -.015, -.005),
    "transition": (-.01, -.025, -.005),
    "combined": (-.01, -.015, -.01),
}
MACRO_DELTA = {"spell_sqrt": .002, "transition": -.01, "combined": 0.0}


def _mean(values):
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def synthetic_rows():
    rows = []
    source_spell_base = (0.40, 0.50, 0.60)
    source_macro_base = (0.72, 0.66, 0.62)
    source_d_base = (0.80, 0.60, 0.30)
    for source_index, source in enumerate(dt.SPLITS):
        for match_index, match_path in enumerate(dt.SPLITS[source]["eval"]):
            match = match_path.stem
            for arm in oa.ARMS:
                for seed in oa.SEEDS:
                    spell_delta = SPELL_DELTA.get(arm, {}).get(seed, 0.0)
                    d_delta = D_DELTA.get(arm, (0.0, 0.0, 0.0))[source_index]
                    macro_delta = MACRO_DELTA.get(arm, 0.0)
                    for view in oa.VIEWS:
                        replicates = ("clean",) if view == "clean" else oa.REPLICATES
                        for replicate_index, replicate in enumerate(replicates):
                            replicate_delta = (0.0 if view == "clean" else
                                               (.006 if replicate_index == 0 else -.006))
                            spell = (source_spell_base[source_index]
                                     + (.02 * (match_index - 1) if source == "skillcorner" else 0)
                                     + (.004 if seed == 2027 else 0)
                                     + (-.12 if view == "feed_like" else 0)
                                     + spell_delta + replicate_delta)
                            macro = (source_macro_base[source_index]
                                     + .002 * match_index
                                     + (-.01 if view == "feed_like" else 0)
                                     + macro_delta)
                            d_f1 = source_d_base[source_index] + d_delta
                            row = dict(
                                source=source, match=match, view=view,
                                replicate=replicate, arm=arm, seed=seed,
                                spell_f1=spell, macro_f1=macro,
                                weighted_70_30=.7 * spell + .3 * macro,
                                d_f1=d_f1, d_precision=d_f1 + .01,
                                d_recall=d_f1 - .02,
                                short_spell_recall=.25 + .5 * spell,
                                false_possessions_per_minute=1.2 - .3 * spell,
                                n_gt_spells=10 + source_index + match_index,
                                n_pred_spells=11 + source_index + match_index,
                                spell_by_iou={iou: {"f1": spell - (index + 1) * .015}
                                              for index, iou in enumerate(IOU_KEYS)},
                            )
                            rows.append(row)
    return rows


def manual_summaries(rows):
    """Build the declared replicate→match→provider means independently."""
    summaries = []
    for arm in oa.ARMS:
        for seed in oa.SEEDS:
            for view in oa.VIEWS:
                providers = {}
                for source in dt.SPLITS:
                    match_rows = []
                    for path in dt.SPLITS[source]["eval"]:
                        cells = [row for row in rows if
                                 (row["source"], row["match"], row["arm"], row["seed"], row["view"])
                                 == (source, path.stem, arm, seed, view)]
                        expected_replicates = 1 if view == "clean" else 2
                        if len(cells) != expected_replicates:
                            raise AssertionError("Synthetic aggregate grid is incomplete")
                        match_rows.append(dict(
                            **{key: _mean([row[key] for row in cells]) for key in METRIC_KEYS},
                            spell_by_iou={iou: _mean([
                                row["spell_by_iou"][iou]["f1"] for row in cells])
                                for iou in IOU_KEYS},
                        ))
                    providers[source] = dict(
                        **{key: _mean([row[key] for row in match_rows])
                           for key in METRIC_KEYS},
                        spell_by_iou={iou: _mean([
                            row["spell_by_iou"][iou] for row in match_rows])
                            for iou in IOU_KEYS},
                        matches=len(match_rows),
                    )
                summaries.append(dict(
                    arm=arm, seed=seed, view=view,
                    per_provider=providers,
                    **{key: _mean([providers[source][key] for source in dt.SPLITS])
                       for key in METRIC_KEYS},
                    spell_by_iou={iou: _mean([
                        providers[source]["spell_by_iou"][iou] for source in dt.SPLITS])
                        for iou in IOU_KEYS},
                ))
    return summaries


def manual_screens():
    screens = []
    for arm in oa.NEW_ARMS:
        for seed in oa.SEEDS:
            clean_delta = SPELL_DELTA[arm][seed]
            stress_delta = SPELL_DELTA[arm][seed]
            worst_provider = min(min(D_DELTA[arm]), MACRO_DELTA[arm])
            screens.append(dict(
                arm=arm, seed=seed,
                clean_spell_delta=clean_delta,
                feed_like_spell_delta=stress_delta,
                worst_provider_clean_d_or_macro_delta=worst_provider,
                passes_positive_screen=clean_delta >= .01,
                passes_guardrails=stress_delta >= -.01 and worst_provider >= -.02,
                passes_all=clean_delta >= .01 and stress_delta >= -.01
                and worst_provider >= -.02,
            ))
    return screens


class ObjectiveVerifierContracts(unittest.TestCase):
    def test_pair_schema_preserves_unswapped_labels_and_swaps_only_symmetry_copy(self):
        y = np.array([0, 1, 2, -1, 1, 0], dtype=np.int8)
        period = np.zeros(len(y), dtype=np.int32)
        names = ["H_metric", "A_metric", "D_metric"]
        base_x = np.column_stack((np.arange(len(y)), 100 + np.arange(len(y)),
                                  -np.arange(len(y)))).astype(np.float32)

        def record(offset):
            return dict(x=base_x + np.float32(offset), y=y.copy(), period=period.copy(),
                        names=names, indices=np.arange(len(y), dtype=np.int64),
                        source="synthetic", match="synthetic")

        metadata = dict(records={
            "clean": record(0), "gap": record(10),
            "dropout_legacy": record(20), "dropout_gap": record(30),
        })
        pairs = verifier._paired_records_from_prepared(metadata, "synthetic")
        self.assertEqual(len(pairs), 4)
        for pair_index, pair in enumerate(pairs):
            team_swap = bool(pair_index % 2)
            expected = verifier._expected_paired_labels(y, team_swap)
            np.testing.assert_array_equal(pair["y"], expected)
            if not team_swap:
                np.testing.assert_array_equal(pair["y"], y)
            else:
                np.testing.assert_array_equal(pair["y"], [1, 0, 2, -1, 0, 1])

    def test_complete_aggregate_and_screen_grid_matches_manual_means(self):
        rows = synthetic_rows()
        report = dict(summaries=manual_summaries(rows), screens=manual_screens())
        result = verifier._audit_aggregates(rows, report)
        self.assertEqual(result, dict(summaries=16, screens=6))

        by_key = {(row["arm"], row["seed"]): row for row in report["screens"]}
        self.assertEqual(by_key[("spell_sqrt", 2026)]["clean_spell_delta"], .02)
        self.assertTrue(by_key[("spell_sqrt", 2026)]["passes_all"])
        self.assertFalse(by_key[("spell_sqrt", 2027)]["passes_positive_screen"])
        self.assertFalse(by_key[("transition", 2027)]["passes_guardrails"])
        self.assertTrue(by_key[("combined", 2027)]["passes_all"])

    def test_screen_identity_rejects_arm_and_seed_mismatches(self):
        rows = synthetic_rows()
        base = dict(summaries=manual_summaries(rows), screens=manual_screens())

        wrong_arm = deepcopy(base)
        wrong_arm["screens"][0]["arm"] = "control"
        with self.assertRaisesRegex(ValueError, "Missing screen"):
            verifier._audit_aggregates(rows, wrong_arm)

        wrong_seed = deepcopy(base)
        wrong_seed["screens"][0]["seed"] = 2028
        with self.assertRaisesRegex(ValueError, "Missing screen"):
            verifier._audit_aggregates(rows, wrong_seed)

        float_seed = deepcopy(base)
        float_seed["screens"][0]["seed"] = float(float_seed["screens"][0]["seed"])
        with self.assertRaisesRegex(ValueError, "seed must be an integer"):
            verifier._audit_aggregates(rows, float_seed)

    def test_official_metric_schema_maps_spell_dead_live_and_population_fields(self):
        labels = np.asarray(list("D" * 25 + "H" * 75 + "A" * 50 + "D" * 25))
        prediction = labels.copy()
        prediction[22:28] = "H"  # One short boundary error exercises the D row.
        scored = metrics.evaluate_tracking(
            labels, prediction, fps=25.0,
            timestamps=np.arange(len(labels), dtype=np.float64) / 25.0)
        mapped = verifier._official_metric_values(
            scored, graded_frames=int(np.isin(labels, ("H", "A", "D")).sum()))

        self.assertEqual(set(mapped), {
            "spell_f1", "macro_f1", "d_f1", "d_precision", "d_recall",
            "n_gt_spells", "n_pred_spells", "graded_frames",
        })
        self.assertEqual(mapped["spell_f1"], scored["spell_f1_mean"])
        self.assertEqual(mapped["macro_f1"], scored["macro_f1"])
        self.assertEqual(mapped["d_f1"], scored["per_class"]["D"]["f1"])
        self.assertEqual(mapped["d_precision"], scored["per_class"]["D"]["precision"])
        self.assertEqual(mapped["d_recall"], scored["per_class"]["D"]["recall"])
        self.assertEqual(mapped["n_gt_spells"], scored["n_gt_spells"])
        self.assertEqual(mapped["n_pred_spells"], scored["n_pred_spells"])
        self.assertEqual(mapped["graded_frames"], len(labels))


if __name__ == "__main__":
    unittest.main()
