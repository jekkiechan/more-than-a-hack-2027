import unittest

import numpy as np

from experiments import v2_spell_balanced_eval as study
from scripts import verify_v2_spell_balanced as verifier


class BalancedVerifierSchemaTests(unittest.TestCase):
    def test_full_cohort_recipe_and_seed_aggregates_match_saved_shape(self):
        cells = [study._cell_key(*cell) for cell in study._expected_cells()]
        recipe_ids = ["recipe_a", "recipe_b"]
        values = {}
        seed_rows = {}
        for index, cell in enumerate(cells):
            metric_row = {metric: .4 + index / 1000 for metric in study.METRICS}
            values[cell] = {recipe_id: dict(metric_row) for recipe_id in recipe_ids}
            seed_rows[cell] = {}
            for recipe_id in recipe_ids:
                for seed in study.SEEDS:
                    seed_rows[cell][f"{recipe_id}|{seed}"] = {
                        metric: .3 + (index + seed % 10) / 1000 for metric in study.METRICS}

        aggregate = verifier.aggregate_cell_metrics(values, recipe_ids)
        self.assertEqual(set(aggregate), set(recipe_ids))
        for recipe_id in recipe_ids:
            self.assertEqual(set(aggregate[recipe_id]), {view for view, _ in study.VIEWS})
            for view, _ in study.VIEWS:
                self.assertEqual(set(aggregate[recipe_id][view]), set(study.METRICS))
                self.assertEqual(set(aggregate[recipe_id][view]["spell_f1"]),
                                 {"equal_provider", "providers", "cells"})

        seed_aggregate = verifier.aggregate_seed_metrics(seed_rows, recipe_ids)
        self.assertEqual(set(seed_aggregate), set(recipe_ids))
        for recipe_id in recipe_ids:
            self.assertEqual(set(seed_aggregate[recipe_id]), {str(seed) for seed in study.SEEDS})
            for seed in study.SEEDS:
                self.assertEqual(set(seed_aggregate[recipe_id][str(seed)]),
                                 {view for view, _ in study.VIEWS})
                self.assertEqual(set(seed_aggregate[recipe_id][str(seed)]["clean"]),
                                 set(study.METRICS))

    def test_independent_mixer_does_not_mutate_frozen_components(self):
        components = {
            "ce": np.array([[.5, .3, .2], [.2, .7, .1]]),
            "calibrated": np.array([[.2, .7, .1], [.6, .3, .1]]),
            "adopted": np.array([[.4, .4, .2], [.4, .5, .1]]),
        }
        before = {key: value.copy() for key, value in components.items()}
        recipe = {"ce_weight": .5, "calibrated_weight": .125,
                  "adopted_weight": .375, "pooling": "log"}
        result = verifier._mix(components, recipe)
        self.assertEqual(result.shape, (2, 2))
        np.testing.assert_allclose(result.sum(axis=1), np.ones(2))
        for name in components:
            np.testing.assert_array_equal(components[name], before[name])


if __name__ == "__main__":
    unittest.main()
