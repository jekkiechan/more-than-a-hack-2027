from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from experiments import v2_extra_data_mix_eval as mix


class ExtraDataMixEvalTests(unittest.TestCase):
    def test_fixed_four_recipes_and_endpoint_weights(self):
        recipes = {row["recipe_id"]: row["components"] for row in mix.RECIPES}
        self.assertEqual(set(recipes), {"fresh_q25", "fresh_q50", "fresh_q75", "weight_q50"})
        self.assertEqual(recipes["fresh_q25"], {"gar_quality50": .25, "fresh_control": .75})
        self.assertEqual(recipes["weight_q50"], {"gar_quality50": .5, "gar_weight625": .5})

    def test_plan_must_match_exact_grid_and_outer_recipe(self):
        plan = {"recipes": {row["recipe_id"]: row["components"] for row in mix.RECIPES},
                "budget": {"candidate_recipes": 4},
                "outer_recipe": {k: v for k, v in mix.BASE_RECIPE.items()}}
        mix._validate_plan(plan)
        plan["recipes"].pop("fresh_q25")
        with self.assertRaises(RuntimeError):
            mix._validate_plan(plan)

    def test_cached_probability_mixture_is_linear_and_endpoints_replay(self):
        cache = {"pooled__fresh_control": np.asarray([[.8, .2, 0.], [.2, .8, 0.]]),
                 "pooled__gar_quality50": np.asarray([[.4, .6, 0.], [.6, .4, 0.]])}
        fresh = mix._combine(cache, {"fresh_control": 1.})
        quality = mix._combine(cache, {"gar_quality50": 1.})
        midpoint = mix._combine(cache, {"fresh_control": .5, "gar_quality50": .5})
        np.testing.assert_array_equal(fresh, cache["pooled__fresh_control"])
        np.testing.assert_array_equal(quality, cache["pooled__gar_quality50"])
        np.testing.assert_array_equal(midpoint, .5 * (fresh + quality))

    def test_score_cell_keeps_adopted_d_mask_and_uses_phase_mean_per_seed(self):
        base = np.tile(np.asarray([[.8, .2, 0.], [.2, .8, 0.]]), (1, 1))
        cache = {"pooled__fresh_control": base, "pooled__gar_quality50": base.copy()}
        for arm in ("fresh_control", "gar_quality50"):
            for seed in mix.SEEDS:
                for phase in range(5):
                    cache[f"phase__{arm}__{seed}__{phase}"] = base
        decoded = np.asarray(["D", "H"])
        control = {"calibrated": base, "adopted": base, "model_period": np.zeros(2), "fps": 25.,
                   "labels": decoded, "period": np.zeros(2), "timestamps": np.arange(2)}
        control.update({f"adop_seed{s}": base for s in mix.SEEDS})
        control.update({f"cand_seed{s}": base for s in mix.SEEDS})
        score = {"reference_spell_count": 1}
        with patch.object(mix.balanced, "recipe_probability", side_effect=lambda c, *a: c["ce"]), \
             patch.object(mix.balanced, "decode_fixed_d", side_effect=lambda p, d, *a: d.copy()) as decode, \
             patch.object(mix.balanced.controls.audit, "score", return_value=score):
            got = mix._score_cell("cell", cache, control, mix.RECIPES[1], decoded)
        self.assertEqual(got[0], score)
        self.assertEqual(decode.call_count, 3)
        np.testing.assert_array_equal(got[2], decoded)
        np.testing.assert_array_equal(got[1], base)
        self.assertEqual(set(got[3]), {str(s) for s in mix.SEEDS})

    def test_two_and_five_phase_native_caches_are_supported(self):
        base = np.asarray([[.7, .3, 0.]])
        for count in (2, 5):
            cache = {"pooled__fresh_control": base}
            for seed in mix.SEEDS:
                for phase in range(count):
                    cache[f"phase__fresh_control__{seed}__{phase}"] = base
            control = {"calibrated": base, "adopted": base, "model_period": np.zeros(1), "fps": float(count * 5),
                       "labels": np.asarray(["H"]), "period": np.zeros(1), "timestamps": np.arange(1)}
            control.update({f"adop_seed{s}": base for s in mix.SEEDS})
            control.update({f"cand_seed{s}": base for s in mix.SEEDS})
            with patch.object(mix.balanced, "recipe_probability", side_effect=lambda c, *a: c["ce"]), \
                 patch.object(mix.balanced, "decode_fixed_d", side_effect=lambda p, d, *a: d.copy()), \
                 patch.object(mix.balanced.controls.audit, "score", return_value={"reference_spell_count": 1}):
                result = mix._score_cell("Metrica|metrica_3|clean|0", cache, control,
                    {"recipe_id": "fresh_control", "components": {"fresh_control": 1.}}, np.asarray(["H"]))
            self.assertEqual(set(result[3]), {str(s) for s in mix.SEEDS})


if __name__ == "__main__":
    unittest.main()
