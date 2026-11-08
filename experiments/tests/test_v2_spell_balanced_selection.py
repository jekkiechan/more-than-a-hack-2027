import unittest
from copy import deepcopy
from experiments import v2_spell_balanced_selection as selection


class FinalSelectionTests(unittest.TestCase):
    def fixture(self):
        return dict(recipes=[{'recipe_id': 'a'}, {'recipe_id': 'b'}],
                    aggregate={key: {'feed_like_v2': {'spell_f1': {'equal_provider': value},
                                'macro_f1': {'equal_provider': .85}}}
                               for key, value in [('a', .59), ('b', .61)]},
                    eligibility={key: dict(eligible=key == 'a', d_metrics_unchanged=True,
                        reference_spell_counts_invariant=True, checks={'metric': {'passed': key == 'a'}})
                                 for key in ('a', 'b')},
                    seed_diagnostics={'b': {'2026': 'intentionally irrelevant'}})

    def test_ineligible_higher_score_and_seed_results_cannot_select(self):
        rows = selection.rank_candidates([(512, self.fixture())])
        self.assertEqual([(r['training_stage'], r['recipe']['recipe_id']) for r in rows], [(512, 'a')])

    def test_final_ranking_compares_both_stages_and_breaks_exact_tie_stably(self):
        a = self.fixture()
        b = deepcopy(a)
        b['aggregate']['a']['feed_like_v2']['spell_f1']['equal_provider'] = .592
        self.assertEqual(selection.rank_candidates([(512, a), (1024, b)])[0]['training_stage'], 1024)
        self.assertEqual(selection.rank_candidates([(1024, a), (512, a)])[0]['training_stage'], 512)

    def test_inconsistent_eligibility_or_inventory_fails_closed(self):
        fixture = self.fixture()
        fixture['eligibility']['b']['eligible'] = True
        with self.assertRaisesRegex(ValueError, 'failed metric'):
            selection.rank_candidates([(512, fixture)])
        fixture = self.fixture()
        del fixture['aggregate']['b']
        with self.assertRaisesRegex(ValueError, 'inventories'):
            selection.rank_candidates([(512, fixture)])
