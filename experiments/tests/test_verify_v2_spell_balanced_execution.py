import unittest
from scripts import verify_v2_spell_balanced_execution as execution


class AuditorIdentityOrderTests(unittest.TestCase):
    def test_exact_grid_identity_is_checked_before_rank_order_normalization(self):
        result={'recipes':[{'recipe_id':x} for x in ['anchor','other','winner']],
                'selected_for_round2_seed_diagnostics':['winner','anchor']}
        original=execution.auditor.ast
        with execution.corrected_identity_order(result):
            self.assertEqual(execution.auditor.ast.literal_eval("['anchor', 'winner']"),['winner','anchor'])
            for invalid in ("['winner', 'anchor']","['anchor']","['anchor','other']"):
                with self.assertRaises(AssertionError):execution.auditor.ast.literal_eval(invalid)
        self.assertIs(execution.auditor.ast,original)

    def test_restores_auditor_on_failure(self):
        original=execution.auditor.ast
        with self.assertRaises(RuntimeError):
            with execution.corrected_identity_order({'recipes':[{'recipe_id':'a'}],
                 'selected_for_round2_seed_diagnostics':['a']}):raise RuntimeError('fixture')
        self.assertIs(execution.auditor.ast,original)
