"""Guard model comparisons against missing metrics and unequal cohorts."""
import unittest
from experiments.model_scorecard import collect, combined_score, scorecard


class ScorecardTests(unittest.TestCase):
    def test_weight_tradeoff_does_not_change_spell_first_order(self):
        models = {'a': {('source', 'match'): (.8, .6)},
                  'b': {('source', 'match'): (.6, .9)}}
        high_spell = scorecard(models, .7)
        high_macro = scorecard(models, .3)
        self.assertEqual(high_spell[0]['model'], 'a')
        self.assertEqual(high_macro[0]['model'], 'a')
        self.assertEqual(next(r for r in high_macro if r['weighted_rank'] == 1)['model'], 'b')
        self.assertAlmostEqual(combined_score(.8, .6, .7), .74)
        for values in ((.8, float('nan'), .7), (.8, .6, -1), (80, .6, .7)):
            with self.assertRaises(ValueError):
                combined_score(*values)

    def test_equal_provider_weight_and_exact_cohort(self):
        rows = {('many', str(i)): (1., 1.) for i in range(3)}
        rows[('singleton', 'only')] = (0., 0.)
        self.assertEqual(scorecard({'model': rows}, .7)[0]['weighted_score'], .5)
        base = dict(fold='without_source', source='source', match='one',
                    spell_f1=.5, macro_f1=.8)
        report = {'rows': [dict(base, model='a'), dict(base, model='b', match='other')]}
        with self.assertRaisesRegex(ValueError, 'same source/match cohort'):
            collect([report], 'provider_holdout_D')
        with self.assertRaises(ValueError):
            collect([{'organizer_check_performed': True, 'rows': [dict(base, model='a')]}],
                    'provider_holdout_D')


if __name__ == '__main__':
    unittest.main()
