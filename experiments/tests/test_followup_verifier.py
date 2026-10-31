"""Guard against missing cohorts and incorrectly reported adoption screens."""
import unittest

from experiments import dead_transfer
from experiments.verify_temporal_followup import exact_keys, verify_guardrail, verify_means


class FollowupVerifierTests(unittest.TestCase):
    def test_exact_cohort_rejects_missing_and_duplicate_rows(self):
        expected={('a',1),('b',2)}
        exact_keys([dict(name='a',seed=1),dict(name='b',seed=2)],('name','seed'),expected,'fixture')
        for rows in ([dict(name='a',seed=1)], [dict(name='a',seed=1),dict(name='a',seed=1)]):
            with self.assertRaises(ValueError):exact_keys(rows,('name','seed'),expected,'fixture')

    def test_guardrail_recomputed_not_trusted(self):
        per={s:dict(d_f1=.8,macro_f1=.8) for s in dead_transfer.SPLITS}
        baseline=dict(spell_f1=.5,per_source=per)
        candidate=dict(spell_f1=.6,per_source=per)
        screen=dict(clean_spell_delta=.1,stress_spell_delta=.1,worst_provider_d_or_macro_delta=0.,passes=True)
        verify_guardrail(screen,candidate,candidate,baseline,baseline)
        with self.assertRaises(ValueError):
            verify_guardrail(dict(screen,passes=False),candidate,candidate,baseline,baseline)
        with self.assertRaises(ValueError):
            verify_guardrail(dict(screen,stress_spell_delta=.2),candidate,candidate,baseline,baseline)

    def test_provider_balance_is_not_pooled_match_mean(self):
        values={'sportec':[.8],'metrica':[.4],'skillcorner':[.2,.2,.2]}
        rows=[dict(model='test',source=s,match=f'{s}_{i}',view='clean',spell_f1=v)
              for s,group in values.items() for i,v in enumerate(group)]
        summary=dict(model='test',view='clean',spell_f1=(.8+.4+.2)/3,
                     per_source={s:dict(spell_f1=v[0]) for s,v in values.items()})
        verify_means(rows,[summary],('model',),'stress',('spell_f1',))
        with self.assertRaises(ValueError):
            verify_means(rows,[dict(summary,spell_f1=(.8+.4+.2+.2+.2)/5)],('model',),'stress',('spell_f1',))


if __name__=='__main__':unittest.main()
