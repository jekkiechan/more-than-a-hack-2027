import unittest
from experiments import v2_spell_frontier_selection as selection


def gates(eligible=True):
    return dict(eligible=eligible,d_metrics_unchanged=True,reference_spell_counts_invariant=True,
                checks={'spell':{'passed':True}})


def aggregate(score,macro=.85):
    return {'feed_like_v2':{'spell_f1':{'equal_provider':score},'macro_f1':{'equal_provider':macro}}}


def fixtures():
    first={'recipes':[{'recipe_id':'a'},{'recipe_id':'b'}],
           'aggregate':{'a':aggregate(.591),'b':aggregate(.59)},
           'eligibility':{'a':gates(),'b':gates()}}
    second={'pipelines':[dict(pipeline_id='fresh__a',recipe={'recipe_id':'a'},training_arm='fresh',training_stage=1536)],
            'aggregate':{'fresh__a':aggregate(.601)},'eligibility':{'fresh__a':gates()}}
    fallback={'recipe':{'recipe_id':'a'},'aggregate':aggregate(.591),'eligibility':gates()}
    return first,second,fallback


class SelectionTests(unittest.TestCase):
    def test_only_strictly_better_eligible_candidate_replaces_current(self):
        a,b,c=fixtures()
        ranked=selection.rank_candidates(a,b,c)
        self.assertEqual([r['pipeline_id'] for r in ranked],['fresh__a','base__a'])
        b['eligibility']['fresh__a']['eligible']=False
        self.assertEqual(selection.rank_candidates(a,b,c)[0]['pipeline_id'],'base__a')
        b['eligibility']['fresh__a']['eligible']=True
        b['aggregate']['fresh__a']=aggregate(.591,.99)
        self.assertEqual(len(selection.rank_candidates(a,b,c)),1)

    def test_inventory_and_failed_guard_rejected(self):
        a,b,c=fixtures();b['eligibility']['fresh__a']['checks']['spell']['passed']=False
        with self.assertRaises(ValueError):selection.rank_candidates(a,b,c)
        a,b,c=fixtures();a['aggregate'].pop('a')
        with self.assertRaises(ValueError):selection.rank_candidates(a,b,c)

    def test_macro_then_pipeline_breaks_exact_spell_tie(self):
        a,b,c=fixtures();a['aggregate']['b']=aggregate(.601,.86)
        self.assertEqual(selection.rank_candidates(a,b,c)[0]['pipeline_id'],'base__b')
        a['aggregate']['b']=aggregate(.601,.85)
        self.assertEqual(selection.rank_candidates(a,b,c)[0]['pipeline_id'],'base__b')

if __name__=='__main__':unittest.main()
