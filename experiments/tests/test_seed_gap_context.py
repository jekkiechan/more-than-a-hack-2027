import unittest
import numpy as np
from experiments import gap_diagnostics as gd, seed_gap_context as study, learned_temporal as lt
from experiments.metrics import evaluate_tracking


def inputs(n=750):
    return dict(ball=np.tile([0.,0.],(n,1)),
                players=np.tile([[0.,0.],[30.,0.]],(n,1,1)),teams=np.array([0,1]),
                period=np.ones(n,int),fps=np.array(25.),timestamps=np.arange(n)/25,
                pitch=np.array([105.,68.]),ball_observed=np.ones(n,bool),
                player_observed=np.ones((n,2),bool))


class SeedGapContextTests(unittest.TestCase):
    def test_full_vector_average_before_decoder(self):
        a=np.array([[.8,.1,.1],[.05,.15,.8]])
        b=np.array([[.2,.6,.2],[.1,.8,.1]])
        np.testing.assert_allclose(study.average_probabilities(a,b),[[.5,.35,.15],[.075,.475,.45]])
        with self.assertRaises(ValueError): study.average_probabilities(a,b[:, :2])
        with self.assertRaises(ValueError): study.average_probabilities(a,b*2)

    def test_context_preserves_core_and_restores_even_on_error(self):
        record=dict(period=np.ones(173,int))
        base=list(lt.chunks(record))
        with self.assertRaises(RuntimeError):
            with study.context(15):
                long=list(lt.chunks(record))
                self.assertEqual([(a,b) for a,b,_ in base],[(a,b) for a,b,_ in long])
                self.assertEqual(len(long[0][2]),200)
                raise RuntimeError('test')
        self.assertEqual(lt.CONTEXT,25)
        self.assertEqual(len(list(lt.chunks(record))[0][2]),100)

    def test_gaps_respect_masks_and_time_breaks(self):
        x=inputs(250)
        x['ball_observed'][50:200]=False
        x['period'][125:]=2
        c=gd.covariates(x)
        self.assertTrue(np.all(c['gap'][50:200]==3))
        self.assertEqual(len(c['gaps']),2)
        self.assertFalse(c['contact'][50:200].any())

    def test_contact_distance_does_not_cross_blocks(self):
        before,after,d=gd.nearest_contacts(np.array([1,0,0,0,0,0],bool),np.array([1,1,1,2,2,2]),1)
        np.testing.assert_array_equal(before,[0,0,0,-1,-1,-1])
        self.assertTrue(np.isinf(d[3:]).all())
        self.assertEqual(d[2],2)

    def test_bracket_visibility_improves_with_context(self):
        x=inputs(1000)
        x['ball'][250:600]=np.nan  # 14s gap, straddles 10s core boundary.
        c=gd.covariates(x)
        self.assertGreater(c['bracket']['15'][250:600].sum(),c['bracket']['5'][250:600].sum())
        self.assertTrue(np.all(c['bracket']['15'] >= c['bracket']['5']))

    def test_error_categories_are_disjoint_and_unknown_excluded(self):
        labels=np.array(list('HHAADDXX'))
        pred=np.array(list('AHDDHDHA'))
        r=gd.frame_counts(labels,pred,np.ones(8,bool))
        self.assertEqual((r['live_frames'],r['correct_team'],r['wrong_team'],r['false_dead']),(4,1,1,2))
        self.assertEqual(r['missed_dead'],1)
        self.assertEqual(r['graded_frames'],6)

    def test_official_global_spell_matching_and_fragmentation(self):
        x=inputs(250)
        labels=np.array(['H']*75+['A']*25+['H']*75+['D']*75)
        pred=labels.copy(); pred[20:30]='A'; pred[75:100]='H'
        d=gd.analyze(x,labels,pred)
        m=evaluate_tracking(labels,pred,x['period'],25,x['timestamps'])
        self.assertEqual(d['spells']['all']['reference_spells'],m['n_gt_spells'])
        self.assertEqual(d['spells']['all']['missed'],m['n_gt_spells']-m['spell_f1'][.5]['n_matched'])
        self.assertGreater(d['spells']['all']['fragmented'],0)
        self.assertEqual(d['spells']['all']['short_reference'],1)

    def test_turnover_timing_and_no_matching_across_unknown(self):
        x=inputs(250)
        labels=np.array(['H']*75+['A']*75+['X']*25+['H']*75)
        pred=labels.copy();pred[75:100]='H'
        d=gd.analyze(x,labels,pred)
        t=d['turnovers']['all']
        self.assertEqual(t['reference'],1)
        self.assertEqual(t['matched_within_5s'],1)
        self.assertEqual(t['mean_signed_seconds'],1)

    def test_covariates_ignore_targets(self):
        x=inputs(250); x['labels']=np.array(['H']*250)
        a=gd.covariates(x); x['labels'][:]='D'; b=gd.covariates(x)
        for key in ('gap','distance','contact','before','after'):
            np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__': unittest.main()
