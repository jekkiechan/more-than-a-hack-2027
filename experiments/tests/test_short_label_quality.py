import copy
import random
import unittest

import numpy as np

from experiments import feed_short_sampler as base, short_label_quality as q
from experiments.tests.test_sn_dense_train import old_record, sn_record, NAMES, teacher


def record(game='g', offset=0):
    r = sn_record(game)
    r['y'][:] = 0
    r['y'][20:30] = 1
    r['y'][60:70] = 1
    r['full_y'] = np.repeat(r['y'], 5)
    r['full_frame_numbers'] = np.arange(len(r['full_y'])) + offset
    r['game_period'] = 1
    r['anchor_full_y'] = np.full(len(r['full_y']), -1)
    r['full_prov'] = np.ones(len(r['full_y']), np.int8)
    r['touch_runs'] = np.array([[100,149,1,1,1,0], [300,349,1,1,1,0]])
    r['x'][:, NAMES.index('H_distance1')] = 4
    r['x'][:, NAMES.index('A_distance1')] = 1
    return r


class QualityContracts(unittest.TestCase):
    def test_quality_rejects_conflict_missing_geometry_and_censored_control(self):
        r = record()
        self.assertEqual(sum(s['quality'] for s in q.spans(r)), 2)
        for intervention in ('anchor', 'ball', 'censored', 'opposing'):
            altered = copy.deepcopy(r)
            if intervention == 'anchor': altered['anchor_full_y'][110] = 0
            if intervention == 'ball': altered['x'][20:30, NAMES.index('ball_observed')] = 0
            if intervention == 'censored': altered['touch_runs'][0,5] = 1
            if intervention == 'opposing': altered['x'][20:30, NAMES.index('H_distance1')] = 0
            self.assertFalse(q.spans(altered)[0]['quality'], intervention)

    def test_team_swap_and_native_censoring(self):
        r = record(); other = copy.deepcopy(r)
        for key in ('y','full_y','anchor_full_y'):
            y = other[key]; other[key] = np.where(np.isin(y,(0,1)),1-y,y)
        other['touch_runs'][:,2] = 1-other['touch_runs'][:,2]
        a,b = NAMES.index('H_distance1'),NAMES.index('A_distance1')
        other['x'][:,[a,b]] = other['x'][:,[b,a]]
        self.assertEqual([s['quality'] for s in q.spans(r)], [s['quality'] for s in q.spans(other)])
        r['full_period'][125:] = 1
        self.assertEqual(sum(s['quality'] for s in q.spans(r)), 1)
        r['full_y'][99] = -1
        self.assertEqual(sum(s['quality'] for s in q.spans(r)), 1)

    def test_base_rng_old_sampling_and_inputs_preserved(self):
        old=[old_record()];new=[record(),record(offset=1000)];before=copy.deepcopy(new)
        reference=base.Resampler(old,new,NAMES,17,1)
        candidate=q.QualitySampler(old,new,NAMES,17,rate=1,diverse=True)
        ar,br=random.Random(17),random.Random(17)
        for _ in range(40):
            a=reference.replace('old',base.cap._draw_old(ar,reference.groups['old'],old))
            b=candidate.replace('old',base.cap._draw_old(br,candidate.groups['old'],old))
            for i in range(5): np.testing.assert_array_equal(a[i],b[i])
            reference.replace('new',base.cap._draw_new(ar,reference.groups['new']))
            candidate.replace('new',base.cap._draw_new(br,candidate.groups['new']))
            self.assertEqual(ar.getstate(),br.getstate())
            self.assertEqual(reference.rng.getstate(),candidate.rng.getstate())
        self.assertEqual(reference.exposures,candidate.exposures)
        self.assertEqual(set(candidate.targets.values()),{10})
        for a,b in zip(before,new):
            for key in a: np.testing.assert_array_equal(a[key],b[key])

    def test_disabled_fit_parity_and_empty_pool_fallback(self):
        from experiments import sn_dense_train as sn
        old=[old_record()];new=[record()];t=teacher()
        kw=dict(width=24,steps=2,batch_size=8,seed=17,ha_target='dense',train_d=True,
                require_full_native=True,verbose=False)
        expected=sn.train(t,old,new,**kw)
        sampler=q.QualitySampler(old,new,NAMES,17,rate=0)
        with base.installed(sampler): actual=sn.train(t,old,new,**kw)
        for part in ('state','d_state'):
            for key,value in expected['model'][part].items():
                np.testing.assert_array_equal(value,actual['model'][part][key])
        new[0]['x'][:,NAMES.index('ball_observed')]=0
        sampler=q.QualitySampler(old,new,NAMES,17,rate=1)
        window=next(iter(sampler.groups['new'].values()))[0]
        self.assertIs(sampler.replace('new',window),window)


if __name__ == '__main__': unittest.main()
