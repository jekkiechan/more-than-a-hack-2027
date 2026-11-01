import copy
import random
import unittest
import numpy as np
from experiments import feed_short_sampler as s
from experiments.tests.test_sn_dense_train import teacher,old_record,sn_record,NAMES

class SamplingContracts(unittest.TestCase):
    def setUp(self):
        self.old=[old_record('a'),old_record('b',2)]
        self.new=[sn_record('c'),sn_record('d',4)]

    def test_disabled_training_is_exact(self):
        t=teacher()
        kw=dict(width=24,steps=2,batch_size=8,seed=17,ha_target='dense',train_d=True,require_full_native=True,verbose=False)
        from experiments import sn_dense_train as sn
        expected=sn.train(t,self.old,self.new,**kw)
        sampler=s.Resampler(self.old,self.new,NAMES,17,0)
        with s.installed(sampler):actual=sn.train(t,self.old,self.new,**kw)
        for part in ('state','d_state'):
            for key,value in expected['model'][part].items():np.testing.assert_array_equal(value,actual['model'][part][key])
        self.assertEqual(expected['exposures'],actual['exposures'])

    def test_base_rng_record_sequence_and_features_preserved(self):
        saved=copy.deepcopy(self.old+self.new)
        a=s.Resampler(self.old,self.new,NAMES,41,0)
        b=s.Resampler(self.old,self.new,NAMES,41,1)
        ar,br=random.Random(41),random.Random(41)
        for _ in range(100):
            aw=s.cap._draw_old(ar,a.groups['old'],self.old)
            bw=s.cap._draw_old(br,b.groups['old'],self.old)
            chosen=b.replace('old',bw)
            self.assertEqual(aw[0],chosen[0]);self.assertEqual(ar.getstate(),br.getstate())
            if bw[0] in b.pool['old']:
                self.assertTrue(b.eligible['old'][chosen[0]][chosen[1]:chosen[1]+chosen[2]].any())
        for before,after in zip(saved,self.old+self.new):
            for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_missing_eligible_pool_falls_back_and_context_restores_after_error(self):
        r=self.old[0];r['full_y'][:]=0;r['y'][:]=0
        a=s.Resampler([r],self.new,NAMES,3,1)
        w=next(iter(a.groups['old'].values()))[0]
        self.assertIs(a.replace('old',w),w)
        before=(s.cap._draw_old,s.cap._draw_new)
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with s.installed(a):raise RuntimeError('fixture')
        self.assertEqual(before,(s.cap._draw_old,s.cap._draw_new))

    def test_complete_native_mask_team_symmetry_and_boundaries(self):
        r=old_record();r['full_y'][:]=-1
        r['full_y'][100:110]=2;r['full_y'][110:160]=0;r['full_y'][160:170]=2
        r['y']=r['full_y'][r['indices']].copy()
        swapped=copy.deepcopy(r)
        for k in ('y','full_y'):
            y=swapped[k];swapped[k]=np.where(y==0,1,np.where(y==1,0,y))
        np.testing.assert_array_equal(s.masks([r])[0],s.masks([swapped])[0])
        self.assertEqual(s.masks([r])[0].sum(),10)
        r['full_period'][135:]=1
        self.assertFalse(s.masks([r])[0].any())

if __name__=='__main__':unittest.main()
