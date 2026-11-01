import random
import unittest
import numpy as np
from experiments import feed_short_sampler as base
from experiments.feed_short_game_sampler import GameResampler
from experiments.tests.test_sn_dense_train import old_record,sn_record,NAMES

class GameSampling(unittest.TestCase):
    def test_cross_record_access_preserves_game_base_rng_and_old_draws(self):
        old=[old_record('o')];a=sn_record('g',1);b=sn_record('g',2)
        a['y'][:]=0;a['full_y'][:]=0
        original=base.Resampler(old,[a,b],NAMES,51,1)
        game=GameResampler(old,[a,b],NAMES,51,1)
        ra,rb=random.Random(51),random.Random(51);changed=0
        for _ in range(100):
            wo=base.cap._draw_old(ra,original.groups['old'],old);wg=base.cap._draw_old(rb,game.groups['old'],old)
            xo=original.replace('old',wo);xg=game.replace('old',wg)
            for i in range(5):np.testing.assert_array_equal(xo[i],xg[i])
            wn=base.cap._draw_new(ra,original.groups['new']);gn=base.cap._draw_new(rb,game.groups['new'])
            original.replace('new',wn);selected=game.replace('new',gn)
            self.assertEqual(selected[0],1);changed+=int(gn[0]!=selected[0])
            self.assertEqual(ra.getstate(),rb.getstate());self.assertEqual(original.rng.getstate(),game.rng.getstate())
        self.assertGreater(changed,0)
        self.assertEqual(original.exposures,game.exposures)

    def test_disabled_and_no_pool_are_identity(self):
        r=sn_record('g');r['y'][:]=0;r['full_y'][:]=0
        for rate in (0,1):
            s=GameResampler([old_record()],[r],NAMES,3,rate)
            window=next(iter(s.groups['new'].values()))[0]
            self.assertIs(s.replace('new',window),window)

if __name__=='__main__':unittest.main()
