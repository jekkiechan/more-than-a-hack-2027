import random
import unittest
import numpy as np
from experiments import short_label_quality as q,feed_short_sampler as base
from experiments.short_original_diversity import OriginalDiversitySampler
from experiments.tests.test_short_label_quality import record
from experiments.tests.test_sn_dense_train import old_record,NAMES


class OriginalDiversityContracts(unittest.TestCase):
    def test_gar_schedule_base_rng_and_record_identity_preserved(self):
        old=[old_record('o1'),old_record('o2',2)];new=[record()]
        for r in old:
            r['y'][45:55]=0
            r['full_y']=np.repeat(r['y'],5)
        parent=q.QualitySampler(old,new,NAMES,31,diverse=True,rate=1)
        candidate=OriginalDiversitySampler(old,new,NAMES,31,rate=1)
        ar,br=random.Random(31),random.Random(31)
        for _ in range(80):
            a=base.cap._draw_old(ar,parent.groups['old'],old)
            b=base.cap._draw_old(br,candidate.groups['old'],old)
            parent.replace('old',a);chosen=candidate.replace('old',b)
            self.assertEqual(chosen[0],b[0])
            a=parent.replace('new',base.cap._draw_new(ar,parent.groups['new']))
            b=candidate.replace('new',base.cap._draw_new(br,candidate.groups['new']))
            for i in range(5):np.testing.assert_array_equal(a[i],b[i])
            self.assertEqual(ar.getstate(),br.getstate())
            self.assertEqual(parent.rng.getstate(),candidate.rng.getstate())
        self.assertEqual(parent.exposures,candidate.exposures)
        self.assertEqual(parent.counts['new'],candidate.counts['new'])
        self.assertEqual(candidate.counts['old']['draws'],80)
        self.assertEqual(candidate.counts['old']['replacement_draws'],80)

    def test_disabled_and_no_eligible_possession_fall_back(self):
        old=[old_record()];old[0]['y'][:]=0;old[0]['full_y'][:]=0
        for rate in (0,1):
            s=OriginalDiversitySampler(old,[record()],NAMES,3,rate)
            w=next(iter(s.groups['old'].values()))[0]
            self.assertIs(s.replace('old',w),w)


if __name__=='__main__':unittest.main()
