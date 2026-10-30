import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experiments import soccernet_extension as study, soccernet_public_check as public

class StudyContracts(unittest.TestCase):
    def summary(self,clean=.6,feed=.5,macro=.8,dead=.8):
        def cell(spell):
            values=dict(spell_f1=spell,macro_f1=macro,d_f1=dead)
            return dict(equal_provider_mean=dict(values),per_provider={p:dict(values) for p in ('sportec','metrica','skillcorner')})
        return dict(clean=cell(clean),feed_like=cell(feed))

    def test_clean_gain_cannot_hide_dead_or_stress_regression(self):
        summaries={'baseline':self.summary(),'old_only':self.summary(.601),
            'gar25':self.summary(.62),'gar50':self.summary(.7)}
        summaries['gar50']['clean']['per_provider']['metrica']['d_f1']=.77
        result=study.select_arm(summaries)
        self.assertEqual(result['selected_arm'],'gar25')
        self.assertTrue(result['improvement_screen_passes'])
        summaries['gar25']['feed_like']['equal_provider_mean']['spell_f1']=.48
        result=study.select_arm(summaries)
        self.assertFalse(result['selected_is_guardrail_eligible'])
        self.assertFalse(result['improvement_screen_passes'])

    def test_gain_must_exceed_equal_update_control(self):
        summaries={'baseline':self.summary(),'old_only':self.summary(.63),
            'gar25':self.summary(.62),'gar50':self.summary(.61)}
        self.assertFalse(study.select_arm(summaries)['improvement_screen_passes'])

    def test_inference_record_excludes_target_and_source_fields(self):
        record={k:np.zeros(1) for k in ('x','names','period','indices','native_n','fps')}
        record.update(y=np.array([0]),full_y=np.array([0]),source='secret',event='PASS')
        a=study.inference_record(record)
        record.update(y=np.array([1]),event='SHOT')
        self.assertEqual(set(a),{'x','names','period','indices','native_n','fps'})
        for k,v in study.inference_record(record).items():np.testing.assert_array_equal(v,a[k])

    def test_gate_and_completed_receipt_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'gate.json'
            public.exclusive(path,{'before_labels':True})
            with self.assertRaises(FileExistsError):public.exclusive(path,{'before_labels':True})
            study.write_once(path,{'before_labels':True})
            with self.assertRaises(ValueError):study.write_once(path,{'before_labels':False})

if __name__=='__main__':unittest.main()
