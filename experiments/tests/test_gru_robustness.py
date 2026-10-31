"""Raw observation and paired-budget contracts for slot dropout study."""
import unittest
from unittest.mock import patch
import numpy as np
from experiments import gru_robustness_experiment as gr, learned_temporal as lt, factorized_had as fh
from experiments.tests.test_tracking_corruption import synthetic


class GruRobustnessContracts(unittest.TestCase):
    def test_slot_mask_preserves_native_ball_targets_and_time(self):
        inputs=synthetic(200)
        # Enough slots per team to activate the legacy masking rule.
        inputs['players']=np.tile(inputs['players'],(1,3,1))
        inputs['teams']=np.tile(inputs['teams'],3)
        inputs['labels']=np.full(200,'H')
        inputs['player_observed']=np.isfinite(inputs['players']).all(axis=-1)
        inputs['player_observed'][20:30,0]=False
        original=inputs['players'].copy()
        a=gr.slot_view(inputs,2026)
        b=gr.slot_view(dict(inputs,labels=np.full(200,'D')),2026)
        np.testing.assert_array_equal(a['players'],b['players'])
        np.testing.assert_array_equal(inputs['players'],original)
        np.testing.assert_array_equal(a['ball'],inputs['ball'])
        np.testing.assert_array_equal(a['period'],inputs['period'])
        self.assertNotIn('labels',a)
        self.assertFalse(a['player_observed'][20:30,0].any())
        self.assertTrue(np.isnan(a['players'][20:30,0]).all())
        self.assertGreater(np.isnan(a['players']).sum(),np.isnan(original).sum())
        self.assertTrue(np.all(np.isfinite(a['players']) <= np.isfinite(original)))

    def test_matched_records_preserve_targets_windows_and_swap_pairs(self):
        n=300
        r=dict(source='s',match='m',x=np.arange(n*3,dtype=np.float32).reshape(n,3),
               y=np.resize([-1,0,1,2],n).astype(np.int8),names=['H_distance','A_distance','D_speed'],
               period=np.repeat([1,2],n//2))
        cache={k:{'joint':dict(r,x=r['x']+i)} for i,k in enumerate(('clean','legacy','gap','dropout_legacy','dropout_gap'))}
        control=gr.paired_records(cache,'m','control');drop=gr.paired_records(cache,'m','slot_dropout')
        for a,b in zip(control,drop):
            for key in ('y','period','replace_rows','replace_windows'):
                np.testing.assert_array_equal(a[key],b[key])
            self.assertEqual(len(list(lt.chunks(a))),len(list(lt.chunks(b))))
        for i in (0,1):
            np.testing.assert_array_equal(control[i]['x'],drop[i]['x'])
            np.testing.assert_array_equal(control[i]['alternate'],drop[i]['alternate'])
        self.assertFalse(np.array_equal(control[2]['x'],drop[2]['x']))
        np.testing.assert_array_equal(drop[3]['x'],drop[2]['x'][:,[1,0,2]])
        with patch.object(lt,'EPOCHS',1):
            a=fh.fit(control,player_only=True,seed=2026)
            b=fh.fit(drop,player_only=True,seed=2026)
        for key in ('parameters','optimizer_updates','windows_per_epoch'):
            self.assertEqual(a[key],b[key])


if __name__=='__main__': unittest.main()
