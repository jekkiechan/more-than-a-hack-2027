"""Information isolation and paired-training contracts for the D repair."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from experiments import factorized_had as fh, learned_temporal as lt, live_gap_experiment as old
from experiments.tests.test_tracking_corruption import synthetic


class DeadballTransferContracts(unittest.TestCase):
    def test_player_block_excludes_ball_and_targets(self):
        inputs=synthetic(150)
        labels=np.full(150,'H')
        first=old.feature_view(inputs,labels,'s','m')['joint']
        changed=dict(inputs,ball=np.full_like(inputs['ball'],np.nan),labels=np.full(150,'D'))
        second=old.feature_view(changed,labels,'s','m')['joint']
        raw=[i for i,n in enumerate(first['names']) if n.startswith('D_')]
        np.testing.assert_array_equal(first['x'][:,raw],second['x'][:,raw])
        self.assertEqual(fh.player_columns(first['names']),raw+[i+len(first['names']) for i in raw])

    def test_dead_probability_has_no_team_input_gradient(self):
        torch.set_num_threads(1)
        torch.manual_seed(2026)
        net=fh.network(6,[2,5],True)
        x=torch.randn(2,20,6,requires_grad=True)
        p=net(x).exp()
        p[...,2].sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[...,[0,1,3,4]]).item(),0)
        self.assertGreater(torch.count_nonzero(x.grad[...,[2,5]]).item(),0)
        np.testing.assert_allclose(p.detach().sum(-1),1,atol=2e-7)

    def test_categorical_likelihood_and_matched_initialization(self):
        torch.manual_seed(9);full=fh.network(6,[2,5],False)
        torch.manual_seed(9);limited=fh.network(6,[2,5],True)
        for key,value in full.state_dict().items():
            torch.testing.assert_close(value,limited.state_dict()[key],rtol=0,atol=0)
        x=torch.randn(2,10,6)
        z=full.output(full.sequence(x)[0])
        y=torch.arange(20).reshape(2,10)%3
        categorical=torch.nn.functional.nll_loss(full(x).transpose(1,2),y,reduction='none')
        dead=torch.nn.functional.binary_cross_entropy_with_logits(z[...,2],(y==2).float(),reduction='none')
        team=torch.nn.functional.cross_entropy(z[...,:2].transpose(1,2),y.clamp(max=1),reduction='none')
        torch.testing.assert_close(categorical,dead+team*(y!=2),rtol=1e-6,atol=1e-6)

    def test_fitted_constraint_swap_and_matched_update_budgets(self):
        x=np.sin(np.arange(1200).reshape(400,3)).astype(np.float32)
        r=dict(source='s',match='m',x=x,y=np.resize([-1,0,1,2],400).astype(np.int8),
               names=['H_distance','A_distance','D_speed'],period=np.repeat([1,2],200))
        records=[old.mixed_record(r,r,'m',0,swap=swap) for swap in (False,True)]
        with patch.object(lt,'EPOCHS',1):
            limited=fh.fit(records,player_only=True,seed=2026)
            full=fh.fit(records,player_only=False,seed=2026)
        for key in ('parameters','windows_per_epoch','optimizer_updates'):
            self.assertEqual(limited[key],full[key])
        p=fh.predict(r,limited)
        changed=x.copy();changed[:,:2]=np.nan
        np.testing.assert_array_equal(p[:,2],fh.predict(dict(r,x=changed),limited)[:,2])
        swapped=fh.predict(dict(r,x=x[:,[1,0,2]]),limited)
        np.testing.assert_allclose(swapped,p[:,[1,0,2]],atol=1e-7)
        np.testing.assert_array_equal(p,fh.predict(dict(r,y=np.full(400,2)),limited))
        np.testing.assert_allclose(p.sum(1),1,atol=2e-7)


if __name__=='__main__': unittest.main()
