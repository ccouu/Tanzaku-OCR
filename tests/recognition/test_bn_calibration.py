import torch
from torch import nn
from src.recognition.bn_calibration import PooledMoments,recalibrate_bn


def test_pooled_moments_handle_different_widths_and_batch_means():
    a=torch.arange(12,dtype=torch.float32).reshape(2,2,3)
    b=torch.arange(4,dtype=torch.float32).reshape(1,2,2)+30
    pooled=PooledMoments();pooled.update(a);pooled.update(b)
    pixels=torch.cat((a.movedim(1,0).reshape(2,-1),b.movedim(1,0).reshape(2,-1)),1).double()
    assert torch.allclose(pooled.mean,pixels.mean(1))
    assert torch.allclose(pooled.variance(),pixels.var(1,correction=1))


def test_calibration_changes_only_bn_buffers_and_restores_modes():
    model=nn.Sequential(nn.BatchNorm1d(2),nn.Dropout(.5)).eval()
    before={k:v.clone() for k,v in model.state_dict().items()}
    batches=[torch.arange(12,dtype=torch.float32).reshape(2,2,3),torch.ones(1,2,4)*20]
    def forward(m,b):
        assert not m[1].training
        return m(b)
    audit=recalibrate_bn(model,batches,forward)
    assert audit['0']['activation_count']==10 and audit['0']['batches']==2
    assert torch.equal(model[0].weight,before['0.weight']) and torch.equal(model[0].bias,before['0.bias'])
    assert not torch.equal(model[0].running_mean,before['0.running_mean'])
    assert int(model[0].num_batches_tracked)==2
    assert model[0].momentum==.1 and not model[0].training and not model[1].training
