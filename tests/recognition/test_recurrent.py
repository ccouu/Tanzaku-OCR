import pytest
import torch
from torch import nn

from src.recognition.model import ResidualBiLSTM, load_recognizer_state


def test_zero_projection_identity_and_gradient_startup():
    torch.set_num_threads(2)
    branch = ResidualBiLSTM()
    x = torch.randn(2,256,11,requires_grad=True)
    lengths = torch.tensor([11,5])
    result = branch(x,lengths)
    assert torch.equal(result,x)
    result.square().mean().backward()
    assert branch.output.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in branch.lstm.parameters())
    with torch.no_grad():
        branch.output.weight -= .01*branch.output.weight.grad
    branch.zero_grad()
    branch(x,lengths).square().mean().backward()
    assert sum(p.grad.abs().sum() for p in branch.lstm.parameters()) > 0


def test_learned_recurrence_does_not_read_padding_or_other_batch_lengths():
    torch.manual_seed(9)
    branch = ResidualBiLSTM().eval()
    nn.init.normal_(branch.output.weight,std=.1)
    nn.init.normal_(branch.output.bias,std=.1)
    x = torch.randn(2,256,13)
    lengths = torch.tensor([5,13])  # deliberately unsorted
    changed = x.clone();changed[0,:,5:] = 1000
    a,b = branch(x,lengths),branch(changed,lengths)
    assert torch.equal(a[0,:,:5],b[0,:,:5])
    alone = branch(x[:1,:,:5],torch.tensor([5]))
    assert torch.allclose(a[:1,:,:5],alone,atol=1e-6)
    assert torch.equal(a[0,:,5:],x[0,:,5:])


@pytest.mark.parametrize('lengths',[None,[0,4],[5,14],[5],[[5,4]],[5.,4.]])
def test_invalid_recurrent_lengths_fail(lengths):
    with pytest.raises(ValueError):
        ResidualBiLSTM()(torch.zeros(2,256,13),lengths)


def test_explicit_warm_start_still_rejects_missing_base_weights():
    class Tiny(nn.Module):
        def __init__(self,recurrent=False):
            super().__init__()
            self.encoder = nn.Linear(2,2)
            self.recurrent = ResidualBiLSTM() if recurrent else None
    base,extended = Tiny(),Tiny(True)
    load_recognizer_state(extended,base.state_dict(),initialize_recurrent=True)
    assert torch.equal(extended.encoder.weight,base.encoder.weight)
    with pytest.raises(RuntimeError):
        load_recognizer_state(extended,{'encoder.bias':base.encoder.bias},initialize_recurrent=True)
    with pytest.raises(RuntimeError):
        load_recognizer_state(extended,base.state_dict())

