import torch
from torch import nn
from src.recognition.partial_classifier import SelectedCTCRows,GroupedCTCRows,export_state


def test_selected_rows_preserve_other_rows_and_export_exactly():
    torch.manual_seed(4)
    original=nn.Conv1d(4,8,1);head=SelectedCTCRows(original,[2,5])
    x=torch.randn(2,4,6,requires_grad=True)
    assert torch.equal(head(x),original(x))
    before=original.weight.detach().clone()
    optimizer=torch.optim.AdamW(head.rows.parameters(),lr=.1,weight_decay=.01)
    head(x).square().mean().backward()
    assert x.grad is not None and original.weight.grad is None
    optimizer.step()
    model=nn.Module();model.classifier=head;state=export_state(model)
    keep=[0,1,3,4,6,7]
    assert torch.equal(state['classifier.weight'][keep],before[keep])
    assert not torch.equal(state['classifier.weight'][[2,5]],before[[2,5]])
    deployed=nn.Conv1d(4,8,1)
    deployed.load_state_dict({k.removeprefix('classifier.'):v for k,v in state.items()})
    assert torch.allclose(deployed(x.detach()),head(x.detach()),atol=1e-6)


def test_grouped_rows_have_independent_learning_rates_and_frozen_blank():
    torch.manual_seed(6)
    base=nn.Conv1d(4,9,1);before=base.weight.detach().clone()
    head=GroupedCTCRows(base,{'latin':[2,5],'digits_brackets':[3,6]})
    optimizer=torch.optim.AdamW([{'params':head.heads['latin'].parameters(),'lr':1e-3},
                                {'params':head.heads['digits_brackets'].parameters(),'lr':1e-5}],weight_decay=0)
    for part in head.heads.values():
        for p in part.parameters():p.grad=torch.ones_like(p)
    optimizer.step();model=nn.Module();model.classifier=head;state=export_state(model)
    assert torch.equal(state['classifier.weight'][[0,1,4,7,8]],before[[0,1,4,7,8]])
    fast=(state['classifier.weight'][[2,5]]-before[[2,5]]).abs().mean()
    slow=(state['classifier.weight'][[3,6]]-before[[3,6]]).abs().mean()
    assert 95<float(fast/slow)<105
    deployed=nn.Conv1d(4,9,1);deployed.load_state_dict({k.removeprefix('classifier.'):v for k,v in state.items()})
    x=torch.randn(2,4,5)
    assert torch.allclose(deployed(x),head(x),atol=1e-6)
