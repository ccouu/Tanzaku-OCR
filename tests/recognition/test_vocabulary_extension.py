import torch
from torch import nn
from src.recognition.vocabulary import extend_vocabulary,ExtendedCTCHead,fused_state


def test_extension_is_append_only_and_preserves_blank():
    old=dict(blank_id=0,char_to_id={'魚':1,'円':2},characters=['魚','円'])
    new,added=extend_vocabulary(old,['円',' ','抹'])
    assert new['char_to_id']['魚']==1 and new['char_to_id']['円']==2
    assert new['char_to_id'][' ']==3 and new['char_to_id']['抹']==4
    assert added==[' ','抹'] and len(old['char_to_id'])==2 and new['blank_id']==0


def test_new_head_and_standard_export_preserve_old_logits_and_parameters():
    torch.manual_seed(40);old=nn.Conv1d(4,3,1);x=torch.randn(2,4,9)
    expected=old(x).detach().clone();model=nn.Module();model.classifier=ExtendedCTCHead(old,2)
    logits=model.classifier(x)
    assert torch.equal(logits[:,:3],expected)
    assert torch.equal(logits[:,3:],torch.full((2,2,9),-12.))
    state=fused_state(model);normal=nn.Conv1d(4,5,1)
    normal.load_state_dict({k.split('.',1)[1]:v for k,v in state.items()})
    assert torch.allclose(normal(x),logits,atol=1e-6)


def test_blank_relative_initialization_cannot_beat_blank_on_weak_frames():
    old=nn.Conv1d(4,3,1)
    with torch.no_grad():old.weight.fill_(-2);old.bias.fill_(-50)
    head=ExtendedCTCHead(old,2,initialization='below_blank',blank_margin=8)
    x=torch.ones(2,4,9);out=head(x)
    assert (out[:,3:]<out[:,:1]).all()
    assert torch.equal(out.argmax(1),old(x).argmax(1))
    assert torch.allclose(out[:,3:],out[:,:1].expand(-1,2,-1)-8)
