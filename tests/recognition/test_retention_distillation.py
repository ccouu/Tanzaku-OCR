import torch
from src.recognition.retention_distillation import old_class_mass_penalty


def test_full_softmax_kl_penalizes_new_mass_and_ignores_padding():
    x=torch.tensor([[[2.,1.,0.]],[[2.,1.,20.]]],requires_grad=True)
    loss=old_class_mass_penalty(x,2,torch.tensor([1]),[True])
    teacher=x[0,0,:2].detach().softmax(-1)
    expected=(teacher*(teacher.log()-x[0,0].log_softmax(-1)[:2])).sum()
    assert torch.allclose(loss,expected)
    loss.backward();assert x.grad[0,0,2]>0 and not x.grad[1].any()
    assert not x.grad[0,0,:2].any()


def test_new_character_labels_are_not_suppressed_by_teacher():
    x=torch.randn(3,2,5,requires_grad=True)
    loss=old_class_mass_penalty(x,3,torch.tensor([3,2]),[False,False])
    loss.backward();assert float(loss.detach())==0 and not x.grad.any()
