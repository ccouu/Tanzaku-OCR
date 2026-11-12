"""Frozen-teacher full-softmax mass penalty on old-only replay frames."""
import torch
from torch.nn import functional as F


def old_class_mass_penalty(logits,old_classes,lengths,old_only):
    # With frozen original logits, KL(teacher_old || student_full) equals
    # log(1 + Z_new/Z_old). Unlike renormalizing student over old classes,
    # this penalizes false new-token probability, including on blank frames.
    x=logits.float()
    penalty=F.softplus(torch.logsumexp(x[...,old_classes:],-1)-torch.logsumexp(x[...,:old_classes].detach(),-1))
    valid=torch.arange(x.shape[0],device=x.device)[:,None]<lengths.to(x.device)[None,:]
    valid=valid & torch.as_tensor(old_only,device=x.device,dtype=torch.bool)[None,:]
    return penalty[valid].mean() if valid.any() else x[...,old_classes:].sum()*0


def teacher_kl(logits,teacher,lengths,temperature=2.):
    """Full-vocabulary teacher consistency on valid sequence frames only."""
    if logits.shape!=teacher.shape or temperature<=0:raise ValueError('Invalid teacher logits or temperature')
    logp=(logits.float()/temperature).log_softmax(-1)
    q=(teacher.detach().float()/temperature).softmax(-1)
    loss=F.kl_div(logp,q,reduction='none').sum(-1)*temperature**2
    valid=torch.arange(logits.shape[0],device=logits.device)[:,None]<lengths.to(logits.device)[None,:]
    return loss[valid].mean()
