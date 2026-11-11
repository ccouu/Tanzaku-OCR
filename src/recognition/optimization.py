"""V27 AdamW parameter groups; no decay on normalization or bias parameters."""
import torch

def build_optimizer(model, policy='all',backbone_lr=3e-5):
    if policy not in ('all','exclude_norm_bias'):
        raise ValueError('Unknown weight-decay policy')
    buckets = {}
    for name,parameter in model.named_parameters():
        role = 'encoder' if name.startswith('encoder.') else 'head'
        decay = not (policy == 'exclude_norm_bias' and (parameter.ndim <= 1 or name.endswith('.bias')))
        buckets.setdefault((role,decay),[]).append(parameter)
    groups = []
    for role in ('encoder','head'):
        for decay in (True,False):
            parameters = buckets.get((role,decay),[])
            if parameters:
                groups.append(dict(params=parameters,role=role,
                                   group_name=f'{role}_{"decay" if decay else "no_decay"}',
                                   lr=backbone_lr if role == 'encoder' else 1e-3,
                                   weight_decay=.01 if decay else 0.0))
    return torch.optim.AdamW(groups,weight_decay=.01)


def group_summary(optimizer):
    return [dict(name=g['group_name'],role=g['role'],weight_decay=g['weight_decay'],
                 tensors=len(g['params']),parameters=sum(p.numel() for p in g['params']))
            for g in optimizer.param_groups]

def pilot_optimizer(model,backbone_lr,head_lr):
    optimizer = build_optimizer(model,'exclude_norm_bias')
    for group in optimizer.param_groups:
        group['lr'] = backbone_lr if group['role'] == 'encoder' else head_lr
    return optimizer
