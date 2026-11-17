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


def cosine_warmup_lr_scale(step, total_steps, warmup_steps=200, min_ratio=0.1):
    """Linear warmup followed by cosine decay down to min_ratio."""
    import math
    if step <= warmup_steps:
        return max(1e-3, float(step) / float(max(1, warmup_steps)))
    progress = min(1.0, float(step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_parameter_controlled_optimizer(
    model,
    module_lrs=None,
    frozen_modules=(),
    freeze_bn_affine=False,
    weight_decay=1e-4,
    default_lr=1e-6,
    weight_decay_policy='all',
):
    """Configure per-module freeze/unfreeze and learning rates purely from parameters.

    Args:
        model: MenuRecognizer (or any nn.Module with named top-level submodules).
        module_lrs: Mapping of module prefix (e.g. 'encoder', 'recurrent') -> lr.
        frozen_modules: Iterable of module prefixes to freeze (requires_grad=False).
        freeze_bn_affine: If True, freeze weight/bias of all BatchNorm layers.
        weight_decay: AdamW weight decay.
        default_lr: Fallback LR for trainable modules not listed in module_lrs.
        weight_decay_policy: 'all' (legacy V72/V73 behavior: apply weight_decay to all
            trainable tensors) or 'exclude_norm_bias' (zero decay on 1D/bias/norm params).
    """
    if weight_decay_policy not in ('all', 'exclude_norm_bias'):
        raise ValueError(f'Unknown weight_decay_policy={weight_decay_policy!r}')
    module_lrs = dict(module_lrs or {})
    frozen_set = set(frozen_modules)
    bn_param_ids = set()
    norm_param_ids = set()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            for p in m.parameters(recurse=False):
                bn_param_ids.add(id(p))
                norm_param_ids.add(id(p))
        elif isinstance(m, (torch.nn.LayerNorm, torch.nn.GroupNorm)) or type(m).__name__ == 'ChannelLayerNorm':
            for p in m.parameters(recurse=False):
                norm_param_ids.add(id(p))

    buckets = {}
    for name, param in model.named_parameters():
        prefix = name.split('.', 1)[0]
        if prefix in frozen_set or (freeze_bn_affine and id(param) in bn_param_ids):
            param.requires_grad = False
            continue
        param.requires_grad = True
        if weight_decay_policy == 'exclude_norm_bias' and (
            param.ndim <= 1 or name.endswith('.bias') or id(param) in norm_param_ids
        ):
            decay_flag = False
        else:
            decay_flag = True
        buckets.setdefault((prefix, decay_flag), []).append(param)

    groups = []
    for (prefix, decay_flag), params in buckets.items():
        lr = float(module_lrs.get(prefix, default_lr))
        wd = float(weight_decay) if decay_flag else 0.0
        gname = prefix if weight_decay_policy == 'all' else f'{prefix}_{"decay" if decay_flag else "no_decay"}'
        groups.append({
            'name': gname,
            'role': prefix,
            'group_name': gname,
            'params': params,
            'lr': lr,
            'weight_decay': wd,
        })

    if not groups:
        raise ValueError('No trainable parameters left after applying freeze policy')

    opt = torch.optim.AdamW(groups, weight_decay=float(weight_decay))
    summary = {
        'frozen_modules': sorted(frozen_set),
        'freeze_bn_affine': bool(freeze_bn_affine),
        'weight_decay_policy': weight_decay_policy,
        'weight_decay': float(weight_decay),
        'trainable_tensors': sum(1 for p in model.parameters() if p.requires_grad),
        'frozen_tensors': sum(1 for p in model.parameters() if not p.requires_grad),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'frozen_params': sum(p.numel() for p in model.parameters() if not p.requires_grad),
        'groups': [
            {'name': g['name'], 'role': g['role'], 'lr': g['lr'], 'weight_decay': g['weight_decay'], 'tensors': len(g['params'])}
            for g in groups
        ],
    }
    return opt, summary


def snapshot_frozen_params(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if not p.requires_grad}


def verify_frozen_params_identical(model, baseline):
    named = dict(model.named_parameters())
    for k, v0 in baseline.items():
        if not torch.equal(named[k].detach().cpu(), v0):
            raise RuntimeError(f'Frozen parameter mutated: {k}')
    return len(baseline)
