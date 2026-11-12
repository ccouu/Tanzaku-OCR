"""Forward-only pooled BN statistics; parameters and non-BN modes untouched."""
import torch
from torch import nn


class PooledMoments:
    def __init__(self):
        self.count=0;self.batches=0;self.mean=None;self.m2=None

    def update(self,x):
        dims=(0,*range(2,x.ndim))
        var,mean=torch.var_mean(x.detach().float(),dim=dims,correction=0)
        var=var.double().cpu();mean=mean.double().cpu()
        n=x.numel()//x.shape[1]
        if self.count==0:self.mean=mean;self.m2=var*n
        else:
            delta=mean-self.mean;total=self.count+n
            self.m2+=var*n+delta.square()*(self.count*n/total)
            self.mean+=delta*(n/total)
        self.count+=n;self.batches+=1

    def variance(self):
        if self.count<2:raise ValueError('Insufficient BN calibration observations')
        return self.m2/(self.count-1)


def recalibrate_bn(model,batches,forward):
    """One pass: BN uses batch statistics, others eval; swap pooled input moments.

    Population aggregation is weighted by actual activation count, including
    varying physical widths/partial batches. Variance includes between-batch
    mean differences. Padding is included just as in ordinary inference BN.
    Deeper-layer inputs are generated with upstream batch-mode BN (AdaBN),
    not an iterative solution for final inference-mode activation moments.
    """
    layers={n:m for n,m in model.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)}
    if not layers:raise ValueError('No BN layers to calibrate')
    if any(not m.track_running_stats for m in layers.values()):raise ValueError('Tracked BN statistics required')
    modes={n:m.training for n,m in model.named_modules()};momenta={n:m.momentum for n,m in layers.items()}
    moments={n:PooledMoments() for n in layers};hooks=[]
    try:
        model.eval()
        for name,layer in layers.items():
            # Avoid recursively enabling dropout children in fused BN/activation.
            layer.training=True;layer.momentum=0.
            hooks.append(layer.register_forward_pre_hook(lambda m,args,n=name:moments[n].update(args[0])))
        with torch.no_grad():
            for batch in batches:forward(model,batch)
        if any(v.count<2 for v in moments.values()):raise ValueError('Unvisited or insufficiently calibrated BN layer')
        with torch.no_grad():
            for name,layer in layers.items():
                stats=moments[name]
                layer.running_mean.copy_(stats.mean.to(layer.running_mean))
                layer.running_var.copy_(stats.variance().to(layer.running_var))
                layer.num_batches_tracked.fill_(stats.batches)
        return {n:dict(activation_count=s.count,batches=s.batches) for n,s in moments.items()}
    finally:
        for hook in hooks:hook.remove()
        for n,layer in layers.items():layer.momentum=momenta[n]
        for n,layer in model.named_modules():layer.training=modes[n]
