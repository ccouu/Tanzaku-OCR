"""V27 atomic, CPU-portable checkpoint snapshots; recovery is in font_training."""
from pathlib import Path
import torch

def cpu_snapshot(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,np.ndarray):
        return value.copy()
    if isinstance(value,dict):
        return {k:cpu_snapshot(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value,tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return value


def save_atomic(path,checkpoint):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint,temporary)
    temporary.replace(path)
