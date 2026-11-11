"""V27 training utilities, extracted unchanged from the historical experiment files."""
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, document):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding='utf-8')
    # Windows readers can briefly hold a handle without delete-sharing permission.
    for attempt in range(10):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(.05)


def training_device(name='cuda', required_gpu='A100'):
    device = torch.device(name)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('A working CUDA GPU is required; CPU fallback is disabled')
    gpu = torch.cuda.get_device_name(device)
    if required_gpu and required_gpu.casefold() not in gpu.casefold():
        raise RuntimeError(f'Expected GPU containing {required_gpu!r}; found {gpu}')
    with torch.cuda.device(device):
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError('This experiment requires native BF16 support')
    return device


def loss_for(model, batch, device, criterion):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = model(batch['images'].to(device), batch['orientation'].to(device), batch['input_lengths'])
    with torch.autocast('cuda', enabled=False):
        loss = criterion(logits.float().log_softmax(-1), batch['targets'].to(device),
                         batch['input_lengths'], batch['lengths'])
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite CTC loss; invalid alignments must not be hidden')
    return loss

def seed_all(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def fit_margins(glyph_width,head,tail,canvas):
    """Keep glyph pixels and proportionally cap only overflowing background."""
    available=canvas-glyph_width
    if min(glyph_width,canvas)<1 or available<0 or min(head,tail)<0:
        raise ValueError('Invalid glyph/canvas/margins')
    if head+tail<=available:
        return head,tail,False
    actual_head=head*available//(head+tail)
    return actual_head,available-actual_head,True
