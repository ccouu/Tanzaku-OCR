"""Atomic, CPU-portable checkpoint snapshots, SHA-256 verification, and key resolution."""
import hashlib
from pathlib import Path
import numpy as np
import torch


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return value


def save_atomic(path, checkpoint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_atomic_with_sha256(path, checkpoint):
    path = Path(path)
    save_atomic(path, checkpoint)
    return sha256_file(path)


def resolve_checkpoint_state(ckpt_path, arm_key=None, expected_sha256=None):
    """Load checkpoint state dict, optionally verifying SHA-256 and resolving raw/ema key aliases."""
    ckpt_path = Path(ckpt_path)
    actual_sha = sha256_file(ckpt_path)
    if expected_sha256 and actual_sha != expected_sha256:
        raise RuntimeError(f'Checkpoint SHA-256 mismatch for {ckpt_path}: expected {expected_sha256}, got {actual_sha}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if arm_key is not None:
        if arm_key in ckpt:
            return ckpt[arm_key], arm_key, actual_sha, ckpt
        aliases = {'raw': 'model', 'model': 'raw', 'ema': 'ema_model', 'ema_model': 'ema'}
        alt = aliases.get(arm_key)
        if alt and alt in ckpt:
            return ckpt[alt], alt, actual_sha, ckpt
        raise KeyError(f'Key {arm_key!r} not found in {ckpt_path}; keys={list(ckpt.keys())}')
    for candidate in ('ema', 'ema_model', 'raw', 'model'):
        if candidate in ckpt:
            return ckpt[candidate], candidate, actual_sha, ckpt
    return ckpt, None, actual_sha, ckpt
