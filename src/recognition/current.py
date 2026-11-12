"""Resolve and load the explicitly promoted recognizer; no silent fallback."""
import json
from pathlib import Path
from .utils import sha

REGISTRY=Path(__file__).with_name('current_model.json')


def current_checkpoint(verify=True):
    metadata=json.loads(REGISTRY.read_text(encoding='utf-8'))
    root=Path(__file__).resolve().parents[2]
    path=(root/metadata['checkpoint']).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f'Promoted recognition checkpoint missing: {path}')
    if verify and sha(path)!=metadata['sha256']:
        raise ValueError('Promoted checkpoint checksum mismatch')
    return path


def load_current_recognizer(device='cpu'):
    import torch
    from .model import MenuRecognizer
    saved=torch.load(current_checkpoint(),map_location='cpu',weights_only=False)
    mapping=saved['vocab']['char_to_id']
    model=MenuRecognizer(len(mapping)+1,use_bilstm=True)
    model.load_state_dict(saved['model'],strict=True);model.vocabulary=mapping
    return model.to(device).eval(),saved
