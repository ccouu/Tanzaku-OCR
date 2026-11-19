"""Resolve and load the explicitly promoted recognizer; no silent fallback."""
import json
from pathlib import Path
from .utils import sha

REGISTRY=Path(__file__).with_name('current_model.json')
ROOT=Path(__file__).resolve().parents[2]


def _registry_asset(metadata, field, hash_field, verify=True):
    path=(ROOT/metadata[field]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise FileNotFoundError(f'Promoted recognition asset missing or outside workspace: {path}')
    if verify and sha(path)!=metadata[hash_field]:
        raise ValueError(f'Promoted {field} checksum mismatch')
    return path


def current_checkpoint(verify=True):
    metadata=json.loads(REGISTRY.read_text(encoding='utf-8'))
    return _registry_asset(metadata,'checkpoint','sha256',verify)


def load_current_recognizer(device='cpu'):
    import torch
    from .model import ARCH_SPECS, MenuRecognizer
    metadata=json.loads(REGISTRY.read_text(encoding='utf-8'))
    path=_registry_asset(metadata,'checkpoint','sha256')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if metadata.get('vocabulary'):
        vocab_path=_registry_asset(metadata,'vocabulary','vocab_sha256')
        vocab=json.loads(vocab_path.read_text(encoding='utf-8'))
    else:
        vocab=saved['vocab']
    mapping=vocab['char_to_id']
    if (len(mapping)!=metadata['characters'] or vocab.get('blank_id',0)!=0
            or sorted(mapping.values())!=list(range(1,len(mapping)+1))):
        raise ValueError('Promoted vocabulary IDs/count/blank are incompatible')
    architecture=metadata.get('architecture','mean')
    if architecture=='mean':
        kwargs={'height_pool':'mean'}
    elif architecture in ARCH_SPECS:
        kwargs=dict(ARCH_SPECS[architecture])
    else:
        raise ValueError(f'Unknown promoted recognizer architecture: {architecture}')
    key=metadata.get('state_key',metadata.get('arm','raw'))
    aliases={'raw':'model','model':'raw','ema':'ema_model','ema_model':'ema'}
    if key not in saved:
        key=aliases.get(key,key)
    if key not in saved:
        raise KeyError(f'Promoted state {metadata.get("state_key",metadata.get("arm"))!r} missing')
    state=saved[key]
    model=MenuRecognizer(len(mapping)+1,pretrained=False,use_bilstm=True,**kwargs)
    model.load_state_dict(state,strict=True);model.vocabulary=mapping
    # Normalize the returned in-memory view for existing callers; never rewrite weights.
    saved={**saved,'vocab':vocab,'model':state,'active_state_key':key,
           'active_architecture':architecture}
    return model.to(device).eval(),saved
