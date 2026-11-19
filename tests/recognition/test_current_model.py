import hashlib
import json

import pytest
import torch

from src.recognition import current, model as model_module


def _fixture(tmp_path, monkeypatch, *, legacy=False):
    vocab={'blank_id':0,'characters':['円'],'char_to_id':{'円':1}}
    checkpoint=tmp_path/'weights.pt'
    state={'weight':torch.tensor([2.0])}
    torch.save({'model':state,'vocab':vocab} if legacy else
               {'raw':{'weight':torch.tensor([1.0])},'ema':state},checkpoint)
    vocabulary=tmp_path/'vocab.json'
    vocabulary.write_text(json.dumps(vocab),encoding='utf-8')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    metadata={'checkpoint':'weights.pt','sha256':digest(checkpoint),'characters':1,
              'arm':'raw' if legacy else 'ema'}
    if not legacy:
        metadata.update(architecture='h4_k41',state_key='ema',vocabulary='vocab.json',
                        vocab_sha256=digest(vocabulary))
    registry=tmp_path/'current.json'
    registry.write_text(json.dumps(metadata),encoding='utf-8')
    monkeypatch.setattr(current,'ROOT',tmp_path)
    monkeypatch.setattr(current,'REGISTRY',registry)

    class TinyModel(torch.nn.Module):
        def __init__(self,classes,**kwargs):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.zeros(1))
            self.kwargs=kwargs
    monkeypatch.setattr(model_module,'MenuRecognizer',TinyModel)
    return metadata,registry,vocabulary


def test_promoted_h4_uses_ema_and_sidecar(tmp_path,monkeypatch):
    _fixture(tmp_path,monkeypatch)
    loaded,saved=current.load_current_recognizer()
    assert loaded.weight.item()==2.0
    assert loaded.kwargs['height_pool']=='conv2d_h4'
    assert loaded.kwargs['expected_h']==4
    assert not loaded.training and saved['active_state_key']=='ema'
    assert saved['vocab']['char_to_id']=={'円':1}


def test_legacy_mean_model_alias_still_loads(tmp_path,monkeypatch):
    _fixture(tmp_path,monkeypatch,legacy=True)
    loaded,saved=current.load_current_recognizer()
    assert loaded.weight.item()==2.0 and loaded.kwargs['height_pool']=='mean'
    assert saved['active_state_key']=='model'


def test_vocab_tampering_rejected(tmp_path,monkeypatch):
    _,_,vocab=_fixture(tmp_path,monkeypatch)
    vocab.write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='checksum'):
        current.load_current_recognizer()


def test_unknown_architecture_and_workspace_escape_rejected(tmp_path,monkeypatch):
    metadata,registry,_=_fixture(tmp_path,monkeypatch)
    metadata['architecture']='unknown'
    registry.write_text(json.dumps(metadata),encoding='utf-8')
    with pytest.raises(ValueError,match='Unknown promoted'):
        current.load_current_recognizer()
    metadata['checkpoint']='../outside.pt'
    registry.write_text(json.dumps(metadata),encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        current.current_checkpoint()
