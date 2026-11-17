import json
import os
import sys
import time
from pathlib import Path
import pytest
from scripts.recognition import cleanup_after_v46 as cleanup


def configure(monkeypatch,tmp_path):
    monkeypatch.setattr(cleanup,'ROOT',tmp_path.resolve())
    monkeypatch.setattr(cleanup,'WORK',tmp_path/'run')
    monkeypatch.setattr(cleanup,'AUDIT',tmp_path/'audit')
    return tmp_path


def test_cleanup_rejects_root_and_external_paths(monkeypatch,tmp_path):
    root=configure(monkeypatch,tmp_path)
    for p in (root,root.parent/'outside'):
        with pytest.raises(AssertionError):cleanup.checked(p)


def test_cleanup_execute_requires_completed_training(monkeypatch,tmp_path):
    root=configure(monkeypatch,tmp_path)
    cache=root/'.cache/test-runs';cache.mkdir(parents=True)
    item=cache/'keep.txt';item.write_text('not completed')
    monkeypatch.setattr(sys,'argv',['cleanup','--execute'])
    with pytest.raises(FileNotFoundError):cleanup.main()
    assert item.read_text()=='not completed'


def test_audit_only_cannot_unlink_and_preserves_unique_dataset(monkeypatch,tmp_path):
    root=configure(monkeypatch,tmp_path)
    registry=root/'registry.json';registry.write_text('{}')
    monkeypatch.setattr(cleanup,'REGISTRY',registry)
    active=root/'checkpoints/menu_recognizer_v43.pt'
    monkeypatch.setattr(cleanup,'current_checkpoint',lambda:active)
    for path in (active,root/'checkpoints/menu_recognizer_v38.pt',root/'outputs/recognition/pilots/v43_digits_prices_2000/final_raw.pt'):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'protected')
    cache=root/'.cache/test-runs';cache.mkdir(parents=True)
    item=cache/'fixture.bin';item.write_bytes(b'disposable')
    os.utime(item,(time.time()-7200,time.time()-7200))
    duplicate=root/'data/processed/gemini_recognition';canonical=root/'data/processed/gemini_banana_mobile_menus_50_crops'
    for folder in (duplicate,canonical):folder.mkdir(parents=True)
    (duplicate/'image.png').write_bytes(b'unique')
    (canonical/'image.png').write_bytes(b'different')
    monkeypatch.setattr(sys,'argv',['cleanup','--audit-only'])
    def forbidden(*args,**kwargs):raise AssertionError('Audit attempted deletion')
    monkeypatch.setattr(Path,'unlink',forbidden)
    cleanup.main()
    manifest=json.loads((root/'audit/deletion_manifest.json').read_text())
    assert len(manifest['candidates'])==1 and manifest['candidates'][0]['path'].endswith('fixture.bin')
    assert any(r['path']=='data/processed/gemini_recognition' for r in manifest['skipped'])
    assert item.exists() and (duplicate/'image.png').read_bytes()==b'unique'
