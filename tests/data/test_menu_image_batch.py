import json

from PIL import Image
import pytest

from scripts.data import menu_image_batch as batch
from src.data.genai_seed_collector import read_ledger


def test_menu_reservation_completion_and_lifetime_cap(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(batch, 'BASE', tmp_path)
    monkeypatch.setattr(batch, 'LEDGER', tmp_path / 'usage_ledger.json')
    batch.reserve(100)
    assert read_ledger(batch.LEDGER)['total_calls'] == 100
    with pytest.raises(ValueError, match='100-call'):
        batch.reserve(1)
    Image.new('RGB', (1024, 1536), 'white').save(tmp_path / 'menu_images/menu-001.png')
    batch.finish(1, True)
    event = read_ledger(batch.LEDGER)['events'][0]
    metadata = json.loads((tmp_path / 'menu_images/menu-001.json').read_text(encoding='utf-8'))
    assert event['status'] == 'completed'
    assert metadata['sha256'] == event['sha256']
    assert metadata['transcription_verified'] is False
    assert metadata['size'] == [1024, 1536]
    batch.finish(2, False)
    assert read_ledger(batch.LEDGER)['total_calls'] == 100
    assert read_ledger(batch.LEDGER)['events'][1]['status'] == 'failed'
