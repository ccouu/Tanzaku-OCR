"""Optional provider-neutral collector. Every attempted call is reserved before dispatch.

The synthetic build makes zero calls. Inject an authenticated provider callable at
the application boundary; neither credentials nor provider exception text is saved.
"""
import contextlib
import json
import os
import time
import uuid
import io
from pathlib import Path

from .common import ROOT, dump_json

CAP = 100
QUOTAS = {'items': 40, 'textures': 40, 'reserve': 20, 'menus': 100}


@contextlib.contextmanager
def ledger_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix('.lock')
    with lock.open('a+b') as f:
        deadline = time.monotonic() + 30
        while True:
            try:
                f.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise TimeoutError('GenAI ledger is locked') from None
                time.sleep(.01)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def read_ledger(path):
    ledger = json.loads(Path(path).read_text(encoding='utf-8'))
    events = ledger['events']
    if ledger['total_calls'] != len(events) or len(events) > CAP:
        raise ValueError('Invalid or over-budget GenAI ledger')
    if any(e['category'] not in QUOTAS or e['id'] != i + 1 for i, e in enumerate(events)):
        raise ValueError('Invalid ledger event')
    for category, cap in QUOTAS.items():
        if sum(e['category'] == category for e in events) > cap:
            raise ValueError('GenAI category budget exceeded')
    return ledger


class SeedCollector:
    def __init__(self, ledger=ROOT / 'data/raw/genai_seeds/usage_ledger.json'):
        self.path = Path(ledger)
        with ledger_lock(self.path):
            if not self.path.exists():
                dump_json(self.path, {'total_calls': 0, 'events': []})
            read_ledger(self.path)

    def call(self, category, provider, prompt):
        if category not in QUOTAS:
            raise ValueError('Unknown collection category')
        with ledger_lock(self.path):
            ledger = read_ledger(self.path)
            if ledger['total_calls'] >= CAP or sum(e['category'] == category for e in ledger['events']) >= QUOTAS[category]:
                raise RuntimeError('GenAI budget exhausted')
            event_id = ledger['total_calls'] + 1
            ledger['events'].append({'id': event_id, 'category': category, 'status': 'reserved'})
            ledger['total_calls'] += 1
            dump_json(self.path, ledger)
        try:
            result = provider(prompt)
        except Exception:
            self._finish(event_id, 'failed')
            raise RuntimeError('Seed provider failed; attempt remains charged') from None
        self._finish(event_id, 'completed')
        return result

    def _finish(self, event_id, status):
        with ledger_lock(self.path):
            ledger = read_ledger(self.path)
            ledger['events'][event_id - 1]['status'] = status
            dump_json(self.path, ledger)

    def collect_items(self, provider):
        prompt = ('Return JSON with an items array containing 50 to 100 authentic regional '
                  'Japanese izakaya, sushi, ramen and yakitori menu entries. Each item must '
                  'have canonical_jp, reading_hiragana, romaji, name_en, description_en, '
                  'category, allergens and dietary_flags. Use unknown for recipe-dependent '
                  'allergens; do not claim unverified food safety properties.')
        result = self.call('items', provider, prompt)
        if isinstance(result, str):
            result = json.loads(result)
        fields = {'canonical_jp', 'reading_hiragana', 'romaji', 'name_en', 'description_en', 'category', 'allergens', 'dietary_flags'}
        items = result['items']
        if not 50 <= len(items) <= 100 or any(not fields <= item.keys() for item in items):
            raise ValueError('Invalid item seed batch; call remains charged')
        output = self.path.parent / ('items-' + uuid.uuid4().hex + '.json')
        dump_json(output, {'reviewed': False, 'items': items})
        return output

    def collect_texture(self, provider):
        prompt = ('Create a high resolution Japanese restaurant menu background texture '
                  'with cedar wood grain, fibrous washi, chalkboard or subtle stain '
                  'overlays. No text, lettering, watermarks or logos. Return image bytes.')
        result = self.call('textures', provider, prompt)
        from PIL import Image
        with Image.open(io.BytesIO(result)) as image:
            if min(image.size) < 640:
                raise ValueError('Texture seed must be at least 640 pixels on each axis')
            output = self.path.parent / ('texture-' + uuid.uuid4().hex + '.png')
            image.convert('RGB').save(output)
        return output

    def collect_headers(self, provider):
        prompt = ('Return JSON with a headers array of rare seasonal Japanese menu '
                  'calligraphy header strings, such as 本日のおすすめ and 店主厳選.')
        result = self.call('reserve', provider, prompt)
        if isinstance(result, str):
            result = json.loads(result)
        if not isinstance(result.get('headers'), list) or not all(isinstance(s, str) and s for s in result['headers']):
            raise ValueError('Invalid header seeds; call remains charged')
        output = self.path.parent / ('headers-' + uuid.uuid4().hex + '.json')
        dump_json(output, {'reviewed': False, 'headers': result['headers']})
        return output


if __name__ == '__main__':
    collector = SeedCollector()
    print(json.dumps(read_ledger(collector.path), indent=2))
