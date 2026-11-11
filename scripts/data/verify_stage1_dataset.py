"""Fail-closed Stage 1 audit. Never imports a model or initiates a network call."""
from src.common.paths import SYNTHETIC_ROOT, asset, dataset_root
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import tarfile
import time

import numpy as np
from PIL import Image

from src.common import ROOT, digest, dump_json, safe_path
from src.data.genai_seed_collector import read_ledger
from src.data.lexicon_builder import CulinaryLookup, normalize, parse_price
from src.data.synthetic_renderer import orientation_tensor


def require(condition, message):
    if not condition:
        raise ValueError(message)


def benchmark_lookup(root):
    lookup = CulinaryLookup(asset(root, 'lexicon/culinary_menu.db'))
    pairs = [('特製鶏肉塩焼き', '特製鳥肉塩焼き'),
             ('自家製豚肉照り焼き', '自家製豚内照り焼き'),
             ('おすすめ牛肉唐揚げ', 'おすすめ牛内唐揚げ')]
    result = {}
    try:
        for kind in ('exact', 'one_typo'):
            times = []
            for i in range(60):
                expected, typo = pairs[i % len(pairs)]
                started = time.perf_counter_ns()
                rows = lookup.search(expected if kind == 'exact' else typo, max_distance=1)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                require(any(r['canonical_jp'] == expected for r in rows), f'Failed {kind} lookup')
                if i >= 6:
                    times.append(elapsed)
            result[kind] = {'median_ms': statistics.median(times), 'p95_ms': sorted(times)[int(len(times)*.95)]}
            require(result[kind]['p95_ms'] < 2, f'{kind} lookup p95 exceeds 2ms: {result[kind]}')
    finally:
        lookup.close()
    return result


def verify(root=SYNTHETIC_ROOT, max_genai_calls=100, report_path=None):
    root = dataset_root(root).resolve()
    manifest = json.loads((root / 'dataset_manifest.json').read_text(encoding='utf-8'))
    mode = manifest['mode']
    require(manifest['stage'] == 1 and mode in ('synthetic', 'hybrid'), 'Invalid Stage 1 mode')
    expected = {'train': 7500, 'validation': 1000, 'test': 500} if mode == 'synthetic' else {'train': 8000, 'train_real': 500, 'validation': 1000, 'test_real': 500}
    ledger = read_ledger(asset(root, 'raw/genai_seeds/usage_ledger.json'))
    require(0 <= max_genai_calls <= 100 and ledger['total_calls'] <= max_genai_calls, 'GenAI cap exceeded')
    generated_images = []
    for event in ledger['events']:
        if event['category'] != 'menus':
            continue
        require(event['status'] == 'completed', 'Incomplete menu image generation request')
        seed_root = asset(root, 'raw/genai_seeds')
        image_path = safe_path(seed_root, event['image'])
        metadata_path = image_path.with_suffix('.json')
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        require(metadata['id'] == event['id'] and metadata['status'] == 'completed', 'Menu seed metadata mismatch')
        require(digest(image_path) == event['sha256'] == metadata['sha256'], 'Menu seed checksum mismatch')
        require(metadata['source'] == 'genai_synthetic' and metadata['reviewed'] is False and metadata['transcription_verified'] is False, 'Generated menus must not be misrepresented as verified real data')
        require(bool(metadata['prompt']) and bool(metadata['intended_text']), 'Missing menu generation prompt')
        with Image.open(image_path) as image:
            require(min(image.size) >= 640 and list(image.size) == metadata['size'], 'Invalid generated menu dimensions')
            image.verify()
        generated_images.append('raw/genai_seeds/' + image_path.relative_to(seed_root).as_posix())
    if manifest.get('requested_menu_image_calls'):
        require(len(generated_images) == manifest['requested_menu_image_calls'] == 100, 'Expected exactly 100 generated menu seeds')
        require(len({e['sha256'] for e in ledger['events'] if e['category'] == 'menus'}) == 100, 'Duplicate generated menu images')
    db = sqlite3.connect(asset(root, 'lexicon/culinary_menu.db'))
    require(db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok', 'Corrupt lexicon')
    require(not db.execute('PRAGMA foreign_key_check').fetchall(), 'Invalid lexicon references')
    size = db.execute('SELECT count(*) FROM menu_dictionary').fetchone()[0]
    require(size >= 20000, 'Lexicon is below 20,000 entries')
    require(db.execute("SELECT count(*) FROM menu_dictionary WHERE source LIKE 'JMdict:%'").fetchone()[0] > 0, 'No public JMdict entries')
    require(db.execute('SELECT count(*) FROM char_bigrams').fetchone()[0] >= size, 'Missing inverted index')
    db.close()
    price_cases = [('八五〇円（税込）', 850, True, None, False, False),
                   ('一串 二百五十円', 250, None, '一串', False, False),
                   ('時価', None, None, None, False, True),
                   ('千二百五十円（税抜）', 1250, False, None, False, False),
                   ('二貫 一、二八〇円〜 外税', 1280, False, '二貫', True, False),
                   ('850〜1200円', 850, None, None, True, False)]
    for text, price, tax, unit, minimum, market in price_cases:
        require(parse_price(text) == dict(price_yen=price, tax_included=tax, unit_counter=unit,
                is_minimum_price=minimum, is_market_price=market), f'Price parser failed: {text}')
    vocab = json.loads((root / 'vocab.json').read_text(encoding='utf-8'))
    chars = vocab['characters']
    require(len(chars) == len(set(chars)) and vocab['blank_id'] == 0, 'Invalid vocabulary')
    require(vocab['char_to_id'] == {c: i + 1 for i, c in enumerate(chars)}, 'Non-contiguous CTC vocabulary')
    cfg = json.loads((root / 'stride_config.json').read_text())
    require(cfg['input_shape'] == [2, 64, 384] and cfg['strides'] == [[2,2],[2,2],[2,1],[2,1],[2,1]], 'Wrong stride configuration')
    width_stride = int(np.prod([s[1] for s in cfg['strides']]))
    require(width_stride == cfg['total_width_stride'] == 4 and 384 // width_stride == cfg['time_steps'] == 96, 'CTC T must equal 96')
    require(cfg['head_pool'] == [2, 1] and cfg['orientation_encoding'] == 'constant_second_channel', 'Invalid head/orientation contract')
    rows = [json.loads(line) for line in (root / 'ctc_lines/labels.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(Counter(r['split'] for r in rows) == expected == manifest['splits'], 'Incorrect split counts')
    require(len(rows) == manifest['line_count'] == sum(expected.values()), 'Wrong line total')
    texts, textures, hashes, photos = defaultdict(set), defaultdict(set), defaultdict(set), defaultdict(set)
    frequency = Counter()
    flags = Counter()
    paths = set()
    for row in rows:
        split = row['split']
        text = normalize(row['text'])
        require(text == row['text'] and 4 <= len(text) <= 10, 'Invalid transcription length/normalization')
        require(set(text) <= set(chars), 'Label outside vocabulary')
        require(text not in texts[split] and row['image'] not in paths, 'Duplicate label or path')
        require(row['orientation_flag'] in (0, 1), 'Invalid orientation flag')
        require(96 >= 2 * len(text) + 1, 'Insufficient CTC time steps')
        image_path = safe_path(root, row['image'])
        sha = digest(image_path)
        require(sha == row['sha256'] and sha not in hashes[split], 'Image checksum mismatch or duplicate')
        with Image.open(image_path) as image:
            require(image.size == (384, 64) and image.mode == 'L', 'Invalid grayscale line dimensions')
            tensor = orientation_tensor(image, row['orientation_flag'])
            require(tensor.shape == (2, 64, 384) and tensor.dtype == np.float32, 'Invalid input tensor')
            require(np.isfinite(tensor).all() and np.all(tensor[1] == row['orientation_flag']), 'Orientation plane mismatch')
            require(tensor[0].min() >= -1 and tensor[0].max() <= 1 and tensor[0].std() > .02, 'Blank or invalid crop')
        if row['source'] == 'real':
            require(row.get('reviewed') is True and row.get('source_photo_id'), 'Unreviewed real line')
            photos[split].add(row['source_photo_id'])
        texts[split].add(text)
        textures[split].add(row['texture_id'])
        hashes[split].add(sha)
        paths.add(row['image'])
        flags[(split, row['orientation_flag'])] += 1
        frequency.update(text)
    for collection in (texts, textures, hashes, photos):
        keys = list(collection)
        for i, left in enumerate(keys):
            for right in keys[i+1:]:
                require(not collection[left] & collection[right], f'{left}/{right} leakage')
    for split in ('train', 'validation') + (('test',) if mode == 'synthetic' else ()):
        require(flags[(split, 0)] == flags[(split, 1)], 'Synthetic orientation split is not 50/50')
    require(all(frequency[c] >= 10 for c in chars), 'Character quota below ten')
    require(json.loads((root / 'character_frequency.json').read_text(encoding='utf-8')) == dict(frequency), 'Stale frequency report')
    pages = json.loads((root / 'yolo_obb/pages.json').read_text(encoding='utf-8'))
    require(len(pages) == manifest['page_count'] == 1000, 'Expected 1,000 OBB pages')
    class_counts = Counter()
    for page in pages:
        with Image.open(safe_path(root, page['image'])) as image:
            require(image.size == (640, 640), 'Invalid full page size')
            image.verify()
        lines = safe_path(root, page['label']).read_text().splitlines()
        require(len(lines) == len(page['items']) >= 3, 'Missing page annotations')
        for line, item in zip(lines, page['items']):
            parts = line.split()
            require(len(parts) == 9 and parts[0] in ('0', '1', '2'), 'Invalid YOLO-OBB row')
            points = np.array([float(p) for p in parts[1:]]).reshape(4, 2)
            require(np.isfinite(points).all() and np.all((points >= 0) & (points <= 1)), 'Out-of-bounds OBB')
            require(int(parts[0]) == item['class_id'] and np.allclose(points * 640, item['corners'], atol=.00001), 'OBB metadata mismatch')
            edges = np.roll(points, -1, axis=0) - points
            next_edges = np.roll(edges, -1, axis=0)
            cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
            require(np.all(cross > 0) or np.all(cross < 0), 'Degenerate/non-convex OBB')
            if item['class_id'] != 2:
                require(item['text'] in texts[page['split']], 'Detector/recognizer cross-split string leakage')
            class_counts[item['class_id']] += 1
    require(set(class_counts) == {0, 1, 2}, 'Missing detector class')
    font_manifest = json.loads((asset(root, 'fonts/manifest.json')).read_text(encoding='utf-8'))
    require(len(font_manifest) == 7, 'Missing font roster')
    for font in font_manifest:
        require(digest(asset(root, 'fonts') / font['file']) == font['sha256'], 'Font checksum mismatch')
        require(digest(asset(root, 'fonts') / (font['file'] + '.license.txt')) == font['license_sha256'], 'Missing font license')
    source = json.loads((asset(root, 'raw/jmdict_source.json')).read_text(encoding='utf-8'))
    require(digest(asset(root, 'raw/JMdict_e.gz')) == source['sha256'], 'JMdict source checksum mismatch')
    archive_path = root / 'menu_dataset_10k.tar.gz'
    archived = set()
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            require(not member.issym() and not member.islnk(), 'Archive symlink')
            safe_path(root, member.name)  # Reject unsafe archive members before resolving shared assets.
            disk_path = asset(root, member.name)
            if member.isfile():
                require(member.name not in archived, 'Duplicate archive member')
                archived.add(member.name)
                h = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(block)
                require(h.hexdigest() == digest(disk_path), 'Archive content mismatch: ' + member.name)
    require(paths <= archived, 'Archive missing line images')
    require(set(generated_images) <= archived and {str(Path(p).with_suffix('.json')).replace('\\', '/') for p in generated_images} <= archived, 'Archive missing generated menu images or prompts')
    require({p['image'] for p in pages} | {p['label'] for p in pages} <= archived, 'Archive missing OBB artifacts')
    require({'vocab.json', 'stride_config.json', 'lexicon/culinary_menu.db', 'dataset_manifest.json', 'DATA_LICENSES.md', 'ctc_lines/labels.jsonl', 'raw/genai_seeds/usage_ledger.json'} <= archived, 'Archive missing required metadata')
    timings = benchmark_lookup(root)
    report = dict(status='passed', stage=1, mode=mode, lines=len(rows), pages=len(pages),
        lexicon_entries=size, vocab_characters=len(chars), minimum_character_frequency=min(frequency[c] for c in chars),
        genai_calls=ledger['total_calls'], generated_menu_images=len(generated_images),
        generated_menu_transcriptions_verified=False, lookup_timings=timings, splits=expected,
        orientation_counts={f'{s}:{f}': n for (s,f),n in flags.items()},
        obb_classes=dict(class_counts), archive_sha256=digest(archive_path),
        synthetic_build_seconds=manifest.get('synthetic_build_seconds'),
        real_data_pending=manifest.get('real_data_pending'), stage2_started=False)
    dump_json(report_path or root / 'verification_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=SYNTHETIC_ROOT)
    parser.add_argument('--max-genai-calls', type=int, default=100)
    parser.add_argument('--report', type=Path, help='Write a new report without replacing historical audit evidence')
    args = parser.parse_args()
    verify(args.data_root, args.max_genai_calls, args.report)
