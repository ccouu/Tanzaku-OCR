"""Transactional hybrid merge: 8k synthetic train + 500 real train + 1k val + 500 real test.

Reviewed JSONL requires: image (relative to JSONL parent), text, split
(train_real/test_real), orientation_flag, reviewed=true, source_photo_id,
texture_id. Images must already be reviewed normalized 384x64 crops.
All crops from a photograph must stay in the same split.
"""
from src.common.paths import SYNTHETIC_ROOT, asset, dataset_root
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import uuid
from contextlib import contextmanager

from PIL import Image

from src.common import ROOT, digest, dump_json, safe_path
from src.data.lexicon_builder import normalize
from src.data.synthetic_renderer import package


@contextmanager
def staging_directory(parent):
    """Inherited ACLs, including when running in the Windows workspace sandbox."""
    parent = Path(parent).resolve()
    if not parent.is_relative_to(ROOT):
        raise ValueError('Merge staging must stay inside the workspace')
    path = parent / ('stage1-merge-' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().is_relative_to(parent) and path.name.startswith('stage1-merge-'):
            shutil.rmtree(path)


def validate_real(records, base, synthetic, vocab):
    if Counter(r['split'] for r in records) != {'train_real': 500, 'test_real': 500}:
        raise ValueError('Exactly 500 reviewed real train and 500 real test lines are required')
    by_split = {}
    for split in ('train_real', 'test_real'):
        subset = [r for r in records if r['split'] == split]
        texts, photos, textures, hashes = set(), set(), set(), set()
        for row in subset:
            if row.get('reviewed') is not True or row.get('orientation_flag') not in (0, 1):
                raise ValueError('Unreviewed real line or invalid orientation')
            if not row.get('source_photo_id') or not row.get('texture_id'):
                raise ValueError('Real provenance and texture identifiers are required')
            text = normalize(row['text'])
            if not 4 <= len(text) <= 10 or not set(text) <= vocab:
                raise ValueError('Real label length or vocabulary mismatch')
            if text in texts:
                raise ValueError('Duplicate real transcription')
            path = safe_path(base, row['image'])
            with Image.open(path) as image:
                if image.size != (384, 64):
                    raise ValueError('Real crops must be normalized to 384x64')
                image.verify()
            sha = digest(path)
            if sha in hashes:
                raise ValueError('Duplicate real image')
            texts.add(text)
            hashes.add(sha)
            photos.add(row['source_photo_id'])
            textures.add(row['texture_id'])
        by_split[split] = (texts, photos, textures, hashes)
    if any(a & b for a, b in zip(by_split['train_real'], by_split['test_real'])):
        raise ValueError('Train/test string, source photograph, texture or image leakage')
    synth_texts = {normalize(r['text']) for r in synthetic}
    if synth_texts & (by_split['train_real'][0] | by_split['test_real'][0]):
        raise ValueError('Real strings overlap synthetic splits; rebuild with exclusions first')
    synth_textures = {r['texture_id'] for r in synthetic}
    if synth_textures & (by_split['train_real'][2] | by_split['test_real'][2]):
        raise ValueError('Real/synthetic texture overlap')


def merge(root, reviewed, output):
    root, reviewed, output = dataset_root(root).resolve(), Path(reviewed).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Choose a new output directory separate from the source dataset')
    records = [json.loads(line) for line in reviewed.read_text(encoding='utf-8').splitlines() if line.strip()]
    synthetic = [json.loads(line) for line in (root / 'ctc_lines/labels.jsonl').read_text(encoding='utf-8').splitlines()]
    if json.loads((root / 'dataset_manifest.json').read_text())['mode'] != 'synthetic':
        raise ValueError('Merge requires a pristine synthetic source build')
    vocab = set(json.loads((root / 'vocab.json').read_text(encoding='utf-8'))['characters'])
    validate_real(records, reviewed.parent, synthetic, vocab)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Work in a sibling directory; publish only after a full hybrid integrity audit.
    with staging_directory(output.parent) as tmp:
        working = Path(tmp) / 'dataset'
        shutil.copytree(root, working, ignore=shutil.ignore_patterns('menu_dataset_10k.tar.gz', 'verification_report.json'))
        # Materialize shared resources so the new hybrid bundle stays self-contained.
        for name in ('lexicon', 'fonts', 'raw'):
            destination = working / name
            if not destination.exists():
                source = asset(root, name)
                if name == 'raw':
                    destination.mkdir()
                    shutil.copytree(source / 'genai_seeds', destination / 'genai_seeds')
                    for filename in ('JMdict_e.gz', 'jmdict_source.json', 'JMdict-LICENSE.html'):
                        shutil.copy2(source / filename, destination / filename)
                else:
                    shutil.copytree(source, destination)
        for row in synthetic:
            if row['split'] == 'test':
                row['split'] = 'train'
        rows = list(synthetic)
        for i, record in enumerate(records):
            row = dict(record)
            target = f'ctc_lines/images/{row["split"]}/{i:04d}.png'
            path = working / target
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(reviewed.parent, row['image']), path)
            row.update(image=target, text=normalize(row['text']), source='real', sha256=digest(path))
            rows.append(row)
        (working / 'ctc_lines/labels.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        # Synthetic test strings are now training data; move their detector pages
        # into train too. Hybrid detector holdout awaits reviewed full pages.
        pages = json.loads((working / 'yolo_obb/pages.json').read_text(encoding='utf-8'))
        for page in pages:
            if page['split'] == 'test':
                page['split'] = 'train'
                for key in ('image', 'label'):
                    old = working / page[key]
                    relative = page[key].replace('/test/', '/train/')
                    destination = working / relative
                    old.replace(destination)
                    page[key] = relative
        dump_json(working / 'yolo_obb/pages.json', pages)
        counts = Counter(r['split'] for r in rows)
        dump_json(working / 'dataset_manifest.json', dict(stage=1, mode='hybrid', line_count=10000,
            page_count=1000, splits=dict(counts), real_data_pending=False,
            detector_holdout_pending=True, note='Initial synthetic test lines and pages reassigned to train; real test lines replace the CTC holdout.'))
        dump_json(working / 'character_frequency.json', dict(Counter(''.join(r['text'] for r in rows))))
        package(working)
        from scripts.data.verify_stage1_dataset import verify
        verify(working, 100)
        working.replace(output)
    print('Verified hybrid dataset:', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=SYNTHETIC_ROOT)
    parser.add_argument('--reviewed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    merge(args.data_root, args.reviewed, args.output)
