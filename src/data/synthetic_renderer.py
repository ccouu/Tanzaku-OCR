"""Stage 1 dual CTC / OBB renderer; Pillow size is (width,height)."""
from src.common.paths import SYNTHETIC_ROOT, asset, dataset_root
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import random
import tarfile
import time
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from fontTools.ttLib import TTFont
from scipy.ndimage import gaussian_filter, map_coordinates

from .common import ROOT, digest, dump_json
from .genai_seed_collector import SeedCollector
from .menu_grammar import generate

VERTICAL_MAP = {'ー': '｜', '「': '﹁', '」': '﹂', '（': '︵', '）': '︶', '(': '︵', ')': '︶'}
SPLITS = {'train': 7500, 'validation': 1000, 'test': 500}


class FontRoster:
    def __init__(self, directory):
        self.paths = sorted(Path(directory).glob('*.ttf'))
        if len(self.paths) != 7:
            raise ValueError('All seven verified fonts are required')
        self.cmaps = []
        for p in self.paths:
            with TTFont(p) as font:
                self.cmaps.append(set(font.getBestCmap()))

    @lru_cache(maxsize=512)
    def font(self, index, size):
        return ImageFont.truetype(str(self.paths[index]), size)

    def index_for(self, char, preferred):
        for i in [preferred] + list(range(len(self.paths))):
            if ord(char) in self.cmaps[i]:
                return i
        raise ValueError(f'No font glyph for U+{ord(char):04X}')


@lru_cache(maxsize=32768)
def glyph_tile(char, roster, preferred, size, vertical=False):
    tile = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(tile)
    if vertical and char == 'ー':
        draw.line((size // 2, size // 8, size // 2, size * 7 // 8), fill=235, width=max(1, size // 16))
        return tile
    mapped = VERTICAL_MAP.get(char, char) if vertical else char
    if mapped != char and not any(ord(mapped) in cmap for cmap in roster.cmaps):
        # Vertical presentation forms are not encoded in every font. Rotate the
        # source bracket in its cell; never emit a tofu replacement glyph.
        return glyph_tile(char, roster, preferred, size, False).transpose(Image.Transpose.ROTATE_270)
    index = roster.index_for(mapped, preferred)
    font = roster.font(index, max(8, int(size * .83)))
    box = draw.textbbox((0, 0), mapped, font=font)
    draw.text(((size - (box[2] - box[0])) / 2 - box[0],
               (size - (box[3] - box[1])) / 2 - box[1]), mapped, font=font, fill=235)
    return tile


def render_upright_vertical(text, roster, font_index=0):
    strip = Image.new('L', (64, 384), 0)
    step = min(52, 372 // len(text))
    offset = (384 - step * len(text)) // 2
    for i, char in enumerate(text):
        tile = glyph_tile(char, roster, font_index, step, True)
        strip.paste(tile, ((64 - step) // 2, offset + i * step))
    return strip


def render_mask(text, orientation_flag, roster, font_index=0):
    if not 4 <= len(text) <= 10:
        raise ValueError('Line labels must have 4–10 characters')
    if orientation_flag == 1:
        return render_upright_vertical(text, roster, font_index).transpose(Image.Transpose.ROTATE_90)
    if orientation_flag != 0:
        raise ValueError('Orientation flag must be 0 or 1')
    strip = Image.new('L', (384, 64), 0)
    step = min(52, 372 // len(text))
    offset = (384 - step * len(text)) // 2
    for i, char in enumerate(text):
        tile = glyph_tile(char, roster, font_index, step)
        strip.paste(tile, (offset + i * step, (64 - step) // 2))
    return strip


def make_background(size, kind, seed):
    rng = np.random.default_rng(seed)
    w, h = size
    y, x = np.mgrid[:h, :w]
    if kind == 'wood':
        base = 188 + 10 * np.sin(y / 4 + np.sin(x / 85)) + 4 * np.sin(y / 1.7)
    elif kind == 'chalk':
        base = np.full((h, w), 42.)
    else:
        base = np.full((h, w), 232.) + gaussian_filter(rng.normal(0, 7, (h, w)), (1, 4))
    gradient = (x / max(w, 1) - .5) * rng.uniform(-30, 30)
    return Image.fromarray(np.uint8(np.clip(base + gradient + rng.normal(0, 2, (h, w)), 0, 255)))


def augment_mask(mask, rng):
    # Elastic stroke displacement in normalized-strip coordinates.
    arr = np.asarray(mask, dtype=np.float32)
    y, x = np.mgrid[:64, :384].astype(np.float32)
    noise = np.random.default_rng(rng.randrange(2**32))
    dx = gaussian_filter(noise.normal(size=(64, 384)), 7) * 9
    dy = gaussian_filter(noise.normal(size=(64, 384)), 7) * 9
    mask = Image.fromarray(np.uint8(map_coordinates(arr, [y + dy, x + dx], order=1, mode='constant')))
    style = rng.randrange(4)
    if style == 0:
        mask = mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(.25))
    elif style == 1:
        mask = Image.blend(mask, mask.filter(ImageFilter.MinFilter(3)), .5)
    # Small perspective coefficients preserve padding and label legibility.
    return mask.transform(mask.size, Image.Transform.PERSPECTIVE,
                          (1, rng.uniform(-.018, .018), 0, rng.uniform(-.005, .005), 1, 0,
                           rng.uniform(-.000025, .000025), rng.uniform(-.000025, .000025)), Image.Resampling.BILINEAR)


def blend(background, mask, kind):
    ink = Image.new('L', background.size, 235 if kind == 'chalk' else 20)
    return Image.composite(ink, background, mask)


def orientation_tensor(image, orientation_flag):
    if orientation_flag not in (0, 1):
        raise ValueError('Invalid orientation')
    arr = np.asarray(image.convert('L'), dtype=np.float32)
    if arr.shape != (64, 384):
        raise ValueError('Incorrect line dimensions')
    return np.stack((arr / 127.5 - 1, np.full_like(arr, orientation_flag)))


def rotated_corners(cx, cy, width, height, angle):
    c, s = math.cos(angle), math.sin(angle)
    return [(cx + dx * c - dy * s, cy + dx * s + dy * c)
            for dx, dy in [(-width/2, -height/2), (width/2, -height/2),
                           (width/2, height/2), (-width/2, height/2)]]


def place_patch(page, patch, cx, cy, angle):
    # PIL positive angles are CCW on screen; OBB uses x-right/y-down.
    rgba = patch.convert('RGBA').rotate(-math.degrees(angle), expand=True, resample=Image.Resampling.BICUBIC)
    page.paste(rgba.convert('L'), (round(cx - rgba.width / 2), round(cy - rgba.height / 2)), rgba.getchannel('A'))
    return rotated_corners(cx, cy, patch.width, patch.height, angle)


def build_pages(root, roster, rows, rng):
    metadata = []
    for page_id in range(1000):
        split = 'train' if page_id < 800 else 'validation' if page_id < 900 else 'test'
        pool = rows[split]
        kind = ['wood', 'washi', 'chalk'][page_id % 3]
        texture_id = f'{split}-page-{page_id:04d}'
        page = make_background((640, 640), kind, 100000 + page_id)
        annotations, items = [], []
        # Isolated horizontal rows at left and tategaki columns at right.
        for j in range(6):
            record = pool[(page_id * 6 + j) % len(pool)]
            flag = j % 2
            mask = render_mask(record['text'], flag, roster, page_id % 7)
            if flag:
                mask = mask.transpose(Image.Transpose.ROTATE_270)
            size = (240, 40) if not flag else (40, 240)
            mask = mask.resize(size, Image.Resampling.LANCZOS)
            patch = blend(make_background(size, kind, 1000000 + page_id * 6 + j), mask, kind)
            cx, cy = ((160, 100 + (j // 2) * 165) if not flag else (360 + (j // 2) * 95, 180))
            angle = rng.uniform(-.04, .04)
            corners = place_patch(page, patch, cx, cy, angle)
            annotations.append(str(flag) + ' ' + ' '.join(f'{v / 640:.8f}' for xy in corners for v in xy))
            items.append(dict(class_id=flag, text=record['text'], corners=corners))
        # Price class rendered independently below the columns.
        for j in range(3):
            text = f'{300 + 50 * ((page_id + j) % 20)}円'
            mask = render_mask(text, 0, roster, page_id % 7).resize((80, 28))
            patch = blend(make_background((80, 28), kind, 2000000 + page_id * 3 + j), mask, kind)
            corners = place_patch(page, patch, 360 + 95 * j, 345, 0)
            annotations.append('2 ' + ' '.join(f'{v / 640:.8f}' for xy in corners for v in xy))
            items.append(dict(class_id=2, text=text, corners=corners))
        image = root / f'yolo_obb/images/{split}/{page_id:04d}.jpg'
        label = root / f'yolo_obb/labels/{split}/{page_id:04d}.txt'
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        page.save(image, quality=90)
        label.write_text('\n'.join(annotations) + '\n', encoding='utf-8')
        metadata.append(dict(image=image.relative_to(root).as_posix(), label=label.relative_to(root).as_posix(),
                             split=split, texture_id=texture_id, items=items))
    dump_json(root / 'yolo_obb/pages.json', metadata)
    (root / 'yolo_obb/dataset.yaml').write_text('path: .\ntrain: images/train\nval: images/validation\ntest: images/test\nnames:\n  0: dish_horizontal\n  1: dish_vertical\n  2: price_tag\n', encoding='utf-8')


def package(root):
    root = dataset_root(root)
    files = ['ctc_lines', 'yolo_obb', 'lexicon', 'vocab.json', 'stride_config.json',
             'character_frequency.json', 'dataset_manifest.json', 'raw/genai_seeds', 'DATA_LICENSES.md']
    tmp = root / 'menu_dataset_10k.tar.gz.tmp'
    with tarfile.open(tmp, 'w:gz', compresslevel=1) as archive:
        for name in files:
            archive.add(asset(root, name), arcname=name)
    tmp.replace(root / 'menu_dataset_10k.tar.gz')


_worker_context = threading.local()


def _worker_init(fontdir):
    _worker_context.roster = FontRoster(fontdir)


def _render_line(task):
    root, split, i, text, seed = task
    roster = _worker_context.roster
    flag = i % 2
    texture_seed = {'train': 1000, 'validation': 20000, 'test': 30000}[split] + i
    kind = ['wood', 'washi', 'chalk'][i % 3]
    rng = random.Random(seed + texture_seed)
    mask = augment_mask(render_mask(text, flag, roster, i % 7), rng)
    image = blend(make_background((384, 64), kind, texture_seed), mask, kind)
    relative = f'ctc_lines/images/{split}/{i:05d}.png'
    p = root / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    image.save(p, compress_level=1)
    return dict(image=relative, text=text, orientation_flag=flag, split=split,
                source='synthetic', texture_id=f'{split}-line-{i:05d}', texture_seed=texture_seed,
                background=kind, preferred_font=roster.paths[i % 7].name, sha256=digest(p))


def build(root=SYNTHETIC_ROOT, seed=20261107, workers=None):
    root = dataset_root(root)
    started = time.perf_counter()
    SeedCollector(asset(root, 'raw/genai_seeds/usage_ledger.json'))
    roster = FontRoster(asset(root, 'fonts'))
    words = generate(root, seed=seed)
    rng = random.Random(seed)
    rows, tasks = {}, []
    cursor = 0
    for split, count in SPLITS.items():
        for i, text in enumerate(words[cursor:cursor + count]):
            tasks.append((root, split, i, text, seed))
        cursor += count
    workers = workers or min(8, os.cpu_count() or 1)
    if workers == 1:
        _worker_init(asset(root, 'fonts'))
        all_rows = list(map(_render_line, tasks))
    else:
        with ThreadPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(asset(root, 'fonts'),)) as pool:
            all_rows = list(pool.map(_render_line, tasks, chunksize=64))
    for split in SPLITS:
        rows[split] = [r for r in all_rows if r['split'] == split]
    labels = root / 'ctc_lines/labels.jsonl'
    labels.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in all_rows), encoding='utf-8')
    build_pages(root, roster, rows, rng)
    dump_json(root / 'stride_config.json', dict(input_shape=[2, 64, 384],
        strides=[[2, 2], [2, 2], [2, 1], [2, 1], [2, 1]],
        head_pool=[2, 1], total_width_stride=4, time_steps=96,
        orientation_encoding='constant_second_channel', blank_id=0))
    (root / 'DATA_LICENSES.md').write_text('''# Stage 1 data sources

This dataset uses JMdict, copyright James William BREEN and the Electronic
Dictionary Research and Development Group (EDRDG), under CC BY-SA 4.0.
Source: https://www.edrdg.org/wiki/index.php/JMdict-EDICT_Dictionary_Project
License: https://www.edrdg.org/edrdg/licence.html
Full code: https://creativecommons.org/licenses/by-sa/4.0/legalcode
The derived lexicon and labels are distributed under CC BY-SA 4.0.
Generated menu variants are not independently attested dishes. Descriptions
are dictionary glosses or deterministic component translations. Allergen and
dietary flags are unknown and require recipe-specific verification.

Five Google Fonts use the SIL Open Font License; Kouzan and Aoyagi fonts
permit commercial use per their publisher. Font binaries are kept separately
in data/resources/fonts with publisher notices, source URLs and checksums. The archive
contains rendered images, not redistributed font binaries. Backgrounds are
procedural and use no external photographs. No GenAI requests were needed.
''', encoding='utf-8')
    dump_json(root / 'dataset_manifest.json', dict(stage=1, mode='synthetic', line_count=9000,
        page_count=1000, splits=SPLITS, seed=seed, real_data_pending=True,
        synthetic_build_seconds=round(time.perf_counter() - started, 3),
        note='Plan initial build is 9,000 lines plus 1,000 pages; archive name is the future hybrid target.'))
    package(root)
    print('Stage 1 synthetic archive built in', round(time.perf_counter() - started, 2), 'seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=SYNTHETIC_ROOT)
    parser.add_argument('--seed', type=int, default=20261107)
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    build(args.data_root, args.seed, args.workers)
