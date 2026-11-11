"""Zero-token line grammar and measured character coverage; no invented rare Kanji."""
from src.common.paths import asset
from collections import Counter, defaultdict
import random
import re
import sqlite3
from pathlib import Path

from fontTools.ttLib import TTFont

from .common import dump_json
from .lexicon_builder import components, MODIFIERS, normalize


def generate(data_root, count=9000, seed=20261107):
    root = Path(data_root)
    rng = random.Random(seed)
    db = sqlite3.connect(asset(root, 'lexicon/culinary_menu.db'))
    words = {r[0] for r in db.execute('SELECT canonical_jp FROM menu_dictionary')
             if 4 <= len(r[0]) <= 10 and re.fullmatch(r'[\u3041-\u3096\u30a1-\u30fa\u3400-\u9fffー々ヶ〇]+', r[0])}
    db.close()
    cmaps = []
    for p in sorted((asset(root, 'fonts')).glob('*.ttf')):
        with TTFont(p) as font:
            cmaps.append(set(font.getBestCmap()))
    supported = set.union(*cmaps)
    words = {w for w in words if all(ord(c) in supported for c in w)}
    # Include price/counter/punctuation examples as full text lines.
    for n in range(100, 2001, 10):
        words.add(f'{n}円（税込）')
        words.add(f'一串{n}円')
    words.update(['八五〇円（税込）', '千二百五十円', '一串二百五十円', '本日の時価',
                  '「ラーメン」', '（チャーハン）', 'らーめん定食', 'ビール一杯', '二貫五百円'])
    words = {normalize(w) for w in words}
    # Repeat scarce culinary characters through contextual prefixes/suffixes,
    # rather than padding labels with isolated or random characters.
    initial = sorted(words)
    frequency = Counter(''.join(initial))
    scarce = {c for c, n in frequency.items() if n < 15}
    prefixes = [m[0] for m in components(MODIFIERS)] + ['お好み', '小皿', '大皿', '珍味', '逸品']
    for word in initial:
        if scarce.intersection(word):
            for prefix in prefixes:
                if len(prefix + word) <= 10:
                    words.add(prefix + word)
            # At most ten-char labels; append numerals to long terms only as prices.
            if len(word) >= 8:
                for suffix in ['並', '大', '小', '上', '特上', '定食', '御膳', '一皿', '一杯', '一人前']:
                    if len(word + suffix) <= 10:
                        words.add(word + suffix)
    # Source terms of length ten with rare characters cannot receive a natural
    # variation. Keep them out of the target vocabulary rather than fake coverage.
    frequency = Counter(''.join(words))
    target = set(c for c, n in frequency.most_common(1800) if n >= 10)
    words = {w for w in words if set(w) <= target}
    target = set(''.join(words))
    pool = sorted(words)
    rng.shuffle(pool)
    by_char = defaultdict(list)
    for word in pool:
        for c in set(word):
            by_char[c].append(word)
    chosen, seen, counts = [], set(), Counter()
    for c in sorted(target, key=lambda c: (len(by_char[c]), c)):
        for word in by_char[c]:
            if counts[c] >= 10:
                break
            if word not in seen:
                chosen.append(word)
                seen.add(word)
                counts.update(word)
    for word in pool:
        if len(chosen) >= count:
            break
        if word not in seen:
            chosen.append(word)
            seen.add(word)
            counts.update(word)
    if len(chosen) != count or any(counts[c] < 10 for c in target):
        raise ValueError('Requested dataset cannot satisfy natural-string character quota')
    rng.shuffle(chosen)
    vocab = {'blank_id': 0, 'characters': sorted(target),
             'char_to_id': {c: i + 1 for i, c in enumerate(sorted(target))},
             'minimum_frequency': 10, 'provenance': 'Characters occurring in licensed culinary lexicon and menu grammar; no arbitrary Kanji padding.'}
    dump_json(root / 'vocab.json', vocab)
    dump_json(root / 'character_frequency.json', dict(sorted(counts.items())))
    return chosen
