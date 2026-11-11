"""Download data assets from publisher-owned HTTPS URLs; never execute downloads."""
from src.common.paths import SYNTHETIC_ROOT, asset, dataset_root
import argparse
import io
import json
import zipfile
from pathlib import Path

import requests
from fontTools.ttLib import TTFont

from src.common import ROOT, digest, dump_json

GOOGLE = 'https://raw.githubusercontent.com/google/fonts/main/ofl/'
FONTS = {
    'KouzanBrushFont.ttf': ('https://opentype.jp/bin/KouzanMouhituFont.zip', 'https://opentype.jp/kouzanmouhitufont.htm'),
    'AoyagiSouseki.ttf': ('https://opentype.jp/bin/AoyagiSosekiFont.zip', 'https://opentype.jp/aoyagisosekifont.htm'),
    'YujiBoku-Regular.ttf': (GOOGLE + 'yujiboku/YujiBoku-Regular.ttf', GOOGLE + 'yujiboku/OFL.txt'),
    'YujiMai-Regular.ttf': (GOOGLE + 'yujimai/YujiMai-Regular.ttf', GOOGLE + 'yujimai/OFL.txt'),
    'ShipporiMincho-Bold.ttf': (GOOGLE + 'shipporimincho/ShipporiMincho-Bold.ttf', GOOGLE + 'shipporimincho/OFL.txt'),
    'HachiMaruPop-Regular.ttf': (GOOGLE + 'hachimarupop/HachiMaruPop-Regular.ttf', GOOGLE + 'hachimarupop/OFL.txt'),
    'KleeOne-SemiBold.ttf': (GOOGLE + 'kleeone/KleeOne-SemiBold.ttf', GOOGLE + 'kleeone/OFL.txt'),
}


def fetch(url):
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def download(root=SYNTHETIC_ROOT, jmdict=False):
    root = dataset_root(root)
    fontdir = asset(root, 'fonts')
    fontdir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (url, license_url) in FONTS.items():
        p = fontdir / name
        if not p.exists():
            blob = fetch(url)
            if url.endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(blob)) as archive:
                    candidates = [n for n in archive.namelist() if n.lower().endswith('.ttf')]
                    if not candidates:
                        raise ValueError('Publisher archive contains no TTF')
                    p.write_bytes(archive.read(candidates[0]))
                    for i, n in enumerate(archive.namelist()):
                        if n.lower().endswith('.txt'):
                            (fontdir / f'{name}.notice-{i}.txt').write_bytes(archive.read(n))
            else:
                p.write_bytes(blob)
        with TTFont(p) as font:
            cmap = font.getBestCmap()
            if not all(ord(c) in cmap for c in '日本料理'):
                raise ValueError(f'Invalid Japanese font: {name}')
            family = font['name'].getDebugName(1)
        license_path = fontdir / (name + '.license.txt')
        if not license_path.exists():
            license_path.write_bytes(fetch(license_url))
        manifest.append(dict(file=name, family=family, sha256=digest(p), source=url,
                             license_url=license_url, license_sha256=digest(license_path)))
        print('Verified font:', name, flush=True)
    dump_json(fontdir / 'manifest.json', manifest)
    if jmdict:
        raw = asset(root, 'raw')
        raw.mkdir(parents=True, exist_ok=True)
        source = 'https://www.edrdg.org/pub/Nihongo/JMdict_e.gz'
        p = raw / 'JMdict_e.gz'
        if not p.exists():
            p.write_bytes(fetch(source))
        license_url = 'https://www.edrdg.org/edrdg/licence.html'
        (raw / 'JMdict-LICENSE.html').write_bytes(fetch(license_url))
        dump_json(raw / 'jmdict_source.json', dict(source=source, sha256=digest(p),
                  license='CC-BY-SA-4.0', copyright='JM Breen and EDRDG', license_url=license_url))
        print('Verified JMdict:', p.stat().st_size, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=SYNTHETIC_ROOT)
    parser.add_argument('--jmdict', action='store_true')
    args = parser.parse_args()
    download(args.data_root, args.jmdict)
