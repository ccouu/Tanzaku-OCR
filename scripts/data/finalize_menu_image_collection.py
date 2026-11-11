"""Package a completed 100-request imagegen collection; performs no API calls."""
from src.common.paths import SYNTHETIC_ROOT, asset, dataset_root
import json
from collections import Counter

from src.common import ROOT, digest, dump_json
from src.data.genai_seed_collector import read_ledger
from src.data.synthetic_renderer import package


def finalize(root=SYNTHETIC_ROOT):
    seeds = asset(root, 'raw/genai_seeds')
    ledger = read_ledger(seeds / 'usage_ledger.json')
    events = ledger['events']
    if ledger['total_calls'] != 100 or Counter((e['category'], e['status']) for e in events) != {('menus', 'completed'): 100}:
        raise ValueError('Exactly 100 completed menu-image calls are required')
    records = []
    for event in events:
        path = seeds / event['image']
        metadata = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        if digest(path) != event['sha256'] or metadata['sha256'] != event['sha256']:
            raise ValueError('Generated image checksum mismatch')
        records.append({key: metadata[key] for key in ('id', 'image', 'sha256', 'size', 'source', 'reviewed', 'transcription_verified')})
    if len({r['sha256'] for r in records}) != 100:
        raise ValueError('Duplicate generated menu images')
    dump_json(seeds / 'menu_images/collection_manifest.json', {
        'provider': 'built_in_imagegen', 'actual_calls': 100, 'completed': 100, 'failed': 0,
        'category_allocation': 'User-requested replacement: 100 handwritten-menu image requests.',
        'review_status': 'unreviewed', 'ocr_ground_truth': False,
        'prompt_location': 'menu-001.json through menu-100.json sidecars',
        'images': records,
    })
    manifest_path = root / 'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(requested_menu_image_calls=100, generated_menu_images=100,
                    generated_menu_images_location='raw/genai_seeds/menu_images',
                    generated_menu_transcriptions_verified=False,
                    generation_provider='built_in_imagegen')
    dump_json(manifest_path, manifest)
    licenses = root / 'DATA_LICENSES.md'
    text = licenses.read_text(encoding='utf-8').replace(
        'No GenAI requests were needed.',
        'The initial procedural build made no GenAI requests. A subsequent user-requested collection made exactly 100 built-in imagegen calls for handwritten-menu image seeds.')
    if '## Generated handwritten-menu collection' not in text:
        text += '''

## Generated handwritten-menu collection

100 images were generated with the built-in imagegen tool and saved under
raw/genai_seeds/menu_images, with original prompts, image checksums, dimensions
and synthetic provenance in JSON sidecars. These images are generated assets,
not real restaurant photographs. Intended prompt text is unverified against
the rendered pixels and is not OCR ground truth. No third-party reference
images were supplied. Review transcriptions and geometry before adding line
crops or detector labels to the supervised splits.
'''
    licenses.write_text(text, encoding='utf-8')
    package(root)
    print('Packaged 100 completed image-generation calls and 100 unique handwritten-menu PNGs.')


if __name__ == '__main__':
    finalize()
