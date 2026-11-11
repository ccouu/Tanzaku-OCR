"""Reserve and finish built-in imagegen requests in the shared Stage 1 ledger."""
import argparse
import json
from pathlib import Path

from PIL import Image

from src.common import ROOT, digest, dump_json
from src.data.genai_seed_collector import ledger_lock, read_ledger, SeedCollector, CAP

BASE = ROOT / 'data/raw/genai_seeds'
LEDGER = BASE / 'usage_ledger.json'
STYLES = ['izakaya brush calligraphy on warm fibrous washi', 'sushi counter handwritten vertical menu on cedar boards',
          'ramen shop casual handwritten blackboard in white chalk', 'yakitori shop energetic sumi brush on cream paper',
          'seasonal kaiseki elegant cursive ink on textured ivory paper', 'rural soba restaurant handwritten menu on aged paper',
          'tempura restaurant bold handwritten vertical menu strips', 'seafood tavern daily specials with blue and black marker',
          'Japanese cafe handwritten lunch menu with rounded pen lettering', 'okonomiyaki shop casual brush menu on kraft paper']
ITEMS = [
    '本日のおすすめ、刺身盛合せ 八五〇円、焼き鳥 一串二百五十円、枝豆 三百円、生ビール 五百円',
    '寿司、まぐろ 二貫六百円、サーモン 二貫五百円、玉子 二貫三百円、穴子 二貫七百円',
    'らーめん、醤油ラーメン 八百円、味噌ラーメン 九百円、塩ラーメン 八五〇円、餃子 四百円',
    '炭火焼、ねぎま 一串二百円、つくね 一串二百五十円、砂肝 一串百八十円、鶏皮 一串百五十円',
    '季節の献立、松茸土瓶蒸し 千二百円、秋刀魚塩焼き 八百円、栗ご飯 六百円、茶碗蒸し 五百円',
    'おしながき、天ざるそば 千二百円、かけそば 七百円、山菜そば 八百円、とろろそば 九百円',
    '揚げたて、海老天丼 千二百五十円、野菜天ぷら 八百円、かき揚げ 六百円、天ぷら定食 千五百円',
    '本日の鮮魚、鯛の煮付け 千二百円、鯖塩焼き 七百円、牡蠣フライ 九百円、旬の刺身 時価',
    'ランチ、唐揚げ定食 八百円、焼き魚定食 九百円、カレーライス 七百円、コーヒー 三百円',
    '鉄板焼、お好み焼き 八百円、豚玉 九百円、海鮮焼き 千二百円、焼きそば 七百円']


def reserve(count):
    SeedCollector(LEDGER)
    jobs = []
    with ledger_lock(LEDGER):
        ledger = read_ledger(LEDGER)
        if count < 1 or ledger['total_calls'] + count > CAP:
            raise ValueError('Reservation would exceed the lifetime 100-call cap')
        for _ in range(count):
            n = ledger['total_calls'] + 1
            i = (n - 1) % 10
            variation = (n - 1) // 10
            prompt = (f'Use case: photorealistic-natural. Asset: synthetic Japanese handwritten restaurant menu for OCR data collection. '
                      f'Create one distinct full menu page, not a collage: {STYLES[i]}. '
                      f'Variation {variation+1}: ' + ['mostly vertical top-to-bottom columns, read right-to-left', 'mostly horizontal handwritten rows',
                      'mixed upright vertical columns and horizontal prices', 'slightly uneven handwritten columns', 'handwritten daily specials with a red heading',
                      'faded ink and light paper stains, still legible', 'confident thick and thin brush strokes', 'fine pen strokes and compact spacing',
                      'warm restaurant lighting and slight camera perspective', 'neutral daylight, visible paper fibers and natural handwriting'][variation] +
                      f'. Japanese menu text to render faithfully: {ITEMS[i]}. Include 円 prices, appropriate vertical punctuation and long vowel marks. '
                      'Portrait composition, the whole menu visible with margins, readable authentic Japanese handwriting, varied natural strokes. '
                      'No English translations, no romanization, no watermark, no people, no collage, no decorative fake characters. '
                      'Keep all text on the page; no clipped edges. High resolution.')
            relative = f'menu_images/menu-{n:03d}.png'
            metadata = {'id': n, 'category': 'menus', 'status': 'reserved', 'provider': 'built_in_imagegen',
                        'image': relative, 'prompt': prompt, 'reviewed': False, 'source': 'genai_synthetic',
                        'intended_text': ITEMS[i], 'transcription_verified': False}
            jobs.append(metadata)
            dump_json(BASE / f'menu_images/menu-{n:03d}.json', metadata)
            ledger['events'].append({k: metadata[k] for k in ('id','category','status','provider','image')})
            ledger['total_calls'] += 1
        dump_json(LEDGER, ledger)
    print(json.dumps(jobs, ensure_ascii=False))


def finish(n, success):
    p = BASE / f'menu_images/menu-{n:03d}.json'
    metadata = json.loads(p.read_text(encoding='utf-8'))
    if success:
        image_path = BASE / metadata['image']
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            metadata['size'] = list(image.size)
        metadata['sha256'] = digest(image_path)
    metadata['status'] = 'completed' if success else 'failed'
    dump_json(p, metadata)
    with ledger_lock(LEDGER):
        ledger = read_ledger(LEDGER)
        event = ledger['events'][n - 1]
        if event['status'] != 'reserved':
            raise ValueError('Request has already been finalized')
        event['status'] = metadata['status']
        if success:
            event['sha256'] = metadata['sha256']
        dump_json(LEDGER, ledger)
    print(json.dumps({'id': n, 'status': metadata['status']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reserve', type=int)
    parser.add_argument('--finish', type=int)
    parser.add_argument('--failed', action='store_true')
    args = parser.parse_args()
    if args.reserve is not None:
        reserve(args.reserve)
    elif args.finish is not None:
        finish(args.finish, not args.failed)
