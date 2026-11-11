"""Public JMdict culinary subset, curated menu grammar, indexed lookup and prices."""
from src.common.paths import SYNTHETIC_ROOT, asset, dataset_root
import argparse
from collections import defaultdict
import gzip
import itertools
import json
import re
import sqlite3
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil

from .common import ROOT, digest, dump_json

DIGITS = dict(zip('〇零一二三四五六七八九', [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
SMALL = {'十': 10, '百': 100, '千': 1000}


def numeral(text):
    text = unicodedata.normalize('NFKC', text).replace(',', '').replace('、', '')
    if not text or any(c not in '0123456789' + ''.join(DIGITS) + '十百千万' for c in text):
        raise ValueError('Invalid price numeral')
    if not any(c in text for c in '十百千万'):
        return int(''.join(str(DIGITS[c]) if c in DIGITS else c for c in text))
    total = section = 0
    pending = ''
    for c in text:
        if c in SMALL:
            section += (int(pending) if pending else 1) * SMALL[c]
            pending = ''
        elif c == '万':
            total += (section + (int(pending) if pending else 0) or 1) * 10000
            section, pending = 0, ''
        else:
            pending += str(DIGITS[c]) if c in DIGITS else c
    return total + section + (int(pending) if pending else 0)


def parse_price(text):
    text = unicodedata.normalize('NFKC', text)
    result = dict(price_yen=None, is_market_price='時価' in text,
                  tax_included=None, unit_counter=None, is_minimum_price=False)
    taxes = re.findall(r'税込|内税|税抜|外税|\+税', text)
    flags = {t in ('税込', '内税') for t in taxes}
    if len(flags) == 1:
        result['tax_included'] = flags.pop()
    text = re.sub(r'税込|内税|税抜|外税|\+税', '', text)
    counters = re.findall(r'[〇零一二三四五六七八九十百千0-9]+(?:人前|グラム|串|貫|本|皿|枚|杯|合|g)', text)
    if counters:
        result['unit_counter'] = ' '.join(counters)
        for counter in counters:
            text = text.replace(counter, '')
    result['is_minimum_price'] = bool(re.search(r'[〜~～]|から', text))
    if result['is_market_price']:
        return result
    # Prefer amounts explicitly adjacent to a currency marker; first range end is minimum.
    number = r'[0-9〇零一二三四五六七八九十百千万][0-9〇零一二三四五六七八九十百千万,、]*'
    minimum = re.search('(' + number + r')\s*(?:円)?\s*(?:[〜~～]|から)', text)
    if minimum:
        result['price_yen'] = numeral(minimum.group(1))
        return result
    matches = re.findall(r'(?:[¥￥]\s*(' + number + r')|(' + number + r')\s*円)', text)
    if matches:
        result['price_yen'] = numeral(next(a or b for a, b in matches))
    else:
        cleaned = re.sub(r'[\s()〜~～]|から', '', text)
        if re.fullmatch(number, cleaned):
            result['price_yen'] = numeral(cleaned)
    return result


def normalize(text):
    return ''.join(unicodedata.normalize('NFKC', text).split())


def bigrams(text):
    return set(text[i:i + 2] for i in range(len(text) - 1)) or {text}


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            row.append(min(row[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = row
    return prev[-1]


def one_edit_distance(a, b):
    """Exact distance for <=1, with 2 as the bounded rejection sentinel."""
    if abs(len(a) - len(b)) > 1:
        return 2
    if len(a) == len(b):
        mismatches = 0
        for x, y in zip(a, b):
            mismatches += x != y
            if mismatches > 1:
                return 2
        return mismatches
    if len(a) > len(b):
        a, b = b, a
    i = j = 0
    skipped = False
    while i < len(a) and j < len(b):
        if a[i] != b[j]:
            if skipped:
                return 2
            skipped = True
            j += 1
        else:
            i += 1
            j += 1
    return 1


def romaji(reading):
    hira = ''.join(chr(ord(c) - 0x60) if 'ァ' <= c <= 'ヶ' else c for c in reading)
    groups = [('あいうえお', 'a i u e o'), ('かきくけこ', 'ka ki ku ke ko'),
              ('さしすせそ', 'sa shi su se so'), ('たちつてと', 'ta chi tsu te to'),
              ('なにぬねの', 'na ni nu ne no'), ('はひふへほ', 'ha hi fu he ho'),
              ('まみむめも', 'ma mi mu me mo'), ('やゆよ', 'ya yu yo'),
              ('らりるれろ', 'ra ri ru re ro'), ('わをん', 'wa o n'),
              ('がぎぐげご', 'ga gi gu ge go'), ('ざじずぜぞ', 'za ji zu ze zo'),
              ('だぢづでど', 'da ji zu de do'), ('ばびぶべぼ', 'ba bi bu be bo'),
              ('ぱぴぷぺぽ', 'pa pi pu pe po'), ('ぁぃぅぇぉゔ', 'a i u e o vu')]
    table = {c: r for chars, rs in groups for c, r in zip(chars, rs.split())}
    out = ''
    geminate = False
    for c in hira:
        if c == 'っ':
            geminate = True
            continue
        if c in 'ゃゅょ' and out:
            out = out[:-1] + ('' if out.endswith(('shi', 'chi', 'ji')) else 'y') + {'ゃ': 'a', 'ゅ': 'u', 'ょ': 'o'}[c]
            continue
        r = table.get(c, c)
        if c == 'ー' and out:
            r = out[-1]
        if geminate:
            r = r[0] + r
            geminate = False
        out += r
    return out


# Curated components: each has Japanese, reading and English; no numeric padding.
INGREDIENTS = '''鶏肉|とりにく|chicken;豚肉|ぶたにく|pork;牛肉|ぎゅうにく|beef;鴨肉|かもにく|duck;羊肉|ひつじにく|lamb;鹿肉|しかにく|venison;馬肉|ばにく|horse meat;鶏手羽|とりてば|chicken wings;鶏もも|とりもも|chicken thigh;砂肝|すなぎも|chicken gizzard;鶏皮|とりかわ|chicken skin;豚バラ|ぶたばら|pork belly;牛タン|ぎゅうたん|beef tongue;牛すじ|ぎゅうすじ|beef tendon;鮭|さけ|salmon;鯖|さば|mackerel;鯛|たい|sea bream;鰤|ぶり|yellowtail;鰯|いわし|sardine;鯵|あじ|horse mackerel;鰹|かつお|bonito;鱈|たら|cod;鰆|さわら|Spanish mackerel;鰈|かれい|flounder;秋刀魚|さんま|Pacific saury;鮎|あゆ|sweetfish;鰻|うなぎ|eel;穴子|あなご|conger eel;鮪|まぐろ|tuna;海老|えび|shrimp;蟹|かに|crab;帆立|ほたて|scallop;牡蠣|かき|oyster;蛤|はまぐり|clam;浅利|あさり|short-neck clam;烏賊|いか|squid;蛸|たこ|octopus;豆腐|とうふ|tofu;厚揚げ|あつあげ|fried tofu;油揚げ|あぶらあげ|tofu pouch;茄子|なす|eggplant;南瓜|かぼちゃ|pumpkin;蓮根|れんこん|lotus root;牛蒡|ごぼう|burdock;人参|にんじん|carrot;大根|だいこん|daikon radish;白菜|はくさい|Chinese cabbage;キャベツ|きゃべつ|cabbage;玉葱|たまねぎ|onion;長葱|ながねぎ|leek;椎茸|しいたけ|shiitake mushroom;舞茸|まいたけ|maitake mushroom;松茸|まつたけ|matsutake mushroom;榎茸|えのきだけ|enoki mushroom;しめじ|しめじ|shimeji mushroom;筍|たけのこ|bamboo shoot;里芋|さといも|taro;山芋|やまいも|yam;長芋|ながいも|Chinese yam;薩摩芋|さつまいも|sweet potato;じゃが芋|じゃがいも|potato;小松菜|こまつな|komatsuna greens;ほうれん草|ほうれんそう|spinach;春菊|しゅんぎく|chrysanthemum greens;菜の花|なのはな|rapeseed greens;アスパラ|あすぱら|asparagus;ピーマン|ぴーまん|green pepper;獅子唐|ししとう|shishito pepper;オクラ|おくら|okra;ズッキーニ|ずっきーに|zucchini;トマト|とまと|tomato;ブロッコリー|ぶろっこりー|broccoli;カリフラワー|かりふらわー|cauliflower;もやし|もやし|bean sprouts;絹さや|きぬさや|snow peas;空豆|そらまめ|broad beans;枝豆|えだまめ|edamame;銀杏|ぎんなん|ginkgo nuts;玉蜀黍|とうもろこし|corn;韮|にら|garlic chives;蒟蒻|こんにゃく|konjac;餅|もち|rice cake;湯葉|ゆば|tofu skin;車麩|くるまふ|wheat gluten;鮑|あわび|abalone;栄螺|さざえ|turban shell;鰊|にしん|herring;鱒|ます|trout;金目鯛|きんめだい|splendid alfonsino;鰤かま|ぶりかま|yellowtail collar;鮭はらす|さけはらす|salmon belly'''
MODIFIERS = '''特製|とくせい|house-special;自家製|じかせい|house-made;本日の|ほんじつの|today's;季節の|きせつの|seasonal;旬の|しゅんの|in-season;厳選|げんせん|selected;名物|めいぶつ|signature;おすすめ|おすすめ|recommended;田舎風|いなかふう|rustic;和風|わふう|Japanese-style;京風|きょうふう|Kyoto-style;家庭風|かていふう|home-style;料理長の|りょうりちょうの|chef's;香味|こうみ|aromatic;柚子香る|ゆずかおる|yuzu-scented;生姜香る|しょうがかおる|ginger-scented;山椒香る|さんしょうかおる|sansho-scented;胡麻香る|ごまかおる|sesame-scented;葱香る|ねぎかおる|leek-scented;大盛り|おおもり|large portion;小盛り|こもり|small portion;一人前の|いちにんまえの|single-serving;二人前の|ににんまえの|two-serving;おまかせ|おまかせ|chef-selected'''
METHODS = '''塩焼き|しおやき|salt-grilled;照り焼き|てりやき|teriyaki;味噌焼き|みそやき|miso-grilled;炭火焼き|すみびやき|charcoal-grilled;醤油焼き|しょうゆやき|soy-grilled;蒸し|むし|steamed;酒蒸し|さかむし|sake-steamed;煮物|にもの|simmered;甘辛煮|あまからに|sweet-soy simmered;唐揚げ|からあげ|deep-fried;天ぷら|てんぷら|tempura;素揚げ|すあげ|plain-fried;炒め|いため|stir-fried;味噌煮|みそに|miso-simmered'''


def components(text):
    return [tuple(part.split('|')) for part in text.split(';')]


def curated_entries():
    for jp, reading, en in components(INGREDIENTS):
        for mj, mr, me in components(MODIFIERS):
            for cj, cr, ce in components(METHODS):
                yield dict(canonical_jp=mj + jp + cj, reading_hiragana=mr + reading + cr,
                           romaji=romaji(mr + reading + cr), name_en=f'{me} {ce} {en}',
                           description_en=f'{me} {ce} {en}. Procedural menu variant; recipe and allergens unverified.',
                           category='curated_grammar', allergens='["unknown"]', dietary_flags='["unknown"]',
                           source='curated_components_v1')


def jmdict_entries(path):
    with gzip.open(path, 'rb') as f:
        # ElementTree reads internal DTD entity declarations but makes no network requests.
        for _, entry in ET.iterparse(f, events=('end',)):
            if entry.tag != 'entry':
                continue
            culinary_words = re.compile(r'\b(food|dish|cooking|cuisine|edible|ingredient|restaurant|noodle|rice|soup|sushi|sashimi|tea|sake|wine|beer|fish|seaweed|mushroom|vegetable|fruit|bean|flour|sauce|seasoning|spice|pickled|fermented|grilled|steamed|fried|boiled|salmon|tuna|mackerel|clam|oyster|shrimp|prawn|crab|scallop|squid|octopus|abalone|cabbage|radish|eggplant|chicken|pork|beef|tofu|bread|pastry|confectionery|sugar|salt|flavor|flavour|taste|sweet|sour|bitter|savory|umami|aroma|fragrance|liquor|alcohol|vinegar|oil|fat|starch|wheat|barley|oat|buckwheat|millet|rye|maize|cereal|grain|soy|miso|kelp|nori|herb|mint|ginger|garlic|pepper|sesame|bamboo|lotus|chestnut|nut|persimmon|pear|peach|plum|apple|apricot|cherry|citrus|yuzu|orange|lemon|lime|grape|strawberry|melon|watermelon|pumpkin|squash|cucumber|tomato|carrot|potato|turnip|onion|leek|spinach|lettuce|mustard|taro|yam|burdock|cinnamon|clove|nutmeg|basil|parsley|dill|coriander|cumin|fennel|saffron|turmeric|wasabi|chopstick|spoon|fork|plate|bowl|tray|kettle|ladle|skewer|kitchen|cook|chef|diner|tavern|izakaya|yakitori|lunch|dinner|breakfast|meal|dessert|meat|poultry|duck|quail|goose|pheasant|boar|deer|roe|caviar|eel|herring|trout|flounder|bream|mussel|cockle|whelk)\b', re.I)
            culinary = [s for s in entry.findall('sense') if any('food' in (n.text or '').lower() or 'cooking' in (n.text or '').lower() for n in s.findall('field')) or any(culinary_words.search(g.text or '') for g in s.findall('gloss'))]
            if culinary:
                reading = entry.findtext('r_ele/reb', '')
                gloss = '; '.join(g.text or '' for s in culinary for g in s.findall('gloss') if g.get('{http://www.w3.org/XML/1998/namespace}lang', 'eng') == 'eng')
                for jp in [n.text for n in entry.findall('k_ele/keb')] or [reading]:
                    yield dict(canonical_jp=jp, reading_hiragana=reading, romaji=romaji(reading),
                               name_en=gloss, description_en=gloss, category='jmdict_food',
                               allergens='["unknown"]', dietary_flags='["unknown"]', source='JMdict:' + entry.findtext('ent_seq'))
            entry.clear()


def build(data_root=SYNTHETIC_ROOT):
    data_root = dataset_root(data_root)
    path = asset(data_root, 'lexicon/culinary_menu.db')
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript('''DROP TABLE IF EXISTS char_bigrams; DROP TABLE IF EXISTS menu_dictionary;
    CREATE TABLE menu_dictionary(id INTEGER PRIMARY KEY, canonical_jp TEXT UNIQUE NOT NULL,
    reading_hiragana TEXT NOT NULL, romaji TEXT NOT NULL, name_en TEXT NOT NULL,
    description_en TEXT NOT NULL, category TEXT NOT NULL, allergens TEXT NOT NULL,
    dietary_flags TEXT NOT NULL, source TEXT NOT NULL);
    CREATE TABLE char_bigrams(bigram TEXT NOT NULL, menu_id INTEGER NOT NULL REFERENCES menu_dictionary(id),
    PRIMARY KEY(bigram,menu_id)) WITHOUT ROWID;
    CREATE INDEX dictionary_length ON menu_dictionary(length(canonical_jp));''')
    records = {}
    for row in curated_entries():
        records[normalize(row['canonical_jp'])] = row
    source_count = 0
    for row in jmdict_entries(asset(data_root, 'raw/JMdict_e.gz')):
        records[normalize(row['canonical_jp'])] = row
        source_count += 1
    columns = 'canonical_jp reading_hiragana romaji name_en description_en category allergens dietary_flags source'.split()
    for i, (jp, row) in enumerate(sorted(records.items()), 1):
        row['canonical_jp'] = jp
        db.execute('INSERT INTO menu_dictionary VALUES (?,?,?,?,?,?,?,?,?,?)', [i] + [row[k] for k in columns])
        db.executemany('INSERT INTO char_bigrams VALUES (?,?)', [(b, i) for b in bigrams(jp)])
    db.commit()
    db.close()
    shutil.copyfile(asset(data_root, 'raw/JMdict-LICENSE.html'), asset(data_root, 'lexicon/JMdict-LICENSE.html'))
    dump_json(asset(data_root, 'lexicon/provenance.json'), dict(total_entries=len(records), jmdict_forms=source_count,
        generated_entries=sum(r['category'] == 'curated_grammar' for r in records.values()),
        source_sha256=digest(asset(data_root, 'raw/JMdict_e.gz')), license='CC-BY-SA-4.0',
        note='JMdict candidate glossary includes food-tagged senses and culinary keyword matches (ingredients, preparation, flavors, service and utensils); review polysemous terms before deployment. Generated variants are component combinations, not independently attested dishes. Allergen and dietary values are unknown.'))
    print('Built lexicon:', len(records), 'entries;', source_count, 'JMdict food forms')


class CulinaryLookup:
    def __init__(self, path=ROOT / 'data/processed/lexicon/culinary_menu.db'):
        from src.common.paths import resolve_path
        self.db = sqlite3.connect(resolve_path(path))
        self.db.row_factory = sqlite3.Row
        self.words = {}
        self.index = defaultdict(set)
        self.lengths = defaultdict(set)
        for row in self.db.execute('SELECT id,canonical_jp FROM menu_dictionary'):
            self.words[row['id']] = row['canonical_jp']
            self.lengths[len(row['canonical_jp'])].add(row['id'])
        self.exact = {word: i for i, word in self.words.items()}
        for row in self.db.execute('SELECT bigram,menu_id FROM char_bigrams'):
            self.index[row['bigram']].add(row['menu_id'])

    def search(self, query, max_distance=2, limit=5):
        query = normalize(query)
        if not query:
            return []
        if query in self.exact:
            return [self._row(self.exact[query], 0)]
        grams = bigrams(query)
        # q-gram lower bound. Short strings have no guaranteed surviving bigram.
        min_shared = max(0, len(grams) - 2 * max_distance)
        if min_shared == 0:
            ids = set().union(*(self.lengths.get(n, set()) for n in range(max(1, len(query) - max_distance), len(query) + max_distance + 1)))
        else:
            # Intersect posting lists in C rather than counting thousands of
            # common culinary bigrams in Python. This retains the q-gram bound.
            postings = sorted((self.index.get(g, set()) for g in grams), key=len)
            ids = set()
            for group in itertools.combinations(postings, min_shared):
                ids.update(set.intersection(*group))
        ids = {i for i in ids if abs(len(self.words[i]) - len(query)) <= max_distance}
        # Known OCR confusions break otherwise indistinguishable ingredient ties;
        # edit distance still controls acceptance, so the original query is retained.
        confusion = str.maketrans({'鳥': '鶏', '烏': '鶏', '内': '肉'})
        folded = query.translate(confusion)
        ranked = []
        for i in ids:
            distance = one_edit_distance(query, self.words[i]) if max_distance == 1 else levenshtein(query, self.words[i])
            if distance <= max_distance:
                shared = len(grams & bigrams(self.words[i]))
                ranked.append((distance, levenshtein(folded, self.words[i].translate(confusion)), -shared, i))
        ranked.sort()
        return [self._row(i, distance) for distance, _, _, i in ranked[:limit]]

    def _row(self, i, distance):
        row = dict(self.db.execute('SELECT * FROM menu_dictionary WHERE id=?', (i,)).fetchone())
        row['distance'] = distance
        return row

    def close(self):
        self.db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=SYNTHETIC_ROOT)
    build(parser.parse_args().data_root)
