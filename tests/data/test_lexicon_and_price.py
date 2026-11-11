from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from src.common import ROOT
from src.data.genai_seed_collector import SeedCollector, read_ledger
from src.data.lexicon_builder import CulinaryLookup, levenshtein, numeral, parse_price


@pytest.mark.parametrize('text,price,tax,counter,minimum,market', [
    ('八五〇円（税込）',850,True,None,False,False),
    ('一串 二百五十円',250,None,'一串',False,False),
    ('時価',None,None,None,False,True),
    ('二貫 千二百五十円（税抜）',1250,False,'二貫',False,False),
    ('一、二八〇円 内税',1280,True,None,False,False),
    ('二皿 １，２００円＋税',1200,False,'二皿',False,False),
    ('850〜1200円',850,None,None,True,False),
    ('八百円から',800,None,None,True,False),
    ('一人前 三千円 外税',3000,False,'一人前',False,False),
    ('100g 九百八十円',980,None,'100g',False,False),
    ('時価 税込 一皿',None,True,'一皿',False,True),
    ('海鮮丼',None,None,None,False,False),
])
def test_price(text, price, tax, counter, minimum, market):
    assert parse_price(text) == dict(price_yen=price,tax_included=tax,unit_counter=counter,is_minimum_price=minimum,is_market_price=market)


@pytest.mark.parametrize('text,expected', [('一万二千三百四十五',12345),('十',10),('百十',110),('八五〇',850),('零',0),('１２５０',1250),('二万五百',20500)])
def test_numeral(text, expected):
    assert numeral(text) == expected


def test_exact_and_two_ocr_errors():
    lookup = CulinaryLookup()
    try:
        assert lookup.search('特製鶏肉塩焼き')[0]['distance'] == 0
        assert any(r['canonical_jp'] == '特製鶏肉塩焼き' for r in lookup.search('特製鳥肉塩焼き'))
        assert any(r['canonical_jp'] == '特製鶏肉塩焼き' for r in lookup.search('特製鳥内塩焼き'))
        assert lookup.search('') == []
        assert levenshtein('焼き鳥', '焼鳥') == 1
    finally:
        lookup.close()


def test_cap_survives_restart_and_failed_calls(tmp_path):
    path = tmp_path / 'usage_ledger.json'
    collector = SeedCollector(path)
    for category, count in [('items',40),('textures',40),('reserve',20)]:
        for _ in range(count):
            assert collector.call(category, lambda _: 'ok', 'prompt') == 'ok'
    assert read_ledger(path)['total_calls'] == 100
    with pytest.raises(RuntimeError, match='budget'):
        SeedCollector(path).call('items', lambda _: pytest.fail('Called provider over budget'), '')
    failed = SeedCollector(tmp_path / 'failed.json')
    with pytest.raises(RuntimeError, match='remains charged'):
        failed.call('items', lambda _: 1/0, '')
    assert read_ledger(failed.path)['total_calls'] == 1


def test_atomic_reservations(tmp_path):
    collector = SeedCollector(tmp_path / 'ledger.json')
    def attempt(_):
        try:
            collector.call('items', lambda _: None, '')
            return 1
        except RuntimeError:
            return 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(48))) == 40
    assert read_ledger(collector.path)['total_calls'] == 40


def test_corrupt_ledger_fails_closed(tmp_path):
    path = tmp_path / 'ledger.json'
    path.write_text('{"total_calls":0,"events":[{"id":1,"category":"items"}]}')
    with pytest.raises(ValueError):
        SeedCollector(path)
