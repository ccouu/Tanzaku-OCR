from src.recognition.label_normalization import normalize_digits


def test_only_fullwidth_decimal_digits_are_normalized():
    assert normalize_digits('１２３４５６７８９０円') == '1234567890円'
    assert normalize_digits('￥４５０　円・①／Ａ') == '￥450　円・①／Ａ'
    assert normalize_digits('¥450円') == '¥450円'


def test_normalization_is_idempotent():
    text='価格１２３円 00'
    assert normalize_digits(normalize_digits(text)) == normalize_digits(text)
