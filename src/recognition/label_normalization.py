"""Narrow label canonicalization; preserve Japanese text and punctuation."""
_DIGITS = str.maketrans('０１２３４５６７８９', '0123456789')


def normalize_digits(text):
    """Map only full-width decimal digits to ASCII, without broad NFKC changes."""
    return text.translate(_DIGITS)
