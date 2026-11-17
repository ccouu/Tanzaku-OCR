"""Narrow label canonicalization; preserve Japanese text and punctuation."""
_DIGITS = str.maketrans('０１２３４５６７８９', '0123456789')
_PARENTHESES = str.maketrans({'（':'(', '）':')'})


def normalize_digits(text):
    """Map only full-width decimal digits to ASCII, without broad NFKC changes."""
    return text.translate(_DIGITS)


def normalize_parentheses(text):
    """Canonicalize width only; preserve bracket direction and repeated pairs."""
    return text.translate(_PARENTHESES)
