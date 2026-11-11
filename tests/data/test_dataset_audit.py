import pytest

from scripts.data.verify_stage1_dataset import require
from src.common import ROOT
from src.data.menu_grammar import generate


def test_audit_fails_closed():
    with pytest.raises(ValueError,match='bad data'):
        require(False,'bad data')


def test_stage1_data_code_has_no_training_dependencies():
    # Stage 1 data preparation stays model-free. Later stages may add training notebooks.
    for path in (ROOT/'src/data').rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        assert 'import torch' not in text
        assert 'nn.Module' not in text
