"""Keep test artifacts in the workspace with inherited Windows sandbox ACLs."""
import uuid

import pytest

from src.common import ROOT


@pytest.fixture
def tmp_path():
    path = ROOT / '.cache/test-runs' / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path
