"""Relocation must preserve source identity and custom dataset isolation."""
from pathlib import Path

import pytest

from src.common import ROOT, safe_path
from src.common.paths import SYNTHETIC_ROOT, asset, resolve_path, source_kind


@pytest.mark.parametrize('old,new', [
    ('data/stage2/manifest.json', 'data/annotations/menu_obb/manifest.json'),
    ('data/stage2_training_v2.zip', 'data/datasets/detection/reviewed_v2/dataset.zip'),
    ('data/augmentation_500/manifest.json', 'data/processed/augmentation/run_500/manifest.json'),
    ('data/fonts/KleeOne-SemiBold.ttf', 'data/resources/fonts/KleeOne-SemiBold.ttf'),
])
def test_legacy_paths_resolve_to_canonical_assets(old, new):
    assert resolve_path(ROOT / old) == ROOT / new


def test_custom_datasets_are_self_contained(tmp_path):
    assert asset(tmp_path, 'fonts') == tmp_path / 'fonts'
    assert asset(tmp_path, 'raw/genai_seeds') == tmp_path / 'raw/genai_seeds'
    assert asset(SYNTHETIC_ROOT, 'fonts') == ROOT / 'data/resources/fonts'
    assert asset(SYNTHETIC_ROOT, 'raw') == ROOT / 'data/raw'
    with pytest.raises(ValueError, match='escapes'):
        safe_path(SYNTHETIC_ROOT, '../outside')


@pytest.mark.parametrize('name,kind', [('tabelog_menus','tabelog_menu'), ('tabelog_interiors','tabelog_interior')])
def test_source_kind_survives_relocation(name, kind):
    legacy = {'source_root': 'data/raw/' + name}
    relocated = {'source_root': 'another/location', 'source_kind': kind}
    assert source_kind(legacy) == source_kind(relocated) == kind
    assert source_kind({'source_root': str(ROOT / 'data/raw' / name)}) == kind


def test_legacy_modules_keep_identity():
    import scripts.stage2_yolo.prepare_training as legacy
    import src.detection.prepare_training as current
    assert legacy is current


def test_consolidation_identity_keeps_original_logical_path():
    from src.common.paths import historical_path
    assert historical_path(ROOT/'data/processed/augmentation/run_500/menu/images/example.png') == 'data/augmentation_500/menu/images/example.png'


def test_source_ratios_use_semantic_source_kind():
    from src.detection.prepare_training import source_ratio_split
    records = [dict(id=str(i), sha256=str(i), source_info={},
                    source_root='relocated/photos', source_kind='tabelog_menu' if i < 4 else 'generated')
               for i in range(14)]
    assignments, groups = source_ratio_split(records)
    assert sum(assignments[str(i)] == 'val' for i in range(4)) == 2
    assert sum(assignments[str(i)] == 'val' for i in range(4,14)) == 1
