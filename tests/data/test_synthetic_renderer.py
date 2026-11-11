import json

import numpy as np
import pytest
from PIL import Image

from src.common import ROOT, safe_path
from src.data.synthetic_renderer import (FontRoster, glyph_tile, orientation_tensor,
    render_mask, render_upright_vertical, rotated_corners, VERTICAL_MAP)
from scripts.prelabel_real_menus import export_annotation
from scripts.merge_real_dataset import validate_real


@pytest.fixture(scope='module')
def roster():
    return FontRoster(ROOT / 'data/resources/fonts')


@pytest.mark.parametrize('text', ['「ラーメン」','（チャーハン）','千二百五十円','特製鶏肉塩焼き'])
def test_tategaki_is_whole_strip_rotation(roster, text):
    upright = render_upright_vertical(text, roster, 2)
    actual = render_mask(text, 1, roster, 2)
    assert upright.size == (64,384)
    assert actual.size == (384,64)
    assert np.array_equal(np.asarray(actual), np.asarray(upright.transpose(Image.Transpose.ROTATE_90)))
    # Inverse rotation restores every glyph and its top-to-bottom reading order.
    assert np.array_equal(np.asarray(actual.transpose(Image.Transpose.ROTATE_270)), np.asarray(upright))


def test_vertical_choonpu_and_ichi_have_distinct_geometry(roster):
    choonpu = np.asarray(glyph_tile('ー', roster, 0, 40, True))
    ys,xs = np.nonzero(choonpu)
    assert np.ptp(ys) > 5 * np.ptp(xs)
    ichi = np.asarray(glyph_tile('一', roster, 0, 40, True))
    ys,xs = np.nonzero(ichi)
    assert np.ptp(xs) > np.ptp(ys)
    assert VERTICAL_MAP['「'] == '﹁' and VERTICAL_MAP['）'] == '︶'


def test_orientation_plane_and_invalid_inputs(roster):
    image = Image.new('L',(384,64),127)
    for flag in (0,1):
        tensor = orientation_tensor(image,flag)
        assert tensor.shape == (2,64,384)
        assert np.all(tensor[1] == flag)
    with pytest.raises(ValueError):
        orientation_tensor(Image.new('L',(64,384)),1)
    with pytest.raises(ValueError):
        render_mask('あ',0,roster)


def test_rotated_obb_area():
    for angle in (0,.2,-.4):
        points = np.array(rotated_corners(320,320,200,40,angle))
        area = abs(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1)))/2
        assert area == pytest.approx(8000)
        assert np.all((points>=0)&(points<=640))


def test_x_anylabeling_output_is_unreviewed(tmp_path):
    path = tmp_path/'menu.png'
    Image.new('L',(640,640)).save(path)
    output = tmp_path/'menu.json'
    export_annotation(path, [[[10,10],[50,10],[50,300],[10,300]]], output)
    data = json.loads(output.read_text())
    assert data['shapes'][0]['flags'] == {'reviewed':False,'vertical':True}
    assert data['shapes'][0]['label'] == ''
    assert data['imageWidth'] == 640


def test_merge_rejects_incomplete_and_unsafe_paths(tmp_path):
    with pytest.raises(ValueError,match='Exactly 500'):
        validate_real([],tmp_path,[],set())
    with pytest.raises(ValueError,match='escapes'):
        safe_path(tmp_path,'../secret')
