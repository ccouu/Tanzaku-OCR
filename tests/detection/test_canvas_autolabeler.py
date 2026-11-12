import json
from pathlib import Path

from PIL import Image
import pytest

from scripts.yolo_crop_canvas_pipeline import (
    DEFAULT_WEIGHTS, build_vlm_prompt, pack_canvases, parse_vlm_transcriptions, run_build,
)


def item(root, ident, vertical=False, review=False):
    display=f'{ident}.png';ctc=f'{ident}_ctc.png'
    Image.new('RGB',(20,100) if vertical else (100,20),'white').save(root/display)
    Image.new('RGB',(384,64),'white').save(root/ctc)
    return dict(crop_id=ident,source_menu='menu-001',source_image='menu-001.jpg',
        detector_class='dish_vertical' if vertical else 'dish_horizontal',confidence=.9,
        is_vertical=vertical,orientation_flag=int(vertical),display_crop=display,ctc_crop=ctc,
        obb_polygon=[[0,0],[100,0],[100,20],[0,20]],padded_polygon=[[-2,-2],[102,-2],[102,22],[-2,22]],
        review_required=review,overlap_source_ids=['other'] if review else [],
        source_ids=['original'],padding_scale=.5,ctc_sha256='fixture',valid_width=100,canvas=384,input_length=25)


def test_grid_prompt_and_parser_preserve_source_mapping(tmp_path):
    crops=[item(tmp_path,'horizontal',review=True),item(tmp_path,'vertical',True)]
    manifest=pack_canvases(crops,tmp_path,items_per_canvas=1)
    assert [c['layout'] for c in manifest['canvases']]==['horizontal_rows','vertical_columns']
    prompt=(tmp_path/manifest['canvases'][0]['prompt_file']).read_text(encoding='utf-8')
    assert '[01]' in prompt and 'UNCERTAIN' in prompt and 'overlap review flags' in prompt
    replies=tmp_path/'responses.txt'
    replies.write_text('canvas_001\n[01] | 魚 | UNCERTAIN\ncanvas_002\n[01] | 円 | OK\n',encoding='utf-8')
    result=parse_vlm_transcriptions(tmp_path,replies)
    assert result['exported_ctc_lines']==1
    records=[json.loads(s) for s in (tmp_path/'labeled_crops.jsonl').read_text(encoding='utf-8').splitlines()]
    assert records[0]['crop_id']=='horizontal' and records[0]['review_required']
    assert records[0]['status']=='UNCERTAIN' and records[0]['text']=='魚'
    assert records[0]['padded_polygon']==crops[0]['padded_polygon']
    assert records[1]['orientation_flag']==1
    assert (tmp_path/'ctc_labels.txt').read_text(encoding='utf-8')=='vertical_ctc.png\t円\n'


def test_existing_build_is_not_overwritten(tmp_path):
    (tmp_path/'crops_manifest.json').write_text('preserve',encoding='utf-8')
    weights=tmp_path/'weights.pt';weights.write_bytes(b'fixture')
    with pytest.raises(FileExistsError):run_build(tmp_path,tmp_path,weights=weights)
    assert (tmp_path/'crops_manifest.json').read_text()=='preserve'


def test_selected_model_and_no_hint_prompt():
    assert 'banana130_finetune_v5' in str(DEFAULT_WEIGHTS)
    prompt=build_vlm_prompt('canvas_001.png',[dict(source_hints=['secret hint'],source_image='source')],False)
    assert 'secret hint' not in prompt and 'Never complete cut-off words' in prompt
    assert 'NOT guaranteed to contain text' in prompt
    assert '[ID] | | NOISE' in prompt and 'never invent a label' in prompt


def test_invalid_canvas_size_rejected(tmp_path):
    with pytest.raises(ValueError):pack_canvases([],tmp_path,items_per_canvas=0)
