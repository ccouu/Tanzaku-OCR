import copy
from pathlib import Path

from PIL import Image
import pytest

from scripts.stage2_yolo.common import annotation, read_json, save_pair, validate_shapes, yolo_text
from scripts.stage2_yolo.consolidate_images import consolidate
from scripts.stage2_yolo.obb_review_ui import ReviewStore, make_server
from scripts.stage2_yolo.predict_initial_boxes import class_hint


def setup_collection(tmp_path):
    source = tmp_path / 'raw'
    source.mkdir()
    Image.new('RGB', (100, 80), 'white').save(source / 'one.png')
    (source / 'one.json').write_text('{"restaurant_id":"123", "text":null}')
    root = tmp_path / 'stage2'
    manifest = consolidate(root, [source])
    record = manifest['records'][0]
    shapes = [dict(label='dish_horizontal', points=[[10,10],[60,10],[60,30],[10,30]], flags={})]
    save_pair(root / 'predictions', record, annotation(record, shapes, {'backend': 'test'}))
    return root, record


def test_consolidation_traceability_and_resume(tmp_path):
    root, record = setup_collection(tmp_path)
    assert record['source_info']['restaurant_id'] == '123'
    assert Path(record['source_path']).exists()
    assert (root / 'images' / record['image']).exists()
    assert read_json(root / 'provenance' / (record['id']+'.json')) == record
    assert consolidate(root, [tmp_path / 'raw'])['records'][0] == record
    (root / 'images' / record['image']).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='overwrite'):
        consolidate(root, [tmp_path / 'raw'])


def test_yolo_and_class_hints():
    points = [[10,10],[60,10],[60,30],[10,30]]
    text = yolo_text([dict(label='price_tag', points=points)], 100, 80)
    assert text.split()[0] == '2'
    assert float(text.split()[2]) == .125
    assert class_hint([[1,1],[5,1],[5,40],[1,40]],100,100)[0] == 'dish_vertical'
    assert class_hint([[86,10],[98,10],[98,20],[86,20]],100,100)[0] == 'price_tag'


@pytest.mark.parametrize('points', [
    [[0,0],[40,40],[40,0],[0,40]],
    [[0,0],[0,0],[1,1],[1,1]],
    [[-1,0],[40,0],[40,10],[0,10]],
    [[0,0],[float('nan'),0],[40,10],[0,10]],
])
def test_invalid_geometry(points):
    with pytest.raises(ValueError):
        validate_shapes([dict(label='dish_horizontal',points=points)],100,80)


def test_review_save_is_explicit_and_preserves_provenance(tmp_path):
    root, record = setup_collection(tmp_path)
    store = ReviewStore(root)
    current = store.load(record['id'])
    assert not (root/'labels').exists()
    saved = store.save(record['id'], dict(revision=current['revision'], shapes=current['annotation']['shapes']))
    assert saved['annotation']['flags']['reviewed']
    assert saved['annotation']['provenance'] == {'backend':'test'}
    assert not read_json(root/'predictions'/(record['id']+'.json'))['flags']['reviewed']
    with pytest.raises(ValueError, match='changed elsewhere'):
        store.save(record['id'], dict(revision=current['revision'], shapes=[]))
    with pytest.raises(ValueError, match='confirmation'):
        store.save(record['id'], dict(revision=saved['revision'], shapes=[]))
    store.save(record['id'], dict(revision=saved['revision'], shapes=[],confirm_empty=True))
    assert (root/'labels'/(record['id']+'.txt')).read_text() == ''


def test_server_routes_and_csrf(tmp_path):
    import json
    import threading
    from urllib.request import urlopen, Request
    from urllib.error import HTTPError
    root, record = setup_collection(tmp_path)
    server = make_server(root, 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f'http://127.0.0.1:{server.server_port}'
    try:
        html = urlopen(url).read().decode()
        assert '__TOKEN__' not in html
        assert json.load(urlopen(url+'/api/images'))[0]['id'] == record['id']
        with pytest.raises(HTTPError) as error:
            urlopen(Request(url+'/api/annotation/'+record['id'], data=b'{}',method='POST'))
        assert error.value.code == 403
        assert urlopen(url+'/image/'+record['id']).read().startswith(b'\x89PNG')
    finally:
        server.shutdown()
        server.server_close()
