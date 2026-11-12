import json
import re
import threading
import urllib.request
import urllib.error
import pytest
from src.common import dump_json
from src.recognition.review.failure_server import make_server


def test_review_is_scoped_preserves_sources_and_exports_corrections(tmp_path):
    root=tmp_path/'review';assets=tmp_path/'images';assets.mkdir()
    image=assets/'crop.txt';image.write_text('unchanged source')
    dump_json(root/'index.json',dict(metadata=dict(version='V42'),rows=[dict(id='crop1',source_menu='menu-001',label='円',prediction='',input_sha256='fixture')],vocab_chars=['円','魚']))
    dump_json(root/'assets.json',dict(allowed_roots=['images'],assets={'input/crop1':'images/crop.txt'}))
    server=make_server(root,0,workspace=tmp_path)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    url=f'http://127.0.0.1:{server.server_port}';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def post(body,token=None):
        headers={'Content-Type':'application/json'}
        if token:headers['X-Review-Token']=token
        return opener.open(urllib.request.Request(url+'/api/review',json.dumps(body).encode(),headers=headers))
    try:
        page=opener.open(url+'/').read().decode();token=re.search("const TOKEN='([^']+)'",page).group(1)
        assert opener.open(url+'/assets/input/crop1').read()==b'unchanged source'
        with pytest.raises(urllib.error.HTTPError) as error:opener.open(url+'/assets/%2e%2e/index.json')
        assert error.value.code==404
        body=dict(id='crop1',category='wrong_label',correction='魚',note='fixture only')
        with pytest.raises(urllib.error.HTTPError) as error:post(body)
        assert error.value.code==403 and not (root/'reviews.json').exists()
        response=json.load(post(body,token));assert response['correction']=='魚' and response['correction_oov']==[]
        saved=json.load(opener.open(url+'/api/reviews'));assert saved['crop1']['category']=='wrong_label'
        with opener.open(url+'/api/export') as response:
            assert 'v42_training_reviews.jsonl' in response.headers['Content-Disposition']
            exported=json.loads(response.read())
        assert exported['original_label']=='円' and exported['review']['correction']=='魚'
        assert image.read_text()=='unchanged source'
        with pytest.raises(urllib.error.HTTPError) as error:post(dict(body,correction='円'),token)
        assert error.value.code==400
        response=json.load(post(dict(body,category='label_correct'),token))
        assert response['correction']=='' and len(response['history'])==1
    finally:server.shutdown();server.server_close();thread.join(timeout=2)


def test_asset_manifest_cannot_escape_allowlisted_roots(tmp_path):
    root=tmp_path/'review'
    dump_json(root/'index.json',dict(rows=[]))
    dump_json(root/'assets.json',dict(allowed_roots=['images'],assets={'secret':'secret.txt'}))
    with pytest.raises(AssertionError):make_server(root,0,workspace=tmp_path)
