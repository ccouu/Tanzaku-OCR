import json
import re
import threading
import urllib.request
import urllib.error
import pytest
from src.common import dump_json
from src.detection.review.crop_review_server import make_server


def test_crop_review_routes_and_feedback_are_scoped(tmp_path):
    root=tmp_path/'review';dump_json(root/'index.json',dict(menus=[dict(id='menu-001')]))
    dump_json(root/'menus/menu-001.json',dict(id='menu-001',processed=[]))
    (tmp_path/'outside.txt').write_text('mustnotbeserved')
    server=make_server(root,0);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    port=server.server_port;url=f'http://127.0.0.1:{port}';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        assert json.load(opener.open(url+'/api/menu/menu-001'))['id']=='menu-001'
        with pytest.raises(urllib.error.HTTPError) as error:opener.open(url+'/assets/%2e%2e/outside.txt')
        assert error.value.code==404
        with pytest.raises(urllib.error.HTTPError) as error:opener.open(url+'/api/menu/unknown')
        assert error.value.code==404
        body=json.dumps(dict(id='menu-001',status='accepted',note='fixtureonly')).encode()
        request=urllib.request.Request(url+'/api/feedback',body,headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as error:opener.open(request)
        assert error.value.code==403 and not (root/'feedback.json').exists()
        html=opener.open(url+'/').read().decode();token=re.search("const token='([^']+)'",html).group(1)
        request=urllib.request.Request(url+'/api/feedback',body,headers={'Content-Type':'application/json','X-Review-Token':token})
        assert json.load(opener.open(request))['status']=='accepted'
        assert json.load(opener.open(url+'/api/feedback'))['menu-001']['note']=='fixtureonly'
    finally:server.shutdown();server.server_close();thread.join(timeout=2)
